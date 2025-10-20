#!/usr/bin/env python3
"""
start_cluster.py

Start multiple instances of your existing server (python -m server.app)
and run an async least-connections load balancer in the same process.

Usage example:
    python start_cluster.py --num 8 --base-port 8000 --lb-port 8080 --model my-model --bitness 16bit

NOTE: This assumes your server will read LLLM_PORT / LLLM_HOST / LLLM_MODEL_ID / LLLM_BITNESS
from environment (your app's input_with_timeout falls back to env vars). If your server still prompts,
set the env vars appropriately or change the invoked command to pass flags if available.
"""

import argparse
import asyncio
import logging
import os
import shutil
import signal
import sys
from aiohttp import web, ClientSession, TCPConnector
from collections import deque
from typing import List

from laurus_llm.app import choose_model

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("start_cluster")


# ----------------------------
# Async proxy (least-connections)
# ----------------------------
class Backend:
    def __init__(self, url: str, max_concurrency: int):
        self.url = url.rstrip("/")
        self.active = 0
        self.lock = asyncio.Lock()
        self.healthy = True
        self.successive_fails = 0
        self.max_concurrency = max_concurrency

    def __repr__(self):
        return f"<Backend {self.url} active={self.active} healthy={self.healthy}>"


class AsyncLoadBalancer:
    def __init__(
        self,
        backends: List[str],
        max_concurrency_per_backend: int = 4,
        retries: int = 1,
        health_path: str = "/health",
        health_interval: float = 5.0,
    ):
        self.backends = [Backend(b, max_concurrency_per_backend) for b in backends]
        self._rr = deque(self.backends)
        self.retries = retries
        self.health_path = health_path
        self.health_interval = health_interval

        self.connector = TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=300)
        self.session = ClientSession(connector=self.connector)

        self._stop = asyncio.Event()
        self.total_requests = 0
        self.total_errors = 0

    async def start_healthchecks(self):
        LOG.info("Starting health checks every %.1fs", self.health_interval)
        while not self._stop.is_set():
            await asyncio.gather(
                *(self._check(b) for b in self.backends), return_exceptions=True
            )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.health_interval)
            except asyncio.TimeoutError:
                continue

    async def _check(self, b: Backend):
        try:
            async with self.session.get(b.url + self.health_path, timeout=3) as r:
                ok = r.status < 500
        except Exception:
            ok = False

        if ok:
            b.healthy = True
            b.successive_fails = 0
        else:
            b.successive_fails += 1
            if b.successive_fails >= 2:
                b.healthy = False

    def choose_backend(self) -> Backend:
        healthy = [b for b in self.backends if b.healthy]
        if not healthy:
            healthy = self.backends
        # least connections first (prefer under-concurrency)
        sorted_bs = sorted(healthy, key=lambda x: (x.active, x.url))
        for b in sorted_bs:
            if b.active < b.max_concurrency:
                return b
        return sorted_bs[0]

    async def proxy_request(self, request: web.Request) -> web.StreamResponse:
        self.total_requests += 1
        last_exc = None
        attempt = 0
        body = await request.read()

        while attempt <= self.retries:
            b = self.choose_backend()
            # small backoff if overloaded
            if b.active >= b.max_concurrency:
                await asyncio.sleep(0.01 * (attempt + 1))
                attempt += 1
                continue

            async with b.lock:
                b.active += 1
            try:
                target = f"{b.url}{request.rel_url}"
                headers = dict(request.headers)
                for h in (
                    "Connection",
                    "Keep-Alive",
                    "Proxy-Authenticate",
                    "Proxy-Authorization",
                    "TE",
                    "Trailers",
                    "Transfer-Encoding",
                    "Upgrade",
                ):
                    headers.pop(h, None)

                async with self.session.request(
                    method=request.method,
                    url=target,
                    headers=headers,
                    params=request.query,
                    data=body,
                    timeout=120,
                ) as resp:
                    # Stream response back
                    response = web.StreamResponse(
                        status=resp.status, reason=resp.reason
                    )
                    # copy headers (minus hop-by-hop)
                    for k, v in resp.headers.items():
                        if k.lower() not in (
                            "connection",
                            "keep-alive",
                            "transfer-encoding",
                        ):
                            response.headers[k] = v
                    await response.prepare(request)
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        await response.write(chunk)
                    await response.write_eof()
                    return response
            except Exception as e:
                LOG.warning("Proxy error to %s: %s", b.url, e)
                last_exc = e
                self.total_errors += 1
                b.successive_fails += 1
                if b.successive_fails >= 1:
                    b.healthy = False
                attempt += 1
            finally:
                async with b.lock:
                    b.active = max(0, b.active - 1)

        LOG.error("All attempts failed: %s", last_exc)
        raise web.HTTPServiceUnavailable(text="No backend available")

    async def metrics(self, request):
        lines = [
            f"total_requests {self.total_requests}",
            f"total_errors {self.total_errors}",
        ]
        for i, b in enumerate(self.backends):
            lines.append(f'backend_active{{idx="{i}",url="{b.url}"}} {b.active}')
            lines.append(
                f'backend_healthy{{idx="{i}",url="{b.url}"}} {1 if b.healthy else 0}'
            )
        return web.Response(text="\n".join(lines), content_type="text/plain")

    async def health(self, request):
        return web.Response(text="ok")

    async def start(self, host: str, port: int):
        app = web.Application()
        app.router.add_get("/health", self.health)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_route("*", "/{tail:.*}", self.proxy_request)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=host, port=port)
        await site.start()
        self._hc_task = asyncio.create_task(self.start_healthchecks())
        LOG.info("Load balancer listening on %s:%s", host, port)
        await self._stop.wait()
        LOG.info("Shutting down LB")
        self._hc_task.cancel()
        await self.session.close()

    def stop(self):
        self._stop.set()


# ----------------------------
# Utilities to start processes
# ----------------------------
async def wait_for_http(url: str, timeout: float = 1.0, tries: int = 30) -> bool:
    import aiohttp

    for i in range(tries):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=timeout) as r:
                    if r.status < 500:
                        return True
        except Exception:
            await asyncio.sleep(0.5)
    return False


async def start_server_instances(
    num: int,
    base_port: int,
    model_id: str,
    bitness: str,
    host: str,
    cores_total: int,
    threads_per_proc: int,
    extra_env: dict,
):
    procs = []
    cores_per_proc = max(1, cores_total // num)
    taskset_path = shutil.which("taskset")

    for i in range(num):
        port = base_port + i
        start_core = i * cores_per_proc
        end_core = start_core + cores_per_proc - 1
        # env for child
        env = os.environ.copy()
        env.update(extra_env)
        env["LLLM_PORT"] = str(port)
        env["LLLM_HOST"] = host
        env["LLLM_MODEL_ID"] = model_id
        env["LLLM_BITNESS"] = bitness
        # threads envs
        env["OMP_NUM_THREADS"] = str(threads_per_proc)
        env["MKL_NUM_THREADS"] = str(threads_per_proc)
        env["OPENBLAS_NUM_THREADS"] = str(threads_per_proc)
        env["TORCH_NUM_THREADS"] = str(threads_per_proc)
        env["TORCH_INTEROP_THREADS"] = str(max(1, threads_per_proc // 4))

        # build command
        cmd = []
        if taskset_path:
            cmd += [taskset_path, "-c", f"{start_core}-{end_core}"]
        # run python -m server.app; ensure module path is available (run from project root)
        cmd += [sys.executable, "-u", "-m", "server.app"]
        LOG.info(
            "Starting server %d on port %d cores %d-%d", i, port, start_core, end_core
        )
        # start subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        procs.append((proc, port))
        # small stagger so they don't all hit disk at once
        await asyncio.sleep(1.0)
    return procs


async def monitor_procs_ready(procs_ports, host, tries=60):
    # wait for all to report /health
    ready = []
    for proc, port in procs_ports:
        url = f"http://{host}:{port}/health"
        ok = await wait_for_http(url, tries=30)
        LOG.info("Server on port %d healthy: %s", port, ok)
        ready.append(ok)
    return all(ready)


# ----------------------------
# Main lifecycle
# ----------------------------
async def main_async(args):
    # Prepare extra env passed to children (if provided)
    extra_env = {}
    if args.access_log is not None:
        extra_env["LLLM_ACCESS_LOG"] = "true" if args.access_log else "false"
    if args.model_auth:
        extra_env["HF_AUTH_TOKEN"] = args.model_auth  # example if needed

    # Start server instances
    procs = await start_server_instances(
        num=args.num,
        base_port=args.base_port,
        model_id=args.model_id,
        bitness=args.bitness,
        host=args.host,
        cores_total=args.cores_total,
        threads_per_proc=args.threads_per_proc,
        extra_env=extra_env,
    )

    try:
        all_ready = await monitor_procs_ready(procs, args.host)
        if not all_ready:
            LOG.error(
                "Some backend instances did not become healthy in time. Check logs."
            )
            # fall through to still start LB but mark unhealthy backends
        # build backends list
        backend_urls = [f"http://{args.host}:{port}" for (_proc, port) in procs]

        # create and start LB
        lb = AsyncLoadBalancer(
            backends=backend_urls,
            max_concurrency_per_backend=args.max_concurrency_per_backend,
            retries=args.retries,
            health_path="/health",
            health_interval=args.health_interval,
        )
        lb_task = asyncio.create_task(lb.start(host=args.host, port=args.lb_port))

        # hook signals to shutdown
        loop = asyncio.get_running_loop()
        stop_ev = asyncio.Event()

        def _signal(_sig):
            LOG.info("Signal received, shutting down...")
            stop_ev.set()

        loop.add_signal_handler(signal.SIGINT, lambda: _signal(signal.SIGINT))
        loop.add_signal_handler(signal.SIGTERM, lambda: _signal(signal.SIGTERM))

        # wait for stop
        await stop_ev.wait()
        LOG.info("Stopping load balancer and servers...")

        # stop LB
        lb.stop()
        await asyncio.sleep(0.1)

    finally:
        # Terminate child processes
        for proc, port in procs:
            try:
                LOG.info("Terminating process serving port %d", port)
                proc.terminate()
            except Exception:
                pass
        # give them a moment and kill if still alive
        await asyncio.sleep(1.0)
        for proc, port in procs:
            if proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass

        # Attempt to read remaining stdout/stderr for debugging
        for proc, port in procs:
            try:
                out, err = await proc.communicate(timeout=0.1)
                if out:
                    LOG.info(
                        "proc[%d] stdout: %s", port, out.decode(errors="ignore")[:400]
                    )
                if err:
                    LOG.info(
                        "proc[%d] stderr: %s", port, err.decode(errors="ignore")[:400]
                    )
            except Exception:
                pass

    LOG.info("Cluster stopped.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=8, help="Number of backend instances")
    p.add_argument("--base-port", type=int, default=8000, help="First backend port")
    p.add_argument("--lb-port", type=int, default=8080, help="LB listening port")
    p.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind backends and LB"
    )
    p.add_argument(
        "--model-id",
        type=str,
        default=choose_model(),
        help="Model id to pass to backends (LLLM_MODEL_ID)",
    )
    p.add_argument(
        "--bitness", type=str, default="16bit", choices=("4bit", "8bit", "16bit")
    )
    p.add_argument(
        "--threads-per-proc",
        type=int,
        default=10,
        help="Intra-process thread count exported to children",
    )
    p.add_argument(
        "--cores-total", type=int, default=80, help="Total cores available on machine"
    )
    p.add_argument(
        "--max-concurrency-per-backend",
        type=int,
        default=4,
        help="Max concurrent requests per backend",
    )
    p.add_argument("--retries", type=int, default=1, help="Proxy retries per request")
    p.add_argument(
        "--health-interval", type=int, default=5, help="Health check interval seconds"
    )
    p.add_argument(
        "--access-log",
        type=int,
        choices=(0, 1),
        default=0,
        help="Set LLLM_ACCESS_LOG for backends (0/1)",
    )
    p.add_argument(
        "--model-auth",
        type=str,
        default=None,
        help="Optional model auth token environment variable name/value",
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        LOG.info("Interrupted")


if __name__ == "__main__":
    main()
