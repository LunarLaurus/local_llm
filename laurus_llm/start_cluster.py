#!/usr/bin/env python3
"""
start_cluster.py - Refactored Cluster Manager

- Staggered backend startup
- Wait for model load and queue readiness
- Async least-connections load balancer
- Full logging, graceful shutdown
"""

import argparse
import asyncio
import logging
import multiprocessing
import os
import shutil
import signal
import sys
from collections import deque
from typing import List

from aiohttp import web, ClientSession, TCPConnector
from laurus_llm.app import choose_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
LOG = logging.getLogger("start_cluster")


# ----------------------------
# Stream reader for subprocess output
# ----------------------------
async def read_stream(name: str, stream: asyncio.StreamReader):
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            LOG.info("[%s] %s", name, line.decode(errors="ignore").rstrip())
    except asyncio.CancelledError:
        return
    except Exception as e:
        LOG.warning("[%s] stream read error: %s", name, e)


# ----------------------------
# Backend / Load Balancer
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
        retries: int = 3,
        health_path: str = "/health",
        health_interval: float = 30.0,
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
            if b.active >= b.max_concurrency:
                await asyncio.sleep(0.01 * (2**attempt))
                attempt += 1
                continue

            async with b.lock:
                b.active += 1
            start = asyncio.get_event_loop().time()
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
                    response = web.StreamResponse(
                        status=resp.status, reason=resp.reason
                    )
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
                    latency = asyncio.get_event_loop().time() - start
                    LOG.info("Proxied request to %s in %.3fs", b.url, latency)
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
# Backend readiness
# ----------------------------
async def wait_for_backend_ready(
    url: str, timeout: float = 1.0, tries: int = 60
) -> bool:
    import aiohttp

    for _ in range(tries):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{url}/health", timeout=timeout) as r:
                    if r.status < 500:
                        data = await r.json()
                        if data.get("model_loaded"):
                            return True
        except Exception:
            await asyncio.sleep(1)
    return False


async def monitor_procs_ready(procs_ports, host, tries=60):
    ready = []
    for _proc, port in procs_ports:
        url = f"http://{host}:{port}"
        ok = await wait_for_backend_ready(url, tries=tries)
        LOG.info("Server on port %d ready: %s", port, ok)
        ready.append(ok)
    return all(ready)


# ----------------------------
# Model pre-download
# ----------------------------
def predownload_model(model_id: str, bitness: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import logging

    logger = logging.getLogger("predownload")
    logger.info("Pre-downloading model %s (%s)...", model_id, bitness)
    AutoTokenizer.from_pretrained(model_id, use_fast=True)
    try:
        if bitness == "4bit":
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, trust_remote_code=True
            )
        elif bitness == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, trust_remote_code=True
            )
        else:
            AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        logger.warning("Failed pre-download with %s, falling back: %s", bitness, e)
        AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    logger.info("Pre-download complete for %s", model_id)


# ----------------------------
# Start server instances
# ----------------------------
async def start_server_instances(
    num: int,
    base_port: int,
    model_id: str,
    bitness: str,
    host: str,
    cores_total: int,
    threads_per_proc: int,
    extra_env: dict,
    stagger: int = 10,
):
    procs = []
    cores_per_proc = max(1, cores_total // num)
    taskset_path = shutil.which("taskset")

    for i in range(num):
        port = base_port + i
        start_core = i * cores_per_proc
        end_core = start_core + cores_per_proc - 1

        env = os.environ.copy()
        env.update(extra_env)
        env.update(
            {
                "LLLM_PORT": str(port),
                "LLLM_HOST": host,
                "LLLM_MODEL_ID": model_id,
                "LLLM_BITNESS": bitness,
                "OMP_NUM_THREADS": str(threads_per_proc),
                "MKL_NUM_THREADS": str(threads_per_proc),
                "OPENBLAS_NUM_THREADS": str(threads_per_proc),
                "TORCH_NUM_THREADS": str(threads_per_proc),
                "TORCH_INTEROP_THREADS": str(max(1, threads_per_proc // 4)),
            }
        )

        cmd = []
        if taskset_path:
            cmd += [taskset_path, "-c", f"{start_core}-{end_core}"]
        cmd += [sys.executable, "-u", "-m", "laurus_llm.app"]

        LOG.info(
            "Starting server %d on port %d cores %d-%d", i, port, start_core, end_core
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        asyncio.create_task(read_stream(f"proc-{port}-stdout", proc.stdout))
        asyncio.create_task(read_stream(f"proc-{port}-stderr", proc.stderr))

        procs.append((proc, port))
        LOG.info("Staggering next backend start by %ds", stagger)
        await asyncio.sleep(stagger)

    return procs


async def shutdown_backends(procs_ports, host):
    import aiohttp

    async with aiohttp.ClientSession() as s:
        for _proc, port in procs_ports:
            url = f"http://{host}:{port}/shutdown"
            try:
                await s.post(url, json={"reason": "Cluster stopping"})
                LOG.info("Sent shutdown to backend %d", port)
            except Exception as e:
                LOG.warning("Failed shutdown request to %d: %s", port, e)


# ----------------------------
# Main async lifecycle
# ----------------------------
async def main_async(args):
    extra_env = {}
    if args.access_log:
        extra_env["LLLM_ACCESS_LOG"] = "true"
    if args.model_auth:
        extra_env["HF_AUTH_TOKEN"] = args.model_auth

    LOG.info("Pre-downloading model cache...")
    predownload_model(args.model_id, args.bitness)

    procs = await start_server_instances(
        num=args.num,
        base_port=args.base_port,
        model_id=args.model_id,
        bitness=args.bitness,
        host=args.host,
        cores_total=args.cores_total,
        threads_per_proc=args.threads_per_proc,
        extra_env=extra_env,
        stagger=10,
    )

    try:
        LOG.info("Waiting for backends to become ready...")
        all_ready = await monitor_procs_ready(procs, args.host, tries=120)
        if not all_ready:
            LOG.warning(
                "Some backends did not become ready in time, LB will still start."
            )

        backend_urls = [f"http://{args.host}:{port}" for (_proc, port) in procs]
        lb = AsyncLoadBalancer(
            backends=backend_urls,
            max_concurrency_per_backend=args.max_concurrency_per_backend,
            retries=args.retries,
            health_path="/health",
            health_interval=args.health_interval,
        )
        lb_task = asyncio.create_task(lb.start(host=args.host, port=args.lb_port))

        loop = asyncio.get_running_loop()
        stop_ev = asyncio.Event()
        loop.add_signal_handler(signal.SIGINT, stop_ev.set)
        loop.add_signal_handler(signal.SIGTERM, stop_ev.set)
        await stop_ev.wait()
        LOG.info("Shutdown signal received, stopping LB and servers...")

        lb.stop()
        await shutdown_backends(procs, args.host)
        await asyncio.sleep(1)

    finally:
        for proc, port in procs:
            LOG.info("Terminating process on port %d", port)
            proc.terminate()
        await asyncio.sleep(1.0)
        for proc, port in procs:
            if proc.returncode is None:
                LOG.info("Killing process on port %d", port)
                proc.kill()

    LOG.info("Cluster stopped.")


# ----------------------------
# Argument parsing
# ----------------------------
def parse_args():
    import multiprocessing

    total_cores = multiprocessing.cpu_count()

    p = argparse.ArgumentParser(description="Start LLM cluster with load balancer")

    # ---------------- Core cluster parameters ----------------
    p.add_argument(
        "--num",
        type=int,
        default=8,
        help="Number of backend instances",
    )
    p.add_argument(
        "--base-port",
        type=int,
        default=8001,
        help="Base port for backend instances",
    )
    p.add_argument(
        "--lb-port",
        type=int,
        default=8000,
        help="Port for the load balancer",
    )
    p.add_argument("--host", type=str, default="0.0.0.0", help="Host for all servers")
    p.add_argument("--model-id", type=str, default=choose_model(), help="Model to load")
    p.add_argument(
        "--bitness",
        type=str,
        default="16bit",
        choices=("4bit", "8bit", "16bit"),
        help="Model precision / quantization",
    )

    # ---------------- Thread / concurrency ----------------
    p.add_argument(
        "--threads-per-proc",
        type=int,
        default=None,
        help="Threads per backend instance (derived from total cores / num if not set)",
    )
    p.add_argument(
        "--cores-total",
        type=int,
        default=total_cores,
        help="Total CPU cores available",
    )
    p.add_argument(
        "--max-concurrency-per-backend",
        type=int,
        default=None,
        help="Max concurrent requests per backend (derived from threads-per-proc if not set)",
    )

    # ---------------- Other options ----------------
    p.add_argument("--retries", type=int, default=3, help="Retries per request in LB")
    p.add_argument(
        "--health-interval", type=int, default=30, help="Health check interval"
    )
    p.add_argument(
        "--access-log", type=int, choices=(0, 1), default=0, help="Enable access log"
    )
    p.add_argument(
        "--model-auth", type=str, default=None, help="HF auth token if needed"
    )

    args = p.parse_args()

    # ---------------- Derived values ----------------
    # Threads per proc based on actual number of instances
    if args.threads_per_proc is None:
        args.threads_per_proc = max(1, args.cores_total // args.num)

    # Max concurrency per backend based on threads per proc
    if args.max_concurrency_per_backend is None:
        args.max_concurrency_per_backend = max(1, args.threads_per_proc // 4)

    # Compute cores per instance for taskset / logging
    args.cores_per_proc = max(1, args.cores_total // args.num)

    # ---------------- Logging ----------------
    LOG.info("Cluster arguments:")
    for k, v in vars(args).items():
        LOG.info("  %s = %s", k, v)

    return args


def main():
    args = parse_args()
    try:
        LOG.info("Starting cluster...")
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")


if __name__ == "__main__":
    main()
