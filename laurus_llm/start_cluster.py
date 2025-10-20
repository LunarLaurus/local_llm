#!/usr/bin/env python3
"""
start_cluster.py - Improved for Ubuntu

- Staggered backend startup to avoid disk/network congestion
- Wait for model load before marking backends ready
- Async least-connections load balancer
- Full logging of backend stdout/stderr
"""

import argparse
import asyncio
import logging
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
# Utilities
# ----------------------------
async def wait_for_http(url: str, timeout: float = 1.0, tries: int = 60) -> bool:
    import aiohttp

    for _ in range(tries):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=timeout) as r:
                    if r.status < 500:
                        return True
        except Exception:
            await asyncio.sleep(1)
    return False


def predownload_model(model_id: str, bitness: str):
    """
    Pre-download the HuggingFace model and tokenizer to populate the cache.
    Call this before starting backend processes to avoid multiple simultaneous downloads.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import logging

    logger = logging.getLogger("predownload")
    logger.info("Pre-downloading model %s (%s)...", model_id, bitness)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    try:
        if bitness == "4bit":
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        elif bitness == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        else:  # full precision
            AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        logger.warning(
            "Failed to pre-download with %s, falling back to float16: %s", bitness, e
        )
        AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    logger.info("Pre-download complete for %s", model_id)


async def read_stream(proc, name):
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        LOG.info("[%s stdout] %s", name, line.decode(errors="ignore").rstrip())


async def start_server_instances(
    num: int,
    base_port: int,
    model_id: str,
    bitness: str,
    host: str,
    cores_total: int,
    threads_per_proc: int,
    extra_env: dict,
    stagger: int = 5,
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
        cmd += [sys.executable, "-u", "-m", "server.app"]

        LOG.info(
            "Starting server %d on port %d cores %d-%d", i, port, start_core, end_core
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        asyncio.create_task(read_stream(proc, f"proc{port}"))
        asyncio.create_task(read_stream(proc, f"proc{port}_err"))

        procs.append((proc, port))
        LOG.info("Staggering next backend start by %ds", stagger)
        await asyncio.sleep(stagger)

    return procs


async def monitor_procs_ready(procs_ports, host, tries=60):
    ready = []
    for proc, port in procs_ports:
        url = f"http://{host}:{port}/health"
        ok = await wait_for_http(url, tries=tries)
        LOG.info("Server on port %d ready: %s", port, ok)
        ready.append(ok)
    return all(ready)


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
        stagger=5,
    )

    try:
        LOG.info(
            "Waiting for backends to become ready (model load may take minutes)..."
        )
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

        # Signal handling
        loop = asyncio.get_running_loop()
        stop_ev = asyncio.Event()
        loop.add_signal_handler(signal.SIGINT, stop_ev.set)
        loop.add_signal_handler(signal.SIGTERM, stop_ev.set)
        await stop_ev.wait()
        LOG.info("Shutdown signal received, stopping LB and servers...")

        lb.stop()
        await asyncio.sleep(0.1)

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--base-port", type=int, default=8000)
    p.add_argument("--lb-port", type=int, default=8080)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--model-id", type=str, default=choose_model())
    p.add_argument(
        "--bitness", type=str, default="16bit", choices=("4bit", "8bit", "16bit")
    )
    p.add_argument("--threads-per-proc", type=int, default=10)
    p.add_argument("--cores-total", type=int, default=80)
    p.add_argument("--max-concurrency-per-backend", type=int, default=4)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--health-interval", type=int, default=5)
    p.add_argument("--access-log", type=int, choices=(0, 1), default=0)
    p.add_argument("--model-auth", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        LOG.info("Interrupted")


if __name__ == "__main__":
    main()
