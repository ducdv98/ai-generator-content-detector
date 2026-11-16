"""Tests for app/limits.py using a tiny ASGI app, so no media code is involved."""
from __future__ import annotations

import asyncio

import httpx

from app.limits import Limits


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make(gate: asyncio.Event | None = None, **kw):
    async def inner(scope, receive, send):
        if gate is not None and scope["path"].startswith("/api/audio/"):
            await gate.wait()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})

    kw.setdefault("enabled", True)
    return Limits(inner, **kw)


def client(app, ip: str = "10.0.0.1") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=(ip, 5000)),
                             base_url="http://t")


def run(coro):
    return asyncio.run(coro)


def test_heavy_route_is_rate_limited_per_ip():
    async def go():
        clock = Clock()
        app = make(heavy=(2, 60), light=(100, 60), clock=clock)
        async with client(app) as c, client(app, "10.0.0.2") as other:
            assert (await c.post("/api/video/make")).status_code == 200
            assert (await c.post("/api/video/check")).status_code == 200
            r = await c.post("/api/video/make")
            assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
            assert (await other.post("/api/video/make")).status_code == 200   # other IP unaffected
            clock.t += 61                                                     # window slides
            assert (await c.post("/api/video/make")).status_code == 200
    run(go())


def test_light_routes_use_the_looser_limit():
    async def go():
        app = make(heavy=(1, 60), light=(3, 60), clock=Clock())
        async with client(app) as c:
            codes = [(await c.get("/api/health")).status_code for _ in range(4)]
            assert codes == [200, 200, 200, 429]
    run(go())


def test_heavy_jobs_beyond_the_cap_get_503_and_release_the_slot():
    async def go():
        gate = asyncio.Event()
        app = make(gate, heavy=(100, 60), light=(100, 60), max_heavy_jobs=1, clock=Clock())
        async with client(app) as c:
            first = asyncio.create_task(c.post("/api/audio/make"))
            await asyncio.sleep(0.05)                      # first request now holds the slot
            busy = await c.post("/api/audio/check")
            assert busy.status_code == 503 and "retry-after" in busy.headers
            assert (await c.get("/api/health")).status_code == 200            # light route unaffected
            gate.set()
            assert (await first).status_code == 200
            assert (await c.post("/api/audio/check")).status_code == 200      # slot was released
    run(go())


def test_oversized_upload_is_refused_before_reading():
    async def go():
        app = make(max_body_bytes=1000, clock=Clock())
        async with client(app) as c:
            r = await c.post("/api/analyze", content=b"x" * 2000)
            assert r.status_code == 413
            assert (await c.post("/api/analyze", content=b"x" * 500)).status_code == 200
    run(go())


def test_client_ip_comes_from_cloudflare_header_only_when_trusted():
    async def go():
        for trust, expected in ((True, [200, 200]), (False, [200, 429])):
            app = make(light=(1, 60), trust_cf=trust, clock=Clock())
            async with client(app, "127.0.0.1") as c:
                codes = []
                for ip in ("1.1.1.1", "2.2.2.2"):
                    codes.append((await c.get("/", headers={"CF-Connecting-IP": ip})).status_code)
                assert codes == expected
        # A non-loopback peer cannot spoof the header, even with trust on.
        app = make(light=(1, 60), trust_cf=True, clock=Clock())
        async with client(app, "10.9.9.9") as c:
            codes = [(await c.get("/", headers={"CF-Connecting-IP": ip})).status_code
                     for ip in ("1.1.1.1", "2.2.2.2")]
            assert codes == [200, 429]
    run(go())


def test_disabled_limiter_passes_everything_through():
    async def go():
        app = make(enabled=False, heavy=(1, 60), light=(1, 60), max_body_bytes=10)
        async with client(app) as c:
            for _ in range(3):
                assert (await c.post("/api/video/make", content=b"x" * 100)).status_code == 200
    run(go())
