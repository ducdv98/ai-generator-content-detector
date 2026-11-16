"""Abuse protection for a public deployment: per-IP rate limits, a cap on concurrent
heavy jobs, and an upload-size check that runs before the body is read.

Pure ASGI (not BaseHTTPMiddleware) so streaming responses such as /api/text/generate
keep their concurrency slot until the last chunk is sent.

Settings (environment variables):
  AIGC_LIMITS          "off" disables everything (the test suite does this).
  AIGC_RATE_HEAVY      "N/SECONDS" per IP for CPU-heavy POST routes   (default 8/60)
  AIGC_RATE_LIGHT      "N/SECONDS" per IP for every other request     (default 240/60)
  AIGC_MAX_HEAVY_JOBS  heavy requests allowed to run at once          (default 2)
  AIGC_MAX_BODY_MB     largest request body accepted                  (default 85)
  AIGC_TRUST_CF        "1": take the client IP from CF-Connecting-IP when the request
                       arrives from loopback (the Cloudflare Tunnel daemon).
"""
from __future__ import annotations

import json
import os
import time
from collections import deque

# POST routes that decode media, run the language model, or re-encode video many times.
HEAVY_PREFIXES = (
    "/api/text/generate",
    "/api/video/",
    "/api/audio/",
    "/api/image/survive",
    "/api/sign",
)
HEAVY_GET = ("/api/video/sample",)      # runs ffmpeg
_MAX_TRACKED_IPS = 20_000


def _parse_rate(text: str) -> tuple[int, float]:
    n, _, seconds = text.partition("/")
    return int(n), float(seconds)


class _Window:
    """Sliding-window counter per key."""

    def __init__(self, limit: int, seconds: float) -> None:
        self.limit, self.seconds = limit, seconds
        self.hits: dict[str, deque[float]] = {}

    def allow(self, key: str, now: float) -> float:
        """Return 0 if the request is allowed, else seconds until it would be."""
        q = self.hits.setdefault(key, deque())
        while q and q[0] <= now - self.seconds:
            q.popleft()
        if len(q) >= self.limit:
            return q[0] + self.seconds - now
        q.append(now)
        return 0.0

    def prune(self, now: float) -> None:
        if len(self.hits) < _MAX_TRACKED_IPS:
            return
        for k in [k for k, q in self.hits.items() if not q or q[-1] <= now - self.seconds]:
            del self.hits[k]


class Limits:
    def __init__(self, app, *, heavy: tuple[int, float] | None = None,
                 light: tuple[int, float] | None = None, max_heavy_jobs: int | None = None,
                 max_body_bytes: int | None = None, trust_cf: bool | None = None,
                 enabled: bool | None = None, clock=time.monotonic) -> None:
        env = os.environ.get
        self.app = app
        self.enabled = (env("AIGC_LIMITS", "on").lower() != "off") if enabled is None else enabled
        self.heavy = _Window(*(heavy or _parse_rate(env("AIGC_RATE_HEAVY", "8/60"))))
        self.light = _Window(*(light or _parse_rate(env("AIGC_RATE_LIGHT", "240/60"))))
        self.max_jobs = max_heavy_jobs or int(env("AIGC_MAX_HEAVY_JOBS", "2"))
        self.max_body = max_body_bytes or int(float(env("AIGC_MAX_BODY_MB", "85")) * 1024 * 1024)
        self.trust_cf = (env("AIGC_TRUST_CF", "0") == "1") if trust_cf is None else trust_cf
        self.running = 0
        self.clock = clock

    # ------------------------------------------------------------------ helpers
    def _client_ip(self, scope) -> str:
        peer = (scope.get("client") or ("unknown", 0))[0]
        if self.trust_cf and peer in ("127.0.0.1", "::1"):
            for name, value in scope["headers"]:
                if name == b"cf-connecting-ip":
                    return value.decode("latin-1").strip() or peer
        return peer

    @staticmethod
    def _is_heavy(method: str, path: str) -> bool:
        if method == "POST":
            return path.startswith(HEAVY_PREFIXES)
        return method == "GET" and path in HEAVY_GET

    @staticmethod
    async def _reject(send, status: int, message: str, retry_after: float | None = None) -> None:
        body = json.dumps({"detail": message}).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
        if retry_after is not None:
            headers.append((b"retry-after", str(max(1, int(retry_after + 0.999))).encode()))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    # -------------------------------------------------------------------- ASGI
    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not self.enabled:
            return await self.app(scope, receive, send)

        method, path = scope["method"], scope["path"]
        now = self.clock()
        ip = self._client_ip(scope)
        heavy = self._is_heavy(method, path)

        # Refuse oversized uploads before the framework spools them to disk.
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    too_big = int(value) > self.max_body
                except ValueError:
                    too_big = True
                if too_big:
                    return await self._reject(send, 413, "Upload is too large.")

        self.light.prune(now)
        self.heavy.prune(now)
        wait = self.light.allow(ip, now)
        if not wait and heavy:
            wait = self.heavy.allow(ip, now)
        if wait:
            return await self._reject(send, 429, "Too many requests. Please slow down.", wait)

        if not heavy:
            return await self.app(scope, receive, send)

        if self.running >= self.max_jobs:
            return await self._reject(send, 503, "The server is busy with other requests. "
                                                  "Try again in a few seconds.", 5)
        self.running += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.running -= 1
