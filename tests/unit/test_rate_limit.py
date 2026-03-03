"""Tests for middleware/rate_limit.py — RateLimitMiddleware."""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from openclaw_memory.middleware.rate_limit import RateLimitMiddleware


def _make_app(rpm: int = 5, enabled: bool = True) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm, enabled=enabled)

    @app.get("/test")
    def test_endpoint():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    return app


class TestRateLimitMiddleware(unittest.TestCase):
    def test_allows_under_limit(self):
        client = TestClient(_make_app(rpm=10))
        for _ in range(10):
            resp = client.get("/test")
            self.assertEqual(resp.status_code, 200)

    def test_blocks_over_limit(self):
        client = TestClient(_make_app(rpm=3))
        for _ in range(3):
            resp = client.get("/test")
            self.assertEqual(resp.status_code, 200)
        resp = client.get("/test")
        self.assertEqual(resp.status_code, 429)

    def test_returns_rate_limit_headers(self):
        client = TestClient(_make_app(rpm=10))
        resp = client.get("/test")
        self.assertIn("x-ratelimit-limit", resp.headers)
        self.assertIn("x-ratelimit-remaining", resp.headers)
        self.assertEqual(resp.headers["x-ratelimit-limit"], "10")

    def test_health_exempt(self):
        client = TestClient(_make_app(rpm=1))
        client.get("/test")  # use up the limit
        client.get("/test")  # this would be 429
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_disabled_allows_all(self):
        client = TestClient(_make_app(rpm=1, enabled=False))
        for _ in range(20):
            resp = client.get("/test")
            self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
