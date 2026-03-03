"""Tests for middleware/auth.py — API key authentication."""

import unittest

from openclaw_memory.middleware.auth import hash_api_key


class TestHashApiKey(unittest.TestCase):
    def test_deterministic(self):
        key = "test-api-key-123"
        h1 = hash_api_key(key)
        h2 = hash_api_key(key)
        self.assertEqual(h1, h2)

    def test_different_keys_different_hashes(self):
        h1 = hash_api_key("key-a")
        h2 = hash_api_key("key-b")
        self.assertNotEqual(h1, h2)

    def test_sha256_length(self):
        h = hash_api_key("anything")
        self.assertEqual(len(h), 64)  # SHA-256 hex = 64 chars

    def test_hex_chars_only(self):
        h = hash_api_key("test")
        self.assertTrue(all(c in "0123456789abcdef" for c in h))


if __name__ == "__main__":
    unittest.main()
