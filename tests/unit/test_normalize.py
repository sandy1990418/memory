"""Tests for pipeline/ingest/normalize.py."""

import unittest

from openclaw_memory.pipeline.ingest.normalize import (
    VALID_MEMORY_TYPES,
    apply_type_fallback,
    normalize_memory_key,
    normalize_value,
    parse_event_time,
)


class TestNormalizeMemoryKey(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(normalize_memory_key(""), "")

    def test_lowercase_and_underscore(self):
        self.assertEqual(normalize_memory_key("Profile Favorite-Food"), "profile_favorite_food")

    def test_strips_invalid_chars(self):
        result = normalize_memory_key("key!@#$%^&*()")
        self.assertNotIn("!", result)
        self.assertNotIn("@", result)

    def test_collapses_dots(self):
        self.assertEqual(normalize_memory_key("a..b...c"), "a.b.c")

    def test_preserves_accented(self):
        result = normalize_memory_key("café")
        self.assertIn("café", result)


class TestNormalizeValue(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(normalize_value("  hello  "), "hello")


class TestParseEventTime(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(parse_event_time(None))

    def test_valid_date(self):
        self.assertEqual(parse_event_time("2026-01-15"), "2026-01-15")

    def test_valid_datetime(self):
        result = parse_event_time("2026-01-15T10:30:00Z")
        self.assertEqual(result, "2026-01-15T10:30:00Z")

    def test_invalid(self):
        self.assertIsNone(parse_event_time("not a date"))

    def test_empty_string(self):
        self.assertIsNone(parse_event_time(""))


class TestApplyTypeFallback(unittest.TestCase):
    def test_valid_types(self):
        for t in VALID_MEMORY_TYPES:
            self.assertEqual(apply_type_fallback(t), t)

    def test_invalid_falls_back(self):
        self.assertEqual(apply_type_fallback("unknown"), "fact")
        self.assertEqual(apply_type_fallback(""), "fact")


if __name__ == "__main__":
    unittest.main()
