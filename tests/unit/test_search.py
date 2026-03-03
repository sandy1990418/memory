"""Tests for pipeline/retrieval/search.py — query expansion and keyword extraction."""

import unittest

from openclaw_memory.pipeline.retrieval.search import extract_keywords


class TestExtractKeywords(unittest.TestCase):
    def test_filters_stop_words_en(self):
        keywords = extract_keywords("what is the best way to find something")
        for sw in ["what", "is", "the", "to", "find", "something"]:
            self.assertNotIn(sw, keywords)

    def test_keeps_meaningful_words(self):
        keywords = extract_keywords("favorite sushi restaurant in Tokyo")
        self.assertIn("favorite", keywords)
        self.assertIn("sushi", keywords)
        self.assertIn("restaurant", keywords)
        self.assertIn("tokyo", keywords)

    def test_chinese_characters(self):
        keywords = extract_keywords("我喜欢寿司")
        # Should extract individual CJK chars and bigrams
        self.assertTrue(len(keywords) > 0)
        # Stop words should be filtered
        self.assertNotIn("我", keywords)

    def test_filters_short_ascii(self):
        keywords = extract_keywords("I am ok at AI")
        # "I", "am", "ok", "at" are < 3 chars or stop words
        for short in ["am", "ok", "at"]:
            self.assertNotIn(short, keywords)

    def test_filters_digits_only(self):
        keywords = extract_keywords("room 42 temperature")
        self.assertNotIn("42", keywords)
        self.assertIn("room", keywords)

    def test_empty_query(self):
        self.assertEqual(extract_keywords(""), [])

    def test_deduplicates(self):
        keywords = extract_keywords("sushi sushi sushi")
        self.assertEqual(keywords.count("sushi"), 1)

    def test_mixed_en_zh(self):
        keywords = extract_keywords("favorite 寿司 restaurant")
        self.assertIn("favorite", keywords)
        self.assertIn("restaurant", keywords)


if __name__ == "__main__":
    unittest.main()
