"""Tests for pipeline/retrieval/hybrid.py."""

import unittest

from openclaw_memory.pipeline.retrieval.hybrid import (
    build_fts_query,
    merge_hybrid_results,
)


class TestBuildFtsQuery(unittest.TestCase):
    def test_single_word(self):
        self.assertEqual(build_fts_query("hello"), "hello")

    def test_multiple_words(self):
        result = build_fts_query("hello world")
        self.assertEqual(result, "hello & world")

    def test_empty(self):
        self.assertIsNone(build_fts_query(""))

    def test_special_chars(self):
        result = build_fts_query("hello! world?")
        self.assertIsNotNone(result)
        self.assertIn("hello", result)
        self.assertIn("world", result)


class TestMergeHybridResults(unittest.TestCase):
    def test_empty(self):
        result = merge_hybrid_results(vector=[], keyword=[])
        self.assertEqual(result, [])

    def test_vector_only(self):
        vector = [
            {"id": "1", "snippet": "memory A", "vector_score": 0.9, "source": "episodic"},
        ]
        result = merge_hybrid_results(vector=vector, keyword=[])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["score"], 0.7 * 0.9)

    def test_keyword_only(self):
        keyword = [
            {"id": "1", "snippet": "memory A", "text_score": 0.8, "source": "episodic"},
        ]
        result = merge_hybrid_results(vector=[], keyword=keyword)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["score"], 0.3 * 0.8)

    def test_both_same_id(self):
        vector = [{"id": "1", "snippet": "test", "vector_score": 0.9, "source": "episodic"}]
        keyword = [{"id": "1", "snippet": "test", "text_score": 0.8, "source": "episodic"}]
        result = merge_hybrid_results(vector=vector, keyword=keyword)
        self.assertEqual(len(result), 1)
        expected = 0.7 * 0.9 + 0.3 * 0.8
        self.assertAlmostEqual(result[0]["score"], expected, places=5)

    def test_different_ids(self):
        vector = [{"id": "1", "snippet": "A", "vector_score": 0.9, "source": "e"}]
        keyword = [{"id": "2", "snippet": "B", "text_score": 0.8, "source": "e"}]
        result = merge_hybrid_results(vector=vector, keyword=keyword)
        self.assertEqual(len(result), 2)

    def test_sorted_by_score_descending(self):
        vector = [
            {"id": "1", "snippet": "low", "vector_score": 0.3, "source": "e"},
            {"id": "2", "snippet": "high", "vector_score": 0.9, "source": "e"},
        ]
        result = merge_hybrid_results(vector=vector, keyword=[])
        self.assertGreater(result[0]["score"], result[1]["score"])

    def test_custom_weights(self):
        vector = [{"id": "1", "snippet": "A", "vector_score": 1.0, "source": "e"}]
        keyword = [{"id": "1", "snippet": "A", "text_score": 1.0, "source": "e"}]
        result = merge_hybrid_results(
            vector=vector, keyword=keyword, vector_weight=0.5, text_weight=0.5,
        )
        self.assertAlmostEqual(result[0]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
