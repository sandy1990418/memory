"""Tests for utils/ modules: similarity, text, tokens."""

import math
import unittest

from openclaw_memory.utils.similarity import (
    cosine_similarity,
    cosine_similarity_any,
    jaccard_similarity,
    text_similarity,
    tokenize,
)
from openclaw_memory.utils.text import (
    coerce_text,
    normalize_text,
    parse_llm_json,
    strip_markdown_fences,
    truncate_utf16_safe,
)
from openclaw_memory.utils.tokens import estimate_tokens


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0)

    def test_cosine_similarity_any_coerces_types(self):
        a = [1.0, 0.0]
        b = (0.0, 1.0)  # tuple, not list
        result = cosine_similarity_any(a, b)
        self.assertAlmostEqual(result, 0.0)


class TestJaccardSimilarity(unittest.TestCase):
    def test_identical_sets(self):
        s = frozenset({"a", "b", "c"})
        self.assertAlmostEqual(jaccard_similarity(s, s), 1.0)

    def test_disjoint_sets(self):
        a = frozenset({"a", "b"})
        b = frozenset({"c", "d"})
        self.assertAlmostEqual(jaccard_similarity(a, b), 0.0)

    def test_both_empty(self):
        self.assertAlmostEqual(jaccard_similarity(frozenset(), frozenset()), 1.0)

    def test_one_empty(self):
        self.assertAlmostEqual(jaccard_similarity(frozenset({"a"}), frozenset()), 0.0)

    def test_partial_overlap(self):
        a = frozenset({"a", "b", "c"})
        b = frozenset({"b", "c", "d"})
        # intersection=2, union=4 => 0.5
        self.assertAlmostEqual(jaccard_similarity(a, b), 0.5)


class TestTokenize(unittest.TestCase):
    def test_basic(self):
        result = tokenize("Hello World 123")
        self.assertEqual(result, frozenset({"hello", "world", "123"}))

    def test_empty(self):
        self.assertEqual(tokenize(""), frozenset())

    def test_special_chars(self):
        result = tokenize("hello-world_test!")
        self.assertIn("hello", result)
        self.assertIn("world_test", result)


class TestTextSimilarity(unittest.TestCase):
    def test_identical_text(self):
        self.assertAlmostEqual(text_similarity("hello world", "hello world"), 1.0)

    def test_different_text(self):
        result = text_similarity("hello world", "foo bar")
        self.assertAlmostEqual(result, 0.0)


class TestNormalizeText(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(normalize_text("  hello   world  "), "hello world")

    def test_strips(self):
        self.assertEqual(normalize_text("  hello  "), "hello")


class TestStripMarkdownFences(unittest.TestCase):
    def test_json_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        self.assertEqual(strip_markdown_fences(raw), '{"key": "value"}')

    def test_no_fence(self):
        raw = '{"key": "value"}'
        self.assertEqual(strip_markdown_fences(raw), '{"key": "value"}')


class TestParseLlmJson(unittest.TestCase):
    def test_valid_json(self):
        result = parse_llm_json('{"action": "merge"}')
        self.assertEqual(result, {"action": "merge"})

    def test_with_fences(self):
        result = parse_llm_json('```json\n{"action": "keep"}\n```')
        self.assertEqual(result, {"action": "keep"})

    def test_invalid_json(self):
        self.assertIsNone(parse_llm_json("not json"))

    def test_empty(self):
        self.assertIsNone(parse_llm_json(""))

    def test_array_returns_none(self):
        self.assertIsNone(parse_llm_json("[1, 2, 3]"))


class TestCoerceText(unittest.TestCase):
    def test_string(self):
        self.assertEqual(coerce_text("  hello  "), "hello")

    def test_none(self):
        self.assertEqual(coerce_text(None), "")

    def test_int(self):
        self.assertEqual(coerce_text(42), "42")

    def test_dict(self):
        result = coerce_text({"a": 1})
        self.assertIn('"a"', result)

    def test_list_of_strings(self):
        result = coerce_text(["hello", "world"])
        self.assertEqual(result, "hello world")


class TestTruncateUtf16Safe(unittest.TestCase):
    def test_no_truncation(self):
        self.assertEqual(truncate_utf16_safe("hello", 10), "hello")

    def test_truncation(self):
        self.assertEqual(truncate_utf16_safe("hello world", 5), "hello")


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_short(self):
        self.assertEqual(estimate_tokens("hi"), 1)

    def test_normal(self):
        self.assertEqual(estimate_tokens("hello world, this is a test"), 6)


if __name__ == "__main__":
    unittest.main()
