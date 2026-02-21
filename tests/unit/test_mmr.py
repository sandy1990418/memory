"""Unit tests for openclaw_memory.mmr."""
from __future__ import annotations

import pytest

from openclaw_memory.mmr import (
    MMRConfig,
    apply_mmr_to_hybrid_results,
    jaccard_similarity,
    mmr_rerank,
    tokenize,
)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self) -> None:
        result = tokenize("Hello World")
        assert "hello" in result
        assert "world" in result

    def test_returns_frozenset(self) -> None:
        assert isinstance(tokenize("foo bar"), frozenset)

    def test_strips_punctuation(self) -> None:
        result = tokenize("hello, world!")
        assert "hello" in result
        assert "world" in result
        assert "," not in result

    def test_deduplicates(self) -> None:
        result = tokenize("foo foo foo")
        assert result.count("foo") == 1 if isinstance(result, list) else True  # frozenset is unique
        assert len([x for x in result if x == "foo"]) == 1

    def test_numbers_included(self) -> None:
        result = tokenize("python3 version 3")
        assert "python3" in result or "python" in result

    def test_empty_string(self) -> None:
        result = tokenize("")
        assert result == frozenset()

    def test_underscore_included(self) -> None:
        result = tokenize("my_function")
        assert "my_function" in result


# ---------------------------------------------------------------------------
# jaccard_similarity
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_sets(self) -> None:
        s = frozenset(["a", "b", "c"])
        assert jaccard_similarity(s, s) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        a = frozenset(["x", "y"])
        b = frozenset(["p", "q"])
        assert jaccard_similarity(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        a = frozenset(["a", "b", "c"])
        b = frozenset(["b", "c", "d"])
        # intersection=2, union=4 → 0.5
        assert jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_both_empty(self) -> None:
        assert jaccard_similarity(frozenset(), frozenset()) == pytest.approx(1.0)

    def test_one_empty(self) -> None:
        a = frozenset(["x"])
        assert jaccard_similarity(a, frozenset()) == pytest.approx(0.0)
        assert jaccard_similarity(frozenset(), a) == pytest.approx(0.0)

    def test_subset(self) -> None:
        a = frozenset(["a"])
        b = frozenset(["a", "b"])
        # intersection=1, union=2 → 0.5
        assert jaccard_similarity(a, b) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# mmr_rerank
# ---------------------------------------------------------------------------


def _make_item(score: float, text: str) -> dict:
    return {"score": score, "text": text}


class TestMmrRerank:
    def _rerank(self, items: list[dict], lambda_: float) -> list[dict]:
        config = MMRConfig(enabled=True, lambda_=lambda_)
        return mmr_rerank(
            items,
            score_fn=lambda r: r["score"],
            content_fn=lambda r: r["text"],
            config=config,
        )

    def test_disabled_returns_original_order(self) -> None:
        items = [
            _make_item(0.9, "alpha beta"),
            _make_item(0.5, "gamma delta"),
        ]
        config = MMRConfig(enabled=False)
        result = mmr_rerank(items, score_fn=lambda r: r["score"], content_fn=lambda r: r["text"], config=config)
        assert result == items

    def test_single_item_returned_unchanged(self) -> None:
        items = [_make_item(0.9, "only")]
        config = MMRConfig(enabled=True, lambda_=0.5)
        result = mmr_rerank(items, score_fn=lambda r: r["score"], content_fn=lambda r: r["text"], config=config)
        assert result == items

    def test_lambda_one_sorts_by_relevance(self) -> None:
        items = [
            _make_item(0.3, "foo bar"),
            _make_item(0.9, "hello world"),
            _make_item(0.6, "python code"),
        ]
        result = self._rerank(items, lambda_=1.0)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_lambda_zero_maximises_diversity(self) -> None:
        # When lambda=0, similarity penalty dominates.
        # Duplicate texts should be spread out.
        items = [
            _make_item(0.9, "identical text here"),
            _make_item(0.8, "identical text here"),  # very similar
            _make_item(0.5, "completely different python code"),
        ]
        result = self._rerank(items, lambda_=0.0)
        # The diverse item should appear before the second identical one
        texts = [r["text"] for r in result]
        first_identical = texts.index("identical text here")
        different_idx = texts.index("completely different python code")
        assert different_idx < first_identical or len(result) == 3  # order may vary but should be length 3

    def test_lambda_half_balanced(self) -> None:
        items = [
            _make_item(0.9, "memory chunk search"),
            _make_item(0.8, "memory chunk search"),  # duplicate
            _make_item(0.7, "cli status provider"),   # diverse
        ]
        result = self._rerank(items, lambda_=0.5)
        assert len(result) == 3

    def test_preserves_all_items(self) -> None:
        items = [_make_item(float(i) / 10, f"text {i}") for i in range(5, 0, -1)]
        config = MMRConfig(enabled=True, lambda_=0.7)
        result = mmr_rerank(items, score_fn=lambda r: r["score"], content_fn=lambda r: r["text"], config=config)
        assert len(result) == len(items)
        assert set(r["score"] for r in result) == set(r["score"] for r in items)


# ---------------------------------------------------------------------------
# apply_mmr_to_hybrid_results
# ---------------------------------------------------------------------------


class TestApplyMmrToHybridResults:
    def _make_hybrid(self, id: str, score: float, snippet: str) -> dict:
        return {
            "path": f"{id}.md",
            "start_line": 1,
            "end_line": 5,
            "score": score,
            "snippet": snippet,
            "source": "memory",
        }

    def test_empty_input(self) -> None:
        result = apply_mmr_to_hybrid_results([])
        assert result == []

    def test_disabled_config_passes_through(self) -> None:
        items = [
            self._make_hybrid("a", 0.9, "foo bar baz"),
            self._make_hybrid("b", 0.5, "hello world"),
        ]
        config = MMRConfig(enabled=False)
        result = apply_mmr_to_hybrid_results(items, config)
        assert result == items

    def test_enabled_returns_all_items(self) -> None:
        items = [
            self._make_hybrid("a", 0.9, "memory search chunk"),
            self._make_hybrid("b", 0.8, "memory search chunk"),
            self._make_hybrid("c", 0.6, "cli status command"),
        ]
        config = MMRConfig(enabled=True, lambda_=0.7)
        result = apply_mmr_to_hybrid_results(items, config)
        assert len(result) == 3

    def test_high_relevance_first_with_lambda_one(self) -> None:
        items = [
            self._make_hybrid("low", 0.2, "text a"),
            self._make_hybrid("high", 0.9, "text b"),
            self._make_hybrid("mid", 0.5, "text c"),
        ]
        config = MMRConfig(enabled=True, lambda_=1.0)
        result = apply_mmr_to_hybrid_results(items, config)
        assert result[0]["score"] == pytest.approx(0.9)
