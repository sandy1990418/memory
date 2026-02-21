"""Unit tests for openclaw_memory.hybrid."""
from __future__ import annotations

import pytest

from openclaw_memory.hybrid import (
    bm25_rank_to_score,
    build_fts_query,
    merge_hybrid_results,
)


# ---------------------------------------------------------------------------
# build_fts_query
# ---------------------------------------------------------------------------


class TestBuildFtsQuery:
    def test_basic_query(self) -> None:
        result = build_fts_query("hello world")
        assert result == '"hello" AND "world"'

    def test_single_token(self) -> None:
        result = build_fts_query("python")
        assert result == '"python"'

    def test_special_chars_stripped(self) -> None:
        # Special chars that are not word chars are stripped
        result = build_fts_query("hello! world?")
        assert result is not None
        assert "!" not in result
        assert "?" not in result

    def test_unbalanced_quotes_handled(self) -> None:
        # Unbalanced quotes should not break parsing
        result = build_fts_query('say "hello')
        assert result is not None
        assert '"' not in result.replace('"hello"', "").replace('"say"', "")

    def test_empty_string_returns_none(self) -> None:
        assert build_fts_query("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert build_fts_query("   ") is None

    def test_tokens_are_double_quoted(self) -> None:
        result = build_fts_query("foo bar")
        assert result is not None
        parts = result.split(" AND ")
        for part in parts:
            assert part.startswith('"')
            assert part.endswith('"')

    def test_numbers_included(self) -> None:
        result = build_fts_query("python 3")
        assert result is not None
        assert '"3"' in result or '"python"' in result


# ---------------------------------------------------------------------------
# bm25_rank_to_score
# ---------------------------------------------------------------------------


class TestBm25RankToScore:
    def test_zero_rank(self) -> None:
        score = bm25_rank_to_score(0.0)
        assert score == pytest.approx(1.0)

    def test_monotone_decreasing(self) -> None:
        scores = [bm25_rank_to_score(float(r)) for r in [0, 1, 5, 10, 100]]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], f"Expected monotone decreasing at index {i}"

    def test_negative_rank_clamped_to_zero(self) -> None:
        # Negative rank should not produce score > 1
        score = bm25_rank_to_score(-5.0)
        assert 0.0 <= score <= 1.0

    def test_large_rank_approaches_zero(self) -> None:
        score = bm25_rank_to_score(1_000_000.0)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_output_range(self) -> None:
        for rank in [0.0, 0.5, 1.0, 10.0, 100.0]:
            score = bm25_rank_to_score(rank)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# merge_hybrid_results
# ---------------------------------------------------------------------------


def _make_vector_result(id: str, path: str, score: float, snippet: str = "") -> dict:
    return {
        "id": id,
        "path": path,
        "start_line": 1,
        "end_line": 5,
        "source": "memory",
        "snippet": snippet or f"snippet for {id}",
        "vector_score": score,
    }


def _make_keyword_result(id: str, path: str, score: float, snippet: str = "") -> dict:
    return {
        "id": id,
        "path": path,
        "start_line": 1,
        "end_line": 5,
        "source": "memory",
        "snippet": snippet or f"snippet for {id}",
        "text_score": score,
    }


class TestMergeHybridResults:
    def test_vector_only(self) -> None:
        vector = [_make_vector_result("a", "MEMORY.md", 0.9)]
        results = merge_hybrid_results(
            vector=vector, keyword=[], vector_weight=0.7, text_weight=0.3
        )
        assert len(results) == 1
        assert results[0]["path"] == "MEMORY.md"
        assert results[0]["score"] == pytest.approx(0.9 * 0.7)

    def test_keyword_only(self) -> None:
        keyword = [_make_keyword_result("b", "memory/notes.md", 0.8)]
        results = merge_hybrid_results(
            vector=[], keyword=keyword, vector_weight=0.7, text_weight=0.3
        )
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.8 * 0.3)

    def test_union_of_both(self) -> None:
        vector = [_make_vector_result("a", "MEMORY.md", 0.9)]
        keyword = [_make_keyword_result("b", "memory/notes.md", 0.8)]
        results = merge_hybrid_results(
            vector=vector, keyword=keyword, vector_weight=0.7, text_weight=0.3
        )
        assert len(results) == 2

    def test_overlap_combines_scores(self) -> None:
        vector = [_make_vector_result("shared", "MEMORY.md", 0.8)]
        keyword = [_make_keyword_result("shared", "MEMORY.md", 0.6)]
        results = merge_hybrid_results(
            vector=vector, keyword=keyword, vector_weight=0.7, text_weight=0.3
        )
        assert len(results) == 1
        expected_score = 0.7 * 0.8 + 0.3 * 0.6
        assert results[0]["score"] == pytest.approx(expected_score)

    def test_sorted_by_score_descending(self) -> None:
        vector = [
            _make_vector_result("low", "a.md", 0.2),
            _make_vector_result("high", "b.md", 0.9),
            _make_vector_result("mid", "c.md", 0.5),
        ]
        results = merge_hybrid_results(
            vector=vector, keyword=[], vector_weight=1.0, text_weight=0.0
        )
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_weight_balance(self) -> None:
        # Equal weights
        vector = [_make_vector_result("x", "x.md", 1.0)]
        keyword = [_make_keyword_result("x", "x.md", 1.0)]
        results = merge_hybrid_results(
            vector=vector, keyword=keyword, vector_weight=0.5, text_weight=0.5
        )
        assert results[0]["score"] == pytest.approx(1.0)

    def test_empty_inputs(self) -> None:
        results = merge_hybrid_results(
            vector=[], keyword=[], vector_weight=0.7, text_weight=0.3
        )
        assert results == []
