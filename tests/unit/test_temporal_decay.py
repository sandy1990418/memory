"""Unit tests for openclaw_memory.temporal_decay."""
from __future__ import annotations

import math
import time

import pytest

from openclaw_memory.temporal_decay import (
    TemporalDecayConfig,
    apply_temporal_decay_to_results,
    apply_temporal_decay_to_score,
    calculate_temporal_decay_multiplier,
    to_decay_lambda,
)


# ---------------------------------------------------------------------------
# to_decay_lambda
# ---------------------------------------------------------------------------


class TestToDecayLambda:
    def test_positive_half_life(self) -> None:
        lam = to_decay_lambda(30.0)
        assert lam == pytest.approx(math.log(2) / 30.0)

    def test_zero_half_life_returns_zero(self) -> None:
        assert to_decay_lambda(0.0) == 0.0

    def test_negative_half_life_returns_zero(self) -> None:
        assert to_decay_lambda(-10.0) == 0.0

    def test_infinite_half_life_returns_zero(self) -> None:
        assert to_decay_lambda(float("inf")) == 0.0


# ---------------------------------------------------------------------------
# calculate_temporal_decay_multiplier
# ---------------------------------------------------------------------------


class TestCalculateTemporalDecayMultiplier:
    def test_age_zero_returns_one(self) -> None:
        multiplier = calculate_temporal_decay_multiplier(0.0, 30.0)
        assert multiplier == pytest.approx(1.0)

    def test_age_equals_half_life_returns_half(self) -> None:
        multiplier = calculate_temporal_decay_multiplier(30.0, 30.0)
        assert multiplier == pytest.approx(0.5, rel=1e-4)

    def test_recent_higher_than_old(self) -> None:
        m_recent = calculate_temporal_decay_multiplier(1.0, 30.0)
        m_old = calculate_temporal_decay_multiplier(90.0, 30.0)
        assert m_recent > m_old

    def test_negative_age_clamped(self) -> None:
        # Negative age should be treated as 0
        m = calculate_temporal_decay_multiplier(-5.0, 30.0)
        assert m == pytest.approx(1.0)

    def test_zero_lambda_returns_one(self) -> None:
        # half_life_days=0 → lambda=0 → multiplier=1
        m = calculate_temporal_decay_multiplier(100.0, 0.0)
        assert m == pytest.approx(1.0)

    def test_large_age_approaches_zero(self) -> None:
        m = calculate_temporal_decay_multiplier(1000.0, 1.0)
        assert m < 0.001


# ---------------------------------------------------------------------------
# apply_temporal_decay_to_score
# ---------------------------------------------------------------------------


class TestApplyTemporalDecayToScore:
    def test_score_scaled_by_multiplier(self) -> None:
        score = 0.8
        result = apply_temporal_decay_to_score(score, 30.0, 30.0)
        assert result == pytest.approx(0.8 * 0.5, rel=1e-4)

    def test_fresh_content_unchanged(self) -> None:
        result = apply_temporal_decay_to_score(0.9, 0.0, 30.0)
        assert result == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# apply_temporal_decay_to_results
# ---------------------------------------------------------------------------


class TestApplyTemporalDecayToResults:
    def _make_result(self, path: str, score: float, source: str = "memory") -> dict:
        return {"path": path, "score": score, "source": source}

    def test_disabled_passes_through(self) -> None:
        results = [self._make_result("MEMORY.md", 0.9)]
        config = TemporalDecayConfig(enabled=False)
        out = apply_temporal_decay_to_results(results, config=config)
        assert out[0]["score"] == pytest.approx(0.9)

    def test_evergreen_memory_not_decayed(self) -> None:
        # MEMORY.md is evergreen — no date, so not decayed
        results = [self._make_result("MEMORY.md", 0.9)]
        config = TemporalDecayConfig(enabled=True, half_life_days=30.0)
        out = apply_temporal_decay_to_results(results, config=config)
        # Evergreen returns unchanged
        assert out[0]["score"] == pytest.approx(0.9)

    def test_dated_memory_file_decayed(self) -> None:
        # Use a very old date so decay is significant
        results = [self._make_result("memory/2020-01-01.md", 0.9)]
        config = TemporalDecayConfig(enabled=True, half_life_days=30.0)
        now_ms = time.time() * 1000
        out = apply_temporal_decay_to_results(results, config=config, now_ms=now_ms)
        # Score should be much lower for a ~5 year old file
        assert out[0]["score"] < 0.9 * 0.01  # should be nearly zero

    def test_recent_date_file_barely_decayed(self) -> None:
        # Use today's date: age ≈ 0 days
        from datetime import datetime, timezone
        today = datetime.now(tz=timezone.utc)
        path = f"memory/{today.year}-{today.month:02d}-{today.day:02d}.md"
        results = [self._make_result(path, 0.9)]
        config = TemporalDecayConfig(enabled=True, half_life_days=30.0)
        now_ms = time.time() * 1000
        out = apply_temporal_decay_to_results(results, config=config, now_ms=now_ms)
        # Nearly no decay for today's file
        assert out[0]["score"] > 0.85

    def test_empty_results(self) -> None:
        config = TemporalDecayConfig(enabled=True)
        out = apply_temporal_decay_to_results([], config=config)
        assert out == []

    def test_multiple_results_ordering_preserved(self) -> None:
        results = [
            self._make_result("memory/2024-01-01.md", 0.9),
            self._make_result("memory/2024-06-01.md", 0.7),
        ]
        config = TemporalDecayConfig(enabled=True, half_life_days=30.0)
        now_ms = time.time() * 1000
        out = apply_temporal_decay_to_results(results, config=config, now_ms=now_ms)
        assert len(out) == 2
        # Older file should have lower absolute score after decay
        assert out[0]["score"] < out[1]["score"] or out[0]["score"] < 0.001
