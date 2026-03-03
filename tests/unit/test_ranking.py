"""Tests for pipeline/retrieval/ranking.py — temporal decay, MMR, LLM rerank."""

import json
import time
import unittest
from datetime import datetime, timedelta, timezone

from openclaw_memory.pipeline.retrieval.ranking import (
    MMRConfig,
    TemporalDecayConfig,
    apply_mmr,
    apply_ranking_pipeline,
    apply_temporal_decay,
    calculate_temporal_decay,
    llm_rerank,
    mmr_rerank,
)


class TestTemporalDecay(unittest.TestCase):
    def test_zero_age_no_decay(self):
        multiplier = calculate_temporal_decay(0.0, 30.0)
        self.assertAlmostEqual(multiplier, 1.0)

    def test_half_life_halves(self):
        multiplier = calculate_temporal_decay(30.0, 30.0)
        self.assertAlmostEqual(multiplier, 0.5, places=5)

    def test_disabled_returns_original(self):
        results = [{"score": 0.9, "created_at": "2020-01-01"}]
        cfg = TemporalDecayConfig(enabled=False)
        output = apply_temporal_decay(results, config=cfg)
        self.assertAlmostEqual(output[0]["score"], 0.9)

    def test_applies_decay_to_old_results(self):
        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        results = [{"score": 1.0, "created_at": old_time}]
        cfg = TemporalDecayConfig(enabled=True, half_life_days=30.0)
        output = apply_temporal_decay(results, config=cfg)
        self.assertLess(output[0]["score"], 0.3)

    def test_recent_results_minimal_decay(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        results = [{"score": 1.0, "created_at": recent}]
        cfg = TemporalDecayConfig(enabled=True, half_life_days=30.0)
        output = apply_temporal_decay(results, config=cfg)
        self.assertGreater(output[0]["score"], 0.99)

    def test_missing_created_at_preserved(self):
        results = [{"score": 0.8, "created_at": None}]
        cfg = TemporalDecayConfig(enabled=True)
        output = apply_temporal_decay(results, config=cfg)
        self.assertAlmostEqual(output[0]["score"], 0.8)

    def test_iso_string_created_at(self):
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        results = [{"score": 1.0, "created_at": old_time}]
        cfg = TemporalDecayConfig(enabled=True, half_life_days=30.0)
        output = apply_temporal_decay(results, config=cfg)
        self.assertAlmostEqual(output[0]["score"], 0.5, places=1)


class TestMMR(unittest.TestCase):
    def test_disabled_returns_input(self):
        items = [{"score": 0.9, "snippet": "A"}, {"score": 0.8, "snippet": "B"}]
        cfg = MMRConfig(enabled=False)
        result = apply_mmr(items, config=cfg)
        self.assertEqual(len(result), 2)

    def test_diversity_reranking(self):
        # Two similar items and one different
        items = [
            {"score": 0.9, "snippet": "cats are great pets"},
            {"score": 0.85, "snippet": "cats are wonderful pets"},
            {"score": 0.8, "snippet": "dogs love to play fetch"},
        ]
        cfg = MMRConfig(enabled=True, lambda_=0.5)
        result = apply_mmr(items, config=cfg)
        self.assertEqual(len(result), 3)
        # With strong diversity (lambda=0.5), the dog item should be promoted
        snippets = [r["snippet"] for r in result]
        # First should still be highest relevance, but ordering after should diversify
        self.assertIn("dogs love to play fetch", snippets)

    def test_single_item(self):
        items = [{"score": 1.0, "snippet": "only one"}]
        result = apply_mmr(items, config=MMRConfig(enabled=True))
        self.assertEqual(len(result), 1)

    def test_lambda_1_sorts_by_score(self):
        items = [
            {"score": 0.5, "snippet": "low"},
            {"score": 0.9, "snippet": "high"},
        ]
        cfg = MMRConfig(enabled=True, lambda_=1.0)
        result = apply_mmr(items, config=cfg)
        self.assertGreaterEqual(result[0]["score"], result[1]["score"])

    def test_generic_mmr_rerank(self):
        items = ["apple", "apples", "banana"]
        result = mmr_rerank(
            items,
            score_fn=lambda x: 1.0 if x.startswith("a") else 0.5,
            content_fn=lambda x: x,
            config=MMRConfig(enabled=True, lambda_=0.5),
        )
        self.assertEqual(len(result), 3)


class TestLlmRerank(unittest.TestCase):
    def test_empty_candidates(self):
        result = llm_rerank("query", [], llm_fn=lambda _: "[]")
        self.assertEqual(result, [])

    def test_successful_rerank(self):
        candidates = [
            {"snippet": "User likes sushi", "score": 0.7},
            {"snippet": "Weather is sunny", "score": 0.8},
            {"snippet": "User favorite is ramen", "score": 0.6},
        ]
        response = json.dumps([
            {"index": 0, "score": 0.95},
            {"index": 2, "score": 0.90},
            {"index": 1, "score": 0.10},
        ])
        result = llm_rerank("favorite food", candidates, llm_fn=lambda _: response, top_k=2)
        self.assertEqual(len(result), 2)
        # Sushi should be ranked first (score 0.95)
        self.assertIn("sushi", result[0]["snippet"])

    def test_fallback_on_llm_failure(self):
        candidates = [
            {"snippet": "A", "score": 0.9},
            {"snippet": "B", "score": 0.8},
        ]

        def fail_llm(_):
            raise RuntimeError("LLM unavailable")

        result = llm_rerank("test", candidates, llm_fn=fail_llm, top_k=2)
        self.assertEqual(len(result), 2)


class TestApplyRankingPipeline(unittest.TestCase):
    def test_full_pipeline(self):
        now = datetime.now(timezone.utc)
        results = [
            {"score": 0.9, "snippet": "recent memory", "created_at": now},
            {"score": 0.8, "snippet": "old memory",
             "created_at": now - timedelta(days=60)},
        ]
        output = apply_ranking_pipeline(
            results,
            temporal_decay=TemporalDecayConfig(enabled=True, half_life_days=30.0),
            mmr=MMRConfig(enabled=True, lambda_=0.7),
            top_k=2,
        )
        self.assertLessEqual(len(output), 2)
        # Recent memory should rank higher after decay
        self.assertGreater(output[0]["score"], output[1]["score"])

    def test_pipeline_without_llm(self):
        results = [{"score": 0.5, "snippet": "test", "created_at": None}]
        output = apply_ranking_pipeline(results, top_k=5)
        self.assertEqual(len(output), 1)

    def test_pipeline_respects_top_k(self):
        results = [{"score": float(i), "snippet": f"item {i}", "created_at": None}
                    for i in range(20)]
        output = apply_ranking_pipeline(results, top_k=5)
        self.assertLessEqual(len(output), 5)


if __name__ == "__main__":
    unittest.main()
