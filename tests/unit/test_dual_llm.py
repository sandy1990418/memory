"""Tests for per-operation LLM support in MemoryService."""

import unittest
from unittest.mock import MagicMock

from openclaw_memory.config import AppSettings
from openclaw_memory.core.service import MemoryService


def _make_mock_emb():
    emb = MagicMock()
    emb.embed_query.return_value = [0.1] * 1536
    emb.dimensions = 1536
    return emb


class TestPerOperationLLMInit(unittest.TestCase):
    """Test that MemoryService correctly assigns per-operation LLM functions."""

    def test_single_llm_fn_used_as_default(self):
        fn = MagicMock()
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
            llm_fn=fn,
        )
        # All operations fall back to the default
        for op in MemoryService.LLM_OPERATIONS:
            self.assertIs(svc._get_llm(op), fn)

    def test_per_operation_overrides(self):
        default_fn = MagicMock(name="default")
        extract_fn = MagicMock(name="extract")
        answer_fn = MagicMock(name="answer")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
            llm_fn=default_fn,
            llm_fns={"extraction": extract_fn, "answer": answer_fn},
        )
        self.assertIs(svc._get_llm("extraction"), extract_fn)
        self.assertIs(svc._get_llm("answer"), answer_fn)
        # Others fall back to default
        self.assertIs(svc._get_llm("conflict"), default_fn)
        self.assertIs(svc._get_llm("rerank"), default_fn)
        self.assertIs(svc._get_llm("consolidation"), default_fn)
        self.assertIs(svc._get_llm("promotion"), default_fn)

    def test_all_six_operations_separate(self):
        fns = {op: MagicMock(name=op) for op in MemoryService.LLM_OPERATIONS}
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
            llm_fns=fns,
        )
        for op in MemoryService.LLM_OPERATIONS:
            self.assertIs(svc._get_llm(op), fns[op])

    def test_no_llm_fn_at_all(self):
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
        )
        for op in MemoryService.LLM_OPERATIONS:
            self.assertIsNone(svc._get_llm(op))

    def test_per_operation_without_default(self):
        extract_fn = MagicMock(name="extract")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
            llm_fns={"extraction": extract_fn},
        )
        self.assertIs(svc._get_llm("extraction"), extract_fn)
        self.assertIsNone(svc._get_llm("answer"))


class TestPerOperationTrackers(unittest.TestCase):
    """Test that _tracked_llm uses the correct model config per operation."""

    def test_extraction_tracker_uses_extraction_model(self):
        settings = AppSettings(extraction_llm_model="gpt-4o", llm_model="gpt-4o-mini")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=settings,
            llm_fn=lambda p: "ok",
        )
        tracker = svc._tracked_llm("extraction")
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.model, "gpt-4o")

    def test_answer_tracker_uses_answer_model(self):
        settings = AppSettings(answer_llm_model="gpt-4o", llm_model="gpt-4o-mini")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=settings,
            llm_fn=lambda p: "ok",
        )
        tracker = svc._tracked_llm("answer")
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.model, "gpt-4o")

    def test_conflict_tracker_uses_conflict_model(self):
        settings = AppSettings(conflict_llm_model="claude-sonnet-4-20250514", llm_model="gpt-4o-mini")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=settings,
            llm_fn=lambda p: "ok",
        )
        tracker = svc._tracked_llm("conflict")
        self.assertEqual(tracker.model, "claude-sonnet-4-20250514")

    def test_tracker_falls_back_to_llm_model(self):
        settings = AppSettings(llm_model="gpt-4o-mini")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=settings,
            llm_fn=lambda p: "ok",
        )
        for op in MemoryService.LLM_OPERATIONS:
            tracker = svc._tracked_llm(op)
            self.assertEqual(tracker.model, "gpt-4o-mini", f"Failed for {op}")

    def test_no_llm_returns_none(self):
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
        )
        for op in MemoryService.LLM_OPERATIONS:
            self.assertIsNone(svc._tracked_llm(op))


class TestPerOperationConfig(unittest.TestCase):
    """Test config settings for per-operation LLM models."""

    def test_all_default_empty(self):
        settings = AppSettings()
        for op in MemoryService.LLM_OPERATIONS:
            self.assertEqual(getattr(settings, f"{op}_llm_model"), "")

    def test_custom_models(self):
        settings = AppSettings(
            extraction_llm_model="gpt-4o-mini",
            conflict_llm_model="gpt-4o",
            rerank_llm_model="gpt-4o-mini",
            answer_llm_model="gpt-4o",
            consolidation_llm_model="claude-sonnet-4-20250514",
            promotion_llm_model="claude-sonnet-4-20250514",
        )
        self.assertEqual(settings.extraction_llm_model, "gpt-4o-mini")
        self.assertEqual(settings.conflict_llm_model, "gpt-4o")
        self.assertEqual(settings.rerank_llm_model, "gpt-4o-mini")
        self.assertEqual(settings.answer_llm_model, "gpt-4o")
        self.assertEqual(settings.consolidation_llm_model, "claude-sonnet-4-20250514")
        self.assertEqual(settings.promotion_llm_model, "claude-sonnet-4-20250514")


if __name__ == "__main__":
    unittest.main()
