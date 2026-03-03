"""Tests for dual-LLM support in MemoryService."""

import json
import unittest
from unittest.mock import MagicMock, patch

from openclaw_memory.config import AppSettings
from openclaw_memory.core.service import MemoryService


def _make_mock_emb():
    emb = MagicMock()
    emb.embed_query.return_value = [0.1] * 1536
    emb.dimensions = 1536
    return emb


class TestDualLLMInit(unittest.TestCase):
    """Test that MemoryService correctly assigns separate LLM functions."""

    def test_single_llm_fn_used_for_both(self):
        fn = MagicMock()
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
            llm_fn=fn,
        )
        self.assertIs(svc._extraction_llm_fn, fn)
        self.assertIs(svc._answer_llm_fn, fn)

    def test_separate_llm_fns(self):
        extract_fn = MagicMock(name="extract")
        answer_fn = MagicMock(name="answer")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
            extraction_llm_fn=extract_fn,
            answer_llm_fn=answer_fn,
        )
        self.assertIs(svc._extraction_llm_fn, extract_fn)
        self.assertIs(svc._answer_llm_fn, answer_fn)

    def test_partial_override_extraction_only(self):
        default_fn = MagicMock(name="default")
        extract_fn = MagicMock(name="extract")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
            llm_fn=default_fn,
            extraction_llm_fn=extract_fn,
        )
        self.assertIs(svc._extraction_llm_fn, extract_fn)
        self.assertIs(svc._answer_llm_fn, default_fn)

    def test_partial_override_answer_only(self):
        default_fn = MagicMock(name="default")
        answer_fn = MagicMock(name="answer")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
            llm_fn=default_fn,
            answer_llm_fn=answer_fn,
        )
        self.assertIs(svc._extraction_llm_fn, default_fn)
        self.assertIs(svc._answer_llm_fn, answer_fn)

    def test_no_llm_fn_at_all(self):
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
        )
        self.assertIsNone(svc._extraction_llm_fn)
        self.assertIsNone(svc._answer_llm_fn)


class TestDualLLMTrackers(unittest.TestCase):
    """Test that tracked LLM helpers use the correct model config."""

    def test_extraction_tracker_uses_extraction_model(self):
        settings = AppSettings(extraction_llm_model="gpt-4o", llm_model="gpt-4o-mini")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=settings,
            llm_fn=lambda p: "ok",
        )
        tracker = svc._tracked_extraction_llm()
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.model, "gpt-4o")

    def test_answer_tracker_uses_answer_model(self):
        settings = AppSettings(answer_llm_model="gpt-4o", llm_model="gpt-4o-mini")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=settings,
            llm_fn=lambda p: "ok",
        )
        tracker = svc._tracked_answer_llm()
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.model, "gpt-4o")

    def test_tracker_falls_back_to_llm_model(self):
        settings = AppSettings(llm_model="gpt-4o-mini")
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=settings,
            llm_fn=lambda p: "ok",
        )
        ext_tracker = svc._tracked_extraction_llm()
        ans_tracker = svc._tracked_answer_llm()
        self.assertEqual(ext_tracker.model, "gpt-4o-mini")
        self.assertEqual(ans_tracker.model, "gpt-4o-mini")

    def test_no_llm_returns_none(self):
        svc = MemoryService(
            embedding_provider=_make_mock_emb(),
            settings=AppSettings(),
        )
        self.assertIsNone(svc._tracked_extraction_llm())
        self.assertIsNone(svc._tracked_answer_llm())


class TestDualLLMConfig(unittest.TestCase):
    """Test config settings for dual LLM models."""

    def test_default_models_empty(self):
        settings = AppSettings()
        self.assertEqual(settings.extraction_llm_model, "")
        self.assertEqual(settings.answer_llm_model, "")

    def test_custom_models(self):
        settings = AppSettings(
            extraction_llm_model="gpt-4o-mini",
            answer_llm_model="gpt-4o",
        )
        self.assertEqual(settings.extraction_llm_model, "gpt-4o-mini")
        self.assertEqual(settings.answer_llm_model, "gpt-4o")


if __name__ == "__main__":
    unittest.main()
