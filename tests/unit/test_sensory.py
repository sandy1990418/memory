"""Tests for pipeline/ingest/sensory.py — LightMem Stage 1."""

import unittest

from openclaw_memory.pipeline.ingest.sensory import (
    SensoryConfig,
    build_session_summary,
    compress_text,
    prepare_for_extraction,
)


class TestCompressText(unittest.TestCase):
    def test_short_text_unchanged(self):
        text = "Hello world"
        result = compress_text(text, 100)
        self.assertEqual(result, text)

    def test_zero_limit(self):
        self.assertEqual(compress_text("Hello", 0), "")

    def test_long_text_compressed(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = compress_text(text, 50)
        self.assertLessEqual(len(result), 50 + 10)  # allow small overshoot
        self.assertTrue(len(result) > 0)

    def test_preserves_first_and_last(self):
        text = "Start here. Middle part. Another middle. End here."
        result = compress_text(text, 80)
        self.assertIn("Start", result)
        self.assertIn("End", result)


class TestPrepareForExtraction(unittest.TestCase):
    def test_empty_conversation(self):
        result = prepare_for_extraction([])
        self.assertEqual(result, [])

    def test_filters_ack_messages(self):
        conversation = [
            {"role": "user", "content": "I like pizza"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "My favorite color is blue"},
        ]
        result = prepare_for_extraction(conversation)
        # "ok" should be filtered out
        contents = [m["content"] for m in result]
        for c in contents:
            self.assertNotEqual(c.strip().lower(), "ok")

    def test_respects_token_budget(self):
        conversation = [
            {"role": "user", "content": "A" * 10000},
            {"role": "assistant", "content": "B" * 10000},
        ]
        cfg = SensoryConfig(max_input_tokens=100)
        result = prepare_for_extraction(conversation, cfg)
        total_chars = sum(len(m["content"]) for m in result)
        self.assertLess(total_chars, 10000)

    def test_user_only_mode(self):
        conversation = [
            {"role": "user", "content": "I like sushi"},
            {"role": "assistant", "content": "That's great!"},
        ]
        cfg = SensoryConfig(messages_use="user_only")
        result = prepare_for_extraction(conversation, cfg)
        roles = {m["role"] for m in result}
        self.assertNotIn("assistant", roles)


class TestBuildSessionSummary(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(build_session_summary([]), "")

    def test_produces_compact_output(self):
        conversation = [
            {"role": "user", "content": "I live in Tokyo"},
            {"role": "assistant", "content": "That's a great city!"},
        ]
        result = build_session_summary(conversation, max_chars=500)
        self.assertTrue(len(result) > 0)
        self.assertLessEqual(len(result), 500 + 50)


class TestSensoryConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SensoryConfig()
        self.assertTrue(cfg.pre_compress)
        self.assertEqual(cfg.messages_use, "all")
        self.assertTrue(cfg.topic_segment)
        self.assertEqual(cfg.max_input_tokens, 4096)

    def test_custom(self):
        cfg = SensoryConfig(pre_compress=False, max_input_tokens=512)
        self.assertFalse(cfg.pre_compress)
        self.assertEqual(cfg.max_input_tokens, 512)


if __name__ == "__main__":
    unittest.main()
