"""Tests for utils/llm.py — TokenTracker."""

import unittest

from openclaw_memory.utils.llm import TokenTracker


class TestTokenTracker(unittest.TestCase):
    def _echo_llm(self, prompt: str) -> str:
        return f"echo: {prompt[:20]}"

    def test_basic_tracking(self):
        tracker = TokenTracker(self._echo_llm, model="test-model", operation="test")
        result = tracker("Hello world, this is a test prompt")
        self.assertTrue(result.startswith("echo:"))
        self.assertEqual(tracker.call_count, 1)
        self.assertGreater(tracker.total_input_tokens, 0)
        self.assertGreater(tracker.total_output_tokens, 0)

    def test_multiple_calls(self):
        tracker = TokenTracker(self._echo_llm, operation="multi")
        tracker("prompt one")
        tracker("prompt two")
        tracker("prompt three")
        self.assertEqual(tracker.call_count, 3)
        self.assertEqual(len(tracker.calls), 3)

    def test_with_operation(self):
        tracker = TokenTracker(self._echo_llm, operation="extraction")
        sub = tracker.with_operation("conflict")
        sub("test prompt")
        # Shared call log
        self.assertEqual(tracker.call_count, 1)
        self.assertEqual(tracker.calls[0].operation, "conflict")

    def test_summary(self):
        tracker = TokenTracker(self._echo_llm, model="gpt-4o-mini", operation="answer")
        tracker("what is my name?")
        summary = tracker.summary()
        self.assertEqual(summary["model"], "gpt-4o-mini")
        self.assertEqual(summary["total_calls"], 1)
        self.assertIn("answer", summary["by_operation"])
        self.assertGreater(summary["total_tokens"], 0)

    def test_failed_call_records_usage(self):
        def fail_llm(prompt: str) -> str:
            raise RuntimeError("LLM down")

        tracker = TokenTracker(fail_llm, operation="test")
        with self.assertRaises(RuntimeError):
            tracker("prompt")
        self.assertEqual(tracker.call_count, 1)
        self.assertGreater(tracker.calls[0].input_tokens, 0)
        self.assertEqual(tracker.calls[0].output_tokens, 0)

    def test_reset(self):
        tracker = TokenTracker(self._echo_llm)
        tracker("test")
        tracker.reset()
        self.assertEqual(tracker.call_count, 0)
        self.assertEqual(tracker.total_tokens, 0)

    def test_duration_recorded(self):
        tracker = TokenTracker(self._echo_llm, operation="speed")
        tracker("test")
        self.assertGreaterEqual(tracker.calls[0].duration_ms, 0.0)

    def test_total_tokens(self):
        tracker = TokenTracker(self._echo_llm)
        tracker("hello")
        self.assertEqual(
            tracker.total_tokens,
            tracker.total_input_tokens + tracker.total_output_tokens,
        )


if __name__ == "__main__":
    unittest.main()
