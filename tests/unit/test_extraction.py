"""Tests for pipeline/ingest/extraction.py — LLM memory extraction."""

import json
import unittest

from openclaw_memory.pipeline.ingest.extraction import extract_memories


class TestExtractMemories(unittest.TestCase):
    def test_empty_conversation(self):
        result = extract_memories([], llm_fn=lambda _: "[]")
        self.assertEqual(result, [])

    def test_parses_valid_response(self):
        response = json.dumps([
            {
                "content": "User likes sushi",
                "memory_type": "preference",
                "confidence": 0.9,
                "memory_key": "food.favorite",
                "value": "sushi",
                "event_time": None,
                "source_refs": [],
            }
        ])
        conversation = [{"role": "user", "content": "I like sushi"}]
        result = extract_memories(conversation, llm_fn=lambda _: response)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "User likes sushi")
        self.assertEqual(result[0].memory_type, "preference")
        self.assertAlmostEqual(result[0].confidence, 0.9)
        self.assertEqual(result[0].memory_key, "food.favorite")

    def test_handles_invalid_json(self):
        result = extract_memories(
            [{"role": "user", "content": "test"}],
            llm_fn=lambda _: "not valid json",
        )
        self.assertEqual(result, [])

    def test_handles_empty_array(self):
        result = extract_memories(
            [{"role": "user", "content": "test"}],
            llm_fn=lambda _: "[]",
        )
        self.assertEqual(result, [])

    def test_strips_markdown_fences(self):
        response = '```json\n' + json.dumps([
            {"content": "test memory", "memory_type": "fact", "confidence": 0.8}
        ]) + '\n```'
        result = extract_memories(
            [{"role": "user", "content": "test"}],
            llm_fn=lambda _: response,
        )
        self.assertEqual(len(result), 1)

    def test_invalid_memory_type_falls_back_to_fact(self):
        response = json.dumps([
            {"content": "test", "memory_type": "unknown_type", "confidence": 0.8}
        ])
        result = extract_memories(
            [{"role": "user", "content": "test"}],
            llm_fn=lambda _: response,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].memory_type, "fact")

    def test_confidence_clamped(self):
        response = json.dumps([
            {"content": "test", "memory_type": "fact", "confidence": 5.0}
        ])
        result = extract_memories(
            [{"role": "user", "content": "test"}],
            llm_fn=lambda _: response,
        )
        self.assertLessEqual(result[0].confidence, 1.0)

    def test_normalizes_memory_key(self):
        response = json.dumps([
            {"content": "test", "memory_type": "fact", "confidence": 0.8,
             "memory_key": "Profile  Favorite-Food"}
        ])
        result = extract_memories(
            [{"role": "user", "content": "test"}],
            llm_fn=lambda _: response,
        )
        # Key should be lowered, spaces->underscores, hyphens->underscores
        key = result[0].memory_key
        self.assertNotIn(" ", key)
        self.assertNotIn("-", key)
        self.assertEqual(key, key.lower())

    def test_prompt_includes_conversation(self):
        captured_prompts = []

        def capture_llm(prompt: str) -> str:
            captured_prompts.append(prompt)
            return "[]"

        extract_memories(
            [{"role": "user", "content": "I live in Tokyo"}],
            llm_fn=capture_llm,
        )
        self.assertEqual(len(captured_prompts), 1)
        self.assertIn("Tokyo", captured_prompts[0])


if __name__ == "__main__":
    unittest.main()
