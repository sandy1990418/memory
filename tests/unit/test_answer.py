"""Tests for pipeline/retrieval/answer.py — RAG answer generation."""

import json
import unittest

from openclaw_memory.pipeline.retrieval.answer import (
    AnswerPayload,
    build_answer_prompt,
    generate_answer,
    parse_answer_response,
    validate_answer_payload,
)


class TestBuildAnswerPrompt(unittest.TestCase):
    def test_basic(self):
        prompt = build_answer_prompt("What food do I like?", [
            {"id": "m1", "snippet": "User likes sushi", "score": 0.9},
        ])
        self.assertIn("What food do I like?", prompt)
        self.assertIn("sushi", prompt)
        self.assertIn("m1", prompt)

    def test_empty_evidence(self):
        prompt = build_answer_prompt("test", [])
        self.assertIn("no evidence available", prompt)

    def test_retry_uses_different_template(self):
        prompt_normal = build_answer_prompt("test", [{"id": "x", "snippet": "y", "score": 0.5}])
        prompt_retry = build_answer_prompt("test", [{"id": "x", "snippet": "y", "score": 0.5}], retry=True)
        self.assertNotEqual(prompt_normal, prompt_retry)
        self.assertIn("Re-evaluate", prompt_retry)


class TestValidateAnswerPayload(unittest.TestCase):
    def test_valid(self):
        data = {
            "answer": "Tokyo",
            "evidence": [{"memory_id": "m1", "quote": "lives in Tokyo", "reason": "direct"}],
            "confidence": 0.9,
            "abstain": False,
        }
        result = validate_answer_payload(data)
        self.assertEqual(result.answer, "Tokyo")
        self.assertFalse(result.abstain)
        self.assertEqual(len(result.evidence), 1)

    def test_missing_keys_raises(self):
        with self.assertRaises(ValueError):
            validate_answer_payload({"answer": "test"})

    def test_force_abstain_on_empty_evidence(self):
        data = {
            "answer": "test",
            "evidence": [],
            "confidence": 0.5,
            "abstain": False,
        }
        result = validate_answer_payload(data)
        self.assertTrue(result.abstain)

    def test_fills_default_abstain_reason(self):
        data = {
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "abstain": True,
        }
        result = validate_answer_payload(data)
        self.assertTrue(len(result.abstain_reason) > 0)

    def test_confidence_clamped(self):
        data = {
            "answer": "test",
            "evidence": [{"memory_id": "m", "quote": "q", "reason": "r"}],
            "confidence": 5.0,
            "abstain": False,
        }
        result = validate_answer_payload(data)
        self.assertLessEqual(result.confidence, 1.0)


class TestParseAnswerResponse(unittest.TestCase):
    def test_valid_json(self):
        response = json.dumps({
            "answer": "sushi",
            "evidence": [{"memory_id": "m1", "quote": "likes sushi", "reason": "pref"}],
            "confidence": 0.9,
            "abstain": False,
        })
        result = parse_answer_response(response)
        self.assertEqual(result.answer, "sushi")
        self.assertFalse(result.abstain)

    def test_with_markdown_fences(self):
        inner = json.dumps({
            "answer": "test", "evidence": [], "confidence": 0.0, "abstain": True,
            "abstain_reason": "no data",
        })
        response = f"```json\n{inner}\n```"
        result = parse_answer_response(response)
        self.assertTrue(result.abstain)

    def test_invalid_json(self):
        result = parse_answer_response("not json at all")
        self.assertTrue(result.abstain)
        self.assertIn("parse", result.abstain_reason.lower())

    def test_empty_response(self):
        result = parse_answer_response("")
        self.assertTrue(result.abstain)

    def test_extracts_embedded_json(self):
        response = 'Here is my answer: {"answer": "test", "evidence": [], "confidence": 0.5, "abstain": true, "abstain_reason": "no info"}'
        result = parse_answer_response(response)
        self.assertTrue(result.abstain)


class TestGenerateAnswer(unittest.TestCase):
    def test_successful_answer(self):
        evidence = [{"id": "m1", "snippet": "User likes sushi", "score": 0.9}]
        response = json.dumps({
            "answer": "sushi",
            "evidence": [{"memory_id": "m1", "quote": "likes sushi", "reason": "preference"}],
            "confidence": 0.9,
            "abstain": False,
        })
        result = generate_answer("favorite food?", evidence, llm_fn=lambda _: response)
        self.assertEqual(result.answer, "sushi")
        self.assertFalse(result.abstain)

    def test_retry_on_abstain_with_evidence(self):
        evidence = [{"id": "m1", "snippet": "User likes sushi", "score": 0.9}]
        call_count = [0]

        def llm_fn(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({
                    "answer": "", "evidence": [], "confidence": 0.0,
                    "abstain": True, "abstain_reason": "not sure",
                })
            return json.dumps({
                "answer": "sushi",
                "evidence": [{"memory_id": "m1", "quote": "likes sushi", "reason": "found it"}],
                "confidence": 0.85, "abstain": False,
            })

        result = generate_answer("favorite food?", evidence, llm_fn=llm_fn)
        self.assertEqual(call_count[0], 2)  # Should have retried

    def test_llm_failure_returns_abstain(self):
        result = generate_answer(
            "test", [{"id": "m1", "snippet": "data", "score": 0.5}],
            llm_fn=lambda _: (_ for _ in ()).throw(RuntimeError("fail")),
        )
        self.assertTrue(result.abstain)

    def test_no_evidence_no_retry(self):
        call_count = [0]

        def llm_fn(prompt):
            call_count[0] += 1
            return json.dumps({
                "answer": "", "evidence": [], "confidence": 0.0,
                "abstain": True, "abstain_reason": "no data",
            })

        result = generate_answer("test", [], llm_fn=llm_fn)
        # Empty evidence -> should not call LLM at all (nothing to search)
        self.assertTrue(result.abstain)


if __name__ == "__main__":
    unittest.main()
