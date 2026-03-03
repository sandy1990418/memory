"""Tests for Pydantic request/response models."""

import unittest

from pydantic import ValidationError


class TestSessionModels(unittest.TestCase):
    def test_start_request(self):
        from openclaw_memory.models.session import SessionStartRequest

        req = SessionStartRequest(user_id="u1", session_id="s1")
        self.assertEqual(req.user_id, "u1")
        self.assertEqual(req.session_id, "s1")

    def test_start_request_requires_fields(self):
        from openclaw_memory.models.session import SessionStartRequest

        with self.assertRaises(ValidationError):
            SessionStartRequest()

    def test_message_request(self):
        from openclaw_memory.models.session import MessageRequest

        req = MessageRequest(user_id="u1", session_id="s1", role="user", content="Hello")
        self.assertEqual(req.role, "user")
        self.assertEqual(req.content, "Hello")

    def test_end_response(self):
        from openclaw_memory.models.session import SessionEndResponse

        resp = SessionEndResponse(user_id="u1", session_id="s1", extracted=3, stored=2)
        self.assertEqual(resp.extracted, 3)
        self.assertEqual(resp.stored, 2)


class TestSearchModels(unittest.TestCase):
    def test_search_request_defaults(self):
        from openclaw_memory.models.search import SearchRequest

        req = SearchRequest(user_id="u1", query="favorite food")
        self.assertEqual(req.limit, 20)

    def test_search_request_custom_limit(self):
        from openclaw_memory.models.search import SearchRequest

        req = SearchRequest(user_id="u1", query="test", limit=5)
        self.assertEqual(req.limit, 5)

    def test_detail_request_requires_ids(self):
        from openclaw_memory.models.search import DetailRequest

        with self.assertRaises(ValidationError):
            DetailRequest(memory_ids=[])

    def test_detail_request_valid(self):
        from openclaw_memory.models.search import DetailRequest

        req = DetailRequest(memory_ids=["abc-123", "def-456"])
        self.assertEqual(len(req.memory_ids), 2)

    def test_answer_request(self):
        from openclaw_memory.models.search import AnswerRequest

        req = AnswerRequest(user_id="u1", query="what is my name?")
        self.assertEqual(req.top_k, 6)
        self.assertIsNone(req.session_id)

    def test_answer_request_with_session(self):
        from openclaw_memory.models.search import AnswerRequest

        req = AnswerRequest(user_id="u1", query="test", session_id="s1", top_k=3)
        self.assertEqual(req.session_id, "s1")
        self.assertEqual(req.top_k, 3)

    def test_search_response(self):
        from openclaw_memory.models.search import SearchResponse

        resp = SearchResponse(results=[{"id": "1", "title": "test", "score": 0.9}])
        self.assertEqual(len(resp.results), 1)

    def test_timeline_response(self):
        from openclaw_memory.models.search import TimelineResponse

        resp = TimelineResponse(
            id="x", content="y", memory_type="fact",
            score=1.0, created_at="2026-01-01", source="episodic",
            neighbors=[],
        )
        self.assertEqual(resp.neighbors, [])

    def test_answer_response(self):
        from openclaw_memory.models.search import AnswerResponse

        resp = AnswerResponse(
            answer="Tokyo", confidence=0.9, abstain=False,
            evidence=[{"memory_id": "m1", "quote": "lives in Tokyo", "reason": "direct match"}],
        )
        self.assertEqual(resp.answer, "Tokyo")
        self.assertFalse(resp.abstain)


class TestMemoryModels(unittest.TestCase):
    def test_add_request_with_conversation(self):
        from openclaw_memory.models.memory import MemoryAddRequest

        req = MemoryAddRequest(
            user_id="u1",
            conversation=[{"role": "user", "content": "I like sushi"}],
        )
        self.assertIsNotNone(req.conversation)
        self.assertIsNone(req.content)

    def test_add_request_with_content(self):
        from openclaw_memory.models.memory import MemoryAddRequest

        req = MemoryAddRequest(user_id="u1", content="User likes sushi")
        self.assertIsNone(req.conversation)
        self.assertEqual(req.content, "User likes sushi")

    def test_add_response(self):
        from openclaw_memory.models.memory import MemoryAddResponse

        resp = MemoryAddResponse(stored=2, extracted=3, memory_id="abc")
        self.assertEqual(resp.stored, 2)
        self.assertEqual(resp.memory_id, "abc")

    def test_memory_item(self):
        from openclaw_memory.models.memory import MemoryItem

        item = MemoryItem(
            id="abc", content="likes sushi", memory_type="preference",
            created_at="2026-01-01", source="episodic",
        )
        self.assertEqual(item.metadata, {})

    def test_list_response(self):
        from openclaw_memory.models.memory import MemoryListResponse

        resp = MemoryListResponse(user_id="u1", memories=[], total=0)
        self.assertEqual(resp.user_id, "u1")

    def test_delete_response(self):
        from openclaw_memory.models.memory import MemoryDeleteResponse

        resp = MemoryDeleteResponse(deleted=True, memory_id="abc-123")
        self.assertTrue(resp.deleted)
        self.assertEqual(resp.memory_id, "abc-123")


if __name__ == "__main__":
    unittest.main()
