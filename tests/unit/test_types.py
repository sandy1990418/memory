"""Tests for core/types.py domain types."""

import unittest

from openclaw_memory.core.types import (
    ConsolidationReport,
    ExtractedMemory,
    MemoryContext,
    MemoryIndex,
    MemorySearchResult,
)


class TestMemoryIndex(unittest.TestCase):
    def test_creation(self):
        idx = MemoryIndex(
            id="abc-123", title="User likes sushi", memory_type="preference",
            score=0.95, created_at="2026-01-01", source="episodic",
        )
        self.assertEqual(idx.id, "abc-123")
        self.assertEqual(idx.title, "User likes sushi")
        self.assertAlmostEqual(idx.score, 0.95)


class TestMemoryContext(unittest.TestCase):
    def test_with_neighbors(self):
        ctx = MemoryContext(
            id="abc-123", content="Full content here",
            memory_type="fact", score=1.0,
            created_at="2026-01-01", source="semantic",
            neighbors=[{"id": "def-456", "title": "Nearby memory", "created_at": "2026-01-02"}],
        )
        self.assertEqual(len(ctx.neighbors), 1)
        self.assertEqual(ctx.neighbors[0]["id"], "def-456")

    def test_default_neighbors_empty(self):
        ctx = MemoryContext(
            id="x", content="y", memory_type="fact",
            score=1.0, created_at="", source="",
        )
        self.assertEqual(ctx.neighbors, [])


class TestMemorySearchResult(unittest.TestCase):
    def test_with_metadata(self):
        r = MemorySearchResult(
            id="abc", content="content", memory_type="fact",
            score=0.8, created_at="2026-01-01", source="episodic",
            metadata={"language": "zh"},
        )
        self.assertEqual(r.metadata["language"], "zh")

    def test_default_metadata_empty(self):
        r = MemorySearchResult(
            id="x", content="y", memory_type="fact",
            score=0.5, created_at="", source="",
        )
        self.assertEqual(r.metadata, {})


class TestExtractedMemory(unittest.TestCase):
    def test_full_creation(self):
        mem = ExtractedMemory(
            content="User prefers dark mode",
            memory_type="preference",
            confidence=0.95,
            metadata={"source": "chat"},
            memory_key="profile.theme",
            value="dark",
            event_time="2026-01-01T10:00:00Z",
            source_refs=["msg-1", "msg-2"],
        )
        self.assertEqual(mem.memory_key, "profile.theme")
        self.assertEqual(len(mem.source_refs), 2)

    def test_defaults(self):
        mem = ExtractedMemory(content="test", memory_type="fact", confidence=0.8)
        self.assertEqual(mem.memory_key, "")
        self.assertEqual(mem.value, "")
        self.assertIsNone(mem.event_time)
        self.assertEqual(mem.source_refs, [])
        self.assertEqual(mem.metadata, {})


class TestConsolidationReport(unittest.TestCase):
    def test_defaults(self):
        r = ConsolidationReport(user_id="user-1")
        self.assertEqual(r.memories_scanned, 0)
        self.assertEqual(r.memories_merged, 0)
        self.assertEqual(r.duration_seconds, 0.0)
        self.assertEqual(r.errors, [])


if __name__ == "__main__":
    unittest.main()
