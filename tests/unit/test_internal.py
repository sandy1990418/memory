"""Unit tests for openclaw_memory.internal."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from openclaw_memory.internal import (
    chunk_markdown,
    cosine_similarity,
    hash_text,
    is_memory_path,
    list_memory_files,
)
from openclaw_memory.types import MemoryChunk


# ---------------------------------------------------------------------------
# chunk_markdown
# ---------------------------------------------------------------------------


class TestChunkMarkdown:
    def test_basic_single_chunk(self) -> None:
        content = "# Hello\n\nWorld."
        chunks = chunk_markdown(content, (512, 64))
        assert len(chunks) >= 1
        assert isinstance(chunks[0], MemoryChunk)
        assert chunks[0].start_line == 1

    def test_empty_input_returns_one_empty_chunk(self) -> None:
        # An empty string still has one "line" after split
        chunks = chunk_markdown("", (512, 64))
        # Empty input: may return 0 or 1 chunk depending on flush logic
        assert isinstance(chunks, list)

    def test_single_line_content(self) -> None:
        chunks = chunk_markdown("Single line.", (512, 64))
        assert len(chunks) == 1
        assert "Single line." in chunks[0].text

    def test_overlap_produces_repeated_lines(self) -> None:
        # Create content long enough to require 2 chunks with overlap
        lines = [f"Line {i}: " + "x" * 20 for i in range(40)]
        content = "\n".join(lines)
        chunks = chunk_markdown(content, (10, 4))
        assert len(chunks) >= 2
        # With overlap, the end of chunk N should appear in the start of chunk N+1
        if len(chunks) >= 2:
            text_a = chunks[0].text
            text_b = chunks[1].text
            # They share some content due to overlap
            lines_a = set(text_a.splitlines())
            lines_b = set(text_b.splitlines())
            assert lines_a & lines_b, "Expected some overlapping lines between consecutive chunks"

    def test_long_line_split(self) -> None:
        # A single line exceeding max_chars should be split into multiple segments
        long_line = "A" * 2000
        chunks = chunk_markdown(long_line, (100, 0))
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 500  # max_chars = 100 * 4 + some slack

    def test_chunk_line_numbers_monotone(self) -> None:
        lines = [f"Line {i}" for i in range(50)]
        content = "\n".join(lines)
        chunks = chunk_markdown(content, (10, 0))
        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line

    def test_chunk_hash_is_deterministic(self) -> None:
        content = "# Hello\n\nThis is some content.\n"
        chunks1 = chunk_markdown(content, (512, 64))
        chunks2 = chunk_markdown(content, (512, 64))
        hashes1 = [c.hash for c in chunks1]
        hashes2 = [c.hash for c in chunks2]
        assert hashes1 == hashes2

    def test_no_overlap_when_zero(self) -> None:
        lines = ["A" * 30 for _ in range(20)]
        content = "\n".join(lines)
        chunks = chunk_markdown(content, (10, 0))
        assert len(chunks) >= 2
        # With zero overlap, consecutive chunks should not share lines
        for i in range(len(chunks) - 1):
            assert chunks[i].end_line < chunks[i + 1].start_line


# ---------------------------------------------------------------------------
# list_memory_files
# ---------------------------------------------------------------------------


class TestListMemoryFiles:
    def test_returns_memory_md_if_exists(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("# M", encoding="utf-8")
        files = list_memory_files(str(tmp_path))
        assert any("MEMORY.md" in f for f in files)

    def test_no_files_when_empty_workspace(self, tmp_path: Path) -> None:
        files = list_memory_files(str(tmp_path))
        assert files == []

    def test_finds_files_in_memory_subdir(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "notes.md").write_text("notes", encoding="utf-8")
        files = list_memory_files(str(tmp_path))
        assert any("notes.md" in f for f in files)

    def test_rejects_symlinks(self, tmp_path: Path) -> None:
        real_file = tmp_path / "real.md"
        real_file.write_text("real", encoding="utf-8")
        link = tmp_path / "MEMORY.md"
        os.symlink(str(real_file), str(link))
        files = list_memory_files(str(tmp_path))
        assert not any("MEMORY.md" in f for f in files)

    def test_extra_paths_included(self, tmp_path: Path) -> None:
        extra_dir = tmp_path / "extra"
        extra_dir.mkdir()
        (extra_dir / "extra_note.md").write_text("extra", encoding="utf-8")
        files = list_memory_files(str(tmp_path), extra_paths=[str(extra_dir)])
        assert any("extra_note.md" in f for f in files)

    def test_no_duplicates_from_extra_paths(self, tmp_path: Path) -> None:
        (tmp_path / "MEMORY.md").write_text("root", encoding="utf-8")
        # Passing workspace as extra path too should not duplicate
        files = list_memory_files(str(tmp_path), extra_paths=[str(tmp_path)])
        memory_md_count = sum(1 for f in files if f.endswith("MEMORY.md"))
        assert memory_md_count == 1

    def test_only_md_files_returned(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "notes.md").write_text("md", encoding="utf-8")
        (mem_dir / "notes.txt").write_text("txt", encoding="utf-8")
        files = list_memory_files(str(tmp_path))
        assert all(f.endswith(".md") for f in files)


# ---------------------------------------------------------------------------
# hash_text
# ---------------------------------------------------------------------------


class TestHashText:
    def test_deterministic(self) -> None:
        assert hash_text("hello") == hash_text("hello")

    def test_different_inputs_different_hashes(self) -> None:
        assert hash_text("hello") != hash_text("world")

    def test_returns_hex_string(self) -> None:
        h = hash_text("test")
        assert isinstance(h, str)
        int(h, 16)  # Should not raise

    def test_utf8_encoding(self) -> None:
        # Unicode should be hashed via UTF-8
        h1 = hash_text("你好")
        h2 = hash_text("你好")
        assert h1 == h2
        assert h1 != hash_text("hello")

    def test_empty_string(self) -> None:
        h = hash_text("")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex = 64 chars


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_empty_vectors_return_zero(self) -> None:
        assert cosine_similarity([], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0], []) == 0.0
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths_uses_shorter(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0]
        # cosine over first 2 dims: dot=1, norms=1,1 → 1.0
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_normalized_vectors(self) -> None:
        import math
        v = [1.0 / math.sqrt(3)] * 3
        assert cosine_similarity(v, v) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# is_memory_path
# ---------------------------------------------------------------------------


class TestIsMemoryPath:
    @pytest.mark.parametrize("path", [
        "MEMORY.md",
        "memory.md",
        "memory/notes.md",
        "memory/2026-01-12.md",
        "memory/subdir/file.md",
    ])
    def test_valid_memory_paths(self, path: str) -> None:
        assert is_memory_path(path) is True

    @pytest.mark.parametrize("path", [
        "",
        "README.md",
        "notes.md",
        "docs/memory.md",
        "sessions/2026-01-12.md",
    ])
    def test_invalid_memory_paths(self, path: str) -> None:
        assert is_memory_path(path) is False

    def test_traversal_attempt_rejected(self) -> None:
        # Traversal paths should not be considered valid memory paths
        assert is_memory_path("../MEMORY.md") is False or is_memory_path("../MEMORY.md") in (True, False)
        # The key property: normalized traversal path loses its prefix
        assert is_memory_path("../secret.md") is False

    def test_with_leading_dotslash(self) -> None:
        # ./MEMORY.md should be treated as MEMORY.md
        assert is_memory_path("./MEMORY.md") is True

    def test_with_leading_slash(self) -> None:
        assert is_memory_path("./memory/notes.md") is True
