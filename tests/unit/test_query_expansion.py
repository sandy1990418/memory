"""Unit tests for openclaw_memory.query_expansion."""
from __future__ import annotations

import pytest

from openclaw_memory.query_expansion import extract_keywords


class TestExtractKeywords:
    def test_basic_query(self) -> None:
        keywords = extract_keywords("search memory files")
        assert "search" in keywords
        assert "memory" in keywords
        assert "files" in keywords

    def test_stop_words_removed(self) -> None:
        keywords = extract_keywords("find the memory file")
        assert "the" not in keywords
        assert "find" not in keywords  # "find" is a request word stop word
        assert "memory" in keywords

    def test_deduplication(self) -> None:
        keywords = extract_keywords("python python python")
        assert keywords.count("python") == 1

    def test_empty_string(self) -> None:
        keywords = extract_keywords("")
        assert keywords == []

    def test_whitespace_only(self) -> None:
        keywords = extract_keywords("   ")
        assert keywords == []

    def test_punctuation_stripped(self) -> None:
        keywords = extract_keywords("hello, world!")
        assert "hello" in keywords
        assert "world" in keywords
        assert "," not in keywords

    def test_short_words_filtered(self) -> None:
        # Pure alpha tokens shorter than 3 chars should be filtered
        keywords = extract_keywords("an is at to for")
        assert "an" not in keywords
        assert "is" not in keywords

    def test_numbers_filtered(self) -> None:
        keywords = extract_keywords("version 123 release")
        assert "123" not in keywords
        assert "version" in keywords

    def test_preserves_order_no_duplicates(self) -> None:
        keywords = extract_keywords("chunking algorithm overlap chunking")
        seen = set()
        for kw in keywords:
            assert kw not in seen, f"Duplicate keyword: {kw}"
            seen.add(kw)

    def test_chinese_stop_words_removed(self) -> None:
        # Chinese stop words should be filtered
        keywords = extract_keywords("我们 的 memory search")
        assert "我们" not in keywords
        assert "的" not in keywords
        assert "memory" in keywords

    def test_meaningful_technical_terms(self) -> None:
        keywords = extract_keywords("sqlite database schema migration")
        assert "sqlite" in keywords
        assert "database" in keywords
        assert "schema" in keywords
        assert "migration" in keywords

    def test_request_phrase(self) -> None:
        # "show me the files about memory" → stop words removed
        keywords = extract_keywords("show me the files about memory")
        assert "show" not in keywords  # request word
        assert "me" not in keywords   # pronoun
        assert "the" not in keywords  # article
        assert "about" not in keywords  # preposition
        assert "memory" in keywords
        assert "files" in keywords
