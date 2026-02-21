"""
File walking, chunking, hashing, and cosine similarity utilities.
Mirrors: src/memory/internal.ts
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Generator

from .types import MemoryChunk, MemoryFileEntry


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def normalize_rel_path(value: str) -> str:
    trimmed = value.strip().lstrip("./")
    return trimmed.replace("\\", "/")


def is_memory_path(rel_path: str) -> bool:
    normalized = normalize_rel_path(rel_path)
    if not normalized:
        return False
    if normalized in ("MEMORY.md", "memory.md"):
        return True
    return normalized.startswith("memory/")


def normalize_extra_memory_paths(workspace_dir: str, extra_paths: list[str] | None) -> list[str]:
    if not extra_paths:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for raw in extra_paths:
        trimmed = raw.strip()
        if not trimmed:
            continue
        if os.path.isabs(trimmed):
            resolved = os.path.realpath(trimmed)
        else:
            resolved = os.path.realpath(os.path.join(workspace_dir, trimmed))
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def ensure_dir(directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _is_symlink(path: str) -> bool:
    try:
        return os.path.islink(path)
    except OSError:
        return False


def _walk_dir(directory: str) -> Generator[str, None, None]:
    """Recursively yield .md file paths, skipping symlinks."""
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    yield from _walk_dir(entry.path)
                elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".md"):
                    yield entry.path
    except OSError:
        return


def list_memory_files(workspace_dir: str, extra_paths: list[str] | None = None) -> list[str]:
    """
    Return all memory .md files for a workspace, in a stable order.
    Mirrors: internal.ts::listMemoryFiles
    """
    result: list[str] = []

    def add_md_file(abs_path: str) -> None:
        if _is_symlink(abs_path):
            return
        if not os.path.isfile(abs_path):
            return
        if not abs_path.endswith(".md"):
            return
        result.append(abs_path)

    memory_file = os.path.join(workspace_dir, "MEMORY.md")
    alt_memory_file = os.path.join(workspace_dir, "memory.md")
    memory_dir = os.path.join(workspace_dir, "memory")

    add_md_file(memory_file)
    add_md_file(alt_memory_file)

    if os.path.isdir(memory_dir) and not _is_symlink(memory_dir):
        result.extend(_walk_dir(memory_dir))

    normalized_extra = normalize_extra_memory_paths(workspace_dir, extra_paths)
    for input_path in normalized_extra:
        if _is_symlink(input_path):
            continue
        if os.path.isdir(input_path):
            result.extend(_walk_dir(input_path))
        elif os.path.isfile(input_path) and input_path.endswith(".md"):
            result.append(input_path)

    # Deduplicate by realpath
    seen: set[str] = set()
    deduped: list[str] = []
    for entry in result:
        try:
            key = os.path.realpath(entry)
        except OSError:
            key = entry
        if key not in seen:
            seen.add(key)
            deduped.append(entry)

    return deduped


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_chunk_id(path: str, start_line: int, end_line: int, chunk_hash: str) -> str:
    """Derive a stable 16-hex-char chunk ID."""
    raw = f"{path}:{start_line}:{end_line}:{chunk_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# File entry builder
# ---------------------------------------------------------------------------


def build_file_entry(abs_path: str, workspace_dir: str) -> MemoryFileEntry:
    st = os.stat(abs_path)
    content = Path(abs_path).read_text(encoding="utf-8")
    file_hash = hash_text(content)
    rel = os.path.relpath(abs_path, workspace_dir).replace("\\", "/")
    return MemoryFileEntry(
        path=rel,
        abs_path=abs_path,
        mtime_ms=st.st_mtime * 1000,
        size=st.st_size,
        hash=file_hash,
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_markdown(
    content: str,
    chunking: tuple[int, int],  # (tokens, overlap)
) -> list[MemoryChunk]:
    """
    Chunk markdown text into overlapping windows.
    Mirrors: internal.ts::chunkMarkdown
    chunk_tokens * 4 ≈ chars (rough approximation).
    """
    chunk_tokens, overlap_tokens = chunking
    max_chars = max(32, chunk_tokens * 4)
    overlap_chars = max(0, overlap_tokens * 4)

    lines = content.split("\n")
    if not lines:
        return []

    chunks: list[MemoryChunk] = []
    # current = list of (line_text, line_no_1indexed)
    current: list[tuple[str, int]] = []
    current_chars = 0

    def flush() -> None:
        if not current:
            return
        text = "\n".join(l for l, _ in current)
        start_line = current[0][1]
        end_line = current[-1][1]
        chunks.append(MemoryChunk(
            start_line=start_line,
            end_line=end_line,
            text=text,
            hash=hash_text(text),
        ))

    def carry_overlap() -> tuple[list[tuple[str, int]], int]:
        if overlap_chars <= 0 or not current:
            return [], 0
        acc = 0
        kept: list[tuple[str, int]] = []
        for line, lineno in reversed(current):
            acc += len(line) + 1
            kept.insert(0, (line, lineno))
            if acc >= overlap_chars:
                break
        new_chars = sum(len(l) + 1 for l, _ in kept)
        return kept, new_chars

    for i, line in enumerate(lines):
        line_no = i + 1
        # Split long lines into segments
        segments = [line[j: j + max_chars] for j in range(0, max(1, len(line)), max_chars)] if line else [""]

        for segment in segments:
            line_size = len(segment) + 1
            if current_chars + line_size > max_chars and current:
                flush()
                current, current_chars = carry_overlap()
            current.append((segment, line_no))
            current_chars += line_size

    flush()
    return chunks


def remap_chunk_lines(chunks: list[MemoryChunk], line_map: list[int] | None) -> None:
    """Remap chunk start/end lines from content-relative to source positions."""
    if not line_map:
        return
    for chunk in chunks:
        idx_start = chunk.start_line - 1
        idx_end = chunk.end_line - 1
        if 0 <= idx_start < len(line_map):
            chunk.start_line = line_map[idx_start]
        if 0 <= idx_end < len(line_map):
            chunk.end_line = line_map[idx_end]


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


def parse_embedding(raw: str) -> list[float]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(length):
        av = a[i]
        bv = b[i]
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def truncate_utf16_safe(text: str, max_chars: int) -> str:
    """Truncate string safely (simple char truncation for Python)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
