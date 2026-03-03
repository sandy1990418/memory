"""
Shared helpers for LongMemEval benchmarks.

Provides CONFIG_MATRIX, LONGMEMEVAL_FILE, _load_dotenv, and build_lme_workspace
used by run_longmemeval_qa.py and other benchmark scripts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


LONGMEMEVAL_FILE = Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"


# ── .env loader (no external dependencies) ──────────────────────────────


def _load_dotenv(env_path: Path | None = None) -> None:
    """
    Load KEY=VALUE pairs from a .env file into os.environ.

    Only sets variables that are not already present in the environment.
    Supports simple KEY=value, quoted values, and # comments.
    """
    if env_path is None:
        candidate = Path(__file__).parent
        for _ in range(6):
            p = candidate / ".env"
            if p.is_file():
                env_path = p
                break
            parent = candidate.parent
            if parent == candidate:
                break
            candidate = parent

    if env_path is None or not env_path.is_file():
        return

    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, raw_value = line.partition("=")
            key = key.strip()
            raw_value = raw_value.strip()
            if (
                len(raw_value) >= 2
                and raw_value[0] in ('"', "'")
                and raw_value[-1] == raw_value[0]
            ):
                raw_value = raw_value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = raw_value


# ── Search configuration presets ─────────────────────────────────────────


CONFIG_MATRIX = {
    "fts_only": {"hybrid_enabled": True},
    "vector_only": {"hybrid_enabled": False},
    "hybrid": {"hybrid_enabled": True, "vector_weight": 0.7, "text_weight": 0.3},
    "hybrid_mmr": {
        "hybrid_enabled": True,
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "mmr_enabled": True,
        "mmr_lambda": 0.7,
    },
}


# ── LongMemEval workspace builder ───────────────────────────────────────


def session_to_markdown(session: list[dict], session_id: str, date: str) -> str:
    lines = [f"# Session: {session_id}", f"Date: {date}", ""]
    for turn in session:
        role = turn["role"].capitalize()
        content = turn["content"].strip()
        lines.append(f"**{role}:** {content}")
        lines.append("")
    return "\n".join(lines)


def build_lme_workspace(instance: dict, workspace_dir: str) -> dict[str, Any]:
    """Build a workspace directory with markdown session files for one LongMemEval instance."""
    memory_dir = os.path.join(workspace_dir, "memory")
    os.makedirs(memory_dir, exist_ok=True)
    sessions = instance["haystack_sessions"]
    session_ids = instance["haystack_session_ids"]
    dates = instance["haystack_dates"]
    answer_ids = set(instance["answer_session_ids"])
    evidence_files: list[dict] = []
    for i, (sess, sid, date) in enumerate(zip(sessions, session_ids, dates)):
        filename = f"session_{i:04d}.md"
        filepath = os.path.join(memory_dir, filename)
        md_content = session_to_markdown(sess, sid, date)
        Path(filepath).write_text(md_content, encoding="utf-8")
        if sid in answer_ids:
            evidence_files.append({"filename": f"memory/{filename}"})
    Path(os.path.join(workspace_dir, "MEMORY.md")).write_text(
        f"# LongMemEval: {instance['question_id']}\n", encoding="utf-8"
    )
    return {"evidence_files": evidence_files}
