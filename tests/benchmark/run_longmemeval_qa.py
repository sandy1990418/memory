"""
LongMemEval end-to-end QA benchmark for openclaw-memory.

Pipeline per instance:
  1) Build workspace with memory sessions
  2) Retrieve top-K chunks
  3) Ask LLM to answer using only retrieved memory
  4) Score with EM / F1 (and optional LLM judge)

Usage:
  python tests/benchmark/run_longmemeval_qa.py --limit 48 --config hybrid
  python tests/benchmark/run_longmemeval_qa.py --limit 100 --balanced
  python tests/benchmark/run_longmemeval_qa.py --judge longmemeval --judge-model gpt-4o-mini
  python tests/benchmark/run_longmemeval_qa.py --provider all --limit 48 --balanced
  python tests/benchmark/run_longmemeval_qa.py --update-report
  python tests/benchmark/run_longmemeval_qa.py --limit 48 --balanced --no-cache
  python tests/benchmark/run_longmemeval_qa.py --limit 48 --balanced --force-reindex
  python tests/benchmark/run_longmemeval_qa.py --pipeline service --service-write-mode distill --distill-batch-sessions 2 --search-k 10 --answer-top-k 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from collections import Counter
from pathlib import Path
from typing import Any

import openai

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

from helpers.embeddings_mock import MockEmbeddingProvider  # noqa: E402
from openclaw_memory.answer_contract import generate_answer  # noqa: E402
from openclaw_memory.config import resolve_memory_search_config  # noqa: E402
from openclaw_memory.embeddings import OpenAIEmbeddingProvider  # noqa: E402
from openclaw_memory.manager import MemoryIndexManager  # noqa: E402
from openclaw_memory.pg_schema import ensure_pg_schema, get_pg_connection  # noqa: E402
from openclaw_memory.service import MemoryService  # noqa: E402
from run_real_embedding_benchmark import (  # noqa: E402
    CONFIG_MATRIX,
    LONGMEMEVAL_FILE,
    _load_dotenv,
    build_lme_workspace,
)

PGVECTOR_DIMS = 1536
LIGHTMEM_BATCH_SYSTEM_PROMPT = "你是有幫助的助手"

_VALID_SERVICE_RESOLVER_MODES = frozenset({"off", "offline", "sync"})
_VALID_SERVICE_DRAIN_QUEUE_MODES = frozenset({"never", "after_ingest", "after_run"})


def _pad_or_truncate(vec: list[float], dims: int = PGVECTOR_DIMS) -> list[float]:
    if len(vec) == dims:
        return vec
    if len(vec) > dims:
        return vec[:dims]
    return vec + ([0.0] * (dims - len(vec)))


class _PaddedEmbeddingProvider:
    """Adapter to match provider output dimensionality to pgvector column dims."""

    def __init__(self, base: Any, dims: int = PGVECTOR_DIMS) -> None:
        self._base = base
        self._dims = dims
        self.id = f"{getattr(base, 'id', 'provider')}-pad{dims}"
        self.model = f"{getattr(base, 'model', 'model')}-pad{dims}"

    def embed_query(self, text: str) -> list[float]:
        return _pad_or_truncate(self._base.embed_query(text), self._dims)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            raw = self._base.embed_batch(texts)
        except Exception:
            raw = [self._base.embed_query(t) for t in texts]
        return [_pad_or_truncate(v, self._dims) for v in raw]


def _normalize_service_resolver_mode(mode: str, service_write_mode: str) -> str:
    parsed = str(mode or "").strip().lower()
    if parsed not in _VALID_SERVICE_RESOLVER_MODES:
        parsed = "offline" if service_write_mode == "distill" else "off"
    if service_write_mode != "distill":
        return "off"
    return parsed


def _normalize_service_drain_queue_mode(mode: str, service_write_mode: str) -> str:
    parsed = str(mode or "").strip().lower()
    if parsed == "auto":
        parsed = "after_run" if service_write_mode == "distill" else "never"
    if parsed not in _VALID_SERVICE_DRAIN_QUEUE_MODES:
        parsed = "never"
    if service_write_mode != "distill":
        return "never"
    return parsed


def _effective_service_drain_queue_mode(
    mode: str,
    service_write_mode: str,
    service_resolver_mode: str,
) -> str:
    normalized = _normalize_service_drain_queue_mode(mode, service_write_mode)
    resolver = _normalize_service_resolver_mode(service_resolver_mode, service_write_mode)
    if resolver != "offline":
        return "never"
    return normalized


def _accumulate_queue_stats(total: dict[str, int], stats: dict[str, Any] | None) -> None:
    if not stats:
        return
    for key in ("claimed", "processed", "retried", "failed"):
        total[key] += int(stats.get(key, 0) or 0)

# ---------------------------------------------------------------------------
# Text normalization and scoring
# ---------------------------------------------------------------------------


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_answer(pred) == _normalize_answer(gold) else 0.0


def _f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------


def _supports_temperature(model: str) -> bool:
    return not model.startswith("gpt-5")


def _openai_text(client: openai.OpenAI, model: str, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        kwargs = {"model": model, "input": messages}
        if _supports_temperature(model):
            kwargs["temperature"] = 0.0
        resp = client.responses.create(**kwargs)
        text = getattr(resp, "output_text", None)
        if text:
            return text.strip()
    except Exception:
        pass

    chat_kwargs = {"model": model, "messages": messages}
    if _supports_temperature(model):
        chat_kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**chat_kwargs)
    return (resp.choices[0].message.content or "").strip()


def _longmemeval_prompt(task: str, question: str, answer: str, response: str,
                        abstention: bool) -> str:
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is "
            "given but the asked information is not.\n"
            f"Question: {question}\n"
            f"Explanation: {answer}\n"
            f"Response: {response}\n"
            "Is the model response correct? Answer yes or no only."
        )

    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no.\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Model Response: {response}\n"
            "Is the model response correct? Answer yes or no only."
        )

    if task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. "
            "If the question asks about the number of days/weeks/months/year between two events, "
            "do not penalize off-by-one errors, which means if the correct answer is 3 days, "
            "the answer can be 2 days or 4 days and it should still be marked as correct.\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Model Response: {response}\n"
            "Is the model response correct? Answer yes or no only."
        )

    if task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is the "
            "required answer.\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Model Response: {response}\n"
            "Is the model response correct? Answer yes or no only."
        )

    if task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response "
            "from a model. Please answer yes if the response satisfies the desired response. "
            "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the user's personal "
            "information correctly.\n\n"
            f"Question: {question}\n"
            f"Rubric: {answer}\n"
            f"Model Response: {response}\n"
            "Is the model response correct? Answer yes or no only."
        )

    raise ValueError(f"Unknown task: {task}")


def _longmemeval_judge(
    client: openai.OpenAI,
    model: str,
    task: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool,
) -> bool:
    if model.startswith("gpt-5"):
        raise ValueError("LongMemEval official judge requires gpt-4o or gpt-4o-mini.")
    prompt = _longmemeval_prompt(task, question, answer, response, abstention)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=10,
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    return "yes" in text


def _build_lme_evidence_files(instance: dict) -> list[dict[str, str]]:
    answer_ids = set(instance.get("answer_session_ids", []))
    evidence_files: list[dict[str, str]] = []
    for i, sid in enumerate(instance.get("haystack_session_ids", [])):
        if sid in answer_ids:
            evidence_files.append({"filename": f"memory/session_{i:04d}.md"})
    return evidence_files


def _result_path(item: Any) -> str:
    if hasattr(item, "path"):
        return str(item.path)
    if isinstance(item, dict):
        return str(item.get("path", ""))
    return ""


def _result_start_line(item: Any) -> int:
    if hasattr(item, "start_line"):
        return int(item.start_line)
    if isinstance(item, dict):
        return int(item.get("start_line", 1))
    return 1


def _result_end_line(item: Any) -> int:
    if hasattr(item, "end_line"):
        return int(item.end_line)
    if isinstance(item, dict):
        return int(item.get("end_line", _result_start_line(item)))
    return _result_start_line(item)


def _result_score(item: Any) -> float:
    if hasattr(item, "score"):
        return float(item.score)
    if isinstance(item, dict):
        return float(item.get("score", 0.0))
    return 0.0


def _result_snippet(item: Any) -> str:
    if hasattr(item, "snippet"):
        return str(item.snippet or "")
    if isinstance(item, dict):
        return str(item.get("snippet", "") or "")
    return ""


def _retrieval_hit_at_k(results: list[Any], evidence_files: list[dict], k: int = 5) -> float:
    evidence_fns = {ef["filename"] for ef in evidence_files}
    paths = {_result_path(r).replace("\\", "/") for r in results[:k]}
    hit = any(any(p.endswith(ef) for p in paths) for ef in evidence_fns)
    return 1.0 if hit else 0.0


def _evidence_coverage_at_k(results: list[Any], evidence_files: list[dict], k: int = 5) -> dict[str, float]:
    """
    Multi-evidence retrieval metrics.

    Returns:
      - any_hit: 1.0 when at least one evidence file is hit
      - coverage: matched_evidence_count / total_evidence_count
      - all_hit: 1.0 when all evidence files are found in top-k
    """
    evidence_fns = {ef["filename"].replace("\\", "/") for ef in evidence_files}
    if not evidence_fns:
        return {"any_hit": 1.0, "coverage": 1.0, "all_hit": 1.0}

    paths = {_result_path(r).replace("\\", "/") for r in results[:k]}
    matched: set[str] = set()
    for p in paths:
        for ef in evidence_fns:
            if p.endswith(ef):
                matched.add(ef)

    coverage = len(matched) / len(evidence_fns)
    return {
        "any_hit": 1.0 if matched else 0.0,
        "coverage": coverage,
        "all_hit": 1.0 if len(matched) == len(evidence_fns) else 0.0,
    }


def _evidence_supported_rate(retrieval_hit: float, pred: str) -> float:
    """Heuristic: supported when retrieval hit and answer is non-empty."""
    if retrieval_hit == 1.0 and pred.strip():
        return 1.0
    return 0.0


def _abstention_precision(pred: str) -> float | None:
    """For abstention questions: 1.0 if pred contains abstention phrase, else 0.0."""
    lower = pred.lower()
    if "don't know" in lower or "i don't know" in lower:
        return 1.0
    return 0.0

def _judge_answer(
    client: openai.OpenAI,
    judge_model: str,
    question: str,
    gold: str,
    pred: str,
) -> bool:
    prompt = (
        "You are a strict evaluator. Decide if the model answer is correct given the question and "
        "the reference answer. Reply with CORRECT or INCORRECT only.\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {gold}\n"
        f"Model Answer: {pred}\n"
    )
    text = _openai_text(client, judge_model, "You are a strict grader.", prompt)
    text = text.strip().upper()
    if "CORRECT" in text and "INCORRECT" not in text:
        return True
    if "INCORRECT" in text:
        return False
    # Fallback: if ambiguous, default to incorrect
    return False


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _select_instances(instances: list[dict], limit: int, balanced: bool) -> list[dict]:
    if limit <= 0 or limit >= len(instances):
        return instances
    if not balanced:
        return instances[:limit]

    by_type: dict[str, list[dict]] = {}
    for d in instances:
        by_type.setdefault(d["question_type"], []).append(d)
    per_type = max(1, limit // len(by_type))
    sampled: list[dict] = []
    for qt in sorted(by_type):
        sampled.extend(by_type[qt][:per_type])
    return sampled[:limit]


def _build_evidence_chunks(
    mgr: MemoryIndexManager | None,
    results: list[Any],
    max_results: int,
    max_chars: int,
    context_lines: int = 0,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for r in results[:max_results]:
        path = _result_path(r)
        start_line = _result_start_line(r)
        end_line = _result_end_line(r)
        try:
            if context_lines > 0 and start_line > 0:
                from_line = max(1, start_line - context_lines)
                lines = max(1, end_line - start_line + 1) + (2 * context_lines)
            else:
                from_line = start_line
                lines = max(1, end_line - start_line + 1)
            if mgr is None:
                raise ValueError("manager unavailable")
            text = mgr.read_file(path, from_line=from_line, lines=lines)["text"]
        except Exception:
            text = _result_snippet(r)
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        chunks.append(
            {
                "id": path,
                "path": path,
                "snippet": text,
                "score": _result_score(r),
            }
        )
    return chunks


def _serialize_results(results: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in results:
        out.append(
            {
                "path": _result_path(r),
                "start_line": _result_start_line(r),
                "end_line": _result_end_line(r),
                "score": _result_score(r),
                "snippet": _result_snippet(r),
            }
        )
    return out


def _session_to_text(session: list[dict[str, Any]], session_id: str, date: str) -> str:
    lines = [f"Session: {session_id}", f"Date: {date}", ""]
    for turn in session:
        role = str(turn.get("role", "unknown")).strip()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _service_reset_user_data(pg_dsn: str, user_id: str) -> None:
    conn = get_pg_connection(pg_dsn)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("DELETE FROM working_messages WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM memory_update_queue WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM canonical_memories WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM semantic_memories WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM episodic_memories WHERE user_id = %s", (user_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _service_user_has_data(pg_dsn: str, user_id: str) -> bool:
    conn = get_pg_connection(pg_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM episodic_memories WHERE user_id = %s) +
                    (SELECT COUNT(*) FROM semantic_memories WHERE user_id = %s) +
                    (SELECT COUNT(*) FROM canonical_memories WHERE user_id = %s) AS total
                """,
                (user_id, user_id, user_id),
            )
            row = cur.fetchone()
            total = int(row[0]) if row and row[0] is not None else 0
            return total > 0
    finally:
        conn.close()


def _service_find_existing_user_ids(
    pg_dsn: str,
    provider_label: str,
    qid: str,
    *,
    limit: int = 5,
) -> list[str]:
    provider_part = provider_label.replace("/", "_")
    pattern = f"lmeqa_{provider_part}_%_{qid}"
    conn = get_pg_connection(pg_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id
                FROM (
                    SELECT DISTINCT user_id
                    FROM episodic_memories
                    WHERE user_id LIKE %s
                    UNION
                    SELECT DISTINCT user_id
                    FROM semantic_memories
                    WHERE user_id LIKE %s
                    UNION
                    SELECT DISTINCT user_id
                    FROM canonical_memories
                    WHERE user_id LIKE %s
                ) AS u
                ORDER BY user_id
                LIMIT %s
                """,
                (pattern, pattern, pattern, max(1, int(limit))),
            )
            rows = cur.fetchall()
        return [str(r[0]) for r in rows if r and r[0] is not None]
    finally:
        conn.close()


def _service_user_id_for_instance(
    provider_label: str,
    qid: str,
    service_write_mode: str,
    distill_batch_sessions: int,
) -> str:
    mode_suffix = (
        f"distill_b{distill_batch_sessions}" if service_write_mode == "distill" else "raw"
    )
    return f"lmeqa_{provider_label}_{mode_suffix}_{qid}".replace("/", "_")


def _build_memory_service(
    *,
    pg_dsn: str,
    provider: Any,
    llm_fn: Callable[[str], str],
    service_write_mode: str,
    service_lightmem: bool,
    service_resolver_mode: str,
) -> MemoryService:
    distill_mode = service_write_mode == "distill"
    resolver_mode = _normalize_service_resolver_mode(service_resolver_mode, service_write_mode)
    enable_resolver = distill_mode and resolver_mode != "off"
    resolver_update_mode = resolver_mode if resolver_mode in ("offline", "sync") else "offline"
    enable_lightmem = distill_mode and service_lightmem

    return MemoryService(
        pg_dsn=pg_dsn,
        embedding_provider=provider,
        llm_fn=llm_fn,
        enable_structured_distill=distill_mode,
        enable_conflict_resolver=enable_resolver,
        enable_answer_contract=True,
        enable_lightmem=enable_lightmem,
        resolver_update_mode=resolver_update_mode,
    )


def _service_ingest_raw_sessions(
    pg_dsn: str,
    user_id: str,
    instance: dict[str, Any],
    provider: Any,
) -> None:
    sessions = instance.get("haystack_sessions", [])
    session_ids = instance.get("haystack_session_ids", [])
    dates = instance.get("haystack_dates", [])

    content_rows: list[tuple[int, str, str, str, str]] = []
    for i, (sess, sid, date) in enumerate(zip(sessions, session_ids, dates)):
        content = _session_to_text(sess, str(sid), str(date))
        content_rows.append((i, str(sid), str(date), content, content[:4000]))

    # Batch embeddings to reduce API round-trips significantly.
    embed_inputs = [row[4] for row in content_rows]
    embeddings: list[list[float]]
    try:
        embeddings = provider.embed_batch(embed_inputs)
    except Exception:
        embeddings = [provider.embed_query(text) for text in embed_inputs]
    if len(embeddings) != len(content_rows):
        embeddings = [provider.embed_query(text) for text in embed_inputs]

    conn = get_pg_connection(pg_dsn)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            for (i, sid, date, content, _), embedding in zip(content_rows, embeddings):
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
                metadata = {
                    "source_path": f"memory/session_{i:04d}.md",
                    "session_id": str(sid),
                    "session_index": i,
                    "date": str(date),
                }
                cur.execute(
                    """
                    INSERT INTO episodic_memories
                    (user_id, thread_id, content, embedding, memory_type, metadata)
                    VALUES (%s, %s, %s, %s::vector, %s, %s::jsonb)
                    """,
                    (
                        user_id,
                        str(sid),
                        content,
                        embedding_str,
                        "session",
                        json.dumps(metadata),
                    ),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _result_to_evidence_chunks(
    results: list[Any], max_results: int, max_chars: int
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for r in results[:max_results]:
        snippet = _result_snippet(r)
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        path = _result_path(r)
        chunks.append(
            {
                "id": path,
                "path": path,
                "snippet": snippet,
                "score": _result_score(r),
            }
        )
    return chunks


def _query_needs_diverse_evidence(query: str) -> bool:
    q = query.lower()
    cues = (
        "how many",
        "how much",
        "how long",
        "when",
        "latest",
        "most recent",
        "current",
        "updated",
        "change",
        "between",
        "total",
        "count",
    )
    return any(c in q for c in cues)


def _diversify_results_by_path(results: list[Any], top_k: int) -> list[Any]:
    """Promote path diversity before duplicates to improve multi-evidence coverage."""
    unique: list[Any] = []
    dup: list[Any] = []
    seen: set[str] = set()
    for r in results:
        p = _result_path(r)
        if p and p not in seen:
            seen.add(p)
            unique.append(r)
        else:
            dup.append(r)
    return (unique + dup)[:top_k]


def _prepare_results_for_qa(
    query: str,
    results: list[Any],
    *,
    answer_top_k: int,
    diversify_paths: bool,
) -> list[Any]:
    if not results:
        return []
    if diversify_paths and _query_needs_diverse_evidence(query):
        return _diversify_results_by_path(results, answer_top_k)
    return results[:answer_top_k]


def _session_to_conversation(
    session: list[dict[str, Any]],
    session_id: str,
    date: str,
) -> list[dict[str, str]]:
    conversation: list[dict[str, str]] = [
        {"role": "system", "content": f"session_id={session_id}; date={date}"}
    ]
    for turn in session:
        role = str(turn.get("role", "user")).strip() or "user"
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        conversation.append({"role": role, "content": content})
    return conversation


def _service_ingest_distilled_sessions(
    service: MemoryService,
    user_id: str,
    instance: dict[str, Any],
    *,
    batch_sessions: int = 2,
    official_batching: bool = True,
    batch_system_prompt: str = LIGHTMEM_BATCH_SYSTEM_PROMPT,
) -> None:
    sessions = instance.get("haystack_sessions", [])
    session_ids = instance.get("haystack_session_ids", [])
    dates = instance.get("haystack_dates", [])
    batch_n = max(1, int(batch_sessions))

    def _session_turns(sess: list[dict[str, Any]]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for turn in sess:
            role = str(turn.get("role", "user")).strip() or "user"
            content = str(turn.get("content", "")).strip()
            if content:
                out.append({"role": role, "content": content})
        return out

    if official_batching:
        if batch_n == 1:
            for i, sess in enumerate(sessions):
                turns = _session_turns(sess)
                if not turns:
                    continue
                sid = str(session_ids[i]) if i < len(session_ids) else f"s{i:04d}"
                service.ingest_conversation(user_id, sid, turns)
            return

        batch_idx = 0
        for i in range(0, len(sessions), batch_n):
            merged: list[dict[str, str]] = []
            if batch_system_prompt.strip():
                merged.append({"role": "system", "content": batch_system_prompt})
            for sess in sessions[i:i + batch_n]:
                merged.extend(_session_turns(sess))
            if not merged:
                continue
            service.ingest_conversation(user_id, f"batch_{batch_idx:04d}", merged)
            batch_idx += 1
        return

    if batch_n == 1:
        for i, sess in enumerate(sessions):
            sid = str(session_ids[i]) if i < len(session_ids) else f"s{i:04d}"
            date = str(dates[i]) if i < len(dates) else ""
            conversation = _session_to_conversation(sess, sid, date)
            if len(conversation) <= 1:
                continue
            service.ingest_conversation(user_id, sid, conversation)
        return

    merged: list[dict[str, str]] = []
    batch_idx = 0
    in_batch = 0
    for i, sess in enumerate(sessions):
        sid = str(session_ids[i]) if i < len(session_ids) else f"s{i:04d}"
        date = str(dates[i]) if i < len(dates) else ""
        merged.append(
            {
                "role": "system",
                "content": f"session_boundary id={sid} date={date}",
            }
        )
        merged.extend(_session_turns(sess))
        in_batch += 1
        if in_batch >= batch_n:
            if len(merged) > 1:
                service.ingest_conversation(user_id, f"batch_{batch_idx:04d}", merged)
            merged = []
            in_batch = 0
            batch_idx += 1
    if len(merged) > 1:
        service.ingest_conversation(user_id, f"batch_{batch_idx:04d}", merged)


def _prepare_service_instance(
    *,
    idx: int,
    total: int,
    inst: dict[str, Any],
    provider: Any,
    provider_label: str,
    pg_dsn: str,
    llm_fn: Callable[[str], str],
    service_write_mode: str,
    distill_batch_sessions: int,
    service_lightmem: bool,
    normalized_resolver_mode: str,
    normalized_drain_mode: str,
    service_drain_queue_limit: int,
    service_drain_queue_max_attempts: int,
    service_official_batching: bool,
    service_batch_system_prompt: str,
    force_reindex: bool,
    reuse_service_ingest: bool,
) -> dict[str, Any]:
    qid = str(inst.get("question_id", f"idx-{idx}"))
    qtype = str(inst.get("question_type", "unknown"))
    item_start = time.time()

    try:
        user_id = _service_user_id_for_instance(
            provider_label,
            qid,
            service_write_mode,
            distill_batch_sessions,
        )
        has_data = _service_user_has_data(pg_dsn, user_id)
        should_ingest = True
        if not force_reindex and reuse_service_ingest and has_data:
            should_ingest = False

        drain_stats: dict[str, int] | None = None
        if should_ingest:
            _service_reset_user_data(pg_dsn, user_id)
            if service_write_mode == "distill":
                service = _build_memory_service(
                    pg_dsn=pg_dsn,
                    provider=provider,
                    llm_fn=llm_fn,
                    service_write_mode=service_write_mode,
                    service_lightmem=service_lightmem,
                    service_resolver_mode=normalized_resolver_mode,
                )
                _service_ingest_distilled_sessions(
                    service,
                    user_id,
                    inst,
                    batch_sessions=distill_batch_sessions,
                    official_batching=service_official_batching,
                    batch_system_prompt=service_batch_system_prompt,
                )
                if normalized_drain_mode == "after_ingest":
                    drain_stats = service.drain_update_queue(
                        limit=service_drain_queue_limit,
                        user_id=user_id,
                        max_attempts=service_drain_queue_max_attempts,
                    )
            else:
                _service_ingest_raw_sessions(pg_dsn, user_id, inst, provider)

        elapsed = time.time() - item_start
        return {
            "idx": idx,
            "question_id": qid,
            "question_type": qtype,
            "prepared": True,
            "ingested": should_ingest,
            "reused": not should_ingest,
            "queue_drain": drain_stats,
            "elapsed_s": round(elapsed, 1),
            "log": (
                f"  [prepare {idx+1}/{total}] {qid} ({qtype})\n"
                f"    prepared in {elapsed:.1f}s | ingested={int(should_ingest)} "
                f"reused={int(not should_ingest)}"
            ),
        }
    except Exception as exc:
        elapsed = time.time() - item_start
        return {
            "idx": idx,
            "question_id": qid,
            "question_type": qtype,
            "prepared": False,
            "error": str(exc),
            "elapsed_s": round(elapsed, 1),
            "log": (
                f"  [prepare {idx+1}/{total}] {qid} ({qtype})\n"
                f"    FAILED in {elapsed:.1f}s: {exc}"
            ),
        }


def prepare_longmemeval_service_data(
    instances: list[dict[str, Any]],
    *,
    provider: Any,
    provider_label: str,
    pg_dsn: str,
    llm_fn: Callable[[str], str],
    service_write_mode: str,
    distill_batch_sessions: int,
    service_lightmem: bool,
    service_resolver_mode: str,
    service_drain_queue_mode: str,
    service_drain_queue_limit: int,
    service_drain_queue_max_attempts: int,
    service_official_batching: bool,
    service_batch_system_prompt: str,
    service_workers: int,
    force_reindex: bool,
    reuse_service_ingest: bool,
) -> dict[str, Any]:
    """Prepare service-side memory once, without running retrieval/answer."""
    normalized_resolver_mode = _normalize_service_resolver_mode(
        service_resolver_mode,
        service_write_mode,
    )
    normalized_drain_mode = _effective_service_drain_queue_mode(
        service_drain_queue_mode,
        service_write_mode,
        normalized_resolver_mode,
    )
    worker_n = max(1, int(service_workers))

    prepared_new = 0
    reused_existing = 0
    errors = 0
    drain_total = {"claimed": 0, "processed": 0, "retried": 0, "failed": 0}
    drain_after_run: dict[str, int] | None = None
    per_instance_by_idx: dict[int, dict[str, Any]] = {}
    t0 = time.time()

    if worker_n == 1:
        for idx, inst in enumerate(instances):
            result = _prepare_service_instance(
                idx=idx,
                total=len(instances),
                inst=inst,
                provider=provider,
                provider_label=provider_label,
                pg_dsn=pg_dsn,
                llm_fn=llm_fn,
                service_write_mode=service_write_mode,
                distill_batch_sessions=distill_batch_sessions,
                service_lightmem=service_lightmem,
                normalized_resolver_mode=normalized_resolver_mode,
                normalized_drain_mode=normalized_drain_mode,
                service_drain_queue_limit=service_drain_queue_limit,
                service_drain_queue_max_attempts=service_drain_queue_max_attempts,
                service_official_batching=service_official_batching,
                service_batch_system_prompt=service_batch_system_prompt,
                force_reindex=force_reindex,
                reuse_service_ingest=reuse_service_ingest,
            )
            print(result["log"], flush=True)
            if result.get("prepared"):
                if result.get("ingested"):
                    prepared_new += 1
                else:
                    reused_existing += 1
                _accumulate_queue_stats(drain_total, result.get("queue_drain"))
            else:
                errors += 1
            result_copy = {k: v for k, v in result.items() if k not in {"idx", "log"}}
            per_instance_by_idx[int(result["idx"])] = result_copy
    else:
        print(
            f"  prepare workers={worker_n} (service pipeline parallel mode)",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=worker_n) as executor:
            future_map = {
                executor.submit(
                    _prepare_service_instance,
                    idx=idx,
                    total=len(instances),
                    inst=inst,
                    provider=provider,
                    provider_label=provider_label,
                    pg_dsn=pg_dsn,
                    llm_fn=llm_fn,
                    service_write_mode=service_write_mode,
                    distill_batch_sessions=distill_batch_sessions,
                    service_lightmem=service_lightmem,
                    normalized_resolver_mode=normalized_resolver_mode,
                    normalized_drain_mode=normalized_drain_mode,
                    service_drain_queue_limit=service_drain_queue_limit,
                    service_drain_queue_max_attempts=service_drain_queue_max_attempts,
                    service_official_batching=service_official_batching,
                    service_batch_system_prompt=service_batch_system_prompt,
                    force_reindex=force_reindex,
                    reuse_service_ingest=reuse_service_ingest,
                ): idx
                for idx, inst in enumerate(instances)
            }
            for future in as_completed(future_map):
                try:
                    result = future.result()
                except Exception as exc:
                    idx = future_map[future]
                    qid = str(instances[idx].get("question_id", f"idx-{idx}"))
                    qtype = str(instances[idx].get("question_type", "unknown"))
                    result = {
                        "idx": idx,
                        "question_id": qid,
                        "question_type": qtype,
                        "prepared": False,
                        "error": str(exc),
                        "elapsed_s": 0.0,
                        "log": (
                            f"  [prepare {idx+1}/{len(instances)}] {qid} ({qtype})\n"
                            f"    FAILED in 0.0s: {exc}"
                        ),
                    }
                print(result["log"], flush=True)
                if result.get("prepared"):
                    if result.get("ingested"):
                        prepared_new += 1
                    else:
                        reused_existing += 1
                    _accumulate_queue_stats(drain_total, result.get("queue_drain"))
                else:
                    errors += 1
                result_copy = {k: v for k, v in result.items() if k not in {"idx", "log"}}
                per_instance_by_idx[int(result["idx"])] = result_copy

    if normalized_drain_mode == "after_run":
        service = _build_memory_service(
            pg_dsn=pg_dsn,
            provider=provider,
            llm_fn=llm_fn,
            service_write_mode=service_write_mode,
            service_lightmem=service_lightmem,
            service_resolver_mode=normalized_resolver_mode,
        )
        drain_after_run = service.drain_update_queue(
            limit=service_drain_queue_limit,
            user_id=None,
            max_attempts=service_drain_queue_max_attempts,
        )
        _accumulate_queue_stats(drain_total, drain_after_run)

    per_instance = [per_instance_by_idx[i] for i in sorted(per_instance_by_idx)]

    return {
        "provider": provider_label,
        "pipeline": "service",
        "service_write_mode": service_write_mode,
        "distill_batch_sessions": distill_batch_sessions,
        "service_lightmem": service_lightmem if service_write_mode == "distill" else None,
        "service_resolver_mode": (
            normalized_resolver_mode if service_write_mode == "distill" else None
        ),
        "service_drain_queue_mode": normalized_drain_mode,
        "service_official_batching": service_official_batching if service_write_mode == "distill" else None,
        "service_workers": worker_n,
        "mode": "prepare_only",
        "instances": len(instances),
        "elapsed_s": round(time.time() - t0, 1),
        "prepared_new": prepared_new,
        "reused_existing": reused_existing,
        "errors": errors,
        "queue_drain_total": drain_total,
        "queue_drain_after_run": drain_after_run,
        "aggregate": {},
        "by_type": {},
        "per_instance": per_instance,
    }


def run_longmemeval_qa(
    instances: list[dict],
    provider,
    provider_label: str,
    answer_client: openai.OpenAI,
    answer_model: str,
    judge: str,
    judge_model: str,
    config_name: str,
    max_results: int,
    max_chars: int,
    search_k: int,
    answer_top_k: int,
    context_lines: int,
    diversify_paths: bool,
    cache_dir: Path | None,
    force_reindex: bool,
    skip_sync_if_cached: bool,
    reuse_retrieval_cache: bool,
    pipeline: str,
    pg_dsn: str | None,
    service_write_mode: str,
    distill_batch_sessions: int,
    service_lightmem: bool,
    service_resolver_mode: str,
    service_drain_queue_mode: str,
    service_drain_queue_limit: int,
    service_drain_queue_max_attempts: int,
    service_official_batching: bool,
    service_batch_system_prompt: str,
    reuse_service_ingest: bool,
    read_answer_only: bool,
) -> dict[str, Any]:
    config_ov = CONFIG_MATRIX.get(config_name, CONFIG_MATRIX["hybrid"])
    use_provider = config_name != "fts_only" or pipeline == "service"
    answer_k = max(1, max(max_results, answer_top_k))

    per_instance: list[dict[str, Any]] = []
    t0 = time.time()

    def _answer_llm_fn(prompt: str) -> str:
        return _openai_text(
            answer_client,
            answer_model,
            "You must return valid JSON only.",
            prompt,
        )

    service: MemoryService | None = None
    service_search_kwargs: dict[str, Any] = {}
    normalized_resolver_mode = _normalize_service_resolver_mode(
        service_resolver_mode,
        service_write_mode,
    )
    normalized_drain_mode = _effective_service_drain_queue_mode(
        service_drain_queue_mode,
        service_write_mode,
        normalized_resolver_mode,
    )
    drain_total = {"claimed": 0, "processed": 0, "retried": 0, "failed": 0}
    drain_after_run: dict[str, int] | None = None
    if pipeline == "service":
        if provider is None:
            raise ValueError("service pipeline requires an embedding provider")
        if not pg_dsn:
            raise ValueError("service pipeline requires --pg-dsn or OPENCLAW_PG_DSN")
        from openclaw_memory.mmr import MMRConfig  # noqa: PLC0415

        if config_name == "fts_only":
            vector_weight, text_weight = 0.0, 1.0
        elif config_name == "vector_only":
            vector_weight, text_weight = 1.0, 0.0
        else:
            vector_weight = float(config_ov.get("vector_weight", 0.7))
            text_weight = float(config_ov.get("text_weight", 0.3))
        mmr_cfg = MMRConfig(
            enabled=bool(config_ov.get("mmr_enabled", True)),
            lambda_=float(config_ov.get("mmr_lambda", 0.7)),
        )
        service_search_kwargs = {
            "max_results": max(10, search_k),
            "include_working_memory": False,
            "enable_llm_rerank": False,
            "vector_weight": vector_weight,
            "text_weight": text_weight,
            "mmr_config": mmr_cfg,
        }
        service = _build_memory_service(
            pg_dsn=pg_dsn,
            provider=provider,
            llm_fn=_answer_llm_fn,
            service_write_mode=service_write_mode,
            service_lightmem=service_lightmem,
            service_resolver_mode=normalized_resolver_mode,
        )

    for idx, inst in enumerate(instances):
        inst_start = time.time()
        qid = inst.get("question_id", f"idx-{idx}")
        qtype = inst.get("question_type", "unknown")
        print(f"  [{idx+1}/{len(instances)}] {qid} ({qtype})", flush=True)
        tmpdir = None
        evidence_files = _build_lme_evidence_files(inst)
        retrieval_cache_file: Path | None = None
        workspace_path: Path | None = None
        db_path: Path | None = None
        if cache_dir is not None:
            inst_root = cache_dir / provider_label / qid
            retrieval_cache_file = (
                inst_root
                / (
                    "retrieval_"
                    f"{pipeline}_{config_name}"
                    f"_search{search_k}_ans{answer_top_k}"
                    f"_c{max_chars}_ctx{context_lines}"
                    f"_div{int(diversify_paths)}.json"
                )
            )
            if pipeline == "manager":
                workspace_path = inst_root / "ws"
                db_path = inst_root / "mem.sqlite"
                if force_reindex and inst_root.exists():
                    shutil.rmtree(inst_root, ignore_errors=True)
                if not workspace_path.exists() or not (workspace_path / "MEMORY.md").exists():
                    workspace_path.mkdir(parents=True, exist_ok=True)
                    build_lme_workspace(inst, str(workspace_path))
        elif pipeline == "manager":
            tmpdir = tempfile.mkdtemp(prefix="ocmem_lmeqa_")
            workspace_path = Path(tmpdir) / "ws"
            workspace_path.mkdir(parents=True, exist_ok=True)
            build_lme_workspace(inst, str(workspace_path))
            db_path = Path(tmpdir) / "mem.sqlite"

        try:
            prov = provider if use_provider else None
            cached_retrieval = False
            evidence_chunks: list[dict[str, Any]] = []
            results: list[Any] = []
            service_user_id: str | None = None
            retrieval_hit = 0.0
            cov5 = {"any_hit": 0.0, "coverage": 0.0, "all_hit": 0.0}
            cov10 = {"any_hit": 0.0, "coverage": 0.0, "all_hit": 0.0}
            should_ingest = False
            drain_stats: dict[str, int] | None = None

            if (
                reuse_retrieval_cache
                and pipeline == "manager"
                and retrieval_cache_file is not None
                and retrieval_cache_file.exists()
                and not force_reindex
            ):
                try:
                    cached = json.loads(retrieval_cache_file.read_text(encoding="utf-8"))
                    if cached.get("question") == inst["question"]:
                        evidence_chunks = list(cached.get("evidence_chunks", []))
                        results = list(cached.get("results", []))
                        retrieval_hit = float(
                            cached.get(
                                "retrieval_hit@5",
                                _retrieval_hit_at_k(results, evidence_files, k=5),
                            )
                        )
                        cov5 = dict(
                            cached.get(
                                "evidence_coverage@5",
                                _evidence_coverage_at_k(results, evidence_files, k=5),
                            )
                        )
                        cov10 = dict(
                            cached.get(
                                "evidence_coverage@10",
                                _evidence_coverage_at_k(results, evidence_files, k=10),
                            )
                        )
                        cached_retrieval = True
                except Exception:
                    cached_retrieval = False

            if not cached_retrieval:
                if pipeline == "manager":
                    if workspace_path is None or db_path is None:
                        raise ValueError("manager pipeline requires workspace/db paths")
                    overrides = {
                        "db_path": str(db_path),
                        "min_score": 0.0,
                        "max_results": max(10, max_results),
                        "cache_enabled": True,
                        **config_ov,
                    }
                    config = resolve_memory_search_config(
                        workspace_dir=str(workspace_path),
                        overrides=overrides,
                    )
                    mgr = MemoryIndexManager(
                        workspace_dir=str(workspace_path),
                        db_path=str(db_path),
                        config=config,
                        provider=prov,
                    )
                    with mgr:
                        should_sync = True
                        if (
                            skip_sync_if_cached
                            and cache_dir is not None
                            and db_path.exists()
                            and not force_reindex
                        ):
                            status = mgr.status()
                            should_sync = status.chunks <= 0
                        if should_sync:
                            mgr.sync(reason="benchmark", force=force_reindex)

                        raw_results = mgr.search(
                            inst["question"],
                            max_results=max(10, search_k),
                            min_score=0.0,
                        )
                        results = _prepare_results_for_qa(
                            inst["question"],
                            raw_results,
                            answer_top_k=answer_k,
                            diversify_paths=diversify_paths,
                        )
                        evidence_chunks = _build_evidence_chunks(
                            mgr,
                            results,
                            answer_k,
                            max_chars,
                            context_lines=context_lines,
                        )
                else:
                    if service is None or pg_dsn is None or prov is None:
                        raise ValueError("service pipeline not initialized")
                    service_user_id = _service_user_id_for_instance(
                        provider_label,
                        qid,
                        service_write_mode,
                        distill_batch_sessions,
                    )

                    should_ingest = True
                    has_data = _service_user_has_data(pg_dsn, service_user_id)
                    if read_answer_only:
                        if not has_data:
                            existing = _service_find_existing_user_ids(
                                pg_dsn,
                                provider_label,
                                str(qid),
                            )
                            existing_hint = (
                                f" Existing prepared IDs: {', '.join(existing)}"
                                if existing
                                else ""
                            )
                            raise ValueError(
                                f"No prepared service data for {qid} "
                                f"(expected user_id={service_user_id}). "
                                "Run with --prepare-only first, and keep "
                                "--service-write-mode/--distill-batch-sessions/"
                                "--provider/--embedding-model consistent."
                                f"{existing_hint}"
                            )
                        should_ingest = False
                    elif reuse_service_ingest and not force_reindex:
                        should_ingest = not has_data
                    if should_ingest:
                        _service_reset_user_data(pg_dsn, service_user_id)
                        if service_write_mode == "distill":
                            _service_ingest_distilled_sessions(
                                service,
                                service_user_id,
                                inst,
                                batch_sessions=distill_batch_sessions,
                                official_batching=service_official_batching,
                                batch_system_prompt=service_batch_system_prompt,
                            )
                        else:
                            _service_ingest_raw_sessions(pg_dsn, service_user_id, inst, prov)
                        if normalized_drain_mode == "after_ingest":
                            drain_stats = service.drain_update_queue(
                                limit=service_drain_queue_limit,
                                user_id=service_user_id,
                                max_attempts=service_drain_queue_max_attempts,
                            )
                            _accumulate_queue_stats(drain_total, drain_stats)
                    raw_results = service.search(
                        service_user_id, inst["question"], **service_search_kwargs
                    )
                    results = _prepare_results_for_qa(
                        inst["question"],
                        raw_results,
                        answer_top_k=answer_k,
                        diversify_paths=diversify_paths,
                    )
                    evidence_chunks = _result_to_evidence_chunks(results, answer_k, max_chars)

                retrieval_hit = _retrieval_hit_at_k(results, evidence_files, k=5)
                cov5 = _evidence_coverage_at_k(results, evidence_files, k=5)
                cov10 = _evidence_coverage_at_k(results, evidence_files, k=10)

                if retrieval_cache_file is not None:
                    try:
                        retrieval_cache_file.parent.mkdir(parents=True, exist_ok=True)
                        retrieval_cache_file.write_text(
                            json.dumps(
                                {
                                    "question": inst["question"],
                                    "results": _serialize_results(results),
                                    "evidence_chunks": evidence_chunks,
                                    "retrieval_hit@5": retrieval_hit,
                                    "evidence_coverage@5": cov5,
                                    "evidence_coverage@10": cov10,
                                },
                                ensure_ascii=False,
                            ),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

            question = inst["question"].strip()
            gold = str(inst["answer"]).strip()
            abstention = str(qid).endswith("_abs")
            payload = generate_answer(
                question,
                evidence_chunks,
                _answer_llm_fn,
            )
            pred = payload.answer.strip()
            if payload.abstain or not pred:
                pred = "I don't know."

            em = _exact_match(pred, gold)
            f1 = _f1_score(pred, gold)
            judged = None
            if judge == "openai":
                judged = _judge_answer(answer_client, judge_model, question, gold, pred)
            elif judge == "longmemeval":
                judged = _longmemeval_judge(
                    answer_client,
                    judge_model,
                    qtype,
                    question,
                    gold,
                    pred,
                    abstention,
                )

            ev_supported = _evidence_supported_rate(retrieval_hit, pred)
            ev_unsupported = 1.0 - ev_supported
            abst_prec: float | None = _abstention_precision(pred) if abstention else None

            per_instance.append(
                {
                    "question_id": qid,
                    "question_type": qtype,
                    "question": question,
                    "answer": gold,
                    "prediction": pred,
                    "retrieval_hit@5": retrieval_hit,
                    "retrieval_coverage@5": cov5["coverage"],
                    "retrieval_all_hit@5": cov5["all_hit"],
                    "retrieval_coverage@10": cov10["coverage"],
                    "retrieval_all_hit@10": cov10["all_hit"],
                    "exact_match": em,
                    "f1": f1,
                    "judge_correct": judged,
                    "evidence_supported_rate": ev_supported,
                    "unsupported_claim_rate": ev_unsupported,
                    "abstention_precision": abst_prec,
                    "answer_contract": {
                        "abstain": payload.abstain,
                        "confidence": payload.confidence,
                        "evidence_count": len(payload.evidence),
                        "abstain_reason": payload.abstain_reason,
                    },
                    "retrieval_cached": cached_retrieval,
                    "service_ingested": should_ingest if pipeline == "service" else None,
                    "queue_drain": drain_stats if pipeline == "service" else None,
                }
            )
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)

        inst_elapsed = time.time() - inst_start
        print(
            "    done in "
            f"{inst_elapsed:.1f}s | EM={em:.3f} F1={f1:.3f} "
            f"cov5={cov5['coverage']:.3f} all5={cov5['all_hit']:.0f}",
            flush=True,
        )

    if (
        pipeline == "service"
        and service_write_mode == "distill"
        and normalized_drain_mode == "after_run"
        and service is not None
    ):
        drain_after_run = service.drain_update_queue(
            limit=service_drain_queue_limit,
            user_id=None,
            max_attempts=service_drain_queue_max_attempts,
        )
        _accumulate_queue_stats(drain_total, drain_after_run)

    elapsed = time.time() - t0

    agg: dict[str, float] = {}
    agg["exact_match"] = sum(m["exact_match"] for m in per_instance) / len(per_instance)
    agg["f1"] = sum(m["f1"] for m in per_instance) / len(per_instance)
    retrieval_vals = [
        m["retrieval_hit@5"]
        for m in per_instance
        if m.get("retrieval_hit@5") is not None and not str(m.get("question_id", "")).endswith("_abs")
    ]
    if retrieval_vals:
        agg["retrieval_hit@5"] = sum(retrieval_vals) / len(retrieval_vals)
    cov5_vals = [
        m["retrieval_coverage@5"]
        for m in per_instance
        if m.get("retrieval_coverage@5") is not None
        and not str(m.get("question_id", "")).endswith("_abs")
    ]
    if cov5_vals:
        agg["retrieval_coverage@5"] = sum(cov5_vals) / len(cov5_vals)
    all5_vals = [
        m["retrieval_all_hit@5"]
        for m in per_instance
        if m.get("retrieval_all_hit@5") is not None
        and not str(m.get("question_id", "")).endswith("_abs")
    ]
    if all5_vals:
        agg["retrieval_all_hit@5"] = sum(all5_vals) / len(all5_vals)
    cov10_vals = [
        m["retrieval_coverage@10"]
        for m in per_instance
        if m.get("retrieval_coverage@10") is not None
        and not str(m.get("question_id", "")).endswith("_abs")
    ]
    if cov10_vals:
        agg["retrieval_coverage@10"] = sum(cov10_vals) / len(cov10_vals)
    all10_vals = [
        m["retrieval_all_hit@10"]
        for m in per_instance
        if m.get("retrieval_all_hit@10") is not None
        and not str(m.get("question_id", "")).endswith("_abs")
    ]
    if all10_vals:
        agg["retrieval_all_hit@10"] = sum(all10_vals) / len(all10_vals)
    if judge == "openai":
        judged_vals = [m["judge_correct"] for m in per_instance if m["judge_correct"] is not None]
        agg["judge_acc"] = sum(1.0 if v else 0.0 for v in judged_vals) / len(judged_vals)
    elif judge == "longmemeval":
        judged_vals = [m["judge_correct"] for m in per_instance if m["judge_correct"] is not None]
        agg["judge_acc"] = sum(1.0 if v else 0.0 for v in judged_vals) / len(judged_vals)

    ev_sup_vals = [
        m["evidence_supported_rate"] for m in per_instance
        if m.get("evidence_supported_rate") is not None
    ]
    if ev_sup_vals:
        agg["evidence_supported_rate"] = sum(ev_sup_vals) / len(ev_sup_vals)
    ev_unsup_vals = [
        m["unsupported_claim_rate"] for m in per_instance
        if m.get("unsupported_claim_rate") is not None
    ]
    if ev_unsup_vals:
        agg["unsupported_claim_rate"] = sum(ev_unsup_vals) / len(ev_unsup_vals)
    abst_vals = [
        m["abstention_precision"] for m in per_instance
        if m.get("abstention_precision") is not None
    ]
    if abst_vals:
        agg["abstention_precision"] = sum(abst_vals) / len(abst_vals)

    by_type: dict[str, dict[str, Any]] = {}
    for m in per_instance:
        qt = m["question_type"]
        if qt not in by_type:
            by_type[qt] = {
                "exact_match": [],
                "f1": [],
                "count": 0,
                "retrieval_hit@5": [],
                "retrieval_coverage@5": [],
                "retrieval_all_hit@5": [],
                "retrieval_coverage@10": [],
                "retrieval_all_hit@10": [],
                "evidence_supported_rate": [],
                "unsupported_claim_rate": [],
                "abstention_precision": [],
            }
            if judge in ("openai", "longmemeval"):
                by_type[qt]["judge_correct"] = []
        by_type[qt]["exact_match"].append(m["exact_match"])
        by_type[qt]["f1"].append(m["f1"])
        if m.get("retrieval_hit@5") is not None and not str(m.get("question_id", "")).endswith("_abs"):
            by_type[qt]["retrieval_hit@5"].append(m["retrieval_hit@5"])
        if m.get("retrieval_coverage@5") is not None and not str(m.get("question_id", "")).endswith("_abs"):
            by_type[qt]["retrieval_coverage@5"].append(m["retrieval_coverage@5"])
        if m.get("retrieval_all_hit@5") is not None and not str(m.get("question_id", "")).endswith("_abs"):
            by_type[qt]["retrieval_all_hit@5"].append(m["retrieval_all_hit@5"])
        if m.get("retrieval_coverage@10") is not None and not str(m.get("question_id", "")).endswith("_abs"):
            by_type[qt]["retrieval_coverage@10"].append(m["retrieval_coverage@10"])
        if m.get("retrieval_all_hit@10") is not None and not str(m.get("question_id", "")).endswith("_abs"):
            by_type[qt]["retrieval_all_hit@10"].append(m["retrieval_all_hit@10"])
        by_type[qt]["count"] += 1
        if judge in ("openai", "longmemeval"):
            by_type[qt]["judge_correct"].append(1.0 if m["judge_correct"] else 0.0)
        if m.get("evidence_supported_rate") is not None:
            by_type[qt]["evidence_supported_rate"].append(m["evidence_supported_rate"])
        if m.get("unsupported_claim_rate") is not None:
            by_type[qt]["unsupported_claim_rate"].append(m["unsupported_claim_rate"])
        if m.get("abstention_precision") is not None:
            by_type[qt]["abstention_precision"].append(m["abstention_precision"])

    type_summary: dict[str, dict[str, Any]] = {}
    for qt, vals in by_type.items():
        summary = {
            "exact_match": sum(vals["exact_match"]) / len(vals["exact_match"]),
            "f1": sum(vals["f1"]) / len(vals["f1"]),
            "count": vals["count"],
        }
        if vals.get("retrieval_hit@5"):
            summary["retrieval_hit@5"] = (
                sum(vals["retrieval_hit@5"]) / len(vals["retrieval_hit@5"])
            )
        if vals.get("retrieval_coverage@5"):
            summary["retrieval_coverage@5"] = (
                sum(vals["retrieval_coverage@5"]) / len(vals["retrieval_coverage@5"])
            )
        if vals.get("retrieval_all_hit@5"):
            summary["retrieval_all_hit@5"] = (
                sum(vals["retrieval_all_hit@5"]) / len(vals["retrieval_all_hit@5"])
            )
        if vals.get("retrieval_coverage@10"):
            summary["retrieval_coverage@10"] = (
                sum(vals["retrieval_coverage@10"]) / len(vals["retrieval_coverage@10"])
            )
        if vals.get("retrieval_all_hit@10"):
            summary["retrieval_all_hit@10"] = (
                sum(vals["retrieval_all_hit@10"]) / len(vals["retrieval_all_hit@10"])
            )
        if judge in ("openai", "longmemeval"):
            summary["judge_acc"] = sum(vals["judge_correct"]) / len(vals["judge_correct"])
        if vals.get("evidence_supported_rate"):
            summary["evidence_supported_rate"] = (
                sum(vals["evidence_supported_rate"]) / len(vals["evidence_supported_rate"])
            )
        if vals.get("unsupported_claim_rate"):
            summary["unsupported_claim_rate"] = (
                sum(vals["unsupported_claim_rate"]) / len(vals["unsupported_claim_rate"])
            )
        if vals.get("abstention_precision"):
            summary["abstention_precision"] = (
                sum(vals["abstention_precision"]) / len(vals["abstention_precision"])
            )
        type_summary[qt] = summary

    return {
        "provider": provider_label,
        "pipeline": pipeline,
        "mode": "read_answer_only" if read_answer_only else "run",
        "service_write_mode": service_write_mode if pipeline == "service" else None,
        "distill_batch_sessions": (
            distill_batch_sessions if pipeline == "service" and service_write_mode == "distill" else None
        ),
        "service_lightmem": service_lightmem if pipeline == "service" and service_write_mode == "distill" else None,
        "service_resolver_mode": (
            normalized_resolver_mode
            if pipeline == "service" and service_write_mode == "distill"
            else None
        ),
        "service_drain_queue_mode": (
            normalized_drain_mode
            if pipeline == "service" and service_write_mode == "distill"
            else "never"
        ),
        "service_official_batching": (
            service_official_batching
            if pipeline == "service" and service_write_mode == "distill"
            else None
        ),
        "reuse_service_ingest": reuse_service_ingest if pipeline == "service" else None,
        "config": config_name,
        "answer_model": answer_model,
        "judge": judge,
        "judge_model": judge_model if judge == "openai" else None,
        "instances": len(instances),
        "elapsed_s": round(elapsed, 1),
        "queue_drain_total": (
            drain_total
            if pipeline == "service" and service_write_mode == "distill"
            else None
        ),
        "queue_drain_after_run": (
            drain_after_run
            if pipeline == "service" and service_write_mode == "distill"
            else None
        ),
        "aggregate": agg,
        "by_type": type_summary,
        "per_instance": per_instance,
    }


def _providers_from_arg(value: str) -> list[str]:
    if value == "all":
        return ["none", "mock", "openai"]
    return [value]


def _label_for_provider(provider_name: str, embedding_model: str) -> str:
    if provider_name == "openai":
        return f"openai/{embedding_model}"
    if provider_name == "mock":
        return "mock"
    return "fts_only"


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval QA (end-to-end) benchmark")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    parser.add_argument("--embedding-model", default="text-embedding-3-small",
                        help="OpenAI embedding model")
    parser.add_argument("--answer-model", default="gpt-5-mini",
                        help="OpenAI model for answering")
    parser.add_argument(
        "--judge",
        choices=["exact", "openai", "longmemeval"],
        default="longmemeval",
        help="Answer evaluation method (longmemeval aligns with official prompts)",
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="OpenAI model for judge (longmemeval uses gpt-4o/4o-mini)")
    parser.add_argument(
        "--pipeline",
        choices=["manager", "service"],
        default="manager",
        help="Execution pipeline: manager(SQLite) or service(PostgreSQL MemoryService)",
    )
    parser.add_argument(
        "--pg-dsn",
        default=None,
        help="PostgreSQL DSN (required when --pipeline service)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "mock", "none", "all"],
        default="openai",
        help="Embedding provider for retrieval (openai|mock|none|all)",
    )
    parser.add_argument(
        "--service-write-mode",
        choices=["raw", "distill"],
        default="raw",
        help="When pipeline=service: raw session rows or full distill/conflict write path",
    )
    parser.add_argument(
        "--service-lightmem",
        action="store_true",
        default=True,
        help="Enable LightMem-style distill preprocessing in service distill mode (default: on)",
    )
    parser.add_argument(
        "--no-service-lightmem",
        dest="service_lightmem",
        action="store_false",
        help="Disable LightMem-style distill preprocessing in service distill mode",
    )
    parser.add_argument(
        "--service-resolver-mode",
        choices=["off", "offline", "sync"],
        default="offline",
        help=(
            "Conflict resolver mode for service distill: "
            "off=disable, offline=enqueue+sleep-time update, sync=inline update (default: offline)"
        ),
    )
    parser.add_argument(
        "--service-drain-queue-mode",
        choices=["auto", "never", "after_ingest", "after_run"],
        default="auto",
        help=(
            "When resolver mode is offline, process queue never/after_ingest/after_run. "
            "auto => after_run for distill, else never"
        ),
    )
    parser.add_argument(
        "--service-drain-queue-limit",
        type=int,
        default=20000,
        help="Max queue items to process per drain cycle (default: 20000)",
    )
    parser.add_argument(
        "--service-drain-queue-max-attempts",
        type=int,
        default=3,
        help="Max retry attempts per queued item (default: 3)",
    )
    parser.add_argument(
        "--service-official-batching",
        action="store_true",
        default=True,
        help=(
            "Use LightMem longmemeval batching style (merge N sessions per ingest call; "
            "prepend system prompt only when batch size > 1) (default: on)"
        ),
    )
    parser.add_argument(
        "--no-service-official-batching",
        dest="service_official_batching",
        action="store_false",
        help="Use metadata-preserving session boundary batching instead of official LightMem style",
    )
    parser.add_argument(
        "--service-batch-system-prompt",
        default=LIGHTMEM_BATCH_SYSTEM_PROMPT,
        help="System prompt injected for official distill batching when batch size > 1",
    )
    parser.add_argument(
        "--service-workers",
        type=int,
        default=1,
        help=(
            "Parallel workers for service prepare phase (default: 1). "
            "Use 2-4 to reduce wall-clock time."
        ),
    )
    parser.add_argument(
        "--distill-batch-sessions",
        type=int,
        default=2,
        help=(
            "When pipeline=service and service-write-mode=distill, merge this many sessions per "
            "extraction call to speed up ingestion (default: 2, aligned with LightMem longmemeval)"
        ),
    )
    parser.add_argument(
        "--reuse-service-ingest",
        action="store_true",
        default=True,
        help=(
            "Reuse previously ingested per-instance service data when available "
            "(default: on; skipped when --force-reindex)"
        ),
    )
    parser.add_argument(
        "--no-reuse-service-ingest",
        dest="reuse_service_ingest",
        action="store_false",
        help="Always re-ingest service data per instance",
    )
    parser.add_argument("--config", default="hybrid",
                        help="Search config (fts_only, vector_only, hybrid, hybrid_mmr, hybrid_decay)")
    parser.add_argument("--limit", type=int, default=48,
                        help="Max LongMemEval instances (default: 48)")
    parser.add_argument("--balanced", action="store_true",
                        help="Balance instances across question types")
    parser.add_argument("--max-results", type=int, default=5,
                        help="Top-K retrieved chunks to include in context")
    parser.add_argument("--search-k", type=int, default=None,
                        help="Top-K retrieval depth before QA post-processing (default: 10 for service+distill, else 30)")
    parser.add_argument("--answer-top-k", type=int, default=None,
                        help="Number of chunks passed to answer generation (default: 3 for service+distill, else 10)")
    parser.add_argument("--context-lines", type=int, default=60,
                        help="Extra lines before/after hit chunk when building evidence (manager mode)")
    parser.add_argument("--diversify-paths", action="store_true", default=True,
                        help="Promote distinct source paths for multi-evidence questions (default: on)")
    parser.add_argument("--no-diversify-paths", dest="diversify_paths", action="store_false",
                        help="Disable path diversity reranking")
    parser.add_argument("--max-chars", type=int, default=2000,
                        help="Max chars per retrieved chunk in context")
    parser.add_argument("--cache-dir", default="tests/benchmark/.cache/longmemeval_qa",
                        help="Persist per-instance workspace/DB to speed up reruns")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable cache and use temporary workspaces")
    parser.add_argument("--force-reindex", action="store_true",
                        help="Force reindex even if cached DB exists")
    parser.add_argument(
        "--skip-sync-if-cached",
        action="store_true",
        default=True,
        help="Skip mgr.sync when cached DB already has chunks (default: on)",
    )
    parser.add_argument(
        "--no-skip-sync-if-cached",
        dest="skip_sync_if_cached",
        action="store_false",
        help="Always run mgr.sync for each instance",
    )
    parser.add_argument(
        "--reuse-retrieval-cache",
        action="store_true",
        default=True,
        help="Reuse cached retrieval/evidence JSON when available (default: on)",
    )
    parser.add_argument(
        "--no-reuse-retrieval-cache",
        dest="reuse_retrieval_cache",
        action="store_false",
        help="Disable retrieval cache reuse and rerun search",
    )
    parser.add_argument("--output", default="tests/benchmark/results_longmemeval_qa.json",
                        help="Write JSON report to file")
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="After writing QA JSON, regenerate docs/benchmark-report.md",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Service pipeline only: prepare/ingest memory and exit without QA scoring",
    )
    parser.add_argument(
        "--read-answer-only",
        action="store_true",
        help="Service pipeline only: skip ingest and run retrieval+answer on prepared data",
    )
    args = parser.parse_args()

    _load_dotenv()
    if args.distill_batch_sessions < 1:
        raise SystemExit("--distill-batch-sessions must be >= 1")
    if args.service_workers < 1:
        raise SystemExit("--service-workers must be >= 1")
    if args.service_drain_queue_limit < 1:
        raise SystemExit("--service-drain-queue-limit must be >= 1")
    if args.service_drain_queue_max_attempts < 1:
        raise SystemExit("--service-drain-queue-max-attempts must be >= 1")
    if args.prepare_only and args.read_answer_only:
        raise SystemExit("--prepare-only and --read-answer-only are mutually exclusive")

    default_service_distill = args.pipeline == "service" and args.service_write_mode == "distill"
    if args.search_k is None:
        args.search_k = 10 if default_service_distill else 30
    if args.answer_top_k is None:
        args.answer_top_k = 3 if default_service_distill else 10
    if args.search_k < 1:
        raise SystemExit("--search-k must be >= 1")
    if args.answer_top_k < 1:
        raise SystemExit("--answer-top-k must be >= 1")

    normalized_service_resolver_mode = _normalize_service_resolver_mode(
        args.service_resolver_mode,
        args.service_write_mode,
    )
    normalized_service_drain_mode = _effective_service_drain_queue_mode(
        args.service_drain_queue_mode,
        args.service_write_mode,
        normalized_service_resolver_mode,
    )

    pg_dsn = args.pg_dsn or os.environ.get("OPENCLAW_PG_DSN")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No OpenAI API key. Use --api-key or set OPENAI_API_KEY.")
        sys.exit(1)

    if not LONGMEMEVAL_FILE.exists():
        print(f"ERROR: LongMemEval not found at {LONGMEMEVAL_FILE}")
        sys.exit(1)

    all_lme = json.loads(LONGMEMEVAL_FILE.read_text(encoding="utf-8"))
    instances = _select_instances(all_lme, args.limit, args.balanced)

    print(f"Loading LongMemEval... {len(instances)} instances")
    if args.balanced:
        dist = Counter(d["question_type"] for d in instances)
        for qt, cnt in sorted(dist.items()):
            print(f"  {qt}: {cnt}")

    answer_client = openai.OpenAI(api_key=api_key)

    if args.pipeline == "service":
        if not pg_dsn:
            print("ERROR: --pipeline service requires --pg-dsn or OPENCLAW_PG_DSN.")
            sys.exit(1)
        conn = get_pg_connection(pg_dsn)
        try:
            ensure_pg_schema(conn)
        finally:
            conn.close()
        if args.service_write_mode == "distill":
            print(
                "Service distill config: "
                f"lightmem={int(args.service_lightmem)} "
                f"resolver={normalized_service_resolver_mode} "
                f"drain_queue={normalized_service_drain_mode} "
                f"batch_sessions={args.distill_batch_sessions} "
                f"official_batching={int(args.service_official_batching)} "
                f"workers={args.service_workers}",
                flush=True,
            )
    elif args.prepare_only or args.read_answer_only:
        raise SystemExit("--prepare-only/--read-answer-only require --pipeline service")

    provider_names = _providers_from_arg(args.provider)
    results: list[dict[str, Any]] = []

    for provider_name in provider_names:
        config_name = args.config
        if provider_name == "none" and config_name != "fts_only":
            print("Note: provider=none forces config=fts_only for retrieval.")
            config_name = "fts_only"

        if provider_name == "openai":
            print(
                f"Verifying OpenAI API (embedding model={args.embedding_model})...",
                flush=True,
            )
            try:
                emb_provider = OpenAIEmbeddingProvider(
                    model=args.embedding_model, api_key=api_key, base_url=None, headers={}
                )
                test_vec = emb_provider.embed_query("test")
                print(f"  OK! dim={len(test_vec)}")
                if args.pipeline == "service" and len(test_vec) != PGVECTOR_DIMS:
                    raise SystemExit(
                        "ERROR: service pipeline expects 1536-dim pgvector embeddings. "
                        f"Got {len(test_vec)} from model {args.embedding_model}. "
                        "Use text-embedding-3-small or migrate schema dims."
                    )
            except Exception as e:
                print(f"  FAILED: {e}")
                sys.exit(1)
            provider_obj = emb_provider
        elif provider_name == "mock":
            provider_obj = MockEmbeddingProvider()
            if args.pipeline == "service":
                provider_obj = _PaddedEmbeddingProvider(provider_obj, PGVECTOR_DIMS)
                print("  Using padded mock embeddings for pgvector(1536) compatibility.")
        else:
            provider_obj = None

        if args.pipeline == "service" and provider_obj is None:
            print("Note: service pipeline requires embeddings; skipping provider=none.")
            continue

        provider_label = _label_for_provider(provider_name, args.embedding_model)
        print(f"\n== QA Run: provider={provider_label} config={config_name} ==", flush=True)
        if args.judge == "longmemeval" and not args.judge_model.startswith("gpt-4o"):
            raise SystemExit(
                "ERROR: LongMemEval official judge requires gpt-4o or gpt-4o-mini. "
                "Use --judge-model gpt-4o-mini."
            )
        cache_dir = None if args.no_cache else Path(args.cache_dir)
        if args.prepare_only:
            def _answer_llm_fn(prompt: str) -> str:
                return _openai_text(
                    answer_client,
                    args.answer_model,
                    "You must return valid JSON only.",
                    prompt,
                )

            result = prepare_longmemeval_service_data(
                instances=instances,
                provider=provider_obj,
                provider_label=provider_label,
                pg_dsn=pg_dsn or "",
                llm_fn=_answer_llm_fn,
                service_write_mode=args.service_write_mode,
                distill_batch_sessions=args.distill_batch_sessions,
                service_lightmem=args.service_lightmem,
                service_resolver_mode=normalized_service_resolver_mode,
                service_drain_queue_mode=normalized_service_drain_mode,
                service_drain_queue_limit=args.service_drain_queue_limit,
                service_drain_queue_max_attempts=args.service_drain_queue_max_attempts,
                service_official_batching=args.service_official_batching,
                service_batch_system_prompt=args.service_batch_system_prompt,
                service_workers=args.service_workers,
                force_reindex=args.force_reindex,
                reuse_service_ingest=args.reuse_service_ingest,
            )
        else:
            result = run_longmemeval_qa(
                instances=instances,
                provider=provider_obj,
                provider_label=provider_label,
                answer_client=answer_client,
                answer_model=args.answer_model,
                judge=args.judge,
                judge_model=args.judge_model,
                config_name=config_name,
                max_results=args.max_results,
                max_chars=args.max_chars,
                search_k=args.search_k,
                answer_top_k=args.answer_top_k,
                context_lines=args.context_lines,
                diversify_paths=args.diversify_paths,
                cache_dir=cache_dir,
                force_reindex=args.force_reindex,
                skip_sync_if_cached=args.skip_sync_if_cached,
                reuse_retrieval_cache=args.reuse_retrieval_cache,
                pipeline=args.pipeline,
                pg_dsn=pg_dsn,
                service_write_mode=args.service_write_mode,
                distill_batch_sessions=args.distill_batch_sessions,
                service_lightmem=args.service_lightmem,
                service_resolver_mode=normalized_service_resolver_mode,
                service_drain_queue_mode=normalized_service_drain_mode,
                service_drain_queue_limit=args.service_drain_queue_limit,
                service_drain_queue_max_attempts=args.service_drain_queue_max_attempts,
                service_official_batching=args.service_official_batching,
                service_batch_system_prompt=args.service_batch_system_prompt,
                reuse_service_ingest=args.reuse_service_ingest,
                read_answer_only=args.read_answer_only,
            )
        results.append(result)

    report = {
        "longmemeval_qa": results,
        "embedding_model": args.embedding_model,
        "answer_model": args.answer_model,
        "judge": args.judge,
        "judge_model": args.judge_model if args.judge == "openai" else None,
        "limit": args.limit,
        "balanced": args.balanced,
        "max_results": args.max_results,
        "search_k": args.search_k,
        "answer_top_k": args.answer_top_k,
        "context_lines": args.context_lines,
        "diversify_paths": args.diversify_paths,
        "max_chars": args.max_chars,
        "pipeline": args.pipeline,
        "service_write_mode": args.service_write_mode if args.pipeline == "service" else None,
        "distill_batch_sessions": (
            args.distill_batch_sessions
            if args.pipeline == "service" and args.service_write_mode == "distill"
            else None
        ),
        "service_lightmem": (
            args.service_lightmem
            if args.pipeline == "service" and args.service_write_mode == "distill"
            else None
        ),
        "service_resolver_mode": (
            normalized_service_resolver_mode
            if args.pipeline == "service" and args.service_write_mode == "distill"
            else None
        ),
        "service_drain_queue_mode": normalized_service_drain_mode,
        "service_drain_queue_limit": (
            args.service_drain_queue_limit
            if args.pipeline == "service" and args.service_write_mode == "distill"
            else None
        ),
        "service_drain_queue_max_attempts": (
            args.service_drain_queue_max_attempts
            if args.pipeline == "service" and args.service_write_mode == "distill"
            else None
        ),
        "service_official_batching": (
            args.service_official_batching
            if args.pipeline == "service" and args.service_write_mode == "distill"
            else None
        ),
        "service_batch_system_prompt": (
            args.service_batch_system_prompt
            if args.pipeline == "service" and args.service_write_mode == "distill"
            else None
        ),
        "service_workers": args.service_workers if args.pipeline == "service" else None,
        "reuse_service_ingest": args.reuse_service_ingest,
        "prepare_only": args.prepare_only,
        "read_answer_only": args.read_answer_only,
        "pg_dsn_set": bool(pg_dsn),
        "skip_sync_if_cached": args.skip_sync_if_cached,
        "reuse_retrieval_cache": args.reuse_retrieval_cache,
    }

    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nReport saved to {args.output}")

    if args.update_report:
        print("Updating docs/benchmark-report.md...", flush=True)
        subprocess.run(
            [sys.executable, "scripts/generate_benchmark_report.py",
             "--qa-input", args.output],
            check=False,
        )
    print("Done.")


if __name__ == "__main__":
    main()
