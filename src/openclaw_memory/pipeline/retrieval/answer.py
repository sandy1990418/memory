"""
RAG answer generation — structured LLM answers grounded in memory evidence.

Provides:
  - EvidenceItem / AnswerPayload dataclasses
  - build_answer_prompt  — formats query + evidence for the LLM
  - parse_answer_response — safely parses LLM output to AnswerPayload
  - generate_answer       — full pipeline: prompt -> LLM -> parse -> validate
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    memory_id: str
    quote: str
    reason: str


@dataclass
class AnswerPayload:
    answer: str
    evidence: list[EvidenceItem]
    confidence: float  # 0.0 - 1.0
    abstain: bool
    abstain_reason: str = ""


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ANSWER_PROMPT = """\
You are a memory-augmented assistant. Answer the user's question using ONLY
the provided evidence chunks.

## User question
{query}

## Evidence
{evidence_block}

## Instructions
- Base your answer ONLY on the evidence above.
- If evidence conflicts, prefer the most recent and highest-confidence fact.
- For counting or temporal questions, combine multiple evidence snippets.
- For recommendation questions, anchor the answer to user-specific preferences.
- Set "abstain": true ONLY when no evidence supports any concrete answer.
- Return strict JSON (no markdown fences, no extra keys):

{{
  "answer": "<concise answer string>",
  "evidence": [
    {{
      "memory_id": "<id from evidence>",
      "quote": "<short verbatim span supporting the answer>",
      "reason": "<one sentence explaining relevance>"
    }}
  ],
  "confidence": <float 0.0-1.0>,
  "abstain": <true|false>,
  "abstain_reason": "<only required when abstain is true>"
}}
"""

_ANSWER_PROMPT_RETRY = """\
You are a memory-augmented assistant. Re-evaluate the question using ONLY the
provided evidence chunks and produce the best-supported answer.

## User question
{query}

## Evidence
{evidence_block}

## Instructions
- Do NOT abstain if at least one evidence chunk supports a concrete answer.
- If evidence conflicts, choose the most recent supported fact and cite it.
- For counting/temporal questions, aggregate across relevant chunks.
- Only set "abstain": true when there is truly no supporting evidence.
- Return strict JSON (no markdown fences, no extra keys):

{{
  "answer": "<concise answer string>",
  "evidence": [
    {{
      "memory_id": "<id from evidence>",
      "quote": "<short verbatim span supporting the answer>",
      "reason": "<one sentence explaining relevance>"
    }}
  ],
  "confidence": <float 0.0-1.0>,
  "abstain": <true|false>,
  "abstain_reason": "<only required when abstain is true>"
}}
"""

_DEFAULT_ABSTAIN_REASON = "Insufficient evidence to answer."

_ABSTAIN_PARSE_FAILURE = AnswerPayload(
    answer="",
    evidence=[],
    confidence=0.0,
    abstain=True,
    abstain_reason="Failed to parse answer",
)
_PARSE_FAILURE_REASON = _ABSTAIN_PARSE_FAILURE.abstain_reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from noisy LLM output."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _format_evidence(evidence_chunks: list[dict[str, Any]]) -> str:
    """Format evidence chunks for the prompt."""
    if not evidence_chunks:
        return "(no evidence available)"
    lines: list[str] = []
    for i, chunk in enumerate(evidence_chunks):
        mem_id = chunk.get("id") or chunk.get("path") or f"chunk_{i}"
        snippet = chunk.get("snippet", chunk.get("content", ""))
        score = chunk.get("score", 0.0)
        lines.append(f"[{i + 1}] id={mem_id!r}  score={score:.3f}\n{snippet}")
    return "\n\n".join(lines)


def _is_parse_failure(payload: AnswerPayload) -> bool:
    return payload.abstain and payload.abstain_reason == _PARSE_FAILURE_REASON


def _build_parse_repair_prompt(
    query: str,
    evidence_chunks: list[dict[str, Any]],
    previous_output: str,
) -> str:
    evidence_block = _format_evidence(evidence_chunks)
    return f"""\
You must repair a malformed answer into STRICT JSON with this schema only:
{{
  "answer": "<concise answer string>",
  "evidence": [
    {{
      "memory_id": "<id from evidence>",
      "quote": "<short verbatim span supporting the answer>",
      "reason": "<one sentence explaining relevance>"
    }}
  ],
  "confidence": <float 0.0-1.0>,
  "abstain": <true|false>,
  "abstain_reason": "<only required when abstain is true>"
}}

Rules:
- Output ONLY valid JSON. No markdown, no prose.
- Keep answer grounded in evidence.
- If evidence includes a concrete answer, set "abstain": false.

Question:
{query}

Evidence:
{evidence_block}

Previous malformed output:
{previous_output}
"""


def _candidate_line_scores(query: str, text: str) -> list[tuple[int, str]]:
    query_terms = {
        tok
        for tok in re.findall(r"[a-z0-9$]+", query.lower())
        if len(tok) >= 3
    }
    out: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(("session:", "date:")):
            continue
        if low.startswith(("user:", "assistant:")) and ":" in line:
            line = line.split(":", 1)[1].strip()
            low = line.lower()
            if not line:
                continue
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if p.strip()]
        if not parts:
            parts = [line]
        for part in parts:
            p_low = part.lower()
            score = sum(1 for t in query_terms if t in p_low)
            out.append((score, part))
    return out


def _best_effort_non_abstain(
    query: str,
    evidence_chunks: list[dict[str, Any]],
) -> AnswerPayload:
    best_score = -1
    best_line = ""
    best_mem_id = ""

    for chunk in evidence_chunks[:3]:
        mem_id = str(chunk.get("id") or chunk.get("path") or "chunk_0")
        snippet = str(chunk.get("snippet", chunk.get("content", "")) or "")
        for score, line in _candidate_line_scores(query, snippet):
            if score > best_score or (score == best_score and best_line and len(line) < len(best_line)):
                best_score = score
                best_line = line
                best_mem_id = mem_id

    if not best_line:
        top = evidence_chunks[0] if evidence_chunks else {}
        best_mem_id = str(top.get("id") or top.get("path") or "chunk_0")
        snippet = str(top.get("snippet", top.get("content", "")) or "").strip()
        best_line = snippet.splitlines()[0].strip() if snippet else "Insufficient structured output."

    if len(best_line) > 260:
        best_line = best_line[:260].rstrip() + "..."

    return AnswerPayload(
        answer=best_line,
        evidence=[
            EvidenceItem(
                memory_id=best_mem_id,
                quote=best_line,
                reason="Best-effort fallback after repeated parse failures.",
            )
        ],
        confidence=0.2,
        abstain=False,
        abstain_reason="",
    )


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------


def build_answer_prompt(
    query: str,
    evidence_chunks: list[dict[str, Any]],
    *,
    retry: bool = False,
) -> str:
    """Build the LLM prompt for structured answer generation."""
    evidence_block = _format_evidence(evidence_chunks)
    template = _ANSWER_PROMPT_RETRY if retry else _ANSWER_PROMPT
    return template.format(query=query, evidence_block=evidence_block)


# ---------------------------------------------------------------------------
# Validate + Parse
# ---------------------------------------------------------------------------


def validate_answer_payload(data: dict[str, Any]) -> AnswerPayload:
    """Validate a raw dict into an AnswerPayload, repairing where possible."""
    required = {"answer", "evidence", "confidence", "abstain"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    answer = str(data["answer"])

    raw_evidence = data["evidence"]
    if not isinstance(raw_evidence, list):
        raise ValueError("'evidence' must be a list")

    evidence: list[EvidenceItem] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        evidence.append(EvidenceItem(
            memory_id=str(item.get("memory_id", "")),
            quote=str(item.get("quote", "")),
            reason=str(item.get("reason", "")),
        ))

    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    abstain = bool(data["abstain"])
    abstain_reason = str(data.get("abstain_reason", ""))

    if not evidence and not abstain:
        abstain = True
    if abstain and not abstain_reason:
        abstain_reason = _DEFAULT_ABSTAIN_REASON

    return AnswerPayload(
        answer=answer,
        evidence=evidence,
        confidence=confidence,
        abstain=abstain,
        abstain_reason=abstain_reason,
    )


def parse_answer_response(response: str) -> AnswerPayload:
    """Parse raw LLM response into AnswerPayload. Never raises."""
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    if not text:
        return _ABSTAIN_PARSE_FAILURE

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(text)
        if extracted is None:
            return _ABSTAIN_PARSE_FAILURE
        data = extracted

    if not isinstance(data, dict):
        return _ABSTAIN_PARSE_FAILURE

    try:
        return validate_answer_payload(data)
    except (ValueError, TypeError):
        return _ABSTAIN_PARSE_FAILURE


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_answer(
    query: str,
    evidence_chunks: list[dict[str, Any]],
    llm_fn: Callable[[str], str],
) -> AnswerPayload:
    """
    Full answer pipeline: build prompt -> call LLM -> parse -> validate.

    Retries once if the LLM abstains despite available evidence.
    """
    try:
        prompt = build_answer_prompt(query, evidence_chunks)
        response = llm_fn(prompt)
        payload = parse_answer_response(response)
        if _is_parse_failure(payload):
            repaired = llm_fn(_build_parse_repair_prompt(query, evidence_chunks, response))
            payload = parse_answer_response(repaired)

        # Retry if abstained with evidence available
        if payload.abstain and evidence_chunks:
            retry_prompt = build_answer_prompt(query, evidence_chunks, retry=True)
            retry_response = llm_fn(retry_prompt)
            retry_payload = parse_answer_response(retry_response)
            if _is_parse_failure(retry_payload):
                repaired_retry = llm_fn(
                    _build_parse_repair_prompt(query, evidence_chunks, retry_response)
                )
                retry_payload = parse_answer_response(repaired_retry)
            if _is_parse_failure(retry_payload):
                return _best_effort_non_abstain(query, evidence_chunks)
            return retry_payload

        return payload
    except Exception:
        return AnswerPayload(
            answer="",
            evidence=[],
            confidence=0.0,
            abstain=True,
            abstain_reason="Failed to generate answer",
        )
