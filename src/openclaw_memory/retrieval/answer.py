"""
Answer contract — structured LLM answer output schema, prompt builder, validator, and parser.

This module is self-contained (stdlib + typing only).  It defines:

- EvidenceItem / AnswerPayload dataclasses
- build_answer_prompt    — formats query + evidence into an LLM prompt
- validate_answer_payload — validates and repairs a raw dict → AnswerPayload
- parse_answer_response  — parses LLM JSON string → AnswerPayload (safe, never raises)
- generate_answer        — full orchestration: prompt → LLM → parse → validate
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
    memory_id: str  # memory path or ID
    quote: str      # short span from the memory supporting the answer
    reason: str     # why this evidence supports the answer


@dataclass
class AnswerPayload:
    answer: str
    evidence: list[EvidenceItem]
    confidence: float   # 0.0–1.0
    abstain: bool
    abstain_reason: str = ""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_ANSWER_PROMPT_TEMPLATE = """\
You are a memory-augmented assistant. Answer the user's question using ONLY
the provided evidence chunks.

## User question
{query}

## Evidence
{evidence_block}

## Instructions
- Base your answer ONLY on the evidence above.
- If evidence conflicts, prefer the most likely latest/updated fact:
  higher score, clearer direct statement, and newer-looking source id/path.
- For counting or temporal questions, combine multiple evidence snippets when needed.
- For questions about "latest/current/updated/changed", prioritize explicit dates/times
  and the newest supported statement.
- For recommendation/personalized questions, explicitly anchor the answer to user-specific
  preferences found in evidence (avoid generic advice).
- Set "abstain": true ONLY when no evidence supports any concrete answer.
- Return strict JSON matching this schema (no markdown fences, no extra keys):

{{
  "answer": "<concise answer string>",
  "evidence": [
    {{
      "memory_id": "<id or path from evidence>",
      "quote": "<short verbatim span that supports the answer>",
      "reason": "<one sentence explaining why this evidence supports the answer>"
    }}
  ],
  "confidence": <float 0.0-1.0>,
  "abstain": <true|false>,
  "abstain_reason": "<only required when abstain is true>"
}}
"""

_ANSWER_PROMPT_TEMPLATE_RETRY = """\
You are a memory-augmented assistant. Re-evaluate the question using ONLY the
provided evidence chunks and produce the best-supported answer.

## User question
{query}

## Evidence
{evidence_block}

## Instructions
- Do NOT abstain if at least one evidence chunk supports a concrete answer.
- If evidence conflicts, choose the most likely latest/updated fact and cite it.
- For counting/temporal questions, aggregate across relevant chunks.
- For latest/current/updated questions, select the most recent supported fact.
- For personalized recommendation questions, use concrete user preferences from evidence.
- Only set "abstain": true when there is truly no supporting evidence.
- Return strict JSON matching this schema (no markdown fences, no extra keys):

{{
  "answer": "<concise answer string>",
  "evidence": [
    {{
      "memory_id": "<id or path from evidence>",
      "quote": "<short verbatim span that supports the answer>",
      "reason": "<one sentence explaining why this evidence supports the answer>"
    }}
  ],
  "confidence": <float 0.0-1.0>,
  "abstain": <true|false>,
  "abstain_reason": "<only required when abstain is true>"
}}
"""

def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction for model outputs that wrap JSON with extra text."""
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
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                if isinstance(parsed, dict):
                    return parsed
                return None
    return None


def build_answer_prompt(
    query: str,
    evidence_chunks: list[dict[str, Any]],
    *,
    retry: bool = False,
) -> str:
    """
    Build an LLM prompt that requests a structured JSON answer.

    Args:
        query:           The user's natural-language question.
        evidence_chunks: List of dicts, each with at minimum keys:
                         ``path`` (or ``id``), ``snippet``, ``score``.

    Returns:
        A ready-to-send prompt string.
    """
    lines: list[str] = []
    for i, chunk in enumerate(evidence_chunks):
        mem_id = chunk.get("path") or chunk.get("id") or f"chunk_{i}"
        snippet = chunk.get("snippet", "")
        score = chunk.get("score", 0.0)
        lines.append(f"[{i + 1}] id={mem_id!r}  score={score:.3f}\n{snippet}")

    evidence_block = "\n\n".join(lines) if lines else "(no evidence available)"
    template = _ANSWER_PROMPT_TEMPLATE_RETRY if retry else _ANSWER_PROMPT_TEMPLATE
    return template.format(query=query, evidence_block=evidence_block)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_DEFAULT_ABSTAIN_REASON = "Insufficient evidence to answer."


def validate_answer_payload(data: dict[str, Any]) -> AnswerPayload:
    """
    Validate a raw dict and return an AnswerPayload.

    Coerces / repairs where possible:
    - Empty evidence + abstain=False → force abstain=True
    - abstain=True with no abstain_reason → fill default reason
    - confidence clamped to [0.0, 1.0]

    Raises:
        ValueError: if required top-level keys are missing.
    """
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
        evidence.append(
            EvidenceItem(
                memory_id=str(item.get("memory_id", "")),
                quote=str(item.get("quote", "")),
                reason=str(item.get("reason", "")),
            )
        )

    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    abstain = bool(data["abstain"])
    abstain_reason = str(data.get("abstain_reason", ""))

    # Repair: force abstain when evidence is empty
    if not evidence and not abstain:
        abstain = True

    # Repair: fill default abstain_reason when abstaining without a reason
    if abstain and not abstain_reason:
        abstain_reason = _DEFAULT_ABSTAIN_REASON

    return AnswerPayload(
        answer=answer,
        evidence=evidence,
        confidence=confidence,
        abstain=abstain,
        abstain_reason=abstain_reason,
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_ABSTAIN_PARSE_FAILURE = AnswerPayload(
    answer="",
    evidence=[],
    confidence=0.0,
    abstain=True,
    abstain_reason="Failed to parse answer",
)


def parse_answer_response(response: str) -> AnswerPayload:
    """
    Parse the raw LLM response string into an AnswerPayload.

    - Strips markdown code fences.
    - Calls validate_answer_payload.
    - Never raises; returns abstain payload on any failure.
    """
    text = response.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    if not text:
        return _ABSTAIN_PARSE_FAILURE

    data: Any
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
    Full pipeline: build prompt → call LLM → parse → validate.

    Args:
        query:           The user's question.
        evidence_chunks: Search results to use as evidence.
        llm_fn:          Callable that takes a prompt string and returns a response string.

    Returns:
        AnswerPayload — guaranteed non-raising; abstains on any failure.
    """
    try:
        prompt = build_answer_prompt(query, evidence_chunks)
        response = llm_fn(prompt)
        payload = parse_answer_response(response)
        if payload.abstain and evidence_chunks:
            retry_prompt = build_answer_prompt(query, evidence_chunks, retry=True)
            retry_response = llm_fn(retry_prompt)
            retry_payload = parse_answer_response(retry_response)
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
