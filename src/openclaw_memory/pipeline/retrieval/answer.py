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

        # Retry if abstained with evidence available
        if payload.abstain and evidence_chunks:
            retry_prompt = build_answer_prompt(query, evidence_chunks, retry=True)
            retry_response = llm_fn(retry_prompt)
            return parse_answer_response(retry_response)

        return payload
    except Exception:
        return AnswerPayload(
            answer="",
            evidence=[],
            confidence=0.0,
            abstain=True,
            abstain_reason="Failed to generate answer",
        )
