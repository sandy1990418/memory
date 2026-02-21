"""
Temporal decay scoring for memory search results.
Mirrors: src/memory/temporal-decay.ts
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class TemporalDecayConfig:
    enabled: bool = False
    half_life_days: float = 30.0


DEFAULT_TEMPORAL_DECAY_CONFIG = TemporalDecayConfig(enabled=False, half_life_days=30.0)

_DAY_MS = 24 * 60 * 60 * 1000
_DATED_MEMORY_PATH_RE = re.compile(r"(?:^|/)memory/(\d{4})-(\d{2})-(\d{2})\.md$")


def to_decay_lambda(half_life_days: float) -> float:
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        return 0.0
    return math.log(2) / half_life_days


def calculate_temporal_decay_multiplier(age_in_days: float, half_life_days: float) -> float:
    lam = to_decay_lambda(half_life_days)
    clamped = max(0.0, age_in_days)
    if lam <= 0 or not math.isfinite(clamped):
        return 1.0
    return math.exp(-lam * clamped)


def apply_temporal_decay_to_score(score: float, age_in_days: float, half_life_days: float) -> float:
    return score * calculate_temporal_decay_multiplier(age_in_days, half_life_days)


def _parse_memory_date_from_path(file_path: str) -> datetime | None:
    normalized = file_path.replace("\\", "/").lstrip("./")
    m = _DATED_MEMORY_PATH_RE.search(normalized)
    if not m:
        return None
    try:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        dt = datetime(year, month, day, tzinfo=timezone.utc)
        if dt.year != year or dt.month != month or dt.day != day:
            return None
        return dt
    except (ValueError, OverflowError):
        return None


def _is_evergreen_memory_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lstrip("./")
    if normalized in ("MEMORY.md", "memory.md"):
        return True
    if not normalized.startswith("memory/"):
        return False
    return not _DATED_MEMORY_PATH_RE.search(normalized)


def _extract_timestamp(
    file_path: str,
    source: str | None,
    workspace_dir: str | None,
) -> datetime | None:
    from_path = _parse_memory_date_from_path(file_path)
    if from_path:
        return from_path

    if source == "memory" and _is_evergreen_memory_path(file_path):
        return None

    if not workspace_dir:
        return None

    abs_path = (
        file_path
        if os.path.isabs(file_path)
        else os.path.join(workspace_dir, file_path)
    )
    try:
        mtime = os.stat(abs_path).st_mtime
        if not math.isfinite(mtime):
            return None
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    except OSError:
        return None


def apply_temporal_decay_to_results(
    results: list[dict],
    *,
    config: TemporalDecayConfig | None = None,
    workspace_dir: str | None = None,
    now_ms: float | None = None,
) -> list[dict]:
    """
    Apply temporal decay to a list of result dicts.
    Each dict must have: path, score, source.
    Mirrors: temporal-decay.ts::applyTemporalDecayToHybridResults
    """
    cfg = config or DEFAULT_TEMPORAL_DECAY_CONFIG
    if not cfg.enabled:
        return list(results)

    import time as _time
    now_ms_actual = now_ms if now_ms is not None else _time.time() * 1000

    timestamp_cache: dict[str, datetime | None] = {}
    out: list[dict] = []

    for entry in results:
        file_path = entry.get("path", "")
        source = entry.get("source", "memory")
        cache_key = f"{source}:{file_path}"

        if cache_key not in timestamp_cache:
            timestamp_cache[cache_key] = _extract_timestamp(file_path, source, workspace_dir)

        ts = timestamp_cache[cache_key]
        if ts is None:
            out.append(entry)
            continue

        age_ms = max(0.0, now_ms_actual - ts.timestamp() * 1000)
        age_days = age_ms / _DAY_MS
        decayed_score = apply_temporal_decay_to_score(entry["score"], age_days, cfg.half_life_days)
        out.append({**entry, "score": decayed_score})

    return out
