"""
Generate/update docs/benchmark-report.md from LongMemEval (real embeddings) output.
Optionally include LongMemEval QA (end-to-end) results if provided.

Usage:
    python scripts/generate_benchmark_report.py
    python scripts/generate_benchmark_report.py --input tests/benchmark/results_real_vs_mock.json
    python scripts/generate_benchmark_report.py --output docs/benchmark-report.md
    python scripts/generate_benchmark_report.py --qa-input tests/benchmark/results_longmemeval_qa.json

Reads JSON produced by:
    python tests/benchmark/run_real_embedding_benchmark.py --longmemeval --limit 50
Optionally reads QA JSON produced by:
    python tests/benchmark/run_longmemeval_qa.py --limit 48 --balanced

No external dependencies — stdlib only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _best_by_metric(results: list[dict], key: str, fallback_key: str | None = None) -> dict:
    def _score(r: dict) -> float:
        agg = r.get("aggregate", {})
        if key in agg:
            return agg.get(key, 0.0)
        if fallback_key and fallback_key in agg:
            return agg.get(fallback_key, 0.0)
        return 0.0

    return max(results, key=_score)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_results_table(results: list[dict]) -> str:
    header = "| Provider | Config | Instances | MRR | Recall@1 | Recall@5 | nDCG@5 | Time (s) |"
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in results:
        a = r["aggregate"]
        rows.append(
            f"| {r['provider']} "
            f"| {r['config']} "
            f"| {r.get('instances', 0)} "
            f"| {_fmt(a['mrr'])} "
            f"| {_fmt(a.get('recall@1', 0.0))} "
            f"| {_fmt(a.get('recall@5', 0.0))} "
            f"| {_fmt(a.get('ndcg@5', 0.0))} "
            f"| {r.get('elapsed_s', 0.0):.1f} |"
        )
    return "\n".join(rows)


def _render_type_breakdown(best: dict) -> str:
    by_type: dict = best.get("by_type", {})
    if not by_type:
        return ""

    header = "| Question Type | Count | MRR | Recall@5 |"
    sep = "|---|---|---|---|"
    rows = [header, sep]
    for qt in sorted(by_type.keys()):
        td = by_type[qt]
        rows.append(
            f"| {qt} "
            f"| {int(td.get('count', 0))} "
            f"| {_fmt(td.get('mrr', 0.0))} "
            f"| {_fmt(td.get('recall@5', 0.0))} |"
        )
    return "\n".join(rows)


def _render_qa_results_table(results: list[dict]) -> str:
    has_evidence = any(
        "evidence_supported_rate" in r.get("aggregate", {}) for r in results
    )
    header = (
        "| Provider | Pipeline | Config | Instances | EM | F1 | Judge Acc | "
        "RetrievalHit@5 | Coverage@5 | AllHit@5 | Time (s) | Answer Model |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    if has_evidence:
        header += " EvidenceSupported | UnsupportedClaim | AbstentionPrec |"
        sep += "---|---|---|"
    rows = [header, sep]
    for r in results:
        a = r.get("aggregate", {})
        pipeline_label = str(r.get("pipeline", "-"))
        if pipeline_label == "service" and r.get("service_write_mode"):
            pipeline_label = f"service/{r.get('service_write_mode')}"
        line = (
            f"| {r.get('provider', '-')} "
            f"| {pipeline_label} "
            f"| {r.get('config', '-')} "
            f"| {r.get('instances', 0)} "
            f"| {_fmt(a.get('exact_match', 0.0))} "
            f"| {_fmt(a.get('f1', 0.0))} "
            f"| {_fmt(a.get('judge_acc', 0.0)) if 'judge_acc' in a else '-'} "
            f"| {_fmt(a.get('retrieval_hit@5', 0.0)) if 'retrieval_hit@5' in a else '-'} "
            f"| {_fmt(a.get('retrieval_coverage@5', 0.0)) if 'retrieval_coverage@5' in a else '-'} "
            f"| {_fmt(a.get('retrieval_all_hit@5', 0.0)) if 'retrieval_all_hit@5' in a else '-'} "
            f"| {r.get('elapsed_s', 0.0):.1f} "
            f"| {r.get('answer_model', '-')} |"
        )
        if has_evidence:
            line += (
                f" {_fmt(a['evidence_supported_rate']) if 'evidence_supported_rate' in a else '-'} "
                f"| {_fmt(a['unsupported_claim_rate']) if 'unsupported_claim_rate' in a else '-'} "
                f"| {_fmt(a['abstention_precision']) if 'abstention_precision' in a else '-'} |"
            )
        rows.append(line)
    return "\n".join(rows)


def _render_evidence_metrics_table(qa_results: list[dict]) -> str:
    """Render a standalone Evidence Support Metrics summary table."""
    rows = [
        "| Provider | Config | EvidenceSupported | UnsupportedClaim | AbstentionPrec |",
        "|---|---|---|---|---|",
    ]
    for r in qa_results:
        a = r.get("aggregate", {})
        ev_keys = ("evidence_supported_rate", "unsupported_claim_rate", "abstention_precision")
        if not any(k in a for k in ev_keys):
            continue
        rows.append(
            f"| {r.get('provider', '-')} "
            f"| {r.get('config', '-')} "
            f"| {_fmt(a['evidence_supported_rate']) if 'evidence_supported_rate' in a else '-'} "
            f"| {_fmt(a['unsupported_claim_rate']) if 'unsupported_claim_rate' in a else '-'} "
            f"| {_fmt(a['abstention_precision']) if 'abstention_precision' in a else '-'} |"
        )
    return "\n".join(rows) if len(rows) > 2 else ""


def _render_evidence_type_breakdown(best: dict) -> str:
    """Render per-type evidence metrics for the best QA run."""
    by_type: dict = best.get("by_type", {})
    if not by_type:
        return ""
    has_ev = any(
        "evidence_supported_rate" in v or "abstention_precision" in v
        for v in by_type.values()
    )
    if not has_ev:
        return ""
    rows = [
        "| Question Type | Count | EvidenceSupported | UnsupportedClaim | AbstentionPrec |",
        "|---|---|---|---|---|",
    ]
    for qt in sorted(by_type.keys()):
        td = by_type[qt]
        rows.append(
            f"| {qt} "
            f"| {int(td.get('count', 0))} "
            f"| {_fmt(td['evidence_supported_rate']) if 'evidence_supported_rate' in td else '-'} "
            f"| {_fmt(td['unsupported_claim_rate']) if 'unsupported_claim_rate' in td else '-'} "
            f"| {_fmt(td['abstention_precision']) if 'abstention_precision' in td else '-'} |"
        )
    return "\n".join(rows)


def _render_qa_type_breakdown(best: dict) -> str:
    by_type: dict = best.get("by_type", {})
    if not by_type:
        return ""

    has_judge = any("judge_acc" in v for v in by_type.values())
    has_retrieval = any("retrieval_hit@5" in v for v in by_type.values())
    has_coverage = any("retrieval_coverage@5" in v for v in by_type.values())
    has_all_hit = any("retrieval_all_hit@5" in v for v in by_type.values())
    header = "| Question Type | Count | EM | F1 |"
    sep = "|---|---|---|---|"
    if has_judge:
        header += " Judge Acc |"
        sep += "---|"
    if has_retrieval:
        header += " RetrievalHit@5 |"
        sep += "---|"
    if has_coverage:
        header += " Coverage@5 |"
        sep += "---|"
    if has_all_hit:
        header += " AllHit@5 |"
        sep += "---|"
    rows = [header, sep]
    for qt in sorted(by_type.keys()):
        td = by_type[qt]
        line = (
            f"| {qt} "
            f"| {int(td.get('count', 0))} "
            f"| {_fmt(td.get('exact_match', 0.0))} "
            f"| {_fmt(td.get('f1', 0.0))} |"
        )
        if has_judge:
            line += f" {_fmt(td.get('judge_acc', 0.0))} |"
        if has_retrieval:
            line += f" {_fmt(td.get('retrieval_hit@5', 0.0))} |"
        if has_coverage:
            line += f" {_fmt(td.get('retrieval_coverage@5', 0.0))} |"
        if has_all_hit:
            line += f" {_fmt(td.get('retrieval_all_hit@5', 0.0))} |"
        rows.append(line)
    return "\n".join(rows)


def _pick_openai(results: list[dict]) -> dict | None:
    for r in results:
        if str(r.get("provider", "")).startswith("openai/"):
            return r
    return None


def build_report(data: dict, generated_at: str, qa_data: dict | None = None) -> str:
    results = data.get("longmemeval", [])
    if not results:
        return "# Benchmark Report: LongMemEval (OpenAI)\n\n(No results found)\n"

    model = data.get("model", "text-embedding-3-small")
    best_overall = _best_by_metric(results, "mrr")
    best_openai = _pick_openai(results) or best_overall

    a = best_openai["aggregate"]

    lines = [
        "# Benchmark Report: LongMemEval (OpenAI)",
        "",
        f"*Generated: {generated_at}*",
        "",
        "> **This file is auto-generated** by `scripts/generate_benchmark_report.py`.",
        "> Do not edit manually — run `scripts/benchmark.sh` to regenerate.",
        "",
        "---",
        "",
        "## Dataset Summary",
        "",
        "- Dataset: `LongMemEval` (official benchmark)",
        f"- Instances evaluated: `{best_openai.get('instances', 0)}`",
        f"- Embedding model: `openai/{model}`",
        "",
        "## OpenAI Results (Primary)",
        "",
        f"**{best_openai['config']}** (provider: `{best_openai['provider']}`)",
        "",
        f"- MRR: **{_fmt(a['mrr'])}**",
        f"- Recall@1: **{_fmt(a.get('recall@1', 0.0))}**",
        f"- Recall@5: **{_fmt(a.get('recall@5', 0.0))}**",
        f"- nDCG@5: **{_fmt(a.get('ndcg@5', 0.0))}**",
        "",
        "---",
        "",
        "## Full LongMemEval Results",
        "",
        _render_results_table(results),
        "",
    ]

    type_table = _render_type_breakdown(best_openai)
    if type_table:
        lines.extend([
            "---",
            "",
            "## Per-Type Breakdown (OpenAI)",
            "",
            type_table,
            "",
        ])

    qa_results = qa_data.get("longmemeval_qa", []) if qa_data else []
    if qa_results:
        best_qa = _pick_openai(qa_results) or _best_by_metric(qa_results, "f1", "exact_match")
        qa_agg = best_qa.get("aggregate", {})
        lines.extend([
            "---",
            "",
            "## LongMemEval QA (End-to-End)",
            "",
            f"- Answer model: `{best_qa.get('answer_model', '-')}`",
            f"- Pipeline: `{best_qa.get('pipeline', '-')}`",
            (
                f"- Service write mode: `{best_qa.get('service_write_mode', '-')}`"
                if best_qa.get("pipeline") == "service"
                else "- Service write mode: `-`"
            ),
            (
                f"- Distill batch sessions: `{best_qa.get('distill_batch_sessions', '-')}`"
                if best_qa.get("pipeline") == "service"
                and best_qa.get("service_write_mode") == "distill"
                else "- Distill batch sessions: `-`"
            ),
            f"- Judge: `{best_qa.get('judge', 'exact')}`",
            f"- Instances evaluated: `{best_qa.get('instances', 0)}`",
            "",
            f"**{best_qa.get('config', '-') }** (provider: `{best_qa.get('provider', '-')}`)",
            "",
            f"- EM: **{_fmt(qa_agg.get('exact_match', 0.0))}**",
            f"- F1: **{_fmt(qa_agg.get('f1', 0.0))}**",
            f"- Judge Acc: **{_fmt(qa_agg.get('judge_acc', 0.0))}**" if "judge_acc" in qa_agg else "- Judge Acc: **-**",
            f"- RetrievalHit@5 (any): **{_fmt(qa_agg.get('retrieval_hit@5', 0.0))}**" if "retrieval_hit@5" in qa_agg else "- RetrievalHit@5 (any): **-**",
            f"- Coverage@5 (all-evidence fraction): **{_fmt(qa_agg.get('retrieval_coverage@5', 0.0))}**" if "retrieval_coverage@5" in qa_agg else "- Coverage@5 (all-evidence fraction): **-**",
            f"- AllHit@5 (strict all evidence): **{_fmt(qa_agg.get('retrieval_all_hit@5', 0.0))}**" if "retrieval_all_hit@5" in qa_agg else "- AllHit@5 (strict all evidence): **-**",
            "",
            "> `RetrievalHit@5` is lenient (any evidence hit). "
            "`Coverage@5` and `AllHit@5` are stricter and better explain multi-hop failures.",
            "",
            _render_qa_results_table(qa_results),
            "",
        ])

        qa_type_table = _render_qa_type_breakdown(best_qa)
        if qa_type_table:
            lines.extend([
                "---",
                "",
                "## LongMemEval QA Per-Type Breakdown",
                "",
                qa_type_table,
                "",
            ])

        evidence_table = _render_evidence_metrics_table(qa_results)
        if evidence_table:
            lines.extend([
                "---",
                "",
                "## Evidence Support Metrics",
                "",
                "> Heuristic evidence metrics: `evidence_supported_rate` is 1.0 when retrieval hit "
                "and answer is non-empty; `unsupported_claim_rate` is the inverse; "
                "`abstention_precision` applies only to abstention questions.",
                "",
                evidence_table,
                "",
            ])

        evidence_type_table = _render_evidence_type_breakdown(best_qa)
        if evidence_type_table:
            lines.extend([
                "---",
                "",
                "## Evidence Support Metrics Per-Type Breakdown",
                "",
                evidence_type_table,
                "",
            ])

    lines.extend([
        "---",
        "",
        "## How To Reproduce",
        "",
        "```bash",
        "# Requires OPENAI_API_KEY (env or .env)",
        "python tests/benchmark/run_real_embedding_benchmark.py --longmemeval --limit 50",
        "python scripts/generate_benchmark_report.py --input tests/benchmark/results_real_vs_mock.json --output docs/benchmark-report.md",
        "```",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate LongMemEval (OpenAI) benchmark report",
    )
    parser.add_argument(
        "--input",
        default=str(repo_root / "tests" / "benchmark" / "results_real_vs_mock.json"),
        help="Path to real-embedding benchmark JSON",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "docs" / "benchmark-report.md"),
        help="Output path for the report markdown",
    )
    parser.add_argument(
        "--qa-input",
        default=str(repo_root / "tests" / "benchmark" / "results_longmemeval_qa.json"),
        help="Optional LongMemEval QA JSON (end-to-end) to include",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    data = _load_json(input_path)
    if data is None:
        raise SystemExit(f"ERROR: Benchmark JSON not found: {input_path}")

    qa_data = None
    if args.qa_input:
        qa_path = Path(args.qa_input)
        qa_data = _load_json(qa_path)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = build_report(data, generated_at, qa_data=qa_data)

    out_path = Path(args.output)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
