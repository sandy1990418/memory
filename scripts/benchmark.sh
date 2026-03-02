#!/usr/bin/env bash
# benchmark.sh — Run LongMemEval with OpenAI embeddings and update the report.
# Exit non-zero if the benchmark or report generation fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_FILE="${REPO_ROOT}/tests/benchmark/results_real_vs_mock.json"
REPORT_GENERATOR="${SCRIPT_DIR}/generate_benchmark_report.py"

cd "${REPO_ROOT}"

echo "=== Benchmark: LongMemEval (OpenAI embeddings) ==="
python tests/benchmark/run_real_embedding_benchmark.py \
    --longmemeval \
    --limit 50 \
    --output "${RESULTS_FILE}"
echo "LongMemEval complete. Results written to ${RESULTS_FILE}"

echo ""
echo "=== Benchmark: Generating Report ==="
python "${REPORT_GENERATOR}" \
    --input "${RESULTS_FILE}" \
    --output docs/benchmark-report.md
echo "Report generated at docs/benchmark-report.md"

echo ""
echo "Benchmark: OK"
