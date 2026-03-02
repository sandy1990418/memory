#!/usr/bin/env bash
# quality.sh — Run lint, type-check, and unit tests.
# Exit non-zero if any step fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "=== ruff check ==="
ruff check src tests
echo "ruff: OK"

echo ""
echo "=== mypy ==="
mypy src/openclaw_memory
echo "mypy: OK"

echo ""
echo "=== pytest ==="
pytest -q
echo "pytest: OK"

echo ""
echo "=== benchmark ==="
bash "${SCRIPT_DIR}/benchmark.sh"
echo "benchmark: OK"

echo ""
echo "All quality checks passed."
