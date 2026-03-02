#!/usr/bin/env bash
# Run LongMemEval QA with service pipeline in a consistent two-phase workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mode="${1:-full}"
if [[ $# -gt 0 ]]; then
  shift
fi

PG_DSN="${OPENCLAW_PG_DSN:-postgresql://memuser:mempass@localhost:5433/memory}"
PROVIDER="${LME_PROVIDER:-openai}"
CONFIG="${LME_CONFIG:-hybrid}"
OUTPUT="${LME_OUTPUT:-tests/benchmark/results_longmemeval_qa.json}"
LIMIT="${LME_LIMIT:-48}"
BALANCED="${LME_BALANCED:-0}"
DISTILL_BATCH="${LME_DISTILL_BATCH:-8}"
LIGHTMEM="${LME_LIGHTMEM:-1}"
RESOLVER_MODE="${LME_RESOLVER_MODE:-off}"
DRAIN_MODE="${LME_DRAIN_MODE:-never}"
OFFICIAL_BATCHING="${LME_OFFICIAL_BATCHING:-1}"
SERVICE_WORKERS="${LME_SERVICE_WORKERS:-2}"
AUTO_PREP_RETRY="${LME_AUTO_PREP_RETRY:-1}"
SEARCH_K="${LME_SEARCH_K:-10}"
ANSWER_TOP_K="${LME_ANSWER_TOP_K:-3}"
PREP_MODEL="${LME_PREP_MODEL:-gpt-4o-mini}"
QA_MODEL="${LME_QA_MODEL:-gpt-5-mini}"
JUDGE="${LME_JUDGE:-longmemeval}"
JUDGE_MODEL="${LME_JUDGE_MODEL:-gpt-4o-mini}"
FAST_MODE="${LME_FAST_MODE:-0}"

if [[ "${FAST_MODE}" == "1" ]]; then
  if [[ -z "${LME_QA_MODEL:-}" ]]; then
    QA_MODEL="gpt-4o-mini"
  fi
  if [[ -z "${LME_JUDGE:-}" ]]; then
    JUDGE="exact"
  fi
fi

usage() {
  cat <<EOF
Usage: bash scripts/run_longmemeval_service.sh [prepare|read|full] [extra benchmark flags...]

Modes:
  prepare  Distill write path only (--prepare-only)
  read     Retrieval + QA only (--read-answer-only)
  full     Run prepare then read (default)

Important:
  - This script keeps service knobs consistent across phases.
  - If you ran prepare with a different distill batch size, set LME_DISTILL_BATCH.

Common env overrides:
  OPENCLAW_PG_DSN=${PG_DSN}
  LME_LIMIT=${LIMIT}
  LME_DISTILL_BATCH=${DISTILL_BATCH}
  LME_PREP_MODEL=${PREP_MODEL}
  LME_QA_MODEL=${QA_MODEL}
  LME_RESOLVER_MODE=${RESOLVER_MODE}
  LME_DRAIN_MODE=${DRAIN_MODE}
  LME_SERVICE_WORKERS=${SERVICE_WORKERS}
  LME_AUTO_PREP_RETRY=${AUTO_PREP_RETRY}
  LME_OUTPUT=${OUTPUT}
  LME_FAST_MODE=${FAST_MODE}
EOF
}

if [[ "${mode}" == "-h" || "${mode}" == "--help" ]]; then
  usage
  exit 0
fi

base_cmd=(
  python tests/benchmark/run_longmemeval_qa.py
  --pipeline service
  --service-write-mode distill
  --pg-dsn "${PG_DSN}"
  --provider "${PROVIDER}"
  --config "${CONFIG}"
  --output "${OUTPUT}"
  --limit "${LIMIT}"
  --distill-batch-sessions "${DISTILL_BATCH}"
  --service-resolver-mode "${RESOLVER_MODE}"
  --service-drain-queue-mode "${DRAIN_MODE}"
  --service-workers "${SERVICE_WORKERS}"
  --search-k "${SEARCH_K}"
  --answer-top-k "${ANSWER_TOP_K}"
  --reuse-service-ingest
)

if [[ "${BALANCED}" == "1" ]]; then
  base_cmd+=(--balanced)
fi
if [[ "${LIGHTMEM}" == "1" ]]; then
  base_cmd+=(--service-lightmem)
else
  base_cmd+=(--no-service-lightmem)
fi
if [[ "${OFFICIAL_BATCHING}" == "1" ]]; then
  base_cmd+=(--service-official-batching)
else
  base_cmd+=(--no-service-official-batching)
fi

run_prepare() {
  echo "== Phase: prepare =="
  echo "prepare config: model=${PREP_MODEL} workers=${SERVICE_WORKERS} distill_batch=${DISTILL_BATCH}"
  "${base_cmd[@]}" \
    --prepare-only \
    --answer-model "${PREP_MODEL}" \
    "$@"
}

run_read() {
  echo "== Phase: read-answer =="
  echo "read config: model=${QA_MODEL} judge=${JUDGE}"
  "${base_cmd[@]}" \
    --read-answer-only \
    --answer-model "${QA_MODEL}" \
    --judge "${JUDGE}" \
    --judge-model "${JUDGE_MODEL}" \
    "$@"
}

prepare_errors() {
  python - "$OUTPUT" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception:
    print(999)
    raise SystemExit(0)
runs = data.get("longmemeval_qa", [])
errs = 0
for run in runs:
    if run.get("mode") == "prepare_only":
        errs += int(run.get("errors", 0) or 0)
print(errs)
PY
}

case "${mode}" in
  prepare)
    run_prepare "$@"
    ;;
  read)
    run_read "$@"
    ;;
  full)
    run_prepare "$@"
    errs="$(prepare_errors)"
    if [[ "${errs}" != "0" ]]; then
      echo "prepare reported ${errs} error(s)."
      if [[ "${AUTO_PREP_RETRY}" == "1" ]]; then
        echo "Retrying prepare once with --service-workers 1 ..."
        run_prepare "$@" --service-workers 1
        errs="$(prepare_errors)"
      fi
    fi
    if [[ "${errs}" != "0" ]]; then
      echo "prepare still has ${errs} error(s); skip read phase."
      echo "Check logs and rerun: bash scripts/run_longmemeval_service.sh prepare"
      exit 1
    fi
    run_read "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
