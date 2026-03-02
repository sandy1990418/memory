#!/usr/bin/env bash
# Local pgvector lifecycle helper for openclaw-memory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.pg.yml"
DEFAULT_DSN="postgresql://memuser:mempass@localhost:5433/memory"
PG_DSN="${OPENCLAW_PG_DSN:-${DEFAULT_DSN}}"

usage() {
  cat <<EOF
Usage: bash scripts/pgvector_local.sh <command>

Commands:
  start   Start local pgvector container
  init    Apply schema (ensure_pg_schema)
  status  Show container health/status
  logs    Tail container logs
  stop    Stop container (keep data volume)
  reset   Stop container and remove data volume
  dsn     Print recommended OPENCLAW_PG_DSN

Current DSN: ${PG_DSN}
EOF
}

cmd="${1:-status}"

case "${cmd}" in
  start)
    docker compose -f "${COMPOSE_FILE}" up -d
    echo "pgvector started."
    echo "OPENCLAW_PG_DSN=${PG_DSN}"
    ;;
  init)
    PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
      python -c "from openclaw_memory.pg_schema import get_pg_connection, ensure_pg_schema; \
conn = get_pg_connection('${PG_DSN}'); \
ensure_pg_schema(conn); \
conn.close(); \
print('schema ready')"
    ;;
  status)
    docker compose -f "${COMPOSE_FILE}" ps
    ;;
  logs)
    docker compose -f "${COMPOSE_FILE}" logs -f pgvector
    ;;
  stop)
    docker compose -f "${COMPOSE_FILE}" down
    ;;
  reset)
    docker compose -f "${COMPOSE_FILE}" down -v
    ;;
  dsn)
    echo "${PG_DSN}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
