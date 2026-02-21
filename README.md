# openclaw-memory

Standalone Python implementation of the [OpenClaw](https://openclaw.ai) memory subsystem.

Provides a library and CLI (`ocmem`) for indexing, searching, and managing plain-Markdown memory files — the same format used by OpenClaw agents.

---

## Features

- Hybrid search: BM25 full-text + vector cosine similarity
- Multiple embedding providers: OpenAI, Gemini, Voyage AI, local (llama-cpp-python)
- FTS-only mode when no embedding provider is configured
- MMR re-ranking for result diversity
- Temporal decay scoring for recency-weighted results
- Query keyword expansion for conversational queries
- SQLite-backed index with embedding cache
- `append-daily` command to write daily notes (`memory/YYYY-MM-DD.md`)

---

## Installation

```bash
pip install -e ".[dev]"       # editable install with dev tools
pip install -e ".[openai]"    # with OpenAI embeddings
pip install -e ".[all]"       # all remote providers
pip install -e ".[local]"     # with local llama-cpp-python embeddings
```

---

## Quick Start

```bash
# Set your workspace and API key
export OPENCLAW_WORKSPACE=~/my-notes
export OPENAI_API_KEY=sk-...

# Index memory files
ocmem index

# Search
ocmem search "deployment notes"

# Read a file
ocmem get MEMORY.md
ocmem get memory/2024-01-15.md --from 10 --lines 20

# Append to today's daily note
ocmem append-daily "Deployed v2.1 to prod. No issues."

# Check status
ocmem status
ocmem status --deep    # probe embedding availability
ocmem status --json    # machine-readable
```

---

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENCLAW_WORKSPACE` | `$PWD` | Directory containing `MEMORY.md` and `memory/` |
| `OPENCLAW_DB_PATH` | `~/.openclaw/memory/main.sqlite` | SQLite index path |
| `OPENCLAW_STATE_DIR` | `~/.openclaw` | Base state directory |
| `OPENCLAW_MEMORY_PROVIDER` | `auto` | `auto`, `openai`, `gemini`, `voyage`, `local` |
| `OPENCLAW_MEMORY_MODEL` | provider default | Embedding model name |
| `OPENCLAW_MEMORY_FALLBACK` | `none` | Fallback provider if primary fails |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `VOYAGE_API_KEY` | — | Voyage AI API key |

Provider auto-selection order (when `OPENCLAW_MEMORY_PROVIDER=auto`):
1. `local` — if a local model file is configured and exists
2. `openai` — if `OPENAI_API_KEY` is set
3. `gemini` — if `GEMINI_API_KEY` is set
4. `voyage` — if `VOYAGE_API_KEY` is set
5. FTS-only mode — keyword search without embeddings

---

## Memory File Layout

```
<workspace>/
├── MEMORY.md              # Long-term curated memory (evergreen)
├── memory.md              # Alternate root (also indexed)
└── memory/
    ├── 2024-01-15.md      # Daily note (temporal decay applies)
    ├── 2024-01-16.md
    └── projects.md        # Topic file (evergreen, no decay)
```

Files are chunked into ~400-token overlapping segments (80-token overlap), embedded, and stored in SQLite with FTS5 support.

---

## CLI Reference

### `ocmem search <query>`

Search the memory index using hybrid BM25 + vector similarity.

```bash
ocmem search "API rate limiting"
ocmem search "deployment checklist" --max-results 10
ocmem search "bug fixes" --min-score 0.5
ocmem search "redis config" --json
ocmem search "notes" --workspace ~/project-notes
```

Options:
- `--max-results N` — maximum results (default: 6)
- `--min-score F` — minimum score 0.0–1.0 (default: 0.35)
- `--json` — output as JSON array
- `--workspace DIR` — override workspace

### `ocmem get <path>`

Read a memory file or slice of it.

```bash
ocmem get MEMORY.md
ocmem get memory/2024-01-15.md
ocmem get memory/projects.md --from 10 --lines 30
```

Options:
- `--from N` — start line (1-indexed)
- `--lines N` — number of lines to read
- `--workspace DIR` — override workspace

### `ocmem append-daily <text>`

Append text to today's daily memory file (`memory/YYYY-MM-DD.md`).
Creates the file and `memory/` directory if they don't exist.
Appends with a blank-line separator if the file already has content.

```bash
ocmem append-daily "Fixed the CORS issue in the API gateway."
ocmem append-daily "Meeting notes: decided to use Redis for sessions." --date 2024-01-15
ocmem append-daily "$(cat my-notes.md)" --workspace ~/project-notes
```

Options:
- `--date YYYY-MM-DD` — override date (default: today UTC)
- `--workspace DIR` — override workspace

### `ocmem index`

Reindex all memory files. Only re-embeds changed files (hash-based).

```bash
ocmem index
ocmem index --force     # force full reindex
ocmem index --verbose   # show progress
```

### `ocmem status`

Show the current index status.

```bash
ocmem status
ocmem status --deep    # also probe embedding availability
ocmem status --json    # machine-readable JSON
```

---

## Python API

```python
from openclaw_memory import MemoryIndexManager

# Create manager (auto-selects embedding provider from env)
mgr = MemoryIndexManager.create(
    workspace_dir="/path/to/workspace",
)

with mgr:
    # Sync index
    mgr.sync()

    # Search
    results = mgr.search("deployment notes", max_results=5)
    for r in results:
        print(f"{r.score:.3f}  {r.path}:{r.start_line}-{r.end_line}")
        print(r.snippet)

    # Read a file
    content = mgr.read_file("MEMORY.md")
    print(content["text"])

    # Status
    st = mgr.status()
    print(f"{st.files} files, {st.chunks} chunks")
```

---

## Architecture

```
MEMORY.md + memory/*.md
        │
        ▼
   list_memory_files()          internal.py
        │
        ▼
   chunk_markdown()             internal.py   (~400 tokens, 80-token overlap)
        │
        ├──► embed_batch()      embeddings.py  (OpenAI / Gemini / Voyage / local)
        │         │
        │         ▼
        │   SQLite chunks       schema.py      (id, path, lines, text, embedding)
        │   FTS5 index          schema.py      (chunks_fts virtual table)
        │
        ▼
   search(query)               manager.py
        ├──► embed_query()      → vector results   (cosine similarity)
        ├──► FTS5 MATCH         → keyword results  (BM25 score)
        ├──► merge_hybrid()     hybrid.py          (weighted sum)
        ├──► temporal_decay()   temporal_decay.py  (optional)
        └──► mmr_rerank()       mmr.py             (optional)
                │
                ▼
        MemorySearchResult[]
```

### TypeScript → Python module mapping

| OpenClaw TS module | Python module | Purpose |
|---|---|---|
| `memory/types.ts` | `types.py` | Core dataclasses and protocols |
| `agents/memory-search.ts` | `config.py` | Config resolution |
| `memory/internal.ts` | `internal.py` | File walking, chunking, hashing |
| `memory/memory-schema.ts` | `schema.py` | SQLite schema creation |
| `memory/embeddings.ts` | `embeddings.py` | Embedding provider adapters |
| `memory/hybrid.ts` | `hybrid.py` | BM25 + vector merging |
| `memory/mmr.ts` | `mmr.py` | MMR diversity re-ranking |
| `memory/temporal-decay.ts` | `temporal_decay.py` | Recency scoring |
| `memory/query-expansion.ts` | `query_expansion.py` | FTS keyword extraction |
| `memory/manager-search.ts` | `manager_search.py` | SQLite search helpers |
| `memory/manager.ts` | `manager.py` | Core MemoryIndexManager |
| `memory/sync-*.ts` | `sync.py` | File sync and indexing |
| `cli/memory-cli.ts` | `cli/main.py` | CLI commands |
| `auto-reply/reply/memory-flush.ts` | `cli/main.py` (append-daily) | Daily note writing |

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,openai]"

# Lint and format
ruff check src/
ruff format src/

# Type check
mypy src/

# Run tests
pytest
pytest --cov=openclaw_memory --cov-report=term-missing
```

---

## Deferred Features (Post-MVP)

- **sqlite-vec acceleration** — vector search in SQL via extension (currently uses in-process cosine similarity)
- **File watcher** — auto-reindex on file changes (watchdog integration)
- **OpenAI / Gemini batch embedding API** — async bulk indexing for large corpora
- **Session transcript indexing** — index OpenClaw `.jsonl` session files
- **QMD backend** — external QMD binary integration
- **Multi-agent scoping** — separate indexes per agent ID
- **Full JSON5 config file** — load config from `~/.openclaw/config.json5`
