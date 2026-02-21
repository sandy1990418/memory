# Memory

This is a sample root memory file used in tests.

## Preferences

- Prefer Python over JavaScript for backend tasks.
- Always write type annotations.
- Use ruff for linting.

## Architecture

The OpenClaw memory subsystem indexes markdown files and supports
hybrid search combining FTS and vector similarity.

## Notes

- SQLite is used as the backing store.
- Embeddings are optional; FTS-only mode is supported without an API key.
