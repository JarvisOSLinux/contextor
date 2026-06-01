# CLAUDE.md — contextor

## What This Is

Long-term memory store for the JARVIS ecosystem. Accepts pre-computed embedding
vectors, stores them in SQLite, and retrieves by cosine similarity. Controlled
by the JARVIS daemon via newline-delimited JSON over stdin/stdout.

## Role in the JARVIS Ecosystem

contextor is the memory layer. The JARVIS daemon sends store/recall/search
commands to contextor as a child process. Embeddings are computed by the daemon
(via Ollama's embedding endpoint) before being sent to contextor for storage.

## Tech Stack

- Rust (2021 edition)
- rusqlite 0.31 (SQLite, bundled)
- serde / serde_json for JSON protocol
- uuid 1 for entry IDs
- dirs 5 for XDG paths

Release profile: LTO enabled, codegen-units=1, symbols stripped.

## Architecture

```
src/
├── main.rs      REPL loop: reads JSON from stdin, dispatches commands, writes JSON to stdout
├── storage.rs   SQLite backend + in-memory vector index (~1078 lines)
└── vector.rs    Cosine similarity computation (auto-vectorized at -O2)
```

### Key Data Models

- **Entry**: theme, content, vector (f32 array), metadata (JSON), timestamps, optional session_id
- **Session**: id, title, user/assistant message counts, rolling summary

### JSON Protocol

One command per line on stdin, one response per line on stdout.

13 commands: `store`, `recall`, `search`, `list`, `delete`, `prune`, `reindex`,
`create_session`, `list_sessions`, `get_session`, `update_session`,
`delete_session`, `status`.

## Build & Test

```bash
cargo build --release
cargo test
cargo clippy
cargo fmt --check
```

## Data Location

`~/.local/share/contextor/contextor.db` (auto-created on first run)

## Conventions

- `cargo fmt` + `cargo clippy` clean before pushing
- Commit messages: imperative mood
- No comments explaining what code does; only non-obvious WHY
