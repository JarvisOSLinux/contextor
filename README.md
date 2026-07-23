# contextor

**Long-term memory store for JARVIS.**

contextor accepts pre-computed embedding vectors and stores them alongside text
in a local SQLite database. Retrieval is by cosine similarity, theme, or session
scope. It is designed to be controlled by a parent process (the JARVIS daemon)
via newline-delimited JSON over stdin/stdout.

## Build

Requires [Rust](https://rustup.rs/).

```bash
cargo build --release
cargo install --path .
```

## Usage

contextor runs as a REPL: one JSON command per line on stdin, one JSON response
per line on stdout.

```bash
echo '{"cmd": "status"}' | ./target/release/contextor
```

## Commands

| Command | Description |
|---------|-------------|
| `store` | Store a memory entry with theme, content, vector, and optional metadata |
| `recall` | Recall the N most recent entries for a theme (returned oldest-first for reading order) |
| `search` | Vector similarity search (optional theme/session filter) |
| `list` | List distinct themes |
| `delete` | Delete all entries for a theme |
| `prune` | Remove old entries beyond age or per-theme limits |
| `reindex` | Rebuild the in-memory vector index from disk |
| `create_session` | Create a new chat session |
| `list_sessions` | List all sessions |
| `get_session` | Get session details (title, summary, message counts) |
| `update_session` | Update session title and/or rolling summary |
| `delete_session` | Delete a session and its scoped entries |
| `status` | Return entry/theme counts, vector dimension info (incl. per-dimension histogram), and the storage directory path |

### Example: Store and Search

```json
{"cmd": "store", "theme": "facts", "content": "The sky is blue", "vector": [0.1, 0.2, 0.3]}
{"cmd": "search", "vector": [0.1, 0.2, 0.3], "top_k": 5, "min_score": 0.7}
```

All stored vectors must share one dimension: `store` rejects empty vectors and
vectors whose dimension differs from the existing index (e.g. after an
embedding-model change) with `{"ok": false, "error": "dimension mismatch: ..."}`
— reindex or prune to switch models.

## Data Storage

- Database: `~/.local/share/jarvis/memory/contextor.db` (SQLite). Override the
  base directory with `JARVIS_DATA_DIR` (database then lives at
  `$JARVIS_DATA_DIR/memory/contextor.db`)
- Schema auto-migrates on startup
- On corruption, the database is renamed to `contextor.db.corrupted` and a
  fresh database is created (a warning is printed to stderr)
- Vector index is held in memory for fast search, rebuilt from disk on startup

## Session Support

Entries can be global or scoped to a `session_id`. Session-scoped searches also
include global entries. Session-scoped entries are exempt from age-based pruning.
`message_count` is maintained automatically — each `store` with a `session_id`
increments it.

## Retention Policies

- **Max age**: 90 days (default, configurable per prune call) — applies to
  global entries only; session-scoped entries are exempt from age-based pruning
- **Max entries**: 500 (default) per theme-per-session bucket — count-based
  pruning applies to session-scoped buckets too; the oldest entries over the
  limit are removed

## License

AGPL-3.0

## Changelog — corrected claims

*2026-07-22:* database path corrected to `~/.local/share/jarvis/memory/contextor.db` (`JARVIS_DATA_DIR` override documented); `delete` deletes a whole theme (there is no by-ID deletion); `update_session` takes title/rolling summary only (message_count is automatic); retention corrected (age pruning skips session entries, count pruning applies per theme-per-session bucket); `store` vector validation and corruption recovery documented; `status` payload described accurately.

*2026-07-23:* `recall` fixed to return the N most recent entries (was oldest-N — `ORDER BY stored_at ASC LIMIT N` selected the head of the table), returned oldest-first within the window; rowid tiebreak for same-instant stores.
