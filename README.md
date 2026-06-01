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
| `recall` | Recall the N most recent entries for a theme (oldest-first) |
| `search` | Vector similarity search (optional theme/session filter) |
| `list` | List distinct themes |
| `delete` | Delete entries by ID |
| `prune` | Remove old entries beyond age or per-theme limits |
| `reindex` | Rebuild the in-memory vector index from disk |
| `create_session` | Create a new chat session |
| `list_sessions` | List all sessions |
| `get_session` | Get session details (title, summary, message counts) |
| `update_session` | Update session title, summary, or message counts |
| `delete_session` | Delete a session and its scoped entries |
| `status` | Return entry count and database path |

### Example: Store and Search

```json
{"cmd": "store", "theme": "facts", "content": "The sky is blue", "vector": [0.1, 0.2, 0.3]}
{"cmd": "search", "vector": [0.1, 0.2, 0.3], "top_k": 5, "min_score": 0.7}
```

## Data Storage

- Database: `~/.local/share/contextor/contextor.db` (SQLite)
- Schema auto-migrates on startup
- Vector index is held in memory for fast search, rebuilt from disk on startup

## Session Support

Entries can be global or scoped to a `session_id`. Session-scoped searches also
include global entries. Session-scoped entries are exempt from age-based pruning.

## Retention Policies

- **Max age**: 90 days (default, configurable per prune call)
- **Max per theme**: 500 entries (default, configurable per prune call)
- Session-scoped entries are never auto-pruned

## License

AGPL-3.0
