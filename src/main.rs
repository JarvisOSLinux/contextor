mod storage;
mod vector;

use std::io::{self, BufRead, Write};

use serde::Deserialize;
use serde_json::Value;
use storage::Store;

// ---------------------------------------------------------------------------
// Command envelope — one JSON line in, one JSON line out
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Command {
    /// Store a memory entry with its pre-computed vector.
    Store {
        theme: String,
        content: String,
        vector: Vec<f32>,
        #[serde(default)]
        metadata: Option<Value>,
        /// Scopes the entry to a session. NULL/absent = global entry.
        #[serde(default)]
        session_id: Option<String>,
    },

    /// Recall the N most recent entries for a theme (returned oldest-first).
    Recall {
        theme: String,
        #[serde(default = "default_limit")]
        limit: usize,
        /// If provided, filter to that session only; absent = all entries for theme.
        #[serde(default)]
        session_id: Option<String>,
    },

    /// Vector similarity search (optionally scoped to a theme / session).
    Search {
        vector: Vec<f32>,
        #[serde(default = "default_top_k")]
        top_k: usize,
        #[serde(default)]
        offset: usize,
        #[serde(default)]
        min_score: f32,
        #[serde(default)]
        theme: Option<String>,
        /// If provided, search that session + global entries; absent = all entries.
        #[serde(default)]
        session_id: Option<String>,
    },

    /// List all themes with entry counts and last-modified timestamps.
    List {
        /// If provided, list themes for that session + global; absent = all themes.
        #[serde(default)]
        session_id: Option<String>,
    },

    /// Delete all entries for a theme.
    Delete { theme: String },

    /// Run retention policy (age-based + max entries per theme).
    Prune {
        #[serde(default = "default_retention_days")]
        retention_days: u64,
        #[serde(default = "default_max_per_theme")]
        max_per_theme: usize,
    },

    /// Replace the active entry for a theme, archiving the old one as a memento.
    /// Empty content = forget (no new active entry created).
    ReplaceActive {
        theme: String,
        #[serde(default)]
        content: String,
        #[serde(default)]
        vector: Vec<f32>,
        #[serde(default)]
        metadata: Option<Value>,
        #[serde(default)]
        session_id: Option<String>,
    },

    /// Return the last N archived (memento) entries for a theme, newest first.
    PeekMemento {
        theme: String,
        #[serde(default = "default_memento_limit")]
        limit: usize,
        #[serde(default)]
        session_id: Option<String>,
    },

    /// Rebuild the in-memory vector index from stored entries.
    Reindex,

    /// Health / status check.
    Status,

    // -----------------------------------------------------------------------
    // Session commands
    // -----------------------------------------------------------------------

    /// Create a new chat session. Returns session_id and created_at.
    CreateSession {
        #[serde(default)]
        title: String,
    },

    /// List sessions ordered by updated_at DESC with pagination.
    ListSessions {
        #[serde(default = "default_sessions_limit")]
        limit: usize,
        #[serde(default)]
        offset: usize,
    },

    /// Get full session metadata including rolling_summary.
    GetSession { session_id: String },

    /// Update session title and/or rolling_summary.
    UpdateSession {
        session_id: String,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        rolling_summary: Option<String>,
    },

    /// Delete a session and all its entries.
    DeleteSession { session_id: String },
}

fn default_limit() -> usize { 20 }
fn default_top_k() -> usize { 5 }
fn default_retention_days() -> u64 { 90 }
fn default_max_per_theme() -> usize { 500 }
fn default_sessions_limit() -> usize { 50 }
fn default_memento_limit() -> usize { 5 }

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let mut store = match Store::open() {
        Ok(s) => s,
        Err(e) => {
            // Fatal: can't proceed without storage
            eprintln!("contextor: failed to initialise store: {e}");
            std::process::exit(1);
        }
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // EOF or I/O error — exit cleanly
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Command>(line) {
            Ok(cmd) => dispatch(&mut store, cmd),
            Err(e) => serde_json::json!({ "ok": false, "error": format!("parse error: {e}") }),
        };

        // Newline-delimited JSON on stdout — one response per command
        if writeln!(out, "{response}").is_err() {
            break; // JARVIS closed the pipe
        }
        let _ = out.flush();
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn dispatch(store: &mut Store, cmd: Command) -> Value {
    match cmd {
        Command::Store { theme, content, vector, metadata, session_id } => {
            store.cmd_store(&theme, &content, vector, metadata, session_id.as_deref())
        }
        Command::Recall { theme, limit, session_id } => {
            store.cmd_recall(&theme, limit, session_id.as_deref())
        }
        Command::Search { vector, top_k, offset, min_score, theme, session_id } => {
            store.cmd_search(&vector, top_k, offset, min_score, theme.as_deref(), session_id.as_deref())
        }
        Command::List { session_id } => store.cmd_list(session_id.as_deref()),
        Command::Delete { theme } => store.cmd_delete(&theme),
        Command::ReplaceActive { theme, content, vector, metadata, session_id } => {
            store.cmd_replace_active(&theme, &content, vector, metadata, session_id.as_deref())
        }
        Command::PeekMemento { theme, limit, session_id } => {
            store.cmd_peek_memento(&theme, limit, session_id.as_deref())
        }
        Command::Prune { retention_days, max_per_theme } => {
            store.cmd_prune(retention_days, max_per_theme)
        }
        Command::Reindex => store.cmd_reindex(),
        Command::Status => store.cmd_status(),
        Command::CreateSession { title } => store.cmd_create_session(&title),
        Command::ListSessions { limit, offset } => store.cmd_list_sessions(limit, offset),
        Command::GetSession { session_id } => store.cmd_get_session(&session_id),
        Command::UpdateSession { session_id, title, rolling_summary } => {
            store.cmd_update_session(&session_id, title, rolling_summary)
        }
        Command::DeleteSession { session_id } => store.cmd_delete_session(&session_id),
    }
}
