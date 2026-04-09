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
    },

    /// Recall the N most recent entries for a theme (returned oldest-first).
    Recall {
        theme: String,
        #[serde(default = "default_limit")]
        limit: usize,
    },

    /// Vector similarity search (optionally scoped to a theme).
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
    },

    /// List all themes with entry counts and last-modified timestamps.
    List,

    /// Delete all entries for a theme.
    Delete { theme: String },

    /// Run retention policy (age-based + max entries per theme).
    Prune {
        #[serde(default = "default_retention_days")]
        retention_days: u64,
        #[serde(default = "default_max_per_theme")]
        max_per_theme: usize,
    },

    /// Rebuild the in-memory vector index from stored entries.
    Reindex,

    /// Health / status check.
    Status,
}

fn default_limit() -> usize { 20 }
fn default_top_k() -> usize { 5 }
fn default_retention_days() -> u64 { 90 }
fn default_max_per_theme() -> usize { 500 }

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
        Command::Store { theme, content, vector, metadata } => {
            store.cmd_store(&theme, &content, vector, metadata)
        }
        Command::Recall { theme, limit } => store.cmd_recall(&theme, limit),
        Command::Search { vector, top_k, offset, min_score, theme } => {
            store.cmd_search(&vector, top_k, offset, min_score, theme.as_deref())
        }
        Command::List => store.cmd_list(),
        Command::Delete { theme } => store.cmd_delete(&theme),
        Command::Prune { retention_days, max_per_theme } => {
            store.cmd_prune(retention_days, max_per_theme)
        }
        Command::Reindex => store.cmd_reindex(),
        Command::Status => store.cmd_status(),
    }
}
