use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde_json::Value;
use uuid::Uuid;

use crate::vector::cosine_similarity;

// ---------------------------------------------------------------------------
// In-memory vector index
// ---------------------------------------------------------------------------

/// Lean entry kept in memory for fast similarity search.
struct IndexEntry {
    id: String,
    theme: String,
    vector: Vec<f32>,
    session_id: Option<String>,
}

/// Flat in-memory index.  Vectors live on the heap; cosine search is a single
/// sequential pass — no tree structures needed at this scale.
struct VectorIndex {
    entries: Vec<IndexEntry>,
}

impl VectorIndex {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    fn push(&mut self, id: String, theme: String, vector: Vec<f32>, session_id: Option<String>) {
        self.entries.push(IndexEntry { id, theme, vector, session_id });
    }

    fn remove_theme(&mut self, theme: &str) {
        self.entries.retain(|e| e.theme != theme);
    }

    fn remove_session(&mut self, session_id: &str) {
        self.entries.retain(|e| e.session_id.as_deref() != Some(session_id));
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Dimension of the first stored vector, or 0 if empty.
    fn dim(&self) -> usize {
        self.entries.first().map(|e| e.vector.len()).unwrap_or(0)
    }

    /// Score every entry with optional theme and session filters.
    ///
    /// `session_filter = Some(sid)`: include entries where `session_id == sid`
    /// OR `session_id IS NULL` (global entries are always visible within a session).
    ///
    /// `session_filter = None`: include all entries regardless of session.
    fn search(
        &self,
        query: &[f32],
        min_score: f32,
        theme_filter: Option<&str>,
        session_filter: Option<&str>,
    ) -> Vec<(f32, &str)> {
        let mut scored: Vec<(f32, &str)> = self
            .entries
            .iter()
            .filter(|e| theme_filter.map_or(true, |t| e.theme == t))
            .filter(|e| match session_filter {
                None => true,
                Some(sid) => e.session_id.as_deref() == Some(sid) || e.session_id.is_none(),
            })
            .map(|e| (cosine_similarity(query, &e.vector), e.id.as_str()))
            .filter(|(s, _)| *s >= min_score)
            .collect();

        scored.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
    }
}

// ---------------------------------------------------------------------------
// Public store
// ---------------------------------------------------------------------------

pub struct Store {
    conn: Connection,
    index: VectorIndex,
    pub storage_path: PathBuf,
}

impl Store {
    /// Open (or create) the store at the configured path.
    /// On corruption, logs to stderr and starts fresh.
    pub fn open() -> Result<Self, Box<dyn std::error::Error>> {
        let storage_path = resolve_storage_path();
        std::fs::create_dir_all(&storage_path)?;

        let db_path = storage_path.join("contextor.db");
        let conn = open_or_recover(&db_path)?;
        migrate(&conn)?;

        let index = load_index(&conn)?;

        Ok(Store { conn, index, storage_path })
    }

    // -----------------------------------------------------------------------
    // Entry commands
    // -----------------------------------------------------------------------

    pub fn cmd_store(
        &mut self,
        theme: &str,
        content: &str,
        vector: Vec<f32>,
        metadata: Option<Value>,
        session_id: Option<&str>,
    ) -> Value {
        let id = Uuid::new_v4().to_string();
        let stored_at = now_f64();
        let blob = encode_vector(&vector);
        let meta_str = metadata.as_ref().map(|m| m.to_string());

        match self.conn.execute(
            "INSERT INTO entries (id, theme, content, vector, stored_at, metadata, session_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![id, theme, content, blob, stored_at, meta_str, session_id],
        ) {
            Ok(_) => {
                if let Some(sid) = session_id {
                    let now = now_iso8601();
                    let _ = self.conn.execute(
                        "UPDATE sessions
                         SET message_count = message_count + 1, updated_at = ?1
                         WHERE id = ?2",
                        params![now, sid],
                    );
                }
                self.index.push(id, theme.to_string(), vector, session_id.map(str::to_string));
                serde_json::json!({ "ok": true, "theme": theme })
            }
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_recall(&self, theme: &str, limit: usize, session_id: Option<&str>) -> Value {
        let result = (|| -> rusqlite::Result<Vec<Value>> {
            let mut stmt = self.conn.prepare(
                "SELECT content, theme, stored_at, metadata, session_id
                 FROM entries
                 WHERE theme = ?1 AND (?2 IS NULL OR session_id = ?2)
                 ORDER BY stored_at ASC
                 LIMIT ?3",
            )?;
            stmt.query_map(params![theme, session_id, limit as i64], |row| {
                let content: String = row.get(0)?;
                let theme: String = row.get(1)?;
                let stored_at: f64 = row.get(2)?;
                let meta_raw: Option<String> = row.get(3)?;
                let session_id: Option<String> = row.get(4)?;
                let metadata: Value = meta_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or(Value::Null);
                Ok(serde_json::json!({
                    "content": content,
                    "theme": theme,
                    "stored_at": stored_at,
                    "metadata": metadata,
                    "session_id": session_id,
                }))
            })
            .and_then(|rows| rows.collect())
        })();

        match result {
            Ok(entries) => {
                let found = !entries.is_empty();
                serde_json::json!({
                    "ok": true,
                    "theme": theme,
                    "entries": entries,
                    "found": found,
                })
            }
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_search(
        &self,
        query: &[f32],
        top_k: usize,
        offset: usize,
        min_score: f32,
        theme_filter: Option<&str>,
        session_id: Option<&str>,
    ) -> Value {
        let scored = self.index.search(query, min_score, theme_filter, session_id);

        // Apply offset + top_k window
        let window: Vec<(f32, &str)> =
            scored.into_iter().skip(offset).take(top_k).collect();

        if window.is_empty() {
            return serde_json::json!({ "ok": true, "results": [], "available": true });
        }

        // Fetch content from DB for the matched IDs
        let ids: Vec<&str> = window.iter().map(|(_, id)| *id).collect();
        let placeholders: String = std::iter::repeat("?")
            .take(ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT id, content, theme, stored_at, session_id FROM entries WHERE id IN ({})",
            placeholders
        );

        let result = (|| -> rusqlite::Result<Vec<(String, String, String, f64, Option<String>)>> {
            let mut stmt = self.conn.prepare(&sql)?;
            stmt.query_map(rusqlite::params_from_iter(ids.iter()), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, Option<String>>(4)?,
                ))
            })
            .and_then(|rows| rows.collect())
        })();

        match result {
            Ok(rows) => {
                // Map id -> row
                let mut content_map: std::collections::HashMap<
                    String,
                    (String, String, f64, Option<String>),
                > = rows
                    .into_iter()
                    .map(|(id, content, theme, stored_at, session_id)| {
                        (id, (content, theme, stored_at, session_id))
                    })
                    .collect();

                // Rebuild in score order (preserving rank)
                let results: Vec<Value> = window
                    .iter()
                    .filter_map(|(score, id)| {
                        content_map.remove(*id).map(|(content, theme, stored_at, session_id)| {
                            serde_json::json!({
                                "content": content,
                                "theme": theme,
                                "score": score,
                                "stored_at": stored_at,
                                "session_id": session_id,
                            })
                        })
                    })
                    .collect();

                serde_json::json!({ "ok": true, "results": results, "available": true })
            }
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_list(&self, session_id: Option<&str>) -> Value {
        let result = (|| -> rusqlite::Result<Vec<Value>> {
            let mut stmt = self.conn.prepare(
                "SELECT theme, COUNT(*) AS entries, MAX(stored_at) AS last_modified
                 FROM entries
                 WHERE (?1 IS NULL OR session_id = ?1 OR session_id IS NULL)
                 GROUP BY theme
                 ORDER BY theme",
            )?;
            stmt.query_map(params![session_id], |row| {
                let theme: String = row.get(0)?;
                let entries: i64 = row.get(1)?;
                let last_modified: f64 = row.get(2)?;
                Ok(serde_json::json!({
                    "theme": theme,
                    "entries": entries,
                    "last_modified": format_unix_secs(last_modified as u64),
                }))
            })
            .and_then(|rows| rows.collect())
        })();

        match result {
            Ok(themes) => serde_json::json!({ "ok": true, "themes": themes }),
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_delete(&mut self, theme: &str) -> Value {
        match self
            .conn
            .execute("DELETE FROM entries WHERE theme = ?1", params![theme])
        {
            Ok(n) => {
                self.index.remove_theme(theme);
                serde_json::json!({
                    "ok": true,
                    "deleted": n > 0,
                    "theme": theme,
                })
            }
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_prune(&mut self, retention_days: u64, max_per_theme: usize) -> Value {
        let cutoff = now_f64() - (retention_days as f64 * 86_400.0);
        let mut pruned = 0usize;

        // 1. Age-based pruning — global entries only (session entries are never age-pruned)
        match self.conn.execute(
            "DELETE FROM entries WHERE stored_at < ?1 AND session_id IS NULL",
            params![cutoff],
        ) {
            Ok(n) => pruned += n,
            Err(e) => return serde_json::json!({ "ok": false, "error": e.to_string() }),
        }

        // 2. Per-theme-per-session count pruning: each (theme, session_id) bucket gets its own budget
        let groups_over_limit: Vec<(String, Option<String>, i64)> = {
            let result = (|| -> rusqlite::Result<Vec<(String, Option<String>, i64)>> {
                let mut stmt = self.conn.prepare(
                    "SELECT theme, session_id, COUNT(*) FROM entries
                     GROUP BY theme, session_id
                     HAVING COUNT(*) > ?1",
                )?;
                stmt.query_map(params![max_per_theme as i64], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })
                .and_then(|rows| rows.collect())
            })();
            match result {
                Ok(v) => v,
                Err(e) => return serde_json::json!({ "ok": false, "error": e.to_string() }),
            }
        };

        for (theme, session_id_opt, count) in groups_over_limit {
            let excess = (count as usize).saturating_sub(max_per_theme);
            let delete_result = match &session_id_opt {
                None => self.conn.execute(
                    "DELETE FROM entries WHERE id IN (
                         SELECT id FROM entries
                         WHERE theme = ?1 AND session_id IS NULL
                         ORDER BY stored_at ASC
                         LIMIT ?2
                     )",
                    params![theme, excess as i64],
                ),
                Some(sid) => self.conn.execute(
                    "DELETE FROM entries WHERE id IN (
                         SELECT id FROM entries
                         WHERE theme = ?1 AND session_id = ?2
                         ORDER BY stored_at ASC
                         LIMIT ?3
                     )",
                    params![theme, sid, excess as i64],
                ),
            };
            match delete_result {
                Ok(n) => pruned += n,
                Err(e) => return serde_json::json!({ "ok": false, "error": e.to_string() }),
            }
        }

        // Rebuild in-memory index to reflect deletions
        if pruned > 0 {
            match load_index(&self.conn) {
                Ok(idx) => self.index = idx,
                Err(e) => return serde_json::json!({ "ok": false, "error": e.to_string() }),
            }
        }

        serde_json::json!({ "ok": true, "pruned": pruned })
    }

    pub fn cmd_replace_active(
        &mut self,
        theme: &str,
        content: &str,
        vector: Vec<f32>,
        metadata: Option<Value>,
        session_id: Option<&str>,
    ) -> Value {
        let archived_at = now_f64();
        let new_id = if content.is_empty() {
            None
        } else {
            Some(Uuid::new_v4().to_string())
        };

        if let Err(e) = self.conn.execute_batch("BEGIN IMMEDIATE") {
            return serde_json::json!({ "ok": false, "error": e.to_string() });
        }

        let result = (|| -> rusqlite::Result<usize> {
            // Copy existing active entries for this theme into mementos.
            // `session_id IS ?` is NULL-safe: matches NULL when param is NULL.
            let archived = self.conn.execute(
                "INSERT INTO mementos
                     (id, theme, content, vector, stored_at, archived_at, metadata, session_id)
                 SELECT id, theme, content, vector, stored_at, ?1, metadata, session_id
                 FROM entries
                 WHERE theme = ?2 AND session_id IS ?3",
                params![archived_at, theme, session_id],
            )?;

            self.conn.execute(
                "DELETE FROM entries WHERE theme = ?1 AND session_id IS ?2",
                params![theme, session_id],
            )?;

            if let Some(ref id) = new_id {
                let stored_at = now_f64();
                let blob = encode_vector(&vector);
                let meta_str = metadata.as_ref().map(|m| m.to_string());
                self.conn.execute(
                    "INSERT INTO entries
                         (id, theme, content, vector, stored_at, metadata, session_id)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![id, theme, content, blob, stored_at, meta_str, session_id],
                )?;
            }

            Ok(archived)
        })();

        match result {
            Ok(archived_count) => {
                if let Err(e) = self.conn.execute_batch("COMMIT") {
                    let _ = self.conn.execute_batch("ROLLBACK");
                    return serde_json::json!({ "ok": false, "error": e.to_string() });
                }
                self.index.remove_theme(theme);
                if let Some(ref id) = new_id {
                    self.index.push(
                        id.clone(),
                        theme.to_string(),
                        vector,
                        session_id.map(str::to_string),
                    );
                }
                let forgotten = content.is_empty();
                serde_json::json!({
                    "ok": true,
                    "archived": archived_count > 0,
                    "forgotten": forgotten,
                    "theme": theme,
                })
            }
            Err(e) => {
                let _ = self.conn.execute_batch("ROLLBACK");
                serde_json::json!({ "ok": false, "error": e.to_string() })
            }
        }
    }

    pub fn cmd_peek_memento(
        &self,
        theme: &str,
        limit: usize,
        session_id: Option<&str>,
    ) -> Value {
        let result = (|| -> rusqlite::Result<Vec<Value>> {
            let mut stmt = self.conn.prepare(
                "SELECT content, stored_at, archived_at, metadata
                 FROM mementos
                 WHERE theme = ?1 AND session_id IS ?2
                 ORDER BY archived_at DESC
                 LIMIT ?3",
            )?;
            stmt.query_map(params![theme, session_id, limit as i64], |row| {
                let content: String = row.get(0)?;
                let stored_at: f64 = row.get(1)?;
                let archived_at: f64 = row.get(2)?;
                let meta_raw: Option<String> = row.get(3)?;
                let metadata: Value = meta_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or(Value::Null);
                Ok(serde_json::json!({
                    "content": content,
                    "stored_at": format_unix_secs(stored_at as u64),
                    "archived_at": format_unix_secs(archived_at as u64),
                    "metadata": metadata,
                }))
            })
            .and_then(|rows| rows.collect())
        })();

        match result {
            Ok(mementos) => serde_json::json!({
                "ok": true,
                "theme": theme,
                "mementos": mementos,
            }),
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_reindex(&mut self) -> Value {
        match load_index(&self.conn) {
            Ok(idx) => {
                let count = idx.len();
                self.index = idx;
                serde_json::json!({ "ok": true, "indexed": count })
            }
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_status(&self) -> Value {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM entries", [], |r| r.get(0))
            .unwrap_or(0);

        let themes: i64 = self
            .conn
            .query_row("SELECT COUNT(DISTINCT theme) FROM entries", [], |r| r.get(0))
            .unwrap_or(0);

        serde_json::json!({
            "ok": true,
            "total_entries": total,
            "themes": themes,
            "vector_dimensions": self.index.dim(),
            "storage_path": self.storage_path.to_string_lossy(),
        })
    }

    // -----------------------------------------------------------------------
    // Session commands
    // -----------------------------------------------------------------------

    pub fn cmd_create_session(&mut self, title: &str) -> Value {
        let id = generate_session_id();
        let effective_title = if title.is_empty() { "New chat" } else { title };
        let now = now_iso8601();

        match self.conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, message_count, rolling_summary)
             VALUES (?1, ?2, ?3, ?4, 0, '')",
            params![id, effective_title, now, now],
        ) {
            Ok(_) => serde_json::json!({
                "ok": true,
                "session_id": id,
                "created_at": now,
            }),
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_list_sessions(&self, limit: usize, offset: usize) -> Value {
        let result = (|| -> rusqlite::Result<Vec<Value>> {
            let mut stmt = self.conn.prepare(
                "SELECT id, title, created_at, updated_at, message_count
                 FROM sessions
                 ORDER BY updated_at DESC
                 LIMIT ?1 OFFSET ?2",
            )?;
            stmt.query_map(params![limit as i64, offset as i64], |row| {
                let id: String = row.get(0)?;
                let title: String = row.get(1)?;
                let created_at: String = row.get(2)?;
                let updated_at: String = row.get(3)?;
                let message_count: i64 = row.get(4)?;
                Ok(serde_json::json!({
                    "id": id,
                    "title": title,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "message_count": message_count,
                }))
            })
            .and_then(|rows| rows.collect())
        })();

        match result {
            Ok(sessions) => serde_json::json!({ "ok": true, "sessions": sessions }),
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_get_session(&self, session_id: &str) -> Value {
        match self.conn.query_row(
            "SELECT id, title, created_at, updated_at, message_count, rolling_summary
             FROM sessions WHERE id = ?1",
            params![session_id],
            |row| {
                let id: String = row.get(0)?;
                let title: String = row.get(1)?;
                let created_at: String = row.get(2)?;
                let updated_at: String = row.get(3)?;
                let message_count: i64 = row.get(4)?;
                let rolling_summary: String = row.get(5)?;
                Ok(serde_json::json!({
                    "id": id,
                    "title": title,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "message_count": message_count,
                    "rolling_summary": rolling_summary,
                }))
            },
        ) {
            Ok(session) => serde_json::json!({ "ok": true, "session": session }),
            Err(rusqlite::Error::QueryReturnedNoRows) => serde_json::json!({
                "ok": false,
                "error": format!("session not found: {session_id}"),
            }),
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_update_session(
        &mut self,
        session_id: &str,
        title: Option<String>,
        rolling_summary: Option<String>,
    ) -> Value {
        let now = now_iso8601();
        let result = match (title, rolling_summary) {
            (Some(t), Some(rs)) => self.conn.execute(
                "UPDATE sessions SET title = ?1, rolling_summary = ?2, updated_at = ?3
                 WHERE id = ?4",
                params![t, rs, now, session_id],
            ),
            (Some(t), None) => self.conn.execute(
                "UPDATE sessions SET title = ?1, updated_at = ?2 WHERE id = ?3",
                params![t, now, session_id],
            ),
            (None, Some(rs)) => self.conn.execute(
                "UPDATE sessions SET rolling_summary = ?1, updated_at = ?2 WHERE id = ?3",
                params![rs, now, session_id],
            ),
            (None, None) => self.conn.execute(
                "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                params![now, session_id],
            ),
        };
        match result {
            Ok(_) => serde_json::json!({ "ok": true }),
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_delete_session(&mut self, session_id: &str) -> Value {
        // Delete all entries belonging to the session
        let deleted_entries = match self.conn.execute(
            "DELETE FROM entries WHERE session_id = ?1",
            params![session_id],
        ) {
            Ok(n) => n,
            Err(e) => return serde_json::json!({ "ok": false, "error": e.to_string() }),
        };

        // Delete the session record
        if let Err(e) = self
            .conn
            .execute("DELETE FROM sessions WHERE id = ?1", params![session_id])
        {
            return serde_json::json!({ "ok": false, "error": e.to_string() });
        }

        // Remove session entries from in-memory index
        self.index.remove_session(session_id);

        serde_json::json!({ "ok": true, "deleted_entries": deleted_entries })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolve_storage_path() -> PathBuf {
    if let Ok(base) = std::env::var("JARVIS_DATA_DIR") {
        PathBuf::from(base).join("memory")
    } else {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("jarvis")
            .join("memory")
    }
}

fn open_or_recover(db_path: &PathBuf) -> Result<Connection, Box<dyn std::error::Error>> {
    match Connection::open(db_path) {
        Ok(conn) => {
            // Quick integrity check — catches most corruption
            match conn.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0)) {
                Ok(ref s) if s == "ok" => Ok(conn),
                Ok(msg) => {
                    eprintln!("contextor: database integrity check failed ({msg}), starting fresh");
                    drop(conn);
                    recover(db_path)
                }
                Err(e) => {
                    eprintln!("contextor: could not run integrity check ({e}), starting fresh");
                    drop(conn);
                    recover(db_path)
                }
            }
        }
        Err(e) => {
            eprintln!("contextor: failed to open database ({e}), starting fresh");
            recover(db_path)
        }
    }
}

fn recover(db_path: &PathBuf) -> Result<Connection, Box<dyn std::error::Error>> {
    // Rename the corrupted file so it isn't lost, then start fresh
    let backup = db_path.with_extension("db.corrupted");
    let _ = std::fs::rename(db_path, &backup);
    Ok(Connection::open(db_path)?)
}

fn migrate(conn: &Connection) -> Result<(), Box<dyn std::error::Error>> {
    // Create base tables
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS entries (
            id         TEXT PRIMARY KEY,
            theme      TEXT NOT NULL,
            content    TEXT NOT NULL,
            vector     BLOB NOT NULL,
            stored_at  REAL NOT NULL,
            metadata   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_theme_stored_at
            ON entries (theme, stored_at);
        CREATE INDEX IF NOT EXISTS idx_stored_at
            ON entries (stored_at);
        CREATE TABLE IF NOT EXISTS sessions (
            id              TEXT PRIMARY KEY,
            title           TEXT NOT NULL DEFAULT '',
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            message_count   INTEGER NOT NULL DEFAULT 0,
            rolling_summary TEXT NOT NULL DEFAULT ''
        );",
    )?;

    // Add session_id column to entries if not present (migration for existing DBs).
    // SQLite does not support IF NOT EXISTS on ALTER TABLE, so we check first.
    let columns: Vec<String> = {
        let mut stmt = conn.prepare("PRAGMA table_info(entries)")?;
        let rows: rusqlite::Result<Vec<String>> = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect();
        rows?
    };

    if !columns.iter().any(|c| c == "session_id") {
        conn.execute_batch(
            "ALTER TABLE entries ADD COLUMN session_id TEXT DEFAULT NULL;",
        )?;
    }

    // Ensure the index exists (safe to run on both fresh and migrated DBs)
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_session_id ON entries (session_id);",
    )?;

    // Mementos table — archived history of replaced/forgotten entries
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS mementos (
            id          TEXT PRIMARY KEY,
            theme       TEXT NOT NULL,
            content     TEXT NOT NULL,
            vector      BLOB NOT NULL,
            stored_at   REAL NOT NULL,
            archived_at REAL NOT NULL,
            metadata    TEXT,
            session_id  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_memento_theme_archived
            ON mementos (theme, archived_at);",
    )?;

    Ok(())
}

fn load_index(conn: &Connection) -> Result<VectorIndex, Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare(
        "SELECT id, theme, vector, session_id FROM entries ORDER BY stored_at ASC",
    )?;

    let rows: rusqlite::Result<Vec<(String, String, Vec<u8>, Option<String>)>> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?
        .collect();

    let mut index = VectorIndex::new();
    for (id, theme, blob, session_id) in rows? {
        index.push(id, theme, decode_vector(&blob), session_id);
    }
    Ok(index)
}

fn encode_vector(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn decode_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn now_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn now_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (year, month, day) = days_to_ymd(secs / 86_400);
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, month, day, h, m, s)
}

fn generate_session_id() -> String {
    let s = Uuid::new_v4().simple().to_string();
    s[..8].to_string()
}

/// Format a Unix timestamp as "YYYY-MM-DD HH:MM:SS" without external crates.
fn format_unix_secs(secs: u64) -> String {
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let (year, month, day) = days_to_ymd(secs / 86_400);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month, day, h, m, s
    )
}

fn days_to_ymd(total_days: u64) -> (u32, u32, u32) {
    let mut remaining = total_days;
    let mut year = 1970u32;

    loop {
        let y_days = if is_leap(year) { 366 } else { 365 };
        if remaining < y_days {
            break;
        }
        remaining -= y_days;
        year += 1;
    }

    let month_lengths: [u32; 12] = [
        31,
        if is_leap(year) { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];

    let mut month = 1u32;
    for &days_in_month in &month_lengths {
        if remaining < days_in_month as u64 {
            break;
        }
        remaining -= days_in_month as u64;
        month += 1;
    }

    (year, month, remaining as u32 + 1)
}

fn is_leap(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> Store {
        // Use an in-memory SQLite database for tests
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        Store {
            conn,
            index: VectorIndex::new(),
            storage_path: PathBuf::from("/tmp/test"),
        }
    }

    #[test]
    fn store_and_recall() {
        let mut s = temp_store();
        let v = vec![1.0_f32, 0.0, 0.0];
        s.cmd_store("test", "hello world", v.clone(), None, None);
        let r = s.cmd_recall("test", 10, None);
        assert_eq!(r["ok"], true);
        assert_eq!(r["found"], true);
        assert_eq!(r["entries"][0]["content"], "hello world");
    }

    #[test]
    fn recall_empty_theme_returns_not_found() {
        let s = temp_store();
        let r = s.cmd_recall("nonexistent", 10, None);
        assert_eq!(r["ok"], true);
        assert_eq!(r["found"], false);
    }

    #[test]
    fn search_finds_similar() {
        let mut s = temp_store();
        s.cmd_store("a", "item a", vec![1.0_f32, 0.0, 0.0], None, None);
        s.cmd_store("b", "item b", vec![0.0_f32, 1.0, 0.0], None, None);
        let r = s.cmd_search(&[1.0, 0.0, 0.0], 5, 0, 0.5, None, None);
        assert_eq!(r["ok"], true);
        assert_eq!(r["results"][0]["content"], "item a");
    }

    #[test]
    fn search_empty_index_returns_available() {
        let s = temp_store();
        let r = s.cmd_search(&[1.0, 0.0], 5, 0, 0.0, None, None);
        assert_eq!(r["ok"], true);
        assert_eq!(r["available"], true);
        assert_eq!(r["results"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn delete_removes_entries() {
        let mut s = temp_store();
        s.cmd_store("del_me", "gone", vec![1.0_f32, 0.0], None, None);
        let r = s.cmd_delete("del_me");
        assert_eq!(r["deleted"], true);
        let r2 = s.cmd_recall("del_me", 10, None);
        assert_eq!(r2["found"], false);
    }

    #[test]
    fn prune_age_based() {
        let mut s = temp_store();
        // Insert a very old entry by manipulating stored_at directly
        let old_ts = 0.0_f64; // 1970-01-01 — definitely expired
        s.conn
            .execute(
                "INSERT INTO entries (id, theme, content, vector, stored_at)
                 VALUES ('old-id', 'theme', 'old', X'0000803F', ?1)",
                params![old_ts],
            )
            .unwrap();
        s.index.push("old-id".into(), "theme".into(), vec![1.0_f32], None);
        let r = s.cmd_prune(30, 1000);
        assert_eq!(r["ok"], true);
        assert!(r["pruned"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn format_unix_secs_epoch() {
        assert_eq!(format_unix_secs(0), "1970-01-01 00:00:00");
    }

    #[test]
    fn format_unix_secs_known_date() {
        // verified: python3 -c "import datetime; print(datetime.datetime.utcfromtimestamp(1736936200))"
        assert_eq!(format_unix_secs(1_736_936_200), "2025-01-15 10:16:40");
    }

    // -----------------------------------------------------------------------
    // Session tests
    // -----------------------------------------------------------------------

    #[test]
    fn create_and_get_session() {
        let mut s = temp_store();
        let r = s.cmd_create_session("Test session");
        assert_eq!(r["ok"], true);
        let sid = r["session_id"].as_str().unwrap().to_string();
        assert_eq!(sid.len(), 8);

        let r2 = s.cmd_get_session(&sid);
        assert_eq!(r2["ok"], true);
        assert_eq!(r2["session"]["title"], "Test session");
        assert_eq!(r2["session"]["message_count"], 0);
    }

    #[test]
    fn create_session_default_title() {
        let mut s = temp_store();
        let r = s.cmd_create_session("");
        assert_eq!(r["ok"], true);
        let sid = r["session_id"].as_str().unwrap().to_string();
        let r2 = s.cmd_get_session(&sid);
        assert_eq!(r2["session"]["title"], "New chat");
    }

    #[test]
    fn get_session_not_found() {
        let s = temp_store();
        let r = s.cmd_get_session("deadbeef");
        assert_eq!(r["ok"], false);
        assert!(r["error"].as_str().unwrap().contains("not found"));
    }

    #[test]
    fn list_sessions_ordered_by_updated_at() {
        let mut s = temp_store();
        s.cmd_create_session("Session A");
        s.cmd_create_session("Session B");
        let r = s.cmd_list_sessions(50, 0);
        assert_eq!(r["ok"], true);
        assert_eq!(r["sessions"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn list_sessions_pagination() {
        let mut s = temp_store();
        for i in 0..5 {
            s.cmd_create_session(&format!("Session {i}"));
        }
        let r = s.cmd_list_sessions(3, 0);
        assert_eq!(r["sessions"].as_array().unwrap().len(), 3);
        let r2 = s.cmd_list_sessions(3, 3);
        assert_eq!(r2["sessions"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn update_session_title() {
        let mut s = temp_store();
        let r = s.cmd_create_session("Original");
        let sid = r["session_id"].as_str().unwrap().to_string();

        let ur = s.cmd_update_session(&sid, Some("Renamed".to_string()), None);
        assert_eq!(ur["ok"], true);

        let gr = s.cmd_get_session(&sid);
        assert_eq!(gr["session"]["title"], "Renamed");
    }

    #[test]
    fn update_session_rolling_summary() {
        let mut s = temp_store();
        let r = s.cmd_create_session("chat");
        let sid = r["session_id"].as_str().unwrap().to_string();

        s.cmd_update_session(&sid, None, Some("User asked about Rust.".to_string()));

        let gr = s.cmd_get_session(&sid);
        assert_eq!(gr["session"]["rolling_summary"], "User asked about Rust.");
    }

    #[test]
    fn store_increments_session_message_count() {
        let mut s = temp_store();
        let r = s.cmd_create_session("counting");
        let sid = r["session_id"].as_str().unwrap().to_string();

        s.cmd_store("log", "msg 1", vec![1.0_f32], None, Some(&sid));
        s.cmd_store("log", "msg 2", vec![1.0_f32], None, Some(&sid));

        let gr = s.cmd_get_session(&sid);
        assert_eq!(gr["session"]["message_count"], 2);
    }

    #[test]
    fn delete_session_removes_entries_and_keeps_global() {
        let mut s = temp_store();
        let r = s.cmd_create_session("to delete");
        let sid = r["session_id"].as_str().unwrap().to_string();

        s.cmd_store("log", "session entry", vec![1.0_f32, 0.0], None, Some(&sid));
        s.cmd_store("log", "global entry", vec![1.0_f32, 0.0], None, None);

        let dr = s.cmd_delete_session(&sid);
        assert_eq!(dr["ok"], true);
        assert_eq!(dr["deleted_entries"], 1);

        // Session record gone
        let gr = s.cmd_get_session(&sid);
        assert_eq!(gr["ok"], false);

        // Global entry survives
        let rr = s.cmd_recall("log", 10, None);
        let entries = rr["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["content"], "global entry");
    }

    #[test]
    fn search_session_scoped_includes_global_excludes_other_sessions() {
        let mut s = temp_store();
        let r = s.cmd_create_session("scoped");
        let sid = r["session_id"].as_str().unwrap().to_string();

        s.cmd_store("t", "global item", vec![1.0_f32, 0.0], None, None);
        s.cmd_store("t", "my session item", vec![1.0_f32, 0.0], None, Some(&sid));
        s.cmd_store("t", "other session item", vec![1.0_f32, 0.0], None, Some("other000"));

        let r = s.cmd_search(&[1.0, 0.0], 10, 0, 0.0, None, Some(&sid));
        let contents: Vec<&str> = r["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["content"].as_str().unwrap())
            .collect();
        assert!(contents.contains(&"global item"));
        assert!(contents.contains(&"my session item"));
        assert!(!contents.contains(&"other session item"));
    }

    #[test]
    fn search_results_include_session_id_field() {
        let mut s = temp_store();
        s.cmd_store("t", "global", vec![1.0_f32, 0.0], None, None);
        let r = s.cmd_search(&[1.0, 0.0], 5, 0, 0.0, None, None);
        let result = &r["results"][0];
        assert!(result.get("session_id").is_some()); // field present (may be null)
    }

    #[test]
    fn recall_session_scoped() {
        let mut s = temp_store();
        let r = s.cmd_create_session("recall test");
        let sid = r["session_id"].as_str().unwrap().to_string();

        s.cmd_store("log", "mine", vec![1.0_f32], None, Some(&sid));
        s.cmd_store("log", "theirs", vec![1.0_f32], None, Some("other000"));

        let rr = s.cmd_recall("log", 10, Some(&sid));
        let entries = rr["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["content"], "mine");
    }

    #[test]
    fn prune_session_entries_not_age_pruned() {
        let mut s = temp_store();
        let r = s.cmd_create_session("active");
        let sid = r["session_id"].as_str().unwrap().to_string();

        let old_ts = 0.0_f64; // 1970-01-01

        // Old session entry — must NOT be pruned
        s.conn
            .execute(
                "INSERT INTO entries (id, theme, content, vector, stored_at, session_id)
                 VALUES ('sess-old', 'theme', 'session old', X'0000803F', ?1, ?2)",
                params![old_ts, sid],
            )
            .unwrap();
        s.index.push("sess-old".into(), "theme".into(), vec![1.0_f32], Some(sid.clone()));

        // Old global entry — MUST be pruned
        s.conn
            .execute(
                "INSERT INTO entries (id, theme, content, vector, stored_at)
                 VALUES ('glob-old', 'theme', 'global old', X'0000803F', ?1)",
                params![old_ts],
            )
            .unwrap();
        s.index.push("glob-old".into(), "theme".into(), vec![1.0_f32], None);

        let r = s.cmd_prune(30, 1000);
        assert_eq!(r["ok"], true);
        assert_eq!(r["pruned"], 1); // only the global entry

        // Session entry still present
        let rr = s.cmd_recall("theme", 10, Some(&sid));
        assert_eq!(rr["entries"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn now_iso8601_format() {
        let s = now_iso8601();
        // Should match YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(s.len(), 20);
        assert_eq!(&s[10..11], "T");
        assert_eq!(&s[19..], "Z");
    }
}
