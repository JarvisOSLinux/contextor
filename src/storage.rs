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

    fn push(&mut self, id: String, theme: String, vector: Vec<f32>) {
        self.entries.push(IndexEntry { id, theme, vector });
    }

    fn remove_theme(&mut self, theme: &str) {
        self.entries.retain(|e| e.theme != theme);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Dimension of the first stored vector, or 0 if empty.
    fn dim(&self) -> usize {
        self.entries.first().map(|e| e.vector.len()).unwrap_or(0)
    }

    /// Score every entry, optionally restricting to a theme.
    /// Returns (score, id) pairs sorted descending.
    fn search(
        &self,
        query: &[f32],
        min_score: f32,
        theme_filter: Option<&str>,
    ) -> Vec<(f32, &str)> {
        let mut scored: Vec<(f32, &str)> = self
            .entries
            .iter()
            .filter(|e| theme_filter.map_or(true, |t| e.theme == t))
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
    // Commands
    // -----------------------------------------------------------------------

    pub fn cmd_store(
        &mut self,
        theme: &str,
        content: &str,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Value {
        let id = Uuid::new_v4().to_string();
        let stored_at = now_f64();
        let blob = encode_vector(&vector);
        let meta_str = metadata.as_ref().map(|m| m.to_string());

        match self.conn.execute(
            "INSERT INTO entries (id, theme, content, vector, stored_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, theme, content, blob, stored_at, meta_str],
        ) {
            Ok(_) => {
                self.index.push(id, theme.to_string(), vector);
                serde_json::json!({ "ok": true, "theme": theme })
            }
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_recall(&self, theme: &str, limit: usize) -> Value {
        let result = (|| -> rusqlite::Result<Vec<Value>> {
            let mut stmt = self.conn.prepare(
                "SELECT content, theme, stored_at, metadata
                 FROM entries
                 WHERE theme = ?1
                 ORDER BY stored_at ASC
                 LIMIT ?2",
            )?;
            stmt.query_map(params![theme, limit as i64], |row| {
                let content: String = row.get(0)?;
                let theme: String = row.get(1)?;
                let stored_at: f64 = row.get(2)?;
                let meta_raw: Option<String> = row.get(3)?;
                let metadata: Value = meta_raw
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or(Value::Null);
                Ok(serde_json::json!({
                    "content": content,
                    "theme": theme,
                    "stored_at": stored_at,
                    "metadata": metadata,
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
    ) -> Value {
        let scored = self.index.search(query, min_score, theme_filter);

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
            "SELECT id, content, theme, stored_at FROM entries WHERE id IN ({})",
            placeholders
        );

        let result = (|| -> rusqlite::Result<Vec<(String, String, String, f64)>> {
            let mut stmt = self.conn.prepare(&sql)?;
            stmt.query_map(rusqlite::params_from_iter(ids.iter()), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?,
                ))
            })
            .and_then(|rows| rows.collect())
        })();

        match result {
            Ok(rows) => {
                // Map id -> row
                let mut content_map: std::collections::HashMap<
                    String,
                    (String, String, f64),
                > = rows
                    .into_iter()
                    .map(|(id, content, theme, stored_at)| {
                        (id, (content, theme, stored_at))
                    })
                    .collect();

                // Rebuild in score order (preserving rank)
                let results: Vec<Value> = window
                    .iter()
                    .filter_map(|(score, id)| {
                        content_map.remove(*id).map(|(content, theme, stored_at)| {
                            serde_json::json!({
                                "content": content,
                                "theme": theme,
                                "score": score,
                                "stored_at": stored_at,
                            })
                        })
                    })
                    .collect();

                serde_json::json!({ "ok": true, "results": results, "available": true })
            }
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        }
    }

    pub fn cmd_list(&self) -> Value {
        let result = (|| -> rusqlite::Result<Vec<Value>> {
            let mut stmt = self.conn.prepare(
                "SELECT theme, COUNT(*) AS entries, MAX(stored_at) AS last_modified
                 FROM entries
                 GROUP BY theme
                 ORDER BY theme",
            )?;
            stmt.query_map([], |row| {
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

        // 1. Age-based pruning
        match self
            .conn
            .execute("DELETE FROM entries WHERE stored_at < ?1", params![cutoff])
        {
            Ok(n) => pruned += n,
            Err(e) => return serde_json::json!({ "ok": false, "error": e.to_string() }),
        }

        // 2. Per-theme count pruning: find themes that still exceed the limit
        let themes_over_limit: Vec<(String, i64)> = {
            let result = (|| -> rusqlite::Result<Vec<(String, i64)>> {
                let mut stmt = self.conn.prepare(
                    "SELECT theme, COUNT(*) FROM entries
                     GROUP BY theme
                     HAVING COUNT(*) > ?1",
                )?;
                stmt.query_map(params![max_per_theme as i64], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })
                .and_then(|rows| rows.collect())
            })();
            match result {
                Ok(v) => v,
                Err(e) => return serde_json::json!({ "ok": false, "error": e.to_string() }),
            }
        };

        for (theme, count) in themes_over_limit {
            let excess = (count as usize).saturating_sub(max_per_theme);
            match self.conn.execute(
                "DELETE FROM entries
                 WHERE id IN (
                     SELECT id FROM entries
                     WHERE theme = ?1
                     ORDER BY stored_at ASC
                     LIMIT ?2
                 )",
                params![theme, excess as i64],
            ) {
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
            ON entries (stored_at);",
    )?;
    Ok(())
}

fn load_index(conn: &Connection) -> Result<VectorIndex, Box<dyn std::error::Error>> {
    let mut stmt =
        conn.prepare("SELECT id, theme, vector FROM entries ORDER BY stored_at ASC")?;

    let rows: rusqlite::Result<Vec<(String, String, Vec<u8>)>> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Vec<u8>>(2)?,
            ))
        })?
        .collect();

    let mut index = VectorIndex::new();
    for (id, theme, blob) in rows? {
        index.push(id, theme, decode_vector(&blob));
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
        s.cmd_store("test", "hello world", v.clone(), None);
        let r = s.cmd_recall("test", 10);
        assert_eq!(r["ok"], true);
        assert_eq!(r["found"], true);
        assert_eq!(r["entries"][0]["content"], "hello world");
    }

    #[test]
    fn recall_empty_theme_returns_not_found() {
        let s = temp_store();
        let r = s.cmd_recall("nonexistent", 10);
        assert_eq!(r["ok"], true);
        assert_eq!(r["found"], false);
    }

    #[test]
    fn search_finds_similar() {
        let mut s = temp_store();
        s.cmd_store("a", "item a", vec![1.0_f32, 0.0, 0.0], None);
        s.cmd_store("b", "item b", vec![0.0_f32, 1.0, 0.0], None);
        let r = s.cmd_search(&[1.0, 0.0, 0.0], 5, 0, 0.5, None);
        assert_eq!(r["ok"], true);
        assert_eq!(r["results"][0]["content"], "item a");
    }

    #[test]
    fn search_empty_index_returns_available() {
        let s = temp_store();
        let r = s.cmd_search(&[1.0, 0.0], 5, 0, 0.0, None);
        assert_eq!(r["ok"], true);
        assert_eq!(r["available"], true);
        assert_eq!(r["results"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn delete_removes_entries() {
        let mut s = temp_store();
        s.cmd_store("del_me", "gone", vec![1.0_f32, 0.0], None);
        let r = s.cmd_delete("del_me");
        assert_eq!(r["deleted"], true);
        let r2 = s.cmd_recall("del_me", 10);
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
        s.index.push(
            "old-id".into(),
            "theme".into(),
            vec![1.0_f32],
        );
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
}
