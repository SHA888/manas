use crate::{TeachItem, answer_stopwords, sentence_split};
use manas_core::{ManasError, Source};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAGIC: u32 = 0x4352_534d; // "MSRC" little-endian
const VERSION: u32 = 1;
const KIND_FILE: u8 = 1;
const KIND_RAW: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StoredSourceKind {
    File,
    Raw,
}

impl StoredSourceKind {
    fn to_u8(self) -> u8 {
        match self {
            StoredSourceKind::File => KIND_FILE,
            StoredSourceKind::Raw => KIND_RAW,
        }
    }

    fn from_u8(value: u8, path: &Path) -> Result<Self, ManasError> {
        match value {
            KIND_FILE => Ok(StoredSourceKind::File),
            KIND_RAW => Ok(StoredSourceKind::Raw),
            _ => Err(corrupt(path, format!("unknown source kind: {}", value))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StoredSourceChunk {
    pub(crate) chunk_id: u64,
    pub(crate) chunk_text: String,
    pub(crate) normalized_text: String,
    pub(crate) tokens: Vec<String>,
    pub(crate) chunk_fingerprint: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StoredSource {
    pub(crate) source_id: u64,
    pub(crate) source_kind: StoredSourceKind,
    pub(crate) source_path: String,
    pub(crate) file_type: String,
    pub(crate) fingerprint: u64,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
    pub(crate) chunks: Vec<StoredSourceChunk>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SourceStore {
    sources: BTreeMap<String, StoredSource>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SourceStoreSnippet {
    pub(crate) text: String,
    pub(crate) normalized_text: String,
    pub(crate) tokens: Vec<String>,
    pub(crate) source: String,
}

pub(crate) fn source_store_path(brain_path: &Path) -> PathBuf {
    let mut p = brain_path.to_path_buf();
    let ext = p
        .extension()
        .map(|e| format!("{}.sources", e.to_string_lossy()))
        .unwrap_or_else(|| "sources".to_string());
    p.set_extension(ext);
    p
}

pub(crate) fn fnv1a64(input: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET;
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

pub(crate) fn normalize_source_text(text: &str) -> String {
    let mut out = String::new();
    let mut last_was_space = true;

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' {
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            last_was_space = false;
        } else if !last_was_space {
            out.push(' ');
            last_was_space = true;
        }
    }

    out.trim().to_string()
}

pub(crate) fn tokenize_source_text(text: &str) -> Vec<String> {
    let normalized = normalize_source_text(text);
    let stopwords = answer_stopwords();
    let mut tokens = Vec::new();
    for token in normalized
        .split_whitespace()
        .filter(|token| !stopwords.contains(*token))
    {
        push_unique_token(&mut tokens, token);
        if let Some(stem) = simple_search_stem(token) {
            push_unique_token(&mut tokens, &stem);
        }
    }
    tokens
}

fn push_unique_token(tokens: &mut Vec<String>, token: &str) {
    if !tokens.iter().any(|existing| existing == token) {
        tokens.push(token.to_string());
    }
}

fn simple_search_stem(token: &str) -> Option<String> {
    if token.len() <= 4
        || token.ends_with("'s")
        || token.ends_with("as")
        || token.ends_with("is")
        || token.ends_with("ss")
        || token.ends_with("us")
    {
        return None;
    }

    token.strip_suffix('s').map(ToString::to_string)
}

pub(crate) fn fingerprint_source(file_type: &str, normalized_text: &str) -> u64 {
    fnv1a64(&format!("{}\0{}", file_type, normalized_text))
}

pub(crate) fn fingerprint_chunk(normalized_text: &str) -> u64 {
    fnv1a64(normalized_text)
}

pub(crate) fn chunk_source_text(text: &str) -> Vec<StoredSourceChunk> {
    sentence_split(text)
        .into_iter()
        .filter_map(|sentence| {
            let chunk_text = sentence.trim().to_string();
            let normalized_text = normalize_source_text(&chunk_text);
            let tokens = tokenize_source_text(&normalized_text);
            if normalized_text.is_empty() || tokens.is_empty() {
                return None;
            }
            Some((chunk_text, normalized_text, tokens))
        })
        .enumerate()
        .map(
            |(idx, (chunk_text, normalized_text, tokens))| StoredSourceChunk {
                chunk_id: idx as u64,
                chunk_fingerprint: fingerprint_chunk(&normalized_text),
                chunk_text,
                normalized_text,
                tokens,
            },
        )
        .collect()
}

impl SourceStore {
    pub(crate) fn new() -> Self {
        Self {
            sources: BTreeMap::new(),
        }
    }

    pub(crate) fn load_from_file(path: &Path) -> Result<Self, ManasError> {
        if !path.exists() {
            return Ok(Self::new());
        }

        let buf = std::fs::read(path).map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut cursor = &buf[..];

        let magic = read_u32(&mut cursor, path)?;
        if magic != MAGIC {
            return Err(corrupt(
                path,
                format!("bad source store magic: {:#x}", magic),
            ));
        }

        let version = read_u32(&mut cursor, path)?;
        if version != VERSION {
            return Err(corrupt(
                path,
                format!("unsupported source store version: {}", version),
            ));
        }

        let source_count = read_u32(&mut cursor, path)? as usize;
        let mut store = Self::new();
        for _ in 0..source_count {
            let source_id = read_u64(&mut cursor, path)?;
            let source_kind = StoredSourceKind::from_u8(read_u8(&mut cursor, path)?, path)?;
            let source_path = read_string(&mut cursor, path)?;
            let file_type = read_string(&mut cursor, path)?;
            let fingerprint = read_u64(&mut cursor, path)?;
            let created_at = read_u64(&mut cursor, path)?;
            let updated_at = read_u64(&mut cursor, path)?;
            let chunk_count = read_u32(&mut cursor, path)? as usize;

            let mut chunks = Vec::with_capacity(chunk_count);
            for _ in 0..chunk_count {
                let chunk_id = read_u64(&mut cursor, path)?;
                let chunk_text = read_string(&mut cursor, path)?;
                let normalized_text = read_string(&mut cursor, path)?;
                let token_count = read_u32(&mut cursor, path)? as usize;
                let mut tokens = Vec::with_capacity(token_count);
                for _ in 0..token_count {
                    tokens.push(read_string(&mut cursor, path)?);
                }
                let chunk_fingerprint = read_u64(&mut cursor, path)?;
                chunks.push(StoredSourceChunk {
                    chunk_id,
                    chunk_text,
                    normalized_text,
                    tokens,
                    chunk_fingerprint,
                });
            }

            let source = StoredSource {
                source_id,
                source_kind,
                source_path,
                file_type,
                fingerprint,
                created_at,
                updated_at,
                chunks,
            };
            store.sources.insert(source.identity_key(), source);
        }

        if !cursor.is_empty() {
            return Err(corrupt(path, "trailing bytes in source store"));
        }

        Ok(store)
    }

    pub(crate) fn save_to_file(&self, path: &Path) -> Result<(), ManasError> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.sources.len() as u32).to_le_bytes());

        for source in self.sources.values() {
            buf.extend_from_slice(&source.source_id.to_le_bytes());
            buf.push(source.source_kind.to_u8());
            write_string(&mut buf, &source.source_path);
            write_string(&mut buf, &source.file_type);
            buf.extend_from_slice(&source.fingerprint.to_le_bytes());
            buf.extend_from_slice(&source.created_at.to_le_bytes());
            buf.extend_from_slice(&source.updated_at.to_le_bytes());
            buf.extend_from_slice(&(source.chunks.len() as u32).to_le_bytes());

            for chunk in &source.chunks {
                buf.extend_from_slice(&chunk.chunk_id.to_le_bytes());
                write_string(&mut buf, &chunk.chunk_text);
                write_string(&mut buf, &chunk.normalized_text);
                buf.extend_from_slice(&(chunk.tokens.len() as u32).to_le_bytes());
                for token in &chunk.tokens {
                    write_string(&mut buf, token);
                }
                buf.extend_from_slice(&chunk.chunk_fingerprint.to_le_bytes());
            }
        }

        std::fs::write(path, &buf).map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }

    pub(crate) fn upsert_teach_item(&mut self, item: &TeachItem) -> bool {
        let (source_kind, source_path, file_type) = match &item.source {
            Source::LocalFile { path } => {
                let file_type = Path::new(path)
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                (StoredSourceKind::File, path.clone(), file_type)
            }
            Source::RawText => (
                StoredSourceKind::Raw,
                "raw text".to_string(),
                "raw".to_string(),
            ),
            Source::Internet { .. } | Source::Unknown => return false,
        };

        let normalized_text = normalize_source_text(&item.text);
        if normalized_text.is_empty() {
            return false;
        }

        let fingerprint = fingerprint_source(&file_type, &normalized_text);
        let identity_key = match source_kind {
            StoredSourceKind::File => source_path.clone(),
            StoredSourceKind::Raw => format!("raw:{:016x}", fingerprint),
        };
        let chunks = chunk_source_text(&item.text);
        if chunks.is_empty() {
            return false;
        }

        let now = current_timestamp();
        if let Some(existing) = self.sources.get_mut(&identity_key) {
            if existing.fingerprint == fingerprint {
                return false;
            }
            existing.file_type = file_type;
            existing.fingerprint = fingerprint;
            existing.updated_at = now;
            existing.chunks = chunks;
            return true;
        }

        self.sources.insert(
            identity_key.clone(),
            StoredSource {
                source_id: fnv1a64(&identity_key),
                source_kind,
                source_path,
                file_type,
                fingerprint,
                created_at: now,
                updated_at: now,
                chunks,
            },
        );
        true
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.sources.len()
    }

    pub(crate) fn chunk_count(&self) -> usize {
        self.sources
            .values()
            .map(|source| source.chunks.len())
            .sum()
    }

    pub(crate) fn all_snippets(&self) -> Vec<SourceStoreSnippet> {
        let mut snippets = Vec::new();
        for source in self.sources.values() {
            for chunk in &source.chunks {
                snippets.push(SourceStoreSnippet {
                    text: chunk.chunk_text.clone(),
                    normalized_text: chunk.normalized_text.clone(),
                    tokens: chunk.tokens.clone(),
                    source: source.source_path.clone(),
                });
            }
        }
        snippets
    }

    pub(crate) fn file_sources(&self) -> Vec<(String, usize)> {
        self.sources
            .values()
            .filter(|source| source.source_kind == StoredSourceKind::File)
            .map(|source| (source.source_path.clone(), source.chunks.len()))
            .collect()
    }
}

impl StoredSource {
    fn identity_key(&self) -> String {
        match self.source_kind {
            StoredSourceKind::File => self.source_path.clone(),
            StoredSourceKind::Raw => format!("raw:{:016x}", self.fingerprint),
        }
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn write_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn read_u8(cursor: &mut &[u8], path: &Path) -> Result<u8, ManasError> {
    if cursor.is_empty() {
        return Err(corrupt(path, "unexpected end of source store"));
    }
    let value = cursor[0];
    *cursor = &cursor[1..];
    Ok(value)
}

fn read_u32(cursor: &mut &[u8], path: &Path) -> Result<u32, ManasError> {
    let bytes = take(cursor, 4, path)?;
    let mut out = [0u8; 4];
    out.copy_from_slice(bytes);
    Ok(u32::from_le_bytes(out))
}

fn read_u64(cursor: &mut &[u8], path: &Path) -> Result<u64, ManasError> {
    let bytes = take(cursor, 8, path)?;
    let mut out = [0u8; 8];
    out.copy_from_slice(bytes);
    Ok(u64::from_le_bytes(out))
}

fn read_string(cursor: &mut &[u8], path: &Path) -> Result<String, ManasError> {
    let len = read_u32(cursor, path)? as usize;
    let bytes = take(cursor, len, path)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| corrupt(path, "invalid UTF-8 in source store"))
}

fn take<'a>(cursor: &mut &'a [u8], len: usize, path: &Path) -> Result<&'a [u8], ManasError> {
    if cursor.len() < len {
        return Err(corrupt(path, "truncated source store"));
    }
    let (head, tail) = cursor.split_at(len);
    *cursor = tail;
    Ok(head)
}

fn corrupt(path: &Path, reason: impl Into<String>) -> ManasError {
    ManasError::CorruptFile {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("manas_source_store_{}_{}", name, nanos));
        path
    }

    fn file_item(path: &str, text: &str) -> TeachItem {
        TeachItem {
            text: text.to_string(),
            source: Source::LocalFile {
                path: path.to_string(),
            },
        }
    }

    fn raw_item(text: &str) -> TeachItem {
        TeachItem {
            text: text.to_string(),
            source: Source::RawText,
        }
    }

    #[test]
    fn source_store_path_derives_sources_sidecar() {
        assert_eq!(
            source_store_path(Path::new("brain.manas")),
            PathBuf::from("brain.manas.sources")
        );
    }

    #[test]
    fn source_store_roundtrip_preserves_ai_ready_fields() {
        let path = temp_path("roundtrip");
        let mut store = SourceStore::new();
        store.upsert_teach_item(&file_item(
            "teach/identity.md",
            "Manas is a local-first AI memory system.",
        ));

        store.save_to_file(&path).unwrap();
        let loaded = SourceStore::load_from_file(&path).unwrap();

        assert_eq!(loaded.entry_count(), 1);
        assert_eq!(loaded.chunk_count(), 1);
        let snippets = loaded.all_snippets();
        assert_eq!(snippets[0].text, "Manas is a local-first AI memory system.");
        assert_eq!(
            snippets[0].normalized_text,
            "manas is a local first ai memory system"
        );
        assert!(snippets[0].tokens.contains(&"manas".to_string()));
        assert!(snippets[0].tokens.contains(&"local".to_string()));
        assert!(snippets[0].tokens.contains(&"first".to_string()));

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn source_store_deduplicates_and_updates_file_sources() {
        let mut store = SourceStore::new();
        assert!(store.upsert_teach_item(&file_item(
            "teach/repeat.md",
            "Manas remembers repeated files.",
        )));
        assert!(!store.upsert_teach_item(&file_item(
            "teach/repeat.md",
            "Manas remembers repeated files.",
        )));
        assert_eq!(store.entry_count(), 1);
        assert_eq!(store.chunk_count(), 1);

        assert!(store.upsert_teach_item(&file_item(
            "teach/repeat.md",
            "Manas updates changed source files safely.",
        )));
        assert_eq!(store.entry_count(), 1);
        assert_eq!(store.chunk_count(), 1);
        assert_eq!(
            store.all_snippets()[0].text,
            "Manas updates changed source files safely."
        );
    }

    #[test]
    fn source_store_deduplicates_raw_text_by_fingerprint() {
        let mut store = SourceStore::new();
        assert!(store.upsert_teach_item(&raw_item("Manas learns raw text.")));
        assert!(!store.upsert_teach_item(&raw_item("Manas learns raw text.")));
        assert!(store.upsert_teach_item(&raw_item("Manas learns different raw text.")));
        assert_eq!(store.entry_count(), 2);
    }

    #[test]
    fn source_store_rejects_corrupt_file() {
        let path = temp_path("corrupt");
        std::fs::write(&path, b"bad").unwrap();
        let result = SourceStore::load_from_file(&path);
        assert!(matches!(result, Err(ManasError::CorruptFile { .. })));
        std::fs::remove_file(path).unwrap();
    }
}
