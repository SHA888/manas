use crate::source_store::{SourceStore, SourceStoreChunkRef};
use manas_core::ManasError;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAGIC: u32 = 0x5849_534d; // "MSIX" little-endian
const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct IndexedChunkRef {
    pub(crate) source_id: u64,
    pub(crate) chunk_id: u64,
    pub(crate) source_fingerprint: u64,
    pub(crate) chunk_fingerprint: u64,
}

impl IndexedChunkRef {
    fn to_store_ref(&self) -> SourceStoreChunkRef {
        SourceStoreChunkRef {
            source_id: self.source_id,
            chunk_id: self.chunk_id,
            source_fingerprint: self.source_fingerprint,
            chunk_fingerprint: self.chunk_fingerprint,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SourceIndex {
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
    pub(crate) source_store_fingerprint: u64,
    pub(crate) indexed_source_count: u32,
    pub(crate) indexed_chunk_count: u32,
    tokens: BTreeMap<String, Vec<IndexedChunkRef>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceIndexStatus {
    Enabled,
    Missing,
    Corrupt,
    Stale,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SourceIndexStats {
    pub(crate) status: SourceIndexStatus,
    pub(crate) indexed_tokens: usize,
    pub(crate) indexed_chunks: usize,
    pub(crate) bytes: Option<u64>,
}

pub(crate) fn source_index_path(brain_path: &Path) -> PathBuf {
    let mut p = brain_path.to_path_buf();
    let ext = p
        .extension()
        .map(|e| format!("{}.sourceindex", e.to_string_lossy()))
        .unwrap_or_else(|| "sourceindex".to_string());
    p.set_extension(ext);
    p
}

impl SourceIndex {
    pub(crate) fn new(
        source_store_fingerprint: u64,
        source_count: usize,
        chunk_count: usize,
    ) -> Self {
        let now = current_timestamp();
        Self {
            created_at: now,
            updated_at: now,
            source_store_fingerprint,
            indexed_source_count: source_count as u32,
            indexed_chunk_count: chunk_count as u32,
            tokens: BTreeMap::new(),
        }
    }

    pub(crate) fn build_from_store(store: &SourceStore) -> Self {
        let mut index = Self::new(
            store.source_store_fingerprint(),
            store.source_count(),
            store.chunk_count(),
        );

        for source in store.sources() {
            for chunk in &source.chunks {
                let reference = IndexedChunkRef {
                    source_id: source.source_id,
                    chunk_id: chunk.chunk_id,
                    source_fingerprint: source.fingerprint,
                    chunk_fingerprint: chunk.chunk_fingerprint,
                };
                let mut seen_tokens = BTreeSet::new();
                for token in &chunk.tokens {
                    if seen_tokens.insert(token.clone()) {
                        index
                            .tokens
                            .entry(token.clone())
                            .or_default()
                            .push(reference.clone());
                    }
                }
            }
        }

        for refs in index.tokens.values_mut() {
            refs.sort();
            refs.dedup();
        }

        index
    }

    pub(crate) fn load_from_file(path: &Path) -> Result<Self, ManasError> {
        if !path.exists() {
            return Ok(Self::new(0, 0, 0));
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
                format!("bad source index magic: {:#x}", magic),
            ));
        }

        let version = read_u32(&mut cursor, path)?;
        if version != VERSION {
            return Err(corrupt(
                path,
                format!("unsupported source index version: {}", version),
            ));
        }

        let created_at = read_u64(&mut cursor, path)?;
        let updated_at = read_u64(&mut cursor, path)?;
        let source_store_fingerprint = read_u64(&mut cursor, path)?;
        let indexed_source_count = read_u32(&mut cursor, path)?;
        let indexed_chunk_count = read_u32(&mut cursor, path)?;
        let token_count = read_u32(&mut cursor, path)? as usize;

        let mut tokens = BTreeMap::new();
        for _ in 0..token_count {
            let token = read_string(&mut cursor, path)?;
            let ref_count = read_u32(&mut cursor, path)? as usize;
            let mut refs = Vec::with_capacity(ref_count);
            for _ in 0..ref_count {
                refs.push(IndexedChunkRef {
                    source_id: read_u64(&mut cursor, path)?,
                    chunk_id: read_u64(&mut cursor, path)?,
                    source_fingerprint: read_u64(&mut cursor, path)?,
                    chunk_fingerprint: read_u64(&mut cursor, path)?,
                });
            }
            refs.sort();
            refs.dedup();
            tokens.insert(token, refs);
        }

        if !cursor.is_empty() {
            return Err(corrupt(path, "trailing bytes in source index"));
        }

        Ok(Self {
            created_at,
            updated_at,
            source_store_fingerprint,
            indexed_source_count,
            indexed_chunk_count,
            tokens,
        })
    }

    pub(crate) fn save_to_file(&self, path: &Path) -> Result<(), ManasError> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&self.created_at.to_le_bytes());
        buf.extend_from_slice(&self.updated_at.to_le_bytes());
        buf.extend_from_slice(&self.source_store_fingerprint.to_le_bytes());
        buf.extend_from_slice(&self.indexed_source_count.to_le_bytes());
        buf.extend_from_slice(&self.indexed_chunk_count.to_le_bytes());
        buf.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());

        for (token, refs) in &self.tokens {
            write_string(&mut buf, token);
            buf.extend_from_slice(&(refs.len() as u32).to_le_bytes());
            for reference in refs {
                buf.extend_from_slice(&reference.source_id.to_le_bytes());
                buf.extend_from_slice(&reference.chunk_id.to_le_bytes());
                buf.extend_from_slice(&reference.source_fingerprint.to_le_bytes());
                buf.extend_from_slice(&reference.chunk_fingerprint.to_le_bytes());
            }
        }

        std::fs::write(path, &buf).map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }

    pub(crate) fn lookup_candidates(
        &self,
        query_tokens: &[String],
        limit: usize,
    ) -> Vec<IndexedChunkRef> {
        if limit == 0 {
            return Vec::new();
        }

        let mut scored: BTreeMap<IndexedChunkRef, u32> = BTreeMap::new();
        let mut seen_tokens = BTreeSet::new();
        for token in query_tokens {
            if !seen_tokens.insert(token) {
                continue;
            }
            if let Some(refs) = self.tokens.get(token) {
                for reference in refs {
                    *scored.entry(reference.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut refs: Vec<(IndexedChunkRef, u32)> = scored.into_iter().collect();
        refs.sort_by(|(a_ref, a_score), (b_ref, b_score)| {
            b_score
                .cmp(a_score)
                .then_with(|| a_ref.source_id.cmp(&b_ref.source_id))
                .then_with(|| a_ref.chunk_id.cmp(&b_ref.chunk_id))
                .then_with(|| a_ref.source_fingerprint.cmp(&b_ref.source_fingerprint))
                .then_with(|| a_ref.chunk_fingerprint.cmp(&b_ref.chunk_fingerprint))
        });
        refs.truncate(limit);
        refs.into_iter().map(|(reference, _)| reference).collect()
    }

    pub(crate) fn is_fresh_for(&self, store: &SourceStore) -> bool {
        self.source_store_fingerprint == store.source_store_fingerprint()
            && self.indexed_source_count as usize == store.source_count()
            && self.indexed_chunk_count as usize == store.chunk_count()
            && self
                .tokens
                .values()
                .flatten()
                .all(|reference| store.resolve_ref(&reference.to_store_ref()).is_some())
    }

    pub(crate) fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn indexed_chunk_count(&self) -> usize {
        self.indexed_chunk_count as usize
    }

    #[cfg(test)]
    pub(crate) fn refs_for_token(&self, token: &str) -> Vec<IndexedChunkRef> {
        self.tokens.get(token).cloned().unwrap_or_default()
    }

    pub(crate) fn resolve_candidates(
        &self,
        store: &SourceStore,
        query_tokens: &[String],
        limit: usize,
    ) -> Vec<crate::source_store::SourceStoreSnippet> {
        self.lookup_candidates(query_tokens, limit)
            .into_iter()
            .filter_map(|reference| store.resolve_ref(&reference.to_store_ref()))
            .collect()
    }
}

pub(crate) fn source_index_stats(
    brain_path: &Path,
    store: Option<&SourceStore>,
) -> SourceIndexStats {
    let path = source_index_path(brain_path);
    let bytes = std::fs::metadata(&path).ok().map(|m| m.len());
    if bytes.is_none() {
        return SourceIndexStats {
            status: SourceIndexStatus::Missing,
            indexed_tokens: 0,
            indexed_chunks: 0,
            bytes,
        };
    }

    match SourceIndex::load_from_file(&path) {
        Ok(index) => {
            let stale = store.is_some_and(|store| !index.is_fresh_for(store));
            SourceIndexStats {
                status: if stale {
                    SourceIndexStatus::Stale
                } else {
                    SourceIndexStatus::Enabled
                },
                indexed_tokens: index.token_count(),
                indexed_chunks: index.indexed_chunk_count(),
                bytes,
            }
        }
        Err(_) => SourceIndexStats {
            status: SourceIndexStatus::Corrupt,
            indexed_tokens: 0,
            indexed_chunks: 0,
            bytes,
        },
    }
}

impl SourceIndexStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            SourceIndexStatus::Enabled => "enabled",
            SourceIndexStatus::Missing => "missing",
            SourceIndexStatus::Corrupt => "corrupt",
            SourceIndexStatus::Stale => "stale",
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
    String::from_utf8(bytes.to_vec()).map_err(|_| corrupt(path, "invalid UTF-8 in source index"))
}

fn take<'a>(cursor: &mut &'a [u8], len: usize, path: &Path) -> Result<&'a [u8], ManasError> {
    if cursor.len() < len {
        return Err(corrupt(path, "unexpected end of source index"));
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
    use crate::TeachItem;
    use manas_core::Source;
    use std::fs;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "manas_source_index_{}_{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("brain.manas.sourceindex")
    }

    fn file_item(path: &str, text: &str) -> TeachItem {
        TeachItem {
            text: text.to_string(),
            source: Source::LocalFile {
                path: path.to_string(),
            },
        }
    }

    fn store_with_sources() -> SourceStore {
        let mut store = SourceStore::new();
        store.upsert_teach_item(&file_item(
            "teach/identity.md",
            "Manas is a local-first AI memory system written in Rust.",
        ));
        store.upsert_teach_item(&file_item(
            "docs/base.md",
            "The source index helps Manas retrieve source chunks faster.",
        ));
        store
    }

    #[test]
    fn source_index_path_derives_sourceindex_sidecar() {
        assert_eq!(
            source_index_path(Path::new("brain.manas")),
            PathBuf::from("brain.manas.sourceindex")
        );
    }

    #[test]
    fn source_index_roundtrip_preserves_tokens_and_refs() {
        let store = store_with_sources();
        let index = SourceIndex::build_from_store(&store);
        let path = temp_path("roundtrip");

        index.save_to_file(&path).unwrap();
        let loaded = SourceIndex::load_from_file(&path).unwrap();

        assert_eq!(loaded.token_count(), index.token_count());
        assert_eq!(loaded.indexed_chunk_count(), 2);
        assert!(!loaded.refs_for_token("manas").is_empty());
        assert!(loaded.is_fresh_for(&store));

        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn source_index_builds_from_store_and_maps_tokens() {
        let store = store_with_sources();
        let index = SourceIndex::build_from_store(&store);

        assert_eq!(index.indexed_source_count, 2);
        assert_eq!(index.indexed_chunk_count(), 2);
        assert_eq!(index.refs_for_token("rust").len(), 1);
        assert_eq!(index.refs_for_token("manas").len(), 2);
    }

    #[test]
    fn source_index_does_not_serialize_full_chunk_text() {
        let store = store_with_sources();
        let index = SourceIndex::build_from_store(&store);
        let path = temp_path("no_text");

        index.save_to_file(&path).unwrap();
        let bytes = fs::read(&path).unwrap();
        let serialized = String::from_utf8_lossy(&bytes);

        assert!(!serialized.contains("local-first AI memory system"));
        assert!(!serialized.contains("source chunks faster"));

        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn source_index_stale_detection_catches_store_changes() {
        let mut store = store_with_sources();
        let index = SourceIndex::build_from_store(&store);
        assert!(index.is_fresh_for(&store));

        store.upsert_teach_item(&file_item(
            "teach/identity.md",
            "Manas changed identity source text.",
        ));

        assert!(!index.is_fresh_for(&store));
    }

    #[test]
    fn source_index_corrupt_file_returns_error() {
        let path = temp_path("corrupt");
        fs::write(&path, b"bad-index").unwrap();

        let result = SourceIndex::load_from_file(&path);

        assert!(matches!(result, Err(ManasError::CorruptFile { .. })));
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
