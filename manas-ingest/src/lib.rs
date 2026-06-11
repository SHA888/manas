pub mod file_reader;
pub mod folder_walker;
pub mod normalizer;
pub mod raw_text;

mod format;

use std::path::PathBuf;
use manas_core::{ManasError, Source};

pub const CHUNK_SIZE: usize = 512;
pub const CHUNK_OVERLAP: usize = 64;

pub enum IngestSource {
    Text(String),
    File(PathBuf),
    Folder(PathBuf),
    Url(String),
}

pub struct TextChunk {
    pub text: String,
    pub source: Source,
    pub chunk_id: u64,
    pub file_path: Option<String>,
    pub url: Option<String>,
}

pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let byte_positions: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let char_count = byte_positions.len();

    if char_count <= chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start_char = 0;

    while start_char < char_count {
        let end_char = (start_char + chunk_size).min(char_count);
        let end_byte = if end_char < char_count { byte_positions[end_char] } else { text.len() };

        if end_char >= char_count {
            chunks.push(text[byte_positions[start_char]..].to_string());
            break;
        }

        let slice_start = byte_positions[start_char];
        let slice = &text[slice_start..end_byte];

        let mut split_byte = end_byte;
        if let Some(pos) = slice.rfind('\n') {
            split_byte = slice_start + pos + 1;
        } else if let Some(pos) = slice.rfind(". ") {
            split_byte = slice_start + pos + 2;
        } else if let Some(pos) = slice.rfind(' ') {
            split_byte = slice_start + pos + 1;
        }

        chunks.push(text[slice_start..split_byte].to_string());

        let split_end = byte_positions.iter().position(|&b| b >= split_byte).unwrap_or(char_count);
        if split_end >= char_count {
            break;
        }

        let overlap_chars = overlap.min(split_end);
        start_char = split_end.saturating_sub(overlap_chars);
    }

    chunks
}

pub struct IngestPipeline;

impl IngestPipeline {
    pub fn new() -> Self {
        IngestPipeline
    }

    pub fn process(&self, source: IngestSource) -> Result<Vec<TextChunk>, ManasError> {
        match source {
            IngestSource::Text(text) => {
                let normalized = normalizer::normalize(&text);
                let chunks = chunk_text(&normalized, CHUNK_SIZE, CHUNK_OVERLAP);
                Ok(chunks.into_iter().enumerate().map(|(i, chunk)| {
                    TextChunk {
                        text: chunk,
                        source: Source::RawText,
                        chunk_id: i as u64,
                        file_path: None,
                        url: None,
                    }
                }).collect())
            }
            IngestSource::File(path) => {
                let content = file_reader::read_file(&path)?;
                let normalized = normalizer::normalize(&content);
                let chunks = chunk_text(&normalized, CHUNK_SIZE, CHUNK_OVERLAP);
                let path_str = path.display().to_string();
                Ok(chunks.into_iter().enumerate().map(|(i, chunk)| {
                    TextChunk {
                        text: chunk,
                        source: Source::LocalFile { path: path_str.clone() },
                        chunk_id: i as u64,
                        file_path: Some(path_str.clone()),
                        url: None,
                    }
                }).collect())
            }
            IngestSource::Folder(path) => {
                let entries = folder_walker::walk_folder(&path)?;
                let mut all_chunks = Vec::new();
                for entry in entries {
                    let normalized = normalizer::normalize(&entry.text);
                    let chunks = chunk_text(&normalized, CHUNK_SIZE, CHUNK_OVERLAP);
                    for (i, chunk) in chunks.into_iter().enumerate() {
                        all_chunks.push(TextChunk {
                            text: chunk,
                            source: Source::LocalFile { path: entry.path.clone() },
                            chunk_id: i as u64,
                            file_path: Some(entry.path.clone()),
                            url: None,
                        });
                    }
                }
                Ok(all_chunks)
            }
            IngestSource::Url(url) => {
                Ok(vec![TextChunk {
                    text: String::new(),
                    source: Source::Internet { url: url.clone() },
                    chunk_id: 0,
                    file_path: None,
                    url: Some(url),
                }])
            }
        }
    }
}

impl Default for IngestPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_short_text() {
        let chunks = chunk_text("hello world", 512, 64);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn chunk_long_text() {
        let long = "A. ".repeat(500);
        let chunks = chunk_text(&long, 512, 64);
        assert!(chunks.len() > 1);
        assert!(!chunks[0].is_empty());
    }

    #[test]
    fn raw_text_pipeline() {
        let pipeline = IngestPipeline::new();
        let chunks = pipeline.process(IngestSource::Text("hello world".into())).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
    }

    #[test]
    fn unsupported_extension() {
        let result = file_reader::parse_by_extension("hello", "xyz");
        assert!(result.is_err());
    }

    #[test]
    fn markdown_strips_syntax() {
        let md = "# Header\n**bold** text\n[link](url)\n```code```";
        let result = crate::format::markdown::parse(md);
        assert!(!result.contains('#'));
        assert!(!result.contains('*'));
        assert!(result.contains("bold"));
        assert!(result.contains("text"));
        assert!(result.contains("link"));
    }

    #[test]
    fn html_strips_tags() {
        let html = "<p>Hello <b>world</b></p><a href='x'>click</a>";
        let result = crate::format::html::parse(html);
        assert!(!result.contains('<'));
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));
        assert!(result.contains("click"));
    }

    #[test]
    fn json_flattens() {
        let json = r#"{"name": "manas", "version": 1}"#;
        let result = crate::format::json::parse(json);
        assert!(result.contains("name: manas"));
        assert!(result.contains("version: 1"));
    }

    #[test]
    fn normalizer_cleans() {
        let dirty = "hello\x00world  extra   spaces\n\n";
        let clean = normalizer::normalize(dirty);
        assert!(!clean.contains('\x00'));
        assert!(!clean.contains("  "));
    }
}
