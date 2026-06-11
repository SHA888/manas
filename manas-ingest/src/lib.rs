pub mod file_reader;
pub mod folder_walker;
pub mod normalizer;
pub mod raw_text;

mod format;

use manas_core::{ManasError, Source};
use std::path::PathBuf;

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
    if text.is_empty() {
        return Vec::new();
    }

    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let char_count = chars.len();

    if char_count <= chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start_char = 0usize;

    while start_char < char_count {
        let hard_end_char = (start_char + chunk_size).min(char_count);

        let start_byte = chars[start_char].0;
        let hard_end_byte = if hard_end_char < char_count {
            chars[hard_end_char].0
        } else {
            text.len()
        };

        if hard_end_char >= char_count {
            let chunk = text[start_byte..].trim();
            if !chunk.is_empty() {
                chunks.push(chunk.to_string());
            }
            break;
        }

        let slice = &text[start_byte..hard_end_byte];

        let mut split_byte = hard_end_byte;

        if let Some(pos) = slice.rfind('\n') {
            split_byte = start_byte + pos + 1;
        } else if let Some(pos) = slice.rfind(". ") {
            split_byte = start_byte + pos + 2;
        } else if let Some(pos) = slice.rfind(' ') {
            split_byte = start_byte + pos + 1;
        }

        let split_char = chars
            .iter()
            .position(|(byte, _)| *byte >= split_byte)
            .unwrap_or(hard_end_char);

        // IMPORTANT: guarantee forward progress.
        // If natural split is too close, ignore it and use hard chunk end.
        let safe_split_char = if split_char <= start_char + overlap {
            hard_end_char
        } else {
            split_char
        };

        let safe_split_byte = if safe_split_char < char_count {
            chars[safe_split_char].0
        } else {
            text.len()
        };

        let chunk = text[start_byte..safe_split_byte].trim();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }

        if safe_split_char >= char_count {
            break;
        }

        let next_start = safe_split_char.saturating_sub(overlap);

        // Extra safety: never allow same/backward start.
        if next_start <= start_char {
            start_char = safe_split_char;
        } else {
            start_char = next_start;
        }
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
                Ok(chunks
                    .into_iter()
                    .enumerate()
                    .map(|(i, chunk)| TextChunk {
                        text: chunk,
                        source: Source::RawText,
                        chunk_id: i as u64,
                        file_path: None,
                        url: None,
                    })
                    .collect())
            }
            IngestSource::File(path) => {
                let content = file_reader::read_file(&path)?;
                let normalized = normalizer::normalize(&content);
                let chunks = chunk_text(&normalized, CHUNK_SIZE, CHUNK_OVERLAP);
                let path_str = path.display().to_string();
                Ok(chunks
                    .into_iter()
                    .enumerate()
                    .map(|(i, chunk)| TextChunk {
                        text: chunk,
                        source: Source::LocalFile {
                            path: path_str.clone(),
                        },
                        chunk_id: i as u64,
                        file_path: Some(path_str.clone()),
                        url: None,
                    })
                    .collect())
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
                            source: Source::LocalFile {
                                path: entry.path.clone(),
                            },
                            chunk_id: i as u64,
                            file_path: Some(entry.path.clone()),
                            url: None,
                        });
                    }
                }
                Ok(all_chunks)
            }
            IngestSource::Url(url) => Ok(vec![TextChunk {
                text: String::new(),
                source: Source::Internet { url: url.clone() },
                chunk_id: 0,
                file_path: None,
                url: Some(url),
            }]),
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
    fn chunk_text_never_loops_on_many_newlines() {
        let text = "# Title\n\n## Section\n\nShort line\n\nAnother short line\n\n".repeat(200);
        let chunks = chunk_text(&text, 512, 64);

        assert!(!chunks.is_empty());
        assert!(chunks.len() < 1000, "too many chunks, possible loop");
    }

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
        let chunks = pipeline
            .process(IngestSource::Text("hello world".into()))
            .unwrap();
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
