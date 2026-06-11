use crate::file_reader;
use manas_core::ManasError;
use std::path::Path;

pub struct FileEntry {
    pub path: String,
    pub text: String,
}

pub fn walk_folder(path: &Path) -> Result<Vec<FileEntry>, ManasError> {
    let mut entries = Vec::new();

    if !path.is_dir() {
        return Err(ManasError::FileReadError {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "not a directory"),
        });
    }

    for entry in walkdir::WalkDir::new(path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let ext = entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !file_reader::supported_extension(&ext) {
            continue;
        }

        match file_reader::read_file(entry.path()) {
            Ok(text) => {
                entries.push(FileEntry {
                    path: entry.path().display().to_string(),
                    text,
                });
            }
            Err(ManasError::UnsupportedFileType(_)) => continue,
            Err(e) => return Err(e),
        }
    }

    Ok(entries)
}
