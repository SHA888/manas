use crate::format;
use manas_core::ManasError;
use std::path::Path;

pub fn read_file(path: &Path) -> Result<String, ManasError> {
    let contents = std::fs::read_to_string(path).map_err(|e| ManasError::FileReadError {
        path: path.to_path_buf(),
        source: e,
    })?;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    parse_by_extension(&contents, &ext)
}

pub fn parse_by_extension(contents: &str, ext: &str) -> Result<String, ManasError> {
    match ext {
        "txt" | "" => Ok(format::plaintext::parse(contents)),
        "md" => Ok(format::markdown::parse(contents)),
        "rs" => Ok(format::rust_source::parse(contents)),
        "json" => Ok(format::json::parse(contents)),
        "toml" => Ok(format::toml::parse(contents)),
        "csv" => Ok(format::csv::parse(contents)),
        "html" | "htm" => Ok(format::html::parse(contents)),
        _ => Err(ManasError::UnsupportedFileType(ext.to_string())),
    }
}

pub fn supported_extension(ext: &str) -> bool {
    matches!(
        ext,
        "txt" | "md" | "rs" | "json" | "toml" | "csv" | "html" | "htm"
    )
}
