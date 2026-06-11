use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum ManasError {
    FileNotFound(PathBuf),
    CorruptFile {
        path: PathBuf,
        reason: String,
    },
    ChecksumMismatch,
    TokenizerError(String),
    EmbeddingError(String),
    BackpropError(String),
    UnsupportedFileType(String),
    FileReadError {
        path: PathBuf,
        source: std::io::Error,
    },
    NetworkError(String),
    ScraperError(String),
    SearchBackendError(String),
    GrowthFailed(String),
    LayerLimitReached,
    NeuronNotFound(u64),
    NeuronFrozen(u64),
}

impl fmt::Display for ManasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManasError::FileNotFound(path) => write!(f, "file not found: {}", path.display()),
            ManasError::CorruptFile { path, reason } => {
                write!(f, "corrupt file {}: {}", path.display(), reason)
            }
            ManasError::ChecksumMismatch => write!(f, "checksum mismatch"),
            ManasError::TokenizerError(msg) => write!(f, "tokenizer error: {}", msg),
            ManasError::EmbeddingError(msg) => write!(f, "embedding error: {}", msg),
            ManasError::BackpropError(msg) => write!(f, "backprop error: {}", msg),
            ManasError::UnsupportedFileType(ext) => write!(f, "unsupported file type: {}", ext),
            ManasError::FileReadError { path, source } => {
                write!(f, "failed to read {}: {}", path.display(), source)
            }
            ManasError::NetworkError(msg) => write!(f, "network error: {}", msg),
            ManasError::ScraperError(msg) => write!(f, "scraper error: {}", msg),
            ManasError::SearchBackendError(msg) => write!(f, "search backend error: {}", msg),
            ManasError::GrowthFailed(msg) => write!(f, "growth failed: {}", msg),
            ManasError::LayerLimitReached => write!(f, "layer limit reached"),
            ManasError::NeuronNotFound(id) => write!(f, "neuron {} not found", id),
            ManasError::NeuronFrozen(id) => write!(f, "neuron {} is frozen", id),
        }
    }
}

impl std::error::Error for ManasError {}

impl From<std::io::Error> for ManasError {
    fn from(err: std::io::Error) -> Self {
        ManasError::FileReadError {
            path: PathBuf::new(),
            source: err,
        }
    }
}
