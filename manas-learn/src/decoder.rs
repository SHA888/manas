use crate::embedder::Embedder;
use crate::tokenizer::Tokenizer;
use manas_core::Network;

pub struct DecodeResult {
    pub tokens: Vec<(String, f32)>,
    pub output_norm: f32,
}

pub fn decode(
    network: &Network,
    embedder: &Embedder,
    tokenizer: &Tokenizer,
    text: &str,
) -> DecodeResult {
    let mut temp_tokenizer = tokenizer.clone();
    let tokens = temp_tokenizer.encode(text);
    if tokens.is_empty() {
        return DecodeResult {
            tokens: Vec::new(),
            output_norm: 0.0,
        };
    }

    let mut temp_embedder = embedder.clone();
    for &id in &tokens {
        temp_embedder.embed_or_init(id);
    }
    let input = temp_embedder.average_embed(&tokens);

    if network.layers.is_empty() {
        return DecodeResult {
            tokens: Vec::new(),
            output_norm: 0.0,
        };
    }

    let output = network.forward(&input);
    let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();

    let mut scored: Vec<(String, f32)> = Vec::new();
    for (&tid, emb) in &embedder.table {
        let sim = cosine_similarity(&output, emb);
        if let Some(word) = tokenizer.decode(tid) {
            scored.push((word.to_string(), sim));
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(20);

    DecodeResult {
        tokens: scored,
        output_norm,
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}
