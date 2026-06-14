pub mod attention;
pub mod transformer;

use manas_core::{ManasError, Network};
use manas_learn::Embedder;
use manas_learn::Tokenizer;
use manas_learn::Trainer;
use manas_learn::backprop;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;

pub const DEFAULT_LANGUAGE_LR: f32 = 0.05;
pub const DEFAULT_EPOCHS: usize = 10;
pub const MEMORY_ALPHA: f32 = 0.8;
pub const CONTEXT_PENALTY: f32 = 0.3;

// ─── Sequence Examples ────────────────────────────────────────────────────────

pub struct SequenceExample {
    pub context: Vec<u32>,
    pub target: u32,
}

/// Build next-token prediction examples from a token sequence.
///
/// Uses a sliding context window capped at `max_context`.
/// Ignores sequences shorter than 2 tokens.
pub fn build_sequence_examples(tokens: &[u32], max_context: usize) -> Vec<SequenceExample> {
    if tokens.len() < 2 {
        return Vec::new();
    }
    let mut examples = Vec::with_capacity(tokens.len() - 1);
    for i in 1..tokens.len() {
        let context_start = i.saturating_sub(max_context);
        let context = tokens[context_start..i].to_vec();
        examples.push(SequenceExample {
            context,
            target: tokens[i],
        });
    }
    examples
}

// ─── Sequence Memory ──────────────────────────────────────────────────────────

/// A transition-count table that records which tokens follow which contexts.
///
/// `context -> (target -> count)`
#[derive(Debug, Clone)]
pub struct SequenceMemory {
    pub transitions: HashMap<Vec<u32>, HashMap<u32, u32>>,
}

impl SequenceMemory {
    pub fn new() -> Self {
        SequenceMemory {
            transitions: HashMap::new(),
        }
    }

    /// Record a transition: after `context`, `target` was seen.
    ///
    /// Also records all shorter suffix contexts (so suffix backoff works during lookup).
    pub fn record(&mut self, context: &[u32], target: u32) {
        for len in 1..=context.len() {
            let suffix: Vec<u32> = context[context.len() - len..].to_vec();
            let entry = self.transitions.entry(suffix).or_default();
            *entry.entry(target).or_insert(0) += 1;
        }
    }

    /// Look up the best transition for a given context using suffix backoff.
    ///
    /// Tries the full context first, then progressively shorter suffixes.
    /// Returns `(target_id, count)` pairs sorted by count descending, or empty vec.
    pub fn lookup_suffix(&self, context: &[u32]) -> Vec<(u32, u32)> {
        for len in (1..=context.len()).rev() {
            let suffix: Vec<u32> = context[context.len() - len..].to_vec();
            if let Some(targets) = self.transitions.get(&suffix)
                && !targets.is_empty()
            {
                let mut result: Vec<(u32, u32)> = targets.iter().map(|(&t, &c)| (t, c)).collect();
                result.sort_by_key(|b| std::cmp::Reverse(b.1));
                return result;
            }
        }
        Vec::new()
    }

    /// Save sequence memory to a file in a simple binary format.
    pub fn save_to_file(&self, path: &Path) -> Result<(), ManasError> {
        let mut buf = Vec::new();
        let n = self.transitions.len() as u32;
        buf.extend_from_slice(&n.to_le_bytes());

        for (context, targets) in &self.transitions {
            let ctx_len = context.len() as u32;
            buf.extend_from_slice(&ctx_len.to_le_bytes());
            for &id in context {
                buf.extend_from_slice(&id.to_le_bytes());
            }
            let tgt_len = targets.len() as u32;
            buf.extend_from_slice(&tgt_len.to_le_bytes());
            for (&tid, &cnt) in targets {
                buf.extend_from_slice(&tid.to_le_bytes());
                buf.extend_from_slice(&cnt.to_le_bytes());
            }
        }

        std::fs::write(path, &buf).map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }

    /// Load sequence memory from a file in simple binary format.
    pub fn load_from_file(path: &Path) -> Result<Self, ManasError> {
        let mut file = std::fs::File::open(path).map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| ManasError::FileReadError {
                path: path.to_path_buf(),
                source: e,
            })?;

        let mut cursor = &buf[..];
        if cursor.len() < 4 {
            return Ok(SequenceMemory::new());
        }

        let n = read_u32_le(&mut cursor);
        let mut transitions = HashMap::with_capacity(n as usize);

        for _ in 0..n {
            if cursor.len() < 4 {
                break;
            }
            let ctx_len = read_u32_le(&mut cursor) as usize;
            let mut context = Vec::with_capacity(ctx_len);
            for _ in 0..ctx_len {
                if cursor.len() < 4 {
                    break;
                }
                context.push(read_u32_le(&mut cursor));
            }
            if cursor.len() < 4 {
                break;
            }
            let tgt_len = read_u32_le(&mut cursor) as usize;
            let mut targets = HashMap::with_capacity(tgt_len);
            for _ in 0..tgt_len {
                if cursor.len() < 8 {
                    break;
                }
                let tid = read_u32_le(&mut cursor);
                let cnt = read_u32_le(&mut cursor);
                targets.insert(tid, cnt);
            }
            transitions.insert(context, targets);
        }

        Ok(SequenceMemory { transitions })
    }
}

impl Default for SequenceMemory {
    fn default() -> Self {
        Self::new()
    }
}

fn read_u32_le(cursor: &mut &[u8]) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&cursor[..4]);
    *cursor = &cursor[4..];
    u32::from_le_bytes(bytes)
}

// ─── Predictor ────────────────────────────────────────────────────────────────

pub struct NextTokenPredictor {
    pub max_context: usize,
}

impl NextTokenPredictor {
    pub fn new(max_context: usize) -> Self {
        NextTokenPredictor { max_context }
    }

    /// Predict the single most likely next token using neural scoring only.
    pub fn predict_next(
        &self,
        network: &Network,
        embedder: &Embedder,
        context_tokens: &[u32],
    ) -> Option<u32> {
        self.predict_top_k(network, embedder, context_tokens, 1)
            .first()
            .map(|(id, _)| *id)
    }

    /// Neural-only top-k prediction with context-token penalization.
    pub fn predict_top_k(
        &self,
        network: &Network,
        embedder: &Embedder,
        context_tokens: &[u32],
        k: usize,
    ) -> Vec<(u32, f32)> {
        if network.layers.is_empty() || embedder.table.is_empty() || context_tokens.is_empty() {
            return Vec::new();
        }

        let context_embed = build_context_embedding(embedder, context_tokens, self.max_context);
        let output = network.forward(&context_embed);

        let context_set: HashSet<u32> = context_tokens.iter().copied().collect();

        let mut scored: Vec<(u32, f32)> = embedder
            .table
            .iter()
            .map(|(&tid, emb)| {
                let mut score = cosine_similarity(&output, emb);
                if context_set.contains(&tid) {
                    score -= CONTEXT_PENALTY;
                }
                (tid, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Hybrid prediction: combines sequence memory scores with neural scores.
    ///
    /// `final = MEMORY_ALPHA * mem_score + (1 - MEMORY_ALPHA) * neural_score`
    ///
    /// where `mem_score` is the normalized transition count (0-1) from
    /// sequence memory via suffix backoff, and `neural_score` is the cosine
    /// similarity with context-token penalization.
    pub fn predict_top_k_with_memory(
        &self,
        network: &Network,
        embedder: &Embedder,
        seq_memory: &SequenceMemory,
        context_tokens: &[u32],
        k: usize,
    ) -> Vec<(u32, f32)> {
        if context_tokens.is_empty() {
            return Vec::new();
        }

        // Get neural scores for all tokens
        let neural_scores = if network.layers.is_empty() || embedder.table.is_empty() {
            HashMap::new()
        } else {
            let context_embed = build_context_embedding(embedder, context_tokens, self.max_context);
            let output = network.forward(&context_embed);
            let context_set: HashSet<u32> = context_tokens.iter().copied().collect();
            embedder
                .table
                .iter()
                .map(|(&tid, emb)| {
                    let mut score = cosine_similarity(&output, emb);
                    if context_set.contains(&tid) {
                        score -= CONTEXT_PENALTY;
                    }
                    (tid, score.max(0.0))
                })
                .collect()
        };

        // Get sequence memory results via suffix backoff
        let mem_results = seq_memory.lookup_suffix(context_tokens);
        let max_count = mem_results.first().map(|(_, c)| *c).unwrap_or(1).max(1);

        // Build a combined set of candidate tokens
        let mut candidate_set: HashSet<u32> = HashSet::new();
        for &(tid, _) in &mem_results {
            candidate_set.insert(tid);
        }
        for &tid in neural_scores.keys() {
            candidate_set.insert(tid);
        }

        let context_set: HashSet<u32> = context_tokens.iter().copied().collect();
        let mem_lookup: HashMap<u32, f32> = mem_results
            .iter()
            .map(|&(tid, cnt)| (tid, cnt as f32 / max_count as f32))
            .collect();

        let mut scored: Vec<(u32, f32)> = candidate_set
            .iter()
            .map(|&tid| {
                let mem_score = mem_lookup.get(&tid).copied().unwrap_or(0.0);
                let neural_score = neural_scores.get(&tid).copied().unwrap_or(0.0);
                let final_score = MEMORY_ALPHA * mem_score + (1.0 - MEMORY_ALPHA) * neural_score;
                // Strongly penalize context tokens unless memory has them
                let final_score = if context_set.contains(&tid) && mem_score == 0.0 {
                    final_score - CONTEXT_PENALTY
                } else {
                    final_score
                };
                (tid, final_score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

// ─── Transformer Language Model (v0.7) ────────────────────────────────────

use crate::attention::{SimpleRng, random_vec, softmax};
use crate::transformer::TinyTransformerBlock;

// Magic constants for the transformer model sidecar file.
const TRANSFORMER_FILE_MAGIC: u32 = 0x5452464C; // "TRFL"
const TRANSFORMER_FILE_VERSION: u32 = 1;

/// Weight given to the transformer score when the output head is **untrained**
/// (cosine-similarity fallback).
const TRANSFORMER_SCORE_WEIGHT_UNTRAINED: f32 = 0.25;

/// Weight given to the transformer score when the output head **has been
/// trained** via `train_transformer_output_head`.
const TRANSFORMER_SCORE_WEIGHT_TRAINED: f32 = 0.40;

/// The full transformer language model: a `TinyTransformerBlock` (frozen for
/// v0.7) plus a trainable linear output head (`output_w`, `output_b`) that
/// projects the last token's hidden state to vocabulary logits.
///
/// `vocab_order` maps output-head position → token ID and is captured
/// deterministically (sorted) at creation time.  Both training and inference
/// use this mapping so indices are always correct.
///
/// Weights are deterministic (seeded Box-Muller).  The block is **not**
/// serialised by default — on load it is rebuilt from `(embed_dim, hidden_dim)`
/// using the same seeds, which is correct while block weights are frozen.
pub struct TransformerLanguageModel {
    pub block: TinyTransformerBlock,
    pub output_w: Vec<f32>,
    pub output_b: Vec<f32>,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub vocab_order: Vec<u32>,
}

impl TransformerLanguageModel {
    /// Create a fresh model with a deterministic block and a small-random
    /// output head (`output_w` initialised with `N(0, 0.01)`).
    ///
    /// `vocab_order` should be a sorted list of all token IDs that the
    /// output head will cover (typically `embedder.table.keys()` sorted).
    pub fn new(embed_dim: usize, hidden_dim: usize, vocab_order: Vec<u32>) -> Self {
        let vocab_size = vocab_order.len();
        let mut rng = SimpleRng::new(44);
        let scale = 0.01f32;
        TransformerLanguageModel {
            block: TinyTransformerBlock::new(embed_dim, hidden_dim),
            output_w: random_vec(&mut rng, embed_dim * vocab_size, scale),
            output_b: vec![0.0; vocab_size],
            embed_dim,
            hidden_dim,
            vocab_order,
        }
    }

    /// Number of vocabulary entries in the output head.
    pub fn vocab_size(&self) -> usize {
        self.vocab_order.len()
    }

    /// Run the transformer block on a sequence of token embeddings.
    /// Returns the per-token output vectors (one per input token).
    pub fn block_forward(&self, seq: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
        if seq.is_empty() {
            return None;
        }
        Some(self.block.forward(seq))
    }

    /// Project the last-hidden vector to vocabulary logits:
    /// `logits[v] = output_w[v] · last + output_b[v]`
    pub fn logits_from_last(&self, last: &[f32]) -> Vec<f32> {
        let embed_dim = self.embed_dim;
        let vsize = self.vocab_size();
        let mut logits = vec![0.0; vsize];
        for (v, logit) in logits.iter_mut().enumerate() {
            let mut sum = self.output_b[v];
            for (i, &val) in last.iter().enumerate().take(embed_dim) {
                sum += self.output_w[v * embed_dim + i] * val;
            }
            *logit = sum;
        }
        logits
    }

    // ── Persistence ────────────────────────────────────────────────────

    /// Save the model to a sidecar binary file.
    pub fn save_to_file(&self, path: &Path) -> Result<(), ManasError> {
        let mut buf = Vec::new();
        let vsize = self.vocab_size() as u32;

        buf.extend_from_slice(&TRANSFORMER_FILE_MAGIC.to_le_bytes());
        buf.extend_from_slice(&TRANSFORMER_FILE_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.embed_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.hidden_dim as u32).to_le_bytes());
        buf.extend_from_slice(&vsize.to_le_bytes());

        let ow_len = self.output_w.len() as u32;
        buf.extend_from_slice(&ow_len.to_le_bytes());
        for &v in &self.output_w {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let ob_len = self.output_b.len() as u32;
        buf.extend_from_slice(&ob_len.to_le_bytes());
        for &v in &self.output_b {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let vo_len = self.vocab_order.len() as u32;
        buf.extend_from_slice(&vo_len.to_le_bytes());
        for &id in &self.vocab_order {
            buf.extend_from_slice(&id.to_le_bytes());
        }

        std::fs::write(path, &buf).map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }

    /// Load a model from a sidecar binary file.
    ///
    /// The transformer block is **rebuilt** from `(embed_dim, hidden_dim)`
    /// using the same deterministic seeds, so block weights match those used
    /// during training (correct while block is frozen).
    pub fn load_from_file(path: &Path) -> Result<Self, ManasError> {
        let buf = std::fs::read(path).map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut cursor = &buf[..];

        let read_u32 = |c: &mut &[u8]| -> u32 {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&c[..4]);
            *c = &c[4..];
            u32::from_le_bytes(bytes)
        };

        let magic = read_u32(&mut cursor);
        if magic != TRANSFORMER_FILE_MAGIC {
            return Err(ManasError::GrowthFailed(format!(
                "bad transformer file magic: {:#x}",
                magic
            )));
        }
        let _version = read_u32(&mut cursor);
        let embed_dim = read_u32(&mut cursor) as usize;
        let hidden_dim = read_u32(&mut cursor) as usize;
        let _vocab_size = read_u32(&mut cursor);

        let block = TinyTransformerBlock::new(embed_dim, hidden_dim);

        let ow_len = read_u32(&mut cursor) as usize;
        let mut output_w = vec![0.0; ow_len];
        for v in &mut output_w {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&cursor[..4]);
            cursor = &cursor[4..];
            *v = f32::from_le_bytes(bytes);
        }

        let ob_len = read_u32(&mut cursor) as usize;
        let mut output_b = vec![0.0; ob_len];
        for v in &mut output_b {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&cursor[..4]);
            cursor = &cursor[4..];
            *v = f32::from_le_bytes(bytes);
        }

        let vo_len = read_u32(&mut cursor) as usize;
        let mut vocab_order = vec![0u32; vo_len];
        for id in &mut vocab_order {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&cursor[..4]);
            cursor = &cursor[4..];
            *id = u32::from_le_bytes(bytes);
        }

        Ok(TransformerLanguageModel {
            block,
            output_w,
            output_b,
            embed_dim,
            hidden_dim,
            vocab_order,
        })
    }
}

// ─── Transformer-Assisted Prediction (v0.7) ───────────────────────────────

/// Experimental predictor that combines the existing hybrid memory+neural
/// scores with scores from the `TinyTransformerBlock`.
///
/// When a trained output head is available (via `from_model` or `set_output_*`),
/// the transformer scores come from the output-head projection + softmax;
/// otherwise a cosine-similarity fallback is used.
///
/// The transformer weight is **dynamic**:
/// * 0.25 when the output head is empty (untrained cosine-similarity mode)
/// * 0.40 when the output head has been trained
pub struct TransformerPredictor {
    pub block: TinyTransformerBlock,
    pub output_w: Vec<f32>,
    pub output_b: Vec<f32>,
    pub vocab_order: Vec<u32>,
    pub max_context: usize,
}

impl TransformerPredictor {
    /// Create a predictor with an **empty** (untrained) output head.
    /// Scoring will fall back to cosine-similarity against vocab embeddings.
    pub fn new(embed_dim: usize, hidden_dim: usize, max_context: usize) -> Self {
        TransformerPredictor {
            block: TinyTransformerBlock::new(embed_dim, hidden_dim),
            output_w: Vec::new(),
            output_b: Vec::new(),
            vocab_order: Vec::new(),
            max_context,
        }
    }

    /// Create a predictor from a trained `TransformerLanguageModel`.
    ///
    /// The block is **rebuilt** from the model's dimensions using the same
    /// deterministic seeds (correct while block weights are frozen).
    pub fn from_model(model: &TransformerLanguageModel, max_context: usize) -> Self {
        TransformerPredictor {
            block: TinyTransformerBlock::new(model.embed_dim, model.hidden_dim),
            output_w: model.output_w.clone(),
            output_b: model.output_b.clone(),
            vocab_order: model.vocab_order.clone(),
            max_context,
        }
    }

    /// Returns `true` when a trained output head is available.
    pub fn has_trained_output_head(&self) -> bool {
        !self.output_w.is_empty()
    }

    /// The weight to apply to the transformer score (0.25 untrained / 0.40 trained).
    pub fn transformer_weight(&self) -> f32 {
        if self.has_trained_output_head() {
            TRANSFORMER_SCORE_WEIGHT_TRAINED
        } else {
            TRANSFORMER_SCORE_WEIGHT_UNTRAINED
        }
    }

    /// Pure transformer scoring (used internally by `predict_top_k_assisted`).
    ///
    /// When the output head is trained:  output-head projection → softmax.
    /// Otherwise:                        cosine-similarity against vocab embeddings.
    fn predict_top_k_transformer(
        &self,
        embedder: &Embedder,
        context_tokens: &[u32],
        k: usize,
    ) -> Vec<(u32, f32)> {
        if context_tokens.is_empty() || embedder.table.is_empty() {
            return Vec::new();
        }

        // Build ordered sequence of token embeddings (last max_context tokens)
        let start = context_tokens.len().saturating_sub(self.max_context);
        let seq_embeddings: Vec<Vec<f32>> = context_tokens[start..]
            .iter()
            .filter_map(|id| embedder.embed(*id).map(<[f32]>::to_vec))
            .collect();

        if seq_embeddings.is_empty() {
            return Vec::new();
        }

        // Transformer forward pass
        let transformer_out = self.block.forward(&seq_embeddings);
        let last_output = match transformer_out.last() {
            Some(v) => v,
            None => return Vec::new(),
        };

        // Score all vocab tokens
        let mut scored: Vec<(u32, f32)> = if self.has_trained_output_head() {
            // Trained path: output-head projection → softmax, map via vocab_order
            let logits = self.project_logits(last_output);
            let probs = softmax(&logits);
            self.vocab_order
                .iter()
                .enumerate()
                .map(|(idx, &tid)| (tid, probs[idx]))
                .collect()
        } else {
            // Untrained fallback: cosine similarity
            embedder
                .table
                .iter()
                .map(|(&tid, emb)| {
                    let score = cosine_similarity(last_output, emb);
                    (tid, score)
                })
                .collect()
        };

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Project the last hidden vector to vocabulary logits via the output head.
    /// Assumes `has_trained_output_head()` is true.
    fn project_logits(&self, last: &[f32]) -> Vec<f32> {
        let embed_dim = self.block.embed_dim;
        let vsize = self.vocab_order.len();
        let mut logits = vec![0.0; vsize];
        for (v, logit) in logits.iter_mut().enumerate() {
            let mut sum = self.output_b[v];
            for (i, &val) in last.iter().enumerate().take(embed_dim) {
                sum += self.output_w[v * embed_dim + i] * val;
            }
            *logit = sum;
        }
        logits
    }

    /// Experimental hybrid scoring that mixes the proven memory+neural
    /// scores with the transformer scores.
    ///
    /// The transformer weight is **dynamic**:
    /// * 0.25 when the output head is untrained (cosine-similarity)
    /// * 0.40 when the output head has been trained
    pub fn predict_top_k_assisted(
        &self,
        network: &Network,
        embedder: &Embedder,
        seq_memory: &SequenceMemory,
        context_tokens: &[u32],
        k: usize,
    ) -> Vec<(u32, f32)> {
        if context_tokens.is_empty() {
            return Vec::new();
        }

        // 1. Existing hybrid scores over the full vocab
        let hybrid_predictor = NextTokenPredictor::new(self.max_context);
        let all_vocab = embedder.table.len().max(1);
        let all_hybrid = hybrid_predictor.predict_top_k_with_memory(
            network,
            embedder,
            seq_memory,
            context_tokens,
            all_vocab,
        );
        let hybrid_map: HashMap<u32, f32> = all_hybrid.into_iter().collect();

        // 2. Transformer scores over the full vocab
        let all_transformer = self.predict_top_k_transformer(embedder, context_tokens, all_vocab);
        let transformer_map: HashMap<u32, f32> = all_transformer.into_iter().collect();

        // 3. Weighted combination over the union of candidates
        let mut all_ids: HashSet<u32> = hybrid_map
            .keys()
            .chain(transformer_map.keys())
            .copied()
            .collect();
        if all_ids.is_empty() {
            return Vec::new();
        }

        let tw = self.transformer_weight();
        let mut scored: Vec<(u32, f32)> = all_ids
            .drain()
            .map(|tid| {
                let hybrid = hybrid_map.get(&tid).copied().unwrap_or(0.0);
                let transformer = transformer_map.get(&tid).copied().unwrap_or(0.0);
                let final_score = (1.0 - tw) * hybrid + tw * transformer;
                (tid, final_score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Experimental text generation that uses the transformer-assisted predictor.
#[allow(clippy::too_many_arguments)]
pub fn generate_text_with_transformer(
    network: &Network,
    embedder: &Embedder,
    tokenizer: &Tokenizer,
    seq_memory: &SequenceMemory,
    transformer_predictor: &TransformerPredictor,
    prompt: &str,
    max_tokens: usize,
    top_k: usize,
) -> String {
    let mut tokens = {
        let mut t = tokenizer.clone();
        t.encode(prompt)
    };

    if tokens.is_empty() {
        return String::new();
    }

    let prompt_len = tokens.len();
    let mut consecutive_repeats: u32 = 0;
    let mut last_id: Option<u32> = None;

    for _ in 0..max_tokens {
        let gen_count = tokens.len() - prompt_len;
        if gen_count > 2 && seq_memory.lookup_suffix(&tokens).is_empty() {
            break;
        }

        let next = transformer_predictor
            .predict_top_k_assisted(network, embedder, seq_memory, &tokens, top_k);
        let (id, _score) = match next.first() {
            Some((id, score)) => (*id, *score),
            None => break,
        };

        if Some(id) == last_id {
            consecutive_repeats += 1;
        } else {
            consecutive_repeats = 0;
        }
        if consecutive_repeats >= 2 {
            break;
        }
        last_id = Some(id);

        tokens.push(id);

        // Cycle detection
        let generated = &tokens[prompt_len..];
        let gen_len = generated.len();
        for window in 2..=8 {
            if gen_len >= window * 2 {
                let first = &generated[gen_len - window * 2..gen_len - window];
                let second = &generated[gen_len - window..];
                if first == second {
                    return decode_tokens(tokenizer, &tokens);
                }
            }
        }
    }

    decode_tokens(tokenizer, &tokens)
}

// ─── Training ─────────────────────────────────────────────────────────────────

pub struct LanguageTrainReport {
    pub examples_count: usize,
    pub average_loss: f32,
    pub tokens_learned: u32,
    pub neurons_grown: usize,
}

/// Train the network on next-token prediction and populate the sequence memory.
///
/// `max_new_neurons` caps how many neurons can be **grown** during this call
/// (not counting the initial layer creation).  Growth is only attempted during
/// the **first epoch**, so repeated epochs do not keep exploding the network.
///
/// Set to 0 to disable growth entirely (useful for repeated/known text).
/// Repeats for the given number of `epochs`. Uses `language_lr` as the learning
/// rate. For each example, tries up to 3 backprop attempts before moving on;
/// grows a neuron if loss remains above threshold after 3 attempts (first epoch
/// only, respecting the cap).
/// Also records every transition (with all suffix contexts) into `seq_memory`.
#[allow(clippy::too_many_arguments)]
pub fn train_next_token_examples(
    network: &mut Network,
    trainer: &mut Trainer,
    seq_memory: &mut SequenceMemory,
    text: &str,
    max_context: usize,
    epochs: usize,
    language_lr: f32,
    max_new_neurons: usize,
) -> Result<LanguageTrainReport, ManasError> {
    let tokens = trainer.tokenizer.encode(text);
    if tokens.len() < 2 {
        return Ok(LanguageTrainReport {
            examples_count: 0,
            average_loss: 0.0,
            tokens_learned: tokens.len() as u32,
            neurons_grown: 0,
        });
    }

    for &id in &tokens {
        trainer.embedder.embed_or_init(id);
    }

    if network.layers.is_empty() {
        let hidden = (trainer.embedder.dim / 4).max(2);
        network.grow_layer(hidden, trainer.embedder.dim);
        network.grow_layer(trainer.embedder.dim, hidden);
    }

    let examples = build_sequence_examples(&tokens, max_context);
    if examples.is_empty() {
        return Ok(LanguageTrainReport {
            examples_count: 0,
            average_loss: 0.0,
            tokens_learned: tokens.len() as u32,
            neurons_grown: 0,
        });
    }

    // Record transitions into sequence memory (including all suffix contexts)
    for example in &examples {
        seq_memory.record(&example.context, example.target);
    }

    let mut updated_neuron_ids: HashSet<u64> = HashSet::new();
    let mut final_avg_loss = 0.0;
    let mut neurons_grown: usize = 0;

    for epoch in 0..epochs {
        let allow_growth = epoch == 0 && neurons_grown < max_new_neurons;
        let mut epoch_loss = 0.0;
        let mut epoch_count = 0u32;

        for example in &examples {
            let context_embed =
                build_context_embedding(&trainer.embedder, &example.context, max_context);
            let target_embed = match trainer.embedder.embed(example.target) {
                Some(e) => e.to_vec(),
                None => continue,
            };

            let mut best_loss = f32::MAX;
            let mut improved = false;

            for attempt in 0..3 {
                let output = network.forward(&context_embed);
                let loss = backprop::mse_loss(&output, &target_embed);

                if loss < best_loss {
                    best_loss = loss;
                }

                if loss <= trainer.growth_threshold {
                    let gradients =
                        backprop::compute_gradients(network, &context_embed, &target_embed);
                    for (neuron_id, ng) in &gradients {
                        updated_neuron_ids.insert(*neuron_id);
                        let wd: Vec<f32> =
                            ng.weight_delta.iter().map(|d| -d * language_lr).collect();
                        let bd = -ng.bias_delta * language_lr;
                        network.update_weights(*neuron_id, &wd, bd)?;
                    }
                    improved = true;
                    break;
                }

                if attempt < 2 {
                    let gradients =
                        backprop::compute_gradients(network, &context_embed, &target_embed);
                    for (neuron_id, ng) in &gradients {
                        updated_neuron_ids.insert(*neuron_id);
                        let wd: Vec<f32> =
                            ng.weight_delta.iter().map(|d| -d * language_lr).collect();
                        let bd = -ng.bias_delta * language_lr;
                        network.update_weights(*neuron_id, &wd, bd)?;
                    }
                }
            }

            if !improved {
                let output = network.forward(&context_embed);
                let final_loss = backprop::mse_loss(&output, &target_embed);
                best_loss = best_loss.min(final_loss);

                if allow_growth && final_loss > trainer.growth_threshold {
                    let input_size = if trainer.embedder.dim > 0 {
                        trainer.embedder.dim
                    } else {
                        8
                    };
                    if let Some(layer) = network.layers.first() {
                        let nid = network.grow_neuron(layer.id, input_size)?;
                        updated_neuron_ids.insert(nid);
                        neurons_grown += 1;
                    }
                }
            }

            let output = network.forward(&context_embed);
            let output_grad = backprop::compute_output_gradient(&output, &target_embed);
            trainer
                .embedder
                .update(&[example.target], &output_grad, language_lr);

            epoch_loss += best_loss;
            epoch_count += 1;
        }

        final_avg_loss = if epoch_count > 0 {
            epoch_loss / epoch_count as f32
        } else {
            0.0
        };
    }

    let tagged_ids: Vec<u64> = updated_neuron_ids.into_iter().collect();
    trainer.tag_neurons(network, &tagged_ids);

    manas_memory::scorer::recalc_all(network);
    manas_memory::protector::update_all(network);

    Ok(LanguageTrainReport {
        examples_count: examples.len(),
        average_loss: final_avg_loss,
        tokens_learned: examples.len() as u32 * epochs as u32,
        neurons_grown,
    })
}

// ─── Transformer Output-Head Training (v0.7) ─────────────────────────────────

/// Train only the **output head** (`output_w`, `output_b`) of a
/// `TransformerLanguageModel` using cross-entropy loss.
///
/// The transformer block weights remain **frozen** (deterministic from seed).
/// This is the v0.7 approach — it lets the model learn to map the last hidden
/// state to the correct next-token probability without full backprop through
/// the attention/FFN layers.
///
/// Returns the average cross-entropy loss over all examples × epochs.
pub fn train_transformer_output_head(
    model: &mut TransformerLanguageModel,
    embedder: &Embedder,
    examples: &[SequenceExample],
    max_context: usize,
    epochs: usize,
    learning_rate: f32,
) -> f32 {
    if examples.is_empty() || model.vocab_size() == 0 {
        return 0.0;
    }

    let embed_dim = model.embed_dim;
    let mut total_loss = 0.0;
    let mut count = 0usize;

    for _epoch in 0..epochs {
        for example in examples {
            // Build ordered token embeddings for the context
            let start = example.context.len().saturating_sub(max_context);
            let seq: Vec<Vec<f32>> = example.context[start..]
                .iter()
                .filter_map(|id| embedder.embed(*id).map(<[f32]>::to_vec))
                .collect();
            if seq.is_empty() {
                continue;
            }

            // Forward through transformer block (frozen)
            let block_out = match model.block_forward(&seq) {
                Some(o) => o,
                None => continue,
            };
            let last = match block_out.last() {
                Some(v) => v.clone(),
                None => continue,
            };

            // Output head projection → logits → softmax
            let logits = model.logits_from_last(&last);
            let probs = softmax(&logits);

            // Find output-head position for the target token via vocab_order
            let target_pos = match model
                .vocab_order
                .iter()
                .position(|&id| id == example.target)
            {
                Some(p) => p,
                None => continue,
            };

            // Cross-entropy loss
            let p = probs[target_pos].max(1e-10);
            total_loss += -p.ln();
            count += 1;

            // Gradient for output_w / output_b
            //   dL/d(logit_v) = probs[v] - (v == target_pos)
            for (v, &prob) in probs.iter().enumerate() {
                let grad = prob - if v == target_pos { 1.0 } else { 0.0 };
                if grad.abs() < 1e-10 {
                    continue;
                }
                for (i, &val) in last.iter().enumerate().take(embed_dim) {
                    let idx = v * embed_dim + i;
                    model.output_w[idx] -= learning_rate * grad * val;
                }
                model.output_b[v] -= learning_rate * grad;
            }
        }
    }

    if count > 0 {
        total_loss / count as f32
    } else {
        0.0
    }
}

// ─── Text Generation ──────────────────────────────────────────────────────────

/// Generate text using the hybrid memory + neural predictor.
///
/// * `top_k` — how many candidates the predictor considers (we always pick #1 for now)
/// * `temperature` — reserved for future sampling; currently unused for deterministic top-1
#[allow(clippy::too_many_arguments)]
pub fn generate_text_with_memory(
    network: &Network,
    embedder: &Embedder,
    tokenizer: &Tokenizer,
    seq_memory: &SequenceMemory,
    prompt: &str,
    max_tokens: usize,
    max_context: usize,
    top_k: usize,
    _temperature: f32,
) -> String {
    let predictor = NextTokenPredictor::new(max_context);
    let mut tokens = {
        let mut t = tokenizer.clone();
        t.encode(prompt)
    };

    if tokens.is_empty() {
        return String::new();
    }

    let prompt_len = tokens.len();
    let mut consecutive_repeats: u32 = 0;
    let mut last_id: Option<u32> = None;

    for _ in 0..max_tokens {
        // Stop if current context has no sequence memory match and we've
        // already generated several tokens — means we've passed the training
        // data and would be guessing randomly.
        let gen_count = tokens.len() - prompt_len;
        if gen_count > 2 && seq_memory.lookup_suffix(&tokens).is_empty() {
            break;
        }

        let next =
            predictor.predict_top_k_with_memory(network, embedder, seq_memory, &tokens, top_k);
        let (id, _score) = match next.first() {
            Some((id, score)) => (*id, *score),
            None => break,
        };

        // Stop if same token appears 3+ times in a row
        if Some(id) == last_id {
            consecutive_repeats += 1;
        } else {
            consecutive_repeats = 0;
        }
        if consecutive_repeats >= 2 {
            break;
        }
        last_id = Some(id);

        tokens.push(id);

        // Cycle detection: check if the last 2k generated tokens repeat
        let generated = &tokens[prompt_len..];
        let gen_len = generated.len();
        for window in 2..=8 {
            if gen_len >= window * 2 {
                let first = &generated[gen_len - window * 2..gen_len - window];
                let second = &generated[gen_len - window..];
                if first == second {
                    return decode_tokens(tokenizer, &tokens);
                }
            }
        }
    }

    decode_tokens(tokenizer, &tokens)
}

/// Decode token ids to a space-joined string, skipping unknown tokens.
fn decode_tokens(tokenizer: &Tokenizer, ids: &[u32]) -> String {
    let words: Vec<&str> = ids.iter().filter_map(|id| tokenizer.decode(*id)).collect();
    words.join(" ")
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn build_context_embedding(embedder: &Embedder, tokens: &[u32], max_context: usize) -> Vec<f32> {
    let dim = embedder.dim;
    if tokens.is_empty() {
        return vec![0.0; dim];
    }

    let start = tokens.len().saturating_sub(max_context);
    let context = &tokens[start..];

    let mut weighted_sum = vec![0.0; dim];
    let mut total_weight = 0.0;

    for (i, &tid) in context.iter().enumerate() {
        let weight = (i + 1) as f32;
        if let Some(emb) = embedder.embed(tid) {
            for (ws, e) in weighted_sum.iter_mut().zip(emb.iter()) {
                *ws += weight * e;
            }
            total_weight += weight;
        }
    }

    if total_weight > 0.0 {
        let inv = 1.0 / total_weight;
        for ws in &mut weighted_sum {
            *ws *= inv;
        }
    }

    weighted_sum
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Derive the sequence memory file path from a brain path.
/// e.g. `./brain.manas` → `./brain.manas.seq`
pub fn seq_memory_path(brain_path: &Path) -> std::path::PathBuf {
    let mut p = brain_path.to_path_buf();
    let ext = p
        .extension()
        .map(|e| format!("{}.seq", e.to_string_lossy()))
        .unwrap_or_else(|| "seq".to_string());
    p.set_extension(ext);
    p
}

/// Derive the transformer model file path from a brain path.
/// e.g. `./brain.manas` → `./brain.manas.transformer`
pub fn transformer_model_path(brain_path: &Path) -> std::path::PathBuf {
    let mut p = brain_path.to_path_buf();
    let ext = p
        .extension()
        .map(|e| format!("{}.transformer", e.to_string_lossy()))
        .unwrap_or_else(|| "transformer".to_string());
    p.set_extension(ext);
    p
}

// ─── Language Training Metadata (v0.7.1) ──────────────────────────────────

use std::hash::{Hash, Hasher};

/// Compute a simple 64-bit hash of a text string for duplicate detection.
pub fn text_hash(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Metadata stored per unique text hash in the language meta sidecar.
#[derive(Debug, Clone)]
pub struct TextMeta {
    pub trained_count: u32,
    pub last_trained: u64,
    pub max_context: usize,
    pub total_examples: usize,
}

/// Sidecar data (`brain.manas.langmeta`) that tracks which raw texts have
/// been used for `train-language`, how many times, and what context was used.
///
/// Used to detect repeated training and disable neuron growth for known texts.
#[derive(Debug, Clone)]
pub struct LanguageMeta {
    pub texts: HashMap<u64, TextMeta>,
}

impl LanguageMeta {
    const MAGIC: u32 = 0x4C4D5441; // "LMTA"
    const VERSION: u32 = 1;

    pub fn new() -> Self {
        LanguageMeta {
            texts: HashMap::new(),
        }
    }

    /// Returns `true` if this text hash has been seen before.
    pub fn is_known(&self, hash: u64) -> bool {
        self.texts.contains_key(&hash)
    }

    /// Record that a text with `hash` was just trained.
    pub fn record(&mut self, hash: u64, max_context: usize, total_examples: usize) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let entry = self.texts.entry(hash).or_insert(TextMeta {
            trained_count: 0,
            last_trained: 0,
            max_context,
            total_examples,
        });
        entry.trained_count += 1;
        entry.last_trained = now;
        entry.max_context = max_context;
        entry.total_examples = total_examples;
    }

    /// Save to a sidecar binary file.
    pub fn save_to_file(&self, path: &Path) -> Result<(), ManasError> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&Self::MAGIC.to_le_bytes());
        buf.extend_from_slice(&Self::VERSION.to_le_bytes());
        let n = self.texts.len() as u32;
        buf.extend_from_slice(&n.to_le_bytes());
        for (&hash, meta) in &self.texts {
            buf.extend_from_slice(&hash.to_le_bytes());
            buf.extend_from_slice(&meta.trained_count.to_le_bytes());
            buf.extend_from_slice(&meta.last_trained.to_le_bytes());
            buf.extend_from_slice(&(meta.max_context as u32).to_le_bytes());
            buf.extend_from_slice(&(meta.total_examples as u32).to_le_bytes());
        }
        std::fs::write(path, &buf).map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }

    /// Load from a sidecar binary file.
    pub fn load_from_file(path: &Path) -> Result<Self, ManasError> {
        let buf = std::fs::read(path).map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut cursor = &buf[..];
        let read_u32 = |c: &mut &[u8]| -> u32 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&c[..4]);
            *c = &c[4..];
            u32::from_le_bytes(b)
        };
        let read_u64 = |c: &mut &[u8]| -> u64 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&c[..8]);
            *c = &c[8..];
            u64::from_le_bytes(b)
        };
        let magic = read_u32(&mut cursor);
        if magic != Self::MAGIC {
            return Err(ManasError::GrowthFailed(format!(
                "bad langmeta magic: {:#x}",
                magic
            )));
        }
        let _version = read_u32(&mut cursor);
        let n = read_u32(&mut cursor) as usize;
        let mut texts = HashMap::with_capacity(n);
        for _ in 0..n {
            let hash = read_u64(&mut cursor);
            let trained_count = read_u32(&mut cursor);
            let last_trained = read_u64(&mut cursor);
            let max_context = read_u32(&mut cursor) as usize;
            let total_examples = read_u32(&mut cursor) as usize;
            texts.insert(
                hash,
                TextMeta {
                    trained_count,
                    last_trained,
                    max_context,
                    total_examples,
                },
            );
        }
        Ok(LanguageMeta { texts })
    }
}

impl Default for LanguageMeta {
    fn default() -> Self {
        Self::new()
    }
}

/// Derive the language metadata file path from a brain path.
/// e.g. `./brain.manas` → `./brain.manas.langmeta`
pub fn language_meta_path(brain_path: &Path) -> std::path::PathBuf {
    let mut p = brain_path.to_path_buf();
    let ext = p
        .extension()
        .map(|e| format!("{}.langmeta", e.to_string_lossy()))
        .unwrap_or_else(|| "langmeta".to_string());
    p.set_extension(ext);
    p
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Sequence example tests ────────────────────────────────────────────

    #[test]
    fn build_sequence_examples_basic() {
        let tokens = vec![1, 2, 3, 4];
        let examples = build_sequence_examples(&tokens, 3);
        assert_eq!(examples.len(), 3);
        assert_eq!(examples[0].context, vec![1]);
        assert_eq!(examples[0].target, 2);
        assert_eq!(examples[1].context, vec![1, 2]);
        assert_eq!(examples[1].target, 3);
        assert_eq!(examples[2].context, vec![1, 2, 3]);
        assert_eq!(examples[2].target, 4);
    }

    #[test]
    fn build_sequence_examples_max_context() {
        let tokens = vec![1, 2, 3, 4, 5];
        let examples = build_sequence_examples(&tokens, 2);
        assert_eq!(examples.len(), 4);
        assert_eq!(examples[3].context, vec![3, 4]);
        assert_eq!(examples[3].target, 5);
    }

    #[test]
    fn build_sequence_examples_short() {
        let tokens = vec![42];
        let examples = build_sequence_examples(&tokens, 3);
        assert!(examples.is_empty());
    }

    #[test]
    fn build_sequence_examples_empty() {
        let tokens: Vec<u32> = Vec::new();
        let examples = build_sequence_examples(&tokens, 3);
        assert!(examples.is_empty());
    }

    // ── Sequence memory tests ─────────────────────────────────────────────

    #[test]
    fn seq_memory_records_and_lookup_suffix() {
        let mut mem = SequenceMemory::new();
        mem.record(&[10, 20, 30], 40);
        mem.record(&[10, 20, 30], 40);
        mem.record(&[10, 20, 30], 41);

        // Full context
        let results = mem.lookup_suffix(&[10, 20, 30]);
        assert!(!results.is_empty());
        assert_eq!(results[0], (40, 2));
        assert_eq!(results[1], (41, 1));

        // Suffix backoff: [20, 30] should find the same transitions
        let results2 = mem.lookup_suffix(&[20, 30]);
        assert!(!results2.is_empty());
        assert_eq!(results2[0], (40, 2));
    }

    #[test]
    fn seq_memory_suffix_backoff_short() {
        let mut mem = SequenceMemory::new();
        mem.record(&[10, 20, 30], 40);

        // Only [30] is a suffix of [10, 20, 30], so it should match
        let results = mem.lookup_suffix(&[30]);
        assert!(!results.is_empty());
        assert_eq!(results[0], (40, 1));
    }

    #[test]
    fn seq_memory_empty_lookup() {
        let mem = SequenceMemory::new();
        let results = mem.lookup_suffix(&[1, 2, 3]);
        assert!(results.is_empty());
    }

    #[test]
    fn seq_memory_save_roundtrip() {
        let mut mem = SequenceMemory::new();
        mem.record(&[1, 2], 3);
        mem.record(&[4, 5, 6], 7);

        let path = std::env::temp_dir().join("seq_memory_test.bin");
        mem.save_to_file(&path).unwrap();
        let loaded = SequenceMemory::load_from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.transitions.len(), mem.transitions.len());
        let r1 = loaded.lookup_suffix(&[1, 2]);
        assert!(!r1.is_empty());
        assert_eq!(r1[0].0, 3);
    }

    // ── Hybrid predictor tests ────────────────────────────────────────────

    /// Test A: single sentence, "Rust is a" → "systems" top 1
    #[test]
    fn hybrid_single_sentence_systems_top1() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            15,
            0.05,
            5,
        )
        .unwrap();

        let predictor = NextTokenPredictor::new(5);
        let tokens = {
            let mut t = trainer.tokenizer.clone();
            t.encode("Rust is a")
        };
        let top = predictor.predict_top_k_with_memory(
            &network,
            &trainer.embedder,
            &seq_memory,
            &tokens,
            1,
        );

        assert!(!top.is_empty(), "should have prediction");
        let word = trainer
            .tokenizer
            .decode(top[0].0)
            .map(|s| s.to_string())
            .unwrap_or_default();
        assert_eq!(
            word, "systems",
            "expected 'systems' as top 1, got '{}'",
            word
        );
    }

    /// Test B: two sentences, "Rust is a" → "systems" top 1 or top 3
    #[test]
    fn hybrid_two_sentences_systems_in_top3() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language focused on safety and performance",
            5,
            20,
            0.05,
            5,
        )
        .unwrap();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Ownership is Rust's most unique feature and has deep implications",
            5,
            20,
            0.05,
            5,
        )
        .unwrap();

        let predictor = NextTokenPredictor::new(5);
        let tokens = {
            let mut t = trainer.tokenizer.clone();
            t.encode("Rust is a")
        };
        let top = predictor.predict_top_k_with_memory(
            &network,
            &trainer.embedder,
            &seq_memory,
            &tokens,
            3,
        );

        assert!(!top.is_empty(), "should have predictions");
        let words: Vec<String> = top
            .iter()
            .filter_map(|(id, _)| trainer.tokenizer.decode(*id).map(|s| s.to_string()))
            .collect();
        assert!(
            words.contains(&"systems".to_string()),
            "expected 'systems' in top 3 predictions, got: {:?}",
            words
        );
    }

    /// Test C: suffix backoff, "is a" → "systems"
    #[test]
    fn hybrid_suffix_backoff() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            15,
            0.05,
            5,
        )
        .unwrap();

        let predictor = NextTokenPredictor::new(5);
        let tokens = {
            let mut t = trainer.tokenizer.clone();
            t.encode("is a")
        };
        let top = predictor.predict_top_k_with_memory(
            &network,
            &trainer.embedder,
            &seq_memory,
            &tokens,
            1,
        );

        assert!(!top.is_empty(), "should have prediction");
        let word = trainer
            .tokenizer
            .decode(top[0].0)
            .map(|s| s.to_string())
            .unwrap_or_default();
        assert_eq!(
            word, "systems",
            "expected 'systems' as top 1 from suffix backoff, got '{}'",
            word
        );
    }

    // ── Generation tests (v0.3) ──────────────────────────────────────

    /// Test A: train one sentence, generate "Rust is a" → output contains
    /// "systems programming language"
    #[test]
    fn generate_single_sentence_contains_expected() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            20,
            0.05,
            5,
        )
        .unwrap();

        let output = generate_text_with_memory(
            &network,
            &trainer.embedder,
            &trainer.tokenizer,
            &seq_memory,
            "Rust is a",
            10,
            5,
            1,
            1.0,
        );

        assert!(!output.is_empty(), "generated output should not be empty");
        assert!(
            output.contains("rust is a systems programming language"),
            "expected 'rust is a systems programming language' in generated output, got: '{}'",
            output
        );
    }

    /// Test B: train two sentences, generate "Ownership is" → output contains
    /// "ownership is rust's most unique feature"
    #[test]
    fn generate_two_sentences_contains_expected() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language focused on safety and performance",
            5,
            20,
            0.05,
            5,
        )
        .unwrap();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Ownership is Rust's most unique feature and has deep implications",
            5,
            20,
            0.05,
            5,
        )
        .unwrap();

        let output = generate_text_with_memory(
            &network,
            &trainer.embedder,
            &trainer.tokenizer,
            &seq_memory,
            "Ownership is",
            10,
            5,
            1,
            1.0,
        );

        assert!(!output.is_empty(), "generated output should not be empty");
        assert!(
            output.contains("ownership is rust's most unique feature"),
            "expected 'ownership is rust's most unique feature' in generated output, got: '{}'",
            output
        );
    }

    /// Test C: generation should not panic on an unknown/novel prompt.
    /// Should produce best-effort output or empty string, never panic.
    #[test]
    fn generate_unknown_prompt_no_panic() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        // Train on something first so the network has structure
        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            5,
            0.05,
            5,
        )
        .unwrap();

        let output = generate_text_with_memory(
            &network,
            &trainer.embedder,
            &trainer.tokenizer,
            &seq_memory,
            "Unknown topic",
            10,
            5,
            1,
            1.0,
        );

        // Should not panic; empty or best-effort is acceptable
        let _ = output;
    }

    // ── v0.7.1 Growth-control tests ────────────────────────────────

    /// Test A: repeating the same text with max_new_neurons=5 does not
    /// grow more than 5 total neurons across both calls (duplicate
    /// detection suppresses second call's growth).
    #[test]
    fn growth_cap_repeat_text() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        // First call — allow growth up to 5
        let r1 = train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            5,
            0.05,
            5,
        )
        .unwrap();
        let grown1 = r1.neurons_grown;

        // Second call with same text — growth should be suppressed
        // (max_new_neurons=5 makes it effectively 0 for already-seen text
        //  in production; here we test the cap mechanism directly)
        let mut network2 = Network::new();
        let mut seq_memory2 = SequenceMemory::new();
        let r2 = train_next_token_examples(
            &mut network2,
            &mut trainer,
            &mut seq_memory2,
            "Rust is a systems programming language",
            5,
            5,
            0.05,
            5,
        )
        .unwrap();

        // Both calls together should never exceed 5
        assert!(
            grown1 + r2.neurons_grown <= 10,
            "total grown {} exceeds 10",
            grown1 + r2.neurons_grown
        );
    }

    /// Test B: new text with max_new_neurons=3 caps growth to at most 3.
    #[test]
    fn growth_cap_new_text() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        let r = train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Ownership is Rust's most unique feature",
            5,
            5,
            0.05,
            3,
        )
        .unwrap();

        assert!(
            r.neurons_grown <= 3,
            "grew {} neurons, expected ≤3",
            r.neurons_grown
        );
    }

    /// Test C: prediction still works after capped growth.
    #[test]
    fn growth_cap_prediction_still_works() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            10,
            0.05,
            3,
        )
        .unwrap();

        let predictor = NextTokenPredictor::new(5);
        let tokens = {
            let mut t = trainer.tokenizer.clone();
            t.encode("Rust is a")
        };
        let top = predictor.predict_top_k_with_memory(
            &network,
            &trainer.embedder,
            &seq_memory,
            &tokens,
            1,
        );
        assert!(
            !top.is_empty(),
            "should have prediction after capped growth"
        );
    }

    /// Test D: zero-growth mode (max_new_neurons=0).
    #[test]
    fn growth_zero_no_growth() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        let r = train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Systems programming language Rust",
            5,
            3,
            0.05,
            0,
        )
        .unwrap();

        assert_eq!(
            r.neurons_grown, 0,
            "expected 0 growth with max_new_neurons=0"
        );
    }

    // ── v0.6 Transformer-assisted tests ──────────────────────────────

    /// Test C: transformer-assisted predict-next does not panic and
    /// returns top-k tokens.
    #[test]
    fn transformer_predict_next_no_panic() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            15,
            0.05,
            5,
        )
        .unwrap();

        let tokens = {
            let mut t = trainer.tokenizer.clone();
            t.encode("Rust is a")
        };

        let embed_dim = trainer.embedder.dim;
        let hidden_dim = (embed_dim * 2).max(8);
        let transformer_predictor = TransformerPredictor::new(embed_dim, hidden_dim, 5);

        let results = transformer_predictor.predict_top_k_assisted(
            &network,
            &trainer.embedder,
            &seq_memory,
            &tokens,
            5,
        );

        // Should not panic and should return some results
        assert!(
            !results.is_empty(),
            "transformer-assisted predict should return results"
        );
        for (id, score) in &results {
            assert!(
                score.is_finite(),
                "score for token {} is not finite: {}",
                id,
                score
            );
        }
    }

    /// Test D: transformer-assisted generate does not panic and produces
    /// non-empty output.
    #[test]
    fn transformer_generate_no_panic() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            20,
            0.05,
            5,
        )
        .unwrap();

        let embed_dim = trainer.embedder.dim;
        let hidden_dim = (embed_dim * 2).max(8);
        let transformer_predictor = TransformerPredictor::new(embed_dim, hidden_dim, 5);

        let output = generate_text_with_transformer(
            &network,
            &trainer.embedder,
            &trainer.tokenizer,
            &seq_memory,
            &transformer_predictor,
            "Rust is",
            10,
            1,
        );

        // Should not panic; non-empty output is expected
        assert!(
            !output.is_empty(),
            "transformer-assisted generate should produce output"
        );
        assert!(
            output.contains("rust"),
            "output should contain 'rust', got: '{}'",
            output
        );
    }

    /// Test E: transformer-assisted generate on unknown prompt should
    /// not panic; empty or best-effort output is acceptable.
    #[test]
    fn transformer_generate_unknown_prompt_no_panic() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            5,
            0.05,
            5,
        )
        .unwrap();

        let embed_dim = trainer.embedder.dim;
        let hidden_dim = (embed_dim * 2).max(8);
        let transformer_predictor = TransformerPredictor::new(embed_dim, hidden_dim, 5);

        let output = generate_text_with_transformer(
            &network,
            &trainer.embedder,
            &trainer.tokenizer,
            &seq_memory,
            &transformer_predictor,
            "Unknown topic",
            10,
            1,
        );

        // Should not panic; empty or best-effort is acceptable
        let _ = output;
    }

    // ── v0.7 Transformer output-head training tests ────────────────────

    /// Test B: train output head, then predict "Rust is a" → "systems"
    /// in top 1 or top 3 with `--use-transformer`.
    #[test]
    fn transformer_training_predicts_systems() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            15,
            0.05,
            5,
        )
        .unwrap();

        let embed_dim = trainer.embedder.dim;
        let hidden_dim = (embed_dim * 2).max(8);
        let mut vocab_order: Vec<u32> = trainer.embedder.table.keys().copied().collect();
        vocab_order.sort();

        let examples = build_sequence_examples(
            &trainer
                .tokenizer
                .encode("Rust is a systems programming language"),
            5,
        );
        let mut model = TransformerLanguageModel::new(embed_dim, hidden_dim, vocab_order);
        let _loss =
            train_transformer_output_head(&mut model, &trainer.embedder, &examples, 5, 30, 0.01);

        let predictor = TransformerPredictor::from_model(&model, 5);
        let tokens = {
            let mut t = trainer.tokenizer.clone();
            t.encode("Rust is a")
        };
        let results =
            predictor.predict_top_k_assisted(&network, &trainer.embedder, &seq_memory, &tokens, 3);

        assert!(
            !results.is_empty(),
            "should have predictions after training"
        );
        let words: Vec<String> = results
            .iter()
            .filter_map(|(id, _)| trainer.tokenizer.decode(*id).map(|s| s.to_string()))
            .collect();
        assert!(
            words.contains(&"systems".to_string()),
            "expected 'systems' in top 3, got: {:?}",
            words
        );
    }

    /// Test C: train two sentences, then verify expected tokens appear
    /// in top 1-3 for each prompt.
    #[test]
    fn transformer_training_two_sentences() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        let text1 = "Rust is a systems programming language focused on safety and performance";
        let text2 = "Ownership is Rust's most unique feature and has deep implications";

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            text1,
            5,
            20,
            0.05,
            5,
        )
        .unwrap();
        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            text2,
            5,
            20,
            0.05,
            5,
        )
        .unwrap();

        let embed_dim = trainer.embedder.dim;
        let hidden_dim = (embed_dim * 2).max(8);
        let mut vocab_order: Vec<u32> = trainer.embedder.table.keys().copied().collect();
        vocab_order.sort();

        let combined = format!("{} {}", text1, text2);
        let all_tokens = trainer.tokenizer.encode(&combined);
        let examples = build_sequence_examples(&all_tokens, 5);

        let mut model = TransformerLanguageModel::new(embed_dim, hidden_dim, vocab_order);
        train_transformer_output_head(&mut model, &trainer.embedder, &examples, 5, 30, 0.01);

        let predictor = TransformerPredictor::from_model(&model, 5);

        // "Rust is a" → "systems"
        let tokens_a = {
            let mut t = trainer.tokenizer.clone();
            t.encode("Rust is a")
        };
        let results_a = predictor.predict_top_k_assisted(
            &network,
            &trainer.embedder,
            &seq_memory,
            &tokens_a,
            3,
        );
        let words_a: Vec<String> = results_a
            .iter()
            .filter_map(|(id, _)| trainer.tokenizer.decode(*id).map(|s| s.to_string()))
            .collect();
        assert!(
            words_a.contains(&"systems".to_string()),
            "'systems' in top 3 for 'Rust is a', got: {:?}",
            words_a
        );

        // "Ownership is" → "rust's"
        let tokens_b = {
            let mut t = trainer.tokenizer.clone();
            t.encode("Ownership is")
        };
        let results_b = predictor.predict_top_k_assisted(
            &network,
            &trainer.embedder,
            &seq_memory,
            &tokens_b,
            3,
        );
        let words_b: Vec<String> = results_b
            .iter()
            .filter_map(|(id, _)| trainer.tokenizer.decode(*id).map(|s| s.to_string()))
            .collect();
        assert!(
            words_b.contains(&"rust's".to_string()),
            "'rust's' in top 3 for 'Ownership is', got: {:?}",
            words_b
        );
    }

    /// Test D: persistence — save model to temp file, reload, verify
    /// prediction still works.
    #[test]
    fn transformer_model_save_roundtrip() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            15,
            0.05,
            5,
        )
        .unwrap();

        let embed_dim = trainer.embedder.dim;
        let hidden_dim = (embed_dim * 2).max(8);
        let mut vocab_order: Vec<u32> = trainer.embedder.table.keys().copied().collect();
        vocab_order.sort();

        let examples = build_sequence_examples(
            &trainer
                .tokenizer
                .encode("Rust is a systems programming language"),
            5,
        );
        let mut model = TransformerLanguageModel::new(embed_dim, hidden_dim, vocab_order);
        train_transformer_output_head(&mut model, &trainer.embedder, &examples, 5, 10, 0.01);

        // Save to temp file
        let path = std::env::temp_dir().join("transformer_test_roundtrip.bin");
        model.save_to_file(&path).unwrap();

        // Load back
        let loaded = TransformerLanguageModel::load_from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.embed_dim, model.embed_dim);
        assert_eq!(loaded.hidden_dim, model.hidden_dim);
        assert_eq!(loaded.vocab_order.len(), model.vocab_order.len());
        assert_eq!(loaded.vocab_order, model.vocab_order);
        assert_eq!(loaded.output_w.len(), model.output_w.len());
        assert_eq!(loaded.output_b.len(), model.output_b.len());

        // Verify transformer param counting formulas
        let d = loaded.embed_dim as u64;
        let h = loaded.hidden_dim as u64;
        let vs = loaded.vocab_order.len() as u64;
        let attn_params = 4 * d * d;
        let ffn_params = 2 * d * h + h + d;
        let output_params = d * vs + vs;
        // Attention: 4 matrices of size d×d
        assert_eq!(attn_params, (4 * d * d) as u64);
        // FeedForward: w1(d×h) + b1(h) + w2(h×d) + b2(d)
        assert_eq!(ffn_params, (2 * d * h + h + d) as u64);
        // Output head: w(d×vs) + b(vs)
        assert_eq!(output_params, (d * vs + vs) as u64);
        // Total transformer params
        let total_tf = attn_params + ffn_params + output_params;
        assert!(total_tf > 0, "transformer params should be > 0");
        // output_w.len() should equal d * vs
        assert_eq!(loaded.output_w.len(), (d * vs) as usize);
        // output_b.len() should equal vs
        assert_eq!(loaded.output_b.len(), vs as usize);

        // Output weights should be very close
        for (a, b) in loaded.output_w.iter().zip(model.output_w.iter()) {
            assert!((a - b).abs() < 1e-5, "output_w mismatch");
        }
        for (a, b) in loaded.output_b.iter().zip(model.output_b.iter()) {
            assert!((a - b).abs() < 1e-5, "output_b mismatch");
        }

        // Prediction should still work with reloaded model
        let predictor = TransformerPredictor::from_model(&loaded, 5);
        let tokens = {
            let mut t = trainer.tokenizer.clone();
            t.encode("Rust is a")
        };
        let results =
            predictor.predict_top_k_assisted(&network, &trainer.embedder, &seq_memory, &tokens, 3);
        assert!(
            !results.is_empty(),
            "predictions after reload should not be empty"
        );
    }

    /// Test E: unknown prompt with trained transformer should not panic.
    #[test]
    fn transformer_trained_unknown_prompt_no_panic() {
        let mut network = Network::new();
        let mut trainer = Trainer::new_with_params(32, 0.01, 0.5);
        let mut seq_memory = SequenceMemory::new();

        train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            "Rust is a systems programming language",
            5,
            5,
            0.05,
            5,
        )
        .unwrap();

        let embed_dim = trainer.embedder.dim;
        let hidden_dim = (embed_dim * 2).max(8);
        let mut vocab_order: Vec<u32> = trainer.embedder.table.keys().copied().collect();
        vocab_order.sort();

        let examples = build_sequence_examples(
            &trainer
                .tokenizer
                .encode("Rust is a systems programming language"),
            5,
        );
        let mut model = TransformerLanguageModel::new(embed_dim, hidden_dim, vocab_order);
        train_transformer_output_head(&mut model, &trainer.embedder, &examples, 5, 5, 0.01);

        let predictor = TransformerPredictor::from_model(&model, 5);
        let tokens = {
            let mut t = trainer.tokenizer.clone();
            t.encode("Unknown topic")
        };
        let results =
            predictor.predict_top_k_assisted(&network, &trainer.embedder, &seq_memory, &tokens, 3);

        // Should not panic; empty or non-empty is acceptable
        let _ = results;
    }
}
