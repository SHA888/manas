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
                result.sort_by(|a, b| b.1.cmp(&a.1));
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

// ─── Training ─────────────────────────────────────────────────────────────────

pub struct LanguageTrainReport {
    pub examples_count: usize,
    pub average_loss: f32,
    pub tokens_learned: u32,
}

/// Train the network on next-token prediction and populate the sequence memory.
///
/// Repeats for the given number of `epochs`. Uses `language_lr` as the learning
/// rate. For each example, tries up to 3 backprop attempts before moving on;
/// grows a neuron if loss remains above threshold after 3 attempts.
/// Also records every transition (with all suffix contexts) into `seq_memory`.
pub fn train_next_token_examples(
    network: &mut Network,
    trainer: &mut Trainer,
    seq_memory: &mut SequenceMemory,
    text: &str,
    max_context: usize,
    epochs: usize,
    language_lr: f32,
) -> Result<LanguageTrainReport, ManasError> {
    let tokens = trainer.tokenizer.encode(text);
    if tokens.len() < 2 {
        return Ok(LanguageTrainReport {
            examples_count: 0,
            average_loss: 0.0,
            tokens_learned: tokens.len() as u32,
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
        });
    }

    // Record transitions into sequence memory (including all suffix contexts)
    for example in &examples {
        seq_memory.record(&example.context, example.target);
    }

    let mut updated_neuron_ids: HashSet<u64> = HashSet::new();
    let mut final_avg_loss = 0.0;

    for _epoch in 0..epochs {
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

                if final_loss > trainer.growth_threshold {
                    let input_size = if trainer.embedder.dim > 0 {
                        trainer.embedder.dim
                    } else {
                        8
                    };
                    if let Some(layer) = network.layers.first() {
                        let nid = network.grow_neuron(layer.id, input_size)?;
                        updated_neuron_ids.insert(nid);
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
    })
}

// ─── Text Generation ──────────────────────────────────────────────────────────

/// Generate text using the hybrid memory + neural predictor.
pub fn generate_text_with_memory(
    network: &Network,
    embedder: &Embedder,
    tokenizer: &Tokenizer,
    seq_memory: &SequenceMemory,
    prompt: &str,
    max_tokens: usize,
    max_context: usize,
) -> String {
    let predictor = NextTokenPredictor::new(max_context);
    let prompt_len = {
        let mut t = tokenizer.clone();
        t.encode(prompt).len()
    };
    let mut tokens = {
        let mut t = tokenizer.clone();
        t.encode(prompt)
    };

    if tokens.is_empty() {
        return String::new();
    }

    for _ in 0..max_tokens {
        let next = predictor.predict_top_k_with_memory(network, embedder, seq_memory, &tokens, 1);
        match next.first() {
            Some((id, _)) => tokens.push(*id),
            None => break,
        }
    }

    let generated: Vec<&str> = tokens[prompt_len..]
        .iter()
        .filter_map(|id| tokenizer.decode(*id))
        .collect();

    generated.join(" ")
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
}
