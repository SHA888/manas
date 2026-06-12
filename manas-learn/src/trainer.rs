use crate::backprop;
use crate::embedder::Embedder;
use crate::tokenizer::Tokenizer;
use manas_core::{ManasError, Network, Source};
use std::collections::HashMap;

pub const DEFAULT_LEARNING_RATE: f32 = 0.01;
pub const DEFAULT_GROWTH_THRESHOLD: f32 = 0.35;
pub const DEFAULT_MAX_UPDATE_ATTEMPTS: u32 = 3;
pub const DEFAULT_EMBED_DIM: usize = 64;

// ─── LearnReport ──────────────────────────────────────────────────────────────

pub struct LearnReport {
    pub loss: f32,
    pub tokens_learned: u32,
    pub growth_occurred: bool,
    pub neurons_updated: usize,
    pub freshness_category: u8,
}

// ─── TrainerSnapshot ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrainerSnapshot {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub embed_table: HashMap<u32, Vec<f32>>,
    pub embed_dim: usize,
}

// ─── Trainer ──────────────────────────────────────────────────────────────────

pub struct Trainer {
    pub tokenizer: Tokenizer,
    pub embedder: Embedder,
    pub learning_rate: f32,
    pub growth_threshold: f32,
    pub max_update_attempts: u32,

    /// Freshness category applied to neurons updated in the current learn step.
    /// Set this before calling `learn()` when you know the content type.
    pub freshness_category: u8,

    /// Source tag applied to neurons updated in the current learn step.
    ///
    /// **Fix 2** — set this before every `learn()` call so neurons carry
    /// their origin:
    /// ```ignore
    /// trainer.source = Source::RawText;
    /// trainer.learn(&mut network, text)?;
    ///
    /// trainer.source = Source::LocalFile { path: path.to_string() };
    /// trainer.learn(&mut network, chunk)?;
    ///
    /// trainer.source = Source::Internet { url: url.to_string() };
    /// trainer.learn(&mut network, scraped)?;
    /// ```
    pub source: Source,
}

// ─── Freshness detection ──────────────────────────────────────────────────────

pub fn detect_freshness_category(text: &str) -> u8 {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    let timeless = [
        "always",
        "formula",
        "law",
        "proof",
        "theorem",
        "definition",
        "never",
        "forever",
        "constant",
    ];
    let realtime = [
        "news", "today", "breaking", "current", "live", "latest", "update",
    ];
    let fast = ["released", "version", "release", "announced", "launched"];
    let slow = [
        "since", "history", "was", "were", "had", "been", "old", "past", "origin",
    ];

    for w in &words {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        if timeless.contains(&w) {
            return 0;
        }
    }
    for w in &words {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        if realtime.contains(&w) {
            return 3;
        }
    }
    for w in &words {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        if fast.contains(&w) {
            return 2;
        }
    }
    for w in &words {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        if slow.contains(&w) {
            return 1;
        }
    }
    1
}

// ─── impl Trainer ─────────────────────────────────────────────────────────────

impl Trainer {
    pub fn new() -> Self {
        Trainer {
            tokenizer: Tokenizer::new(),
            embedder: Embedder::new(DEFAULT_EMBED_DIM),
            learning_rate: DEFAULT_LEARNING_RATE,
            growth_threshold: DEFAULT_GROWTH_THRESHOLD,
            max_update_attempts: DEFAULT_MAX_UPDATE_ATTEMPTS,
            freshness_category: 1,
            source: Source::Unknown,
        }
    }

    pub fn new_with_params(embed_dim: usize, learning_rate: f32, growth_threshold: f32) -> Self {
        Trainer {
            tokenizer: Tokenizer::new(),
            embedder: Embedder::new(embed_dim),
            learning_rate,
            growth_threshold,
            max_update_attempts: DEFAULT_MAX_UPDATE_ATTEMPTS,
            freshness_category: 1,
            source: Source::Unknown,
        }
    }

    /// Export current vocab + embeddings for persistence.
    pub fn snapshot(&self) -> TrainerSnapshot {
        TrainerSnapshot {
            vocab: self.tokenizer.vocab.clone(),
            id_to_token: self.tokenizer.id_to_token.clone(),
            embed_table: self.embedder.table.clone(),
            embed_dim: self.embedder.dim,
        }
    }

    /// Restore vocab + embeddings from a saved snapshot.
    pub fn restore(&mut self, snapshot: &TrainerSnapshot) {
        self.tokenizer.vocab = snapshot.vocab.clone();
        self.tokenizer.id_to_token = snapshot.id_to_token.clone();
        self.tokenizer.vocab_size = snapshot.vocab.len() as u32;
        self.embedder.table = snapshot.embed_table.clone();
        self.embedder.dim = snapshot.embed_dim;
    }

    /// Ensure at least one neuron carries the current file/URL source.
    ///
    /// Grows one neuron in layer 0 when:
    /// - `self.source` is `LocalFile` or `Internet`
    /// - No existing neuron has this exact source
    ///
    /// Returns `true` if a new neuron was grown.
    pub fn ensure_source_neuron(&mut self, network: &mut Network) -> Result<bool, ManasError> {
        let is_new_source = match &self.source {
            Source::LocalFile { path } => !network
                .layers
                .iter()
                .flat_map(|l| &l.neurons)
                .any(|n| matches!(&n.source, Source::LocalFile { path: p } if p == path)),
            Source::Internet { url } => !network
                .layers
                .iter()
                .flat_map(|l| &l.neurons)
                .any(|n| matches!(&n.source, Source::Internet { url: u } if u == url)),
            _ => return Ok(false),
        };

        if !is_new_source {
            return Ok(false);
        }

        let nid = network.grow_neuron(0, self.embedder.dim)?;

        for layer in &mut network.layers {
            for neuron in &mut layer.neurons {
                if neuron.id == nid {
                    neuron.source = self.source.clone();
                    neuron.freshness_category = self.freshness_category;
                }
            }
        }

        manas_memory::scorer::recalc_all(network);
        manas_memory::protector::update_all(network);

        Ok(true)
    }

    // ── Core learning loop ────────────────────────────────────────────────────

    /// Learn from a text string.
    ///
    /// Before calling, set:
    /// - `self.freshness_category` — detected or forced category
    /// - `self.source`            — where this text came from
    ///
    /// After each call this function:
    /// 1. Applies backprop weight updates (or grows a neuron if loss is high)
    /// 2. Tags updated neurons with `self.source` and `self.freshness_category`
    /// 3. **Fix 1** — recalculates importance scores for every neuron
    /// 4. **Fix 2** — updates protection levels based on new importance scores
    pub fn learn(&mut self, network: &mut Network, text: &str) -> Result<LearnReport, ManasError> {
        // ── Tokenize ──────────────────────────────────────────────────────────
        let tokens = self.tokenizer.encode(text);
        if tokens.is_empty() {
            return Ok(LearnReport {
                loss: 0.0,
                tokens_learned: 0,
                growth_occurred: false,
                neurons_updated: 0,
                freshness_category: self.freshness_category,
            });
        }

        for &id in &tokens {
            self.embedder.embed_or_init(id);
        }

        let input_vec = self.embedder.average_embed(&tokens);

        // ── Auto-create layers on first learn ─────────────────────────────────
        if network.layers.is_empty() {
            let hidden = (self.embedder.dim / 4).max(2);
            network.grow_layer(hidden, self.embedder.dim);
            network.grow_layer(self.embedder.dim, hidden);
        }

        // ── Update loop ───────────────────────────────────────────────────────
        let mut growth_occurred = false;
        let mut neurons_updated = 0;
        let mut updated_ids: Vec<u64> = Vec::new();
        let mut grown_neuron_id: Option<u64> = None;

        for attempt in 0..self.max_update_attempts {
            let output = network.forward(&input_vec);
            let loss = backprop::mse_loss(&output, &input_vec);

            if loss <= self.growth_threshold {
                // Loss acceptable — do a normal gradient update
                let gradients = backprop::compute_gradients(network, &input_vec, &input_vec);
                neurons_updated = gradients.len();
                updated_ids = gradients.iter().map(|(id, _)| *id).collect();

                for (neuron_id, ng) in &gradients {
                    let lr = self.learning_rate;
                    let wd: Vec<f32> = ng.weight_delta.iter().map(|d| -d * lr).collect();
                    let bd = -ng.bias_delta * lr;
                    network.update_weights(*neuron_id, &wd, bd)?;
                }

                let output_grad = backprop::compute_output_gradient(&output, &input_vec);
                self.embedder
                    .update(&tokens, &output_grad, self.learning_rate);

                // Tag neurons with source + freshness
                tag_neurons(network, &updated_ids, &self.source, self.freshness_category);

                // ── FIX 1 — recalculate importance + protection ───────────────
                manas_memory::scorer::recalc_all(network);
                manas_memory::protector::update_all(network);

                return Ok(LearnReport {
                    loss,
                    tokens_learned: tokens.len() as u32,
                    growth_occurred: false,
                    neurons_updated,
                    freshness_category: self.freshness_category,
                });
            }

            // Loss still too high
            if attempt == self.max_update_attempts - 1 {
                // Final attempt — grow a neuron instead of giving up
                let target = find_highest_loss_layer(network, &input_vec, &input_vec);
                if let Some(layer_id) = target {
                    let input_size = if layer_id == 0 {
                        self.embedder.dim
                    } else {
                        let prev = layer_id as usize - 1;
                        if prev < network.layers.len() {
                            network.layers[prev].neurons.len()
                        } else {
                            self.embedder.dim
                        }
                    };
                    let nid = network.grow_neuron(layer_id, input_size)?;
                    grown_neuron_id = Some(nid);
                    growth_occurred = true;
                }
            } else {
                // Not the last attempt — update weights and try again
                let gradients = backprop::compute_gradients(network, &input_vec, &input_vec);
                neurons_updated = gradients.len();
                updated_ids = gradients.iter().map(|(id, _)| *id).collect();

                for (neuron_id, ng) in &gradients {
                    let lr = self.learning_rate;
                    let wd: Vec<f32> = ng.weight_delta.iter().map(|d| -d * lr).collect();
                    let bd = -ng.bias_delta * lr;
                    network.update_weights(*neuron_id, &wd, bd)?;
                }
            }
        }

        // ── Final pass after growth ───────────────────────────────────────────
        let final_output = network.forward(&input_vec);
        let final_loss = backprop::mse_loss(&final_output, &input_vec);
        let output_grad = backprop::compute_output_gradient(&final_output, &input_vec);
        self.embedder
            .update(&tokens, &output_grad, self.learning_rate);

        if let Some(nid) = grown_neuron_id {
            updated_ids.push(nid);
        }

        // Tag neurons with source + freshness
        tag_neurons(network, &updated_ids, &self.source, self.freshness_category);

        // ── FIX 1 — recalculate importance + protection ───────────────────────
        manas_memory::scorer::recalc_all(network);
        manas_memory::protector::update_all(network);

        Ok(LearnReport {
            loss: final_loss,
            tokens_learned: tokens.len() as u32,
            growth_occurred,
            neurons_updated,
            freshness_category: self.freshness_category,
        })
    }
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Tag a set of neurons with their knowledge origin and freshness category.
/// FIX 2 — source is now stamped onto every neuron that participated in learning.
fn tag_neurons(network: &mut Network, ids: &[u64], source: &Source, freshness_category: u8) {
    for layer in &mut network.layers {
        for neuron in &mut layer.neurons {
            if ids.contains(&neuron.id) {
                neuron.freshness_category = freshness_category;
                if matches!(neuron.source, Source::Unknown) {
                    neuron.source = source.clone();
                }
            }
        }
    }
}

/// Find the layer whose output has the highest MSE loss against the target.
/// Used to decide where to grow a new neuron.
fn find_highest_loss_layer(network: &Network, input: &[f32], target: &[f32]) -> Option<u32> {
    let mut current = input.to_vec();
    let mut highest_loss = 0.0_f32;
    let mut best_layer: Option<u32> = None;

    for layer in &network.layers {
        let output = layer.forward(&current);
        if output.len() == target.len() {
            let loss: f32 = output
                .iter()
                .zip(target)
                .map(|(o, t)| (o - t) * (o - t))
                .sum::<f32>()
                / output.len() as f32;

            if loss > highest_loss {
                highest_loss = loss;
                best_layer = Some(layer.id);
            }
        }
        current = output;
    }

    best_layer.or_else(|| network.layers.last().map(|l| l.id))
}
