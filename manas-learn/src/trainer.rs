use std::collections::HashMap;
use manas_core::{ManasError, Network};
use crate::tokenizer::Tokenizer;
use crate::embedder::Embedder;
use crate::backprop::{self};

pub const DEFAULT_LEARNING_RATE: f32 = 0.01;
pub const DEFAULT_GROWTH_THRESHOLD: f32 = 0.35;
pub const DEFAULT_MAX_UPDATE_ATTEMPTS: u32 = 3;
pub const DEFAULT_EMBED_DIM: usize = 64;

pub struct LearnReport {
    pub loss: f32,
    pub tokens_learned: u32,
    pub growth_occurred: bool,
    pub neurons_updated: usize,
    pub freshness_category: u8,
}

#[derive(Debug, Clone)]
pub struct TrainerSnapshot {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub embed_table: HashMap<u32, Vec<f32>>,
    pub embed_dim: usize,
}

pub struct Trainer {
    pub tokenizer: Tokenizer,
    pub embedder: Embedder,
    pub learning_rate: f32,
    pub growth_threshold: f32,
    pub max_update_attempts: u32,
    pub freshness_category: u8,
}

pub fn detect_freshness_category(text: &str) -> u8 {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    let realtime_keywords = ["news", "today", "breaking", "current", "live", "latest", "update"];
    let fast_keywords = ["released", "version", "release", "announced", "launched"];
    let slow_keywords = ["since", "history", "was", "were", "had", "been", "old", "past", "origin"];
    let timeless_keywords = ["always", "formula", "law", "proof", "theorem", "definition", "always", "never", "forever", "constant"];

    for w in &words {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        if timeless_keywords.contains(&w) {
            return 0;
        }
    }
    for w in &words {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        if realtime_keywords.contains(&w) {
            return 3;
        }
    }
    for w in &words {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        if fast_keywords.contains(&w) {
            return 2;
        }
    }
    for w in &words {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        if slow_keywords.contains(&w) {
            return 1;
        }
    }
    1
}

impl Trainer {
    pub fn new() -> Self {
        Trainer {
            tokenizer: Tokenizer::new(),
            embedder: Embedder::new(DEFAULT_EMBED_DIM),
            learning_rate: DEFAULT_LEARNING_RATE,
            growth_threshold: DEFAULT_GROWTH_THRESHOLD,
            max_update_attempts: DEFAULT_MAX_UPDATE_ATTEMPTS,
            freshness_category: 1,
        }
    }

    pub fn new_with_params(
        embed_dim: usize,
        learning_rate: f32,
        growth_threshold: f32,
    ) -> Self {
        Trainer {
            tokenizer: Tokenizer::new(),
            embedder: Embedder::new(embed_dim),
            learning_rate,
            growth_threshold,
            max_update_attempts: DEFAULT_MAX_UPDATE_ATTEMPTS,
            freshness_category: 1,
        }
    }

    pub fn snapshot(&self) -> TrainerSnapshot {
        TrainerSnapshot {
            vocab: self.tokenizer.vocab.clone(),
            id_to_token: self.tokenizer.id_to_token.clone(),
            embed_table: self.embedder.table.clone(),
            embed_dim: self.embedder.dim,
        }
    }

    pub fn restore(&mut self, snapshot: &TrainerSnapshot) {
        self.tokenizer.vocab = snapshot.vocab.clone();
        self.tokenizer.id_to_token = snapshot.id_to_token.clone();
        self.tokenizer.vocab_size = snapshot.vocab.len() as u32;
        self.embedder.table = snapshot.embed_table.clone();
        self.embedder.dim = snapshot.embed_dim;
    }

    pub fn learn(&mut self, network: &mut Network, text: &str) -> Result<LearnReport, ManasError> {
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

        if network.layers.is_empty() {
            let hidden = (self.embedder.dim / 4).max(2);
            network.grow_layer(hidden, self.embedder.dim);
            network.grow_layer(self.embedder.dim, hidden);
        }

        let mut growth_occurred = false;
        let mut neurons_updated = 0;
        let mut updated_ids: Vec<u64> = Vec::new();
        let mut grown_neuron_id: Option<u64> = None;

        for attempt in 0..self.max_update_attempts {
            let output = network.forward(&input_vec);
            let loss = backprop::mse_loss(&output, &input_vec);

            if loss <= self.growth_threshold {
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
                self.embedder.update(&tokens, &output_grad, self.learning_rate);

                set_freshness_on_neurons(network, &updated_ids, self.freshness_category);

                return Ok(LearnReport {
                    loss,
                    tokens_learned: tokens.len() as u32,
                    growth_occurred: false,
                    neurons_updated,
                    freshness_category: self.freshness_category,
                });
            }

            if attempt == self.max_update_attempts - 1 && loss > self.growth_threshold {
                let target_layer = find_highest_loss_layer(network, &input_vec, &input_vec);
                if let Some(layer_id) = target_layer {
                    let input_size = if layer_id == 0 {
                        self.embedder.dim
                    } else {
                        let prev_idx = layer_id as usize - 1;
                        if prev_idx < network.layers.len() {
                            network.layers[prev_idx].neurons.len()
                        } else {
                            self.embedder.dim
                        }
                    };
                    let nid = network.grow_neuron(layer_id, input_size)?;
                    grown_neuron_id = Some(nid);
                    growth_occurred = true;
                }
            } else {
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

        let final_output = network.forward(&input_vec);
        let final_loss = backprop::mse_loss(&final_output, &input_vec);
        let output_grad = backprop::compute_output_gradient(&final_output, &input_vec);
        self.embedder.update(&tokens, &output_grad, self.learning_rate);

        if let Some(nid) = grown_neuron_id {
            updated_ids.push(nid);
        }
        set_freshness_on_neurons(network, &updated_ids, self.freshness_category);

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

fn set_freshness_on_neurons(network: &mut Network, ids: &[u64], category: u8) {
    for layer in &mut network.layers {
        for neuron in &mut layer.neurons {
            if ids.contains(&neuron.id) {
                neuron.freshness_category = category;
            }
        }
    }
}

fn find_highest_loss_layer(network: &Network, input: &[f32], target: &[f32]) -> Option<u32> {
    let mut current = input.to_vec();
    let mut highest_loss = 0.0f32;
    let mut best_layer: Option<u32> = None;

    for layer in &network.layers {
        let output = layer.forward(&current);
        if output.len() == target.len() {
            let loss: f32 = output.iter()
                .zip(target)
                .map(|(o, t)| (o - t) * (o - t))
                .sum::<f32>() / output.len() as f32;
            if loss > highest_loss {
                highest_loss = loss;
                best_layer = Some(layer.id);
            }
        }
        current = output;
    }

    best_layer.or_else(|| network.layers.last().map(|l| l.id))
}
