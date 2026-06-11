use manas_core::{Network, Neuron};
use std::collections::HashMap;

pub fn importance_for_neuron(neuron: &Neuron, now: u64) -> f32 {
    let freq = activation_frequency(neuron);
    let recency = recency_score(neuron.last_activated, now);
    let magnitude = weight_magnitude(neuron);
    let age = age_grace(neuron.born_at, now);

    0.40 * freq + 0.30 * recency + 0.20 * magnitude + 0.10 * age
}

fn activation_frequency(neuron: &Neuron) -> f32 {
    (neuron.activation_count as f32 / 10_000.0).clamp(0.0, 1.0)
}

fn recency_score(last_activated: u64, now: u64) -> f32 {
    if last_activated == 0 {
        return 0.0;
    }
    let days_since = (now - last_activated) as f32 / 86400.0;
    (-0.1 * days_since).exp()
}

fn weight_magnitude(neuron: &Neuron) -> f32 {
    let l2: f32 = neuron.weights.iter().map(|w| w * w).sum::<f32>().sqrt();
    (l2 / 10.0).clamp(0.0, 1.0)
}

fn age_grace(born_at: u64, now: u64) -> f32 {
    let age_seconds = now.saturating_sub(born_at);
    if age_seconds < 7 * 86400 { 1.0 } else { 0.0 }
}

pub fn recalc_neuron(neuron: &mut Neuron) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    neuron.importance_score = importance_for_neuron(neuron, now);
}

pub fn recalc_all(network: &mut Network) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    for layer in &mut network.layers {
        for neuron in &mut layer.neurons {
            neuron.importance_score = importance_for_neuron(neuron, now);
        }
    }
}

pub fn find_low_importance(network: &Network, threshold: f32) -> Vec<u64> {
    let mut result = Vec::new();
    for layer in &network.layers {
        for neuron in &layer.neurons {
            if neuron.importance_score < threshold {
                result.push(neuron.id);
            }
        }
    }
    result
}

pub fn score_all_to_map(network: &Network) -> HashMap<u64, f32> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut scores = HashMap::new();
    for layer in &network.layers {
        for neuron in &layer.neurons {
            scores.insert(neuron.id, importance_for_neuron(neuron, now));
        }
    }
    scores
}
