use manas_core::{Activation, Network, Neuron, ProtectionLevel};
use std::collections::HashMap;

pub struct CompressionReport {
    pub candidates_found: usize,
    pub clusters_formed: usize,
    pub neurons_removed: usize,
    pub neurons_created: usize,
}

pub fn find_candidates(network: &Network, threshold: f32) -> Vec<u64> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut candidates = Vec::new();
    for layer in &network.layers {
        for neuron in &layer.neurons {
            if neuron.importance_score < threshold {
                let age = now.saturating_sub(neuron.born_at);
                if age >= 7 * 86400 {
                    candidates.push(neuron.id);
                }
            }
        }
    }
    candidates
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

fn neuron_by_id<'a>(network: &'a Network, id: u64) -> Option<&'a Neuron> {
    for layer in &network.layers {
        for neuron in &layer.neurons {
            if neuron.id == id {
                return Some(neuron);
            }
        }
    }
    None
}

fn neuron_by_id_mut<'a>(network: &'a mut Network, id: u64) -> Option<&'a mut Neuron> {
    for layer in &mut network.layers {
        for neuron in &mut layer.neurons {
            if neuron.id == id {
                return Some(neuron);
            }
        }
    }
    None
}

pub fn cluster_candidates(
    network: &Network,
    candidates: &[u64],
    similarity_threshold: f32,
) -> Vec<Vec<u64>> {
    let mut visited: HashMap<u64, bool> = HashMap::new();
    for &id in candidates {
        visited.insert(id, false);
    }

    let mut clusters: Vec<Vec<u64>> = Vec::new();

    for &id in candidates {
        if visited.get(&id) == Some(&true) {
            continue;
        }

        let neuron_a = match neuron_by_id(network, id) {
            Some(n) => n,
            None => continue,
        };

        let mut cluster = vec![id];
        visited.insert(id, true);

        for &other_id in candidates {
            if other_id == id || visited.get(&other_id) == Some(&true) {
                continue;
            }
            let neuron_b = match neuron_by_id(network, other_id) {
                Some(n) => n,
                None => continue,
            };

            if neuron_a.weights.len() == neuron_b.weights.len()
                && cosine_similarity(&neuron_a.weights, &neuron_b.weights) >= similarity_threshold
            {
                cluster.push(other_id);
                visited.insert(other_id, true);
            }
        }

        if cluster.len() > 1 {
            clusters.push(cluster);
        }
    }

    clusters
}

pub fn merge_cluster(network: &Network, cluster: &[u64]) -> Option<Neuron> {
    if cluster.is_empty() {
        return None;
    }

    let neurons: Vec<&Neuron> = cluster
        .iter()
        .filter_map(|&id| neuron_by_id(network, id))
        .collect();

    if neurons.is_empty() {
        return None;
    }

    let n = neurons.len() as f32;
    let weight_count = neurons[0].weights.len();
    let mut avg_weights = vec![0.0f32; weight_count];
    let mut avg_bias = 0.0f32;
    let mut avg_importance = 0.0f32;
    let mut activation_sum = Activation::ReLU;
    let mut total_activation_count: u64 = 0;

    for neuron in &neurons {
        for (i, w) in neuron.weights.iter().enumerate() {
            avg_weights[i] += w / n;
        }
        avg_bias += neuron.bias / n;
        avg_importance += neuron.importance_score / n;
        total_activation_count += neuron.activation_count;
        activation_sum = neuron.activation;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let merged_id = cluster[0];

    Some(Neuron {
        id: merged_id,
        weights: avg_weights,
        bias: avg_bias,
        activation: activation_sum,
        importance_score: avg_importance,
        born_at: neurons.iter().map(|n| n.born_at).min().unwrap_or(now),
        last_activated: now,
        activation_count: total_activation_count,
        learned_at: now,
        last_verified: now,
        freshness_category: 1,
        source: manas_core::Source::Unknown,
        is_protected: false,
        protection_level: ProtectionLevel::Open,
    })
}

pub fn compress(
    network: &mut Network,
    threshold: f32,
    similarity_threshold: f32,
) -> CompressionReport {
    let candidates = find_candidates(network, threshold);
    if candidates.is_empty() {
        return CompressionReport {
            candidates_found: 0,
            clusters_formed: 0,
            neurons_removed: 0,
            neurons_created: 0,
        };
    }

    let clusters = cluster_candidates(network, &candidates, similarity_threshold);
    let mut removed = Vec::new();

    for cluster in &clusters {
        if let Some(merged) = merge_cluster(network, cluster) {
            for &id in cluster.iter().skip(1) {
                for layer in &mut network.layers {
                    layer.neurons.retain(|n| n.id != id);
                }
                removed.push(id);
                network.total_neurons -= 1;
            }

            if let Some(target) = neuron_by_id_mut(network, cluster[0]) {
                *target = merged;
            }
        }
    }

    CompressionReport {
        candidates_found: candidates.len(),
        clusters_formed: clusters.len(),
        neurons_removed: removed.len(),
        neurons_created: clusters.len(),
    }
}
