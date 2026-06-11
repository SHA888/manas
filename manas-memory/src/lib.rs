pub mod compressor;
pub mod protector;
pub mod scorer;

pub use compressor::{CompressionReport, compress};
use manas_core::{Network, Neuron};
pub use protector::{protection_from_importance, update_all, update_neuron};
pub use scorer::{
    find_low_importance, importance_for_neuron, recalc_all, recalc_neuron, score_all_to_map,
};

pub struct MemoryManager;

impl MemoryManager {
    pub fn score_neuron(neuron: &Neuron) -> f32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        scorer::importance_for_neuron(neuron, now)
    }

    pub fn score_all(network: &mut Network) {
        scorer::recalc_all(network);
    }

    pub fn update_protection(neuron: &mut Neuron) {
        protector::update_neuron(neuron);
    }

    pub fn update_all_protections(network: &mut Network) {
        protector::update_all(network);
    }

    pub fn find_compress_candidates(network: &Network, threshold: f32) -> Vec<u64> {
        compressor::find_candidates(network, threshold)
    }

    pub fn compress_network(
        network: &mut Network,
        importance_threshold: f32,
        similarity_threshold: f32,
    ) -> compressor::CompressionReport {
        compressor::compress(network, importance_threshold, similarity_threshold)
    }

    pub fn full_maintenance(network: &mut Network) {
        scorer::recalc_all(network);
        protector::update_all(network);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::{Network, ProtectionLevel};

    fn test_network() -> Network {
        let mut net = Network::new();
        net.grow_layer(4, 8);
        net.grow_neuron(0, 8).unwrap();
        net
    }

    #[test]
    fn importance_in_range() {
        let net = test_network();
        let neuron = &net.layers[0].neurons[0];
        let score = MemoryManager::score_neuron(neuron);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn recalc_updates_scores() {
        let mut net = test_network();
        MemoryManager::score_all(&mut net);
        for layer in &net.layers {
            for neuron in &layer.neurons {
                assert!((0.0..=1.0).contains(&neuron.importance_score));
            }
        }
    }

    #[test]
    fn protection_bands() {
        assert_eq!(
            protection_from_importance(0.90, 9999999),
            ProtectionLevel::Frozen
        );
        assert_eq!(
            protection_from_importance(0.70, 0),
            ProtectionLevel::Guarded
        );
        assert_eq!(
            protection_from_importance(0.30, 9999999),
            ProtectionLevel::Open
        );
    }

    #[test]
    fn new_neurons_are_guarded() {
        let mut net = test_network();
        MemoryManager::full_maintenance(&mut net);
        for layer in &net.layers {
            for neuron in &layer.neurons {
                let age = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    .saturating_sub(neuron.born_at);
                if age < 7 * 86400 && neuron.importance_score < 0.85 {
                    assert_eq!(neuron.protection_level, ProtectionLevel::Guarded);
                }
            }
        }
    }

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = compressor::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = compressor::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn compress_merges_similar_neurons() {
        let mut net = Network::new();
        net.grow_layer(3, 4);

        for neuron in &mut net.layers[0].neurons {
            neuron.importance_score = 0.05;
            neuron.born_at = 0;
            neuron.weights = vec![0.5, 0.3, 0.1, 0.7];
        }

        let report = MemoryManager::compress_network(&mut net, 0.10, 0.7);
        assert!(report.neurons_removed > 0);
        assert!(report.clusters_formed > 0);
    }

    #[test]
    fn find_low_importance_neurons() {
        let mut net = test_network();
        scorer::recalc_all(&mut net);
        let low = scorer::find_low_importance(&net, 0.5);
        assert!(!low.is_empty());
    }
}
