use crate::activation::Activation;
use crate::error::ManasError;
use crate::layer::Layer;
use crate::neuron::{Neuron, ProtectionLevel};

pub struct Network {
    pub layers: Vec<Layer>,
    pub total_neurons: u64,
    pub created_at: u64,
    pub version: u8,
    pub total_texts_learned: u64,

    /// Monotonic counter — the ONLY source of truth for new neuron ids.
    ///
    /// Always equals (highest id ever assigned + 1). Never scans neurons.
    /// Set to 0 on `new()`. After loading from disk, call `recompute_next_id()`
    /// once to restore the counter from the saved neuron ids.
    pub next_id: u64,
}

impl Network {
    pub fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Network {
            layers: Vec::new(),
            total_neurons: 0,
            created_at: now,
            version: 1,
            total_texts_learned: 0,
            next_id: 0,
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Forward pass that also returns per-layer neuron activations.
    /// Used by `manas-cli trace` and `manas-agent` freshness system.
    pub fn forward_with_activations(&self, input: &[f32]) -> (Vec<f32>, Vec<Vec<(u64, f32)>>) {
        let mut all_layer_acts: Vec<Vec<(u64, f32)>> = Vec::new();
        let mut current = input.to_vec();

        for layer in &self.layers {
            let mut layer_acts: Vec<(u64, f32)> = Vec::with_capacity(layer.neurons.len());
            let mut output = Vec::with_capacity(layer.neurons.len());

            for neuron in &layer.neurons {
                let val = neuron.activate(&current);
                layer_acts.push((neuron.id, val));
                output.push(val);
            }

            all_layer_acts.push(layer_acts);
            current = output;
        }

        (current, all_layer_acts)
    }

    // ── Growth ────────────────────────────────────────────────────────────────

    /// Grow a new neuron into the layer with the given id.
    ///
    /// Returns the new neuron's id.
    /// O(L) — scans layers to find the target, L = layer count (tiny).
    pub fn grow_neuron(&mut self, layer_id: u32, input_size: usize) -> Result<u64, ManasError> {
        let id = self.alloc_id();

        let layer = self
            .layers
            .iter_mut()
            .find(|l| l.id == layer_id)
            .ok_or_else(|| ManasError::GrowthFailed(format!("layer {} not found", layer_id)))?;

        layer
            .neurons
            .push(Neuron::new(id, input_size, layer.activation));
        self.total_neurons += 1;
        Ok(id)
    }

    /// Add a new layer with `neuron_count` neurons each expecting `input_size` inputs.
    ///
    /// Returns the new layer's id.
    pub fn grow_layer(&mut self, neuron_count: usize, input_size: usize) -> u32 {
        let layer_id = self.layers.last().map(|l| l.id + 1).unwrap_or(0);

        let mut layer = Layer {
            id: layer_id,
            neurons: Vec::with_capacity(neuron_count),
            activation: Activation::ReLU,
        };

        for _ in 0..neuron_count {
            let nid = self.alloc_id();
            layer
                .neurons
                .push(Neuron::new(nid, input_size, Activation::ReLU));
            self.total_neurons += 1;
        }

        self.layers.push(layer);
        layer_id
    }

    // ── Weight updates ────────────────────────────────────────────────────────

    /// Apply weight and bias deltas to a specific neuron.
    ///
    /// Respects the neuron's current protection level:
    /// - `Frozen`  → returns `NeuronFrozen` error, no update applied
    /// - `Guarded` → deltas clamped to ±0.01
    /// - `Open`    → full delta applied
    pub fn update_weights(
        &mut self,
        neuron_id: u64,
        weight_delta: &[f32],
        bias_delta: f32,
    ) -> Result<(), ManasError> {
        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                if neuron.id != neuron_id {
                    continue;
                }
                match neuron.protection_level {
                    ProtectionLevel::Frozen => {
                        return Err(ManasError::NeuronFrozen(neuron_id));
                    }
                    ProtectionLevel::Guarded => {
                        for (w, d) in neuron.weights.iter_mut().zip(weight_delta) {
                            *w += d.clamp(-0.01, 0.01);
                        }
                        neuron.bias += bias_delta.clamp(-0.01, 0.01);
                    }
                    ProtectionLevel::Open => {
                        for (w, d) in neuron.weights.iter_mut().zip(weight_delta) {
                            *w += d;
                        }
                        neuron.bias += bias_delta;
                    }
                }
                return Ok(());
            }
        }
        Err(ManasError::NeuronNotFound(neuron_id))
    }

    /// Record that a neuron fired — updates its `last_activated` timestamp
    /// and increments `activation_count`.
    pub fn record_activation(&mut self, neuron_id: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                if neuron.id == neuron_id {
                    neuron.last_activated = now;
                    neuron.activation_count += 1;
                    return;
                }
            }
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Total number of learnable parameters (weights + biases) across all neurons.
    pub fn parameter_count(&self) -> u64 {
        self.layers
            .iter()
            .flat_map(|l| l.neurons.iter())
            .map(|n| n.weights.len() as u64 + 1)
            .sum()
    }

    /// All neurons in the network as `(layer_id, &Neuron)` pairs.
    pub fn all_neurons(&self) -> Vec<(u32, &Neuron)> {
        self.layers
            .iter()
            .flat_map(|l| l.neurons.iter().map(move |n| (l.id, n)))
            .collect()
    }

    // ── Counter management ────────────────────────────────────────────────────

    /// Allocate the next neuron id — O(1), never scans neurons.
    ///
    /// This is the ONLY place a new id is created. Every growth path
    /// (grow_neuron, grow_layer) goes through here.
    #[inline]
    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Rebuild `next_id` by scanning all neurons once.
    ///
    /// **Call this exactly once after loading a Network from a `.manas` file.**
    /// After that, `alloc_id()` is O(1) for the lifetime of the network.
    ///
    /// In `manas-store/src/reader.rs`, add:
    /// ```ignore
    /// let mut network = Network { layers, total_neurons: ..., ... };
    /// network.recompute_next_id();
    /// Ok(network)
    /// ```
    pub fn recompute_next_id(&mut self) {
        let max_id = self
            .layers
            .iter()
            .flat_map(|l| l.neurons.iter())
            .map(|n| n.id)
            .max()
            .unwrap_or(0);

        // next_id must be strictly greater than every existing id
        self.next_id = max_id + 1;
    }

    /// The next id that will be assigned without allocating it.
    /// Useful for tests and assertions.
    #[inline]
    pub fn peek_next_id(&self) -> u64 {
        self.next_id
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── id counter ────────────────────────────────────────────────────────────

    #[test]
    fn ids_are_unique_and_monotonic() {
        let mut net = Network::new();
        net.grow_layer(4, 8);
        net.grow_layer(2, 4);

        let ids: Vec<u64> = net.all_neurons().iter().map(|(_, n)| n.id).collect();

        // all ids unique
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "duplicate neuron ids detected");

        // counter is ahead of all existing ids
        assert!(net.next_id > *ids.iter().max().unwrap());
    }

    #[test]
    fn grow_neuron_increments_counter_by_one() {
        let mut net = Network::new();
        net.grow_layer(2, 4);

        let before = net.peek_next_id();
        let new_id = net.grow_neuron(0, 4).unwrap();

        assert_eq!(new_id, before);
        assert_eq!(net.peek_next_id(), before + 1);
    }

    #[test]
    fn grow_layer_increments_counter_by_neuron_count() {
        let mut net = Network::new();

        let before = net.peek_next_id();
        net.grow_layer(5, 8);

        assert_eq!(net.peek_next_id(), before + 5);
    }

    #[test]
    fn counter_never_reuses_ids_after_many_growths() {
        let mut net = Network::new();
        net.grow_layer(10, 8);
        net.grow_layer(10, 10);
        net.grow_layer(10, 10);

        // grow 20 more individual neurons
        let layer_id = net.layers[1].id;
        for _ in 0..20 {
            net.grow_neuron(layer_id, 10).unwrap();
        }

        let ids: Vec<u64> = net.all_neurons().iter().map(|(_, n)| n.id).collect();
        let total = ids.len();
        let mut deduped = ids;
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(deduped.len(), total, "id collision after many growths");
    }

    // ── recompute_next_id (simulates load from disk) ───────────────────────

    #[test]
    fn recompute_restores_counter_correctly() {
        // Simulate: build network, "serialize" it (keep layers), reconstruct
        // with next_id=0 (as if loaded from file before fix), then recompute.
        let mut original = Network::new();
        original.grow_layer(4, 8);
        original.grow_layer(2, 4);

        let max_id_before = original
            .all_neurons()
            .iter()
            .map(|(_, n)| n.id)
            .max()
            .unwrap();

        // Simulate a load that doesn't restore next_id (old reader)
        let mut loaded = Network {
            layers: original.layers.clone(),
            total_neurons: original.total_neurons,
            created_at: original.created_at,
            version: original.version,
            total_texts_learned: original.total_texts_learned,
            next_id: 0, // ← what an old reader would produce
        };

        // After recompute, counter must be max_id + 1
        loaded.recompute_next_id();
        assert_eq!(loaded.next_id, max_id_before + 1);
    }

    #[test]
    fn no_id_collision_after_recompute_and_grow() {
        let mut net = Network::new();
        net.grow_layer(5, 8);

        let existing_ids: Vec<u64> = net.all_neurons().iter().map(|(_, n)| n.id).collect();

        // Simulate reload
        net.next_id = 0;
        net.recompute_next_id();

        // Grow more neurons after reload
        let layer_id = net.layers[0].id;
        for _ in 0..5 {
            net.grow_neuron(layer_id, 8).unwrap();
        }

        let all_ids: Vec<u64> = net.all_neurons().iter().map(|(_, n)| n.id).collect();

        // None of the new ids should collide with existing ones
        for id in &all_ids[existing_ids.len()..] {
            assert!(
                !existing_ids.contains(id),
                "new neuron id {} collides with existing id",
                id
            );
        }
    }

    // ── forward pass ──────────────────────────────────────────────────────────

    #[test]
    fn forward_output_size_matches_last_layer() {
        let mut net = Network::new();
        net.grow_layer(8, 4);
        net.grow_layer(4, 8);

        let output = net.forward(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn forward_no_nan() {
        let mut net = Network::new();
        net.grow_layer(16, 8);
        net.grow_layer(8, 16);

        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let output = net.forward(&input);
        for v in &output {
            assert!(!v.is_nan(), "NaN in forward output");
        }
    }

    #[test]
    fn forward_with_activations_counts_match() {
        let mut net = Network::new();
        net.grow_layer(4, 3);
        net.grow_layer(2, 4);

        let (output, acts) = net.forward_with_activations(&[0.5, -0.3, 0.1]);
        assert_eq!(acts.len(), 2);
        assert_eq!(acts[0].len(), 4); // layer 0 has 4 neurons
        assert_eq!(acts[1].len(), 2); // layer 1 has 2 neurons
        assert_eq!(output.len(), 2);
    }

    // ── weight updates ────────────────────────────────────────────────────────

    #[test]
    fn update_weights_open_neuron() {
        let mut net = Network::new();
        net.grow_layer(2, 3);

        let nid = net.layers[0].neurons[0].id;

        // Force Open protection so the update actually applies
        net.layers[0].neurons[0].protection_level = ProtectionLevel::Open;

        let before: Vec<f32> = net.layers[0].neurons[0].weights.clone();
        let deltas = vec![0.1, 0.2, 0.3];
        net.update_weights(nid, &deltas, 0.01).unwrap();
        let after = &net.layers[0].neurons[0].weights;

        for i in 0..3 {
            assert!(
                (after[i] - (before[i] + deltas[i])).abs() < 1e-6,
                "weight {} not updated correctly",
                i
            );
        }
    }

    #[test]
    fn update_weights_frozen_returns_error() {
        let mut net = Network::new();
        net.grow_layer(2, 3);

        let nid = net.layers[0].neurons[0].id;
        net.layers[0].neurons[0].protection_level = ProtectionLevel::Frozen;

        let result = net.update_weights(nid, &[0.1, 0.2, 0.3], 0.0);
        assert!(matches!(result, Err(ManasError::NeuronFrozen(_))));
    }

    #[test]
    fn update_weights_guarded_clamps_delta() {
        let mut net = Network::new();
        net.grow_layer(1, 2);

        let nid = net.layers[0].neurons[0].id;
        net.layers[0].neurons[0].protection_level = ProtectionLevel::Guarded;

        let before: Vec<f32> = net.layers[0].neurons[0].weights.clone();
        // Large delta — should be clamped to 0.01
        net.update_weights(nid, &[100.0, 100.0], 0.0).unwrap();
        let after = &net.layers[0].neurons[0].weights;

        for i in 0..2 {
            let change = (after[i] - before[i]).abs();
            assert!(
                change <= 0.01 + 1e-6,
                "guarded delta not clamped: change was {}",
                change
            );
        }
    }

    #[test]
    fn update_weights_neuron_not_found() {
        let mut net = Network::new();
        net.grow_layer(2, 3);

        let result = net.update_weights(9999, &[0.1, 0.2, 0.3], 0.0);
        assert!(matches!(result, Err(ManasError::NeuronNotFound(9999))));
    }

    // ── parameter count ───────────────────────────────────────────────────────

    #[test]
    fn parameter_count_correct() {
        let mut net = Network::new();
        // Layer 0: 4 neurons × (8 weights + 1 bias) = 36
        net.grow_layer(4, 8);
        // Layer 1: 2 neurons × (4 weights + 1 bias) = 10
        net.grow_layer(2, 4);

        assert_eq!(net.parameter_count(), 36 + 10);
    }

    // ── all_neurons ───────────────────────────────────────────────────────────

    #[test]
    fn all_neurons_returns_correct_count() {
        let mut net = Network::new();
        net.grow_layer(3, 4);
        net.grow_layer(2, 3);

        assert_eq!(net.all_neurons().len(), 5);
    }

    #[test]
    fn grow_neuron_invalid_layer_errors() {
        let mut net = Network::new();
        net.grow_layer(2, 4);
        assert!(net.grow_neuron(999, 4).is_err());
    }
}
