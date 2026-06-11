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
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    pub fn grow_neuron(&mut self, layer_id: u32, input_size: usize) -> Result<u64, ManasError> {
        let id = self.next_neuron_id();
        let layer = self.layers.iter_mut()
            .find(|l| l.id == layer_id)
            .ok_or_else(|| ManasError::GrowthFailed(format!("layer {} not found", layer_id)))?;

        let neuron = Neuron::new(id, input_size, layer.activation);
        layer.neurons.push(neuron);
        self.total_neurons += 1;
        Ok(id)
    }

    pub fn grow_layer(&mut self, neuron_count: usize, input_size: usize) -> u32 {
        let id = self.layers.last().map(|l| l.id + 1).unwrap_or(0);
        self.layers.push(Layer {
            id,
            neurons: Vec::with_capacity(neuron_count),
            activation: Activation::ReLU,
        });
        for _ in 0..neuron_count {
            let nid = self.next_neuron_id();
            self.layers.last_mut().unwrap().neurons.push(
                Neuron::new(nid, input_size, Activation::ReLU)
            );
            self.total_neurons += 1;
        }
        id
    }

    pub fn update_weights(&mut self, neuron_id: u64, weight_delta: &[f32], bias_delta: f32) -> Result<(), ManasError> {
        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                if neuron.id == neuron_id {
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
        }
        Err(ManasError::NeuronNotFound(neuron_id))
    }

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

    pub fn parameter_count(&self) -> u64 {
        self.layers.iter()
            .flat_map(|l| l.neurons.iter())
            .map(|n| n.weights.len() as u64 + 1)
            .sum()
    }

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

    pub fn all_neurons(&self) -> Vec<(u32, &Neuron)> {
        self.layers.iter()
            .flat_map(|l| l.neurons.iter().map(move |n| (l.id, n)))
            .collect()
    }

    fn next_neuron_id(&self) -> u64 {
        self.layers.iter()
            .flat_map(|l| l.neurons.iter())
            .map(|n| n.id)
            .max()
            .unwrap_or(0) + 1
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}
