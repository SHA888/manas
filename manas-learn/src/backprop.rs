use manas_core::Network;

pub fn mse_loss(prediction: &[f32], target: &[f32]) -> f32 {
    if prediction.is_empty() || target.is_empty() {
        return 0.0;
    }
    let n = prediction.len().min(target.len()) as f32;
    prediction
        .iter()
        .zip(target)
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f32>()
        / n
}

#[derive(Debug, Clone)]
pub struct ForwardCache {
    pub layer_inputs: Vec<Vec<f32>>,
    pub pre_activations: Vec<Vec<f32>>,
    pub layer_outputs: Vec<Vec<f32>>,
}

pub fn forward_with_cache(network: &Network, input: &[f32]) -> ForwardCache {
    let layer_count = network.layers.len();
    let mut cache = ForwardCache {
        layer_inputs: Vec::with_capacity(layer_count + 1),
        pre_activations: Vec::with_capacity(layer_count),
        layer_outputs: Vec::with_capacity(layer_count),
    };

    cache.layer_inputs.push(input.to_vec());
    let mut current = cache.layer_inputs[0].clone();

    for layer in &network.layers {
        let mut zs = Vec::with_capacity(layer.neurons.len());
        let mut outs = Vec::with_capacity(layer.neurons.len());

        for neuron in &layer.neurons {
            let z: f32 = neuron
                .weights
                .iter()
                .zip(&current)
                .map(|(w, i)| w * i)
                .sum::<f32>()
                + neuron.bias;
            zs.push(z);
            outs.push(neuron.activation.apply(z));
        }

        cache.pre_activations.push(zs);
        cache.layer_inputs.push(outs.clone());
        cache.layer_outputs.push(outs);
        current = cache.layer_inputs.last().unwrap().clone();
    }

    cache
}

pub struct NeuronGradients {
    pub weight_delta: Vec<f32>,
    pub bias_delta: f32,
}

pub fn compute_gradients(
    network: &Network,
    input: &[f32],
    target: &[f32],
) -> Vec<(u64, NeuronGradients)> {
    let cache = forward_with_cache(network, input);

    if network.layers.is_empty() {
        return Vec::new();
    }

    let num_layers = network.layers.len();
    let output_size = cache.layer_outputs[num_layers - 1].len();

    let mut deltas: Vec<Vec<f32>> = Vec::new();

    let output = &cache.layer_outputs[num_layers - 1];
    let mut delta_out = Vec::with_capacity(output_size);
    let last_layer = &network.layers[num_layers - 1];

    for (i, output_value) in output.iter().enumerate() {
        let target_value = target.get(i).copied().unwrap_or(0.0);
        let da = 2.0 * (*output_value - target_value) / output_size.max(1) as f32;
        let act = last_layer.neurons[i].activation;
        delta_out.push(da * act.derivative(cache.pre_activations[num_layers - 1][i]));
    }
    deltas.push(delta_out);

    for l in (0..num_layers - 1).rev() {
        let next_layer = &network.layers[l + 1];
        let current_delta = &deltas[num_layers - 2 - l];
        let num_neurons = network.layers[l].neurons.len();
        let layer_input = &cache.pre_activations[l];

        let mut delta = Vec::with_capacity(num_neurons);
        for (j, pre_act) in layer_input.iter().enumerate().take(num_neurons) {
            let mut error = 0.0;
            for (k, delta_val) in current_delta.iter().enumerate() {
                if j < next_layer.neurons[k].weights.len() {
                    error += next_layer.neurons[k].weights[j] * delta_val;
                }
            }
            let act = network.layers[l].neurons[j].activation;
            delta.push(error * act.derivative(*pre_act));
        }
        deltas.push(delta);
    }

    let total_neurons: usize = network.layers.iter().map(|l| l.neurons.len()).sum();
    let mut result = Vec::with_capacity(total_neurons);
    for l in 0..num_layers {
        let layer = &network.layers[l];
        let input_a = &cache.layer_inputs[l];
        let layer_delta = &deltas[num_layers - 1 - l];

        for (i, neuron) in layer.neurons.iter().enumerate() {
            let delta_i = layer_delta.get(i).copied().unwrap_or(0.0);

            let mut weight_delta = Vec::with_capacity(neuron.weights.len());
            for j in 0..neuron.weights.len() {
                let g = if j < input_a.len() {
                    delta_i * input_a[j]
                } else {
                    0.0
                };
                weight_delta.push(g);
            }

            let bias_delta = delta_i;
            result.push((
                neuron.id,
                NeuronGradients {
                    weight_delta,
                    bias_delta,
                },
            ));
        }
    }

    result
}

pub fn compute_output_gradient(output: &[f32], target: &[f32]) -> Vec<f32> {
    let n = output.len().max(1) as f32;
    output
        .iter()
        .zip(target)
        .map(|(p, t)| 2.0 * (p - t) / n)
        .collect()
}
