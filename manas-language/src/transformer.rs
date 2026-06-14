use crate::attention::{CausalSelfAttention, SimpleRng, mat_vec_mul, random_vec};

// ─── Feed‑Forward ────────────────────────────────────────────────

#[derive(Clone)]
pub struct FeedForward {
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl FeedForward {
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = SimpleRng::new(43);
        let scale = (1.0 / (embed_dim as f32).sqrt()).min(0.02);
        FeedForward {
            embed_dim,
            hidden_dim,
            w1: random_vec(&mut rng, embed_dim * hidden_dim, scale),
            b1: vec![0.0; hidden_dim],
            w2: random_vec(&mut rng, hidden_dim * embed_dim, scale),
            b2: vec![0.0; embed_dim],
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        // w1 @ input + b1 → ReLU
        let hidden = mat_vec_mul(&self.w1, self.hidden_dim, self.embed_dim, input);
        let mut hidden = add_vectors(&hidden, &self.b1);
        relu_vec(&mut hidden);

        // w2 @ hidden + b2
        let out = mat_vec_mul(&self.w2, self.embed_dim, self.hidden_dim, &hidden);
        add_vectors(&out, &self.b2)
    }

    pub fn train_step(&mut self, input: &[f32], grad_output: &[f32], learning_rate: f32) {
        // Forward with caching
        let hidden_pre = mat_vec_mul(&self.w1, self.hidden_dim, self.embed_dim, input);
        let hidden_pre_bias = add_vectors(&hidden_pre, &self.b1);
        let hidden_post: Vec<f32> = hidden_pre_bias.iter().map(|x| x.max(0.0)).collect();

        // Backprop through w2: grad_w2[r][c] = grad_output[r] * hidden_post[c]
        let mut grad_w2 = vec![0.0; self.embed_dim * self.hidden_dim];
        for (r, &go) in grad_output.iter().enumerate() {
            let base = r * self.hidden_dim;
            for (c, &hp) in hidden_post.iter().enumerate() {
                grad_w2[base + c] = go * hp;
            }
        }

        // Backprop through b2 = grad_output
        let mut grad_b2 = grad_output.to_vec();

        // Backprop through w2^T: grad_hidden[c] = sum_r w2[r][c] * grad_output[r]
        let mut grad_hidden = vec![0.0; self.hidden_dim];
        for (c, gh) in grad_hidden.iter_mut().enumerate() {
            let mut s = 0.0;
            for (r, &go) in grad_output.iter().enumerate() {
                s += self.w2[r * self.hidden_dim + c] * go;
            }
            *gh = s;
        }

        // ReLU derivative
        for (i, gh) in grad_hidden.iter_mut().enumerate() {
            if hidden_pre_bias[i] <= 0.0 {
                *gh = 0.0;
            }
        }

        // Backprop through w1: grad_w1[r][c] = grad_hidden[r] * input[c]
        let mut grad_w1 = vec![0.0; self.hidden_dim * self.embed_dim];
        for (r, &gh) in grad_hidden.iter().enumerate() {
            let base = r * self.embed_dim;
            for (c, &inp) in input.iter().enumerate() {
                grad_w1[base + c] = gh * inp;
            }
        }

        // Backprop through b1 = grad_hidden (reuse, renamed)
        let mut grad_b1 = grad_hidden;

        // Gradient clipping to [-1.0, 1.0]
        for g in &mut grad_w1 {
            *g = g.clamp(-1.0, 1.0);
        }
        for g in &mut grad_b1 {
            *g = g.clamp(-1.0, 1.0);
        }
        for g in &mut grad_w2 {
            *g = g.clamp(-1.0, 1.0);
        }
        for g in &mut grad_b2 {
            *g = g.clamp(-1.0, 1.0);
        }

        // NaN/inf check — skip update if any gradient is not finite
        let has_nan = grad_w1
            .iter()
            .chain(&grad_b1)
            .chain(&grad_w2)
            .chain(&grad_b2)
            .any(|&g| !g.is_finite());
        if has_nan {
            return;
        }

        // Update weights
        for (i, &g) in grad_w1.iter().enumerate() {
            self.w1[i] -= learning_rate * g;
        }
        for (i, &g) in grad_b1.iter().enumerate() {
            self.b1[i] -= learning_rate * g;
        }
        for (i, &g) in grad_w2.iter().enumerate() {
            self.w2[i] -= learning_rate * g;
        }
        for (i, &g) in grad_b2.iter().enumerate() {
            self.b2[i] -= learning_rate * g;
        }
    }
}

// ─── Tiny Transformer Block ──────────────────────────────────────

#[derive(Clone)]
pub struct TinyTransformerBlock {
    pub embed_dim: usize,
    pub attention: CausalSelfAttention,
    pub feed_forward: FeedForward,
}

impl TinyTransformerBlock {
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        TinyTransformerBlock {
            embed_dim,
            attention: CausalSelfAttention::new(embed_dim),
            feed_forward: FeedForward::new(embed_dim, hidden_dim),
        }
    }

    /// Forward pass: self‑attention → residual add → FFN → residual add.
    pub fn forward(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if inputs.is_empty() || self.embed_dim == 0 {
            return Vec::new();
        }

        // 1. Causal self‑attention
        let attn_out = self.attention.forward(inputs);

        // 2. Residual add: x + attention_output
        let mut x: Vec<Vec<f32>> = inputs
            .iter()
            .zip(attn_out.iter())
            .map(|(inp, att)| add_vectors(inp, att))
            .collect();

        // 3. Feed‑forward per token
        let ff_out: Vec<Vec<f32>> = x.iter().map(|v| self.feed_forward.forward(v)).collect();

        // 4. Residual add: x + feed_forward_output
        x.iter_mut().zip(ff_out.iter()).for_each(|(xv, ffv)| {
            for (a, b) in xv.iter_mut().zip(ffv.iter()) {
                *a += b;
            }
        });

        x
    }

    /// Forward pass that also returns the per-position FFN inputs (residual
    /// after attention) so callers can later backprop through the FFN.
    pub fn forward_with_ffn_inputs(&self, inputs: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        if inputs.is_empty() || self.embed_dim == 0 {
            return (Vec::new(), Vec::new());
        }

        let attn_out = self.attention.forward(inputs);

        let ffn_inputs: Vec<Vec<f32>> = inputs
            .iter()
            .zip(attn_out.iter())
            .map(|(inp, att)| add_vectors(inp, att))
            .collect();

        let ff_out: Vec<Vec<f32>> = ffn_inputs
            .iter()
            .map(|v| self.feed_forward.forward(v))
            .collect();

        let block_out: Vec<Vec<f32>> = ffn_inputs
            .iter()
            .zip(ff_out.iter())
            .map(|(xv, ffv)| add_vectors(xv, ffv))
            .collect();

        (block_out, ffn_inputs)
    }
}

// ─── Internal helpers ────────────────────────────────────────────

fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn relu_vec(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.max(0.0);
    }
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: shape test.
    #[test]
    fn forward_shape() {
        let block = TinyTransformerBlock::new(4, 8);
        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let output = block.forward(&inputs);
        assert_eq!(output.len(), 3);
        for v in &output {
            assert_eq!(v.len(), 4);
        }
    }

    /// Test 2: empty input.
    #[test]
    fn empty_input() {
        let block = TinyTransformerBlock::new(4, 8);
        let output = block.forward(&[]);
        assert!(output.is_empty());
    }

    /// Test 3: no NaN in output.
    #[test]
    fn no_nan() {
        let block = TinyTransformerBlock::new(8, 16);
        let inputs = vec![
            vec![0.5, -0.2, 0.1, 0.0, 0.3, -0.1, 0.4, 0.0],
            vec![-0.3, 0.6, 0.0, 0.2, -0.1, 0.0, 0.5, -0.2],
            vec![0.1, 0.0, -0.4, 0.7, 0.0, 0.3, -0.2, 0.1],
        ];
        let output = block.forward(&inputs);
        assert_eq!(output.len(), 3);
        for (i, v) in output.iter().enumerate() {
            for (j, &val) in v.iter().enumerate() {
                assert!(!val.is_nan(), "output[{}][{}] is NaN", i, j);
                assert!(!val.is_infinite(), "output[{}][{}] is infinite", i, j);
            }
        }
    }

    /// Test 4: residual connection changes output (output != input).
    #[test]
    fn residual_changes_output() {
        let block = TinyTransformerBlock::new(4, 8);
        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let output = block.forward(&inputs);
        // Output should differ from input at every position
        for (i, (inp, out)) in inputs.iter().zip(output.iter()).enumerate() {
            let diff: f32 = inp.iter().zip(out).map(|(a, b)| (a - b).abs()).sum();
            assert!(
                diff > 1e-6,
                "position {} output equals input (no residual effect)",
                i
            );
        }
    }

    /// Test 5: causal property — changing a future token does not affect
    /// earlier output positions.
    #[test]
    fn causal_property() {
        let block = TinyTransformerBlock::new(8, 16);

        let inputs_a = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let mut inputs_b = inputs_a.clone();
        // Change token 2 (position 2)
        inputs_b[2] = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let out_a = block.forward(&inputs_a);
        let out_b = block.forward(&inputs_b);

        // Positions 0 and 1 should be very close
        let eps = 1e-4;
        for pos in 0..=1 {
            let diff: f32 = out_a[pos]
                .iter()
                .zip(out_b[pos].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                diff < eps,
                "position {} differs too much (diff={}) — causal property violated",
                pos,
                diff
            );
        }
        // Position 2 may differ (future token changed)
        let diff2: f32 = out_a[2]
            .iter()
            .zip(out_b[2].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff2 > 1e-6,
            "position 2 should be allowed to differ (future token changed)"
        );
    }
}
