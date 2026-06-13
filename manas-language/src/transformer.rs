use crate::attention::{CausalSelfAttention, SimpleRng, mat_vec_mul, random_vec};

// ─── Feed‑Forward ────────────────────────────────────────────────

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
}

// ─── Tiny Transformer Block ──────────────────────────────────────

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
