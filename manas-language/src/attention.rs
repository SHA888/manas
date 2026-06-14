// ─── Helpers ────────────────────────────────────────────────────

pub(crate) fn mat_vec_mul(matrix: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; rows];
    for r in 0..rows {
        let mut sum = 0.0;
        for c in 0..cols {
            sum += matrix[r * cols + c] * input[c];
        }
        out[r] = sum;
    }
    out
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().copied().reduce(f32::max).unwrap_or(0.0);
    let mut exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

// ─── CausalSelfAttention ─────────────────────────────────────────

/// Single-head causal self-attention module.
///
/// Shapes (row-major):
///   w_q: embed_dim × embed_dim
///   w_k: embed_dim × embed_dim
///   w_v: embed_dim × embed_dim
///   w_o: embed_dim × embed_dim
#[derive(Clone)]
pub struct CausalSelfAttention {
    pub embed_dim: usize,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
}

impl CausalSelfAttention {
    /// Create a new attention module with small random weights.
    ///
    /// Weights are initialized with `N(0, 0.02)` scaled by `1/sqrt(embed_dim)`.
    pub fn new(embed_dim: usize) -> Self {
        let n = embed_dim * embed_dim;
        // Deterministic pseudo-random init for reproducibility
        let mut rng = SimpleRng::new(42);
        let scale = (1.0 / (embed_dim as f32).sqrt()).min(0.02);
        CausalSelfAttention {
            embed_dim,
            w_q: random_vec(&mut rng, n, scale),
            w_k: random_vec(&mut rng, n, scale),
            w_v: random_vec(&mut rng, n, scale),
            w_o: random_vec(&mut rng, n, scale),
        }
    }

    /// Forward pass with causal masking.
    ///
    /// `inputs`: sequence of token embeddings, shape `seq_len × embed_dim`.
    /// Returns output vectors of the same shape.
    pub fn forward(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = inputs.len();
        if seq_len == 0 || self.embed_dim == 0 {
            return Vec::new();
        }

        let d = self.embed_dim;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();

        // 1. Compute Q, K, V
        let mut qs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut ks: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut vs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        for input in inputs {
            qs.push(mat_vec_mul(&self.w_q, d, d, input));
            ks.push(mat_vec_mul(&self.w_k, d, d, input));
            vs.push(mat_vec_mul(&self.w_v, d, d, input));
        }

        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        for (i, qi) in qs.iter().enumerate() {
            // 2. Compute scaled dot-product scores for positions 0..=i (causal)
            let mut scores = Vec::with_capacity(i + 1);
            for kj in ks.iter().take(i + 1) {
                scores.push(dot(qi, kj) * inv_sqrt_d);
            }

            // 3. Softmax over allowed positions
            let attn_weights = softmax(&scores);

            // 4. Weighted sum of V
            let mut out = vec![0.0; d];
            for (j, weight) in attn_weights.iter().enumerate() {
                if *weight > 0.0 {
                    for k in 0..d {
                        out[k] += weight * vs[j][k];
                    }
                }
            }

            // 5. Output projection
            outputs.push(mat_vec_mul(&self.w_o, d, d, &out));
        }

        outputs
    }

    /// Expose attention weights for a single position (for testing).
    ///
    /// Returns a vector of length `inputs.len()`, with 0.0 at future positions.
    pub fn attention_weights_for_position(&self, inputs: &[Vec<f32>], position: usize) -> Vec<f32> {
        if position >= inputs.len() || self.embed_dim == 0 {
            return vec![0.0; inputs.len()];
        }

        let d = self.embed_dim;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();

        let q = mat_vec_mul(&self.w_q, d, d, &inputs[position]);

        let mut scores = Vec::with_capacity(position + 1);
        for input_j in inputs.iter().take(position + 1) {
            let k = mat_vec_mul(&self.w_k, d, d, input_j);
            scores.push(dot(&q, &k) * inv_sqrt_d);
        }

        let allowed = softmax(&scores);

        let mut full = vec![0.0; inputs.len()];
        for (j, w) in allowed.iter().enumerate() {
            full[j] = *w;
        }
        full
    }
}

// ─── Deterministic pseudo-random helpers ─────────────────────────

pub(crate) struct SimpleRng {
    seed: u64,
}

impl SimpleRng {
    pub(crate) fn new(seed: u64) -> Self {
        SimpleRng { seed }
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        self.seed = self
            .seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.seed >> 33) as u32;
        u as f32 / u32::MAX as f32
    }
}

pub(crate) fn random_vec(rng: &mut SimpleRng, n: usize, scale: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let u1 = rng.next_f32();
        if u1 < 1e-10 {
            v.push(0.0);
            continue;
        }
        let u2 = rng.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let z = r * (2.0 * std::f32::consts::PI * u2).cos();
        v.push(z * scale);
    }
    v
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: shape test — forward output should match input dimensions.
    #[test]
    fn forward_shape() {
        let attn = CausalSelfAttention::new(4);
        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let output = attn.forward(&inputs);
        assert_eq!(output.len(), 3, "should have 3 output vectors");
        for v in &output {
            assert_eq!(v.len(), 4, "each output vector should have embed_dim=4");
        }
    }

    /// Test 2: causal mask behavior.
    ///
    /// Position 0 must only attend to token 0 (future tokens get 0 weight).
    /// Position 1 can attend to 0 and 1.
    /// Position 2 can attend to 0, 1, 2.
    #[test]
    fn causal_mask() {
        let attn = CausalSelfAttention::new(4);
        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        // Position 0: only token 0 has non-zero attention
        let w0 = attn.attention_weights_for_position(&inputs, 0);
        assert_eq!(w0.len(), 3);
        assert!(w0[0] > 0.0, "position 0 should attend to token 0");
        assert_eq!(
            w0[1], 0.0,
            "position 0 should NOT attend to token 1 (future)"
        );
        assert_eq!(
            w0[2], 0.0,
            "position 0 should NOT attend to token 2 (future)"
        );

        // Position 1: tokens 0 and 1 can have non-zero attention
        let w1 = attn.attention_weights_for_position(&inputs, 1);
        assert_eq!(w1.len(), 3);
        assert!(w1[1] >= 0.0, "position 1 can attend to token 1");
        assert_eq!(
            w1[2], 0.0,
            "position 1 should NOT attend to token 2 (future)"
        );

        // Position 2: all three tokens can have non-zero attention
        let w2 = attn.attention_weights_for_position(&inputs, 2);
        assert_eq!(w2.len(), 3);
        // All weights should be non-negative (softmax output)
        for (j, &w) in w2.iter().enumerate() {
            assert!(
                w >= 0.0,
                "position 2 weight at {} should be >= 0, got {}",
                j,
                w
            );
        }
    }

    /// Test 3: empty input returns empty output without panic.
    #[test]
    fn empty_input_no_panic() {
        let attn = CausalSelfAttention::new(4);
        let output = attn.forward(&[]);
        assert!(output.is_empty(), "empty input should give empty output");
    }

    /// Test 4: deterministic dimensions — no NaN in output.
    #[test]
    fn no_nan_in_output() {
        let attn = CausalSelfAttention::new(8);
        let inputs = vec![
            vec![0.5, -0.2, 0.1, 0.0, 0.3, -0.1, 0.4, 0.0],
            vec![-0.3, 0.6, 0.0, 0.2, -0.1, 0.0, 0.5, -0.2],
            vec![0.1, 0.0, -0.4, 0.7, 0.0, 0.3, -0.2, 0.1],
            vec![0.0, 0.2, 0.3, -0.1, 0.5, -0.3, 0.0, 0.4],
        ];
        let output = attn.forward(&inputs);
        assert_eq!(output.len(), 4);
        for (i, v) in output.iter().enumerate() {
            assert_eq!(v.len(), 8, "output vector {} should have embed_dim=8", i);
            for (j, &val) in v.iter().enumerate() {
                assert!(!val.is_nan(), "output[{}][{}] is NaN", i, j);
            }
        }
    }
}
