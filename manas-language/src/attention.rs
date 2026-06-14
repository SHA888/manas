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

/// Cached intermediate values from a causal self-attention forward pass.
///
/// This is forward-only state for future training work.  It must not change
/// inference behavior.
#[derive(Clone, Debug)]
pub struct AttentionForwardCache {
    pub qs: Vec<Vec<f32>>,
    pub ks: Vec<Vec<f32>>,
    pub vs: Vec<Vec<f32>>,
    pub attention_weights: Vec<Vec<f32>>,
    pub weighted_values: Vec<Vec<f32>>,
}

/// Result of one attention output-projection training step.
#[derive(Clone, Debug)]
pub struct AttentionTrainStepReport {
    pub applied: bool,
    pub clipped: bool,
    pub invalid: bool,
    pub grad_norm: f32,
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
        self.forward_with_cache(inputs).0
    }

    /// Forward pass with causal masking and reusable intermediate cache.
    ///
    /// Returns the same output as `forward()` plus per-position Q/K/V,
    /// full-length causal attention weights, and weighted value vectors.
    pub fn forward_with_cache(
        &self,
        inputs: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, AttentionForwardCache) {
        let seq_len = inputs.len();
        if seq_len == 0 || self.embed_dim == 0 {
            return (
                Vec::new(),
                AttentionForwardCache {
                    qs: Vec::new(),
                    ks: Vec::new(),
                    vs: Vec::new(),
                    attention_weights: Vec::new(),
                    weighted_values: Vec::new(),
                },
            );
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
        let mut attention_weights: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut weighted_values: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        for (i, qi) in qs.iter().enumerate() {
            // 2. Compute scaled dot-product scores for positions 0..=i (causal)
            let mut scores = Vec::with_capacity(i + 1);
            for kj in ks.iter().take(i + 1) {
                scores.push(dot(qi, kj) * inv_sqrt_d);
            }

            // 3. Softmax over allowed positions
            let attn_weights = softmax(&scores);
            let mut full_weights = vec![0.0; seq_len];
            for (j, &weight) in attn_weights.iter().enumerate() {
                full_weights[j] = weight;
            }

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
            attention_weights.push(full_weights);
            weighted_values.push(out);
        }

        (
            outputs,
            AttentionForwardCache {
                qs,
                ks,
                vs,
                attention_weights,
                weighted_values,
            },
        )
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

    /// Train only the output projection `w_o`.
    ///
    /// `context` is the cached weighted value vector before `w_o`, and
    /// `grad_output` is the gradient flowing into the attention output.
    pub fn train_output_projection_step(
        &mut self,
        context: &[f32],
        grad_output: &[f32],
        learning_rate: f32,
        max_grad_norm: f32,
    ) -> AttentionTrainStepReport {
        let d = self.embed_dim;
        if d == 0
            || context.len() != d
            || grad_output.len() != d
            || self.w_o.len() != d * d
            || !learning_rate.is_finite()
            || context.iter().any(|&v| !v.is_finite())
            || grad_output.iter().any(|&v| !v.is_finite())
        {
            return AttentionTrainStepReport {
                applied: false,
                clipped: false,
                invalid: true,
                grad_norm: f32::NAN,
            };
        }

        let mut grad_w_o = vec![0.0; d * d];
        for (r, &go) in grad_output.iter().enumerate() {
            let base = r * d;
            for (c, &ctx) in context.iter().enumerate() {
                grad_w_o[base + c] = go * ctx;
            }
        }

        let grad_norm = vector_norm(&grad_w_o);
        if !grad_norm.is_finite() || grad_w_o.iter().any(|&g| !g.is_finite()) {
            return AttentionTrainStepReport {
                applied: false,
                clipped: false,
                invalid: true,
                grad_norm,
            };
        }

        let clipped = clip_vector_by_norm(&mut grad_w_o, max_grad_norm);
        let mut next_w_o = self.w_o.clone();
        for (w, &g) in next_w_o.iter_mut().zip(grad_w_o.iter()) {
            *w -= learning_rate * g;
        }
        if next_w_o.iter().any(|&w| !w.is_finite()) {
            return AttentionTrainStepReport {
                applied: false,
                clipped,
                invalid: true,
                grad_norm,
            };
        }

        self.w_o = next_w_o;
        AttentionTrainStepReport {
            applied: true,
            clipped,
            invalid: false,
            grad_norm,
        }
    }

    /// Train only the value projection `w_v`.
    ///
    /// `attention_weights_last` is the final-position causal attention row, and
    /// `grad_context_last` is the gradient flowing into the weighted value sum.
    pub fn train_value_projection_step(
        &mut self,
        inputs: &[Vec<f32>],
        attention_weights_last: &[f32],
        grad_context_last: &[f32],
        learning_rate: f32,
        max_grad_norm: f32,
    ) -> AttentionTrainStepReport {
        let d = self.embed_dim;
        if d == 0
            || inputs.is_empty()
            || attention_weights_last.len() != inputs.len()
            || grad_context_last.len() != d
            || self.w_v.len() != d * d
            || !learning_rate.is_finite()
            || inputs
                .iter()
                .any(|input| input.len() != d || input.iter().any(|&v| !v.is_finite()))
            || attention_weights_last.iter().any(|&v| !v.is_finite())
            || grad_context_last.iter().any(|&v| !v.is_finite())
        {
            return AttentionTrainStepReport {
                applied: false,
                clipped: false,
                invalid: true,
                grad_norm: f32::NAN,
            };
        }

        let mut grad_w_v = vec![0.0; d * d];
        for (input, &weight) in inputs.iter().zip(attention_weights_last.iter()) {
            if weight.abs() < 1e-10 {
                continue;
            }
            for (r, &grad_context) in grad_context_last.iter().enumerate() {
                let grad_v = weight * grad_context;
                if grad_v.abs() < 1e-10 {
                    continue;
                }
                let base = r * d;
                for (c, &input_value) in input.iter().enumerate() {
                    grad_w_v[base + c] += grad_v * input_value;
                }
            }
        }

        let grad_norm = vector_norm(&grad_w_v);
        if !grad_norm.is_finite() || grad_w_v.iter().any(|&g| !g.is_finite()) {
            return AttentionTrainStepReport {
                applied: false,
                clipped: false,
                invalid: true,
                grad_norm,
            };
        }

        let clipped = clip_vector_by_norm(&mut grad_w_v, max_grad_norm);
        let mut next_w_v = self.w_v.clone();
        for (w, &g) in next_w_v.iter_mut().zip(grad_w_v.iter()) {
            *w -= learning_rate * g;
        }
        if next_w_v.iter().any(|&w| !w.is_finite()) {
            return AttentionTrainStepReport {
                applied: false,
                clipped,
                invalid: true,
                grad_norm,
            };
        }

        self.w_v = next_w_v;
        AttentionTrainStepReport {
            applied: true,
            clipped,
            invalid: false,
            grad_norm,
        }
    }
}

fn vector_norm(values: &[f32]) -> f32 {
    values.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

fn clip_vector_by_norm(values: &mut [f32], max_norm: f32) -> bool {
    if max_norm <= 0.0 || values.is_empty() {
        return false;
    }
    let norm = vector_norm(values);
    if norm.is_finite() && norm > max_norm {
        let scale = max_norm / norm;
        for value in values {
            *value *= scale;
        }
        true
    } else {
        false
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

    #[test]
    fn forward_with_cache_matches_forward() {
        let attn = CausalSelfAttention::new(6);
        let inputs = vec![
            vec![0.5, -0.2, 0.1, 0.0, 0.3, -0.1],
            vec![-0.3, 0.6, 0.0, 0.2, -0.1, 0.0],
            vec![0.1, 0.0, -0.4, 0.7, 0.0, 0.3],
        ];

        let direct = attn.forward(&inputs);
        let (cached, _cache) = attn.forward_with_cache(&inputs);

        assert_eq!(cached.len(), direct.len());
        for (a, b) in cached.iter().zip(direct.iter()) {
            assert_eq!(a.len(), b.len());
            for (&x, &y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-7, "cached forward mismatch");
            }
        }
    }

    #[test]
    fn forward_cache_shapes_are_correct() {
        let attn = CausalSelfAttention::new(4);
        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let (_output, cache) = attn.forward_with_cache(&inputs);

        assert_eq!(cache.qs.len(), 3);
        assert_eq!(cache.ks.len(), 3);
        assert_eq!(cache.vs.len(), 3);
        assert_eq!(cache.attention_weights.len(), 3);
        assert_eq!(cache.weighted_values.len(), 3);

        for row in &cache.qs {
            assert_eq!(row.len(), 4);
        }
        for row in &cache.ks {
            assert_eq!(row.len(), 4);
        }
        for row in &cache.vs {
            assert_eq!(row.len(), 4);
        }
        for row in &cache.attention_weights {
            assert_eq!(row.len(), 3);
        }
        for row in &cache.weighted_values {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn forward_cache_preserves_causal_mask() {
        let attn = CausalSelfAttention::new(4);
        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let (_output, cache) = attn.forward_with_cache(&inputs);

        for (i, weights) in cache.attention_weights.iter().enumerate() {
            let sum: f32 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "attention row should sum to 1");
            for &future_weight in weights.iter().skip(i + 1) {
                assert_eq!(future_weight, 0.0, "future positions must stay masked");
            }
        }
    }

    #[test]
    fn output_projection_step_updates_only_w_o() {
        let mut attn = CausalSelfAttention::new(4);
        let q_before = attn.w_q.clone();
        let k_before = attn.w_k.clone();
        let v_before = attn.w_v.clone();
        let o_before = attn.w_o.clone();

        let report = attn.train_output_projection_step(
            &[1.0, 0.5, -0.25, 0.75],
            &[0.2, -0.1, 0.3, 0.4],
            0.05,
            10.0,
        );

        assert!(report.applied, "w_o update should be applied");
        assert!(!report.invalid, "w_o update should be valid");
        assert_eq!(attn.w_q, q_before, "w_q must remain frozen");
        assert_eq!(attn.w_k, k_before, "w_k must remain frozen");
        assert_eq!(attn.w_v, v_before, "w_v must remain frozen");
        assert_ne!(attn.w_o, o_before, "w_o should change");
    }

    #[test]
    fn output_projection_step_reports_clipping() {
        let mut attn = CausalSelfAttention::new(2);
        let before = attn.w_o.clone();

        let report = attn.train_output_projection_step(&[10.0, 10.0], &[10.0, -10.0], 0.01, 1.0);

        assert!(report.applied, "clipped update should still apply");
        assert!(report.clipped, "large gradient should be clipped");
        assert!(report.grad_norm > 1.0, "pre-clip norm should be tracked");
        assert_ne!(attn.w_o, before, "w_o should change after clipped update");
    }

    #[test]
    fn output_projection_step_rejects_non_finite_gradient() {
        let mut attn = CausalSelfAttention::new(2);
        let before = attn.w_o.clone();

        let report = attn.train_output_projection_step(&[1.0, 0.0], &[f32::NAN, 1.0], 0.01, 1.0);

        assert!(!report.applied, "invalid update should not apply");
        assert!(report.invalid, "non-finite gradient should be invalid");
        assert_eq!(attn.w_o, before, "w_o must not change on invalid update");
    }

    #[test]
    fn value_projection_step_updates_only_w_v() {
        let mut attn = CausalSelfAttention::new(3);
        let q_before = attn.w_q.clone();
        let k_before = attn.w_k.clone();
        let v_before = attn.w_v.clone();
        let o_before = attn.w_o.clone();
        let inputs = vec![
            vec![1.0, 0.0, 0.5],
            vec![0.0, -1.0, 0.25],
            vec![0.5, 0.5, 1.0],
        ];

        let report = attn.train_value_projection_step(
            &inputs,
            &[0.2, 0.3, 0.5],
            &[0.4, -0.2, 0.1],
            0.05,
            10.0,
        );

        assert!(report.applied, "w_v update should be applied");
        assert!(!report.invalid, "w_v update should be valid");
        assert_eq!(attn.w_q, q_before, "w_q must remain frozen");
        assert_eq!(attn.w_k, k_before, "w_k must remain frozen");
        assert_eq!(attn.w_o, o_before, "w_o must not change");
        assert_ne!(attn.w_v, v_before, "w_v should change");
    }

    #[test]
    fn value_projection_step_reports_clipping() {
        let mut attn = CausalSelfAttention::new(2);
        let before = attn.w_v.clone();
        let inputs = vec![vec![10.0, 10.0], vec![5.0, -5.0]];

        let report =
            attn.train_value_projection_step(&inputs, &[0.5, 0.5], &[10.0, -10.0], 0.01, 1.0);

        assert!(report.applied, "clipped update should still apply");
        assert!(report.clipped, "large gradient should be clipped");
        assert!(report.grad_norm > 1.0, "pre-clip norm should be tracked");
        assert_ne!(attn.w_v, before, "w_v should change after clipped update");
    }

    #[test]
    fn value_projection_step_rejects_non_finite_gradient() {
        let mut attn = CausalSelfAttention::new(2);
        let before = attn.w_v.clone();
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let report = attn.train_value_projection_step(
            &inputs,
            &[0.5, 0.5],
            &[f32::INFINITY, 1.0],
            0.01,
            1.0,
        );

        assert!(!report.applied, "invalid update should not apply");
        assert!(report.invalid, "non-finite gradient should be invalid");
        assert_eq!(attn.w_v, before, "w_v must not change on invalid update");
    }
}
