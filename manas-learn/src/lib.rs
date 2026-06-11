pub mod backprop;
pub mod decoder;
pub mod embedder;
pub mod tokenizer;
pub mod trainer;

pub use backprop::{mse_loss, compute_gradients, compute_output_gradient, ForwardCache, NeuronGradients};
pub use decoder::{decode, DecodeResult};
pub use embedder::Embedder;
pub use tokenizer::Tokenizer;
pub use trainer::{Trainer, LearnReport, TrainerSnapshot, detect_freshness_category, DEFAULT_EMBED_DIM, DEFAULT_GROWTH_THRESHOLD, DEFAULT_LEARNING_RATE};

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::Network;

    #[test]
    fn tokenizer_grows_vocab() {
        let mut tok = Tokenizer::new();
        let ids = tok.encode("hello world hello");
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], ids[2]);
        assert_eq!(tok.token_count(), 2);
    }

    #[test]
    fn embedder_averages() {
        let mut embedder = Embedder::new(8);
        embedder.embed_or_init(0);
        embedder.embed_or_init(1);
        let avg = embedder.average_embed(&[0, 1]);
        assert_eq!(avg.len(), 8);
    }

    #[test]
    fn mse_loss_basic() {
        let p = vec![1.0, 2.0, 3.0];
        let t = vec![1.0, 2.0, 3.0];
        assert!((mse_loss(&p, &t) - 0.0).abs() < 1e-6);

        let p2 = vec![0.0, 0.0, 0.0];
        let t2 = vec![1.0, 1.0, 1.0];
        assert!((mse_loss(&p2, &t2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn end_to_end_learn() {
        let mut trainer = Trainer::new_with_params(16, 0.01, 0.5);
        let mut network = Network::new();

        for _ in 0..5 {
            let report = trainer.learn(&mut network, "rust is a systems programming language").unwrap();
            assert!(report.tokens_learned > 0);
        }

        assert!(network.total_neurons > 0);
        assert!(trainer.embedder.embedding_count() > 0);
        assert!(trainer.tokenizer.token_count() > 0);
    }
}
