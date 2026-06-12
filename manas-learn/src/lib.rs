pub mod backprop;
pub mod decoder;
pub mod embedder;
pub mod tokenizer;
pub mod trainer;

pub use backprop::{
    ForwardCache, NeuronGradients, compute_gradients, compute_output_gradient, mse_loss,
};
pub use decoder::{DecodeResult, decode};
pub use embedder::Embedder;
pub use tokenizer::Tokenizer;
pub use trainer::{
    DEFAULT_EMBED_DIM, DEFAULT_GROWTH_THRESHOLD, DEFAULT_LEARNING_RATE, LearnReport, Trainer,
    TrainerSnapshot, detect_freshness_category,
};

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::{Network, Source};

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
            let report = trainer
                .learn(&mut network, "rust is a systems programming language")
                .unwrap();
            assert!(report.tokens_learned > 0);
        }

        assert!(network.total_neurons > 0);
        assert!(trainer.embedder.embedding_count() > 0);
        assert!(trainer.tokenizer.token_count() > 0);
    }

    #[test]
    fn source_not_overwritten_by_learn() {
        let mut trainer = Trainer::new_with_params(16, 0.01, 0.5);
        let mut network = Network::new();

        // First learn: all neurons get src=RawText
        trainer.source = Source::RawText;
        trainer
            .learn(&mut network, "rust is safe and fast")
            .unwrap();
        assert!(network.total_neurons > 0);

        let raw_count = network
            .layers
            .iter()
            .flat_map(|l| &l.neurons)
            .filter(|n| matches!(n.source, Source::RawText))
            .count();
        assert_eq!(raw_count, network.total_neurons as usize);

        // Second learn with different source: must NOT overwrite existing neurons
        trainer.source = Source::Internet {
            url: "https://example.com".into(),
        };
        trainer
            .learn(&mut network, "some more content here")
            .unwrap();

        let raw_after = network
            .layers
            .iter()
            .flat_map(|l| &l.neurons)
            .filter(|n| matches!(n.source, Source::RawText))
            .count();
        let url_after = network
            .layers
            .iter()
            .flat_map(|l| &l.neurons)
            .filter(
                |n| matches!(&n.source, Source::Internet { url } if url == "https://example.com"),
            )
            .count();

        assert_eq!(raw_after, network.total_neurons as usize - url_after);
        assert!(raw_after > 0, "raw-text neurons were overwritten");
    }

    #[test]
    fn ensure_source_neuron_grows_for_new_file() {
        let mut trainer = Trainer::new_with_params(16, 0.01, 0.5);
        let mut network = Network::new();

        // Need at least one layer to grow into
        trainer.source = Source::RawText;
        trainer
            .learn(&mut network, "seed content to create layers")
            .unwrap();
        let before = network.total_neurons;
        assert!(before > 0);

        // Now request a source neuron for a new file
        trainer.source = Source::LocalFile {
            path: "/tmp/test.md".into(),
        };
        let grown = trainer.ensure_source_neuron(&mut network).unwrap();
        assert!(grown, "should have grown a source neuron");

        let after = network.total_neurons;
        assert_eq!(after, before + 1, "exactly 1 neuron should be added");

        // Verify the new neuron has the correct source
        let file_neurons: Vec<_> = network
            .layers
            .iter()
            .flat_map(|l| &l.neurons)
            .filter(|n| matches!(&n.source, Source::LocalFile { path } if path == "/tmp/test.md"))
            .collect();
        assert_eq!(file_neurons.len(), 1);
    }

    #[test]
    fn ensure_source_neuron_bounded() {
        let mut trainer = Trainer::new_with_params(16, 0.01, 0.5);
        let mut network = Network::new();

        trainer.source = Source::RawText;
        trainer.learn(&mut network, "seed content").unwrap();
        let before = network.total_neurons;

        // Call grow multiple times for the same file
        trainer.source = Source::LocalFile {
            path: "/tmp/test.md".into(),
        };
        let g1 = trainer.ensure_source_neuron(&mut network).unwrap();
        let g2 = trainer.ensure_source_neuron(&mut network).unwrap();
        let g3 = trainer.ensure_source_neuron(&mut network).unwrap();

        assert!(g1, "first call should grow");
        assert!(!g2, "second call should NOT grow (duplicate)");
        assert!(!g3, "third call should NOT grow (duplicate)");
        assert_eq!(network.total_neurons, before + 1);
    }

    #[test]
    fn ensure_source_neuron_skips_raw_text() {
        let mut trainer = Trainer::new_with_params(16, 0.01, 0.5);
        let mut network = Network::new();

        trainer.source = Source::RawText;
        trainer.learn(&mut network, "seed content").unwrap();
        let before = network.total_neurons;

        // Ensure source neuron with RawText should do nothing
        let grown = trainer.ensure_source_neuron(&mut network).unwrap();
        assert!(!grown, "RawText should not trigger source-aware growth");
        assert_eq!(network.total_neurons, before);
    }
}
