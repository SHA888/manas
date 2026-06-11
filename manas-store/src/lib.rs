mod format;
mod integrity;
mod patcher;
mod reader;
mod writer;

use manas_core::{ManasError, Network, Neuron};
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::Activation;
    use std::path::Path;

    fn test_network() -> Network {
        let mut net = Network::new();
        net.grow_layer(3, 4);
        net.grow_neuron(0, 4).unwrap();
        net
    }

    #[test]
    fn round_trip() {
        let path = Path::new("/tmp/test_manas.manas");
        let brain = ManasBrain::new(path);

        let original = test_network();
        brain.save(&original).unwrap();
        assert!(brain.verify().unwrap());

        let loaded = brain.load().unwrap();
        assert_eq!(loaded.total_neurons, original.total_neurons);
        assert_eq!(loaded.layers.len(), original.layers.len());
        assert_eq!(loaded.created_at, original.created_at);

        for (ol, ll) in original.layers.iter().zip(loaded.layers.iter()) {
            assert_eq!(ol.neurons.len(), ll.neurons.len());
            for (on, ln) in ol.neurons.iter().zip(ll.neurons.iter()) {
                assert_eq!(on.id, ln.id);
                assert_eq!(on.weights.len(), ln.weights.len());
                assert_eq!(on.bias.to_bits(), ln.bias.to_bits());
                assert_eq!(on.activation as u8, ln.activation as u8);
            }
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn verify_corrupt() {
        let path = Path::new("/tmp/test_corrupt.manas");
        let brain = ManasBrain::new(path);

        brain.save(&test_network()).unwrap();

        let mut data = std::fs::read(path).unwrap();
        data[30] ^= 0xFF;
        std::fs::write(path, &data).unwrap();

        assert_eq!(brain.verify().unwrap(), false);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn inspect_stats() {
        let path = Path::new("/tmp/test_inspect.manas");
        let brain = ManasBrain::new(path);

        let net = test_network();
        brain.save(&net).unwrap();

        let stats = brain.inspect().unwrap();
        assert_eq!(stats.neuron_count, net.total_neurons);
        assert_eq!(stats.layer_count, net.layers.len() as u32);
        assert!(stats.brain_size > 0);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn archive_round_trip() {
        use manas_core::Neuron;
        let path = Path::new("/tmp/test_archive.manas");
        let brain = ManasBrain::new(path);

        let net = test_network();
        let archived = vec![Neuron::new(999, 4, Activation::ReLU)];

        crate::writer::write_to_path_with_archive(&net, &archived, path).unwrap();

        let loaded_archive = brain.load_archive().unwrap();
        assert_eq!(loaded_archive.len(), 1);
        assert_eq!(loaded_archive[0].id, 999);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn archive_no_flag_returns_empty() {
        let path = Path::new("/tmp/test_noarchive.manas");
        let brain = ManasBrain::new(path);

        let net = test_network();
        brain.save(&net).unwrap();

        let loaded_archive = brain.load_archive().unwrap();
        assert!(loaded_archive.is_empty());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn append_neuron_round_trip() {
        use manas_core::Neuron;
        let path = Path::new("/tmp/test_append.manas");
        let brain = ManasBrain::new(path);

        let net = test_network();
        brain.save(&net).unwrap();
        let before = brain.inspect().unwrap();

        let new_n = Neuron::new(100, 4, Activation::ReLU);
        brain.append_neuron(0, &new_n).unwrap();

        assert!(brain.verify().unwrap());
        let stats = brain.inspect().unwrap();
        assert_eq!(stats.neuron_count, before.neuron_count + 1);
        assert_eq!(stats.layer_count, before.layer_count);

        let loaded = brain.load().unwrap();
        let found = loaded.all_neurons().iter().any(|(_, n)| n.id == 100);
        assert!(found, "appended neuron 100 not found in loaded network");

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn update_neuron_in_place() {
        let path = Path::new("/tmp/test_update.manas");
        let brain = ManasBrain::new(path);

        let net = test_network();
        brain.save(&net).unwrap();

        let neurons = net.all_neurons();
        let (_, old) = neurons[0];
        let mut updated = old.clone();
        updated.bias = 42.0;
        updated.last_activated = 999999;

        brain.update_neuron(old.id, &updated).unwrap();

        assert!(brain.verify().unwrap());
        let loaded = brain.load().unwrap();
        let found: Vec<&manas_core::Neuron> = loaded
            .all_neurons()
            .iter()
            .map(|(_, n)| *n)
            .filter(|n| n.id == old.id)
            .collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].bias.to_bits(), 42.0f32.to_bits());
        assert_eq!(found[0].last_activated, 999999);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn append_multiple_neurons() {
        use manas_core::Neuron;
        let path = Path::new("/tmp/test_append_multi.manas");
        let brain = ManasBrain::new(path);

        let net = test_network();
        brain.save(&net).unwrap();
        let before = brain.inspect().unwrap();

        for i in 0..5 {
            let n = Neuron::new(200 + i, 4, Activation::ReLU);
            brain.append_neuron(0, &n).unwrap();
        }

        let stats = brain.inspect().unwrap();
        assert_eq!(stats.neuron_count, before.neuron_count + 5);

        let loaded = brain.load().unwrap();
        let ids: Vec<u64> = loaded.all_neurons().iter().map(|(_, n)| n.id).collect();
        for i in 0..5 {
            assert!(ids.contains(&(200 + i)), "neuron {} not found", 200 + i);
        }

        std::fs::remove_file(path).ok();
    }
}

pub struct BrainStats {
    pub neuron_count: u64,
    pub layer_count: u32,
    pub vocab_size: u32,
    pub total_texts_learned: u64,
    pub brain_size: u64,
    pub last_modified: u64,
    pub file_path: String,
}

pub struct ManasBrain {
    pub path: PathBuf,
}

impl ManasBrain {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        ManasBrain { path: path.into() }
    }

    pub fn load(&self) -> Result<Network, ManasError> {
        reader::read_from_path(&self.path)
    }

    pub fn save(&self, network: &Network) -> Result<(), ManasError> {
        writer::write_to_path(network, &self.path)
    }

    pub fn append_neuron(&self, layer_id: u32, neuron: &Neuron) -> Result<(), ManasError> {
        patcher::append_neuron_to_file(&self.path, layer_id, neuron)
    }

    pub fn update_neuron(&self, neuron_id: u64, neuron: &Neuron) -> Result<(), ManasError> {
        patcher::update_neuron_in_file(&self.path, neuron_id, neuron)
    }

    pub fn save_with_vocab(
        &self,
        network: &Network,
        vocab: &HashMap<u32, (String, Vec<f32>)>,
    ) -> Result<(), ManasError> {
        writer::write_to_path_with_vocab(network, vocab, &self.path)
    }

    pub fn load_vocab(&self) -> Result<HashMap<u32, (String, Vec<f32>)>, ManasError> {
        let data = std::fs::read(&self.path).map_err(|e| ManasError::FileReadError {
            path: self.path.clone(),
            source: e,
        })?;
        reader::read_vocab_from_bytes(&data)
    }

    pub fn verify(&self) -> Result<bool, ManasError> {
        let data = std::fs::read(&self.path).map_err(|e| ManasError::FileReadError {
            path: self.path.clone(),
            source: e,
        })?;
        match integrity::verify_checksum(&data) {
            Ok(_) => Ok(true),
            Err(ManasError::ChecksumMismatch) => Ok(false),
            Err(e) => Err(e),
        }
    }

    pub fn load_archive(&self) -> Result<Vec<Neuron>, ManasError> {
        let mut file = std::fs::File::open(&self.path).map_err(|e| ManasError::FileReadError {
            path: self.path.clone(),
            source: e,
        })?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| ManasError::FileReadError {
                path: self.path.clone(),
                source: e,
            })?;
        reader::read_archived_neurons(&data)
    }

    pub fn inspect(&self) -> Result<BrainStats, ManasError> {
        let data = std::fs::read(&self.path).map_err(|e| ManasError::FileReadError {
            path: self.path.clone(),
            source: e,
        })?;

        let header = format::read_header(&data).ok_or_else(|| ManasError::CorruptFile {
            path: self.path.clone(),
            reason: "invalid header".into(),
        })?;

        Ok(BrainStats {
            neuron_count: header.total_neurons,
            layer_count: header.total_layers,
            vocab_size: header.vocab_size,
            total_texts_learned: header.total_texts_learned,
            brain_size: data.len() as u64,
            last_modified: header.last_modified,
            file_path: self.path.display().to_string(),
        })
    }
}
