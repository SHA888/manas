use crate::format;
use crate::integrity;
use manas_core::{Activation, Layer, ManasError, Network};
use std::collections::HashMap;

pub fn write_to_path(network: &Network, path: &std::path::Path) -> Result<(), ManasError> {
    let bytes = build_bytes(network, &[], &HashMap::new());
    std::fs::write(path, &bytes).map_err(|e| ManasError::FileReadError {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(())
}

#[allow(dead_code)]
pub fn write_to_path_with_archive(
    network: &Network,
    archived: &[manas_core::Neuron],
    path: &std::path::Path,
) -> Result<(), ManasError> {
    let bytes = build_bytes(network, archived, &HashMap::new());
    std::fs::write(path, &bytes).map_err(|e| ManasError::FileReadError {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(())
}

pub fn write_to_path_with_vocab(
    network: &Network,
    vocab: &HashMap<u32, (String, Vec<f32>)>,
    path: &std::path::Path,
) -> Result<(), ManasError> {
    let bytes = build_bytes(network, &[], vocab);
    std::fs::write(path, &bytes).map_err(|e| ManasError::FileReadError {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(())
}

fn build_bytes(
    network: &Network,
    archived: &[manas_core::Neuron],
    vocab: &HashMap<u32, (String, Vec<f32>)>,
) -> Vec<u8> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut buf = Vec::new();

    let header = format::BrainHeader {
        magic: *format::MAGIC,
        version: network.version,
        created_at: network.created_at,
        last_modified: now,
        total_neurons: network.total_neurons,
        total_layers: network.layers.len() as u32,
        vocab_size: vocab.len() as u32,
        total_texts_learned: network.total_texts_learned,
        flags: if archived.is_empty() { 0 } else { 1 },
        checksum_offset: 0,
    };
    format::write_header(&mut buf, &header);

    buf.extend_from_slice(&(vocab.len() as u32).to_le_bytes());
    let mut sorted: Vec<(&u32, &(String, Vec<f32>))> = vocab.iter().collect();
    sorted.sort_by_key(|(id, _)| **id);
    for (id, (token, embedding)) in &sorted {
        buf.extend_from_slice(&id.to_le_bytes());
        let token_bytes = token.as_bytes();
        buf.push(token_bytes.len() as u8);
        buf.extend_from_slice(token_bytes);
        let embed_dim = embedding.len() as u16;
        buf.extend_from_slice(&embed_dim.to_le_bytes());
        for v in embedding {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    let layer_index_offset = buf.len() as u64;

    let mut layer_offsets: Vec<u64> = Vec::new();
    let mut neuron_counts: Vec<u32> = Vec::new();
    let mut current_offset = layer_index_offset + network.layers.len() as u64 * 16;
    for layer in &network.layers {
        layer_offsets.push(current_offset);
        neuron_counts.push(layer.neurons.len() as u32);
        current_offset += 9;
        for neuron in &layer.neurons {
            current_offset += format::neuron_binary_size(neuron) as u64;
        }
    }

    for (i, layer) in network.layers.iter().enumerate() {
        buf.extend_from_slice(&layer.id.to_le_bytes());
        buf.extend_from_slice(&layer_offsets[i].to_le_bytes());
        buf.extend_from_slice(&neuron_counts[i].to_le_bytes());
    }

    for layer in &network.layers {
        format::write_layer_block(&mut buf, layer);
    }

    let archive_layer = Layer {
        id: u32::MAX,
        neurons: archived.to_vec(),
        activation: Activation::Linear,
    };
    format::write_layer_block(&mut buf, &archive_layer);

    let checksum_offset = buf.len() as u32;
    buf[48..52].copy_from_slice(&checksum_offset.to_le_bytes());

    let crc = integrity::compute_crc32(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    buf
}
