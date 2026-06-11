use std::collections::HashMap;
use std::io::Read;
use manas_core::{ManasError, Network, Neuron};
use crate::format;
use crate::integrity;

pub fn read_from_path(path: &std::path::Path) -> Result<Network, ManasError> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;

    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| ManasError::FileReadError {
            path: path.to_path_buf(),
            source: e,
        })?;

    read_from_bytes(&data)
}

pub fn read_from_bytes(data: &[u8]) -> Result<Network, ManasError> {
    integrity::verify_checksum(data)?;

    let header = format::read_header(data)
        .ok_or_else(|| ManasError::CorruptFile {
            path: std::path::PathBuf::new(),
            reason: "invalid or missing file header".into(),
        })?;

    let mut layers = Vec::with_capacity(header.total_layers as usize);

    let mut offset = format::HEADER_SIZE;

    let vocab_count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    for _ in 0..vocab_count {
        offset += 4;
        let token_len = data[offset] as usize;
        offset += 1 + token_len + 2;
        let embed_count = u16::from_le_bytes(data[offset - 2..offset].try_into().unwrap()) as usize;
        offset += embed_count * 4;
    }

    let layer_index_size = header.total_layers as usize * 16;
    offset += layer_index_size;

    for _ in 0..header.total_layers {
        let layer = format::read_layer_block(data, &mut offset)
            .ok_or_else(|| ManasError::CorruptFile {
                path: std::path::PathBuf::new(),
                reason: format!("failed to read layer block at offset {}", offset),
            })?;
        layers.push(layer);
    }

    Ok(Network {
        layers,
        total_neurons: header.total_neurons,
        created_at: header.created_at,
        version: header.version,
    })
}

pub fn read_vocab_from_bytes(data: &[u8]) -> Result<HashMap<u32, (String, Vec<f32>)>, ManasError> {
    let mut offset = format::HEADER_SIZE;
    let vocab_count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;

    let mut vocab = HashMap::with_capacity(vocab_count as usize);
    for _ in 0..vocab_count {
        if offset + 4 > data.len() { break; }
        let token_id = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if offset >= data.len() { break; }
        let token_len = data[offset] as usize;
        offset += 1;
        if offset + token_len > data.len() { break; }
        let token = String::from_utf8_lossy(&data[offset..offset + token_len]).to_string();
        offset += token_len;
        if offset + 2 > data.len() { break; }
        let embed_dim = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;
        if offset + embed_dim * 4 > data.len() { break; }
        let mut embedding = Vec::with_capacity(embed_dim);
        for _ in 0..embed_dim {
            let v = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            embedding.push(v);
            offset += 4;
        }
        vocab.insert(token_id, (token, embedding));
    }

    Ok(vocab)
}

pub fn read_archived_neurons(data: &[u8]) -> Result<Vec<Neuron>, ManasError> {
    let header = format::read_header(data)
        .ok_or_else(|| ManasError::CorruptFile {
            path: std::path::PathBuf::new(),
            reason: "invalid header".into(),
        })?;

    if header.flags & 1 == 0 {
        return Ok(Vec::new());
    }

    let mut offset = format::HEADER_SIZE;

    let vocab_count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;
    for _ in 0..vocab_count {
        offset += 4;
        let token_len = data[offset] as usize;
        offset += 1 + token_len + 2;
        let embed_count = u16::from_le_bytes(data[offset - 2..offset].try_into().unwrap()) as usize;
        offset += embed_count * 4;
    }

    let layer_index_size = header.total_layers as usize * 16;
    offset += layer_index_size;

    for _ in 0..header.total_layers {
        let _ = format::read_layer_block(data, &mut offset)
            .ok_or_else(|| ManasError::CorruptFile {
                path: std::path::PathBuf::new(),
                reason: format!("failed to read layer block at offset {}", offset),
            })?;
    }

    let archive_layer = format::read_layer_block(data, &mut offset)
        .ok_or_else(|| ManasError::CorruptFile {
            path: std::path::PathBuf::new(),
            reason: "failed to read archive block".into(),
        })?;

    Ok(archive_layer.neurons)
}
