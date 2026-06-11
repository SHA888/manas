use crate::format;
use crate::integrity;
use manas_core::{Activation, ManasError, Network, Neuron, ProtectionLevel, Source};
use std::collections::HashMap;
use std::io::Read;

pub struct LayerLocation {
    pub id: u32,
    pub neuron_count: u32,
    /// Byte offset where the layer block begins (9-byte header: id + count + activation)
    pub block_start: u64,
    /// Byte offset where the first neuron's data starts
    pub neuron_data_start: u64,
}

pub fn find_layer_locations(data: &[u8]) -> Result<Vec<LayerLocation>, ManasError> {
    let header = format::read_header(data).ok_or_else(|| ManasError::CorruptFile {
        path: std::path::PathBuf::new(),
        reason: "invalid header".into(),
    })?;

    let mut offset = format::HEADER_SIZE as u64;
    let vocab_count = u32::from_le_bytes(
        data[offset as usize..offset as usize + 4]
            .try_into()
            .unwrap(),
    );
    offset += 4;
    for _ in 0..vocab_count {
        offset += 4;
        let token_len = data[offset as usize] as usize;
        offset += 1 + token_len as u64 + 2;
        let embed_count = u16::from_le_bytes(
            data[offset as usize - 2..offset as usize]
                .try_into()
                .unwrap(),
        ) as usize;
        offset += (embed_count * 4) as u64;
    }

    let layer_index_start = offset;

    let mut locations = Vec::with_capacity(header.total_layers as usize);
    for i in 0..header.total_layers {
        let idx_off = (layer_index_start + i as u64 * 16) as usize;
        let layer_id = u32::from_le_bytes(data[idx_off..idx_off + 4].try_into().unwrap());
        let block_offset = u64::from_le_bytes(data[idx_off + 4..idx_off + 12].try_into().unwrap());
        let neuron_count = u32::from_le_bytes(data[idx_off + 12..idx_off + 16].try_into().unwrap());

        let neuron_data_start = block_offset + 9;
        locations.push(LayerLocation {
            id: layer_id,
            neuron_count,
            block_start: block_offset,
            neuron_data_start,
        });

        let mut npos = neuron_data_start;
        for _ in 0..neuron_count {
            let nsize = compute_neuron_size_bytes(data, npos)?;
            npos += nsize;
        }
    }

    Ok(locations)
}

pub fn find_neuron_offset(data: &[u8], neuron_id: u64) -> Result<u64, ManasError> {
    let layers = find_layer_locations(data)?;
    for layer in &layers {
        let mut npos = layer.neuron_data_start;
        for _ in 0..layer.neuron_count {
            if npos as usize + 8 > data.len() {
                return Err(ManasError::NeuronNotFound(neuron_id));
            }
            let nid =
                u64::from_le_bytes(data[npos as usize..npos as usize + 8].try_into().unwrap());
            if nid == neuron_id {
                return Ok(npos);
            }
            let nsize = compute_neuron_size_bytes(data, npos)?;
            npos += nsize;
        }
    }
    Err(ManasError::NeuronNotFound(neuron_id))
}

pub fn compute_neuron_size_bytes(data: &[u8], offset: u64) -> Result<u64, ManasError> {
    let off = offset as usize;
    if off + 10 > data.len() {
        return Err(ManasError::CorruptFile {
            path: std::path::PathBuf::new(),
            reason: "truncated neuron".into(),
        });
    }
    let weight_count = u16::from_le_bytes(data[off + 8..off + 10].try_into().unwrap());
    let source_len_off = 61 + weight_count as usize * 4;
    if off + source_len_off + 2 > data.len() {
        return Err(ManasError::CorruptFile {
            path: std::path::PathBuf::new(),
            reason: "truncated neuron (source_len)".into(),
        });
    }
    let source_len = u16::from_le_bytes(
        data[off + source_len_off..off + source_len_off + 2]
            .try_into()
            .unwrap(),
    );
    Ok(10
        + weight_count as u64 * 4
        + 4
        + 1
        + 4
        + 8
        + 8
        + 8
        + 8
        + 8
        + 1
        + 1
        + 2
        + source_len as u64
        + 1
        + 1)
}

#[allow(dead_code)]
pub fn read_raw_neuron(data: &[u8], offset: u64) -> Result<Neuron, ManasError> {
    let off = offset as usize;
    let mut pos = off;
    let id = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;
    let weight_count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
    pos += 2;
    let mut weights = Vec::with_capacity(weight_count as usize);
    for _ in 0..weight_count {
        weights.push(f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
        pos += 4;
    }
    let bias = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let activation = Activation::from_u8(data[pos]).ok_or_else(|| ManasError::CorruptFile {
        path: std::path::PathBuf::new(),
        reason: "invalid activation".into(),
    })?;
    pos += 1;
    let importance_score = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let born_at = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;
    let last_activated = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;
    let activation_count = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;
    let learned_at = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;
    let last_verified = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;
    let freshness_category = data[pos];
    pos += 1;
    let source_type = data[pos];
    pos += 1;
    let source_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
    pos += 2;
    let source_bytes = if source_len > 0 {
        data[pos..pos + source_len as usize].to_vec()
    } else {
        Vec::new()
    };
    pos += source_len as usize;
    let is_protected = data[pos] != 0;
    pos += 1;
    let protection_level = ProtectionLevel::from_u8(data[pos]).unwrap_or(ProtectionLevel::Open);

    Ok(Neuron {
        id,
        weights,
        bias,
        activation,
        importance_score,
        born_at,
        last_activated,
        activation_count,
        learned_at,
        last_verified,
        freshness_category,
        source: Source::from_bytes(source_type, &source_bytes),
        is_protected,
        protection_level,
    })
}

pub fn read_from_path(path: &std::path::Path) -> Result<Network, ManasError> {
    let mut file = std::fs::File::open(path).map_err(|e| ManasError::FileReadError {
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

    let header = format::read_header(data).ok_or_else(|| ManasError::CorruptFile {
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
        let layer =
            format::read_layer_block(data, &mut offset).ok_or_else(|| ManasError::CorruptFile {
                path: std::path::PathBuf::new(),
                reason: format!("failed to read layer block at offset {}", offset),
            })?;
        layers.push(layer);
    }

    let mut network = Network {
        layers,
        total_neurons: header.total_neurons,
        created_at: header.created_at,
        version: header.version,
        total_texts_learned: header.total_texts_learned,
        next_id: 0, // ← new field, starts at 0
    };
    network.recompute_next_id(); // ← one scan, O(1) alloc_id() forever after
    Ok(network)
}

pub fn read_vocab_from_bytes(data: &[u8]) -> Result<HashMap<u32, (String, Vec<f32>)>, ManasError> {
    let mut offset = format::HEADER_SIZE;
    let vocab_count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    offset += 4;

    let mut vocab = HashMap::with_capacity(vocab_count as usize);
    for _ in 0..vocab_count {
        if offset + 4 > data.len() {
            break;
        }
        let token_id = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if offset >= data.len() {
            break;
        }
        let token_len = data[offset] as usize;
        offset += 1;
        if offset + token_len > data.len() {
            break;
        }
        let token = String::from_utf8_lossy(&data[offset..offset + token_len]).to_string();
        offset += token_len;
        if offset + 2 > data.len() {
            break;
        }
        let embed_dim = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;
        if offset + embed_dim * 4 > data.len() {
            break;
        }
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
    let header = format::read_header(data).ok_or_else(|| ManasError::CorruptFile {
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
        let _ =
            format::read_layer_block(data, &mut offset).ok_or_else(|| ManasError::CorruptFile {
                path: std::path::PathBuf::new(),
                reason: format!("failed to read layer block at offset {}", offset),
            })?;
    }

    let archive_layer =
        format::read_layer_block(data, &mut offset).ok_or_else(|| ManasError::CorruptFile {
            path: std::path::PathBuf::new(),
            reason: "failed to read archive block".into(),
        })?;

    Ok(archive_layer.neurons)
}
