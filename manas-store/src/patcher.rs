use std::io::Read;
use manas_core::{ManasError, Neuron};
use crate::format;
use crate::integrity;
use crate::reader;

fn read_all(path: &std::path::Path) -> Result<Vec<u8>, ManasError> {
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
    Ok(data)
}

fn strip_checksum(data: &mut Vec<u8>) {
    if data.len() >= 4 {
        data.truncate(data.len() - 4);
    }
}

pub fn append_neuron_to_file(path: &std::path::Path, layer_id: u32, neuron: &Neuron) -> Result<(), ManasError> {
    let mut data = read_all(path)?;

    integrity::verify_checksum(&data)?;
    strip_checksum(&mut data);

    let header = format::read_header(&data).ok_or_else(|| ManasError::CorruptFile {
        path: path.to_path_buf(),
        reason: "invalid header".into(),
    })?;

    let layers = reader::find_layer_locations(&data)?;
    let layer = layers.iter().find(|l| l.id == layer_id).ok_or_else(|| {
        ManasError::GrowthFailed(format!("layer {} not found", layer_id))
    })?;

    let mut neuron_bytes = Vec::new();
    format::write_neuron(&mut neuron_bytes, neuron);

    // Calculate insert position: end of target layer's last neuron
    let mut insert_pos = layer.neuron_data_start as usize;
    let mut npos = layer.neuron_data_start;
    for _ in 0..layer.neuron_count {
        if let Ok(ns) = reader::compute_neuron_size_bytes(&data, npos) {
            insert_pos = (npos + ns) as usize;
            npos += ns;
        }
    }

    // Insert neuron bytes at insert_pos (shifts subsequent data)
    let shift = neuron_bytes.len();
    let after = data.split_off(insert_pos);
    data.extend_from_slice(&neuron_bytes);
    data.extend_from_slice(&after);

    // Update this layer's neuron count in the layer block header
    let count_off = (layer.block_start + 4) as usize;
    let new_count = layer.neuron_count + 1;
    data[count_off..count_off + 4].copy_from_slice(&new_count.to_le_bytes());

    // Update layer index: update neuron count and shift offsets for subsequent layers
    let layer_index_start = (layers[0].block_start - header.total_layers as u64 * 16) as usize;
    let layer_index_entries = header.total_layers as usize;
    let mut found_target = false;
    for i in 0..layer_index_entries {
        let entry_off = layer_index_start + i * 16;
        if entry_off + 16 > data.len() { break; }
        let lid = u32::from_le_bytes(data[entry_off..entry_off + 4].try_into().unwrap());
        if lid == layer_id {
            // Update this layer's neuron count in the index
            let old_count = u32::from_le_bytes(data[entry_off + 12..entry_off + 16].try_into().unwrap());
            data[entry_off + 12..entry_off + 16].copy_from_slice(&(old_count + 1).to_le_bytes());
            found_target = true;
        } else if found_target {
            // Shift block offset for layers after the modified one
            let mut block_off = u64::from_le_bytes(data[entry_off + 4..entry_off + 12].try_into().unwrap());
            block_off += shift as u64;
            data[entry_off + 4..entry_off + 12].copy_from_slice(&block_off.to_le_bytes());
        }
    }

    // Update total_neurons in header
    let new_total = header.total_neurons + 1;
    data[22..30].copy_from_slice(&new_total.to_le_bytes());

    // Update last_modified
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    data[14..22].copy_from_slice(&now.to_le_bytes());

    // Update checksum offset and compute final CRC
    let checksum_offset = data.len() as u32;
    data[48..52].copy_from_slice(&checksum_offset.to_le_bytes());
    let crc = integrity::compute_crc32(&data);
    data.extend_from_slice(&crc.to_le_bytes());

    std::fs::write(path, &data).map_err(|e| ManasError::FileReadError {
        path: path.to_path_buf(),
        source: e,
    })?;

    Ok(())
}

pub fn update_neuron_in_file(path: &std::path::Path, neuron_id: u64, neuron: &Neuron) -> Result<(), ManasError> {
    let mut data = read_all(path)?;

    integrity::verify_checksum(&data)?;
    strip_checksum(&mut data);

    let neuron_offset = reader::find_neuron_offset(&data, neuron_id)?;
    let old_size = reader::compute_neuron_size_bytes(&data, neuron_offset)?;

    let mut new_bytes = Vec::new();
    format::write_neuron(&mut new_bytes, neuron);

    if new_bytes.len() as u64 != old_size {
        return Err(ManasError::CorruptFile {
            path: path.to_path_buf(),
            reason: format!(
                "neuron {} size mismatch: old={} new={}",
                neuron_id, old_size, new_bytes.len()
            ),
        });
    }

    let off = neuron_offset as usize;
    data[off..off + new_bytes.len()].copy_from_slice(&new_bytes);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    data[14..22].copy_from_slice(&now.to_le_bytes());

    let checksum_offset = data.len() as u32;
    data[48..52].copy_from_slice(&checksum_offset.to_le_bytes());
    let crc = integrity::compute_crc32(&data);
    data.extend_from_slice(&crc.to_le_bytes());

    std::fs::write(path, &data).map_err(|e| ManasError::FileReadError {
        path: path.to_path_buf(),
        source: e,
    })?;

    Ok(())
}
