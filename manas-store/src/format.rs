use manas_core::{Activation, Layer, Neuron, ProtectionLevel, Source};

pub const MAGIC: &[u8; 5] = b"MANAS";
pub const HEADER_SIZE: usize = 64;
pub struct BrainHeader {
    pub magic: [u8; 5],
    pub version: u8,
    pub created_at: u64,
    pub last_modified: u64,
    pub total_neurons: u64,
    pub total_layers: u32,
    pub vocab_size: u32,
    pub total_texts_learned: u64,
    pub flags: u16,
    pub checksum_offset: u32,
}

pub fn read_header(data: &[u8]) -> Option<BrainHeader> {
    if data.len() < HEADER_SIZE {
        return None;
    }
    let magic: [u8; 5] = data[0..5].try_into().ok()?;
    if &magic != MAGIC {
        return None;
    }
    Some(BrainHeader {
        magic,
        version: data[5],
        created_at: u64::from_le_bytes(data[6..14].try_into().ok()?),
        last_modified: u64::from_le_bytes(data[14..22].try_into().ok()?),
        total_neurons: u64::from_le_bytes(data[22..30].try_into().ok()?),
        total_layers: u32::from_le_bytes(data[30..34].try_into().ok()?),
        vocab_size: u32::from_le_bytes(data[34..38].try_into().ok()?),
        total_texts_learned: u64::from_le_bytes(data[38..46].try_into().ok()?),
        flags: u16::from_le_bytes(data[46..48].try_into().ok()?),
        checksum_offset: u32::from_le_bytes(data[48..52].try_into().ok()?),
    })
}

pub fn write_header(buf: &mut Vec<u8>, header: &BrainHeader) {
    buf.extend_from_slice(&header.magic);
    buf.push(header.version);
    buf.extend_from_slice(&header.created_at.to_le_bytes());
    buf.extend_from_slice(&header.last_modified.to_le_bytes());
    buf.extend_from_slice(&header.total_neurons.to_le_bytes());
    buf.extend_from_slice(&header.total_layers.to_le_bytes());
    buf.extend_from_slice(&header.vocab_size.to_le_bytes());
    buf.extend_from_slice(&header.total_texts_learned.to_le_bytes());
    buf.extend_from_slice(&header.flags.to_le_bytes());
    buf.extend_from_slice(&header.checksum_offset.to_le_bytes());
    buf.extend_from_slice(&[0u8; 12]);
}

pub fn neuron_binary_size(neuron: &Neuron) -> usize {
    let (_stype, sbytes) = neuron.source.to_bytes();
    8 + 2
        + neuron.weights.len() * 4
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
        + sbytes.len()
        + 1
        + 1
}

pub fn write_neuron(buf: &mut Vec<u8>, neuron: &Neuron) {
    buf.extend_from_slice(&neuron.id.to_le_bytes());
    buf.extend_from_slice(&(neuron.weights.len() as u16).to_le_bytes());
    for w in &neuron.weights {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    buf.extend_from_slice(&neuron.bias.to_le_bytes());
    buf.push(neuron.activation.to_u8());
    buf.extend_from_slice(&neuron.importance_score.to_le_bytes());
    buf.extend_from_slice(&neuron.born_at.to_le_bytes());
    buf.extend_from_slice(&neuron.last_activated.to_le_bytes());
    buf.extend_from_slice(&neuron.activation_count.to_le_bytes());
    buf.extend_from_slice(&neuron.learned_at.to_le_bytes());
    buf.extend_from_slice(&neuron.last_verified.to_le_bytes());
    buf.push(neuron.freshness_category);
    let (stype, sbytes) = neuron.source.to_bytes();
    buf.push(stype);
    buf.extend_from_slice(&(sbytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(&sbytes);
    buf.push(neuron.is_protected as u8);
    buf.push(neuron.protection_level.to_u8());
}

pub fn read_neuron(data: &[u8], offset: &mut usize) -> Option<Neuron> {
    if *offset + 8 > data.len() {
        return None;
    }
    let id = u64::from_le_bytes(data[*offset..*offset + 8].try_into().ok()?);
    *offset += 8;

    if *offset + 2 > data.len() {
        return None;
    }
    let weight_count = u16::from_le_bytes(data[*offset..*offset + 2].try_into().ok()?);
    *offset += 2;

    let mut weights = Vec::with_capacity(weight_count as usize);
    for _ in 0..weight_count {
        if *offset + 4 > data.len() {
            return None;
        }
        weights.push(f32::from_le_bytes(
            data[*offset..*offset + 4].try_into().ok()?,
        ));
        *offset += 4;
    }

    if *offset + 4 > data.len() {
        return None;
    }
    let bias = f32::from_le_bytes(data[*offset..*offset + 4].try_into().ok()?);
    *offset += 4;

    let activation = Activation::from_u8(data[*offset])?;
    *offset += 1;

    if *offset + 4 > data.len() {
        return None;
    }
    let importance_score = f32::from_le_bytes(data[*offset..*offset + 4].try_into().ok()?);
    *offset += 4;

    if *offset + 8 > data.len() {
        return None;
    }
    let born_at = u64::from_le_bytes(data[*offset..*offset + 8].try_into().ok()?);
    *offset += 8;

    if *offset + 8 > data.len() {
        return None;
    }
    let last_activated = u64::from_le_bytes(data[*offset..*offset + 8].try_into().ok()?);
    *offset += 8;

    if *offset + 8 > data.len() {
        return None;
    }
    let activation_count = u64::from_le_bytes(data[*offset..*offset + 8].try_into().ok()?);
    *offset += 8;

    if *offset + 8 > data.len() {
        return None;
    }
    let learned_at = u64::from_le_bytes(data[*offset..*offset + 8].try_into().ok()?);
    *offset += 8;

    if *offset + 8 > data.len() {
        return None;
    }
    let last_verified = u64::from_le_bytes(data[*offset..*offset + 8].try_into().ok()?);
    *offset += 8;

    let freshness_category = data[*offset];
    *offset += 1;

    let source_type = data[*offset];
    *offset += 1;

    if *offset + 2 > data.len() {
        return None;
    }
    let source_len = u16::from_le_bytes(data[*offset..*offset + 2].try_into().ok()?);
    *offset += 2;

    let source_data = if source_len > 0 {
        if *offset + source_len as usize > data.len() {
            return None;
        }
        let s = data[*offset..*offset + source_len as usize].to_vec();
        *offset += source_len as usize;
        s
    } else {
        Vec::new()
    };
    let source = Source::from_bytes(source_type, &source_data);

    let is_protected = data[*offset] != 0;
    *offset += 1;

    let protection_level = ProtectionLevel::from_u8(data[*offset]).unwrap_or(ProtectionLevel::Open);
    *offset += 1;

    Some(Neuron {
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
        source,
        is_protected,
        protection_level,
    })
}

pub fn write_layer_block(buf: &mut Vec<u8>, layer: &Layer) {
    buf.extend_from_slice(&layer.id.to_le_bytes());
    buf.extend_from_slice(&(layer.neurons.len() as u32).to_le_bytes());
    buf.push(layer.activation.to_u8());
    for neuron in &layer.neurons {
        write_neuron(buf, neuron);
    }
}

pub fn read_layer_block(data: &[u8], offset: &mut usize) -> Option<Layer> {
    if *offset + 9 > data.len() {
        return None;
    }
    let id = u32::from_le_bytes(data[*offset..*offset + 4].try_into().ok()?);
    *offset += 4;
    let neuron_count = u32::from_le_bytes(data[*offset..*offset + 4].try_into().ok()?);
    *offset += 4;
    let act = Activation::from_u8(data[*offset])?;
    *offset += 1;

    let mut neurons = Vec::with_capacity(neuron_count as usize);
    for _ in 0..neuron_count {
        let neuron = read_neuron(data, offset)?;
        neurons.push(neuron);
    }

    Some(Layer {
        id,
        neurons,
        activation: act,
    })
}
