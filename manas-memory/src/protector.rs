use manas_core::{Network, Neuron, ProtectionLevel};

pub fn protection_from_importance(importance: f32, age_seconds: u64) -> ProtectionLevel {
    if importance >= 0.85 {
        ProtectionLevel::Frozen
    } else if importance >= 0.60 || age_seconds < 7 * 86400 {
        ProtectionLevel::Guarded
    } else {
        ProtectionLevel::Open
    }
}

pub fn update_neuron(neuron: &mut Neuron) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if neuron.is_protected {
        neuron.protection_level = ProtectionLevel::Frozen;
        return;
    }

    let age = now.saturating_sub(neuron.born_at);
    neuron.protection_level = protection_from_importance(neuron.importance_score, age);
}

pub fn update_all(network: &mut Network) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    for layer in &mut network.layers {
        for neuron in &mut layer.neurons {
            if neuron.is_protected {
                neuron.protection_level = ProtectionLevel::Frozen;
                continue;
            }
            let age = now.saturating_sub(neuron.born_at);
            neuron.protection_level = protection_from_importance(neuron.importance_score, age);
        }
    }
}
