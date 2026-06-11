use crate::activation::Activation;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProtectionLevel {
    Open,
    Guarded,
    Frozen,
}

impl ProtectionLevel {
    pub fn to_u8(&self) -> u8 {
        match self {
            ProtectionLevel::Open => 0,
            ProtectionLevel::Guarded => 1,
            ProtectionLevel::Frozen => 2,
        }
    }

    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(ProtectionLevel::Open),
            1 => Some(ProtectionLevel::Guarded),
            2 => Some(ProtectionLevel::Frozen),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Source {
    RawText,
    LocalFile { path: String },
    Internet { url: String },
    Unknown,
}

impl Source {
    pub fn to_bytes(&self) -> (u8, Vec<u8>) {
        match self {
            Source::RawText => (0, Vec::new()),
            Source::LocalFile { path } => (1, path.as_bytes().to_vec()),
            Source::Internet { url } => (2, url.as_bytes().to_vec()),
            Source::Unknown => (3, Vec::new()),
        }
    }

    pub fn from_bytes(source_type: u8, data: &[u8]) -> Self {
        let s = String::from_utf8_lossy(data).to_string();
        match source_type {
            0 => Source::RawText,
            1 => Source::LocalFile { path: s },
            2 => Source::Internet { url: s },
            _ => Source::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Neuron {
    pub id: u64,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub activation: Activation,
    pub importance_score: f32,
    pub born_at: u64,
    pub last_activated: u64,
    pub activation_count: u64,
    pub learned_at: u64,
    pub last_verified: u64,
    pub freshness_category: u8,
    pub source: Source,
    pub is_protected: bool,
    pub protection_level: ProtectionLevel,
}

impl Neuron {
    pub fn new(id: u64, input_size: usize, activation: Activation) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weights: Vec<f32> = (0..input_size)
            .map(|_| {
                let u: f32 = rng.r#gen();
                let z = (-2.0 * u.ln()).sqrt()
                    * (2.0 * std::f32::consts::PI * rng.r#gen::<f32>()).cos();
                z * 0.1
            })
            .collect();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Neuron {
            id,
            weights,
            bias: 0.0,
            activation,
            importance_score: 0.5,
            born_at: now,
            last_activated: 0,
            activation_count: 0,
            learned_at: now,
            last_verified: now,
            freshness_category: 1,
            source: Source::Unknown,
            is_protected: false,
            protection_level: ProtectionLevel::Guarded,
        }
    }

    pub fn activate(&self, input: &[f32]) -> f32 {
        let sum: f32 = self
            .weights
            .iter()
            .zip(input)
            .map(|(w, i)| w * i)
            .sum::<f32>()
            + self.bias;
        self.activation.apply(sum)
    }
}
