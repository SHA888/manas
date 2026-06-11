#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

impl Activation {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Linear => x,
        }
    }

    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Linear => 1.0,
        }
    }

    pub fn to_u8(&self) -> u8 {
        match self {
            Activation::ReLU => 0,
            Activation::Sigmoid => 1,
            Activation::Tanh => 2,
            Activation::Linear => 3,
        }
    }

    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Activation::ReLU),
            1 => Some(Activation::Sigmoid),
            2 => Some(Activation::Tanh),
            3 => Some(Activation::Linear),
            _ => None,
        }
    }
}
