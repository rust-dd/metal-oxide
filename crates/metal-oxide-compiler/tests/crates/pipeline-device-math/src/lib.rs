#![no_std]

pub struct Affine {
    pub scale: f32,
    pub offset: f32,
}

impl Affine {
    pub fn apply(self, value: f32) -> f32 {
        value * self.scale + self.offset
    }
}

pub fn first<const SCALE: u32>(value: f32) -> f32 {
    Affine {
        scale: SCALE as f32,
        offset: 3.0,
    }
    .apply(value)
}

pub fn second(value: f32) -> f32 {
    Affine {
        scale: 5.0,
        offset: 7.0,
    }
    .apply(value)
}
