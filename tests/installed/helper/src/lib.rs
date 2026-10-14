//! Math that an application's kernels and its CPU reference share.
#![no_std]

/// The position after `steps` steps at `velocity`.
pub fn advance(mut position: [f32; 2], velocity: [f32; 2], dt: f32, steps: u32) -> [f32; 2] {
    let mut step = 0;
    while step < steps {
        position[0] += velocity[0] * dt;
        position[1] += velocity[1] * dt;
        step += 1;
    }
    position
}
