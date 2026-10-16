#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub mod atomics;
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub mod half;
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub mod wide;

pub use crate::generated_support::{Random, compile};

pub const SEEDS: [u64; 8] = [
    0,
    1,
    7,
    31,
    255,
    0xdead_beef,
    0x0123_4567_89ab_cdef,
    u64::MAX,
];
