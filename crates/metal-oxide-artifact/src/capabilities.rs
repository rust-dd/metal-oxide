/// A shader feature's minimum language, OS and GPU requirements.
///
/// Sources: Apple Metal Shading Language Specification, sections 2.1, 6.10 and 6.16,
/// and the Metal Feature Set Tables. Runtime support also requires macOS 15 or later.
#[derive(Clone, Copy, Debug)]
pub struct Capability {
    pub name: &'static str,
    pub msl: (u8, u8),
    pub macos: u32,
    pub apple_family: u32,
    pub mac2: bool,
}

/// Requirements shared by artifact validation, code generation and device loading.
pub const CAPABILITIES: &[Capability] = &[
    Capability {
        name: "int32_atomics",
        msl: (1, 0),
        macos: 11,
        apple_family: 1,
        mac2: true,
    },
    Capability {
        name: "simd_groups",
        msl: (2, 0),
        macos: 11,
        apple_family: 7,
        mac2: true,
    },
    Capability {
        name: "dynamic_threadgroup_memory",
        msl: (1, 0),
        macos: 11,
        apple_family: 1,
        mac2: true,
    },
    Capability {
        name: "logging",
        msl: (3, 2),
        macos: 15,
        apple_family: 6,
        mac2: true,
    },
    Capability {
        name: "bfloat16",
        msl: (3, 1),
        macos: 14,
        apple_family: 6,
        mac2: false,
    },
];

impl Capability {
    pub fn named(name: &str) -> Option<&'static Self> {
        CAPABILITIES.iter().find(|feature| feature.name == name)
    }
}
