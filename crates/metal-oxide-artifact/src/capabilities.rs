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
        name: "float32_atomics",
        msl: (3, 0),
        macos: 13,
        apple_family: 7,
        mac2: false,
    },
    Capability {
        name: "uint64_atomics",
        msl: (2, 4),
        macos: 12,
        apple_family: 8,
        mac2: false,
    },
    Capability {
        name: "atomic_acquire_release",
        msl: (4, 1),
        macos: 27,
        apple_family: 7,
        mac2: false,
    },
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
        name: "simdgroup_matrix",
        msl: (2, 3),
        macos: 11,
        apple_family: 7,
        mac2: false,
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

impl crate::Abi {
    /// The MSL version required by this ABI's features and the compiler's baseline.
    pub fn msl_version(&self) -> String {
        let (major, minor) = self
            .required_features
            .iter()
            .filter_map(|f| Capability::named(f))
            .map(|f| f.msl)
            .fold((3, 2), std::cmp::max);
        format!("{major}.{minor}")
    }

    fn minimum_macos(&self) -> u32 {
        self.required_features
            .iter()
            .filter_map(|f| Capability::named(f))
            .map(|f| f.macos)
            .fold(15, std::cmp::max)
    }

    pub fn target_flags(&self) -> [String; 2] {
        [
            format!("-std=metal{}", self.msl_version()),
            format!("-mmacosx-version-min={}.0", self.minimum_macos()),
        ]
    }
}
