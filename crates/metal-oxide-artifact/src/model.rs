use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scalar {
    F32,
    F16,
    U32,
    U64,
    I32,
    U8,
    U16,
    I8,
    I16,
}

impl Scalar {
    pub const fn size(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::U64 => 8,
        }
    }

    pub const fn alignment(self) -> usize {
        self.size()
    }

    pub const fn is_integer(self) -> bool {
        !matches!(self, Self::F32 | Self::F16)
    }

    pub const fn rust_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "metal_oxide::F16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I32 => "i32",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::I8 => "i8",
            Self::I16 => "i16",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
    ReadWrite,
    Atomic,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParameterType {
    Value {
        layout: crate::Layout,
    },
    Buffer {
        element: crate::Layout,
        stride: u64,
        access: Access,
    },
    Threadgroup {
        element: crate::Layout,
        stride: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameter {
    pub name: String,
    pub binding: u32,
    pub ty: ParameterType,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub required_block: Option<[u32; 3]>,
    /// The buffer slot after the parameters: a word that is nonzero once an earlier dispatch
    /// of the command buffer has failed, telling this one to skip its work.
    pub gate_binding: u32,
    /// The kernel can fail a runtime check; it then also takes the command buffer's failure
    /// word in the slot after its gate.
    pub fallible: bool,
    /// The kernel reads buffer lengths; it then takes the element count of each parameter, as
    /// a `ulong` table, in the slot after its gate and failure words.
    pub lengths: bool,
    /// The kernel reads its thread index, so it accepts only one-dimensional launches.
    pub one_dimensional: bool,
    /// The kernel is a safe `fn`, so launching it is safe.
    pub safe: bool,
    /// Relations between argument lengths and values that a launch checks before encoding.
    pub requires: Vec<crate::Contract>,
}

/// Serialized with layouts and parameter lists shared across kernels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "crate::tables::AbiTables",
    into = "crate::tables::AbiTables"
)]
pub struct Abi {
    pub version: u32,
    pub math_mode: crate::MathMode,
    pub required_features: Vec<String>,
    pub kernels: Vec<Kernel>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildInfo {
    pub fingerprint: String,
    pub compiler: String,
    pub rustc: String,
    pub metal: String,
    pub sdk: String,
    pub rustc_args: Vec<String>,
    pub metal_flags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Files {
    pub msl: String,
    pub oxide_ir: String,
    pub ir: String,
    pub metallib: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub abi: Abi,
    pub checks: Vec<crate::Check>,
    pub target: String,
    pub msl_version: String,
    pub files: Files,
    pub build: BuildInfo,
}
