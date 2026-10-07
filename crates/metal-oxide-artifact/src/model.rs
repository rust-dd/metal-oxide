use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scalar {
    F32,
    U32,
    I32,
    U8,
    U16,
}

impl Scalar {
    pub const fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
        }
    }

    pub const fn alignment(self) -> usize {
        self.size()
    }

    pub const fn rust_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U8 => "u8",
            Self::U16 => "u16",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Abi {
    pub version: u32,
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
    pub rust_flags: Vec<String>,
    pub metal_flags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Files {
    pub msl: String,
    pub oxide_ir: String,
    pub ir: String,
    pub metallib: String,
    pub bindings: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub abi: Abi,
    pub target: String,
    pub msl_version: String,
    pub required_features: Vec<String>,
    pub files: Files,
    pub build: BuildInfo,
}
