use std::collections::HashSet;

use crate::{
    ABI_VERSION, Abi, Access, DEVICE_TARGET, Error, MSL_VERSION, Manifest, ParameterType, Scalar,
};

pub(crate) fn identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Abi {
    pub fn validate(&self) -> Result<(), Error> {
        if self.version != ABI_VERSION {
            return Err(Error(format!(
                "unsupported kernel ABI version {}",
                self.version
            )));
        }
        if self.kernels.is_empty() {
            return Err(Error("artifact contains no kernels".into()));
        }
        let mut features = HashSet::new();
        if self.required_features.iter().any(|feature| {
            !matches!(
                feature.as_str(),
                "simd_groups" | "int32_atomics" | "dynamic_threadgroup_memory"
            ) || !features.insert(feature.as_str())
        }) {
            return Err(Error("unsupported or duplicate artifact feature".into()));
        }
        let mut names = HashSet::new();
        for kernel in &self.kernels {
            if !identifier(&kernel.name) || !names.insert(&kernel.name) {
                return Err(Error(format!(
                    "invalid or duplicate kernel name {:?}",
                    kernel.name
                )));
            }
            let mut parameters = HashSet::new();
            let mut slots = [0; 2];
            for parameter in &kernel.parameters {
                match &parameter.ty {
                    ParameterType::Value { layout } => layout.validate()?,
                    ParameterType::Threadgroup { element, stride } => {
                        element.validate()?;
                        if *stride != element.size {
                            return Err(Error(
                                "threadgroup stride does not match its element layout".into(),
                            ));
                        }
                        if !features.contains("dynamic_threadgroup_memory") {
                            return Err(Error(
                                "threadgroup argument requires dynamic_threadgroup_memory".into(),
                            ));
                        }
                    }
                    ParameterType::Buffer {
                        element,
                        stride,
                        access,
                    } => {
                        element.validate()?;
                        if *stride != element.size {
                            return Err(Error(
                                "buffer stride does not match its element layout".into(),
                            ));
                        }
                        if *access == Access::Atomic
                            && !matches!(element.scalar_type(), Some(Scalar::U32 | Scalar::I32))
                        {
                            return Err(Error(format!(
                                "kernel {} requires integer atomic elements",
                                kernel.name
                            )));
                        }
                    }
                }
                let namespace =
                    usize::from(matches!(parameter.ty, ParameterType::Threadgroup { .. }));
                if parameter.binding != slots[namespace] {
                    return Err(Error(format!(
                        "kernel {} has a nonsequential binding",
                        kernel.name
                    )));
                }
                slots[namespace] += 1;
                if slots[namespace] > 31 {
                    return Err(Error(format!(
                        "kernel {} exceeds 31 bindings in one argument table",
                        kernel.name
                    )));
                }
                if !identifier(&parameter.name) || !parameters.insert(&parameter.name) {
                    return Err(Error(format!(
                        "kernel {} has invalid parameter names",
                        kernel.name
                    )));
                }
            }
            if let Some(block) = kernel.required_block
                && (block.contains(&0)
                    || block
                        .into_iter()
                        .try_fold(1_u32, |a, b| a.checked_mul(b))
                        .is_none())
            {
                return Err(Error(format!(
                    "kernel {} has an invalid block requirement",
                    kernel.name
                )));
            }
        }
        Ok(())
    }
}

impl Manifest {
    pub fn validate(&self) -> Result<(), Error> {
        self.abi.validate()?;
        for expected in self.abi.math_mode.metal_flags() {
            let prefix = expected.split_once('=').unwrap().0;
            let actual = self
                .build
                .metal_flags
                .iter()
                .filter(|flag| flag.split_once('=').is_some_and(|(name, _)| name == prefix))
                .map(String::as_str)
                .collect::<Vec<_>>();
            if actual != [expected] {
                return Err(Error(
                    "Metal flags do not match the artifact math mode".into(),
                ));
            }
        }
        if self
            .build
            .metal_flags
            .iter()
            .any(|flag| matches!(flag.as_str(), "-ffast-math" | "-fno-fast-math"))
        {
            return Err(Error("legacy Metal math flags are unsupported".into()));
        }
        if self.target != DEVICE_TARGET || self.msl_version != MSL_VERSION {
            return Err(Error("unsupported artifact target or MSL version".into()));
        }
        if self
            .files
            .entries()
            .into_iter()
            .map(|(_, value)| value)
            .chain(std::iter::once(self.build.fingerprint.as_str()))
            .any(|v| !hash(v))
        {
            return Err(Error(
                "artifact hashes must be lowercase SHA-256 values".into(),
            ));
        }
        if [
            &self.build.compiler,
            &self.build.rustc,
            &self.build.metal,
            &self.build.sdk,
        ]
        .into_iter()
        .any(|v| v.trim().is_empty())
        {
            return Err(Error("artifact build identity is incomplete".into()));
        }
        Ok(())
    }
}
