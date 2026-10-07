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
        if self
            .required_features
            .iter()
            .any(|feature| feature != "simd_groups" || !features.insert(feature))
        {
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
            if kernel.parameters.len() > 31 {
                return Err(Error(format!(
                    "kernel {} exceeds 31 buffer bindings",
                    kernel.name
                )));
            }
            let mut parameters = HashSet::new();
            for (index, parameter) in kernel.parameters.iter().enumerate() {
                match &parameter.ty {
                    ParameterType::Value { layout } => layout.validate()?,
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
                if parameter.binding != index as u32 {
                    return Err(Error(format!(
                        "kernel {} has a nonsequential binding",
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
        if self.target != DEVICE_TARGET || self.msl_version != MSL_VERSION {
            return Err(Error("unsupported artifact target or MSL version".into()));
        }
        if self.required_features != self.abi.required_features {
            return Err(Error(format!(
                "artifact features do not match the ABI: {:?}",
                self.required_features
            )));
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
