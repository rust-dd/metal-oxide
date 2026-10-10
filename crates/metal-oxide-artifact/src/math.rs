use serde::{Deserialize, Serialize};

/// Floating-point contract used by every kernel and helper in an artifact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathMode {
    /// Precise math functions, ordered arithmetic, and no implicit contraction.
    #[default]
    Precise,
    /// Fast math functions with reassociation, reciprocal approximations, and contraction.
    ///
    /// Arithmetic honors NaN and infinity; zero signs need not be preserved.
    /// Math functions follow the domains and accuracy of MSL's fast variants.
    /// Subnormal arithmetic may flush to zero in either mode.
    Fast,
}

impl MathMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Precise => "precise",
            Self::Fast => "fast",
        }
    }

    /// Metal options required to compile this numerical contract.
    pub const fn metal_flags(self) -> [&'static str; 3] {
        match self {
            Self::Precise => [
                "-fmetal-math-mode=safe",
                "-fmetal-math-fp32-functions=precise",
                "-ffp-contract=off",
            ],
            Self::Fast => [
                "-fmetal-math-mode=relaxed",
                "-fmetal-math-fp32-functions=fast",
                "-ffp-contract=fast",
            ],
        }
    }
}

impl std::str::FromStr for MathMode {
    type Err = crate::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "precise" => Ok(Self::Precise),
            "fast" => Ok(Self::Fast),
            _ => Err(crate::Error(format!(
                "invalid math mode {value:?}; expected precise or fast"
            ))),
        }
    }
}
