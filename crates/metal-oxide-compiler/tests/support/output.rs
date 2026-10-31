pub const POISON: u32 = 0xdead_beef;
const GUARDS: usize = 17;

/// Where a generated kernel's output words are: every padded thread's elements, then guards
/// that nothing writes.
pub struct Layout {
    pub threads: usize,
    /// Threads per block, which pad the last block.
    pub block: usize,
    /// Output elements per thread.
    pub elements: usize,
    /// Words per element: two for 64-bit elements, low word first.
    pub width: usize,
    /// The word of an element that still holds the poison.
    pub poison: u32,
}

impl Layout {
    /// The output buffer's length in elements.
    pub fn length(&self) -> usize {
        self.threads.div_ceil(self.block) * self.block * self.elements + GUARDS
    }

    /// The words of an untouched output buffer.
    pub fn initial(&self) -> Vec<u32> {
        vec![self.poison; self.length() * self.width]
    }

    /// Checks the padding before exposing the active threads' words to the oracle.
    pub fn payload<'a>(&self, output: &'a [u32]) -> Result<&'a [u32], String> {
        let length = self.length() * self.width;
        if output.len() != length {
            return Err(format!("{} output words, expected {length}", output.len()));
        }
        let active = self.threads * self.elements * self.width;
        if let Some(index) = output[active..]
            .iter()
            .position(|&word| word != self.poison)
        {
            return Err(format!(
                "output guard word {} was overwritten: {:#x}",
                active + index,
                output[active + index]
            ));
        }
        Ok(&output[..active])
    }
}

/// The low `bits` of POISON, which a poisoned element that narrow holds.
pub fn poison(bits: u32) -> u32 {
    POISON & (u32::MAX >> 32_u32.saturating_sub(bits))
}
