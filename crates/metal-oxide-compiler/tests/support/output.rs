pub const POISON: u32 = 0xdead_beef;
const GUARDS: usize = 17;

/// Includes every padded thread's output and trailing guard words.
pub fn initial(threads: usize, words: usize) -> Vec<u32> {
    vec![POISON; threads.div_ceil(64) * 64 * words + GUARDS]
}

/// Checks the padding before exposing the active threads' words to the oracle.
pub fn payload(output: &[u32], threads: usize, words: usize) -> Result<&[u32], String> {
    let length = threads.div_ceil(64) * 64 * words + GUARDS;
    if output.len() != length {
        return Err(format!("{} output words, expected {length}", output.len()));
    }
    let active = threads * words;
    if let Some(index) = output[active..].iter().position(|&word| word != POISON) {
        return Err(format!(
            "output guard word {} was overwritten: {:#x}",
            active + index,
            output[active + index]
        ));
    }
    Ok(&output[..active])
}
