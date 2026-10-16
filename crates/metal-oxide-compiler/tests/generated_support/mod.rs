use std::path::PathBuf;

pub fn directory(name: &str) -> PathBuf {
    let directory = crate::support::root()
        .join("target/generated-numeric")
        .join(std::process::id().to_string())
        .join(name);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

pub fn compile(name: &str, source: &str) -> PathBuf {
    let directory = directory(name);
    let path = directory.join("case.rs");
    std::fs::write(&path, source).unwrap();
    let generated = directory.join("generated");
    let result = crate::support::emit_into(
        path.to_str().unwrap(),
        &["-C", "overflow-checks=off"],
        &generated,
    );
    std::fs::write(directory.join("compiler.stdout"), &result.stdout).unwrap();
    std::fs::write(directory.join("compiler.stderr"), &result.stderr).unwrap();
    assert!(
        result.status.success(),
        "{}: {}",
        path.display(),
        String::from_utf8_lossy(&result.stderr)
    );
    generated
}

pub struct Random(u64);

impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let other = self.below(index + 1);
            items.swap(index, other);
        }
    }
}
