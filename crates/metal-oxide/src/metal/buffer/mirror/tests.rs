use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};

use metal_oxide_artifact::Layout;

use super::*;

std::thread_local! {
    static PANIC_NEXT: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Word(u32);

impl GpuValue for Word {
    const SIZE: usize = 4;
    const ALIGNMENT: usize = 4;
    fn layout() -> crate::Result<Layout> {
        u32::layout()
    }
    fn zeroed() -> Self {
        Self(0)
    }
    fn encode(self, bytes: &mut [u8]) {
        self.0.encode(bytes);
    }
    fn decode(bytes: &[u8]) -> Self {
        assert!(!PANIC_NEXT.replace(false), "decoder failure");
        Self(u32::decode(bytes))
    }
}

fn bytes(storage: &mut [u8]) -> NonNull<u8> {
    NonNull::new(storage.as_mut_ptr()).unwrap()
}

#[test]
fn gpu_writes_decode_and_cpu_writes_encode() {
    let access = AccessState::default();
    let mut storage = [7, 0, 0, 0, 9, 0, 0, 0];
    let mut mirror = Mirror::new(vec![Word(0); 2], Location::Synced);
    access.mark_written();
    // SAFETY: the storage holds two canonical values and no GPU work exists.
    let values = unsafe { mirror.download(&access, bytes(&mut storage)) };
    assert_eq!(values, [Word(7), Word(9)]);
    // SAFETY: as above.
    let values = unsafe { mirror.download_mut(&access, bytes(&mut storage)) };
    values[1] = Word(11);
    // SAFETY: as above.
    unsafe { mirror.prepare(Access::Write, bytes(&mut storage)) };
    assert_eq!(storage, [7, 0, 0, 0, 11, 0, 0, 0]);
    storage[0] = 5;
    // SAFETY: as above; the prepared write left the GPU copy authoritative.
    let values = unsafe { mirror.download(&access, bytes(&mut storage)) };
    assert_eq!(values, [Word(5), Word(11)]);
}

#[test]
fn a_decoder_panic_keeps_the_gpu_copy_authoritative() {
    let access = AccessState::default();
    let mut storage = [3, 0, 0, 0];
    let storage = bytes(&mut storage);
    let mirror = Mirror::new(vec![Word(0)], Location::Gpu);
    PANIC_NEXT.with(|panic| panic.set(true));
    // SAFETY: the storage holds one canonical value and no GPU work exists.
    let failed = catch_unwind(AssertUnwindSafe(|| unsafe {
        mirror.download(&access, storage).to_vec()
    }));
    assert!(failed.is_err());
    // SAFETY: as above.
    assert_eq!(unsafe { mirror.download(&access, storage) }, [Word(3)]);
}
