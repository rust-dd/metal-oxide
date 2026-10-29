//! The CPU model of blocks that cooperative programs include: a thread per block thread, barriers,
//! and 32-lane SIMD groups that exchange values through slots.

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    marker::PhantomData,
    ops::{BitAnd, BitOr, BitXor},
    panic::{AssertUnwindSafe, Location},
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

const LANES: u32 = 32;

#[derive(Clone, Copy)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// A value that lanes exchange and threadgroup memory holds, as the bits of a word.
pub trait Bits: Copy {
    fn into_word(self) -> u64;
    fn from_word(word: u64) -> Self;
}

impl Bits for u32 {
    fn into_word(self) -> u64 {
        self.into()
    }
    fn from_word(word: u64) -> Self {
        word as u32
    }
}

impl Bits for i32 {
    fn into_word(self) -> u64 {
        (self as u32).into()
    }
    fn from_word(word: u64) -> Self {
        word as u32 as i32
    }
}

impl Bits for f32 {
    fn into_word(self) -> u64 {
        self.to_bits().into()
    }
    fn from_word(word: u64) -> Self {
        f32::from_bits(word as u32)
    }
}

impl Bits for bool {
    fn into_word(self) -> u64 {
        self.into()
    }
    fn from_word(word: u64) -> Self {
        word != 0
    }
}

/// The integers that SIMD reductions and threadgroup atomics combine; sums wrap.
pub trait Integer:
    Bits + Ord + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self>
{
    const ZERO: Self;
    fn plus(self, other: Self) -> Self;
    fn minus(self, other: Self) -> Self;
}

impl Integer for u32 {
    const ZERO: Self = 0;
    fn plus(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    fn minus(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }
}

impl Integer for i32 {
    const ZERO: Self = 0;
    fn plus(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    fn minus(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }
}

struct Block {
    barrier: Barrier,
    groups: Vec<Group>,
    /// Threadgroup memory by the call site that allocates it.
    memory: Mutex<HashMap<Location<'static>, &'static [AtomicU64]>>,
}

/// The lanes of a SIMD group meet at the barrier around each exchange of their slots.
struct Group {
    barrier: Barrier,
    slots: Vec<AtomicU64>,
}

impl Block {
    fn new(size: u32) -> Self {
        let groups = (0..size.div_ceil(LANES))
            .map(|group| {
                let lanes = (size - group * LANES).min(LANES) as usize;
                Group {
                    barrier: Barrier::new(lanes),
                    slots: (0..lanes).map(|_| AtomicU64::new(0)).collect(),
                }
            })
            .collect::<Vec<_>>();
        Self {
            barrier: Barrier::new(size as usize),
            groups,
            memory: Mutex::default(),
        }
    }
}

thread_local! {
    static THREAD: RefCell<Option<(Arc<Block>, u32)>> = const { RefCell::new(None) };
    static PANIC: Cell<(u32, u32)> = const { Cell::new((0, 0)) };
}

/// Calls `f` with the block of the current thread and the thread's index in it.
fn current<R>(f: impl FnOnce(&Block, u32) -> R) -> R {
    THREAD.with_borrow(|thread| {
        let (block, index) = thread.as_ref().expect("a launched thread");
        f(block, *index)
    })
}

pub fn thread_idx() -> Dim3 {
    current(|_, index| Dim3 {
        x: index,
        y: 0,
        z: 0,
    })
}

/// Runs `kernel` for threads `0..threads`, in blocks of `size` threads on OS threads; the last
/// block runs whole. Each outcome is the thread's result or where it panicked.
pub fn launch<R: Send>(
    size: u32,
    threads: u32,
    kernel: impl Fn(u32) -> R + Sync,
) -> Vec<Result<R, (u32, u32)>> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        let location = info.location().unwrap();
        PANIC.set((location.line(), location.column()));
    }));
    let mut outcomes = Vec::new();
    for first in (0..threads.div_ceil(size)).map(|block| block * size) {
        let block = Arc::new(Block::new(size));
        std::thread::scope(|scope| {
            let threads = (0..size)
                .map(|index| {
                    let (block, kernel) = (Arc::clone(&block), &kernel);
                    scope.spawn(move || {
                        THREAD.set(Some((block, index)));
                        std::panic::catch_unwind(AssertUnwindSafe(|| kernel(first + index)))
                            .map_err(|_| PANIC.get())
                    })
                })
                .collect::<Vec<_>>();
            outcomes.extend(threads.into_iter().map(|thread| thread.join().unwrap()));
        });
    }
    std::panic::set_hook(hook);
    outcomes
}

/// The block's storage for the allocation at `site`.
fn storage(site: &'static Location<'static>, length: usize) -> &'static [AtomicU64] {
    current(|block, _| {
        let mut memory = block.memory.lock().unwrap();
        *memory.entry(*site).or_insert_with(|| {
            let cells = (0..length).map(|_| AtomicU64::new(0)).collect::<Vec<_>>();
            cells.leak()
        })
    })
}

#[derive(Clone, Copy)]
pub struct ThreadgroupBuffer<T, const N: usize> {
    cells: &'static [AtomicU64],
    element: PhantomData<T>,
}

impl<T: Bits, const N: usize> ThreadgroupBuffer<T, N> {
    pub unsafe fn load_unchecked(self, index: u32) -> T {
        T::from_word(self.cells[index as usize].load(Relaxed))
    }

    pub unsafe fn store_unchecked(self, index: u32, value: T) {
        self.cells[index as usize].store(value.into_word(), Relaxed);
    }
}

/// The initial words of the device counters that cooperative programs may count in.
pub const COUNTERS: [u32; 4] = [0, u32::MAX, 0x7fff_ffff, 0x8000_0000];

macro_rules! atomics {
    () => {
        pub unsafe fn load(self, index: u32, _order: core::sync::atomic::Ordering) -> T {
            T::from_word(self.cells[index as usize].load(Relaxed))
        }

        pub unsafe fn store(self, index: u32, value: T, _order: core::sync::atomic::Ordering) {
            self.cells[index as usize].store(value.into_word(), Relaxed);
        }

        fn update(self, index: u32, change: impl Fn(T) -> T) -> T {
            let cell = &self.cells[index as usize];
            let mut word = cell.load(Relaxed);
            loop {
                let next = change(T::from_word(word)).into_word();
                match cell.compare_exchange_weak(word, next, Relaxed, Relaxed) {
                    Ok(previous) => return T::from_word(previous),
                    Err(current) => word = current,
                }
            }
        }

        pub unsafe fn fetch_add(
            self,
            index: u32,
            value: T,
            _order: core::sync::atomic::Ordering,
        ) -> T {
            self.update(index, |old| old.plus(value))
        }

        pub unsafe fn fetch_sub(
            self,
            index: u32,
            value: T,
            _order: core::sync::atomic::Ordering,
        ) -> T {
            self.update(index, |old| old.minus(value))
        }

        pub unsafe fn fetch_min(
            self,
            index: u32,
            value: T,
            _order: core::sync::atomic::Ordering,
        ) -> T {
            self.update(index, |old| old.min(value))
        }

        pub unsafe fn fetch_max(
            self,
            index: u32,
            value: T,
            _order: core::sync::atomic::Ordering,
        ) -> T {
            self.update(index, |old| old.max(value))
        }

        pub unsafe fn fetch_and(
            self,
            index: u32,
            value: T,
            _order: core::sync::atomic::Ordering,
        ) -> T {
            self.update(index, |old| old & value)
        }

        pub unsafe fn fetch_or(
            self,
            index: u32,
            value: T,
            _order: core::sync::atomic::Ordering,
        ) -> T {
            self.update(index, |old| old | value)
        }

        pub unsafe fn fetch_xor(
            self,
            index: u32,
            value: T,
            _order: core::sync::atomic::Ordering,
        ) -> T {
            self.update(index, |old| old ^ value)
        }
    };
}

/// Device atomics that the threads of every block share.
#[derive(Clone, Copy)]
pub struct AtomicBuffer<T> {
    cells: &'static [AtomicU64],
    element: PhantomData<T>,
}

impl<T: Integer> AtomicBuffer<T> {
    pub fn new(words: &[u32]) -> Self {
        let cells = words.iter().map(|&word| AtomicU64::new(word.into()));
        Self {
            cells: cells.collect::<Vec<_>>().leak(),
            element: PhantomData,
        }
    }

    pub fn words(self) -> Vec<u32> {
        let words = self.cells.iter().map(|cell| cell.load(Relaxed) as u32);
        words.collect()
    }

    atomics!();
}

#[derive(Clone, Copy)]
pub struct AtomicThreadgroupBuffer<T, const N: usize> {
    cells: &'static [AtomicU64],
    element: PhantomData<T>,
}

impl<T: Integer, const N: usize> AtomicThreadgroupBuffer<T, N> {
    atomics!();
}

pub mod threadgroup {
    use super::{AtomicThreadgroupBuffer, Bits, ThreadgroupBuffer, current, storage};
    use std::{marker::PhantomData, panic::Location};

    #[track_caller]
    pub fn shared<T: Bits, const N: usize>() -> ThreadgroupBuffer<T, N> {
        ThreadgroupBuffer {
            cells: storage(Location::caller(), N),
            element: PhantomData,
        }
    }

    #[track_caller]
    pub fn shared_atomic<T: Bits, const N: usize>() -> AtomicThreadgroupBuffer<T, N> {
        AtomicThreadgroupBuffer {
            cells: storage(Location::caller(), N),
            element: PhantomData,
        }
    }

    pub unsafe fn barrier() {
        current(|block, _| {
            block.barrier.wait();
        });
    }
}

/// Every lane's value in the current SIMD group, in lane order, and the current lane.
fn exchange<T: Bits>(value: T) -> (Vec<T>, usize) {
    current(|block, index| {
        let group = &block.groups[(index / LANES) as usize];
        let lane = (index % LANES) as usize;
        group.slots[lane].store(value.into_word(), Relaxed);
        group.barrier.wait();
        let values = group
            .slots
            .iter()
            .map(|slot| T::from_word(slot.load(Relaxed)));
        let values = values.collect::<Vec<_>>();
        // The next exchange reuses the slots once every lane has read them.
        group.barrier.wait();
        (values, lane)
    })
}

pub mod simdgroup {
    use super::{Bits, Integer, exchange};

    pub unsafe fn sum<T: Integer>(value: T) -> T {
        exchange(value).0.into_iter().fold(T::ZERO, T::plus)
    }

    pub unsafe fn min<T: Integer>(value: T) -> T {
        exchange(value).0.into_iter().min().unwrap()
    }

    pub unsafe fn max<T: Integer>(value: T) -> T {
        exchange(value).0.into_iter().max().unwrap()
    }

    pub unsafe fn and<T: Integer>(value: T) -> T {
        exchange(value).0.into_iter().reduce(|a, b| a & b).unwrap()
    }

    pub unsafe fn or<T: Integer>(value: T) -> T {
        exchange(value).0.into_iter().reduce(|a, b| a | b).unwrap()
    }

    pub unsafe fn xor<T: Integer>(value: T) -> T {
        exchange(value).0.into_iter().reduce(|a, b| a ^ b).unwrap()
    }

    pub unsafe fn inclusive_sum<T: Integer>(value: T) -> T {
        let (values, lane) = exchange(value);
        values[..=lane].iter().copied().fold(T::ZERO, T::plus)
    }

    pub unsafe fn exclusive_sum<T: Integer>(value: T) -> T {
        let (values, lane) = exchange(value);
        values[..lane].iter().copied().fold(T::ZERO, T::plus)
    }

    pub unsafe fn shuffle<T: Bits>(value: T, source: u32) -> T {
        exchange(value).0[source as usize]
    }

    pub unsafe fn shuffle_up<T: Bits>(value: T, delta: u32) -> T {
        let (values, lane) = exchange(value);
        lane.checked_sub(delta as usize)
            .map_or(value, |source| values[source])
    }

    pub unsafe fn shuffle_down<T: Bits>(value: T, delta: u32) -> T {
        let (values, lane) = exchange(value);
        values.get(lane + delta as usize).copied().unwrap_or(value)
    }

    pub unsafe fn shuffle_xor<T: Bits>(value: T, mask: u32) -> T {
        let (values, lane) = exchange(value);
        values[lane ^ mask as usize]
    }

    pub unsafe fn any(predicate: bool) -> bool {
        exchange(predicate).0.contains(&true)
    }

    pub unsafe fn all(predicate: bool) -> bool {
        !exchange(predicate).0.contains(&false)
    }

    pub unsafe fn ballot(predicate: bool) -> [u32; 2] {
        let votes = exchange(predicate).0.into_iter().enumerate();
        let bits = votes.fold(0_u64, |bits, (lane, vote)| bits | u64::from(vote) << lane);
        [bits as u32, (bits >> 32) as u32]
    }
}
