// words: float exact exact exact
// overflow checks: on
// cooperative
// counters: i32
#![cfg_attr(not(cpu_reference), no_std)]
#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, unused_imports, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]

fn cooperate(input: u32, lane: u32, counters: AtomicBuffer<i32>, a8: AtomicThreadgroupBuffer<i32, 1>) -> [u32; 1] {
    unsafe { a8.store(lane % 1, 1_i32, core::sync::atomic::Ordering::Relaxed) };
    unsafe { threadgroup::barrier() };
    unsafe { a8.fetch_or(0_u32 % 1, 0_i32, core::sync::atomic::Ordering::Relaxed) };
    unsafe { threadgroup::barrier() };
    let mut l7: i32 = unsafe { a8.load(input % 1, core::sync::atomic::Ordering::Relaxed) };
    unsafe { threadgroup::barrier() };
    [(l7 as u32)]
}

fn run(input: u32, k0: u32) -> [u32; 4] {
    let mut l23: [bool; 3] = [true, true, false];
    l23[((u64::MAX / (k0 as u64)) as usize) % 3] = l23[(k0 as u16) as usize];
    [(-224.0_f32).to_bits(), 0_u32, 0_u32, 0_u32]
}

#[cfg(not(cpu_reference))]
use metal_oxide_device::{AtomicBuffer, AtomicThreadgroupBuffer, ReadBuffer, ThreadgroupBuffer, WriteBuffer, block_idx, kernel, simdgroup, thread_idx, threadgroup};

#[cfg(not(cpu_reference))]
#[kernel(block = (64, 1, 1))]
pub unsafe fn generated(input: ReadBuffer<u32>, output: WriteBuffer<u32>, n: u32, counters: AtomicBuffer<i32>) {
    let i = block_idx().x * 64 + thread_idx().x;
    let value = if i < n { unsafe { input.load_unchecked(i) } } else { 0 };
    let a8 = threadgroup::shared_atomic::<i32, 1>();
    let k = cooperate(value, thread_idx().x, counters, a8);
    if i < n {
        let words = run(value, k[0]);
        unsafe {
        output.store_unchecked(i * 4 + 0, words[0]);
        output.store_unchecked(i * 4 + 1, words[1]);
        output.store_unchecked(i * 4 + 2, words[2]);
        output.store_unchecked(i * 4 + 3, words[3]);
        }
    }
}

#[cfg(cpu_reference)]
use model::{AtomicBuffer, AtomicThreadgroupBuffer, ThreadgroupBuffer, simdgroup, thread_idx, threadgroup};

#[cfg(cpu_reference)]
mod model {
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
            .collect();
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
        let values = values.collect();
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
}

#[cfg(cpu_reference)]
fn main() {
    let inputs = [0_u32, 1_u32, 4294967295_u32, 2147483647_u32, 2147483648_u32, 65535_u32, 65536_u32, 2451366367_u32, 950814130_u32, 2404739433_u32, 3607503796_u32, 378305411_u32, 1787980550_u32, 1880671917_u32, 868153384_u32, 803831143_u32, 4065352602_u32, 2165110321_u32, 4277664220_u32, 1662012299_u32, 586099566_u32, 2342517749_u32, 2900319440_u32, 319916527_u32, 2845781634_u32, 3502119929_u32, 1351390468_u32, 3830848659_u32, 4002150614_u32, 1804982845_u32, 3086243448_u32, 1610819447_u32, 955276906_u32, 513538753_u32, 1830589740_u32, 3147697819_u32, 711619390_u32, 248480133_u32, 1351648544_u32, 3908072959_u32, 1703906130_u32, 826126985_u32, 3239352916_u32, 1378632099_u32, 1941977766_u32, 1858474445_u32, 2224720072_u32, 3863975303_u32, 1044141370_u32, 3256771409_u32, 3866444924_u32, 1506334123_u32, 2927030030_u32, 3558255381_u32, 3201106288_u32, 3210389007_u32, 1235901474_u32, 3659345177_u32, 1465070500_u32, 3711953587_u32, 3685494902_u32, 3137498461_u32, 1630965528_u32, 3801073559_u32, 2231451658_u32, 3187486689_u32, 1884751820_u32, 1625399483_u32, 3224477406_u32, 1206810789_u32, 2948190656_u32, 2922453535_u32, 844931314_u32, 1793428393_u32, 3275579636_u32, 318904259_u32, 1527568966_u32, 1273228525_u32, 177039720_u32, 1757724071_u32, 2001477338_u32, 2732679281_u32, 3109478172_u32, 209285067_u32, 737641134_u32, 3641698869_u32, 2025584144_u32, 1234489903_u32, 3411237314_u32, 2304757305_u32, 2248680004_u32, 2703476947_u32, 1239365654_u32, 1732481149_u32, 2191792056_u32, 2435807159_u32, 912912810_u32, 3484677377_u32, 1766443628_u32, 3227599579_u32, 1231395454_u32, 2289303493_u32, 1551396448_u32, 2984693311_u32, 1601930898_u32, 1905587401_u32, 1109948308_u32, 1704328675_u32, 4083173862_u32, 758798349_u32, 1623169544_u32, 2287007175_u32, 1793570938_u32, 27127185_u32, 1900333500_u32, 413800939_u32, 3411600974_u32, 2035530069_u32, 3671341744_u32, 2747797071_u32, 2835490658_u32, 2244869977_u32, 190013668_u32, 2122954483_u32, 1712132022_u32, 673268637_u32, 560263256_u32, 2666149847_u32, 345142090_u32, 3825748513_u32, 1537006860_u32, 4188337403_u32, 2414765598_u32, 2166888165_u32, 16014080_u32, 2283377247_u32, 2706652210_u32, 3734236649_u32, 3061702196_u32, 1517286403_u32, 1134725510_u32, 3700511533_u32, 1985551016_u32, 1509703143_u32, 1517756954_u32, 2694580913_u32, 2615608412_u32, 2682442763_u32, 1887640046_u32, 3081377909_u32, 3107835728_u32, 2214353519_u32, 1360971010_u32, 3668951161_u32, 702345092_u32, 3355030803_u32, 2434643798_u32, 1499694781_u32, 2527140088_u32, 2194147319_u32, 1524810986_u32, 1866608449_u32, 3740811180_u32, 2956244763_u32, 297009598_u32, 114474501_u32, 458594208_u32, 2295424639_u32, 2689515986_u32, 2938795785_u32, 2900726996_u32, 102234659_u32, 2260441382_u32, 3350257229_u32, 2390879048_u32, 919508487_u32, 4217527226_u32, 639875025_u32, 1524993788_u32, 1202429483_u32, 3009289614_u32, 3134916501_u32, 3156259824_u32, 1681503887_u32, 638661282_u32, 4149023129_u32, 944557604_u32, 3556492083_u32, 1610166006_u32, 4208089565_u32, 2407567768_u32, 4157663255_u32, 3099780746_u32, 3383338081_u32, 520352332_u32, 194545979_u32, 10382686_u32, 305919269_u32, 4257489984_u32, 3491121823_u32, 4174583666_u32, 1151373353_u32, 3322387316_u32, 2165706819_u32, 731084998_u32, 6355309_u32, 1080167400_u32, 3150970407_u32, 1997628762_u32, 477951217_u32, 1676175772_u32, 965507147_u32, 3755077934_u32, 3450767029_u32, 1604642960_u32, 2189959855_u32, 2118217794_u32, 2934829753_u32, 2706046148_u32, 4246200659_u32, 2609346198_u32, 2956311805_u32, 220944952_u32, 4245477431_u32, 1369102378_u32, 797354369_u32, 3681023212_u32, 3076025179_u32, 1117104382_u32, 3224091717_u32, 3462922464_u32, 1039154879_u32, 994655506_u32, 1681799497_u32, 3062624788_u32, 3891410531_u32, 3574736998_u32, 1013556365_u32, 1998683272_u32, 3389552199_u32, 1793867514_u32, 4059021841_u32, 9933884_u32, 588079723_u32, 4123432142_u32, 1592563157_u32, 4092751152_u32, 3773800143_u32, 2753002978_u32, 953836505_u32, 3816782692_u32, 2715160435_u32, 1085312566_u32, 3243618333_u32, 1690923736_u32, 3582187607_u32, 3167918538_u32];
    let n = inputs.len() as u32;
    let counters = model::AtomicBuffer::<i32>::new(&model::COUNTERS);
    let outcomes = model::launch(64, n, |i| {
        let value = if i < n { inputs[i as usize] } else { 0 };
        let a8 = threadgroup::shared_atomic::<i32, 1>();
        let k = cooperate(value, thread_idx().x, counters, a8);
        (i < n).then(|| run(value, k[0]))
    });
    for outcome in outcomes {
        match outcome {
            Ok(Some(words)) => println!("{}", words.map(|word| word.to_string()).join(" ")),
            Ok(None) => {}
            Err((line, column)) => println!("panic {line}:{column}"),
        }
    }
    let words = counters.words().iter().map(u32::to_string).collect::<Vec<_>>();
    println!("counters {}", words.join(" "));
}
