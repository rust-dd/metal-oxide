// Minimized from seed 0xe7455123b3977ba2: early returns in loops repeated the empty shared return
// block past the duplication limit, so the compiler rejected the program.
// words: exact
// input: u16
// output: i32
// block: 8 8 1
// overflow checks: off
// cooperative
// counters: i32
#![cfg_attr(not(cpu_reference), no_std)]
#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, unused_imports, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]

fn cooperate(input: u16, lane: u32, counters: AtomicBuffer<i32>, a6: AtomicThreadgroupBuffer<i32, 2>, a8: AtomicThreadgroupBuffer<u32, 3>) -> [u32; 3] {
    [0_u32, 0_u32, 0_u32]
}

fn run(input: u16, k0: u32, k1: u32, k2: u32) -> [i32; 1] {
    let mut l9: i64 = 0_i64;
    if matches!(Ok::<i8, u16>(0_i8), Err(..)) {
        return [0_i32];
    }
    if (if (if true { false } else { false }) { true } else { false }) {
        if (l9.checked_div(l9).unwrap_or(0_i64) != 0_i64) {
            let mut l18: i32 = 0_i32;
            return [l18];
        }
    } else {
        'o23: for x24 in (0_u32 .. 0_u32) {
            if ((-7017282516908321438_isize) | 6402177006034088453_isize).overflowing_shr(((-1_i32) as u32)).1 {
                return [0_i32];
            }
        }
    }
    'o29: for c30 in 0..4_u32 {
    }
    let mut w32 = 0_u32;
    'o31: while w32 < 0_u64.count_ones() && (0_usize < (match Ok::<u32, isize>(0_u32) { Ok(b33) => 0_usize, Err(b34) => 14085676656752857559_usize })) {
        w32 += 1;
        return [0_i32];
    }
    'o50: for c51 in 0..4_u32 {
        assert!(((input as isize) < 0_isize), "generated assertion");
        return [0_i32];
    }
    [0_i32]
}

#[cfg(not(cpu_reference))]
use metal_oxide_device::{AtomicBuffer, AtomicThreadgroupBuffer, Float, ReadBuffer, ThreadgroupBuffer, WriteBuffer, block_idx, kernel, simdgroup, thread_idx, threadgroup};

#[cfg(not(cpu_reference))]
#[kernel(block = (8, 8, 1))]
pub unsafe fn generated(input: ReadBuffer<u16>, output: WriteBuffer<i32>, n: u32, counters: AtomicBuffer<i32>) {
    let lane = (thread_idx().z * 8 + thread_idx().y) * 8 + thread_idx().x;
    let i = block_idx().x.wrapping_mul(64).wrapping_add(lane);
    let value = if i < n { unsafe { input.load_unchecked(i) } } else { 0_u16 };
    let a6 = threadgroup::shared_atomic::<i32, 2>();
    let a8 = threadgroup::shared_atomic::<u32, 3>();
    let k = cooperate(value, lane, counters, a6, a8);
    if i < n {
        let words = run(value, k[0], k[1], k[2]);
        unsafe {
        output.store_unchecked(i * 1 + 0, words[0]);
        }
    }
}

#[cfg(cpu_reference)]
use model::{AtomicBuffer, AtomicThreadgroupBuffer, ThreadgroupBuffer, simdgroup, thread_idx, threadgroup};

#[cfg(cpu_reference)]
#[path = "../../support/generate/model.rs"]
mod model;

#[cfg(cpu_reference)]
fn main() {
    let inputs = [0_u32, 1_u32, 4294967295_u32, 2147483647_u32, 2147483648_u32, 65535_u32, 65536_u32, 1856209388_u32, 2962003035_u32, 3012673022_u32, 4199992645_u32, 2489179616_u32, 3598350271_u32, 2651070994_u32, 1006230089_u32, 641277716_u32, 4171988835_u32, 2727402854_u32, 3572946317_u32, 1087568264_u32, 4107915079_u32, 2986591226_u32, 3927933713_u32, 2062180668_u32, 1542441835_u32, 1344006606_u32, 1109386965_u32, 3637742128_u32, 116135887_u32, 4213154530_u32, 2147632345_u32, 1563158628_u32, 671455347_u32, 1655735094_u32, 1676878109_u32, 1791897560_u32, 2081386839_u32, 3252641482_u32, 1447378849_u32, 1812354188_u32, 2149385851_u32, 4034939294_u32, 3233129573_u32, 1929742976_u32, 3479567327_u32, 4290680754_u32, 4193227625_u32, 3348514228_u32, 225064323_u32, 1978719494_u32, 3343945901_u32, 2517645864_u32, 5625703_u32, 2108485018_u32, 3102498865_u32, 864836572_u32, 3215279499_u32, 2619071854_u32, 3822126581_u32, 2874728144_u32, 453583855_u32, 3264086146_u32, 3196765689_u32, 2309830404_u32, 2643089043_u32, 3478055638_u32, 602416189_u32, 396236920_u32, 3505258871_u32, 3518591082_u32, 3221087425_u32, 2085713708_u32, 2774108315_u32, 2777392446_u32, 1802177413_u32, 2003994400_u32, 557223935_u32, 3111837010_u32, 953998473_u32, 2833628244_u32, 3176594339_u32, 352824486_u32, 953342925_u32, 2579286728_u32, 291041159_u32, 4052870970_u32, 933416273_u32, 4201293436_u32, 3024801707_u32, 2058133774_u32, 4189893909_u32, 3392635760_u32, 1972322319_u32, 2310155810_u32, 3123631897_u32, 1542772132_u32, 1375806643_u32, 1889018486_u32, 4016682845_u32, 1810977048_u32, 3377963415_u32, 671306250_u32, 2588154337_u32, 3140282828_u32, 2009733819_u32, 1956368606_u32, 2738815653_u32, 3821851584_u32, 401880095_u32, 3617649394_u32, 168408489_u32, 1523548916_u32, 1358549443_u32, 1589464134_u32, 3947214573_u32, 3540036456_u32, 4109047719_u32, 2803544282_u32, 2693450353_u32, 1965900060_u32, 2473093579_u32, 2813736110_u32, 2125132853_u32, 2711058448_u32, 2223402031_u32, 385100738_u32, 4130764857_u32, 782943300_u32, 3488744147_u32, 2138319382_u32, 440392317_u32, 1492113848_u32, 2488553911_u32, 1478820778_u32, 3512721153_u32, 3628117100_u32, 3539385563_u32, 3013135486_u32, 888768965_u32, 3460067424_u32, 3282495551_u32, 1519966354_u32, 3804110537_u32, 2180749716_u32, 349878243_u32, 1710865382_u32, 1148264973_u32, 486792200_u32, 3155447751_u32, 2880391802_u32, 300701585_u32, 3715902396_u32, 981865451_u32, 2468386894_u32, 2439234389_u32, 1906359472_u32, 2046208079_u32, 616298850_u32, 1508485465_u32, 1886847716_u32, 2783244531_u32, 1758134710_u32, 2620558749_u32, 599896664_u32, 2766934487_u32, 1770423626_u32, 899232801_u32, 389332748_u32, 375877371_u32, 1818925086_u32, 291676389_u32, 552164608_u32, 4166429791_u32, 919244338_u32, 2621577193_u32, 3608281140_u32, 1501670915_u32, 1906637702_u32, 1310530861_u32, 2800639144_u32, 3151796199_u32, 2159522842_u32, 2383377585_u32, 16737884_u32, 187420171_u32, 1624493038_u32, 1957634677_u32, 1316704592_u32, 1273492591_u32, 1634834178_u32, 814770809_u32, 2751566212_u32, 1717694227_u32, 3448023382_u32, 285855933_u32, 1703860984_u32, 2988892663_u32, 3616528106_u32, 2067274049_u32, 1337008556_u32, 631517467_u32, 1559835582_u32, 1296189445_u32, 2583402912_u32, 1309376639_u32, 1419645906_u32, 1943904521_u32, 649770708_u32, 1937159203_u32, 4238805798_u32, 3054610509_u32, 4092079432_u32, 2370564103_u32, 463203770_u32, 4215022033_u32, 1096741116_u32, 3940653099_u32, 3904360334_u32, 2404716949_u32, 542062064_u32, 3026342031_u32, 1870464162_u32, 695384985_u32, 1609723940_u32, 2817588531_u32, 2190024950_u32, 3096290269_u32, 192893848_u32, 3071066647_u32, 2443769994_u32, 981765729_u32, 3982349388_u32, 1748304699_u32, 4146896734_u32, 559972133_u32, 3415942720_u32, 550331551_u32, 2424007026_u32, 181975593_u32, 1369292148_u32, 3141651011_u32, 3051875014_u32, 3457598317_u32, 3675868648_u32, 225073191_u32, 3854825306_u32, 599416561_u32, 2487405468_u32, 1127182923_u32, 3065291566_u32, 1813876917_u32, 2739169936_u32, 829275311_u32, 3846586946_u32, 1473813689_u32, 1324457668_u32, 85597011_u32];
    let n = inputs.len() as u32;
    let counters = model::AtomicBuffer::<i32>::new(&model::COUNTERS);
    let outcomes = model::launch(64, n, |i| {
        flush_subnormals();
        let value = if i < n { inputs[i as usize] as u16 } else { 0_u16 };
        let a6 = threadgroup::shared_atomic::<i32, 2>();
        let a8 = threadgroup::shared_atomic::<u32, 3>();
        let k = cooperate(value, thread_idx().x, counters, a6, a8);
        (i < n).then(|| run(value, k[0], k[1], k[2]))
    });
    for outcome in outcomes {
        match outcome {
            Ok(Some(words)) => println!("{}", words.map(|word| (word as u32).to_string()).join(" ")),
            Ok(None) => {}
            Err((line, column)) => println!("panic {line}:{column}"),
        }
    }
    let words = counters.words().iter().map(u32::to_string).collect::<Vec<_>>();
    println!("counters {}", words.join(" "));
}

#[cfg(cpu_reference)]
fn flush_subnormals() {
    #[cfg(target_arch = "aarch64")]
    if std::env::var_os("METAL_OXIDE_FLUSH_SUBNORMALS").is_some() {
        let control: u64;
        // SAFETY: FPCR.FZ only makes this thread's float operations flush subnormals to zero.
        unsafe {
            core::arch::asm!("mrs {}, fpcr", out(reg) control);
            core::arch::asm!("msr fpcr, {}", in(reg) control | 1 << 24);
        }
    }
}
