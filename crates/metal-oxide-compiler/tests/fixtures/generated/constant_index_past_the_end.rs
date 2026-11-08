// Minimized from seed 0xb28a46fd4619c216: an array indexed past its end by a constant, behind a
// bounds check that always fails, was rejected as an invalid projection.
// words: exact exact exact exact
// input: i64
// output: i64
// block: 128 1 1
// overflow checks: off
// cooperative
// counters: u32
#![cfg_attr(not(cpu_reference), no_std)]
#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, unused_imports, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]

fn h1<const K: u32>(p0: [i64; 1], p1: Option<isize>) -> i64 {
    p0[1_usize as usize]
}

fn cooperate(input: i64, lane: u32, counters: AtomicBuffer<u32>, s6: ThreadgroupBuffer<f32, 128>) -> [u32; 1] {
    [0_u32]
}

fn run(input: i64, k0: u32) -> [i64; 2] {
    let mut l21: isize = 0_isize;
    [h1::<35>([input], false.then_some(l21)), 0_i64]
}

#[cfg(not(cpu_reference))]
use metal_oxide_device::{AtomicBuffer, AtomicThreadgroupBuffer, Float, ReadBuffer, ThreadgroupBuffer, WriteBuffer, block_idx, kernel, simdgroup, thread_idx, threadgroup};

#[cfg(not(cpu_reference))]
#[kernel(block = (128, 1, 1))]
pub unsafe fn generated(input: ReadBuffer<i64>, output: WriteBuffer<i64>, n: u32, counters: AtomicBuffer<u32>) {
    let i = block_idx().x * 128 + thread_idx().x;
    let value = if i < n { unsafe { input.load_unchecked(i) } } else { 0_i64 };
    let s6 = threadgroup::shared::<f32, 128>();
    let k = cooperate(value, thread_idx().x, counters, s6);
    if i < n {
        let words = run(value, k[0]);
        unsafe {
        output.store_unchecked(i * 2 + 0, words[0]);
        output.store_unchecked(i * 2 + 1, words[1]);
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
    let inputs = [0_u32, 1_u32, 4294967295_u32, 2147483647_u32, 2147483648_u32, 65535_u32, 65536_u32, 1775588686_u32, 4041170005_u32, 2061337008_u32, 1703586127_u32, 749078114_u32, 1190810201_u32, 2661652452_u32, 446726643_u32, 991387318_u32, 3424732829_u32, 3247904600_u32, 2348823255_u32, 1672622666_u32, 901709089_u32, 3358979084_u32, 3147112443_u32, 1296209182_u32, 1076394469_u32, 2255184384_u32, 1794043231_u32, 191617842_u32, 841023721_u32, 87676212_u32, 1566932739_u32, 3531380870_u32, 1990071853_u32, 1773207976_u32, 1249704167_u32, 511877402_u32, 2429255089_u32, 4045527900_u32, 1686748939_u32, 3785296110_u32, 3998169973_u32, 3063060048_u32, 2543089007_u32, 2375689218_u32, 1540199289_u32, 896706180_u32, 3833462803_u32, 213354070_u32, 531433917_u32, 3180249080_u32, 2086896375_u32, 4257874922_u32, 4279919169_u32, 1279579820_u32, 2151834139_u32, 3347558590_u32, 2974589189_u32, 2130189984_u32, 2296136063_u32, 1461575890_u32, 2231962121_u32, 2462387156_u32, 1524303139_u32, 2651138086_u32, 1013391693_u32, 2775929416_u32, 1798623495_u32, 588602042_u32, 1658104529_u32, 2880693756_u32, 399509803_u32, 987335822_u32, 2707008149_u32, 3498113776_u32, 3093778831_u32, 3139703202_u32, 3575407769_u32, 837158180_u32, 3954019891_u32, 1038253558_u32, 3961871581_u32, 787069080_u32, 3299992343_u32, 1194435978_u32, 3429062497_u32, 3133409612_u32, 3162747963_u32, 3803290718_u32, 3371128869_u32, 1766992704_u32, 2350294431_u32, 2350495346_u32, 1771086633_u32, 684364404_u32, 1382456131_u32, 1997309894_u32, 2290138221_u32, 3702584040_u32, 2866526503_u32, 2318132314_u32, 175073265_u32, 1309294748_u32, 2548982603_u32, 2708267054_u32, 3032918453_u32, 617656208_u32, 1712012719_u32, 3953905474_u32, 3861824953_u32, 3735532484_u32, 3937872979_u32, 2004895126_u32, 894582781_u32, 4140826936_u32, 3287503671_u32, 1129387818_u32, 971103361_u32, 3009121260_u32, 2078348891_u32, 1688754174_u32, 64528197_u32, 1268877280_u32, 3030784447_u32, 449272850_u32, 83872841_u32, 1542613268_u32, 3230679395_u32, 3766138726_u32, 1890772877_u32, 3666600840_u32, 1249489223_u32, 3509468666_u32, 1440986385_u32, 30206780_u32, 4067216747_u32, 4114678734_u32, 4150147285_u32, 3819549744_u32, 42482127_u32, 1234787554_u32, 642561753_u32, 2589962852_u32, 2686638707_u32, 2524503350_u32, 2734404381_u32, 1248434648_u32, 489673559_u32, 1771067594_u32, 3353211297_u32, 1773362828_u32, 1306691707_u32, 3344163742_u32, 47216229_u32, 930944_u32, 110271967_u32, 1737413042_u32, 3768485225_u32, 3571748788_u32, 635434883_u32, 2436347654_u32, 2932187821_u32, 3190694952_u32, 3474372967_u32, 3917768602_u32, 4151939633_u32, 2225333724_u32, 3100854155_u32, 3393058670_u32, 1541953525_u32, 288622800_u32, 3018212847_u32, 4082416258_u32, 2428385273_u32, 3619029252_u32, 3464009875_u32, 2880718038_u32, 2162800189_u32, 3000926840_u32, 3077530487_u32, 2773969514_u32, 3667891905_u32, 2775810348_u32, 1175456411_u32, 949763902_u32, 2325667205_u32, 885010720_u32, 1239691775_u32, 3403159378_u32, 3920940681_u32, 1346963028_u32, 115195299_u32, 1949009574_u32, 2692804045_u32, 4206867656_u32, 29020551_u32, 949379386_u32, 2239299409_u32, 674705532_u32, 11129259_u32, 1749793550_u32, 4180477717_u32, 2241310064_u32, 1680695823_u32, 4215662626_u32, 2227915033_u32, 491954084_u32, 1010853555_u32, 4229594230_u32, 3202632029_u32, 223787800_u32, 3183084439_u32, 1443777546_u32, 3127823329_u32, 3259233228_u32, 4226915515_u32, 2590599902_u32, 2215367845_u32, 1809808832_u32, 178443807_u32, 2433428722_u32, 1904912297_u32, 2663894260_u32, 3960574915_u32, 2822645318_u32, 1202505965_u32, 1466471784_u32, 4016961959_u32, 2546739930_u32, 2049571953_u32, 3526282012_u32, 1668836299_u32, 3411168942_u32, 167003701_u32, 3975979536_u32, 3244658223_u32, 1742005698_u32, 3317426745_u32, 98405956_u32, 3020047571_u32, 9667606_u32, 4038290557_u32, 2331657144_u32, 2669130679_u32, 1327300010_u32, 2475921665_u32, 2949428844_u32, 2333032155_u32, 2191746686_u32, 4225719237_u32, 4220787296_u32, 2564196927_u32, 4203745938_u32, 761794761_u32, 2768823188_u32, 1644394979_u32, 2153223654_u32, 1242641421_u32];
    let n = inputs.len() as u32;
    let counters = model::AtomicBuffer::<u32>::new(&model::COUNTERS);
    let outcomes = model::launch(128, n, |i| {
        flush_subnormals();
        let value = if i < n { inputs[i as usize] as i64 } else { 0_i64 };
        let s6 = threadgroup::shared::<f32, 128>();
        let k = cooperate(value, thread_idx().x, counters, s6);
        (i < n).then(|| run(value, k[0]))
    });
    for outcome in outcomes {
        match outcome {
            Ok(Some(words)) => println!("{}", words.map(|word| format!("{} {}", word as u32, word as u64 >> 32)).join(" ")),
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
