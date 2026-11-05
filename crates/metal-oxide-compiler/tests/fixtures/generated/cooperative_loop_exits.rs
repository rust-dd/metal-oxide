// Minimized from seed 0xa79722095e43ea5a: consecutive loops with early returns repeated the code
// after them per exit until the compiler rejected the duplication.
// words: exact
// input: i8
// output: u32
// block: 16 2 2
// overflow checks: on
// cooperative
#![cfg_attr(not(cpu_reference), no_std)]
#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, unused_imports, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]

fn cooperate(input: i8, lane: u32) -> [u32; 2] {
    [0_u32, 0_u32]
}

fn run(input: i8, k0: u32, k1: u32) -> [u32; 1] {
    let mut w10 = 0_u32;
    'o9: while w10 < 0_isize.trailing_zeros() && (0_i64 < 0_i64.wrapping_add(0_i64)) {
        w10 += 1;
        if ((0_u32 .. 0_u32).map((move |a14: u32| -> Option<usize> { Some(0_usize) })).fold(0_u16, (move |a15: u16, a16: Option<usize>| -> u16 { a15 })) < 0_u16) {
            return [k1];
        }
    }
    if (0_u32 ..= k1).all((move |a25: u32| -> bool { true })) {
        return [w10];
    }
    let mut w29 = 0_u32;
    'o28: while w29 < 0_u32.checked_sub(k1).unwrap() && (0.0_f32 > 0.0_f32) {
        w29 += 1;
        'o30: for c31 in 0..1_u32 {
        }
        return [0_u32];
    }
    if ((k0 as usize) != 0_usize) {
        return [0_u32];
    }
    [(if true { w10 } else { 1_u32 }).wrapping_add((3848686535_u32 >> (k0 as u32))).overflowing_sub((match 4346124106238897296_usize.checked_pow((usize::MAX as u32)) { Some(b58) => 0_u32.wrapping_div((k0 | 1_u32)), None => k0 })).0]
}

#[cfg(not(cpu_reference))]
use metal_oxide_device::{AtomicBuffer, AtomicThreadgroupBuffer, Float, ReadBuffer, ThreadgroupBuffer, WriteBuffer, block_idx, kernel, simdgroup, thread_idx, threadgroup};

#[cfg(not(cpu_reference))]
#[kernel(block = (16, 2, 2))]
pub unsafe fn generated(input: ReadBuffer<i8>, output: WriteBuffer<u32>, n: u32) {
    let lane = (thread_idx().z * 2 + thread_idx().y) * 16 + thread_idx().x;
    let i = block_idx().x.wrapping_mul(64).wrapping_add(lane);
    let value = if i < n { unsafe { input.load_unchecked(i) } } else { 0_i8 };
    let k = cooperate(value, lane);
    if i < n {
        let words = run(value, k[0], k[1]);
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
    let inputs = [0_u32, 1_u32, 4294967295_u32, 2147483647_u32, 2147483648_u32, 65535_u32, 65536_u32, 1143550358_u32, 4092476413_u32, 1730399544_u32, 3346837303_u32, 670221098_u32, 211785857_u32, 1541806060_u32, 3937546843_u32, 309249022_u32, 1911823173_u32, 4067349472_u32, 1700459967_u32, 2677966866_u32, 3413444681_u32, 30307604_u32, 92593507_u32, 319726438_u32, 4265468813_u32, 117616520_u32, 3452570951_u32, 2095770106_u32, 117469457_u32, 3475666748_u32, 1375005035_u32, 3327291342_u32, 306754773_u32, 3405381680_u32, 736368079_u32, 2028701922_u32, 2533421785_u32, 4285447780_u32, 3970856563_u32, 1438107958_u32, 2000006941_u32, 3351471576_u32, 772222807_u32, 254280906_u32, 1796844961_u32, 3701729932_u32, 2816582779_u32, 3127922590_u32, 977924709_u32, 1419870336_u32, 1851095519_u32, 3424386482_u32, 2270282089_u32, 2447809460_u32, 3150091139_u32, 3017908998_u32, 829333165_u32, 1361768488_u32, 3946124647_u32, 1807160218_u32, 2425638449_u32, 1307013596_u32, 2265539467_u32, 2384814958_u32, 263246837_u32, 801789136_u32, 2412782063_u32, 3353142914_u32, 3912266745_u32, 3043691780_u32, 1601320083_u32, 3095998678_u32, 163689021_u32, 1346260600_u32, 3432300407_u32, 2531595882_u32, 1566134977_u32, 2781482284_u32, 1916610203_u32, 739158846_u32, 175528325_u32, 2853797152_u32, 1068830207_u32, 4196090706_u32, 1604717193_u32, 1198592596_u32, 3031325091_u32, 2716562086_u32, 2001201613_u32, 3343024328_u32, 1839672711_u32, 3695119674_u32, 3583097681_u32, 1885387900_u32, 3834506667_u32, 2584291086_u32, 2228454165_u32, 242481520_u32, 2309307919_u32, 537138210_u32, 1945863449_u32, 1454222244_u32, 2561147571_u32, 830826614_u32, 458900829_u32, 1572636440_u32, 1311596439_u32, 2660366346_u32, 2879817697_u32, 2466283468_u32, 1632417979_u32, 3375486686_u32, 1262571685_u32, 4110338496_u32, 3850482207_u32, 4127480050_u32, 632939433_u32, 1930811636_u32, 4174393283_u32, 2081690182_u32, 1368507629_u32, 2960220520_u32, 3794294183_u32, 669701850_u32, 3493902449_u32, 2621331228_u32, 2215282635_u32, 1709554350_u32, 746111541_u32, 2468277776_u32, 1198221871_u32, 130617794_u32, 2057971257_u32, 4259224132_u32, 3805798611_u32, 3113512982_u32, 3217516669_u32, 2707820472_u32, 2715357111_u32, 2039916970_u32, 38358273_u32, 334447212_u32, 278420187_u32, 2814497406_u32, 2306004933_u32, 192505440_u32, 801334847_u32, 2151628434_u32, 248859849_u32, 2038230932_u32, 2494567907_u32, 356745702_u32, 3860082701_u32, 2816446984_u32, 432049607_u32, 472019066_u32, 1592346001_u32, 3408413116_u32, 50803179_u32, 564338254_u32, 3848835413_u32, 3186832048_u32, 3140134479_u32, 4187217762_u32, 2760136537_u32, 3651524836_u32, 2357841651_u32, 2107525046_u32, 2932939677_u32, 543352920_u32, 4204777431_u32, 4015729482_u32, 4052328993_u32, 742282508_u32, 2678590715_u32, 2443615774_u32, 1976757989_u32, 2544980736_u32, 2994874975_u32, 1295250482_u32, 3024111081_u32, 3545126452_u32, 2564164611_u32, 1952563590_u32, 4271319853_u32, 2629112488_u32, 695719399_u32, 4199406106_u32, 1137926833_u32, 2328263772_u32, 2198596619_u32, 2972275182_u32, 2993352821_u32, 1662568272_u32, 2599088751_u32, 1674345730_u32, 4241627257_u32, 749430660_u32, 3896888595_u32, 3733647190_u32, 1145003709_u32, 869994744_u32, 3481903095_u32, 994540778_u32, 3987703617_u32, 1939191724_u32, 2986594075_u32, 80230846_u32, 3844707845_u32, 399413152_u32, 3318087295_u32, 88766930_u32, 323145481_u32, 221383892_u32, 432673315_u32, 3562459430_u32, 149075533_u32, 3524911944_u32, 1784032775_u32, 2010452922_u32, 625414097_u32, 4240508668_u32, 2845881899_u32, 2891994510_u32, 3125272469_u32, 716680176_u32, 1622429327_u32, 2832919202_u32, 4074465689_u32, 1301759524_u32, 1284758323_u32, 1900177142_u32, 2145006045_u32, 2297858456_u32, 3684597783_u32, 214146698_u32, 1325595745_u32, 362637900_u32, 1865654587_u32, 856597854_u32, 3493776677_u32, 3724090432_u32, 2241258143_u32, 1156239218_u32, 2368006185_u32, 3352302452_u32, 806627395_u32, 2752164038_u32, 381604205_u32, 3744859112_u32, 2036569639_u32, 3189775706_u32, 2374782193_u32, 3602899356_u32, 2689437771_u32, 1596991790_u32, 1204248245_u32];
    let n = inputs.len() as u32;
    let outcomes = model::launch(64, n, |i| {
        flush_subnormals();
        let value = if i < n { inputs[i as usize] as i8 } else { 0_i8 };
        let k = cooperate(value, thread_idx().x);
        (i < n).then(|| run(value, k[0], k[1]))
    });
    for outcome in outcomes {
        match outcome {
            Ok(Some(words)) => println!("{}", words.map(|word| word.to_string()).join(" ")),
            Ok(None) => {}
            Err((line, column)) => println!("panic {line}:{column}"),
        }
    }
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
