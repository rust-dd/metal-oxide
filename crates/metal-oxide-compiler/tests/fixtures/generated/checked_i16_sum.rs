// Minimized from seed 0xcd7bf5c723fd40ff: LLVM turned the range test of an i16 checked sum into
// llvm.sadd.with.overflow.i16, whose overflow Apple GPUs reported for sums that fit.
// words: exact
// input: i16
// output: u16
// block: 128 1 1
// overflow checks: off
// cooperative
// counters: u32
#![cfg_attr(not(cpu_reference), no_std)]
#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, unused_imports, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]

fn h0(p0: i16, p1: u32, p2: &bool) -> Result<f32, usize> {
    let mut l1: i16 = p0.checked_add((p0 / 16383_i16)).expect("generated expectation");
    Ok::<f32, usize>(0.0_f32)
}

fn cooperate(input: i16, lane: u32, counters: AtomicBuffer<u32>, s21: ThreadgroupBuffer<f32, 128>, a28: AtomicThreadgroupBuffer<u32, 3>) -> [u32; 3] {
    [0_u32, 0_u32, 0_u32]
}

fn run(input: i16, k0: u32, k1: u32, k2: u32) -> [u16; 1] {
    let mut l34: bool = true;
    let mut l49: Result<f32, usize> = h0(input, k1.checked_mul(k2).unwrap(), &l34);
    [0_u16]
}

#[cfg(not(cpu_reference))]
use metal_oxide_device::{AtomicBuffer, AtomicThreadgroupBuffer, Float, ReadBuffer, ThreadgroupBuffer, WriteBuffer, block_idx, kernel, simdgroup, thread_idx, threadgroup};

#[cfg(not(cpu_reference))]
#[kernel(block = (128, 1, 1))]
pub unsafe fn generated(input: ReadBuffer<i16>, output: WriteBuffer<u16>, n: u32, counters: AtomicBuffer<u32>) {
    let i = block_idx().x * 128 + thread_idx().x;
    let value = if i < n { unsafe { input.load_unchecked(i) } } else { 0_i16 };
    let s21 = threadgroup::shared::<f32, 128>();
    let a28 = threadgroup::shared_atomic::<u32, 3>();
    let k = cooperate(value, thread_idx().x, counters, s21, a28);
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
    let inputs = [0_u32, 1_u32, 4294967295_u32, 2147483647_u32, 2147483648_u32, 65535_u32, 65536_u32, 3086120759_u32, 3550341930_u32, 3728777345_u32, 4064509932_u32, 3974134363_u32, 2809687038_u32, 1147249477_u32, 2385427424_u32, 1820966335_u32, 682536978_u32, 423039049_u32, 198761748_u32, 3581683043_u32, 827149158_u32, 3864816525_u32, 1119852424_u32, 3268531527_u32, 2415923706_u32, 1426290961_u32, 3263291196_u32, 2832544107_u32, 2489718734_u32, 3325574357_u32, 2790516784_u32, 3387219407_u32, 1453699298_u32, 3982818009_u32, 95626852_u32, 2811739763_u32, 3460329782_u32, 1897281309_u32, 911887832_u32, 2485188439_u32, 203893962_u32, 3590239649_u32, 1400197772_u32, 1207168123_u32, 739112862_u32, 943422053_u32, 3189074048_u32, 2678604255_u32, 3506447794_u32, 2530724201_u32, 2330264500_u32, 3860716419_u32, 1093946118_u32, 3046188717_u32, 287037480_u32, 1323359591_u32, 1965046682_u32, 3638253105_u32, 246602204_u32, 2228071307_u32, 2915896174_u32, 3760479221_u32, 366492880_u32, 1895098863_u32, 1718292098_u32, 2186836985_u32, 933459204_u32, 2646572179_u32, 3039156438_u32, 2963683901_u32, 537474680_u32, 3662841719_u32, 1865956970_u32, 1669487297_u32, 386889004_u32, 36925083_u32, 2755679038_u32, 1441522053_u32, 2604754208_u32, 220976639_u32, 1154706258_u32, 1928336009_u32, 2304997972_u32, 3412959651_u32, 139598502_u32, 4184764877_u32, 90663112_u32, 4059493767_u32, 3858511162_u32, 2585576273_u32, 2560631932_u32, 1530279339_u32, 3041185550_u32, 332073749_u32, 959832432_u32, 2683176463_u32, 3378920482_u32, 300387609_u32, 786043812_u32, 1817791155_u32, 2620788854_u32, 1363332445_u32, 146501400_u32, 898574231_u32, 3694731274_u32, 1326682081_u32, 414836684_u32, 858194107_u32, 1912045278_u32, 104519845_u32, 668643776_u32, 3239868959_u32, 1351647474_u32, 2127031209_u32, 1476149492_u32, 2381543363_u32, 630110790_u32, 867978477_u32, 1724533096_u32, 958143911_u32, 1006370522_u32, 2467283057_u32, 3320954652_u32, 1167511499_u32, 649421486_u32, 468994613_u32, 1016387088_u32, 2228761135_u32, 100612546_u32, 3747229241_u32, 100598340_u32, 1575823571_u32, 2086213654_u32, 1723068541_u32, 2711156664_u32, 2497632183_u32, 2794574250_u32, 1157256449_u32, 3357321836_u32, 1985389275_u32, 2870704766_u32, 3589300165_u32, 973688416_u32, 2340564543_u32, 3030345362_u32, 4020720841_u32, 1532450708_u32, 976707043_u32, 1808935398_u32, 2319628301_u32, 3496770056_u32, 4116207047_u32, 1149737082_u32, 2717666705_u32, 2421139900_u32, 3782703595_u32, 839304782_u32, 3613923669_u32, 538778288_u32, 297496143_u32, 2526938978_u32, 2448973657_u32, 3954852068_u32, 3238205171_u32, 2188832694_u32, 2831262621_u32, 4022980696_u32, 726243287_u32, 2510967626_u32, 3581848097_u32, 685751564_u32, 3947517179_u32, 429147678_u32, 1976858341_u32, 2079668992_u32, 6615647_u32, 627579954_u32, 1591102953_u32, 203210292_u32, 3770766339_u32, 3791939974_u32, 3335107373_u32, 829815464_u32, 1001627111_u32, 1090978330_u32, 2301258417_u32, 2237785180_u32, 1348482059_u32, 3139500526_u32, 1223587957_u32, 3086397264_u32, 4238326383_u32, 3920274690_u32, 889856121_u32, 582209412_u32, 3894613267_u32, 259495766_u32, 1932781245_u32, 1282799864_u32, 1876387831_u32, 3840582890_u32, 1961398081_u32, 3534430124_u32, 898242331_u32, 994697662_u32, 3132038661_u32, 1808168864_u32, 2009908863_u32, 1263706578_u32, 3382499081_u32, 4143215828_u32, 2518243875_u32, 3272440102_u32, 1461272141_u32, 3440298824_u32, 1698035207_u32, 3873586106_u32, 4007794641_u32, 3335548668_u32, 936967723_u32, 1508638094_u32, 2538563477_u32, 2890503152_u32, 3213694607_u32, 1636635298_u32, 936866201_u32, 26455588_u32, 701834035_u32, 3091402486_u32, 3318921693_u32, 1690661272_u32, 495994903_u32, 1136313994_u32, 2071987297_u32, 4045853260_u32, 2090723643_u32, 1109675358_u32, 2638763301_u32, 1537541184_u32, 230795935_u32, 3267819378_u32, 2437083177_u32, 3162930036_u32, 1925673027_u32, 2111133894_u32, 1291409773_u32, 979299304_u32, 250043943_u32, 1602307418_u32, 1567315185_u32, 172181916_u32, 3244939339_u32, 1515180334_u32, 223536821_u32, 2034092176_u32, 2559279791_u32, 1420644418_u32];
    let n = inputs.len() as u32;
    let counters = model::AtomicBuffer::<u32>::new(&model::COUNTERS);
    let outcomes = model::launch(128, n, |i| {
        flush_subnormals();
        let value = if i < n { inputs[i as usize] as i16 } else { 0_i16 };
        let s21 = threadgroup::shared::<f32, 128>();
        let a28 = threadgroup::shared_atomic::<u32, 3>();
        let k = cooperate(value, thread_idx().x, counters, s21, a28);
        (i < n).then(|| run(value, k[0], k[1], k[2]))
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
