// Minimized from seed 0xf93f37bb76ff56a4: a call between the overflow check of a remainder and
// the remainder dropped the facts about its dividend, copied from an array the call could change.
// words: exact
// input: u64
// output: u16
// block: 7 3 1
// overflow checks: off
#![cfg_attr(not(cpu_reference), no_std)]
#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, unused_imports, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]

fn run(input: u64) -> [u16; 1] {
    let mut l35: [(u8, isize, usize,); 3] = [(0_u8, 0_isize, 0_usize,), (0_u8, 0_isize, 0_usize,), (0_u8, 0_isize, 0_usize,)];
    let mut w37 = 0_u32;
    'o36: while w37 < 0_u32 && true {
        w37 += 1;
    }
    let mut l49: ((f32, i8, i32,),) = None::<((f32, i8, i32,),)>.unwrap_or(((0.0_f32, 0_i8, 0_i32,),));
    l35[(l49.0.1.overflowing_pow((input as u32)).0 as usize) % 3].1 = (l35[(1_i16 as usize) % 3].1 % (input as isize));
    let mut w51 = 0_u32;
    'o50: while w51 < w37 && l35[(input as u16) as usize].2.overflowing_neg().1 {
        w51 += 1;
        &mut l35[2..3];
    }
    [0_u16]
}

#[cfg(not(cpu_reference))]
use metal_oxide_device::{Float, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[cfg(not(cpu_reference))]
#[kernel(instances(generated(1)))]
pub unsafe fn generated_body<const ELEMENTS: u32>(input: ReadBuffer<u64>, output: WriteBuffer<u16>, n: u32) {
    let (block, thread) = (block_dim(), thread_idx());
    let i = block_idx().x * (block.x * block.y * block.z) + (thread.z * block.y + thread.y) * block.x + thread.x;
    if i < n {
        let words = run(unsafe { input.load_unchecked(i) });
        unsafe {
        output.store_unchecked(i * ELEMENTS + 0, words[0]);
        }
    }
}

#[cfg(cpu_reference)]
fn main() {
    flush_subnormals();
    let location = std::sync::Arc::new(std::sync::Mutex::new((0, 0)));
    let recorded = std::sync::Arc::clone(&location);
    std::panic::set_hook(Box::new(move |info| {
        let panicked = info.location().unwrap();
        *recorded.lock().unwrap() = (panicked.line(), panicked.column());
    }));
    let inputs = [0_u32, 1_u32, 4294967295_u32, 2147483647_u32, 2147483648_u32, 65535_u32, 65536_u32, 2884451314_u32, 1476356777_u32, 3815224308_u32, 2906279619_u32, 4107380038_u32, 1455700973_u32, 2926275688_u32, 2067588263_u32, 368056794_u32, 2318868337_u32, 1238077980_u32, 1555597003_u32, 2706746798_u32, 2264537397_u32, 861559056_u32, 2316425519_u32, 2440475842_u32, 2456644921_u32, 4027790660_u32, 1212530643_u32, 1845674774_u32, 4185104253_u32, 2349783736_u32, 2689425079_u32, 1945676970_u32, 292910081_u32, 1068973420_u32, 2059540955_u32, 1425704318_u32, 1738927813_u32, 3996881248_u32, 1136760127_u32, 1642176914_u32, 3800454089_u32, 1900300948_u32, 2064757987_u32, 882021606_u32, 3651806989_u32, 3857128712_u32, 3565460679_u32, 756102010_u32, 1740319889_u32, 1859845308_u32, 3432820971_u32, 1439994190_u32, 2059233365_u32, 535567792_u32, 3580925263_u32, 2203779682_u32, 1720917593_u32, 112194532_u32, 2144284147_u32, 3271432886_u32, 4257272477_u32, 2222535512_u32, 1151573719_u32, 1829154378_u32, 1953581345_u32, 1837879308_u32, 3533261819_u32, 1845547294_u32, 2444854757_u32, 2710633984_u32, 67098975_u32, 2605700914_u32, 4078750953_u32, 290140468_u32, 3773770499_u32, 654464134_u32, 1716560429_u32, 1703550376_u32, 1370186983_u32, 2263227674_u32, 2638268849_u32, 3143546716_u32, 1699277579_u32, 1658602734_u32, 3226683253_u32, 697250384_u32, 1351639407_u32, 1584209922_u32, 2733399929_u32, 55247492_u32, 2300750867_u32, 3516678742_u32, 2654115261_u32, 203348984_u32, 2698833655_u32, 3304882154_u32, 1444064833_u32, 287879852_u32, 3313274395_u32, 3305368766_u32, 3968052485_u32, 2609625760_u32, 142814591_u32, 1621089490_u32, 1728763401_u32, 2955175892_u32, 3545936163_u32, 1728635942_u32, 1249881421_u32, 3497815624_u32, 733560071_u32, 953603770_u32, 2970613457_u32, 2969485820_u32, 2890216747_u32, 3226781838_u32, 1585690261_u32, 1483446000_u32, 1434008975_u32, 3843362210_u32, 2529243289_u32, 2626445604_u32, 2596813363_u32, 2395290102_u32, 3130058973_u32, 807916696_u32, 1342263063_u32, 2336366986_u32, 4003570529_u32, 3056985420_u32, 1525932091_u32, 3958102110_u32, 1355572261_u32, 2387855168_u32, 1297322399_u32, 2923016818_u32, 2140696361_u32, 1316482676_u32, 1255950147_u32, 3280880582_u32, 2013218925_u32, 501616360_u32, 3753256231_u32, 3427491930_u32, 225489905_u32, 1700994204_u32, 2870579019_u32, 438705198_u32, 2148971957_u32, 2292796304_u32, 36907439_u32, 3451571010_u32, 4116287929_u32, 2635861956_u32, 4014463059_u32, 593559958_u32, 3601058813_u32, 1371282744_u32, 823705399_u32, 1128239914_u32, 1516644481_u32, 2086365164_u32, 512180827_u32, 681612286_u32, 3143347013_u32, 1123296_u32, 2457404863_u32, 1954895890_u32, 3792541769_u32, 15582484_u32, 1140583779_u32, 1064989542_u32, 2224518029_u32, 2565799816_u32, 782900551_u32, 1051441658_u32, 11206929_u32, 2184431420_u32, 4014864747_u32, 3493347278_u32, 1943291093_u32, 1785745456_u32, 952509903_u32, 3267778786_u32, 3590161113_u32, 3819048548_u32, 3307832947_u32, 3965163830_u32, 439566109_u32, 637777368_u32, 1730887511_u32, 3831289034_u32, 2577939873_u32, 390150796_u32, 532591739_u32, 1963597726_u32, 1291420261_u32, 1256968320_u32, 1543211487_u32, 2548747698_u32, 2545045865_u32, 4029541300_u32, 3208749955_u32, 297104134_u32, 2853187245_u32, 3770630184_u32, 496837991_u32, 23061402_u32, 3171343921_u32, 1159250396_u32, 2322232203_u32, 4134744942_u32, 2894357493_u32, 2179507408_u32, 2671391215_u32, 1654634114_u32, 3019179001_u32, 1657167108_u32, 1593132179_u32, 2056323286_u32, 1359430205_u32, 3055984248_u32, 1497419639_u32, 3657557610_u32, 2427445953_u32, 3465501996_u32, 4062328475_u32, 2131786558_u32, 1248717189_u32, 2634796320_u32, 4058193407_u32, 4040299346_u32, 231643785_u32, 1491100244_u32, 3241503139_u32, 2770813606_u32, 3325131213_u32, 3900625096_u32, 4120329607_u32, 4191149370_u32, 1489783633_u32, 2142899324_u32, 301399467_u32, 1421801230_u32, 3236893461_u32, 658131312_u32, 2677493263_u32, 1068426274_u32, 1854412057_u32, 556858276_u32, 53678771_u32, 2465544310_u32, 3941061981_u32, 1598871320_u32, 2992730007_u32, 60213258_u32, 130357217_u32];
    for input in inputs {
        match std::panic::catch_unwind(|| run(input as u64)) {
            Ok(words) => println!("{}", words.map(|word| word.to_string()).join(" ")),
            Err(_) => {
                let (line, column) = *location.lock().unwrap();
                println!("panic {line}:{column}");
            }
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
