// Minimized from seed 0x8299ec2428b9cf85: the GPU compiler of hosted macOS runners followed the
// wrong exit of the loop in filter's next, so the for loop ran its body without a match.
// words: float float float float
// input: u16
// output: f32
// block: 32 1 1
// overflow checks: off
#![cfg_attr(not(cpu_reference), no_std)]
#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, unused_imports, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]

fn run(input: u16) -> [f32; 4] {
    let mut l1: [f32; 1] = [0.0_f32];
    let mut l5: [bool; 2] = [false, false];
    'o6: for x7 in l5.iter().take(3).filter((move |a8: &&bool| -> bool { Some((*(*a8))).unwrap_or(true) })) {
        return [0.0_f32, 0.0_f32, l1[(0_i64 as usize) % 1], l1[(0_i8 as usize) % 1]];
    }
    let mut l17: f32 = l1[(0_isize as usize) % 1];
    [0.0_f32, l1[(0_i32 as usize) % 1], l1[(0_i32 as usize) % 1], (-l17).sqrt()]
}

#[cfg(not(cpu_reference))]
use metal_oxide_device::{Float, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[cfg(not(cpu_reference))]
#[kernel(block = (32, 1, 1), instances(generated(4)))]
pub unsafe fn generated_body<const ELEMENTS: u32>(input: ReadBuffer<u16>, output: WriteBuffer<f32>, n: u32) {
    let i = block_idx().x * 32 + thread_idx().x;
    if i < n {
        let words = run(unsafe { input.load_unchecked(i) });
        unsafe {
        output.store_unchecked(i * ELEMENTS + 0, words[0]);
        output.store_unchecked(i * ELEMENTS + 1, words[1]);
        output.store_unchecked(i * ELEMENTS + 2, words[2]);
        output.store_unchecked(i * ELEMENTS + 3, words[3]);
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
    let inputs = [0_u32, 1_u32, 4294967295_u32, 2147483647_u32, 2147483648_u32, 65535_u32, 65536_u32, 687597708_u32, 1703774843_u32, 1338965406_u32, 2272053349_u32, 111828608_u32, 2939994079_u32, 331254706_u32, 3441882985_u32, 1463703988_u32, 1156266371_u32, 320213254_u32, 2834052269_u32, 307203624_u32, 3204782951_u32, 2779472282_u32, 1997554737_u32, 1815539676_u32, 189188491_u32, 2984743278_u32, 3168872949_u32, 2743251664_u32, 329312239_u32, 4267373698_u32, 1160456697_u32, 195709700_u32, 22829715_u32, 4069596886_u32, 1660733501_u32, 2298637432_u32, 3229500791_u32, 455402602_u32, 2661992641_u32, 4059106092_u32, 2808876187_u32, 85219646_u32, 1360277381_u32, 157375264_u32, 1724863487_u32, 1146409298_u32, 4045815945_u32, 108515412_u32, 2780430243_u32, 3114721446_u32, 396394445_u32, 3716554440_u32, 3993645959_u32, 1806326586_u32, 2098770257_u32, 3191814780_u32, 3440268203_u32, 2268222734_u32, 1983830293_u32, 2066438000_u32, 1281950735_u32, 1524179490_u32, 2992716569_u32, 3462130084_u32, 657795251_u32, 1641305718_u32, 1076910941_u32, 2942280984_u32, 1600775575_u32, 2986919434_u32, 482541025_u32, 779517388_u32, 394174139_u32, 1633584350_u32, 3503957669_u32, 2298849216_u32, 3561945119_u32, 3059017458_u32, 3764722089_u32, 1800115956_u32, 2336150979_u32, 936804422_u32, 272980717_u32, 2471755624_u32, 547249063_u32, 4031587546_u32, 3327307377_u32, 1227609372_u32, 978185675_u32, 2012604590_u32, 127866933_u32, 1616202768_u32, 2325884975_u32, 2591367106_u32, 3640319033_u32, 3061569604_u32, 4152886995_u32, 2880069142_u32, 95977085_u32, 1667771832_u32, 1105761719_u32, 3754194858_u32, 2427503361_u32, 238305388_u32, 290275547_u32, 982876286_u32, 1401335237_u32, 465519712_u32, 2277747775_u32, 3928411282_u32, 2418770633_u32, 2858528148_u32, 3254893539_u32, 546107366_u32, 2023933453_u32, 2397122568_u32, 3888747463_u32, 3237723770_u32, 4043717521_u32, 3332685756_u32, 2962069483_u32, 1947880526_u32, 264890197_u32, 321425584_u32, 2153016399_u32, 4153863522_u32, 541068633_u32, 2998015716_u32, 2161964275_u32, 3170579894_u32, 727527837_u32, 1782842968_u32, 1526879703_u32, 734001482_u32, 1253840929_u32, 1423069964_u32, 1952445179_u32, 446778398_u32, 1239532773_u32, 733360384_u32, 3577049183_u32, 1412596274_u32, 1505852393_u32, 4234286132_u32, 445297155_u32, 984756102_u32, 3665871149_u32, 2136322216_u32, 412343271_u32, 3743295514_u32, 2138790065_u32, 2219965020_u32, 290804235_u32, 528973806_u32, 1367819893_u32, 667755856_u32, 2743580783_u32, 2695269122_u32, 1400426105_u32, 3316032900_u32, 3585864467_u32, 3064913238_u32, 3437655229_u32, 3709546232_u32, 3787400695_u32, 2439203562_u32, 2133654849_u32, 1509471660_u32, 2755573019_u32, 1063905214_u32, 1219718149_u32, 4145180064_u32, 3568059519_u32, 1558185938_u32, 4201579785_u32, 2918501076_u32, 1768011811_u32, 2167291686_u32, 1671908429_u32, 1742307656_u32, 3422893063_u32, 3223085498_u32, 3623486929_u32, 1675330812_u32, 3257736235_u32, 3525673870_u32, 3013555605_u32, 107678192_u32, 776213647_u32, 3087792290_u32, 2716671897_u32, 100762660_u32, 212664627_u32, 3987559670_u32, 2853980125_u32, 689285016_u32, 1361511959_u32, 2148953226_u32, 1179349601_u32, 156198988_u32, 2289141563_u32, 2597802846_u32, 147039013_u32, 2415655488_u32, 1647323295_u32, 1620532594_u32, 2474384937_u32, 532795764_u32, 2265894467_u32, 585703110_u32, 3856557933_u32, 2677318120_u32, 1596300327_u32, 1298033498_u32, 1154106097_u32, 3672727452_u32, 2895967819_u32, 2547799854_u32, 1088009397_u32, 3945414288_u32, 512852143_u32, 1412380226_u32, 2665808057_u32, 2657792708_u32, 1077419859_u32, 3940457622_u32, 650554109_u32, 257650744_u32, 1749153335_u32, 3678494250_u32, 2623502209_u32, 3799935724_u32, 2357225819_u32, 3536948990_u32, 1725590085_u32, 3701135072_u32, 499719359_u32, 2948629266_u32, 2855323465_u32, 3329304596_u32, 1343878243_u32, 2194329190_u32, 3605926541_u32, 3517911688_u32, 1929188423_u32, 3830203642_u32, 4097052689_u32, 4079192636_u32, 1043128427_u32, 815004366_u32, 671007701_u32, 3862086448_u32, 913833167_u32, 2138588130_u32, 1101558233_u32, 3448385892_u32, 2397392243_u32];
    for input in inputs {
        match std::panic::catch_unwind(|| run(input as u16)) {
            Ok(words) => println!("{}", words.map(|word| word.to_bits().to_string()).join(" ")),
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
