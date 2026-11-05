// Minimized from seed 0xc35863645c6ba6ca: loop exits that continue at the same code repeated it
// per exit until the compiler rejected the duplication.
// words: exact exact exact
// input: i8
// output: i16
// block: 16 2 2
// overflow checks: off
#![cfg_attr(not(cpu_reference), no_std)]
#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, unused_imports, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]

fn run(input: i8) -> [i16; 3] {
    'o11: for c12 in 0..4_u32 {
        if ((input as u64) != 0_u64) {
            return [0_i16, 0_i16.checked_add(0_i16).expect("generated expectation"), 0_i16];
        }
        if (0_u32 ..= 0_u32).any((move |a15: u32| -> bool { true })) { break 'o11; }
    }
    if (4125821218786266465_usize.overflowing_shr(((0_u32 ..= (((1560180009_u32 << (input as u32)).rotate_left((4227532631226876917_isize.overflowing_neg().0 as u32)) ^ (input as u32)) % 8_u32)).rev().skip(1).take(2).map((move |a16: u32| -> f32 { 932.75_f32 })).fold(0_u64, (move |a17: u64, a18: f32| -> u64 { a17 })) as u32)).0 != 0_usize) {
        return true.then_some([0_i16, 0_i16, 0_i16]).expect("generated expectation");
    }
    let mut l24: u32 = 0_i32.leading_zeros();
    if ((input as u32) != 0_u32) {
        return [0_i16, 0_i16, 0_i16];
    }
    let mut l32: Option<i8> = input.checked_shr(0_u32);
    assert!((((0_u32 ..= 0_u32).count() as i8) != 0_i8), "generated assertion");
    if l32.is_none() {
    } else {
        return [0_i16, 0_i16, 0_i16];
    }
    [0_i16, 0_i16.checked_abs().expect("generated expectation"), 0_i16]
}

#[cfg(not(cpu_reference))]
use metal_oxide_device::{Float, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[cfg(not(cpu_reference))]
#[kernel(instances(generated(3)))]
pub unsafe fn generated_body<const ELEMENTS: u32>(input: ReadBuffer<i8>, output: WriteBuffer<i16>, n: u32) {
    let (block, thread) = (block_dim(), thread_idx());
    let i = block_idx().x * (block.x * block.y * block.z) + (thread.z * block.y + thread.y) * block.x + thread.x;
    if i < n {
        let words = run(unsafe { input.load_unchecked(i) });
        unsafe {
        output.store_unchecked(i * ELEMENTS + 0, words[0]);
        output.store_unchecked(i * ELEMENTS + 1, words[1]);
        output.store_unchecked(i * ELEMENTS + 2, words[2]);
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
    let inputs = [0_u32, 1_u32, 4294967295_u32, 2147483647_u32, 2147483648_u32, 65535_u32, 65536_u32, 1087459637_u32, 3684184336_u32, 3511013679_u32, 468339906_u32, 2161911097_u32, 553815364_u32, 3111991251_u32, 4288928534_u32, 3862026109_u32, 3557274296_u32, 2248169143_u32, 2766095530_u32, 1169894401_u32, 2794558828_u32, 1256755675_u32, 4072578430_u32, 3828033221_u32, 3444533600_u32, 2547223871_u32, 127613330_u32, 394464201_u32, 3412697748_u32, 2282171619_u32, 1658432742_u32, 3738590989_u32, 1064695048_u32, 1658164423_u32, 3791506298_u32, 3025069201_u32, 631985340_u32, 2957060331_u32, 620669262_u32, 493920341_u32, 1129706928_u32, 3856698703_u32, 769301090_u32, 3181237849_u32, 2945141732_u32, 268609011_u32, 1332425398_u32, 4007332509_u32, 2904427352_u32, 454203095_u32, 3212396106_u32, 617808161_u32, 1238509580_u32, 2705048571_u32, 1611720990_u32, 2652552677_u32, 1993397760_u32, 1615712607_u32, 2484432690_u32, 666207465_u32, 2388429108_u32, 3152061187_u32, 857042054_u32, 4141406765_u32, 2059728296_u32, 4021837031_u32, 4032738586_u32, 3706749361_u32, 4070248284_u32, 3597230859_u32, 3377618158_u32, 1508015989_u32, 2116356688_u32, 1748785519_u32, 1324922882_u32, 1351817081_u32, 974281348_u32, 1448223763_u32, 3738524246_u32, 1638452669_u32, 2339353592_u32, 1711793911_u32, 519772138_u32, 814769729_u32, 953879212_u32, 1889207835_u32, 1365247166_u32, 381740293_u32, 2633470624_u32, 722281855_u32, 1383167186_u32, 155160073_u32, 3860971476_u32, 440936739_u32, 1458043942_u32, 3026445645_u32, 2539865672_u32, 1468151047_u32, 3168497338_u32, 299609809_u32, 3198622204_u32, 149007659_u32, 2216091790_u32, 4243517077_u32, 2486005488_u32, 2992715151_u32, 1101833634_u32, 2298734745_u32, 2000664868_u32, 3270750771_u32, 2731168246_u32, 509812957_u32, 2082778264_u32, 31999767_u32, 3536632202_u32, 2705021793_u32, 4283710796_u32, 3230554171_u32, 1486090334_u32, 652581925_u32, 4058750784_u32, 4095314335_u32, 1058490994_u32, 4251526953_u32, 3546367604_u32, 2613428035_u32, 2637781958_u32, 155156589_u32, 2356733672_u32, 389750055_u32, 3504076890_u32, 3176688625_u32, 2675405980_u32, 1362233163_u32, 20198446_u32, 828239285_u32, 1637294992_u32, 3797361071_u32, 3160302402_u32, 439896505_u32, 834365380_u32, 2423214163_u32, 3291617686_u32, 2832335869_u32, 3764712760_u32, 3451631415_u32, 1582705450_u32, 2468077697_u32, 3169173484_u32, 480098907_u32, 3147082750_u32, 3411075909_u32, 4225039328_u32, 2071657919_u32, 1248784402_u32, 2843365449_u32, 1791171860_u32, 1021404515_u32, 150047590_u32, 2335419277_u32, 2771244936_u32, 4025161031_u32, 700994042_u32, 766425361_u32, 1051991868_u32, 1021473131_u32, 2693945294_u32, 1173847253_u32, 2525636656_u32, 3659931087_u32, 1769337058_u32, 460864217_u32, 1210934884_u32, 4249681523_u32, 2683613494_u32, 683505437_u32, 1834520024_u32, 3968481111_u32, 3103747274_u32, 108656033_u32, 624398988_u32, 2488347771_u32, 1951020958_u32, 617265765_u32, 349939840_u32, 1161396703_u32, 1491101106_u32, 381356393_u32, 3572450228_u32, 4263713667_u32, 3505949446_u32, 1946707629_u32, 2158630952_u32, 3868859751_u32, 956857242_u32, 2502334001_u32, 3657767388_u32, 3715820427_u32, 1981369198_u32, 2911083509_u32, 3073277136_u32, 1071000047_u32, 3881520770_u32, 134986745_u32, 2906502404_u32, 1276427411_u32, 3338279126_u32, 1777171005_u32, 740783736_u32, 3311126391_u32, 1311800938_u32, 3751647937_u32, 2146546988_u32, 3140529819_u32, 815567678_u32, 2015043973_u32, 1797760288_u32, 2573014527_u32, 1121168210_u32, 2993902217_u32, 1317911124_u32, 3826442659_u32, 1614174886_u32, 2709893581_u32, 303409352_u32, 1146107271_u32, 3825497402_u32, 168297297_u32, 1125278844_u32, 3364058539_u32, 1236340494_u32, 4274670357_u32, 464265584_u32, 2104444431_u32, 3363208226_u32, 3208301849_u32, 1737773988_u32, 3276704435_u32, 4264185974_u32, 3693743453_u32, 2046406424_u32, 54028183_u32, 4249968650_u32, 3872313313_u32, 2830903244_u32, 2831433915_u32, 1383317214_u32, 772413605_u32, 14741952_u32, 2213394975_u32, 3568353522_u32, 1307257769_u32, 675709172_u32, 4147689411_u32, 1435695686_u32, 2507377901_u32, 2694351208_u32];
    for input in inputs {
        match std::panic::catch_unwind(|| run(input as i8)) {
            Ok(words) => println!("{}", words.map(|word| (word as u16).to_string()).join(" ")),
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
