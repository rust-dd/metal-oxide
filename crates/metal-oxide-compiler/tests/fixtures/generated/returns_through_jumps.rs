// Minimized from seed 0x07ece2c35150a2cb: early returns reached the shared return through an
// empty jump, which counted toward the duplication limit, so the compiler rejected the program.
// words: exact exact exact exact
// input: u64
// output: u8
// block: 3 5 7
// overflow checks: off
#![cfg_attr(not(cpu_reference), no_std)]
#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, unused_imports, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]

fn run(input: u64) -> [u8; 4] {
    if (0.0_f32 > 0.0_f32) {
        'o17: for c18 in (0_u32 .. 0_u32) {
            assert!((c18 != 0_u32), "generated assertion");
        }
    }
    let mut l35: u16 = 0_u16;
    if (match Ok::<i64, f32>(0_i64) { Ok(b38) => false, _ => true }) {
        return [0_u8, 0_u8, 0_u8, 0_u8];
    }
    if ((input as i16) >= 0_i16) {
        return [0_u8, 0_u8, 0_u8, 0_u8];
    }
    if (0.0_f32 < 0.0_f32) {
    } else {
        'o52: for x53 in (0_u32..3) {
            if ((input as u16) != 0_u16) {
                return [0_u8.checked_pow(0_u32).expect("generated expectation"), 0_u8, 0_u8, 0_u8];
            }
        }
    }
    if ((input as i64) < 0_i64) {
        return [0_u8, 0_u8, 0_u8, 0_u8];
    }
    [0_u8, 0_u8, 0_u8, 0_u8]
}

#[cfg(not(cpu_reference))]
use metal_oxide_device::{Float, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[cfg(not(cpu_reference))]
#[kernel(block = (3, 5, 7), instances(generated(4)))]
pub unsafe fn generated_body<const ELEMENTS: u32>(input: ReadBuffer<u64>, output: WriteBuffer<u8>, n: u32) {
    let i = block_idx().x * 105 + (thread_idx().z * 5 + thread_idx().y) * 3 + thread_idx().x;
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
    let inputs = [0_u32, 1_u32, 4294967295_u32, 2147483647_u32, 2147483648_u32, 65535_u32, 65536_u32, 3863569351_u32, 3931721338_u32, 1646693265_u32, 891867068_u32, 2074241003_u32, 1594416206_u32, 457652053_u32, 1717936304_u32, 2664315983_u32, 1554414946_u32, 2238340441_u32, 2176329444_u32, 4270572787_u32, 364627382_u32, 1988395421_u32, 3038975576_u32, 2562074071_u32, 1119979850_u32, 623929377_u32, 2493145868_u32, 4159226619_u32, 2608061470_u32, 3388151013_u32, 1651982592_u32, 425934943_u32, 1533415986_u32, 1388301289_u32, 3308004404_u32, 1596817923_u32, 3856056198_u32, 750522669_u32, 507035816_u32, 846965735_u32, 3703814170_u32, 1749264561_u32, 4133358172_u32, 978461195_u32, 1061031918_u32, 3140261493_u32, 856831312_u32, 1248430191_u32, 1660820226_u32, 625679993_u32, 154449284_u32, 1850917651_u32, 3711397206_u32, 993379517_u32, 3775020792_u32, 3034248695_u32, 2930563818_u32, 1531530561_u32, 2672346540_u32, 2679080219_u32, 1191475134_u32, 1357691909_u32, 131825056_u32, 1531057279_u32, 861640658_u32, 706041097_u32, 759618260_u32, 585926691_u32, 2350567206_u32, 1124768845_u32, 566530376_u32, 1968505863_u32, 1980363194_u32, 3429520849_u32, 1567593724_u32, 4245810219_u32, 1252266894_u32, 1332512149_u32, 2377652720_u32, 1368253583_u32, 1001097378_u32, 4100419481_u32, 3256325156_u32, 4096512307_u32, 2248008950_u32, 2408814557_u32, 704952216_u32, 766170647_u32, 2275933322_u32, 3088040545_u32, 3626465356_u32, 2949273403_u32, 1295639390_u32, 1831138085_u32, 1053624896_u32, 523106463_u32, 1784343922_u32, 231370281_u32, 1503388020_u32, 3917585987_u32, 832417478_u32, 2791946093_u32, 3095900648_u32, 199060519_u32, 2677275482_u32, 3638727409_u32, 3758452636_u32, 2909390411_u32, 2367874862_u32, 3805217973_u32, 2994687632_u32, 2990756015_u32, 4246128194_u32, 2248626361_u32, 3607675588_u32, 2862542675_u32, 2566335638_u32, 3613784829_u32, 1364361272_u32, 3257779767_u32, 2971332138_u32, 936101761_u32, 3202251500_u32, 2478913883_u32, 1409031934_u32, 4217616965_u32, 3983795936_u32, 86961343_u32, 1355551506_u32, 2200377161_u32, 3201513492_u32, 2406794339_u32, 960978534_u32, 3433128589_u32, 2376737416_u32, 2535253063_u32, 3066680570_u32, 3153354769_u32, 2278005308_u32, 3101798507_u32, 3332541134_u32, 2753303509_u32, 2978992944_u32, 781307087_u32, 2980565986_u32, 3513958873_u32, 2259698020_u32, 2956205427_u32, 3540730934_u32, 443870749_u32, 1813223640_u32, 1969997399_u32, 2188226506_u32, 3133496481_u32, 1729482124_u32, 991701883_u32, 3040029342_u32, 4235268453_u32, 3455587200_u32, 1900867807_u32, 722929842_u32, 1936937065_u32, 3851738804_u32, 4115528323_u32, 973069830_u32, 3480853933_u32, 2689852200_u32, 4229655655_u32, 2366865050_u32, 2280111409_u32, 4066251996_u32, 285321867_u32, 1795882606_u32, 3860574965_u32, 1863428048_u32, 1283545327_u32, 1472639362_u32, 837927673_u32, 599134212_u32, 661871507_u32, 622996438_u32, 76050749_u32, 3815768440_u32, 2389497463_u32, 3339207018_u32, 88375745_u32, 2020962348_u32, 425930139_u32, 2126972478_u32, 2501179525_u32, 1853984800_u32, 2341471487_u32, 832818770_u32, 737614217_u32, 1787352404_u32, 4287971491_u32, 4272893350_u32, 1824155853_u32, 1135408072_u32, 1675691143_u32, 1742272570_u32, 855943761_u32, 1366351740_u32, 726774955_u32, 2722382350_u32, 529916437_u32, 2737622128_u32, 1444541711_u32, 3789250338_u32, 1724666905_u32, 348270244_u32, 920955315_u32, 3022250870_u32, 3848832093_u32, 1955576344_u32, 2553869975_u32, 3193086730_u32, 3213965025_u32, 782590668_u32, 639516603_u32, 1923068382_u32, 236416933_u32, 825777344_u32, 1073752351_u32, 2430231538_u32, 1576694441_u32, 4267121652_u32, 2530383555_u32, 2252164422_u32, 103531501_u32, 517821544_u32, 997567655_u32, 8536538_u32, 2543003505_u32, 1789393436_u32, 22662859_u32, 1211520430_u32, 3948060981_u32, 3658039568_u32, 1499701551_u32, 911072450_u32, 2471130425_u32, 3564936516_u32, 152045523_u32, 3640158998_u32, 2967283581_u32, 815448760_u32, 1641555639_u32, 464835754_u32, 1978891265_u32, 1498260844_u32, 3704971739_u32, 111488382_u32, 3560994501_u32, 1279555936_u32, 1051024703_u32, 1763887506_u32];
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
