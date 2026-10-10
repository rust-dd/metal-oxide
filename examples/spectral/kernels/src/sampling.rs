use metal_oxide_device::{WriteBuffer, kernel, math};

pub fn avalanche(mut word: u32) -> u32 {
    word ^= word >> 16;
    word *= 2_246_822_519;
    word ^= word >> 13;
    word *= 3_266_489_917;
    word ^ (word >> 16)
}

fn uniform(key: u32, salt: u32, seed: u32) -> f32 {
    avalanche((key ^ salt) ^ (seed * 2_654_435_761)) as f32 * 2.328_306_4e-10
}

fn positive_uniform(key: u32, salt: u32, seed: u32) -> f32 {
    uniform(key, salt, seed) * 0.999_998 + 1.0e-6
}

fn gamma(key: u32, seed: u32, law: u32, shape: f32, scale: f32) -> f32 {
    if shape <= 0.0 {
        return 0.0;
    }
    let mut a = shape;
    let mut boost = 1.0;
    if a < 1.0 {
        boost = math::powf(
            positive_uniform(key, 2_246_822_519 + law * 97, seed),
            1.0 / shape,
        );
        a += 1.0;
    }
    let dd = a - 1.0 / 3.0;
    let cc = 1.0 / math::sqrt(9.0 * dd);
    let mut val = dd;
    let mut attempt = 0;
    while attempt < 24 {
        let offset = law * 7919 + attempt * 104_729;
        let ga = positive_uniform(key, 1_103_515_245 + offset, seed);
        let gb = positive_uniform(key, 1_013_904_223 + offset, seed);
        let zg = math::sqrt(-2.0 * math::ln(ga)) * math::cos(core::f32::consts::TAU * gb);
        let vv = 1.0 + cc * zg;
        let vv = vv * vv * vv;
        if vv > 0.0 {
            let ug = positive_uniform(key, 668_265_263 + offset, seed);
            val = dd * vv;
            if math::ln(ug) < 0.5 * zg * zg + dd - dd * vv + dd * math::ln(vv) {
                break;
            }
        }
        attempt += 1;
    }
    scale * boost * val
}

/// # Safety
/// out covers rows * steps * 4 values, with products representable in u32.
#[kernel]
pub unsafe fn sample_jumps(
    out: WriteBuffer<f32>,
    rows: u32,
    steps: u32,
    first_path: u32,
    seed: u32,
    shape: f32,
    intensity: f32,
) {
    let cell = crate::index();
    if steps != 0 && cell < rows * steps {
        let path = first_path + cell / steps;
        let step = cell % steps;
        let key = avalanche((path * 2_654_435_761) ^ ((step + 2_246_822_519) * 2_246_822_519));
        let threshold = math::exp(-intensity);
        let mut product = 1.0;
        let mut count = 0;
        while count < 64 {
            let word =
                avalanche((key ^ (2_166_136_261 + count * 16_777_619)) ^ (seed * 374_761_393));
            product *= word as f32 * 2.328_306_4e-10;
            if product <= threshold {
                break;
            }
            count += 1;
        }
        let nj = count as f32;
        let ua = positive_uniform(key, 1_103_515_245, seed);
        let ub = uniform(key, 1_013_904_223, seed);
        let zj = math::sqrt(-2.0 * math::ln(ua)) * math::cos(core::f32::consts::TAU * ub);
        // SAFETY: the active cell owns four consecutive output values.
        unsafe {
            out.store_unchecked(cell * 4, nj);
            out.store_unchecked(cell * 4 + 1, gamma(key, seed, 0, shape + 0.25 * nj, 0.5));
            out.store_unchecked(cell * 4 + 2, gamma(key, seed, 1, 2.5 + 0.1 * nj, 0.75));
            out.store_unchecked(cell * 4 + 3, 0.125 * nj + 0.25 * math::sqrt(nj) * zj);
        }
    }
}
