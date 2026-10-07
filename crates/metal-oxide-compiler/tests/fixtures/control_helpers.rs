pub fn classify(value: i32) -> u32 {
    let signed = match value {
        i32::MIN => 1,
        -7 | -3 => 2,
        0 => 3,
        1 | 2 => match value {
            1 => 4,
            2 => 5,
            _ => 6,
        },
        9..=12 => 7,
        i32::MAX => 8,
        _ => 9,
    };
    let byte = match value as u8 {
        0 => 1,
        1 | 7 => 2,
        255 => 3,
        _ => 4,
    };
    let short = match value as u16 {
        0 => 1,
        1 | 7 => 2,
        65535 => 3,
        _ => 4,
    };
    let unsigned = match value as u32 {
        0 => 1,
        1 | 7 => 2,
        u32::MAX => 3,
        _ => 4,
    };
    signed | (byte << 8) | (short << 16) | (unsigned << 24)
}

pub fn search(limit: u32) -> u32 {
    let mut total = 0;
    let mut i = 0;
    'outer: while i < limit {
        i += 1;
        if i & 3 == 0 {
            continue;
        }
        let mut j = 0;
        'inner: while j < 7 {
            j += 1;
            if j == 2 {
                continue;
            }
            if i == 3 && j == 3 {
                break 'inner;
            }
            if i == 5 && j == 4 {
                continue 'outer;
            }
            if i == 7 && j == 5 {
                break 'outer;
            }
            if limit == 9 && j == 3 {
                return total + 1000;
            }
            total += i * 10 + j;
        }
        total += 100;
    }
    total
}

pub fn nested(mode: u32) -> u32 {
    let mut sum = 0;
    let mut i = 0;
    'outer: while i < 3 {
        i += 1;
        let mut j = 0;
        while j < 3 {
            j += 1;
            let mut k = 0;
            loop {
                k += 1;
                if k == 2 {
                    continue;
                }
                if k >= 4 {
                    break;
                }
                match mode {
                    0 if i == 2 && j == 2 && k == 1 => break 'outer,
                    1 if j == 2 && k == 1 => continue 'outer,
                    2 if k == 3 => break,
                    3 if i == 3 => return sum + 1000,
                    _ => {}
                }
                sum += i * 100 + j * 10 + k;
            }
        }
    }
    sum
}
