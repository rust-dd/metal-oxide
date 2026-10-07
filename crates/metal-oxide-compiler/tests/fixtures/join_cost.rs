#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};
#[kernel]
pub unsafe fn join_cost(out: WriteBuffer<u32>, selector: u32, skip: u32, bits: u32) {
    let mut value = 0_u32;
    if selector == 0 {
        if skip == 0 {
            return;
        }
        if bits & 1 != 0 {
            value = value.wrapping_add(1);
        } else {
            value = value.wrapping_add(2);
        }
        if bits & 2 != 0 {
            value = value.wrapping_add(3);
        } else {
            value = value.wrapping_add(4);
        }
        if bits & 4 != 0 {
            value = value.wrapping_add(5);
        } else {
            value = value.wrapping_add(6);
        }
        if bits & 8 != 0 {
            value = value.wrapping_add(7);
        } else {
            value = value.wrapping_add(8);
        }
        if bits & 16 != 0 {
            value = value.wrapping_add(9);
        } else {
            value = value.wrapping_add(10);
        }
        if bits & 32 != 0 {
            value = value.wrapping_add(11);
        } else {
            value = value.wrapping_add(12);
        }
        if bits & 64 != 0 {
            value = value.wrapping_add(13);
        } else {
            value = value.wrapping_add(14);
        }
        if bits & 128 != 0 {
            value = value.wrapping_add(15);
        } else {
            value = value.wrapping_add(16);
        }
        if bits & 256 != 0 {
            value = value.wrapping_add(17);
        } else {
            value = value.wrapping_add(18);
        }
        if bits & 512 != 0 {
            value = value.wrapping_add(19);
        } else {
            value = value.wrapping_add(20);
        }
        if bits & 1024 != 0 {
            value = value.wrapping_add(21);
        } else {
            value = value.wrapping_add(22);
        }
        if bits & 2048 != 0 {
            value = value.wrapping_add(23);
        } else {
            value = value.wrapping_add(24);
        }
        if bits & 4096 != 0 {
            value = value.wrapping_add(25);
        } else {
            value = value.wrapping_add(26);
        }
        if bits & 8192 != 0 {
            value = value.wrapping_add(27);
        } else {
            value = value.wrapping_add(28);
        }
        if bits & 16384 != 0 {
            value = value.wrapping_add(29);
        } else {
            value = value.wrapping_add(30);
        }
        if bits & 32768 != 0 {
            value = value.wrapping_add(31);
        } else {
            value = value.wrapping_add(32);
        }
        if bits & 65536 != 0 {
            value = value.wrapping_add(33);
        } else {
            value = value.wrapping_add(34);
        }
        if bits & 131072 != 0 {
            value = value.wrapping_add(35);
        } else {
            value = value.wrapping_add(36);
        }
        if bits & 262144 != 0 {
            value = value.wrapping_add(37);
        } else {
            value = value.wrapping_add(38);
        }
        if bits & 524288 != 0 {
            value = value.wrapping_add(39);
        } else {
            value = value.wrapping_add(40);
        }
        if bits & 1048576 != 0 {
            value = value.wrapping_add(41);
        } else {
            value = value.wrapping_add(42);
        }
        if bits & 2097152 != 0 {
            value = value.wrapping_add(43);
        } else {
            value = value.wrapping_add(44);
        }
    } else {
        if skip == 1 {
            return;
        }
        if bits & 1 != 0 {
            value = value.wrapping_add(3);
        } else {
            value = value.wrapping_add(4);
        }
        if bits & 2 != 0 {
            value = value.wrapping_add(5);
        } else {
            value = value.wrapping_add(6);
        }
        if bits & 4 != 0 {
            value = value.wrapping_add(7);
        } else {
            value = value.wrapping_add(8);
        }
        if bits & 8 != 0 {
            value = value.wrapping_add(9);
        } else {
            value = value.wrapping_add(10);
        }
        if bits & 16 != 0 {
            value = value.wrapping_add(11);
        } else {
            value = value.wrapping_add(12);
        }
        if bits & 32 != 0 {
            value = value.wrapping_add(13);
        } else {
            value = value.wrapping_add(14);
        }
        if bits & 64 != 0 {
            value = value.wrapping_add(15);
        } else {
            value = value.wrapping_add(16);
        }
        if bits & 128 != 0 {
            value = value.wrapping_add(17);
        } else {
            value = value.wrapping_add(18);
        }
        if bits & 256 != 0 {
            value = value.wrapping_add(19);
        } else {
            value = value.wrapping_add(20);
        }
        if bits & 512 != 0 {
            value = value.wrapping_add(21);
        } else {
            value = value.wrapping_add(22);
        }
        if bits & 1024 != 0 {
            value = value.wrapping_add(23);
        } else {
            value = value.wrapping_add(24);
        }
        if bits & 2048 != 0 {
            value = value.wrapping_add(25);
        } else {
            value = value.wrapping_add(26);
        }
        if bits & 4096 != 0 {
            value = value.wrapping_add(27);
        } else {
            value = value.wrapping_add(28);
        }
        if bits & 8192 != 0 {
            value = value.wrapping_add(29);
        } else {
            value = value.wrapping_add(30);
        }
        if bits & 16384 != 0 {
            value = value.wrapping_add(31);
        } else {
            value = value.wrapping_add(32);
        }
        if bits & 32768 != 0 {
            value = value.wrapping_add(33);
        } else {
            value = value.wrapping_add(34);
        }
        if bits & 65536 != 0 {
            value = value.wrapping_add(35);
        } else {
            value = value.wrapping_add(36);
        }
        if bits & 131072 != 0 {
            value = value.wrapping_add(37);
        } else {
            value = value.wrapping_add(38);
        }
        if bits & 262144 != 0 {
            value = value.wrapping_add(39);
        } else {
            value = value.wrapping_add(40);
        }
        if bits & 524288 != 0 {
            value = value.wrapping_add(41);
        } else {
            value = value.wrapping_add(42);
        }
        if bits & 1048576 != 0 {
            value = value.wrapping_add(43);
        } else {
            value = value.wrapping_add(44);
        }
        if bits & 2097152 != 0 {
            value = value.wrapping_add(45);
        } else {
            value = value.wrapping_add(46);
        }
    }
    unsafe {
        out.store_unchecked(0, value);
    }
}
