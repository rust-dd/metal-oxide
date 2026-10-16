#![no_std]
#![feature(core_intrinsics, portable_simd)]

use core::{intrinsics::simd, simd::Simd};
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn rejected(output: WriteBuffer<u32>, value: u32) {
    let lanes = Simd::<u32, 2>::splat(value);
    #[cfg(nonvector)]
    let result = unsafe { simd::simd_add(value, value) };
    #[cfg(representation)]
    let result = unsafe { simd::simd_add((value, value), (value, value)) }.0;
    #[cfg(array)]
    let result = unsafe { simd::simd_add([value; 2], [value; 2]) }[0];
    #[cfg(lane_mismatch)]
    let result = unsafe { simd::simd_eq::<_, Simd<i32, 4>>(lanes, lanes) }[0] as u32;
    #[cfg(comparison_float)]
    let result = unsafe { simd::simd_eq::<_, Simd<f32, 2>>(lanes, lanes) }[0] as u32;
    #[cfg(comparison_width)]
    let result = unsafe { simd::simd_eq::<_, Simd<i16, 2>>(lanes, lanes) }[0] as u32;
    #[cfg(destination)]
    let result =
        unsafe { simd::simd_shuffle::<_, _, Simd<u32, 2>>(lanes, lanes, const { [0_u32, 1, 2] }) }
            [0];
    #[cfg(shuffle_bounds)]
    let result =
        unsafe { simd::simd_shuffle::<_, _, Simd<u32, 2>>(lanes, lanes, const { [0_u32, 4] }) }[0];
    #[cfg(shuffle_dynamic)]
    let result =
        unsafe { simd::simd_shuffle::<_, _, Simd<u32, 2>>(lanes, lanes, [value, value]) }[0];
    #[cfg(shuffle_signed)]
    let result =
        unsafe { simd::simd_shuffle::<_, _, Simd<u32, 2>>(lanes, lanes, const { [0_i32, 1] }) }[0];
    #[cfg(bitmask_array)]
    let result = unsafe { simd::simd_bitmask::<_, [u8; 1]>(Simd::<i32, 2>::splat(-1)) }[0] as u32;
    #[cfg(bitmask_float)]
    let result = unsafe { simd::simd_bitmask::<_, f32>(lanes) } as u32;
    #[cfg(bitmask_bool)]
    let result = unsafe { simd::simd_bitmask::<_, bool>(lanes) } as u32;
    #[cfg(bitmask_wide)]
    let result = unsafe { simd::simd_bitmask::<_, u32>(lanes) };
    #[cfg(bitmask_float_lanes)]
    let result = unsafe { simd::simd_bitmask::<_, u8>(Simd::<f32, 2>::splat(1.0)) } as u32;
    #[cfg(select_float)]
    let result = unsafe { simd::simd_select(Simd::<f32, 2>::splat(1.0), lanes, lanes) }[0];
    #[cfg(select_lanes)]
    let result = unsafe { simd::simd_select(Simd::<i32, 4>::splat(-1), lanes, lanes) }[0];
    #[cfg(reduction_output)]
    let result = unsafe { simd::simd_reduce_add_unordered::<_, f32>(lanes) } as u32;
    #[cfg(reduction_initial)]
    let result = unsafe { simd::simd_reduce_add_ordered(lanes, 1.0_f32) } as u32;
    #[cfg(reduction_boolean)]
    let result = unsafe { simd::simd_reduce_all(Simd::<f32, 2>::splat(1.0)) } as u32;
    #[cfg(extract_output)]
    let result = unsafe { simd::simd_extract::<_, f32>(lanes, 0) } as u32;
    #[cfg(insert_value)]
    let result = unsafe { simd::simd_insert(lanes, 0, 1.0_f32) }[0];
    #[cfg(cast_lanes)]
    let result = unsafe { simd::simd_as::<_, Simd<i32, 4>>(lanes) }[0] as u32;
    #[cfg(splat_type)]
    let result = unsafe { simd::simd_splat::<Simd<f32, 2>, _>(value) }[0] as u32;
    // SAFETY: the caller supplies one output element and launches one thread.
    unsafe { output.store_unchecked(0, result) };
}
