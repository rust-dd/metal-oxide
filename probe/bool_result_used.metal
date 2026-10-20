#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma STDC FP_CONTRACT OFF

struct metal_oxide_context {
    uint3 thread_idx;
    uint3 block_idx;
    uint3 block_dim;
    uint3 grid_dim;
    uint thread_index;
    uint simd_lane;
    uint simd_size;
    uint simd_group;
    uint simd_count;
    const device uint* gate;
    device atomic_uint* failure;
    thread bool* failed;
};

struct metal_oxide_checked_u32 { uint value; bool overflow; };
static inline metal_oxide_checked_u32 metal_oxide_add_checked_u32(uint a, uint b) {
    uint value = a + b;
    return {value, value < a};
}

static inline metal_oxide_checked_u32 metal_oxide_mul_checked_u32(uint a, uint b) {
    uint value = a * b;
    return {value, mulhi(a, b) != 0u};
}

struct metal_oxide_aggregate_0 {
    uint elements[1];
};
static_assert(sizeof(metal_oxide_aggregate_0) == 4 && alignof(metal_oxide_aggregate_0) == 4, "metal_oxide_aggregate_0 layout");

struct metal_oxide_aggregate_1 {
    uint elements[4];
};
static_assert(sizeof(metal_oxide_aggregate_1) == 16 && alignof(metal_oxide_aggregate_1) == 4, "metal_oxide_aggregate_1 layout");

struct metal_oxide_aggregate_2 {
    uchar f0;
};
static_assert(sizeof(metal_oxide_aggregate_2) == 1 && alignof(metal_oxide_aggregate_2) == 1, "metal_oxide_aggregate_2 layout");

struct metal_oxide_aggregate_3 {
    bool elements[3];
};
static_assert(sizeof(metal_oxide_aggregate_3) == 3 && alignof(metal_oxide_aggregate_3) == 1, "metal_oxide_aggregate_3 layout");

static void metal_oxide_fail(metal_oxide_context ctx, uint check) {
    *ctx.failed = true;
    if (*ctx.gate == 0u) {
        atomic_store_explicit(ctx.failure, check + 2u, memory_order_relaxed);
    }
}

static inline uint3 metal_oxide_fn_1(metal_oxide_context metal_oxide_ctx);
static inline uint3 metal_oxide_fn_2(metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_0 metal_oxide_fn_3(uint v1, uint v2, device atomic_int * v3, threadgroup atomic_int * v4, metal_oxide_context metal_oxide_ctx);
static inline void metal_oxide_fn_4(metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_1 metal_oxide_fn_5(uint v1, uint v2, metal_oxide_context metal_oxide_ctx);

kernel void generated(device const uint * metal_oxide_arg_1 [[buffer(0)]], device uint * metal_oxide_arg_2 [[buffer(1)]], constant uint & metal_oxide_arg_3 [[buffer(2)]], device atomic_int * metal_oxide_arg_4 [[buffer(3)]], uint3 metal_oxide_thread_idx [[thread_position_in_threadgroup]], uint3 metal_oxide_block_idx [[threadgroup_position_in_grid]], uint3 metal_oxide_block_dim [[threads_per_threadgroup]], uint3 metal_oxide_grid_dim [[threadgroups_per_grid]], const device uint* metal_oxide_gate [[buffer(4)]], device atomic_uint* metal_oxide_failure [[buffer(5)]]) {
    bool metal_oxide_failed = false;
    metal_oxide_context metal_oxide_ctx = {metal_oxide_thread_idx, metal_oxide_block_idx, metal_oxide_block_dim, metal_oxide_grid_dim, 0, 0, 0, 0, 0, metal_oxide_gate, metal_oxide_failure, &metal_oxide_failed};
    threadgroup atomic_int metal_oxide_shared_0[1];
    device const uint * v1 = metal_oxide_arg_1;
    device uint * v2 = metal_oxide_arg_2;
    uint v3 = metal_oxide_arg_3;
    device atomic_int * v4 = metal_oxide_arg_4;
    uint v5;
    uint v6;
    uint v7;
    uint3 v8;
    metal_oxide_checked_u32 v9;
    uint v10;
    uint3 v11;
    metal_oxide_checked_u32 v12;
    uint v13;
    bool v14;
    uint v15;
    uint v16;
    device const uint * v17;
    uint v18;
    threadgroup atomic_int * v19;
    metal_oxide_aggregate_0 v20;
    uint v21;
    uint v22;
    uint3 v23;
    device atomic_int * v24;
    threadgroup atomic_int * v25;
    bool v26;
    uint v27;
    uint v28;
    metal_oxide_aggregate_1 v29;
    uint v30;
    uint v31;
    device uint * v35;
    uint v36;
    uint v37;
    uint v38;
    metal_oxide_checked_u32 v39;
    metal_oxide_checked_u32 v40;
    uint v41;
    device uint * v45;
    uint v46;
    uint v47;
    uint v48;
    metal_oxide_checked_u32 v49;
    metal_oxide_checked_u32 v50;
    uint v51;
    device uint * v55;
    uint v56;
    uint v57;
    uint v58;
    metal_oxide_checked_u32 v59;
    metal_oxide_checked_u32 v60;
    uint v61;
    device uint * v65;
    uint v66;
    uint v67;
    uint v68;
    metal_oxide_checked_u32 v69;
    metal_oxide_checked_u32 v70;
    uint v71;
    v8 = metal_oxide_fn_1(metal_oxide_ctx);
    v7 = v8.x;
    v9 = metal_oxide_mul_checked_u32(v7, 64u);
    v6 = v9.value;
    v11 = metal_oxide_fn_2(metal_oxide_ctx);
    v10 = v11.x;
    v12 = metal_oxide_add_checked_u32(v6, v10);
    v5 = v12.value;
    v15 = v5;
    v16 = v3;
    v14 = (v15 < v16);
    if (*metal_oxide_gate != 0u) { return; }
    if (v14) {
        v17 = v1;
        v18 = v5;
        v13 = v17[v18];
    }
    else {
        v13 = 0u;
    }
    v19 = metal_oxide_shared_0;
    v21 = v13;
    v23 = metal_oxide_fn_2(metal_oxide_ctx);
    v22 = v23.x;
    v24 = v4;
    v25 = v19;
    v20 = metal_oxide_fn_3(v21, v22, v24, v25, metal_oxide_ctx);
    v27 = v5;
    v28 = v3;
    v26 = (v27 < v28);
    if (v26) {
        v30 = v13;
        v31 = v20.elements[0];
        v29 = metal_oxide_fn_5(v30, v31, metal_oxide_ctx);
        if (*metal_oxide_ctx.failed) { return; }
        v35 = v2;
        v38 = v5;
        v39 = metal_oxide_mul_checked_u32(v38, 4u);
        if (v39.overflow) {
            metal_oxide_fail(metal_oxide_ctx, 0u);
            return;
        }
        v37 = v39.value;
        v40 = metal_oxide_add_checked_u32(v37, 0u);
        v36 = v40.value;
        v41 = v29.elements[0];
        v35[v36] = v41;
        v45 = v2;
        v48 = v5;
        v49 = metal_oxide_mul_checked_u32(v48, 4u);
        if (v49.overflow) {
            metal_oxide_fail(metal_oxide_ctx, 1u);
            return;
        }
        v47 = v49.value;
        v50 = metal_oxide_add_checked_u32(v47, 1u);
        if (v50.overflow) {
            metal_oxide_fail(metal_oxide_ctx, 2u);
            return;
        }
        v46 = v50.value;
        v51 = v29.elements[1];
        v45[v46] = v51;
        v55 = v2;
        v58 = v5;
        v59 = metal_oxide_mul_checked_u32(v58, 4u);
        if (v59.overflow) {
            metal_oxide_fail(metal_oxide_ctx, 3u);
            return;
        }
        v57 = v59.value;
        v60 = metal_oxide_add_checked_u32(v57, 2u);
        if (v60.overflow) {
            metal_oxide_fail(metal_oxide_ctx, 4u);
            return;
        }
        v56 = v60.value;
        v61 = v29.elements[2];
        v55[v56] = v61;
        v65 = v2;
        v68 = v5;
        v69 = metal_oxide_mul_checked_u32(v68, 4u);
        if (v69.overflow) {
            metal_oxide_fail(metal_oxide_ctx, 5u);
            return;
        }
        v67 = v69.value;
        v70 = metal_oxide_add_checked_u32(v67, 3u);
        if (v70.overflow) {
            metal_oxide_fail(metal_oxide_ctx, 6u);
            return;
        }
        v66 = v70.value;
        v71 = v29.elements[3];
        v65[v66] = v71;
    }
    else {
    }
    return;
}

static inline uint3 metal_oxide_fn_1(metal_oxide_context metal_oxide_ctx) {
    uint3 v0;
    v0 = metal_oxide_ctx.block_idx;
    return v0;
}

static inline uint3 metal_oxide_fn_2(metal_oxide_context metal_oxide_ctx) {
    uint3 v0;
    v0 = metal_oxide_ctx.thread_idx;
    return v0;
}

static inline metal_oxide_aggregate_0 metal_oxide_fn_3(uint v1, uint v2, device atomic_int * v3, threadgroup atomic_int * v4, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_0 v0;
    threadgroup atomic_int * v6;
    uint v7;
    uint v8;
    int v12;
    threadgroup atomic_int * v13;
    uint v14;
    int v18;
    threadgroup atomic_int * v19;
    uint v20;
    uint v21;
    uint v25;
    int v26;
    v6 = v4;
    v8 = v2;
    v7 = (v8 % 1u);
    atomic_store_explicit(&v6[v7], as_type<int>(0x00000001u), memory_order_relaxed);
    metal_oxide_fn_4(metal_oxide_ctx);
    v13 = v4;
    v14 = (0u % 1u);
    v12 = atomic_fetch_or_explicit(&v13[v14], as_type<int>(0x00000000u), memory_order_relaxed);
    metal_oxide_fn_4(metal_oxide_ctx);
    v19 = v4;
    v21 = v1;
    v20 = (v21 % 1u);
    v18 = atomic_load_explicit(&v19[v20], memory_order_relaxed);
    metal_oxide_fn_4(metal_oxide_ctx);
    v26 = v18;
    v25 = uint(v26);
    v0 = metal_oxide_aggregate_0{{v25}};
    return v0;
}

static inline void metal_oxide_fn_4(metal_oxide_context metal_oxide_ctx) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
}

static inline metal_oxide_aggregate_1 metal_oxide_fn_5(uint v1, uint v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_1 v0;
    metal_oxide_aggregate_3 v3;
    bool v4;
    ulong v5;
    ushort v6;
    uint v7;
    bool v8;
    ulong v9;
    ulong v10;
    ulong v11;
    ulong v12;
    uint v13;
    bool v14;
    uint v17;
    v3 = metal_oxide_aggregate_3{{true, true, false}};
    v7 = v2;
    v6 = ushort(v7);
    v5 = ulong(v6);
    v8 = (v5 < 3ul);
    if (!(v8)) {
        metal_oxide_fail(metal_oxide_ctx, 7u);
        return {};
    }
    v4 = v3.elements[v5];
    v13 = v2;
    v12 = ulong(v13);
    v14 = (v12 == 0ul);
    if (v14) {
        metal_oxide_fail(metal_oxide_ctx, 8u);
        return {};
    }
    v11 = (18446744073709551615ul / v12);
    v10 = ulong(v11);
    v9 = (v10 % 3ul);
    v3.elements[v9] = v4;
    v17 = as_type<uint>(as_type<float>(0xc3600000u));
    v0 = metal_oxide_aggregate_1{{v17, uint(v3.elements[0]), uint(v3.elements[1]), uint(v3.elements[2])}};
    return v0;
}

