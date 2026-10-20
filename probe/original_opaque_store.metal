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

struct metal_oxide_aggregate_0;
struct metal_oxide_aggregate_1;
struct metal_oxide_aggregate_2;
struct metal_oxide_aggregate_3;
struct metal_oxide_aggregate_4;
struct metal_oxide_aggregate_5;
struct metal_oxide_aggregate_6;
struct metal_oxide_aggregate_7;
struct metal_oxide_aggregate_8;
struct metal_oxide_aggregate_9;
struct metal_oxide_aggregate_10;
struct metal_oxide_aggregate_11;
struct metal_oxide_aggregate_12;
struct metal_oxide_aggregate_13;
struct metal_oxide_aggregate_14;
struct metal_oxide_aggregate_15;
struct metal_oxide_aggregate_16;
struct metal_oxide_aggregate_17;
struct metal_oxide_aggregate_18;
struct metal_oxide_aggregate_19;
struct metal_oxide_aggregate_20;
struct metal_oxide_aggregate_21;
struct metal_oxide_aggregate_22;
struct metal_oxide_aggregate_23;
struct metal_oxide_aggregate_24;
struct metal_oxide_aggregate_25;
struct metal_oxide_aggregate_26;
struct metal_oxide_aggregate_27;
struct metal_oxide_aggregate_28;
struct metal_oxide_aggregate_29;
struct metal_oxide_aggregate_30;
struct metal_oxide_aggregate_31;
struct metal_oxide_aggregate_32;
struct metal_oxide_aggregate_33;
struct metal_oxide_aggregate_34;
struct metal_oxide_aggregate_35;
struct metal_oxide_aggregate_36;
struct metal_oxide_aggregate_37;
struct metal_oxide_aggregate_38;
struct metal_oxide_aggregate_39;
struct metal_oxide_aggregate_40;
struct metal_oxide_aggregate_41;
struct metal_oxide_aggregate_42;
struct metal_oxide_aggregate_43;
struct metal_oxide_aggregate_44;
struct metal_oxide_aggregate_45;
struct metal_oxide_aggregate_46;
struct metal_oxide_aggregate_47;
struct metal_oxide_aggregate_48;
struct metal_oxide_aggregate_49;
struct metal_oxide_aggregate_50;
struct metal_oxide_aggregate_51;
struct metal_oxide_aggregate_52;
struct metal_oxide_aggregate_53;
struct metal_oxide_aggregate_54;
struct metal_oxide_aggregate_55;
struct metal_oxide_aggregate_56;
struct metal_oxide_aggregate_57;
struct metal_oxide_aggregate_58;
struct metal_oxide_aggregate_59;
struct metal_oxide_aggregate_60;
struct metal_oxide_aggregate_61;
struct metal_oxide_aggregate_62;
struct metal_oxide_aggregate_63;
struct metal_oxide_aggregate_64;
typedef thread const metal_oxide_aggregate_10* metal_oxide_pointer_0;
struct metal_oxide_pointer_1 {
    thread const bool* data;
    ulong length;
};

struct metal_oxide_pointer_2 {
    thread bool* data;
    ulong length;
};

typedef thread bool* metal_oxide_pointer_3;
typedef thread const bool* metal_oxide_pointer_4;
typedef thread const metal_oxide_aggregate_8* metal_oxide_pointer_5;
typedef thread metal_oxide_aggregate_8* metal_oxide_pointer_6;
typedef thread const uint* metal_oxide_pointer_7;
typedef thread const metal_oxide_aggregate_12* metal_oxide_pointer_8;
typedef thread metal_oxide_aggregate_32* metal_oxide_pointer_9;
typedef thread metal_oxide_aggregate_24* metal_oxide_pointer_10;
typedef thread ulong* metal_oxide_pointer_11;
typedef thread metal_oxide_aggregate_23* metal_oxide_pointer_12;
typedef thread metal_oxide_aggregate_22* metal_oxide_pointer_13;
typedef thread metal_oxide_aggregate_37* metal_oxide_pointer_14;
typedef thread const metal_oxide_pointer_4* metal_oxide_pointer_15;
typedef thread metal_oxide_aggregate_35* metal_oxide_pointer_16;
typedef thread metal_oxide_aggregate_51* metal_oxide_pointer_17;
typedef thread metal_oxide_aggregate_17* metal_oxide_pointer_18;
typedef thread metal_oxide_aggregate_56* metal_oxide_pointer_19;
typedef thread metal_oxide_aggregate_55* metal_oxide_pointer_20;
typedef thread metal_oxide_aggregate_59* metal_oxide_pointer_21;
typedef thread metal_oxide_aggregate_19* metal_oxide_pointer_22;
typedef thread metal_oxide_aggregate_63* metal_oxide_pointer_23;
typedef thread metal_oxide_aggregate_62* metal_oxide_pointer_24;

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
    uchar f0;
    ushort f1;
};
static_assert(sizeof(metal_oxide_aggregate_3) == 4 && alignof(metal_oxide_aggregate_3) == 2, "metal_oxide_aggregate_3 layout");

struct metal_oxide_aggregate_4 {
    char f0;
    uchar metal_oxide_padding_1[3];
    float f1;
};
static_assert(sizeof(metal_oxide_aggregate_4) == 8 && alignof(metal_oxide_aggregate_4) == 4, "metal_oxide_aggregate_4 layout");

struct metal_oxide_aggregate_5 {
    uchar f0;
    metal_oxide_aggregate_4 f1;
    uchar f2;
};
static_assert(sizeof(metal_oxide_aggregate_5) == 16 && alignof(metal_oxide_aggregate_5) == 4, "metal_oxide_aggregate_5 layout");

struct metal_oxide_aggregate_6 {
    uchar f0;
    uchar f1;
};
static_assert(sizeof(metal_oxide_aggregate_6) == 2 && alignof(metal_oxide_aggregate_6) == 1, "metal_oxide_aggregate_6 layout");

struct metal_oxide_aggregate_7 {
    uchar f0;
    int f1;
};
static_assert(sizeof(metal_oxide_aggregate_7) == 8 && alignof(metal_oxide_aggregate_7) == 4, "metal_oxide_aggregate_7 layout");

struct metal_oxide_aggregate_8 {
    uint f0;
    uint f1;
};
static_assert(sizeof(metal_oxide_aggregate_8) == 8 && alignof(metal_oxide_aggregate_8) == 4, "metal_oxide_aggregate_8 layout");

struct metal_oxide_aggregate_9 {
    uchar f0;
    uint f1;
};
static_assert(sizeof(metal_oxide_aggregate_9) == 8 && alignof(metal_oxide_aggregate_9) == 4, "metal_oxide_aggregate_9 layout");

struct metal_oxide_aggregate_10 {
    bool elements[3];
};
static_assert(sizeof(metal_oxide_aggregate_10) == 3 && alignof(metal_oxide_aggregate_10) == 1, "metal_oxide_aggregate_10 layout");

struct metal_oxide_aggregate_11 {
    uchar f0;
    metal_oxide_aggregate_4 f1;
    uint f2;
};
static_assert(sizeof(metal_oxide_aggregate_11) == 16 && alignof(metal_oxide_aggregate_11) == 4, "metal_oxide_aggregate_11 layout");

struct metal_oxide_aggregate_12 {
    metal_oxide_aggregate_8 f0;
    ulong f1;
    bool f2;
};
static_assert(sizeof(metal_oxide_aggregate_12) == 24 && alignof(metal_oxide_aggregate_12) == 8, "metal_oxide_aggregate_12 layout");

struct metal_oxide_aggregate_13 {
    uchar f0;
    int f1;
    uint f2;
};
static_assert(sizeof(metal_oxide_aggregate_13) == 12 && alignof(metal_oxide_aggregate_13) == 4, "metal_oxide_aggregate_13 layout");

struct metal_oxide_aggregate_14 {
    ulong f0;
    ulong f1;
};
static_assert(sizeof(metal_oxide_aggregate_14) == 16 && alignof(metal_oxide_aggregate_14) == 8, "metal_oxide_aggregate_14 layout");

struct metal_oxide_aggregate_15 {
    float elements[4];
};
static_assert(sizeof(metal_oxide_aggregate_15) == 16 && alignof(metal_oxide_aggregate_15) == 4, "metal_oxide_aggregate_15 layout");

struct metal_oxide_aggregate_16 {
    uchar f0;
    metal_oxide_aggregate_15 f1;
    short f2;
};
static_assert(sizeof(metal_oxide_aggregate_16) == 24 && alignof(metal_oxide_aggregate_16) == 4, "metal_oxide_aggregate_16 layout");

struct metal_oxide_aggregate_17 {
    metal_oxide_aggregate_8 f0;
};
static_assert(sizeof(metal_oxide_aggregate_17) == 8 && alignof(metal_oxide_aggregate_17) == 4, "metal_oxide_aggregate_17 layout");

struct metal_oxide_aggregate_18 {
    metal_oxide_aggregate_17 f0;
};
static_assert(sizeof(metal_oxide_aggregate_18) == 8 && alignof(metal_oxide_aggregate_18) == 4, "metal_oxide_aggregate_18 layout");

struct metal_oxide_aggregate_19 {
    ushort f0;
};
static_assert(sizeof(metal_oxide_aggregate_19) == 2 && alignof(metal_oxide_aggregate_19) == 2, "metal_oxide_aggregate_19 layout");

struct metal_oxide_aggregate_20 {
    uchar f0;
    float f1;
    char f2;
};
static_assert(sizeof(metal_oxide_aggregate_20) == 12 && alignof(metal_oxide_aggregate_20) == 4, "metal_oxide_aggregate_20 layout");

struct metal_oxide_aggregate_21 {
    uchar f0;
    metal_oxide_aggregate_7 f1;
};
static_assert(sizeof(metal_oxide_aggregate_21) == 12 && alignof(metal_oxide_aggregate_21) == 4, "metal_oxide_aggregate_21 layout");

struct metal_oxide_aggregate_22 {
    metal_oxide_pointer_4 f0;
    metal_oxide_pointer_4 f1;
};
static_assert(sizeof(metal_oxide_aggregate_22) == 16 && alignof(metal_oxide_aggregate_22) == 8, "metal_oxide_aggregate_22 layout");

struct metal_oxide_aggregate_23 {
    metal_oxide_aggregate_22 f0;
    ulong f1;
};
static_assert(sizeof(metal_oxide_aggregate_23) == 24 && alignof(metal_oxide_aggregate_23) == 8, "metal_oxide_aggregate_23 layout");

struct metal_oxide_aggregate_24 {
    metal_oxide_aggregate_23 f0;
    ulong f1;
};
static_assert(sizeof(metal_oxide_aggregate_24) == 32 && alignof(metal_oxide_aggregate_24) == 8, "metal_oxide_aggregate_24 layout");

struct metal_oxide_aggregate_25 {
    metal_oxide_aggregate_24 f0;
};
static_assert(sizeof(metal_oxide_aggregate_25) == 32 && alignof(metal_oxide_aggregate_25) == 8, "metal_oxide_aggregate_25 layout");

struct metal_oxide_aggregate_26 {
    uchar f0;
    ulong f1;
};
static_assert(sizeof(metal_oxide_aggregate_26) == 16 && alignof(metal_oxide_aggregate_26) == 8, "metal_oxide_aggregate_26 layout");

struct metal_oxide_aggregate_27 {
    ulong f0;
    metal_oxide_aggregate_26 f1;
};
static_assert(sizeof(metal_oxide_aggregate_27) == 24 && alignof(metal_oxide_aggregate_27) == 8, "metal_oxide_aggregate_27 layout");

struct metal_oxide_aggregate_28 {
    uchar f0;
};
static_assert(sizeof(metal_oxide_aggregate_28) == 1 && alignof(metal_oxide_aggregate_28) == 1, "metal_oxide_aggregate_28 layout");

struct metal_oxide_aggregate_29 {
    metal_oxide_aggregate_28 f0;
};
static_assert(sizeof(metal_oxide_aggregate_29) == 1 && alignof(metal_oxide_aggregate_29) == 1, "metal_oxide_aggregate_29 layout");

struct metal_oxide_aggregate_30 {
    uchar f0;
    uint f1;
    metal_oxide_aggregate_29 f2;
};
static_assert(sizeof(metal_oxide_aggregate_30) == 12 && alignof(metal_oxide_aggregate_30) == 4, "metal_oxide_aggregate_30 layout");

struct metal_oxide_aggregate_31 {
    bool f0;
    uint f1;
};
static_assert(sizeof(metal_oxide_aggregate_31) == 8 && alignof(metal_oxide_aggregate_31) == 4, "metal_oxide_aggregate_31 layout");

struct metal_oxide_aggregate_32 {
    bool f0;
};
static_assert(sizeof(metal_oxide_aggregate_32) == 1 && alignof(metal_oxide_aggregate_32) == 1, "metal_oxide_aggregate_32 layout");

struct metal_oxide_aggregate_33 {
    float f0;
};
static_assert(sizeof(metal_oxide_aggregate_33) == 4 && alignof(metal_oxide_aggregate_33) == 4, "metal_oxide_aggregate_33 layout");

struct metal_oxide_aggregate_34 {
    uchar f0;
    float f1;
    metal_oxide_aggregate_33 f2;
};
static_assert(sizeof(metal_oxide_aggregate_34) == 12 && alignof(metal_oxide_aggregate_34) == 4, "metal_oxide_aggregate_34 layout");

struct metal_oxide_aggregate_35 {
    metal_oxide_pointer_11 f0;
};
static_assert(sizeof(metal_oxide_aggregate_35) == 8 && alignof(metal_oxide_aggregate_35) == 8, "metal_oxide_aggregate_35 layout");

struct metal_oxide_aggregate_36 {
    uchar f0;
    float f1;
    metal_oxide_aggregate_34 f2;
};
static_assert(sizeof(metal_oxide_aggregate_36) == 20 && alignof(metal_oxide_aggregate_36) == 4, "metal_oxide_aggregate_36 layout");

struct metal_oxide_aggregate_37 {
    metal_oxide_pointer_11 f0;
    metal_oxide_aggregate_35 f1;
};
static_assert(sizeof(metal_oxide_aggregate_37) == 16 && alignof(metal_oxide_aggregate_37) == 8, "metal_oxide_aggregate_37 layout");

struct metal_oxide_aggregate_38 {
    uchar f0;
    metal_oxide_aggregate_34 f1;
};
static_assert(sizeof(metal_oxide_aggregate_38) == 16 && alignof(metal_oxide_aggregate_38) == 4, "metal_oxide_aggregate_38 layout");

struct metal_oxide_aggregate_39 {
    uchar f0;
    float f1;
    metal_oxide_aggregate_38 f2;
};
static_assert(sizeof(metal_oxide_aggregate_39) == 24 && alignof(metal_oxide_aggregate_39) == 4, "metal_oxide_aggregate_39 layout");

struct metal_oxide_aggregate_40 {
    uchar f0;
    metal_oxide_pointer_4 f1;
};
static_assert(sizeof(metal_oxide_aggregate_40) == 16 && alignof(metal_oxide_aggregate_40) == 8, "metal_oxide_aggregate_40 layout");

struct metal_oxide_aggregate_41 {
    float f0;
    metal_oxide_pointer_4 f1;
};
static_assert(sizeof(metal_oxide_aggregate_41) == 16 && alignof(metal_oxide_aggregate_41) == 8, "metal_oxide_aggregate_41 layout");

struct metal_oxide_aggregate_42 {
    float f0;
    metal_oxide_aggregate_4 f1;
};
static_assert(sizeof(metal_oxide_aggregate_42) == 12 && alignof(metal_oxide_aggregate_42) == 4, "metal_oxide_aggregate_42 layout");

struct metal_oxide_aggregate_43 {
    metal_oxide_pointer_4 f0;
};
static_assert(sizeof(metal_oxide_aggregate_43) == 8 && alignof(metal_oxide_aggregate_43) == 8, "metal_oxide_aggregate_43 layout");

struct metal_oxide_aggregate_44 {
    metal_oxide_aggregate_8 f0;
    ulong f1;
};
static_assert(sizeof(metal_oxide_aggregate_44) == 16 && alignof(metal_oxide_aggregate_44) == 8, "metal_oxide_aggregate_44 layout");

struct metal_oxide_aggregate_45 {
    float f0;
    uint f1;
};
static_assert(sizeof(metal_oxide_aggregate_45) == 8 && alignof(metal_oxide_aggregate_45) == 4, "metal_oxide_aggregate_45 layout");

struct metal_oxide_aggregate_46 {
    uchar f0;
    float f1;
};
static_assert(sizeof(metal_oxide_aggregate_46) == 8 && alignof(metal_oxide_aggregate_46) == 4, "metal_oxide_aggregate_46 layout");

struct metal_oxide_aggregate_47 {
    uchar f0;
    metal_oxide_aggregate_33 f1;
};
static_assert(sizeof(metal_oxide_aggregate_47) == 8 && alignof(metal_oxide_aggregate_47) == 4, "metal_oxide_aggregate_47 layout");

struct metal_oxide_aggregate_48 {
    uchar f0;
    float f1;
    metal_oxide_aggregate_47 f2;
};
static_assert(sizeof(metal_oxide_aggregate_48) == 16 && alignof(metal_oxide_aggregate_48) == 4, "metal_oxide_aggregate_48 layout");

struct metal_oxide_aggregate_49 {
    uchar f0;
    ulong f1;
};
static_assert(sizeof(metal_oxide_aggregate_49) == 16 && alignof(metal_oxide_aggregate_49) == 8, "metal_oxide_aggregate_49 layout");

struct metal_oxide_aggregate_50 {
    metal_oxide_aggregate_49 f0;
};
static_assert(sizeof(metal_oxide_aggregate_50) == 16 && alignof(metal_oxide_aggregate_50) == 8, "metal_oxide_aggregate_50 layout");

struct metal_oxide_aggregate_51 {
    metal_oxide_aggregate_17 f0;
    ulong f1;
};
static_assert(sizeof(metal_oxide_aggregate_51) == 16 && alignof(metal_oxide_aggregate_51) == 8, "metal_oxide_aggregate_51 layout");

struct metal_oxide_aggregate_52 {
    metal_oxide_aggregate_51 f0;
    ulong f1;
    bool f2;
};
static_assert(sizeof(metal_oxide_aggregate_52) == 32 && alignof(metal_oxide_aggregate_52) == 8, "metal_oxide_aggregate_52 layout");

struct metal_oxide_aggregate_53 {
    metal_oxide_aggregate_52 f0;
};
static_assert(sizeof(metal_oxide_aggregate_53) == 32 && alignof(metal_oxide_aggregate_53) == 8, "metal_oxide_aggregate_53 layout");

struct metal_oxide_aggregate_54 {
    int f0;
    uint f1;
};
static_assert(sizeof(metal_oxide_aggregate_54) == 8 && alignof(metal_oxide_aggregate_54) == 4, "metal_oxide_aggregate_54 layout");

struct metal_oxide_aggregate_55 {
    metal_oxide_pointer_17 f0;
    ulong f1;
};
static_assert(sizeof(metal_oxide_aggregate_55) == 16 && alignof(metal_oxide_aggregate_55) == 8, "metal_oxide_aggregate_55 layout");

struct metal_oxide_aggregate_56 {
    metal_oxide_aggregate_55 f0;
};
static_assert(sizeof(metal_oxide_aggregate_56) == 16 && alignof(metal_oxide_aggregate_56) == 8, "metal_oxide_aggregate_56 layout");

struct metal_oxide_aggregate_57 {
    int f0;
    uchar f1;
    uchar metal_oxide_padding_2[3];
};
static_assert(sizeof(metal_oxide_aggregate_57) == 8 && alignof(metal_oxide_aggregate_57) == 4, "metal_oxide_aggregate_57 layout");

struct metal_oxide_aggregate_58 {
    uint f0;
};
static_assert(sizeof(metal_oxide_aggregate_58) == 4 && alignof(metal_oxide_aggregate_58) == 4, "metal_oxide_aggregate_58 layout");

struct metal_oxide_aggregate_59 {
    metal_oxide_aggregate_19 f0;
};
static_assert(sizeof(metal_oxide_aggregate_59) == 2 && alignof(metal_oxide_aggregate_59) == 2, "metal_oxide_aggregate_59 layout");

struct metal_oxide_aggregate_60 {
    float f0;
    float f1;
};
static_assert(sizeof(metal_oxide_aggregate_60) == 8 && alignof(metal_oxide_aggregate_60) == 4, "metal_oxide_aggregate_60 layout");

struct metal_oxide_aggregate_61 {
    metal_oxide_aggregate_8 f0;
    metal_oxide_aggregate_19 f1;
    uchar metal_oxide_padding_2[2];
};
static_assert(sizeof(metal_oxide_aggregate_61) == 12 && alignof(metal_oxide_aggregate_61) == 4, "metal_oxide_aggregate_61 layout");

struct metal_oxide_aggregate_62 {
    float f0;
};
static_assert(sizeof(metal_oxide_aggregate_62) == 4 && alignof(metal_oxide_aggregate_62) == 4, "metal_oxide_aggregate_62 layout");

struct metal_oxide_aggregate_63 {
    metal_oxide_aggregate_62 f0;
    metal_oxide_aggregate_19 f1;
    uchar metal_oxide_padding_2[2];
};
static_assert(sizeof(metal_oxide_aggregate_63) == 8 && alignof(metal_oxide_aggregate_63) == 4, "metal_oxide_aggregate_63 layout");

struct metal_oxide_aggregate_64 {
    float f0;
    ushort f1;
    uchar metal_oxide_padding_2[2];
};
static_assert(sizeof(metal_oxide_aggregate_64) == 8 && alignof(metal_oxide_aggregate_64) == 4, "metal_oxide_aggregate_64 layout");

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
static inline int metal_oxide_fn_5(int v1, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_6(uint v1, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_7(int v1, int v2, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_8(int v1, uint v2, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_9(uint v1, uint v2, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_10(int v1, int v2, metal_oxide_context metal_oxide_ctx);
static inline bool metal_oxide_fn_11(bool v1, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_12(int v1, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_13(uint v1, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_14(metal_oxide_aggregate_7 v1, int v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_1 metal_oxide_fn_15(uint v1, uint v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_22 metal_oxide_fn_16(metal_oxide_pointer_1 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_22 metal_oxide_fn_17(metal_oxide_pointer_1 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_pointer_1 metal_oxide_fn_18(metal_oxide_pointer_1 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_pointer_4 metal_oxide_fn_19(metal_oxide_pointer_1 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_pointer_2 metal_oxide_fn_20(metal_oxide_pointer_1 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_pointer_3 metal_oxide_fn_21(metal_oxide_pointer_4 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_pointer_3 metal_oxide_fn_22(metal_oxide_pointer_3 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_23 metal_oxide_fn_23(metal_oxide_aggregate_22 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_23 metal_oxide_fn_24(metal_oxide_aggregate_22 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_24 metal_oxide_fn_25(metal_oxide_aggregate_23 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_24 metal_oxide_fn_26(metal_oxide_aggregate_23 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_25 metal_oxide_fn_27(metal_oxide_aggregate_24 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_25 metal_oxide_fn_28(metal_oxide_aggregate_24 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_12 metal_oxide_fn_29(metal_oxide_aggregate_8 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_12 metal_oxide_fn_30(metal_oxide_aggregate_8 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_8 metal_oxide_fn_31(metal_oxide_aggregate_8 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_pointer_6 metal_oxide_fn_32(metal_oxide_pointer_6 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_27 metal_oxide_fn_33(metal_oxide_pointer_5 v1, metal_oxide_context metal_oxide_ctx);
static inline bool metal_oxide_fn_34(metal_oxide_pointer_7 v1, metal_oxide_pointer_7 v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_27 metal_oxide_fn_35(metal_oxide_pointer_7 v1, metal_oxide_pointer_7 v2, metal_oxide_context metal_oxide_ctx);
static inline ulong metal_oxide_fn_36(ulong v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline bool metal_oxide_fn_37(metal_oxide_aggregate_12 v1, bool v2, metal_oxide_context metal_oxide_ctx);
static inline bool metal_oxide_fn_38(metal_oxide_aggregate_12 v1, bool v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_pointer_5 metal_oxide_fn_39(metal_oxide_pointer_5 v1, metal_oxide_context metal_oxide_ctx);
static inline ulong metal_oxide_fn_40(metal_oxide_pointer_8 v1, metal_oxide_context metal_oxide_ctx);
static inline ulong metal_oxide_fn_41(ulong v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_26 metal_oxide_fn_42(ulong v1, metal_oxide_context metal_oxide_ctx);
static inline ulong metal_oxide_fn_43(ulong v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_30 metal_oxide_fn_44(ulong v1, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_45(metal_oxide_aggregate_30 v1, uint v2, metal_oxide_context metal_oxide_ctx);
static inline bool metal_oxide_fn_46(bool v2, uint v3, metal_oxide_context metal_oxide_ctx);
static inline bool metal_oxide_fn_47(metal_oxide_aggregate_8 v1, bool v2, metal_oxide_aggregate_32 v3, metal_oxide_context metal_oxide_ctx);
static inline bool metal_oxide_fn_48(metal_oxide_pointer_9 v1, bool v2, uint v3, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_49(uint v1, uint v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_50(metal_oxide_aggregate_25 v1, float v2, metal_oxide_context metal_oxide_ctx);
static inline void metal_oxide_fn_51(metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_52(metal_oxide_aggregate_24 v1, float v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_53(metal_oxide_aggregate_24 v1, float v2, metal_oxide_context metal_oxide_ctx);
static inline void metal_oxide_fn_54(metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_33 metal_oxide_fn_55(metal_oxide_pointer_10 v1, float v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_35 metal_oxide_fn_56(metal_oxide_pointer_11 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_33 metal_oxide_fn_57(float v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_34 metal_oxide_fn_58(metal_oxide_pointer_12 v1, float v2, metal_oxide_aggregate_35 v3, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_37 metal_oxide_fn_59(metal_oxide_pointer_11 v1, metal_oxide_aggregate_35 v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_34 metal_oxide_fn_60(float v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_36 metal_oxide_fn_61(metal_oxide_pointer_13 v1, float v2, metal_oxide_aggregate_37 v3, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_40 metal_oxide_fn_62(metal_oxide_pointer_13 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_pointer_4 metal_oxide_fn_63(metal_oxide_pointer_4 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_pointer_4 metal_oxide_fn_64(metal_oxide_pointer_15 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_pointer_4 metal_oxide_fn_65(metal_oxide_pointer_3 v1, metal_oxide_context metal_oxide_ctx);
static inline bool metal_oxide_fn_66(metal_oxide_pointer_15 v1, metal_oxide_pointer_15 v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_36 metal_oxide_fn_67(metal_oxide_pointer_14 v1, float v2, metal_oxide_pointer_4 v3, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_34 metal_oxide_fn_68(metal_oxide_pointer_16 v1, float v2, metal_oxide_pointer_4 v3, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_33 metal_oxide_fn_69(float v2, metal_oxide_pointer_4 v3, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_70(float v2, metal_oxide_pointer_4 v3, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_4 metal_oxide_fn_71(metal_oxide_pointer_4 v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_72(float v2, metal_oxide_aggregate_4 v3, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_44 metal_oxide_fn_73(metal_oxide_aggregate_8 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_44 metal_oxide_fn_74(metal_oxide_aggregate_8 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_75(metal_oxide_aggregate_44 v1, float v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_76(metal_oxide_aggregate_44 v1, float v2, metal_oxide_context metal_oxide_ctx);
static inline ulong metal_oxide_fn_77(metal_oxide_pointer_5 v1, metal_oxide_context metal_oxide_ctx);
static inline ulong metal_oxide_fn_78(ulong v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_79(metal_oxide_pointer_6 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_80(metal_oxide_pointer_7 v1, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_81(uint v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_82(uint v1, uint v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_83(float v2, uint v3, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_34 metal_oxide_fn_84(metal_oxide_aggregate_33 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_46 metal_oxide_fn_85(metal_oxide_aggregate_33 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_36 metal_oxide_fn_86(metal_oxide_aggregate_34 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_48 metal_oxide_fn_87(metal_oxide_aggregate_34 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_34 metal_oxide_fn_88(metal_oxide_aggregate_47 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_39 metal_oxide_fn_89(metal_oxide_aggregate_36 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_36 metal_oxide_fn_90(metal_oxide_aggregate_38 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_36 metal_oxide_fn_91(float v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_34 metal_oxide_fn_92(metal_oxide_aggregate_36 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_33 metal_oxide_fn_93(metal_oxide_aggregate_34 v1, metal_oxide_context metal_oxide_ctx);
static inline ushort metal_oxide_fn_94(ushort v1, ushort v2, metal_oxide_context metal_oxide_ctx);
static inline ushort metal_oxide_fn_95(ushort v1, ushort v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_13 metal_oxide_fn_96(float v1, metal_oxide_pointer_1 v2, ushort v3, metal_oxide_context metal_oxide_ctx);
static inline char metal_oxide_fn_97(char v1, char v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_50 metal_oxide_fn_98(metal_oxide_aggregate_4 v1, float v2, metal_oxide_aggregate_50 v3, metal_oxide_context metal_oxide_ctx);
static inline ulong metal_oxide_fn_99(metal_oxide_aggregate_8 v1, metal_oxide_context metal_oxide_ctx);
static inline ulong metal_oxide_fn_100(metal_oxide_aggregate_26 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_17 metal_oxide_fn_101(metal_oxide_aggregate_8 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_17 metal_oxide_fn_102(metal_oxide_aggregate_8 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_51 metal_oxide_fn_103(metal_oxide_aggregate_17 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_51 metal_oxide_fn_104(metal_oxide_aggregate_17 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_52 metal_oxide_fn_105(metal_oxide_aggregate_51 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_52 metal_oxide_fn_106(metal_oxide_aggregate_51 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_51 metal_oxide_fn_107(metal_oxide_aggregate_51 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_53 metal_oxide_fn_108(metal_oxide_aggregate_52 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_53 metal_oxide_fn_109(metal_oxide_aggregate_52 v1, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_110(metal_oxide_aggregate_53 v1, int v2, metal_oxide_context metal_oxide_ctx);
static inline void metal_oxide_fn_111(metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_112(metal_oxide_aggregate_52 v1, int v2, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_113(metal_oxide_aggregate_52 v1, int v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_55 metal_oxide_fn_114(metal_oxide_pointer_17 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_56 metal_oxide_fn_115(metal_oxide_aggregate_55 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_116(metal_oxide_pointer_17 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_117(metal_oxide_pointer_18 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_118(metal_oxide_pointer_6 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_119(metal_oxide_pointer_6 v1, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_120(uint v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline uint metal_oxide_fn_121(uint v1, uint v2, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_122(int v2, uint v3, metal_oxide_context metal_oxide_ctx);
static inline uchar metal_oxide_fn_123(uint v2, metal_oxide_context metal_oxide_ctx);
static inline uchar metal_oxide_fn_124(uchar v1, uchar v2, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_125(int v2, uchar v3, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_126(int v1, int v2, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_127(metal_oxide_aggregate_56 v1, int v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_128(metal_oxide_pointer_19 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_129(metal_oxide_pointer_20 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_130(metal_oxide_pointer_17 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_131(metal_oxide_pointer_18 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_132(metal_oxide_pointer_6 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_133(metal_oxide_pointer_6 v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_134(uint v1, ulong v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_9 metal_oxide_fn_135(uint v1, uint v2, metal_oxide_context metal_oxide_ctx);
static inline bool metal_oxide_fn_136(metal_oxide_pointer_7 v1, metal_oxide_pointer_7 v2, metal_oxide_context metal_oxide_ctx);
static inline ushort metal_oxide_fn_137(ushort v1, metal_oxide_context metal_oxide_ctx);
static inline ushort metal_oxide_fn_138(ushort v1, ushort v2, metal_oxide_context metal_oxide_ctx);
static inline ushort metal_oxide_fn_139(ushort v1, ushort v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_18 metal_oxide_fn_140(metal_oxide_aggregate_17 v1, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_18 metal_oxide_fn_141(metal_oxide_aggregate_17 v1, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_142(metal_oxide_aggregate_18 v1, float v2, metal_oxide_aggregate_19 v3, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_59 metal_oxide_fn_143(metal_oxide_aggregate_19 v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_144(metal_oxide_aggregate_17 v1, float v2, metal_oxide_aggregate_59 v3, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_145(metal_oxide_aggregate_8 v1, float v2, metal_oxide_aggregate_59 v3, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_146(metal_oxide_pointer_21 v1, float v2, uint v3, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_147(uint v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_148(metal_oxide_pointer_22 v1, float v2, float v3, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_61 metal_oxide_fn_149(metal_oxide_aggregate_8 v1, metal_oxide_aggregate_19 v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_61 metal_oxide_fn_150(metal_oxide_aggregate_8 v1, metal_oxide_aggregate_19 v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_151(metal_oxide_aggregate_61 v1, float v2, metal_oxide_aggregate_62 v3, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_63 metal_oxide_fn_152(metal_oxide_aggregate_19 v1, metal_oxide_aggregate_62 v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_153(metal_oxide_aggregate_8 v1, float v2, metal_oxide_aggregate_63 v3, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_154(metal_oxide_pointer_23 v1, float v2, uint v3, metal_oxide_context metal_oxide_ctx);
static inline ushort metal_oxide_fn_155(metal_oxide_pointer_22 v1, uint v2, metal_oxide_context metal_oxide_ctx);
static inline float metal_oxide_fn_156(metal_oxide_pointer_24 v1, float v2, ushort v3, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_157(int v1, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_158(int v1, int v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_21 metal_oxide_fn_159(bool v1, metal_oxide_aggregate_7 v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_7 metal_oxide_fn_160(metal_oxide_aggregate_21 v1, metal_oxide_aggregate_7 v2, metal_oxide_context metal_oxide_ctx);
static inline metal_oxide_aggregate_7 metal_oxide_fn_161(bool v1, int v2, metal_oxide_context metal_oxide_ctx);
static inline int metal_oxide_fn_162(metal_oxide_aggregate_7 v1, metal_oxide_context metal_oxide_ctx);
static inline short metal_oxide_fn_163(short v1, metal_oxide_context metal_oxide_ctx);
static inline ushort metal_oxide_fn_164(ushort v1, metal_oxide_context metal_oxide_ctx);

kernel void generated(device const uint * metal_oxide_arg_1 [[buffer(0)]], device uint * metal_oxide_arg_2 [[buffer(1)]], constant uint & metal_oxide_arg_3 [[buffer(2)]], device atomic_int * metal_oxide_arg_4 [[buffer(3)]], uint3 metal_oxide_thread_idx [[thread_position_in_threadgroup]], uint3 metal_oxide_block_idx [[threadgroup_position_in_grid]], uint3 metal_oxide_block_dim [[threads_per_threadgroup]], uint3 metal_oxide_grid_dim [[threadgroups_per_grid]], uint metal_oxide_simd_lane [[thread_index_in_simdgroup]], uint metal_oxide_simd_size [[threads_per_simdgroup]], uint metal_oxide_simd_group [[simdgroup_index_in_threadgroup]], uint metal_oxide_simd_count [[simdgroups_per_threadgroup]], const device uint* metal_oxide_gate [[buffer(4)]], device atomic_uint* metal_oxide_failure [[buffer(5)]]) {
    bool metal_oxide_failed = false;
    metal_oxide_context metal_oxide_ctx = {metal_oxide_thread_idx, metal_oxide_block_idx, metal_oxide_block_dim, metal_oxide_grid_dim, 0, metal_oxide_simd_lane, metal_oxide_simd_size, metal_oxide_simd_group, metal_oxide_simd_count, metal_oxide_gate, metal_oxide_failure, &metal_oxide_failed};
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
        v29 = metal_oxide_fn_15(v30, v31, metal_oxide_ctx);
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
    uint v15;
    uchar v16;
    bool v17;
    metal_oxide_aggregate_3 v18;
    uchar v19;
    int v21;
    int v22;
    int v23;
    metal_oxide_aggregate_5 v24;
    uchar v25;
    int v27;
    int v28;
    char v29;
    char v30;
    char v31;
    char v32;
    int v37;
    int v38;
    int v43;
    int v44;
    int v45;
    uint v50;
    int v53;
    threadgroup atomic_int * v54;
    uint v55;
    uint v56;
    uint v57;
    uint v58;
    uint v59;
    uint v60;
    uint v61;
    bool v66;
    bool v67;
    metal_oxide_aggregate_6 v69;
    uchar v70;
    int v74;
    device atomic_int * v75;
    uint v76;
    uint v77;
    uint v78;
    uchar v79;
    uint v80;
    uint v81;
    int v84;
    metal_oxide_aggregate_7 v85;
    int v86;
    metal_oxide_aggregate_8 v89;
    metal_oxide_aggregate_8 v90;
    metal_oxide_aggregate_8 v91;
    int v100;
    uint v101;
    int v102;
    uint v103;
    bool v104;
    v6 = v4;
    v8 = v2;
    v7 = (v8 % 1u);
    atomic_store_explicit(&v6[v7], as_type<int>(0x00000001u), memory_order_relaxed);
    metal_oxide_fn_4(metal_oxide_ctx);
    v13 = v4;
    v18 = metal_oxide_aggregate_3{0, {}};
    v19 = v18.f0;
    if (v19 == uchar(0u)) {
        v17 = true;
    }
    else {
        v17 = false;
    }
    v16 = uchar(v17);
    v15 = uint(v16);
    v14 = (v15 % 1u);
    v24 = metal_oxide_aggregate_5{1, {}, uchar(2u)};
    v25 = v24.f0;
    if (v25 == uchar(0u)) {
        v23 = as_type<int>(0x9bda28e4u);
    }
    else {
        v23 = as_type<int>(0x9bd72713u);
    }
    v27 = metal_oxide_fn_5(as_type<int>(0xffffffffu), metal_oxide_ctx);
    v22 = metal_oxide_fn_7(v23, v27, metal_oxide_ctx);
    v32 = as_type<char>(uchar(uint(as_type<char>(uchar(1u))) & uint(as_type<char>(uchar(63u)))));
    v31 = as_type<char>(uchar(uint(v32) | uint(as_type<char>(uchar(1u)))));
    v30 = (as_type<char>(uchar(128u)) / v31);
    v29 = as_type<char>(uchar(uint(v30) & uint(as_type<char>(uchar(3u)))));
    if (v29 == as_type<char>(uchar(0u))) {
        v38 = (as_type<int>(0x5fac1755u) & as_type<int>(0x3fffffffu));
        v37 = (v38 | as_type<int>(0x00000001u));
        v28 = ((as_type<int>(0xffffffffu) ^ as_type<int>(uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x))) / v37);
    }
    else if (v29 == as_type<char>(uchar(1u))) {
        v43 = (as_type<int>(0x84bba52cu) & as_type<int>(0x00000003u));
        if (v43 == as_type<int>(0x00000000u)) {
            v28 = as_type<int>(0x00000000u);
        }
        else if (v43 == as_type<int>(0x00000001u)) {
            v28 = as_type<int>(0x461da5b4u);
        }
        else if (v43 == as_type<int>(0x00000002u)) {
            v28 = as_type<int>(0x640f8471u);
        }
        else {
            v28 = as_type<int>(0xe2ec7d81u);
        }
    }
    else if (v29 == as_type<char>(uchar(2u))) {
        v45 = (as_type<int>(0x00000000u) & as_type<int>(0x3fffffffu));
        v44 = (v45 | as_type<int>(0x00000001u));
        v28 = ((as_type<int>(0x00000001u) ^ as_type<int>(uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x))) / v44);
    }
    else {
        v50 = 4294967295u;
        v28 = metal_oxide_fn_8(as_type<int>(0x7f4ea156u), v50, metal_oxide_ctx);
    }
    v21 = metal_oxide_fn_10(v22, v28, metal_oxide_ctx);
    v12 = atomic_fetch_or_explicit(&v13[v14], v21, memory_order_relaxed);
    metal_oxide_fn_4(metal_oxide_ctx);
    v54 = v4;
    v58 = v1;
    v57 = (v58 | 1u);
    v61 = v1;
    v60 = uint(~uint(v61));
    v59 = (v60 | 1u);
    v56 = ((v57 ^ uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x)) / v59);
    v55 = (v56 % 1u);
    v53 = atomic_load_explicit(&v54[v55], memory_order_relaxed);
    metal_oxide_fn_4(metal_oxide_ctx);
    v69 = metal_oxide_aggregate_6{0, uchar(187u)};
    v70 = v69.f0;
    if (v70 == uchar(0u)) {
    }
    else {
    }
    v67 = false;
    v66 = metal_oxide_fn_11(v67, metal_oxide_ctx);
    v75 = v3;
    v79 = uchar(uint(uchar(251u)) & uint(uchar(3u)));
    if (v79 == uchar(0u)) {
        v78 = 0u;
    }
    else if (v79 == uchar(1u)) {
        v78 = v2;
    }
    else if (v79 == uchar(2u)) {
        v78 = 0u;
    }
    else {
        v78 = v1;
    }
    v81 = metal_oxide_fn_12(as_type<int>(0x00000000u), metal_oxide_ctx);
    v80 = (v81 | 1u);
    v77 = ((v78 ^ uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x)) % v80);
    v76 = (v77 % 4u);
    v85 = metal_oxide_aggregate_7{1, as_type<int>(0x00000001u)};
    v86 = v53;
    v84 = metal_oxide_fn_14(v85, v86, metal_oxide_ctx);
    v74 = atomic_fetch_or_explicit(&v75[v76], v84, memory_order_relaxed);
    v90 = metal_oxide_aggregate_8{0u, 3u};
    v89 = v90;
    v91 = v89;
    while (true) {
        v103 = v91.f0;
        v104 = (v103 < v91.f1);
        if (v104) {
        }
        else {
            break;
        }
        v91.f0 = (v103 + 1u);
        v100 = v53;
        v53 = v100;
        continue;
    }
    v102 = v53;
    v101 = uint(v102);
    v0 = metal_oxide_aggregate_0{{v101}};
    return v0;
}

static inline void metal_oxide_fn_4(metal_oxide_context metal_oxide_ctx) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
}

static inline int metal_oxide_fn_5(int v1, metal_oxide_context metal_oxide_ctx) {
    int v0;
    uint v2;
    uint v3;
    int v4;
    v4 = v1;
    v3 = uint(v4);
    v2 = metal_oxide_fn_6(v3, metal_oxide_ctx);
    v0 = as_type<int>(uint(v2));
    return v0;
}

static inline uint metal_oxide_fn_6(uint v1, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uint v2;
    v2 = v1;
    v0 = reverse_bits(uint((v2 ^ uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x))));
    return v0;
}

static inline int metal_oxide_fn_7(int v1, int v2, metal_oxide_context metal_oxide_ctx) {
    int v0;
    int v3;
    int v4;
    v3 = v1;
    v4 = v2;
    v0 = as_type<int>(uint(uint(int(v3)) * uint(int(v4))));
    return v0;
}

static inline int metal_oxide_fn_8(int v1, uint v2, metal_oxide_context metal_oxide_ctx) {
    int v0;
    uint v3;
    uint v4;
    int v5;
    uint v6;
    v5 = v1;
    v4 = uint(v5);
    v6 = v2;
    v3 = metal_oxide_fn_9(v4, v6, metal_oxide_ctx);
    v0 = as_type<int>(uint(v3));
    return v0;
}

static inline uint metal_oxide_fn_9(uint v1, uint v2, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uint v4;
    uint v5;
    v4 = v1;
    v5 = v2;
    v0 = rotate(v4, uint((0u - v5) & 31u));
    return v0;
}

static inline int metal_oxide_fn_10(int v1, int v2, metal_oxide_context metal_oxide_ctx) {
    int v0;
    int v3;
    int v4;
    v3 = v1;
    v4 = v2;
    v0 = as_type<int>(uint(uint(int(v3)) + uint(int(v4))));
    return v0;
}

static inline bool metal_oxide_fn_11(bool v1, metal_oxide_context metal_oxide_ctx) {
    bool v0;
    bool v2;
    v2 = v1;
    v0 = simd_all(v2);
    return v0;
}

static inline uint metal_oxide_fn_12(int v1, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uint v2;
    int v3;
    v3 = v1;
    v2 = uint(v3);
    v0 = metal_oxide_fn_13(v2, metal_oxide_ctx);
    return v0;
}

static inline uint metal_oxide_fn_13(uint v1, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uint v3;
    v3 = v1;
    v0 = uint(popcount(v3));
    return v0;
}

static inline int metal_oxide_fn_14(metal_oxide_aggregate_7 v1, int v2, metal_oxide_context metal_oxide_ctx) {
    int v0;
    uchar v3;
    int v4;
    bool v5;
    v5 = false;
    v5 = true;
    v3 = v1.f0;
    if (v3 == uchar(0u)) {
        v5 = false;
        v0 = v2;
    }
    else {
        v4 = v1.f1;
        v0 = v4;
    }
    if (v5) {
    }
    else {
    }
    return v0;
}

static inline metal_oxide_aggregate_1 metal_oxide_fn_15(uint v1, uint v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_1 v0;
    metal_oxide_aggregate_8 v4;
    metal_oxide_aggregate_8 v5;
    metal_oxide_aggregate_8 v6;
    metal_oxide_aggregate_10 v15;
    bool v16;
    metal_oxide_aggregate_11 v17;
    uint v18;
    uchar v19;
    bool v21;
    metal_oxide_aggregate_12 v22;
    metal_oxide_aggregate_8 v23;
    metal_oxide_aggregate_13 v27;
    float v28;
    metal_oxide_aggregate_25 v29;
    metal_oxide_aggregate_24 v30;
    metal_oxide_aggregate_23 v31;
    metal_oxide_aggregate_22 v32;
    metal_oxide_pointer_1 v33;
    metal_oxide_pointer_0 v34;
    float v36;
    metal_oxide_pointer_1 v38;
    metal_oxide_pointer_1 v39;
    metal_oxide_pointer_1 v40;
    metal_oxide_pointer_0 v41;
    metal_oxide_aggregate_14 v42;
    ulong v43;
    ulong v44;
    ushort v45;
    uint v46;
    ulong v48;
    ulong v49;
    uchar v50;
    uint v51;
    ushort v53;
    ushort v54;
    ushort v55;
    bool v56;
    ulong v57;
    ushort v58;
    uint v59;
    bool v60;
    ulong v61;
    ulong v62;
    ulong v63;
    ulong v64;
    uint v65;
    bool v66;
    ushort v69;
    metal_oxide_aggregate_16 v70;
    metal_oxide_aggregate_15 v71;
    uchar v72;
    ushort v74;
    bool v78;
    short v79;
    short v80;
    uint v81;
    short v82;
    uint v83;
    uint v87;
    float v88;
    metal_oxide_aggregate_18 v89;
    metal_oxide_aggregate_17 v90;
    metal_oxide_aggregate_8 v91;
    metal_oxide_aggregate_19 v93;
    uint v94;
    int v95;
    int v96;
    int v97;
    ulong v98;
    int v99;
    uint v100;
    int v101;
    metal_oxide_aggregate_7 v102;
    bool v103;
    metal_oxide_aggregate_20 v104;
    uchar v105;
    metal_oxide_aggregate_21 v106;
    metal_oxide_aggregate_7 v107;
    metal_oxide_aggregate_7 v108;
    int v110;
    metal_oxide_aggregate_7 v111;
    bool v112;
    int v113;
    int v114;
    uint v115;
    int v116;
    uchar v117;
    uint v118;
    short v119;
    short v120;
    uint v123;
    uint v126;
    bool v127;
    metal_oxide_pointer_1 v128;
    ulong v129;
    bool v130;
    bool v131;
    v5 = metal_oxide_aggregate_8{0u, 2u};
    v4 = v5;
    v6 = v4;
    #pragma clang loop unroll(disable)
    while (true) {
        v126 = v6.f0;
        v127 = (v126 < v6.f1);
        if (v127) {
        }
        else {
            break;
        }
        v6.f0 = (v126 + 1u);
        v18 = v2;
        v17 = metal_oxide_aggregate_11{1, {}, v18};
        v19 = v17.f0;
        if (v19 == uchar(1u)) {
            v16 = true;
        }
        else {
            v16 = false;
        }
        v23 = metal_oxide_aggregate_8{0u, 4u};
        v22 = metal_oxide_fn_29(v23, 3ul, metal_oxide_ctx);
        v21 = metal_oxide_fn_37(v22, false, metal_oxide_ctx);
        v15 = metal_oxide_aggregate_10{{v16, false, v21}};
        v34 = &v15;
        v33 = metal_oxide_pointer_1{(*v34).elements, ((3ul) ^ ulong(uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x)))};
        v32 = metal_oxide_fn_16(v33, metal_oxide_ctx);
        v31 = metal_oxide_fn_23(v32, 4ul, metal_oxide_ctx);
        v30 = metal_oxide_fn_25(v31, 4ul, metal_oxide_ctx);
        v29 = metal_oxide_fn_27(v30, metal_oxide_ctx);
        v36 = float(uchar(1u));
        v28 = metal_oxide_fn_50(v29, v36, metal_oxide_ctx);
        v41 = &v15;
        v46 = v2;
        v45 = ushort(v46);
        v44 = ulong(v45);
        v43 = (v44 % 5ul);
        v51 = v2;
        v50 = uchar(v51);
        v49 = ulong(v50);
        v48 = (v49 % 5ul);
        v42 = metal_oxide_aggregate_14{v43, v48};
        v128 = metal_oxide_pointer_1{(*v41).elements, ((3ul) ^ ulong(uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x)))};
        v129 = v128.length;
        v131 = (v42.f0 <= v42.f1);
        if (!(v131)) {
            metal_oxide_fail(metal_oxide_ctx, 7u);
            return {};
        }
        v130 = (v42.f1 <= v129);
        if (!(v130)) {
            metal_oxide_fail(metal_oxide_ctx, 11u);
            return {};
        }
        v40 = metal_oxide_pointer_1{v128.data + v42.f0, ((v42.f1 - v42.f0) ^ ulong(uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x)))};
        v39 = v40;
        v38 = v39;
        v54 = ushort(~uint(ushort(65535u)));
        v55 = metal_oxide_fn_94(ushort(56408u), ushort(0u), metal_oxide_ctx);
        v53 = metal_oxide_fn_95(v54, v55, metal_oxide_ctx);
        v27 = metal_oxide_fn_96(v28, v38, v53, metal_oxide_ctx);
        if (*metal_oxide_ctx.failed) { return {}; }
        v59 = v2;
        v58 = ushort(v59);
        v57 = ulong(v58);
        v60 = (v57 < 3ul);
        if (!(v60)) {
            metal_oxide_fail(metal_oxide_ctx, 8u);
            return {};
        }
        v56 = v15.elements[v57];
        v65 = v2;
        v64 = ulong(v65);
        v66 = (v64 == 0ul);
        if (v66) {
            metal_oxide_fail(metal_oxide_ctx, 9u);
            return {};
        }
        v63 = ((18446744073709551615ul ^ ulong(uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x))) / v64);
        v62 = ulong(v63);
        v61 = (v62 % 3ul);
        v15.elements[v61] = bool(uint(v56) ^ uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x));
        continue;
    }
    v71 = metal_oxide_aggregate_15{{as_type<float>(0x43fe8000u), as_type<float>(0xc45b9000u), as_type<float>(0x44586000u), as_type<float>(0x43a6e000u)}};
    v70 = metal_oxide_aggregate_16{0, v71, {}};
    v72 = v70.f0;
    if (v72 == uchar(0u)) {
        v74 = metal_oxide_fn_137(ushort(65535u), metal_oxide_ctx);
        v69 = metal_oxide_fn_139(v74, ushort(20990u), metal_oxide_ctx);
    }
    else {
        v69 = metal_oxide_fn_94(ushort(18634u), ushort(65535u), metal_oxide_ctx);
    }
    v81 = (4294967295u & 3u);
    if (v81 == 0u) {
        v80 = as_type<short>(ushort(uint(as_type<short>(ushort(0u))) ^ uint(as_type<short>(ushort(65535u)))));
    }
    else if (v81 == 1u) {
        v80 = as_type<short>(ushort(14306u));
    }
    else if (v81 == 2u) {
        v80 = as_type<short>(ushort(uint(as_type<short>(ushort(46164u))) ^ uint(as_type<short>(ushort(0u)))));
    }
    else {
        v80 = as_type<short>(ushort(uint(as_type<short>(ushort(1u))) & uint(as_type<short>(ushort(65535u)))));
    }
    v83 = v1;
    v82 = as_type<short>(ushort(v83));
    v79 = as_type<short>(ushort(uint(v80) ^ uint(v82)));
    v78 = (v79 < as_type<short>(ushort(0u)));
    if (v78) {
    }
    else {
        metal_oxide_fail(metal_oxide_ctx, 10u);
        return {};
    }
    v91 = metal_oxide_aggregate_8{0u, 7u};
    v90 = metal_oxide_fn_101(v91, metal_oxide_ctx);
    v89 = metal_oxide_fn_140(v90, metal_oxide_ctx);
    v93 = metal_oxide_aggregate_19{v69};
    v88 = metal_oxide_fn_142(v89, as_type<float>(0xc3600000u), v93, metal_oxide_ctx);
    v87 = as_type<uint>(v88);
    v98 = (1ul & 3ul);
    if (v98 == 0ul) {
        v97 = as_type<int>(0x04611400u);
    }
    else if (v98 == 1ul) {
        v97 = as_type<int>(0x42d1c95fu);
    }
    else if (v98 == 2ul) {
        v97 = as_type<int>(0x6807b1ffu);
    }
    else {
        v97 = as_type<int>(0x00000000u);
    }
    v96 = metal_oxide_fn_5(v97, metal_oxide_ctx);
    v99 = metal_oxide_fn_157(as_type<int>(0xe6bc7fd2u), metal_oxide_ctx);
    v95 = metal_oxide_fn_126(v96, v99, metal_oxide_ctx);
    v94 = uint(v95);
    v104 = metal_oxide_aggregate_20{0, as_type<float>(0xc455b000u), {}};
    v105 = v104.f0;
    if (v105 == uchar(1u)) {
        v103 = true;
    }
    else {
        v103 = false;
    }
    if (v103) {
        v107 = metal_oxide_aggregate_7{1, as_type<int>(0x00000000u)};
        v106 = metal_oxide_fn_159(false, v107, metal_oxide_ctx);
        v108 = metal_oxide_aggregate_7{1, as_type<int>(0x80000000u)};
        v102 = metal_oxide_fn_160(v106, v108, metal_oxide_ctx);
    }
    else {
        v102 = metal_oxide_aggregate_7{1, as_type<int>(0x80000000u)};
    }
    v115 = v1;
    v114 = as_type<int>(uint(v115));
    v113 = (as_type<int>(0xffffffffu) ^ v114);
    v112 = (v113 >= as_type<int>(0x3215a8f1u));
    v117 = uchar(uint(uchar(89u)) & uint(uchar(3u)));
    if (v117 == uchar(0u)) {
        v116 = as_type<int>(0x24f1878au);
    }
    else if (v117 == uchar(1u)) {
        v116 = as_type<int>(0x1cd69074u);
    }
    else if (v117 == uchar(2u)) {
        v116 = as_type<int>(0x00000000u);
    }
    else {
        v116 = as_type<int>(0x80000000u);
    }
    v111 = metal_oxide_fn_161(v112, v116, metal_oxide_ctx);
    v110 = metal_oxide_fn_162(v111, metal_oxide_ctx);
    if (*metal_oxide_ctx.failed) { return {}; }
    v101 = metal_oxide_fn_14(v102, v110, metal_oxide_ctx);
    v100 = uint(v101);
    v123 = uint(as_type<int>(0xffffffffu));
    v120 = as_type<short>(ushort(uint(as_type<short>(ushort(8907u))) << (uint(v123) & 15u)));
    v119 = metal_oxide_fn_163(v120, metal_oxide_ctx);
    v118 = uint(v119);
    v0 = metal_oxide_aggregate_1{{v87, v94, v100, v118}};
    return v0;
}

static inline metal_oxide_aggregate_22 metal_oxide_fn_16(metal_oxide_pointer_1 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_22 v0;
    metal_oxide_pointer_1 v2;
    v2 = v1;
    v0 = metal_oxide_fn_17(v2, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_22 metal_oxide_fn_17(metal_oxide_pointer_1 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_22 v0;
    ulong v2;
    metal_oxide_pointer_1 v3;
    metal_oxide_pointer_4 v4;
    metal_oxide_pointer_1 v5;
    metal_oxide_pointer_1 v6;
    metal_oxide_pointer_4 v7;
    metal_oxide_pointer_3 v10;
    metal_oxide_pointer_3 v11;
    metal_oxide_pointer_4 v12;
    ulong v13;
    metal_oxide_pointer_4 v14;
    metal_oxide_pointer_4 v15;
    v3 = v1;
    v2 = v3.length;
    v6 = v1;
    v5 = metal_oxide_fn_18(v6, metal_oxide_ctx);
    v4 = metal_oxide_fn_19(v5, metal_oxide_ctx);
    v12 = v4;
    v11 = metal_oxide_fn_21(v12, metal_oxide_ctx);
    v13 = v2;
    v10 = metal_oxide_fn_22(v11, v13, metal_oxide_ctx);
    v7 = (metal_oxide_pointer_4)(v10);
    v14 = v4;
    v15 = v7;
    v0 = metal_oxide_aggregate_22{v14, v15};
    return v0;
}

static inline metal_oxide_pointer_1 metal_oxide_fn_18(metal_oxide_pointer_1 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_pointer_1 v0;
    metal_oxide_pointer_1 v2;
    metal_oxide_pointer_1 v3;
    v3 = v1;
    v2 = v3;
    v0 = v2;
    return v0;
}

static inline metal_oxide_pointer_4 metal_oxide_fn_19(metal_oxide_pointer_1 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_pointer_4 v0;
    metal_oxide_pointer_3 v2;
    metal_oxide_pointer_3 v3;
    metal_oxide_pointer_2 v4;
    metal_oxide_pointer_1 v5;
    v5 = v1;
    v4 = metal_oxide_fn_20(v5, metal_oxide_ctx);
    v3 = (metal_oxide_pointer_3)(v4.data);
    v2 = v3;
    v0 = (metal_oxide_pointer_4)(v2);
    return v0;
}

static inline metal_oxide_pointer_2 metal_oxide_fn_20(metal_oxide_pointer_1 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_pointer_2 v0;
    metal_oxide_pointer_1 v2;
    v2 = v1;
    v0 = metal_oxide_pointer_2{(thread bool*)(v2.data), v2.length};
    return v0;
}

static inline metal_oxide_pointer_3 metal_oxide_fn_21(metal_oxide_pointer_4 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_pointer_3 v0;
    metal_oxide_pointer_4 v2;
    v2 = v1;
    v0 = (metal_oxide_pointer_3)(v2);
    return v0;
}

static inline metal_oxide_pointer_3 metal_oxide_fn_22(metal_oxide_pointer_3 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_pointer_3 v0;
    metal_oxide_pointer_3 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = (v3 + v4);
    return v0;
}

static inline metal_oxide_aggregate_23 metal_oxide_fn_23(metal_oxide_aggregate_22 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_23 v0;
    metal_oxide_aggregate_22 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_fn_24(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_23 metal_oxide_fn_24(metal_oxide_aggregate_22 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_23 v0;
    metal_oxide_aggregate_22 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_aggregate_23{v3, v4};
    return v0;
}

static inline metal_oxide_aggregate_24 metal_oxide_fn_25(metal_oxide_aggregate_23 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_24 v0;
    metal_oxide_aggregate_23 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_fn_26(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_24 metal_oxide_fn_26(metal_oxide_aggregate_23 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_24 v0;
    metal_oxide_aggregate_23 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_aggregate_24{v3, v4};
    return v0;
}

static inline metal_oxide_aggregate_25 metal_oxide_fn_27(metal_oxide_aggregate_24 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_25 v0;
    metal_oxide_aggregate_24 v3;
    v3 = v1;
    v0 = metal_oxide_fn_28(v3, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_25 metal_oxide_fn_28(metal_oxide_aggregate_24 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_25 v0;
    metal_oxide_aggregate_24 v3;
    v3 = v1;
    v0 = metal_oxide_aggregate_25{v3};
    return v0;
}

static inline metal_oxide_aggregate_12 metal_oxide_fn_29(metal_oxide_aggregate_8 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_12 v0;
    metal_oxide_aggregate_8 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_fn_30(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_12 metal_oxide_fn_30(metal_oxide_aggregate_8 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_12 v0;
    bool v4;
    ulong v5;
    metal_oxide_aggregate_8 v7;
    metal_oxide_aggregate_8 v8;
    ulong v9;
    metal_oxide_aggregate_8 v10;
    ulong v11;
    ulong v12;
    v5 = v2;
    v4 = (v5 != 0ul);
    v8 = v1;
    v9 = v2;
    v7 = metal_oxide_fn_31(v8, v9, metal_oxide_ctx);
    v10 = v7;
    v12 = v2;
    v11 = (v12 - 1ul);
    v0 = metal_oxide_aggregate_12{v10, v11, true};
    return v0;
}

static inline metal_oxide_aggregate_8 metal_oxide_fn_31(metal_oxide_aggregate_8 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_8 v0;
    ulong v3;
    metal_oxide_aggregate_27 v4;
    metal_oxide_pointer_5 v5;
    ulong v6;
    ulong v7;
    ulong v8;
    ulong v9;
    metal_oxide_pointer_6 v10;
    metal_oxide_pointer_6 v11;
    v5 = &v1;
    v4 = metal_oxide_fn_33(v5, metal_oxide_ctx);
    v3 = v4.f0;
    v7 = v3;
    v8 = v2;
    v6 = metal_oxide_fn_36(v7, v8, metal_oxide_ctx);
    v9 = v6;
    v11 = &v1;
    v10 = metal_oxide_fn_32(v11, metal_oxide_ctx);
    (*v10).f1 = uint(v9);
    v0 = v1;
    return v0;
}

static inline metal_oxide_pointer_6 metal_oxide_fn_32(metal_oxide_pointer_6 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_pointer_6 v0;
    metal_oxide_pointer_6 v2;
    v2 = &(*v1);
    v0 = &(*v2);
    return v0;
}

static inline metal_oxide_aggregate_27 metal_oxide_fn_33(metal_oxide_pointer_5 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_27 v0;
    bool v2;
    metal_oxide_pointer_7 v3;
    metal_oxide_pointer_7 v4;
    metal_oxide_pointer_7 v5;
    metal_oxide_pointer_7 v6;
    metal_oxide_pointer_7 v7;
    metal_oxide_pointer_7 v8;
    metal_oxide_aggregate_26 v9;
    v3 = &(*v1).f0;
    v4 = &(*v1).f1;
    v2 = metal_oxide_fn_34(v3, v4, metal_oxide_ctx);
    if (v2) {
        v6 = &(*v1).f0;
        v5 = &(*v6);
        v8 = &(*v1).f1;
        v7 = &(*v8);
        v0 = metal_oxide_fn_35(v5, v7, metal_oxide_ctx);
    }
    else {
        v9 = metal_oxide_aggregate_26{1, 0ul};
        v0 = metal_oxide_aggregate_27{0ul, v9};
    }
    return v0;
}

static inline bool metal_oxide_fn_34(metal_oxide_pointer_7 v1, metal_oxide_pointer_7 v2, metal_oxide_context metal_oxide_ctx) {
    bool v0;
    uint v3;
    uint v4;
    v3 = (*v1);
    v4 = (*v2);
    v0 = (v3 < v4);
    return v0;
}

static inline metal_oxide_aggregate_27 metal_oxide_fn_35(metal_oxide_pointer_7 v1, metal_oxide_pointer_7 v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_27 v0;
    bool v3;
    uint v4;
    uint v5;
    ulong v6;
    uint v7;
    uint v8;
    uint v9;
    ulong v10;
    metal_oxide_aggregate_26 v11;
    ulong v12;
    metal_oxide_aggregate_26 v13;
    v4 = (*v1);
    v5 = (*v2);
    v3 = (v4 <= v5);
    if (v3) {
        v8 = (*v2);
        v9 = (*v1);
        v7 = (v8 - v9);
        v6 = ulong(v7);
        v10 = v6;
        v12 = v6;
        v11 = metal_oxide_aggregate_26{1, v12};
        v0 = metal_oxide_aggregate_27{v10, v11};
    }
    else {
        v13 = metal_oxide_aggregate_26{0, {}};
        v0 = metal_oxide_aggregate_27{0ul, v13};
    }
    return v0;
}

static inline ulong metal_oxide_fn_36(ulong v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    ulong v0;
    ulong v3;
    ulong v4;
    ulong v5;
    ulong v7;
    ulong v8;
    ulong v9;
    bool v11;
    ulong v12;
    ulong v13;
    v4 = v1;
    v5 = v2;
    v3 = ((v4 ^ ulong(uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x))) / v5);
    v8 = v1;
    v9 = v2;
    v7 = ((v8 ^ ulong(uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x))) % v9);
    v12 = v7;
    v11 = (v12 > 0ul);
    if (v11) {
        v13 = v3;
        v0 = (v13 + 1ul);
    }
    else {
        v0 = v3;
    }
    return v0;
}

static inline bool metal_oxide_fn_37(metal_oxide_aggregate_12 v1, bool v2, metal_oxide_context metal_oxide_ctx) {
    bool v0;
    metal_oxide_aggregate_12 v4;
    bool v5;
    v4 = v1;
    v5 = v2;
    v0 = metal_oxide_fn_38(v4, v5, metal_oxide_ctx);
    return v0;
}

static inline bool metal_oxide_fn_38(metal_oxide_aggregate_12 v1, bool v2, metal_oxide_context metal_oxide_ctx) {
    bool v0;
    uint v4;
    metal_oxide_aggregate_30 v5;
    ulong v6;
    ulong v7;
    metal_oxide_pointer_8 v8;
    metal_oxide_pointer_5 v9;
    metal_oxide_pointer_5 v10;
    uint v11;
    bool v12;
    uint v13;
    metal_oxide_aggregate_8 v15;
    metal_oxide_aggregate_8 v16;
    uint v17;
    metal_oxide_aggregate_8 v18;
    bool v26;
    metal_oxide_aggregate_31 v28;
    bool v29;
    uint v30;
    uint v31;
    uint v32;
    uint v33;
    uint v36;
    bool v37;
    v8 = &v1;
    v7 = metal_oxide_fn_40(v8, metal_oxide_ctx);
    v6 = metal_oxide_fn_43(v7, metal_oxide_ctx);
    v5 = metal_oxide_fn_44(v6, metal_oxide_ctx);
    v4 = metal_oxide_fn_45(v5, 4294967295u, metal_oxide_ctx);
    v10 = &v1.f0;
    v9 = metal_oxide_fn_39(v10, metal_oxide_ctx);
    v11 = (*v9).f1;
    v12 = v2;
    v13 = (*v9).f0;
    v17 = v11;
    v16 = metal_oxide_aggregate_8{0u, v17};
    v15 = v16;
    v18 = v15;
    while (true) {
        v36 = v18.f0;
        v37 = (v36 < v18.f1);
        if (v37) {
        }
        else {
            break;
        }
        v18.f0 = (v36 + 1u);
        v29 = v12;
        v30 = v13;
        v28 = metal_oxide_aggregate_31{v29, v30};
        v26 = metal_oxide_fn_46(v28.f0, v28.f1, metal_oxide_ctx);
        v12 = v26;
        v32 = v13;
        v33 = v4;
        v31 = metal_oxide_fn_49(v32, v33, metal_oxide_ctx);
        v13 = v31;
        continue;
    }
    v0 = v12;
    return v0;
}

static inline metal_oxide_pointer_5 metal_oxide_fn_39(metal_oxide_pointer_5 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_pointer_5 v0;
    v0 = v1;
    return v0;
}

static inline ulong metal_oxide_fn_40(metal_oxide_pointer_8 v1, metal_oxide_context metal_oxide_ctx) {
    ulong v0;
    ulong v2;
    ulong v3;
    v3 = (*v1).f1;
    v2 = (v3 + 1ul);
    v0 = metal_oxide_fn_41(v2, metal_oxide_ctx);
    return v0;
}

static inline ulong metal_oxide_fn_41(ulong v1, metal_oxide_context metal_oxide_ctx) {
    ulong v0;
    metal_oxide_aggregate_26 v2;
    ulong v3;
    ulong v5;
    v3 = v1;
    v2 = metal_oxide_fn_42(v3, metal_oxide_ctx);
    v5 = v2.f1;
    v0 = v5;
    return v0;
}

static inline metal_oxide_aggregate_26 metal_oxide_fn_42(ulong v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_26 v0;
    ulong v2;
    bool v3;
    v2 = v1;
    v0 = metal_oxide_aggregate_26{1, v2};
    v3 = (v2 != 0ul);
    v0.f0 = uchar(v3);
    return v0;
}

static inline ulong metal_oxide_fn_43(ulong v1, metal_oxide_context metal_oxide_ctx) {
    ulong v0;
    ulong v2;
    v2 = v1;
    v0 = v2;
    return v0;
}

static inline metal_oxide_aggregate_30 metal_oxide_fn_44(ulong v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_30 v0;
    bool v2;
    ulong v3;
    ulong v4;
    metal_oxide_aggregate_29 v5;
    metal_oxide_aggregate_28 v6;
    uint v7;
    ulong v8;
    v3 = v1;
    v4 = 4294967295ul;
    v2 = (v3 > v4);
    if (v2) {
        v6 = metal_oxide_aggregate_28{2};
        v5 = metal_oxide_aggregate_29{v6};
        v0 = metal_oxide_aggregate_30{1, {}, v5};
    }
    else {
        v8 = v1;
        v7 = uint(v8);
        v0 = metal_oxide_aggregate_30{0, v7, {}};
    }
    return v0;
}

static inline uint metal_oxide_fn_45(metal_oxide_aggregate_30 v1, uint v2, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uchar v3;
    uint v4;
    bool v5;
    uchar v7;
    v5 = false;
    v5 = true;
    v3 = v1.f0;
    if (v3 == uchar(0u)) {
        v4 = v1.f1;
        v0 = v4;
    }
    else {
        v5 = false;
        v0 = v2;
    }
    if (v5) {
    }
    else {
    }
    v7 = v1.f0;
    if (v7 == uchar(0u)) {
    }
    else {
    }
    return v0;
}

static inline bool metal_oxide_fn_46(bool v2, uint v3, metal_oxide_context metal_oxide_ctx) {
    bool v0;
    metal_oxide_aggregate_8 v4;
    metal_oxide_aggregate_32 v5;
    v4 = metal_oxide_aggregate_8{0u, 6u};
    v5 = metal_oxide_aggregate_32{v2};
    v0 = metal_oxide_fn_47(v4, true, v5, metal_oxide_ctx);
    return v0;
}

static inline bool metal_oxide_fn_47(metal_oxide_aggregate_8 v1, bool v2, metal_oxide_aggregate_32 v3, metal_oxide_context metal_oxide_ctx) {
    bool v0;
    bool v4;
    metal_oxide_aggregate_9 v7;
    uint v10;
    bool v11;
    metal_oxide_pointer_9 v12;
    metal_oxide_aggregate_31 v13;
    bool v14;
    uint v15;
    uint v20;
    bool v21;
    v4 = v2;
    while (true) {
        v20 = v1.f0;
        v21 = (v20 < v1.f1);
        if (v21) {
        }
        else {
            break;
        }
        v7 = metal_oxide_aggregate_9{1, v20};
        v1.f0 = (v20 + 1u);
        v10 = v7.f1;
        v12 = &v3;
        v14 = v4;
        v15 = v10;
        v13 = metal_oxide_aggregate_31{v14, v15};
        v11 = metal_oxide_fn_48(v12, v13.f0, v13.f1, metal_oxide_ctx);
        v4 = v11;
        continue;
    }
    v7 = metal_oxide_aggregate_9{0, {}};
    v0 = v4;
    return v0;
}

static inline bool metal_oxide_fn_48(metal_oxide_pointer_9 v1, bool v2, uint v3, metal_oxide_context metal_oxide_ctx) {
    bool v0;
    v0 = (*v1).f0;
    return v0;
}

static inline uint metal_oxide_fn_49(uint v1, uint v2, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uint v3;
    uint v4;
    v3 = v1;
    v4 = v2;
    v0 = (v3 + v4);
    return v0;
}

static inline float metal_oxide_fn_50(metal_oxide_aggregate_25 v1, float v2, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_aggregate_24 v4;
    float v5;
    v4 = v1.f0;
    v5 = v2;
    metal_oxide_fn_51(metal_oxide_ctx);
    v0 = metal_oxide_fn_52(v4, v5, metal_oxide_ctx);
    return v0;
}

static inline void metal_oxide_fn_51(metal_oxide_context metal_oxide_ctx) {
    return;
}

static inline float metal_oxide_fn_52(metal_oxide_aggregate_24 v1, float v2, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_aggregate_24 v4;
    float v5;
    v4 = v1;
    v5 = v2;
    v0 = metal_oxide_fn_53(v4, v5, metal_oxide_ctx);
    return v0;
}

static inline float metal_oxide_fn_53(metal_oxide_aggregate_24 v1, float v2, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_aggregate_33 v4;
    metal_oxide_pointer_10 v5;
    float v6;
    v5 = &v1;
    v6 = v2;
    metal_oxide_fn_54(metal_oxide_ctx);
    v4 = metal_oxide_fn_55(v5, v6, metal_oxide_ctx);
    v0 = v4.f0;
    return v0;
}

static inline void metal_oxide_fn_54(metal_oxide_context metal_oxide_ctx) {
    return;
}

static inline metal_oxide_aggregate_33 metal_oxide_fn_55(metal_oxide_pointer_10 v1, float v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_33 v0;
    bool v4;
    ulong v5;
    float v6;
    metal_oxide_pointer_11 v7;
    metal_oxide_aggregate_34 v8;
    metal_oxide_pointer_12 v9;
    float v10;
    metal_oxide_aggregate_35 v11;
    metal_oxide_pointer_11 v12;
    bool v14;
    v14 = false;
    v14 = true;
    v5 = (*v1).f1;
    v4 = (v5 == 0ul);
    if (v4) {
        v6 = v2;
        v0 = metal_oxide_fn_57(v6, metal_oxide_ctx);
    }
    else {
        v7 = &(*v1).f1;
        v9 = &(*v1).f0;
        v10 = v2;
        v12 = &(*v7);
        v14 = false;
        v11 = metal_oxide_fn_56(v12, metal_oxide_ctx);
        v8 = metal_oxide_fn_58(v9, v10, v11, metal_oxide_ctx);
        v0 = metal_oxide_fn_93(v8, metal_oxide_ctx);
    }
    if (v14) {
    }
    else {
    }
    return v0;
}

static inline metal_oxide_aggregate_35 metal_oxide_fn_56(metal_oxide_pointer_11 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_35 v0;
    v0 = metal_oxide_aggregate_35{v1};
    return v0;
}

static inline metal_oxide_aggregate_33 metal_oxide_fn_57(float v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_33 v0;
    float v2;
    v2 = v1;
    v0 = metal_oxide_aggregate_33{v2};
    return v0;
}

static inline metal_oxide_aggregate_34 metal_oxide_fn_58(metal_oxide_pointer_12 v1, float v2, metal_oxide_aggregate_35 v3, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_34 v0;
    bool v4;
    ulong v5;
    float v6;
    metal_oxide_pointer_11 v7;
    metal_oxide_aggregate_36 v8;
    metal_oxide_pointer_13 v9;
    float v10;
    metal_oxide_aggregate_37 v11;
    metal_oxide_pointer_11 v12;
    metal_oxide_aggregate_35 v13;
    bool v14;
    v14 = false;
    v14 = true;
    v5 = (*v1).f1;
    v4 = (v5 == 0ul);
    if (v4) {
        v6 = v2;
        v0 = metal_oxide_fn_60(v6, metal_oxide_ctx);
    }
    else {
        v7 = &(*v1).f1;
        v9 = &(*v1).f0;
        v10 = v2;
        v12 = &(*v7);
        v14 = false;
        v13 = v3;
        v11 = metal_oxide_fn_59(v12, v13, metal_oxide_ctx);
        v8 = metal_oxide_fn_61(v9, v10, v11, metal_oxide_ctx);
        v0 = metal_oxide_fn_92(v8, metal_oxide_ctx);
    }
    if (v14) {
    }
    else {
    }
    return v0;
}

static inline metal_oxide_aggregate_37 metal_oxide_fn_59(metal_oxide_pointer_11 v1, metal_oxide_aggregate_35 v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_37 v0;
    v0 = metal_oxide_aggregate_37{v1, v2};
    return v0;
}

static inline metal_oxide_aggregate_34 metal_oxide_fn_60(float v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_34 v0;
    float v2;
    v2 = v1;
    v0 = metal_oxide_aggregate_34{0, v2, {}};
    return v0;
}

static inline metal_oxide_aggregate_36 metal_oxide_fn_61(metal_oxide_pointer_13 v1, float v2, metal_oxide_aggregate_37 v3, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_36 v0;
    float v4;
    metal_oxide_aggregate_40 v7;
    metal_oxide_pointer_13 v8;
    uchar v9;
    metal_oxide_pointer_4 v10;
    float v11;
    metal_oxide_aggregate_39 v12;
    metal_oxide_aggregate_36 v13;
    metal_oxide_pointer_14 v14;
    metal_oxide_aggregate_41 v15;
    float v16;
    metal_oxide_pointer_4 v17;
    uchar v18;
    metal_oxide_aggregate_38 v19;
    metal_oxide_aggregate_38 v21;
    float v22;
    float v26;
    v4 = v2;
    uint metal_oxide_exit_1 = 0u;
    while (true) {
        v8 = &(*v1);
        v7 = metal_oxide_fn_62(v8, metal_oxide_ctx);
        v9 = v7.f0;
        if (v9 == uchar(1u)) {
        }
        else {
            metal_oxide_exit_1 = 10u;
            break;
        }
        v10 = v7.f1;
        v14 = &v3;
        v16 = v4;
        v17 = v10;
        v15 = metal_oxide_aggregate_41{v16, v17};
        v13 = metal_oxide_fn_67(v14, v15.f0, v15.f1, metal_oxide_ctx);
        v12 = metal_oxide_fn_89(v13, metal_oxide_ctx);
        v18 = v12.f0;
        if (v18 == uchar(0u)) {
        }
        else {
            metal_oxide_exit_1 = 8u;
            break;
        }
        v22 = v12.f1;
        v11 = v22;
        v4 = v11;
        continue;
    }
    if (metal_oxide_exit_1 == 8u) {
        v19 = v12.f2;
        v21 = v19;
        v0 = metal_oxide_fn_90(v21, metal_oxide_ctx);
    }
    else {
        v26 = v4;
        v0 = metal_oxide_fn_91(v26, metal_oxide_ctx);
    }
    return v0;
}

static inline metal_oxide_aggregate_40 metal_oxide_fn_62(metal_oxide_pointer_13 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_40 v0;
    metal_oxide_pointer_4 v2;
    metal_oxide_pointer_4 v3;
    bool v16;
    metal_oxide_pointer_15 v17;
    metal_oxide_pointer_15 v18;
    metal_oxide_pointer_4 v19;
    metal_oxide_pointer_4 v20;
    metal_oxide_pointer_4 v22;
    metal_oxide_pointer_4 v23;
    metal_oxide_pointer_4 v24;
    metal_oxide_pointer_4 v25;
    metal_oxide_pointer_15 v26;
    metal_oxide_pointer_4 v27;
    v2 = (*v1).f0;
    v3 = (*v1).f1;
    v17 = &v2;
    v20 = v3;
    v19 = v20;
    v18 = &v19;
    v16 = metal_oxide_fn_66(v17, v18, metal_oxide_ctx);
    if (v16) {
        v0 = metal_oxide_aggregate_40{0, {}};
    }
    else {
        v23 = v2;
        v22 = metal_oxide_fn_63(v23, 1ul, metal_oxide_ctx);
        (*v1).f0 = v22;
        v27 = v2;
        v26 = &v27;
        v25 = metal_oxide_fn_64(v26, metal_oxide_ctx);
        v24 = &(*v25);
        v0 = metal_oxide_aggregate_40{1, v24};
    }
    return v0;
}

static inline metal_oxide_pointer_4 metal_oxide_fn_63(metal_oxide_pointer_4 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_pointer_4 v0;
    metal_oxide_pointer_3 v3;
    metal_oxide_pointer_3 v4;
    metal_oxide_pointer_4 v5;
    ulong v6;
    v5 = v1;
    v4 = metal_oxide_fn_21(v5, metal_oxide_ctx);
    v6 = v2;
    v3 = (v4 + v6);
    v0 = (metal_oxide_pointer_4)(v3);
    return v0;
}

static inline metal_oxide_pointer_4 metal_oxide_fn_64(metal_oxide_pointer_15 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_pointer_4 v0;
    metal_oxide_pointer_4 v2;
    metal_oxide_pointer_4 v3;
    metal_oxide_pointer_3 v4;
    metal_oxide_pointer_4 v5;
    v5 = (*v1);
    v4 = metal_oxide_fn_21(v5, metal_oxide_ctx);
    v3 = metal_oxide_fn_65(v4, metal_oxide_ctx);
    v2 = &(*v3);
    v0 = &(*v2);
    return v0;
}

static inline metal_oxide_pointer_4 metal_oxide_fn_65(metal_oxide_pointer_3 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_pointer_4 v0;
    metal_oxide_pointer_4 v2;
    metal_oxide_pointer_4 v3;
    metal_oxide_pointer_3 v4;
    v4 = v1;
    v3 = (metal_oxide_pointer_4)(v4);
    v2 = v3;
    v0 = v2;
    return v0;
}

static inline bool metal_oxide_fn_66(metal_oxide_pointer_15 v1, metal_oxide_pointer_15 v2, metal_oxide_context metal_oxide_ctx) {
    bool v0;
    metal_oxide_pointer_3 v3;
    metal_oxide_pointer_4 v4;
    metal_oxide_pointer_3 v5;
    metal_oxide_pointer_4 v6;
    v4 = (*v1);
    v3 = metal_oxide_fn_21(v4, metal_oxide_ctx);
    v6 = (*v2);
    v5 = metal_oxide_fn_21(v6, metal_oxide_ctx);
    v0 = (v3 == v5);
    return v0;
}

static inline metal_oxide_aggregate_36 metal_oxide_fn_67(metal_oxide_pointer_14 v1, float v2, metal_oxide_pointer_4 v3, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_36 v0;
    metal_oxide_aggregate_34 v4;
    metal_oxide_pointer_16 v5;
    metal_oxide_aggregate_41 v6;
    float v7;
    metal_oxide_pointer_4 v8;
    bool v9;
    ulong v10;
    metal_oxide_aggregate_34 v11;
    metal_oxide_aggregate_34 v12;
    metal_oxide_pointer_11 v13;
    metal_oxide_pointer_11 v14;
    metal_oxide_pointer_11 v15;
    v13 = (*v1).f0;
    v14 = (*v1).f0;
    (*v13) = ((*v14) - 1ul);
    v5 = &(*v1).f1;
    v7 = v2;
    v8 = v3;
    v6 = metal_oxide_aggregate_41{v7, v8};
    v4 = metal_oxide_fn_68(v5, v6.f0, v6.f1, metal_oxide_ctx);
    v15 = (*v1).f0;
    v10 = (*v15);
    v9 = (v10 == 0ul);
    if (v9) {
        v11 = v4;
        v0 = metal_oxide_aggregate_36{1, {}, v11};
    }
    else {
        v12 = v4;
        v0 = metal_oxide_fn_86(v12, metal_oxide_ctx);
    }
    return v0;
}

static inline metal_oxide_aggregate_34 metal_oxide_fn_68(metal_oxide_pointer_16 v1, float v2, metal_oxide_pointer_4 v3, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_34 v0;
    metal_oxide_aggregate_33 v4;
    metal_oxide_aggregate_41 v6;
    float v7;
    metal_oxide_pointer_4 v8;
    bool v9;
    ulong v10;
    metal_oxide_aggregate_33 v11;
    metal_oxide_aggregate_33 v12;
    metal_oxide_pointer_11 v13;
    metal_oxide_pointer_11 v14;
    metal_oxide_pointer_11 v15;
    v13 = (*v1).f0;
    v14 = (*v1).f0;
    (*v13) = ((*v14) - 1ul);
    v7 = v2;
    v8 = v3;
    v6 = metal_oxide_aggregate_41{v7, v8};
    v4 = metal_oxide_fn_69(v6.f0, v6.f1, metal_oxide_ctx);
    v15 = (*v1).f0;
    v10 = (*v15);
    v9 = (v10 == 0ul);
    if (v9) {
        v11 = v4;
        v0 = metal_oxide_aggregate_34{1, {}, v11};
    }
    else {
        v12 = v4;
        v0 = metal_oxide_fn_84(v12, metal_oxide_ctx);
    }
    return v0;
}

static inline metal_oxide_aggregate_33 metal_oxide_fn_69(float v2, metal_oxide_pointer_4 v3, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_33 v0;
    float v4;
    metal_oxide_aggregate_41 v6;
    float v7;
    metal_oxide_pointer_4 v8;
    v7 = v2;
    v8 = v3;
    v6 = metal_oxide_aggregate_41{v7, v8};
    v4 = metal_oxide_fn_70(v6.f0, v6.f1, metal_oxide_ctx);
    v0 = metal_oxide_aggregate_33{v4};
    return v0;
}

static inline float metal_oxide_fn_70(float v2, metal_oxide_pointer_4 v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_aggregate_42 v5;
    float v6;
    metal_oxide_aggregate_4 v7;
    metal_oxide_aggregate_43 v9;
    metal_oxide_pointer_4 v10;
    v6 = v2;
    v10 = v3;
    v9 = metal_oxide_aggregate_43{v10};
    v7 = metal_oxide_fn_71(v9.f0, metal_oxide_ctx);
    v5 = metal_oxide_aggregate_42{v6, v7};
    v0 = metal_oxide_fn_72(v5.f0, v5.f1, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_4 metal_oxide_fn_71(metal_oxide_pointer_4 v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_4 v0;
    v0 = metal_oxide_aggregate_4{as_type<char>(uchar(0u)), {}, as_type<float>(0x443d5000u)};
    return v0;
}

static inline float metal_oxide_fn_72(float v2, metal_oxide_aggregate_4 v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_aggregate_44 v4;
    metal_oxide_aggregate_8 v5;
    float v6;
    v5 = metal_oxide_aggregate_8{0u, 3u};
    v4 = metal_oxide_fn_73(v5, 3ul, metal_oxide_ctx);
    v6 = v3.f1;
    v0 = metal_oxide_fn_75(v4, v6, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_44 metal_oxide_fn_73(metal_oxide_aggregate_8 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_44 v0;
    metal_oxide_aggregate_8 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_fn_74(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_44 metal_oxide_fn_74(metal_oxide_aggregate_8 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_44 v0;
    metal_oxide_aggregate_8 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_aggregate_44{v3, v4};
    return v0;
}

static inline float metal_oxide_fn_75(metal_oxide_aggregate_44 v1, float v2, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_aggregate_44 v4;
    float v5;
    v4 = v1;
    v5 = v2;
    v0 = metal_oxide_fn_76(v4, v5, metal_oxide_ctx);
    return v0;
}

static inline float metal_oxide_fn_76(metal_oxide_aggregate_44 v1, float v2, metal_oxide_context metal_oxide_ctx) {
    float v0;
    float v4;
    ulong v5;
    ulong v6;
    ulong v7;
    metal_oxide_pointer_5 v8;
    metal_oxide_aggregate_14 v10;
    metal_oxide_aggregate_14 v11;
    ulong v12;
    metal_oxide_aggregate_14 v13;
    metal_oxide_aggregate_26 v16;
    ulong v21;
    uint v22;
    metal_oxide_pointer_6 v23;
    ulong v24;
    float v25;
    metal_oxide_aggregate_45 v27;
    float v28;
    uint v29;
    ulong v31;
    bool v32;
    v4 = v2;
    v6 = v1.f1;
    v8 = &v1.f0;
    v7 = metal_oxide_fn_77(v8, metal_oxide_ctx);
    v5 = metal_oxide_fn_78(v6, v7, metal_oxide_ctx);
    v12 = v5;
    v11 = metal_oxide_aggregate_14{0ul, v12};
    v10 = v11;
    v13 = v10;
    while (true) {
        v31 = v13.f0;
        v32 = (v31 < v13.f1);
        if (v32) {
        }
        else {
            break;
        }
        v16 = metal_oxide_aggregate_26{1, v31};
        v13.f0 = (v31 + 1ul);
        v21 = v16.f1;
        v23 = &v1.f0;
        v24 = v21;
        v22 = metal_oxide_fn_79(v23, v24, metal_oxide_ctx);
        v28 = v4;
        v29 = v22;
        v27 = metal_oxide_aggregate_45{v28, v29};
        v25 = metal_oxide_fn_83(v27.f0, v27.f1, metal_oxide_ctx);
        v4 = v25;
        continue;
    }
    v16 = metal_oxide_aggregate_26{0, {}};
    v0 = v4;
    return v0;
}

static inline ulong metal_oxide_fn_77(metal_oxide_pointer_5 v1, metal_oxide_context metal_oxide_ctx) {
    ulong v0;
    metal_oxide_aggregate_27 v2;
    metal_oxide_pointer_5 v3;
    v3 = &(*v1);
    v2 = metal_oxide_fn_33(v3, metal_oxide_ctx);
    v0 = v2.f0;
    return v0;
}

static inline ulong metal_oxide_fn_78(ulong v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    ulong v0;
    ulong v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = min(v3, v4);
    return v0;
}

static inline uint metal_oxide_fn_79(metal_oxide_pointer_6 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uint v3;
    metal_oxide_pointer_7 v4;
    ulong v5;
    v4 = &(*v1).f0;
    v3 = metal_oxide_fn_80(v4, metal_oxide_ctx);
    v5 = v2;
    v0 = metal_oxide_fn_81(v3, v5, metal_oxide_ctx);
    return v0;
}

static inline uint metal_oxide_fn_80(metal_oxide_pointer_7 v1, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    v0 = (*v1);
    return v0;
}

static inline uint metal_oxide_fn_81(uint v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uint v3;
    uint v4;
    ulong v5;
    v3 = v1;
    v5 = v2;
    v4 = uint(v5);
    v0 = metal_oxide_fn_82(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline uint metal_oxide_fn_82(uint v1, uint v2, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uint v8;
    uint v9;
    v8 = v1;
    v9 = v2;
    v0 = (v8 + v9);
    return v0;
}

static inline float metal_oxide_fn_83(float v2, uint v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    v0 = as_type<float>(0x433e0000u);
    return v0;
}

static inline metal_oxide_aggregate_34 metal_oxide_fn_84(metal_oxide_aggregate_33 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_34 v0;
    metal_oxide_aggregate_46 v2;
    metal_oxide_aggregate_33 v3;
    float v5;
    float v6;
    v3 = v1;
    v2 = metal_oxide_fn_85(v3, metal_oxide_ctx);
    v5 = v2.f1;
    v6 = v5;
    v0 = metal_oxide_aggregate_34{0, v6, {}};
    return v0;
}

static inline metal_oxide_aggregate_46 metal_oxide_fn_85(metal_oxide_aggregate_33 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_46 v0;
    float v2;
    v2 = v1.f0;
    v0 = metal_oxide_aggregate_46{0, v2};
    return v0;
}

static inline metal_oxide_aggregate_36 metal_oxide_fn_86(metal_oxide_aggregate_34 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_36 v0;
    metal_oxide_aggregate_48 v2;
    metal_oxide_aggregate_34 v3;
    uchar v4;
    float v5;
    float v6;
    metal_oxide_aggregate_47 v7;
    metal_oxide_aggregate_34 v8;
    metal_oxide_aggregate_47 v9;
    v3 = v1;
    v2 = metal_oxide_fn_87(v3, metal_oxide_ctx);
    v4 = v2.f0;
    if (v4 == uchar(0u)) {
        v5 = v2.f1;
        v6 = v5;
        v0 = metal_oxide_aggregate_36{0, v6, {}};
    }
    else {
        v7 = v2.f2;
        v9 = v7;
        v8 = metal_oxide_fn_88(v9, metal_oxide_ctx);
        v0 = metal_oxide_aggregate_36{1, {}, v8};
    }
    return v0;
}

static inline metal_oxide_aggregate_48 metal_oxide_fn_87(metal_oxide_aggregate_34 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_48 v0;
    uchar v2;
    float v3;
    float v4;
    metal_oxide_aggregate_33 v5;
    metal_oxide_aggregate_47 v6;
    metal_oxide_aggregate_33 v7;
    v2 = v1.f0;
    if (v2 == uchar(0u)) {
        v3 = v1.f1;
        v4 = v3;
        v0 = metal_oxide_aggregate_48{0, v4, {}};
    }
    else {
        v5 = v1.f2;
        v7 = v5;
        v6 = metal_oxide_aggregate_47{1, v7};
        v0 = metal_oxide_aggregate_48{1, {}, v6};
    }
    return v0;
}

static inline metal_oxide_aggregate_34 metal_oxide_fn_88(metal_oxide_aggregate_47 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_34 v0;
    metal_oxide_aggregate_33 v3;
    metal_oxide_aggregate_33 v4;
    v3 = v1.f1;
    v4 = v3;
    v0 = metal_oxide_aggregate_34{1, {}, v4};
    return v0;
}

static inline metal_oxide_aggregate_39 metal_oxide_fn_89(metal_oxide_aggregate_36 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_39 v0;
    uchar v2;
    float v3;
    float v4;
    metal_oxide_aggregate_34 v5;
    metal_oxide_aggregate_38 v6;
    metal_oxide_aggregate_34 v7;
    v2 = v1.f0;
    if (v2 == uchar(0u)) {
        v3 = v1.f1;
        v4 = v3;
        v0 = metal_oxide_aggregate_39{0, v4, {}};
    }
    else {
        v5 = v1.f2;
        v7 = v5;
        v6 = metal_oxide_aggregate_38{1, v7};
        v0 = metal_oxide_aggregate_39{1, {}, v6};
    }
    return v0;
}

static inline metal_oxide_aggregate_36 metal_oxide_fn_90(metal_oxide_aggregate_38 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_36 v0;
    metal_oxide_aggregate_34 v3;
    metal_oxide_aggregate_34 v4;
    v3 = v1.f1;
    v4 = v3;
    v0 = metal_oxide_aggregate_36{1, {}, v4};
    return v0;
}

static inline metal_oxide_aggregate_36 metal_oxide_fn_91(float v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_36 v0;
    float v2;
    v2 = v1;
    v0 = metal_oxide_aggregate_36{0, v2, {}};
    return v0;
}

static inline metal_oxide_aggregate_34 metal_oxide_fn_92(metal_oxide_aggregate_36 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_34 v0;
    uchar v2;
    float v3;
    float v4;
    metal_oxide_aggregate_34 v5;
    v2 = v1.f0;
    if (v2 == uchar(0u)) {
        v3 = v1.f1;
        v4 = v3;
        v0 = metal_oxide_fn_60(v4, metal_oxide_ctx);
    }
    else {
        v5 = v1.f2;
        v0 = v5;
    }
    return v0;
}

static inline metal_oxide_aggregate_33 metal_oxide_fn_93(metal_oxide_aggregate_34 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_33 v0;
    uchar v2;
    float v3;
    float v4;
    metal_oxide_aggregate_33 v5;
    v2 = v1.f0;
    if (v2 == uchar(0u)) {
        v3 = v1.f1;
        v4 = v3;
        v0 = metal_oxide_fn_57(v4, metal_oxide_ctx);
    }
    else {
        v5 = v1.f2;
        v0 = v5;
    }
    return v0;
}

static inline ushort metal_oxide_fn_94(ushort v1, ushort v2, metal_oxide_context metal_oxide_ctx) {
    ushort v0;
    ushort v3;
    ushort v4;
    v3 = v1;
    v4 = v2;
    v0 = ushort(uint(v3) * uint(v4));
    return v0;
}

static inline ushort metal_oxide_fn_95(ushort v1, ushort v2, metal_oxide_context metal_oxide_ctx) {
    ushort v0;
    ushort v3;
    ushort v4;
    v3 = v1;
    v4 = v2;
    v0 = max(v3, v4);
    return v0;
}

static inline metal_oxide_aggregate_13 metal_oxide_fn_96(float v1, metal_oxide_pointer_1 v2, ushort v3, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_13 v0;
    metal_oxide_aggregate_50 v4;
    metal_oxide_aggregate_4 v5;
    char v6;
    float v7;
    bool v8;
    ulong v9;
    ulong v10;
    ulong v11;
    metal_oxide_pointer_1 v12;
    bool v13;
    ulong v14;
    bool v15;
    float v16;
    metal_oxide_aggregate_50 v17;
    metal_oxide_aggregate_49 v18;
    int v19;
    ulong v20;
    metal_oxide_pointer_1 v21;
    ulong v23;
    metal_oxide_aggregate_8 v24;
    int v25;
    metal_oxide_aggregate_53 v26;
    metal_oxide_aggregate_52 v27;
    metal_oxide_aggregate_51 v28;
    metal_oxide_aggregate_17 v29;
    metal_oxide_aggregate_8 v30;
    int v32;
    v6 = metal_oxide_fn_97(as_type<char>(uchar(59u)), as_type<char>(uchar(1u)), metal_oxide_ctx);
    v10 = 18446744073709530536ul;
    v12 = v2;
    v11 = v12.length;
    v13 = (v11 == 0ul);
    if (v13) {
        metal_oxide_fail(metal_oxide_ctx, 12u);
        return {};
    }
    v9 = ((v10 ^ ulong(uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x))) % v11);
    v14 = v2.length;
    v15 = (v9 < v14);
    if (!(v15)) {
        metal_oxide_fail(metal_oxide_ctx, 13u);
        return {};
    }
    v8 = v2.data[v9];
    if (v8) {
        v7 = v1;
    }
    else {
        v7 = v1;
    }
    v5 = metal_oxide_aggregate_4{v6, {}, v7};
    v16 = v1;
    v18 = metal_oxide_aggregate_49{1, 18446744073709551615ul};
    v17 = metal_oxide_aggregate_50{v18};
    v4 = metal_oxide_fn_98(v5, v16, v17, metal_oxide_ctx);
    v21 = v2;
    v20 = v21.length;
    v19 = as_type<int>(uint(v20));
    v24 = metal_oxide_aggregate_8{0u, 6u};
    v23 = metal_oxide_fn_99(v24, metal_oxide_ctx);
    v30 = metal_oxide_aggregate_8{0u, 6u};
    v29 = metal_oxide_fn_101(v30, metal_oxide_ctx);
    v28 = metal_oxide_fn_103(v29, 2ul, metal_oxide_ctx);
    v27 = metal_oxide_fn_105(v28, 1ul, metal_oxide_ctx);
    v26 = metal_oxide_fn_108(v27, metal_oxide_ctx);
    v32 = v19;
    v25 = metal_oxide_fn_110(v26, v32, metal_oxide_ctx);
    v0 = metal_oxide_aggregate_13{0, v25, {}};
    return v0;
}

static inline char metal_oxide_fn_97(char v1, char v2, metal_oxide_context metal_oxide_ctx) {
    char v0;
    char v3;
    char v4;
    v3 = v1;
    v4 = v2;
    v0 = as_type<char>(uchar(uint(int(v3)) + uint(int(v4))));
    return v0;
}

static inline metal_oxide_aggregate_50 metal_oxide_fn_98(metal_oxide_aggregate_4 v1, float v2, metal_oxide_aggregate_50 v3, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_50 v0;
    metal_oxide_aggregate_49 v4;
    ulong v5;
    v5 = (1ul % 14930209768842974708ul);
    v4 = metal_oxide_aggregate_49{1, v5};
    v0 = metal_oxide_aggregate_50{v4};
    return v0;
}

static inline ulong metal_oxide_fn_99(metal_oxide_aggregate_8 v1, metal_oxide_context metal_oxide_ctx) {
    ulong v0;
    bool v2;
    metal_oxide_pointer_7 v3;
    metal_oxide_pointer_7 v4;
    metal_oxide_aggregate_26 v5;
    metal_oxide_aggregate_27 v6;
    metal_oxide_pointer_7 v7;
    metal_oxide_pointer_7 v8;
    metal_oxide_pointer_7 v9;
    metal_oxide_pointer_7 v10;
    v3 = &v1.f0;
    v4 = &v1.f1;
    v2 = metal_oxide_fn_34(v3, v4, metal_oxide_ctx);
    if (v2) {
        v8 = &v1.f0;
        v7 = &(*v8);
        v10 = &v1.f1;
        v9 = &(*v10);
        v6 = metal_oxide_fn_35(v7, v9, metal_oxide_ctx);
        v5 = v6.f1;
        v0 = metal_oxide_fn_100(v5, metal_oxide_ctx);
    }
    else {
        v0 = 0ul;
    }
    return v0;
}

static inline ulong metal_oxide_fn_100(metal_oxide_aggregate_26 v1, metal_oxide_context metal_oxide_ctx) {
    ulong v0;
    uchar v3;
    ulong v4;
    v3 = v1.f0;
    v4 = v1.f1;
    v0 = v4;
    return v0;
}

static inline metal_oxide_aggregate_17 metal_oxide_fn_101(metal_oxide_aggregate_8 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_17 v0;
    metal_oxide_aggregate_8 v2;
    v2 = v1;
    v0 = metal_oxide_fn_102(v2, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_17 metal_oxide_fn_102(metal_oxide_aggregate_8 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_17 v0;
    metal_oxide_aggregate_8 v2;
    v2 = v1;
    v0 = metal_oxide_aggregate_17{v2};
    return v0;
}

static inline metal_oxide_aggregate_51 metal_oxide_fn_103(metal_oxide_aggregate_17 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_51 v0;
    metal_oxide_aggregate_17 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_fn_104(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_51 metal_oxide_fn_104(metal_oxide_aggregate_17 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_51 v0;
    metal_oxide_aggregate_17 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_aggregate_51{v3, v4};
    return v0;
}

static inline metal_oxide_aggregate_52 metal_oxide_fn_105(metal_oxide_aggregate_51 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_52 v0;
    metal_oxide_aggregate_51 v3;
    ulong v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_fn_106(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_52 metal_oxide_fn_106(metal_oxide_aggregate_51 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_52 v0;
    bool v4;
    ulong v5;
    metal_oxide_aggregate_51 v7;
    metal_oxide_aggregate_51 v8;
    ulong v9;
    metal_oxide_aggregate_51 v10;
    ulong v11;
    ulong v12;
    v5 = v2;
    v4 = (v5 != 0ul);
    v8 = v1;
    v9 = v2;
    v7 = metal_oxide_fn_107(v8, v9, metal_oxide_ctx);
    v10 = v7;
    v12 = v2;
    v11 = (v12 - 1ul);
    v0 = metal_oxide_aggregate_52{v10, v11, true};
    return v0;
}

static inline metal_oxide_aggregate_51 metal_oxide_fn_107(metal_oxide_aggregate_51 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_51 v0;
    v0 = v1;
    return v0;
}

static inline metal_oxide_aggregate_53 metal_oxide_fn_108(metal_oxide_aggregate_52 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_53 v0;
    metal_oxide_aggregate_52 v3;
    v3 = v1;
    v0 = metal_oxide_fn_109(v3, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_53 metal_oxide_fn_109(metal_oxide_aggregate_52 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_53 v0;
    metal_oxide_aggregate_52 v3;
    v3 = v1;
    v0 = metal_oxide_aggregate_53{v3};
    return v0;
}

static inline int metal_oxide_fn_110(metal_oxide_aggregate_53 v1, int v2, metal_oxide_context metal_oxide_ctx) {
    int v0;
    metal_oxide_aggregate_52 v4;
    int v5;
    v4 = v1.f0;
    v5 = v2;
    metal_oxide_fn_111(metal_oxide_ctx);
    v0 = metal_oxide_fn_112(v4, v5, metal_oxide_ctx);
    return v0;
}

static inline void metal_oxide_fn_111(metal_oxide_context metal_oxide_ctx) {
    return;
}

static inline int metal_oxide_fn_112(metal_oxide_aggregate_52 v1, int v2, metal_oxide_context metal_oxide_ctx) {
    int v0;
    metal_oxide_aggregate_52 v4;
    int v5;
    v4 = v1;
    v5 = v2;
    v0 = metal_oxide_fn_113(v4, v5, metal_oxide_ctx);
    return v0;
}

static inline int metal_oxide_fn_113(metal_oxide_aggregate_52 v1, int v2, metal_oxide_context metal_oxide_ctx) {
    int v0;
    bool v5;
    metal_oxide_aggregate_9 v6;
    metal_oxide_pointer_17 v7;
    uchar v8;
    uint v10;
    int v11;
    metal_oxide_aggregate_54 v13;
    int v14;
    uint v15;
    metal_oxide_aggregate_56 v16;
    metal_oxide_aggregate_55 v17;
    metal_oxide_pointer_17 v18;
    metal_oxide_pointer_17 v19;
    ulong v20;
    int v21;
    v5 = v1.f2;
    if (v5) {
        v1.f2 = false;
        v7 = &v1.f0;
        v6 = metal_oxide_fn_116(v7, metal_oxide_ctx);
        v8 = v6.f0;
        if (v8 == uchar(0u)) {
            v0 = v2;
        }
        else {
            v10 = v6.f1;
            v14 = v2;
            v15 = v10;
            v13 = metal_oxide_aggregate_54{v14, v15};
            v11 = metal_oxide_fn_122(v13.f0, v13.f1, metal_oxide_ctx);
            v2 = v11;
            v19 = &v1.f0;
            v18 = &(*v19);
            v20 = v1.f1;
            v17 = metal_oxide_fn_114(v18, v20, metal_oxide_ctx);
            v16 = metal_oxide_fn_115(v17, metal_oxide_ctx);
            v21 = v2;
            v0 = metal_oxide_fn_127(v16, v21, metal_oxide_ctx);
        }
    }
    else {
        v19 = &v1.f0;
        v18 = &(*v19);
        v20 = v1.f1;
        v17 = metal_oxide_fn_114(v18, v20, metal_oxide_ctx);
        v16 = metal_oxide_fn_115(v17, metal_oxide_ctx);
        v21 = v2;
        v0 = metal_oxide_fn_127(v16, v21, metal_oxide_ctx);
    }
    return v0;
}

static inline metal_oxide_aggregate_55 metal_oxide_fn_114(metal_oxide_pointer_17 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_55 v0;
    v0 = metal_oxide_aggregate_55{v1, v2};
    return v0;
}

static inline metal_oxide_aggregate_56 metal_oxide_fn_115(metal_oxide_aggregate_55 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_56 v0;
    metal_oxide_aggregate_55 v2;
    v2 = v1;
    v0 = metal_oxide_aggregate_56{v2};
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_116(metal_oxide_pointer_17 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    bool v2;
    ulong v3;
    metal_oxide_pointer_18 v4;
    v3 = (*v1).f1;
    v2 = (v3 != 0ul);
    if (v2) {
        (*v1).f1 = ((*v1).f1 - 1ul);
        v4 = &(*v1).f0;
        v0 = metal_oxide_fn_117(v4, metal_oxide_ctx);
    }
    else {
        v0 = metal_oxide_aggregate_9{0, {}};
    }
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_117(metal_oxide_pointer_18 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    metal_oxide_pointer_6 v2;
    v2 = &(*v1).f0;
    v0 = metal_oxide_fn_118(v2, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_118(metal_oxide_pointer_6 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    metal_oxide_pointer_6 v2;
    v2 = &(*v1);
    v0 = metal_oxide_fn_119(v2, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_119(metal_oxide_pointer_6 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    bool v2;
    metal_oxide_pointer_7 v3;
    metal_oxide_pointer_7 v4;
    uint v5;
    uint v6;
    uint v7;
    v3 = &(*v1).f0;
    v4 = &(*v1).f1;
    v2 = metal_oxide_fn_34(v3, v4, metal_oxide_ctx);
    if (v2) {
        v6 = (*v1).f1;
        v5 = metal_oxide_fn_120(v6, 1ul, metal_oxide_ctx);
        (*v1).f1 = v5;
        v7 = (*v1).f1;
        v0 = metal_oxide_aggregate_9{1, v7};
    }
    else {
        v0 = metal_oxide_aggregate_9{0, {}};
    }
    return v0;
}

static inline uint metal_oxide_fn_120(uint v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uint v3;
    uint v4;
    ulong v5;
    v3 = v1;
    v5 = v2;
    v4 = uint(v5);
    v0 = metal_oxide_fn_121(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline uint metal_oxide_fn_121(uint v1, uint v2, metal_oxide_context metal_oxide_ctx) {
    uint v0;
    uint v8;
    uint v9;
    v8 = v1;
    v9 = v2;
    v0 = (v8 - v9);
    return v0;
}

static inline int metal_oxide_fn_122(int v2, uint v3, metal_oxide_context metal_oxide_ctx) {
    int v0;
    metal_oxide_aggregate_57 v5;
    int v6;
    uchar v7;
    metal_oxide_aggregate_58 v9;
    uint v10;
    v6 = v2;
    v10 = v3;
    v9 = metal_oxide_aggregate_58{v10};
    v7 = metal_oxide_fn_123(v9.f0, metal_oxide_ctx);
    v5 = metal_oxide_aggregate_57{v6, v7, {}};
    v0 = metal_oxide_fn_125(v5.f0, v5.f1, metal_oxide_ctx);
    return v0;
}

static inline uchar metal_oxide_fn_123(uint v2, metal_oxide_context metal_oxide_ctx) {
    uchar v0;
    v0 = metal_oxide_fn_124(uchar(155u), uchar(255u), metal_oxide_ctx);
    return v0;
}

static inline uchar metal_oxide_fn_124(uchar v1, uchar v2, metal_oxide_context metal_oxide_ctx) {
    uchar v0;
    uchar v3;
    uchar v4;
    v3 = v1;
    v4 = v2;
    v0 = max(v3, v4);
    return v0;
}

static inline int metal_oxide_fn_125(int v2, uchar v3, metal_oxide_context metal_oxide_ctx) {
    int v0;
    int v4;
    int v5;
    v4 = v2;
    v5 = v2;
    v0 = metal_oxide_fn_126(v4, v5, metal_oxide_ctx);
    return v0;
}

static inline int metal_oxide_fn_126(int v1, int v2, metal_oxide_context metal_oxide_ctx) {
    int v0;
    int v3;
    int v4;
    v3 = v1;
    v4 = v2;
    v0 = max(v3, v4);
    return v0;
}

static inline int metal_oxide_fn_127(metal_oxide_aggregate_56 v1, int v2, metal_oxide_context metal_oxide_ctx) {
    int v0;
    int v4;
    metal_oxide_aggregate_9 v7;
    metal_oxide_pointer_19 v8;
    uchar v9;
    uint v10;
    int v11;
    metal_oxide_aggregate_54 v13;
    int v14;
    uint v15;
    v4 = v2;
    while (true) {
        v8 = &v1;
        v7 = metal_oxide_fn_128(v8, metal_oxide_ctx);
        v9 = v7.f0;
        if (v9 == uchar(1u)) {
        }
        else {
            break;
        }
        v10 = v7.f1;
        v14 = v4;
        v15 = v10;
        v13 = metal_oxide_aggregate_54{v14, v15};
        v11 = metal_oxide_fn_122(v13.f0, v13.f1, metal_oxide_ctx);
        v4 = v11;
        continue;
    }
    v0 = v4;
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_128(metal_oxide_pointer_19 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    metal_oxide_pointer_20 v2;
    v2 = &(*v1).f0;
    v0 = metal_oxide_fn_129(v2, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_129(metal_oxide_pointer_20 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    metal_oxide_pointer_17 v2;
    ulong v3;
    metal_oxide_pointer_17 v4;
    v4 = (*v1).f0;
    v2 = &(*v4);
    v3 = (*v1).f1;
    v0 = metal_oxide_fn_130(v2, v3, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_130(metal_oxide_pointer_17 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    bool v3;
    ulong v4;
    ulong v5;
    ulong v6;
    ulong v7;
    metal_oxide_pointer_18 v8;
    ulong v9;
    bool v11;
    ulong v12;
    metal_oxide_aggregate_9 v13;
    metal_oxide_pointer_18 v14;
    ulong v15;
    ulong v16;
    v4 = (*v1).f1;
    v5 = v2;
    v3 = (v4 > v5);
    if (v3) {
        v7 = v2;
        v6 = (v7 + 1ul);
        (*v1).f1 = ((*v1).f1 - v6);
        v8 = &(*v1).f0;
        v9 = v2;
        v0 = metal_oxide_fn_131(v8, v9, metal_oxide_ctx);
    }
    else {
        v12 = (*v1).f1;
        v11 = (v12 > 0ul);
        if (v11) {
            v14 = &(*v1).f0;
            v16 = (*v1).f1;
            v15 = (v16 - 1ul);
            v13 = metal_oxide_fn_131(v14, v15, metal_oxide_ctx);
            (*v1).f1 = 0ul;
        }
        else {
        }
        v0 = metal_oxide_aggregate_9{0, {}};
    }
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_131(metal_oxide_pointer_18 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    metal_oxide_pointer_6 v3;
    ulong v4;
    v3 = &(*v1).f0;
    v4 = v2;
    v0 = metal_oxide_fn_132(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_132(metal_oxide_pointer_6 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    metal_oxide_pointer_6 v3;
    ulong v4;
    v3 = &(*v1);
    v4 = v2;
    v0 = metal_oxide_fn_133(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_133(metal_oxide_pointer_6 v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    metal_oxide_aggregate_9 v4;
    uint v5;
    ulong v6;
    uchar v7;
    uint v8;
    bool v9;
    metal_oxide_pointer_7 v10;
    metal_oxide_pointer_7 v11;
    uint v13;
    uint v14;
    uint v15;
    uint v16;
    v5 = (*v1).f1;
    v6 = v2;
    v4 = metal_oxide_fn_134(v5, v6, metal_oxide_ctx);
    v7 = v4.f0;
    if (v7 == uchar(1u)) {
        v8 = v4.f1;
        v10 = &v8;
        v11 = &(*v1).f0;
        v9 = metal_oxide_fn_136(v10, v11, metal_oxide_ctx);
        if (v9) {
            v14 = v8;
            v13 = metal_oxide_fn_120(v14, 1ul, metal_oxide_ctx);
            (*v1).f1 = v13;
            v15 = (*v1).f1;
            v0 = metal_oxide_aggregate_9{1, v15};
        }
        else {
            v16 = (*v1).f0;
            (*v1).f1 = v16;
            v0 = metal_oxide_aggregate_9{0, {}};
        }
    }
    else {
        v16 = (*v1).f0;
        (*v1).f1 = v16;
        v0 = metal_oxide_aggregate_9{0, {}};
    }
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_134(uint v1, ulong v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    metal_oxide_aggregate_30 v3;
    ulong v4;
    uchar v5;
    uint v6;
    uint v7;
    uint v8;
    v4 = v2;
    v3 = metal_oxide_fn_44(v4, metal_oxide_ctx);
    v5 = v3.f0;
    if (v5 == uchar(0u)) {
        v6 = v3.f1;
        v7 = v1;
        v8 = v6;
        v0 = metal_oxide_fn_135(v7, v8, metal_oxide_ctx);
    }
    else {
        v0 = metal_oxide_aggregate_9{0, {}};
    }
    return v0;
}

static inline metal_oxide_aggregate_9 metal_oxide_fn_135(uint v1, uint v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_9 v0;
    bool v3;
    uint v4;
    uint v5;
    uint v6;
    uint v7;
    uint v8;
    v4 = v1;
    v5 = v2;
    v3 = (v4 < v5);
    if (v3) {
        v0 = metal_oxide_aggregate_9{0, {}};
    }
    else {
        v7 = v1;
        v8 = v2;
        v6 = (v7 - v8);
        v0 = metal_oxide_aggregate_9{1, v6};
    }
    return v0;
}

static inline bool metal_oxide_fn_136(metal_oxide_pointer_7 v1, metal_oxide_pointer_7 v2, metal_oxide_context metal_oxide_ctx) {
    bool v0;
    uint v3;
    uint v4;
    v3 = (*v1);
    v4 = (*v2);
    v0 = (v3 > v4);
    return v0;
}

static inline ushort metal_oxide_fn_137(ushort v1, metal_oxide_context metal_oxide_ctx) {
    ushort v0;
    ushort v3;
    v3 = v1;
    v0 = metal_oxide_fn_138(ushort(0u), v3, metal_oxide_ctx);
    return v0;
}

static inline ushort metal_oxide_fn_138(ushort v1, ushort v2, metal_oxide_context metal_oxide_ctx) {
    ushort v0;
    ushort v3;
    ushort v4;
    v3 = v1;
    v4 = v2;
    v0 = ushort(uint(v3) - uint(v4));
    return v0;
}

static inline ushort metal_oxide_fn_139(ushort v1, ushort v2, metal_oxide_context metal_oxide_ctx) {
    ushort v0;
    ushort v3;
    ushort v4;
    v3 = v1;
    v4 = v2;
    v0 = min(v3, v4);
    return v0;
}

static inline metal_oxide_aggregate_18 metal_oxide_fn_140(metal_oxide_aggregate_17 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_18 v0;
    metal_oxide_aggregate_17 v3;
    v3 = v1;
    v0 = metal_oxide_fn_141(v3, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_18 metal_oxide_fn_141(metal_oxide_aggregate_17 v1, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_18 v0;
    metal_oxide_aggregate_17 v3;
    v3 = v1;
    v0 = metal_oxide_aggregate_18{v3};
    return v0;
}

static inline float metal_oxide_fn_142(metal_oxide_aggregate_18 v1, float v2, metal_oxide_aggregate_19 v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_aggregate_17 v4;
    float v5;
    metal_oxide_aggregate_59 v6;
    metal_oxide_aggregate_19 v8;
    v4 = v1.f0;
    v5 = v2;
    v8 = v3;
    v6 = metal_oxide_fn_143(v8, metal_oxide_ctx);
    v0 = metal_oxide_fn_144(v4, v5, v6, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_59 metal_oxide_fn_143(metal_oxide_aggregate_19 v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_59 v0;
    v0 = metal_oxide_aggregate_59{v2};
    return v0;
}

static inline float metal_oxide_fn_144(metal_oxide_aggregate_17 v1, float v2, metal_oxide_aggregate_59 v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_aggregate_8 v4;
    float v5;
    metal_oxide_aggregate_59 v6;
    v4 = v1.f0;
    v5 = v2;
    v6 = v3;
    v0 = metal_oxide_fn_145(v4, v5, v6, metal_oxide_ctx);
    return v0;
}

static inline float metal_oxide_fn_145(metal_oxide_aggregate_8 v1, float v2, metal_oxide_aggregate_59 v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    float v4;
    metal_oxide_aggregate_9 v7;
    metal_oxide_pointer_6 v8;
    uchar v9;
    uint v10;
    float v11;
    metal_oxide_pointer_21 v12;
    metal_oxide_aggregate_45 v13;
    float v14;
    uint v15;
    v4 = v2;
    while (true) {
        v8 = &v1;
        v7 = metal_oxide_fn_118(v8, metal_oxide_ctx);
        v9 = v7.f0;
        if (v9 == uchar(1u)) {
        }
        else {
            break;
        }
        v10 = v7.f1;
        v12 = &v3;
        v14 = v4;
        v15 = v10;
        v13 = metal_oxide_aggregate_45{v14, v15};
        v11 = metal_oxide_fn_146(v12, v13.f0, v13.f1, metal_oxide_ctx);
        v4 = v11;
        continue;
    }
    v0 = v4;
    return v0;
}

static inline float metal_oxide_fn_146(metal_oxide_pointer_21 v1, float v2, uint v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_pointer_22 v4;
    metal_oxide_aggregate_60 v5;
    float v6;
    float v7;
    metal_oxide_aggregate_58 v9;
    uint v10;
    v4 = &(*v1).f0;
    v6 = v2;
    v10 = v3;
    v9 = metal_oxide_aggregate_58{v10};
    v7 = metal_oxide_fn_147(v9.f0, metal_oxide_ctx);
    v5 = metal_oxide_aggregate_60{v6, v7};
    v0 = metal_oxide_fn_148(v4, v5.f0, v5.f1, metal_oxide_ctx);
    return v0;
}

static inline float metal_oxide_fn_147(uint v2, metal_oxide_context metal_oxide_ctx) {
    float v0;
    v0 = as_type<float>(0x4446a000u);
    return v0;
}

static inline float metal_oxide_fn_148(metal_oxide_pointer_22 v1, float v2, float v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    float v4;
    metal_oxide_aggregate_61 v5;
    metal_oxide_aggregate_8 v6;
    metal_oxide_aggregate_19 v7;
    metal_oxide_aggregate_62 v8;
    float v9;
    v6 = metal_oxide_aggregate_8{0u, 2u};
    v7 = metal_oxide_aggregate_19{(*v1).f0};
    v5 = metal_oxide_fn_149(v6, v7, metal_oxide_ctx);
    v8 = metal_oxide_aggregate_62{v3};
    v4 = metal_oxide_fn_151(v5, as_type<float>(0xc4716000u), v8, metal_oxide_ctx);
    v9 = float(as_type<int>(0x00000001u));
    v0 = fmin(v4, v9);
    return v0;
}

static inline metal_oxide_aggregate_61 metal_oxide_fn_149(metal_oxide_aggregate_8 v1, metal_oxide_aggregate_19 v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_61 v0;
    metal_oxide_aggregate_8 v3;
    metal_oxide_aggregate_19 v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_fn_150(v3, v4, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_61 metal_oxide_fn_150(metal_oxide_aggregate_8 v1, metal_oxide_aggregate_19 v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_61 v0;
    metal_oxide_aggregate_8 v3;
    metal_oxide_aggregate_19 v4;
    v3 = v1;
    v4 = v2;
    v0 = metal_oxide_aggregate_61{v3, v4, {}};
    return v0;
}

static inline float metal_oxide_fn_151(metal_oxide_aggregate_61 v1, float v2, metal_oxide_aggregate_62 v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_aggregate_8 v4;
    float v5;
    metal_oxide_aggregate_63 v6;
    metal_oxide_aggregate_19 v7;
    metal_oxide_aggregate_62 v8;
    v4 = v1.f0;
    v5 = v2;
    v7 = v1.f1;
    v8 = v3;
    v6 = metal_oxide_fn_152(v7, v8, metal_oxide_ctx);
    v0 = metal_oxide_fn_153(v4, v5, v6, metal_oxide_ctx);
    return v0;
}

static inline metal_oxide_aggregate_63 metal_oxide_fn_152(metal_oxide_aggregate_19 v1, metal_oxide_aggregate_62 v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_63 v0;
    v0 = metal_oxide_aggregate_63{v2, v1, {}};
    return v0;
}

static inline float metal_oxide_fn_153(metal_oxide_aggregate_8 v1, float v2, metal_oxide_aggregate_63 v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    float v4;
    metal_oxide_aggregate_9 v7;
    uint v10;
    float v11;
    metal_oxide_pointer_23 v12;
    metal_oxide_aggregate_45 v13;
    float v14;
    uint v15;
    uint v20;
    bool v21;
    v4 = v2;
    while (true) {
        v20 = v1.f0;
        v21 = (v20 < v1.f1);
        if (v21) {
        }
        else {
            break;
        }
        v7 = metal_oxide_aggregate_9{1, v20};
        v1.f0 = (v20 + 1u);
        v10 = v7.f1;
        v12 = &v3;
        v14 = v4;
        v15 = v10;
        v13 = metal_oxide_aggregate_45{v14, v15};
        v11 = metal_oxide_fn_154(v12, v13.f0, v13.f1, metal_oxide_ctx);
        v4 = v11;
        continue;
    }
    v7 = metal_oxide_aggregate_9{0, {}};
    v0 = v4;
    return v0;
}

static inline float metal_oxide_fn_154(metal_oxide_pointer_23 v1, float v2, uint v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    metal_oxide_pointer_24 v4;
    metal_oxide_aggregate_64 v5;
    float v6;
    ushort v7;
    metal_oxide_pointer_22 v8;
    metal_oxide_aggregate_58 v9;
    uint v10;
    v4 = &(*v1).f0;
    v6 = v2;
    v8 = &(*v1).f1;
    v10 = v3;
    v9 = metal_oxide_aggregate_58{v10};
    v7 = metal_oxide_fn_155(v8, v9.f0, metal_oxide_ctx);
    v5 = metal_oxide_aggregate_64{v6, v7, {}};
    v0 = metal_oxide_fn_156(v4, v5.f0, v5.f1, metal_oxide_ctx);
    return v0;
}

static inline ushort metal_oxide_fn_155(metal_oxide_pointer_22 v1, uint v2, metal_oxide_context metal_oxide_ctx) {
    ushort v0;
    v0 = (*v1).f0;
    return v0;
}

static inline float metal_oxide_fn_156(metal_oxide_pointer_24 v1, float v2, ushort v3, metal_oxide_context metal_oxide_ctx) {
    float v0;
    v0 = (*v1).f0;
    return v0;
}

static inline int metal_oxide_fn_157(int v1, metal_oxide_context metal_oxide_ctx) {
    int v0;
    int v3;
    v3 = v1;
    v0 = metal_oxide_fn_158(as_type<int>(0x00000000u), v3, metal_oxide_ctx);
    return v0;
}

static inline int metal_oxide_fn_158(int v1, int v2, metal_oxide_context metal_oxide_ctx) {
    int v0;
    int v3;
    int v4;
    v3 = v1;
    v4 = v2;
    v0 = as_type<int>(uint(uint(int(v3)) - uint(int(v4))));
    return v0;
}

static inline metal_oxide_aggregate_21 metal_oxide_fn_159(bool v1, metal_oxide_aggregate_7 v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_21 v0;
    bool v3;
    metal_oxide_aggregate_7 v4;
    bool v5;
    v5 = false;
    v5 = true;
    v3 = v1;
    if (v3) {
        v5 = false;
        v4 = v2;
        v0 = metal_oxide_aggregate_21{1, v4};
    }
    else {
        v0 = metal_oxide_aggregate_21{0, {}};
    }
    if (v5) {
    }
    else {
    }
    return v0;
}

static inline metal_oxide_aggregate_7 metal_oxide_fn_160(metal_oxide_aggregate_21 v1, metal_oxide_aggregate_7 v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_7 v0;
    uchar v3;
    metal_oxide_aggregate_7 v4;
    bool v5;
    v5 = false;
    v5 = true;
    v3 = v1.f0;
    if (v3 == uchar(0u)) {
        v5 = false;
        v0 = v2;
    }
    else {
        v4 = v1.f1;
        v0 = v4;
    }
    if (v5) {
    }
    else {
    }
    return v0;
}

static inline metal_oxide_aggregate_7 metal_oxide_fn_161(bool v1, int v2, metal_oxide_context metal_oxide_ctx) {
    metal_oxide_aggregate_7 v0;
    bool v3;
    int v4;
    bool v5;
    v5 = false;
    v5 = true;
    v3 = v1;
    if (v3) {
        v5 = false;
        v4 = v2;
        v0 = metal_oxide_aggregate_7{1, v4};
    }
    else {
        v0 = metal_oxide_aggregate_7{0, {}};
    }
    if (v5) {
    }
    else {
    }
    return v0;
}

static inline int metal_oxide_fn_162(metal_oxide_aggregate_7 v1, metal_oxide_context metal_oxide_ctx) {
    int v0;
    uchar v2;
    int v3;
    v2 = v1.f0;
    if (v2 == uchar(0u)) {
        metal_oxide_fail(metal_oxide_ctx, 14u);
        return {};
    }
    else {
        v3 = v1.f1;
        v0 = v3;
        return v0;
    }
}

static inline short metal_oxide_fn_163(short v1, metal_oxide_context metal_oxide_ctx) {
    short v0;
    ushort v2;
    ushort v3;
    short v4;
    v4 = v1;
    v3 = ushort(v4);
    v2 = metal_oxide_fn_164(v3, metal_oxide_ctx);
    v0 = as_type<short>(ushort(v2));
    return v0;
}

static inline ushort metal_oxide_fn_164(ushort v1, metal_oxide_context metal_oxide_ctx) {
    ushort v0;
    ushort v2;
    v2 = v1;
    v0 = reverse_bits(ushort((v2 ^ uint(metal_oxide_ctx.thread_idx.x >= metal_oxide_ctx.block_dim.x))));
    return v0;
}

