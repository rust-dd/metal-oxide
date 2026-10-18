use super::run;
use sort::{Step::*, padded_len, plan};

#[test]
#[ignore = "requires a Metal device"]
fn generated_sort_matches_cpu_and_msl() -> metal_oxide::Result<()> {
    run::verify()
}

#[test]
fn keys_pad_to_a_power_of_two_of_at_least_one_tile() {
    let lengths = [0, 1, 2, 255, 512, 513, 4096, 100_003, 1 << 31].map(padded_len);
    assert_eq!(
        lengths,
        [0, 512, 512, 512, 512, 1024, 4096, 131_072, 1 << 31]
    );
}

#[test]
fn each_stage_above_a_tile_merges_globally_down_to_a_tile_then_locally() {
    assert!(plan(0).is_empty());
    assert_eq!(plan(512), [SortLocal]);
    let stages = [
        SortLocal,
        MergeGlobal { j: 512, k: 1024 },
        MergeLocal { k: 1024 },
        MergeGlobal { j: 1024, k: 2048 },
        MergeGlobal { j: 512, k: 2048 },
        MergeLocal { k: 2048 },
    ];
    assert_eq!(plan(2048), stages);
    assert_eq!(plan(1 << 20).len(), 1 + 66 + 11);
}
