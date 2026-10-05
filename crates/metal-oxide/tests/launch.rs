use metal_oxide::{Dim3, Error, LaunchConfig};

const AXIS_LIMITS: Dim3 = Dim3::new(1024, 1024, 1024);

#[test]
fn grid_counts_blocks_and_block_counts_threads() {
    let config = LaunchConfig {
        grid: Dim3::new(4, 1, 1),
        block: Dim3::new(256, 1, 1),
    };
    assert_eq!(config.total_threads().unwrap(), 1024);
    config.validate(1024, AXIS_LIMITS).unwrap();
}

#[test]
fn multidimensional_launches_count_all_axes() {
    let two_dimensional = LaunchConfig::new(Dim3::xy(2, 3), Dim3::xy(4, 5));
    assert_eq!(two_dimensional.total_threads().unwrap(), 120);
    two_dimensional.validate(1024, AXIS_LIMITS).unwrap();

    let three_dimensional = LaunchConfig::new(Dim3::new(2, 3, 4), Dim3::new(3, 2, 2));
    assert_eq!(three_dimensional.total_threads().unwrap(), 288);
    three_dimensional.validate(1024, AXIS_LIMITS).unwrap();
}

#[test]
fn element_helper_rounds_up_to_complete_blocks() {
    for (elements, blocks) in [
        (0, 0),
        (1, 1),
        (255, 1),
        (256, 1),
        (257, 2),
        (1_000_003, 3907),
    ] {
        let config = LaunchConfig::for_elements(elements, 256).unwrap();
        assert_eq!(config.grid, Dim3::x(blocks));
        assert_eq!(config.block, Dim3::x(256));
        assert_eq!(config.total_threads().unwrap(), u64::from(blocks) * 256);
    }
    assert!(LaunchConfig::for_elements(1, 0).is_err());
}

#[test]
fn block_volume_must_fit_the_pipeline_limit() {
    let config = LaunchConfig::new(Dim3::x(1), Dim3::new(16, 8, 8));
    config.validate(1024, AXIS_LIMITS).unwrap();
    assert!(matches!(
        config.validate(512, AXIS_LIMITS),
        Err(Error::InvalidLaunch(_))
    ));
    assert!(config.validate(0, AXIS_LIMITS).is_err());
}

#[test]
fn each_block_axis_must_fit_the_device_limit() {
    let limits = Dim3::new(8, 4, 2);
    for block in [Dim3::new(9, 1, 1), Dim3::new(1, 5, 1), Dim3::new(1, 1, 3)] {
        assert!(
            LaunchConfig::new(Dim3::x(1), block)
                .validate(1024, limits)
                .is_err()
        );
    }
    LaunchConfig::new(Dim3::x(1), limits)
        .validate(1024, limits)
        .unwrap();
}

#[test]
fn empty_grid_is_a_no_op_but_block_dimensions_must_be_nonzero() {
    for grid in [Dim3::new(0, 3, 4), Dim3::new(2, 0, 4), Dim3::new(2, 3, 0)] {
        let config = LaunchConfig::new(grid, Dim3::new(2, 3, 2));
        assert!(config.is_empty());
        assert_eq!(config.total_threads().unwrap(), 0);
        config.validate(1024, AXIS_LIMITS).unwrap();
    }
    for block in [Dim3::new(0, 1, 1), Dim3::new(1, 0, 1), Dim3::new(1, 1, 0)] {
        assert!(
            LaunchConfig::new(Dim3::x(0), block)
                .validate(1024, AXIS_LIMITS)
                .is_err()
        );
    }
}

#[test]
fn global_coordinate_overflow_is_rejected_on_each_axis() {
    for grid in [
        Dim3::new(u32::MAX, 1, 1),
        Dim3::new(1, u32::MAX, 1),
        Dim3::new(1, 1, u32::MAX),
    ] {
        let config = LaunchConfig::new(grid, Dim3::new(2, 2, 2));
        assert!(config.validate(1024, AXIS_LIMITS).is_err());
    }
    assert!(LaunchConfig::for_elements(u32::MAX, 256).is_err());
    assert_eq!(
        LaunchConfig::for_elements(u32::MAX, 1)
            .unwrap()
            .total_threads()
            .unwrap(),
        u64::from(u32::MAX)
    );
}

#[test]
fn total_thread_count_overflow_is_rejected() {
    let config = LaunchConfig::new(Dim3::new(u32::MAX, u32::MAX, u32::MAX), Dim3::x(1));
    assert!(config.total_threads().is_err());
    assert!(config.validate(1024, AXIS_LIMITS).is_err());
}
