use super::{Dim3, DynamicLaunchConfig, Error, LaunchConfig};

const AXIS_LIMITS: Dim3 = Dim3::new(1024, 1024, 1024);

#[test]
fn grid_counts_blocks_and_block_counts_threads() {
    let config = LaunchConfig::<256>::new(Dim3::x(4));
    assert_eq!(config.total_threads().unwrap(), 1024);
    DynamicLaunchConfig::from(config)
        .validate(1024, AXIS_LIMITS)
        .unwrap();
}

#[test]
fn multidimensional_launches_count_all_axes() {
    let two_dimensional = LaunchConfig::<4, 5>::new(Dim3::xy(2, 3));
    assert_eq!(two_dimensional.total_threads().unwrap(), 120);
    DynamicLaunchConfig::from(two_dimensional)
        .validate(1024, AXIS_LIMITS)
        .unwrap();

    let three_dimensional = LaunchConfig::<3, 2, 2>::new(Dim3::new(2, 3, 4));
    assert_eq!(three_dimensional.total_threads().unwrap(), 288);
    DynamicLaunchConfig::from(three_dimensional)
        .validate(1024, AXIS_LIMITS)
        .unwrap();
}

#[test]
fn element_helper_rounds_up_to_complete_blocks() {
    for (elements, blocks, threads) in [
        (0, 0, 0),
        (1, 1, 256),
        (255, 1, 256),
        (256, 1, 256),
        (257, 2, 512),
        (1_000_003, 3907, 1_000_192),
    ] {
        let config = LaunchConfig::<256>::for_elements(elements).unwrap();
        assert_eq!(config.grid, Dim3::x(blocks));
        assert_eq!(config.total_threads().unwrap(), threads);
    }
    assert!(LaunchConfig::<0>::for_elements(1).is_err());
}

#[test]
fn block_volume_must_fit_the_pipeline_limit() {
    let config = LaunchConfig::<16, 8, 8>::new(Dim3::x(1));
    DynamicLaunchConfig::from(config)
        .validate(1024, AXIS_LIMITS)
        .unwrap();
    assert!(matches!(
        DynamicLaunchConfig::from(config).validate(512, AXIS_LIMITS),
        Err(Error::InvalidLaunch(_))
    ));
    assert!(
        DynamicLaunchConfig::from(config)
            .validate(0, AXIS_LIMITS)
            .is_err()
    );
}

#[test]
fn each_block_axis_must_fit_the_device_limit() {
    let limits = Dim3::new(8, 4, 2);
    assert!(
        DynamicLaunchConfig::from(LaunchConfig::<9>::new(Dim3::x(1)))
            .validate(1024, limits)
            .is_err()
    );
    assert!(
        DynamicLaunchConfig::from(LaunchConfig::<1, 5>::new(Dim3::x(1)))
            .validate(1024, limits)
            .is_err()
    );
    assert!(
        DynamicLaunchConfig::from(LaunchConfig::<1, 1, 3>::new(Dim3::x(1)))
            .validate(1024, limits)
            .is_err()
    );
    DynamicLaunchConfig::from(LaunchConfig::<8, 4, 2>::new(Dim3::x(1)))
        .validate(1024, limits)
        .unwrap();
}

#[test]
fn empty_grid_is_a_no_op_but_block_dimensions_must_be_nonzero() {
    for grid in [Dim3::new(0, 3, 4), Dim3::new(2, 0, 4), Dim3::new(2, 3, 0)] {
        let config = LaunchConfig::<2, 3, 2>::new(grid);
        assert!(config.is_empty());
        assert_eq!(config.total_threads().unwrap(), 0);
        DynamicLaunchConfig::from(config)
            .validate(1024, AXIS_LIMITS)
            .unwrap();
    }
    assert!(
        DynamicLaunchConfig::from(LaunchConfig::<0>::new(Dim3::x(0)))
            .validate(1024, AXIS_LIMITS)
            .is_err()
    );
    assert!(
        DynamicLaunchConfig::from(LaunchConfig::<1, 0>::new(Dim3::x(0)))
            .validate(1024, AXIS_LIMITS)
            .is_err()
    );
    assert!(
        DynamicLaunchConfig::from(LaunchConfig::<1, 1, 0>::new(Dim3::x(0)))
            .validate(1024, AXIS_LIMITS)
            .is_err()
    );
}

#[test]
fn global_coordinate_overflow_is_rejected_on_each_axis() {
    for grid in [
        Dim3::new(u32::MAX, 1, 1),
        Dim3::new(1, u32::MAX, 1),
        Dim3::new(1, 1, u32::MAX),
    ] {
        let config = LaunchConfig::<2, 2, 2>::new(grid);
        assert!(
            DynamicLaunchConfig::from(config)
                .validate(1024, AXIS_LIMITS)
                .is_err()
        );
    }
    assert!(LaunchConfig::<256>::for_elements(u32::MAX).is_err());
    assert_eq!(
        LaunchConfig::<1>::for_elements(u32::MAX)
            .unwrap()
            .total_threads()
            .unwrap(),
        u64::from(u32::MAX)
    );
}

#[test]
fn total_thread_count_overflow_is_rejected() {
    let config = LaunchConfig::<1>::new(Dim3::new(u32::MAX, u32::MAX, u32::MAX));
    assert!(config.total_threads().is_err());
    assert!(
        DynamicLaunchConfig::from(config)
            .validate(1024, AXIS_LIMITS)
            .is_err()
    );
}

#[test]
fn dynamic_element_helper_accepts_runtime_block_sizes() {
    let config = DynamicLaunchConfig::for_elements(257, 32).unwrap();
    assert_eq!(config.grid, Dim3::x(9));
    assert_eq!(config.total_threads().unwrap(), 288);
    config.validate(32, AXIS_LIMITS).unwrap();
    assert!(config.validate(31, AXIS_LIMITS).is_err());
    assert!(DynamicLaunchConfig::for_elements(1, 0).is_err());
    assert!(DynamicLaunchConfig::for_elements(u32::MAX, 256).is_err());
}

#[test]
fn dynamic_block_axes_use_the_same_device_limits() {
    let limits = Dim3::new(8, 4, 2);
    for block in [Dim3::new(9, 1, 1), Dim3::new(1, 5, 1), Dim3::new(1, 1, 3)] {
        assert!(
            DynamicLaunchConfig::new(Dim3::x(1), block)
                .validate(1024, limits)
                .is_err()
        );
    }
    let config = DynamicLaunchConfig::new(Dim3::xy(2, 3), Dim3::xy(4, 5));
    assert_eq!(config.total_threads().unwrap(), 120);
    config.validate(20, AXIS_LIMITS).unwrap();
}
