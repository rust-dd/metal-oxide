use metal_oxide::{Dispatch1d, Error};

#[test]
fn default_group_uses_pipeline_execution_width() {
    assert_eq!(Dispatch1d::new(257).group_width(32, 1024).unwrap(), 32);
}

#[test]
fn explicit_group_must_fit_pipeline_limit() {
    let dispatch = Dispatch1d::new(257).with_group_width(256);
    assert_eq!(dispatch.group_width(32, 256).unwrap(), 256);
    assert!(matches!(
        dispatch.group_width(32, 128),
        Err(Error::InvalidDispatch(_))
    ));
}

#[test]
fn zero_and_invalid_pipeline_widths_are_rejected() {
    assert!(
        Dispatch1d::new(1)
            .with_group_width(0)
            .group_width(32, 1024)
            .is_err()
    );
    assert!(Dispatch1d::new(1).group_width(0, 1024).is_err());
    assert!(Dispatch1d::new(1).group_width(32, 0).is_err());
    assert!(Dispatch1d::new(1).group_width(64, 32).is_err());
}

#[test]
fn empty_grid_preserves_zero_threads() {
    assert_eq!(Dispatch1d::new(0).threads(), 0);
}
