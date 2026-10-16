use metal_oxide_artifact::{AtomicOp::*, AtomicOrdering::*, Scalar};

#[test]
fn capabilities_follow_the_operation_and_address_space() {
    for op in [
        Load,
        Store,
        Exchange,
        CompareExchangeWeak,
        Add,
        Sub,
        Min,
        Max,
        And,
        Or,
        Xor,
    ] {
        for scalar in [Scalar::I32, Scalar::U32] {
            for shared in [false, true] {
                assert_eq!(op.capability(scalar, shared).unwrap().name, "int32_atomics");
            }
        }
        let float = matches!(
            op,
            Load | Store | Exchange | CompareExchangeWeak | Add | Sub
        );
        assert_eq!(op.capability(Scalar::F32, false).is_some(), float);
        assert!(op.capability(Scalar::F32, true).is_none());
        assert!(op.capability(Scalar::U64, false).is_none());
    }
    for op in [MinStore, MaxStore] {
        let capability = op.capability(Scalar::U64, false).unwrap();
        assert_eq!(capability.apple_family, 8);
        assert!(op.capability(Scalar::U64, true).is_none());
        assert!(op.capability(Scalar::U32, false).is_none());
    }
}

#[test]
fn orderings_are_validated_before_lowering() {
    for order in [Relaxed, Acquire, Release, AcqRel] {
        assert!(Add.validate_order(order, Relaxed).is_ok());
        assert_eq!(
            Load.validate_order(order, Relaxed).is_ok(),
            matches!(order, Relaxed | Acquire)
        );
        assert_eq!(
            Store.validate_order(order, Relaxed).is_ok(),
            matches!(order, Relaxed | Release)
        );
        assert_eq!(
            MinStore.validate_order(order, Relaxed).is_ok(),
            order == Relaxed
        );
    }
    for success in [Relaxed, Acquire, Release, AcqRel, SeqCst] {
        for failure in [Relaxed, Acquire, Release, AcqRel, SeqCst] {
            assert_eq!(
                CompareExchangeWeak.validate_order(success, failure).is_ok(),
                success != SeqCst
                    && (failure == Relaxed
                        || (failure == Acquire && matches!(success, Acquire | AcqRel)))
            );
        }
    }
    assert!(Add.validate_order(SeqCst, Relaxed).is_err());
}
