use std::{
    sync::Arc,
    task::{Context, Poll, Waker},
};

use super::super::sync::{Mutex, lock};

pub(in crate::metal) const LIMIT: usize = 64;

/// Reservations include command buffers being encoded, before their queue running count starts.
#[derive(Default)]
pub(in crate::metal) struct Capacity(Mutex<State>);

#[derive(Default)]
struct State {
    used: usize,
    wakers: Vec<Waker>,
}

pub(in crate::metal) struct Permit(Arc<Capacity>);

impl Capacity {
    pub(in crate::metal) fn poll(self: &Arc<Self>, cx: &mut Context<'_>) -> Poll<Permit> {
        let mut state = lock(&self.0);
        if state.used == LIMIT {
            if !state.wakers.iter().any(|waker| waker.will_wake(cx.waker())) {
                state.wakers.push(cx.waker().clone());
            }
            return Poll::Pending;
        }
        state.used += 1;
        Poll::Ready(Permit(Arc::clone(self)))
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        let wakers = {
            let mut state = lock(&self.0.0);
            state.used -= 1;
            std::mem::take(&mut state.wakers)
        };
        for waker in wakers {
            // A user waker must not unwind through a native callback or permit destructor.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests;
