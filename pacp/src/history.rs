//! Query History Tracker — per-principal release accounting.
//!
//! PAC bounds compose over repeated releases, so privacy must be charged
//! cumulatively, not per-query. This tracks, per principal, how many sensitive-
//! channel releases have been made (the budget meter) and a short window of
//! recent input digests (for the richer composition M3 will add). M1 uses it
//! for the fail-closed release cap.

use std::collections::HashMap;

/// Returned when a principal has spent its release budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetExhausted;

/// Per-principal accounting state.
#[derive(Debug, Clone, Default)]
pub struct PrincipalState {
    /// Sensitive-channel releases charged so far.
    pub releases_spent: u64,
    /// Recent input digests (most-recent-last), capped to the window.
    pub recent_inputs: Vec<u64>,
}

/// Tracks release budgets across principals.
pub struct QueryHistory {
    window: usize,
    principals: HashMap<u64, PrincipalState>,
}

impl QueryHistory {
    pub fn new(window: usize) -> Self {
        QueryHistory { window: window.max(1), principals: HashMap::new() }
    }

    /// Charge one release against `principal`. Fails closed once the cap is
    /// reached (the caller then suppresses the sensitive channel rather than
    /// releasing it un-noised).
    pub fn charge(
        &mut self,
        principal: u64,
        max_releases: u64,
        input_digest: u64,
    ) -> Result<(), BudgetExhausted> {
        let st = self.principals.entry(principal).or_default();
        if st.releases_spent >= max_releases {
            return Err(BudgetExhausted);
        }
        st.releases_spent += 1;
        st.recent_inputs.push(input_digest);
        if st.recent_inputs.len() > self.window {
            let overflow = st.recent_inputs.len() - self.window;
            st.recent_inputs.drain(0..overflow);
        }
        Ok(())
    }

    pub fn spent(&self, principal: u64) -> u64 {
        self.principals.get(&principal).map(|s| s.releases_spent).unwrap_or(0)
    }

    /// Re-seed a principal's release count. Required when a node resumes a
    /// session mid-stream (snapshot restore / replay starting after epoch N):
    /// the budget meter is not carried in the NDAL log, so it must be restored
    /// to the snapshot's value or the suppress decision diverges from the
    /// recorded run. (Oracle-draw alignment is unaffected — the layer draws once
    /// per release regardless of budget — but the perturb-vs-suppress choice
    /// depends on this count.)
    pub fn restore(&mut self, principal: u64, releases_spent: u64) {
        self.principals.entry(principal).or_default().releases_spent = releases_spent;
    }

    pub fn recent_inputs(&self, principal: u64) -> &[u64] {
        self.principals.get(&principal).map(|s| s.recent_inputs.as_slice()).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_until_cap_then_fails_closed() {
        let mut h = QueryHistory::new(8);
        assert!(h.charge(1, 2, 0xaa).is_ok());
        assert!(h.charge(1, 2, 0xbb).is_ok());
        assert_eq!(h.charge(1, 2, 0xcc), Err(BudgetExhausted));
        assert_eq!(h.spent(1), 2);
    }

    #[test]
    fn principals_are_independent() {
        let mut h = QueryHistory::new(8);
        h.charge(1, 1, 0x1).unwrap();
        // principal 2 has its own budget
        assert!(h.charge(2, 1, 0x2).is_ok());
        assert_eq!(h.charge(1, 1, 0x3), Err(BudgetExhausted));
    }

    #[test]
    fn input_window_is_bounded() {
        let mut h = QueryHistory::new(3);
        for i in 0..10 {
            h.charge(1, u64::MAX, i).unwrap();
        }
        assert_eq!(h.recent_inputs(1), &[7, 8, 9]);
    }
}
