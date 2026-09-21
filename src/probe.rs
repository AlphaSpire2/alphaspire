//! The search's own stopwatch: where a decision's time actually goes.
//!
//! Off unless the `ALPHASPIRE_PROBE` environment variable is set to anything
//! other than `0`, and free when off — every call site is a closure the
//! optimizer inlines around a single relaxed atomic load. Armed, it charges
//! wall time to a fixed set of phases and counts the structural work each
//! decision does, so a cost claim about the search is a measurement rather
//! than a reading of the code.
//!
//! Counters are process-global atomics rather than thread locals precisely
//! because a batch fans out: a parallel batch's report is the batch's, and
//! the relaxed adds are far below the cost of anything they time. A
//! measurement run still wants `--jobs 1` if per-decision *latency* is the
//! question, since the histogram mixes workers.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static ARMED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("ALPHASPIRE_PROBE").is_ok_and(|value| !value.is_empty() && value != "0")
});

/// Whether the probe is armed for this process.
#[must_use]
pub fn armed() -> bool {
    *ARMED
}

/// The phases a search decision's time is charged to. Together they cover
/// every call a decision makes that is not pure bookkeeping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    /// Erasing the authoritative state to a belief (`*BeliefState::from_simulator`).
    Erase,
    /// Dealing one possible world out of the belief (`Determinizer::sample`).
    Sample,
    /// Keying a decision point (`observation_key`/`state_key`).
    NodeKey,
    /// Pricing a decision point on first sight: its canonical action classes,
    /// and a net's priors where one guides.
    Expand,
    /// Asking whether a sampled world has walked past the horizon.
    Horizon,
    /// Stepping the simulator inside the tree.
    TreeStep,
    /// The rollout policy choosing an action past the tree.
    RolloutPick,
    /// Stepping the simulator past the tree.
    RolloutStep,
    /// A checkpoint's value head at a leaf.
    LeafNet,
    /// The objective scoring a state where a walk stopped.
    Peek,
}

/// How many phases there are.
pub const PHASES: usize = 10;

/// The phase names, in the order [`Phase`] discriminates them.
pub const PHASE_NAMES: [&str; PHASES] = [
    "erase",
    "sample",
    "node_key",
    "expand",
    "horizon",
    "tree_step",
    "rollout_pick",
    "rollout_step",
    "leaf_net",
    "peek",
];

impl Phase {
    const fn index(self) -> usize {
        match self {
            Self::Erase => 0,
            Self::Sample => 1,
            Self::NodeKey => 2,
            Self::Expand => 3,
            Self::Horizon => 4,
            Self::TreeStep => 5,
            Self::RolloutPick => 6,
            Self::RolloutStep => 7,
            Self::LeafNet => 8,
            Self::Peek => 9,
        }
    }
}

/// The structural counts a decision runs up, beside the time they cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tally {
    /// Rollouts started.
    Iterations,
    /// Tree-descent steps taken (one per node the descent stood on).
    DescentSteps,
    /// Rollout steps taken past the tree.
    RolloutSteps,
    /// Decision points priced for the first time.
    NodesPriced,
    /// Decisions the search downgraded to the rollout policy under a budget.
    Downgrades,
    /// Rollouts that ended on the depth cap, still mid-line.
    RolloutCapped,
    /// Rollouts that ended on a dead player (or a won run).
    RolloutTerminal,
    /// Rollouts that ended on the mode's own horizon.
    RolloutHorizon,
    /// Rollouts that ended where the world offered nothing, or refused.
    RolloutDeadEnd,
    /// Tree descents that stopped because they had walked back onto a
    /// decision point already on their own path.
    DescentCycle,
    /// Decisions that stopped starting simulations early because the run's
    /// budget ceiling came down mid-decision.
    DecisionCapped,
    /// Live observation token slots summed over every pricing, against the
    /// padded ceiling the plan runs at.
    LiveTokens,
    /// Actions summed over every pricing, against the padded action axis.
    PricedActions,
    /// Net leaf values raised to the least the objective can pay for the
    /// state: a head reading below a defeat, which is nothing worse than a
    /// rounding of zero.
    LeafClippedLow,
    /// Net leaf values lowered to the most the objective can pay for the
    /// state: a head extrapolating past a payoff no line could reach.
    LeafClippedHigh,
    /// Counterfactual branches played to the fight's end and recorded.
    BranchesRecorded,
    /// Counterfactual branches dropped unrecorded at their step cap.
    BranchesDropped,
}

/// How many tallies there are.
pub const TALLIES: usize = 17;

/// The tally names, in the order [`Tally`] discriminates them.
pub const TALLY_NAMES: [&str; TALLIES] = [
    "iterations",
    "descent_steps",
    "rollout_steps",
    "nodes_priced",
    "downgrades",
    "rollout_capped",
    "rollout_terminal",
    "rollout_horizon",
    "rollout_dead_end",
    "descent_cycle",
    "decision_capped",
    "live_tokens",
    "priced_actions",
    "leaf_clipped_low",
    "leaf_clipped_high",
    "branches_recorded",
    "branches_dropped",
];

impl Tally {
    const fn index(self) -> usize {
        match self {
            Self::Iterations => 0,
            Self::DescentSteps => 1,
            Self::RolloutSteps => 2,
            Self::NodesPriced => 3,
            Self::Downgrades => 4,
            Self::RolloutCapped => 5,
            Self::RolloutTerminal => 6,
            Self::RolloutHorizon => 7,
            Self::RolloutDeadEnd => 8,
            Self::DescentCycle => 9,
            Self::DecisionCapped => 10,
            Self::LiveTokens => 11,
            Self::PricedActions => 12,
            Self::LeafClippedLow => 13,
            Self::LeafClippedHigh => 14,
            Self::BranchesRecorded => 15,
            Self::BranchesDropped => 16,
        }
    }
}

/// Which search a decision belonged to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// An out-of-combat decision searched over act-scoped worlds.
    Act,
    /// An in-combat decision searched over fight-scoped worlds.
    Combat,
}

/// How many decision kinds there are.
pub const KINDS: usize = 2;

/// The kind names, in the order [`Kind`] discriminates them.
pub const KIND_NAMES: [&str; KINDS] = ["act", "combat"];

impl Kind {
    const fn index(self) -> usize {
        match self {
            Self::Act => 0,
            Self::Combat => 1,
        }
    }
}

/// Histogram buckets: decision latency by power of two microseconds, from
/// under a microsecond to over nine minutes.
pub const BUCKETS: usize = 24;

/// Where phase time is charged when no decision is standing.
const LOOSE: usize = KINDS;

/// The arms time is charged to: one per decision kind, plus whatever runs
/// outside a decision.
const ARMS: usize = KINDS + 1;

const ARM_NAMES: [&str; ARMS] = ["act", "combat", "loose"];

thread_local! {
    /// Which arm the thread is inside. Set by [`Decision`], restored when it
    /// drops; the arms never nest (an act decision that lands in a fight
    /// hands the whole decision to the combat search, it does not wrap one).
    static ARM: std::cell::Cell<usize> = const { std::cell::Cell::new(LOOSE) };
}

fn arm() -> usize {
    ARM.with(std::cell::Cell::get)
}

static NANOS: [[AtomicU64; PHASES]; ARMS] = [const { [const { AtomicU64::new(0) }; PHASES] }; ARMS];
static CALLS: [[AtomicU64; PHASES]; ARMS] = [const { [const { AtomicU64::new(0) }; PHASES] }; ARMS];
static TALLIED: [[AtomicU64; TALLIES]; ARMS] =
    [const { [const { AtomicU64::new(0) }; TALLIES] }; ARMS];
static DECISIONS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static DECISION_NANOS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static DECISION_MAX: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static HISTOGRAM: [[AtomicU64; BUCKETS]; KINDS] =
    [const { [const { AtomicU64::new(0) }; BUCKETS] }; KINDS];

fn nanos_of(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Runs `work`, charging its wall time to `phase` when the probe is armed.
pub fn timed<T>(phase: Phase, work: impl FnOnce() -> T) -> T {
    if !armed() {
        return work();
    }
    let start = Instant::now();
    let value = work();
    let nanos = nanos_of(start);
    let arm = arm();
    NANOS[arm][phase.index()].fetch_add(nanos, Ordering::Relaxed);
    CALLS[arm][phase.index()].fetch_add(1, Ordering::Relaxed);
    value
}

/// Counts `amount` of structural work.
pub fn tally(which: Tally, amount: u64) {
    if !armed() {
        return;
    }
    TALLIED[arm()][which.index()].fetch_add(amount, Ordering::Relaxed);
}

/// The stopwatch on one whole decision, settled when it is dropped. Held by
/// value at the call site so an early return still settles it.
pub struct Decision {
    kind: Kind,
    start: Option<Instant>,
    restore: usize,
}

/// Starts timing one decision of `kind`, and charges everything the thread
/// does until it is dropped to that arm.
#[must_use]
pub fn decision(kind: Kind) -> Decision {
    let restore = arm();
    if armed() {
        ARM.with(|cell| cell.set(kind.index()));
    }
    Decision {
        kind,
        start: armed().then(Instant::now),
        restore,
    }
}

impl Drop for Decision {
    fn drop(&mut self) {
        ARM.with(|cell| cell.set(self.restore));
        let Some(start) = self.start else {
            return;
        };
        let nanos = nanos_of(start);
        let index = self.kind.index();
        DECISIONS[index].fetch_add(1, Ordering::Relaxed);
        DECISION_NANOS[index].fetch_add(nanos, Ordering::Relaxed);
        DECISION_MAX[index].fetch_max(nanos, Ordering::Relaxed);
        let micros = nanos / 1_000;
        let bucket =
            (BUCKETS - 1).min(usize::try_from(micros.checked_ilog2().unwrap_or(0)).unwrap_or(0));
        HISTOGRAM[index][bucket].fetch_add(1, Ordering::Relaxed);
    }
}

/// What the probe measured, as a report to print, or nothing when it was
/// never armed.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "wall-clock nanosecond totals are far below f64 precision"
)]
pub fn report() -> Option<String> {
    use std::fmt::Write as _;
    if !armed() {
        return None;
    }
    let mut lines = String::from("probe: where the search spent its time\n");
    let total: u64 = (0..ARMS)
        .flat_map(|arm| (0..PHASES).map(move |phase| NANOS[arm][phase].load(Ordering::Relaxed)))
        .sum();
    let _ = writeln!(
        lines,
        "  {:<8} {:<14} {:>10} {:>12} {:>10} {:>8}",
        "arm", "phase", "calls", "total ms", "per call", "share"
    );
    for (arm, name) in ARM_NAMES.iter().enumerate() {
        for (phase, phase_name) in PHASE_NAMES.iter().enumerate() {
            let calls = CALLS[arm][phase].load(Ordering::Relaxed);
            let nanos = NANOS[arm][phase].load(Ordering::Relaxed);
            if calls == 0 {
                continue;
            }
            let _ = writeln!(
                lines,
                "  {name:<8} {phase_name:<14} {calls:>10} {:>12.1} {:>9.1}\u{b5}s {:>7.1}%",
                nanos as f64 / 1e6,
                nanos as f64 / calls as f64 / 1e3,
                100.0 * nanos as f64 / total.max(1) as f64,
            );
        }
    }
    let _ = writeln!(lines, "  measured total {:.1} ms", total as f64 / 1e6);
    for (kind, name) in KIND_NAMES.iter().enumerate() {
        let count = DECISIONS[kind].load(Ordering::Relaxed);
        if count == 0 {
            continue;
        }
        let nanos = DECISION_NANOS[kind].load(Ordering::Relaxed);
        let _ = writeln!(
            lines,
            "  {name} decisions: {count} | total {:.1} s | mean {:.1} ms | max {:.1} ms",
            nanos as f64 / 1e9,
            nanos as f64 / count as f64 / 1e6,
            DECISION_MAX[kind].load(Ordering::Relaxed) as f64 / 1e6,
        );
        let mut row = String::new();
        for (bucket, slot) in HISTOGRAM[kind].iter().enumerate() {
            let hits = slot.load(Ordering::Relaxed);
            if hits > 0 {
                let _ = write!(row, " 2^{bucket}\u{b5}s:{hits}");
            }
        }
        let _ = writeln!(lines, "    latency{row}");
    }
    for (arm, name) in ARM_NAMES.iter().enumerate() {
        let mut row = String::new();
        for (which, tally_name) in TALLY_NAMES.iter().enumerate() {
            let count = TALLIED[arm][which].load(Ordering::Relaxed);
            if count > 0 {
                let _ = write!(row, " {tally_name}={count}");
            }
        }
        if !row.is_empty() {
            let _ = writeln!(lines, "  {name} work:{row}");
        }
    }
    Some(lines)
}

#[cfg(test)]
mod tests {
    use super::{Kind, Phase, Tally, armed, decision, report, tally, timed};

    /// Disarmed — the ordinary case — the probe is a pass-through that
    /// reports nothing: no test run pays for it, and no batch prints it.
    #[test]
    fn disarmed_probe_passes_work_through_and_reports_nothing() {
        if armed() {
            // The suite is running under an armed probe; the claim under
            // test is about the disarmed default, so there is nothing here
            // to check.
            return;
        }
        let answer = timed(Phase::Peek, || 2 + 2);
        assert_eq!(answer, 4);
        tally(Tally::Iterations, 7);
        drop(decision(Kind::Act));
        assert!(report().is_none());
    }
}
