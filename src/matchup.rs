//! A match: two search configurations over identical games, and a paired
//! verdict.
//!
//! The name is the chess testing ecosystem's — fishtest runs SPRT matches
//! between two engine versions over shared openings — and the design is the
//! paired one that makes small samples honest: both arms play the *same
//! games*, seed for seed (a run being a function of its index, see
//! [`crate::selfplay::run_seeds`]), so a pair's difference is attributable
//! to the arms and to nothing about which games were drawn. The verdict is a
//! one-sided sign test over the non-tied pairs: expert iteration's gate,
//! where only a measured win promotes a checkpoint.

use std::sync::Arc;

use crate::net::PolicyValueNet;
use crate::objective::{CombatStrength, Objective};
use crate::policy::RolloutPolicy;
use crate::search::{BeliefSearch, Degradations, SearchConfig, Selection};
use crate::selfplay::{Batch, OutcomeTally, play_batch};

/// One arm of a match: belief search under these priors. `None` is the
/// uniform-prior arm.
pub type Arm = Option<Arc<PolicyValueNet>>;

/// How a match is played: the shared search configuration and the budget.
/// Everything here applies to both arms identically — that is the point.
pub struct MatchConfig<'a> {
    pub batch: Batch<'a>,
    pub search: SearchConfig,
    pub selection: &'a (dyn Fn() -> Box<dyn Selection> + Sync),
    /// The rollout each search leans on — and, past `fight_over`, the whole
    /// out-of-combat policy of both arms. One builder for both arms, like
    /// everything else here.
    pub rollout: &'a (dyn Fn() -> Box<dyn RolloutPolicy> + Sync),
}

/// One game both arms finished: how deep each got.
#[derive(Clone, Debug)]
pub struct Pair {
    pub seed: String,
    pub baseline: u32,
    pub candidate: u32,
}

/// A finished match.
#[derive(Clone, Debug, Default)]
pub struct MatchReport {
    pub pairs: Vec<Pair>,
    /// Games one arm lost to a panic, and the loss lines. A dropped pair is
    /// reported, never silently truncated.
    pub dropped: Vec<String>,
    /// Each arm's outcomes in aggregate — act-1 clear rate and death table —
    /// over every run that arm finished, dropped pairs included.
    pub baseline_outcomes: OutcomeTally,
    pub candidate_outcomes: OutcomeTally,
    /// Each arm's degraded pricings, summed over its runs. `None` where the
    /// arm's policy reports no spend at all — an arm that counted nothing
    /// says nothing rather than zero.
    pub baseline_degradations: Option<Degradations>,
    pub candidate_degradations: Option<Degradations>,
}

impl MatchReport {
    /// Pairs the candidate went deeper on.
    #[must_use]
    pub fn ups(&self) -> usize {
        self.pairs
            .iter()
            .filter(|pair| pair.candidate > pair.baseline)
            .count()
    }

    /// Pairs the candidate fell short on.
    #[must_use]
    pub fn downs(&self) -> usize {
        self.pairs
            .iter()
            .filter(|pair| pair.candidate < pair.baseline)
            .count()
    }

    /// Pairs that came out even.
    #[must_use]
    pub fn ties(&self) -> usize {
        self.pairs.len() - self.ups() - self.downs()
    }

    /// Mean floors, baseline then candidate.
    #[must_use]
    #[allow(clippy::cast_precision_loss, reason = "floor counts are small")]
    pub fn means(&self) -> (f64, f64) {
        if self.pairs.is_empty() {
            return (0.0, 0.0);
        }
        let n = self.pairs.len() as f64;
        (
            self.pairs
                .iter()
                .map(|pair| f64::from(pair.baseline))
                .sum::<f64>()
                / n,
            self.pairs
                .iter()
                .map(|pair| f64::from(pair.candidate))
                .sum::<f64>()
                / n,
        )
    }

    /// The one-sided sign test over the non-tied pairs: the chance of at
    /// least this many ups under a coin-flip null. Ties carry no evidence
    /// either way and are set aside, which is the standard treatment.
    #[must_use]
    pub fn p_value(&self) -> f64 {
        sign_test(self.ups() as u64, self.downs() as u64)
    }

    /// Whether the candidate measurably won: more ups than downs, at the
    /// conventional threshold. The gate expert iteration promotes on.
    #[must_use]
    pub fn candidate_wins(&self) -> bool {
        self.ups() > self.downs() && self.p_value() < 0.05
    }
}

/// P(X >= ups) for X ~ Binomial(ups + downs, 1/2): the one-sided tail,
/// computed exactly. Zero informative pairs is no evidence, which reads as
/// certainty of nothing: p = 1.
#[must_use]
pub fn sign_test(ups: u64, downs: u64) -> f64 {
    let n = ups + downs;
    if n == 0 {
        return 1.0;
    }
    // Sum C(n, k) / 2^n for k in ups..=n, the binomial coefficient carried
    // multiplicatively — exact in f64 well past any match size.
    let mut tail = 0.0_f64;
    let mut coefficient = 1.0_f64;
    #[allow(clippy::cast_precision_loss, reason = "match sizes are small")]
    for k in 0..=n {
        if k >= ups {
            tail += coefficient;
        }
        coefficient *= (n - k) as f64 / (k + 1) as f64;
    }
    #[allow(clippy::cast_possible_truncation, reason = "n is a match size")]
    {
        tail / 2.0_f64.powi(n as i32)
    }
}

/// One arm played out: each run's seed and floor in run order, the arm's
/// outcome tally, the loss lines of any run a panic took, and what the arm
/// answered with priors no checkpoint produced.
struct ArmOutcome {
    runs: Vec<Option<(String, u32)>>,
    tally: OutcomeTally,
    lost: Vec<String>,
    degradations: Option<Degradations>,
}

/// Plays one arm over `batch` with `policy`, and reads off how deep each run
/// got.
///
/// The half every kind of arm shares: what an arm *is* — a combat search
/// under one checkpoint, a run policy playing greedily against a frozen
/// resolver — is entirely the policy builder's business, and everything from
/// the seeds down to the pairing is identical either way. That is what keeps
/// a second kind of gate from being a second copy of the statistics.
fn play_arm(
    batch: &Batch<'_>,
    policy: &(dyn Fn() -> Box<dyn RolloutPolicy> + Sync),
    label: &str,
) -> ArmOutcome {
    let make_objective = || -> Box<dyn Objective> { Box::new(CombatStrength::default()) };
    let mut runs: Vec<Option<(String, u32)>> = vec![None; batch.runs];
    let mut tally = OutcomeTally::default();
    let mut lost = Vec::new();
    let mut degradations: Option<Degradations> = None;
    play_batch(
        batch,
        policy,
        &make_objective,
        &mut |index, played| match played {
            Ok(report) => {
                tally.record(&report);
                if let Some(spend) = &report.budget {
                    degradations
                        .get_or_insert_default()
                        .merge(spend.degradations);
                }
                runs[index] = Some((report.seed, report.floor));
            }
            Err(error) => lost.push(format!("{label}: {error}")),
        },
    );
    ArmOutcome {
        runs,
        tally,
        lost,
        degradations,
    }
}

/// Plays one combat arm: belief search under this arm's priors, with the
/// out-of-combat screens answered by the shared rollout policy.
fn play_combat_arm(config: &MatchConfig<'_>, net: &Arm, label: &str) -> ArmOutcome {
    let make_policy = || -> Box<dyn RolloutPolicy> {
        let mut search = BeliefSearch::with_rollout(
            config.search,
            CombatStrength::default(),
            (config.rollout)(),
        )
        .selecting((config.selection)());
        if let Some(net) = net {
            search = search.with_net(Arc::clone(net) as Arc<dyn crate::net::Evaluate>);
        }
        Box::new(search)
    };
    play_arm(&config.batch, &make_policy, label)
}

/// How a *macro* match is played: the run checkpoints are the arms, and
/// everything else is held identical between them.
///
/// The gate for a run policy, beside the one that grades a combat net. What
/// makes it a gate rather than two batches is what it fixes: the same seeds,
/// the same resolver on the same frozen combat checkpoint, and both arms
/// playing their macro decisions by argmax — which is what a deployed
/// checkpoint does, and so what a promotion decision must be made on. A
/// sampled arm would let two arms differ by their draws as much as by their
/// weights.
pub struct MacroMatchConfig<'a> {
    pub batch: Batch<'a>,
    /// The frozen combat checkpoint both arms resolve their fights with.
    pub combat_net: Arc<PolicyValueNet>,
    /// How those fights are answered. One description for both arms, built
    /// fresh per run, so the environment is the same environment.
    pub resolver: crate::actor::Resolver,
}

/// One arm of a macro match: this run checkpoint, played greedily.
///
/// Not an `Option` as [`Arm`] is. A combat match's netless arm still searches,
/// so uniform priors are a real policy to beat; a greedy macro arm over
/// uniform priors is always the *last* action on the screen — `max_by`
/// answers with the last maximum among equals — which is not a baseline any
/// promotion decision should be made against. The gate takes two checkpoints.
pub type MacroArm = Arc<PolicyValueNet>;

/// Plays one macro arm: this run checkpoint by argmax, against the shared
/// frozen resolver.
fn play_macro_arm(config: &MacroMatchConfig<'_>, net: &MacroArm, label: &str) -> ArmOutcome {
    let make_policy = || -> Box<dyn RolloutPolicy> {
        Box::new(crate::actor::MacroGreedy::new(
            Arc::clone(net) as Arc<dyn crate::net::Evaluate>,
            config
                .resolver
                .build(Arc::clone(&config.combat_net) as Arc<dyn crate::net::Evaluate>),
        ))
    };
    play_arm(&config.batch, &make_policy, label)
}

/// Plays a macro match: both run checkpoints over the identical games,
/// paired, and graded by exactly the statistics a combat match is graded by.
#[must_use]
pub fn play_macro_match(
    config: &MacroMatchConfig<'_>,
    baseline: &MacroArm,
    candidate: &MacroArm,
) -> MatchReport {
    paired(
        play_macro_arm(config, baseline, "baseline"),
        play_macro_arm(config, candidate, "candidate"),
    )
}

/// Plays one *student* arm: the checkpoint's own argmax inside fights, with
/// the shared rollout answering everything outside them.
fn play_student_arm(
    config: &MatchConfig<'_>,
    net: &Arc<PolicyValueNet>,
    label: &str,
) -> ArmOutcome {
    let make_policy = || -> Box<dyn RolloutPolicy> {
        Box::new(crate::actor::GreedyCombat::new(
            Arc::clone(net) as Arc<dyn crate::net::Evaluate>,
            (config.rollout)(),
        ))
    };
    play_arm(&config.batch, &make_policy, label)
}

/// Plays the policy-improvement gate: one checkpoint, searched against
/// itself unsearched, over the identical games.
///
/// The question expert iteration never asks itself and depends on entirely.
/// A generation trains the net toward the search's improved policy on the
/// premise that the search improves on the net; if it does not, the
/// generation teaches the net to be worse, and every downstream measurement —
/// the promotion match included — compares two checkpoints that were both
/// taught by the same broken teacher and so cannot see it.
///
/// The baseline arm is the *student*: the checkpoint answering its own
/// fights by argmax, one forward pass a decision. The candidate arm is the
/// *teacher*: the same checkpoint, the same games, the same out-of-combat
/// policy, with the fights searched at `config.search`. Everything else is
/// held identical, so `candidate_wins` reads as "the search at this budget
/// is worth running", and its absence reads as "this generation's labels
/// are not an improvement and should not be trained on".
#[must_use]
pub fn play_improvement(config: &MatchConfig<'_>, net: &Arc<PolicyValueNet>) -> MatchReport {
    paired(
        play_student_arm(config, net, "student"),
        play_combat_arm(config, &Some(Arc::clone(net)), "teacher"),
    )
}

/// Plays the match: both arms over the identical games, paired.
#[must_use]
pub fn play_match(config: &MatchConfig<'_>, baseline: &Arm, candidate: &Arm) -> MatchReport {
    paired(
        play_combat_arm(config, baseline, "baseline"),
        play_combat_arm(config, candidate, "candidate"),
    )
}

/// Two played arms, paired seed for seed.
fn paired(baseline_arm: ArmOutcome, candidate_arm: ArmOutcome) -> MatchReport {
    let mut report = MatchReport {
        dropped: baseline_arm
            .lost
            .into_iter()
            .chain(candidate_arm.lost)
            .collect(),
        baseline_outcomes: baseline_arm.tally,
        candidate_outcomes: candidate_arm.tally,
        baseline_degradations: baseline_arm.degradations,
        candidate_degradations: candidate_arm.degradations,
        ..MatchReport::default()
    };
    for slot in baseline_arm.runs.into_iter().zip(candidate_arm.runs) {
        // A slot either arm lost stays out of the pairing; its loss line is
        // already recorded.
        if let (Some((seed, baseline)), Some((candidate_seed, candidate))) = slot {
            assert_eq!(
                seed, candidate_seed,
                "both arms of a match play the same games"
            );
            report.pairs.push(Pair {
                seed,
                baseline,
                candidate,
            });
        }
    }
    report
}
