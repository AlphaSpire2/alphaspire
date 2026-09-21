//! The search core: information-set MCTS with per-rollout determinization.
//!
//! [`TrueState`] reduces the information set to a
//! singleton for coverage fuzzing; belief determinizers sample worlds from
//! player-visible information. Both use the same tree implementation.
//!
//! UCT selects by visits and mean value. Gumbel selection uses top-k root
//! sampling, sequential halving and completed Q values with policy priors.
//!
//! Every search run takes an explicit analysis seed: a generated script is
//! reproducible from (run seed, analysis seed, configuration).

/// One edge of the tree: an action out of a decision point, and what the
/// search has learned about it.
///
/// The fields are exactly what both selection families read. UCT reads
/// visits and mean value; Gumbel-style selection reads completed Q over the
/// prior. The prior is uniform until a learned policy fills it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeStats {
    /// How many rollouts descended this edge.
    pub visits: u64,
    /// How many descents through the parent offered this edge at all. Under
    /// `TrueState` every determinization offers every edge and this marches
    /// with the parent's visits; under a belief mode a world only offers the
    /// actions its hidden state allows, and subset-armed UCB explores against
    /// availability rather than the parent total (single-observer ISMCTS).
    pub availability: u64,
    /// The sum of rollout values that came back through it.
    pub total_value: f64,
    /// The probability a policy prior assigns this action. Uniform until a
    /// learned prior exists; the slot is carried so Gumbel selection is
    /// additive.
    pub prior: f64,
}

impl EdgeStats {
    /// A fresh edge under a prior.
    #[must_use]
    pub const fn fresh(prior: f64) -> Self {
        Self {
            visits: 0,
            availability: 0,
            total_value: 0.0,
            prior,
        }
    }

    /// The mean value of the rollouts that descended here.
    #[must_use]
    pub fn mean_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            #[allow(
                clippy::cast_precision_loss,
                reason = "visit counts are far below f64 precision"
            )]
            {
                self.total_value / self.visits as f64
            }
        }
    }
}

/// What selection reads about the decision point itself, beside its edges.
///
/// UCT reads only the visit total. Completed-Q needs the node's own value
/// too: where a checkpoint priced the node, its value head is what stands in
/// for an edge nobody has descended yet.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NodeStats {
    /// Descents through this decision point.
    pub visits: u64,
    /// The node's own value estimate, where one exists — the checkpoint's
    /// value head, read once when the node was priced. `None` leaves the
    /// children's own evidence as the only estimate there is.
    pub value: Option<f64>,
}

/// Which edge a descent takes, given the decision point's totals and every
/// edge's statistics. UCT and Gumbel-style selection are both
/// implementations; which one runs is configuration, not architecture.
pub trait Selection {
    /// The index of the edge to descend. `edges` is never empty.
    fn descend(&mut self, node: NodeStats, edges: &[EdgeStats]) -> usize;

    /// The root schedule this selection wants, where it has one.
    ///
    /// Gumbel's top-k sampling and sequential halving are a budget plan over
    /// the root's actions rather than a per-descent rule, so the root is
    /// driven by the schedule and everything below it by [`Self::descend`].
    /// UCT has nothing to say here: it answers `None`, and its root is
    /// descended like any other node.
    fn root_schedule(&self) -> Option<Gumbel> {
        None
    }
}

impl Selection for Box<dyn Selection> {
    fn descend(&mut self, node: NodeStats, edges: &[EdgeStats]) -> usize {
        (**self).descend(node, edges)
    }

    fn root_schedule(&self) -> Option<Gumbel> {
        (**self).root_schedule()
    }
}

/// Upper confidence bounds applied to trees, the first-pass selection policy:
/// exploitation by mean value, exploration by visit imbalance, priors as a
/// multiplier on the exploration term so a filled prior steers without
/// overriding evidence.
#[derive(Clone, Copy, Debug)]
pub struct Uct {
    /// The exploration constant; higher wanders wider.
    pub exploration: f64,
}

impl Default for Uct {
    fn default() -> Self {
        Self {
            exploration: std::f64::consts::SQRT_2,
        }
    }
}

impl Selection for Uct {
    #[allow(
        clippy::cast_precision_loss,
        reason = "visit counts are far below f64 precision"
    )]
    fn descend(&mut self, _node: NodeStats, edges: &[EdgeStats]) -> usize {
        assert!(!edges.is_empty(), "a decision point offers an action");
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (index, edge) in edges.iter().enumerate() {
            let score = if edge.visits == 0 {
                f64::INFINITY
            } else {
                let offered = (edge.availability.max(1)) as f64;
                edge.mean_value()
                    + self.exploration * edge.prior * (offered.ln() / edge.visits as f64).sqrt()
            };
            if score > best_score {
                best_score = score;
                best = index;
            }
        }
        best
    }
}

/// Gumbel search: the selection policy the learned prior was waiting for.
///
/// Two halves, both reading statistics [`EdgeStats`] has carried since day
/// one. At the root, `considered` actions are drawn from the prior *without
/// replacement* by the Gumbel-top-k trick — one variate per action, no
/// rejection loop — and the whole simulation budget is spent on those, by
/// sequential halving: every survivor gets the same number of simulations,
/// then the weakest half is dropped and the freed budget goes to the rest.
/// Inside the tree, a descent walks toward the improved policy that
/// completed-Q defines.
///
/// Why it replaces UCT once a net guides the search: UCT scores an unvisited
/// edge at infinity, so a prior can only order the expansions and season the
/// exploration term afterwards — at 64 simulations over 20 actions the
/// budget is spread by the tree, not by the policy. Gumbel spends the budget
/// where the prior points and still improves on it: its answer is provably
/// no worse in expected value than the prior itself at any budget, which is
/// the guarantee expert iteration needs at the small budgets self-play can
/// afford.
///
/// The root answer is already a sample from the improved policy — the Gumbel
/// variates are the randomness — so [`SearchConfig::temperature`] does not
/// apply under this selection.
#[derive(Clone, Copy, Debug)]
pub struct Gumbel {
    /// How many root actions the budget is spread over (`m` in the paper),
    /// clamped to what the decision actually offers.
    pub considered: usize,
    /// `c_visit`: how far evidence outgrows the prior as the busiest edge
    /// fills up.
    pub visit_scale: f64,
    /// `c_scale`: the weight of value against prior in the transform.
    ///
    /// Ten times the reference implementation's default, and left there
    /// deliberately: paired with [`Self::value_span`] the product is what
    /// sets the weight, and a decision of typical spread lands where the
    /// reference constant would have put it — but proportionally, which the
    /// reference does not.
    pub value_scale: f64,
    /// The objective's own unit: how far apart a decision's values must be
    /// before the transform lets them fill its whole range. See
    /// `value_offsets`.
    pub value_span: f64,
}

impl Default for Gumbel {
    fn default() -> Self {
        Self {
            considered: 16,
            visit_scale: 50.0,
            value_scale: 1.0,
            // What both learned objectives score on: `CombatStrength` and
            // `ActBoundary` each pay one for the outcome, one for hit points
            // retained, and small change for potions and floors, so a whole
            // unit is the difference between winning the fight intact and
            // losing it. `Coverage`, which nothing trains on, spans more
            // than this per novel model and is unaffected.
            value_span: 1.0,
        }
    }
}

impl Selection for Gumbel {
    #[allow(
        clippy::cast_precision_loss,
        reason = "visit counts are far below f64 precision"
    )]
    fn descend(&mut self, node: NodeStats, edges: &[EdgeStats]) -> usize {
        assert!(!edges.is_empty(), "a decision point offers an action");
        let policy = improved_policy(*self, node, edges);
        // Deterministic, and the reason no exploration constant appears: the
        // edge whose share of the descents most lags the improved policy is
        // the one that moves the visit distribution toward it, so over many
        // descents the visits *become* the policy.
        let total: u64 = edges.iter().map(|edge| edge.visits).sum();
        let denominator = 1.0 + total as f64;
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (index, edge) in edges.iter().enumerate() {
            let score = policy[index] - edge.visits as f64 / denominator;
            if score > best_score {
                best_score = score;
                best = index;
            }
        }
        best
    }

    fn root_schedule(&self) -> Option<Self> {
        Some(*self)
    }
}

/// The priors of `edges` as a distribution: what the checkpoint said,
/// renormalized over the edges actually standing. A uniform prior survives
/// the normalization as a uniform distribution.
fn edge_priors(edges: &[EdgeStats]) -> Vec<f64> {
    let total: f64 = edges.iter().map(|edge| edge.prior.max(0.0)).sum();
    if total > 0.0 {
        edges
            .iter()
            .map(|edge| edge.prior.max(0.0) / total)
            .collect()
    } else {
        #[allow(clippy::cast_precision_loss, reason = "action counts are small")]
        {
            vec![1.0 / edges.len() as f64; edges.len()]
        }
    }
}

/// The value an edge nobody has descended is credited with: the node's own
/// estimate pulled toward whatever its visited siblings have found. Without
/// it an unvisited edge scores as worthless and the halving starves the very
/// actions the prior likes; with it, a node whose visited children are all
/// losing lowers its unvisited ones too, which is what "completed" means.
#[allow(
    clippy::cast_precision_loss,
    reason = "visit counts are far below f64 precision"
)]
fn mixed_value(node: NodeStats, edges: &[EdgeStats], priors: &[f64]) -> f64 {
    let visits: u64 = edges.iter().map(|edge| edge.visits).sum();
    let own = node.value.unwrap_or_else(|| {
        if visits == 0 {
            0.0
        } else {
            edges.iter().map(|edge| edge.total_value).sum::<f64>() / visits as f64
        }
    });
    let descended = || edges.iter().zip(priors).filter(|(edge, _)| edge.visits > 0);
    let weight: f64 = descended().map(|(_, prior)| prior).sum();
    let weighted: f64 = descended()
        .map(|(edge, prior)| prior * edge.mean_value())
        .sum();
    if visits == 0 || weight <= 0.0 {
        own
    } else {
        (own + visits as f64 * weighted / weight) / (1.0 + visits as f64)
    }
}

/// Completed Q: an edge that has been descended answers with its own mean,
/// one that has not answers with `mixed_value`.
fn completed_q(node: NodeStats, edges: &[EdgeStats], priors: &[f64]) -> Vec<f64> {
    let mixed = mixed_value(node, edges, priors);
    edges
        .iter()
        .map(|edge| {
            if edge.visits > 0 {
                edge.mean_value()
            } else {
                mixed
            }
        })
        .collect()
}

/// How heavily value counts against prior at this node: `c_visit + max_b N(b)`
/// times `c_scale`, so a decision the search has barely touched trusts its
/// prior and one it has hammered trusts its evidence.
#[allow(
    clippy::cast_precision_loss,
    reason = "visit counts are far below f64 precision"
)]
fn value_weight(schedule: Gumbel, edges: &[EdgeStats]) -> f64 {
    let busiest = edges.iter().map(|edge| edge.visits).max().unwrap_or(0);
    (schedule.visit_scale + busiest as f64) * schedule.value_scale
}

/// How far each value stands above the decision's worst, measured against
/// whichever is larger: the decision's own span, or `unit`.
///
/// Divides by the larger of the observed span and `unit`. The floor prevents
/// small differences from being amplified into confident policy targets.
/// Equal values contribute zero, leaving the prior unchanged.
fn value_offsets(values: &[f64], unit: f64) -> Vec<f64> {
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (high - low).max(unit);
    values
        .iter()
        .map(|value| {
            if span > f64::EPSILON {
                (value - low) / span
            } else {
                0.0
            }
        })
        .collect()
}

/// `log prior + σ(completed Q)`: the improved policy's logits, and the one
/// quantity Gumbel search reads everywhere — interior selection descends
/// toward its softmax, the root's halving ranks by it under each candidate's
/// Gumbel variate, and expert iteration trains toward it.
fn improved_logits(schedule: Gumbel, node: NodeStats, edges: &[EdgeStats]) -> Vec<f64> {
    let priors = edge_priors(edges);
    let transformed = value_offsets(&completed_q(node, edges, &priors), schedule.value_span);
    let weight = value_weight(schedule, edges);
    priors
        .iter()
        .zip(transformed)
        .map(|(prior, value)| logit(*prior) + weight * value)
        .collect()
}

/// The improved policy over `edges`: the softmax of `improved_logits`.
///
/// The one distribution Gumbel search is defined by. Interior selection
/// descends toward it, the root records it, and expert iteration trains
/// toward what the root recorded — so it is public, and pinned by tests
/// against the statistics it is a function of rather than only through a
/// whole search.
#[must_use]
pub fn improved_policy(schedule: Gumbel, node: NodeStats, edges: &[EdgeStats]) -> Vec<f64> {
    softmax(&improved_logits(schedule, node, edges))
}

/// The log of a probability, floored so a prior of zero is merely hopeless
/// rather than a `NaN` waiting to happen.
fn logit(probability: f64) -> f64 {
    probability.max(f64::MIN_POSITIVE).ln()
}

/// How far apart a decision's completed values actually are: `max − min`,
/// read before any transform normalizes them away.
///
/// This is the evidence a label rests on, and the quantity
/// `value_offsets` keeps rather than divides out. Recorded on every
/// searched decision so a learner can weight a row by it and so the question
/// "how much did this label actually know?" is a query rather than an
/// investigation.
fn value_spread(values: &[f64]) -> f64 {
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if high > low { high - low } else { 0.0 }
}

/// A numerically settled softmax.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|logit| (logit - peak).exp()).collect();
    let total: f64 = weights.iter().sum();
    if total > 0.0 {
        weights.iter().map(|weight| weight / total).collect()
    } else {
        #[allow(clippy::cast_precision_loss, reason = "action counts are small")]
        {
            vec![1.0 / logits.len() as f64; logits.len()]
        }
    }
}

/// One draw from the standard Gumbel distribution, off the analysis stream:
/// `-log(-log u)`. The double is pinched away from both ends so neither
/// logarithm can reach infinity.
fn gumbel_variate(rng: &mut MegaRandom) -> f64 {
    let uniform = rng
        .next_double()
        .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    -(-uniform.ln()).ln()
}

use std::collections::{HashMap, HashSet};

use sts2_engine::{Action, Simulator};
use sts2_rng::MegaRandom;

/// The search's own random stream, split off the harness's on the first
/// searched decision and never rejoined.
///
/// The harness hands every policy one stream per run, and the rollout
/// policy answers the run's out-of-combat screens off it. Were the search
/// to draw its analysis seeds and Gumbel variates off that same stream,
/// how much it searched would move what the rollout drew next — and two
/// arms of a match, differing only in their priors, would make *different*
/// card picks and path choices from the first fight on, which is a
/// comparison of nothing. One draw seeds this stream where the first
/// searched decision falls, identically for both arms; after that the
/// harness stream is the rollout's alone, and the run's macro play pairs.
fn search_stream<'a>(
    stream: &'a mut Option<MegaRandom>,
    harness: &mut MegaRandom,
) -> &'a mut MegaRandom {
    stream.get_or_insert_with(|| MegaRandom::new(harness.next_u64()))
}

use crate::env::{Belief, Determinizer, NodeKey, TrueState, fight_over};
use crate::objective::Objective;
use crate::policy::RolloutPolicy;

/// The budgets and temperaments of one search.
#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    /// Rollouts per decision.
    pub iterations: u32,
    /// How far past the tree a rollout walks before it is scored.
    pub rollout_depth: u32,
    /// Root sampling temperature over visit counts: zero takes the most
    /// visited edge, higher wanders across the plausible ones. Diversity
    /// beats strength for coverage, so the default wanders a little.
    pub temperature: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            iterations: 64,
            rollout_depth: 30,
            temperature: 0.5,
        }
    }
}

/// The value of a freshly expanded leaf: the net where one guides, a
/// rollout where none does, and the objective's own word wherever the walk
/// has already ended.
///
/// A rollout can also end at an *engine dead end*: a live screen the
/// engine enumerates no answer for (an empty offer), or an action it
/// enumerates and then refuses to apply (`step_quietly` rolls the state
/// back, so the world still stands where it stood). Either shape has been
/// an engine fault before, and the handling does not depend on one being
/// open: unsupported content and future faults can still dead-end a sampled
/// line, and neither shape is knowable before the state stands (the static
/// guard in `policy::permitted_actions` covers what is knowable — the
/// EVENT.TRIAL notes there).
///
/// In both cases the honest answer is the same leaf [`SearchNode::select`]
/// gives when a world offers nothing a node knows: stop and score the state
/// where it stands — exactly as at the depth cap — never fabricate a step
/// past it, and never void the whole search over one impossible sampled
/// line. The rollout policy is never asked to answer a screen that offers
/// nothing, and its own "a live run offers an action" invariant stays: on
/// the *authoritative* run these faults must surface loudly, and they do —
/// through the policy's panic or the harness's own step error, either way
/// with the run's seed as the reproducer. The descent loop's steps keep the
/// same discipline inline.
#[allow(
    clippy::too_many_arguments,
    reason = "one argument per thing a leaf is scored against"
)]
fn leaf_value(
    rollout_depth: u32,
    determinizer: &dyn Determinizer,
    objective: &dyn Objective,
    rollout: &mut dyn RolloutPolicy,
    net: Option<&dyn crate::net::Evaluate>,
    simulator: &mut Simulator,
    rng: &mut MegaRandom,
    spent: &mut Spent,
) -> f64 {
    if simulator.state().terminal.is_none()
        && !past_horizon(determinizer, simulator)
        && let Some(net) = net
    {
        let value =
            crate::probe::timed(crate::probe::Phase::LeafNet, || net.state_value(simulator));
        return within_reach(objective, simulator, value);
    }
    let mut walked = 0;
    let ending = loop {
        if walked == rollout_depth {
            break crate::probe::Tally::RolloutCapped;
        }
        if simulator.state().terminal.is_some() {
            break crate::probe::Tally::RolloutTerminal;
        }
        if past_horizon(determinizer, simulator) {
            break crate::probe::Tally::RolloutHorizon;
        }
        if simulator.legal_actions().is_empty() {
            break crate::probe::Tally::RolloutDeadEnd;
        }
        let action = crate::probe::timed(crate::probe::Phase::RolloutPick, || {
            rollout.choose(simulator, rng)
        });
        spent.steps += 1;
        if crate::probe::timed(crate::probe::Phase::RolloutStep, || {
            simulator.step_quietly(&action)
        })
        .is_err()
        {
            break crate::probe::Tally::RolloutDeadEnd;
        }
        crate::probe::tally(crate::probe::Tally::RolloutSteps, 1);
        walked += 1;
    };
    if ending == crate::probe::Tally::RolloutDeadEnd {
        spent.degradations.dead_ends += 1;
    }
    crate::probe::tally(ending, 1);
    crate::probe::timed(crate::probe::Phase::Peek, || objective.peek(simulator))
}

/// A net's estimate held inside what the objective can pay for the state.
///
/// The value head is a plain linear output with no bound of its own; at a
/// state far from its training pool it extrapolates, and a maximising
/// search then plays for a payoff that cannot exist (found on turn-40 stalls
/// priced above 2 against a ceiling near 1). The clamp removes only the
/// impossible part — an overshoot that stays under the ceiling is still the
/// head's error — and the tallies say how often each end of the bound had to
/// act: the floor mostly catches a head reading a lost position a hair under
/// zero, the ceiling is the extrapolation the bound exists for.
fn within_reach(objective: &dyn Objective, simulator: &Simulator, value: f64) -> f64 {
    let range = objective.attainable(simulator);
    if value < *range.start() {
        crate::probe::tally(crate::probe::Tally::LeafClippedLow, 1);
        *range.start()
    } else if value > *range.end() {
        crate::probe::tally(crate::probe::Tally::LeafClippedHigh, 1);
        *range.end()
    } else {
        value
    }
}

/// The gameplay identity of an action: which copy of a card carries a play is
/// not part of what the play does. Instance ids and pile positions inside a
/// card handle are scrubbed, and a multi-card answer names a set rather than
/// an order — so two same-print Strikes in hand are one class, and choosing
/// either of two identical copies on a screen is one choice. Everything else
/// an action carries (the full fingerprint — upgrade, enchantment,
/// properties — the target, an event option's place on its page) stays, so
/// two actions share a class only when no rule could tell their outcomes
/// apart.
#[must_use]
pub fn action_class(action: &Action) -> String {
    let mut value = serde_json::to_value(action).expect("an action serializes");
    scrub_copy_identity(&mut value);
    value.to_string()
}

fn scrub_copy_identity(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            // A card handle is the one shape that names a copy: strip the
            // instance id and its pile position, keep the fingerprint.
            if map.contains_key("card_id") && map.contains_key("fingerprint") {
                map.remove("card_id");
                map.remove("index");
            }
            for entry in map.values_mut() {
                scrub_copy_identity(entry);
            }
            // A multi-card answer is a set: sort the scrubbed handles so two
            // orders of the same picks share a class.
            if let Some(serde_json::Value::Array(cards)) = map.get_mut("cards") {
                cards.sort_by_cached_key(serde_json::Value::to_string);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                scrub_copy_identity(item);
            }
        }
        _ => {}
    }
}

/// One representative per gameplay class, first occurrence kept, order
/// preserved: what the search branches over instead of the raw legal list.
/// Which copy the representative names is deterministic, so an emitted
/// script still replays exactly.
#[must_use]
pub fn canonical_actions(legal: &[Action]) -> Vec<Action> {
    canonical_actions_with_classes(legal).0
}

/// [`canonical_actions`] with the class each representative stands for,
/// aligned: computed once here so a node can match a world's offers by
/// class without serializing its own list again.
fn canonical_actions_with_classes(legal: &[Action]) -> (Vec<Action>, Vec<String>) {
    let mut seen = HashSet::new();
    let mut actions = Vec::new();
    let mut classes = Vec::new();
    for action in legal {
        let class = action_class(action);
        if seen.insert(class.clone()) {
            actions.push(action.clone());
            classes.push(class);
        }
    }
    (actions, classes)
}

/// One decision point the search has seen.
///
/// The canonical action list is computed once, from the first world that
/// reached the key: node keys carry what a player sees and legality is a
/// function of it, so every world at one key offers the same action
/// *classes*. The key folds copy identity and hand order, so another world
/// at the same key may name the same class through another copy in another
/// place — a node therefore matches each world's offers to its classes
/// (see [`SearchNode::world_actions`]) and steps the world's own action,
/// never its representative. Selection still checks the current world's
/// offers before stepping — the guard that keeps the search honest where a
/// key is coarser than the screen.
#[derive(Clone, Debug)]
struct SearchNode {
    visits: u64,
    edges: Vec<SearchEdge>,
    /// The class representatives this decision offers, in offer order.
    actions: Vec<Action>,
    /// The class of each representative, aligned with `actions`.
    classes: Vec<String>,
    /// A prior per action, aligned with `actions`: the checkpoint's pricing
    /// where a net guides the search, uniform where none does.
    priors: Vec<f64>,
    /// What the checkpoint's value head said about this decision when it
    /// priced the priors, kept rather than discarded: completed-Q spends it
    /// on the edges no descent has reached. `None` where no net guides.
    value: Option<f64>,
}

impl SearchNode {
    /// What selection reads about the decision point itself.
    const fn stats(&self) -> NodeStats {
        NodeStats {
            visits: self.visits,
            value: self.value,
        }
    }

    /// This world's own action for each of the node's classes, aligned with
    /// `actions`: the offer equal to the representative where the world
    /// names the same copy, the offer of the same class where it names
    /// another, and nothing where the world does not offer the class. The
    /// classes are serialized only where equality left something unmatched
    /// and the world holds an offer the node's list does not — the common
    /// case, one world naming the copies the node was built from, pays a
    /// comparison per action and nothing more.
    fn world_actions<'a>(&self, legal: &'a [Action]) -> Vec<Option<&'a Action>> {
        let mut matched: Vec<Option<&'a Action>> = self
            .actions
            .iter()
            .map(|action| legal.iter().find(|offered| *offered == action))
            .collect();
        if matched.iter().any(Option::is_none)
            && legal.iter().any(|offered| !self.actions.contains(offered))
        {
            let classes: Vec<String> = legal.iter().map(action_class).collect();
            for (index, slot) in matched.iter_mut().enumerate() {
                if slot.is_none() {
                    *slot = classes
                        .iter()
                        .position(|class| *class == self.classes[index])
                        .map(|position| &legal[position]);
                }
            }
        }
        matched
    }

    /// Expansion: of the classes this world offers that have no edge yet,
    /// the one the prior likes best — offer order under a uniform prior.
    /// Another world at the same key may offer classes this one does not;
    /// they expand when their world comes up.
    fn next_expansion(&self, world: &[Option<&Action>]) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (index, offered) in world.iter().enumerate() {
            if offered.is_some()
                && !self.edges.iter().any(|edge| edge.index == index)
                && best.is_none_or(|current| self.priors[index] > self.priors[current])
            {
                best = Some(index);
            }
        }
        best
    }

    /// Selection among the edges this world offers, against their
    /// availability: an edge is judged over the descents that could have
    /// taken it, not over descents whose worlds never offered it. Pays the
    /// availability of every offered edge and answers with the index of the
    /// edge to descend.
    ///
    /// `None` where this world offers no action the node has an edge for:
    /// nothing to descend, and the caller scores the state where it stands.
    /// Reachable only where the observation key is coarser than the offers —
    /// the `Rewards` context keys by set id, not contents, so under an
    /// act-scoped belief two worlds' differently-rolled reward screens can
    /// share a node while offering disjoint claims. The node's cached
    /// classes and this world's screen then miss each other entirely, and
    /// the honest answer is a leaf, never a step the world refuses.
    fn select(
        &mut self,
        world: &[Option<&Action>],
        selection: &mut dyn Selection,
    ) -> Option<usize> {
        let available = self.offer(world);
        if available.is_empty() {
            return None;
        }
        let stats: Vec<EdgeStats> = available
            .iter()
            .map(|&index| self.edges[index].stats)
            .collect();
        Some(available[selection.descend(self.stats(), &stats)])
    }

    /// Pays the availability of every edge this world offers and answers with
    /// their indices. Every descent that reaches this node pays it, whichever
    /// rule then picks the edge — a root schedule's pick is still a descent
    /// the other offered edges were passed over by.
    fn offer(&mut self, world: &[Option<&Action>]) -> Vec<usize> {
        let available: Vec<usize> = self
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| world[edge.index].is_some())
            .map(|(index, _)| index)
            .collect();
        for &index in &available {
            self.edges[index].stats.availability += 1;
        }
        available
    }

    /// Which of this decision's action classes the given world offers,
    /// aligned with `actions`.
    fn offered(world: &[Option<&Action>]) -> Vec<bool> {
        world.iter().map(Option::is_some).collect()
    }
}

/// What a descent does at the node it stands on: open the action class at
/// this index, or walk the edge at this one.
#[derive(Clone, Copy, Debug)]
enum Step {
    Expand(usize),
    Descend(usize),
}

/// An action out of a decision point: what the search has learned about it,
/// and — once a deterministic descent has keyed it — where it leads.
#[derive(Clone, Debug)]
struct SearchEdge {
    /// The action class this edge opens, as an index into the node's list.
    index: usize,
    /// That class's representative, for whoever reads the tree; what a
    /// descent steps is the current world's own action of the class.
    action: Action,
    stats: EdgeStats,
    /// The key this edge reaches. Cached only under deterministic
    /// transitions, where one action from one state always lands on one key,
    /// so a descent pays for each key once instead of every visit.
    child: Option<NodeKey>,
}

/// One decision's Gumbel plan at the root: the actions the top-k trick
/// sampled, the budget sequential halving spreads over them, and the answer
/// the survivor gives.
///
/// It lives for one call to [`Mcts::decide_with_policy`] — the Gumbel
/// variates are drawn once per decision, and every simulation of that
/// decision is dealt against them.
struct GumbelPlan {
    schedule: Gumbel,
    /// The sampled candidates: an index into the root's action list, and
    /// `g + log prior` — the variate that sampled it plus its prior's logit.
    /// Ranking adds σ(completed Q) to this and nothing else.
    candidates: Vec<(usize, f64)>,
    /// Candidate slots still in the running, as indices into `candidates`.
    alive: Vec<usize>,
    /// This phase's deal, in reverse: simulations are taken off the back.
    queue: Vec<usize>,
    /// Halvings left, this phase included.
    phases_left: u32,
    /// Simulations left in the decision's whole budget.
    budget: u32,
    /// Whether a phase has been dealt yet, so the first deal does not halve
    /// a field nothing has been spent on.
    dealt: bool,
}

impl GumbelPlan {
    /// Samples `considered` root actions from `priors` without replacement
    /// and sets up the halving schedule that will spend `budget` on them.
    #[allow(clippy::cast_precision_loss, reason = "action counts are small")]
    fn new(
        schedule: Gumbel,
        priors: &[f64],
        budget: u32,
        pinned: &[usize],
        rng: &mut MegaRandom,
    ) -> Self {
        // Gumbel-top-k: the k largest of `g_i + log p_i` are a draw of k
        // actions from p without replacement. One variate per action, no
        // rejection loop, and the variates are kept — they are what makes
        // the final answer a sample rather than an argmax.
        let total: f64 = priors.iter().map(|prior| prior.max(0.0)).sum();
        let mut candidates: Vec<(usize, f64)> = priors
            .iter()
            .enumerate()
            .map(|(index, prior)| {
                let probability = if total > 0.0 {
                    prior.max(0.0) / total
                } else {
                    1.0 / priors.len() as f64
                };
                (index, gumbel_variate(rng) + logit(probability))
            })
            .collect();
        candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
        // A pinned action moves to the front of the draw and keeps its key,
        // so the field it is dealt into is the same field it would have
        // been ranked against had the prior favoured it.
        let (mut front, back): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|(index, _)| pinned.contains(index));
        let width = schedule
            .considered
            .clamp(1, priors.len().max(1))
            .max(front.len());
        front.extend(back);
        front.truncate(width);
        let mut candidates = front;
        candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
        let alive = (0..candidates.len()).collect();
        // ceil(log2 m) halvings take m candidates down to one.
        let phases_left = candidates.len().next_power_of_two().trailing_zeros().max(1);
        Self {
            schedule,
            candidates,
            alive,
            queue: Vec::new(),
            phases_left,
            budget,
            dealt: false,
        }
    }

    /// The root action the next simulation should take, as an index into the
    /// root's action list. Skips any candidate the sampled world does not
    /// offer, and answers `None` when the budget is spent or nothing dealt
    /// can be played here — the caller then descends the root like any other
    /// node, which is the same guard selection already keeps.
    fn next(&mut self, offered: &[bool], node: &SearchNode) -> Option<usize> {
        if self.budget == 0 || self.candidates.is_empty() {
            return None;
        }
        // At most one refill: a phase whose whole deal this world refuses is
        // dropped rather than spun on.
        for _ in 0..2 {
            if let Some(position) = self
                .queue
                .iter()
                .rposition(|&slot| offered[self.candidates[slot].0])
            {
                let slot = self.queue.remove(position);
                self.budget -= 1;
                return Some(self.candidates[slot].0);
            }
            self.queue.clear();
            self.deal(node);
        }
        None
    }

    /// Ends the phase: drops the weaker half of the field, then deals the
    /// next phase's simulations evenly over the survivors. The last phase
    /// spends whatever the budget has left, so no simulation is wasted.
    #[allow(clippy::cast_possible_truncation, reason = "action counts are small")]
    fn deal(&mut self, node: &SearchNode) {
        if self.dealt {
            if self.alive.len() > 1 {
                let mut ranked = self.ranking(node);
                ranked.truncate(self.alive.len().div_ceil(2));
                self.alive = ranked;
            }
            self.phases_left = self.phases_left.saturating_sub(1);
        }
        self.dealt = true;
        let alive = self.alive.len().max(1) as u32;
        let each = if self.phases_left <= 1 {
            self.budget.div_ceil(alive)
        } else {
            self.budget / (self.phases_left * alive)
        }
        .max(1);
        for _ in 0..each {
            self.queue.extend_from_slice(&self.alive);
        }
        self.queue.reverse();
    }

    /// The surviving candidates, best first.
    fn ranking(&self, node: &SearchNode) -> Vec<usize> {
        let scores = self.scores(node);
        let mut alive = self.alive.clone();
        alive.sort_by(|left, right| scores[*right].total_cmp(&scores[*left]));
        alive
    }

    /// `g + log prior + σ(completed Q)` per candidate slot: what the halving
    /// and the final answer both rank by. A candidate the search has not
    /// expanded yet is scored on the mixed value, exactly as an unvisited
    /// edge is — its Gumbel key is not silently docked for being late.
    fn scores(&self, node: &SearchNode) -> Vec<f64> {
        let stats: Vec<EdgeStats> = node.edges.iter().map(|edge| edge.stats).collect();
        let priors = edge_priors(&stats);
        let mut values = completed_q(node.stats(), &stats, &priors);
        let mixed = mixed_value(node.stats(), &stats, &priors);
        // The mixed value rides along through the normalization so an
        // unexpanded candidate is measured on the same scale as the rest.
        values.push(mixed);
        let scaled = value_offsets(&values, self.schedule.value_span);
        let weight = value_weight(self.schedule, &stats);
        let unexpanded = *scaled.last().expect("the mixed value was pushed");
        self.candidates
            .iter()
            .map(|&(action, key)| {
                let value = node
                    .edges
                    .iter()
                    .position(|edge| edge.index == action)
                    .map_or(unexpanded, |edge| scaled[edge]);
                key + weight * value
            })
            .collect()
    }

    /// The decision's answer: the best surviving candidate, as an index into
    /// the root's action list. Under Gumbel this is already a sample from the
    /// improved policy, so no temperature is applied on top of it.
    fn answer(&self, node: &SearchNode) -> Option<usize> {
        self.ranking(node)
            .first()
            .map(|&slot| self.candidates[slot].0)
    }
}

/// The objective's word about the state a walk stopped on, on the probe's
/// clock. Every leaf in this file is scored through here.
fn peek(objective: &dyn Objective, simulator: &Simulator) -> f64 {
    crate::probe::timed(crate::probe::Phase::Peek, || objective.peek(simulator))
}

/// Whether a sampled world has walked out of what its mode samples.
fn past_horizon(determinizer: &dyn Determinizer, simulator: &Simulator) -> bool {
    crate::probe::timed(crate::probe::Phase::Horizon, || {
        determinizer.beyond_horizon(simulator)
    })
}

/// Whether a sampled world stands on a decision its mode plays rather than
/// branches over — the fight fence in act mode.
fn played_out(determinizer: &dyn Determinizer, simulator: &Simulator) -> bool {
    crate::probe::timed(crate::probe::Phase::Horizon, || {
        determinizer.plays_out(simulator)
    })
}

/// How many tree edges one descent may walk. A fight's tree outlives the
/// decision that grew it, and inside one turn a line is deterministic, so
/// where a play is free and hands itself back — a card made free for the
/// combat that returns to the hand — every simulation of every later decision
/// re-walks the same line and adds an edge to its end: nothing repeats,
/// because each play moves a counter, and nothing runs out. The line's depth
/// then grows with the square of the decisions taken and one decision comes
/// to cost minutes. Ordinary descents are a few dozen edges deep at any
/// budget in use, so the ceiling binds only there.
const DESCENT_CEILING: usize = 128;

/// How many decisions the search answers inside one turn before it ends the
/// turn itself. The same free play that returns to the hand is worth a little
/// more each time it is made, so no value ever prefers ending the turn and
/// the fight would stand on one turn until the harness's step limit. A human
/// turn is tens of decisions; past this many the turn is over.
const TURN_DECISION_CEILING: usize = 128;

/// The key of the decision point a sampled world stands on.
fn keyed(determinizer: &dyn Determinizer, simulator: &Simulator) -> NodeKey {
    crate::probe::timed(crate::probe::Phase::NodeKey, || {
        determinizer
            .node_key(simulator)
            .expect("a settled state keys")
    })
}

/// One step taken inside the tree; false where the world refused it (see
/// [`leaf_value`]). A refusal at the legality gate moves nothing; an
/// application that escapes mid-way leaves the simulator poisoned, which is
/// fine — every failed walk is scored where it stands and its sampled world
/// is dropped with the iteration.
fn tree_step(simulator: &mut Simulator, action: &Action) -> bool {
    crate::probe::timed(crate::probe::Phase::TreeStep, || {
        simulator.step_quietly(action)
    })
    .is_ok()
}

/// The node for a key, priced on first sight: the decision's canonical action
/// classes, a prior each, and — where a checkpoint guides the search — the
/// value its head reads off the position, held inside what the objective can
/// pay for it ([`within_reach`]).
///
/// A free function over the table rather than a method, so the borrow it
/// hands back stays disjoint from the search's selection policy.
///
/// A node the checkpoint could not price is counted into `degradations`: the
/// selection policy is left with a flat prior over that decision and nothing
/// downstream can tell that from a genuinely flat one.
fn node_for<'a>(
    nodes: &'a mut HashMap<NodeKey, SearchNode>,
    key: NodeKey,
    simulator: &Simulator,
    objective: &dyn Objective,
    net: Option<&dyn crate::net::Evaluate>,
    degradations: &mut Degradations,
) -> &'a mut SearchNode {
    nodes.entry(key).or_insert_with(|| {
        crate::probe::tally(crate::probe::Tally::NodesPriced, 1);
        crate::probe::timed(crate::probe::Phase::Expand, || {
            // Filtered before the node ever exists: an action the engine
            // would refuse (see `policy::permitted_actions` — EVENT.TRIAL's
            // poisoned line) never gets an action slot, so no schedule,
            // expansion, or descent can step it.
            let (actions, classes) =
                canonical_actions_with_classes(&crate::policy::permitted_actions(simulator));
            let (priors, value) = match net {
                Some(net) => {
                    let (priors, value, degraded) =
                        net.priors_and_value(simulator, &actions).into_parts();
                    degradations.priced(degraded);
                    let value = within_reach(objective, simulator, value);
                    (priors.into_iter().map(f64::from).collect(), Some(value))
                }
                None => (vec![1.0; actions.len()], None),
            };
            SearchNode {
                visits: 0,
                edges: Vec::new(),
                actions,
                classes,
                priors,
                value,
            }
        })
    })
}

/// What a descent does at the node it stands on: the root schedule's pick
/// where one has been dealt, else the first action class no edge covers yet,
/// else the edge the selection policy wants. `None` where this world offers
/// nothing the node has an edge for and nothing left to expand — a state the
/// search scores where it stands.
fn choose_step(
    node: &mut SearchNode,
    world: &[Option<&Action>],
    scheduled: Option<usize>,
    selection: &mut dyn Selection,
) -> Option<Step> {
    if let Some(index) = scheduled {
        let available = node.offer(world);
        return Some(
            node.edges
                .iter()
                .position(|edge| edge.index == index)
                .filter(|edge| available.contains(edge))
                .map_or(Step::Expand(index), Step::Descend),
        );
    }
    if let Some(index) = node.next_expansion(world) {
        return Some(Step::Expand(index));
    }
    node.select(world, selection).map(Step::Descend)
}

/// Credits every node and edge the descent walked with the value that came
/// back to it.
fn back_up(nodes: &mut HashMap<NodeKey, SearchNode>, path: &[(NodeKey, usize)], value: f64) {
    for &(key, edge) in path {
        let node = nodes.get_mut(&key).expect("a path node stands");
        node.visits += 1;
        let stats = &mut node.edges[edge].stats;
        stats.visits += 1;
        stats.total_value += value;
    }
}

/// What one descent walks with: the sources of worlds, values, and
/// randomness a rollout needs, gathered so the descent takes one argument for
/// them instead of five.
struct Walk<'a> {
    determinizer: &'a mut dyn Determinizer,
    objective: &'a dyn Objective,
    rollout: &'a mut dyn RolloutPolicy,
    net: Option<&'a dyn crate::net::Evaluate>,
    rng: &'a mut MegaRandom,
}

/// What one decision's simulations run up as they walk: the engine steps a
/// budget is measured against, and what the walks could not search or price.
#[derive(Debug, Default)]
struct Spent {
    steps: u64,
    degradations: Degradations,
}

/// What the root of one search answers with: the improved policy over the
/// actions the decision stood on, and how much evidence that improvement
/// rests on.
///
/// The two travel together because a label is only as good as its evidence
/// and the transform that writes the label discards the evidence. See
/// `value_spread`.
#[derive(Clone, Debug, Default)]
pub struct RootPolicy {
    /// The improved policy, one weight per action the root gave an edge.
    /// Empty where the search never expanded the root at all.
    pub actions: Vec<(Action, f64)>,
    /// What the search learned about each root edge, aligned with
    /// `actions`: the evidence the improved policy was read off, kept for
    /// whoever wants the table rather than the answer.
    pub candidates: Vec<RootCandidate>,
    /// `max Q − min Q` over the root's completed values, before the
    /// transform. `None` where there was no root to read it off.
    pub q_spread: Option<f64>,
    /// The search's own value of the decision's state: the root's
    /// `mixed_value`, in the objective's units. Where `z` is one number
    /// per fight, this is one per decision — it moves when a play moves the
    /// state, which is what a bootstrapped value target needs. `None` where
    /// there was no root to ask.
    pub root_value: Option<f64>,
}

/// One root edge as the search left it: the prior it started from, the
/// visits it was given, the mean of what came back, and the completed value
/// the root schedule ranked it by.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RootCandidate {
    pub prior: f64,
    pub visits: u64,
    pub mean_value: f64,
    pub completed_value: f64,
}

/// Information-set MCTS with per-rollout determinization. With `TrueState`
/// every determinization is the same world and this is plain UCT; a belief
/// determinizer changes the worlds and the keys, and nothing here.
#[derive(Clone, Debug)]
pub struct Mcts<S: Selection> {
    pub config: SearchConfig,
    selection: S,
    nodes: HashMap<NodeKey, SearchNode>,
    /// The decision searched most recently, so what the budget bought can be
    /// read back off the tree.
    root: Option<NodeKey>,
    /// Engine steps this tree has spent since it was built — inside the tree
    /// and past it, over every decision it has searched. The unit a per-run
    /// budget counts in, and the one thing a search spends that is both
    /// dominant and reproducible: a step is a step whatever the box is doing.
    steps: u64,
    /// What this tree has answered with something other than what it was
    /// asked for, since it was built. See [`Degradations`].
    degradations: Degradations,
    /// What stops the decision now being searched part-way through, where a
    /// caller has a budget to enforce. See [`Ceiling`].
    ceiling: Ceiling,
}

impl<S: Selection> Mcts<S> {
    #[must_use]
    pub fn new(config: SearchConfig, selection: S) -> Self {
        Self {
            config,
            selection,
            nodes: HashMap::new(),
            root: None,
            steps: 0,
            degradations: Degradations::default(),
            ceiling: Ceiling::default(),
        }
    }

    /// The ceiling the decisions from here on are searched under: a cap that
    /// stands *inside* one decision rather than only between decisions. See
    /// [`Ceiling`].
    pub const fn under(&mut self, ceiling: Ceiling) {
        self.ceiling = ceiling;
    }

    /// How many engine steps this tree has spent, all decisions since it was
    /// built. What [`Budget::steps`] is measured against.
    #[must_use]
    pub const fn steps_taken(&self) -> u64 {
        self.steps
    }

    /// What this tree could not search or could not price, all decisions
    /// since it was built.
    #[must_use]
    pub const fn degradations(&self) -> Degradations {
        self.degradations
    }

    /// What the last decision's budget actually bought: each root action
    /// class with the visits the search spent on it. Under UCT the budget
    /// spreads over everything on offer; under a root schedule it lands on
    /// the candidates the prior sampled, which is the whole point of one.
    #[must_use]
    pub fn root_visits(&self) -> Vec<(Action, u64)> {
        self.root
            .and_then(|key| self.nodes.get(&key))
            .map_or_else(Vec::new, |node| {
                node.edges
                    .iter()
                    .map(|edge| (edge.action.clone(), edge.stats.visits))
                    .collect()
            })
    }

    /// How many decision points the tree currently holds. The table persists
    /// across decisions, which is what re-rooting on the taken edge comes to
    /// when nodes are keyed by state.
    #[must_use]
    pub fn tree_size(&self) -> usize {
        self.nodes.len()
    }

    /// Searches the decision the determinizer is rooted on and answers with
    /// the action to take. Deterministic given the determinizer and `rng`.
    pub fn decide(
        &mut self,
        determinizer: &mut dyn Determinizer,
        objective: &dyn Objective,
        rollout: &mut dyn RolloutPolicy,
        rng: &mut MegaRandom,
    ) -> Action {
        self.decide_with_policy(determinizer, objective, rollout, None, rng)
            .0
    }

    /// The same search, answering also with the root's improved policy per
    /// action — a distribution, and what expert iteration trains toward. A
    /// net, when one is handed in, prices priors at every new node and stands
    /// in for the rollout as the leaf evaluator.
    pub fn decide_with_policy(
        &mut self,
        determinizer: &mut dyn Determinizer,
        objective: &dyn Objective,
        rollout: &mut dyn RolloutPolicy,
        net: Option<&dyn crate::net::Evaluate>,
        rng: &mut MegaRandom,
    ) -> (Action, RootPolicy) {
        self.decide_pinned(determinizer, objective, rollout, net, &[], rng)
    }

    /// The same search with `pinned` actions guaranteed a place among the
    /// root schedule's candidates, whatever their prior.
    ///
    /// A root schedule samples its candidates from the prior and never visits
    /// the rest, so an action the checkpoint dislikes has no searched value
    /// at all. Analysis of a recorded move needs that value more than any
    /// other, so the move is pinned: it keeps its own Gumbel key and is
    /// ranked like every other candidate, it simply cannot be left out of the
    /// draw. An action matching no class the root offers is ignored. Under
    /// UCT there is no draw and nothing changes.
    pub fn decide_pinned(
        &mut self,
        determinizer: &mut dyn Determinizer,
        objective: &dyn Objective,
        rollout: &mut dyn RolloutPolicy,
        net: Option<&dyn crate::net::Evaluate>,
        pinned: &[Action],
        rng: &mut MegaRandom,
    ) -> (Action, RootPolicy) {
        let root = crate::probe::timed(crate::probe::Phase::Sample, || determinizer.sample(0));
        let root_key = keyed(determinizer, &root);
        self.root = Some(root_key);
        // A root schedule samples its candidates from the priors, so the root
        // is priced before the first simulation rather than lazily on the way
        // past. Under UCT there is no schedule and nothing changes.
        let mut spent = Spent::default();
        let mut plan = self.selection.root_schedule().map(|schedule| {
            let node = node_for(
                &mut self.nodes,
                root_key,
                &root,
                objective,
                net,
                &mut spent.degradations,
            );
            let classes: Vec<String> = pinned.iter().map(action_class).collect();
            let pins: Vec<usize> = node
                .actions
                .iter()
                .enumerate()
                .filter(|(_, action)| classes.contains(&action_class(action)))
                .map(|(index, _)| index)
                .collect();
            GumbelPlan::new(schedule, &node.priors, self.config.iterations, &pins, rng)
        });
        for iteration in 0..self.config.iterations {
            if self.ceiling.reached(self.steps + spent.steps) {
                // The run's budget came down mid-decision. The plan answers
                // over the simulations it did get, which is what a Gumbel
                // plan is built to do at any budget, and the caller's next
                // decision sees the budget spent and downgrades.
                crate::probe::tally(crate::probe::Tally::DecisionCapped, 1);
                break;
            }
            crate::probe::tally(crate::probe::Tally::Iterations, 1);
            let mut simulator = crate::probe::timed(crate::probe::Phase::Sample, || {
                determinizer.sample(u64::from(iteration))
            });
            let mut path: Vec<(NodeKey, usize)> = Vec::new();
            let value = self.descend(
                Walk {
                    determinizer,
                    objective,
                    rollout,
                    net,
                    rng,
                },
                plan.as_mut(),
                // The root's key is carried into every descent, not only the
                // deterministic ones: a sampled world is consistent with the
                // observation by construction (the engine's belief tests pin
                // that a sample keys like its original), so the walk's first
                // node is this key in every world and re-hashing it per
                // iteration bought nothing.
                Some(root_key),
                &mut simulator,
                &mut path,
                &mut spent,
            );
            back_up(&mut self.nodes, &path, value);
        }
        self.steps += spent.steps;
        self.degradations.merge(spent.degradations);
        let policy = self.root_policy(&root_key);
        (self.pick_root(&root, &root_key, plan.as_ref(), rng), policy)
    }

    /// One rollout: descend the tree from the root the sampled world stands
    /// on, expand where the tree ends, and come back with the value the walk
    /// was worth. The path it walked is left in `path` for the back-up, and
    /// every engine step it spent is added to `spent`.
    ///
    /// A descent breaks with the objective's word wherever the walk stops
    /// having a decision to branch over: past the mode's horizon, on a state
    /// the mode plays rather than searches, where a world offers nothing the
    /// node has an edge for, or on an engine dead end. The last two, and a
    /// walk that closes a circle, are counted into `spent`.
    #[allow(
        clippy::too_many_lines,
        reason = "one loop, one arm per kind of step, in walk order"
    )]
    fn descend(
        &mut self,
        walk: Walk<'_>,
        mut plan: Option<&mut GumbelPlan>,
        mut carried_key: Option<NodeKey>,
        simulator: &mut Simulator,
        path: &mut Vec<(NodeKey, usize)>,
        spent: &mut Spent,
    ) -> f64 {
        let Walk {
            determinizer,
            objective,
            rollout,
            net,
            rng,
        } = walk;
        let deterministic = determinizer.deterministic_transitions();
        loop {
            crate::probe::tally(crate::probe::Tally::DescentSteps, 1);
            if simulator.state().terminal.is_some() || past_horizon(determinizer, simulator) {
                break peek(objective, simulator);
            }
            if path.len() >= DESCENT_CEILING {
                spent.degradations.deep += 1;
                break peek(objective, simulator);
            }
            if !path.is_empty() && played_out(determinizer, simulator) {
                // Inside the horizon but below the tree: this mode plays such
                // a state rather than branching over it (see
                // `Determinizer::plays_out`), so the line continues on the
                // rollout policy and comes back as this leaf's value.
                break leaf_value(
                    self.config.rollout_depth,
                    determinizer,
                    objective,
                    rollout,
                    net,
                    simulator,
                    rng,
                    spent,
                );
            }
            let key = carried_key
                .take()
                .unwrap_or_else(|| keyed(determinizer, simulator));
            if path.iter().any(|&(stood, _)| stood == key) {
                // A circle. Selection is a pure function of what the node
                // and its edges hold, and a descent updates neither until it
                // has finished and backed up — so a walk that arrives a
                // second time at a decision point it already stood on will
                // make the same choice it made the first time, and every
                // time after that, forever. The screens that close the loop
                // are ordinary ones: a rest site's smith opened and
                // cancelled, a shop entered and left, any back button an
                // observation key cannot tell from the screen it returned
                // to. Bounding the tree is the search's job and not the
                // content's, so the walk is scored where it stands and this
                // simulation ends here.
                spent.degradations.cycles += 1;
                crate::probe::tally(crate::probe::Tally::DescentCycle, 1);
                break peek(objective, simulator);
            }
            let node = node_for(
                &mut self.nodes,
                key,
                simulator,
                objective,
                net,
                &mut spent.degradations,
            );
            // This world's own offers, one per class the node knows.
            let world = node.world_actions(simulator.legal_actions());
            // The schedule drives the root and only the root: below it, the
            // tree is the selection policy's.
            let scheduled = if path.is_empty() {
                plan.as_mut().and_then(|plan| {
                    let offered = SearchNode::offered(&world);
                    plan.next(&offered, node)
                })
            } else {
                None
            };
            let Some(step) = choose_step(node, &world, scheduled, &mut self.selection) else {
                // This world offers nothing the node has an edge for and
                // nothing left to expand: the line stops where it stands.
                spent.degradations.dead_ends += 1;
                crate::probe::tally(crate::probe::Tally::RolloutDeadEnd, 1);
                break peek(objective, simulator);
            };
            let (edge, action) = match step {
                Step::Expand(index) => {
                    let action = world[index]
                        .expect("an expanded class is one this world offers")
                        .clone();
                    let prior = node.priors[index];
                    let edge = node.edges.len();
                    node.edges.push(SearchEdge {
                        index,
                        action: node.actions[index].clone(),
                        stats: EdgeStats::fresh(prior),
                        child: None,
                    });
                    node.edges[edge].stats.availability += 1;
                    path.push((key, edge));
                    spent.steps += 1;
                    if !tree_step(simulator, &action) {
                        // An engine dead end (see `leaf_value`): the world
                        // refused what it enumerated, so the opened edge
                        // learns exactly that — this line goes nowhere.
                        spent.degradations.dead_ends += 1;
                        crate::probe::tally(crate::probe::Tally::RolloutDeadEnd, 1);
                        break peek(objective, simulator);
                    }
                    break self
                        .table_priced_leaf(
                            determinizer,
                            objective,
                            net,
                            simulator,
                            &mut spent.degradations,
                        )
                        .unwrap_or_else(|| {
                            leaf_value(
                                self.config.rollout_depth,
                                determinizer,
                                objective,
                                rollout,
                                net,
                                simulator,
                                rng,
                                spent,
                            )
                        });
                }
                Step::Descend(edge) => {
                    let action = world[node.edges[edge].index]
                        .expect("an offered edge is one this world offers")
                        .clone();
                    (edge, action)
                }
            };
            path.push((key, edge));
            let cached_child = node.edges[edge].child;
            spent.steps += 1;
            if !tree_step(simulator, &action) {
                // The same engine dead end on an edge already open: a world
                // can refuse dynamically (the fault depends on relics held or
                // effects pending), so an edge other worlds walked fine is
                // still a leaf in this one.
                spent.degradations.dead_ends += 1;
                crate::probe::tally(crate::probe::Tally::RolloutDeadEnd, 1);
                break peek(objective, simulator);
            }
            carried_key = if !deterministic || simulator.state().terminal.is_some() {
                None
            } else if cached_child.is_some() {
                cached_child
            } else {
                let child = keyed(determinizer, simulator);
                self.nodes
                    .get_mut(&key)
                    .expect("the descended node stands")
                    .edges[edge]
                    .child = Some(child);
                Some(child)
            };
        }
    }

    /// Where a net guides, an expanded leaf is priced through the table: the
    /// child node is made (or found) now, and its value head's word is the
    /// leaf value. A position the table has seen — this fight's earlier
    /// decisions included — pays no forward pass at all, and a fresh one
    /// pays one (priors and value together) instead of one at the leaf and
    /// another when a later descent walks in. The value head reads the
    /// observation alone, so the number is what `state_value` says, held
    /// inside the objective's reach. `None` where no net guides or the state
    /// is past pricing — the caller falls back to [`leaf_value`].
    fn table_priced_leaf(
        &mut self,
        determinizer: &dyn Determinizer,
        objective: &dyn Objective,
        net: Option<&dyn crate::net::Evaluate>,
        simulator: &Simulator,
        degradations: &mut Degradations,
    ) -> Option<f64> {
        if net.is_none()
            || simulator.state().terminal.is_some()
            || past_horizon(determinizer, simulator)
        {
            return None;
        }
        let child_key = keyed(determinizer, simulator);
        node_for(
            &mut self.nodes,
            child_key,
            simulator,
            objective,
            net,
            degradations,
        )
        .value
    }

    /// The root's improved policy, as a distribution over the actions the
    /// search gave edges: what expert iteration trains toward.
    ///
    /// Under UCT that is the visit distribution. Under a root schedule it is
    /// not — sequential halving deliberately starves the candidates it drops,
    /// so raw visits would teach the net that a pruned action is worthless
    /// rather than merely unexamined. Completed-Q is the policy the halving
    /// was ranking by all along, so that is what is recorded.
    #[allow(
        clippy::cast_precision_loss,
        reason = "visit counts are far below f64 precision"
    )]
    fn root_policy(&self, root_key: &NodeKey) -> RootPolicy {
        let Some(node) = self.nodes.get(root_key) else {
            return RootPolicy::default();
        };
        let stats: Vec<EdgeStats> = node.edges.iter().map(|edge| edge.stats).collect();
        if stats.is_empty() {
            return RootPolicy::default();
        }
        let priors = edge_priors(&stats);
        let completed = completed_q(node.stats(), &stats, &priors);
        let q_spread = value_spread(&completed);
        let root_value = mixed_value(node.stats(), &stats, &priors);
        let candidates = stats
            .iter()
            .zip(&priors)
            .zip(&completed)
            .map(|((edge, prior), completed_value)| RootCandidate {
                prior: *prior,
                visits: edge.visits,
                mean_value: edge.mean_value(),
                completed_value: *completed_value,
            })
            .collect();
        let weights = if let Some(schedule) = self.selection.root_schedule() {
            improved_policy(schedule, node.stats(), &stats)
        } else {
            let total: u64 = stats.iter().map(|edge| edge.visits).sum();
            stats
                .iter()
                .map(|edge| {
                    if total == 0 {
                        0.0
                    } else {
                        edge.visits as f64 / total as f64
                    }
                })
                .collect()
        };
        RootPolicy {
            actions: node
                .edges
                .iter()
                .map(|edge| edge.action.clone())
                .zip(weights)
                .collect(),
            candidates,
            q_spread: Some(q_spread),
            root_value: Some(root_value),
        }
    }

    /// The root move: sampled over visit counts under the configured
    /// temperature, so repeated searches of one seed spread across the
    /// plausible lines instead of replaying the argmax.
    #[allow(
        clippy::cast_precision_loss,
        reason = "visit counts are far below f64 precision"
    )]
    fn pick_root(
        &self,
        root: &Simulator,
        root_key: &NodeKey,
        plan: Option<&GumbelPlan>,
        rng: &mut MegaRandom,
    ) -> Action {
        let Some(node) = self.nodes.get(root_key) else {
            return crate::policy::permitted_actions(root)
                .first()
                .cloned()
                .unwrap_or_else(|| {
                    // The root is the authoritative screen: an empty
                    // enumeration here is the engine's fault, reported with
                    // the screen named (see `Heuristic::choose`).
                    panic!(
                        "a live run offers an action; the engine enumerated none at {:?}",
                        root.decision()
                    )
                });
        };
        // A root schedule already answered: its surviving candidate is a
        // sample from the improved policy, and a temperature on top of it
        // would be a second helping of the same randomness. The legality
        // check is the same guard selection keeps — a node's action list is
        // cached from the first world that keyed it, and a schedule can
        // answer with a candidate no world ever expanded, so the real screen
        // gets the last word.
        let world = node.world_actions(root.legal_actions());
        if let Some(action) = plan
            .and_then(|plan| plan.answer(node))
            .and_then(|index| world[index])
        {
            return action.clone();
        }
        // The same last word for the visit-count answer: a node can be
        // reached under a key coarser than the screen (the reward set id),
        // so its visited edges may have been walked in worlds whose screens
        // offered what this one does not. An answer must be the real
        // screen's own.
        let visited: Vec<(&Action, u64)> = node
            .edges
            .iter()
            .filter(|edge| edge.stats.visits > 0)
            .filter_map(|edge| world[edge.index].map(|action| (action, edge.stats.visits)))
            .collect();
        if visited.is_empty() {
            return crate::policy::permitted_actions(root)
                .first()
                .cloned()
                .unwrap_or_else(|| {
                    // The root is the authoritative screen: an empty
                    // enumeration here is the engine's fault, reported with
                    // the screen named (see `Heuristic::choose`).
                    panic!(
                        "a live run offers an action; the engine enumerated none at {:?}",
                        root.decision()
                    )
                });
        }
        if self.config.temperature <= f64::EPSILON {
            return visited
                .iter()
                .max_by_key(|(_, visits)| *visits)
                .expect("visited is non-empty")
                .0
                .clone();
        }
        let weights: Vec<f64> = visited
            .iter()
            .map(|(_, visits)| (*visits as f64).powf(1.0 / self.config.temperature))
            .collect();
        let total: f64 = weights.iter().sum();
        let mut draw = rng.next_double() * total;
        for ((action, _), weight) in visited.iter().zip(&weights) {
            draw -= weight;
            if draw <= 0.0 {
                return (*action).clone();
            }
        }
        visited.last().expect("visited is non-empty").0.clone()
    }
}

/// The search as a policy: at each real decision it roots a `TrueState`
/// determinizer on the authentic state and searches. Plugged into the same
/// harness the dumb policies use, so a search run and a random run differ by
/// one flag.
pub struct TrueStateSearch<O: Objective> {
    mcts: Mcts<Box<dyn Selection>>,
    rollout: Box<dyn RolloutPolicy>,
    objective: O,
    /// See [`search_stream`].
    stream: Option<MegaRandom>,
}

impl<O: Objective> TrueStateSearch<O> {
    #[must_use]
    pub fn new(config: SearchConfig, objective: O) -> Self {
        Self::with_rollout(config, objective, Box::new(crate::policy::UniformRandom))
    }

    /// The same search rolling out on another policy: a stronger rollout is a
    /// stronger evaluator, at the price of its bias.
    #[must_use]
    pub fn with_rollout(
        config: SearchConfig,
        objective: O,
        rollout: Box<dyn RolloutPolicy>,
    ) -> Self {
        Self {
            mcts: Mcts::new(config, Box::new(Uct::default())),
            rollout,
            objective,
            stream: None,
        }
    }

    /// The same search under another selection policy — Gumbel, where a
    /// checkpoint's prior is worth spending a budget on.
    #[must_use]
    pub fn selecting(mut self, selection: Box<dyn Selection>) -> Self {
        self.mcts = Mcts::new(self.mcts.config, selection);
        self
    }

    #[must_use]
    pub fn tree_size(&self) -> usize {
        self.mcts.tree_size()
    }
}

impl<O: Objective> RolloutPolicy for TrueStateSearch<O> {
    fn choose(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> Action {
        let mut determinizer = TrueState::new(simulator.clone());
        self.mcts.decide(
            &mut determinizer,
            &self.objective,
            self.rollout.as_mut(),
            search_stream(&mut self.stream, rng),
        )
    }
}

/// A counterfactual branch a recording search plays out beside the line it
/// answers with. At a searched decision, an action whose completed value
/// stands within `margin` of the answer's while its prior is at most
/// `max_prior` is a line the head rates and the policy never plays — the
/// pool holds no such positions, so the head is extrapolating there and
/// nothing corrects it. The branch steps that action on a clone of the
/// real state, plays the fight to its end with the same search, and
/// records every decision on the way with the outcome the branch actually
/// reached, flagged [`counterfactual`](crate::training::Decision::counterfactual).
/// The pending line's own samples are untouched. At most `per_fight`
/// branches are taken per fight, and a branch that has not ended inside
/// `max_steps` decisions is dropped unrecorded.
///
/// A second rule branches on the **pass**: at a decision where the search
/// did not end the turn but could have, while some affordable attack
/// would lower an enemy's HP, the branch ends the turn instead. The state
/// the branch then records first — the next turn, reached with energy
/// unspent and attacks in hand — is the leaf a belief search prices when
/// it weighs passing, and one self-play never reaches. Each qualifying
/// decision branches with probability `pass_rate`, at most
/// `pass_per_fight` times per fight, so the branch points spread over the
/// fight instead of landing on its first turn.
#[derive(Clone, Copy, Debug)]
pub struct Counterfactual {
    pub margin: f64,
    pub max_prior: f64,
    pub per_fight: usize,
    pub max_steps: usize,
    pub pass_per_fight: usize,
    pub pass_rate: f64,
}

/// Which rule a counterfactual branch was taken under; the name each
/// branch row carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchKind {
    /// The low-prior rival whose value stood near the answer's.
    Rival,
    /// The turn ended with an attack still affordable.
    Pass,
}

impl BranchKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rival => "rival",
            Self::Pass => "pass",
        }
    }
}

/// The search as a fair player: at each in-combat decision it erases the
/// authentic state to a belief (model `combat-v0`) and searches possible
/// worlds keyed by what a player can see. Outside a fight — beyond the
/// model's scope — it plays its rollout policy directly. Plugged into the
/// same harness the other policies use, so a belief run and a `TrueState`
/// run differ by one flag; the search core they share is the same code.
pub struct BeliefSearch<O: Objective> {
    mcts: Mcts<Box<dyn Selection>>,
    rollout: Box<dyn RolloutPolicy>,
    objective: O,
    recorder: Option<Recorder>,
    /// Where set, a recording search also plays out counterfactual
    /// branches; see [`Counterfactual`].
    counterfactual: Option<Counterfactual>,
    /// Rival branches played out of the fight now pending.
    branches: usize,
    /// Pass branches played out of the fight now pending.
    pass_branches: usize,
    /// Whether the search is inside a branch — a branch never branches.
    in_branch: bool,
    /// The macro recorder, where this search was asked to clone its
    /// out-of-combat teacher. See [`BeliefSearch::recording_macro`].
    macro_recorder: Option<MacroRecorder>,
    /// What a macro sample's `z` is scored by — the act horizon, never the
    /// fight's. Carried rather than taken as `O` because `O` is the
    /// *combat* objective this search plays fights under, and the two are
    /// deliberately different quantities.
    boundary: crate::objective::ActBoundary,
    net: Option<std::sync::Arc<dyn crate::net::Evaluate>>,
    /// See [`search_stream`].
    stream: Option<MegaRandom>,
    /// The combat round the last decision stood in, and how many decisions
    /// have been answered inside it; see [`TURN_DECISION_CEILING`].
    turn: (u32, usize),
}

/// Sample recording for expert iteration: one pending sample per searched
/// decision, settled with the objective's score where the fight ends.
struct Recorder {
    /// Decisions of the fight still running, value not yet known.
    pending: Vec<crate::training::Decision>,
    complete: Vec<crate::training::Decision>,
    /// Which fight of the run the pending samples belong to: the provenance
    /// stamp that pairs a sample with its fight — and with the fight
    /// library's banked entry for the same run, which counts the same way.
    fight: usize,
}

impl Recorder {
    /// The fight the pending samples belong to ended on `value`.
    fn settle(&mut self, value: f64) {
        #[allow(clippy::cast_possible_truncation, reason = "values are small")]
        for mut sample in self.pending.drain(..) {
            sample.z = value as f32;
            self.complete.push(sample);
        }
        self.fight += 1;
    }
}

impl<O: Objective> BeliefSearch<O> {
    #[must_use]
    pub fn new(config: SearchConfig, objective: O) -> Self {
        Self::with_rollout(config, objective, Box::new(crate::policy::UniformRandom))
    }

    #[must_use]
    pub fn with_rollout(
        config: SearchConfig,
        objective: O,
        rollout: Box<dyn RolloutPolicy>,
    ) -> Self {
        Self {
            mcts: Mcts::new(config, Box::new(Uct::default())),
            rollout,
            objective,
            recorder: None,
            counterfactual: None,
            branches: 0,
            pass_branches: 0,
            in_branch: false,
            macro_recorder: None,
            boundary: crate::objective::ActBoundary::default(),
            net: None,
            stream: None,
            turn: (0, 0),
        }
    }

    /// The same search under another selection policy — Gumbel, where a
    /// checkpoint's prior is worth spending a budget on.
    #[must_use]
    pub fn selecting(mut self, selection: Box<dyn Selection>) -> Self {
        self.mcts = Mcts::new(self.mcts.config, selection);
        self
    }

    /// The same search guided by a checkpoint: priors at every new node, net
    /// value at every leaf. The evaluator may be the loaded net itself or an
    /// inference server's handle; the search cannot tell.
    #[must_use]
    pub fn with_net(mut self, net: std::sync::Arc<dyn crate::net::Evaluate>) -> Self {
        self.net = Some(net);
        self
    }

    /// Searches one in-fight decision and answers with the root's table
    /// rather than a move: what the search thought of every candidate,
    /// `pinned` ones included whatever their prior. Nothing is recorded and
    /// nothing is played. Refused outside a fight, where the belief model has
    /// nothing honest to sample.
    pub fn analyze(
        &mut self,
        simulator: &Simulator,
        pinned: &[Action],
        rng: &mut MegaRandom,
    ) -> Result<RootPolicy, sts2_engine::EngineError> {
        let stream = search_stream(&mut self.stream, rng);
        let analysis_seed = stream.next_u64();
        let mut determinizer = Belief::from_simulator(simulator, analysis_seed)?;
        let (_, policy) = self.mcts.decide_pinned(
            &mut determinizer,
            &self.objective,
            self.rollout.as_mut(),
            self.net
                .as_deref()
                .map(|net| net as &dyn crate::net::Evaluate),
            pinned,
            stream,
        );
        Ok(policy)
    }

    /// The same search, recording a [`Decision`](crate::training::Decision)
    /// at every decision it searches: the observation and canonical actions
    /// as the player saw them, the root visit distribution, and — once the
    /// fight ends — its value. Nothing is encoded here; the sink that
    /// writes the decision encodes it, or keeps it as it is.
    #[must_use]
    pub fn recording(mut self) -> Self {
        self.recorder = Some(Recorder {
            pending: Vec::new(),
            complete: Vec::new(),
            fight: 0,
        });
        self
    }

    /// The same recording search playing out counterfactual branches; see
    /// [`Counterfactual`]. Without [`recording`](Self::recording) a branch
    /// would record nothing, so it is never taken.
    #[must_use]
    pub const fn branching(mut self, counterfactual: Counterfactual) -> Self {
        self.counterfactual = Some(counterfactual);
        self
    }

    /// The same search, recording a *macro* training sample at every
    /// out-of-combat decision it hands to its rollout policy — the
    /// heuristic-imitation source of the macro track.
    ///
    /// The observation is encoded as in act mode, the target policy is a
    /// softmax of the rollout teacher's action scores, and the value target
    /// is the act's [`ActBoundary`](crate::objective::ActBoundary) score,
    /// settled at `act_over`. These samples bootstrap macro priors without
    /// an additional search.
    ///
    /// One recorder serves both sources — this is `MacroRecorder`, the
    /// same type [`ActSearch`] settles — and the shard header's `source`
    /// says which teacher answered
    /// ([`SOURCE_HEURISTIC`](crate::training::SOURCE_HEURISTIC)).
    ///
    /// A rollout policy with no distribution to teach
    /// (`macro_teacher_policy` answering `None`) records nothing: there is
    /// no target. The CLI refuses the configuration up front rather than
    /// letting a batch write an empty set.
    #[must_use]
    pub fn recording_macro(mut self) -> Self {
        self.macro_recorder = Some(MacroRecorder {
            pending: Vec::new(),
            complete: Vec::new(),
        });
        self
    }

    #[must_use]
    pub fn tree_size(&self) -> usize {
        self.mcts.tree_size()
    }

    /// How many engine steps this search has spent since it was built, over
    /// every decision it has answered. What a caller enforcing a
    /// [`Budget`] measures against.
    #[must_use]
    pub const fn steps_taken(&self) -> u64 {
        self.mcts.steps_taken()
    }

    /// What this search could not search or could not price, since it was
    /// built. The owner carries it out on its [`BudgetSpend`].
    #[must_use]
    pub const fn degradations(&self) -> Degradations {
        self.mcts.degradations()
    }

    /// The ceiling the decisions from here on are searched under — a cap that
    /// stands *inside* one decision rather than only between decisions, which
    /// is the difference between a bound and a hope. See [`Ceiling`].
    ///
    /// The budget itself lives with whoever owns the search, because what a
    /// downgraded decision falls back to is that owner's business: the act
    /// arm has a rollout policy to hand the screen to, and a combat resolver
    /// has the checkpoint's own answer.
    pub const fn under(&mut self, ceiling: Ceiling) {
        self.mcts.under(ceiling);
    }

    /// The word that whatever act this search was walking through has
    /// ended, with the state it ended on: settles any pending macro samples
    /// there, exactly as [`ActSearch::settle_act`] does and for the same
    /// reason. Idempotent when nothing is pending.
    pub fn settle_act(&mut self, simulator: &Simulator) {
        if let Some(recorder) = self.macro_recorder.as_mut()
            && !recorder.pending.is_empty()
        {
            recorder.settle(crate::objective::Objective::peek(&self.boundary, simulator));
        }
    }

    /// The word that whatever fight this search was deciding has ended, with
    /// the state it ended on: settles any pending recorded samples there — at
    /// the fight's own end, where the turn count still stands and the loot
    /// has not yet inflated the belt. Idempotent when nothing is pending.
    /// `choose` says it itself on any past-the-fight state it is handed; the
    /// composed act-mode policy must say it instead, because there the
    /// out-of-combat decisions no longer pass through this search.
    pub fn settle_fight(&mut self, simulator: &Simulator) {
        if let Some(recorder) = self.recorder.as_mut()
            && !recorder.pending.is_empty()
        {
            recorder.settle(self.objective.peek(simulator));
        }
        self.branches = 0;
        self.pass_branches = 0;
        self.turn = (0, 0);
    }

    /// Counts this decision into its turn and, past the ceiling, answers it
    /// by ending the turn where the turn can be ended. Unsearched and
    /// unrecorded: no tree stood behind the answer.
    fn turn_ceiling(&mut self, simulator: &Simulator) -> Option<Action> {
        let round = simulator.state().combat.as_ref()?.round;
        if self.turn.0 != round {
            self.turn = (round, 0);
        }
        self.turn.1 += 1;
        if self.turn.1 <= TURN_DECISION_CEILING {
            return None;
        }
        let end_turn = simulator
            .legal_actions()
            .iter()
            .find(|action| matches!(action, Action::EndTurn { .. }))?
            .clone();
        self.mcts.degradations.long_turns += 1;
        Some(end_turn)
    }

    /// Whether a decision qualifies for a pass branch: the turn can be
    /// ended, and some affordable attack, stepped on a clone of the real
    /// state, lowers the enemies' HP — so passing leaves damage on the
    /// table rather than hitting block, a shell, or nothing.
    fn pass_qualifies(simulator: &Simulator) -> bool {
        let legal = simulator.legal_actions();
        if !legal
            .iter()
            .any(|action| matches!(action, Action::EndTurn { .. }))
        {
            return false;
        }
        let before = enemy_hp(simulator);
        legal.iter().any(|action| {
            let Action::PlayCard { card, .. } = action else {
                return false;
            };
            if simulator.content().card_type_of(&card.fingerprint)
                != Some(sts2_engine::CardType::Attack)
            {
                return false;
            }
            let mut world = simulator.clone();
            world.step_quietly(action).is_ok() && enemy_hp(&world) < before
        })
    }

    /// The counterfactual rival at a decision, where the configuration
    /// asks for one: the best-valued action class whose prior the policy
    /// all but refused and whose completed value stands within the margin
    /// of the answer's. `None` where nothing qualifies, or where the answer
    /// itself is not among the root's candidates.
    fn rival<'a>(
        counterfactual: Counterfactual,
        policy: &'a RootPolicy,
        chosen: &Action,
    ) -> Option<&'a Action> {
        let chosen_class = action_class(chosen);
        let rows = || policy.actions.iter().zip(&policy.candidates);
        let answer = rows()
            .find(|((action, _), _)| action_class(action) == chosen_class)
            .map(|(_, candidate)| candidate.completed_value)?;
        rows()
            .filter(|((action, _), candidate)| {
                candidate.visits > 0
                    && candidate.prior <= counterfactual.max_prior
                    && candidate.completed_value >= answer - counterfactual.margin
                    && action_class(action) != chosen_class
            })
            .max_by(|(_, left), (_, right)| left.completed_value.total_cmp(&right.completed_value))
            .map(|((action, _), _)| action)
    }

    /// Plays `rival` out from a clone of `simulator` to the fight's end and
    /// records the branch's decisions with that outcome. The pending line's
    /// own samples are parked for the duration and restored after.
    fn branch(
        &mut self,
        simulator: &Simulator,
        rival: &Action,
        kind: BranchKind,
        counterfactual: Counterfactual,
        rng: &mut MegaRandom,
    ) {
        let rival_class = action_class(rival);
        let Some(step) = simulator
            .legal_actions()
            .iter()
            .find(|offered| action_class(offered) == rival_class)
            .cloned()
        else {
            return;
        };
        let mut world = simulator.clone();
        if world.step_quietly(&step).is_err() {
            return;
        }
        let parked = {
            let recorder = self.recorder.as_mut().expect("a branch is recorded");
            std::mem::take(&mut recorder.pending)
        };
        self.in_branch = true;
        let mut steps = 0;
        while world.state().terminal.is_none()
            && !fight_over(&world)
            && steps < counterfactual.max_steps
        {
            let action = RolloutPolicy::choose(self, &world, rng);
            if world.step_quietly(&action).is_err() {
                break;
            }
            steps += 1;
        }
        self.in_branch = false;
        match kind {
            BranchKind::Rival => self.branches += 1,
            BranchKind::Pass => self.pass_branches += 1,
        }
        let ended = world.state().terminal.is_some() || fight_over(&world);
        let value = self.objective.peek(&world);
        let recorder = self.recorder.as_mut().expect("a branch is recorded");
        if ended {
            crate::probe::tally(crate::probe::Tally::BranchesRecorded, 1);
            #[allow(clippy::cast_possible_truncation, reason = "values are small")]
            for mut sample in recorder.pending.drain(..) {
                sample.z = value as f32;
                sample.counterfactual = true;
                sample.counterfactual_kind = Some(kind.as_str().to_owned());
                recorder.complete.push(sample);
            }
        } else {
            crate::probe::tally(crate::probe::Tally::BranchesDropped, 1);
            recorder.pending.clear();
        }
        recorder.pending = parked;
    }

    /// Records one macro decision the rollout policy is about to answer: the
    /// teacher's own distribution over the canonical actions, banked pending
    /// the act's horizon score.
    ///
    /// A teacher with no distribution to give records nothing — see
    /// [`BeliefSearch::recording_macro`].
    fn record_macro(&mut self, simulator: &Simulator) {
        let canonical = canonical_actions(&crate::policy::permitted_actions(simulator));
        let Some(teacher) = self.rollout.macro_teacher_policy(simulator, &canonical) else {
            return;
        };
        // A teacher's own distribution, not a search's: there is no tree
        // here, so no completed values to have spread and no root to value.
        let policy = RootPolicy {
            actions: canonical.into_iter().zip(teacher).collect(),
            candidates: Vec::new(),
            q_spread: None,
            root_value: None,
        };
        let Some(recorder) = self.macro_recorder.as_mut() else {
            return;
        };
        let decision = record_decision(
            simulator,
            &policy,
            // A macro decision belongs to no fight, and the act index is
            // what stands in the fight stamp's place.
            None,
            0,
            simulator.state().run.as_ref().map(|run| run.current_act),
        );
        recorder.pending.push(decision);
    }
}

/// One searched decision as it was seen: the observation, the canonical
/// actions, and the search's improved policy aligned onto them. `z` is left
/// at zero for whoever settles the horizon.
///
/// One function for both recorders because a decision is a decision: the
/// combat recorder stamps the fight it was searched in, the macro recorder
/// stamps the act, and nothing else about the shape differs.
/// The living enemies' HP, summed.
fn enemy_hp(simulator: &Simulator) -> i32 {
    simulator.state().combat.as_ref().map_or(0, |combat| {
        combat
            .creatures
            .iter()
            .filter(|creature| creature.id != combat.player.creature_id)
            .map(|creature| creature.current_hp.max(0))
            .sum()
    })
}

fn record_decision(
    simulator: &Simulator,
    policy: &RootPolicy,
    encounter: Option<sts2_core::ModelId>,
    fight: usize,
    act: Option<usize>,
) -> crate::training::Decision {
    let observation = simulator.agent_observation();
    let canonical = canonical_actions(&crate::policy::permitted_actions(simulator));
    #[allow(clippy::cast_possible_truncation, reason = "a policy is bounded")]
    let mut pi: Vec<f32> = canonical
        .iter()
        .map(|action| {
            policy
                .actions
                .iter()
                .find(|(edge, _)| edge == action)
                .map_or(0.0, |(_, weight)| *weight as f32)
        })
        .collect();
    let total: f32 = pi.iter().sum();
    if total > 0.0 {
        for weight in &mut pi {
            *weight /= total;
        }
    } else {
        // A root nobody visited: uniform is the honest answer.
        #[allow(clippy::cast_precision_loss, reason = "counts are small")]
        let uniform = 1.0 / canonical.len() as f32;
        pi.fill(uniform);
    }
    crate::training::Decision {
        observation,
        // The search enumerates single actions and nothing else — a plan is
        // a policy-layer decision, and the tree never stands at a screen that
        // offers one.
        actions: canonical
            .into_iter()
            .map(crate::plan::ActionPlan::Single)
            .collect(),
        pi,
        z: 0.0,
        encounter,
        // Stamped by the sink, which knows the batch's run order.
        run: 0,
        fight,
        act,
        // A searched decision teaches its whole improved policy and names no
        // single action taken, so it carries none of the trajectory keys —
        // which is what keeps its line byte-identical to format 3's.
        chosen: None,
        logp: None,
        value: None,
        reward: None,
        done: false,
        bootstrap: None,
        degraded: false,
        counterfactual: false,
        counterfactual_kind: None,
        exploration_epsilon: None,
        #[allow(clippy::cast_possible_truncation, reason = "a value is bounded")]
        q_spread: policy.q_spread.map(|spread| spread as f32),
        #[allow(clippy::cast_possible_truncation, reason = "a value is bounded")]
        root_value: policy.root_value.map(|value| value as f32),
    }
}

impl<O: Objective> RolloutPolicy for BeliefSearch<O> {
    fn choose(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> Action {
        if fight_over(simulator) {
            // Past the fight — between fights, or on the victory screen the
            // engine keeps the combat alive behind: whatever fight was
            // pending is over, and this state is its horizon. Score it here
            // and hand the decision back to the rollout, which reads the
            // real screen instead of a sampled one.
            self.settle_fight(simulator);
            if crate::env::act_over(simulator) {
                // The act's horizon: the macro samples of the act settle on
                // exactly this state, before the transition is stepped and
                // the next act's map is read. Nothing is recorded here —
                // the transition's `AdvanceAct` is forced and the boss's
                // last reward claims are act mode's own unsearched rim, so
                // the two sources record the same decisions.
                self.settle_act(simulator);
                return self.rollout.choose(simulator, rng);
            }
            if self.macro_recorder.is_some() {
                self.record_macro(simulator);
            }
            return self.rollout.choose(simulator, rng);
        }
        if let Some(end_turn) = self.turn_ceiling(simulator) {
            return end_turn;
        }
        let _timing = crate::probe::decision(crate::probe::Kind::Combat);
        // A per-decision analysis seed off the search's own stream: the walk
        // stays reproducible from (run seed, analysis seed, configuration),
        // no two decisions share their sampled worlds, and the harness
        // stream is left to the rollout — see `search_stream`.
        let stream = search_stream(&mut self.stream, rng);
        let analysis_seed = stream.next_u64();
        let mut determinizer = crate::probe::timed(crate::probe::Phase::Erase, || {
            Belief::from_simulator(simulator, analysis_seed).expect("a fight erases")
        });
        let (action, policy) = self.mcts.decide_with_policy(
            &mut determinizer,
            &self.objective,
            self.rollout.as_mut(),
            self.net
                .as_deref()
                .map(|net| net as &dyn crate::net::Evaluate),
            stream,
        );
        if let Some(recorder) = self.recorder.as_mut() {
            let decision = record_decision(
                simulator,
                &policy,
                // The fight's own identity, not the room shell around it:
                // two rooms can spawn the same multiset of enemies.
                simulator
                    .state()
                    .combat
                    .as_ref()
                    .map(|combat| combat.encounter_model.clone()),
                recorder.fight,
                None,
            );
            recorder.pending.push(decision);
        }
        if let Some(counterfactual) = self.counterfactual
            && self.recorder.is_some()
            && !self.in_branch
            && self.branches < counterfactual.per_fight
            && let Some(rival) = Self::rival(counterfactual, &policy, &action).cloned()
        {
            self.branch(simulator, &rival, BranchKind::Rival, counterfactual, rng);
        }
        if let Some(counterfactual) = self.counterfactual
            && self.recorder.is_some()
            && !self.in_branch
            && self.pass_branches < counterfactual.pass_per_fight
            && !matches!(action, Action::EndTurn { .. })
            && rng.next_double() < counterfactual.pass_rate
            && Self::pass_qualifies(simulator)
            && let Some(pass) = simulator
                .legal_actions()
                .iter()
                .find(|offered| matches!(offered, Action::EndTurn { .. }))
                .cloned()
        {
            self.branch(simulator, &pass, BranchKind::Pass, counterfactual, rng);
        }
        action
    }

    fn run_ended(&mut self, simulator: &Simulator) {
        // A run that ended inside a fight — a death, a step cap — settles the
        // pending samples on the state it ended with.
        self.settle_fight(simulator);
        // And a run that ended inside its act settles that act's macro
        // samples the same way: the set carries its losses rather than
        // quietly dropping them.
        self.settle_act(simulator);
    }

    fn drain_decisions(&mut self) -> Vec<crate::training::Decision> {
        self.recorder
            .as_mut()
            .map_or_else(Vec::new, |recorder| std::mem::take(&mut recorder.complete))
    }

    fn drain_macro_decisions(&mut self) -> Vec<crate::training::Decision> {
        self.macro_recorder
            .as_mut()
            .map_or_else(Vec::new, |recorder| std::mem::take(&mut recorder.complete))
    }
}

/// Combines combat belief search with act-scoped search of out-of-combat
/// decisions under the [`ActBoundary`](crate::objective::ActBoundary) objective.
/// At [`act_over`](crate::env::act_over), the rollout policy handles the
/// transition and remaining boss rewards because they are outside the
/// act belief model's horizon.
///
/// Inside act rollouts, fights are played by the rollout policy under the
/// depth cap, never searched by the act tree. The boundary is enforced by
/// [`Determinizer::plays_out`].
/// With [`ActSearch::with_net`],
/// [`ActEvaluator`](crate::net::ActEvaluator) instead prices macro leaves
/// directly and translates combat values at fight-entry leaves into
/// act-boundary units. Both kinds of leaf must use the same value scale.
///
/// Combat samples and searched macro samples have different horizons and
/// value targets, so they are written to separate shard directories.
pub struct ActSearch<O: Objective> {
    mcts: Mcts<Box<dyn Selection>>,
    rollout: Box<dyn RolloutPolicy>,
    objective: O,
    combat: BeliefSearch<crate::objective::CombatStrength>,
    recorder: Option<MacroRecorder>,
    /// The act tree's own evaluator, where one was handed in: priors and
    /// leaf values at macro nodes from a macro checkpoint, fight-entry
    /// leaves from the combat checkpoint translated into act-boundary
    /// units. See [`ActEvaluator`](crate::net::ActEvaluator) and
    /// [`ActSearch::with_net`].
    net: Option<std::sync::Arc<dyn crate::net::Evaluate>>,
    budget: Budget,
    /// When this run's clock started — the first decision this policy was
    /// asked for, since a policy is built per run.
    started: Option<std::time::Instant>,
    searched: usize,
    downgraded: usize,
    /// See [`search_stream`].
    stream: Option<MegaRandom>,
}

/// A soft ceiling on what one run may spend inside one of its searches.
///
/// Soft in the one way that matters: a run out of budget is neither killed
/// nor truncated. Its searched decisions are answered by something cheaper
/// from there on — the rollout policy for the act arm, the combat
/// checkpoint's own best action for a rollout's resolver
/// ([`SearchedCombat`](crate::actor::SearchedCombat)) — and the run walks to
/// its own end, still recording, still counting in the batch's outcome
/// tally. The straggler tail this exists for is a *latency* problem, and a
/// downgraded tail costs the batch nothing but the strength of one run's late
/// play. A batch's wall time is its slowest run, and one pathological fight
/// is enough to be it.
///
/// Two ceilings, because they answer different questions:
///
/// - [`steps`](Budget::steps) counts the search's own engine steps and is
///   **reproducible**: a run under a step budget is still a function of its
///   seed pair alone, so a batch at any `--jobs` plays the same games and
///   emits byte-identical shards. This is the ceiling an emission batch
///   wants, and the only one a batch that writes training data may have.
/// - [`seconds`](Budget::seconds) counts wall-clock time from the run's
///   first decision and is **not** reproducible — what it downgrades
///   depends on what else the box was doing. It is the honest circuit
///   breaker for an eval batch under a hard timeout, and nothing that
///   writes training data may use it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Budget {
    /// Engine steps the search may spend over one run.
    pub steps: Option<u64>,
    /// Wall-clock seconds one run may spend before its macro decisions
    /// downgrade.
    pub seconds: Option<f64>,
}

impl Budget {
    /// Whether this budget constrains anything at all.
    #[must_use]
    pub const fn is_off(self) -> bool {
        self.steps.is_none() && self.seconds.is_none()
    }

    /// This budget as one decision of a run whose clock started at `started`
    /// sees it. A wall cap that cannot arrive without an answer from the
    /// search is no cap, so the seconds become a deadline the decision's own
    /// simulation loop reads. See [`Ceiling`].
    #[must_use]
    pub fn ceiling(self, started: std::time::Instant) -> Ceiling {
        Ceiling {
            steps: self.steps,
            deadline: self
                .seconds
                .and_then(|seconds| {
                    seconds.is_finite().then(|| {
                        started.checked_add(std::time::Duration::from_secs_f64(seconds.max(0.0)))
                    })
                })
                .flatten(),
        }
    }
}

/// A [`Budget`] resolved against the run's clock, as one decision sees it.
///
/// The distinction it exists for: a budget read only *between* decisions is
/// not a cap at all, because a single decision that never ends is never
/// asked about. That is not hypothetical — an act tree whose descent walked
/// a screen loop ran one decision for hours under a ninety-second wall cap,
/// and the run never printed a line. A ceiling is read between the
/// simulations of one decision, so the longest a cap can be overshot is one
/// simulation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ceiling {
    /// Engine steps the tree may have spent in total before the decision
    /// stops starting simulations.
    steps: Option<u64>,
    /// The instant past which the decision starts no more simulations.
    deadline: Option<std::time::Instant>,
}

impl Ceiling {
    /// Whether a decision that has brought the tree's step total to `steps`
    /// should stop starting simulations.
    #[must_use]
    fn reached(self, steps: u64) -> bool {
        self.steps.is_some_and(|cap| steps >= cap)
            || self
                .deadline
                .is_some_and(|deadline| std::time::Instant::now() >= deadline)
    }
}

/// What one run's search came to, for the run's own summary line.
#[derive(Clone, Copy, Debug, Default)]
pub struct BudgetSpend {
    /// Decisions the search actually searched.
    pub searched: usize,
    /// Decisions a budget handed to the cheaper answer instead.
    pub downgraded: usize,
    /// Engine steps the search spent.
    pub steps: u64,
    /// What the run answered with something other than what it asked for.
    pub degradations: Degradations,
}

impl BudgetSpend {
    /// Adds `other` in, for a caller summing a batch or a policy summing the
    /// searches it owns.
    pub const fn merge(&mut self, other: Self) {
        self.searched += other.searched;
        self.downgraded += other.downgraded;
        self.steps += other.steps;
        self.degradations.merge(other.degradations);
    }
}

/// What a run answered with something other than what it was asked for,
/// counted always rather than under [`crate::probe`]: each of these is a
/// correctness signal, and a signal nobody can see without an environment
/// variable is one nobody sees.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Degradations {
    /// Decisions priced uniform because they offered more actions than the
    /// checkpoint's axis.
    pub over_axis: u64,
    /// Decisions priced off a checkpoint answer that was not a number.
    pub non_finite: u64,
    /// Decisions the evaluator holds no checkpoint for.
    pub out_of_scope: u64,
    /// Lines that stopped where a world offered nothing, or refused what it
    /// had offered — inside the tree and past it.
    pub dead_ends: u64,
    /// Descents that stopped on a decision point already on their own path.
    pub cycles: u64,
    /// Descents that stopped at the depth ceiling: a line inside one turn
    /// that never repeats a state and never runs out.
    pub deep: u64,
    /// Turns ended for the player at the per-turn decision ceiling.
    pub long_turns: u64,
}

impl Degradations {
    /// Counts one pricing under the cause that degraded it, and nothing for
    /// a pricing the checkpoint made itself.
    pub const fn priced(&mut self, cause: Option<crate::net::Degraded>) {
        match cause {
            Some(crate::net::Degraded::OverAxis) => self.over_axis += 1,
            Some(crate::net::Degraded::NonFinite) => self.non_finite += 1,
            Some(crate::net::Degraded::OutOfScope) => self.out_of_scope += 1,
            None => {}
        }
    }

    /// Adds `other` in.
    pub const fn merge(&mut self, other: Self) {
        self.over_axis += other.over_axis;
        self.non_finite += other.non_finite;
        self.out_of_scope += other.out_of_scope;
        self.dead_ends += other.dead_ends;
        self.cycles += other.cycles;
        self.deep += other.deep;
        self.long_turns += other.long_turns;
    }

    /// Decisions answered with priors no checkpoint produced, all causes.
    #[must_use]
    pub const fn priced_total(&self) -> u64 {
        self.over_axis + self.non_finite + self.out_of_scope
    }

    /// Whether anything at all was counted.
    #[must_use]
    pub const fn any(&self) -> bool {
        self.priced_total() + self.dead_ends + self.cycles + self.deep + self.long_turns > 0
    }
}

/// The counts that are not zero, named; `none` where none of them is. One
/// rendering for the episode line, the batch report and a match verdict, so
/// the three read the same.
impl std::fmt::Display for Degradations {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let counts = [
            ("over axis", self.over_axis),
            ("non-finite", self.non_finite),
            ("out of scope", self.out_of_scope),
            ("dead ends", self.dead_ends),
            ("circles", self.cycles),
            ("deep descents", self.deep),
            ("long turns", self.long_turns),
        ];
        let mut wrote = false;
        for (name, count) in counts {
            if count == 0 {
                continue;
            }
            if wrote {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{name} {count}")?;
            wrote = true;
        }
        if wrote {
            Ok(())
        } else {
            formatter.write_str("none")
        }
    }
}

/// Macro-sample recording for expert iteration on the *run* net: one pending
/// sample per searched out-of-combat decision, settled with the act-boundary
/// objective's score where the act ends.
///
/// The shape is [`Recorder`]'s, one rung up the horizon ladder: there a
/// fight's samples share the fight's exit score, here an act's macro samples
/// share the act's. Deaths settle the same way and for the same reason — a
/// run that ends mid-act ends its samples on the state it died in, so the
/// set carries its losses instead of quietly dropping them.
struct MacroRecorder {
    /// Decisions of the act still being played, value not yet known.
    pending: Vec<crate::training::Decision>,
    complete: Vec<crate::training::Decision>,
}

impl MacroRecorder {
    /// The act the pending samples belong to ended on `value`.
    fn settle(&mut self, value: f64) {
        #[allow(clippy::cast_possible_truncation, reason = "values are small")]
        for mut sample in self.pending.drain(..) {
            sample.z = value as f32;
            self.complete.push(sample);
        }
    }
}

impl<O: Objective> ActSearch<O> {
    /// Composes the act arm over `combat`, the in-fight search built exactly
    /// as combat-only belief mode builds it (selection, net, recorder). The
    /// act arm searches under its own `config` — its rollouts are priced in
    /// rooms rather than turns — and rolls out on `rollout`, which also
    /// answers the real screens at the act horizon.
    #[must_use]
    pub fn new(
        config: SearchConfig,
        objective: O,
        rollout: Box<dyn RolloutPolicy>,
        combat: BeliefSearch<crate::objective::CombatStrength>,
    ) -> Self {
        Self {
            mcts: Mcts::new(config, Box::new(Uct::default())),
            rollout,
            objective,
            combat,
            recorder: None,
            net: None,
            budget: Budget::default(),
            started: None,
            searched: 0,
            downgraded: 0,
            stream: None,
        }
    }

    /// The same search under another selection policy for the act tree.
    #[must_use]
    pub fn selecting(mut self, selection: Box<dyn Selection>) -> Self {
        self.mcts = Mcts::new(self.mcts.config, selection);
        self
    }

    /// Guides the act tree with priors at macro nodes and values at leaves.
    /// The evaluator should be an [`ActEvaluator`](crate::net::ActEvaluator),
    /// which prices macro states directly and translates combat values at
    /// fight entries into act-boundary units. Non-terminal leaves use this
    /// evaluator instead of playing out the rollout policy.
    #[must_use]
    pub fn with_net(mut self, net: std::sync::Arc<dyn crate::net::Evaluate>) -> Self {
        self.net = Some(net);
        self
    }

    /// The same search under a per-run soft budget: past it the macro
    /// decisions are the rollout policy's, and the run walks on. See
    /// [`Budget`].
    #[must_use]
    pub const fn within(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// The same search, recording a macro
    /// [`Decision`](crate::training::Decision) at every out-of-combat
    /// decision it searches: the observation and canonical actions as
    /// seen, the root's improved policy, and — once the act ends — the
    /// act-boundary score. Decisions a budget downgraded record nothing:
    /// there is no search to record.
    #[must_use]
    pub fn recording_macro(mut self) -> Self {
        self.recorder = Some(MacroRecorder {
            pending: Vec::new(),
            complete: Vec::new(),
        });
        self
    }

    /// How many decision points the act tree holds.
    #[must_use]
    pub fn tree_size(&self) -> usize {
        self.mcts.tree_size()
    }

    /// The word that whatever act this search was deciding has ended, with
    /// the state it ended on: settles any pending macro samples there — at
    /// the act's own horizon, where `act_over` first stands, before the next
    /// act's map is generated or read. Idempotent when nothing is pending.
    pub fn settle_act(&mut self, simulator: &Simulator) {
        if let Some(recorder) = self.recorder.as_mut()
            && !recorder.pending.is_empty()
        {
            recorder.settle(self.objective.peek(simulator));
        }
    }

    /// Whether this run has spent its macro search budget. Starts the run's
    /// clock on the first ask, which is the first decision the harness makes
    /// — a policy is built per run, so that is the run's own start.
    fn over_budget(&mut self) -> bool {
        let started = *self.started.get_or_insert_with(std::time::Instant::now);
        self.budget
            .steps
            .is_some_and(|cap| self.mcts.steps_taken() >= cap)
            || self
                .budget
                .seconds
                .is_some_and(|cap| started.elapsed().as_secs_f64() >= cap)
    }
}

impl<O: Objective> RolloutPolicy for ActSearch<O> {
    fn choose(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> Action {
        if !fight_over(simulator) {
            // Inside a fight the combat search owns the decision, exactly as
            // it does in combat-only belief mode.
            return self.combat.choose(simulator, rng);
        }
        // The first past-the-fight state settles the combat recorder's
        // pending fight, which no longer sees these states through its own
        // `choose`. Idempotent between fights.
        self.combat.settle_fight(simulator);
        if crate::env::act_over(simulator) {
            // The act's horizon: nothing of the act is left to sample, so
            // the rollout policy answers the real screen directly — and the
            // act's macro samples settle on exactly this state, before the
            // transition is stepped and the next act's map is read.
            self.settle_act(simulator);
            return self.rollout.choose(simulator, rng);
        }
        if self.over_budget() {
            // Out of budget: the decision is the rollout policy's, and the
            // run walks on to its own end. Nothing is recorded — there is no
            // search here to record.
            self.downgraded += 1;
            crate::probe::tally(crate::probe::Tally::Downgrades, 1);
            return self.rollout.choose(simulator, rng);
        }
        self.searched += 1;
        // The budget check above started the run's clock; the same budget
        // now stands over this one decision's simulations, so a run that
        // walks past its cap mid-decision stops there rather than at the
        // next decision that never comes.
        let started = self.started.expect("the budget check started the clock");
        self.mcts.under(self.budget.ceiling(started));
        let _timing = crate::probe::decision(crate::probe::Kind::Act);
        // A per-decision analysis seed off the act tree's own stream, the
        // same discipline as the combat arm: reproducible from (run seed,
        // analysis seed, configuration), no two decisions sharing worlds,
        // and the harness stream left to the rollout — see `search_stream`.
        let stream = search_stream(&mut self.stream, rng);
        let analysis_seed = stream.next_u64();
        let mut determinizer = crate::probe::timed(crate::probe::Phase::Erase, || {
            crate::env::ActBelief::from_simulator(simulator, analysis_seed)
                .expect("an act in progress erases")
        });
        let (action, policy) = self.mcts.decide_with_policy(
            &mut determinizer,
            &self.objective,
            self.rollout.as_mut(),
            self.net
                .as_deref()
                .map(|net| net as &dyn crate::net::Evaluate),
            stream,
        );
        if let Some(recorder) = self.recorder.as_mut() {
            let decision = record_decision(
                simulator,
                &policy,
                // A macro decision belongs to no fight: the encounter stamp
                // is what a fight would have written there, and the act
                // index is what stands in its place.
                None,
                0,
                simulator.state().run.as_ref().map(|run| run.current_act),
            );
            recorder.pending.push(decision);
        }
        action
    }

    fn run_ended(&mut self, simulator: &Simulator) {
        self.combat.run_ended(simulator);
        // A run that ended inside its act — a death, a step cap — settles
        // its macro samples on the state it ended with.
        self.settle_act(simulator);
    }

    fn drain_decisions(&mut self) -> Vec<crate::training::Decision> {
        self.combat.drain_decisions()
    }

    fn drain_macro_decisions(&mut self) -> Vec<crate::training::Decision> {
        self.recorder
            .as_mut()
            .map_or_else(Vec::new, |recorder| std::mem::take(&mut recorder.complete))
    }

    fn budget_spent(&self) -> Option<BudgetSpend> {
        // Both trees the act arm owns: the act tree itself, and the combat
        // search it hands its live fights to.
        let mut degradations = self.mcts.degradations();
        degradations.merge(self.combat.degradations());
        Some(BudgetSpend {
            searched: self.searched,
            downgraded: self.downgraded,
            steps: self.mcts.steps_taken(),
            degradations,
        })
    }
}
