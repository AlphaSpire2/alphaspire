//! What a trajectory is worth.
//!
//! Validation mode optimizes coverage, not strength. Strength arrived with
//! the belief phase as [`CombatStrength`], behind the same trait;
//! [`ActBoundary`] widens it to the act horizon for `act-v0.5` search.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use sts2_engine::{RunResult, Simulator};

/// Scores the state a rollout ended on. `&mut` because a coverage objective
/// carries batch-wide memory: novelty pays for content nobody has exercised
/// yet, and paying consumes it.
pub trait Objective {
    /// Scores and settles: novelty found here is paid and consumed.
    fn reward(&mut self, simulator: &Simulator) -> f64;
    /// Scores without settling: what a search rollout asks, thousands of
    /// times, without draining the batch table it is steering by.
    fn peek(&self, simulator: &Simulator) -> f64;
    /// The range this objective can pay for any continuation of the state:
    /// the band a leaf evaluator's estimate is held inside before the search
    /// reads it. A number outside it is not an optimistic or pessimistic
    /// read of the position but one the objective cannot produce, and a
    /// maximising search hunts for exactly those. The default is unbounded,
    /// for an objective whose reach cannot be stated from the state alone.
    fn attainable(&self, simulator: &Simulator) -> std::ops::RangeInclusive<f64> {
        let _ = simulator;
        f64::NEG_INFINITY..=f64::INFINITY
    }
}

/// The validation objective: a weighted sum of terminal result, floor depth,
/// and novelty against a batch-wide table of content exercised. A
/// novelty-seeking search turns into frontier exploration — it deliberately
/// fights the elite nobody has fought and buys the relic nobody has bought.
#[derive(Clone, Debug)]
pub struct Coverage {
    pub win_weight: f64,
    pub floor_weight: f64,
    pub novelty_weight: f64,
    /// Content already paid for, shared across every rollout of a batch.
    seen: BTreeSet<String>,
}

impl Default for Coverage {
    fn default() -> Self {
        Self {
            win_weight: 10.0,
            floor_weight: 0.1,
            novelty_weight: 1.0,
            seen: BTreeSet::new(),
        }
    }
}

impl Coverage {
    /// Everything this state exercised that the batch had not seen yet:
    /// rooms walked (by resolved model), relics held, and deck cards carried.
    fn fresh(&self, simulator: &Simulator) -> Vec<String> {
        let state = simulator.state();
        let mut touched: Vec<String> = Vec::new();
        if let Some(run) = &state.run {
            for act in &run.map_history {
                for point in &act.points {
                    for room in &point.rooms {
                        if let Some(model) = &room.model_id {
                            touched.push(model.to_string());
                        }
                    }
                }
            }
        }
        for relic in &state.run_player.relics {
            touched.push(relic.model_id.to_string());
        }
        for card in &state.run_player.deck {
            touched.push(card.model_id.to_string());
        }
        touched.retain(|model| !self.seen.contains(model));
        touched.sort();
        touched.dedup();
        touched
    }

    /// How much of the batch's novelty table is filled.
    #[must_use]
    pub fn seen_count(&self) -> usize {
        self.seen.len()
    }
}

impl Coverage {
    #[allow(
        clippy::cast_precision_loss,
        reason = "floors and novelty counts are far below f64 precision"
    )]
    fn score(&self, simulator: &Simulator, novelty: f64) -> f64 {
        let state = simulator.state();
        let floor = state.run.as_ref().map_or(0, |run| run.floor);
        let win = match state.terminal {
            Some(RunResult::Victory) => 1.0,
            _ => 0.0,
        };
        self.win_weight * win + self.floor_weight * f64::from(floor) + self.novelty_weight * novelty
    }
}

impl Objective for Coverage {
    #[allow(
        clippy::cast_precision_loss,
        reason = "novelty counts are far below f64 precision"
    )]
    fn reward(&mut self, simulator: &Simulator) -> f64 {
        let fresh = self.fresh(simulator);
        let novelty = fresh.len() as f64;
        self.seen.extend(fresh);
        self.score(simulator, novelty)
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "novelty counts are far below f64 precision"
    )]
    fn peek(&self, simulator: &Simulator) -> f64 {
        self.score(simulator, self.fresh(simulator).len() as f64)
    }
}

/// One coverage table shared between the harness (which settles rewards) and
/// a search (which peeks while steering): the batch is one batch.
#[derive(Clone, Debug)]
pub struct SharedCoverage(Arc<Mutex<Coverage>>);

impl Default for SharedCoverage {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Coverage::default())))
    }
}

impl SharedCoverage {
    #[must_use]
    pub fn seen_count(&self) -> usize {
        self.0.lock().expect("coverage lock").seen_count()
    }
}

impl Objective for SharedCoverage {
    fn reward(&mut self, simulator: &Simulator) -> f64 {
        self.0.lock().expect("coverage lock").reward(simulator)
    }

    fn peek(&self, simulator: &Simulator) -> f64 {
        self.0.lock().expect("coverage lock").peek(simulator)
    }
}

/// The fair-analysis objective for combat-scoped belief search: a fight is
/// worth what the player walks out with. Leaving the combat alive pays the
/// win weight, hit points retained pay proportionally, potions still held
/// pay a shadow price each, and a defeat pays nothing. Stateless — strength
/// has no batch table to remember — so `reward` and `peek` agree.
///
/// The potion term is the stage-one answer to a combat-scoped objective's
/// blind spot: inside one fight a potion is free value, so without a price
/// on the future the search drinks everything every fight. The weight is a
/// hand-tuned guess at run-level worth; the principled price arrives when a
/// run-value function can score the fight's exit state instead.
///
/// The turn term prices the objective's other blind spot: time. The game
/// permits fights nobody can end — a deck whose attacks True Grit exhausted
/// cycles Defends against a survivor forever — and a time-neutral objective
/// makes cycling value-stable, so a search dithers into and inside the trap
/// (found by the first net-guided gate: one run, forty minutes, three cards,
/// an enemy at 15 HP nobody could touch). A price per turn makes every turn
/// bleed value, so ending the fight dominates outlasting it on any horizon.
///
/// The price *discounts* rather than subtracts, and that is the whole of
/// `v3`. A subtracted price floored at zero — which is what `v2` was — stops
/// discriminating the moment the price exceeds what the fight can possibly
/// pay: past `(win_weight + 1 + potions·potion_weight) / turn_weight` turns
/// every line clamps to the same 0.0, winning and stalling included, and the
/// search falls through to its priors and cycles forever. That is not a
/// hypothetical: a Regent at 4/86 HP holding two Strikes against a 3 HP
/// Twig Slime cycled Defends from turn 53 (where its ceiling of 1.047 met
/// the price) to the step cap at 146. Dividing by `1 + turn_weight · turns`
/// keeps the ordering strict at every turn count and stays above zero, so a
/// long fight can never price below the defeat it is not.
#[derive(Clone, Copy, Debug)]
pub struct CombatStrength {
    pub win_weight: f64,
    pub hp_weight: f64,
    /// What holding onto one potion past the fight is worth.
    pub potion_weight: f64,
    /// What one elapsed turn costs.
    pub turn_weight: f64,
}

/// The name of what `z` means under this objective's current terms, stamped
/// into sample files: a checkpoint trained on one pricing must not be read
/// as if it scored another. `v2` added the turn price and moved the win
/// settle from the map back to the fight's own end. `v3` turned that price
/// from a subtraction floored at zero into a discount, because the floor
/// collapsed the objective to a constant in exactly the long fights the
/// price exists to end.
pub const VALUE_SEMANTICS: &str = "combat-strength-v3";

impl Default for CombatStrength {
    fn default() -> Self {
        Self {
            win_weight: 1.0,
            hp_weight: 1.0,
            potion_weight: 0.1,
            turn_weight: 0.02,
        }
    }
}

/// The fraction of maximum hit points the player stands on. In a fight the
/// player's hit points live on their creature; outside one they have been
/// settled back onto the run.
fn retained_hp_fraction(state: &sts2_engine::GameState) -> f64 {
    let (current, max) = state
        .combat
        .as_ref()
        .and_then(|combat| {
            combat
                .creatures
                .iter()
                .find(|creature| creature.id == combat.player.creature_id)
                .map(|creature| (creature.current_hp, creature.max_hp))
        })
        .unwrap_or((state.run_player.current_hp, state.run_player.max_hp));
    if max > 0 {
        f64::from(current.max(0)) / f64::from(max)
    } else {
        0.0
    }
}

impl CombatStrength {
    fn score(self, simulator: &Simulator) -> f64 {
        let state = simulator.state();
        if state.terminal == Some(RunResult::Defeat) {
            return 0.0;
        }
        let retained = retained_hp_fraction(state);
        // A state past the fight is a fight survived — the belief horizon
        // scores exactly here, victory screen included: the fight is won
        // where the engine says it ended, not where the room is left.
        let won = if crate::env::fight_over(simulator) {
            1.0
        } else {
            0.0
        };
        #[allow(
            clippy::cast_precision_loss,
            reason = "a potion belt holds single digits"
        )]
        let held = state.run_player.potions.iter().flatten().count() as f64;
        let turns = state
            .combat
            .as_ref()
            .map_or(0.0, |combat| f64::from(combat.player.turn));
        // Discounted, not docked: every term is non-negative and the divisor
        // only grows, so a pathological hundred-turn line decays toward
        // defeat's score without ever reaching it — and, unlike a subtraction
        // floored at zero, the ordering between its own continuations stays
        // strict however long the fight runs.
        (self.win_weight * won + self.hp_weight * retained + self.potion_weight * held)
            / (1.0 + self.turn_weight * turns)
    }
}

impl CombatStrength {
    /// The most any continuation of this state can score: the fight
    /// survived at full health with every potion slot filled, discounted by
    /// the turns already elapsed. Every term stands at its maximum and the
    /// divisor only grows from here, so no line out of this state pays more.
    /// A value head that says otherwise is extrapolating, not predicting.
    #[allow(
        clippy::cast_precision_loss,
        reason = "a potion belt holds single digits"
    )]
    fn ceiling(self, simulator: &Simulator) -> f64 {
        let state = simulator.state();
        if state.terminal == Some(RunResult::Defeat) {
            return 0.0;
        }
        let slots = state.run_player.potions.len() as f64;
        let turns = state
            .combat
            .as_ref()
            .map_or(0.0, |combat| f64::from(combat.player.turn));
        (self.win_weight + self.hp_weight + self.potion_weight * slots)
            / (1.0 + self.turn_weight * turns)
    }
}

impl Objective for CombatStrength {
    fn reward(&mut self, simulator: &Simulator) -> f64 {
        self.score(simulator)
    }

    fn peek(&self, simulator: &Simulator) -> f64 {
        self.score(simulator)
    }

    /// Nothing below a defeat, nothing above the fight survived whole: the
    /// bound the search holds a net's leaf value inside.
    fn attainable(&self, simulator: &Simulator) -> std::ops::RangeInclusive<f64> {
        0.0..=self.ceiling(simulator)
    }
}

/// Scores act-scoped belief search by crossing the act, retaining hit points
/// and potions, and climbing floors. The turn discount penalizes time spent
/// in the fight containing a leaf. Stateless, so `reward` and `peek` agree.
///
/// Defeats still earn floor progress: this distinguishes failed rollouts
/// that reached different floors. Truncated rollouts also retain a progress
/// signal when they have not reached the act boundary.
///
/// Every macro sample in an act shares this horizon score as its `z`, settled
/// at `act_over` or the end of the run. [`ACT_VALUE_SEMANTICS`] identifies its
/// scale; it must not be mixed with combat value targets.
#[derive(Clone, Copy, Debug)]
pub struct ActBoundary {
    /// What reaching the act's exit is worth.
    pub crossing_weight: f64,
    pub hp_weight: f64,
    /// What holding onto one potion past the act is worth.
    pub potion_weight: f64,
    /// What one floor of progress is worth to a rollout the cap truncated.
    pub floor_weight: f64,
    /// What one elapsed turn of the fight a leaf stands in costs.
    pub turn_weight: f64,
}

/// Identifies macro value targets under [`ActBoundary`]. Version 2 discounts
/// the crossing terms by turns spent in combat and adds floor progress
/// outside that discount. Its scale differs from [`VALUE_SEMANTICS`], so
/// combat and macro samples must not share a checkpoint's training data.
pub const ACT_VALUE_SEMANTICS: &str = "act-boundary-v2";

/// The name of what a *run-level* critic predicts, stamped into PPO
/// trajectory files exactly as [`VALUE_SEMANTICS`] and
/// [`ACT_VALUE_SEMANTICS`] are stamped into the other two nets' data: the
/// return of the whole remaining run under
/// [`RunReward`](crate::reward::RunReward) — the floors still to be climbed,
/// plus the victory or defeat term the run will end on.
///
/// It is neither of the other two, and the distance is not a matter of
/// weights. `combat-strength-v3` scores one fight and stops where the fight
/// does. `act-boundary-v2` scores one act and stops at its exit. This scores
/// a whole climb and stops only where the run does, which makes it the one
/// of the three whose horizon can see an act-three boss from act one. It is
/// also the only one that goes negative, because a defeat costs rather than
/// merely paying nothing, and the only one whose scale grows with the length
/// of the climb rather than sitting inside a fixed band.
///
/// The three checkpoints are interchangeable on the wire and nowhere else —
/// one graph shape, one vocabulary, one file layout — so each gets a string
/// and a loader of its own ([`load_run`](crate::net::PolicyValueNet::load_run)
/// for this one). A checkpoint is never reinterpreted, only refused.
pub const RUN_VALUE_SEMANTICS: &str = "run-return-v1";

impl Default for ActBoundary {
    fn default() -> Self {
        Self {
            crossing_weight: 1.0,
            hp_weight: 1.0,
            potion_weight: 0.1,
            floor_weight: 0.02,
            turn_weight: 0.02,
        }
    }
}

impl ActBoundary {
    fn score(self, simulator: &Simulator) -> f64 {
        let state = simulator.state();
        let floor = state.run.as_ref().map_or(0, |run| run.floor);
        if state.terminal == Some(RunResult::Defeat) {
            // Nothing survives a defeat but the climb itself: no crossing,
            // no hit points, and a belt of potions nobody will drink.
            return self.floor_weight * f64::from(floor);
        }
        // The act is crossed where its sample's horizon stands: the engine's
        // own transition, the boss's exit screens, or a run won outright —
        // `act_over` on a live state is exactly that set.
        let crossed = if crate::env::act_over(simulator) {
            1.0
        } else {
            0.0
        };
        #[allow(
            clippy::cast_precision_loss,
            reason = "a potion belt holds single digits"
        )]
        let held = state.run_player.potions.iter().flatten().count() as f64;
        let turns = state
            .combat
            .as_ref()
            .map_or(0.0, |combat| f64::from(combat.player.turn));
        // `CombatStrength`'s discipline, on this horizon: the time price
        // discounts what the act is worth rather than docking it, so no
        // amount of dithering drives a live act under the dead one it is
        // still ahead of. The climb sits outside the discount deliberately —
        // it is what a defeat is paid, so a live act must never price below
        // it, and floors already climbed do not become worth less because
        // the fight standing on them ran long.
        (self.crossing_weight * crossed
            + self.hp_weight * retained_hp_fraction(state)
            + self.potion_weight * held)
            / (1.0 + self.turn_weight * turns)
            + self.floor_weight * f64::from(floor)
    }
}

impl Objective for ActBoundary {
    fn reward(&mut self, simulator: &Simulator) -> f64 {
        self.score(simulator)
    }

    fn peek(&self, simulator: &Simulator) -> f64 {
        self.score(simulator)
    }
}
