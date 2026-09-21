//! The value the search reads off a net stays inside what the objective can
//! pay.
//!
//! `CombatStrength`'s ceiling is a fact of its arithmetic — the fight
//! survived at full health with every potion slot filled, discounted by the
//! turns elapsed — and a value head is a plain linear output that knows
//! nothing of it. At a state far from its training pool the head
//! extrapolates, and a maximising search plays for the payoff it names. The
//! search holds such a leaf at the ceiling instead.

use std::sync::Arc;

use alphaspire::net::{Evaluate, Priced};
use alphaspire::objective::{CombatStrength, Objective};
use alphaspire::policy::RolloutPolicy;
use alphaspire::search::{BeliefSearch, Gumbel, SearchConfig};
use sts2_core::UnlockPresetManifest;
use sts2_engine::{Action, Simulator};
use sts2_rng::MegaRandom;

fn fresh_run(seed: &str) -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at(seed, &character, &preset, 0).unwrap()
}

fn config() -> SearchConfig {
    SearchConfig {
        iterations: 32,
        rollout_depth: 5,
        temperature: 0.5,
    }
}

/// The action a plain walk takes: a non-empty card pick where one is
/// offered, else the first thing on the screen.
fn next_action(simulator: &Simulator) -> Action {
    simulator
        .legal_actions()
        .iter()
        .find(|action| matches!(action, Action::ChooseCards { cards, .. } if !cards.is_empty()))
        .or_else(|| simulator.legal_actions().first())
        .cloned()
        .expect("a live run offers an action")
}

/// The seeded run walked to its first in-combat decision.
fn first_combat(seed: &str) -> Simulator {
    let mut simulator = fresh_run(seed);
    for _ in 0..40 {
        if simulator.state().combat.is_some() {
            return simulator;
        }
        simulator.step_quietly(&next_action(&simulator)).unwrap();
    }
    panic!("the seed reaches a combat inside forty decisions");
}

/// A checkpoint that prices every position at one number, with flat priors:
/// a value head that has run away, everywhere at once.
struct Flat(f64);

impl Evaluate for Flat {
    #[allow(
        clippy::cast_precision_loss,
        reason = "action counts are far below f32 precision"
    )]
    fn priors_and_value(&self, _: &Simulator, actions: &[Action]) -> Priced {
        let width = actions.len().max(1) as f32;
        Priced::real(vec![1.0 / width; actions.len()], self.0)
    }

    fn state_value(&self, _: &Simulator) -> f64 {
        self.0
    }
}

#[test]
fn the_objective_never_pays_outside_its_attainable_range() {
    // Walk a run through its fights and read the objective at every state
    // it stands on: whatever it pays sits inside the range it declares, and
    // the range is a real bound, not the unbounded default.
    let objective = CombatStrength::default();
    let mut simulator = fresh_run("NLD6VZXP94");
    let mut in_combat = 0;
    for _ in 0..400 {
        if simulator.state().terminal.is_some() {
            break;
        }
        let range = objective.attainable(&simulator);
        assert!(
            range.end().is_finite() && *range.start() == 0.0,
            "the combat objective states a finite ceiling over a floor of nothing: {range:?}"
        );
        let paid = objective.peek(&simulator);
        assert!(
            range.contains(&paid),
            "the objective paid {paid} outside {range:?} at turn {:?}",
            simulator
                .state()
                .combat
                .as_ref()
                .map(|combat| combat.player.turn)
        );
        if simulator.state().combat.is_some() {
            in_combat += 1;
        }
        simulator.step_quietly(&next_action(&simulator)).unwrap();
    }
    assert!(in_combat > 20, "the walk stood inside fights: {in_combat}");
}

#[test]
fn a_leaf_the_net_overprices_is_read_at_the_objectives_ceiling() {
    // A head that says 5.0 at every state: past anything the fight can pay
    // at any turn. The search's root value is a mix of the root's own read
    // and the leaves' backed-up values, every one of them net-priced or the
    // objective's own word, so a root value inside the ceiling is every
    // leaf having been held there.
    let simulator = first_combat("NLD6VZXP94");
    let objective = CombatStrength::default();
    let ceiling = *objective.attainable(&simulator).end();
    assert!(
        ceiling < 5.0,
        "the fake head prices past the ceiling, or the test proves nothing: {ceiling}"
    );

    let mut policy = BeliefSearch::new(config(), objective)
        .selecting(Box::new(Gumbel::default()))
        .with_net(Arc::new(Flat(5.0)))
        .recording();
    let mut rng = MegaRandom::new(4);
    policy.choose(&simulator, &mut rng);
    policy.run_ended(&simulator);

    let decisions = policy.drain_decisions();
    let recorded = decisions.first().expect("the decision was recorded");
    let value = f64::from(
        recorded
            .root_value
            .expect("a searched decision has a root value"),
    );
    assert!(
        value.is_finite() && value > 0.0 && value <= ceiling + 1e-6,
        "the root value is held inside what the fight can pay ({ceiling}): {value}"
    );
}
