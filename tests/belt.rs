//! The belt outside a fight: a bare discard is withheld from the policy,
//! a trade still names its own discard, a discard inside a fight stays —
//! and a relic left on a reward screen is counted as one.

use alphaspire::actor::{MacroActor, Resolver};
use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::{Evaluate, PolicyValueNet};
use alphaspire::objective::CombatStrength;
use alphaspire::plan::{ActionPlan, permitted_plans};
use alphaspire::policy::RolloutPolicy;
use alphaspire::policy::permitted_actions;
use alphaspire::summary::{BatchSummary, EpisodeMetrics, Provenance};
use std::path::Path;
use std::sync::Arc;
use sts2_core::UnlockPresetManifest;
use sts2_engine::{Action, DecisionContext, RewardFingerprint, RewardItem, Simulator};
use sts2_rng::MegaRandom;

const SEED: &str = "NLD6VZXP94";
/// `PotionUsage.CombatOnly`: held, discardable, and never drinkable outside a
/// fight.
const THROWN: &str = "POTION.VULNERABLE_POTION";
const RELIC: &str = "RELIC.AKABEKO";

fn id(value: &str) -> sts2_core::ModelId {
    value.parse().expect("static model ID")
}

fn fresh_run() -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at(SEED, &character, &preset, 0).unwrap()
}

fn next_action(simulator: &Simulator) -> Action {
    simulator
        .legal_actions()
        .iter()
        .find(|action| matches!(action, Action::ChooseCards { cards, .. } if !cards.is_empty()))
        .or_else(|| simulator.legal_actions().first())
        .cloned()
        .expect("a live run offers an action")
}

fn walk_until(mut simulator: Simulator, stop: impl Fn(&Simulator) -> bool) -> Simulator {
    for _ in 0..300 {
        if stop(&simulator) {
            return simulator;
        }
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }
    panic!("the seed reaches the stop inside three hundred decisions");
}

fn rebuilt(simulator: &Simulator, edit: impl FnOnce(&mut sts2_engine::GameState)) -> Simulator {
    let mut state = simulator.state().clone();
    edit(&mut state);
    Simulator::from_scenario(state, sts2_content::standard_registry()).expect("the state rebuilds")
}

fn hold(state: &mut sts2_engine::GameState, slots: usize, potions: &[&str]) {
    state.run_player.potions = (0..slots)
        .map(|slot| potions.get(slot).map(|model| id(model)))
        .collect();
}

fn in_a_fight(simulator: &Simulator) -> bool {
    simulator
        .state()
        .combat
        .as_ref()
        .is_some_and(|combat| combat.in_progress)
}

/// A won fight's reward screen offering exactly `rewards`, the belt resized
/// to `slots` and holding `potions`.
fn reward_screen(slots: usize, potions: &[&str], rewards: Vec<RewardItem>) -> Simulator {
    let screen = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::Rewards { .. })
    });
    rebuilt(&screen, |state| {
        hold(state, slots, potions);
        let set = state
            .combat
            .as_mut()
            .and_then(|combat| combat.reward_set.as_mut())
            .or_else(|| state.run.as_mut().and_then(|run| run.reward_set.as_mut()))
            .expect("the screen is open");
        set.rewards = rewards;
    })
}

fn reward(index: usize, kind: &str, model: &str) -> RewardItem {
    RewardItem {
        index,
        fingerprint: RewardFingerprint::of(kind).with_model(id(model)),
        selected: false,
        passed_over: false,
        opened: false,
    }
}

fn discards(actions: &[Action]) -> Vec<&Action> {
    actions
        .iter()
        .filter(|action| matches!(action, Action::DiscardPotion { .. }))
        .collect()
}

#[test]
fn a_bare_discard_outside_a_fight_is_withheld_from_the_policy() {
    let simulator = reward_screen(3, &[THROWN], vec![reward(0, "potion", THROWN)]);
    assert_eq!(
        discards(simulator.legal_actions()).len(),
        1,
        "the engine offers the held potion's discard"
    );
    assert!(
        discards(&permitted_actions(&simulator)).is_empty(),
        "and the policy is not: {:?}",
        permitted_actions(&simulator)
    );
    assert!(
        permitted_plans(&simulator)
            .iter()
            .all(|plan| !matches!(plan.lead(), Action::DiscardPotion { .. })),
        "no plan leads with one either"
    );
}

#[test]
fn a_full_belt_still_trades_through_its_own_discard() {
    let simulator = reward_screen(
        3,
        &[THROWN, THROWN, THROWN],
        vec![reward(0, "potion", THROWN)],
    );
    let plans = permitted_plans(&simulator);
    let frees: Vec<&Action> = plans
        .iter()
        .filter_map(|plan| match plan {
            ActionPlan::Trade { free, .. } => Some(free),
            _ => None,
        })
        .collect();
    assert_eq!(
        frees.len(),
        3,
        "one trade per slot, each freed by a discard: {plans:?}"
    );
    assert!(
        frees
            .iter()
            .all(|free| matches!(free, Action::DiscardPotion { .. })),
        "the discard the trade names is the engine's own"
    );
    assert!(
        plans
            .iter()
            .all(|plan| !matches!(plan, ActionPlan::Single(Action::DiscardPotion { .. }))),
        "and none stands on its own"
    );
}

#[test]
fn a_discard_inside_a_fight_is_left_to_the_fight() {
    let fight = walk_until(fresh_run(), in_a_fight);
    let simulator = rebuilt(&fight, |state| hold(state, 3, &[THROWN]));
    assert!(
        in_a_fight(&simulator),
        "the rebuilt state is still mid-fight"
    );
    assert_eq!(
        discards(simulator.legal_actions()).len(),
        1,
        "the engine offers the discard in the fight"
    );
    assert_eq!(
        discards(&permitted_actions(&simulator)).len(),
        1,
        "and so does the policy's list"
    );
}

#[test]
fn a_relic_left_on_a_reward_screen_is_counted_and_a_claimed_one_is_not() {
    let simulator = reward_screen(
        3,
        &[],
        vec![reward(0, "relic", RELIC), reward(1, "gold", RELIC)],
    );
    let plans = permitted_plans(&simulator);
    let leave = plans
        .iter()
        .find(|plan| matches!(plan.lead(), Action::Proceed))
        .cloned()
        .expect("the screen can be walked off");
    let claim = plans
        .iter()
        .find(|plan| {
            matches!(plan.lead(), Action::ClaimReward { fingerprint, .. } if fingerprint.reward_type == "relic")
        })
        .cloned()
        .expect("the relic can be claimed");

    let mut left = EpisodeMetrics::default();
    left.decided(&simulator, &plans, &leave);
    let waste = left.waste();
    assert_eq!(
        (waste.relic_rewards, waste.relic_rewards_left),
        (1, 1),
        "walking off with the relic standing leaves one: {waste:?}"
    );

    let mut taken = EpisodeMetrics::default();
    taken.decided(&simulator, &plans, &claim);
    let waste = taken.waste();
    assert_eq!(
        (waste.relic_rewards, waste.relic_rewards_left),
        (1, 0),
        "claiming it settles one and leaves none: {waste:?}"
    );
}

/// The batch's own view of the counter: what one episode settled is what
/// the batch reports, so a field the merge forgot would read zero over any
/// number of runs — which is exactly how the first build of this counter
/// read over four hundred runs and a thousand relics.
#[test]
fn the_batch_carries_the_relic_counters_of_its_episodes() {
    let simulator = reward_screen(3, &[], vec![reward(0, "relic", RELIC)]);
    let plans = permitted_plans(&simulator);
    let leave = plans
        .iter()
        .find(|plan| matches!(plan.lead(), Action::Proceed))
        .cloned()
        .expect("the screen can be walked off");
    let mut metrics = EpisodeMetrics::default();
    metrics.decided(&simulator, &plans, &leave);

    let registry = sts2_content::standard_registry();
    let encoder = Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ));
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let net: Arc<dyn Evaluate> =
        Arc::new(PolicyValueNet::load(&base, encoder).expect("the fixture checkpoint loads"));
    let mut policy: Box<dyn RolloutPolicy> = Box::new(MacroActor::new(
        Arc::clone(&net),
        Resolver::Greedy.build(net),
    ));
    let mut rng = MegaRandom::new(7);
    let mut objective = CombatStrength::default();
    let mut report = alphaspire::selfplay::play_run(
        SEED,
        &"CHARACTER.IRONCLAD".parse().unwrap(),
        0,
        policy.as_mut(),
        &mut rng,
        &mut objective,
        40,
    )
    .expect("the run walks");
    report.metrics = Some(metrics);

    let mut batch = BatchSummary::new(Provenance {
        run_net: "ckpts/run".into(),
        combat_net: "ckpts/combat".into(),
        resolver: Resolver::Greedy,
        character: "CHARACTER.IRONCLAD".into(),
        ascension: 0,
        runs: 1,
        analysis_seed: 1,
        max_steps: 40,
    });
    batch.record(&report);
    let json = batch.json();
    assert_eq!(
        (
            json["waste"]["relic_rewards"].as_u64(),
            json["waste"]["relic_rewards_left"].as_u64()
        ),
        (Some(1), Some(1)),
        "the episode's one relic left is the batch's: {}",
        json["waste"]
    );
    assert!(
        batch.report().contains("relic rewards 1 of 1"),
        "and the report line says so: {}",
        batch.report()
    );
}
