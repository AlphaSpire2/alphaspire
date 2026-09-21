//! A recording search asked for pass branches ends the turn beside the
//! line it plays, at a decision where an attack would have landed, and
//! records the branch's decisions with the branch's own outcome under the
//! `pass` kind — the next-turn position self-play never reaches.

use alphaspire::objective::CombatStrength;
use alphaspire::policy::{RolloutPolicy, UniformRandom};
use alphaspire::search::{BeliefSearch, Counterfactual, Gumbel, SearchConfig};
use alphaspire::training::Decision;
use sts2_engine::{CardFingerprint, PileName, ScenarioBuilder, Simulator};
use sts2_rng::MegaRandom;

fn id(value: &str) -> sts2_core::ModelId {
    value.parse().unwrap()
}

/// A beetle fight where Strikes are in hand on turn one and the draw pile
/// holds more: every turn can attack, so every turn can also be passed.
fn fight() -> Simulator {
    let mut builder = ScenarioBuilder::new("PASSBRANCH1", id("ENCOUNTER.SHRINKER_BEETLE_WEAK"))
        .player_hp(80, 80)
        .energy(3, 3)
        .turn(1, 1)
        .shuffle_counter(0)
        .enemy(1, id("MONSTER.SHRINKER_BEETLE"), 39, 39, "SHRINKER_MOVE");
    for (index, card) in [
        (10, "CARD.STRIKE_IRONCLAD"),
        (11, "CARD.STRIKE_IRONCLAD"),
        (12, "CARD.DEFEND_IRONCLAD"),
        (13, "CARD.BASH"),
        (14, "CARD.DEFEND_IRONCLAD"),
    ] {
        let cost = if card == "CARD.BASH" { 2 } else { 1 };
        builder = builder.card(PileName::Hand, index, CardFingerprint::base(id(card)), cost);
    }
    for (index, card) in [
        (20, "CARD.STRIKE_IRONCLAD"),
        (21, "CARD.STRIKE_IRONCLAD"),
        (22, "CARD.DEFEND_IRONCLAD"),
        (23, "CARD.STRIKE_IRONCLAD"),
        (24, "CARD.DEFEND_IRONCLAD"),
    ] {
        builder = builder.card(PileName::Draw, index, CardFingerprint::base(id(card)), 1);
    }
    builder.build(sts2_content::standard_registry()).unwrap()
}

fn play(counterfactual: Counterfactual) -> Vec<Decision> {
    let mut search = BeliefSearch::with_rollout(
        SearchConfig {
            iterations: 16,
            ..SearchConfig::default()
        },
        CombatStrength::default(),
        Box::new(UniformRandom),
    )
    .selecting(Box::new(Gumbel {
        considered: 4,
        ..Gumbel::default()
    }))
    .recording()
    .branching(counterfactual);
    let mut simulator = fight();
    let mut rng = MegaRandom::new(7);
    for _ in 0..400 {
        if simulator.state().terminal.is_some() || alphaspire::env::fight_over(&simulator) {
            break;
        }
        let action = search.choose(&simulator, &mut rng);
        simulator.step_quietly(&action).unwrap();
    }
    search.run_ended(&simulator);
    search.drain_decisions()
}

#[test]
fn a_pass_branch_records_the_next_turn_under_its_own_kind() {
    let decisions = play(Counterfactual {
        margin: 0.05,
        max_prior: 0.1,
        per_fight: 0,
        max_steps: 500,
        pass_per_fight: 1,
        pass_rate: 1.0,
    });
    let main: Vec<&Decision> = decisions.iter().filter(|d| !d.counterfactual).collect();
    let branch: Vec<&Decision> = decisions.iter().filter(|d| d.counterfactual).collect();
    assert!(!main.is_empty(), "the fight itself is recorded");
    assert!(
        !branch.is_empty(),
        "with every qualifying decision branching, the fight holds a pass branch"
    );
    assert!(
        branch
            .iter()
            .all(|d| d.counterfactual_kind.as_deref() == Some("pass")),
        "every branch row names the pass rule: {:?}",
        branch
            .iter()
            .map(|d| d.counterfactual_kind.clone())
            .collect::<Vec<_>>()
    );
    // The branch's first decision is the turn after the pass: a fresh turn
    // with energy unspent, later than the turn the main line was on when
    // the branch was taken.
    let first = branch[0];
    assert!(
        first.observation.turn.is_some_and(|turn| turn >= 2),
        "turn {:?}",
        first.observation.turn
    );
    assert_eq!(first.observation.energy, first.observation.max_energy);
    // Main-line rows carry no kind.
    assert!(main.iter().all(|d| d.counterfactual_kind.is_none()));
}

#[test]
fn no_pass_branch_without_the_rule() {
    let decisions = play(Counterfactual {
        margin: 0.05,
        max_prior: 0.1,
        per_fight: 0,
        max_steps: 500,
        pass_per_fight: 0,
        pass_rate: 1.0,
    });
    assert!(decisions.iter().all(|d| !d.counterfactual));
}

#[test]
fn the_cap_bounds_the_pass_branches_per_fight() {
    let decisions = play(Counterfactual {
        margin: 0.05,
        max_prior: 0.1,
        per_fight: 0,
        max_steps: 500,
        pass_per_fight: 2,
        pass_rate: 1.0,
    });
    // Branch rows are drained in the order the branches ended, each branch
    // contiguous and its turns rising to the fight's end; a new branch
    // starts on a turn no later than the previous branch's last. Count
    // those restarts: the cap bounds them.
    let mut starts = 0;
    let mut previous_turn: Option<u32> = None;
    for decision in decisions.iter().filter(|d| d.counterfactual) {
        let turn = decision.observation.turn.unwrap_or(0);
        if previous_turn.is_none_or(|previous| turn < previous) {
            starts += 1;
        }
        previous_turn = Some(turn);
    }
    assert!((1..=2).contains(&starts), "{starts} branches");
}
