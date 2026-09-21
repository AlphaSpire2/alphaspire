//! The determinizer boundary produces reproducible worlds and mode-correct keys,
//! policies are deterministic under an analysis seed, the coverage objective
//! pays novelty once, and UCT explores before it exploits.

use alphaspire::env::{Belief, Determinizer, NodeKey, TrueState};
use alphaspire::objective::{Coverage, Objective};
use alphaspire::policy::{FirstLegal, RolloutPolicy, UniformRandom};
use alphaspire::search::{EdgeStats, NodeStats, Selection, Uct};
use sts2_core::UnlockPresetManifest;
use sts2_engine::Simulator;
use sts2_rng::MegaRandom;

fn fresh_run(seed: &str) -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at(seed, &character, &preset, 0).unwrap()
}

#[test]
fn true_state_samples_are_the_same_world_under_an_exact_key() {
    let mut determinizer = TrueState::new(fresh_run("NLD6VZXP94"));
    let one = determinizer.sample(0);
    let two = determinizer.sample(7);
    assert_eq!(
        one.state_key().unwrap(),
        two.state_key().unwrap(),
        "true state hides nothing, so every rollout gets the same world"
    );
    let NodeKey::Exact(exact) = determinizer.node_key(&one).unwrap() else {
        panic!("true-state search keys exactly");
    };
    assert_eq!(exact, one.state_key().unwrap());
}

/// The first legal action, preferring a non-empty card pick so a skippable
/// reward cannot re-offer itself.
fn next_action(simulator: &Simulator) -> sts2_engine::Action {
    simulator
        .legal_actions()
        .iter()
        .find(
            |action| matches!(action, sts2_engine::Action::ChooseCards { cards, .. } if !cards.is_empty()),
        )
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
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }
    panic!("the seed reaches a combat inside forty decisions");
}

#[test]
fn belief_keys_by_what_a_player_can_see() {
    let simulator = first_combat("NLD6VZXP94");
    let belief = Belief::from_simulator(&simulator, 3).unwrap();
    let NodeKey::Observation(key) = belief.node_key(&simulator).unwrap() else {
        panic!("belief search keys by observation");
    };
    assert_eq!(key, simulator.observation_key().unwrap());
}

#[test]
fn belief_worlds_are_reproducible_and_agree_on_what_is_visible() {
    let original = first_combat("NLD6VZXP94");
    let mut determinizer = Belief::from_simulator(&original, 5).unwrap();
    let one = determinizer.sample(2);
    let again = determinizer.sample(2);
    assert_eq!(
        one.state_key().unwrap(),
        again.state_key().unwrap(),
        "the same rollout index deals the same world"
    );
    let other = determinizer.sample(3);
    assert_eq!(
        one.observation_key().unwrap(),
        other.observation_key().unwrap(),
        "every world shows the player the same screen"
    );
    assert_eq!(
        one.legal_actions(),
        original.legal_actions(),
        "and offers the same actions"
    );
    assert!(
        !determinizer.deterministic_transitions(),
        "hidden state varies by rollout, so edges cache no keys"
    );
}

#[test]
fn the_belief_horizon_is_the_end_of_the_fight() {
    let in_combat = first_combat("NLD6VZXP94");
    let determinizer = Belief::from_simulator(&in_combat, 1).unwrap();
    assert!(!determinizer.beyond_horizon(&in_combat));
    assert!(
        determinizer.beyond_horizon(&fresh_run("NLD6VZXP94")),
        "outside the fight the sample's hidden state was carried, not sampled"
    );
    assert!(
        Belief::from_simulator(&fresh_run("NLD6VZXP94"), 1).is_err(),
        "combat-v0 refuses to erase a state outside a fight"
    );
}

#[test]
fn a_policy_walk_is_reproducible_from_its_analysis_seed() {
    let walk = |analysis_seed: u64| {
        let mut simulator = fresh_run("NLD6VZXP94");
        let mut rng = MegaRandom::new(analysis_seed);
        let mut policy = UniformRandom;
        for _ in 0..60 {
            if simulator.state().terminal.is_some() {
                break;
            }
            let action = policy.choose(&simulator, &mut rng);
            simulator.step(action).unwrap();
        }
        simulator.state_key().unwrap()
    };
    assert_eq!(walk(11), walk(11), "one seed, one trajectory");
    assert_ne!(walk(11), walk(12), "another seed wanders elsewhere");
}

#[test]
fn coverage_pays_novelty_once_and_depth_always() {
    let mut objective = Coverage::default();
    let start = fresh_run("NLD6VZXP94");
    let mut walked = start.clone();
    let mut rng = MegaRandom::new(3);
    let mut policy = FirstLegal;
    for _ in 0..80 {
        if walked.state().terminal.is_some() {
            break;
        }
        let action = policy.choose(&walked, &mut rng);
        walked.step(action).unwrap();
    }
    let first = objective.reward(&walked);
    let second = objective.reward(&walked);
    assert!(
        first > second,
        "novelty is paid once: {first} then {second}"
    );
    assert!(
        second > objective.reward(&start),
        "depth still counts once novelty is spent"
    );
    assert!(objective.seen_count() > 0, "the batch table filled");
}

#[test]
fn uct_explores_the_unvisited_and_then_exploits() {
    let mut uct = Uct::default();
    let edges = [
        EdgeStats {
            visits: 10,
            availability: 10,
            total_value: 9.0,
            prior: 1.0,
        },
        EdgeStats::fresh(1.0),
    ];
    assert_eq!(
        uct.descend(NodeStats::default(), &edges),
        1,
        "an unvisited edge goes first"
    );
    let edges = [
        EdgeStats {
            visits: 10,
            availability: 20,
            total_value: 9.0,
            prior: 1.0,
        },
        EdgeStats {
            visits: 10,
            availability: 20,
            total_value: 1.0,
            prior: 1.0,
        },
    ];
    assert_eq!(
        uct.descend(
            NodeStats {
                visits: 20,
                value: None,
            },
            &edges
        ),
        0,
        "with equal visits the better mean wins"
    );
}

#[test]
fn a_decision_script_round_trips() {
    let mut simulator = fresh_run("NLD6VZXP94");
    let mut rng = MegaRandom::new(1);
    let mut policy = FirstLegal;
    let mut actions = Vec::new();
    for _ in 0..10 {
        let action = policy.choose(&simulator, &mut rng);
        actions.push(action.clone());
        simulator.step(action).unwrap();
    }
    let script = alphaspire::script::DecisionScript {
        seed: "NLD6VZXP94".to_owned(),
        character: "CHARACTER.IRONCLAD".to_owned(),
        ascension: 0,
        unlock_preset: "v0107_1_veteran".to_owned(),
        analysis_seed: 1,
        actions,
    };
    let json = serde_json::to_string(&script).unwrap();
    let back: alphaspire::script::DecisionScript = serde_json::from_str(&json).unwrap();
    assert_eq!(back, script);

    // A script replays in-process to the same state: the reproducibility the
    // whole validation loop leans on.
    let mut replayed = fresh_run("NLD6VZXP94");
    for action in back.actions {
        replayed.step(action).unwrap();
    }
    assert_eq!(
        replayed.state_key().unwrap(),
        simulator.state_key().unwrap()
    );
}

/// An emitted script must parse as Format v1 and replay through the
/// simulator's adapter to the same state as the original run.
#[test]
fn an_emitted_script_replays_through_the_adapter_to_the_same_state() {
    let mut coverage = Coverage::default();
    let mut rng = MegaRandom::new(21);
    let mut policy = UniformRandom;
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let report = alphaspire::selfplay::play_run(
        "NLD6VZXP94",
        &character,
        0,
        &mut policy,
        &mut rng,
        &mut coverage,
        200,
    )
    .unwrap();
    let script = report.script.expect("this walk stays scriptable");

    let mut walked = fresh_run("NLD6VZXP94");
    for action in &report.actions {
        walked.step(action.clone()).unwrap();
    }

    let mut parser = sts2_replay::Parser::new(std::io::Cursor::new(script.clone())).unwrap();
    assert_eq!(
        parser.header("Producer"),
        Some(alphaspire::script::PRODUCER)
    );
    assert_eq!(parser.header("Mode"), Some("CUSTOM"));
    let mut adapter = sts2_replay::ReplayAdapter::new(fresh_run("NLD6VZXP94"));
    let mut applied = 0_usize;
    for event in parser.by_ref() {
        match event.unwrap() {
            sts2_replay::Event::Record(record) => {
                if record.kind == "run.start" {
                    continue;
                }
                match adapter.accept(&record).unwrap() {
                    sts2_replay::AdapterOutcome::Applied(_) => applied += 1,
                    sts2_replay::AdapterOutcome::Ignored(_) => {}
                }
            }
            sts2_replay::Event::Terminal(token) => assert_eq!(token, "*"),
        }
    }
    assert_eq!(applied, report.actions.len(), "every decision applied");
    assert_eq!(
        adapter.simulator().state_key().unwrap(),
        walked.state_key().unwrap(),
        "the script drives the adapter to the walked state"
    );
}
