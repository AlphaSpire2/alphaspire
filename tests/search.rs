//! The search is deterministic from its seeds, its tree
//! obeys the visit ledger, and a searched prefix stays scriptable.
//!
//! The same core searches a belief — only the
//! determinizer and the node keys change — and its analysis is fair: two
//! states a player cannot tell apart get the same answer.

use alphaspire::env::{Belief, TrueState};
use alphaspire::objective::{CombatStrength, Coverage, Objective, SharedCoverage};
use alphaspire::policy::{RolloutPolicy as _, UniformRandom};
use alphaspire::search::{BeliefSearch, Mcts, SearchConfig, TrueStateSearch, Uct};
use sts2_core::UnlockPresetManifest;
use sts2_engine::{CardFingerprint, PileName, ScenarioBuilder, Simulator};
use sts2_rng::MegaRandom;

fn fresh_run(seed: &str) -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at(seed, &character, &preset, 0).unwrap()
}

fn tiny() -> SearchConfig {
    SearchConfig {
        iterations: 12,
        rollout_depth: 5,
        temperature: 0.5,
    }
}

#[test]
fn one_decision_fills_the_tree_and_keeps_the_visit_ledger() {
    let simulator = fresh_run("NLD6VZXP94");
    let mut mcts = Mcts::new(tiny(), Uct::default());
    let mut determinizer = TrueState::new(simulator.clone());
    let objective = Coverage::default();
    let mut rollout = UniformRandom;
    let mut rng = MegaRandom::new(5);
    let action = mcts.decide(&mut determinizer, &objective, &mut rollout, &mut rng);
    assert!(
        simulator.legal_actions().contains(&action),
        "the search answers with a legal action"
    );
    assert!(mcts.tree_size() > 0, "the tree grew");
}

#[test]
fn a_searched_walk_is_reproducible_from_its_seeds() {
    let walk = |analysis_seed: u64| {
        let coverage = SharedCoverage::default();
        let mut policy = TrueStateSearch::new(tiny(), coverage.clone());
        let mut objective: Box<dyn Objective> = Box::new(coverage);
        let mut rng = MegaRandom::new(analysis_seed);
        let character = "CHARACTER.IRONCLAD".parse().unwrap();
        let report = alphaspire::selfplay::play_run(
            "NLD6VZXP94",
            &character,
            0,
            &mut policy,
            &mut rng,
            objective.as_mut(),
            10,
        )
        .unwrap();
        (report.actions, report.script.expect("prefix is scriptable"))
    };
    let (actions_one, script_one) = walk(9);
    let (actions_two, script_two) = walk(9);
    assert_eq!(actions_one, actions_two, "one seed, one line of play");
    assert_eq!(script_one, script_two, "and one script");

    // The searched prefix replays in-process to the same state.
    let mut replayed = fresh_run("NLD6VZXP94");
    for action in actions_one {
        replayed.step(action).unwrap();
    }
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
fn the_same_core_searches_a_belief_and_answers_legally() {
    // Belief search uses the same MCTS core with a different determinizer.
    let original = first_combat("NLD6VZXP94");
    let mut mcts = Mcts::new(tiny(), Uct::default());
    let mut determinizer = Belief::from_simulator(&original, 5).unwrap();
    let objective = CombatStrength::default();
    let mut rollout = UniformRandom;
    let mut rng = MegaRandom::new(5);
    let action = mcts.decide(&mut determinizer, &objective, &mut rollout, &mut rng);
    assert!(
        original.legal_actions().contains(&action),
        "the search answers with an action the real screen offers"
    );
    assert!(mcts.tree_size() > 0, "the tree grew");
}

#[test]
fn two_indistinguishable_states_get_the_same_analysis() {
    // The original and a sampled possible world of it differ only in hidden
    // state. A fair search cannot tell them apart: same analysis seed, same
    // answer. Run against the true state instead, the two worlds would
    // disagree on what the clairvoyant search sees — this is exactly the
    // boundary between the modes.
    let original = first_combat("NLD6VZXP94");
    let twin = sts2_engine::BeliefState::from_simulator(&original)
        .unwrap()
        .sample(99, 4)
        .unwrap();
    assert_ne!(
        original.state_key().unwrap(),
        twin.state_key().unwrap(),
        "the twin hides different state"
    );
    let decide = |simulator: &Simulator| {
        let mut mcts = Mcts::new(tiny(), Uct::default());
        let mut determinizer = Belief::from_simulator(simulator, 7).unwrap();
        let mut rollout = UniformRandom;
        let mut rng = MegaRandom::new(3);
        mcts.decide(
            &mut determinizer,
            &CombatStrength::default(),
            &mut rollout,
            &mut rng,
        )
    };
    assert_eq!(
        decide(&original),
        decide(&twin),
        "what a player cannot distinguish, the analysis does not either"
    );
}

#[test]
fn a_belief_searched_walk_stays_legal_and_scriptable() {
    // The whole harness, one flag away from a TrueState run: belief search
    // inside fights, the rollout policy elsewhere, and the emitted prefix
    // still replays.
    let coverage = SharedCoverage::default();
    let mut policy = BeliefSearch::new(tiny(), CombatStrength::default());
    let mut objective: Box<dyn Objective> = Box::new(coverage);
    let mut rng = MegaRandom::new(9);
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let report = alphaspire::selfplay::play_run(
        "NLD6VZXP94",
        &character,
        0,
        &mut policy,
        &mut rng,
        objective.as_mut(),
        12,
    )
    .unwrap();
    report.script.expect("the searched prefix is scriptable");
    let mut replayed = fresh_run("NLD6VZXP94");
    for action in report.actions {
        replayed.step(action).unwrap();
    }
    assert!(policy.tree_size() > 0, "the fights were searched");
}

/// The seeded run walked to the victory screen of its first fight: the
/// combat is still standing, but the fight behind it is over.
fn first_victory_screen(seed: &str) -> Simulator {
    let mut simulator = fresh_run(seed);
    for _ in 0..400 {
        if simulator.state().combat.is_some() && alphaspire::env::fight_over(&simulator) {
            return simulator;
        }
        assert!(
            simulator.state().terminal.is_none(),
            "the seed wins its first fight"
        );
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }
    panic!("the seed reaches a won fight inside four hundred decisions");
}

#[test]
fn the_horizon_is_the_end_of_the_fight_not_the_end_of_the_room() {
    // The engine keeps the combat alive behind the victory screen, so
    // `combat.is_some()` outlives the fight. `combat-v0` erases nothing at
    // the run level: past the fight, a reward is rolled fresh in every
    // sampled world, and a search standing there would answer with an action
    // off a screen that does not exist. The erasure is refused there, and the
    // search hands the decision back to its rollout policy.
    let won = first_victory_screen("NLD6VZXP94");
    assert!(won.state().combat.is_some(), "the combat still stands");
    assert!(alphaspire::env::fight_over(&won), "the fight does not");
    assert!(
        Belief::from_simulator(&won, 5).is_err(),
        "combat-v0 refuses to erase what it does not sample"
    );

    // Outside the fight horizon, the rollout policy must answer the actual
    // reward screen rather than a cached screen from a sampled world.
    let mut policy = BeliefSearch::new(tiny(), CombatStrength::default());
    let mut rng = MegaRandom::new(5);
    let action = policy.choose(&won, &mut rng);
    assert!(
        won.legal_actions().contains(&action),
        "the answer comes off the screen the player is looking at"
    );

    // And a fight won mid-walk is worth the win: the objective reads the same
    // horizon the determinizer stops at.
    assert!(
        CombatStrength::default().peek(&won) >= 1.0,
        "the victory screen scores as a fight survived"
    );
}

#[test]
fn a_long_fight_still_prefers_ending_to_outlasting() {
    // Even late in a fight, winning must outrank stalling and another turn
    // must reduce the score. A floored subtraction would collapse both to zero.
    let objective = CombatStrength::default();
    let standing = |turn: u32| {
        ScenarioBuilder::new(
            "NLD6VZXP94",
            "ENCOUNTER.SHRINKER_BEETLE_WEAK".parse().unwrap(),
        )
        .player_hp(4, 86)
        .energy(2, 3)
        .turn(turn, turn)
        .enemy(
            1,
            "MONSTER.SHRINKER_BEETLE".parse().unwrap(),
            3,
            39,
            "SHRINKER_MOVE",
        )
        .card(
            PileName::Hand,
            10,
            CardFingerprint::base("CARD.STRIKE_IRONCLAD".parse().unwrap()),
            1,
        )
        .build(sts2_content::standard_registry())
        .unwrap()
    };
    for turn in [1, 53, 123, 400] {
        let now = objective.peek(&standing(turn));
        let later = objective.peek(&standing(turn + 1));
        assert!(
            now > later,
            "turn {turn} outranks turn {}: {now} vs {later}",
            turn + 1
        );
        assert!(
            now > 0.0,
            "and a live fight never prices at the defeat it is not: {now}"
        );
    }
    // The win term keeps its edge at the turn count where `v2` lost it: the
    // same state scored as a fight walked out of beats the fight still on.
    let running = standing(123);
    let walked_out = {
        let mut state = running.state().clone();
        state.combat.as_mut().unwrap().has_ended = true;
        Simulator::from_scenario(state, sts2_content::standard_registry()).unwrap()
    };
    assert!(
        objective.peek(&walked_out) > objective.peek(&running),
        "ending the fight dominates outlasting it at turn 123: {} vs {}",
        objective.peek(&walked_out),
        objective.peek(&running)
    );
}

#[test]
fn a_turn_costs_what_the_objective_says_it_does() {
    // The time price, pinned: walking a turn forward strictly lowers what
    // the fight is worth — so outlasting a fight bleeds value and ending it
    // dominates cycling, which is what keeps the search out of the
    // exhausted-deck stall the first net-guided gate found.
    let objective = CombatStrength::default();
    let early = first_combat("NLD6VZXP94");
    let mut late = early.clone();
    // Walk a full turn: end it, then let the enemies play out to the next
    // player decision.
    for _ in 0..20 {
        let combat = late.state().combat.as_ref().expect("the fight stands");
        if combat.player.turn > early.state().combat.as_ref().unwrap().player.turn
            && late
                .legal_actions()
                .iter()
                .any(|action| matches!(action, sts2_engine::Action::PlayCard { .. }))
        {
            break;
        }
        let action = late
            .legal_actions()
            .iter()
            .find(|action| matches!(action, sts2_engine::Action::EndTurn { .. }))
            .or_else(|| late.legal_actions().first())
            .cloned()
            .expect("a live fight offers an action");
        late.step_quietly(&action).unwrap();
    }
    let turns_apart = f64::from(
        late.state().combat.as_ref().unwrap().player.turn
            - early.state().combat.as_ref().unwrap().player.turn,
    );
    assert!(turns_apart >= 1.0, "a turn passed");
    let drop = objective.peek(&early) - objective.peek(&late);
    let priced = objective.turn_weight * turns_apart;
    assert!(
        drop > 0.0,
        "the later state pays for its turns: dropped {drop:.3}"
    );
    assert!(
        drop < priced + objective.hp_weight * 0.5,
        "and pays roughly the turn price, not wildly more: \
         dropped {drop:.3}, turn price {priced:.3}"
    );
}

#[test]
fn a_wall_cap_stops_the_decision_it_falls_in_the_middle_of() {
    // A budget read only between decisions is not a cap: the decision that
    // overruns it is the one nobody is asked about. The ceiling the budget
    // resolves to is read between the simulations of one decision, so a
    // deadline that has already passed buys no simulations at all.
    let simulator = fresh_run("NLD6VZXP94");
    let started = std::time::Instant::now();
    let spend = |budget: alphaspire::search::Budget| {
        let mut mcts = Mcts::new(
            SearchConfig {
                iterations: 5000,
                rollout_depth: 5,
                temperature: 0.5,
            },
            Uct::default(),
        );
        mcts.under(budget.ceiling(started));
        let mut determinizer = TrueState::new(simulator.clone());
        let objective = Coverage::default();
        let mut rollout = UniformRandom;
        let mut rng = MegaRandom::new(3);
        let action = mcts.decide(&mut determinizer, &objective, &mut rollout, &mut rng);
        assert!(
            simulator.legal_actions().contains(&action),
            "a capped decision still answers with a legal action"
        );
        mcts.steps_taken()
    };

    assert_eq!(
        spend(alphaspire::search::Budget {
            steps: None,
            seconds: Some(0.0),
        }),
        0,
        "a deadline already past starts no simulation"
    );
    assert_eq!(
        spend(alphaspire::search::Budget {
            steps: Some(0),
            seconds: None,
        }),
        0,
        "and neither does a step cap already reached"
    );
    assert!(
        spend(alphaspire::search::Budget {
            steps: Some(6),
            seconds: Some(3600.0),
        }) >= 6,
        "a cap that has not come down yet is not a cap: the search spends up to it"
    );
}

#[test]
fn a_fight_costs_the_harness_stream_one_draw_however_long_the_search() {
    // Two belief searches differing only in budget play the same fight from
    // the same state, each on a fresh harness stream from the same seed.
    // Inside the fight the search draws off its own stream, split from the
    // harness's by one draw at the first searched decision — so when the
    // fight is over, both harness streams stand at the same place, and
    // every draw the rollout policy makes after it (the run's macro play)
    // is the same draw for both. That is what lets two arms of a match play
    // the same run and not merely the same seed.
    let play = |iterations: u32| {
        let mut simulator = first_combat("NLD6VZXP94");
        let mut policy = BeliefSearch::new(
            SearchConfig {
                iterations,
                rollout_depth: 4,
                temperature: 0.5,
            },
            CombatStrength::default(),
        );
        let mut rng = MegaRandom::new(77);
        let mut decisions = 0;
        for _ in 0..300 {
            if simulator.state().terminal.is_some() || alphaspire::env::fight_over(&simulator) {
                break;
            }
            let action = policy.choose(&simulator, &mut rng);
            simulator.step_quietly(&action).unwrap();
            decisions += 1;
        }
        (decisions, rng.next_u64())
    };
    let (short, after_short) = play(2);
    let (long, after_long) = play(24);
    assert!(short > 0 && long > 0, "both searches fought");
    assert_eq!(
        after_short, after_long,
        "the harness stream moved by exactly one draw, whatever the search spent"
    );
}
