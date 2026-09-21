//! What a run-level step is paid.
//!
//! Five properties of [`RunReward`]: a run standing still owes nothing, a
//! climb pays per floor, a victory and a defeat each pay their term *on top
//! of* the floors already climbed, and a deep death therefore returns more
//! over the episode than a shallow one — which is the whole reason the
//! terminal term is added to the climb instead of replacing it. Then the
//! boss term: a boss killed pays its act's weight when the fight is left
//! alive, the final kill included, and the weights name themselves.

use alphaspire::reward::{
    GoldScope, RunReward, RunTerms, checkpoint_terms, resolve_run_terms, run_terms, value_semantics,
};
use sts2_core::UnlockPresetManifest;
use sts2_engine::{RunResult, Simulator};

fn fresh_run() -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at("NLD6VZXP94", &character, &preset, 0).unwrap()
}

/// The reward a run begins with, and the floor it begins on.
fn starting() -> (RunReward, u32) {
    let simulator = fresh_run();
    let floor = simulator.state().run.as_ref().unwrap().floor;
    (RunReward::starting(&simulator), floor)
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-9
}

#[test]
fn a_run_that_has_not_moved_is_owed_nothing() {
    // The baseline is the floor the run starts on, which is one rather than
    // zero: read against zero, the first macro decision of every episode
    // would be paid for a floor nobody climbed.
    let simulator = fresh_run();
    let mut reward = RunReward::starting(&simulator);
    assert!(
        close(reward.paid(&simulator), 0.0),
        "a state read twice with nothing between pays nothing"
    );
}

#[test]
fn a_climb_pays_the_floor_weight_for_every_floor() {
    let (mut reward, floor) = starting();
    let weight = reward.floor_weight;
    assert!(close(reward.priced(floor + 3, None), 3.0 * weight));
    assert!(
        close(reward.priced(floor + 4, None), weight),
        "each floor is paid once: the baseline moves with the climb"
    );
}

#[test]
fn a_victory_pays_its_term_on_top_of_the_floors_it_climbed() {
    let (mut reward, floor) = starting();
    let (climb, victory) = (reward.floor_weight, reward.victory_weight);
    let paid = reward.priced(floor + 2, Some(RunResult::Victory));
    assert!(close(paid, 2.0 * climb + victory), "paid {paid}");
    assert!(
        close(reward.priced(floor + 2, Some(RunResult::Victory)), 0.0),
        "a run ends once, so the terminal term is paid once"
    );
}

#[test]
fn a_defeat_pays_its_term_on_top_of_the_floors_it_climbed() {
    let (mut reward, floor) = starting();
    let (climb, defeat) = (reward.floor_weight, reward.defeat_weight);
    let paid = reward.priced(floor + 2, Some(RunResult::Defeat));
    assert!(close(paid, 2.0 * climb + defeat), "paid {paid}");
    assert!(
        defeat < 0.0,
        "a defeat costs rather than merely paying nothing"
    );
}

#[test]
fn a_deep_death_returns_more_over_the_episode_than_a_shallow_one() {
    // The episode return is the sum of what its steps were paid, so the
    // floors a losing run climbed have to survive its loss. Zeroing them
    // would make every death in a generation the same number, which is the
    // failure a dense reward exists to avoid.
    let died_at = |floor: u32| -> f64 {
        let (mut reward, start) = starting();
        let mut total = 0.0;
        for step in (start + 1)..=floor {
            total += reward.priced(step, None);
        }
        total + reward.priced(floor, Some(RunResult::Defeat))
    };
    let shallow = died_at(3);
    let deep = died_at(30);
    assert!(
        deep > shallow,
        "a death on floor 30 returns {deep}, a death on floor 3 returns {shallow}"
    );
}

#[test]
fn a_boss_killed_pays_the_weight_of_its_act_when_the_fight_is_left_alive() {
    let (mut reward, _) = starting();
    reward = reward.with_boss_terms([0.0, 3.0, 3.0]);
    // Standing in the act-two boss fight pays nothing yet; leaving it alive
    // pays the act's weight, once.
    assert!(close(reward.fought(false, Some(1), None), 0.0));
    assert!(close(reward.fought(false, Some(1), None), 0.0));
    assert!(close(reward.fought(false, None, None), 3.0));
    assert!(close(reward.fought(false, None, None), 0.0));
    // An act whose weight is zero is left for nothing.
    assert!(close(reward.fought(false, Some(0), None), 0.0));
    assert!(close(reward.fought(false, None, None), 0.0));
    // Dying in the fight pays nothing for it.
    assert!(close(reward.fought(false, Some(2), None), 0.0));
    assert!(close(
        reward.fought(false, None, Some(RunResult::Defeat)),
        0.0
    ));
}

#[test]
fn the_final_kill_is_paid_on_top_of_the_victory() {
    let (mut reward, floor) = starting();
    reward = reward.with_boss_terms([0.0, 3.0, 3.0]);
    assert!(close(reward.fought(false, Some(2), None), 0.0));
    // A won run has left its last fight: the state a victory leaves stands
    // in no combat, so the kill and the victory settle on the same step.
    let paid = reward.priced(floor, Some(RunResult::Victory))
        + reward.fought(false, None, Some(RunResult::Victory));
    assert!(close(paid, reward.victory_weight + 3.0), "paid {paid}");
}

#[test]
fn an_elite_and_a_boss_are_paid_independently() {
    let (mut reward, _) = starting();
    reward = reward
        .with_terms(0.25, 0.0, 0.0)
        .with_boss_terms([1.0, 1.0, 1.0]);
    assert!(close(reward.fought(true, None, None), 0.0));
    assert!(close(reward.fought(false, None, None), 0.25));
    assert!(close(reward.fought(false, Some(0), None), 0.0));
    assert!(close(reward.fought(false, None, None), 1.0));
}

#[test]
fn the_boss_term_names_itself_and_leaves_older_names_alone() {
    let none = [0.0; 3];
    assert_eq!(
        value_semantics(0.0, 0.0, 0.0, GoldScope::Any, none),
        "run-return-v1"
    );
    assert_eq!(
        value_semantics(0.25, 0.0, 0.0, GoldScope::Any, none),
        "run-return-v2;elite=0.25;relic=0"
    );
    assert_eq!(
        value_semantics(0.25, 0.0, 0.003, GoldScope::Any, none),
        "run-return-v2;elite=0.25;relic=0;gold=0.003"
    );
    assert_eq!(
        value_semantics(0.25, 0.0, 0.003, GoldScope::Rewards, none),
        "run-return-v2;elite=0.25;relic=0;gold=0.003;gold_scope=rewards"
    );
    assert_eq!(
        value_semantics(0.25, 0.0, 0.003, GoldScope::Any, [0.0, 3.0, 3.0]),
        "run-return-v2;elite=0.25;relic=0;gold=0.003;boss=0/3/3"
    );
    assert_eq!(
        value_semantics(0.0, 0.0, 0.0, GoldScope::Any, [0.0, 3.0, 3.0]),
        "run-return-v2;elite=0;relic=0;boss=0/3/3"
    );
    let (reward, _) = starting();
    assert_eq!(
        reward.with_boss_terms([0.0, 3.0, 3.0]).value_semantics(),
        "run-return-v2;elite=0;relic=0;boss=0/3/3"
    );
}

#[test]
fn run_terms_round_trip_and_refuse_the_unknown() {
    assert_eq!(
        run_terms("run-return-v1"),
        Some(RunTerms {
            elite: 0.0,
            relic: 0.0,
            gold: 0.0,
            gold_rewards_only: false,
            boss: [0.0; 3],
        })
    );
    assert_eq!(
        run_terms("run-return-v2;elite=0.25;relic=0;gold=0.003;gold_scope=rewards"),
        Some(RunTerms {
            elite: 0.25,
            relic: 0.0,
            gold: 0.003,
            gold_rewards_only: true,
            boss: [0.0; 3],
        })
    );
    assert_eq!(
        run_terms("run-return-v2;elite=0.25;relic=0;gold=0.003;boss=0/3/3"),
        Some(RunTerms {
            elite: 0.25,
            relic: 0.0,
            gold: 0.003,
            gold_rewards_only: false,
            boss: [0.0, 3.0, 3.0],
        })
    );
    assert_eq!(
        run_terms("run-return-v2;elite=0.25;relic=0;boss=0/3"),
        None,
        "a boss term names all three acts"
    );
    assert_eq!(run_terms("act-boundary-v2"), None);
}

#[test]
fn a_run_pays_its_checkpoints_reward_unless_the_flags_contradict_it() {
    let semantics = "run-return-v2;elite=0.25;relic=0;gold=0.003";
    let trained = run_terms(semantics).expect("a name this build writes");
    assert_eq!(
        resolve_run_terms(None, semantics, trained),
        Ok(trained),
        "no flags: the checkpoint's own reward"
    );
    assert_eq!(
        resolve_run_terms(Some(trained), semantics, trained),
        Ok(trained),
        "flags stating the checkpoint's reward assert it and change nothing"
    );
    let contradicting = RunTerms {
        elite: 0.5,
        ..trained
    };
    let refused = resolve_run_terms(Some(contradicting), semantics, trained)
        .expect_err("flags naming another reward are refused");
    assert!(
        refused.contains("run-return-v2;elite=0.5;relic=0;gold=0.003"),
        "{refused}"
    );
    assert!(refused.contains(semantics), "{refused}");
    let partial = RunTerms {
        gold: 0.0,
        ..trained
    };
    assert!(
        resolve_run_terms(Some(partial), semantics, trained).is_err(),
        "a half-stated reward names a different one and is refused too"
    );
    assert_eq!(
        resolve_run_terms(Some(RunTerms::NONE), "run-return-v1", RunTerms::NONE),
        Ok(RunTerms::NONE),
        "explicit zeros against the base reward are its statement"
    );
}

#[test]
fn checkpoint_terms_read_the_provenance_and_refuse_what_this_build_cannot_pay() {
    let dir = std::env::temp_dir().join(format!(
        "alphaspire-checkpoint-terms-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let write = |stem: &str, body: &str| {
        let path = dir.join(stem);
        std::fs::write(path.with_extension("json"), body).unwrap();
        path
    };
    let priced = write(
        "priced",
        r#"{"scope":"macro","value_semantics":"run-return-v2;elite=0.25;relic=0;gold=0.003;boss=0/3/3"}"#,
    );
    let (semantics, terms) = checkpoint_terms(&priced).expect("a run reward this build writes");
    assert_eq!(
        semantics,
        "run-return-v2;elite=0.25;relic=0;gold=0.003;boss=0/3/3"
    );
    assert_eq!(
        terms,
        run_terms(&semantics).unwrap(),
        "the terms are the name's own round trip"
    );
    assert_eq!(terms.scope(), GoldScope::Any);
    let boundary = write(
        "boundary",
        r#"{"scope":"macro","value_semantics":"act-boundary-v2"}"#,
    );
    let refused =
        checkpoint_terms(&boundary).expect_err("an act-boundary checkpoint pays no run reward");
    assert!(refused.contains("act-boundary-v2"), "{refused}");
    let unnamed = write("unnamed", r#"{"scope":"macro"}"#);
    assert!(
        checkpoint_terms(&unnamed).is_err(),
        "a provenance naming no semantics"
    );
    assert!(
        checkpoint_terms(&dir.join("missing")).is_err(),
        "no provenance at all"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
