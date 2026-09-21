//! The batch: seeds derived rather than drawn, and runs that may play at
//! once because nothing couples them.

use alphaspire::objective::{CombatStrength, Objective};
use alphaspire::policy::{FirstLegal, RolloutPolicy, UniformRandom};
use alphaspire::selfplay::{Batch, RunReport, play_batch, run_seeds};
use sts2_core::UnlockPresetManifest;

/// One batch walked under the given policy and job count, answering with each
/// run's seed and the line of play it took.
fn walk(
    jobs: usize,
    runs: usize,
    policy: &(dyn Fn() -> Box<dyn RolloutPolicy> + Sync),
) -> Vec<(String, String)> {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let batch = Batch {
        preset: &preset,
        character: &character,
        ascension: 0,
        analysis_seed: 4,
        seed: None,
        runs,
        max_steps: 25,
        jobs,
        harvest: None,
        force_wins: None,
    };
    let objective = || -> Box<dyn Objective> { Box::new(CombatStrength::default()) };
    let mut walked = Vec::new();
    play_batch(&batch, policy, &objective, &mut |index, played| {
        let report: RunReport = played.unwrap();
        assert_eq!(walked.len(), index, "reports arrive in run order");
        walked.push((report.seed, format!("{:?}", report.actions)));
    });
    walked
}

#[test]
fn a_run_is_a_function_of_its_index_and_nothing_else() {
    let (seed, analysis) = run_seeds(4, 7);
    assert_eq!(
        (seed.clone(), analysis),
        run_seeds(4, 7),
        "one pair, one run"
    );
    assert_ne!(
        run_seeds(4, 7).0,
        run_seeds(4, 8).0,
        "another index, another game"
    );
    assert_ne!(
        run_seeds(5, 7).0,
        run_seeds(4, 7).0,
        "another batch, another game"
    );
    // Zero-valued seeds and indices must still produce distinct games.
    assert_ne!(run_seeds(0, 0).0, run_seeds(0, 1).0, "no collapse at zero");
}

#[test]
fn what_the_policy_spends_no_longer_moves_the_batch_onto_other_games() {
    // Policy RNG consumption must not change the game seeds in a batch.
    let random = walk(
        1,
        4,
        &(|| Box::new(UniformRandom) as Box<dyn RolloutPolicy>),
    );
    let first = walk(1, 4, &(|| Box::new(FirstLegal) as Box<dyn RolloutPolicy>));
    let games: Vec<&String> = random.iter().map(|(seed, _)| seed).collect();
    let others: Vec<&String> = first.iter().map(|(seed, _)| seed).collect();
    assert_eq!(games, others, "both arms play the same four games");
    assert_ne!(
        random.iter().map(|(_, line)| line).collect::<Vec<_>>(),
        first.iter().map(|(_, line)| line).collect::<Vec<_>>(),
        "and still play them differently, which is what is being compared"
    );
}

#[test]
fn a_parallel_batch_plays_the_same_games_as_a_serial_one() {
    let policy = || Box::new(UniformRandom) as Box<dyn RolloutPolicy>;
    let serial = walk(1, 6, &policy);
    let parallel = walk(4, 6, &policy);
    assert_eq!(
        serial, parallel,
        "the jobs count is a throughput lever, not a seed"
    );
}

#[test]
fn a_run_that_panics_is_lost_loudly_and_the_batch_keeps_walking() {
    // The searches exist to walk into engine faults; a fault's panic must
    // cost its run, not the batch. The poisoned policy stands in for an
    // engine invariant tripping mid-run.
    struct Poisoned;
    impl RolloutPolicy for Poisoned {
        fn choose(
            &mut self,
            simulator: &sts2_engine::Simulator,
            rng: &mut sts2_rng::MegaRandom,
        ) -> sts2_engine::Action {
            let _ = (simulator, rng);
            panic!("an engine invariant tripped");
        }
    }
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let batch = Batch {
        preset: &preset,
        character: &character,
        ascension: 0,
        analysis_seed: 4,
        seed: None,
        runs: 3,
        max_steps: 10,
        jobs: 2,
        harvest: None,
        force_wins: None,
    };
    let poison_second = std::sync::atomic::AtomicUsize::new(0);
    let policy = move || -> Box<dyn RolloutPolicy> {
        // Runs claim policies in index order off the shared cursor, so the
        // second policy built belongs to run 1.
        if poison_second.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 1 {
            Box::new(Poisoned)
        } else {
            Box::new(UniformRandom)
        }
    };
    let objective = || -> Box<dyn Objective> { Box::new(CombatStrength::default()) };
    let mut outcomes = Vec::new();
    play_batch(&batch, &policy, &objective, &mut |index, played| {
        outcomes.push((
            index,
            played
                .map(|report| report.seed)
                .map_err(|error| error.to_string()),
        ));
    });
    assert_eq!(
        outcomes.len(),
        3,
        "every run reports, the lost one included"
    );
    let lost: Vec<_> = outcomes
        .iter()
        .filter(|(_, played)| played.is_err())
        .collect();
    assert_eq!(lost.len(), 1, "one poisoned run, one loss");
    let (index, error) = (lost[0].0, lost[0].1.as_ref().unwrap_err());
    assert!(
        error.contains(&format!("run {index} on ")) && error.contains("panicked"),
        "the loss names its run and seed — the reproducer: {error}"
    );
}

/// A hand-built report: the tally reads reports, it never plays.
fn report(floor: u32, act: usize, acts_cleared: usize, ending: Ending) -> RunReport {
    use sts2_engine::RunResult;
    let (terminal, death) = match ending {
        Ending::Unfinished => (None, None),
        Ending::Victory => (Some(RunResult::Victory), None),
        Ending::Death(encounter, room) => (
            Some(RunResult::Defeat),
            Some(alphaspire::selfplay::Death {
                encounter: Some(encounter.parse().unwrap()),
                room: Some(room),
            }),
        ),
    };
    RunReport {
        seed: "SEED".into(),
        actions: Vec::new(),
        floor,
        act,
        acts_cleared,
        terminal,
        death,
        reward: 0.0,
        script: Ok(String::new()),
        decisions: Vec::new(),
        macro_decisions: Vec::new(),
        budget: None,
        metrics: None,
        fights: Vec::new(),
    }
}

#[derive(Clone, Copy)]
enum Ending {
    Unfinished,
    Victory,
    Death(&'static str, sts2_engine::MapPointType),
}

#[test]
fn the_tally_counts_clears_and_names_the_killers() {
    use alphaspire::selfplay::OutcomeTally;
    use sts2_engine::MapPointType;
    let mut tally = OutcomeTally::default();
    // Two deaths to one elite on floors 5 and 7, one to a hallway fight,
    // one act-2 death past the cleared boss, a victory, and a stalled run.
    tally.record(&report(
        5,
        0,
        0,
        Ending::Death("MONSTER.NOB", MapPointType::Elite),
    ));
    tally.record(&report(
        7,
        0,
        0,
        Ending::Death("MONSTER.NOB", MapPointType::Elite),
    ));
    tally.record(&report(
        3,
        0,
        0,
        Ending::Death("MONSTER.CULTIST", MapPointType::Monster),
    ));
    tally.record(&report(
        18,
        1,
        1,
        Ending::Death("MONSTER.CULTIST", MapPointType::Monster),
    ));
    tally.record(&report(50, 3, 4, Ending::Victory));
    tally.record(&report(11, 0, 0, Ending::Unfinished));
    assert_eq!(tally.act1_clears(), 2, "the act-2 death and the victory");
    let summary = tally.summary();
    assert!(
        summary.contains("act 1 cleared 2/6 (33.3%)"),
        "the clear rate leads: {summary}"
    );
    assert!(
        summary.contains("1 victories") && summary.contains("1 unfinished"),
        "the other endings are counted: {summary}"
    );
    let table = tally.death_table(10);
    assert_eq!(table.len(), 3, "one row per killer per act: {table:?}");
    assert!(
        table[0].contains("MONSTER.NOB")
            && table[0].contains("elite")
            && table[0].contains("floors 5-7"),
        "the deadliest row leads, with its floor span: {}",
        table[0]
    );
    assert!(
        table
            .iter()
            .any(|line| line.contains("act 1") && line.contains("floor 18")),
        "the act-2 death keeps its act index: {table:?}"
    );
    let capped = tally.death_table(1);
    assert_eq!(capped.len(), 2, "a cap leaves one row and the remainder");
    assert!(
        capped[1].contains("2 deaths") && capped[1].contains("2 more rows"),
        "the remainder counts what the cap cut: {}",
        capped[1]
    );
}

#[test]
fn a_cleared_act_is_a_beaten_boss() {
    assert!(
        !report(16, 0, 0, Ending::Unfinished).cleared_act1(),
        "standing in act 1 is not through it"
    );
    assert!(
        report(16, 0, 1, Ending::Unfinished).cleared_act1(),
        "the transition screen counts its beaten boss"
    );
    assert!(
        report(17, 1, 1, Ending::Unfinished).cleared_act1(),
        "act 2 is past the boss"
    );
    assert!(
        report(50, 3, 4, Ending::Victory).cleared_act1(),
        "a won run cleared everything"
    );
}
