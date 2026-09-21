//! The network-free summary of a trace, witnessed by a script the test
//! writes itself: the floor curve and act breakdown follow the replay, the
//! run spec reproduces the starting world, and a resumed trace is summarized
//! through its reload.

use std::path::PathBuf;

use alphaspire::objective::Coverage;
use alphaspire::policy::UniformRandom;
use alphaspire::selfplay::RunReport;
use alphaspire::trace_summary::{TRACE_SUMMARY_FORMAT, run_spec, summarize_trace};
use sts2_core::UnlockPresetManifest;
use sts2_engine::RunResult;
use sts2_rng::MegaRandom;

const SEED: &str = "QEY5K1P4LY";

/// A random walk on the pinned profile, and the script it wrote.
fn scripted_run(steps: usize) -> (RunReport, String) {
    let mut coverage = Coverage::default();
    let mut rng = MegaRandom::new(21);
    let mut policy = UniformRandom;
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let report = alphaspire::selfplay::play_run(
        SEED,
        &character,
        0,
        &mut policy,
        &mut rng,
        &mut coverage,
        steps,
    )
    .unwrap();
    let script = report
        .script
        .clone()
        .expect("a random walk stays scriptable");
    (report, script)
}

fn written(name: &str, text: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "alphaspire-trace-summary-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join(format!("{SEED}.sts2pgn"));
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn a_script_yields_a_floor_curve_and_act_breakdown_that_follow_its_replay() {
    let (report, script) = scripted_run(400);
    let summary = summarize_trace(&written("curve", &script)).expect("the script summarizes");
    assert_eq!(summary.trace_summary_format, TRACE_SUMMARY_FORMAT);
    assert_eq!(summary.source, "script");
    assert!(
        !summary.verified,
        "a script carries no observations to verify against"
    );
    assert_eq!(summary.seed, SEED);
    assert_eq!(summary.character, "CHARACTER.IRONCLAD");
    assert_eq!(summary.ascension, 0);

    assert_eq!(summary.floor_reached, report.floor);
    assert!(
        summary.floor_reached >= 2,
        "the walk went deep enough to witness a curve: floor {}",
        summary.floor_reached
    );
    assert_eq!(summary.act_reached, report.act + 1);
    let expected = match report.terminal {
        Some(RunResult::Victory) => "victory",
        Some(RunResult::Defeat) => "defeat",
        None => "unfinished",
    };
    assert_eq!(summary.result, expected);

    let last = summary.hp_curve.last().expect("the walk visited a floor");
    assert_eq!(last.floor, report.floor);
    assert!(
        summary
            .hp_curve
            .windows(2)
            .all(|pair| pair[0].floor < pair[1].floor),
        "one point per floor, in order"
    );
    if report.terminal == Some(RunResult::Defeat) {
        assert_eq!(last.hp, 0, "a defeat ends at zero");
    }

    assert_eq!(summary.acts.len(), summary.act_reached);
    let floors: usize = summary.acts.iter().map(|act| act.floors_visited).sum();
    assert_eq!(
        floors,
        summary.hp_curve.len(),
        "every floor lands in one act"
    );
    assert_eq!(summary.acts[0].hp_start, summary.hp_curve[0].hp);
}

#[test]
fn a_script_exposes_the_configuration_a_rollout_needs() {
    let (_, script) = scripted_run(40);
    let spec = run_spec(&written("spec", &script)).expect("the script has a run spec");
    assert_eq!(spec.seed, SEED);
    assert_eq!(spec.character.to_string(), "CHARACTER.IRONCLAD");
    assert_eq!(spec.ascension, 0);
    let pinned = UnlockPresetManifest::pinned().unwrap();
    assert_eq!(spec.preset.preset_id, pinned.preset_id);
    assert_eq!(spec.preset.unlocks, pinned.unlocks);
}

#[test]
fn a_resumed_trace_is_summarized_through_its_reload() {
    // The script with a `run.resume` record spliced in after `run.start`,
    // the records renumbered so the sequence stays consecutive.
    let (_, script) = scripted_run(40);
    let is_record = |line: &str| {
        line.split_once(' ')
            .is_some_and(|(head, _)| head.parse::<u64>().is_ok())
    };
    let records = script.lines().filter(|line| is_record(line)).count();
    let mut sequence = 0;
    let mut lines = Vec::new();
    for line in script.lines() {
        let Some((_, rest)) = line.split_once(' ').filter(|_| is_record(line)) else {
            lines.push(line.to_owned());
            continue;
        };
        sequence += 1;
        lines.push(format!("{sequence} {rest}"));
        if rest.starts_with("run.start ") {
            sequence += 1;
            lines.push(format!(
                "{sequence} run.resume {{\"actor\":null,\"data\":{{}}}}"
            ));
        }
    }
    assert_eq!(sequence, records + 1, "one record was spliced in");
    let resumed = summarize_trace(&written("resumed", &lines.join("\n")))
        .expect("a reload is replayed, not refused");
    let plain = summarize_trace(&written("unresumed", &script)).unwrap();
    assert_eq!(resumed.floor_reached, plain.floor_reached);
    assert_eq!(resumed.result, plain.result);
    assert_eq!(resumed.hp_curve.len(), plain.hp_curve.len());
}
