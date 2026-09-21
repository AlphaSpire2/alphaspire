//! The forced-win walk's contracts: the checked-in config parses and a bad
//! one is refused whole; a walk plays no combat and still banks fights at
//! depths the policy could never fight to, every entry stamped for the walk
//! that reached it; and a batch is a function of its seeds, byte-identical
//! at any job count.

use std::path::{Path, PathBuf};

use alphaspire::forcewins::ForceWins;
use alphaspire::heuristics::Heuristic;
use alphaspire::library::{FightOrigin, LibrarySink};
use alphaspire::objective::{CombatStrength, Objective};
use alphaspire::policy::RolloutPolicy;
use alphaspire::selfplay::{Batch, RunReport, play_batch};
use sts2_core::UnlockPresetManifest;

fn checked_in_config() -> ForceWins {
    ForceWins::load(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/configs/force-wins.toml"
    )))
    .expect("the checked-in config is the example that parses")
}

#[test]
fn a_config_is_refused_whole_on_anything_it_cannot_honor() {
    fn config(act_scale: &str, min_hp: &str, elite_span: &str, hallway_use: &str) -> String {
        format!(
            "act_scale = {act_scale}\nmin_hp = {min_hp}\n\
             [hp_loss.hallway]\nmin = 0.0\nmax = 0.1\n\
             [hp_loss.elite]\n{elite_span}\n\
             [hp_loss.boss]\nmin = 0.1\nmax = 0.5\n\
             [potion_use]\nhallway = {hallway_use}\nelite = 0.3\nboss = 0.6\n"
        )
    }
    let good = config("[1.0]", "1", "min = 0.1\nmax = 0.3", "0.05");
    let cases: [(String, &str); 4] = [
        // An unknown field is a typo'd knob, not a comment.
        (format!("typo = 1\n{good}"), "typo"),
        (
            config("[1.0]", "1", "min = 0.5\nmax = 0.2", "0.05"),
            "hp_loss.elite",
        ),
        (
            config("[1.0]", "1", "min = 0.1\nmax = 0.3", "1.5"),
            "potion_use.hallway",
        ),
        (
            config("[1.0]", "0", "min = 0.1\nmax = 0.3", "0.05"),
            "min_hp",
        ),
    ];
    for (index, (text, named)) in cases.into_iter().enumerate() {
        let path = std::env::temp_dir().join(format!(
            "alphaspire-forcewins-config-{index}-{}.toml",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        let message = ForceWins::load(&path).unwrap_err().to_string();
        assert!(message.contains(named), "case {index}: {message}");
        std::fs::remove_file(&path).ok();
    }
}

/// One forced-win batch banked to a fresh directory, answering the shard
/// bytes and the reports.
fn banked(name: &str, jobs: usize) -> (PathBuf, Vec<RunReport>) {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let config = checked_in_config();
    let batch = Batch {
        preset: &preset,
        character: &character,
        ascension: 0,
        analysis_seed: 77,
        seed: None,
        runs: 2,
        max_steps: 3000,
        jobs,
        harvest: Some(alphaspire::library::HarvestAs::Setup),
        force_wins: Some(&config),
    };
    let policy = || Box::new(Heuristic) as Box<dyn RolloutPolicy>;
    let objective = || Box::new(CombatStrength::default()) as Box<dyn Objective>;
    let directory = std::env::temp_dir().join(format!("alphaspire-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let mut sink = LibrarySink::create(&directory, 32).unwrap();
    sink.annotate("force_wins", config.echo());
    let mut reports = Vec::new();
    play_batch(&batch, &policy, &objective, &mut |_, played| {
        let mut report = played.expect("a walk that fights nothing loses nothing");
        sink.write_run(&mut report.fights).unwrap();
        reports.push(report);
    });
    sink.finish().unwrap();
    (directory, reports)
}

#[test]
fn a_forced_win_walk_banks_the_depths_no_policy_could_fight_to() {
    let (directory, reports) = banked("forcewins-walk", 1);
    for report in &reports {
        assert!(
            report.script.is_err(),
            "a forced-win walk says it cannot be scripted"
        );
        assert!(
            report.decisions.is_empty(),
            "no combat was played, so no combat decision exists"
        );
    }
    let fights: Vec<_> = reports.iter().flat_map(|report| &report.fights).collect();
    assert!(
        fights
            .iter()
            .all(|fight| fight.meta.origin == FightOrigin::ForcedWin),
        "every entry names the walk that reached it"
    );
    assert!(
        fights.iter().any(|fight| fight.meta.act >= 2),
        "the walk reaches the third act — the depth it exists for"
    );
    let entry_hp: std::collections::BTreeSet<i32> =
        fights.iter().map(|fight| fight.meta.entry_hp).collect();
    assert!(
        entry_hp.len() > 1,
        "the injected walk shows up in the banked entry hit points"
    );
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(directory.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["origins"]["natural"], serde_json::json!(0));
    assert!(manifest["force_wins"]["min_hp"].is_number());
    assert!(
        manifest["classes"]["act3"].as_u64().unwrap() > 0,
        "the act-three lane is nameable and fed"
    );
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn a_forced_win_batch_is_a_function_of_its_seeds_not_of_jobs() {
    let (serial, _) = banked("forcewins-serial", 1);
    let (parallel, _) = banked("forcewins-parallel", 2);
    for name in ["manifest.json", "entries-00000.jsonl"] {
        assert_eq!(
            std::fs::read(serial.join(name)).unwrap(),
            std::fs::read(parallel.join(name)).unwrap(),
            "{name} is a function of (seed, config), not of --jobs"
        );
    }
    std::fs::remove_dir_all(&serial).unwrap();
    std::fs::remove_dir_all(&parallel).unwrap();
}
