use std::process::Command;

#[test]
fn completions_prints_a_bash_completion_script() {
    let output = Command::new(env!("CARGO_BIN_EXE_alphaspire"))
        .args(["completions", "bash"])
        .output()
        .expect("alphaspire runs");

    assert!(output.status.success());
    assert!(
        String::from_utf8(output.stdout)
            .expect("completion output is UTF-8")
            .contains("_alphaspire")
    );
}

/// What the harness refuses to be told, and why each refusal exists.
fn refused(args: &[&str]) -> String {
    let (code, message) = ran(args);
    assert_ne!(code, Some(0), "{args:?} should have been refused");
    message
}

/// The exit code and stderr of one invocation.
fn ran(args: &[&str]) -> (Option<i32>, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_alphaspire"))
        .args(args)
        .output()
        .expect("alphaspire runs");
    (
        output.status.code(),
        String::from_utf8(output.stderr).expect("the refusal is UTF-8"),
    )
}

#[test]
fn macro_emission_takes_a_teacher_with_a_ranking_to_teach() {
    // Belief-mode macro emission clones the out-of-combat policy's own
    // scores. Uniform random has none, and the batch is refused up front
    // rather than writing an empty set.
    let out = std::env::temp_dir().join("alphaspire-refused-macro");
    let out = out.to_str().unwrap();
    let message = refused(&[
        "run",
        "--runs",
        "1",
        "--mode",
        "belief",
        "--policy",
        "random",
        "--emit-macro-samples",
        out,
    ]);
    assert!(
        message.contains("--policy heuristic"),
        "the refusal names the fix: {message}"
    );
    let message = refused(&[
        "run",
        "--runs",
        "1",
        "--mode",
        "true-state",
        "--emit-macro-samples",
        out,
    ]);
    assert!(
        message.contains("--mode act") || message.contains("--mode belief"),
        "and a clairvoyant search is refused outright: {message}"
    );
}

#[test]
fn the_act_tree_is_never_left_to_value_a_fight_by_playing_it_at_random() {
    // `--macro-net` without `--combat-net` would leave the act tree's fight-entry
    // leaves to the rollout policy, which plays fights at uniform random.
    // The nets are the evaluators, or the configuration is refused.
    let message = refused(&[
        "run",
        "--runs",
        "1",
        "--mode",
        "act",
        "--macro-net",
        "tests/fixtures/tiny",
    ]);
    assert!(
        message.contains("--combat-net"),
        "the refusal names what is missing: {message}"
    );
    let message = refused(&[
        "run",
        "--runs",
        "1",
        "--mode",
        "belief",
        "--macro-net",
        "tests/fixtures/tiny",
        "--combat-net",
        "tests/fixtures/tiny",
    ]);
    assert!(
        message.contains("--mode act"),
        "and the run net guides the act tree, which belief mode has none of: {message}"
    );
}

#[test]
fn a_mistyped_character_is_refused_once_before_anything_is_played() {
    // `ModelId` promises only CATEGORY.ENTRY, so CHARACTER.SILNET parses.
    // Unchecked it would fail once inside every run of the batch; checked,
    // it costs one refusal, and the refusal names what the ids are.
    for command in [vec!["run"], vec!["match", "combat"]] {
        let mut args = command.clone();
        args.extend(["--runs", "1", "--character", "CHARACTER.SILNET"]);
        if command[0] == "match" {
            args.extend(["--candidate-net", "tests/fixtures/tiny"]);
        }
        let (code, message) = ran(&args);
        assert_eq!(code, Some(2), "a bad argument is exit 2: {message}");
        assert!(
            message.contains("CHARACTER.SILENT") && message.contains("CHARACTER.IRONCLAD"),
            "the refusal lists the ids: {message}"
        );
    }
}

#[test]
fn run_accepts_a_lowercase_bare_character_name() {
    let (code, message) = ran(&["run", "--runs", "0", "--character", "silent"]);
    assert_eq!(
        code,
        Some(0),
        "a bare lowercase character is accepted: {message}"
    );
}

#[test]
fn run_names_replays_for_the_character_seed_and_ascension() {
    let directory =
        std::env::temp_dir().join(format!("alphaspire-replay-name-{}", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_alphaspire"))
        .args([
            "run",
            "--runs",
            "1",
            "--max-steps",
            "0",
            "--character",
            "silent",
            "--seed",
            "9MCU0647KN",
            "--ascension",
            "7",
            "--out",
        ])
        .arg(&directory)
        .output()
        .expect("alphaspire runs");

    assert!(
        output.status.success(),
        "zero-step search succeeds: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        directory.join("SILENT-9MCU0647KN-a7.sts2pgn").is_file(),
        "the replay name uses the canonical character name, seed, and ascension"
    );
    std::fs::remove_dir_all(directory).expect("test output is removed");
}

#[test]
fn a_bad_argument_and_a_file_that_would_not_open_exit_differently() {
    // Usage errors and file failures have distinct exit codes.
    let directory =
        std::env::temp_dir().join(format!("alphaspire-missing-input-{}", std::process::id()));
    assert!(!directory.exists(), "the test requires missing input files");
    let bank = directory.join("bank");
    let raw = directory.join("raw");
    let out = directory.join("out");
    let (usage, _) = ran(&["run", "--runs", "1", "--preset", "no-such-preset"]);
    assert_eq!(usage, Some(2));
    let (runtime, message) = ran(&[
        "fights",
        "--library",
        bank.to_str().unwrap(),
        "--mix",
        "boss=1",
    ]);
    assert_eq!(runtime, Some(3), "a file that would not open: {message}");
    let (runtime, message) = ran(&[
        "encode",
        "--decisions",
        raw.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(
        runtime,
        Some(3),
        "and so is a set that is not there: {message}"
    );
}

#[test]
fn match_does_not_take_the_two_arguments_it_could_not_honour() {
    // Both arms play every game, so one --out directory would take two
    // different scripts under one name and a bank would hold each entry
    // twice with nothing saying which arm reached it. The gate does not
    // offer either flag rather than accepting one and ignoring it.
    let out = std::env::temp_dir().join("alphaspire-match-refused");
    let out = out.to_str().unwrap();
    for flag in ["--out", "--harvest-fights"] {
        let message = refused(&[
            "match",
            "combat",
            "--candidate-net",
            "tests/fixtures/tiny",
            flag,
            out,
        ]);
        assert!(
            message.contains("unexpected argument"),
            "clap refuses {flag}: {message}"
        );
    }
}

/// A run-scoped checkpoint built out of the combat fixture: the same graph
/// and the same vocabulary, with the provenance a PPO export carries.
///
/// The two files are interchangeable on the wire and differ only in what
/// their provenance claims the value head predicts, which is exactly the
/// thing `load_run` checks — so a fixture that differs only there is the
/// honest way to exercise both sides of the refusal without a trained net.
fn run_checkpoint(name: &str) -> std::path::PathBuf {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let directory = std::env::temp_dir().join(format!("alphaspire-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("the fixture directory is made");
    let base = directory.join("run");
    std::fs::copy(fixtures.join("tiny.onnx"), base.with_extension("onnx"))
        .expect("the graph is copied");
    let mut provenance: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join("tiny.json")).unwrap())
            .expect("the fixture provenance parses");
    let object = provenance.as_object_mut().unwrap();
    object.insert("scope".into(), "macro".into());
    object.insert("value_semantics".into(), "run-return-v1".into());
    std::fs::write(base.with_extension("json"), format!("{provenance}\n"))
        .expect("the provenance is written");
    base
}

#[test]
fn a_rollout_refuses_a_checkpoint_pointed_at_the_wrong_flag() {
    // The run net and the combat net are the same bytes with different
    // provenance, so nothing about the file stops one being passed for the
    // other. Refused rather than reinterpreted: a combat checkpoint sampled
    // for macro actions, or an act-boundary critic used as a PPO one, would
    // answer plausible nonsense for a whole generation. And refused as a
    // *file that would not load* — exit 3, with a message — rather than as a
    // panic out of a worker.
    let run = run_checkpoint("rollout-wrong-flag");
    let (code, message) = ran(&[
        "run",
        "--runs",
        "1",
        "--run-net",
        "tests/fixtures/tiny",
        "--combat-net",
        "tests/fixtures/tiny",
    ]);
    assert_eq!(code, Some(3), "a checkpoint that would not load: {message}");
    assert!(
        message.contains("--run-net") && message.contains("scope"),
        "the refusal names the flag and what it wanted: {message}"
    );
    let (code, message) = ran(&[
        "run",
        "--runs",
        "1",
        "--run-net",
        run.to_str().unwrap(),
        "--combat-net",
        run.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(3), "and so does a macro-scoped resolver net");
    assert!(
        message.contains("--combat-net"),
        "naming the other flag: {message}"
    );
    std::fs::remove_dir_all(run.parent().unwrap()).expect("test output is removed");
}

#[test]
fn a_greedy_rollout_writes_the_script_its_walk_already_recorded() {
    // Nothing here replays the run: the walk that plays a PPO episode opens a
    // `ScriptWriter` like every other walk in the crate, so `--out` only keeps
    // what it wrote — under the same name a search batch gives it, because a
    // playback driver takes either one into the real game the same way.
    let run = run_checkpoint("rollout-script");
    let out = run.parent().unwrap().join("scripts");
    let (code, message) = ran(&[
        "run",
        "--runs",
        "1",
        "--max-steps",
        "200",
        "--seed",
        "9MCU0647KN",
        "--ascension",
        "7",
        "--greedy",
        "--resolver",
        "greedy",
        "--run-net",
        run.to_str().unwrap(),
        "--combat-net",
        "tests/fixtures/tiny",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "the episode walks: {message}");
    let script = std::fs::read_to_string(out.join("IRONCLAD-9MCU0647KN-a7.sts2pgn"))
        .expect("the episode's script is kept");
    assert!(
        script.contains("[Producer \"alphaspire/") && script.contains("[Mode \"CUSTOM\"]"),
        "and it is the decision-script profile a driver plays: {script:.200}"
    );
}

#[test]
fn a_run_can_copy_a_traces_run_configuration() {
    // `--like` reads the seed, character, ascension and the complete unlock
    // projection off a trace — a recording or a decision script — so the
    // copied run opens exactly the game the trace did.
    let run = run_checkpoint("run-like");
    let out = run.parent().unwrap().join("scripts");
    let (code, message) = ran(&[
        "run",
        "--runs",
        "1",
        "--max-steps",
        "0",
        "--character",
        "silent",
        "--seed",
        "QEY5K1P4LY",
        "--ascension",
        "3",
        "--runs-count",
        "84",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "the trace to copy is written: {message}");
    let trace = out.join("SILENT-QEY5K1P4LY-a3.sts2pgn");
    let copied = run.parent().unwrap().join("copied");
    let output = Command::new(env!("CARGO_BIN_EXE_alphaspire"))
        .args([
            "run",
            "--max-steps",
            "0",
            "--like",
            trace.to_str().unwrap(),
            "--greedy",
            "--resolver",
            "greedy",
            "--run-net",
            run.to_str().unwrap(),
            "--combat-net",
            "tests/fixtures/tiny",
            "--out",
            copied.to_str().unwrap(),
        ])
        .output()
        .expect("alphaspire runs");
    assert!(
        output.status.success(),
        "the copied run opens: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let script = std::fs::read_to_string(copied.join("SILENT-QEY5K1P4LY-a3.sts2pgn"))
        .expect("the copied configuration names the script");
    assert!(
        script.contains("\"number_of_runs\":84"),
        "the complete unlock projection is copied: {script:.400}"
    );
    std::fs::remove_dir_all(run.parent().unwrap()).expect("test output is removed");
}

#[test]
fn a_rollout_banks_the_fights_its_checkpoint_reached() {
    // Combat data does not come off a rollout's own resolver — it searches at
    // a deliberately low budget — but the *entries* it reaches are exactly
    // what the fight library wants, because the bank's distribution is the
    // reaching policy's. Harvested here, replayed at full budget by `fights`.
    let run = run_checkpoint("rollout-harvest");
    let bank = run.parent().unwrap().join("bank");
    let (code, message) = ran(&[
        "run",
        "--runs",
        "2",
        "--max-steps",
        "300",
        "--resolver",
        "greedy",
        "--run-net",
        run.to_str().unwrap(),
        "--combat-net",
        "tests/fixtures/tiny",
        "--harvest-fights",
        bank.to_str().unwrap(),
        "--harvest-as",
        "both",
    ]);
    assert_eq!(code, Some(0), "the batch walks: {message}");
    // Both halves stand up: the setup through the engine's combat start,
    // the exact state as the very fight that was banked.
    for from in ["setup", "state"] {
        let (code, message) = ran(&[
            "fights",
            "--library",
            bank.to_str().unwrap(),
            "--from",
            from,
            "--fights",
            "2",
            "--mix",
            "hallway=1.0",
            "--iterations",
            "4",
        ]);
        assert_eq!(
            code,
            Some(0),
            "and `fights --from {from}` honors what it banked: {message}"
        );
    }
    // Rewritten as setups alone, the bank loses its exact states and says
    // so to a consumer that asks for them.
    let setups = bank.with_file_name("bank-setups");
    let (code, message) = ran(&[
        "fights",
        "--library",
        bank.to_str().unwrap(),
        "--convert",
        setups.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "the bank converts: {message}");
    let (code, message) = ran(&[
        "fights",
        "--library",
        setups.to_str().unwrap(),
        "--from",
        "state",
        "--fights",
        "1",
        "--mix",
        "hallway=1.0",
        "--iterations",
        "4",
    ]);
    assert_ne!(
        code,
        Some(0),
        "a setup-only bank has no exact state: {message}"
    );
    assert!(message.contains("no exact state"), "{message}");
}

#[test]
fn a_rollout_can_walk_its_checkpoint_through_forced_wins() {
    // The forced-win walk under the run checkpoint: the checkpoint makes
    // every macro decision, every fight is banked at its entry and resolved
    // as won at the config's price, so the bank holds the checkpoint's own
    // decks at depths a natural batch reaches only through survival. The
    // walk is refused without the bank it exists for, and with anything
    // that would keep its unreplayable script or its skewed decisions.
    let run = run_checkpoint("rollout-forced-wins");
    let bank = run.parent().unwrap().join("bank");
    let config = concat!(env!("CARGO_MANIFEST_DIR"), "/configs/force-wins.toml");
    let common = [
        "run",
        "--runs",
        "2",
        "--max-steps",
        "3000",
        "--greedy",
        "--resolver",
        "greedy",
        "--run-net",
        run.to_str().unwrap(),
        "--combat-net",
        "tests/fixtures/tiny",
        "--force-wins",
        config,
    ];
    let message = refused(&common);
    assert!(
        message.contains("--harvest-fights"),
        "the walk exists for its bank: {message}"
    );
    let out = std::env::temp_dir().join("alphaspire-forced-refused");
    let out = out.to_str().unwrap();
    for (flag, want) in [
        ("--out", "not scriptable"),
        ("--emit-raw-ppo", "emit nothing"),
    ] {
        let mut args = common.to_vec();
        args.extend(["--harvest-fights", bank.to_str().unwrap(), flag, out]);
        let message = refused(&args);
        assert!(
            message.contains(want),
            "{flag} is refused for why: {message}"
        );
    }
    let mut args = common.to_vec();
    args.extend(["--harvest-fights", bank.to_str().unwrap()]);
    let (code, message) = ran(&args);
    assert_eq!(code, Some(0), "the walk banks: {message}");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(bank.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["origins"]["natural"], serde_json::json!(0));
    assert!(
        manifest["origins"]["forced_win"].as_u64().unwrap() > 0,
        "every entry names the walk that reached it: {manifest}"
    );
    assert!(
        manifest["force_wins"]["min_hp"].is_number(),
        "and the bank names the config that shaped it"
    );
    assert!(
        manifest["classes"]["act2"].as_u64().unwrap() > 0,
        "the checkpoint's walk reaches past the first act: {manifest}"
    );
    std::fs::remove_dir_all(run.parent().unwrap()).expect("test output is removed");
}

#[test]
fn a_rollout_refuses_a_resolver_budget_that_buys_nothing() {
    // Every refusal here is a flag combination that would otherwise be
    // accepted and quietly do the wrong thing: a greedy resolver has no
    // budget to spread, a search over no candidates has nothing to spread it
    // over, a search at zero iterations is the greedy resolver reached
    // expensively, and a greedy macro arm writing PPO shards would train a
    // ratio against a distribution nothing was drawn from.
    for (args, want) in [
        (
            vec!["--resolver", "greedy", "--resolver-iterations", "8"],
            "--resolver greedy",
        ),
        (
            vec!["--resolver", "greedy", "--resolver-considered", "4"],
            "--resolver greedy",
        ),
        (
            vec!["--resolver", "greedy", "--resolver-budget-steps", "100"],
            "--resolver greedy",
        ),
        (vec!["--resolver-considered", "0"], "at least one action"),
        (vec!["--resolver-iterations", "0"], "--resolver greedy"),
        (
            vec!["--resolver", "greedy", "--resolver-boss-iterations", "8"],
            "--resolver greedy",
        ),
        (
            vec!["--resolver-boss-iterations", "0"],
            "buys no simulation",
        ),
        (
            vec!["--resolver-elite-considered", "0"],
            "at least one action",
        ),
        (vec!["--policy", "heuristic"], "--policy names nothing"),
        (vec!["--greedy", "--emit-ppo", "shards"], "not on-policy"),
        (
            vec!["--greedy", "--emit-raw-ppo", "shards"],
            "not on-policy",
        ),
    ] {
        let mut line = vec![
            "run",
            "--runs",
            "1",
            "--run-net",
            "tests/fixtures/tiny",
            "--combat-net",
            "tests/fixtures/tiny",
        ];
        line.extend(args.iter().copied());
        let (code, message) = ran(&line);
        assert_eq!(code, Some(2), "a bad command line is exit 2: {message}");
        assert!(
            message.contains(want),
            "the refusal says which lever is idle: {message}"
        );
    }
}

#[test]
fn a_rollout_writes_trajectory_shards_a_ppo_trainer_will_accept() {
    // The three header keys are what the trainer dispatches its loss on, and
    // a batch that wrote shards missing one of them would be refused after
    // the generation had been paid for rather than before.
    let run = run_checkpoint("rollout-shards");
    let out = run.parent().unwrap().join("shards");
    let (code, message) = ran(&[
        "run",
        "--runs",
        "2",
        "--max-steps",
        "200",
        "--resolver",
        "greedy",
        "--run-net",
        run.to_str().unwrap(),
        "--combat-net",
        "tests/fixtures/tiny",
        "--shard-runs",
        "4",
        "--emit-ppo",
        out.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "the batch walks: {message}");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap())
            .expect("the manifest parses");
    assert_eq!(manifest["scope"], "macro");
    assert_eq!(manifest["value_semantics"], "run-return-v1");
    assert_eq!(manifest["training_mode"], "ppo");
    assert_eq!(manifest["runs"], 2, "one line-group per episode");
    let shard = std::fs::read_to_string(out.join("samples-00000.jsonl")).unwrap();
    let sample: serde_json::Value =
        serde_json::from_str(shard.lines().nth(1).expect("an episode recorded a step")).unwrap();
    for key in ["chosen", "logp", "value", "reward"] {
        assert!(sample.get(key).is_some(), "a trajectory line carries {key}");
    }
    std::fs::remove_dir_all(run.parent().unwrap()).expect("test output is removed");
}

#[test]
fn the_macro_gate_takes_two_run_checkpoints_and_one_frozen_resolver() {
    // A combat match's netless arm still searches, so uniform priors are a
    // real policy to beat; a greedy macro arm over uniform priors is "always
    // the first action on the screen", so there is no honest netless arm here
    // and the baseline is required rather than optional.
    let message = refused(&[
        "match",
        "macro",
        "--runs",
        "1",
        "--candidate-net",
        "tests/fixtures/tiny",
        "--combat-net",
        "tests/fixtures/tiny",
    ]);
    assert!(
        message.contains("--baseline-net"),
        "clap names the missing arm: {message}"
    );
    // And each checkpoint loads through the loader that names the net it is.
    let run = run_checkpoint("match-macro-wrong-flag");
    let (code, message) = ran(&[
        "match",
        "macro",
        "--runs",
        "1",
        "--candidate-net",
        "tests/fixtures/tiny",
        "--baseline-net",
        run.to_str().unwrap(),
        "--combat-net",
        "tests/fixtures/tiny",
    ]);
    assert_eq!(code, Some(3), "a checkpoint that would not load: {message}");
    assert!(
        message.contains("--candidate-net"),
        "the refusal names the flag: {message}"
    );
    std::fs::remove_dir_all(run.parent().unwrap()).expect("test output is removed");
}

#[test]
fn a_retired_spelling_names_the_one_that_replaced_it() {
    // No alias answers to an old command: the refusal names the new
    // spelling and nothing else, so a dated driver run against a new
    // binary is told exactly what changed. A name that was never a command
    // is still clap's own unrecognized-subcommand error.
    for (old, new) in [
        ("rollout", "run --run-net"),
        ("search", "run"),
        ("selfplay", "run"),
        ("match-macro", "match macro"),
        ("improve", "match self"),
        ("convert-fights", "fights --convert"),
        ("summarize", "analyze --no-net"),
        ("vocabulary", "encode --vocabulary"),
    ] {
        let (code, message) = ran(&[old, "--runs", "1"]);
        assert_eq!(code, Some(2), "{old} is a bad command line: {message}");
        assert!(
            message.contains(&format!("`{new}`")),
            "{old} names {new}: {message}"
        );
    }
    let (code, message) = ran(&["bogus"]);
    assert_eq!(code, Some(2));
    assert!(message.contains("unrecognized subcommand"), "{message}");
}

#[test]
fn a_run_takes_one_macro_and_only_that_macros_flags() {
    // The macro is chosen by what is named: `--run-net` plays the PPO
    // checkpoint, and without it the run is searched. A flag of the other
    // macro names nothing and is refused before anything is played — by
    // clap, naming the flag it wants or the flag it cannot sit beside.
    let message = refused(&["run", "--runs", "1", "--greedy"]);
    assert!(
        message.contains("--run-net"),
        "a run-checkpoint flag wants the run checkpoint: {message}"
    );
    let message = refused(&["run", "--runs", "1", "--resolver", "greedy"]);
    assert!(message.contains("--run-net"), "{message}");
    let message = refused(&[
        "run",
        "--runs",
        "1",
        "--run-net",
        "tests/fixtures/tiny",
        "--combat-net",
        "tests/fixtures/tiny",
        "--iterations",
        "8",
    ]);
    assert!(
        message.contains("--iterations") && message.contains("cannot be used"),
        "a search flag has no search to sit on under a run checkpoint: {message}"
    );
    let message = refused(&["run", "--runs", "1", "--run-net", "tests/fixtures/tiny"]);
    assert!(
        message.contains("--combat-net"),
        "and the run checkpoint needs its resolver's net: {message}"
    );
}

#[test]
fn encode_prints_the_vocabulary_the_checkpoints_hash() {
    let output = Command::new(env!("CARGO_BIN_EXE_alphaspire"))
        .args(["encode", "--vocabulary"])
        .output()
        .expect("alphaspire runs");
    assert!(output.status.success());
    let listing = String::from_utf8(output.stdout).expect("model ids are UTF-8");
    let lines: Vec<&str> = listing.lines().collect();
    assert!(lines.len() > 1000, "one model id per line: {}", lines.len());
    assert!(
        lines.iter().all(|line| line.contains('.')),
        "every line is a CATEGORY.ENTRY id"
    );
    // The listing is the vocabulary a checkpoint names by hash, so it
    // cannot be asked for beside an encoding.
    let out = std::env::temp_dir().join("alphaspire-vocab");
    let out = out.to_str().unwrap();
    let message = refused(&["encode", "--vocabulary", "--out", out]);
    assert!(message.contains("cannot be used"), "{message}");
}

#[test]
fn analyze_without_a_net_writes_the_trace_summary() {
    // A decision script is a trace too: it carries actions rather than
    // observation snapshots, so the summary replays them through the
    // simulator to recover the same run-level facts, loading no net.
    let out = std::env::temp_dir().join(format!("alphaspire-summary-{}", std::process::id()));
    let scripts = out.join("scripts");
    let (code, message) = ran(&[
        "run",
        "--runs",
        "1",
        "--max-steps",
        "40",
        "--iterations",
        "2",
        "--seed",
        "9MCU0647KN",
        "--out",
        scripts.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "the script is written: {message}");
    let script = scripts.join("IRONCLAD-9MCU0647KN-a0.sts2pgn");
    let output = Command::new(env!("CARGO_BIN_EXE_alphaspire"))
        .args(["analyze", "--no-net", "--out"])
        .arg(&out)
        .arg(&script)
        .output()
        .expect("alphaspire runs");
    assert!(
        output.status.success(),
        "the script replays: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("IRONCLAD-9MCU0647KN-a0.summary.json"))
            .expect("the summary is written where an analysis would go"),
    )
    .expect("the summary is JSON");
    assert_eq!(summary["trace_summary_format"], 1);
    assert_eq!(summary["seed"], "9MCU0647KN");
    assert_eq!(summary["source"], "script");
    assert!(
        summary["hp_curve"]
            .as_array()
            .is_some_and(|curve| !curve.is_empty())
    );
    let line = String::from_utf8(output.stdout).expect("the line is UTF-8");
    assert!(
        line.contains("summary.json"),
        "the line names the file: {line}"
    );
    std::fs::remove_dir_all(out).expect("test output is removed");
    // A summary loads no net, so it takes none.
    let message = refused(&[
        "analyze",
        "--no-net",
        "--combat-net",
        "tests/fixtures/tiny",
        "x",
    ]);
    assert!(message.contains("cannot be used"), "{message}");
}
