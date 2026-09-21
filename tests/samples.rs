//! The sample wire format and the batch's sample sink.
//!
//! Two properties, both about size rather than content: an observation is
//! written without its padding and comes back padded, and a batch's samples
//! reach disk as they are made instead of piling up in memory.

use alphaspire::encoding::{
    MAX_TOKENS, ObservationEncoding, PolicyEncoder, TOKEN_FEATURES, standard_vocabulary,
};
use alphaspire::training::{
    DECISION_FORMAT, Decision, DecisionSet, DecisionSink, SAMPLE_FORMAT, SCOPE_COMBAT, SCOPE_MACRO,
    SampleSink, TRAINING_MODE_PPO, TrainingSample,
};
use sts2_core::UnlockPresetManifest;
use sts2_engine::Simulator;

/// A character list to stamp a set with.
fn silent() -> [sts2_core::ModelId; 1] {
    ["CHARACTER.SILENT".parse().unwrap()]
}

fn encoder() -> PolicyEncoder {
    let registry = sts2_content::standard_registry();
    PolicyEncoder::new(standard_vocabulary(&registry), &registry)
}

/// A decision as a recorder would have kept it at the fixture's first
/// in-combat state: the observation, the canonical actions, a uniform π.
fn decision(simulator: &Simulator) -> Decision {
    let actions: Vec<alphaspire::plan::ActionPlan> =
        alphaspire::search::canonical_actions(simulator.legal_actions())
            .into_iter()
            .map(alphaspire::plan::ActionPlan::Single)
            .collect();
    #[allow(clippy::cast_precision_loss, reason = "a handful of actions")]
    let uniform = 1.0 / actions.len() as f32;
    Decision {
        observation: simulator.agent_observation(),
        pi: vec![uniform; actions.len()],
        actions,
        z: 0.25,
        encounter: Some("ENCOUNTER.EXORDIUM_HORDLINGS".parse().unwrap()),
        run: 0,
        fight: 0,
        act: None,
        chosen: None,
        logp: None,
        value: None,
        reward: None,
        done: false,
        bootstrap: None,
        degraded: false,
        counterfactual: false,
        counterfactual_kind: None,
        exploration_epsilon: None,
        q_spread: None,
        root_value: None,
    }
}

fn scratch(name: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!("alphaspire-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    directory
}

/// Every file of a sink's directory, name to bytes.
fn files(directory: &std::path::Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().to_string_lossy().into_owned(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

fn first_combat() -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let mut simulator =
        sts2_content::standard_run_on_preset_at("NLD6VZXP94", &character, &preset, 0).unwrap();
    for _ in 0..40 {
        if simulator.state().combat.is_some() {
            return simulator;
        }
        let action = simulator
            .legal_actions()
            .iter()
            .find(|action| matches!(action, sts2_engine::Action::ChooseCards { cards, .. } if !cards.is_empty()))
            .or_else(|| simulator.legal_actions().first())
            .cloned()
            .unwrap();
        simulator.step_quietly(&action).unwrap();
    }
    panic!("the seed reaches a combat inside forty decisions");
}

#[test]
fn an_observation_goes_out_ragged_and_comes_back_padded() {
    let simulator = first_combat();
    let encoded = encoder().encode_observation(&simulator.agent_observation());
    assert_eq!(encoded.tokens.len(), MAX_TOKENS, "in memory it is padded");
    assert_eq!(encoded.features.len(), MAX_TOKENS * TOKEN_FEATURES);

    let live = encoded.live_tokens();
    assert!(
        live > 0 && live < MAX_TOKENS,
        "the fixture decision fills some of the slots, not all: {live}"
    );

    let ragged = serde_json::to_string(&encoded).unwrap();
    let wire: serde_json::Value = serde_json::from_str(&ragged).unwrap();
    assert_eq!(
        wire["tokens"].as_array().unwrap().len(),
        live,
        "the padding does not go on the wire"
    );
    assert_eq!(
        wire["features"].as_array().unwrap().len(),
        live * TOKEN_FEATURES
    );

    let back: ObservationEncoding = serde_json::from_str(&ragged).unwrap();
    assert_eq!(back, encoded, "and the tensors are the ones that went out");

    // The saving is the whole reason: the padded form is more than nine
    // tenths zeroes, and a dense binary encoding would not have shrunk them.
    let padded = serde_json::to_string(&serde_json::json!({
        "scalars": encoded.scalars,
        "tokens": encoded.tokens,
        "features": encoded.features,
    }))
    .unwrap();
    assert!(
        ragged.len() * 5 < padded.len(),
        "ragged {} vs padded {}",
        ragged.len(),
        padded.len()
    );
}

#[test]
fn a_file_written_before_the_padding_was_dropped_still_reads() {
    // Format 1 wrote the full padded arrays. Padding an already-padded block
    // is a no-op, so those files parse unchanged.
    let simulator = first_combat();
    let encoded = encoder().encode_observation(&simulator.agent_observation());
    let dense = serde_json::to_string(&serde_json::json!({
        "scalars": encoded.scalars,
        "tokens": encoded.tokens,
        "features": encoded.features,
    }))
    .unwrap();
    let back: ObservationEncoding = serde_json::from_str(&dense).unwrap();
    assert_eq!(back, encoded);
}

#[test]
fn the_sink_shards_on_run_boundaries_and_counts_what_it_wrote() {
    let directory = scratch("sink");
    let simulator = first_combat();
    let sample = || decision(&simulator);

    let mut sink = SampleSink::create(&directory, &encoder(), 2, &[]).unwrap();
    // Five runs, two per shard, and one of them recorded nothing: the empty
    // run still spends its place, so shard boundaries stay at run indices.
    for run in 0..5 {
        let mut samples: Vec<Decision> = (0..run).map(|_| sample()).collect();
        sink.write_run(&mut samples).unwrap();
        assert!(
            samples.iter().all(|sample| sample.run == run),
            "the sink stamps run order"
        );
    }
    let total = sink.finish().unwrap();
    assert_eq!(total, 10, "every run's samples were written: 0+1+2+3+4");

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(directory.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["format"], SAMPLE_FORMAT);
    assert_eq!(manifest["runs"], 5);
    assert_eq!(manifest["samples"], 10);
    let shards = manifest["shards"].as_array().unwrap();
    assert_eq!(shards.len(), 3, "five runs, two per shard");
    assert_eq!(shards[0]["samples"], 1, "runs 0 and 1 hold one sample");
    assert_eq!(shards[2]["runs"], 1, "the last shard is the short one");

    // Every shard opens with the full header, so one alone is loadable, and
    // its line count is the manifest's word.
    for shard in shards {
        let path = directory.join(shard["file"].as_str().unwrap());
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        let header: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(header["max_tokens"], MAX_TOKENS);
        assert_eq!(header["vocabulary_hash"], *encoder().vocabulary_hash());
        assert_eq!(
            u64::try_from(lines.count()).unwrap(),
            shard["samples"].as_u64().unwrap(),
            "the manifest counts what the shard holds"
        );
    }
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn a_line_written_before_the_stamps_still_reads() {
    // Format 2 carried no provenance. Dropping the stamped fields from a
    // format 3 line is exactly what such a file looks like, and it parses:
    // the stamps read back absent, zero, and zero.
    let simulator = first_combat();
    let stamped = TrainingSample {
        observation: encoder().encode_observation(&simulator.agent_observation()),
        actions: Vec::new(),
        pi: vec![1.0],
        z: 0.25,
        encounter: Some("ENCOUNTER.EXORDIUM_HORDLINGS".parse().unwrap()),
        run: 3,
        fight: 1,
        act: None,
        chosen: None,
        logp: None,
        value: None,
        reward: None,
        done: false,
        bootstrap: None,
        degraded: false,
        counterfactual: false,
        exploration_epsilon: None,
        q_spread: None,
        root_value: None,
    };
    let mut wire: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&stamped).unwrap()).unwrap();
    let line = wire.as_object_mut().unwrap();
    for field in ["encounter", "run", "fight"] {
        line.remove(field).expect("format 3 writes the stamp");
    }
    let back: TrainingSample = serde_json::from_str(&wire.to_string()).unwrap();
    assert_eq!(back.encounter, None);
    assert_eq!((back.run, back.fight), (0, 0));
    assert_eq!(back.observation, stamped.observation);
}

#[test]
fn a_recorded_run_stamps_each_sample_with_its_fight() {
    // The recorder counts fights where it settles them, so a run's samples
    // carry (fight, encounter) pairs a reader can group by — no more
    // reconstructing fights by segmenting run-ordered samples.
    let mut policy = alphaspire::search::BeliefSearch::with_rollout(
        alphaspire::search::SearchConfig {
            iterations: 2,
            rollout_depth: 4,
            temperature: 0.5,
        },
        alphaspire::objective::CombatStrength::default(),
        Box::new(alphaspire::policy::UniformRandom),
    )
    .recording();
    let mut objective = alphaspire::objective::CombatStrength::default();
    let mut rng = sts2_rng::MegaRandom::new(9);
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let report = alphaspire::selfplay::play_run(
        "NLD6VZXP94",
        &character,
        0,
        &mut policy,
        &mut rng,
        &mut objective,
        400,
    )
    .unwrap();
    assert!(!report.decisions.is_empty(), "the fights were searched");
    assert!(
        report
            .decisions
            .iter()
            .all(|sample| sample.encounter.is_some()),
        "every sample names its encounter"
    );
    let fights: Vec<usize> = report.decisions.iter().map(|sample| sample.fight).collect();
    assert_eq!(fights[0], 0, "the run's first fight is fight zero");
    assert!(
        fights.windows(2).all(|pair| pair[1] >= pair[0]),
        "fight indices march with the run: {fights:?}"
    );
    assert!(
        *fights.last().unwrap() >= 1,
        "the walk crossed a fight boundary: {fights:?}"
    );
    // Samples of one fight share their encounter stamp.
    for pair in report.decisions.windows(2) {
        if pair[0].fight == pair[1].fight {
            assert_eq!(pair[0].encounter, pair[1].encounter);
        }
    }
}

#[test]
fn the_macro_sink_names_its_own_scope_and_its_own_value() {
    // The two nets' data never share a directory, and a reader can tell
    // which it is holding without reading a line: the header names the
    // scope and what `z` meant when it was scored.
    let directory = scratch("macro-sink");
    let simulator = first_combat();
    let recorded = Decision {
        encounter: None,
        act: Some(1),
        ..decision(&simulator)
    };
    let sample = recorded.encode(&encoder());
    let mut sink = SampleSink::create_macro(
        &directory,
        &encoder(),
        4,
        alphaspire::training::SOURCE_HEURISTIC,
        &[],
    )
    .unwrap();
    sink.write_run(&mut [recorded]).unwrap();
    assert_eq!(sink.finish().unwrap(), 1);

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(directory.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(
        manifest["format"], SAMPLE_FORMAT,
        "no format bump was needed"
    );
    assert_eq!(manifest["scope"], SCOPE_MACRO);
    // And which teacher answered the decisions, on the same idiom: a set
    // cloned from the heuristic is not a set the act search improved on.
    assert_eq!(
        manifest["source"],
        alphaspire::training::SOURCE_HEURISTIC,
        "the header names its teacher"
    );
    assert_eq!(
        manifest["macro_policy_temperature"],
        serde_json::json!(alphaspire::heuristics::MACRO_POLICY_TEMPERATURE),
        "and the sharpness the teacher's scores were read at"
    );
    assert_eq!(manifest["value_semantics"], "act-boundary-v2");
    assert_ne!(
        manifest["value_semantics"],
        serde_json::json!("combat-strength-v3"),
        "an act score is not a fight score"
    );

    // The act stamp rides on the macro line and on no other: a combat line
    // is byte-identical to what format 3 always wrote.
    let wire = serde_json::to_string(&sample).unwrap();
    assert!(wire.contains("\"act\":1"), "the macro line names its act");
    let combat = TrainingSample {
        act: None,
        ..sample
    };
    assert!(
        !serde_json::to_string(&combat).unwrap().contains("\"act\":"),
        "and an in-combat line carries no act key at all"
    );
    let back: TrainingSample = serde_json::from_str(&wire).unwrap();
    assert_eq!(back.act, Some(1));

    let combat_directory = scratch("combat-sink");
    let sink = SampleSink::create(&combat_directory, &encoder(), 4, &[]).unwrap();
    assert_eq!(sink.finish().unwrap(), 0);
    let combat_manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(combat_directory.join("manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(combat_manifest["scope"], SCOPE_COMBAT);
    assert_eq!(combat_manifest["value_semantics"], "combat-strength-v3");

    std::fs::remove_dir_all(&directory).unwrap();
    std::fs::remove_dir_all(&combat_directory).unwrap();
}

#[test]
fn a_recorded_set_encodes_to_the_shards_the_batch_would_have_written() {
    // The same runs through both sinks — across a shard boundary, with one
    // run that recorded nothing — and the recorded set encoded afterwards
    // is the direct sink's output byte for byte: shard names, headers, run
    // stamps, manifest. What `--emit-raw` keeps is the search; what
    // `encode` adds is exactly what `--emit-samples` would have.
    let simulator = first_combat();
    let direct = scratch("direct");
    let raw = scratch("raw");
    let mut samples = SampleSink::create(&direct, &encoder(), 2, &silent()).unwrap();
    let mut decisions = DecisionSink::create(&raw, 2, &silent()).unwrap();
    for (index, count) in [2_usize, 0, 1, 3].into_iter().enumerate() {
        let mut recorded: Vec<Decision> = (0..count)
            .map(|fight| Decision {
                fight,
                #[allow(clippy::cast_precision_loss, reason = "a handful of fights")]
                z: 0.5 - fight as f32 / 10.0,
                ..decision(&simulator)
            })
            .collect();
        samples.write_run(&mut recorded.clone()).unwrap();
        decisions.write_run(&mut recorded).unwrap();
        assert!(
            recorded.iter().all(|decision| decision.run == index),
            "both sinks stamp the run index"
        );
    }
    assert_eq!(samples.finish().unwrap(), 6);
    assert_eq!(decisions.finish().unwrap(), 6);

    let set = DecisionSet::open(&raw).unwrap();
    assert_eq!(
        (set.runs(), set.decisions(), set.runs_per_shard()),
        (4, 6, 2)
    );
    assert_eq!(set.scope(), SCOPE_COMBAT);
    assert_eq!(set.characters(), silent(), "the set says whose decisions");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(raw.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["format"], DECISION_FORMAT);
    assert_eq!(
        manifest["characters"],
        serde_json::json!(["CHARACTER.SILENT"])
    );
    assert!(
        manifest.get("policy_encoding_version").is_none(),
        "a recorded set is pinned to no encoding"
    );
    assert_eq!(
        manifest["observation_version"],
        sts2_engine::AGENT_OBSERVATION_VERSION
    );

    let encoded = scratch("encoded");
    let mut sink = SampleSink::create_for(&encoded, &encoder(), &set).unwrap();
    let mut runs = 0;
    set.for_each_run(|mut run| {
        runs += 1;
        sink.write_run(&mut run)
    })
    .unwrap();
    assert_eq!(runs, 4, "the empty run is handed over too");
    assert_eq!(sink.finish().unwrap(), 6);
    assert_eq!(
        files(&encoded),
        files(&direct),
        "the encoded set is the direct set"
    );

    for directory in [&direct, &raw, &encoded] {
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn a_recorded_set_from_another_observation_version_is_refused() {
    let raw = scratch("foreign");
    DecisionSink::create(&raw, 1, &[])
        .unwrap()
        .finish()
        .unwrap();
    let path = raw.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    manifest["observation_version"] = serde_json::json!(sts2_engine::AGENT_OBSERVATION_VERSION - 1);
    std::fs::write(&path, manifest.to_string()).unwrap();
    let error = DecisionSet::open(&raw).err().expect("refused");
    assert!(
        error.to_string().contains("observation_version"),
        "the refusal names the pin: {error}"
    );
    std::fs::remove_dir_all(&raw).unwrap();
}

#[test]
fn a_searched_batch_records_and_encodes_the_same_shards_it_emits() {
    // End to end through the harness: one batch, both sinks, then the
    // recorded set encoded — the direct set again, byte for byte.
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character: sts2_core::ModelId = "CHARACTER.IRONCLAD".parse().unwrap();
    let make_policy = || -> Box<dyn alphaspire::policy::RolloutPolicy> {
        Box::new(
            alphaspire::search::BeliefSearch::new(
                alphaspire::search::SearchConfig {
                    iterations: 2,
                    rollout_depth: 3,
                    temperature: 0.5,
                },
                alphaspire::objective::CombatStrength::default(),
            )
            .recording(),
        )
    };
    let make_objective = || -> Box<dyn alphaspire::objective::Objective> {
        Box::new(alphaspire::objective::CombatStrength::default())
    };
    let direct = scratch("batch-direct");
    let raw = scratch("batch-raw");
    let mut samples =
        SampleSink::create(&direct, &encoder(), 2, std::slice::from_ref(&character)).unwrap();
    let mut decisions = DecisionSink::create(&raw, 2, std::slice::from_ref(&character)).unwrap();
    let mut recorded = 0;
    alphaspire::selfplay::play_batch(
        &alphaspire::selfplay::Batch {
            preset: &preset,
            character: &character,
            ascension: 0,
            analysis_seed: 3,
            seed: None,
            runs: 3,
            max_steps: 30,
            jobs: 3,
            harvest: None,
            force_wins: None,
        },
        &make_policy,
        &make_objective,
        &mut |_, played| {
            let mut report = played.unwrap();
            recorded += report.decisions.len();
            samples.write_run(&mut report.decisions.clone()).unwrap();
            decisions.write_run(&mut report.decisions).unwrap();
        },
    );
    assert!(recorded > 0, "the batch fought");
    assert_eq!(samples.finish().unwrap(), recorded);
    assert_eq!(decisions.finish().unwrap(), recorded);

    let set = DecisionSet::open(&raw).unwrap();
    let encoded = scratch("batch-encoded");
    let mut sink = SampleSink::create_for(&encoded, &encoder(), &set).unwrap();
    set.for_each_run(|mut run| sink.write_run(&mut run))
        .unwrap();
    assert_eq!(sink.finish().unwrap(), recorded);
    assert_eq!(files(&encoded), files(&direct));

    for directory in [&direct, &raw, &encoded] {
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn a_set_that_names_no_characters_encodes_to_what_this_build_always_wrote() {
    // The stamp is omitted rather than written empty where the batch did
    // not know, so a set recorded before it existed re-encodes byte for
    // byte: records written before the stamp stay readable, and stay the
    // same when read.
    let simulator = first_combat();
    let raw = scratch("unstamped-raw");
    let mut decisions = DecisionSink::create(&raw, 2, &[]).unwrap();
    decisions.write_run(&mut [decision(&simulator)]).unwrap();
    decisions.finish().unwrap();

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(raw.join("manifest.json")).unwrap()).unwrap();
    assert!(manifest.get("characters").is_none(), "absent, not empty");

    let set = DecisionSet::open(&raw).expect("an unstamped set still opens");
    assert!(set.characters().is_empty(), "which reads as does not say");
    let encoded = scratch("unstamped-encoded");
    let mut sink = SampleSink::create_for(&encoded, &encoder(), &set).unwrap();
    set.for_each_run(|mut run| sink.write_run(&mut run))
        .unwrap();
    sink.finish().unwrap();
    let shard = std::fs::read_to_string(encoded.join("samples-00000.jsonl")).unwrap();
    let header: serde_json::Value = serde_json::from_str(shard.lines().next().unwrap()).unwrap();
    assert!(header.get("characters").is_none());

    for directory in [&raw, &encoded] {
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn format_four_puts_the_trajectory_keys_on_a_trajectory_line_and_nowhere_else() {
    // The bump is additive: every added key is omitted where it does not
    // apply, so an in-combat line and a macro-imitation line are byte for
    // byte what format 3 wrote — and a reader that knows only format 3 still
    // parses a trajectory line, reading the keys it does not know as absent.
    let simulator = first_combat();
    let searched = decision(&simulator).encode(&encoder());
    let wire = serde_json::to_string(&searched).unwrap();
    for key in [
        "chosen",
        "logp",
        "exploration_epsilon",
        "value",
        "reward",
        "done",
        "bootstrap",
        "degraded",
    ] {
        assert!(
            !wire.contains(&format!("\"{key}\"")),
            "a searched line carries no {key}"
        );
    }

    let recorded = Decision {
        chosen: Some(1),
        logp: Some(-0.75),
        value: Some(0.5),
        reward: Some(0.1),
        done: true,
        degraded: true,
        exploration_epsilon: Some(0.25),
        ..decision(&simulator)
    };
    let line = serde_json::to_string(&recorded.encode(&encoder())).unwrap();
    let back: TrainingSample = serde_json::from_str(&line).unwrap();
    assert_eq!(back.chosen, Some(1));
    assert_eq!(back.logp, Some(-0.75));
    assert_eq!(back.exploration_epsilon, Some(0.25));
    assert_eq!(back.value, Some(0.5));
    assert_eq!(back.reward, Some(0.1));
    assert!(
        back.done && back.degraded,
        "the flags survive the round trip"
    );
    assert_eq!(
        back.bootstrap, None,
        "a step that ended the episode needs no bootstrap"
    );
    assert!(
        !line.contains("\"bootstrap\""),
        "and does not write the key at all"
    );

    // A trajectory line read by a build that never heard of the keys is the
    // same line with them dropped, which is exactly a format 3 line.
    let mut wire: serde_json::Value = serde_json::from_str(&line).unwrap();
    let object = wire.as_object_mut().unwrap();
    for key in [
        "chosen",
        "logp",
        "exploration_epsilon",
        "value",
        "reward",
        "done",
        "degraded",
    ] {
        object.remove(key).expect("format 4 writes the key");
    }
    let plain: TrainingSample = serde_json::from_str(&wire.to_string()).unwrap();
    assert_eq!(plain.chosen, None);
    assert_eq!(plain.exploration_epsilon, None);
    assert!(!plain.done && !plain.degraded);
    assert!(
        (plain.z - searched.z).abs() < f32::EPSILON,
        "and the rest of the line is unmoved"
    );
}

#[test]
fn the_ppo_sink_names_the_three_keys_a_trajectory_set_is_known_by() {
    // A PPO set and an imitation set share a scope and differ in every way
    // that matters to a loss. The header is where a reader settles which it
    // holds before parsing a line: `scope` says which net, `value_semantics`
    // says what that net's critic predicts, and `training_mode` says whether
    // a line teaches a distribution or an action actually drawn from one.
    let directory = scratch("ppo-sink");
    let simulator = first_combat();
    let recorded = Decision {
        encounter: None,
        act: Some(0),
        chosen: Some(0),
        logp: Some(-1.5),
        value: Some(0.25),
        reward: Some(0.1),
        ..decision(&simulator)
    };
    let mut sink = SampleSink::create_ppo(&directory, &encoder(), 4, &silent()).unwrap();
    sink.write_run(&mut [recorded]).unwrap();
    assert_eq!(sink.finish().unwrap(), 1);

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(directory.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["scope"], SCOPE_MACRO);
    assert_eq!(manifest["value_semantics"], "run-return-v1");
    assert_eq!(manifest["training_mode"], TRAINING_MODE_PPO);
    assert!(
        manifest.get("source").is_none(),
        "a PPO line's teacher is the checkpoint that acted, which `source` does not name"
    );
    // The shard opens with the same header, so one shard alone says what
    // loss it may be read under.
    let shard = std::fs::read_to_string(directory.join("samples-00000.jsonl")).unwrap();
    let header: serde_json::Value = serde_json::from_str(shard.lines().next().unwrap()).unwrap();
    assert_eq!(header["training_mode"], TRAINING_MODE_PPO);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn an_expert_iteration_set_carries_no_training_mode_at_all() {
    // The key is absent rather than written with some other value, so every
    // shard this build wrote before PPO existed is byte-identical to what it
    // writes now — and a trainer that requires the key refuses those sets
    // instead of reading them as some default.
    let directory = scratch("no-training-mode");
    let simulator = first_combat();
    let mut sink = SampleSink::create_macro(
        &directory,
        &encoder(),
        4,
        alphaspire::training::SOURCE_HEURISTIC,
        &silent(),
    )
    .unwrap();
    sink.write_run(&mut [decision(&simulator)]).unwrap();
    sink.finish().unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(directory.join("manifest.json")).unwrap())
            .unwrap();
    assert!(manifest.get("training_mode").is_none());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn encoding_a_recorded_trajectory_keeps_it_a_trajectory_set() {
    // The search is paid once and the encoding is cheap, which is why a
    // rollout may record raw and encode later. What must survive that trip
    // is the header: an encoded set whose `training_mode` had been dropped
    // would be an imitation set on the wire, and the surrogate would form
    // ratios against log-probabilities it had thrown away the meaning of.
    let raw = scratch("ppo-raw");
    let encoded = scratch("ppo-encoded");
    let simulator = first_combat();
    let recorded = Decision {
        encounter: None,
        act: Some(0),
        chosen: Some(0),
        logp: Some(-1.5),
        value: Some(0.25),
        reward: Some(0.1),
        done: true,
        ..decision(&simulator)
    };
    let mut sink = DecisionSink::create_ppo(&raw, 4, &silent()).unwrap();
    sink.write_run(&mut [recorded]).unwrap();
    sink.finish().unwrap();

    let set = DecisionSet::open(&raw).unwrap();
    assert_eq!(set.training_mode(), Some(TRAINING_MODE_PPO));
    assert_eq!(set.value_semantics(), "run-return-v1");
    let mut sink = SampleSink::create_for(&encoded, &encoder(), &set).unwrap();
    set.for_each_run(|mut run| sink.write_run(&mut run))
        .unwrap();
    assert_eq!(sink.finish().unwrap(), 1);

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(encoded.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["scope"], SCOPE_MACRO);
    assert_eq!(manifest["value_semantics"], "run-return-v1");
    assert_eq!(manifest["training_mode"], TRAINING_MODE_PPO);
    let shard = std::fs::read_to_string(encoded.join("samples-00000.jsonl")).unwrap();
    let sample: TrainingSample = serde_json::from_str(shard.lines().nth(1).unwrap()).unwrap();
    assert_eq!(sample.chosen, Some(0));
    assert_eq!(sample.logp, Some(-1.5));
    assert!(sample.done, "and the trajectory keys came through with it");
    for directory in [raw, encoded] {
        std::fs::remove_dir_all(directory).unwrap();
    }
}
