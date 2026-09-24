//! Equivalent analyses of synthetic observations stored full, delta, or gzip.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use alphaspire::actor::TierBudget;
use alphaspire::analyze::{Nets, Settings, analyze_trace};
use alphaspire::trace_summary::{run_spec, summarize_trace};
use flate2::{Compression, write::GzEncoder};
use serde_json::{Value, json};
use sha2::Digest as _;
use sts2_replay::{Event, Parser, Record, ReplayAdapter, observation, projection_digest};

const SEED: &str = "NLD6VZXP94";

fn nets() -> Nets {
    let registry = sts2_content::standard_registry();
    let encoder = Arc::new(alphaspire::encoding::PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ));
    let stem = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    Nets {
        combat: Arc::new(alphaspire::net::PolicyValueNet::load(&stem, encoder).unwrap()),
        combat_stem: stem,
        run: None,
    }
}

fn settings() -> Settings {
    let tier = TierBudget {
        iterations: 1,
        considered: 1,
    };
    Settings {
        budgets: alphaspire::analyze::Budgets {
            hallway: tier,
            elite: tier,
            boss: tier,
            deep: tier,
            deep_top: 0,
            deep_cutoff: 0.1,
            min_visits: 1,
        },
        net_only: true,
        playouts: false,
        lookahead: false,
        full_observation: false,
        analysis_seed: 7,
        return_discount: 1.0,
        threads: 1,
    }
}

fn append(records: &mut Vec<Record>, kind: &str, payload: Value) {
    records.push(Record {
        sequence: records.len() as u64 + 1,
        kind: kind.into(),
        payload,
    });
}

fn observe(records: &mut Vec<Record>, state: &Value, context: &str) {
    append(
        records,
        "state.observe",
        json!({"state":state, "after":records.len(),
        "context":context, "projection_version":1, "digest":projection_digest(state).unwrap()}),
    );
}

/// Walk into a fight, take two unsaved actions, then reload its own save.
/// The final two observations are identical; the latter can be an empty delta.
fn scenario(divergent: bool) -> (String, Vec<Record>) {
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let mut policy = alphaspire::policy::UniformRandom;
    let report = alphaspire::selfplay::play_run(
        SEED,
        &character,
        0,
        &mut policy,
        &mut sts2_rng::MegaRandom::new(21),
        &mut alphaspire::objective::Coverage::default(),
        100,
    )
    .unwrap();
    let script = report.script.unwrap();
    let header = script.split_once("\n\n").unwrap().0;
    let header = header
        .lines()
        .filter(|line| !line.starts_with("[Producer "))
        .collect::<Vec<_>>()
        .join("\n");
    let header = format!(
        "{header}\n[RecorderVersion \"0.10.0\"]\n[RunId \"synthetic\"]\n[Mods \"[]\"]\n[Result \"*\"]\n\n"
    );
    let preset = sts2_core::UnlockPresetManifest::pinned().unwrap();
    let simulator =
        sts2_content::run_on_preset_at(SEED, &character, &preset, 0, sts2_engine::RunMode::Custom)
            .unwrap();
    let mut driver = ReplayAdapter::new(simulator);
    let mut records = Vec::new();
    for event in Parser::new(script.as_bytes()).unwrap() {
        let Event::Record(record) = event.unwrap() else {
            continue;
        };
        append(&mut records, &record.kind, record.payload);
        driver.accept(records.last().unwrap()).unwrap();
        observe(
            &mut records,
            &observation::render(driver.simulator()),
            "action.settled",
        );
        if driver.actions_since_save() >= 2 && driver.simulator().state().combat.is_some() {
            append(&mut records, "run.resume", json!({}));
            driver.accept(records.last().unwrap()).unwrap();
            assert_eq!(driver.rewound_actions(), 2);
            let mut state = observation::render(driver.simulator());
            if divergent {
                state["players"][0]["gold"] = json!(123_456);
            }
            observe(&mut records, &state, "run.resume");
            observe(&mut records, &state, "action.settled");
            return (header, records);
        }
    }
    panic!("the synthetic walk must reach two unsaved combat actions");
}

fn encode(header: &str, records: &[Record], version: u8, deltas: bool) -> String {
    let mut output = header.replace(
        "[FormatVersion \"1\"]",
        &format!("[FormatVersion \"{version}\"]"),
    );
    let mut baseline: Option<(u64, Value)> = None;
    for record in records {
        let mut wire = record.clone();
        if matches!(
            record.kind.as_str(),
            "run.start" | "run.resume" | "room.enter" | "act.enter"
        ) {
            baseline = None;
        }
        if record.kind == "state.observe" {
            let state = &record.payload["state"];
            if let Some((base, previous)) = &baseline
                && deltas
            {
                let mut patch = Vec::new();
                for (key, value) in state.as_object().unwrap() {
                    if previous.get(key) != Some(value) {
                        patch.push(json!(["set", [key], value]));
                    }
                }
                for key in previous.as_object().unwrap().keys() {
                    if state.get(key).is_none() {
                        patch.push(json!(["remove", [key]]));
                    }
                }
                let mut envelope = record.payload.clone();
                envelope.as_object_mut().unwrap().remove("state");
                wire.kind = "state.delta".into();
                wire.payload = json!({"base":base,"observation":envelope,"patch":patch});
            }
            baseline = Some((record.sequence, state.clone()));
        }
        writeln!(output, "{} {} {}", wire.sequence, wire.kind, wire.payload).unwrap();
    }
    output.push_str("*\n");
    output
}

fn gzip(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap()
}

fn directory(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("alphaspire-storage-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

fn normalized_analysis(analysis: &alphaspire::analyze::Analysis) -> Value {
    let mut summary = serde_json::to_value(&analysis.summary).unwrap();
    summary.as_object_mut().unwrap().remove("timing");
    summary["trace"].as_object_mut().unwrap().remove("path");
    summary["trace"].as_object_mut().unwrap().remove("sha256");
    json!({"summary":summary,"lines":analysis.lines})
}

#[test]
fn full_delta_and_gzip_have_identical_analysis_summary_and_run_spec() {
    let (header, records) = scenario(false);
    let directory = directory("equivalent");
    let nets = nets();
    let mut expected = None;
    let mut expected_summary = None;
    for (name, version, deltas, compressed) in [
        ("legacy.sts2pgn", 1, false, false),
        ("full.sts2pgn", 2, false, false),
        ("delta.sts2pgn", 2, true, false),
        ("legacy.sts2pgn.gz", 1, false, true),
        ("full.sts2pgn.gz", 2, false, true),
        ("delta.data", 2, true, true),
    ] {
        let text = encode(&header, &records, version, deltas);
        if deltas {
            assert!(text.contains(" state.delta "));
        }
        let bytes = if compressed {
            gzip(text.as_bytes())
        } else {
            text.into_bytes()
        };
        let path = directory.join(name);
        std::fs::write(&path, &bytes).unwrap();
        let spec = run_spec(&path).unwrap();
        assert_eq!(spec.seed, SEED);
        assert_eq!(spec.ascension, 0);
        assert_eq!(spec.character.to_string(), "CHARACTER.IRONCLAD");
        assert_eq!(
            spec.preset.unlocks,
            sts2_core::UnlockPresetManifest::pinned().unwrap().unlocks
        );
        let mut summary = serde_json::to_value(summarize_trace(&path).unwrap()).unwrap();
        assert_eq!(summary["verified"], true);
        summary.as_object_mut().unwrap().remove("path");
        assert_eq!(
            &summary,
            expected_summary.get_or_insert_with(|| summary.clone()),
            "{name}"
        );
        let analysis = analyze_trace(&path, &nets, settings()).unwrap();
        assert!(analysis.summary.coverage.divergence.is_none());
        assert_eq!(
            (
                analysis.summary.coverage.resumes,
                analysis.summary.coverage.rewound
            ),
            (1, 2)
        );
        // Provenance identifies the actual input artifact, including compression.
        assert_eq!(
            analysis.summary.trace.sha256,
            format!("sha256:{:x}", sha2::Sha256::digest(&bytes))
        );
        let actual = normalized_analysis(&analysis);
        assert_eq!(
            &actual,
            expected.get_or_insert_with(|| actual.clone()),
            "{name}"
        );
    }
}

#[test]
fn unchanged_delta_fields_still_report_the_same_divergence() {
    let (header, records) = scenario(true);
    let directory = directory("divergence");
    let nets = nets();
    let mut expected = None;
    for (version, deltas, compressed) in [
        (1, false, false),
        (2, false, false),
        (2, true, false),
        (2, true, true),
    ] {
        let text = encode(&header, &records, version, deltas);
        let bytes = if compressed {
            gzip(text.as_bytes())
        } else {
            text.into_bytes()
        };
        let path = directory.join(format!("{version}-{deltas}-{compressed}.sts2pgn"));
        std::fs::write(&path, bytes).unwrap();
        assert!(summarize_trace(&path).is_err());
        let analysis = analyze_trace(&path, &nets, settings()).unwrap();
        assert!(analysis.summary.coverage.divergence.is_some());
        let actual = normalized_analysis(&analysis);
        assert_eq!(&actual, expected.get_or_insert_with(|| actual.clone()));
    }
}

#[test]
fn a_terminal_record_does_not_hide_a_corrupt_gzip_trailer() {
    let (header, records) = scenario(false);
    let text = encode(&header, &records, 2, true);
    let directory = directory("corrupt");
    let nets = nets();
    for truncated in [false, true] {
        let mut bytes = gzip(text.as_bytes());
        if truncated {
            bytes.truncate(bytes.len() - 4);
        } else {
            let crc = bytes.len() - 8;
            bytes[crc] ^= 1;
        }
        let path = directory.join(format!("{truncated}.sts2pgn.gz"));
        std::fs::write(&path, bytes).unwrap();
        assert!(summarize_trace(&path).is_err());
        assert!(analyze_trace(&path, &nets, settings()).is_err());
    }
}
