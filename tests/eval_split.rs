//! Throwaway probe: where one leaf evaluation's time goes — observation
//! build, encoding, tensor assembly, forward. Run by hand; point
//! `ALPHASPIRE_PROBE_CKPT` at a real checkpoint base to measure it, else the
//! tiny fixture answers.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::{Evaluate as _, INFER_BATCH, PolicyValueNet, encode_request};
use sts2_core::UnlockPresetManifest;

#[test]
#[ignore = "a probe, not a property"]
fn where_a_leaf_evaluation_spends() {
    let base = std::env::var("ALPHASPIRE_PROBE_CKPT").map_or_else(
        |_| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny"),
        PathBuf::from,
    );
    let registry = sts2_content::standard_registry();
    let net = PolicyValueNet::load_with_batch(
        &base,
        Arc::new(PolicyEncoder::new(
            alphaspire::encoding::standard_vocabulary(&registry),
            &registry,
        )),
        INFER_BATCH,
    )
    .unwrap();
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let mut simulator =
        sts2_content::standard_run_on_preset_at("NLD6VZXP94", &character, &preset, 0).unwrap();
    for _ in 0..40 {
        if simulator.state().combat.is_some() {
            break;
        }
        let action = simulator.legal_actions().first().cloned().unwrap();
        simulator.step_quietly(&action).unwrap();
    }
    let encoder = net.encoder_handle();
    let actions = simulator.legal_actions().to_vec();
    let rounds = 500_u32;

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        let _ = std::hint::black_box(simulator.agent_observation());
    }
    println!("agent_observation      {:?}", start.elapsed() / rounds);

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        let _ = std::hint::black_box(encode_request(&encoder, &simulator, &actions));
    }
    println!("encode_request         {:?}", start.elapsed() / rounds);

    let request = [encode_request(&encoder, &simulator, &actions)];
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        let _ = std::hint::black_box(net.run_batch(&request));
    }
    println!("run_batch              {:?}", start.elapsed() / rounds);

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        let _ = std::hint::black_box(net.priors_and_value(&simulator, &actions));
    }
    println!("priors_and_value       {:?}", start.elapsed() / rounds);

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        let _ = std::hint::black_box(net.state_value(&simulator));
    }
    println!("state_value            {:?}", start.elapsed() / rounds);
}
