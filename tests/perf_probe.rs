//! Throwaway probe: what a padded forward costs at each fill. Run by hand.

use std::path::Path;
use std::sync::Arc;

use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::{INFER_BATCH, PolicyValueNet, encode_request};
use sts2_core::UnlockPresetManifest;

#[test]
#[ignore = "a probe, not a property"]
fn forward_cost_by_fill() {
    let registry = sts2_content::standard_registry();
    let load = |batch: usize| {
        PolicyValueNet::load_with_batch(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny"),
            Arc::new(PolicyEncoder::new(
                alphaspire::encoding::standard_vocabulary(&registry),
                &registry,
            )),
            batch,
        )
        .unwrap()
    };
    let net = load(INFER_BATCH);
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
    let request = || encode_request(&encoder, &simulator, simulator.legal_actions());
    for shape in [1, 4, 8, 16, INFER_BATCH] {
        let net = load(shape);
        let batch: Vec<_> = (0..shape).map(|_| request()).collect();
        let start = std::time::Instant::now();
        let rounds = 200;
        for _ in 0..rounds {
            let _ = net.run_batch(&batch);
        }
        let full = start.elapsed() / rounds;
        let lone = [request()];
        let start = std::time::Instant::now();
        for _ in 0..rounds {
            let _ = net.run_batch(&lone);
        }
        let thin = start.elapsed() / rounds;
        println!(
            "plan {shape:2}: full fill {full:?}/fwd ({:?}/row), fill-1 {thin:?}/fwd",
            full / u32::try_from(shape).unwrap()
        );
    }
}
