//! Checks that each row's inference result is independent of the other
//! rows in its batch. Tests use batches larger than the production size
//! of one to exercise this property.

use std::path::Path;
use std::sync::Arc;

use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::{PolicyValueNet, encode_request};
use sts2_core::UnlockPresetManifest;
use sts2_engine::Simulator;

fn first_combat(seed: &str) -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let mut simulator =
        sts2_content::standard_run_on_preset_at(seed, &character, &preset, 0).unwrap();
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
fn a_row_is_deaf_to_its_batch_mates() {
    let registry = sts2_content::standard_registry();
    let net = PolicyValueNet::load_with_batch(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny"),
        Arc::new(PolicyEncoder::new(
            alphaspire::encoding::standard_vocabulary(&registry),
            &registry,
        )),
        4,
    )
    .expect("the fixture checkpoint loads");
    let one = first_combat("NLD6VZXP94");
    let two = first_combat("176YDEH492");
    let encoder = net.encoder_handle();
    let request =
        |simulator: &Simulator| encode_request(&encoder, simulator, simulator.legal_actions());
    let alone = net.run_batch(&[request(&one)]);
    let together = net.run_batch(&[request(&one), request(&two)]);
    assert_eq!(
        alone[0], together[0],
        "the first row's answer is bitwise the same with and without company"
    );
    let other_alone = net.run_batch(&[request(&two)]);
    assert_eq!(other_alone[0], together[1], "and so is the second's");
}
