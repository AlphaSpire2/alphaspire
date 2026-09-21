//! Alphaspire can build a seeded run and step it through
//! the simulator's public API alone — the same surface a bindings consumer
//! gets. No engine internals, no privileged hooks.

use sts2_core::UnlockPresetManifest;
use sts2_engine::Action;

/// Walks a run forward on the dumbest possible policy — the first legal
/// action, preferring a non-empty card pick so a skippable reward cannot
/// re-offer itself — and asserts the run genuinely progresses.
#[test]
fn a_seeded_run_steps_through_the_public_api() {
    let preset = UnlockPresetManifest::pinned().expect("pinned preset parses");
    let character = "CHARACTER.IRONCLAD".parse().expect("character id parses");
    let mut simulator =
        sts2_content::standard_run_on_preset_at("NLD6VZXP94", &character, &preset, 0)
            .expect("the pinned baseline generates");

    let mut applied = 0_u32;
    for _ in 0_u32..400 {
        if simulator.state().terminal.is_some() {
            break;
        }
        let action = simulator
            .legal_actions()
            .iter()
            .find(|action| matches!(action, Action::ChooseCards { cards, .. } if !cards.is_empty()))
            .or_else(|| simulator.legal_actions().first())
            .cloned()
            .expect("a live run always offers an action");
        simulator.step(action).expect("a legal action applies");
        applied += 1;
    }

    let floor = simulator.state().run.as_ref().map_or(0, |run| run.floor);
    assert!(applied >= 25, "only {applied} actions applied");
    assert!(
        floor >= 2 || simulator.state().terminal.is_some(),
        "the run never left the first floor (floor {floor})"
    );

    // Snapshot and restore round-trip: the state a search node would hold.
    let snapshot = simulator.snapshot();
    let key_before = simulator.state_key().expect("a settled state has a key");
    simulator
        .restore(&snapshot)
        .expect("a snapshot restores onto its own simulator");
    assert_eq!(
        simulator.state_key().expect("a settled state has a key"),
        key_before,
        "restore reproduces the exact state"
    );
}

#[test]
fn every_character_the_registry_names_can_start_a_run() {
    // The harness checks `--character` against this list before a batch
    // plays anything, so a name on it that cannot be started would be a
    // typo waved through — the failure the check exists to prevent. The
    // list is read off the card-pool map and the starting loadouts are a
    // separate table, and this is what holds the two together.
    let preset = UnlockPresetManifest::pinned().unwrap();
    let registry = sts2_content::standard_registry();
    let characters = registry.registered_character_ids();
    assert!(!characters.is_empty());
    for character in characters {
        assert_eq!(character.category(), "CHARACTER", "{character}");
        sts2_content::standard_run_on_preset_at("NLD6VZXP94", &character, &preset, 0)
            .unwrap_or_else(|error| panic!("{character} starts a run: {error:?}"));
    }
}
