//! The learned prior behind the search: a provenance-checked checkpoint
//! prices priors and values deterministically, a mismatched checkpoint is
//! refused, and a net-guided search answers with a legal action off the
//! same core.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use alphaspire::encoding::PolicyEncoder;
use alphaspire::env::Belief;
use alphaspire::net::{Evaluate as _, PolicyValueNet};
use alphaspire::objective::CombatStrength;
use alphaspire::policy::UniformRandom;
use alphaspire::search::{Mcts, SearchConfig, Uct, canonical_actions};
use sts2_core::UnlockPresetManifest;
use sts2_engine::{Action, Simulator};
use sts2_rng::MegaRandom;

fn encoder() -> Arc<PolicyEncoder> {
    let registry = sts2_content::standard_registry();
    Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ))
}

fn fixture() -> PolicyValueNet {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    PolicyValueNet::load(&base, encoder()).expect("the fixture checkpoint loads")
}

fn fresh_run(seed: &str) -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at(seed, &character, &preset, 0).unwrap()
}

fn next_action(simulator: &Simulator) -> Action {
    simulator
        .legal_actions()
        .iter()
        .find(|action| matches!(action, Action::ChooseCards { cards, .. } if !cards.is_empty()))
        .or_else(|| simulator.legal_actions().first())
        .cloned()
        .expect("a live run offers an action")
}

fn first_combat() -> Simulator {
    let mut simulator = fresh_run("NLD6VZXP94");
    for _ in 0..40 {
        if simulator.state().combat.is_some() {
            return simulator;
        }
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }
    panic!("the seed reaches a combat inside forty decisions");
}

#[test]
fn a_checkpoint_prices_priors_and_a_value_deterministically() {
    let net = fixture();
    let simulator = first_combat();
    let actions = canonical_actions(simulator.legal_actions());
    let (priors, value, degraded) = net.priors_and_value(&simulator, &actions).into_parts();
    assert_eq!(
        degraded, None,
        "a decision inside the axis is the net's own"
    );
    assert_eq!(priors.len(), actions.len());
    let total: f32 = priors.iter().sum();
    assert!((total - 1.0).abs() < 1e-4, "priors sum to one: {total}");
    assert!(
        priors
            .iter()
            .all(|prior| prior.is_finite() && *prior >= 0.0)
    );
    assert!(value.is_finite());
    let (again, value_again, _) = net.priors_and_value(&simulator, &actions).into_parts();
    assert_eq!(priors, again, "the same position prices the same priors");
    assert!((value - value_again).abs() < 1e-9);
    assert!(net.state_value(&simulator).is_finite());
}

/// A copy of the fixture checkpoint with `edit` applied to its provenance,
/// so a test can say what the provenance claims. Returns the base path: a
/// checkpoint that loads and one that is refused are told apart by the
/// caller, not here.
fn checkpoint_claiming(name: &str, edit: impl FnOnce(&mut serde_json::Value)) -> PathBuf {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let directory = std::env::temp_dir().join(format!("alphaspire-{name}"));
    std::fs::create_dir_all(&directory).unwrap();
    let base = directory.join("tiny");
    std::fs::copy(source.with_extension("onnx"), base.with_extension("onnx")).unwrap();
    let mut provenance: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(source.with_extension("json")).unwrap())
            .unwrap();
    edit(&mut provenance);
    std::fs::write(base.with_extension("json"), provenance.to_string()).unwrap();
    base
}

#[test]
fn a_mismatched_checkpoint_is_refused() {
    let stale = checkpoint_claiming("stale-checkpoint", |provenance| {
        provenance["vocabulary_hash"] = serde_json::json!("sha256:not-this-content");
    });
    let refused = PolicyValueNet::load(&stale, encoder());
    assert!(refused.is_err(), "another vocabulary is another world");
}

#[test]
fn a_checkpoint_from_another_character_warns_and_loads_anyway() {
    // Cross-character loading is how a character without a checkpoint is
    // bootstrapped, so the mismatch is a warning and never a refusal. What
    // it buys is that a mis-pointed --net says so at second zero.
    let silent = "CHARACTER.SILENT".parse().unwrap();
    let ironclad = "CHARACTER.IRONCLAD".parse().unwrap();
    let base = checkpoint_claiming("foreign-checkpoint", |provenance| {
        provenance["characters"] = serde_json::json!(["CHARACTER.IRONCLAD"]);
    });
    let net = PolicyValueNet::load(&base, encoder()).expect("the checkpoint loads");
    assert_eq!(net.characters(), ["CHARACTER.IRONCLAD"]);
    assert!(net.foreign_to(&ironclad).is_none(), "its own character");
    let warning = net.foreign_to(&silent).expect("a foreign character warns");
    assert!(
        warning.contains("CHARACTER.IRONCLAD") && warning.contains("CHARACTER.SILENT"),
        "the warning names both: {warning}"
    );
}

#[test]
fn a_checkpoint_that_names_no_character_reads_as_saying_nothing() {
    // Every checkpoint written before provenance carried the field. It
    // loads, and it warns about nobody: absent is "does not say", not
    // "trained on nothing".
    let net = fixture();
    assert_eq!(net.characters(), [] as [String; 0]);
    for name in ["CHARACTER.IRONCLAD", "CHARACTER.SILENT"] {
        assert!(net.foreign_to(&name.parse().unwrap()).is_none());
    }
}

#[test]
fn a_net_guided_search_answers_legally_off_the_same_core() {
    let net = fixture();
    let original = first_combat();
    let mut mcts = Mcts::new(
        SearchConfig {
            iterations: 12,
            rollout_depth: 5,
            temperature: 0.5,
        },
        Uct::default(),
    );
    let mut determinizer = Belief::from_simulator(&original, 5).unwrap();
    let mut rollout = UniformRandom;
    let mut rng = MegaRandom::new(5);
    let (action, visits) = mcts.decide_with_policy(
        &mut determinizer,
        &CombatStrength::default(),
        &mut rollout,
        Some(&net),
        &mut rng,
    );
    assert!(
        original.legal_actions().contains(&action),
        "the guided search answers with an action the real screen offers"
    );
    assert!(!visits.actions.is_empty(), "the root was searched");
    assert!(mcts.tree_size() > 0);
}

#[test]
fn a_run_checkpoint_and_an_act_checkpoint_refuse_each_other() {
    // Both are `scope: macro`, both price the same actions off the same
    // encoder, and their value heads answer different quantities: an act's
    // horizon score inside a fixed band against a whole run's return, which
    // goes negative and grows with the climb. Nothing in the file tells them
    // apart, so the loaders do — and refuse rather than reinterpret.
    let act = checkpoint_claiming("act-boundary-checkpoint", |provenance| {
        provenance["scope"] = serde_json::json!("macro");
        provenance["value_semantics"] = serde_json::json!("act-boundary-v2");
    });
    let run = checkpoint_claiming("run-return-checkpoint", |provenance| {
        provenance["scope"] = serde_json::json!("macro");
        provenance["value_semantics"] =
            serde_json::json!(alphaspire::objective::RUN_VALUE_SEMANTICS);
    });
    assert!(
        PolicyValueNet::load_macro(&act, encoder()).is_ok(),
        "the act tree's own checkpoint loads for the act tree"
    );
    assert!(
        PolicyValueNet::load_run(&run, encoder()).is_ok(),
        "and the run checkpoint loads for the run net"
    );
    let message = PolicyValueNet::load_run(&act, encoder())
        .err()
        .expect("an act-boundary critic is not a run critic")
        .to_string();
    assert!(
        message.contains("value_semantics") && message.contains("run-return-v1"),
        "the refusal names what it wanted: {message}"
    );
    assert!(
        PolicyValueNet::load_macro(&run, encoder()).is_err(),
        "and a run critic is not an act-boundary critic"
    );
    assert!(
        PolicyValueNet::load(&run, encoder()).is_err(),
        "nor is it a combat net"
    );
}
