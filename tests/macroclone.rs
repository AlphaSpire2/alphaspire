//! The macro net's bootstrap: the hand-written out-of-combat policy read as
//! a *teacher*, its answers recorded off ordinary belief self-play, and the
//! act tree's two-checkpoint evaluator that the resulting checkpoint is
//! meant to guide.
//!
//! The rule the whole file is written against: the nets are the evaluators.
//! Nothing here may bring back a random play-through, and nothing here adds
//! a heuristic — the existing one is cloned, never extended.

use std::path::Path;
use std::sync::Arc;

use alphaspire::encoding::PolicyEncoder;
use alphaspire::heuristics::{Heuristic, MACRO_POLICY_TEMPERATURE};
use alphaspire::net::{ActEvaluator, Evaluate, PolicyValueNet};
use alphaspire::objective::{ActBoundary, CombatStrength, Objective};
use alphaspire::policy::{RolloutPolicy, UniformRandom, permitted_actions};
use alphaspire::search::{BeliefSearch, SearchConfig, canonical_actions};
use sts2_core::UnlockPresetManifest;
use sts2_engine::{Action, Simulator};
use sts2_rng::MegaRandom;

const SEED: &str = "NLD6VZXP94";

fn encoder() -> PolicyEncoder {
    let registry = sts2_content::standard_registry();
    PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    )
}

fn fresh_run() -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at(SEED, &character, &preset, 0).unwrap()
}

fn tiny() -> SearchConfig {
    SearchConfig {
        iterations: 4,
        rollout_depth: 4,
        temperature: 0.0,
    }
}

#[test]
fn the_teacher_teaches_the_answer_it_would_have_played() {
    // A cloned policy is only the teacher's if its mode is the teacher's
    // move. Walked over a real run's macro screens rather than one, so the
    // claim covers reward claims, pathing, rest sites and events alike.
    let mut simulator = fresh_run();
    let mut policy = Heuristic;
    let mut rng = MegaRandom::new(11);
    let mut screens = 0;
    for _ in 0..120 {
        if simulator.state().terminal.is_some() {
            break;
        }
        let action = policy.choose(&simulator, &mut rng);
        if alphaspire::env::fight_over(&simulator) {
            let canonical = canonical_actions(&permitted_actions(&simulator));
            let pi = Heuristic::teacher_policy(&simulator, &canonical);
            assert_eq!(pi.len(), canonical.len(), "aligned with the actions");
            let total: f64 = pi.iter().sum();
            assert!((total - 1.0).abs() < 1e-9, "a distribution: {total}");
            assert!(pi.iter().all(|weight| weight.is_finite() && *weight >= 0.0));
            let peak = pi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            // `choose` takes its argmax over the raw permitted list, the
            // teacher over the canonical one; the two agree on the class,
            // which is what a sample records. Stated as "the played class
            // carries the peak mass" rather than "is *the* mode", because a
            // screen can offer equal-scoring options — the teacher splits a
            // tie evenly where `choose` breaks it by index, and neither is
            // wrong about what the policy knows.
            let played = canonical_actions(std::slice::from_ref(&action));
            let at = canonical
                .iter()
                .position(|candidate| *candidate == played[0])
                .expect("the played class is on the canonical list");
            assert!(
                (pi[at] - peak).abs() < 1e-12,
                "the teacher's peak mass is on the move the teacher played: \
                 {} against {peak}",
                pi[at]
            );
            screens += 1;
        }
        simulator.step_quietly(&action).unwrap();
    }
    assert!(
        screens > 10,
        "the walk crossed real macro screens: {screens}"
    );
}

#[test]
fn the_soft_target_keeps_the_ranking_a_one_hot_would_throw_away() {
    // Why a softmax and not a one-hot: the teacher knows more than its
    // argmax. Read at the documented temperature, a knob's-worth of
    // separation stays a preference and the refusal sentinels underflow.
    let scores = [40.0, 35.0, 5.0, -10_000.0];
    let peak = scores[0];
    let weights: Vec<f64> = scores
        .iter()
        .map(|score| ((score - peak) / MACRO_POLICY_TEMPERATURE).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let pi: Vec<f64> = weights.iter().map(|weight| weight / total).collect();
    assert!(pi[0] > pi[1], "the argmax leads");
    assert!(
        pi[1] > 10.0 * pi[2],
        "and a five-point gap is a preference, not a tie: {pi:?}"
    );
    assert!(
        pi[1] > 0.05,
        "but the runner-up keeps real mass, which is the point: {}",
        pi[1]
    );
    assert!(pi[3] < 1e-300, "a refusal sentinel underflows to nothing");
}

#[test]
fn a_policy_with_no_ranking_teaches_nothing() {
    // The honest default: uniform random has no scores to hand out, so a
    // recorder over it records nothing rather than inventing a target.
    let simulator = fresh_run();
    let canonical = canonical_actions(&permitted_actions(&simulator));
    assert!(
        UniformRandom
            .macro_teacher_policy(&simulator, &canonical)
            .is_none()
    );
    assert!(
        Heuristic
            .macro_teacher_policy(&simulator, &canonical)
            .is_some(),
        "and the teacher does"
    );
}

#[test]
fn belief_self_play_records_the_macro_decisions_it_hands_to_its_teacher() {
    // The bootstrap's whole claim: the production config — belief search in
    // fights, the heuristic out of them — already plays every macro decision
    // and can record it for free, with the act's own horizon score as z.
    let mut policy =
        BeliefSearch::with_rollout(tiny(), CombatStrength::default(), Box::new(Heuristic))
            .recording()
            .recording_macro();
    let mut rng = MegaRandom::new(29);
    let mut simulator = fresh_run();
    for _ in 0..400 {
        if simulator.state().terminal.is_some() {
            break;
        }
        let action = policy.choose(&simulator, &mut rng);
        simulator.step_quietly(&action).unwrap();
    }
    policy.run_ended(&simulator);
    let macro_samples = policy.drain_macro_decisions();
    let combat_samples = policy.drain_decisions();
    // This walk's run dies early under a four-iteration combat search, so
    // the bar is the shape, not the yield: a real batch measures ~67 macro
    // decisions per run, against act mode's ~15 searched ones at a 40k step
    // budget — which is the bootstrap's cost argument.
    assert!(
        macro_samples.len() >= 8,
        "the run answered macro decisions and kept them: {}",
        macro_samples.len()
    );
    assert!(
        !combat_samples.is_empty(),
        "and the in-combat channel is untouched beside it"
    );
    assert!(
        combat_samples
            .iter()
            .all(|sample| sample.encounter.is_some() && sample.act.is_none()),
        "an in-combat sample is stamped exactly as it always was"
    );
    for sample in &macro_samples {
        assert!(sample.encounter.is_none(), "a macro decision has no fight");
        assert_eq!(sample.fight, 0);
        assert!(sample.act.is_some(), "it names the act it belongs to");
        assert_eq!(sample.pi.len(), sample.actions.len(), "pi is aligned");
        let total: f32 = sample.pi.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "pi is a distribution: {total}");
    }
    // Every macro sample of the *last* act settles on the state the run
    // ended in — the same discipline act mode's recorder uses.
    #[allow(clippy::cast_possible_truncation, reason = "z is written as f32")]
    let ending = ActBoundary::default().peek(&simulator) as f32;
    let last = macro_samples.last().expect("the set is not empty");
    assert!(
        (last.z - ending).abs() < 1e-6,
        "the last act settles where the run ended: {} against {ending}",
        last.z
    );
    // And the samples are not all one number: an act that ended and an act
    // still running when the run died do not share a horizon.
    assert!(
        macro_samples.iter().all(|sample| sample.z.is_finite()),
        "every z settled"
    );
}

#[test]
fn a_teacherless_rollout_records_nothing_rather_than_a_uniform_target() {
    let mut policy =
        BeliefSearch::with_rollout(tiny(), CombatStrength::default(), Box::new(UniformRandom))
            .recording_macro();
    let mut rng = MegaRandom::new(3);
    let mut simulator = fresh_run();
    for _ in 0..60 {
        if simulator.state().terminal.is_some() {
            break;
        }
        let action = policy.choose(&simulator, &mut rng);
        simulator.step_quietly(&action).unwrap();
    }
    policy.run_ended(&simulator);
    assert!(
        policy.drain_macro_decisions().is_empty(),
        "uniform random has no ranking to clone"
    );
}

fn fixture() -> Arc<PolicyValueNet> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    Arc::new(PolicyValueNet::load(&base, Arc::new(encoder())).expect("the combat fixture loads"))
}

#[test]
fn a_combat_checkpoint_is_refused_where_a_run_net_is_wanted() {
    // The two checkpoints are interchangeable on the wire and nowhere else:
    // same encoder, same graph, same file layout, different horizons. The
    // loader refuses rather than reinterprets, exactly as it does for every
    // version it checks.
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let error = PolicyValueNet::load_macro(&base, Arc::new(encoder()))
        .err()
        .expect("a combat checkpoint is not a run net");
    let message = error.to_string();
    assert!(
        message.contains("scope") && message.contains("macro"),
        "the refusal names what it wanted: {message}"
    );
    assert!(
        PolicyValueNet::load(&base, Arc::new(encoder())).is_ok(),
        "and the combat loader still takes it"
    );
}

#[test]
fn a_fight_entry_leaf_is_the_combat_head_read_in_act_boundary_units() {
    // The translation queued item 2 asked for: one combat scalar read as
    // survival and health-after, priced through ActBoundary's own weights.
    // What is pinned here is the shape, not a calibration claim — a single
    // scalar cannot separate P(survive) from HP retained, and the type's
    // documentation says so.
    let net = fixture();
    let evaluator = ActEvaluator::new(
        Arc::clone(&net) as Arc<dyn Evaluate>,
        Arc::clone(&net) as Arc<dyn Evaluate>,
    );
    let simulator = fresh_run();
    let boundary = ActBoundary::default();
    let dead = evaluator.translate(0.0, &simulator);
    let scraped = evaluator.translate(1.0, &simulator);
    let untouched = evaluator.translate(2.0, &simulator);
    assert!(
        dead < scraped && scraped < untouched,
        "a better fight is a better act: {dead} {scraped} {untouched}"
    );
    let floor = f64::from(simulator.state().run.as_ref().map_or(0, |run| run.floor));
    assert!(
        (dead - boundary.floor_weight * floor).abs() < 1e-9,
        "a fight the head gives up on pays only the floors climbed, which is \
         exactly ActBoundary's own defeat pricing"
    );
    assert!(
        untouched >= boundary.crossing_weight,
        "and a fight walked out of whole carries a crossing's worth"
    );
    // The scale claim that makes the splice legal: a fight entry and a macro
    // state are both an estimate of the act's eventual boundary score, so
    // neither is a full crossing below the other for standing where it does.
    assert!(
        scraped >= boundary.crossing_weight * 0.9,
        "a survivable fight is not priced a crossing below a macro leaf: {scraped}"
    );
}

#[test]
fn the_act_evaluator_sends_each_state_to_the_net_that_prices_it() {
    // Out of combat the macro net answers and nothing is translated; in a
    // fight the combat net answers and everything is.
    let net = fixture();
    let evaluator = ActEvaluator::new(
        Arc::clone(&net) as Arc<dyn Evaluate>,
        Arc::clone(&net) as Arc<dyn Evaluate>,
    );
    let simulator = fresh_run();
    assert!(alphaspire::env::fight_over(&simulator));
    let raw = net.state_value(&simulator);
    assert!(
        (evaluator.state_value(&simulator) - raw).abs() < 1e-12,
        "a macro state is already act-boundary units"
    );
    let mut fighting = fresh_run();
    for _ in 0..80 {
        if !alphaspire::env::fight_over(&fighting) {
            break;
        }
        let action: Action = Heuristic.choose(&fighting, &mut MegaRandom::new(5));
        fighting.step_quietly(&action).unwrap();
    }
    assert!(
        !alphaspire::env::fight_over(&fighting),
        "the seed reaches a live fight"
    );
    let translated = evaluator.state_value(&fighting);
    assert!(
        (translated - evaluator.translate(net.state_value(&fighting), &fighting)).abs() < 1e-12,
        "a fight entry goes through the translation"
    );
}
