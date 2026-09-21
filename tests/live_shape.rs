//! The forward pass runs the graph at a position's own shape, with its
//! embedding lookups done host-side, and answers exactly what the graph as
//! exported answers when run padded to its ceiling. Pinned on the fixture
//! net: one lookup by the `tokens` input that the loader lifts, two by
//! ids sliced inside the graph that stay tract's. A current export lifts
//! both of its lookups; the loader takes either form.

use std::path::Path;
use std::sync::Arc;

use alphaspire::encoding::{
    ACTION_FEATURES, ACTION_TOKENS, ActionEncoding, MAX_TOKENS, OBSERVATION_SCALARS,
    ObservationEncoding, PolicyEncoder, TOKEN_FEATURES,
};
use alphaspire::net::{EncodedRequest, Evaluate, MAX_ACTIONS, PolicyValueNet, encode_request};
use sts2_core::UnlockPresetManifest;
use sts2_engine::Simulator;
use tract_onnx::prelude::*;

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}

fn load() -> PolicyValueNet {
    let registry = sts2_content::standard_registry();
    PolicyValueNet::load(
        &fixture(),
        Arc::new(PolicyEncoder::new(
            alphaspire::encoding::standard_vocabulary(&registry),
            &registry,
        )),
    )
    .expect("the fixture checkpoint loads")
}

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

/// Reference evaluation of the exported graph with every axis padded to its
/// ceiling and every embedding lookup executed inside tract.
struct Padded(TypedRunnableModel<TypedModel>);

impl Padded {
    fn load() -> Self {
        let facts = [
            InferenceFact::dt_shape(f32::datum_type(), [1, OBSERVATION_SCALARS]),
            InferenceFact::dt_shape(i64::datum_type(), [1, MAX_TOKENS]),
            InferenceFact::dt_shape(f32::datum_type(), [1, MAX_TOKENS, TOKEN_FEATURES]),
            InferenceFact::dt_shape(i64::datum_type(), [1, MAX_ACTIONS, ACTION_TOKENS]),
            InferenceFact::dt_shape(f32::datum_type(), [1, MAX_ACTIONS, ACTION_FEATURES]),
            InferenceFact::dt_shape(f32::datum_type(), [1, MAX_ACTIONS]),
        ];
        let mut model = tract_onnx::onnx()
            .model_for_path(fixture().with_extension("onnx"))
            .unwrap();
        for (slot, fact) in facts.into_iter().enumerate() {
            model = model.with_input_fact(slot, fact).unwrap();
        }
        Self(model.into_optimized().unwrap().into_runnable().unwrap())
    }

    /// The row's logits over the actions it offered, and its value.
    fn run(&self, request: &EncodedRequest) -> (Vec<f32>, f32) {
        let mut scalars = tract_ndarray::Array2::<f32>::zeros((1, OBSERVATION_SCALARS));
        let mut tokens = tract_ndarray::Array2::<i64>::zeros((1, MAX_TOKENS));
        let mut token_features =
            tract_ndarray::Array3::<f32>::zeros((1, MAX_TOKENS, TOKEN_FEATURES));
        let mut action_tokens =
            tract_ndarray::Array3::<i64>::zeros((1, MAX_ACTIONS, ACTION_TOKENS));
        let mut action_features =
            tract_ndarray::Array3::<f32>::zeros((1, MAX_ACTIONS, ACTION_FEATURES));
        let mut action_mask = tract_ndarray::Array2::<f32>::zeros((1, MAX_ACTIONS));
        for (column, &scalar) in request.observation.scalars.iter().enumerate() {
            scalars[(0, column)] = scalar;
        }
        for (column, &token) in request.observation.tokens.iter().enumerate() {
            tokens[(0, column)] = i64::from(token);
        }
        for (column, &feature) in request.observation.features.iter().enumerate() {
            token_features[(0, column / TOKEN_FEATURES, column % TOKEN_FEATURES)] = feature;
        }
        for (slot, action) in request.actions.iter().enumerate() {
            for (position, &token) in action.tokens.iter().enumerate() {
                action_tokens[(0, slot, position)] = i64::from(token);
            }
            for (position, &feature) in action.features.iter().enumerate() {
                action_features[(0, slot, position)] = feature;
            }
            action_mask[(0, slot)] = 1.0;
        }
        let outputs = self
            .0
            .run(tvec!(
                Tensor::from(scalars).into(),
                Tensor::from(tokens).into(),
                Tensor::from(token_features).into(),
                Tensor::from(action_tokens).into(),
                Tensor::from(action_features).into(),
                Tensor::from(action_mask).into(),
            ))
            .unwrap();
        let logits = outputs[0].as_slice::<f32>().unwrap()[..request.actions.len()].to_vec();
        let value = outputs[1].as_slice::<f32>().unwrap()[0];
        (logits, value)
    }
}

/// Two answers to the same position agree to the last place floating-point
/// reassociation leaves alone.
fn assert_close(what: &str, live: &[f32], padded: &[f32]) {
    assert_eq!(live.len(), padded.len(), "{what}: one answer per action");
    for (slot, (a, b)) in live.iter().zip(padded).enumerate() {
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= 1e-5 * scale,
            "{what}[{slot}]: live shape {a}, padded {b}"
        );
    }
}

#[test]
fn the_live_shape_answers_what_the_padded_graph_answers() {
    let net = load();
    let padded = Padded::load();
    let encoder = net.encoder_handle();
    for seed in ["NLD6VZXP94", "176YDEH492"] {
        let simulator = first_combat(seed);
        let request = encode_request(&encoder, &simulator, simulator.legal_actions());
        assert!(
            request.observation.live_tokens() < MAX_TOKENS && request.actions.len() > 1,
            "a first combat is smaller than the ceiling on both axes",
        );
        let (logits, value) = net.run_batch(std::slice::from_ref(&request)).pop().unwrap();
        let (padded_logits, padded_value) = padded.run(&request);
        assert_close(
            &format!("{seed} logits"),
            &logits[..request.actions.len()],
            &padded_logits,
        );
        #[allow(clippy::cast_possible_truncation, reason = "f64 from f32")]
        assert_close(&format!("{seed} value"), &[value as f32], &[padded_value]);
    }
}

#[test]
fn the_lookup_by_tokens_leaves_the_graph_and_the_rest_stay() {
    // One lookup is by the `tokens` input; the two by ids sliced out of
    // `action_tokens` inside the graph are tract's in this export.
    assert_eq!(load().lifted_lookups(), 1);
}

/// A request assembled by hand: `live` tokens, id 7 in every slot but the
/// last live one, which holds `last`; then `width` actions.
fn synthetic(live: usize, last: u32, width: usize) -> EncodedRequest {
    let mut tokens = vec![0; MAX_TOKENS];
    tokens[..live].fill(7);
    tokens[live - 1] = last;
    EncodedRequest {
        observation: ObservationEncoding::from_parts(
            vec![0.5; OBSERVATION_SCALARS],
            tokens,
            vec![0.25; MAX_TOKENS * TOKEN_FEATURES],
            live,
        ),
        actions: (0..width)
            .map(|slot| ActionEncoding {
                tokens: vec![1 + u32::try_from(slot).unwrap(); ACTION_TOKENS],
                features: vec![0.125; ACTION_FEATURES],
            })
            .collect(),
    }
}

#[test]
fn the_last_live_token_is_priced() {
    let net = load();
    let padded = Padded::load();
    // Two positions that differ only in their last live token, at a live
    // count far below the ceiling and at the ceiling itself: each agrees
    // with the padded graph, and the pair differ from each other, so the
    // axis the plan runs at reaches the last live slot.
    for live in [10, MAX_TOKENS] {
        let mut answers = Vec::new();
        for last in [7, 9] {
            let request = synthetic(live, last, 3);
            let (logits, value) = net.run_batch(std::slice::from_ref(&request)).pop().unwrap();
            let (padded_logits, padded_value) = padded.run(&request);
            assert_close(
                &format!("live {live} last {last}"),
                &logits[..3],
                &padded_logits,
            );
            #[allow(clippy::cast_possible_truncation, reason = "f64 from f32")]
            assert_close(
                &format!("live {live} last {last}"),
                &[value as f32],
                &[padded_value],
            );
            answers.push((logits, value));
        }
        assert_ne!(
            answers[0], answers[1],
            "the token in live slot {live} changes the answer"
        );
    }
}

#[test]
fn a_value_only_request_runs_without_an_action() {
    let net = load();
    let padded = Padded::load();
    let simulator = first_combat("NLD6VZXP94");
    let request = encode_request(net.encoder(), &simulator, &[]);
    let value = net.state_value(&simulator);
    let (_, padded_value) = padded.run(&request);
    #[allow(clippy::cast_possible_truncation, reason = "f64 from f32")]
    assert_close("value", &[value as f32], &[padded_value]);
}

#[test]
fn the_same_position_prices_the_same_twice() {
    let net = load();
    let simulator = first_combat("176YDEH492");
    let priced = |net: &PolicyValueNet| {
        net.priors_and_value(&simulator, simulator.legal_actions())
            .into_parts()
    };
    let (priors, value, degraded) = priced(&net);
    let (again, value_again, degraded_again) = priced(&net);
    assert_eq!(priors, again);
    assert_eq!(value.to_bits(), value_again.to_bits());
    assert_eq!(degraded, degraded_again);
    let reloaded = load();
    let (from_reload, value_reloaded, _) = priced(&reloaded);
    assert_eq!(priors, from_reload, "and across a reload");
    assert_eq!(value.to_bits(), value_reloaded.to_bits());
}
