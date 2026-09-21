//! Degradation is counted wherever it happens.
//!
//! A checkpoint that cannot price a decision answers uniform, and argmax over
//! uniform priors is the last action on the screen — a fixed arbitrary pick,
//! not a policy. Every path that takes such an answer counts it: the two
//! greedy policies, the actor that records training lines, and the batch
//! summary that reports on all of them.

use std::path::Path;
use std::sync::Arc;

use alphaspire::actor::{MacroActor, MacroGreedy, NetGreedy, Resolver};
use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::{Degraded, Evaluate, MAX_ACTIONS, PolicyValueNet, Priced};
use alphaspire::objective::CombatStrength;
use alphaspire::plan::ActionPlan;
use alphaspire::policy::RolloutPolicy;
use alphaspire::selfplay::RunReport;
use alphaspire::summary::{BatchSummary, Provenance};
use sts2_engine::{Action, Simulator};
use sts2_rng::MegaRandom;

const SEED: &str = "NLD6VZXP94";

fn fixture() -> Arc<dyn Evaluate> {
    let registry = sts2_content::standard_registry();
    let encoder = Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ));
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    Arc::new(PolicyValueNet::load(&base, encoder).expect("the fixture checkpoint loads"))
}

/// The fixture checkpoint behind an action axis of `axis`: past it the
/// decision is priced uniform and the pricing says so.
///
/// `PolicyValueNet`'s own rule at a width every ordinary screen crosses, so a
/// short run degrades many decisions instead of the one in two hundred a real
/// axis of [`MAX_ACTIONS`] degrades.
struct TinyAxis {
    net: Arc<dyn Evaluate>,
    axis: usize,
}

impl TinyAxis {
    #[allow(
        clippy::cast_precision_loss,
        reason = "action counts are far below f32 precision"
    )]
    fn uniform(&self, simulator: &Simulator, width: usize) -> Priced {
        Priced::fallback(
            vec![1.0 / width as f32; width],
            self.net.state_value(simulator),
            Degraded::OverAxis,
        )
    }
}

impl Evaluate for TinyAxis {
    fn priors_and_value(&self, simulator: &Simulator, actions: &[Action]) -> Priced {
        if actions.len() > self.axis {
            return self.uniform(simulator, actions.len());
        }
        self.net.priors_and_value(simulator, actions)
    }

    fn plan_priors_and_value(&self, simulator: &Simulator, plans: &[ActionPlan]) -> Priced {
        if plans.len() > self.axis {
            return self.uniform(simulator, plans.len());
        }
        self.net.plan_priors_and_value(simulator, plans)
    }

    fn state_value(&self, simulator: &Simulator) -> f64 {
        self.net.state_value(simulator)
    }
}

/// A checkpoint whose every answer holds a number that is not one: what
/// `settled` leaves behind once it has read the row as flat.
struct NonFiniteRows(Arc<dyn Evaluate>);

impl Evaluate for NonFiniteRows {
    fn priors_and_value(&self, simulator: &Simulator, actions: &[Action]) -> Priced {
        let (priors, value, _) = self.0.priors_and_value(simulator, actions).into_parts();
        Priced::fallback(priors, value, Degraded::NonFinite)
    }

    fn plan_priors_and_value(&self, simulator: &Simulator, plans: &[ActionPlan]) -> Priced {
        let (priors, value, _) = self.0.plan_priors_and_value(simulator, plans).into_parts();
        Priced::fallback(priors, value, Degraded::NonFinite)
    }

    fn state_value(&self, simulator: &Simulator) -> f64 {
        self.0.state_value(simulator)
    }
}

/// One run played by `policy`, bounded so an argmax arm on uniform priors
/// cannot walk forever.
fn play(policy: &mut dyn RolloutPolicy, max_steps: usize) -> RunReport {
    let mut rng = MegaRandom::new(7);
    let mut objective = CombatStrength::default();
    alphaspire::selfplay::play_run(
        SEED,
        &"CHARACTER.IRONCLAD".parse().unwrap(),
        0,
        policy,
        &mut rng,
        &mut objective,
        max_steps,
    )
    .expect("the run walks")
}

fn provenance() -> Provenance {
    Provenance {
        run_net: "ckpts/run".into(),
        combat_net: "ckpts/combat".into(),
        resolver: Resolver::Greedy,
        character: "CHARACTER.IRONCLAD".into(),
        ascension: 0,
        runs: 1,
        analysis_seed: 7,
        max_steps: 40,
    }
}

#[test]
fn a_decision_wider_than_the_axis_is_priced_uniform_and_says_so() {
    let net = fixture();
    let preset = sts2_core::UnlockPresetManifest::pinned().unwrap();
    let simulator = sts2_content::standard_run_on_preset_at(
        SEED,
        &"CHARACTER.IRONCLAD".parse().unwrap(),
        &preset,
        0,
    )
    .unwrap();
    let one = simulator.legal_actions()[0].clone();

    let wide = vec![one.clone(); MAX_ACTIONS + 1];
    let (priors, _, degraded) = net.priors_and_value(&simulator, &wide).into_parts();
    assert_eq!(degraded, Some(Degraded::OverAxis));
    assert_eq!(priors.len(), wide.len());
    assert!((priors.iter().sum::<f32>() - 1.0).abs() < 1e-4, "a policy");

    let plans: Vec<ActionPlan> = wide.into_iter().map(ActionPlan::Single).collect();
    let (_, _, degraded) = net.plan_priors_and_value(&simulator, &plans).into_parts();
    assert_eq!(degraded, Some(Degraded::OverAxis), "and on the plan path");
}

#[test]
fn every_greedy_caller_counts_what_the_checkpoint_did_not_price() {
    let net: Arc<dyn Evaluate> = Arc::new(TinyAxis {
        net: fixture(),
        axis: 2,
    });

    // The in-fight resolver, which answers through `greedy_action`.
    let mut resolver = NetGreedy::new(Arc::clone(&net));
    play(&mut resolver, 40);
    assert!(
        resolver.degradations().over_axis > 0,
        "the resolver counted the screens it could not price"
    );

    // The gate's arm, which answers through `greedy_plan`.
    let mut greedy = MacroGreedy::new(Arc::clone(&net), Resolver::Greedy.build(Arc::clone(&net)));
    play(&mut greedy, 40);
    assert!(greedy.degradations().over_axis > 0, "and the greedy arm");

    // The actor, which counted this one already and now counts it by cause.
    let mut actor = MacroActor::new(Arc::clone(&net), Resolver::Greedy.build(Arc::clone(&net)));
    play(&mut actor, 40);
    assert!(actor.degradations().over_axis > 0, "and the actor");
}

#[test]
fn a_non_finite_row_flags_the_line_it_recorded() {
    let net: Arc<dyn Evaluate> = Arc::new(NonFiniteRows(fixture()));
    let mut actor = MacroActor::new(Arc::clone(&net), Resolver::Greedy.build(Arc::clone(&net)));
    let report = play(&mut actor, 40);

    assert!(!report.macro_decisions.is_empty(), "the actor recorded");
    assert!(
        report.macro_decisions.iter().all(|step| step.degraded),
        "a line priced off a row that was not a number is flagged, so the \
         surrogate drops it and the value target keeps it"
    );
    let degradations = actor.degradations();
    assert!(degradations.non_finite > 0);
    assert_eq!(
        degradations.over_axis, 0,
        "a broken checkpoint is not a wide screen"
    );
}

#[test]
fn a_greedy_batch_reports_the_decisions_it_degraded() {
    let net: Arc<dyn Evaluate> = Arc::new(TinyAxis {
        net: fixture(),
        axis: 2,
    });
    let mut greedy = MacroGreedy::new(Arc::clone(&net), Resolver::Greedy.build(Arc::clone(&net)));
    let report = play(&mut greedy, 40);

    let mut summary = BatchSummary::new(provenance());
    summary.record(&report);
    let json = summary.json();
    let of = json["degraded"]["of"].as_u64().unwrap();
    let degraded = json["degraded"]["decisions"].as_u64().unwrap();
    assert!(of > 0, "the arm made decisions and the summary says so");
    assert!(degraded > 0, "and says how many it could not price");
    assert!(
        json["degraded"]["over_axis"].as_u64().unwrap() > 0,
        "with the cause beside the count"
    );
    assert!(
        summary
            .report()
            .contains(&format!("degraded {degraded} of {of}")),
        "the headline a batch prints carries the same two numbers"
    );
}
