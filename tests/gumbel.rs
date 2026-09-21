//! Gumbel selection contracts.
//!
//! The properties that separate it from UCT, each pinned on its own: the
//! budget lands on the candidates the prior sampled rather than spreading
//! over everything on offer; an edge nobody has descended is completed, not
//! condemned; the recorded root policy is the improved one rather than a
//! visit count; and none of it costs the fairness the belief mode is for.

use alphaspire::env::{Belief, TrueState};
use alphaspire::objective::{CombatStrength, Coverage};
use alphaspire::policy::RolloutPolicy;
use alphaspire::policy::UniformRandom;
use alphaspire::search::{
    BeliefSearch, EdgeStats, Gumbel, Mcts, NodeStats, SearchConfig, Selection, Uct,
    canonical_actions,
};
use sts2_core::UnlockPresetManifest;
use sts2_engine::Simulator;
use sts2_rng::MegaRandom;

const ITERATIONS: u32 = 32;

fn fresh_run(seed: &str) -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at(seed, &character, &preset, 0).unwrap()
}

fn config() -> SearchConfig {
    SearchConfig {
        iterations: ITERATIONS,
        rollout_depth: 5,
        temperature: 0.5,
    }
}

/// The seeded run walked to its first in-combat decision — the widest
/// branching the search meets early, and the one a schedule is for.
fn first_combat(seed: &str) -> Simulator {
    let mut simulator = fresh_run(seed);
    for _ in 0..40 {
        if simulator.state().combat.is_some() {
            return simulator;
        }
        let action = simulator
            .legal_actions()
            .iter()
            .find(
                |action| matches!(action, sts2_engine::Action::ChooseCards { cards, .. } if !cards.is_empty()),
            )
            .or_else(|| simulator.legal_actions().first())
            .cloned()
            .expect("a live run offers an action");
        simulator.step_quietly(&action).unwrap();
    }
    panic!("the seed reaches a combat inside forty decisions");
}

/// What one searched decision answers with: the action, what the budget
/// bought at the root, and the policy the search recorded there.
struct Searched {
    action: sts2_engine::Action,
    visits: Vec<(sts2_engine::Action, u64)>,
    policy: Vec<(sts2_engine::Action, f64)>,
}

/// One belief decision searched under the given selection.
fn search(selection: Box<dyn Selection>, simulator: &Simulator) -> Searched {
    let mut mcts = Mcts::new(config(), selection);
    let mut determinizer = Belief::from_simulator(simulator, 5).unwrap();
    let mut rollout = UniformRandom;
    let mut rng = MegaRandom::new(5);
    let (action, policy) = mcts.decide_with_policy(
        &mut determinizer,
        &CombatStrength::default(),
        &mut rollout,
        None,
        &mut rng,
    );
    Searched {
        action,
        visits: mcts.root_visits(),
        policy: policy.actions,
    }
}

#[test]
fn the_budget_lands_on_the_candidates_the_prior_sampled() {
    let original = first_combat("NLD6VZXP94");
    let offered = canonical_actions(original.legal_actions()).len();
    assert!(
        offered > 4,
        "the fixture decision branches wide enough to be worth concentrating: {offered}"
    );

    let considered = 2;
    let searched = search(
        Box::new(Gumbel {
            considered,
            ..Gumbel::default()
        }),
        &original,
    );
    let (action, visits) = (&searched.action, &searched.visits);
    assert!(
        original.legal_actions().contains(action),
        "the search answers with an action the real screen offers"
    );
    let spent: u64 = visits.iter().map(|(_, count)| count).sum();
    assert_eq!(
        spent,
        u64::from(ITERATIONS),
        "sequential halving spends the whole budget and no more"
    );
    let touched = visits.iter().filter(|(_, count)| *count > 0).count();
    assert!(
        touched <= considered,
        "the budget stayed on the {considered} sampled candidates, not {touched} actions"
    );

    // The contrast that makes the point: UCT at the same budget owes every
    // action on offer a first visit before it may prefer any of them.
    let spread = search(Box::new(Uct::default()), &original);
    let uct_touched = spread.visits.iter().filter(|(_, count)| *count > 0).count();
    assert!(
        uct_touched > touched,
        "UCT spread over {uct_touched} actions where the schedule held {touched}"
    );
}

#[test]
fn an_unvisited_edge_is_completed_not_condemned() {
    let mut gumbel = Gumbel::default();

    // UCT scores an unvisited edge at infinity, so it must be tried before
    // anything may be preferred. Completed-Q gives it the node's mixed value
    // instead — good enough to be worth a look under an even prior...
    let even = [
        EdgeStats {
            visits: 10,
            availability: 10,
            total_value: 5.0,
            prior: 0.5,
        },
        EdgeStats::fresh(0.5),
    ];
    assert_eq!(
        gumbel.descend(
            NodeStats {
                visits: 10,
                value: Some(0.5),
            },
            &even
        ),
        1,
        "an unvisited edge the prior likes as much is where the descents lag"
    );

    // ...and not worth one when the prior all but rules it out. UCT would
    // have taken it anyway, and at a small budget that is the whole budget.
    let lopsided = [
        EdgeStats {
            visits: 1,
            availability: 1,
            total_value: 1.0,
            prior: 0.99,
        },
        EdgeStats::fresh(0.01),
    ];
    assert_eq!(
        gumbel.descend(
            NodeStats {
                visits: 1,
                value: Some(1.0),
            },
            &lopsided
        ),
        0,
        "no edge is scored at infinity for the sole merit of being new"
    );
}

#[test]
fn the_recorded_policy_is_completed_not_counted() {
    let original = first_combat("NLD6VZXP94");
    let searched = search(
        Box::new(Gumbel {
            considered: 4,
            ..Gumbel::default()
        }),
        &original,
    );
    let (visits, policy) = (&searched.visits, &searched.policy);
    assert_eq!(visits.len(), policy.len(), "one weight per root edge");
    let total: f64 = policy.iter().map(|(_, weight)| weight).sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "the recorded policy is a distribution: {total}"
    );
    assert!(
        policy.iter().all(|(_, weight)| *weight > 0.0),
        "a candidate the halving starved is unexamined, not worthless"
    );

    // The point of recording completed-Q rather than counts: the halving
    // deliberately skews the visits, so the two distributions differ.
    let spent: u64 = visits.iter().map(|(_, count)| count).sum();
    #[allow(clippy::cast_precision_loss, reason = "visit counts are small")]
    let counted: Vec<f64> = visits
        .iter()
        .map(|(_, count)| *count as f64 / spent as f64)
        .collect();
    assert!(
        policy
            .iter()
            .zip(&counted)
            .any(|((_, weight), share)| (weight - share).abs() > 1e-6),
        "the improved policy is not the visit distribution it was measured beside"
    );
}

#[test]
fn a_gumbel_analysis_is_reproducible_and_still_fair() {
    // The keystone belief property, re-pinned under the new selection: two
    // states a player cannot tell apart get the same answer, and one seed
    // gives one answer.
    let original = first_combat("NLD6VZXP94");
    let twin = sts2_engine::BeliefState::from_simulator(&original)
        .unwrap()
        .sample(99, 4)
        .unwrap();
    assert_ne!(
        original.state_key().unwrap(),
        twin.state_key().unwrap(),
        "the twin hides different state"
    );
    let decide = |simulator: &Simulator| search(Box::new(Gumbel::default()), simulator).action;
    assert_eq!(decide(&original), decide(&original), "one seed, one answer");
    assert_eq!(
        decide(&original),
        decide(&twin),
        "what a player cannot distinguish, the analysis does not either"
    );
}

#[test]
fn the_schedule_drives_the_clairvoyant_mode_too() {
    // Nothing about the schedule is belief-specific: the same selection
    // behind the same core searches the true state, where the information
    // set is a singleton.
    let simulator = fresh_run("NLD6VZXP94");
    let mut mcts = Mcts::new(config(), Gumbel::default());
    let mut determinizer = TrueState::new(simulator.clone());
    let mut rollout = UniformRandom;
    let mut rng = MegaRandom::new(5);
    let action = mcts.decide(
        &mut determinizer,
        &Coverage::default(),
        &mut rollout,
        &mut rng,
    );
    assert!(
        simulator.legal_actions().contains(&action),
        "the search answers with a legal action"
    );
    assert!(mcts.tree_size() > 0, "the tree grew");
}

#[test]
fn a_gumbel_belief_walk_stays_legal_and_scriptable() {
    // The whole harness one flag over: Gumbel inside fights, the emitted
    // prefix still replays.
    let mut policy = BeliefSearch::new(config(), CombatStrength::default())
        .selecting(Box::new(Gumbel::default()));
    let mut objective = CombatStrength::default();
    let mut rng = MegaRandom::new(9);
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let report = alphaspire::selfplay::play_run(
        "NLD6VZXP94",
        &character,
        0,
        &mut policy,
        &mut rng,
        &mut objective,
        12,
    )
    .unwrap();
    report.script.expect("the searched prefix is scriptable");
    let mut replayed = fresh_run("NLD6VZXP94");
    for action in report.actions {
        replayed.step(action).unwrap();
    }
    assert!(policy.tree_size() > 0, "the fights were searched");
}

/// The statistics of one decision: four actions, a flat prior, and the
/// completed values a search came back with. `visits` is what
/// `value_weight` reads — the busiest edge at a 64-simulation root sits
/// around a dozen.
#[allow(clippy::cast_precision_loss, reason = "a dozen visits")]
fn decision(values: [f64; 4], visits: u64) -> (NodeStats, Vec<EdgeStats>) {
    let edges = values
        .iter()
        .map(|value| EdgeStats {
            visits,
            availability: visits,
            total_value: value * visits as f64,
            prior: 0.25,
        })
        .collect();
    (
        NodeStats {
            visits: visits * 4,
            value: Some(values.iter().sum::<f64>() / 4.0),
        },
        edges,
    )
}

#[test]
fn a_label_is_only_as_confident_as_the_evidence_under_it() {
    // The improvement transform floors its denominator at the objective's
    // own unit, so a decision's own span is what reaches the softmax until
    // that span is a whole unit wide. Two decisions whose evidence differs a
    // hundredfold must not write the same label.
    let schedule = Gumbel::default();
    let (weak_node, weak) = decision([0.500, 0.506, 0.499, 0.501], 12);
    let (strong_node, strong) = decision([0.10, 0.70, 0.05, 0.15], 12);

    let weak_label = alphaspire::search::improved_policy(schedule, weak_node, &weak);
    let strong_label = alphaspire::search::improved_policy(schedule, strong_node, &strong);

    assert!(
        weak_label[1] < 0.5,
        "six thousandths of evidence do not carry a confident label: {weak_label:?}"
    );
    assert!(
        strong_label[1] > 0.99,
        "six tenths of evidence do: {strong_label:?}"
    );
    // Both still rank the same action first — the ordering is evidence, the
    // confidence is what the evidence has to earn.
    for label in [&weak_label, &strong_label] {
        let best = (0..4)
            .max_by(|l, r| label[*l].total_cmp(&label[*r]))
            .unwrap();
        assert_eq!(best, 1, "the transform still ranks by value: {label:?}");
    }
}

#[test]
fn a_decision_wider_than_the_objective_is_min_maxed_as_before() {
    // The floor only ever raises the denominator. A decision that already
    // spans a unit or more transforms exactly as an unfloored min-max would,
    // so what an objective happens to report on still never sets the
    // exploration constant.
    let schedule = Gumbel::default();
    let (node, edges) = decision([0.0, 3.0, 1.0, 2.0], 12);
    let label = alphaspire::search::improved_policy(schedule, node, &edges);
    assert!(
        label[1] > 0.999,
        "a three-unit span is a one-hot, as it was: {label:?}"
    );
}

#[test]
fn a_searched_decision_records_the_evidence_its_label_rests_on() {
    // What the label does not carry, the sample file does: the completed
    // values' raw span, in the objective's own units, before the transform
    // normalized it away. `CombatStrength` scores inside [0, ~1.1], so a
    // spread is a fraction of that and a search that found its actions
    // genuinely alike reports a small one.
    let simulator = first_combat("NLD6VZXP94");
    let mut policy = BeliefSearch::new(config(), CombatStrength::default())
        .selecting(Box::new(Gumbel::default()))
        .recording();
    let mut rng = MegaRandom::new(4);
    policy.choose(&simulator, &mut rng);
    policy.run_ended(&simulator);

    let decisions = policy.drain_decisions();
    let recorded = decisions.first().expect("the decision was recorded");
    let spread = recorded.q_spread.expect("a searched decision has a spread");
    assert!(
        spread.is_finite() && (0.0..=1.2).contains(&spread),
        "the spread is in the objective's units, not the transform's: {spread}"
    );
}

#[test]
fn a_searched_decision_records_the_searchs_own_value_of_its_state() {
    // `z` arrives once per fight, smeared over every decision inside it; the
    // root's mixed value arrives once per decision, in the same units, and
    // is what a bootstrapped value target blends against. `CombatStrength`
    // scores inside [0, ~1.1], so the recorded value must too.
    let simulator = first_combat("NLD6VZXP94");
    let mut policy = BeliefSearch::new(config(), CombatStrength::default())
        .selecting(Box::new(Gumbel::default()))
        .recording();
    let mut rng = MegaRandom::new(4);
    policy.choose(&simulator, &mut rng);
    policy.run_ended(&simulator);

    let decisions = policy.drain_decisions();
    let recorded = decisions.first().expect("the decision was recorded");
    let value = recorded
        .root_value
        .expect("a searched decision has a root value");
    assert!(
        value.is_finite() && (0.0..=1.2).contains(&value),
        "the root value is in the objective's units: {value}"
    );
}
