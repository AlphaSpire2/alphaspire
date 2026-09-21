//! The run-level PPO actor and the trajectory it records.
//!
//! Five properties. A recorded step names an action actually sampled from
//! the policy on the line beside it. The reward of a macro step is everything
//! the run collected until the *next* macro step, so a decision that walks
//! into a fight is paid for the fight. An episode that reaches a terminal is
//! `done` and carries no bootstrap; an episode the step cap stops carries a
//! bootstrap and is not `done`. `z` is the undiscounted suffix sum of the
//! rewards. And the whole trajectory is a function of the seed pair, so two
//! runs of one index are the same run.

use std::path::Path;
use std::sync::Arc;

use alphaspire::actor::{MacroActor, Resolver};
use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::{Evaluate, PolicyValueNet};
use alphaspire::objective::CombatStrength;
use alphaspire::policy::RolloutPolicy;
use alphaspire::search::{Gumbel, SearchConfig};
use alphaspire::selfplay::RunReport;
use alphaspire::training::Decision;
use sts2_rng::MegaRandom;

fn encoder() -> Arc<PolicyEncoder> {
    let registry = sts2_content::standard_registry();
    Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ))
}

/// The fixture checkpoint, standing in for both the run net and the frozen
/// combat net: what is under test is the actor's bookkeeping, and a tiny net
/// prices a position as honestly as a trained one does.
fn fixture() -> Arc<dyn Evaluate> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    Arc::new(PolicyValueNet::load(&base, encoder()).expect("the fixture checkpoint loads"))
}

/// One episode: an actor built fresh, a run played to its terminal or to
/// `max_steps`, and the trajectory it drained.
fn episode(net: &Arc<dyn Evaluate>, resolver: Resolver, max_steps: usize) -> RunReport {
    let mut policy: Box<dyn RolloutPolicy> = Box::new(MacroActor::new(
        Arc::clone(net),
        resolver.build(Arc::clone(net)),
    ));
    let mut rng = MegaRandom::new(7);
    let mut objective = CombatStrength::default();
    alphaspire::selfplay::play_run(
        "NLD6VZXP94",
        &"CHARACTER.IRONCLAD".parse().unwrap(),
        0,
        policy.as_mut(),
        &mut rng,
        &mut objective,
        max_steps,
    )
    .expect("the run walks")
}

/// A resolver cheap enough to compose one whole run out of, without asking a
/// debug build for a generation's worth of search.
fn cheap_search() -> Resolver {
    Resolver::Searched {
        config: SearchConfig {
            iterations: 2,
            rollout_depth: 4,
            ..SearchConfig::default()
        },
        selection: Gumbel::default(),
        budget: alphaspire::search::Budget::default(),
        elite: None,
        boss: None,
    }
}

fn rewards(trajectory: &[Decision]) -> Vec<f32> {
    trajectory
        .iter()
        .map(|step| step.reward.expect("a trajectory line carries its reward"))
        .collect()
}

#[test]
fn every_recorded_step_names_an_action_sampled_from_the_policy_beside_it() {
    let trajectory = episode(&fixture(), Resolver::Greedy, 4000).macro_decisions;
    assert!(!trajectory.is_empty(), "a run makes macro decisions");
    for step in &trajectory {
        assert_eq!(step.exploration_epsilon, Some(0.0));
        let chosen = step.chosen.expect("a trajectory line names its action");
        assert!(chosen < step.actions.len(), "the index is into `actions`");
        assert_eq!(step.pi.len(), step.actions.len(), "the policy is masked");
        let total: f32 = step.pi.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "the policy sums to one: {total}"
        );
        let logp = step.logp.expect("a trajectory line names its probability");
        assert!(
            (logp - step.pi[chosen].ln()).abs() < 1e-4,
            "the log-probability is of the action taken: {logp}"
        );
        assert!(
            step.value.expect("the critic priced the state").is_finite(),
            "the value is a number"
        );
        assert!(
            !step.degraded,
            "no act-one decision offers more actions than the net prices"
        );
    }
}

#[test]
fn a_macro_step_is_paid_for_everything_the_run_collects_before_the_next_one() {
    // Only out-of-combat decisions are recorded, and a run takes far more
    // steps than it records: everything in between is the fight the recorded
    // decision walked into, and its reward has to land on that decision.
    let report = episode(&fixture(), Resolver::Greedy, 4000);
    let trajectory = &report.macro_decisions;
    assert!(
        trajectory.len() < report.actions.len(),
        "the resolver answered the steps between the recorded ones"
    );
    assert!(
        report.decisions.is_empty(),
        "a rollout records the macro half and nothing else"
    );
    let paid = rewards(trajectory);
    assert!(
        paid.iter().any(|reward| *reward > 0.0),
        "some decision was paid for the floors it climbed: {paid:?}"
    );
}

#[test]
fn an_episode_that_reaches_a_terminal_is_done_and_needs_no_bootstrap() {
    let report = episode(&fixture(), Resolver::Greedy, 4000);
    assert!(
        report.terminal.is_some(),
        "the seed's run ends inside the cap"
    );
    let trajectory = &report.macro_decisions;
    let last = trajectory.last().expect("the run recorded a decision");
    assert!(last.done, "the last step of a terminal episode is done");
    assert_eq!(
        last.bootstrap, None,
        "a run that ended needs no value for what came after it"
    );
    assert!(
        trajectory[..trajectory.len() - 1]
            .iter()
            .all(|step| !step.done && step.bootstrap.is_none()),
        "and no earlier step claims either"
    );
    // The terminal term is on the wire, on the step that walked into it.
    let paid = rewards(trajectory);
    assert!(
        *paid.last().unwrap() < 0.0,
        "the defeat term was paid: {paid:?}"
    );
}

#[test]
fn an_episode_the_step_cap_stops_is_truncated_rather_than_finished() {
    // A climb the batch stopped watching is not a climb that ended. Read as
    // terminal it would teach the critic that the run dies wherever the cap
    // falls, so the last line says `done: false` and carries the value of the
    // state it stopped on instead.
    let report = episode(&fixture(), Resolver::Greedy, 40);
    assert_eq!(report.terminal, None, "the cap stopped the run early");
    let last = report
        .macro_decisions
        .last()
        .expect("the run recorded a decision");
    assert!(!last.done, "a truncated episode is not done");
    assert!(
        last.bootstrap.is_some_and(f32::is_finite),
        "and it carries the value of the state it stopped on"
    );
}

#[test]
fn z_is_the_undiscounted_suffix_sum_of_the_episode_rewards() {
    // The discount is the learner's and lives on exactly one side of the
    // boundary. What the rollout writes is the plain suffix sum, which is
    // what keeps a collapsed return distribution visible to the audit.
    let trajectory = episode(&fixture(), cheap_search(), 400).macro_decisions;
    let paid = rewards(&trajectory);
    let mut carry = 0.0;
    for (step, reward) in trajectory.iter().zip(&paid).rev() {
        carry += reward;
        assert!(
            (step.z - carry).abs() < 1e-4,
            "z is the suffix sum: {} against {carry}",
            step.z
        );
    }
    assert!(
        (trajectory[0].z - paid.iter().sum::<f32>()).abs() < 1e-4,
        "so the first line's z is the whole episode's return"
    );
}

#[test]
fn one_seed_pair_plays_one_episode() {
    // A run is a function of its index and nothing else, sampled macro
    // decisions included: a batch that could not reproduce its own episodes
    // could not pair two arms over them either.
    let net = fixture();
    let first = episode(&net, cheap_search(), 400).macro_decisions;
    let again = episode(&net, cheap_search(), 400).macro_decisions;
    assert_eq!(first.len(), again.len(), "the same run made the same calls");
    assert_eq!(first, again, "and recorded the same trajectory");
}
