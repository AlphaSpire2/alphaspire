//! A card reward declined is counted as one, and the loot beside it cannot
//! be left.
//!
//! Backing out of the pick screen is withheld from the run policy, so the
//! only skip a policy can take is to proceed off the reward screen with the
//! card's claim still standing. The waste counters read the skip off that
//! leaving step — a counter that watched only for an empty pick sat at zero
//! for every batch ever played, whatever the policy did. The gold and the
//! potions the belt has room for are claimed by the harness before the
//! policy is asked, so a policy that prices the exit above everything still
//! leaves with them: the counters read them as settled and never as left.

use std::path::Path;
use std::sync::Arc;

use alphaspire::actor::{MacroGreedy, Resolver};
use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::{Evaluate, PolicyValueNet, Priced};
use alphaspire::objective::CombatStrength;
use alphaspire::plan::ActionPlan;
use alphaspire::policy::RolloutPolicy;
use sts2_engine::{Action, DecisionContext, Simulator};
use sts2_rng::MegaRandom;

const SEED: &str = "L4XKYLQCFM";
const ANALYSIS: u64 = 3_402_165_191_353_373_922;

/// The fixture checkpoint with one habit laid over it: on a reward screen,
/// the exit is priced above everything on the screen, so every reward is
/// walked away from with all of its lines standing.
struct LeavesEveryReward(Arc<dyn Evaluate>);

impl Evaluate for LeavesEveryReward {
    fn priors_and_value(&self, simulator: &Simulator, actions: &[Action]) -> Priced {
        self.0.priors_and_value(simulator, actions)
    }

    fn plan_priors_and_value(&self, simulator: &Simulator, plans: &[ActionPlan]) -> Priced {
        let (mut priors, value, degraded) =
            self.0.plan_priors_and_value(simulator, plans).into_parts();
        if matches!(simulator.decision(), DecisionContext::Rewards { .. }) {
            for (prior, plan) in priors.iter_mut().zip(plans) {
                if matches!(plan.lead(), Action::Proceed | Action::AdvanceAct) {
                    *prior = 3.0;
                }
            }
        }
        match degraded {
            Some(cause) => Priced::fallback(priors, value, cause),
            None => Priced::real(priors, value),
        }
    }

    fn state_value(&self, simulator: &Simulator) -> f64 {
        self.0.state_value(simulator)
    }
}

#[test]
fn walking_off_a_reward_screen_counts_the_card_gold_and_potions_left_on_it() {
    let registry = sts2_content::standard_registry();
    let encoder = Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ));
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let fixture: Arc<dyn Evaluate> =
        Arc::new(PolicyValueNet::load(&base, encoder).expect("the fixture checkpoint loads"));
    let net: Arc<dyn Evaluate> = Arc::new(LeavesEveryReward(fixture));
    let mut policy: Box<dyn RolloutPolicy> = Box::new(MacroGreedy::new(
        Arc::clone(&net),
        Resolver::Greedy.build(net),
    ));
    let mut rng = MegaRandom::new(ANALYSIS);
    let mut objective = CombatStrength::default();
    let report = alphaspire::selfplay::play_run(
        SEED,
        &"CHARACTER.IRONCLAD".parse().unwrap(),
        0,
        policy.as_mut(),
        &mut rng,
        &mut objective,
        4000,
    )
    .expect("the run walks");

    let metrics = report.metrics.expect("the greedy arm keeps its metrics");
    let waste = metrics.waste();
    assert!(
        waste.card_rewards > 0,
        "the run won a fight and stood at its reward: {waste:?}"
    );
    assert_eq!(
        waste.card_rewards_skipped, waste.card_rewards,
        "and every card reward it stood at was left standing: {waste:?}"
    );
    assert!(
        waste.gold_rewards > 0,
        "the gold beside it was on the screen: {waste:?}"
    );
    assert_eq!(
        (waste.gold_rewards_left, waste.gold_left),
        (0, 0),
        "and was claimed before the policy could leave it: {waste:?}"
    );
    assert_eq!(
        waste.potion_rewards_left, 0,
        "as was every potion it had room for: {waste:?}"
    );
}
