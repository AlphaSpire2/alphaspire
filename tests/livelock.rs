//! The greedy arm cannot stall on a decision that changes nothing.
//!
//! Two shapes reach the same place. A run checkpoint that prices a potion
//! claim above everything beside it, played by argmax onto a belt with
//! nothing free on it, answers with the same claim forever: the step is
//! accepted, the screen stands where it stood, and the next decision is the
//! one just made. And a policy that claims a card reward and then skips
//! the pick screen leaves the reward unanswered, so the claim is offered
//! again — a cycle no single step is a no-op in. Sampled play escapes such a
//! self-loop geometrically; argmax does not, and a policy whose entropy has
//! collapsed samples like argmax.

use std::path::Path;
use std::sync::Arc;

use alphaspire::actor::{MacroGreedy, Resolver};
use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::{Evaluate, PolicyValueNet, Priced};
use alphaspire::objective::CombatStrength;
use alphaspire::plan::ActionPlan;
use alphaspire::policy::RolloutPolicy;
use sts2_engine::{Action, ChoiceScreen, DecisionContext, Simulator};
use sts2_rng::MegaRandom;

/// The analysis seed of the traced generation-zero batch. The macro arm is
/// argmax, so this only steers the fights under it.
const ANALYSIS: u64 = 17_518_610_013_652_630_564;

/// Runs to walk, the traced one first.
///
/// More than one because reaching the livelock's precondition — a belt with
/// nothing free on it, standing at another potion — is a property of the
/// *walk*, and the walk is the fixture checkpoint's. Every rebuild of that
/// fixture, which every encoding bump forces, moves which seeds arrive there:
/// the traced pair reached it under the v7 fixture and does not under the v8
/// one, and pinning a single seed made the guard silently vacuous rather than
/// failing. So the run is what varies and the property is what is asserted —
/// at least one of these stands at the offer, and every one that does answers
/// it with a trade and walks to an ending of its own.
const SEEDS: [&str; 4] = ["P3SH3CG2XR", "NLD6VZXP94", "L4XKYLQCFM", "N3Z9EE2CXB"];

/// The fixture checkpoint with the two habits the trace showed: it takes
/// every potion offered and gives none up. The traced row read 0.516 on the
/// claim, 0.484 on proceeding and 1e-14 on every discard, and those two
/// habits together are what leave a belt with nothing free on it — the only
/// state the inert claim is reachable from.
struct TracedHabits(Arc<dyn Evaluate>);

impl Evaluate for TracedHabits {
    fn priors_and_value(&self, simulator: &Simulator, actions: &[Action]) -> Priced {
        let (mut priors, value, degraded) =
            self.0.priors_and_value(simulator, actions).into_parts();
        for (prior, action) in priors.iter_mut().zip(actions) {
            match action {
                Action::ClaimReward { fingerprint, .. } if fingerprint.reward_type == "potion" => {
                    *prior = 2.0;
                }
                // A potion on a shelf is a potion offered. Left to the
                // fixture's own weights this was whatever that checkpoint
                // happened to think of the price, so whether the habit held
                // at a shop moved with every retrain of the fixture; the
                // habit is the premise of the test and belongs here.
                Action::BuyShopItem {
                    fingerprint: sts2_engine::ShopItem::Potion(_),
                    ..
                } => *prior = 2.0,
                Action::UsePotion { .. } | Action::DiscardPotion { .. } => *prior = 0.0,
                _ => {}
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
fn the_livelock_reproducer_runs_to_a_terminal() {
    let registry = sts2_content::standard_registry();
    let encoder = Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ));
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let mut stood_at_the_offer = 0;
    for seed in SEEDS {
        let fixture: Arc<dyn Evaluate> = Arc::new(
            PolicyValueNet::load(&base, Arc::clone(&encoder))
                .expect("the fixture checkpoint loads"),
        );
        let net: Arc<dyn Evaluate> = Arc::new(TracedHabits(fixture));
        let mut policy: Box<dyn RolloutPolicy> = Box::new(MacroGreedy::new(
            Arc::clone(&net),
            Resolver::Greedy.build(net),
        ));
        let mut rng = MegaRandom::new(ANALYSIS);
        let mut objective = CombatStrength::default();
        let report = alphaspire::selfplay::play_run(
            seed,
            &"CHARACTER.IRONCLAD".parse().unwrap(),
            0,
            policy.as_mut(),
            &mut rng,
            &mut objective,
            4000,
        )
        .expect("the run walks");

        assert!(
            report.terminal.is_some(),
            "{seed} ended on its own rather than on the step cap, at floor {}",
            report.floor
        );
        let metrics = report.metrics.expect("the greedy arm keeps its metrics");
        let trades = metrics.trades();
        if trades.reward_offered > 0 {
            stood_at_the_offer += 1;
        }
        assert_eq!(
            trades.taken, trades.offered,
            "{seed} answered each with a trade rather than with a claim that \
             changes nothing: {trades:?}"
        );
        // The plan block is what widens a macro decision, so the run says how
        // wide it got against the axis the checkpoint prices.
        let widest = metrics.widest();
        assert!(
            widest.any <= alphaspire::net::MAX_ACTIONS,
            "{seed}'s widest decision fits the priced axis: {widest:?}"
        );
    }
    assert!(
        stood_at_the_offer > 0,
        "and at least one of them did stand at a potion offer with the belt \
         full, which is the only state the inert claim is reachable from — \
         without it the assertions above pass by never arriving"
    );
}

/// The two habits the screen livelock needs, over the same fixture
/// checkpoint: a card reward's claim priced above everything beside it, and
/// the pick screen it opens answered by taking nothing. Every other decision
/// keeps the fixture's own prices, and everything inside a fight is left
/// alone — only the run policy asks for plans.
struct ScreenHabits(Arc<dyn Evaluate>);

/// Whether this plan claims a card reward: the button that spends nothing
/// and opens a pick screen.
fn claims_a_card_reward(plan: &ActionPlan) -> bool {
    plan.steps().any(|action| match action {
        Action::ClaimReward { fingerprint, .. } => !fingerprint.offered_cards.is_empty(),
        _ => false,
    })
}

impl Evaluate for ScreenHabits {
    fn priors_and_value(&self, simulator: &Simulator, actions: &[Action]) -> Priced {
        self.0.priors_and_value(simulator, actions)
    }

    fn plan_priors_and_value(&self, simulator: &Simulator, plans: &[ActionPlan]) -> Priced {
        let (mut priors, value, degraded) =
            self.0.plan_priors_and_value(simulator, plans).into_parts();
        let picking = matches!(
            simulator.decision(),
            DecisionContext::ChooseCards {
                screen: ChoiceScreen::CardReward,
                ..
            }
        );
        for (prior, plan) in priors.iter_mut().zip(plans) {
            if claims_a_card_reward(plan) {
                *prior = 3.0;
            } else if picking && let Action::ChooseCards { cards, .. } = plan.outcome() {
                *prior = if cards.is_empty() { 2.0 } else { 1.0 };
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

/// The run the cycle caught, off the traced generation-zero batch.
const SCREEN_SEED: &str = "L4XKYLQCFM";
const SCREEN_ANALYSIS: u64 = 3_402_165_191_353_373_922;

#[test]
fn the_screen_livelock_reproducer_runs_to_a_terminal() {
    let registry = sts2_content::standard_registry();
    let encoder = Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ));
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let fixture: Arc<dyn Evaluate> =
        Arc::new(PolicyValueNet::load(&base, encoder).expect("the fixture checkpoint loads"));
    let net: Arc<dyn Evaluate> = Arc::new(ScreenHabits(fixture));
    let mut policy: Box<dyn RolloutPolicy> = Box::new(MacroGreedy::new(
        Arc::clone(&net),
        Resolver::Greedy.build(net),
    ));
    let mut rng = MegaRandom::new(SCREEN_ANALYSIS);
    let mut objective = CombatStrength::default();
    let report = alphaspire::selfplay::play_run(
        SCREEN_SEED,
        &"CHARACTER.IRONCLAD".parse().unwrap(),
        0,
        policy.as_mut(),
        &mut rng,
        &mut objective,
        500,
    )
    .expect("the run walks");

    assert!(
        report.terminal.is_some(),
        "the run ended on its own rather than on the step cap, at floor {}",
        report.floor
    );
    let metrics = report.metrics.expect("the greedy arm keeps its metrics");
    let waste = metrics.waste();
    assert!(
        waste.card_rewards > 0,
        "and it did answer a card reward: {waste:?}"
    );
    assert_eq!(
        waste.card_rewards_skipped, 0,
        "with a card every time: the habit prices the claim above the exit, \
         so it never walks off the screen with one standing: {waste:?}"
    );
    let widest = metrics.widest();
    assert!(
        widest.any <= alphaspire::net::MAX_ACTIONS,
        "and the widest decision it stood at fits the priced axis: {widest:?}"
    );
}
