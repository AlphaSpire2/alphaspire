//! The run-level PPO actor: a policy that plays whole runs, hands its fights
//! to a frozen resolver, and records the trajectory a learner updates on.
//!
//! One episode is one run. [`MacroActor`] answers the out-of-combat
//! decisions itself — pathing, card picks, events, rest sites, shops, reward
//! screens — by sampling from a checkpoint's priors, and hands every in-fight
//! decision to a [`Resolver`] held frozen for the length of a generation.
//! Frozen because a deployed model's behaviour *is* the environment: a
//! resolver that improved between one batch and the next would make the
//! environment non-stationary, and an advantage measured against a moving
//! environment measures nothing.
//!
//! **Sampling, not argmax**, and that is the whole difference between this
//! and every other net-driven policy in the crate. PPO forms a ratio between
//! the policy that acted and the policy being updated, which needs the
//! log-probability of an action genuinely drawn from a distribution. Argmax
//! would record a certainty the checkpoint never expressed and a ratio that
//! means nothing.
//!
//! Nothing here reads `simulator.state()`. The actor sees what a player sees,
//! through `agent_observation`; [`RunReward`] is the one privileged reader,
//! which is what an environment is.

use std::sync::Arc;

use sts2_engine::{Action, MapPointType, RestSiteOption, Simulator, VisibleMap};
use sts2_rng::MegaRandom;

use crate::net::{Degraded, Evaluate};
use crate::objective::CombatStrength;
use crate::plan::{ActionPlan, permitted_plans};
use crate::policy::{RolloutPolicy, permitted_actions};
use crate::reward::RunReward;
use crate::search::{
    BeliefSearch, Budget, BudgetSpend, Degradations, Gumbel, SearchConfig, canonical_actions,
};
use crate::training::Decision;

/// Default iteration budget for [`SearchedCombat`]. Kept low because combat
/// resolution runs at every in-fight decision of a macro training episode.
pub const RESOLVER_ITERATIONS: u32 = 16;

/// The cheap in-fight resolver: one forward pass per decision, and the action
/// the checkpoint likes best.
///
/// It is the bottom rung of the resolver ladder and what a budget downgrade
/// falls back to — a batch that cannot afford a search iteration can still
/// afford the prior the search would have started from. Argmax rather than a
/// sample, because a resolver is not the thing being trained: nothing records
/// its log-probabilities and nothing forms a ratio against them, so the
/// single strongest answer is the right one and a sampled one would only add
/// variance to the environment the learner is fitting.
pub struct NetGreedy {
    net: Arc<dyn Evaluate>,
    degradations: Degradations,
}

impl NetGreedy {
    /// This resolver over `net` — the frozen combat checkpoint.
    #[must_use]
    pub fn new(net: Arc<dyn Evaluate>) -> Self {
        Self {
            net,
            degradations: Degradations::default(),
        }
    }

    /// What this resolver answered with priors the checkpoint never
    /// produced. Read by whoever owns it, since a fallback resolver reports
    /// through its owner.
    #[must_use]
    pub const fn degradations(&self) -> Degradations {
        self.degradations
    }
}

impl RolloutPolicy for NetGreedy {
    fn choose(&mut self, simulator: &Simulator, _rng: &mut MegaRandom) -> Action {
        let (action, degraded) = greedy_action(self.net.as_ref(), simulator);
        self.degradations.priced(degraded);
        action
    }

    /// No search and so no budget, but the pricings it degraded are the
    /// run's all the same: a resolver answering by argmax off uniform priors
    /// is answering with the last action on the screen.
    fn budget_spent(&self) -> Option<BudgetSpend> {
        Some(BudgetSpend {
            degradations: self.degradations,
            ..BudgetSpend::default()
        })
    }
}

/// The single action `net` likes best at `simulator`, over the canonical
/// list.
///
/// One function for both greedy policies because the arithmetic is the same
/// arithmetic: what differs between resolving a fight this way and playing a
/// run this way is which decisions reach it and which checkpoint prices
/// them.
///
/// The second half of the answer is what stood in for the checkpoint where
/// anything did: over uniform priors argmax is the *last* action on the
/// screen, which is a fixed arbitrary pick and not a policy. A caller
/// discarding it is claiming a checkpoint chose.
fn greedy_action(net: &dyn Evaluate, simulator: &Simulator) -> (Action, Option<Degraded>) {
    let actions = canonical_actions(&permitted_actions(simulator));
    // See `Heuristic::choose`: an empty enumeration on a live run is an
    // engine fault to report, and only the authoritative run can reach a
    // policy with one.
    assert!(
        !actions.is_empty(),
        "a live run offers an action; the engine enumerated none at {:?}",
        simulator.decision()
    );
    let (priors, _, degraded) = net.priors_and_value(simulator, &actions).into_parts();
    let chosen = actions
        .iter()
        .zip(&priors)
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or_else(|| actions[0].clone(), |(action, _)| action.clone());
    (chosen, degraded)
}

/// The checkpoint's own answer inside a fight and the shared rollout's
/// everywhere else: the *student* half of a policy-improvement gate.
///
/// [`NetGreedy`] alone will not do as that half. It answers whatever it is
/// asked, so out of combat it plays the run by the combat net's argmax over
/// map steps and card picks — a policy the searched arm never runs, and a
/// difference between the arms that has nothing to do with what is being
/// measured. This delegates on the same `fight_over` fence
/// [`BeliefSearch`] delegates on, so the two
/// arms differ in exactly one thing: whether the fight's decisions were
/// searched.
pub struct GreedyCombat {
    resolver: NetGreedy,
    rollout: Box<dyn RolloutPolicy>,
}

impl GreedyCombat {
    /// This student over `net`, with `rollout` answering everything outside a
    /// fight.
    #[must_use]
    pub fn new(net: Arc<dyn Evaluate>, rollout: Box<dyn RolloutPolicy>) -> Self {
        Self {
            resolver: NetGreedy::new(net),
            rollout,
        }
    }
}

impl RolloutPolicy for GreedyCombat {
    fn choose(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> Action {
        if crate::env::fight_over(simulator) {
            self.rollout.choose(simulator, rng)
        } else {
            self.resolver.choose(simulator, rng)
        }
    }

    fn run_ended(&mut self, simulator: &Simulator) {
        self.rollout.run_ended(simulator);
    }

    fn stepped(&mut self, simulator: &Simulator) {
        self.rollout.stepped(simulator);
    }

    fn forced(&mut self, simulator: &Simulator, action: &Action) {
        self.rollout.forced(simulator, action);
    }

    fn budget_spent(&self) -> Option<BudgetSpend> {
        Some(BudgetSpend {
            degradations: self.resolver.degradations(),
            ..BudgetSpend::default()
        })
    }
}

/// One macro decision played by argmax: what the screen offered, the policy
/// the checkpoint priced it with, which of the offers argmax took, and what
/// stood in for the checkpoint where anything did.
struct GreedyPlan {
    plans: Vec<ActionPlan>,
    pi: Vec<f32>,
    chosen: usize,
    degraded: Option<Degraded>,
}

/// The plan `net` likes best at a macro screen, over the list
/// [`permitted_plans`] offers — which is [`greedy_action`]'s list with a
/// screen-opening button replaced by the answers behind it, plus a trade
/// wherever the belt is full and a potion is on offer.
fn greedy_plan(net: &dyn Evaluate, simulator: &Simulator) -> GreedyPlan {
    let plans = permitted_plans(simulator);
    // See `Heuristic::choose`: an empty enumeration on a live run is an
    // engine fault to report, and only the authoritative run can reach a
    // policy with one.
    assert!(
        !plans.is_empty(),
        "a live run offers an action; the engine enumerated none at {:?}",
        simulator.decision()
    );
    let (priors, _, degraded) = net.plan_priors_and_value(simulator, &plans).into_parts();
    let pi = distribution(&priors, plans.len());
    let chosen = pi
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index);
    GreedyPlan {
        plans,
        pi,
        chosen,
        degraded,
    }
}

/// The default in-fight resolver: belief search over sampled worlds under
/// Gumbel selection, with the frozen combat checkpoint supplying priors at
/// every node and a value at every leaf.
///
/// A wrapper rather than a bare [`BeliefSearch`] so that the composition has
/// a name and somewhere to say what it is for. What it fixes is exactly what
/// must not vary across a generation: [`CombatStrength`] as the objective,
/// Gumbel as the selection, one checkpoint as the evaluator. Gumbel because
/// its answer is provably no worse in expectation than the prior it starts
/// from at any budget, which is the guarantee a resolver running at a dozen
/// iterations needs.
///
/// The search core's own rollout policy sits underneath and is very nearly
/// unreachable: with an evaluator in hand every freshly expanded leaf costs
/// one forward pass instead of a walk to the end of the fight.
pub struct SearchedCombat {
    search: BeliefSearch<CombatStrength>,
    /// The base tree's settings, which a tier's own tree inherits everything
    /// but its budget from. See [`SearchedCombat::budgeting`].
    config: SearchConfig,
    selection: Gumbel,
    net: Arc<dyn Evaluate>,
    /// A tree of its own for the fights of a tier that spends a different
    /// budget, where a batch named one; the base tree answers the rest.
    elite: Option<BeliefSearch<CombatStrength>>,
    boss: Option<BeliefSearch<CombatStrength>>,
    /// What a downgraded decision falls back to: the same checkpoint, asked
    /// once instead of searched. See [`SearchedCombat::within`].
    fallback: NetGreedy,
    budget: Budget,
    /// When this run's clock started — the first decision this resolver was
    /// asked for, since a policy is built per run.
    started: Option<std::time::Instant>,
    searched: usize,
    downgraded: usize,
}

impl SearchedCombat {
    /// This resolver over `net` — the frozen combat checkpoint — at `config`
    /// under `selection`.
    #[must_use]
    pub fn new(net: Arc<dyn Evaluate>, config: SearchConfig, selection: Gumbel) -> Self {
        Self {
            search: BeliefSearch::new(config, CombatStrength::default())
                .selecting(Box::new(selection))
                .with_net(Arc::clone(&net)),
            config,
            selection,
            net: Arc::clone(&net),
            elite: None,
            boss: None,
            fallback: NetGreedy::new(net),
            budget: Budget::default(),
            started: None,
            searched: 0,
            downgraded: 0,
        }
    }

    /// The same resolver spending `budget` inside the fights of `tier`
    /// instead of the base budget.
    ///
    /// A boss is met once an act and decides the run, and a search that is
    /// barely past its prior at sixteen iterations in a hallway is the same
    /// search at the boss; the two are worth different budgets. The tier
    /// gets a tree of its own at its budget and the base tree is untouched,
    /// so a resolver that names no tier budget plays exactly as before. The
    /// run's step ceiling stays one ceiling over every tree.
    #[must_use]
    pub fn budgeting(mut self, tier: crate::library::Tier, budget: TierBudget) -> Self {
        let tree = BeliefSearch::new(
            SearchConfig {
                iterations: budget.iterations,
                ..self.config
            },
            CombatStrength::default(),
        )
        .selecting(Box::new(Gumbel {
            considered: budget.considered,
            ..self.selection
        }))
        .with_net(Arc::clone(&self.net));
        match tier {
            crate::library::Tier::Hallway => self.search = tree,
            crate::library::Tier::Elite => self.elite = Some(tree),
            crate::library::Tier::Boss => self.boss = Some(tree),
        }
        self
    }

    /// The tree that answers the standing fight: the tier's own where the
    /// tier has one, the base tree otherwise.
    fn tree_for(&mut self, simulator: &Simulator) -> &mut BeliefSearch<CombatStrength> {
        let room = simulator
            .state()
            .run
            .as_ref()
            .and_then(sts2_engine::RunState::standing_room_type);
        match crate::library::Tier::of(room) {
            crate::library::Tier::Elite if self.elite.is_some() => {
                self.elite.as_mut().expect("checked above")
            }
            crate::library::Tier::Boss if self.boss.is_some() => {
                self.boss.as_mut().expect("checked above")
            }
            _ => &mut self.search,
        }
    }

    /// Every tree's tier-budgeted and base steps together: the run's spend.
    fn steps_taken(&self) -> u64 {
        [Some(&self.search), self.elite.as_ref(), self.boss.as_ref()]
            .into_iter()
            .flatten()
            .map(BeliefSearch::steps_taken)
            .sum()
    }

    /// The same resolver under a per-run soft ceiling: past it every in-fight
    /// decision is answered by [`NetGreedy`] instead, and the run walks on.
    ///
    /// The combat arm had no ceiling of any kind, and a batch's wall time is
    /// its slowest run: one fight that searches its way into a long line —
    /// a loop, a screen the tree keeps reopening, a boss whose turns never
    /// stop — is enough to hold a whole unattended generation. The
    /// [`Budget`] is the act arm's own machinery rather than a second copy,
    /// including the part that matters most: a ceiling read only *between*
    /// decisions is not a ceiling, because the decision that never ends is
    /// never asked about, so it stands inside the decision's own simulation
    /// loop too.
    ///
    /// **The downgrade falls back to the checkpoint, not to nothing.** The
    /// act arm hands a downgraded screen to its rollout policy; a resolver
    /// has no such thing, and the search core's own rollout plays fights at
    /// uniform random. What a run out of search budget can still afford is
    /// the prior the search would have started from, which is exactly
    /// [`NetGreedy`].
    ///
    /// Only [`Budget::steps`] is reachable from the command line. A rollout
    /// writes training data, and a wall-clock cap makes what it downgrades
    /// depend on what else the box was doing, so the episodes it recorded
    /// would no longer be a function of their seed pairs.
    #[must_use]
    pub const fn within(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Whether this run has spent its resolver budget. Starts the run's clock
    /// on the first ask, which is the first in-fight decision of the run — a
    /// policy is built per run, so that is the run's own start.
    fn over_budget(&mut self) -> bool {
        let started = *self.started.get_or_insert_with(std::time::Instant::now);
        self.budget
            .steps
            .is_some_and(|cap| self.steps_taken() >= cap)
            || self
                .budget
                .seconds
                .is_some_and(|cap| started.elapsed().as_secs_f64() >= cap)
    }
}

impl RolloutPolicy for SearchedCombat {
    fn choose(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> Action {
        if self.budget.is_off() {
            self.searched += 1;
            return self.tree_for(simulator).choose(simulator, rng);
        }
        if self.over_budget() {
            self.downgraded += 1;
            crate::probe::tally(crate::probe::Tally::Downgrades, 1);
            return self.fallback.choose(simulator, rng);
        }
        self.searched += 1;
        // The budget check above started the run's clock; the same budget now
        // stands over this one decision's simulations, so a fight that walks
        // past the cap mid-decision stops there rather than at the next
        // decision that never comes. The ceiling is the run's, over every
        // tree: what the other trees have spent comes off this one's.
        let started = self.started.expect("the budget check started the clock");
        let spent = self.steps_taken();
        let budget = self.budget;
        let tree = self.tree_for(simulator);
        let elsewhere = spent - tree.steps_taken();
        tree.under(
            Budget {
                steps: budget.steps.map(|cap| cap.saturating_sub(elsewhere)),
                ..budget
            }
            .ceiling(started),
        );
        tree.choose(simulator, rng)
    }

    fn run_ended(&mut self, simulator: &Simulator) {
        self.search.run_ended(simulator);
        for tree in [self.elite.as_mut(), self.boss.as_mut()]
            .into_iter()
            .flatten()
        {
            tree.run_ended(simulator);
        }
    }

    fn budget_spent(&self) -> Option<BudgetSpend> {
        // Every tree's degradations and the fallback's: a downgraded
        // decision is answered by the fallback, so all of them are this
        // resolver's.
        let mut degradations = self.search.degradations();
        for tree in [self.elite.as_ref(), self.boss.as_ref()]
            .into_iter()
            .flatten()
        {
            degradations.merge(tree.degradations());
        }
        degradations.merge(self.fallback.degradations());
        Some(BudgetSpend {
            searched: self.searched,
            downgraded: self.downgraded,
            steps: self.steps_taken(),
            degradations,
        })
    }
}

/// Which of the two resolvers answers the decisions inside a fight while the
/// macro actor plays the run around them.
///
/// A small owned description rather than a built policy, because the batch
/// harness builds one policy per run and a run must be a function of its
/// index: the resolver is described once and constructed fresh for every
/// episode.
/// The search budget one tier's fights spend, where it differs from the
/// resolver's base budget. See [`SearchedCombat::budgeting`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TierBudget {
    pub iterations: u32,
    pub considered: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Resolver {
    /// [`NetGreedy`]: one forward pass per in-fight decision.
    Greedy,
    /// [`SearchedCombat`]: belief search under Gumbel, which is the default
    /// and what a generation is expected to run.
    Searched {
        config: SearchConfig,
        selection: Gumbel,
        /// The per-run ceiling past which the search downgrades to
        /// [`NetGreedy`], off unless a batch set one. See
        /// [`SearchedCombat::within`].
        budget: Budget,
        /// A budget of the elite fights' own, where a batch named one.
        elite: Option<TierBudget>,
        /// A budget of the boss fights' own, where a batch named one.
        boss: Option<TierBudget>,
    },
}

impl Default for Resolver {
    fn default() -> Self {
        Self::Searched {
            config: SearchConfig {
                iterations: RESOLVER_ITERATIONS,
                ..SearchConfig::default()
            },
            selection: Gumbel::default(),
            budget: Budget::default(),
            elite: None,
            boss: None,
        }
    }
}

impl Resolver {
    /// This resolver over `net`, the frozen combat checkpoint, as a policy
    /// one episode can be handed.
    #[must_use]
    pub fn build(self, net: Arc<dyn Evaluate>) -> Box<dyn RolloutPolicy> {
        match self {
            Self::Greedy => Box::new(NetGreedy::new(net)),
            Self::Searched {
                config,
                selection,
                budget,
                elite,
                boss,
            } => {
                let mut searched = SearchedCombat::new(net, config, selection).within(budget);
                if let Some(tier) = elite {
                    searched = searched.budgeting(crate::library::Tier::Elite, tier);
                }
                if let Some(tier) = boss {
                    searched = searched.budgeting(crate::library::Tier::Boss, tier);
                }
                Box::new(searched)
            }
        }
    }
}

/// The PPO learner's own policy: it plays the run, samples every macro
/// decision from a checkpoint, hands its fights to a [`Resolver`], and
/// records the episode as a trajectory.
///
/// **What one recorded step is.** At an out-of-combat decision the actor
/// encodes what the player sees, prices the canonical action list through
/// the checkpoint, samples an action from the resulting distribution, and
/// records the observation, the action list, the whole masked policy, the
/// index sampled, its log-probability and the critic's value. In-fight
/// decisions are the resolver's and are recorded nowhere: they are the
/// environment, not the policy.
///
/// **What a step is paid.** A macro decision is not one engine step. Stepping
/// a path choice enters the room, and the engine then offers a whole fight's
/// worth of decisions the resolver answers before the actor is asked again.
/// Everything the run collects across that span — floors climbed, the act
/// crossed, the death walked into — is the consequence of the decision that
/// entered the room, so it accumulates through [`RolloutPolicy::stepped`] and
/// is charged to that decision when the actor next stands at a macro screen.
/// Charging it to whichever in-fight step happened to be adjacent to the
/// floor advance would put the signal somewhere the macro net never looks,
/// and elite-versus-rest routing — the thing a run policy exists to learn —
/// would not be learnable at all.
///
/// **Where the episode ends.** A run that reaches a terminal ends its last
/// recorded step with `done`. A run the harness's step cap stops is
/// *truncated* instead: `done` stays false and the last line carries a
/// `bootstrap` value for the state it stopped on, so nothing teaches the
/// critic that a climb ends where a batch ran out of patience. Every episode
/// is bounded, because the harness bounds it; a run that cannot terminate
/// truncates rather than failing.
///
/// **`z`** is settled at the end of the episode as the plain, undiscounted
/// suffix sum of the recorded rewards. The discount belongs to the learner
/// and to nothing else — two places to keep it is two places for it to
/// disagree — so no discount appears here. The learner computes its own
/// returns and advantages from `reward` and reads `z` as the diagnostic that
/// makes a collapsed return distribution visible at a glance.
pub struct MacroActor {
    net: Arc<dyn Evaluate>,
    resolver: Box<dyn RolloutPolicy>,
    /// The run's reward, opened on the first decision so that its floor
    /// baseline is the floor the run actually starts on.
    reward: Option<RunReward>,
    /// The weights of the reward's optional terms, applied when the reward
    /// is opened on the first decision.
    reward_terms: (f64, f64, f64),
    /// Which gold the reward's gold term pays for.
    gold_scope: crate::reward::GoldScope,
    /// What the reward pays a boss killed, by act.
    boss_terms: crate::reward::BossWeights,
    /// What the environment has paid since the last recorded step, waiting to
    /// be charged to it. See [`MacroActor::charge`].
    accrued: f64,
    steps: Vec<Decision>,
    /// What this actor answered with priors the checkpoint never produced,
    /// by cause. The same fact rides on each affected line as
    /// [`Decision::degraded`](crate::training::Decision); this is the half
    /// that says why.
    degradations: Degradations,
    /// What the episode is throwing away and what it is fighting, collected
    /// as it plays. The actor is the only thing that sees a fight start and
    /// end, so it is the only thing that can record one.
    metrics: crate::summary::EpisodeMetrics,
    /// Exploration at rest sites: the fraction of the sampled distribution
    /// handed to uniform over the screen's plans, there and nowhere else.
    /// What is recorded is the *mixed* distribution — the one the action was
    /// actually drawn from — so the ratio a learner forms against these
    /// lines is against the true behaviour policy. Uniform over plans
    /// rather than options on purpose: the smith stands one plan per
    /// upgradable card, so a uniform share reaches every card the deck can
    /// upgrade — the exposure the exploration exists to buy.
    explore_rest: Option<f64>,
    /// The same mixing fraction over map-navigation decisions: `epsilon` of
    /// the sampled distribution handed to uniform over the destinations, and
    /// the mix recorded as the behaviour policy. Exploration for the paths —
    /// the elite branches above all — a risk-averse habit stops walking; the
    /// payoff it exists to expose is the relic and gold a fought elite banks.
    explore_map: Option<f64>,
    /// Whether the episode has been settled. `run_ended` is the harness's
    /// word and comes once, but settling twice would double the suffix sums.
    settled: bool,
}

impl MacroActor {
    /// An actor playing one run: `net` is the run checkpoint it samples its
    /// macro decisions from and prices its states with, and `resolver`
    /// answers everything inside a fight.
    ///
    /// Built fresh per episode, which is what makes a run a function of its
    /// index: the actor owns its trajectory, its reward and its resolver, and
    /// shares no mutable state with any other run of the batch.
    #[must_use]
    pub fn new(net: Arc<dyn Evaluate>, resolver: Box<dyn RolloutPolicy>) -> Self {
        Self {
            net,
            resolver,
            reward: None,
            reward_terms: (0.0, 0.0, 0.0),
            gold_scope: crate::reward::GoldScope::Any,
            boss_terms: [0.0; 3],
            accrued: 0.0,
            steps: Vec::new(),
            degradations: Degradations::default(),
            metrics: crate::summary::EpisodeMetrics::default(),
            explore_rest: None,
            explore_map: None,
            settled: false,
        }
    }

    /// The same actor paying the reward's two optional terms — an elite
    /// fight won, a relic gained — at these weights. What the critic then
    /// predicts is named by [`crate::reward::value_semantics`], and the
    /// checkpoint on the other side must carry the same name.
    #[must_use]
    pub const fn with_reward_terms(
        mut self,
        elite_weight: f64,
        relic_weight: f64,
        gold_weight: f64,
    ) -> Self {
        self.reward_terms = (elite_weight, relic_weight, gold_weight);
        self
    }

    /// The same actor paying the gold term for `scope` only.
    #[must_use]
    pub const fn with_gold_scope(mut self, scope: crate::reward::GoldScope) -> Self {
        self.gold_scope = scope;
        self
    }

    /// The same actor paying a boss killed at these weights, by the act it
    /// was killed in.
    #[must_use]
    pub const fn with_boss_terms(mut self, weights: crate::reward::BossWeights) -> Self {
        self.boss_terms = weights;
        self
    }

    /// This run's macro decisions the checkpoint did not price, by cause.
    ///
    /// Whether wide macro decisions occur at all is a measurement rather than
    /// an assumption, and this is where the measurement comes from.
    #[must_use]
    pub const fn degradations(&self) -> Degradations {
        self.degradations
    }

    /// Charges everything accrued since the last recorded step to that step,
    /// and opens the accumulator again.
    ///
    /// The first macro decision of a run has nothing before it, so the
    /// accumulator is empty and there is nothing to charge.
    fn charge(&mut self) {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a run's reward is a handful of floors and one terminal term"
        )]
        let paid = std::mem::replace(&mut self.accrued, 0.0) as f32;
        if let Some(last) = self.steps.last_mut()
            && let Some(reward) = last.reward.as_mut()
        {
            *reward += paid;
        }
    }

    /// Records the macro decision standing at `simulator` and answers with
    /// the plan sampled from the checkpoint's policy.
    fn act(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> ActionPlan {
        let actions = permitted_plans(simulator);
        // See `Heuristic::choose`: an empty enumeration on a live run is an
        // engine fault to report, and only the authoritative run can reach a
        // policy with one.
        assert!(
            !actions.is_empty(),
            "a live run offers an action; the engine enumerated none at {:?}",
            simulator.decision()
        );
        // Priors the checkpoint did not produce are honest degradation for a
        // search, which loses guidance on one node, and dishonest for a
        // learner, which would form its ratio against a log-probability no
        // net emitted. The evaluator says which it handed back; the line is
        // kept, flagged, for the value target it is still good for.
        let (priors, value, degraded) = self
            .net
            .plan_priors_and_value(simulator, &actions)
            .into_parts();
        self.degradations.priced(degraded);
        let mut pi = distribution(&priors, actions.len());
        let mut exploration_epsilon = 0.0;
        let exploring = match simulator.decision() {
            sts2_engine::DecisionContext::RestSite { .. } => self.explore_rest,
            sts2_engine::DecisionContext::MapNavigation { .. } => self.explore_map,
            _ => None,
        };
        if let Some(epsilon) = exploring
            && pi.len() > 1
        {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss,
                reason = "a mixing weight over tens of plans is far inside f32's range"
            )]
            let uniform = (epsilon / pi.len() as f64) as f32;
            #[allow(clippy::cast_possible_truncation, reason = "epsilon is in [0, 1]")]
            {
                exploration_epsilon = epsilon as f32;
            }
            #[allow(
                clippy::cast_possible_truncation,
                reason = "a mixing weight is far inside f32's range"
            )]
            for weight in &mut pi {
                *weight = (1.0 - epsilon) as f32 * *weight + uniform;
            }
        }
        let chosen = sampled(&pi, rng);
        let plan = actions[chosen].clone();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a log-probability and a run value are far inside f32's range"
        )]
        let step = Decision {
            observation: simulator.agent_observation(),
            logp: Some(f64::from(pi[chosen]).max(f64::MIN_POSITIVE).ln() as f32),
            exploration_epsilon: Some(exploration_epsilon),
            chosen: Some(chosen),
            value: Some(value as f32),
            actions,
            pi,
            // Settled at the end of the episode; see `run_ended`.
            z: 0.0,
            // A macro decision belongs to no fight and to no searched act:
            // the episode is the run, and the run stamp the sink writes is
            // the grouping key a trajectory needs.
            encounter: None,
            run: 0,
            fight: 0,
            act: None,
            // Opened at zero and filled by `charge` when the next macro
            // decision arrives, or when the run ends.
            reward: Some(0.0),
            done: false,
            bootstrap: None,
            degraded: degraded.is_some(),
            counterfactual: false,
            counterfactual_kind: None,
            // No tree stood here: this is a checkpoint's own distribution,
            // not a search's improvement on one, and the critic's read
            // already rides in `value`.
            q_spread: None,
            root_value: None,
        };
        self.metrics.decided(simulator, &step.actions, &plan);
        self.steps.push(step);
        plan
    }
}

impl MacroActor {
    /// The same actor with `epsilon` of its rest-site distribution handed
    /// to uniform over the screen's plans. See the field: the mix is the
    /// recorded behaviour policy, so these episodes stay honest training
    /// data — the exploration is *in* the distribution, not an override of
    /// it.
    #[must_use]
    pub const fn exploring_rest(mut self, epsilon: f64) -> Self {
        self.explore_rest = Some(epsilon);
        self
    }

    /// The same actor with `epsilon` of its map-navigation distribution
    /// handed to uniform over the destinations. As with the rest mix, the
    /// blend is the recorded behaviour policy, so the episodes stay honest
    /// training data.
    #[must_use]
    pub const fn exploring_map(mut self, epsilon: f64) -> Self {
        self.explore_map = Some(epsilon);
        self
    }
}

impl RolloutPolicy for MacroActor {
    fn choose(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> Action {
        self.choose_plan(simulator, rng).lead().clone()
    }

    fn choose_plan(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> ActionPlan {
        let (elite_weight, relic_weight, gold_weight) = self.reward_terms;
        let gold_scope = self.gold_scope;
        let boss_terms = self.boss_terms;
        self.reward.get_or_insert_with(|| {
            RunReward::starting(simulator)
                .with_terms(elite_weight, relic_weight, gold_weight)
                .with_gold_scope(gold_scope)
                .with_boss_terms(boss_terms)
        });
        if crate::env::fight_over(simulator) {
            self.charge();
            self.act(simulator, rng)
        } else {
            ActionPlan::Single(self.resolver.choose(simulator, rng))
        }
    }

    fn stepped(&mut self, simulator: &Simulator) {
        self.resolver.stepped(simulator);
        self.metrics.stepped(simulator);
        if let Some(reward) = self.reward.as_mut() {
            self.accrued += reward.paid(simulator);
        }
    }

    fn forced(&mut self, simulator: &Simulator, action: &Action) {
        self.metrics.forced(simulator, action);
    }

    fn run_ended(&mut self, simulator: &Simulator) {
        self.resolver.run_ended(simulator);
        if self.settled {
            return;
        }
        self.settled = true;
        self.metrics.ended(simulator);
        // Whatever the run collected after its last macro decision belongs to
        // that decision, exactly as it would have had another one followed.
        self.charge();
        let terminal = crate::reward::terminated(simulator);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a run value is far inside f32's range"
        )]
        let bootstrap =
            (!terminal && !self.steps.is_empty()).then(|| self.net.state_value(simulator) as f32);
        if let Some(last) = self.steps.last_mut() {
            last.done = terminal;
            last.bootstrap = bootstrap;
        }
        let mut carry = 0.0;
        for step in self.steps.iter_mut().rev() {
            carry += step.reward.unwrap_or(0.0);
            step.z = carry;
        }
    }

    fn drain_decisions(&mut self) -> Vec<Decision> {
        self.resolver.drain_decisions()
    }

    fn drain_macro_decisions(&mut self) -> Vec<Decision> {
        std::mem::take(&mut self.steps)
    }

    fn drain_run_metrics(&mut self) -> Option<crate::summary::EpisodeMetrics> {
        Some(std::mem::take(&mut self.metrics))
    }

    /// The resolver's spend, because the resolver is the only thing in an
    /// episode with a search budget: the actor prices its own decisions with
    /// one forward pass each. Its own degraded pricings ride out on the same
    /// struct, which is the run's one channel for them.
    fn budget_spent(&self) -> Option<BudgetSpend> {
        let mut spend = self.resolver.budget_spent().unwrap_or_default();
        spend.degradations.merge(self.degradations);
        Some(spend)
    }
}

/// The run policy as it is *deployed*: every out-of-combat decision the
/// checkpoint's single best action, every in-fight decision the resolver's,
/// and nothing recorded.
///
/// The arm a macro promotion gate grades, and deliberately not
/// [`MacroActor`]. An actor samples because PPO needs an on-policy
/// distribution to form a ratio against, and a sampled arm would grade a
/// checkpoint on play nobody will ever deploy — two arms would then differ by
/// their draws as much as by their weights, and a paired comparison over
/// identical seeds exists precisely to remove everything that is not the
/// difference under test. Argmax is what the checkpoint will actually do, so
/// argmax is what is measured.
///
/// It pays no reward — a gate reads how deep each arm got, which the run's
/// own report already carries — but it keeps the run's own
/// [`EpisodeMetrics`](crate::summary::EpisodeMetrics) and the decisions it
/// made. Both are observations of what the run walked through rather than
/// anything a learner consumes, and a batch summary reporting "degraded 0 of
/// 0 decisions" over a run that made two hundred states a falsehood about the
/// play it just watched.
///
/// What its recorded lines carry is the policy it played by and none of the
/// trajectory keys: argmax names no action drawn from a distribution, so
/// there is no log-probability to record and no ratio to form against one.
/// A rollout refuses `--greedy` beside an emit flag for exactly that reason,
/// and these lines reach no sink.
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent ablation switches, not mutually exclusive states"
)]
pub struct MacroGreedy {
    net: Arc<dyn Evaluate>,
    resolver: Box<dyn RolloutPolicy>,
    metrics: crate::summary::EpisodeMetrics,
    steps: Vec<Decision>,
    degradations: Degradations,
    /// Ablation override: at a rest site offering a smith, take a smith
    /// plan whenever hit points stand at or above this fraction of maximum.
    /// See [`MacroGreedy::forcing_smith`].
    force_smith: Option<f64>,
    /// Under the override, smith a uniformly random upgradable card instead
    /// of the checkpoint's preferred one. The control that separates the
    /// timing of a smith from the choice of its card.
    random_smith_card: bool,
    /// Ablation override: claim a relic standing on a reward screen.
    force_relics: bool,
    /// Ablation override: from this act (numbered from one) onward, a card
    /// reward's claim is never the answer — the screen's other lines and
    /// its exit stay the policy's own choice. See
    /// [`MacroGreedy::forcing_card_skips`].
    force_skip_cards: Option<u32>,
    /// Ablation override: wherever a shop offers a card removal, buy one.
    /// See [`MacroGreedy::forcing_removals`].
    force_remove: bool,
    /// Under the removal override, remove a uniformly random card instead
    /// of the checkpoint's preferred one — the same control the smith has.
    random_removal_card: bool,
    /// Ablation override: walk into an elite the map offers as a next room.
    /// See [`MacroGreedy::forcing_elites`].
    force_elite: Option<EliteForcing>,
}

/// Where the elite forcing applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EliteForcing {
    /// Wherever the map offers an elite, whatever stands beside it — a rest
    /// site, a shop, a treasure. The upper bound on elite exposure, and a
    /// measurement of elites *instead of everything*.
    Everywhere,
    /// Only where every other next room is a fight or an unknown room: the
    /// elite-versus-monster choice, which is the one the checkpoint faces
    /// most often and paths around — measured without giving up the rest
    /// sites and shops it would otherwise have walked to.
    OverFights,
}

impl MacroGreedy {
    /// This policy over `net` — the run checkpoint on trial — with `resolver`
    /// answering everything inside a fight.
    ///
    /// The resolver is the environment and must be identical across the arms
    /// of a gate: a match whose arms resolved their fights differently would
    /// measure the resolvers, and the run checkpoints only incidentally.
    #[must_use]
    pub fn new(net: Arc<dyn Evaluate>, resolver: Box<dyn RolloutPolicy>) -> Self {
        Self {
            net,
            resolver,
            metrics: crate::summary::EpisodeMetrics::default(),
            steps: Vec::new(),
            degradations: Degradations::default(),
            force_smith: None,
            random_smith_card: false,
            force_relics: false,
            force_skip_cards: None,
            force_remove: false,
            random_removal_card: false,
            force_elite: None,
        }
    }

    /// The same policy with a rest site that offers a smith answered by the
    /// checkpoint's best smith plan whenever hit points stand at or above
    /// `threshold` of maximum, whatever the checkpoint preferred. Zero
    /// forces the smith everywhere.
    ///
    /// An ablation arm, not a policy anyone deploys: it measures what the
    /// smith the policy refuses is worth by paying it wherever the
    /// threshold says, paired against the unforced arm over identical
    /// seeds. The card smithed is still the policy's own choice — the
    /// override picks *that* a smith happens, and the checkpoint's
    /// preference among smith plans picks which — so the arms differ by
    /// exactly the decision under test.
    #[must_use]
    pub const fn forcing_smith(mut self, threshold: f64) -> Self {
        self.force_smith = Some(threshold);
        self
    }

    /// The forcing with its card chosen uniformly at random instead of by
    /// the checkpoint. Paired against [`MacroGreedy::forcing_smith`] over
    /// identical seeds, the two arms differ by exactly the card choice —
    /// which is how an uninformed preference is told apart from a real one.
    #[must_use]
    pub const fn smithing_random_cards(mut self) -> Self {
        self.random_smith_card = true;
        self
    }

    /// Claim a relic on a reward screen, choosing the checkpoint's preferred
    /// relic when several stand there. Treasure-room relics are mandatory.
    #[must_use]
    pub const fn forcing_relics(mut self) -> Self {
        self.force_relics = true;
        self
    }

    /// The same policy never answering a card reward's claim from act `act`
    /// (numbered from one) onward. Gold, potions and relics on the same
    /// screen stay the policy's own play, and so does the exit — only the
    /// draft is withheld. An ablation arm: it measures what deck discipline
    /// is worth by paying it everywhere, against the unforced arm over
    /// identical seeds.
    #[must_use]
    pub const fn forcing_card_skips(mut self, act: u32) -> Self {
        self.force_skip_cards = Some(act);
        self
    }

    /// The same policy buying a card removal wherever a shop offers one it
    /// can afford, the checkpoint choosing which card goes. An ablation arm
    /// for the removal the policy almost never buys on its own.
    #[must_use]
    pub const fn forcing_removals(mut self) -> Self {
        self.force_remove = true;
        self
    }

    /// The removal forcing with its card chosen uniformly at random instead
    /// of by the checkpoint — the control that separates buying removals at
    /// all from knowing what to remove.
    /// The same policy walking into an elite the map offers as a next room,
    /// the checkpoint choosing among elites where there are several, and
    /// `where` saying which alternatives it overrides. An ablation arm for
    /// the elites the policy paths around: against the unforced arm over
    /// identical seeds it measures what an elite actually returns under
    /// this resolver — the cost the critic charges for one and the relic
    /// and gold it never sees, both realised.
    #[must_use]
    pub const fn forcing_elites(mut self, r#where: EliteForcing) -> Self {
        self.force_elite = Some(r#where);
        self
    }

    #[must_use]
    pub const fn removing_random_cards(mut self) -> Self {
        self.random_removal_card = true;
        self
    }

    /// This run's macro decisions the checkpoint did not price, by cause.
    #[must_use]
    pub const fn degradations(&self) -> Degradations {
        self.degradations
    }
}

/// The indexes of the smith plans on a screen, in offer order.
fn smith_plans(plans: &[ActionPlan]) -> Vec<usize> {
    plans
        .iter()
        .enumerate()
        .filter(|(_, plan)| {
            matches!(
                plan.lead(),
                Action::RestOption {
                    option: RestSiteOption::Smith,
                    ..
                }
            )
        })
        .map(|(index, _)| index)
        .collect()
}

/// The smith plan the policy likes best, where the screen offers one.
fn best_smith(plans: &[ActionPlan], pi: &[f32]) -> Option<usize> {
    smith_plans(plans)
        .into_iter()
        .max_by(|left, right| pi[*left].total_cmp(&pi[*right]))
}

/// The checkpoint's preferred relic claim on a reward screen, if any.
fn best_relic_claim(plans: &[ActionPlan], pi: &[f32]) -> Option<usize> {
    plans
        .iter()
        .enumerate()
        .filter(|(_, plan)| {
            matches!(
                plan.lead(),
                Action::ClaimReward { fingerprint, .. } if fingerprint.reward_type == "relic"
            )
        })
        .map(|(index, _)| index)
        .max_by(|left, right| pi[*left].total_cmp(&pi[*right]))
}

/// Whether this plan answers a card reward's claim — the draft the skip
/// ablation withholds. Reward lines carrying gold, potions or relics are
/// not drafts and stay out of it.
fn claims_a_card(plan: &ActionPlan) -> bool {
    matches!(
        plan.lead(),
        Action::ClaimReward { fingerprint, .. } if !fingerprint.offered_cards.is_empty()
    )
}

/// The indexes of the card-removal plans on a screen, in offer order.
fn removal_plans(plans: &[ActionPlan]) -> Vec<usize> {
    plans
        .iter()
        .enumerate()
        .filter(|(_, plan)| matches!(plan.lead(), Action::BuyCardRemoval))
        .map(|(index, _)| index)
        .collect()
}

/// The indexes of the map plans whose destination is an elite room, where
/// the act's map is visible and the forcing applies to what stands beside
/// them. Empty where it does not, so the checkpoint's own choice stands.
fn elite_plans(
    plans: &[ActionPlan],
    map: Option<&VisibleMap>,
    forcing: EliteForcing,
) -> Vec<usize> {
    let Some(map) = map else {
        return Vec::new();
    };
    let point_type = |plan: &ActionPlan| match plan.lead() {
        Action::ChooseMap { destination } => map
            .points
            .iter()
            .find(|point| point.coord == *destination)
            .map(|point| point.point_type),
        _ => None,
    };
    let is_elite = |plan: &ActionPlan| matches!(point_type(plan), Some(MapPointType::Elite));
    if forcing == EliteForcing::OverFights
        && plans.iter().any(|plan| {
            !is_elite(plan)
                && !matches!(
                    point_type(plan),
                    Some(MapPointType::Monster | MapPointType::Unknown)
                )
        })
    {
        return Vec::new();
    }
    plans
        .iter()
        .enumerate()
        .filter(|(_, plan)| is_elite(plan))
        .map(|(index, _)| index)
        .collect()
}

/// The plan the policy likes best outside the card claims, where one
/// stands. There always is one on a reward screen — the exit at minimum —
/// but the caller still falls back rather than trusting that.
fn best_non_claim(plans: &[ActionPlan], pi: &[f32]) -> Option<usize> {
    (0..plans.len())
        .filter(|index| !claims_a_card(&plans[*index]))
        .max_by(|left, right| pi[*left].total_cmp(&pi[*right]))
}

impl RolloutPolicy for MacroGreedy {
    fn choose(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> Action {
        self.choose_plan(simulator, rng).lead().clone()
    }

    fn choose_plan(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> ActionPlan {
        if !crate::env::fight_over(simulator) {
            return ActionPlan::Single(self.resolver.choose(simulator, rng));
        }
        let decided = greedy_plan(self.net.as_ref(), simulator);
        let observation = simulator.agent_observation();
        let healthy = f64::from(observation.current_hp.unwrap_or(0))
            >= self.force_smith.unwrap_or(f64::INFINITY)
                * f64::from(observation.max_hp.unwrap_or(1));
        let chosen = if healthy {
            let forced = if self.random_smith_card {
                let smiths = smith_plans(&decided.plans);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss,
                    reason = "a screen offers tens of plans"
                )]
                (!smiths.is_empty()).then(|| {
                    smiths[(rng.next_double() * smiths.len() as f64) as usize % smiths.len()]
                })
            } else {
                best_smith(&decided.plans, &decided.pi)
            };
            forced.unwrap_or(decided.chosen)
        } else {
            decided.chosen
        };
        let chosen = if self.force_relics {
            best_relic_claim(&decided.plans, &decided.pi).unwrap_or(chosen)
        } else {
            chosen
        };
        // The deck-discipline ablations, each on its own screen: a shop's
        // removal is bought wherever one stands, and from the named act
        // onward a card reward's claim is never the answer.
        let chosen = if self.force_remove {
            let removals = removal_plans(&decided.plans);
            if removals.is_empty() {
                chosen
            } else if self.random_removal_card {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss,
                    reason = "a screen offers tens of plans"
                )]
                {
                    removals[(rng.next_double() * removals.len() as f64) as usize % removals.len()]
                }
            } else {
                removals
                    .into_iter()
                    .max_by(|left, right| decided.pi[*left].total_cmp(&decided.pi[*right]))
                    .unwrap_or(chosen)
            }
        } else {
            chosen
        };
        let chosen = match self.force_skip_cards {
            Some(act)
                if observation.current_act.unwrap_or(0) + 1 >= act as usize
                    && claims_a_card(&decided.plans[chosen]) =>
            {
                best_non_claim(&decided.plans, &decided.pi).unwrap_or(chosen)
            }
            _ => chosen,
        };
        let chosen = match self.force_elite {
            Some(forcing) => elite_plans(&decided.plans, observation.map.as_ref(), forcing)
                .into_iter()
                .max_by(|left, right| decided.pi[*left].total_cmp(&decided.pi[*right]))
                .unwrap_or(chosen),
            None => chosen,
        };
        let plan = decided.plans[chosen].clone();
        self.degradations.priced(decided.degraded);
        self.metrics.decided(simulator, &decided.plans, &plan);
        self.steps.push(Decision {
            observation,
            actions: decided.plans,
            pi: decided.pi,
            z: 0.0,
            encounter: None,
            run: 0,
            fight: 0,
            act: None,
            chosen: None,
            logp: None,
            exploration_epsilon: None,
            value: None,
            reward: None,
            done: false,
            bootstrap: None,
            degraded: decided.degraded.is_some(),
            counterfactual: false,
            counterfactual_kind: None,
            q_spread: None,
            root_value: None,
        });
        plan
    }

    fn stepped(&mut self, simulator: &Simulator) {
        self.resolver.stepped(simulator);
        self.metrics.stepped(simulator);
    }

    fn forced(&mut self, simulator: &Simulator, action: &Action) {
        self.metrics.forced(simulator, action);
    }

    fn run_ended(&mut self, simulator: &Simulator) {
        self.resolver.run_ended(simulator);
        self.metrics.ended(simulator);
    }

    fn drain_macro_decisions(&mut self) -> Vec<Decision> {
        std::mem::take(&mut self.steps)
    }

    fn drain_run_metrics(&mut self) -> Option<crate::summary::EpisodeMetrics> {
        Some(std::mem::take(&mut self.metrics))
    }

    fn budget_spent(&self) -> Option<BudgetSpend> {
        let mut spend = self.resolver.budget_spent().unwrap_or_default();
        spend.degradations.merge(self.degradations);
        Some(spend)
    }
}

/// `priors` as a distribution over `count` actions: normalized where it sums
/// to anything at all, uniform where it does not.
///
/// Normalized once, here, because the recorded `pi` and the `logp` the
/// learner forms its ratio against have to be the same numbers. A checkpoint
/// answering a degenerate row and an over-wide decision's uniform fallback
/// both land on the same answer, which is the only distribution that claims
/// nothing.
fn distribution(priors: &[f32], count: usize) -> Vec<f32> {
    #[allow(
        clippy::cast_precision_loss,
        reason = "action counts are far below f32 precision"
    )]
    let uniform = 1.0 / count.max(1) as f32;
    if priors.len() != count {
        return vec![uniform; count];
    }
    let total: f32 = priors.iter().sum();
    if total > 0.0 {
        priors.iter().map(|weight| weight / total).collect()
    } else {
        vec![uniform; count]
    }
}

/// One index drawn from `pi`.
///
/// The draw comes off the stream the harness hands the policy rather than off
/// a split analysis stream, because the actor *is* the run's policy: its
/// randomness is play, not analysis. A resolver that searches splits its own
/// stream underneath, which is what keeps two arms differing only in their
/// search from drawing different macro actions.
fn sampled(pi: &[f32], rng: &mut MegaRandom) -> usize {
    let total: f64 = pi.iter().map(|weight| f64::from(*weight)).sum();
    let mut draw = rng.next_double() * total;
    for (index, weight) in pi.iter().enumerate() {
        draw -= f64::from(*weight);
        if draw <= 0.0 {
            return index;
        }
    }
    pi.len().saturating_sub(1)
}
