//! Rollout and baseline policies.
//!
//! A policy owns no randomness: it is handed a deterministic RNG seeded from
//! the explicit analysis seed, so a trajectory is reproducible from
//! (run seed, analysis seed, configuration). A learned policy slots in behind
//! the same trait later.

use std::borrow::Cow;

use sts2_core::ModelId;
use sts2_engine::{
    Action, ChoicePurpose, Command, ContentRegistry, DecisionContext, EventEffect, EventOption,
    ShopItem, Simulator, SuspendedChoice,
};
use sts2_rng::MegaRandom;

use crate::plan::ActionPlan;

/// Whether obtaining this relic is a step the engine refuses: a relic whose
/// `on_obtained` body is `Command::Unsupported` — the default
/// `offered_only` leaves when a definition does not override it
/// (`sts2-content/src/relics.rs`). One carries it at this baseline:
/// `RELIC.MASSIVE_SCROLL`, multiplayer-only and never offered here. The
/// check stays general because the shape is the TRIAL trap's exact shape —
/// a legal choice the engine then refuses, voiding the run — and a later
/// content change can leave a relic refused again.
fn relic_refused_on_obtain(content: &ContentRegistry, relic: &ModelId) -> bool {
    content.relic(relic).is_some_and(|definition| {
        definition
            .on_obtained
            .iter()
            .any(|command| matches!(command, Command::Unsupported(_)))
    })
}

/// Whether choosing this event option would step straight into a body the
/// engine refuses. Two shapes exist at this baseline:
///
/// - An option whose own effects carry [`EventEffect::Unsupported`]:
///   EVENT.TRIAL's Reject→Double-down line (`sts2-content/src/events.rs`) —
///   the build's double-down opens the abandon-run confirmation, the
///   simulator carries it as `Unsupported`, and stepping it voids the
///   run.
/// - An option that obtains a relic whose `AfterObtained` body the simulator
///   refuses (see `relic_refused_on_obtain`): nothing a single-player run
///   is offered today, and any relic a later content change leaves refused.
///
/// The check is over the option's *own* effect list, not the pages behind
/// it — an option that merely opens a page holding a poisoned option is fine
/// to take, because the poisoned option is filtered again on its own page.
/// It deliberately does not walk arbitrary `Run` command bodies: nothing
/// registered hides an `Unsupported` there, and simulating command bodies is
/// the engine's job, not a filter's.
#[must_use]
pub fn poisoned_event_option(content: &ContentRegistry, option: &EventOption) -> bool {
    option.effects.iter().any(|effect| match effect {
        EventEffect::Unsupported(_) => true,
        EventEffect::ObtainRelic(relic)
        | EventEffect::ObtainRelicWithCard { relic, .. }
        | EventEffect::ObtainSeaGlass { relic, .. }
        | EventEffect::ObtainRelicAndRandomCard { relic, .. }
        | EventEffect::ObtainRelicAndGeneratedRewards { relic, .. } => {
            relic_refused_on_obtain(content, relic)
        }
        EventEffect::TradeRelic { taken, .. } => relic_refused_on_obtain(content, taken),
        // A roll over candidates may land on any of them, so one refused
        // candidate poisons the whole option: stepping it can void the run.
        EventEffect::ObtainRelicRolledFrom { candidates } => candidates
            .iter()
            .any(|relic| relic_refused_on_obtain(content, relic)),
        _ => false,
    })
}

/// Whether the engine would refuse to step this action even though it
/// enumerated it. Anything that chooses actions — a policy, a search node,
/// a fallback — must never take one.
#[must_use]
pub fn engine_refuses(simulator: &Simulator, action: &Action) -> bool {
    let Action::ChooseEvent { index, .. } = action else {
        return false;
    };
    let DecisionContext::Event { options, .. } = simulator.decision() else {
        return false;
    };
    options
        .get(*index)
        .is_some_and(|option| poisoned_event_option(simulator.content(), option))
}

/// Whether this option's body has already run, which makes clicking it a
/// step that does nothing at all: the engine returns without running
/// anything and the page stands where it stood.
///
/// Public for the same reason [`poisoned_event_option`] is — the rule is
/// worth pinning against hand-built pages rather than only against a run
/// that happens to walk onto one.
#[must_use]
pub fn spent_event_option(option: &EventOption) -> bool {
    option.was_chosen
}

/// Whether the belt holds a potion in every slot it has.
#[must_use]
pub fn belt_is_full(simulator: &Simulator) -> bool {
    let belt = &simulator.state().run_player.potions;
    !belt.is_empty() && belt.iter().all(Option::is_some)
}

/// Whether stepping this action would leave the state exactly as it found
/// it: a potion acquisition with no slot to put the potion in, or an event
/// option whose body has already run.
///
/// The engine enumerates such a click and accepts it, because the game does
/// — a shop potion bought onto a full belt fails for lack of space with
/// nothing paid and the offer still on the shelf, and a potion reward
/// leaves its potion on the table the same way. A policy has no use for a
/// step whose only effect is to return the screen it was taken on; freeing a
/// slot first is [`crate::plan::ActionPlan::Trade`].
///
/// The event case is the stronger of the two, because the engine says so
/// itself rather than leaving it to be inferred off the belt: an option
/// already marked chosen returns without running anything, and the page
/// stands exactly where it stood. Left on the list it is a livelock waiting
/// for an argmax policy — one that priced the option highest once prices it
/// highest forever, and sampled play is the only thing that escapes. A run
/// was caught doing precisely that, spending 4,843 of its 4,906 decisions
/// re-clicking one spent Orobas option.
#[must_use]
pub fn step_is_inert(simulator: &Simulator, action: &Action) -> bool {
    if let Action::ChooseEvent { index, .. } = action {
        let DecisionContext::Event { options, .. } = simulator.decision() else {
            return false;
        };
        return options.get(*index).is_some_and(spent_event_option);
    }
    let takes_a_potion = match action {
        Action::ClaimReward { fingerprint, .. } => fingerprint.reward_type == "potion",
        Action::BuyShopItem { fingerprint, .. } => matches!(fingerprint, ShopItem::Potion(_)),
        _ => return false,
    };
    takes_a_potion && belt_is_full(simulator)
}

/// The card-selection screen standing over the run, wherever it is held.
///
/// A fight's own screen is the one the player is answering while a fight is
/// open, and the run's is the one they answer outside a fight — the order
/// `compute_decision` reads them in.
#[must_use]
pub fn open_choice(simulator: &Simulator) -> Option<&SuspendedChoice> {
    let state = simulator.state();
    match &state.combat {
        Some(combat) => combat.suspended_choice.as_ref(),
        None => state
            .run
            .as_ref()
            .and_then(|run| run.suspended_choice.as_ref()),
    }
}

/// Whether backing out of the screen this answer stands on costs nothing and
/// re-offers the button that opened it.
///
/// Five screens have that shape: a card reward, a removal reward, the
/// merchant's removal, the smith and the cook. Their openers are spent only
/// by an answer, so the empty answer closes the screen, leaves the opener
/// where it was, and a policy that keeps choosing the same way takes the
/// pair forever. Declining is still expressible — it is `Proceed` on the
/// screen the opener stood on, or the rest site's other options.
///
/// Every other cancelable screen is a card offer that is *skipped* rather
/// than backed out of: the page behind it has moved on either way, and
/// taking nothing is a real answer.
fn cancel_re_offers_the_opener(simulator: &Simulator, action: &Action) -> bool {
    let Action::ChooseCards { cards, .. } = action else {
        return false;
    };
    if !cards.is_empty() {
        return false;
    }
    open_choice(simulator).is_some_and(|choice| {
        matches!(
            choice.purpose,
            ChoicePurpose::AddRewardCard { .. }
                | ChoicePurpose::RemoveDeckCardAtShop { .. }
                | ChoicePurpose::RestSmith { .. }
                | ChoicePurpose::RestCook { .. }
                | ChoicePurpose::RemoveDeckCardForReward { .. }
        )
    })
}

/// Whether this action throws a potion off the belt outside a fight.
///
/// The engine offers a discard for every held potion at every screen, and
/// out of a fight the step only ever destroys the potion: there is nothing
/// to make room for except an offer, and taking an offer over a full belt is
/// [`crate::plan::ActionPlan::Trade`], which names its own discard. A bare
/// discard is therefore dominated by not discarding — the potion kept can
/// still be thrown away later, or drunk — and a policy has no use for it.
/// Left on the list it was taken: five potions per run thrown away under
/// greedy play, nine in ten with no potion picked up afterwards on the same
/// screen and six in ten with two slots still free, and the act-3 boss
/// entered with an empty belt. A discard inside a fight stays, where it is
/// the engine's own answer to a belt that must be emptied.
#[must_use]
pub fn discards_off_the_belt_outside_a_fight(simulator: &Simulator, action: &Action) -> bool {
    matches!(action, Action::DiscardPotion { .. })
        && !simulator
            .state()
            .combat
            .as_ref()
            .is_some_and(|combat| combat.in_progress)
}

/// Whether this action is kept off the list a policy chooses from: the
/// engine would refuse it, stepping it would change nothing, it only
/// closes a screen its own opener will put straight back up, or it throws a
/// potion away outside a fight.
pub(crate) fn withheld(simulator: &Simulator, action: &Action) -> bool {
    engine_refuses(simulator, action)
        || step_is_inert(simulator, action)
        || cancel_re_offers_the_opener(simulator, action)
        || discards_off_the_belt_outside_a_fight(simulator, action)
}

/// The legal actions worth choosing between: `legal_actions` with the ones
/// the engine would refuse, the ones that would change nothing, and the
/// cancels that re-offer their own opener filtered out. Borrowed and
/// allocation-free on the vast majority of states, where nothing is
/// filtered.
///
/// Every consumer goes through here — the actor, the greedy arm, the
/// search — so an inert step is unreachable from all of them at once.
///
/// One list is emptied by filtering: a selection screen over a deck holding
/// nothing it would accept, whose only answer is the cancel. The raw list
/// comes back instead — cancelling out of a screen with nothing on it is
/// what a player does, and strictly better than a policy with no action to
/// choose. A collapsed opener never leads there, because a screen with no
/// answer leaves nothing behind it (see
/// [`permitted_plans`](crate::plan::permitted_plans)).
///
/// The same fallback covers the filters that have never emptied a list: no
/// registered event lays out a page of only poisoned options, and a screen
/// offering nothing but a potion there is no room for still offers
/// `Proceed`.
#[must_use]
pub fn permitted_actions(simulator: &Simulator) -> Cow<'_, [Action]> {
    let legal = simulator.legal_actions();
    if !legal.iter().any(|action| withheld(simulator, action)) {
        return Cow::Borrowed(legal);
    }
    let filtered: Vec<Action> = legal
        .iter()
        .filter(|action| !withheld(simulator, action))
        .cloned()
        .collect();
    if filtered.is_empty() {
        Cow::Borrowed(legal)
    } else {
        Cow::Owned(filtered)
    }
}

/// Chooses among the actions a simulator enumerates. The simulator handed in
/// always comes from a determinizer, never from the authoritative state.
pub trait RolloutPolicy {
    fn choose(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> Action;

    /// The plan this policy takes at `simulator`, which the harness expands
    /// into engine steps in order.
    ///
    /// The default names one action, which is what every in-fight policy and
    /// every baseline has to say: only a run-level policy standing at a macro
    /// screen has a second step to name. A policy that overrides this owes
    /// [`RolloutPolicy::choose`] the plan's lead step and nothing more, so a
    /// caller that steps only `choose` takes half a trade — the harness calls
    /// this.
    fn choose_plan(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> ActionPlan {
        ActionPlan::Single(self.choose(simulator, rng))
    }

    /// This policy's own answer as a *distribution* over `actions`, where it
    /// has one to expose — the soft target a macro net is cloned from.
    ///
    /// `None` is the honest default: a policy that only ever names one
    /// action has no ranking of the rest to teach, and a recorder handed
    /// `None` has nothing to record. Only
    /// [`Heuristic`](crate::heuristics::Heuristic) answers today, with a
    /// softmax of its own scores (see
    /// [`MACRO_POLICY_TEMPERATURE`](crate::heuristics::MACRO_POLICY_TEMPERATURE)).
    /// `actions` is the canonical list the recorder will encode, so the
    /// answer is aligned with it index for index and sums to one.
    fn macro_teacher_policy(
        &self,
        _simulator: &Simulator,
        _actions: &[Action],
    ) -> Option<Vec<f64>> {
        None
    }

    /// The harness's word that the walk is over, with the state it ended on.
    /// A policy that owes bookkeeping to the end of a run — a recorder
    /// settling its last fight's outcome — settles it here. Most policies owe
    /// nothing.
    fn run_ended(&mut self, _simulator: &Simulator) {}

    /// The harness's word that the action this policy just named has been
    /// applied, with the state it produced.
    ///
    /// [`RolloutPolicy::choose`] sees the state *before* the step, which is
    /// everything a policy choosing an action needs and not enough for a
    /// policy being *paid* for one: what a step is worth is a function of
    /// the state it left behind — the floor now stood on, the run now over —
    /// and no caller of `choose` can see it. The harness is the only thing
    /// that can, so the harness says so.
    ///
    /// It fires on every accepted step, in-fight steps included, and one
    /// out-of-combat decision is followed by a whole fight's worth of them.
    /// A policy accumulating a reward accumulates across all of them and
    /// charges the total to the decision that walked into the room, because
    /// that is the decision that bought it.
    ///
    /// Most policies owe nothing here, and the default no-op is the honest
    /// answer for every one of them.
    fn stepped(&mut self, _simulator: &Simulator) {}

    /// The harness's word that it is about to claim a free reward line on
    /// this policy's behalf — gold, or a potion there is room for (see
    /// [`crate::plan::forced_step`]) — with the screen it
    /// stands on and the claim. Not a decision, never recorded as one, and
    /// followed by [`RolloutPolicy::stepped`] like any other step; a policy
    /// keeping count of what its screens offered reads the offer here,
    /// since it will never see the claim on a list of its own.
    ///
    /// Most policies keep no such count, and the default no-op is the honest
    /// answer for every one of them.
    fn forced(&mut self, _simulator: &Simulator, _action: &Action) {}

    /// Whatever in-combat decisions the policy recorded, drained. Policies
    /// that record nothing answer with nothing.
    fn drain_decisions(&mut self) -> Vec<crate::training::Decision> {
        Vec::new()
    }

    /// Whatever *macro* decisions the policy recorded, drained. A separate
    /// channel from the in-combat one because the two nets use different
    /// value targets and separate shard directories.
    fn drain_macro_decisions(&mut self) -> Vec<crate::training::Decision> {
        Vec::new()
    }

    /// What this policy's per-run search budget came to, for the run's own
    /// summary line. A policy with no budget to spend answers with nothing.
    fn budget_spent(&self) -> Option<crate::search::BudgetSpend> {
        None
    }

    /// Whatever the policy observed about the run while playing it, drained:
    /// the fights it entered and what it threw away at the screens between
    /// them.
    ///
    /// Collected here rather than read off the finished report because it
    /// cannot be read off the finished report. A run that died on floor 30
    /// carries the room it died in and nothing about the fifteen fights it
    /// won on the way, and the policy is the only thing that saw them. Most
    /// policies are not being diagnosed and owe nothing.
    fn drain_run_metrics(&mut self) -> Option<crate::summary::EpisodeMetrics> {
        None
    }
}

/// The first legal action, preferring a non-empty card pick so a skippable
/// reward cannot re-offer itself. Fully deterministic; the walking policy the
/// smoke tests and benchmarks use.
#[derive(Clone, Copy, Debug, Default)]
pub struct FirstLegal;

impl RolloutPolicy for FirstLegal {
    fn choose(&mut self, simulator: &Simulator, _rng: &mut MegaRandom) -> Action {
        let permitted = permitted_actions(simulator);
        permitted
            .iter()
            .find(|action| matches!(action, Action::ChooseCards { cards, .. } if !cards.is_empty()))
            .or_else(|| permitted.first())
            .cloned()
            .unwrap_or_else(|| {
                // See `Heuristic::choose`: an empty enumeration on a live
                // run is an engine fault to report, and only the
                // authoritative run can reach a policy with one.
                panic!(
                    "a live run offers an action; the engine enumerated none at {:?}",
                    simulator.decision()
                )
            })
    }
}

/// Uniformly random over the enumerated actions: the breadth policy. Diversity
/// beats strength for coverage, and this is the most diverse policy there is.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformRandom;

impl RolloutPolicy for UniformRandom {
    fn choose(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> Action {
        let legal = permitted_actions(simulator);
        // See `Heuristic::choose`: an empty enumeration on a live run is an
        // engine fault to report, and only the authoritative run can reach
        // a policy with one.
        assert!(
            !legal.is_empty(),
            "a live run offers an action; the engine enumerated none at {:?}",
            simulator.decision()
        );
        let index = usize::try_from(rng.next_u64()).unwrap_or(usize::MAX) % legal.len();
        legal[index].clone()
    }
}
