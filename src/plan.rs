//! Action plans: what the run policy scores as one decision.
//!
//! Almost every decision is one click, and a plan wrapping one click is
//! [`ActionPlan::Single`]. Two shapes are not. A potion the belt has no room
//! for is taken by giving one up first, which is a strict loss on its own
//! step. A button that only opens a screen — a card reward, a removal, the
//! smith — is not a decision at all: the decision is what the screen is
//! answered with, and backing out of it re-offers the button forever.
//!
//! A third shape is not a decision either, and is not scored at all. A
//! reward screen's gold and a potion the belt has room for are free: nothing
//! is given up by claiming them and nothing makes the claim a loss. They are
//! claimed by the harness before any policy is asked ([`forced_step`]), so
//! what a policy scores on a reward screen is what is actually open on it —
//! the card offer, the relic, a card handed back, and walking away.
//!
//! The simulator gains no notion of a plan. It goes on enumerating one click
//! at a time, a plan expands into actions the engine already offers, and an
//! emitted decision script is still a sequence of legal engine actions.

use std::collections::BTreeSet;

use sts2_engine::{Action, DecisionContext, RestSiteOption, Simulator};

use crate::policy::{belt_is_full, permitted_actions, step_is_inert, withheld};

/// Whether this step claims a reward line that is free to take: gold, or a
/// potion.
///
/// Nothing is given up by claiming either, and nothing in the content makes
/// the claim a loss — a relic that blocks gold or potions keeps the line off
/// the screen rather than turning the claim into one. So neither is a
/// decision, and a policy asked to price one against the exit is being asked
/// a question with one answer, which it then gets wrong a fifth of the time
/// under sampled play. The card offer is not free — it is one pick among
/// several — and neither is a relic: an elite drops any relic in the pool,
/// and a few are a net loss beside the deck in hand. Nor is a card handed
/// back rather than chosen between: what a thief stole comes back as a line
/// of its own, and leaving it there is the one free removal a run is ever
/// offered, which a deck that has outgrown the card wants. All three stay
/// plans, and walking past any of them is `Proceed`.
///
/// A potion claim is free only where the belt has room. That is not read
/// here: over a full belt the claim is inert and `withheld` before this is
/// asked, which is why every caller reads this over [`permitted_actions`]
/// rather than the engine's raw list.
#[must_use]
pub fn free_claim(action: &Action) -> bool {
    match action {
        Action::ClaimReward { fingerprint, .. } => {
            fingerprint.offered_cards.is_empty()
                && fingerprint.special_card.is_none()
                && matches!(fingerprint.reward_type.as_str(), "gold" | "potion")
        }
        _ => false,
    }
}

/// Whether `action` is a step the harness takes on this screen without asking
/// a policy: a [`free_claim`] the policy layer permits here.
#[must_use]
pub fn is_forced_step(simulator: &Simulator, action: &Action) -> bool {
    free_claim(action)
        && matches!(simulator.decision(), DecisionContext::Rewards { .. })
        && permitted_actions(simulator).contains(action)
}

/// The step the harness takes on this screen before any policy is asked, if
/// there is one: the first free reward line still standing, in screen order.
/// Gold before a potion or the reverse changes nothing the engine tracks.
///
/// One at a time rather than the whole screen at once, so each claim is
/// read off the screen the previous one left — the engine re-enumerates
/// after every step, and this reads what it enumerates. `None` everywhere
/// but a reward screen with a free line on it, which is where a policy is
/// asked instead.
#[must_use]
pub fn forced_step(simulator: &Simulator) -> Option<Action> {
    if !matches!(simulator.decision(), DecisionContext::Rewards { .. }) {
        return None;
    }
    permitted_actions(simulator)
        .iter()
        .filter(|action| free_claim(action))
        .min_by_key(|action| match action {
            Action::ClaimReward { reward_index, .. } => *reward_index,
            _ => usize::MAX,
        })
        .cloned()
}
use crate::search::canonical_actions;

/// One policy-layer decision, which expands into an ordered sequence of
/// engine actions.
///
/// A plan makes no claim about which trade is good: it says what counts as
/// one decision, the way playing a card is one decision despite being
/// several engine steps.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
#[allow(
    clippy::large_enum_variant,
    reason = "an Action is already 416 bytes and a decision holds tens of plans; boxing would buy a heap allocation per trade"
)]
pub enum ActionPlan {
    /// One engine action.
    Single(Action),
    /// A held potion given up so an offered one can be taken. `free` empties
    /// a belt slot and pays no gold, so it cannot price `take` out of reach —
    /// but a drink is not only a slot emptied. Its body runs, and a body that
    /// starts a fight leaves `take` answering a screen that is no longer up,
    /// so the pair is probed for on a copy the way a collapse's answers are.
    Trade { free: Action, take: Action },
    /// A screen opened and answered in one decision. `open` is spent only by
    /// an answer, so cancelling the screen leaves `open` on offer and a
    /// policy that scores single actions can take the pair forever; `pick`
    /// is one of the answers that screen accepts.
    Collapse { open: Action, pick: Action },
}

impl ActionPlan {
    /// The engine steps this plan expands into, in the order they are taken.
    pub fn steps(&self) -> impl Iterator<Item = &Action> {
        match self {
            Self::Single(action) => [Some(action), None],
            Self::Trade { free, take } => [Some(free), Some(take)],
            Self::Collapse { open, pick } => [Some(open), Some(pick)],
        }
        .into_iter()
        .flatten()
    }

    /// The first engine step, which every plan has.
    #[must_use]
    pub const fn lead(&self) -> &Action {
        match self {
            Self::Single(action)
            | Self::Trade { free: action, .. }
            | Self::Collapse { open: action, .. } => action,
        }
    }

    /// What the plan is taken for: a trade's acquisition, a collapse's
    /// answer, or the single action itself.
    #[must_use]
    pub const fn outcome(&self) -> &Action {
        match self {
            Self::Single(action)
            | Self::Trade { take: action, .. }
            | Self::Collapse { pick: action, .. } => action,
        }
    }

    /// The step that frees the belt slot, on a trade and nothing else.
    #[must_use]
    pub const fn surrender(&self) -> Option<&Action> {
        match self {
            Self::Single(_) | Self::Collapse { .. } => None,
            Self::Trade { free, .. } => Some(free),
        }
    }

    /// The step that opens the screen, on a collapse and nothing else.
    #[must_use]
    pub const fn opener(&self) -> Option<&Action> {
        match self {
            Self::Single(_) | Self::Trade { .. } => None,
            Self::Collapse { open, .. } => Some(open),
        }
    }

    /// Whether this plan gives a held potion up for an offered one.
    #[must_use]
    pub const fn is_trade(&self) -> bool {
        matches!(self, Self::Trade { .. })
    }
}

/// Whether this step empties a belt slot: a potion thrown away, or one drunk
/// where the belt will pour it.
fn frees_a_slot(action: &Action) -> Option<(bool, usize)> {
    match action {
        Action::DiscardPotion { slot, .. } => Some((false, *slot)),
        Action::UsePotion { slot, .. } => Some((true, *slot)),
        _ => None,
    }
}

/// Whether this step only opens a card-selection screen, spending nothing
/// until that screen is answered.
///
/// The engine has five such buttons — a card reward, a removal reward, the
/// merchant's removal, the smith and the cook. Three collapse into one
/// decision here. The cook does not: it asks for two cards at once, so
/// collapsing it would emit one plan per pair — about 190 at a twenty-card
/// deck against an axis of [`MAX_ACTIONS`](crate::net::MAX_ACTIONS) — and a
/// permanently uniform-prior screen is worse than the two steps it replaces.
/// The smith does not either, for a different reason: collapsed, its
/// probability mass splits one plan per upgradable card while the heal
/// stands whole on a single plan, so a policy can want the smith in
/// aggregate and still never rank any one upgrade first. As one arm of the
/// rest decision — with the card the next decision's own choice — every
/// smith outcome trains and competes as one class. It is still probed below
/// so an unanswerable smith is dropped, not offered as a repeatable step.
fn opens_a_screen(action: &Action) -> bool {
    match action {
        // `claim_reward`: a removal reward always opens the deck, and every
        // other reward opens a screen exactly where it carries cards to
        // choose between.
        Action::ClaimReward { fingerprint, .. } => {
            fingerprint.reward_type == "card_removal" || !fingerprint.offered_cards.is_empty()
        }
        Action::BuyCardRemoval
        | Action::RestOption {
            option: RestSiteOption::Smith,
            ..
        } => true,
        _ => false,
    }
}

/// The answers the screen `open` puts up accepts, or `None` where `open`
/// puts up no screen.
///
/// Read off a throwaway copy of the run stepped onto that screen. The engine
/// is deterministic, so the copy's step is the same function of the same
/// state the authoritative step will be and the answers read here are the
/// answers the real screen will offer; the copy is discarded, so the run
/// itself takes only the steps its policy names.
///
/// Only what settles the screen counts: the picks and the buttons beside
/// them, named by the screen's own choice. The engine appends the belt to
/// every decision, and drinking a potion at a selection screen leaves that
/// screen standing — the very shape the collapse exists to remove. It is
/// still offered on the screen the opener stands on.
///
/// The empty answer is not among them either — `withheld` drops it — which
/// is what makes the collapse total: no answer here returns the screen the
/// opener was taken on.
fn collapsed_answers(simulator: &Simulator, open: &Action) -> Option<Vec<Action>> {
    if !opens_a_screen(open) {
        return None;
    }
    let mut probe = simulator.clone();
    probe.step_quietly(open).ok()?;
    let DecisionContext::ChooseCards { choice_id, .. } = probe.decision() else {
        // A screen with no candidate to offer never opens, and the button
        // settles nothing: the reward stays unanswered, the removal unbought.
        // No answers, so nothing stands where the button did.
        //
        // A state key cannot be used to see this. Every step advances the
        // run's visible history, so the key is fresh after a step that
        // changed nothing else — which is the whole reason a screen with a
        // free cancel loops a policy invisibly.
        return Some(Vec::new());
    };
    let opened = *choice_id;
    let answers: Vec<Action> = probe
        .legal_actions()
        .iter()
        .filter(|action| match action {
            Action::ChooseCards { choice_id, .. } | Action::ChooseAlternative { choice_id, .. } => {
                *choice_id == opened
            }
            _ => false,
        })
        .filter(|action| !withheld(&probe, action))
        .cloned()
        .collect();
    Some(canonical_actions(&answers))
}

/// Everything the run policy may choose at `simulator`.
///
/// The engine's own offers, less what [`permitted_actions`] withholds and
/// less the reward lines that are free to take ([`free_claim`]) — those are
/// the harness's steps, not the policy's — each as an [`ActionPlan::Single`],
/// with two substitutions on top.
///
/// A button that only opens a card-selection screen is replaced by one
/// [`ActionPlan::Collapse`] per answer that screen accepts, so the policy
/// scores the answer rather than the click that reaches it. Declining is
/// still expressible: it is `Proceed` on the screen the button stood on.
///
/// Where the belt is full and a potion is on offer anyway, one
/// [`ActionPlan::Trade`] per way of making room for it. The freeing steps
/// are exactly the ones the engine is offering: a discard for every potion
/// held, and a drink only for the potions the belt will pour outside a
/// fight, which is where `CombatOnly` potions drop out.
#[must_use]
pub fn permitted_plans(simulator: &Simulator) -> Vec<ActionPlan> {
    let offered = permitted_actions(simulator);
    let canonical = canonical_actions(&offered);
    let mut plans: Vec<ActionPlan> = Vec::with_capacity(canonical.len());
    for action in &canonical {
        if free_claim(action) {
            continue;
        }
        match collapsed_answers(simulator, action) {
            // The smith class arm: the button stands as one plan where its
            // screen has an answer, and disappears where it does not — a
            // smith over a fully upgraded deck opens nothing, settles
            // nothing, and would otherwise be an infinitely repeatable step.
            Some(answers)
                if matches!(
                    action,
                    Action::RestOption {
                        option: RestSiteOption::Smith,
                        ..
                    }
                ) =>
            {
                if !answers.is_empty() {
                    plans.push(ActionPlan::Single(action.clone()));
                }
            }
            Some(answers) => plans.extend(answers.into_iter().map(|pick| ActionPlan::Collapse {
                open: action.clone(),
                pick,
            })),
            None => plans.push(ActionPlan::Single(action.clone())),
        }
    }
    // A screen with no answer worth taking leaves nothing where its opener
    // stood — which is right for a smith over a fully upgraded deck, and
    // must never be the whole decision: a policy with no plan has nothing to
    // answer with, so the uncollapsed list stands instead.
    if plans.is_empty() {
        return canonical.into_iter().map(ActionPlan::Single).collect();
    }
    if !belt_is_full(simulator) {
        return plans;
    }
    let legal = simulator.legal_actions();
    let mut seen = BTreeSet::new();
    let frees: Vec<&Action> = legal
        .iter()
        .filter(|action| frees_a_slot(action).is_some_and(|key| seen.insert(key)))
        .collect();
    let takes: Vec<&Action> = legal
        .iter()
        .filter(|action| step_is_inert(simulator, action))
        .collect();
    for free in &frees {
        let Some(surviving) = survives_the_free(simulator, free, &takes) else {
            continue;
        };
        for take in surviving {
            plans.push(ActionPlan::Trade {
                free: (*free).clone(),
                take,
            });
        }
    }
    plans
}

/// Which of `takes` the screen still offers once `free` has been stepped.
///
/// A discard empties a slot and does nothing else, but a drink runs its body,
/// and one body walks off the screen entirely: a foul potion thrown at the
/// fake merchant turns him on the player and the fight starts where the
/// reward screen stood. The claim behind it was chosen at a screen that no
/// longer exists, and the engine refuses it — rightly — as an answer to the
/// wrong decision.
///
/// Read off a throwaway copy, as [`collapsed_answers`] reads a collapse's
/// answers off one. The engine is deterministic, so the copy's step is the
/// same function of the same state the authoritative step will be, and what
/// the copy offers afterwards is what the run will offer. `None` where the
/// freeing step will not apply at all, which leaves no trade to name.
fn survives_the_free(
    simulator: &Simulator,
    free: &Action,
    takes: &[&Action],
) -> Option<Vec<Action>> {
    if takes.is_empty() {
        return None;
    }
    let mut probe = simulator.clone();
    probe.step_quietly(free).ok()?;
    let offered = probe.legal_actions();
    Some(
        takes
            .iter()
            .filter(|take| offered.contains(**take))
            .map(|take| (*take).clone())
            .collect(),
    )
}
