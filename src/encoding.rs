//! The policy encoding: what a player sees, as tensors.
//!
//! A numeric rendering of [`AgentObservation`] and of each legal [`Action`],
//! for a learned policy to read. The simulator's half of the contract is the
//! versioned observation itself (`AGENT_OBSERVATION_VERSION`) and the legal
//! actions; the tensor rendering is deliberately this repo's — it is built
//! purely against the public API, and its layout churns with training
//! experiments, which is churn the simulator should never see. It is built *only* from the
//! observation, so everything the observation guarantees carries over — the
//! encoding is invariant under hidden-state changes a player cannot
//! distinguish, and two indistinguishable decision points encode
//! identically.
//!
//! The action side is pointer-shaped: there is no fixed action space. Each
//! legal action encodes to its own token indices and feature vector; a net
//! scores the offered actions and a softmax over exactly that set is the
//! policy. Masking is therefore structural rather than a mask tensor.
//!
//! Everything visible is a token: cards in every zone, relics, potions,
//! orbs, creatures, the powers they carry, the lines of the intent they
//! show, the run deck, the act's map, and whatever a standing screen is
//! offering. The vocabulary is supplied by the caller — see
//! [`standard_vocabulary`], which is what every call site must build it from
//! — and hashed, so a checkpoint names the vocabulary it was trained
//! against.

use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

use crate::plan::ActionPlan;
use sts2_core::{CardInstanceId, ModelId};
use sts2_engine::{
    Action, AgentObservation, CardFingerprint, CardPreview, CombatSide, DecisionContext, MapCoord,
    MapPointType, RestSiteOption, ShopItem, TargetHandle, VisibleMap,
};

/// The encoding contract version. Changes to dimensions, slot meanings or
/// normalization require a new version and compatible checkpoint metadata.
///
/// Version 9 encodes the rewards left behind by `Proceed` and `AdvanceAct`.
/// It has the same tensor dimensions as version 8 but different macro action
/// semantics, so macro checkpoints require retraining. Combat action
/// semantics are unchanged.
pub const POLICY_ENCODING_VERSION: u32 = 9;

/// Fixed-position scalar features per observation.
pub const OBSERVATION_SCALARS: usize = SCALAR_SLOTS + DECISION_KINDS;
/// The scalars ahead of the decision-kind one-hot.
const SCALAR_SLOTS: usize = 47;

/// The decision contexts, in the order their one-hot occupies the tail of the
/// observation scalars. This is `DecisionContext`'s full range — a closed set
/// of twelve, which is why nothing is hashed.
pub const DECISION_KIND_NAMES: [&str; DECISION_KINDS] = [
    "map_navigation",
    "event",
    "shop",
    "rest_site",
    "treasure",
    "crystal_sphere",
    "act_transition",
    "combat_priority",
    "choose_cards",
    "rewards",
    "room_proceed",
    "terminal",
];
/// One-hot slots for the decision kind.
pub const DECISION_KINDS: usize = 12;

/// The most tokens an observation carries; beyond it, later tokens are
/// dropped and the truncation flag scalar is set.
///
/// Out of combat the act's map is on the screen — around fifty points — and
/// the run deck is beside it, so the budget is set for the run-level
/// decisions rather than the combat ones.
pub const MAX_TOKENS: usize = 256;

/// The intent types, in the order their one-hot occupies a token's intent
/// block: the game's closed set of fifteen, under the simulator's own
/// snake-case names.
pub const INTENT_TYPE_NAMES: [&str; INTENT_TYPES] = [
    "attack",
    "buff",
    "debuff",
    "debuff_strong",
    "defend",
    "escape",
    "heal",
    "hidden",
    "summon",
    "sleep",
    "stun",
    "status_card",
    "card_debuff",
    "death_blow",
    "unknown",
];
/// One-hot slots for an intent type.
pub const INTENT_TYPES: usize = 15;

/// Which intent lines count as blows, for the damage total and the hit
/// count. Every other line kind is a rider that says something else — a
/// defend, a buff, a summon — and counting its `repeats` into the hits was
/// how a twelve-damage attack with a defend rider came to look like a
/// six-times-two multi-attack.
///
/// Typed by kind, with any line that advertises a damage number counted too,
/// so a line that carries damage under some other icon is never silently
/// dropped from the total.
fn is_a_blow(entry: &sts2_engine::MoveIntent) -> bool {
    matches!(entry.intent_type.as_str(), "attack" | "death_blow") || entry.damage.is_some()
}

/// How many of a creature's performed moves ride along as tokens of their
/// own. The engine keeps [`sts2_engine::MOVE_HISTORY_TAIL`] of them; three is
/// as deep as any cooldown or no-repeat weight in the registry reads, and
/// three tokens per creature is what the [`MAX_TOKENS`] budget affords in a
/// fight.
pub const MOVE_HISTORY_DEPTH: usize = 3;

/// Features per token: the zone one-hot, the shared value slots, the owner
/// one-hot, the intent type one-hot, the log-scaled hit-point pair, the card
/// face, the registry's static card numbers, the map point-type one-hot, and
/// the per-zone extras. Slot meanings are documented on
/// [`PolicyEncoder::encode_observation`].
pub const TOKEN_FEATURES: usize = EXTRA_BASE + EXTRA_SLOTS;
const SHARED_BASE: usize = ZONES;
const SHARED_SLOTS: usize = 7;
/// Which creature slot a row belongs to, over the same nine slots
/// [`ACTION_TARGET_BASE`] uses: 0-7 name the creature's index and 8 is the
/// overflow an untargeted action also lands in. A consumer pools token rows
/// by this block and gathers with the action's target block, which is how an
/// action reaches the creature it points at.
const OWNER_BASE: usize = SHARED_BASE + SHARED_SLOTS;
const OWNER_SLOTS: usize = 9;
const INTENT_BASE: usize = OWNER_BASE + OWNER_SLOTS;
const HP_BASE: usize = INTENT_BASE + INTENT_TYPES;
const HP_SLOTS: usize = 2;
const FACE_BASE: usize = HP_BASE + HP_SLOTS;
const FACE_SLOTS: usize = 15;
/// The damage and the block the game prints on the card, hook-resolved:
/// [`sts2_engine::CardPreview`], never re-derived here. Written on a card in the
/// hand, which is the only pile the build runs the preview passes for; every
/// other card row leaves the pair empty.
const FACE_PREVIEW_BASE: usize = FACE_BASE + 13;
/// What the registry says the card does before anything on the board scales
/// it: the card-type one-hot, the target-type one-hot, then base damage, hit
/// count, base block and granted-power count.
const STATIC_BASE: usize = FACE_BASE + FACE_SLOTS;
/// The preview pair is the whole tail of the face block, and nothing else
/// writes it.
const _: () = assert!(FACE_PREVIEW_BASE + 2 == STATIC_BASE);
const STATIC_TARGET_BASE: usize = STATIC_BASE + CARD_TYPES;
const STATIC_NUMBER_BASE: usize = STATIC_TARGET_BASE + TARGET_TYPES;
const STATIC_SLOTS: usize = CARD_TYPES + TARGET_TYPES + 4;
const MAP_TYPE_BASE: usize = STATIC_BASE + STATIC_SLOTS;
const EXTRA_BASE: usize = MAP_TYPE_BASE + MAP_TYPES;
const EXTRA_SLOTS: usize = 8;

/// The card types, in the order their one-hot occupies a card token's static
/// block and a card play's. `CardType`'s full range.
pub const CARD_TYPE_NAMES: [&str; CARD_TYPES] =
    ["attack", "skill", "power", "status", "curse", "quest"];
/// One-hot slots for a card type.
pub const CARD_TYPES: usize = 6;

/// The target types, in the order their one-hot occupies the same blocks.
/// `TargetType`'s full range.
pub const TARGET_TYPE_NAMES: [&str; TARGET_TYPES] = [
    "none",
    "self",
    "any_player",
    "any_enemy",
    "all_enemies",
    "random_enemy",
    "targeted_no_creature",
    "all_allies",
    "any_ally",
];
/// One-hot slots for a target type.
pub const TARGET_TYPES: usize = 9;

fn card_type_slot(card_type: sts2_engine::CardType) -> usize {
    match card_type {
        sts2_engine::CardType::Attack => 0,
        sts2_engine::CardType::Skill => 1,
        sts2_engine::CardType::Power => 2,
        sts2_engine::CardType::Status => 3,
        sts2_engine::CardType::Curse => 4,
        sts2_engine::CardType::Quest => 5,
    }
}

fn target_type_slot(target: sts2_engine::TargetType) -> usize {
    match target {
        sts2_engine::TargetType::None => 0,
        sts2_engine::TargetType::SelfTarget => 1,
        sts2_engine::TargetType::AnyPlayer => 2,
        sts2_engine::TargetType::AnyEnemy => 3,
        sts2_engine::TargetType::AllEnemies => 4,
        sts2_engine::TargetType::RandomEnemy => 5,
        sts2_engine::TargetType::TargetedNoCreature => 6,
        sts2_engine::TargetType::AllAllies => 7,
        sts2_engine::TargetType::AnyAlly => 8,
    }
}

/// Which of the nine owner slots a creature index lands in. The last slot is
/// the overflow, which is also where an action aiming at nobody points.
const fn owner_slot(slot: usize) -> usize {
    if slot < OWNER_SLOTS - 1 {
        slot
    } else {
        OWNER_SLOTS - 1
    }
}

/// The map point types, in the order their one-hot occupies a map token's
/// block and a map action's. `MapPointType`'s full range.
pub const MAP_TYPE_NAMES: [&str; MAP_TYPES] = [
    "ancient",
    "monster",
    "elite",
    "unknown",
    "shop",
    "rest_site",
    "treasure",
    "boss",
];
/// One-hot slots for a map point type.
pub const MAP_TYPES: usize = 8;

fn map_type_slot(point_type: MapPointType) -> usize {
    match point_type {
        MapPointType::Ancient => 0,
        MapPointType::Monster => 1,
        MapPointType::Elite => 2,
        MapPointType::Unknown => 3,
        MapPointType::Shop => 4,
        MapPointType::RestSite => 5,
        MapPointType::Treasure => 6,
        MapPointType::Boss => 7,
    }
}

/// Embedding slots per action: [`ACTION_NAMING_TOKENS`] naming what the
/// action is about, then [`ACTION_SURRENDERED_TOKEN`].
pub const ACTION_TOKENS: usize = 5;
/// The slots naming what an action is about — cards it plays or picks,
/// potion models, map points, shop items, reward lines. Every writer that
/// fills slots in a loop stops here, which is what keeps the surrendered
/// slot's meaning exclusive.
pub const ACTION_NAMING_TOKENS: usize = ACTION_TOKENS - 1;
/// The potion a [`Trade`](crate::plan::ActionPlan::Trade) gives up, on a slot
/// of its own.
///
/// The token embedding is mean-pooled over the naming slots, and a mean is
/// symmetric: surrendering a Fire Potion for a Weak Potion and surrendering a
/// Weak Potion for a Fire Potion would pool to the same vector and the
/// direction of the trade would be gone. Read apart from the pool, it is a
/// difference to learn rather than an average to be confused by.
///
/// Index 0 on every action that is not a trade, `nn.Embedding`'s
/// `padding_idx`, which embeds to an exact zero vector.
pub const ACTION_SURRENDERED_TOKEN: usize = ACTION_TOKENS - 1;
/// Features per action.
pub const ACTION_FEATURES: usize = ACTION_EVENT_BASE + ACTION_EVENT_SLOTS;
/// The action families, in the order their one-hot occupies the head of the
/// action features. This is [`Action::family`]'s full range.
pub const ACTION_FAMILY_NAMES: [&str; ACTION_FAMILIES] = [
    "map_navigation",
    "event",
    "shop",
    "rest_site",
    "treasure",
    "crystal_sphere",
    "potion",
    "act_transition",
    "combat_priority",
    "choose_cards",
    "choose_alternative",
    "rewards",
    "room_proceed",
];
const ACTION_FAMILIES: usize = 13;
/// The creature an action aims at, over the same nine slots `OWNER_BASE`
/// uses on the token side: 0-7 name the target's index and 8 is aiming at
/// nobody. The two blocks share a numbering so that a consumer can gather the
/// aimed creature's pooled summary with one matmul.
const ACTION_TARGET_BASE: usize = ACTION_FAMILIES;
const ACTION_TARGET_SLOTS: usize = 9;
const ACTION_CARD_BASE: usize = ACTION_TARGET_BASE + ACTION_TARGET_SLOTS;
const ACTION_CARD_SLOTS: usize = 10;
const ACTION_INDEX_BASE: usize = ACTION_CARD_BASE + ACTION_CARD_SLOTS;
const ACTION_INDEX_SLOTS: usize = 2;
/// The map point a step names: normalized column and row, how many ways on
/// the point opens, whether the walk has stood there, how far up the act it
/// is from where the player stands, whether it is the boss's own coordinate,
/// and — at `+6` — whether the act's map carries no point at that coordinate
/// at all.
///
/// That last flag is not redundant with the empty type one-hot beside it.
/// `MapPointType` is closed, so a point the map does carry always lights
/// exactly one slot of [`ACTION_MAP_TYPE_BASE`]; a destination the map cannot
/// resolve leaves them all dark — and so does every action that is not a map
/// step, which is most of them. Without the flag the net reads "a step onto a
/// coordinate nothing is known about" and "not a map step" off the same row.
const ACTION_MAP_BASE: usize = ACTION_INDEX_BASE + ACTION_INDEX_SLOTS;
const ACTION_MAP_SLOTS: usize = 7;
const ACTION_MAP_TYPE_BASE: usize = ACTION_MAP_BASE + ACTION_MAP_SLOTS;
const ACTION_PRICE_BASE: usize = ACTION_MAP_TYPE_BASE + MAP_TYPES;
const ACTION_PRICE_SLOTS: usize = 4;
const ACTION_KIND_BASE: usize = ACTION_PRICE_BASE + ACTION_PRICE_SLOTS;
const ACTION_KIND_SLOTS: usize = 6;
/// The same registry statics the card token carries, on the play itself: the
/// pointer scorer is additive in the state and the action vector, so what the
/// card token knows never reaches the action that plays it.
const ACTION_STATIC_BASE: usize = ACTION_KIND_BASE + ACTION_KIND_SLOTS;
const ACTION_STATIC_TARGET_BASE: usize = ACTION_STATIC_BASE + CARD_TYPES;
const ACTION_STATIC_NUMBER_BASE: usize = ACTION_STATIC_TARGET_BASE + TARGET_TYPES;
const ACTION_STATIC_SLOTS: usize = CARD_TYPES + TARGET_TYPES + 4;
/// Per-target preview: the damage this play deals to the creature it is aimed at,
/// that as a fraction of what the creature has left, and whether it kills.
/// Off the engine's own preview pass, so an aim into a Vulnerable enemy
/// reads what the screen would show; a play with no number on its face, and
/// every action outside a combat, leaves the block empty.
const ACTION_PREVIEW_BASE: usize = ACTION_STATIC_BASE + ACTION_STATIC_SLOTS;
const ACTION_PREVIEW_SLOTS: usize = 3;
/// Whether aiming this play or this potion turns the player around, which
/// is the one thing the facing counter on the Surrounded power token cannot
/// say — the counter is the *current* facing, and what a back attack costs
/// depends on the facing the play leaves behind. It is
/// `POWER.SURROUNDED_POWER`'s own rule (`turn_to_face` in the simulator's
/// content crate): facing right is
/// counter zero and a target carrying `POWER.BACK_ATTACK_LEFT_POWER` turns
/// the player, facing left is counter one and the right marker does.
const ACTION_FLIP: usize = ACTION_PREVIEW_BASE + ACTION_PREVIEW_SLOTS;
/// The route a map step opens, as [`Route`] reads it: [`MAP_TYPES`] slots of
/// the fewest rooms of each type the walk can still stand on, then
/// [`MAP_TYPES`] slots of the most, then the longest walk still ahead at
/// [`ACTION_ROUTE_DEPTH`]. Empty on every action that is not a map step, and
/// on a step whose destination the map cannot resolve.
///
/// Counts share the `/8` the rest of the small integers here use: an act is
/// fourteen to sixteen rows, so no type can appear more than about sixteen
/// times on one path and a typical branch holds two to six of the type that
/// matters. The depth divides by sixteen, which is the longest act.
const ACTION_ROUTE_BASE: usize = ACTION_FLIP + 1;
const ACTION_ROUTE_MOST_BASE: usize = ACTION_ROUTE_BASE + MAP_TYPES;
const ACTION_ROUTE_DEPTH: usize = ACTION_ROUTE_MOST_BASE + MAP_TYPES;
/// How an [`ActionPlan::Trade`](crate::plan::ActionPlan::Trade) made room:
/// that it is one at all, whether the belt slot is freed by throwing the
/// potion away or by drinking it, and which slot as a fraction of the belt's
/// own length. The acquired side needs no slots here — a trade carries its
/// `take` action's own block unchanged, so the offered potion's identity,
/// face and price arrive with it.
///
/// Empty on every plan that is not a trade, which is what lets a checkpoint
/// built at the narrower width be zero-padded to this one instead of
/// retrained.
/// The two slots past the three flags and the index are the exit's: how
/// many reward lines `Proceed` leaves standing, and whether a relic is among
/// them (v9). Empty on every other action.
const ACTION_PLAN_BASE: usize = ACTION_ROUTE_DEPTH + 1;
const ACTION_PLAN_SLOTS: usize = 6;
/// What an event option's body does: [`EVENT_EFFECTS`] slots of multi-hot over
/// the effects it is made of, then [`ACTION_EVENT_NUMBER_BASE`]'s resources.
///
/// Options share the event's identity, so their effects and resource changes
/// must appear in each action's own features to distinguish them.
/// Empty on every action that is not an event option.
const ACTION_EVENT_BASE: usize = ACTION_PLAN_BASE + ACTION_PLAN_SLOTS;
/// The resources an option's body moves, signed: hit points, maximum hit
/// points, gold, then cards gained, removed and upgraded, relics gained and
/// lost, and potions given up. The last slot is raised where an amount could
/// not be read at all — an event var the page settled and the option only
/// points at — so an unreadable cost is distinguishable from a free one.
///
/// Counts and gold divide by what an event deals in rather than by what a run
/// holds: a page moves one or two relics and a hundred gold, never a deck.
const ACTION_EVENT_NUMBER_BASE: usize = ACTION_EVENT_BASE + EVENT_EFFECTS;
const ACTION_EVENT_NUMBER_SLOTS: usize = 10;
const ACTION_EVENT_SLOTS: usize = EVENT_EFFECTS + ACTION_EVENT_NUMBER_SLOTS;

/// Zones, in the order their one-hot occupies the head of the token
/// features.
const ZONES: usize = 14;
const ZONE_HAND: usize = 0;
const ZONE_DRAW: usize = 1;
const ZONE_DISCARD: usize = 2;
const ZONE_EXHAUST: usize = 3;
const ZONE_PLAY: usize = 4;
const ZONE_RELIC: usize = 5;
const ZONE_POTION: usize = 6;
const ZONE_ORB: usize = 7;
const ZONE_CREATURE: usize = 8;
const ZONE_POWER: usize = 9;
/// One shown line of a creature's advertised move, carrying the move id in
/// its token slot.
const ZONE_INTENT: usize = 10;
/// The run deck, out of combat, where a card pick is priced against it.
const ZONE_DECK: usize = 11;
/// The current act's map, one token per point.
const ZONE_MAP: usize = 12;
/// What a standing screen is offering — reward lines, card candidates, shop
/// shelves, treasure relics, event options, rest options. The zone the value
/// head reads a choice screen through.
const ZONE_OFFER: usize = 13;

/// The synthetic model ids the encoding names things with that the content
/// registry has no model for: a map point of each type, a reward line of
/// each kind, and a rest-site option of each kind. They live in categories
/// (`MAP`, `REWARD`, `REST`) no registered model uses, and they exist
/// because a token index of zero is the padding row the net masks out —
/// every token the encoding pushes has to name something.
#[must_use]
pub fn synthetic_ids() -> Vec<ModelId> {
    let mut ids = Vec::new();
    for name in MAP_TYPE_NAMES {
        ids.push(synthetic("MAP", &name.to_uppercase()));
    }
    for name in [
        "CARD",
        "CARD_REMOVAL",
        "GOLD",
        "POTION",
        "RELIC",
        "SPECIAL_CARD",
        "OTHER",
    ] {
        ids.push(synthetic("REWARD", name));
    }
    for option in [
        RestSiteOption::Heal,
        RestSiteOption::Smith,
        RestSiteOption::Cook,
        RestSiteOption::Clone,
        RestSiteOption::Dig,
        RestSiteOption::Hatch,
        RestSiteOption::Kindle,
        RestSiteOption::Lift,
    ] {
        ids.push(synthetic("REST", option.option_id()));
    }
    ids
}

fn synthetic(category: &str, entry: &str) -> ModelId {
    ModelId::new(category, entry).expect("a synthetic id is well shaped")
}

/// The vocabulary every call site must build its encoder from: the
/// registry's own models, its characters, the move states of every
/// registered monster (an intent's `move_id` is a rotation position, and
/// hashing cannot separate one monster's own states reliably), and the
/// encoding's [`synthetic_ids`].
///
/// One function so that a checkpoint's vocabulary hash cannot depend on which
/// call site built the encoder.
///
/// Character IDs must be included explicitly: they are registered separately
/// from other models. Missing IDs encode as zero, which is the padding token
/// and is masked out by the network.
#[must_use]
pub fn standard_vocabulary(registry: &sts2_engine::ContentRegistry) -> Vec<ModelId> {
    let mut ids = registry.registered_model_ids();
    ids.extend(registry.registered_character_ids());
    ids.extend(registry.registered_move_ids());
    ids.extend(synthetic_ids());
    ids.sort();
    ids.dedup();
    ids
}

/// The vocabulary index a move state is named by, matching
/// `ContentRegistry::registered_move_ids`.
fn move_id(state: &str) -> Option<ModelId> {
    ModelId::new("MOVE", state).ok()
}

/// One observation as tensors: fixed scalars, and a token sequence of
/// vocabulary indices with per-token features. `tokens` and `features` are
/// padded to [`MAX_TOKENS`] (index 0 is the padding token; real vocabulary
/// starts at 1), `features` laid out row-major, [`TOKEN_FEATURES`] per
/// token.
///
/// In memory this is always the padded form, because that is the shape the
/// net's inputs are declared at. On the wire it is not: see the hand-written
/// [`serde::Serialize`] below.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationEncoding {
    pub scalars: Vec<f32>,
    pub tokens: Vec<u32>,
    pub features: Vec<f32>,
    /// How many of the token slots carry a real token. Counted rather than
    /// derived from the last non-zero entry: a token whose model is outside
    /// the vocabulary indexes zero and would otherwise read as padding.
    live: usize,
}

impl ObservationEncoding {
    /// How many token slots carry a real token.
    #[must_use]
    pub const fn live_tokens(&self) -> usize {
        self.live
    }

    /// One encoding assembled from parts, with `live` real tokens ahead of
    /// the padding. What a caller building an encoding by hand — a test, a
    /// fixture — goes through, so that the live count can never disagree
    /// with the tokens beside it by accident.
    #[must_use]
    pub fn from_parts(
        scalars: Vec<f32>,
        mut tokens: Vec<u32>,
        mut features: Vec<f32>,
        live: usize,
    ) -> Self {
        tokens.resize(MAX_TOKENS, 0);
        features.resize(MAX_TOKENS * TOKEN_FEATURES, 0.0);
        Self {
            scalars,
            tokens,
            features,
            live: live.min(MAX_TOKENS),
        }
    }
}

/// Serialized without its padding: a decision fills a fraction of the
/// [`MAX_TOKENS`] slots, so the padded form spends most of its bytes writing
/// `0.0` — and, because a zero costs three bytes as text and four as an
/// `f32`, a dense binary encoding would not have saved any of them. Dropping
/// the pad is the whole win; the shape on the wire is otherwise unchanged, so
/// a reader only has to pad after parsing, and the live count is the length
/// of the token array it read.
impl serde::Serialize for ObservationEncoding {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let live = self.live.min(self.tokens.len());
        let rows = (live * TOKEN_FEATURES).min(self.features.len());
        let mut state = serializer.serialize_struct("ObservationEncoding", 3)?;
        state.serialize_field("scalars", &self.scalars)?;
        state.serialize_field("tokens", &self.tokens[..live])?;
        state.serialize_field("features", &self.features[..rows])?;
        state.end()
    }
}

/// Padded back on the way in.
impl<'de> serde::Deserialize<'de> for ObservationEncoding {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Wire {
            scalars: Vec<f32>,
            tokens: Vec<u32>,
            features: Vec<f32>,
        }
        let mut wire = Wire::deserialize(deserializer)?;
        // A file that carries the full padded array says nothing about where
        // the real tokens stopped; there the trailing run of zeroes is the
        // padding. A file that carries exactly its real tokens has the count
        // in its length.
        let live = if wire.tokens.len() >= MAX_TOKENS {
            wire.tokens
                .iter()
                .rposition(|&token| token != 0)
                .map_or(0, |last| last + 1)
        } else {
            wire.tokens.len()
        };
        wire.tokens.resize(MAX_TOKENS, 0);
        wire.features.resize(MAX_TOKENS * TOKEN_FEATURES, 0.0);
        Ok(Self {
            scalars: wire.scalars,
            tokens: wire.tokens,
            features: wire.features,
            live,
        })
    }
}

/// One legal action as tensors: up to [`ACTION_TOKENS`] vocabulary indices
/// (padded with 0) and a fixed feature vector.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ActionEncoding {
    pub tokens: Vec<u32>,
    pub features: Vec<f32>,
}

/// The encoder: a vocabulary, the registry's static card numbers, and the
/// versioned layout above.
#[derive(Clone, Debug)]
pub struct PolicyEncoder {
    vocabulary: BTreeMap<ModelId, u32>,
    /// Each registered card's numbers in both of its forms, base first. Only
    /// levels 0 and 1 exist — `supported_upgrade_levels` never holds another
    /// — and an upgrade replaces the body whole, so a Strike+ reads nine
    /// rather than the base form's six.
    statics: BTreeMap<ModelId, [CardStatics; 2]>,
    hash: String,
}

/// What the registry says a card does before anything on the board scales it,
/// walked once at construction rather than per encode.
///
/// The card type and the target type are the ones the model declares; the two
/// cards of the build that work theirs out at play time (`type_from_property`,
/// `target_widens_with`) read here as what they were registered as, which is
/// what an untinkered copy does. The numbers are counted off the body's
/// unconditional top-level run alone: a command inside an `if`, a loop or a
/// bound local is not statically readable, and neither is an amount that is
/// not [`sts2_engine::Amount::Fixed`]. Those read as zero, and the face flags
/// and the model-id embedding carry what the walk could not.
#[derive(Clone, Copy, Debug)]
struct CardStatics {
    card_type: usize,
    target: usize,
    /// The whole blow before modifiers: each attack's amount times its hits.
    damage: i32,
    hits: i32,
    block: i32,
    granted_powers: i32,
}

impl CardStatics {
    /// The card in one of its forms, off the commands that form runs.
    fn of(definition: &sts2_engine::CardDefinition, effects: &[sts2_engine::Command]) -> Self {
        use sts2_engine::{Amount, Command, CreatureCmd, DamageCmd, PowerCmd};
        let fixed = |amount: &Amount| match amount {
            Amount::Fixed(value) => Some(*value),
            _ => None,
        };
        let mut statics = Self {
            card_type: card_type_slot(definition.card_type),
            target: target_type_slot(definition.target),
            damage: 0,
            hits: 0,
            block: 0,
            granted_powers: 0,
        };
        for command in effects {
            match command {
                Command::Damage(
                    DamageCmd::Attack { amount, hits, .. }
                    | DamageCmd::AttackFromPet { amount, hits, .. },
                ) => {
                    if let (Some(amount), Some(hits)) = (fixed(amount), fixed(hits)) {
                        statics.damage += amount * hits;
                        statics.hits += hits;
                    }
                }
                Command::Creature(CreatureCmd::GainBlock { amount, .. }) => {
                    statics.block += fixed(amount).unwrap_or(0);
                }
                Command::Power(PowerCmd::Apply { .. }) => statics.granted_powers += 1,
                _ => {}
            }
        }
        statics
    }
}

/// A raw hit-point count, log-scaled.
///
/// The Waterfall Giant's invulnerable phase carries 999,999,999 hit points
/// (the game shows it as an infinity rather than a number), which under a
/// plain `/100` arrived at the net as 1e7 and put a whole
/// training epoch's value head at |v|~4e4. `ln(1+hp)/20` puts the sentinel at
/// 1.04 and an ordinary forty-hit-point monster at 0.19, so the sentinel is
/// simply the top of the scale rather than an outlier the trainer has to
/// clip.
fn log_hp(hp: i32) -> f32 {
    #[allow(clippy::cast_precision_loss, reason = "hit points are far below f32")]
    let value = hp.max(0) as f32;
    value.ln_1p() / 20.0
}

/// The board's threat, summed across the creature rows.
///
/// Plain arithmetic over what the screen shows, and it is encoded because the
/// consumer cannot do it: a mean-pool over one bag of tokens recovers a sum
/// only as `mean × n`, a product of two of its own inputs. "A Defend into a
/// buff intent is worth nothing" is the read that costs.
#[derive(Clone, Copy, Debug, Default)]
struct BoardThreat {
    /// The damage the shown intents advertise, counted off attacking lines
    /// alone — the same discipline [`is_a_blow`] holds the creature token's
    /// own total to.
    incoming: i64,
    /// What the player's block does not stop, floored at nothing.
    unblocked: i64,
    player_hp: i32,
    enemies_alive: usize,
    allies_alive: usize,
    enemy_block: i64,
    enemy_hp: i64,
}

impl BoardThreat {
    fn of(observation: &AgentObservation) -> Self {
        let mut threat = Self {
            player_hp: observation.current_hp.unwrap_or(0).max(0),
            ..Self::default()
        };
        for creature in &observation.creatures {
            if creature.side == CombatSide::Player {
                threat.allies_alive += usize::from(creature.current_hp > 0);
                continue;
            }
            threat.enemies_alive += usize::from(creature.current_hp > 0);
            threat.enemy_block += i64::from(creature.block.max(0));
            threat.enemy_hp += i64::from(creature.current_hp.max(0));
            for entry in creature.intent.iter().flat_map(|intent| &intent.intents) {
                if is_a_blow(entry) {
                    threat.incoming += i64::from(entry.damage.unwrap_or(0))
                        * i64::from(entry.repeats.unwrap_or(1));
                }
            }
        }
        threat.unblocked =
            (threat.incoming - i64::from(observation.block.unwrap_or(0).max(0))).max(0);
        threat
    }

    /// The unblocked blow as a share of what the player has left, capped the
    /// way the gold scalar is: past twice over, the bit below says the rest.
    fn lethality(self) -> f32 {
        if self.player_hp <= 0 {
            return 0.0;
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "damage and hit points are small"
        )]
        {
            (self.unblocked as f32 / self.player_hp as f32).min(2.0)
        }
    }

    fn would_die(self) -> bool {
        self.player_hp > 0 && self.unblocked >= i64::from(self.player_hp)
    }

    /// The enemy side's hit points, log-scaled like every other raw count:
    /// several Waterfall Giants' worth of sentinel still has to land on the
    /// scale rather than off it.
    fn enemy_hp(self) -> i32 {
        i32::try_from(self.enemy_hp).unwrap_or(i32::MAX)
    }
}

/// What a walk that steps onto a map point can still meet.
///
/// The act's map is a graph of points with paths one row up, and a step
/// commits the walk to the sub-graph above the point it lands on. Everything
/// that separates one step from the step beside it lives there — which rooms
/// the branch cannot avoid, which it merely offers, and how much act is left
/// — and none of it is on the point itself. The observation's [`ZONE_MAP`]
/// tokens carry the whole graph, but a pointer scorer is additive in the
/// option's own block and a context every option on the screen shares, so a
/// branch described only by those tokens cannot move one option's logit
/// relative to another's. Hence the read is on the action.
///
/// Per type and per bound rather than per path: the fewest rooms of a type
/// over all the paths from here is what the branch forces, the most is what
/// it offers, and the pair brackets the branch without enumerating paths, of
/// which a map holds exponentially many.
#[derive(Clone, Copy, Debug, Default)]
struct Route {
    /// The fewest rooms of each type standing on any path from this point to
    /// the end of the act, this point counted.
    least: [u16; MAP_TYPES],
    /// The most rooms of each type, over the same paths.
    most: [u16; MAP_TYPES],
    /// The most points such a walk stands on, this one included. Every path
    /// climbs exactly one row a step, so on a well-formed map this is also
    /// the rows left between here and the act's end.
    depth: u16,
}

impl Route {
    /// A walk that ends where it starts: one room, of this point's own type.
    fn of_point(point_type: MapPointType) -> Self {
        let mut counted = [0_u16; MAP_TYPES];
        counted[map_type_slot(point_type)] = 1;
        Self {
            least: counted,
            most: counted,
            depth: 1,
        }
    }
}

/// How a mark stands during the walk below: never reached, on the stack, or
/// settled. The middle state is what makes an edge back into the walk
/// recognizable as one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mark {
    Fresh,
    Open,
    Done,
}

/// The [`Route`] of every point of an act's map, settled in one pass.
///
/// The dynamic program is over the point set, not over the actions: a map
/// screen offers up to four steps onto one map of fifty to seventy points, so
/// walking it per action would pay for the same sub-graphs four times.
/// [`PolicyEncoder::encode_actions`] threads one table through a whole
/// decision's action list and it is built on the first map step encoded, so a
/// decision that offers none never pays for it at all. The table is a plain
/// local passed by `&mut`: the encoder itself stays free of interior
/// mutability, which is what lets the determinism audit and the parallel
/// batch harness share one encoder across threads.
#[derive(Debug, Default)]
struct RouteTable {
    routes: BTreeMap<MapCoord, Route>,
}

impl RouteTable {
    /// The table for the act's map, or an empty one where no map is on
    /// screen.
    ///
    /// The walk is an explicit-stack post-order with a three-colour mark
    /// rather than recursion. A map is a DAG a dozen or so rows deep, so
    /// recursion would in fact fit; the point is that a graph arriving
    /// malformed — a cycle, a child naming a coordinate that is not on the
    /// map — must cost a wrong number rather than a hung encoder or a blown
    /// stack. An edge into a point still on the stack closes a cycle and
    /// contributes nothing, which is the right answer for a path that cannot
    /// be completed; a child the map does not carry is simply not an edge.
    fn of(map: Option<&VisibleMap>) -> Self {
        let Some(map) = map else {
            return Self::default();
        };
        let position: BTreeMap<MapCoord, usize> = map
            .points
            .iter()
            .enumerate()
            .map(|(index, point)| (point.coord, index))
            .collect();
        let children: Vec<Vec<usize>> = map
            .points
            .iter()
            .map(|point| {
                point
                    .children
                    .iter()
                    .filter_map(|coord| position.get(coord).copied())
                    .collect()
            })
            .collect();
        let mut mark = vec![Mark::Fresh; map.points.len()];
        let mut routes = vec![Route::default(); map.points.len()];
        let mut stack: Vec<usize> = Vec::new();
        for root in 0..map.points.len() {
            if mark[root] != Mark::Fresh {
                continue;
            }
            stack.push(root);
            while let Some(&node) = stack.last() {
                match mark[node] {
                    // Reached again through another parent, or pushed twice
                    // by one.
                    Mark::Done => {
                        stack.pop();
                    }
                    Mark::Fresh => {
                        mark[node] = Mark::Open;
                        for &child in &children[node] {
                            if mark[child] == Mark::Fresh {
                                stack.push(child);
                            }
                        }
                    }
                    // Everything pushed above this point has settled, so
                    // every child that is on a completable path is `Done`.
                    Mark::Open => {
                        stack.pop();
                        routes[node] = settle_route(
                            map.points[node].point_type,
                            &children[node],
                            &mark,
                            &routes,
                        );
                        mark[node] = Mark::Done;
                    }
                }
            }
        }
        Self {
            routes: map
                .points
                .iter()
                .map(|point| point.coord)
                .zip(routes)
                .collect(),
        }
    }

    /// The route from a coordinate, or nothing where the map carries no point
    /// there.
    fn at(&self, coord: MapCoord) -> Option<Route> {
        self.routes.get(&coord).copied()
    }
}

/// One point's route, off its own type and its settled children's.
///
/// A point with no completable child is where a walk ends and carries its own
/// room alone. Otherwise the bounds compose the way bounds do: the fewest of
/// a type ahead is this point's own contribution plus the least any one
/// branch forces, the most is its contribution plus the most any one branch
/// offers, and the depth is one more than the deepest branch.
fn settle_route(
    point_type: MapPointType,
    children: &[usize],
    mark: &[Mark],
    routes: &[Route],
) -> Route {
    let mut settled = Route::of_point(point_type);
    let mut least = [u16::MAX; MAP_TYPES];
    let mut most = [0_u16; MAP_TYPES];
    let mut depth = 0_u16;
    let mut walks_on = false;
    for &child in children {
        if mark[child] != Mark::Done {
            continue;
        }
        let route = routes[child];
        walks_on = true;
        for (bound, count) in least.iter_mut().zip(route.least) {
            *bound = (*bound).min(count);
        }
        for (bound, count) in most.iter_mut().zip(route.most) {
            *bound = (*bound).max(count);
        }
        depth = depth.max(route.depth);
    }
    if walks_on {
        // Saturating because a malformed graph is the only way these can
        // approach the type's range, and a saturated count is a bounded
        // wrong answer where a wrap would be an unbounded one.
        for (own, ahead) in settled.least.iter_mut().zip(least) {
            *own = own.saturating_add(ahead);
        }
        for (own, ahead) in settled.most.iter_mut().zip(most) {
            *own = own.saturating_add(ahead);
        }
        settled.depth = settled.depth.saturating_add(depth);
    }
    settled
}

/// The route lookahead, on the step that opens it.
fn write_route(features: &mut [f32], route: Route) {
    for (slot, count) in route.least.into_iter().enumerate() {
        features[ACTION_ROUTE_BASE + slot] = f32::from(count) / 8.0;
    }
    for (slot, count) in route.most.into_iter().enumerate() {
        features[ACTION_ROUTE_MOST_BASE + slot] = f32::from(count) / 8.0;
    }
    features[ACTION_ROUTE_DEPTH] = f32::from(route.depth) / 16.0;
}

impl PolicyEncoder {
    /// An encoder over a vocabulary of model ids, reading the registry's own
    /// card bodies for the static numbers a card token carries. The ids are
    /// sorted and deduplicated, so the index a model lands on — and the
    /// vocabulary hash — is a function of the set alone. Index 0 is reserved
    /// for padding and for models outside the vocabulary.
    ///
    /// Build the set with [`standard_vocabulary`]; a call site that passes
    /// only the registry's models will hash to a vocabulary no checkpoint
    /// was trained against, which is exactly what the load-time check exists
    /// to catch. The statics ride outside the hash: they are the registry's,
    /// and the registry is already pinned by the compatibility manifest.
    #[must_use]
    pub fn new(
        model_ids: impl IntoIterator<Item = ModelId>,
        registry: &sts2_engine::ContentRegistry,
    ) -> Self {
        let sorted: std::collections::BTreeSet<ModelId> = model_ids.into_iter().collect();
        let mut digest = Sha256::new();
        let mut vocabulary = BTreeMap::new();
        for (index, id) in sorted.into_iter().enumerate() {
            digest.update(id.to_string().as_bytes());
            digest.update(b"\n");
            #[allow(
                clippy::cast_possible_truncation,
                reason = "vocabularies are far below u32"
            )]
            vocabulary.insert(id, index as u32 + 1);
        }
        let hash = format!("sha256:{:x}", digest.finalize());
        let mut statics = BTreeMap::new();
        for id in registry.registered_model_ids() {
            let Some(definition) = registry.card(&id) else {
                continue;
            };
            let base = CardStatics::of(definition, &definition.effects);
            let upgraded = definition.upgrade.as_ref().map_or(base, |upgrade| {
                CardStatics::of(definition, &upgrade.effects)
            });
            statics.insert(id, [base, upgraded]);
        }
        Self {
            vocabulary,
            statics,
            hash,
        }
    }

    /// The hash naming the vocabulary, carried beside every emitted sample
    /// and every checkpoint.
    #[must_use]
    pub fn vocabulary_hash(&self) -> &str {
        &self.hash
    }

    /// One past the largest vocabulary index: the embedding-table size a
    /// consumer allocates.
    #[must_use]
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len() + 1
    }

    fn index(&self, model_id: &ModelId) -> u32 {
        self.vocabulary.get(model_id).copied().unwrap_or(0)
    }

    fn move_index(&self, state: &str) -> u32 {
        move_id(state).map_or(0, |id| self.index(&id))
    }

    /// A card's registry numbers in the form it is standing in. `None` for
    /// anything that is not a registered card — a relic, a potion, a map
    /// point — whose static block stays empty rather than lighting some
    /// card type it is not.
    fn statics(&self, model_id: &ModelId, upgrade_level: u8) -> Option<&CardStatics> {
        self.statics
            .get(model_id)
            .map(|forms| &forms[usize::from(upgrade_level.min(1))])
    }

    /// The registry's static card numbers, on a token row.
    fn write_statics(&self, row: &mut [f32; TOKEN_FEATURES], model_id: &ModelId, upgrade: u8) {
        let Some(statics) = self.statics(model_id, upgrade) else {
            return;
        };
        row[STATIC_BASE + statics.card_type] = 1.0;
        row[STATIC_TARGET_BASE + statics.target] = 1.0;
        #[allow(clippy::cast_precision_loss, reason = "card numbers are small")]
        {
            row[STATIC_NUMBER_BASE] = statics.damage as f32 / 50.0;
            row[STATIC_NUMBER_BASE + 1] = statics.hits as f32 / 10.0;
            row[STATIC_NUMBER_BASE + 2] = statics.block as f32 / 50.0;
            row[STATIC_NUMBER_BASE + 3] = statics.granted_powers as f32 / 4.0;
        }
    }

    /// The same numbers on an action's feature vector.
    fn write_action_statics(&self, features: &mut [f32], model_id: &ModelId, upgrade: u8) {
        let Some(statics) = self.statics(model_id, upgrade) else {
            return;
        };
        features[ACTION_STATIC_BASE + statics.card_type] = 1.0;
        features[ACTION_STATIC_TARGET_BASE + statics.target] = 1.0;
        #[allow(clippy::cast_precision_loss, reason = "card numbers are small")]
        {
            features[ACTION_STATIC_NUMBER_BASE] = statics.damage as f32 / 50.0;
            features[ACTION_STATIC_NUMBER_BASE + 1] = statics.hits as f32 / 10.0;
            features[ACTION_STATIC_NUMBER_BASE + 2] = statics.block as f32 / 50.0;
            features[ACTION_STATIC_NUMBER_BASE + 3] = statics.granted_powers as f32 / 4.0;
        }
    }

    /// The index of `model_id`, or of `fallback` where the model is outside
    /// the vocabulary.
    ///
    /// A token index of zero is the padding row, which the net masks out of
    /// its pool — so a token that names nothing is a token that is not there.
    /// That is right for a card the vocabulary has never heard of and wrong
    /// for a map point, which is on the screen whatever stands on it: act
    /// one's ancient is `EVENT.NEOW`, which the registry carries no event
    /// definition for, and the point still has to be a point.
    fn index_or(&self, model_id: &ModelId, fallback: &ModelId) -> u32 {
        match self.index(model_id) {
            0 => self.index(fallback),
            index => index,
        }
    }

    /// The observation as tensors.
    ///
    /// The token feature row, block by block:
    ///
    /// | slots | block |
    /// |---|---|
    /// | `0..14` | the zone one-hot |
    /// | `14..21` | the shared value slots |
    /// | `21..30` | the owner one-hot (`OWNER_BASE`) |
    /// | `30..45` | the intent-type one-hot |
    /// | `45..47` | log-scaled current and maximum hit points |
    /// | `47..62` | the card face |
    /// | `62..81` | the registry's static card numbers |
    /// | `81..89` | the map point-type one-hot |
    /// | `89..97` | the per-zone extras |
    ///
    /// Per-zone meaning of the shared feature slots (`ZONES + n`):
    /// `+0` upgrade (cards), `+1` enchanted flag (cards),
    /// `+2` hp fraction / power amount / resolved orb passive / relic
    /// counter, `+3` block / power counter / resolved orb evoke / relic
    /// second counter, `+4` player side (creatures), latch (relics), rider
    /// flag (enchantment and affliction riders), `+5` the shown intent's
    /// total damage, counted off attacking lines alone, `+6` its hit count,
    /// counted the same way.
    ///
    /// Then the owner one-hot: which creature slot a creature, power, intent
    /// line or move-history token belongs to, in the nine slots an action's
    /// target block shares. Then the intent-type one-hot — on a creature, the
    /// union of the kinds its shown lines advertise; on an intent-line token,
    /// that line's own kind. Then the log-scaled current and maximum hit
    /// points: the fraction above says how hurt, this pair says what is
    /// lethal. Then the card face: the four displayed cost components (kept
    /// apart, because "cheaper until played" is not "cheaper this combat"),
    /// the X flag, replay marks, ethereal, retain, sly, grown damage, star
    /// cost, the damage the card's enchantment has grown, whether that
    /// enchantment is spent, and the `FACE_PREVIEW_BASE` pair — the damage
    /// and the block the card in hand actually prints, hook-resolved, empty
    /// in every other pile. Then the static block: the card-type one-hot, the
    /// target-type one-hot, base damage / 50, hit count / 10, base block / 50
    /// and granted powers / 4. Then the map point-type one-hot. Then the
    /// per-zone extras (`EXTRA_BASE + n`):
    ///
    /// | n | map | offer | intent line | creature | other |
    /// |---|-----|-------|-------------|----------|-------|
    /// | 0 | column / width | price / 100 | line damage / 50 | unblocked blows / 10 | potion slot / 4, deck floor / 60 |
    /// | 1 | row / height | affordable | line repeats / 10 | | |
    /// | 2 | children / 4 | offer index / 8 | line index / 4 | | |
    /// | 3 | walked | gold on the line / 100 | recency of a performed move | | |
    /// | 4 | stood on now | options on the line / 4 | a move already performed | | |
    /// | 5 | offered as a step now | taken, sold, or chosen | | | |
    /// | 6 | rows ahead / 8 | a card on a choice | | | |
    /// | 7 | | already picked | | | |
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "one block per token zone, in layout order"
    )]
    pub fn encode_observation(&self, observation: &AgentObservation) -> ObservationEncoding {
        let mut tokens: Vec<u32> = Vec::new();
        let mut features: Vec<f32> = Vec::new();
        let mut truncated = false;
        let mut push = |token: u32, row: [f32; TOKEN_FEATURES]| {
            if tokens.len() < MAX_TOKENS {
                tokens.push(token);
                features.extend_from_slice(&row);
            } else {
                truncated = true;
            }
        };
        let card_row = |zone: usize,
                        fingerprint: &CardFingerprint,
                        face: &sts2_engine::CardFace,
                        preview: Option<&CardPreview>| {
            let mut row = [0.0; TOKEN_FEATURES];
            row[zone] = 1.0;
            row[SHARED_BASE] = f32::from(fingerprint.upgrade_level);
            row[SHARED_BASE + 1] = f32::from(fingerprint.enchantment.is_some());
            write_face(&mut row, face);
            #[allow(clippy::cast_precision_loss, reason = "displayed numbers are small")]
            if let Some(preview) = preview {
                row[FACE_PREVIEW_BASE] = preview.damage.unwrap_or(0) as f32 / 50.0;
                row[FACE_PREVIEW_BASE + 1] = preview.block.unwrap_or(0) as f32 / 50.0;
            }
            self.write_statics(&mut row, &fingerprint.model_id, fingerprint.upgrade_level);
            row
        };
        // An enchanted card is two tokens: the card, then its enchantment as
        // a rider in the same zone, marked and carrying its amount.
        let rider_row = |zone: usize, amount: i32| {
            let mut row = [0.0; TOKEN_FEATURES];
            row[zone] = 1.0;
            row[SHARED_BASE + 4] = 1.0;
            #[allow(clippy::cast_precision_loss, reason = "amounts are small")]
            {
                row[SHARED_BASE + 2] = amount as f32 / 10.0;
            }
            row
        };
        // Creatures first — their slot order is what action targets point
        // into — then the fight's piles, then the run's holdings, then the
        // run-level sights, then whatever screen is standing.
        for (slot, creature) in observation.creatures.iter().enumerate() {
            let mut row = [0.0; TOKEN_FEATURES];
            row[ZONE_CREATURE] = 1.0;
            #[allow(clippy::cast_precision_loss, reason = "hit points are small")]
            {
                row[SHARED_BASE + 2] = if creature.max_hp > 0 {
                    creature.current_hp.max(0) as f32 / creature.max_hp as f32
                } else {
                    0.0
                };
                row[SHARED_BASE + 3] = creature.block as f32 / 50.0;
            }
            row[HP_BASE] = log_hp(creature.current_hp);
            row[HP_BASE + 1] = log_hp(creature.max_hp);
            row[SHARED_BASE + 4] = f32::from(creature.side == CombatSide::Player);
            row[OWNER_BASE + owner_slot(slot)] = 1.0;
            #[allow(clippy::cast_precision_loss, reason = "blow counts are small")]
            {
                row[EXTRA_BASE] = creature.unblocked_blows_taken as f32 / 10.0;
            }
            if let Some(intent) = &creature.intent {
                let mut damage = 0_i64;
                let mut hits = 0_i64;
                for entry in &intent.intents {
                    let repeats = i64::from(entry.repeats.unwrap_or(1));
                    // Only a blow adds to the damage total and the hit
                    // count; a defend or a buff line says what it is and
                    // nothing more.
                    if is_a_blow(entry) {
                        damage += i64::from(entry.damage.unwrap_or(0)) * repeats;
                        hits += repeats;
                    }
                    if let Some(slot) = intent_type_slot(&entry.intent_type) {
                        row[INTENT_BASE + slot] = 1.0;
                    }
                }
                #[allow(clippy::cast_precision_loss, reason = "intents are small")]
                {
                    row[SHARED_BASE + 5] = damage as f32 / 50.0;
                    row[SHARED_BASE + 6] = hits as f32 / 10.0;
                }
            }
            push(self.index(&creature.model_id), row);
            for power in &creature.powers {
                let mut power_row = [0.0; TOKEN_FEATURES];
                power_row[ZONE_POWER] = 1.0;
                power_row[OWNER_BASE + owner_slot(slot)] = 1.0;
                #[allow(clippy::cast_precision_loss, reason = "amounts are small")]
                {
                    power_row[SHARED_BASE + 2] = power.amount as f32 / 10.0;
                    power_row[SHARED_BASE + 3] = power.counter as f32 / 10.0;
                }
                push(self.index(&power.model_id), power_row);
            }
            // One token per shown line, each naming the move it belongs to.
            // The move id is the whole rotation position: every one of the
            // eight bosses runs a deterministic machine, so two states that
            // advertise the same numbers still have different futures.
            if let Some(intent) = &creature.intent {
                let token = self.move_index(&intent.move_id);
                if intent.intents.is_empty() {
                    let mut line = [0.0; TOKEN_FEATURES];
                    line[ZONE_INTENT] = 1.0;
                    line[OWNER_BASE + owner_slot(slot)] = 1.0;
                    push(token, line);
                }
                for (position, entry) in intent.intents.iter().enumerate() {
                    let mut line = [0.0; TOKEN_FEATURES];
                    line[ZONE_INTENT] = 1.0;
                    line[OWNER_BASE + owner_slot(slot)] = 1.0;
                    if let Some(kind) = intent_type_slot(&entry.intent_type) {
                        line[INTENT_BASE + kind] = 1.0;
                    }
                    #[allow(clippy::cast_precision_loss, reason = "damages and repeats are small")]
                    {
                        line[EXTRA_BASE] = entry.damage.unwrap_or(0) as f32 / 50.0;
                        line[EXTRA_BASE + 1] = entry.repeats.unwrap_or(1) as f32 / 10.0;
                        line[EXTRA_BASE + 2] = position as f32 / 4.0;
                    }
                    push(token, line);
                }
            }
            // What the machine has already done, most recent first. A move
            // state is a vocabulary index rather than a number, so the only
            // place the history fits is a token slot of its own; it rides in
            // the intent zone under a marker, because the zone one-hot is
            // fourteen wide and a consumer's per-zone pooling depends on it
            // staying that way. The states entered are not encoded beside
            // them: they differ from these by exactly the move the intent
            // above already advertises.
            for (position, token) in creature
                .move_history
                .recent_performed
                .iter()
                .rev()
                .map(|state| self.move_index(state))
                .filter(|token| *token != 0)
                .take(MOVE_HISTORY_DEPTH)
                .enumerate()
            {
                let mut line = [0.0; TOKEN_FEATURES];
                line[ZONE_INTENT] = 1.0;
                line[OWNER_BASE + owner_slot(slot)] = 1.0;
                line[EXTRA_BASE + 4] = 1.0;
                #[allow(clippy::cast_precision_loss, reason = "the depth is three")]
                {
                    line[EXTRA_BASE + 3] =
                        (MOVE_HISTORY_DEPTH - position) as f32 / MOVE_HISTORY_DEPTH as f32;
                }
                push(token, line);
            }
        }
        // Matched by card id rather than by position: the list runs in hand
        // order, but an old sample deserializes it empty (serde default) and
        // a preview that stands alone is exactly what the id is for.
        let previews = preview_index(observation);
        let piles = [
            (
                ZONE_HAND,
                observation
                    .hand
                    .iter()
                    .map(|handle| {
                        (
                            &handle.fingerprint,
                            &handle.face,
                            previews.get(&handle.card_id).copied(),
                        )
                    })
                    .collect(),
            ),
            (
                ZONE_DRAW,
                observation
                    .draw_pile
                    .iter()
                    .map(|card| (&card.fingerprint, &card.face, None))
                    .collect(),
            ),
            (ZONE_DISCARD, handle_pairs(&observation.discard)),
            (ZONE_EXHAUST, handle_pairs(&observation.exhaust)),
            (ZONE_PLAY, handle_pairs(&observation.play)),
        ];
        for (zone, cards) in piles {
            for (fingerprint, face, preview) in cards {
                push(
                    self.index(&fingerprint.model_id),
                    card_row(zone, fingerprint, face, preview),
                );
                if let Some(enchantment) = &fingerprint.enchantment {
                    push(
                        self.index(&enchantment.model_id),
                        rider_row(zone, enchantment.amount),
                    );
                }
                // An affliction is a rider too: its identity is a model.
                if let Some(affliction) = &face.affliction {
                    push(self.index(affliction), rider_row(zone, 0));
                }
            }
        }
        for relic in &observation.relics {
            let mut row = [0.0; TOKEN_FEATURES];
            row[ZONE_RELIC] = 1.0;
            #[allow(clippy::cast_precision_loss, reason = "counters are small")]
            {
                row[SHARED_BASE + 2] = relic.counter as f32 / 10.0;
                row[SHARED_BASE + 3] = relic.second_counter as f32 / 10.0;
            }
            row[SHARED_BASE + 4] = f32::from(relic.latched);
            push(self.index(&relic.model_id), row);
        }
        // The belt's empty slots are part of the sight: which slot a potion
        // stands in is what a discard names.
        for (slot, potion) in observation.potions.iter().enumerate() {
            let Some(potion) = potion else { continue };
            let mut row = [0.0; TOKEN_FEATURES];
            row[ZONE_POTION] = 1.0;
            #[allow(clippy::cast_precision_loss, reason = "belts are four slots")]
            {
                row[EXTRA_BASE] = slot as f32 / 4.0;
            }
            push(self.index(potion), row);
        }
        if let Some(orbs) = &observation.orbs {
            for orb in &orbs.orbs {
                let mut row = [0.0; TOKEN_FEATURES];
                row[ZONE_ORB] = 1.0;
                #[allow(clippy::cast_precision_loss, reason = "orb values are small")]
                {
                    // The resolved values, which are the numbers the orb
                    // shows — not the bases the modifiers start from.
                    row[SHARED_BASE + 2] = orb.passive as f32 / 10.0;
                    row[SHARED_BASE + 3] = orb.evoke as f32 / 10.0;
                }
                push(self.index(&orb.model_id), row);
            }
        }
        // The run deck and the act's map are on the screen at every moment of
        // a run, but they are what the *run-level* decisions are made of, and
        // a fifty-point map pooled in with a fight's forty tokens would drown
        // it. While an enemy stands, the piles carry the deck's combat copies
        // and the map is not what is being answered, so both zones wait.
        let fighting = observation
            .creatures
            .iter()
            .any(|creature| creature.side != CombatSide::Player);
        let threat = BoardThreat::of(observation);
        if !fighting {
            for card in &observation.deck {
                let mut row = [0.0; TOKEN_FEATURES];
                row[ZONE_DECK] = 1.0;
                row[SHARED_BASE] = f32::from(card.upgrade_level);
                row[SHARED_BASE + 1] = f32::from(card.enchantment.is_some());
                self.write_statics(&mut row, &card.model_id, card.upgrade_level);
                #[allow(clippy::cast_precision_loss, reason = "floors are small")]
                {
                    row[EXTRA_BASE] = card.floor_added_to_deck.unwrap_or(0) as f32 / 60.0;
                }
                push(self.index(&card.model_id), row);
                if let Some(enchantment) = &card.enchantment {
                    push(
                        self.index(&enchantment.model_id),
                        rider_row(ZONE_DECK, enchantment.amount),
                    );
                }
            }
            if let Some(map) = &observation.map {
                let destinations: &[MapCoord] = match &observation.decision {
                    DecisionContext::MapNavigation { destinations, .. } => destinations,
                    _ => &[],
                };
                for point in &map.points {
                    let mut row = [0.0; TOKEN_FEATURES];
                    row[ZONE_MAP] = 1.0;
                    row[MAP_TYPE_BASE + map_type_slot(point.point_type)] = 1.0;
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "map coordinates are single digits"
                    )]
                    {
                        row[EXTRA_BASE] = f32::from(point.coord.col) / f32::from(map.width.max(1));
                        row[EXTRA_BASE + 1] =
                            f32::from(point.coord.row) / f32::from(map.height.max(1));
                        row[EXTRA_BASE + 2] = point.children.len() as f32 / 4.0;
                        row[EXTRA_BASE + 6] = observation.map_coord.map_or(0.0, |here| {
                            f32::from(point.coord.row.saturating_sub(here.row)) / 8.0
                        });
                    }
                    row[EXTRA_BASE + 3] = f32::from(point.visited);
                    row[EXTRA_BASE + 4] = f32::from(observation.map_coord == Some(point.coord));
                    row[EXTRA_BASE + 5] = f32::from(destinations.contains(&point.coord));
                    // The boss and the ancient are drawn as themselves on
                    // their own points, so they are named as themselves.
                    let fallback = map_point_id(point.point_type);
                    let token = match point.point_type {
                        MapPointType::Boss if Some(point.coord) == map.second_boss_coord => {
                            self.index_or(map.second_boss.as_ref().unwrap_or(&map.boss), &fallback)
                        }
                        MapPointType::Boss => self.index_or(&map.boss, &fallback),
                        MapPointType::Ancient => self.index_or(&map.ancient, &fallback),
                        _ => self.index(&fallback),
                    };
                    push(token, row);
                }
            }
        }
        self.push_offers(observation, &mut push);
        let filled = tokens.len();
        tokens.resize(MAX_TOKENS, 0);
        features.resize(MAX_TOKENS * TOKEN_FEATURES, 0.0);

        let mut scalars = Vec::with_capacity(OBSERVATION_SCALARS);
        let hp_fraction = match (observation.current_hp, observation.max_hp) {
            (Some(current), Some(max)) if max > 0 => {
                #[allow(clippy::cast_precision_loss, reason = "hit points are small")]
                {
                    current.max(0) as f32 / max as f32
                }
            }
            _ => 0.0,
        };
        let odds = observation
            .odds
            .unwrap_or_else(sts2_engine::PlayerOdds::start_of_run);
        let room_odds = observation
            .unknown_room_odds
            .unwrap_or_else(sts2_engine::UnknownRoomOdds::start_of_act)
            .weights();
        #[allow(
            clippy::cast_precision_loss,
            reason = "counts and totals are far below f32 precision"
        )]
        {
            scalars.push(hp_fraction);
            scalars.push(log_hp(observation.current_hp.unwrap_or(0)));
            scalars.push(log_hp(observation.max_hp.unwrap_or(0)));
            scalars.push(observation.block.unwrap_or(0) as f32 / 50.0);
            scalars.push(observation.energy.unwrap_or(0) as f32 / 10.0);
            scalars.push(observation.max_energy.unwrap_or(0) as f32 / 10.0);
            scalars.push(observation.stars.unwrap_or(0) as f32 / 10.0);
            scalars.push((observation.gold.unwrap_or(0) as f32 / 1000.0).min(2.0));
            scalars.push(observation.current_act.unwrap_or(0) as f32 / 4.0);
            scalars.push(observation.floor.unwrap_or(0) as f32 / 60.0);
            scalars.push(observation.turn.unwrap_or(0) as f32 / 30.0);
            scalars.push(observation.hand.len() as f32 / 10.0);
            scalars.push(observation.draw_pile_size as f32 / 50.0);
            scalars.push(observation.discard.len() as f32 / 50.0);
            scalars.push(observation.exhaust.len() as f32 / 50.0);
            scalars.push(observation.play.len() as f32 / 4.0);
            scalars.push(observation.relics.len() as f32 / 20.0);
            scalars.push(observation.potions.iter().flatten().count() as f32 / 4.0);
            scalars.push(
                observation
                    .orbs
                    .as_ref()
                    .map_or(0.0, |orbs| orbs.capacity as f32 / 10.0),
            );
            scalars.push(
                observation
                    .orbs
                    .as_ref()
                    .map_or(0.0, |orbs| orbs.orbs.len() as f32 / 10.0),
            );
            scalars.push(filled as f32 / MAX_TOKENS as f32);
            scalars.push(f32::from(truncated));
            scalars.push(f32::from(observation.ascension.unwrap_or(0)) / 10.0);
            scalars.push(observation.deck.len() as f32 / 40.0);
            scalars.push(f32::from(fighting));
            scalars.push(observation.tallies.energy_spent_this_turn as f32 / 10.0);
            scalars.push(observation.tallies.status_cards_drawn_this_turn as f32 / 5.0);
            scalars.push(observation.tallies.first_plays_started_this_turn as f32 / 10.0);
            scalars.push(observation.tallies.zero_cost_attack_plays_this_turn as f32 / 5.0);
            scalars.push(observation.tallies.orbs_channeled_this_combat.len() as f32 / 10.0);
            scalars.push(odds.card_rarity_offset());
            scalars.push(odds.potion_reward());
            for weight in room_odds {
                scalars.push(weight);
            }
            scalars.push(
                observation
                    .map
                    .as_ref()
                    .map_or(0.0, |map| map.points.len() as f32 / 64.0),
            );
            scalars.push(match &observation.decision {
                DecisionContext::MapNavigation { destinations, .. } => {
                    destinations.len() as f32 / 4.0
                }
                _ => 0.0,
            });
            scalars.push(offer_count(observation) as f32 / 8.0);
            scalars.push(threat.incoming as f32 / 50.0);
            scalars.push(threat.unblocked as f32 / 50.0);
            scalars.push(threat.lethality());
            scalars.push(f32::from(threat.would_die()));
            scalars.push(threat.enemies_alive as f32 / 8.0);
            scalars.push(threat.allies_alive as f32 / 8.0);
            scalars.push(threat.enemy_block as f32 / 50.0);
        }
        scalars.push(log_hp(threat.enemy_hp()));
        debug_assert_eq!(scalars.len(), SCALAR_SLOTS);
        let mut decision = [0.0_f32; DECISION_KINDS];
        decision[decision_kind(&observation.decision)] = 1.0;
        scalars.extend_from_slice(&decision);
        debug_assert_eq!(scalars.len(), OBSERVATION_SCALARS);

        ObservationEncoding {
            scalars,
            tokens,
            features,
            live: filled,
        }
    }

    /// What the standing screen offers, as tokens.
    ///
    /// The value head reads observation tokens, so offered choices must be
    /// represented here as well as in the policy's per-action encodings.
    #[allow(
        clippy::too_many_lines,
        reason = "one block per screen kind, in variant order"
    )]
    fn push_offers(
        &self,
        observation: &AgentObservation,
        push: &mut impl FnMut(u32, [f32; TOKEN_FEATURES]),
    ) {
        let gold = observation.gold.unwrap_or(0);
        let blank = || {
            let mut row = [0.0; TOKEN_FEATURES];
            row[ZONE_OFFER] = 1.0;
            row
        };
        #[allow(
            clippy::cast_precision_loss,
            reason = "prices, indices and counts are small"
        )]
        match &observation.decision {
            DecisionContext::ChooseCards {
                candidates, picked, ..
            } => {
                for (index, candidate) in candidates.iter().enumerate() {
                    let mut row = blank();
                    row[SHARED_BASE] = f32::from(candidate.fingerprint.upgrade_level);
                    row[SHARED_BASE + 1] = f32::from(candidate.fingerprint.enchantment.is_some());
                    write_face(&mut row, &candidate.face);
                    self.write_statics(
                        &mut row,
                        &candidate.fingerprint.model_id,
                        candidate.fingerprint.upgrade_level,
                    );
                    row[EXTRA_BASE + 2] = index as f32 / 8.0;
                    row[EXTRA_BASE + 6] = 1.0;
                    row[EXTRA_BASE + 7] =
                        f32::from(picked.iter().any(|held| held.card_id == candidate.card_id));
                    push(self.index(&candidate.fingerprint.model_id), row);
                }
            }
            DecisionContext::Rewards { offers, .. } => {
                for offer in offers {
                    let mut row = blank();
                    row[EXTRA_BASE + 2] = offer.index as f32 / 8.0;
                    row[EXTRA_BASE + 3] = offer.gold_amount as f32 / 100.0;
                    row[EXTRA_BASE + 4] = offer.option_count as f32 / 4.0;
                    row[EXTRA_BASE + 5] = f32::from(offer.selected || offer.passed_over);
                    let token = offer.model_id.as_ref().map_or_else(
                        || self.index(&reward_line_id(&offer.reward_type)),
                        |id| self.index(id),
                    );
                    push(token, row);
                    for card in offer
                        .offered_cards
                        .iter()
                        .chain(offer.special_card.as_ref())
                    {
                        let mut card_row = blank();
                        card_row[SHARED_BASE] = f32::from(card.upgrade_level);
                        card_row[SHARED_BASE + 1] = f32::from(card.enchantment.is_some());
                        self.write_statics(&mut card_row, &card.model_id, card.upgrade_level);
                        card_row[EXTRA_BASE + 2] = offer.index as f32 / 8.0;
                        card_row[EXTRA_BASE + 6] = 1.0;
                        push(self.index(&card.model_id), card_row);
                    }
                }
            }
            DecisionContext::Shop {
                offers,
                removal_price,
                removal_used,
                ..
            } => {
                for offer in offers {
                    let mut row = blank();
                    row[EXTRA_BASE] = offer.price as f32 / 100.0;
                    row[EXTRA_BASE + 1] = f32::from(gold >= offer.price);
                    row[EXTRA_BASE + 2] = offer.index as f32 / 8.0;
                    row[EXTRA_BASE + 5] = f32::from(offer.sold);
                    let token = match &offer.item {
                        ShopItem::Card(card) => {
                            row[SHARED_BASE] = f32::from(card.upgrade_level);
                            row[SHARED_BASE + 1] = f32::from(card.enchantment.is_some());
                            self.write_statics(&mut row, &card.model_id, card.upgrade_level);
                            row[EXTRA_BASE + 6] = 1.0;
                            self.index(&card.model_id)
                        }
                        ShopItem::Potion(id) | ShopItem::Relic(id) => self.index(id),
                    };
                    push(token, row);
                }
                let mut row = blank();
                row[EXTRA_BASE] = *removal_price as f32 / 100.0;
                row[EXTRA_BASE + 1] = f32::from(gold >= *removal_price);
                row[EXTRA_BASE + 5] = f32::from(*removal_used);
                push(self.index(&reward_line_id("card_removal")), row);
            }
            DecisionContext::Treasure { relics } => {
                for (index, relic) in relics.iter().enumerate() {
                    let mut row = blank();
                    row[EXTRA_BASE + 2] = index as f32 / 8.0;
                    push(self.index(relic), row);
                }
            }
            DecisionContext::Event { model_id, options } => {
                for (index, option) in options.iter().enumerate() {
                    let mut row = blank();
                    let body = EventBody::of(&option.effects, observation);
                    // The two an event page is usually deciding between, on the
                    // row's own free extras. The action block carries the whole
                    // body; a token row cannot without widening all 256 of them
                    // for the sake of the three or four an event stands up.
                    #[allow(clippy::cast_precision_loss, reason = "event numbers are small")]
                    {
                        row[EXTRA_BASE] = body.hp as f32 / 32.0;
                        row[EXTRA_BASE + 1] = body.gold as f32 / 100.0;
                    }
                    row[EXTRA_BASE + 2] = index as f32 / 8.0;
                    row[EXTRA_BASE + 3] = f32::from(body.unread);
                    row[EXTRA_BASE + 4] = option.effects.len() as f32 / 4.0;
                    row[EXTRA_BASE + 5] = f32::from(option.was_chosen);
                    row[EXTRA_BASE + 6] = f32::from(option.is_proceed);
                    push(self.index(model_id), row);
                }
            }
            DecisionContext::RestSite { options, left_open } => {
                for (index, option) in options.iter().enumerate() {
                    let mut row = blank();
                    row[EXTRA_BASE + 2] = index as f32 / 8.0;
                    row[EXTRA_BASE + 5] = f32::from(!option.standing);
                    row[EXTRA_BASE + 7] = f32::from(*left_open);
                    push(self.index(&rest_option_id(option.option)), row);
                }
            }
            DecisionContext::MapNavigation { .. }
            | DecisionContext::CrystalSphere { .. }
            | DecisionContext::ActTransition { .. }
            | DecisionContext::CombatPriority { .. }
            | DecisionContext::RoomProceed
            | DecisionContext::Terminal { .. } => {}
        }
    }

    /// One legal action as tensors. `observation` supplies the creature slot
    /// order that action targets point into, and the screen an action names a
    /// place on — a map point, a shop shelf, an event page — so that an
    /// action carries what the player reads off the thing they are clicking.
    ///
    /// A caller with a whole decision's action list in hand should hand it to
    /// [`Self::encode_actions`] instead, which pays the map's route
    /// lookahead once for the screen rather than once for every step on it.
    #[must_use]
    pub fn encode_action(&self, observation: &AgentObservation, action: &Action) -> ActionEncoding {
        self.encode_action_with(observation, action, &mut None)
    }

    /// Every action of one decision as tensors, in the order given.
    ///
    /// The entry a decision goes through. A map step's route lookahead is a
    /// walk of the act's whole map (`RouteTable`), and a map screen offers
    /// as many as four steps onto that one map; encoded together the walk is
    /// paid once for the decision instead of once for each step, and a
    /// decision offering no map step never walks anything.
    #[must_use]
    pub fn encode_actions(
        &self,
        observation: &AgentObservation,
        actions: &[Action],
    ) -> Vec<ActionEncoding> {
        let mut routes = None;
        actions
            .iter()
            .map(|action| self.encode_action_with(observation, action, &mut routes))
            .collect()
    }

    /// One plan as tensors: its [`ActionPlan::outcome`] block, plus what it
    /// gives up where it gives anything up.
    #[must_use]
    pub fn encode_plan(&self, observation: &AgentObservation, plan: &ActionPlan) -> ActionEncoding {
        self.encode_plan_with(observation, plan, &mut None)
    }

    /// Every plan of one decision as tensors, in the order given — the entry
    /// a run-policy decision goes through, sharing one route table exactly as
    /// [`Self::encode_actions`] does.
    #[must_use]
    pub fn encode_plans(
        &self,
        observation: &AgentObservation,
        plans: &[ActionPlan],
    ) -> Vec<ActionEncoding> {
        let mut routes = None;
        plans
            .iter()
            .map(|plan| self.encode_plan_with(observation, plan, &mut routes))
            .collect()
    }

    /// A plan carries its [`ActionPlan::outcome`]'s own block unchanged —
    /// the card it picks, the offer it takes — and a trade writes what it
    /// gives up beside it, so a trade says what it takes and what it costs in
    /// one row.
    #[allow(clippy::cast_precision_loss, reason = "a belt slot index is tiny")]
    fn encode_plan_with(
        &self,
        observation: &AgentObservation,
        plan: &ActionPlan,
        routes: &mut Option<RouteTable>,
    ) -> ActionEncoding {
        let mut encoding = self.encode_action_with(observation, plan.outcome(), routes);
        let Some(free) = plan.surrender() else {
            return encoding;
        };
        let (drunk, slot, model_id) = match free {
            Action::DiscardPotion { slot, model_id } => (false, *slot, model_id),
            Action::UsePotion { slot, model_id, .. } => (true, *slot, model_id),
            // Nothing else frees a belt slot, and the enumeration pairs
            // nothing else; such a plan encodes as the bare acquisition
            // rather than as a half-described trade.
            _ => return encoding,
        };
        encoding.features[ACTION_PLAN_BASE] = 1.0;
        encoding.features[ACTION_PLAN_BASE + 1] = f32::from(!drunk);
        encoding.features[ACTION_PLAN_BASE + 2] = f32::from(drunk);
        // Normalized by the belt the player actually has, never a constant:
        // it opens at three, is two under Tight Belt, and grows with no fixed
        // bound (`RELIC.POTION_BELT`, `RELIC.PHIAL_HOLSTER`,
        // `RELIC.ALCHEMICAL_COFFER`).
        encoding.features[ACTION_PLAN_BASE + 3] =
            slot as f32 / observation.potions.len().max(1) as f32;
        encoding.tokens[ACTION_SURRENDERED_TOKEN] = self.index(model_id);
        encoding
    }

    /// One action, against a route table the caller carries across the
    /// decision. `None` until a map step asks for it.
    #[allow(
        clippy::too_many_lines,
        reason = "one block per action variant, in variant order"
    )]
    fn encode_action_with(
        &self,
        observation: &AgentObservation,
        action: &Action,
        routes: &mut Option<RouteTable>,
    ) -> ActionEncoding {
        let mut tokens = vec![0_u32; ACTION_TOKENS];
        let mut features = vec![0.0_f32; ACTION_FEATURES];
        let family = action.family();
        if let Some(position) = ACTION_FAMILY_NAMES.iter().position(|name| *name == family) {
            features[position] = 1.0;
        }
        let set_target = |target: Option<&TargetHandle>, features: &mut Vec<f32>| match target
            .and_then(|target| {
                observation
                    .creatures
                    .iter()
                    .position(|creature| creature.combat_id == target.combat_id)
            }) {
            Some(slot) if slot < 8 => features[ACTION_TARGET_BASE + slot] = 1.0,
            _ => features[ACTION_TARGET_BASE + 8] = 1.0,
        };
        #[allow(
            clippy::cast_precision_loss,
            reason = "counts, prices and indices are small"
        )]
        match action {
            Action::PlayCard { card, target } => {
                tokens[0] = self.index(&card.fingerprint.model_id);
                if let Some(enchantment) = &card.fingerprint.enchantment {
                    tokens[1] = self.index(&enchantment.model_id);
                }
                write_action_face(&mut features, &card.face);
                self.write_action_statics(
                    &mut features,
                    &card.fingerprint.model_id,
                    card.fingerprint.upgrade_level,
                );
                features[ACTION_CARD_BASE] = f32::from(card.fingerprint.upgrade_level);
                features[ACTION_CARD_BASE + 1] = 1.0 / 4.0;
                features[ACTION_CARD_BASE + 2] = 1.0;
                set_target(target.as_ref(), &mut features);
                if let Some(preview) = observation
                    .hand_previews
                    .iter()
                    .find(|preview| preview.card_id == card.card_id)
                {
                    write_action_preview(observation, &mut features, preview, target.as_ref());
                }
                write_flip(observation, &mut features, target.as_ref());
            }
            Action::EndTurn { .. } => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                features[ACTION_CARD_BASE + 3] = 1.0;
            }
            Action::ChooseCards { cards, .. } => {
                let mut indices: Vec<u32> = cards
                    .iter()
                    .map(|handle| self.index(&handle.fingerprint.model_id))
                    .collect();
                indices.sort_unstable();
                for (slot, index) in indices.iter().take(ACTION_NAMING_TOKENS).enumerate() {
                    tokens[slot] = *index;
                }
                if let Some(first) = cards.first() {
                    features[ACTION_CARD_BASE] = f32::from(first.fingerprint.upgrade_level);
                    write_action_face(&mut features, &first.face);
                    self.write_action_statics(
                        &mut features,
                        &first.fingerprint.model_id,
                        first.fingerprint.upgrade_level,
                    );
                }
                features[ACTION_CARD_BASE + 1] = cards.len() as f32 / 4.0;
                features[ACTION_TARGET_BASE + 8] = 1.0;
            }
            Action::UsePotion {
                model_id, target, ..
            } => {
                tokens[0] = self.index(model_id);
                set_target(target.as_ref(), &mut features);
                write_flip(observation, &mut features, target.as_ref());
            }
            Action::DiscardPotion { slot, model_id } => {
                tokens[0] = self.index(model_id);
                features[ACTION_TARGET_BASE + 8] = 1.0;
                features[ACTION_INDEX_BASE] = *slot as f32 / 8.0;
                features[ACTION_KIND_BASE + 5] = 1.0;
            }
            // A map step carries the point it lands on, whole, and the
            // branch that point opens. The branch is what tells two steps
            // apart when the points themselves look alike, so it rides on
            // the action rather than being left to the map tokens every step
            // on the screen shares.
            Action::ChooseMap { destination } => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                let point = observation
                    .map
                    .as_ref()
                    .and_then(|map| map.points.iter().find(|point| point.coord == *destination));
                let map = observation.map.as_ref();
                features[ACTION_MAP_BASE] =
                    f32::from(destination.col) / f32::from(map.map_or(1, |map| map.width.max(1)));
                features[ACTION_MAP_BASE + 1] =
                    f32::from(destination.row) / f32::from(map.map_or(1, |map| map.height.max(1)));
                if let Some(point) = point {
                    features[ACTION_MAP_BASE + 2] = point.children.len() as f32 / 4.0;
                    features[ACTION_MAP_BASE + 3] = f32::from(point.visited);
                    features[ACTION_MAP_TYPE_BASE + map_type_slot(point.point_type)] = 1.0;
                    let fallback = map_point_id(point.point_type);
                    tokens[0] = match point.point_type {
                        MapPointType::Boss => map.map_or_else(
                            || self.index(&fallback),
                            |map| self.index_or(&map.boss, &fallback),
                        ),
                        MapPointType::Ancient => map.map_or_else(
                            || self.index(&fallback),
                            |map| self.index_or(&map.ancient, &fallback),
                        ),
                        _ => self.index(&fallback),
                    };
                    if let Some(route) = routes
                        .get_or_insert_with(|| RouteTable::of(map))
                        .at(*destination)
                    {
                        write_route(&mut features, route);
                    }
                } else {
                    // Nothing is known about where this step lands, which is
                    // a different sight from an action that names no map
                    // point — and the type one-hot above cannot say which,
                    // because both leave it empty.
                    features[ACTION_MAP_BASE + 6] = 1.0;
                }
                features[ACTION_MAP_BASE + 4] = observation.map_coord.map_or(0.0, |here| {
                    f32::from(destination.row.saturating_sub(here.row)) / 8.0
                });
                features[ACTION_MAP_BASE + 5] =
                    f32::from(map.is_some_and(|map| map.boss_coord == *destination));
            }
            Action::ChooseEvent { index, option_id } => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                features[ACTION_INDEX_BASE] = *index as f32 / 8.0;
                if let DecisionContext::Event { model_id, options } = &observation.decision {
                    tokens[0] = self.index(model_id);
                    // An event may lay out two options that share an id, so
                    // the place on the page is what names one — and the
                    // option's own shape is what the player reads off it.
                    if let Some(option) = options
                        .get(*index)
                        .filter(|option| option.option_id == *option_id)
                        .or_else(|| options.iter().find(|option| option.option_id == *option_id))
                    {
                        features[ACTION_PRICE_BASE + 3] = option.effects.len() as f32 / 4.0;
                        features[ACTION_KIND_BASE + 4] = f32::from(option.was_chosen);
                        features[ACTION_KIND_BASE + 5] = f32::from(option.is_proceed);
                        // What taking it does. Until v8 the three numbers above
                        // were the whole of an option, and a count of effects
                        // is the same count whether the one effect hands over a
                        // relic or takes unblockable damage.
                        let body = EventBody::of(&option.effects, observation);
                        body.write(&mut features);
                        // The naming slots past the event's own: which relic,
                        // which card, which fight. Two options that obtain a
                        // relic each are the same action until these are read.
                        for (slot, named) in body.names.iter().take(ACTION_TOKENS - 2).enumerate() {
                            tokens[slot + 1] = self.index(named);
                        }
                        if let Some(given) = &body.surrendered {
                            tokens[ACTION_SURRENDERED_TOKEN] = self.index(given);
                        }
                    }
                }
            }
            Action::BuyShopItem {
                offer_index,
                fingerprint,
            } => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                features[ACTION_INDEX_BASE] = *offer_index as f32 / 8.0;
                match fingerprint {
                    ShopItem::Card(card) => {
                        tokens[0] = self.index(&card.model_id);
                        if let Some(enchantment) = &card.enchantment {
                            tokens[1] = self.index(&enchantment.model_id);
                        }
                        features[ACTION_CARD_BASE] = f32::from(card.upgrade_level);
                        features[ACTION_KIND_BASE] = 1.0;
                        self.write_action_statics(
                            &mut features,
                            &card.model_id,
                            card.upgrade_level,
                        );
                    }
                    ShopItem::Potion(id) => {
                        tokens[0] = self.index(id);
                        features[ACTION_KIND_BASE + 2] = 1.0;
                    }
                    ShopItem::Relic(id) => {
                        tokens[0] = self.index(id);
                        features[ACTION_KIND_BASE + 1] = 1.0;
                    }
                }
                if let DecisionContext::Shop { offers, .. } = &observation.decision
                    && let Some(offer) = offers.get(*offer_index)
                {
                    features[ACTION_PRICE_BASE] = offer.price as f32 / 100.0;
                    features[ACTION_PRICE_BASE + 1] =
                        f32::from(observation.gold.unwrap_or(0) >= offer.price);
                    features[ACTION_PRICE_BASE + 2] = offer.base_price as f32 / 100.0;
                }
            }
            Action::BuyCardRemoval => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                tokens[0] = self.index(&reward_line_id("card_removal"));
                if let DecisionContext::Shop {
                    removal_price,
                    removal_used,
                    ..
                } = &observation.decision
                {
                    features[ACTION_PRICE_BASE] = *removal_price as f32 / 100.0;
                    features[ACTION_PRICE_BASE + 1] =
                        f32::from(observation.gold.unwrap_or(0) >= *removal_price);
                    features[ACTION_KIND_BASE + 4] = f32::from(*removal_used);
                }
            }
            Action::RestOption { index, option } => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                features[ACTION_INDEX_BASE] = *index as f32 / 8.0;
                tokens[0] = self.index(&rest_option_id(*option));
            }
            Action::TakeTreasure { relic } => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                tokens[0] = self.index(relic);
                features[ACTION_KIND_BASE + 1] = 1.0;
            }
            Action::ClaimReward {
                reward_index,
                fingerprint,
                ..
            } => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                features[ACTION_INDEX_BASE] = *reward_index as f32 / 8.0;
                tokens[0] = fingerprint.model_id.as_ref().map_or_else(
                    || self.index(&reward_line_id(&fingerprint.reward_type)),
                    |id| self.index(id),
                );
                if let Some(card) = fingerprint
                    .offered_cards
                    .first()
                    .or(fingerprint.special_card.as_ref())
                {
                    tokens[1] = self.index(&card.model_id);
                    features[ACTION_CARD_BASE] = f32::from(card.upgrade_level);
                    self.write_action_statics(&mut features, &card.model_id, card.upgrade_level);
                }
                features[ACTION_PRICE_BASE + 2] = fingerprint.gold_amount as f32 / 100.0;
                features[ACTION_PRICE_BASE + 3] = fingerprint.option_count as f32 / 4.0;
                features[ACTION_KIND_BASE] = f32::from(!fingerprint.offered_cards.is_empty());
                features[ACTION_KIND_BASE + 3] = f32::from(fingerprint.gold_amount > 0);
                features[ACTION_KIND_BASE + 5] = f32::from(fingerprint.can_reroll);
            }
            Action::UncoverCrystalSphere { x, y, tool } => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                features[ACTION_MAP_BASE] = *x as f32 / 8.0;
                features[ACTION_MAP_BASE + 1] = *y as f32 / 8.0;
                features[ACTION_KIND_BASE + 5] =
                    f32::from(*tool == sts2_engine::DivinationTool::Big);
            }
            // The buttons standing beside a card reward's own cards: the
            // reward's reroll, and whatever a relic put there. What separates
            // one from the cards it competes against — and from the button
            // beside it — is the relic behind it, its place on the page, and
            // how much of the screen it declines.
            Action::ChooseAlternative { option_id, .. } => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                features[ACTION_INDEX_BASE + 1] = option_slot(option_id);
                if let DecisionContext::ChooseCards {
                    alternatives,
                    candidates,
                    cancelable,
                    ..
                } = &observation.decision
                {
                    if let Some((index, alternative)) = alternatives
                        .iter()
                        .enumerate()
                        .find(|(_, alternative)| alternative.option_id == *option_id)
                    {
                        features[ACTION_INDEX_BASE] = index as f32 / 8.0;
                        // A relic's button and the reward's own reroll are
                        // told apart by the flag rather than by the token
                        // alone: a relic outside the vocabulary indexes zero,
                        // which is the padding row a rerolled screen writes.
                        if let Some(relic) = &alternative.relic {
                            tokens[0] = self.index(relic);
                            features[ACTION_KIND_BASE + 1] = 1.0;
                        }
                    }
                    features[ACTION_CARD_BASE + 1] = candidates.len() as f32 / 4.0;
                    features[ACTION_PRICE_BASE + 3] = alternatives.len() as f32 / 4.0;
                    features[ACTION_KIND_BASE + 4] = f32::from(*cancelable);
                }
            }
            // The exit from a reward screen names what it leaves behind:
            // each line still standing, by its own model where it has one
            // and by its kind where it does not, and on the plan block how
            // many stand and whether a relic is among them. On any other
            // screen, and on a reward screen with nothing left, it is the
            // bare exit it always was.
            Action::AdvanceAct | Action::Proceed => {
                features[ACTION_TARGET_BASE + 8] = 1.0;
                if let DecisionContext::Rewards { offers, .. } = &observation.decision {
                    let standing: Vec<_> = offers
                        .iter()
                        .filter(|offer| !offer.selected && !offer.passed_over)
                        .collect();
                    for (slot, offer) in standing.iter().take(ACTION_NAMING_TOKENS).enumerate() {
                        tokens[slot] = offer.model_id.as_ref().map_or_else(
                            || self.index(&reward_line_id(&offer.reward_type)),
                            |id| self.index(id),
                        );
                    }
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "a screen holds a handful of lines"
                    )]
                    {
                        features[ACTION_PLAN_BASE + 4] = standing.len() as f32 / 4.0;
                    }
                    features[ACTION_PLAN_BASE + 5] =
                        f32::from(standing.iter().any(|offer| offer.reward_type == "relic"));
                }
            }
        }
        ActionEncoding { tokens, features }
    }
}

/// The card face, on a token row. The two `FACE_PREVIEW_BASE` slots are
/// written beside this, from the hand's own previews, because a card carries
/// a face everywhere and a preview only in hand.
fn write_face(row: &mut [f32; TOKEN_FEATURES], face: &sts2_engine::CardFace) {
    #[allow(clippy::cast_precision_loss, reason = "costs and marks are small")]
    {
        // The four displayed components stay apart: "cheaper until played"
        // and "cheaper this combat" are not the same card.
        row[FACE_BASE] = face.energy_cost as f32 / 3.0;
        row[FACE_BASE + 1] = face.cost_this_turn as f32 / 3.0;
        row[FACE_BASE + 2] = face.cost_this_combat as f32 / 3.0;
        row[FACE_BASE + 3] = face.cost_until_played as f32 / 3.0;
        row[FACE_BASE + 5] = face.replay_count as f32;
        row[FACE_BASE + 9] = face.damage_bonus as f32 / 10.0;
        row[FACE_BASE + 10] = face.star_cost as f32 / 3.0;
        row[FACE_BASE + 11] = face.enchantment_damage_bonus as f32 / 10.0;
    }
    row[FACE_BASE + 4] = f32::from(face.costs_x);
    row[FACE_BASE + 6] = f32::from(face.ethereal);
    row[FACE_BASE + 7] = f32::from(face.retain);
    row[FACE_BASE + 8] = f32::from(face.sly);
    row[FACE_BASE + 12] =
        f32::from(face.enchantment_status == sts2_engine::EnchantmentStatus::Disabled);
}

/// What the play resolves to against the creature it names, on an action's
/// feature vector.
///
/// The number is the engine's own preview pass, never re-derived: a targeted
/// card reads the [`sts2_engine::AimedDamage`] entry for the creature it aims
/// at, and one that lands on all or a random one of them reads the shared
/// multi-target number. A card whose face carries no damage — and every
/// action outside a combat — leaves the block empty.
fn write_action_preview(
    observation: &AgentObservation,
    features: &mut [f32],
    preview: &CardPreview,
    target: Option<&TargetHandle>,
) {
    let damage = match target {
        Some(target) => preview
            .aimed_damage
            .iter()
            .find(|aimed| aimed.combat_id == target.combat_id)
            .map(|aimed| aimed.damage),
        None => preview.multi_target_damage,
    };
    let Some(damage) = damage else { return };
    let fraction = match target {
        Some(target) => observation
            .creatures
            .iter()
            .find(|creature| creature.combat_id == target.combat_id)
            .map(|creature| kill_fraction(damage, creature)),
        // The same number lands on every one of them, so what the play is
        // worth is whichever of the enemies still standing it comes closest
        // to killing.
        None => observation
            .creatures
            .iter()
            .filter(|creature| creature.side != CombatSide::Player && creature.current_hp > 0)
            .map(|creature| kill_fraction(damage, creature))
            .fold(None, |best: Option<f32>, share| {
                Some(best.map_or(share, |best| best.max(share)))
            }),
    };
    #[allow(clippy::cast_precision_loss, reason = "displayed damage is small")]
    {
        features[ACTION_PREVIEW_BASE] = damage as f32 / 50.0;
    }
    if let Some(fraction) = fraction {
        // Capped, and the bit says which side of the cap it was on: past
        // lethal, how far past is not a decision.
        features[ACTION_PREVIEW_BASE + 1] = fraction.min(1.0);
        features[ACTION_PREVIEW_BASE + 2] = f32::from(fraction >= 1.0);
    }
}

/// How much of what a creature has left this blow takes: block stands in
/// front of the hit points, and a creature with neither is already dead.
fn kill_fraction(damage: i32, creature: &sts2_engine::VisibleCreature) -> f32 {
    let standing = i64::from(creature.current_hp.max(0)) + i64::from(creature.block.max(0));
    #[allow(
        clippy::cast_precision_loss,
        reason = "damage and hit points are small"
    )]
    {
        damage.max(0) as f32 / standing.max(1) as f32
    }
}

/// Whether aiming this play or this potion turns the player around.
///
/// `POWER.SURROUNDED_POWER`'s own rule, as the simulator's content crate
/// states it: its `BeforeCardPlayed` and
/// `BeforePotionUsed` hooks fire on `OwnerAimedAtSomebody`, and each of the
/// two `turn_to_face` arms sets the counter to the other facing when the
/// counter is its own and the target carries the marker for the side behind
/// the player — counter zero (facing right) reads
/// `POWER.BACK_ATTACK_LEFT_POWER`, counter one reads
/// `POWER.BACK_ATTACK_RIGHT_POWER`. The counter is the facing; the amount is
/// not, and reading the amount here would answer for a power that is simply
/// on.
fn write_flip(observation: &AgentObservation, features: &mut [f32], target: Option<&TargetHandle>) {
    let Some(target) = target else { return };
    let Some(counter) = surrounded_counter(observation) else {
        return;
    };
    let marker = match counter {
        0 => "BACK_ATTACK_LEFT_POWER",
        1 => "BACK_ATTACK_RIGHT_POWER",
        _ => return,
    };
    let turns = observation
        .creatures
        .iter()
        .find(|creature| creature.combat_id == target.combat_id)
        .is_some_and(|creature| {
            creature
                .powers
                .iter()
                .any(|power| is_power(&power.model_id, marker))
        });
    features[ACTION_FLIP] = f32::from(turns);
}

/// The facing the player is standing in, or nothing where they carry no
/// Surrounded. The power is the Kaiser Crab fight's own debuff and it lands
/// on the player, so the player's side of the board is where it is read
/// from.
fn surrounded_counter(observation: &AgentObservation) -> Option<i32> {
    observation
        .creatures
        .iter()
        .filter(|creature| creature.side == CombatSide::Player)
        .find_map(|creature| {
            creature
                .powers
                .iter()
                .find(|power| is_power(&power.model_id, "SURROUNDED_POWER"))
                .map(|power| power.counter)
        })
}

/// Whether a model id names this power, without building one to compare
/// against — an action encode is the search's inner loop.
fn is_power(model_id: &ModelId, entry: &str) -> bool {
    model_id.category() == "POWER" && model_id.entry() == entry
}

/// The card face, on an action's feature vector.
fn write_action_face(features: &mut [f32], face: &sts2_engine::CardFace) {
    #[allow(clippy::cast_precision_loss, reason = "costs and marks are small")]
    {
        features[ACTION_CARD_BASE + 4] = face.energy_cost as f32 / 3.0;
        features[ACTION_CARD_BASE + 5] =
            (face.cost_this_turn + face.cost_this_combat + face.cost_until_played) as f32 / 3.0;
        features[ACTION_CARD_BASE + 6] = face.replay_count as f32;
        features[ACTION_CARD_BASE + 7] = face.star_cost as f32 / 3.0;
        features[ACTION_CARD_BASE + 8] =
            (face.damage_bonus + face.enchantment_damage_bonus) as f32 / 10.0;
    }
    features[ACTION_CARD_BASE + 9] = f32::from(face.costs_x);
}

/// A handled pile as (print, face, preview) triples. Only the hand carries a
/// preview — the build runs the hook passes for no other pile — so every pile
/// that goes through here reads the empty pair.
fn handle_pairs(
    handles: &[sts2_engine::CardHandle],
) -> Vec<(
    &CardFingerprint,
    &sts2_engine::CardFace,
    Option<&CardPreview>,
)> {
    handles
        .iter()
        .map(|handle| (&handle.fingerprint, &handle.face, None))
        .collect()
}

/// The hand's previews by card id.
fn preview_index(observation: &AgentObservation) -> BTreeMap<CardInstanceId, &CardPreview> {
    observation
        .hand_previews
        .iter()
        .map(|preview| (preview.card_id, preview))
        .collect()
}

/// Which slot an intent type occupies, over the closed enum. `None` for a
/// name outside it, which the pinned content never produces — the encoding
/// simply leaves the block empty rather than folding an unknown kind onto
/// some other kind's slot, which is what hashing did.
fn intent_type_slot(intent_type: &str) -> Option<usize> {
    INTENT_TYPE_NAMES
        .iter()
        .position(|name| *name == intent_type)
}

/// Which slot a decision kind occupies, over the closed context set. The
/// internally-tagged serialization names the variant in `kind`, so this reads
/// the tag rather than hashing the first key of the serialized map — which is
/// what put nine of the twelve variants in one bucket.
fn decision_kind(decision: &DecisionContext) -> usize {
    let name = match decision {
        DecisionContext::MapNavigation { .. } => "map_navigation",
        DecisionContext::Event { .. } => "event",
        DecisionContext::Shop { .. } => "shop",
        DecisionContext::RestSite { .. } => "rest_site",
        DecisionContext::Treasure { .. } => "treasure",
        DecisionContext::CrystalSphere { .. } => "crystal_sphere",
        DecisionContext::ActTransition { .. } => "act_transition",
        DecisionContext::CombatPriority { .. } => "combat_priority",
        DecisionContext::ChooseCards { .. } => "choose_cards",
        DecisionContext::Rewards { .. } => "rewards",
        DecisionContext::RoomProceed => "room_proceed",
        DecisionContext::Terminal { .. } => "terminal",
    };
    DECISION_KIND_NAMES
        .iter()
        .position(|entry| *entry == name)
        .expect("every decision kind has a slot")
}

/// How many things the standing screen is offering, for the scalar.
fn offer_count(observation: &AgentObservation) -> usize {
    match &observation.decision {
        DecisionContext::ChooseCards { candidates, .. } => candidates.len(),
        DecisionContext::Rewards { offers, .. } => offers.len(),
        DecisionContext::Shop { offers, .. } => offers.len(),
        DecisionContext::Treasure { relics } => relics.len(),
        DecisionContext::Event { options, .. } => options.len(),
        DecisionContext::RestSite { options, .. } => options.len(),
        DecisionContext::MapNavigation { destinations, .. } => destinations.len(),
        DecisionContext::CrystalSphere { .. }
        | DecisionContext::ActTransition { .. }
        | DecisionContext::CombatPriority { .. }
        | DecisionContext::RoomProceed
        | DecisionContext::Terminal { .. } => 0,
    }
}

/// The synthetic id a reward line of this kind is named by.
fn reward_line_id(reward_type: &str) -> ModelId {
    let entry = match reward_type {
        "card" => "CARD",
        "card_removal" => "CARD_REMOVAL",
        "gold" => "GOLD",
        "potion" => "POTION",
        "relic" => "RELIC",
        "special_card" => "SPECIAL_CARD",
        _ => "OTHER",
    };
    synthetic("REWARD", entry)
}

/// The synthetic id a map point of this type is named by, which is also the
/// fallback for a boss or ancient the vocabulary does not carry.
fn map_point_id(point_type: MapPointType) -> ModelId {
    synthetic(
        "MAP",
        &MAP_TYPE_NAMES[map_type_slot(point_type)].to_uppercase(),
    )
}

/// The synthetic id a rest-site option is named by.
fn rest_option_id(option: RestSiteOption) -> ModelId {
    synthetic("REST", option.option_id())
}

/// A stable small number for an alternative's option id. The alternatives a
/// relic puts on a card-reward screen are a handful of fixed strings, and
/// nothing reads this as an identity — it separates two buttons on one
/// screen, which is all the action list needs.
fn option_slot(option_id: &str) -> f32 {
    #[allow(clippy::cast_sign_loss, reason = "bucketed modulo a small constant")]
    let bucket = (sts2_rng::deterministic_hash(option_id).unsigned_abs() as usize) % 8;
    #[allow(clippy::cast_precision_loss, reason = "buckets are single digits")]
    {
        bucket as f32 / 8.0
    }
}

/// The event effects an option's body is made of, in the order their multi-hot
/// occupies `ACTION_EVENT_BASE`. `EventEffect`'s full range, one slot a
/// variant, in the order the enum declares them.
///
/// One slot a variant rather than a grouping into families — "you get a relic",
/// "it costs health" — on purpose. The enum is closed, so the full range costs
/// fifty slots and carries no judgement; a family table would be somebody's
/// opinion about which effects play alike, baked in where the net cannot
/// disagree with it. What two variants have in common is for the net to find.
pub const EVENT_EFFECT_NAMES: [&str; EVENT_EFFECTS] = [
    "gain_gold",
    "lose_hp",
    "obtain_relic",
    "obtain_relic_with_card",
    "obtain_sea_glass",
    "obtain_rolled_relic",
    "choose_generated_cards",
    "choose_created_cards",
    "heal",
    "gain_max_hp",
    "lose_max_hp",
    "lose_max_hp_and_upgrade",
    "trade_potion_for_upgraded_cards",
    "lose_gold",
    "remove_deck_cards_for_gold",
    "remove_rolled_tradable_relic",
    "remove_relic",
    "discard_rolled_potion",
    "discard_potion",
    "set_page",
    "set_shuffled_page",
    "set_rolled_page",
    "run",
    "add_card",
    "add_generated_card",
    "obtain_relic_and_random_card",
    "obtain_relic_rolled_from",
    "add_card_rolled_from",
    "obtain_relic_and_generated_rewards",
    "rewards",
    "remove_the_rolled_card",
    "grab_off_the_belt",
    "upgrade_a_rolled_card",
    "unless_the_fight_timed_out",
    "upgrade_shuffled_deck_cards",
    "trade_relic",
    "buy_from_the_bag",
    "downgrade_a_rolled_card",
    "play_the_crystal_sphere",
    "hold_on_to_the_bridge",
    "immerse_in_the_baths",
    "offer_shuffled_relics",
    "offer_rolled_potion",
    "offer_card_rewards",
    "offer_created_card_rewards",
    "enter_combat_without_leaving",
    "read_the_event_stream",
    "unsupported",
    "proceed",
    "win_run",
];
const EVENT_EFFECTS: usize = 50;

/// Where an effect's slot stands in [`EVENT_EFFECT_NAMES`].
///
/// Exhaustive with no catch-all, so an effect added to the engine stops this
/// crate from compiling until it has a slot and a name. A wildcard would leave
/// the new effect silently encoded as whatever it fell through to.
fn event_effect_slot(effect: &sts2_engine::EventEffect) -> usize {
    use sts2_engine::EventEffect as E;
    match effect {
        E::GainGold(_) => 0,
        E::LoseHp(_) => 1,
        E::ObtainRelic(_) => 2,
        E::ObtainRelicWithCard { .. } => 3,
        E::ObtainSeaGlass { .. } => 4,
        E::ObtainRolledRelic => 5,
        E::ChooseGeneratedCards { .. } => 6,
        E::ChooseCreatedCards { .. } => 7,
        E::Heal(_) => 8,
        E::GainMaxHp(_) => 9,
        E::LoseMaxHp(_) => 10,
        E::LoseMaxHpAndUpgrade { .. } => 11,
        E::TradePotionForUpgradedCards { .. } => 12,
        E::LoseGold(_) => 13,
        E::RemoveDeckCardsForGold { .. } => 14,
        E::RemoveRolledTradableRelic => 15,
        E::RemoveRelic(_) => 16,
        E::DiscardRolledPotion => 17,
        E::DiscardPotion { .. } => 18,
        E::SetPage(_) => 19,
        E::SetShuffledPage { .. } => 20,
        E::SetRolledPage(_) => 21,
        E::Run(_) => 22,
        E::AddCard(_) => 23,
        E::AddGeneratedCard { .. } => 24,
        E::ObtainRelicAndRandomCard { .. } => 25,
        E::ObtainRelicRolledFrom { .. } => 26,
        E::AddCardRolledFrom { .. } => 27,
        E::ObtainRelicAndGeneratedRewards { .. } => 28,
        E::Rewards(_) => 29,
        E::RemoveTheRolledCard { .. } => 30,
        E::GrabOffTheBelt { .. } => 31,
        E::UpgradeARolledCard => 32,
        E::UnlessTheFightTimedOut(_) => 33,
        E::UpgradeShuffledDeckCards { .. } => 34,
        E::TradeRelic { .. } => 35,
        E::BuyFromTheBag { .. } => 36,
        E::DowngradeARolledCard => 37,
        E::PlayTheCrystalSphere { .. } => 38,
        E::HoldOnToTheBridge { .. } => 39,
        E::ImmerseInTheBaths { .. } => 40,
        E::OfferShuffledRelics { .. } => 41,
        E::OfferRolledPotion { .. } => 42,
        E::OfferCardRewards { .. } => 43,
        E::OfferCreatedCardRewards { .. } => 44,
        E::EnterCombatWithoutLeaving { .. } => 45,
        E::ReadTheEventStream(_) => 46,
        E::Unsupported(_) => 47,
        E::Proceed => 48,
        E::WinRun => 49,
    }
}

/// What an event option's body comes to: which effects it is made of, and the
/// resources those effects move.
///
/// Built by walking the option's effects, and nested bodies with them — an
/// effect guarded by how a fight ended, or run after one, is still what taking
/// the option does.
struct EventBody {
    kinds: [f32; EVENT_EFFECTS],
    hp: i32,
    max_hp: i32,
    gold: i32,
    cards_gained: i32,
    cards_removed: i32,
    cards_upgraded: i32,
    relics_gained: i32,
    relics_lost: i32,
    potions_lost: i32,
    /// An amount the page settled and the option only points at, which nothing
    /// here can read. Raised so that a cost this cannot price is not read off
    /// the same zeros as no cost at all.
    unread: bool,
    /// What the option is about, by name, in the order the body names them:
    /// the relic obtained, the card added, the fight walked into. Filled into
    /// the action's free naming slots, which is what they are for.
    ///
    /// A roll over candidates names nothing here. `ObtainRolledRelic` and
    /// `AddCardRolledFrom` settle after the option is taken, and a player
    /// reading the page cannot see which one either — the effect's own slot
    /// says a relic is coming, and inventing which would be clairvoyance.
    names: Vec<ModelId>,
    /// What the option gives up, by name. One, because a page that takes two
    /// things takes them as one price, and because the slot it lands in is one
    /// slot: see [`ACTION_SURRENDERED_TOKEN`].
    surrendered: Option<ModelId>,
}

impl Default for EventBody {
    fn default() -> Self {
        Self {
            kinds: [0.0; EVENT_EFFECTS],
            hp: 0,
            max_hp: 0,
            gold: 0,
            cards_gained: 0,
            cards_removed: 0,
            cards_upgraded: 0,
            relics_gained: 0,
            relics_lost: 0,
            potions_lost: 0,
            unread: false,
            names: Vec::new(),
            surrendered: None,
        }
    }
}

impl EventBody {
    fn of(effects: &[sts2_engine::EventEffect], observation: &AgentObservation) -> Self {
        let mut body = Self::default();
        body.walk(effects, observation);
        body
    }

    fn walk(&mut self, effects: &[sts2_engine::EventEffect], observation: &AgentObservation) {
        for effect in effects {
            self.add(effect, observation);
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "deck and pick counts are small"
    )]
    fn add(&mut self, effect: &sts2_engine::EventEffect, observation: &AgentObservation) {
        use sts2_engine::EventEffect as E;
        self.kinds[event_effect_slot(effect)] = 1.0;
        self.name(effect, observation);
        // Only what the effect itself certainly moves. An effect that puts a
        // screen up rather than paying something out — the `Offer` family, a
        // crystal sphere, a page swap — moves nothing here and is carried by
        // its slot above; what the player then picks off that screen is that
        // screen's own decision.
        match effect {
            E::GainGold(amount) => self.gold += self.amount(*amount, observation),
            E::LoseGold(amount) => self.gold -= self.amount(*amount, observation),
            E::Heal(amount) => self.hp += self.amount(*amount, observation),
            E::LoseHp(amount) => self.hp -= self.amount(*amount, observation),
            // Raising the maximum heals by as much.
            E::GainMaxHp(amount) => {
                self.max_hp += amount;
                self.hp += amount;
            }
            // The maximum comes down and takes the current hit points with it
            // only where they no longer fit under it.
            E::LoseMaxHp(amount) => {
                let amount = self.amount(*amount, observation);
                self.max_hp -= amount;
                let max_hp = observation.max_hp.unwrap_or(0);
                let current = observation.current_hp.unwrap_or(max_hp);
                self.hp += (max_hp - amount - current).min(0);
            }
            E::LoseMaxHpAndUpgrade { amount, whole_deck } => {
                self.max_hp -= self.amount(*amount, observation);
                self.cards_upgraded += if *whole_deck {
                    observation.deck.len() as i32
                } else {
                    1
                };
            }
            E::HoldOnToTheBridge { cost } => self.hp -= cost,
            E::ImmerseInTheBaths { damage } => {
                self.max_hp += 2;
                self.hp += 2 - damage;
            }
            E::BuyFromTheBag { cost, .. } => {
                self.gold -= cost;
                self.relics_gained += 1;
            }
            E::RemoveDeckCardsForGold { count, cost } => {
                self.gold -= cost;
                self.cards_removed += *count as i32;
            }
            E::ObtainRelic(_)
            | E::ObtainRolledRelic
            | E::ObtainRelicRolledFrom { .. }
            | E::ObtainSeaGlass { .. }
            | E::ObtainRelicAndGeneratedRewards { .. } => self.relics_gained += 1,
            E::ObtainRelicWithCard { .. } | E::ObtainRelicAndRandomCard { .. } => {
                self.relics_gained += 1;
                self.cards_gained += 1;
            }
            E::TradeRelic { .. } => {
                self.relics_gained += 1;
                self.relics_lost += 1;
            }
            E::RemoveRelic(_) | E::RemoveRolledTradableRelic => self.relics_lost += 1,
            E::DiscardPotion { .. }
            | E::DiscardRolledPotion
            | E::TradePotionForUpgradedCards { .. } => self.potions_lost += 1,
            E::AddCard(_) | E::AddGeneratedCard { .. } | E::AddCardRolledFrom { .. } => {
                self.cards_gained += 1;
            }
            E::ChooseGeneratedCards { picks, .. } | E::ChooseCreatedCards { picks, .. } => {
                self.cards_gained += *picks as i32;
            }
            E::RemoveTheRolledCard { .. } => self.cards_removed += 1,
            E::UpgradeARolledCard => self.cards_upgraded += 1,
            E::DowngradeARolledCard => self.cards_upgraded -= 1,
            E::UpgradeShuffledDeckCards { count } => self.cards_upgraded += *count as i32,
            // A guarded body and a body run after a fight are both bodies.
            E::UnlessTheFightTimedOut(effects) => self.walk(effects, observation),
            E::EnterCombatWithoutLeaving { resume, .. } => self.walk(resume, observation),
            // Everything else moves nothing this counts.
            _ => {}
        }
    }

    /// What the effect names: the things it is about, and the one thing it
    /// gives up.
    fn name(&mut self, effect: &sts2_engine::EventEffect, observation: &AgentObservation) {
        use sts2_engine::EventEffect as E;
        match effect {
            E::ObtainRelic(relic)
            | E::ObtainSeaGlass { relic, .. }
            | E::ObtainRelicAndRandomCard { relic, .. }
            | E::ObtainRelicAndGeneratedRewards { relic, .. } => self.names.push(relic.clone()),
            E::ObtainRelicWithCard { relic, card } => {
                self.names.push(relic.clone());
                self.names.push(card.clone());
            }
            E::AddCard(card) => self.names.push(card.model_id.clone()),
            E::EnterCombatWithoutLeaving { encounter, .. } => self.names.push(encounter.clone()),
            // The two sides of a trade go to different slots on purpose: a
            // mean over the naming slots is symmetric, so a relic given for a
            // relic taken would pool to the vector of its own reverse.
            E::TradeRelic { given, taken } => {
                self.names.push(taken.clone());
                self.give(given.clone());
            }
            E::RemoveRelic(relic) => self.give(relic.clone()),
            E::RemoveTheRolledCard { card: Some(card) } => self.give(card.model_id.clone()),
            E::DiscardPotion { slot } | E::TradePotionForUpgradedCards { slot, .. } => {
                if let Some(Some(potion)) = observation.potions.get(*slot) {
                    self.give(potion.clone());
                }
            }
            _ => {}
        }
    }

    /// The first thing the body gives up keeps the slot. A body that hands over
    /// two things hands them over as one price, and there is one slot.
    fn give(&mut self, given: ModelId) {
        if self.surrendered.is_none() {
            self.surrendered = Some(given);
        }
    }

    /// An amount as this can read it: what it says, or what the run it is
    /// standing in makes it. A var the page settled reads as nothing and raises
    /// [`EventBody::unread`].
    fn amount(&mut self, amount: sts2_engine::EventAmount, observation: &AgentObservation) -> i32 {
        use sts2_engine::EventAmount as A;
        let max_hp = observation.max_hp.unwrap_or(0);
        match amount {
            A::Fixed(value) => value,
            A::MaxHpLessOne => max_hp - 1,
            A::MaxHpFraction {
                numerator,
                denominator,
            } => {
                if denominator == 0 {
                    self.unread = true;
                    0
                } else {
                    max_hp * numerator / denominator
                }
            }
            // A roll reads as its middle: the page has not rolled yet, and the
            // mean is the only summary of a range that is not a guess about
            // which way it will land.
            A::Rolled { min, max } | A::RolledOnto { base: 0, min, max } => min.midpoint(max),
            A::RolledOnto { base, min, max } => base + min.midpoint(max),
            A::RolledAround { base, .. } => base,
            A::RolledOffOf { base, min, max } => base - min.midpoint(max),
            A::MissingHp => max_hp - observation.current_hp.unwrap_or(max_hp),
            A::Var(_) => {
                self.unread = true;
                0
            }
        }
    }

    /// This body onto an action's own feature block.
    fn write(&self, features: &mut [f32]) {
        features[ACTION_EVENT_BASE..ACTION_EVENT_BASE + EVENT_EFFECTS].copy_from_slice(&self.kinds);
        #[allow(clippy::cast_precision_loss, reason = "event numbers are small")]
        let numbers = [
            self.hp as f32 / 32.0,
            self.max_hp as f32 / 32.0,
            self.gold as f32 / 100.0,
            self.cards_gained as f32 / 4.0,
            self.cards_removed as f32 / 4.0,
            self.cards_upgraded as f32 / 8.0,
            self.relics_gained as f32 / 2.0,
            self.relics_lost as f32 / 2.0,
            self.potions_lost as f32 / 2.0,
            f32::from(self.unread),
        ];
        features[ACTION_EVENT_NUMBER_BASE..ACTION_EVENT_NUMBER_BASE + ACTION_EVENT_NUMBER_SLOTS]
            .copy_from_slice(&numbers);
    }
}
