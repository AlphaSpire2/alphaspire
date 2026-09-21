//! The hand-written out-of-combat policy: every decision the belief search
//! does not own, decided by legible rules instead of a die roll.
//!
//! Scores card rewards, pathing, events, rest sites and shops. It supplies
//! rollout actions and imitation targets for training a macro policy.
//!
//! In combat it defers: standalone it plays uniform random, and under
//! `BeliefSearch` the search owns every in-combat decision anyway. An A/B
//! against `UniformRandom` therefore differs only out of combat.
//!
//! Every knob and table lives in [`knobs`], with scores based on the
//! simulator's registered content.

use sts2_engine::{
    Action, CardFingerprint, CardType, ChoicePurpose, DecisionContext, EventAmount, EventEffect,
    EventOption, MapCoord, MapPointType, RestSiteOption, ShopItem, Simulator,
};
use sts2_rng::MegaRandom;

use crate::env::fight_over;
use crate::policy::{RolloutPolicy, UniformRandom, permitted_actions, poisoned_event_option};

/// Every tunable number the policy reads, in one place.
///
/// Scores are comparable only within one decision screen: the policy takes
/// the argmax over the actions that screen offers, first index winning ties,
/// so what matters is each screen's internal ordering plus the few
/// cross-family baselines (`PROCEED`, the potion penalties) that share a
/// screen with everything else.
pub mod knobs {
    /// Walking on (`Proceed` / `AdvanceAct`): the do-nothing baseline every
    /// claim and purchase must beat.
    pub const PROCEED: f64 = 5.0;

    /// Drinking a potion out of combat. Potions are held for fights, where
    /// the search prices them; the engine only offers the ones with an
    /// out-of-combat body at all, and none of those beats saving the slot at
    /// this baseline's stakes.
    pub const USE_POTION_OOC: f64 = -500.0;
    /// Throwing a potion away. Never: a full belt simply stops pickups.
    pub const DISCARD_POTION: f64 = -800.0;
    /// The reroll button on a card reward (`RELIC.DRIFTWOOD`, once per
    /// offer): three fresh cards for nothing. Worth exactly a card sitting
    /// at the bar the table already calls worth taking, so an offer with
    /// nothing at that bar is redrawn and an offer with something at it is
    /// taken. It is on the same scale as a pick ([`SCREEN_COMMIT`] plus the
    /// card's value), because it is ranked against picks.
    pub const CHOICE_REROLL: f64 = SCREEN_COMMIT + TAKE_TIER_FLOOR;
    /// The sacrifice button on a card reward (`RELIC.PAELS_WING`): the offer
    /// is thrown away and every second one thrown away pays a relic. Half a
    /// relic against a whole card reward is a loss at this baseline's stakes,
    /// so it ranks below every pick.
    pub const CHOICE_SACRIFICE: f64 = -50.0;
    /// An event option whose body already ran (`was_chosen`): clicking it
    /// again does nothing at all, as in the game, so choosing it can loop
    /// the policy in place.
    pub const EVENT_OPTION_SPENT: f64 = -900.0;
    /// A step the engine refuses, or that kills the player outright.
    pub const NEVER: f64 = -10_000.0;

    // --- rest sites -------------------------------------------------------
    /// Rest below this fraction of max HP, else smith. The heal is
    /// `PlayerCmd::MimicRestSiteHeal`; an upgrade compounds over every
    /// remaining fight, so it wins whenever HP is not the binding resource.
    pub const REST_HP_FRACTION: f64 = 0.60;
    pub const REST_HEAL_BASE: f64 = 60.0;
    pub const REST_HEAL_PER_MISSING: f64 = 40.0;
    pub const REST_HEAL_HEALTHY: f64 = 4.0;
    pub const REST_SMITH: f64 = 30.0;
    /// Dig is a free relic out of the bag (`RelicCmd::ObtainFromTheBag`) —
    /// better than an upgrade.
    pub const REST_DIG: f64 = 45.0;
    pub const REST_HATCH: f64 = 20.0;
    pub const REST_LIFT: f64 = 15.0;
    pub const REST_KINDLE: f64 = 12.0;
    pub const REST_CLONE: f64 = 10.0;
    /// Cook: two cards leave the deck and max HP rises nine.
    /// Worth it exactly when the deck holds two cards worth losing.
    pub const REST_COOK_GOOD: f64 = 35.0;
    pub const REST_COOK_BAD: f64 = 2.0;

    // --- pathing ----------------------------------------------------------
    /// Base worth of a rest-site node, plus how hard missing HP pulls
    /// toward it, plus the campfire-before-boss bonus for a rest site whose
    /// children include the boss node.
    pub const MAP_REST_BASE: f64 = 12.0;
    pub const MAP_REST_PER_MISSING: f64 = 30.0;
    pub const MAP_REST_BEFORE_BOSS: f64 = 25.0;
    pub const MAP_TREASURE: f64 = 20.0;
    pub const MAP_ANCIENT: f64 = 15.0;
    /// `?` rooms: mostly events, mildly preferred over one more hallway
    /// fight when healthy (they cost less HP on average).
    pub const MAP_UNKNOWN: f64 = 10.0;
    pub const MAP_UNKNOWN_HEALTHY_BONUS: f64 = 3.0;
    pub const MAP_SHOP: f64 = 6.0;
    /// A shop is worth walking to once there is gold to spend in it.
    pub const MAP_SHOP_RICH_BONUS: f64 = 8.0;
    pub const MAP_SHOP_RICH_GOLD: i32 = 150;
    pub const MAP_MONSTER: f64 = 8.0;
    /// A hallway fight while nearly dead is still a fight.
    pub const MAP_MONSTER_HURT_PENALTY: f64 = 6.0;
    pub const MAP_MONSTER_HURT_FRACTION: f64 = 0.35;
    /// Elites: a relic when strong, a death when weak. "Strong" is HP above
    /// the fraction and a deck that has actually grown (see
    /// `Ctx::deck_strength`).
    pub const MAP_ELITE_STRONG: f64 = 13.0;
    pub const MAP_ELITE_WEAK: f64 = -25.0;
    pub const MAP_ELITE_HP_FRACTION: f64 = 0.65;
    pub const MAP_ELITE_DECK_STRENGTH: usize = 4;
    pub const MAP_BOSS: f64 = 1.0;

    // --- card evaluation --------------------------------------------------
    /// A reward card must clear this tier to join the deck at all...
    pub const TAKE_TIER_FLOOR: f64 = 35.0;
    /// ...and the bar rises per card past this deck size: deck bloat dilutes
    /// every draw of the cards that win fights.
    pub const BLOAT_DECK_SIZE: usize = 18;
    pub const BLOAT_PENALTY_PER_CARD: f64 = 2.0;
    /// Attacks win act 1: the starter deck's damage is five Strikes and a
    /// Bash, and hallway fights are lost to slowness. While fewer than this
    /// many attacks have been added, attack-type rewards get a bump.
    pub const EARLY_ATTACK_COUNT: usize = 3;
    pub const EARLY_ATTACK_BONUS: f64 = 8.0;
    /// A second copy of the same print is worth less than the first;
    /// a power played once a fight especially so.
    pub const DUPLICATE_PENALTY: f64 = 6.0;
    pub const DUPLICATE_POWER_PENALTY: f64 = 14.0;
    /// An upgraded copy is worth this much more per level — mostly so
    /// removal screens keep their hands off upgraded cards.
    pub const UPGRADE_LEVEL_VALUE: f64 = 3.0;
    /// Removing a card pays off when the card is worth less than this pivot
    /// (curses, statuses, and the basic Strikes/Defends sit below it).
    pub const REMOVAL_PIVOT: f64 = 20.0;
    /// Bash is the starter deck's engine — `attack(8)` plus
    /// `POWER.VULNERABLE_POWER 2`, and the upgrade raises both — so it
    /// jumps the smith queue early.
    pub const UPGRADE_BASH_BONUS: f64 = 20.0;

    // --- unknown-card defaults (resolved by card type, cards.rs) ----------
    pub const TIER_UNKNOWN_PLAYABLE: f64 = 25.0;

    // --- derived tiers (`derived_tier`): any playable card the table does
    // not name, scored off its registry definition on the table's scale ---
    /// What a card is worth before its body says anything, by type.
    pub const DERIVED_BASE_ATTACK: f64 = 22.0;
    pub const DERIVED_BASE_SKILL: f64 = 22.0;
    pub const DERIVED_BASE_POWER: f64 = 30.0;
    /// Per point of fixed damage and block in the body, before the cost
    /// divisor.
    pub const DERIVED_PER_DAMAGE: f64 = 1.5;
    pub const DERIVED_PER_BLOCK: f64 = 1.3;
    /// Damage aimed at every enemy counts this many times over.
    pub const DERIVED_AOE_MULTIPLIER: f64 = 1.7;
    /// A power that reads as a debuff where it lands (Vulnerable, Weak) and
    /// one that reads as a buff, for standing at all, plus this much per
    /// stack past the first. Which way a power reads is the registry's
    /// answer for the amount applied, not the side it lands on.
    pub const DERIVED_PER_ENEMY_DEBUFF: f64 = 10.0;
    pub const DERIVED_PER_SELF_BUFF: f64 = 3.0;
    pub const DERIVED_PER_ENEMY_DEBUFF_STACK: f64 = 1.5;
    pub const DERIVED_PER_SELF_BUFF_STACK: f64 = 1.5;
    /// The most stacks one application is priced for. A stack count the board
    /// settles rather than the body states reads as one, and a body asking
    /// for more than a few is counting in a unit of its own, the way a
    /// Cruelty's 25 is a percentage rather than twenty-five stacks.
    pub const DERIVED_MAX_PRICED_STACKS: f64 = 4.0;
    /// What a power's own listener bodies are worth, walked as part of the
    /// card that applies it: what the power does every turn it stands is the
    /// rest of what the card bought, read in the frame of whoever ends up
    /// carrying it. Every hook is walked, whatever condition gates it.
    /// Bounded, because a power's body applies powers.
    pub const DERIVED_POWER_BODY_WEIGHT: f64 = 0.2;
    pub const DERIVED_POWER_BODY_DEPTH: u32 = 1;
    /// Per card drawn and per energy gained.
    pub const DERIVED_PER_DRAW: f64 = 10.0;
    pub const DERIVED_PER_ENERGY: f64 = 9.0;
    /// Per point healed in the fight, and per point of the owner's own HP a
    /// body spends to pay for the rest of itself.
    pub const DERIVED_PER_HEAL: f64 = 1.5;
    pub const DERIVED_PER_SELF_HARM: f64 = 2.0;
    /// Per star gained (`PlayerCmd::GainStars`): the Regent's second
    /// resource, worth rather less per point than energy because only
    /// their own cards charge it.
    pub const DERIVED_PER_STAR_GAINED: f64 = 5.0;
    /// Per point of pet handed to `OstyCmd::Summon`: the Necrobinder's pet is
    /// a body that keeps standing, so a point of it is worth about a point of
    /// block that does not expire.
    pub const DERIVED_PER_PET_HP: f64 = 1.5;
    /// Per point a `ForgeCmd::Forge` sharpens the Regent's blade by: damage
    /// the rest of the fight swings with rather than damage now.
    pub const DERIVED_PER_FORGED_POINT: f64 = 2.0;
    /// An orb channeled works every turn it stands, so it is worth more than
    /// the one-shot firing it — evoking it, or ticking its passive — cashes
    /// it in for; a slot is only room for another one.
    pub const DERIVED_PER_ORB_CHANNELED: f64 = 15.0;
    pub const DERIVED_PER_ORB_FIRED: f64 = 8.0;
    pub const DERIVED_PER_ORB_SLOT: f64 = 5.0;
    /// How many passes a loop is read as making when its count is one the
    /// board settles rather than one the body states: a `Repeat` whose count
    /// is worked out, a `ForEachOrb`, a `ForEachCard`.
    pub const DERIVED_UNCOUNTED_LOOP_PASSES: f64 = 1.0;
    /// One arm of a two-armed `if`, only one of which runs. A bare guard —
    /// an `if` with no else — is worth its whole body instead: it is what the
    /// card is for rather than one of two things it might do.
    pub const DERIVED_WHEN_ARM_WEIGHT: f64 = 0.5;
    /// What a body is worth on a card that puts a condition on its own play
    /// (`playable_when`). The formula reads that the card is sometimes dead
    /// in hand, not how often; such a card is also worth none of the tempo a
    /// card that costs nothing is otherwise worth.
    pub const DERIVED_PLAY_RESTRICTED_FACTOR: f64 = 0.25;
    /// A damage or block amount the registry can only give as a formula
    /// (scaled by strength, by cards played, ...): worth about this much
    /// fixed, since such numbers usually grow.
    pub const DERIVED_SCALED_AMOUNT: f64 = 9.0;
    /// The body's value is divided by this per energy the card costs, so a
    /// 2-cost body must do more than twice a 1-cost one to score the same,
    /// and by this per star (`CardDefinition::star_cost`), the second
    /// resource nineteen of the Regent's cards charge on top of energy.
    pub const DERIVED_COST_DIVISOR_PER_ENERGY: f64 = 0.4;
    pub const DERIVED_COST_DIVISOR_PER_STAR: f64 = 0.3;
    /// X-cost bodies scale with what is spent; a flat allowance.
    pub const DERIVED_X_COST_BONUS: f64 = 8.0;
    /// A card that costs nothing is tempo whatever it does.
    pub const DERIVED_ZERO_COST_BONUS: f64 = 8.0;
    pub const DERIVED_RARITY_UNCOMMON: f64 = 5.0;
    pub const DERIVED_RARITY_RARE: f64 = 10.0;
    /// A card that removes itself when played thins the deck around the
    /// cards that stay; ethereal does the same, less reliably.
    pub const DERIVED_EXHAUST_BONUS: f64 = 4.0;
    pub const DERIVED_ETHEREAL_BONUS: f64 = 2.0;
    /// A body that adds curses to the deck.
    pub const DERIVED_CURSE_ADDING_PENALTY: f64 = 15.0;
    /// The scale's ends: nothing derived leaves them.
    pub const DERIVED_MIN: f64 = 10.0;
    pub const DERIVED_MAX: f64 = 75.0;
    /// `status_cards()`: unplayable dead draws (Wound, Dazed...), some of
    /// which hurt to hold (Toxic burns its holder for 5 at turn end).
    pub const TIER_STATUS: f64 = -60.0;
    /// Curses (Doubt, Clumsy, Greed...): dead draws the run carries forever.
    pub const TIER_CURSE: f64 = -80.0;

    // --- rewards ----------------------------------------------------------
    pub const CLAIM_GOLD: f64 = 40.0;
    pub const CLAIM_RELIC: f64 = 45.0;
    pub const CLAIM_POTION: f64 = 35.0;
    /// A potion with no slot to hold it is left on the table rather than
    /// forced through a discard.
    pub const CLAIM_POTION_FULL_BELT: f64 = -20.0;
    pub const CLAIM_REMOVAL_GOOD: f64 = 42.0;
    pub const CLAIM_REMOVAL_BAD: f64 = -5.0;
    pub const CLAIM_CARD_BASE: f64 = 30.0;
    /// A card reward whose whole offer is below the take bar is skipped —
    /// walking on beats deck bloat. The claim is gated *before* opening the
    /// pick screen, so a skippable reward is never claim-cancel looped.
    pub const CLAIM_CARD_ALL_BAD: f64 = -20.0;
    pub const CLAIM_SPECIAL_CARD: f64 = 25.0;
    pub const CLAIM_UNKNOWN: f64 = 10.0;
    pub const TAKE_TREASURE: f64 = 50.0;

    // --- shops ------------------------------------------------------------
    /// Removal first: at A0 the starter deck's dead weight (and any curse)
    /// is the cheapest permanent upgrade gold buys.
    pub const SHOP_REMOVAL: f64 = 60.0;
    pub const SHOP_RELIC: f64 = 50.0;
    pub const SHOP_RELIC_MAX_PRICE: i32 = 180;
    pub const SHOP_POTION: f64 = 45.0;
    pub const SHOP_POTION_MAX_PRICE: i32 = 80;
    pub const SHOP_CARD_BASE: f64 = 40.0;
    pub const SHOP_CARD_MAX_PRICE: i32 = 120;
    pub const SHOP_SKIP: f64 = -5.0;

    // --- events -----------------------------------------------------------
    /// Marginal value of one point of each resource inside an event body.
    pub const EVENT_GOLD_PER_POINT: f64 = 0.15;
    pub const EVENT_HEAL_PER_POINT: f64 = 0.5;
    pub const EVENT_MAX_HP_PER_POINT: f64 = 1.2;
    pub const EVENT_MAX_HP_LOSS_PER_POINT: f64 = 1.5;
    /// HP paid to an event, per point — and the multiplier once HP is low,
    /// which is what "avoid HP-gambles when low" means numerically.
    pub const EVENT_HP_LOSS_PER_POINT: f64 = 0.6;
    pub const EVENT_HP_LOSS_LOW_MULTIPLIER: f64 = 2.5;
    pub const EVENT_HP_LOW_FRACTION: f64 = 0.4;
    /// When an event's cost is a var this policy cannot read, assume this
    /// fraction of max HP rather than assuming it free.
    pub const EVENT_UNKNOWN_HP_COST_FRACTION: i32 = 8;
    pub const EVENT_RELIC: f64 = 18.0;
    pub const EVENT_RELIC_PAGE: f64 = 15.0;
    pub const EVENT_CARD_OFFER: f64 = 6.0;
    pub const EVENT_REWARD_EACH: f64 = 8.0;
    pub const EVENT_REMOVAL_EACH: f64 = 10.0;
    pub const EVENT_REMOVE_ROLLED: f64 = 6.0;
    pub const EVENT_UPGRADE_EACH: f64 = 8.0;
    pub const EVENT_DOWNGRADE: f64 = -12.0;
    pub const EVENT_LOSE_RELIC: f64 = -10.0;
    pub const EVENT_LOSE_POTION: f64 = -4.0;
    pub const EVENT_POTION_TRADE: f64 = 4.0;
    pub const EVENT_GRAB_BELT: f64 = 8.0;
    pub const EVENT_BAG_BUY: f64 = 10.0;
    pub const EVENT_CANNOT_AFFORD: f64 = -5.0;
    /// `LoseMaxHpAndUpgrade` (Tablet of Truth and kin): at A0 act 1, max HP
    /// is the resource runs die for lack of; the upgrade does not pay it
    /// back fast enough to bootstrap on.
    pub const EVENT_MAX_HP_FOR_UPGRADE: f64 = -5.0;
    /// An event fight (`EnterCombatWithoutLeaving`): rewards behind HP risk.
    pub const EVENT_FIGHT: f64 = -6.0;
    pub const EVENT_FIGHT_LOW_HP: f64 = -20.0;
    pub const EVENT_CRYSTAL_SPHERE: f64 = 5.0;
    /// A page an option opens is worth its best line, discounted for being
    /// one click further away.
    pub const EVENT_PAGE_DISCOUNT: f64 = 0.85;
    pub const EVENT_SHUFFLED_PAGE_DISCOUNT: f64 = 0.5;
    pub const EVENT_PAGE_DEPTH: u32 = 3;
    /// `is_proceed` options (leave / continue): a whisker above zero so an
    /// all-negative page is left rather than paid.
    pub const EVENT_PROCEED_BASELINE: f64 = 1.0;

    // --- choose-cards screens --------------------------------------------
    /// Screens that must be completed once opened (a cancel would re-offer
    /// the opener and loop): committing beats any pick, and the empty answer
    /// is poisoned.
    pub const SCREEN_COMMIT: f64 = 100.0;
    pub const SCREEN_NEVER_CANCEL: f64 = -100.0;
}

/// The Ironclad tier table: model id → tier, 0–100, judged from each card's
/// body in `sts2-content/src/cards.rs` for an
/// A0 act-1 bootstrap — front-loaded damage, `AoE`, and tempo over engines
/// that need a developed deck. Cards not in the table — every other class's
/// pool, colorless cards, anything picked up off-class — are scored from
/// their registry definition by [`derived_tier`].
#[must_use]
#[allow(clippy::too_many_lines, reason = "one entry per card, kept legible")]
#[allow(
    clippy::match_same_arms,
    reason = "one commented entry per card beats arms merged by coincidence of value"
)]
pub fn card_tier(model_id: &str) -> Option<f64> {
    Some(match model_id {
        // --- starters: kept low so removal prefers them and rewards beat
        // them. Strike is `attack(6)`; Defend is `block(5)`.
        "CARD.STRIKE_IRONCLAD" => 5.0,
        "CARD.DEFEND_IRONCLAD" => 8.0,
        // Bash: `attack(8)` + Vulnerable 2 — the starter deck's damage
        // multiplier. Not offered by pools (Basic), tiered for smith/removal.
        "CARD.BASH" => 30.0,

        // --- commons ------------------------------------------------------
        // `attack(9)` + `draw(1)`: damage that replaces itself.
        "CARD.POMMEL_STRIKE" => 62.0,
        // `block(8)` + `draw(1)`: the defensive twin.
        "CARD.SHRUG_IT_OFF" => 60.0,
        // `sweep(4)` + Vulnerable 1 on everything: act-1 packs (slimes,
        // crawlers) are exactly what a 1-cost AoE + mass shake beats.
        "CARD.THUNDERCLAP" => 58.0,
        // `block(5), attack(5)` for 1: tempo both ways.
        "CARD.IRON_WAVE" => 48.0,
        // 0-cost `attack(6)` that leaves a combat-only copy in the discard.
        "CARD.ANGER" => 45.0,
        // `6+2/Strike-tag ×1` at 2E: the starter's five Strikes plus tagged
        // pickups make this ~16 on floor 1 and it grows.
        "CARD.PERFECTED_STRIKE" => 45.0,
        // `3×3` random targets: nine damage for one.
        "CARD.SWORD_BOOMERANG" => 44.0,
        // `5×2` for 1E.
        "CARD.TWIN_STRIKE" => 42.0,
        // `attack(9)` then one discard-pile card on top of the draw pile.
        "CARD.HEADBUTT" => 40.0,
        // `attack(18)` at 2E, burns a random hand card: big hit, real cost.
        "CARD.CINDER" => 40.0,
        // `self_hp_loss(1), sweep(9)` at 1E: cheap AoE.
        "CARD.BREAKTHROUGH" => 40.0,
        // `block(5)` + upgrade a hand card for the turn's fight.
        "CARD.ARMAMENTS" => 40.0,
        // `attack(7)` + 2 temporary strength.
        "CARD.SETUP_STRIKE" => 38.0,
        // `attack(10)` + doubles existing Vulnerable, exhausts: Bash-dependent.
        "CARD.MOLTEN_FIST" => 35.0,
        // Vulnerable 3, exhausts: sets up a big turn once.
        "CARD.TREMBLE" => 35.0,
        // 2HP → `block(16)` at 2E.
        "CARD.BLOOD_WALL" => 35.0,
        // 3HP → 2 energy at 0E: an enabler that wants a deck to enable.
        "CARD.BLOODLETTING" => 30.0,
        // `block(7)` + a *random* hand card exhausted: can eat a good card.
        "CARD.TRUE_GRIT" => 28.0,
        // Auto-plays the top draw: uncontrolled with statuses/curses around.
        "CARD.HAVOC" => 22.0,
        // Damage = current block at 1E (0E upgraded): the starter deck holds
        // almost no block to convert.
        "CARD.BODY_SLAM" => 15.0,

        // --- uncommons ----------------------------------------------------
        // 0-cost `draw(3)` (no more draws this turn): tempo that finds the
        // deck's few good cards.
        "CARD.BATTLE_TRANCE" => 68.0,
        // `attack(13)` + Weak 1 + Vulnerable 1: damage plus both defensive
        // and offensive debuffs in one card.
        "CARD.UPPERCUT" => 66.0,
        // 2HP → `attack(15)` at 1E: best damage-per-energy in the pool.
        "CARD.HEMOKINESIS" => 62.0,
        // Strength 2: scales every hit of every remaining fight.
        "CARD.INFLAME" => 60.0,
        // `attack(32)` at 3E: elite/boss killer.
        "CARD.BLUDGEON" => 58.0,
        // X-cost `5×X` on everything: scaling AoE.
        "CARD.WHIRLWIND" => 55.0,
        // Plating 4: block every single turn (powers.rs pays it at turn end).
        "CARD.STONE_ARMOR" => 55.0,
        // `attack(14)` + a free attack owed (`POWER.FREE_ATTACK_POWER`).
        "CARD.UNRELENTING" => 52.0,
        // `8×2` against a Vulnerable target: Bash makes it 16 for 1E.
        "CARD.DISMANTLE" => 50.0,
        // Pulls attacks back from the discard *upgraded* each turn
        // (aggression_power in powers.rs): a free card and a free upgrade.
        "CARD.AGGRESSION" => 50.0,
        // `sweep(16)` at 3E that replays itself out of the exhaust pile.
        "CARD.HOWL_FROM_BEYOND" => 48.0,
        // Block per exhaust: needs an exhaust package to feed it.
        "CARD.FEEL_NO_PAIN" => 45.0,
        // Vulnerable + strength equal to the target's Vulnerable.
        "CARD.DOMINATE" => 45.0,
        // Grows 5 damage per play, permanent within the fight.
        "CARD.RAMPAGE" => 45.0,
        // Auto-plays attacks from hand free at end of turn (stampede_power).
        "CARD.STAMPEDE" => 45.0,
        // Draw per Vulnerable application: pairs with Bash/Thunderclap.
        "CARD.VICIOUS" => 45.0,
        // `sweep(12)` whose cost shrinks per attack played.
        "CARD.STOMP" => 42.0,
        // `attack(6)` + draw-while-attacks chain.
        "CARD.PILLAGE" => 42.0,
        // Auto-plays every Strike on draw (hellraiser_power): free Strikes
        // in a Strike-heavy starter deck.
        "CARD.HELLRAISER" => 40.0,
        // `4+2/Vulnerable-stack` at 0E: a Bash payoff card.
        "CARD.BULLY" => 40.0,
        // `5×2` + 3 strength — but hands the target 1 strength too.
        "CARD.FIGHT_ME" => 38.0,
        // `block(7)` + Vulnerable 1 on a target.
        "CARD.TAUNT" => 35.0,
        // Block twice if something burned this turn: package card.
        "CARD.EVIL_EYE" => 35.0,
        // Exhaust one, draw two: card-quality filter.
        "CARD.BURNING_PACT" => 35.0,
        // `draw(2)` now, energy whenever it burns: package card.
        "CARD.DRUM_OF_BATTLE" => 35.0,
        // Copies the third attack each turn (juggling_power).
        "CARD.JUGGLING" => 35.0,
        // Next attack plays twice (one_two_punch_power).
        "CARD.ONE_TWO_PUNCH" => 45.0,
        // Strength per unblocked self-HP-loss on own turn (rupture_power):
        // needs the self-damage package.
        "CARD.RUPTURE" => 30.0,
        // Burns every non-attack in hand for 5 block each: eats Defends.
        "CARD.SECOND_WIND" => 30.0,
        // `5×2` only if the owner already bled this turn.
        "CARD.SPITE" => 30.0,
        // Rolls a random free attack into hand, exhausts.
        "CARD.INFERNAL_BLADE" => 30.0,
        // Energy per attack in hand, then no energy gains: anti-synergy
        // with everything else here.
        "CARD.EXPECT_A_FIGHT" => 25.0,
        // 6 + 3-per-exhausted-card: needs an exhaust deck.
        "CARD.ASHEN_STRIKE" => 30.0,
        // Doubles block gains up to N times a turn (unmovable_power).
        "CARD.UNMOVABLE" => 40.0,
        // Block 5/8 + halves a shaken attacker's damage for a turn.
        "CARD.COLOSSUS" => 35.0,
        // `block(12)` + burns attackers for 4: strong into multi-hit acts.
        "CARD.FLAME_BARRIER" => 40.0,
        // Inferno: strength engine that also burns its owner harder per
        // laying-down (`POWER.INFERNO_POWER` counter).
        "CARD.INFERNO" => 35.0,
        // Rage: 3 block per attack played this turn, 0-cost.
        "CARD.RAGE" => 40.0,

        // --- rares --------------------------------------------------------
        // 6HP → 2 energy + 3 cards, 0-cost: the tempo rare.
        "CARD.OFFERING" => 70.0,
        // `attack(10)` + 3 max HP per counting kill: compounds across the
        // whole run, exactly what a bootstrap policy wants.
        "CARD.FEED" => 65.0,
        // +1 max energy every turn (pyre power reads `ModifyMaxEnergy`).
        "CARD.PYRE" => 65.0,
        // `block(30)` once: the elite/boss turn-saver.
        "CARD.IMPERVIOUS" => 62.0,
        // Strength 2 per turn, every turn.
        "CARD.DEMON_FORM" => 60.0,
        // Burns the hand, hits once per card burned: huge with any hand.
        "CARD.FIEND_FIRE" => 55.0,
        // `heal(10)` in-fight, exhausts: real HP at A0.
        "CARD.NOT_YET" => 45.0,
        // `4×2` that eats an attack and keeps its damage: grows all fight.
        "CARD.THRASH" => 45.0,
        // Damage per block gained (juggernaut_power).
        "CARD.JUGGERNAUT" => 45.0,
        // `attack(15)` + Weak-alike (MANGLE_POWER strips strength) at 3E.
        "CARD.MANGLE" => 40.0,
        // Keeps block across turns; upgrade only cheapens it.
        "CARD.BARRICADE" => 40.0,
        // 0-cost `sweep(17)` gated on 3+ cards exhausted.
        "CARD.PACTS_END" => 35.0,
        // +25% damage onto Vulnerable targets (the Vulnerable power reads it).
        "CARD.CRUELTY" => 35.0,
        // Draw per exhaust: package engine.
        "CARD.DARK_EMBRACE" => 35.0,
        // Stacking block power that costs 1 more HP per laying-down.
        "CARD.CRIMSON_MANTLE" => 35.0,
        // 1-per-hit `5×history-of-unblocked-blows`: pays for getting hit.
        "CARD.TEAR_ASUNDER" => 35.0,
        // `2×4` sweep at rare rarity: numbers a common would carry.
        "CARD.CONFLAGRATION" => 30.0,
        // Burns the hand, rolls that many pool cards back: a gamble.
        "CARD.STOKE" => 30.0,
        // 1HP + exhaust a card → strength: slow.
        "CARD.BRAND" => 30.0,
        // X-cost: auto-play X off the draw pile.
        "CARD.CASCADE" => 30.0,
        // Every attack in hand becomes a Giant Rock (16 for 1E): swingy.
        "CARD.PRIMAL_FORCE" => 25.0,

        // --- act pickups / tokens outside the rolled pool ------------------
        // `attack(16)` for 1E, Token rarity: what PRIMAL_FORCE makes.
        "CARD.GIANT_ROCK" => 30.0,
        // Ancient: `attack(20)` + Vulnerable 5 (Archaic Tooth's card).
        "CARD.BREAK" => 55.0,
        // Ancient: Corruption — skills cost 0 and exhaust; needs a deck
        // built for it, dangerous to a starter deck's Defends.
        "CARD.CORRUPTION" => 30.0,
        _ => return None,
    })
}

/// How sharply the teacher's own scores are read as a distribution, in the
/// units [`knobs`] is written in: one *e*-fold of probability per this many
/// points of score.
///
/// Five, because five is the granularity the scores are actually authored
/// at — the tier table steps in fives, and the pathing knobs separate
/// families by single-digit margins (`MAP_SHOP` 6 against `MAP_MONSTER` 8).
/// At this temperature a knob's-worth of separation is a 1.2× preference, a
/// tier's-worth (20 points: a rare against a common) is 55×, and the
/// screen-commit and `NEVER` sentinels underflow to zero — so the cloned
/// distribution is peaked on exactly what the teacher would have played
/// while still carrying its ranking of everything it did not.
///
/// Stamped into every heuristic-sourced macro shard's header
/// (`macro_policy_temperature`), because a net trained toward one sharpness
/// must not be read as if it had been trained toward another.
pub const MACRO_POLICY_TEMPERATURE: f64 = 5.0;

/// The stateless out-of-combat policy. Deterministic given the simulator
/// state (the RNG is consumed only by the in-combat uniform fallback), so
/// batch reproducibility — byte-identical shards across `--jobs` — holds
/// exactly as it does for `UniformRandom`.
///
/// It is also a *teacher*: [`Heuristic::action_scores`] hands out the very
/// numbers `choose` takes its argmax over, and
/// [`Heuristic::teacher_policy`] reads them as the soft target a macro net
/// is cloned from (the `AlphaGo` SL-policy pattern). A one-hot target would
/// throw away everything the policy knows about the actions it did *not*
/// pick; the softmax keeps the ranking and the margins.
#[derive(Clone, Copy, Debug, Default)]
pub struct Heuristic;

impl Heuristic {
    /// This policy's own score for each of `actions`, in [`knobs`] units.
    ///
    /// The same `score_action` `choose` ranks by, over a caller-chosen
    /// action list rather than over `permitted_actions` — so a recorder can
    /// score exactly the canonical list it is about to encode. Scores are
    /// comparable only within one decision screen, which is all either
    /// caller needs.
    #[must_use]
    pub fn action_scores(simulator: &Simulator, actions: &[Action]) -> Vec<f64> {
        let ctx = Ctx::read(simulator);
        actions
            .iter()
            .map(|action| score_action(&ctx, action))
            .collect()
    }

    /// The teacher's answer as a distribution over `actions`: a softmax of
    /// [`Heuristic::action_scores`] at [`MACRO_POLICY_TEMPERATURE`],
    /// numerically settled by subtracting the peak.
    ///
    /// Its argmax is `choose`'s argmax whenever the two are handed the same
    /// list — ties included, since a softmax splits equal scores evenly and
    /// `choose` breaks them by index.
    #[must_use]
    pub fn teacher_policy(simulator: &Simulator, actions: &[Action]) -> Vec<f64> {
        softmax_at(
            &Self::action_scores(simulator, actions),
            MACRO_POLICY_TEMPERATURE,
        )
    }
}

/// A settled softmax at `temperature`. An empty list answers empty; a list
/// whose weights all underflow answers uniform, which cannot happen after
/// the peak subtraction but is the honest fallback if it ever did.
#[must_use]
fn softmax_at(scores: &[f64], temperature: f64) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores
        .iter()
        .map(|score| ((score - peak) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if total > 0.0 && total.is_finite() {
        weights.iter().map(|weight| weight / total).collect()
    } else {
        #[allow(clippy::cast_precision_loss, reason = "action counts are small")]
        {
            vec![1.0 / scores.len() as f64; scores.len()]
        }
    }
}

impl RolloutPolicy for Heuristic {
    fn macro_teacher_policy(&self, simulator: &Simulator, actions: &[Action]) -> Option<Vec<f64>> {
        // Only out of combat: in a fight this policy is the uniform
        // baseline, and a macro recorder never asks there anyway.
        fight_over(simulator).then(|| Self::teacher_policy(simulator, actions))
    }

    fn choose(&mut self, simulator: &Simulator, rng: &mut MegaRandom) -> Action {
        // In-combat decisions are the search's (or, in rollouts, chance's):
        // deferring to the uniform baseline keeps an A/B against
        // `UniformRandom` attributable to out-of-combat play alone.
        if !fight_over(simulator) {
            return UniformRandom.choose(simulator, rng);
        }
        let permitted = permitted_actions(simulator);
        // A policy cannot answer a screen that offers nothing; an empty
        // enumeration on a live run is the engine's fault to report, not
        // this policy's to paper over. Search rollouts never ask (an empty
        // offer is a leaf there — see `search::leaf_value`), so tripping
        // this means the *authoritative* run stands on the dead screen, and
        // the harness reports it with the run's seed as the reproducer.
        assert!(
            !permitted.is_empty(),
            "a live run offers an action; the engine enumerated none at {:?}",
            simulator.decision()
        );
        let ctx = Ctx::read(simulator);
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (index, action) in permitted.iter().enumerate() {
            let score = score_action(&ctx, action);
            // Strict: the first of equals wins, deterministically.
            if score > best_score {
                best = index;
                best_score = score;
            }
        }
        permitted[best].clone()
    }
}

/// What every scorer reads about the run, gathered once per decision.
struct Ctx<'a> {
    simulator: &'a Simulator,
    hp: i32,
    max_hp: i32,
    hp_frac: f64,
    gold: i32,
    free_potion_slots: usize,
    deck_size: usize,
    /// Attacks added since the start (starters carry no
    /// `floor_added_to_deck`): the "does the deck deal damage yet" proxy.
    added_attacks: usize,
    /// Added cards + upgrade levels: the "has the deck grown" proxy the
    /// elite gate reads.
    deck_strength: usize,
}

/// What the body of a card adds up to, read off its registry definition.
#[derive(Default)]
struct CardBody {
    damage: f64,
    block: f64,
    heal: f64,
    self_harm: f64,
    enemy_debuffs: f64,
    enemy_debuff_stacks: f64,
    self_buffs: f64,
    self_buff_stacks: f64,
    draws: f64,
    energy: f64,
    stars: f64,
    pet_hp: f64,
    forged: f64,
    orbs_channeled: f64,
    orbs_fired: f64,
    orb_slots: f64,
    scaled_amounts: f64,
    curses_added: f64,
}

/// Which side of the fight a selector names, where the body plainly says.
/// Most selectors name a creature by the part it played rather than by the
/// side it stands on, and a card body means those the way it aims: outward.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Ours,
    Theirs,
    Unsaid,
}

impl Side {
    /// The same side read from the other end of the fight.
    fn swapped(self) -> Self {
        match self {
            Self::Ours => Self::Theirs,
            Self::Theirs => Self::Ours,
            Self::Unsaid => Self::Unsaid,
        }
    }
}

/// One pass of the walk over a run of commands: the registry the bodies it
/// reaches are read out of, how much of the body this run is — how many times
/// the loops around it run it, and which arm of a two-armed `if` it stands in
/// — whether the card aims at the whole enemy side, how deep into the powers
/// it applies the walk has gone, and whose side `CreatureSelector::Owner`
/// names, which is the card's own until the walk steps into a power standing
/// on an enemy.
#[derive(Clone, Copy)]
struct Pass<'a> {
    content: &'a sts2_engine::ContentRegistry,
    weight: f64,
    all_enemies: bool,
    depth: u32,
    owner_is_enemy: bool,
}

impl<'a> Pass<'a> {
    fn root(content: &'a sts2_engine::ContentRegistry, all_enemies: bool) -> Self {
        Self {
            content,
            weight: 1.0,
            all_enemies,
            depth: 0,
            owner_is_enemy: false,
        }
    }

    /// The same pass over a run that is worth a fraction of the body.
    fn scaled(self, by: f64) -> Self {
        Self {
            weight: self.weight * by,
            ..self
        }
    }

    /// The pass into an applied power's own listener bodies, in the frame of
    /// whichever creature ends up carrying it.
    fn into_power(self, on_enemy: bool) -> Self {
        Self {
            weight: self.weight * knobs::DERIVED_POWER_BODY_WEIGHT,
            all_enemies: false,
            depth: self.depth + 1,
            owner_is_enemy: on_enemy,
            ..self
        }
    }

    /// Which side a selector names, from where this pass is standing.
    fn side(self, target: &sts2_engine::CreatureSelector) -> Side {
        use sts2_engine::CreatureSelector as Sel;
        let side = match target {
            Sel::Owner
            | Sel::Applier
            | Sel::AllAllies
            | Sel::OtherAllies
            | Sel::Pet
            | Sel::AppliersPet
            | Sel::DoomedOnOwnersSide => Side::Ours,
            Sel::Target
            | Sel::OtherOpponents
            | Sel::AllOpponents
            | Sel::RandomOpponent
            | Sel::TargetOrRandomOpponent
            | Sel::WeakestOpponent
            | Sel::EnemyCarryingThePower
            | Sel::DoomedAmongOpponents(_) => Side::Theirs,
            _ => Side::Unsaid,
        };
        if self.owner_is_enemy {
            side.swapped()
        } else {
            side
        }
    }

    /// How many times over a blow aimed this way counts, which the card's own
    /// target type says as well as the command's selector. A power standing
    /// on an enemy faces one player, so nothing it does spreads.
    fn spread(self, target: &sts2_engine::CreatureSelector) -> f64 {
        let whole_side =
            self.all_enemies || matches!(target, sts2_engine::CreatureSelector::AllOpponents);
        if whole_side && !self.owner_is_enemy {
            knobs::DERIVED_AOE_MULTIPLIER
        } else {
            1.0
        }
    }
}

/// What a size states outright, and whether it is worked out on the board
/// instead — scaled by strength, by cards played, by what is standing. This
/// formula can read the first and only allow for the second.
fn stated(amount: &sts2_engine::Amount) -> (f64, bool) {
    match amount {
        sts2_engine::Amount::Fixed(value) => (f64::from(*value), false),
        _ => (0.0, true),
    }
}

/// The whole number a body states outright, for the registry calls that read
/// an amount rather than a size.
fn stated_i32(amount: &sts2_engine::Amount) -> Option<i32> {
    match amount {
        sts2_engine::Amount::Fixed(value) => Some(*value),
        _ => None,
    }
}

/// A size the body states outright and does not grow.
fn fixed(amount: &sts2_engine::Amount) -> Option<f64> {
    match stated(amount) {
        (value, false) => Some(value),
        (_, true) => None,
    }
}

/// The `then` a card-selection screen runs over what was chosen. Every
/// variant of the family has one; the screen itself is the prompt, and what
/// the body is worth is what it does with the answer.
fn selection_body(command: &sts2_engine::CardSelectCmd) -> &[sts2_engine::Command] {
    use sts2_engine::CardSelectCmd;
    match command {
        CardSelectCmd::FromHand { then, .. }
        | CardSelectCmd::FromHandForDiscard { then, .. }
        | CardSelectCmd::FromHandForUpgrade { then }
        | CardSelectCmd::FromCombatPile { then, .. }
        | CardSelectCmd::FromDeckForUpgrade { then, .. }
        | CardSelectCmd::FromDeckForRemoval { then, .. }
        | CardSelectCmd::FromDeckGeneric { then, .. }
        | CardSelectCmd::ChooseACard { then, .. }
        | CardSelectCmd::SimpleGridOfGeneratedCards { then, .. }
        | CardSelectCmd::ChooseAGeneratedCard { then, .. }
        | CardSelectCmd::SimpleGrid { then, .. }
        | CardSelectCmd::SimpleGridForRewards { then, .. }
        | CardSelectCmd::ChooseACreatedCard { then, .. }
        | CardSelectCmd::FromDeckForTransformation { then, .. }
        | CardSelectCmd::FromDeckForEnchantment { then, .. }
        | CardSelectCmd::ChooseAGeneratedBundle { then, .. } => then,
    }
}

/// How many times a loop runs its body. A count the board settles rather than
/// states is worth [`knobs::DERIVED_UNCOUNTED_LOOP_PASSES`] passes.
fn passes(times: &sts2_engine::Amount) -> f64 {
    fixed(times).unwrap_or(knobs::DERIVED_UNCOUNTED_LOOP_PASSES)
}

impl CardBody {
    /// A swing: amount per hit, over as many hits, over as many targets.
    fn attack(
        &mut self,
        amount: &sts2_engine::Amount,
        hits: &sts2_engine::Amount,
        target: &sts2_engine::CreatureSelector,
        pass: Pass,
    ) {
        let hits = fixed(hits).unwrap_or(1.0).max(1.0);
        self.hurt(target, amount, pass.scaled(hits));
    }

    /// Damage, whatever the command that deals it: at an enemy it is damage,
    /// at the player's own side it is the price the card charges for the rest
    /// of its body, which no allowance for a growing number is owed for.
    fn hurt(
        &mut self,
        target: &sts2_engine::CreatureSelector,
        amount: &sts2_engine::Amount,
        pass: Pass,
    ) {
        let (amount, grows) = stated(amount);
        if pass.side(target) == Side::Ours {
            self.self_harm += pass.weight * amount;
        } else {
            self.damage += pass.weight * amount * pass.spread(target);
            if grows {
                self.scaled_amounts += pass.weight;
            }
        }
    }

    /// Block gained, whatever the command that gains it. Block standing in
    /// front of an enemy is the same number the other way about.
    fn gain_block(
        &mut self,
        target: &sts2_engine::CreatureSelector,
        amount: &sts2_engine::Amount,
        pass: Pass,
    ) {
        let (amount, grows) = stated(amount);
        let ours = if pass.side(target) == Side::Theirs {
            -1.0
        } else {
            1.0
        };
        self.block += ours * pass.weight * amount;
        if grows {
            self.scaled_amounts += ours * pass.weight;
        }
    }

    /// HP put back, on whichever side the command aims at.
    fn heal(
        &mut self,
        target: &sts2_engine::CreatureSelector,
        amount: &sts2_engine::Amount,
        pass: Pass,
    ) {
        let (amount, grows) = stated(amount);
        let ours = if pass.side(target) == Side::Theirs {
            -1.0
        } else {
            1.0
        };
        self.heal += ours * pass.weight * amount;
        if grows {
            self.scaled_amounts += ours * pass.weight;
        }
    }

    /// A power applied: its stacks on the scale of the side it stands on,
    /// signed by which way the registry says it reads for that amount, plus
    /// what the power's own listener bodies do once it is standing.
    fn apply_power(
        &mut self,
        power: &sts2_core::ModelId,
        target: &sts2_engine::CreatureSelector,
        amount: &sts2_engine::Amount,
        pass: Pass,
    ) {
        let model = pass.content.power(power);
        let applied = stated_i32(amount);
        let stacks =
            f64::from(applied.unwrap_or(1).abs()).clamp(1.0, knobs::DERIVED_MAX_PRICED_STACKS);
        let on_us = pass.side(target) == Side::Ours;
        // A Strength put on an enemy is not a debuff and a Strength worked
        // backwards is not a buff; the registry answers both, for the amount
        // applied. Where the body works that amount out rather than stating
        // it — a Malaise takes off as much Strength as it was paid — the
        // registry has no answer to give and the side it lands on is what is
        // left to read.
        let debuff = match (applied, model) {
            (Some(applied), Some(model)) => matches!(
                model.type_for_amount(applied),
                sts2_engine::PowerType::Debuff
            ),
            _ => !on_us,
        };
        // Helping is a buff on our side or a debuff on theirs.
        let helps = if debuff == on_us { -1.0 } else { 1.0 };
        if on_us {
            self.self_buffs += helps * pass.weight;
            self.self_buff_stacks += helps * pass.weight * (stacks - 1.0);
        } else {
            let spread = helps * pass.weight * pass.spread(target);
            self.enemy_debuffs += spread;
            self.enemy_debuff_stacks += spread * (stacks - 1.0);
        }
        if pass.depth < knobs::DERIVED_POWER_BODY_DEPTH
            && let Some(model) = model
        {
            let into = pass.into_power(!on_us);
            for hook in &model.hooks {
                self.walk(&hook.commands, into);
            }
        }
    }

    /// An orb command: a channel stands an orb up for the rest of the fight,
    /// firing one cashes in what is already standing, and slots are room for
    /// more.
    fn orb(&mut self, command: &sts2_engine::OrbCmd, weight: f64) {
        use sts2_engine::OrbCmd;
        match command {
            OrbCmd::Channel(_) => self.orbs_channeled += weight,
            OrbCmd::EvokeNext { .. } | OrbCmd::EvokeLast { .. } | OrbCmd::Passive { .. } => {
                self.orbs_fired += weight;
            }
            OrbCmd::AddSlots(amount) => self.orb_slots += weight * fixed(amount).unwrap_or(1.0),
            OrbCmd::RemoveSlots(amount) => self.orb_slots -= weight * fixed(amount).unwrap_or(1.0),
            // `SetValue` writes an orb's own number and `Reached` is an
            // evoke's return: neither is a gain of its own.
            OrbCmd::SetValue { .. } | OrbCmd::Reached(_) => {}
        }
    }

    /// Reads a run of commands into the tally, under the [`Pass`] that says
    /// how much of the body the run is and whose side it acts on.
    #[allow(
        clippy::too_many_lines,
        clippy::match_same_arms,
        reason = "one arm per command family, each priced for its own reason"
    )]
    fn walk(&mut self, commands: &[sts2_engine::Command], pass: Pass) {
        use sts2_engine::{
            CardPileCmd, Command, CreatureCmd, DamageCmd, ForgeCmd, OstyCmd, PlayerCmd, PowerCmd,
        };
        for command in commands {
            match command {
                Command::Damage(
                    DamageCmd::Attack {
                        amount,
                        hits,
                        target,
                        ..
                    }
                    | DamageCmd::AttackFromPet {
                        amount,
                        hits,
                        target,
                        ..
                    },
                ) => self.attack(amount, hits, target, pass),
                Command::Strike {
                    amount,
                    hits,
                    target,
                    then,
                    ..
                } => {
                    self.attack(amount, hits, target, pass);
                    self.walk(then, pass);
                }
                Command::Creature(CreatureCmd::Damage { target, amount, .. }) => {
                    self.hurt(target, amount, pass);
                }
                Command::Deal {
                    target,
                    amount,
                    then,
                    ..
                } => {
                    self.hurt(target, amount, pass);
                    self.walk(then, pass);
                }
                Command::Creature(CreatureCmd::GainBlock { target, amount, .. }) => {
                    self.gain_block(target, amount, pass);
                }
                Command::Shield {
                    target,
                    amount,
                    then,
                    ..
                } => {
                    self.gain_block(target, amount, pass);
                    self.walk(then, pass);
                }
                // A Glitterstream works out what block would come to and
                // hands the number to `then`; it gains none of its own.
                Command::WorkOutBlock { then, .. } => self.walk(then, pass),
                Command::Creature(CreatureCmd::Heal { target, amount }) => {
                    self.heal(target, amount, pass);
                }
                Command::Power(PowerCmd::Apply {
                    power,
                    target,
                    amount,
                    ..
                }) => self.apply_power(power, target, amount, pass),
                Command::CardPile(CardPileCmd::Draw { count }) => {
                    self.draws += pass.weight * fixed(count).unwrap_or(1.0);
                }
                // A bound draw hands what it drew to `then`, which keeps some
                // and throws the rest back through the pile commands this
                // formula does not read: an Escape Plan pays block only for a
                // skill, a Scrape discards whatever costs something. What the
                // draw was worth is what `then` did with it.
                Command::Drawn { then, .. } => self.walk(then, pass),
                Command::CardPile(CardPileCmd::AddCursesToDeck { .. }) => {
                    self.curses_added += pass.weight;
                }
                Command::Player(PlayerCmd::GainEnergy(amount)) => {
                    self.energy += pass.weight * fixed(amount).unwrap_or(1.0);
                }
                Command::Player(PlayerCmd::LoseEnergy(amount)) => {
                    self.energy -= pass.weight * fixed(amount).unwrap_or(1.0);
                }
                Command::Player(PlayerCmd::GainStars(amount)) => {
                    self.stars += pass.weight * fixed(amount).unwrap_or(1.0);
                }
                Command::Player(PlayerCmd::LoseStars(amount)) => {
                    self.stars -= pass.weight * fixed(amount).unwrap_or(1.0);
                }
                Command::Osty(OstyCmd::Summon { amount, .. }) => {
                    self.pet_hp += pass.weight * fixed(amount).unwrap_or(1.0);
                }
                Command::Forge(ForgeCmd::Forge(amount)) => {
                    self.forged += pass.weight * fixed(amount).unwrap_or(1.0);
                }
                Command::Orb(command) => self.orb(command, pass.weight),
                Command::When {
                    then, otherwise, ..
                } => {
                    if otherwise.is_empty() {
                        // A bare guard: the body it guards is the point of
                        // the card, so it is worth what it says.
                        self.walk(then, pass);
                    } else {
                        // Two arms, one of which runs.
                        let arm = pass.scaled(knobs::DERIVED_WHEN_ARM_WEIGHT);
                        self.walk(then, arm);
                        self.walk(otherwise, arm);
                    }
                }
                Command::Repeat { times, then } => {
                    self.walk(then, pass.scaled(passes(times).max(0.0)));
                }
                Command::RepeatWithLast { times, then, last } => {
                    // Every pass but the last runs `then`, and a count of
                    // nothing runs neither arm. A count the board settles
                    // runs both, the way a Multi-Cast leaves every orb but
                    // the final one standing.
                    if let Some(times) = fixed(times) {
                        let times = times.max(0.0);
                        self.walk(then, pass.scaled((times - 1.0).max(0.0)));
                        self.walk(last, pass.scaled(times.min(1.0)));
                    } else {
                        self.walk(then, pass.scaled(knobs::DERIVED_UNCOUNTED_LOOP_PASSES));
                        self.walk(last, pass);
                    }
                }
                // The body runs once and buys itself a pass for every
                // creature that pass killed, so one is what it is owed.
                Command::RepeatPerKill { then } => self.walk(then, pass),
                Command::ForEachOrb { then, .. } | Command::ForEachCard { then, .. } => {
                    self.walk(then, pass.scaled(knobs::DERIVED_UNCOUNTED_LOOP_PASSES));
                }
                Command::Select { then, .. }
                | Command::WithAmount { then, .. }
                | Command::WithCreatures { then, .. }
                | Command::WithTargetsDebuffs { then } => self.walk(then, pass),
                Command::CardSelect(command) => self.walk(selection_body(command), pass),
                // Deliberately unpriced, because their worth is the worth of
                // cards this formula is not looking at: the piles a body
                // shuffles, moves, exhausts, discards, upgrades, transforms
                // or auto-plays (`CardCmd`, the rest of `CardPileCmd`), and
                // the cards a body makes into the fight, which can as easily
                // be a Wound as a Shiv. Relics, potions and rewards a body
                // hands out are run-level gains a fight card is not judged
                // on, and `Command::Modify` is a listener's return value
                // rather than a gain at all.
                _ => {}
            }
        }
    }
}

/// A playable card's tier worked out from its registry definition, for
/// every card the hand table does not name: another character's pool, a
/// colorless card, anything a run picks up off-class. On the table's scale,
/// so the two compete honestly on one reward screen — a table entry, where
/// there is one, still wins.
///
/// Deliberately a formula over what the body plainly says: damage, block,
/// healing and self-inflicted HP loss, whether a blow hits the whole side,
/// the stacks of the powers it stands up and what those powers' own bodies
/// then do every turn, cards drawn, energy and stars gained, orbs channeled
/// and fired, the pet summoned and the blade forged. Each is counted as
/// many times as the loops around it run, discounted where it stands in one
/// arm of a two-armed `if`, and signed by the side it lands on, so a power
/// the owner takes on themselves and one put on an enemy read opposite ways.
/// On top of the body sit the card's own terms: what it costs in energy and
/// in stars, its rarity, the self-removing keywords, which thin the deck,
/// and the condition it puts on its own play, which is worth a fraction of a
/// body that may not be allowed to run. There is no notion of synergy in any
/// of it. A starter Strike or Defend (rarity Basic, tagged) scores exactly as
/// the Ironclad's do, so removal prefers them for every character. Curses and
/// statuses keep their tiers.
#[must_use]
pub fn derived_tier(
    content: &sts2_engine::ContentRegistry,
    definition: &sts2_engine::CardDefinition,
) -> f64 {
    use sts2_engine::{CardRarity, CardTag, CardType, TargetType};
    match definition.card_type {
        CardType::Curse => return knobs::TIER_CURSE,
        CardType::Status => return knobs::TIER_STATUS,
        _ => {}
    }
    if definition.unplayable {
        return knobs::TIER_STATUS;
    }
    if definition.rarity == CardRarity::Basic {
        if definition.tags.contains(&CardTag::Strike) {
            return card_tier("CARD.STRIKE_IRONCLAD").expect("the starter Strike is tiered");
        }
        if definition.tags.contains(&CardTag::Defend) {
            return card_tier("CARD.DEFEND_IRONCLAD").expect("the starter Defend is tiered");
        }
    }
    let mut body = CardBody::default();
    body.walk(
        &definition.effects,
        Pass::root(content, matches!(definition.target, TargetType::AllEnemies)),
    );
    let mut effects = body.damage * knobs::DERIVED_PER_DAMAGE
        + body.block * knobs::DERIVED_PER_BLOCK
        + body.heal * knobs::DERIVED_PER_HEAL
        - body.self_harm * knobs::DERIVED_PER_SELF_HARM
        + body.enemy_debuffs * knobs::DERIVED_PER_ENEMY_DEBUFF
        + body.enemy_debuff_stacks * knobs::DERIVED_PER_ENEMY_DEBUFF_STACK
        + body.self_buffs * knobs::DERIVED_PER_SELF_BUFF
        + body.self_buff_stacks * knobs::DERIVED_PER_SELF_BUFF_STACK
        + body.draws * knobs::DERIVED_PER_DRAW
        + body.energy * knobs::DERIVED_PER_ENERGY
        + body.stars * knobs::DERIVED_PER_STAR_GAINED
        + body.pet_hp * knobs::DERIVED_PER_PET_HP
        + body.forged * knobs::DERIVED_PER_FORGED_POINT
        + body.orbs_channeled * knobs::DERIVED_PER_ORB_CHANNELED
        + body.orbs_fired * knobs::DERIVED_PER_ORB_FIRED
        + body.orb_slots * knobs::DERIVED_PER_ORB_SLOT
        + body.scaled_amounts * knobs::DERIVED_SCALED_AMOUNT;
    // A card that asks a question before it may be played is sometimes dead
    // in hand: it is worth a fraction of its body, and none of the tempo a
    // card that costs nothing is otherwise worth. Only a fraction of what a
    // body is worth, never a fraction of what it costs.
    let restricted = definition.playable_when.is_some();
    if restricted {
        effects = effects.min(effects * knobs::DERIVED_PLAY_RESTRICTED_FACTOR);
    }
    let energy = if definition.costs_x {
        0
    } else {
        definition.energy_cost.max(0)
    };
    // `star_cost` is the `-1` sentinel on a card with no star cost at all,
    // which `max(0)` reads as the nothing it charges.
    let stars = if definition.costs_star_x {
        0
    } else {
        definition.star_cost.max(0)
    };
    let divisor = 1.0
        + knobs::DERIVED_COST_DIVISOR_PER_ENERGY * f64::from(energy.max(1) - 1)
        + knobs::DERIVED_COST_DIVISOR_PER_STAR * f64::from(stars);
    let base = match definition.card_type {
        CardType::Attack => knobs::DERIVED_BASE_ATTACK,
        CardType::Power => knobs::DERIVED_BASE_POWER,
        _ => knobs::DERIVED_BASE_SKILL,
    };
    let rarity = match definition.rarity {
        CardRarity::Uncommon => knobs::DERIVED_RARITY_UNCOMMON,
        CardRarity::Rare | CardRarity::Ancient => knobs::DERIVED_RARITY_RARE,
        _ => 0.0,
    };
    let mut tier = base + effects / divisor + rarity;
    if definition.costs_x || definition.costs_star_x {
        tier += knobs::DERIVED_X_COST_BONUS;
    } else if energy == 0 && stars == 0 && !restricted {
        tier += knobs::DERIVED_ZERO_COST_BONUS;
    }
    if definition.exhausts && definition.card_type != CardType::Power {
        tier += knobs::DERIVED_EXHAUST_BONUS;
    }
    if definition.ethereal {
        tier += knobs::DERIVED_ETHEREAL_BONUS;
    }
    if body.curses_added > 0.0 {
        tier -= body.curses_added.min(1.0) * knobs::DERIVED_CURSE_ADDING_PENALTY;
    }
    tier.clamp(knobs::DERIVED_MIN, knobs::DERIVED_MAX)
}

impl<'a> Ctx<'a> {
    fn read(simulator: &'a Simulator) -> Self {
        let state = simulator.state();
        let player = &state.run_player;
        let max_hp = player.max_hp.max(1);
        let mut added_attacks = 0;
        let mut deck_strength = 0;
        for card in &player.deck {
            let added = card.floor_added_to_deck.is_some();
            if added {
                deck_strength += 1;
                if card_type_of(simulator, card) == Some(CardType::Attack) {
                    added_attacks += 1;
                }
            }
            deck_strength += usize::from(card.upgrade_level);
        }
        Self {
            simulator,
            hp: player.current_hp,
            max_hp,
            hp_frac: f64::from(player.current_hp) / f64::from(max_hp),
            gold: player.gold,
            free_potion_slots: player.potions.iter().filter(|slot| slot.is_none()).count(),
            deck_size: player.deck.len(),
            added_attacks,
            deck_strength,
        }
    }

    /// One card's worth to this run: the tier table, the tier derived from
    /// the registry definition for anything unlisted, plus the upgrade's
    /// premium and the duplicate discount.
    fn card_value(&self, card: &CardFingerprint) -> f64 {
        let model = card.model_id.to_string();
        let base = card_tier(&model)
            .or_else(|| {
                self.simulator
                    .content()
                    .card(&card.model_id)
                    .map(|definition| derived_tier(self.simulator.content(), definition))
            })
            // Not a registered card at all: mid-low, honest about ignorance.
            .unwrap_or(knobs::TIER_UNKNOWN_PLAYABLE);
        let copies = self
            .simulator
            .state()
            .run_player
            .deck
            .iter()
            .filter(|held| held.model_id == card.model_id)
            .count();
        let duplicate_penalty = if copies == 0 {
            0.0
        } else {
            let per_copy = if card_type_of(self.simulator, card) == Some(CardType::Power) {
                knobs::DUPLICATE_POWER_PENALTY
            } else {
                knobs::DUPLICATE_PENALTY
            };
            #[allow(clippy::cast_precision_loss, reason = "deck counts are tiny")]
            {
                per_copy * copies as f64
            }
        };
        base + knobs::UPGRADE_LEVEL_VALUE * f64::from(card.upgrade_level) - duplicate_penalty
    }

    /// Whether adding this reward card beats walking on: its value against
    /// the take bar (which rises with deck bloat), with the early-attacks
    /// bump. Positive means take.
    fn take_score(&self, card: &CardFingerprint) -> f64 {
        let bloat = self.deck_size.saturating_sub(knobs::BLOAT_DECK_SIZE);
        #[allow(clippy::cast_precision_loss, reason = "deck counts are tiny")]
        let threshold = knobs::TAKE_TIER_FLOOR + knobs::BLOAT_PENALTY_PER_CARD * bloat as f64;
        let attack_bonus = if self.added_attacks < knobs::EARLY_ATTACK_COUNT
            && card_type_of(self.simulator, card) == Some(CardType::Attack)
        {
            knobs::EARLY_ATTACK_BONUS
        } else {
            0.0
        };
        self.card_value(card) + attack_bonus - threshold
    }

    /// What removing this card is worth. Positive for the deck's dead
    /// weight — curses, statuses, basics — negative for anything real.
    fn removal_gain(&self, card: &CardFingerprint) -> f64 {
        knobs::REMOVAL_PIVOT - self.card_value(card)
    }

    /// The smith's ordering: the best card wants the upgrade, and Bash
    /// jumps the queue (see [`knobs::UPGRADE_BASH_BONUS`]).
    fn upgrade_gain(&self, card: &CardFingerprint) -> f64 {
        let bash_bonus = if card.model_id.to_string() == "CARD.BASH" {
            knobs::UPGRADE_BASH_BONUS
        } else {
            0.0
        };
        self.card_value(card) + bash_bonus
    }

    /// How hard an HP cost hurts right now.
    fn hp_loss_score(&self, amount: i32) -> f64 {
        if amount >= self.hp {
            return knobs::NEVER;
        }
        let multiplier = if self.hp_frac < knobs::EVENT_HP_LOW_FRACTION {
            knobs::EVENT_HP_LOSS_LOW_MULTIPLIER
        } else {
            1.0
        };
        -f64::from(amount) * knobs::EVENT_HP_LOSS_PER_POINT * multiplier
    }
}

fn card_type_of(simulator: &Simulator, card: &CardFingerprint) -> Option<CardType> {
    simulator
        .content()
        .card(&card.model_id)
        .map(|definition| definition.card_type)
}

/// The point the map lays out at a coordinate.
fn map_point(simulator: &Simulator, coord: MapCoord) -> Option<(&'static str, MapPointType, bool)> {
    let run = simulator.state().run.as_ref()?;
    let map = run.maps.get(run.current_act)?;
    let point = map.points.iter().find(|point| point.coord == coord)?;
    let feeds_boss = point.children.contains(&map.boss)
        || map
            .second_boss
            .is_some_and(|second| point.children.contains(&second));
    Some(("", point.point_type, feeds_boss))
}

/// One destination's worth (see the `MAP_*` knobs for each rule's why).
fn map_score(ctx: &Ctx<'_>, destination: MapCoord) -> f64 {
    let Some((_, point_type, feeds_boss)) = map_point(ctx.simulator, destination) else {
        // A coordinate the visible map does not name: the fresh act's start
        // point, scored as the plain walk it is.
        return knobs::PROCEED;
    };
    match point_type {
        MapPointType::RestSite => {
            let mut score =
                knobs::MAP_REST_BASE + knobs::MAP_REST_PER_MISSING * (1.0 - ctx.hp_frac);
            // Campfire before the boss: heal or smith on the last node.
            if feeds_boss && ctx.hp_frac < 0.9 {
                score += knobs::MAP_REST_BEFORE_BOSS;
            }
            score
        }
        MapPointType::Treasure => knobs::MAP_TREASURE,
        MapPointType::Ancient => knobs::MAP_ANCIENT,
        MapPointType::Unknown => {
            knobs::MAP_UNKNOWN
                + if ctx.hp_frac >= 0.6 {
                    knobs::MAP_UNKNOWN_HEALTHY_BONUS
                } else {
                    0.0
                }
        }
        MapPointType::Shop => {
            knobs::MAP_SHOP
                + if ctx.gold >= knobs::MAP_SHOP_RICH_GOLD {
                    knobs::MAP_SHOP_RICH_BONUS
                } else {
                    0.0
                }
        }
        MapPointType::Monster => {
            knobs::MAP_MONSTER
                - if ctx.hp_frac < knobs::MAP_MONSTER_HURT_FRACTION {
                    knobs::MAP_MONSTER_HURT_PENALTY
                } else {
                    0.0
                }
        }
        MapPointType::Elite => {
            if ctx.hp_frac >= knobs::MAP_ELITE_HP_FRACTION
                && ctx.deck_strength >= knobs::MAP_ELITE_DECK_STRENGTH
            {
                knobs::MAP_ELITE_STRONG
            } else {
                knobs::MAP_ELITE_WEAK
            }
        }
        MapPointType::Boss => knobs::MAP_BOSS,
    }
}

/// A rolled or var-carried event amount, estimated. `unknown` is the
/// caller's pessimism for a var this policy cannot read.
fn amount_estimate(ctx: &Ctx<'_>, amount: EventAmount, unknown: i32) -> i32 {
    match amount {
        EventAmount::Fixed(value) => value,
        EventAmount::MaxHpLessOne => ctx.max_hp - 1,
        EventAmount::MaxHpFraction {
            numerator,
            denominator,
        } => {
            if denominator == 0 {
                unknown
            } else {
                ctx.max_hp * numerator / denominator
            }
        }
        EventAmount::Rolled { min, max } | EventAmount::RolledOnto { base: 0, min, max } => {
            min.midpoint(max)
        }
        EventAmount::RolledOnto { base, min, max } => base + min.midpoint(max),
        EventAmount::RolledAround { base, .. } => base,
        EventAmount::RolledOffOf { base, min, max } => base - min.midpoint(max),
        EventAmount::MissingHp => ctx.max_hp - ctx.hp,
        EventAmount::Var(_) => unknown,
    }
}

/// One event effect's worth. The catch-all is zero: an effect this policy
/// does not price neither lures nor scares it, and the option's other
/// effects (plus the proceed baseline) decide.
#[allow(clippy::too_many_lines, reason = "one arm per effect family")]
fn event_effect_score(ctx: &Ctx<'_>, effect: &EventEffect, depth: u32) -> f64 {
    match effect {
        EventEffect::GainGold(amount) => {
            f64::from(amount_estimate(ctx, *amount, 0)) * knobs::EVENT_GOLD_PER_POINT
        }
        EventEffect::LoseGold(amount) => {
            -f64::from(amount_estimate(ctx, *amount, 0)) * knobs::EVENT_GOLD_PER_POINT
        }
        EventEffect::Heal(amount) => {
            let missing = ctx.max_hp - ctx.hp;
            let healed = amount_estimate(ctx, *amount, ctx.max_hp / 3).min(missing);
            f64::from(healed) * knobs::EVENT_HEAL_PER_POINT
        }
        EventEffect::GainMaxHp(amount) => f64::from(*amount) * knobs::EVENT_MAX_HP_PER_POINT,
        EventEffect::LoseMaxHp(amount) => {
            -f64::from(amount_estimate(ctx, *amount, 7)) * knobs::EVENT_MAX_HP_LOSS_PER_POINT
        }
        EventEffect::LoseMaxHpAndUpgrade { .. } => knobs::EVENT_MAX_HP_FOR_UPGRADE,
        EventEffect::LoseHp(amount) => {
            let cost = amount_estimate(
                ctx,
                *amount,
                ctx.max_hp / knobs::EVENT_UNKNOWN_HP_COST_FRACTION,
            );
            ctx.hp_loss_score(cost.max(0))
        }
        EventEffect::ImmerseInTheBaths { damage } => ctx.hp_loss_score((*damage).max(0)),
        EventEffect::ObtainRelic(_)
        | EventEffect::ObtainRelicWithCard { .. }
        | EventEffect::ObtainSeaGlass { .. }
        | EventEffect::ObtainRolledRelic
        | EventEffect::ObtainRelicRolledFrom { .. }
        | EventEffect::ObtainRelicAndRandomCard { .. }
        | EventEffect::ObtainRelicAndGeneratedRewards { .. } => knobs::EVENT_RELIC,
        EventEffect::OfferShuffledRelics { .. } => knobs::EVENT_RELIC_PAGE,
        EventEffect::AddCard(card) => {
            // A specific card forced into the deck: a curse is the classic
            // event price (EVENT.TRIAL's Doubt), a good card a real payment.
            let value = ctx.card_value(card);
            if value < 0.0 {
                value * 0.3
            } else {
                (value - knobs::TAKE_TIER_FLOOR) * 0.2
            }
        }
        EventEffect::ChooseGeneratedCards { .. }
        | EventEffect::ChooseCreatedCards { .. }
        | EventEffect::AddGeneratedCard { .. }
        | EventEffect::AddCardRolledFrom { .. }
        | EventEffect::OfferCardRewards { .. }
        | EventEffect::OfferCreatedCardRewards { .. } => knobs::EVENT_CARD_OFFER,
        EventEffect::OfferRolledPotion { .. } => {
            if ctx.free_potion_slots > 0 {
                knobs::EVENT_CARD_OFFER
            } else {
                0.0
            }
        }
        EventEffect::Rewards(rewards) => {
            #[allow(clippy::cast_precision_loss, reason = "reward counts are tiny")]
            {
                knobs::EVENT_REWARD_EACH * rewards.len().min(3) as f64
            }
        }
        EventEffect::RemoveDeckCardsForGold { count, cost } => {
            if *cost <= ctx.gold {
                #[allow(clippy::cast_precision_loss, reason = "counts are tiny")]
                {
                    knobs::EVENT_REMOVAL_EACH * (*count).min(3) as f64
                }
            } else {
                knobs::EVENT_CANNOT_AFFORD
            }
        }
        EventEffect::RemoveTheRolledCard { .. } => knobs::EVENT_REMOVE_ROLLED,
        EventEffect::UpgradeARolledCard => knobs::EVENT_UPGRADE_EACH,
        EventEffect::UpgradeShuffledDeckCards { count } => {
            #[allow(clippy::cast_precision_loss, reason = "counts are tiny")]
            {
                knobs::EVENT_UPGRADE_EACH * (*count).min(3) as f64
            }
        }
        EventEffect::DowngradeARolledCard => knobs::EVENT_DOWNGRADE,
        EventEffect::RemoveRelic(_) | EventEffect::RemoveRolledTradableRelic => {
            knobs::EVENT_LOSE_RELIC
        }
        EventEffect::DiscardRolledPotion | EventEffect::DiscardPotion { .. } => {
            knobs::EVENT_LOSE_POTION
        }
        EventEffect::TradePotionForUpgradedCards { .. } => knobs::EVENT_POTION_TRADE,
        EventEffect::GrabOffTheBelt { .. } => knobs::EVENT_GRAB_BELT,
        EventEffect::BuyFromTheBag { cost, .. } => {
            if *cost <= ctx.gold {
                knobs::EVENT_BAG_BUY
            } else {
                knobs::EVENT_CANNOT_AFFORD
            }
        }
        EventEffect::SetPage(options) => {
            page_score(ctx, options, depth) * knobs::EVENT_PAGE_DISCOUNT
        }
        EventEffect::SetShuffledPage { options, .. } => {
            page_score(ctx, options, depth) * knobs::EVENT_SHUFFLED_PAGE_DISCOUNT
        }
        EventEffect::UnlessTheFightTimedOut(effects) => effects
            .iter()
            .map(|inner| event_effect_score(ctx, inner, depth))
            .sum(),
        EventEffect::EnterCombatWithoutLeaving { .. } => {
            if ctx.hp_frac < knobs::EVENT_HP_LOW_FRACTION {
                knobs::EVENT_FIGHT_LOW_HP
            } else {
                knobs::EVENT_FIGHT
            }
        }
        EventEffect::PlayTheCrystalSphere { .. } => knobs::EVENT_CRYSTAL_SPHERE,
        // The page advertises what a hold costs, so it is priced as the HP it
        // is rather than as a constant standing in for one — which also means
        // a hold that would kill reads as `NEVER` rather than as cheap.
        EventEffect::HoldOnToTheBridge { cost } => ctx.hp_loss_score(*cost),
        EventEffect::Unsupported(_) => knobs::NEVER,
        EventEffect::WinRun => -knobs::NEVER,
        // `Run` bodies, relic trades, rolled pages, stream reads, and
        // anything unpriced: zero, so the option's other effects and the
        // proceed baseline decide. Deliberate: pricing opaque command bodies
        // would be pretending to knowledge this table does not have.
        _ => 0.0,
    }
}

/// The best line a page offers, for pricing the option that opens it.
fn page_score(ctx: &Ctx<'_>, options: &[EventOption], depth: u32) -> f64 {
    if depth >= knobs::EVENT_PAGE_DEPTH {
        return 0.0;
    }
    options
        .iter()
        .filter(|option| !poisoned_event_option(ctx.simulator.content(), option))
        .map(|option| event_option_score(ctx, option, depth + 1))
        .fold(0.0, f64::max)
}

/// One event option's worth: the sum of its effects, with the proceed
/// baseline keeping "walk away" ahead of any all-negative page.
fn event_option_score(ctx: &Ctx<'_>, option: &EventOption, depth: u32) -> f64 {
    if option.was_chosen {
        return knobs::EVENT_OPTION_SPENT;
    }
    let effects: f64 = option
        .effects
        .iter()
        .map(|effect| event_effect_score(ctx, effect, depth))
        .sum();
    if option.is_proceed {
        effects + knobs::EVENT_PROCEED_BASELINE
    } else {
        effects
    }
}

/// The suspended card-selection screen's purpose, wherever it is held.
fn choice_purpose(simulator: &Simulator) -> Option<&ChoicePurpose> {
    crate::policy::open_choice(simulator).map(|choice| &choice.purpose)
}

/// A card-selection answer's worth, by what the screen was opened for.
fn choose_cards_score(ctx: &Ctx<'_>, cards: &[sts2_engine::CardHandle]) -> f64 {
    let fingerprints = || cards.iter().map(|handle| &handle.fingerprint);
    match choice_purpose(ctx.simulator) {
        // A claimed card reward: the claim gate already decided taking beats
        // skipping, so commit to the best card. Canceling would leave the
        // reward on the table and loop the policy against it.
        Some(ChoicePurpose::AddRewardCard { .. }) => {
            if cards.is_empty() {
                knobs::SCREEN_NEVER_CANCEL
            } else {
                knobs::SCREEN_COMMIT + fingerprints().map(|card| ctx.card_value(card)).sum::<f64>()
            }
        }
        // The smith: upgrade the card the table likes best. Backing out
        // leaves the rest site standing and loops.
        Some(ChoicePurpose::RestSmith { .. }) => {
            if cards.is_empty() {
                knobs::SCREEN_NEVER_CANCEL
            } else {
                knobs::SCREEN_COMMIT
                    + fingerprints()
                        .map(|card| ctx.upgrade_gain(card))
                        .sum::<f64>()
            }
        }
        // Removal screens: the shop's removal (nothing was paid until a card
        // is chosen — canceling re-offers the purchase and loops), a
        // removal reward, an event's remove-for-gold, and the cook. All
        // commit, all pick the deck's worst.
        Some(
            ChoicePurpose::RemoveDeckCardAtShop { .. }
            | ChoicePurpose::RemoveDeckCardsForGold { .. }
            | ChoicePurpose::RemoveDeckCardForReward { .. }
            | ChoicePurpose::RestCook { .. },
        ) => {
            if cards.is_empty() {
                knobs::SCREEN_NEVER_CANCEL
            } else {
                knobs::SCREEN_COMMIT
                    + fingerprints()
                        .map(|card| ctx.removal_gain(card))
                        .sum::<f64>()
            }
        }
        // An event's pick-to-deck: skipping is a real option (the event's
        // page has moved on either way), so all-bad offers are declined.
        Some(ChoicePurpose::AddChosenCardsToDeck) => {
            fingerprints().map(|card| ctx.take_score(card)).sum::<f64>()
        }
        // A command-driven screen (event transforms and kin), or a screen
        // whose purpose is not visible: over the run deck it is a
        // remove/transform shape, so pick the worst cards; over an offer it
        // is a gain, so pick the best or nothing.
        Some(ChoicePurpose::RunCommands { .. }) | None => {
            let over_deck = matches!(
                ctx.simulator.decision(),
                DecisionContext::ChooseCards {
                    identity: sts2_engine::ChoiceIdentity::DeckCard,
                    ..
                }
            );
            if over_deck {
                if cards.is_empty() {
                    knobs::SCREEN_NEVER_CANCEL
                } else {
                    knobs::SCREEN_COMMIT
                        + fingerprints()
                            .map(|card| ctx.removal_gain(card))
                            .sum::<f64>()
                }
            } else {
                fingerprints().map(|card| ctx.take_score(card)).sum::<f64>()
            }
        }
    }
}

/// A card reward's button beside its cards, by which button it is
/// (`CardRewardAlternative.Generate` writes the ids).
///
/// The two registered buttons point opposite ways and cannot share a score:
/// the reroll costs nothing and the sacrifice throws the reward away. An id
/// this baseline does not know is priced as the sacrifice, which is the
/// conservative half.
fn alternative_score(option_id: &str) -> f64 {
    match option_id {
        "reroll" => knobs::CHOICE_REROLL,
        _ => knobs::CHOICE_SACRIFICE,
    }
}

/// A claim's worth, by what the reward is (`RewardFingerprint.reward_type`
/// strings as the engine writes them).
fn claim_score(ctx: &Ctx<'_>, fingerprint: &sts2_engine::RewardFingerprint) -> f64 {
    match fingerprint.reward_type.as_str() {
        "gold" => knobs::CLAIM_GOLD,
        "relic" => knobs::CLAIM_RELIC,
        "potion" => {
            if ctx.free_potion_slots > 0 {
                knobs::CLAIM_POTION
            } else {
                knobs::CLAIM_POTION_FULL_BELT
            }
        }
        "card_removal" => {
            let worth_removing = ctx
                .simulator
                .state()
                .run_player
                .deck
                .iter()
                .any(|card| ctx.removal_gain(card) > 0.0);
            if worth_removing {
                knobs::CLAIM_REMOVAL_GOOD
            } else {
                knobs::CLAIM_REMOVAL_BAD
            }
        }
        "card" => {
            // A handed-over card (thief loot) is taken as it stands.
            if fingerprint.special_card.is_some() {
                return knobs::CLAIM_SPECIAL_CARD;
            }
            // The offer is on the fingerprint before the claim, so the
            // take/skip decision happens here — never claim-then-cancel,
            // which would re-offer the reward forever.
            let best = fingerprint
                .offered_cards
                .iter()
                .map(|card| ctx.take_score(card))
                .fold(f64::NEG_INFINITY, f64::max);
            if best > 0.0 {
                knobs::CLAIM_CARD_BASE + best
            } else {
                knobs::CLAIM_CARD_ALL_BAD
            }
        }
        _ => knobs::CLAIM_UNKNOWN,
    }
}

/// A shop purchase's worth. Prices come off the shop decision's own offers;
/// a `BuyShopItem` outside a shop decision (an event's shelf) is unpriced
/// and declined.
fn shop_score(ctx: &Ctx<'_>, offer_index: usize, item: &ShopItem) -> f64 {
    let DecisionContext::Shop { offers, .. } = ctx.simulator.decision() else {
        return knobs::SHOP_SKIP;
    };
    let Some(offer) = offers.iter().find(|offer| offer.index == offer_index) else {
        return knobs::SHOP_SKIP;
    };
    // Legality already filtered unaffordable offers; the caps below are
    // taste, not affordability.
    match item {
        ShopItem::Relic(_) => {
            if offer.price <= knobs::SHOP_RELIC_MAX_PRICE {
                knobs::SHOP_RELIC
            } else {
                knobs::SHOP_SKIP
            }
        }
        ShopItem::Potion(_) => {
            if offer.price <= knobs::SHOP_POTION_MAX_PRICE {
                knobs::SHOP_POTION
            } else {
                knobs::SHOP_SKIP
            }
        }
        ShopItem::Card(card) => {
            let take = ctx.take_score(card);
            if take > 0.0 && offer.price <= knobs::SHOP_CARD_MAX_PRICE {
                knobs::SHOP_CARD_BASE + take * 0.1
            } else {
                knobs::SHOP_SKIP
            }
        }
    }
}

/// One rest-site option's worth (see the `REST_*` knobs).
fn rest_score(ctx: &Ctx<'_>, option: RestSiteOption) -> f64 {
    match option {
        RestSiteOption::Heal => {
            if ctx.hp_frac < knobs::REST_HP_FRACTION {
                knobs::REST_HEAL_BASE + knobs::REST_HEAL_PER_MISSING * (1.0 - ctx.hp_frac)
            } else {
                knobs::REST_HEAL_HEALTHY
            }
        }
        RestSiteOption::Smith => knobs::REST_SMITH,
        RestSiteOption::Dig => knobs::REST_DIG,
        RestSiteOption::Hatch => knobs::REST_HATCH,
        RestSiteOption::Lift => knobs::REST_LIFT,
        RestSiteOption::Kindle => knobs::REST_KINDLE,
        RestSiteOption::Clone => knobs::REST_CLONE,
        RestSiteOption::Cook => {
            let mut gains: Vec<f64> = ctx
                .simulator
                .state()
                .run_player
                .deck
                .iter()
                .map(|card| ctx.removal_gain(card))
                .collect();
            gains.sort_by(|left, right| right.total_cmp(left));
            // Two cards leave the deck for nine max HP: worth it exactly
            // when the two worst cards are worth losing.
            if gains.len() >= 2 && gains[0] > 0.0 && gains[1] > 0.0 {
                knobs::REST_COOK_GOOD
            } else {
                knobs::REST_COOK_BAD
            }
        }
    }
}

/// The argmax's scorer: one number per legal out-of-combat action.
fn score_action(ctx: &Ctx<'_>, action: &Action) -> f64 {
    match action {
        Action::ChooseMap { destination } => map_score(ctx, *destination),
        Action::ChooseEvent { index, .. } => {
            let DecisionContext::Event { options, .. } = ctx.simulator.decision() else {
                return 0.0;
            };
            options
                .get(*index)
                .map_or(0.0, |option| event_option_score(ctx, option, 0))
        }
        Action::BuyShopItem {
            offer_index,
            fingerprint,
        } => shop_score(ctx, *offer_index, fingerprint),
        Action::BuyCardRemoval => {
            // Only worth opening when the deck holds something worth paying
            // to lose; the screen, once open, commits (see
            // `choose_cards_score`).
            let worth_removing = ctx
                .simulator
                .state()
                .run_player
                .deck
                .iter()
                .any(|card| ctx.removal_gain(card) > 0.0);
            if worth_removing {
                knobs::SHOP_REMOVAL
            } else {
                knobs::SHOP_SKIP
            }
        }
        Action::RestOption { option, .. } => rest_score(ctx, *option),
        Action::TakeTreasure { .. } => knobs::TAKE_TREASURE,
        Action::ClaimReward { fingerprint, .. } => claim_score(ctx, fingerprint),
        Action::ChooseCards { cards, .. } => choose_cards_score(ctx, cards),
        Action::ChooseAlternative { option_id, .. } => alternative_score(option_id),
        Action::Proceed | Action::AdvanceAct => knobs::PROCEED,
        // The sphere offers only cells (plus the belt): first hidden cell,
        // big tool — deterministic, and the board pays out either way.
        Action::UncoverCrystalSphere { .. } => 1.0,
        Action::UsePotion { .. } => knobs::USE_POTION_OOC,
        Action::DiscardPotion { .. } => knobs::DISCARD_POTION,
        // Unreachable out of combat; scored so the match is total.
        Action::PlayCard { .. } | Action::EndTurn { .. } => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two buttons a card reward carries point opposite ways, so the
    /// teacher's softmax must not give them the same mass. A pick scores
    /// [`knobs::SCREEN_COMMIT`] plus the card's own value, so these are the
    /// picks the reroll must beat and the ones it must lose to.
    #[test]
    fn the_reroll_and_the_sacrifice_are_scored_apart() {
        let under_the_floor = knobs::SCREEN_COMMIT + knobs::TAKE_TIER_FLOOR - 1.0;
        let over_the_floor = knobs::SCREEN_COMMIT + knobs::TAKE_TIER_FLOOR + 1.0;
        let reroll = alternative_score("reroll");
        assert!(
            reroll > under_the_floor,
            "a free redraw beats an offer with nothing worth taking on it"
        );
        assert!(
            reroll < over_the_floor,
            "and loses to an offer that has something"
        );
        let sacrifice = alternative_score("sacrifice");
        assert!(
            sacrifice < under_the_floor && sacrifice < 0.0,
            "half a relic for a whole reward loses to every pick: {sacrifice}"
        );
        assert_eq!(
            alternative_score("a_button_this_baseline_has_not_met").total_cmp(&sacrifice),
            std::cmp::Ordering::Equal,
            "and an unknown button is priced as the conservative half"
        );
    }

    #[test]
    fn the_tier_table_orders_the_obvious() {
        // Sanity on the endpoints the rules lean on hardest.
        assert!(card_tier("CARD.POMMEL_STRIKE") > card_tier("CARD.STRIKE_IRONCLAD"));
        assert!(card_tier("CARD.OFFERING") > card_tier("CARD.HAVOC"));
        assert!(card_tier("CARD.STRIKE_IRONCLAD").unwrap() < knobs::REMOVAL_PIVOT);
        assert!(card_tier("CARD.SHRUG_IT_OFF").unwrap() > knobs::TAKE_TIER_FLOOR);
    }

    #[test]
    fn every_tiered_card_is_registered_content() {
        // A typo in the table would silently fall back to the unknown
        // default; pin every id to the actual registry.
        let registry = sts2_content::standard_registry();
        let mut checked = 0;
        for id in registry.registered_model_ids() {
            let name = id.to_string();
            if card_tier(&name).is_some() {
                checked += 1;
            }
        }
        // Every table entry resolves: count the table by probing the
        // registry the other way around.
        let table_size = {
            // The table is a match, so enumerate through the registry: every
            // entry must be a registered card id.
            let registered: std::collections::HashSet<String> = registry
                .registered_model_ids()
                .into_iter()
                .map(|id| id.to_string())
                .collect();
            let mut missing = Vec::new();
            for name in TIERED_IDS {
                if !registered.contains(*name) {
                    missing.push(*name);
                }
            }
            assert!(missing.is_empty(), "unregistered tier entries: {missing:?}");
            TIERED_IDS.len()
        };
        assert_eq!(checked, table_size, "every tiered id is registered");
    }

    #[test]
    fn every_registered_card_derives_a_tier_on_the_scale() {
        let registry = sts2_content::standard_registry();
        let mut playable = 0;
        for id in registry.registered_model_ids() {
            let Some(definition) = registry.card(&id) else {
                continue;
            };
            let tier = derived_tier(&registry, definition);
            assert!(tier.is_finite(), "{id}: {tier}");
            let fixed = match definition.card_type {
                CardType::Curse => Some(knobs::TIER_CURSE),
                CardType::Status => Some(knobs::TIER_STATUS),
                _ if definition.unplayable => Some(knobs::TIER_STATUS),
                _ => None,
            };
            if let Some(want) = fixed {
                assert_eq!(tier.to_bits(), want.to_bits(), "{id}");
                continue;
            }
            let starter = [
                card_tier("CARD.STRIKE_IRONCLAD").unwrap(),
                card_tier("CARD.DEFEND_IRONCLAD").unwrap(),
            ]
            .iter()
            .any(|value| value.to_bits() == tier.to_bits());
            assert!(
                (knobs::DERIVED_MIN..=knobs::DERIVED_MAX).contains(&tier) || starter,
                "{id}: {tier} is off the scale"
            );
            playable += 1;
        }
        assert!(playable > 300, "the registry's playable cards: {playable}");
    }

    #[test]
    fn derived_tiers_spread_so_a_reward_screen_is_not_a_tie() {
        // Cards without explicit tiers should still receive distinct scores.
        let registry = sts2_content::standard_registry();
        let mut distinct = std::collections::BTreeSet::new();
        for id in registry.registered_model_ids() {
            if let Some(definition) = registry.card(&id)
                && card_tier(&id.to_string()).is_none()
                && !matches!(definition.card_type, CardType::Curse | CardType::Status)
                && !definition.unplayable
            {
                distinct.insert(derived_tier(&registry, definition).to_bits());
            }
        }
        assert!(
            distinct.len() >= 40,
            "unlisted playable cards take {} distinct tiers",
            distinct.len()
        );
    }

    #[test]
    fn every_characters_starters_sit_below_the_removal_pivot() {
        // Basic Strikes and Defends of every class score as the Ironclad's
        // do, so removal prefers them whoever is playing.
        use sts2_engine::{CardRarity, CardTag};
        let registry = sts2_content::standard_registry();
        let mut starters = 0;
        for id in registry.registered_model_ids() {
            let Some(definition) = registry.card(&id) else {
                continue;
            };
            if definition.rarity == CardRarity::Basic
                && (definition.tags.contains(&CardTag::Strike)
                    || definition.tags.contains(&CardTag::Defend))
            {
                assert!(
                    derived_tier(&registry, definition) < knobs::REMOVAL_PIVOT,
                    "{id}"
                );
                starters += 1;
            }
        }
        assert!(
            starters >= 8,
            "four or more classes' Strike and Defend: {starters}"
        );
    }

    #[test]
    fn derived_tiers_track_the_hand_table() {
        // The derived formula must put an unlisted card on the table's
        // scale, or off-class cards win or lose reward screens for the wrong
        // reason. Judged against the table it will never be asked about: the
        // Ironclad cards, where both answers exist. Run with --nocapture to
        // see the comparison when tuning the knobs.
        let registry = sts2_content::standard_registry();
        let mut rows = Vec::new();
        for name in TIERED_IDS {
            let id: sts2_core::ModelId = name.parse().unwrap();
            let definition = registry.card(&id).unwrap();
            let table = card_tier(name).unwrap();
            let derived = derived_tier(&registry, definition);
            rows.push((name, table, derived));
        }
        rows.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (name, table, derived) in &rows {
            println!(
                "{name:32} table {table:5.1}  derived {derived:5.1}  {:+5.1}",
                derived - table
            );
        }
        #[allow(clippy::cast_precision_loss, reason = "a table of dozens")]
        let mean_abs_error = rows
            .iter()
            .map(|(_, table, derived)| (derived - table).abs())
            .sum::<f64>()
            / rows.len() as f64;
        let mut agree = 0;
        let mut pairs = 0;
        for (i, a) in rows.iter().enumerate() {
            for b in &rows[i + 1..] {
                if a.1.to_bits() != b.1.to_bits() {
                    pairs += 1;
                    if (a.2 < b.2) == (a.1 < b.1) {
                        agree += 1;
                    }
                }
            }
        }
        #[allow(clippy::cast_precision_loss, reason = "a table of dozens")]
        let concordance = f64::from(agree) / f64::from(pairs);
        println!("mean |error| {mean_abs_error:.3}, pairwise order agreement {concordance:.4}");
        // The bars the knobs are tuned against: loosening either one is
        // giving the formula's agreement with the table away. Both read the
        // card bodies in the sibling simulator, so a content change there
        // moves them with no change here.
        assert!(mean_abs_error < 8.55, "mean |error| {mean_abs_error:.3}");
        assert!(concordance > 0.67, "order agreement {concordance:.4}");
    }

    /// Every registered card's derived tier, by model id, for the tests that
    /// name particular cards.
    fn derived_tiers() -> std::collections::BTreeMap<String, f64> {
        let registry = sts2_content::standard_registry();
        registry
            .registered_model_ids()
            .into_iter()
            .filter_map(|id| {
                registry
                    .card(&id)
                    .map(|definition| (id.to_string(), derived_tier(&registry, definition)))
            })
            .collect()
    }

    #[test]
    fn a_loop_runs_its_body_as_many_times_as_it_says() {
        use sts2_engine::{Amount, Command, CreatureSelector, DamageCmd, ValueProps};
        let registry = sts2_content::standard_registry();
        let swing = |amount| {
            Command::Damage(DamageCmd::Attack {
                amount: Amount::Fixed(amount),
                hits: Amount::Fixed(1),
                target: CreatureSelector::Target,
                props: ValueProps::NONE,
            })
        };
        let mut looped = CardBody::default();
        looped.walk(
            &[Command::Repeat {
                times: Amount::Fixed(3),
                then: vec![swing(5)],
            }],
            Pass::root(&registry, false),
        );
        let mut written_out = CardBody::default();
        written_out.walk(
            &[swing(5), swing(5), swing(5)],
            Pass::root(&registry, false),
        );
        assert!((looped.damage - written_out.damage).abs() < 1e-9);
        assert!(looped.damage > 0.0);
    }

    /// A block command aimed at the owner, for the walker tests below.
    fn block(amount: i32) -> sts2_engine::Command {
        sts2_engine::Command::Creature(sts2_engine::CreatureCmd::GainBlock {
            target: sts2_engine::CreatureSelector::Owner,
            amount: sts2_engine::Amount::Fixed(amount),
            props: sts2_engine::ValueProps::NONE,
        })
    }

    #[test]
    fn a_bare_guard_is_worth_its_whole_body() {
        use sts2_engine::{Command, Condition};
        let registry = sts2_content::standard_registry();
        let mut guarded = CardBody::default();
        guarded.walk(
            &[Command::When {
                condition: Condition::OwnerLostHpThisTurn,
                then: vec![block(10)],
                otherwise: Vec::new(),
            }],
            Pass::root(&registry, false),
        );
        assert!((guarded.block - 10.0).abs() < 1e-9, "{}", guarded.block);
    }

    #[test]
    fn both_arms_of_a_choice_are_walked() {
        use sts2_engine::{Command, Condition};
        let registry = sts2_content::standard_registry();
        let mut two_armed = CardBody::default();
        two_armed.walk(
            &[Command::When {
                condition: Condition::OwnerLostHpThisTurn,
                then: vec![block(10)],
                otherwise: vec![block(4)],
            }],
            Pass::root(&registry, false),
        );
        let arm = knobs::DERIVED_WHEN_ARM_WEIGHT;
        assert!(
            (two_armed.block - (10.0 + 4.0) * arm).abs() < 1e-9,
            "{}",
            two_armed.block
        );
    }

    #[test]
    fn an_orb_body_is_worth_taking() {
        // Both of the Defect's starters are orb bodies and nothing else; a
        // formula that reads no orbs leaves them under the take bar and the
        // character never assembles a queue.
        let tiers = derived_tiers();
        for id in ["CARD.ZAP", "CARD.DUALCAST"] {
            assert!(tiers[id] >= knobs::TAKE_TIER_FLOOR, "{id}: {}", tiers[id]);
        }
    }

    #[test]
    fn a_powers_stacks_are_worth_something() {
        // Poison 5 is not poison 1, and a power laid on the whole enemy side
        // beats one that only lends its owner a stat.
        let tiers = derived_tiers();
        assert!(
            tiers["CARD.DEADLY_POISON"] >= knobs::TAKE_TIER_FLOOR,
            "{}",
            tiers["CARD.DEADLY_POISON"]
        );
        assert!(
            tiers["CARD.NOXIOUS_FUMES"] > tiers["CARD.FOOTWORK"],
            "fumes {} against footwork {}",
            tiers["CARD.NOXIOUS_FUMES"],
            tiers["CARD.FOOTWORK"]
        );
    }

    #[test]
    fn a_star_cost_is_paid_for() {
        // The Regent charges a second resource the energy cost says nothing
        // about. Whatever else is written on a card, the same card also
        // charging stars is worth less than the one that does not.
        let registry = sts2_content::standard_registry();
        let mut compared = 0;
        for id in registry.registered_model_ids() {
            let Some(definition) = registry.card(&id) else {
                continue;
            };
            if definition.star_cost > 0 || definition.costs_star_x {
                continue;
            }
            let free = derived_tier(&registry, definition);
            // Only where the body is worth something and the scale's ends are
            // not already holding the answer still.
            let mut bare = definition.clone();
            bare.effects = Vec::new();
            if free >= knobs::DERIVED_MAX || derived_tier(&registry, &bare) >= free {
                continue;
            }
            let mut charged = definition.clone();
            charged.star_cost = 3;
            let paid = derived_tier(&registry, &charged);
            assert!(paid < free, "{id}: {paid} against {free}");
            compared += 1;
        }
        assert!(compared > 100, "only {compared} cards compared");
    }

    #[test]
    fn a_power_reads_the_way_the_registry_says_and_not_the_side_it_lands_on() {
        // A No Draw the owner takes on themselves is the price of the body
        // around it, not a buff for standing there. Read off powers the
        // registry declares, with no listener bodies of their own, so what is
        // measured is the sign and nothing else.
        use sts2_engine::{Amount, Command, CreatureSelector, PowerCmd, PowerType};
        let registry = sts2_content::standard_registry();
        let of_type = |wanted| {
            registry
                .registered_model_ids()
                .into_iter()
                .find(|id| {
                    registry.power(id).is_some_and(|model| {
                        model.hooks.is_empty() && model.type_for_amount(1) == wanted
                    })
                })
                .expect("the registry declares powers of both types")
        };
        let apply = |power, target| {
            [Command::Power(PowerCmd::Apply {
                power,
                target,
                applier: None,
                amount: Amount::Fixed(1),
                marks: None,
            })]
        };
        let walked = |commands: &[Command]| {
            let mut body = CardBody::default();
            body.walk(commands, Pass::root(&registry, false));
            body
        };
        let debuff = of_type(PowerType::Debuff);
        let buff = of_type(PowerType::Buff);
        // On the owner, a debuff costs where a buff pays.
        assert!(walked(&apply(debuff.clone(), CreatureSelector::Owner)).self_buffs < 0.0);
        assert!(walked(&apply(buff.clone(), CreatureSelector::Owner)).self_buffs > 0.0);
        // On an enemy, the same two the other way about.
        assert!(walked(&apply(debuff, CreatureSelector::Target)).enemy_debuffs > 0.0);
        assert!(walked(&apply(buff, CreatureSelector::Target)).enemy_debuffs < 0.0);
    }

    #[test]
    fn a_card_that_may_not_be_playable_does_not_top_the_scale() {
        // A Grand Finale is 60 damage to the whole side at no cost, and
        // playable only on an empty draw pile.
        let tiers = derived_tiers();
        let finale = tiers["CARD.GRAND_FINALE"];
        let best = tiers.values().copied().fold(f64::MIN, f64::max);
        assert!(finale < best, "{finale} is still the best of {best}");
    }

    /// The table's ids, listed once for the registration test. Kept beside
    /// the table; the test above fails if the two drift apart.
    const TIERED_IDS: &[&str] = &[
        "CARD.STRIKE_IRONCLAD",
        "CARD.DEFEND_IRONCLAD",
        "CARD.BASH",
        "CARD.POMMEL_STRIKE",
        "CARD.SHRUG_IT_OFF",
        "CARD.THUNDERCLAP",
        "CARD.IRON_WAVE",
        "CARD.ANGER",
        "CARD.PERFECTED_STRIKE",
        "CARD.SWORD_BOOMERANG",
        "CARD.TWIN_STRIKE",
        "CARD.HEADBUTT",
        "CARD.CINDER",
        "CARD.BREAKTHROUGH",
        "CARD.ARMAMENTS",
        "CARD.SETUP_STRIKE",
        "CARD.MOLTEN_FIST",
        "CARD.TREMBLE",
        "CARD.BLOOD_WALL",
        "CARD.BLOODLETTING",
        "CARD.TRUE_GRIT",
        "CARD.HAVOC",
        "CARD.BODY_SLAM",
        "CARD.BATTLE_TRANCE",
        "CARD.UPPERCUT",
        "CARD.HEMOKINESIS",
        "CARD.INFLAME",
        "CARD.BLUDGEON",
        "CARD.WHIRLWIND",
        "CARD.STONE_ARMOR",
        "CARD.UNRELENTING",
        "CARD.DISMANTLE",
        "CARD.AGGRESSION",
        "CARD.HOWL_FROM_BEYOND",
        "CARD.FEEL_NO_PAIN",
        "CARD.DOMINATE",
        "CARD.RAMPAGE",
        "CARD.STAMPEDE",
        "CARD.VICIOUS",
        "CARD.STOMP",
        "CARD.PILLAGE",
        "CARD.HELLRAISER",
        "CARD.BULLY",
        "CARD.FIGHT_ME",
        "CARD.TAUNT",
        "CARD.EVIL_EYE",
        "CARD.BURNING_PACT",
        "CARD.DRUM_OF_BATTLE",
        "CARD.JUGGLING",
        "CARD.ONE_TWO_PUNCH",
        "CARD.RUPTURE",
        "CARD.SECOND_WIND",
        "CARD.SPITE",
        "CARD.INFERNAL_BLADE",
        "CARD.EXPECT_A_FIGHT",
        "CARD.ASHEN_STRIKE",
        "CARD.UNMOVABLE",
        "CARD.COLOSSUS",
        "CARD.FLAME_BARRIER",
        "CARD.INFERNO",
        "CARD.RAGE",
        "CARD.OFFERING",
        "CARD.FEED",
        "CARD.PYRE",
        "CARD.IMPERVIOUS",
        "CARD.DEMON_FORM",
        "CARD.FIEND_FIRE",
        "CARD.NOT_YET",
        "CARD.THRASH",
        "CARD.JUGGERNAUT",
        "CARD.MANGLE",
        "CARD.BARRICADE",
        "CARD.PACTS_END",
        "CARD.CRUELTY",
        "CARD.DARK_EMBRACE",
        "CARD.CRIMSON_MANTLE",
        "CARD.TEAR_ASUNDER",
        "CARD.CONFLAGRATION",
        "CARD.STOKE",
        "CARD.BRAND",
        "CARD.CASCADE",
        "CARD.PRIMAL_FORCE",
        "CARD.GIANT_ROCK",
        "CARD.BREAK",
        "CARD.CORRUPTION",
    ];
}
