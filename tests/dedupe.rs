//! Action-class dedupe: which copy of a card
//! carries a play is not part of what the play does, so the search branches
//! over gameplay classes, not instances — and the potion shadow price keeps
//! a combat-scoped objective from drinking the belt for free.

use alphaspire::objective::{CombatStrength, Objective};
use alphaspire::search::canonical_actions;
use sts2_engine::{Action, CardFingerprint, PileName, ScenarioBuilder, Simulator};

const SEED: &str = "NLD6VZXP94";

fn id(value: &str) -> sts2_core::ModelId {
    value.parse().unwrap()
}

fn card(model: &str) -> CardFingerprint {
    CardFingerprint::base(id(model))
}

fn fight(hand: &[(u64, &str)]) -> Simulator {
    let mut scenario = ScenarioBuilder::new(SEED, id("ENCOUNTER.SHRINKER_BEETLE_WEAK"))
        .player_hp(80, 80)
        .energy(9, 9)
        .turn(1, 1)
        .enemy(1, id("MONSTER.SHRINKER_BEETLE"), 39, 39, "SHRINKER_MOVE");
    for (card_id, model) in hand {
        scenario = scenario.card(PileName::Hand, *card_id, card(model), 1);
    }
    scenario.build(sts2_content::standard_registry()).unwrap()
}

#[test]
fn three_strikes_are_one_class_and_an_upgrade_is_another() {
    // Three same-print Strikes, one upgraded Strike, one Defend: the raw
    // legal list offers a play per copy, the canonical list one per class.
    let mut upgraded = card("CARD.STRIKE_IRONCLAD");
    upgraded.upgrade_level = 1;
    let simulator = ScenarioBuilder::new(SEED, id("ENCOUNTER.SHRINKER_BEETLE_WEAK"))
        .player_hp(80, 80)
        .energy(9, 9)
        .turn(1, 1)
        .enemy(1, id("MONSTER.SHRINKER_BEETLE"), 39, 39, "SHRINKER_MOVE")
        .card(PileName::Hand, 10, card("CARD.STRIKE_IRONCLAD"), 1)
        .card(PileName::Hand, 11, card("CARD.STRIKE_IRONCLAD"), 1)
        .card(PileName::Hand, 12, card("CARD.STRIKE_IRONCLAD"), 1)
        .card(PileName::Hand, 13, upgraded, 1)
        .card(PileName::Hand, 14, card("CARD.DEFEND_IRONCLAD"), 1)
        .build(sts2_content::standard_registry())
        .unwrap();
    let raw_plays = simulator
        .legal_actions()
        .iter()
        .filter(|action| matches!(action, Action::PlayCard { .. }))
        .count();
    assert_eq!(raw_plays, 5, "one raw play per copy");
    let canonical = canonical_actions(simulator.legal_actions());
    let canonical_plays = canonical
        .iter()
        .filter(|action| matches!(action, Action::PlayCard { .. }))
        .count();
    assert_eq!(
        canonical_plays, 3,
        "strike, strike-plus, and defend: a class per distinguishable play"
    );
    // Every representative is itself a legal action, so a script that names
    // it replays.
    for action in &canonical {
        assert!(simulator.legal_actions().contains(action));
    }
}

#[test]
fn a_choice_between_identical_copies_is_one_choice() {
    // Armaments over two same-print Strikes: the screen offers a pick per
    // copy, the canonical list one per class.
    let mut simulator = fight(&[
        (10, "CARD.ARMAMENTS"),
        (11, "CARD.STRIKE_IRONCLAD"),
        (12, "CARD.STRIKE_IRONCLAD"),
    ]);
    let play = simulator
        .legal_actions()
        .iter()
        .find(|action| {
            matches!(action, Action::PlayCard { card, .. }
                if card.fingerprint.model_id == id("CARD.ARMAMENTS"))
        })
        .cloned()
        .expect("armaments is playable");
    simulator.step_quietly(&play).unwrap();
    let raw_choices = simulator
        .legal_actions()
        .iter()
        .filter(|action| matches!(action, Action::ChooseCards { .. }))
        .count();
    let canonical_choices = canonical_actions(simulator.legal_actions())
        .iter()
        .filter(|action| matches!(action, Action::ChooseCards { .. }))
        .count();
    assert!(raw_choices > canonical_choices, "the copies collapsed");
    assert_eq!(canonical_choices, 1, "two identical Strikes are one pick");
}

#[test]
fn distinct_targets_and_prints_stay_distinct() {
    // Two enemies: a Strike at each is two classes — the target is part of
    // what the play does.
    let simulator = ScenarioBuilder::new(SEED, id("ENCOUNTER.SHRINKER_BEETLE_PAIR"))
        .player_hp(80, 80)
        .energy(9, 9)
        .turn(1, 1)
        .enemy(1, id("MONSTER.SHRINKER_BEETLE"), 39, 39, "SHRINKER_MOVE")
        .enemy(2, id("MONSTER.SHRINKER_BEETLE"), 39, 39, "SHRINKER_MOVE")
        .card(PileName::Hand, 10, card("CARD.STRIKE_IRONCLAD"), 1)
        .build(sts2_content::standard_registry())
        .unwrap();
    let canonical_plays = canonical_actions(simulator.legal_actions())
        .iter()
        .filter(|action| matches!(action, Action::PlayCard { .. }))
        .count();
    assert_eq!(canonical_plays, 2, "one class per target");
}

#[test]
fn a_held_potion_is_worth_its_shadow_price() {
    let bare = fight(&[(10, "CARD.STRIKE_IRONCLAD")]);
    let carrying = ScenarioBuilder::new(SEED, id("ENCOUNTER.SHRINKER_BEETLE_WEAK"))
        .player_hp(80, 80)
        .energy(9, 9)
        .turn(1, 1)
        .enemy(1, id("MONSTER.SHRINKER_BEETLE"), 39, 39, "SHRINKER_MOVE")
        .card(PileName::Hand, 10, card("CARD.STRIKE_IRONCLAD"), 1)
        .potions(vec![Some(id("POTION.HEALTH_POTION")), None])
        .build(sts2_content::standard_registry())
        .unwrap();
    let objective = CombatStrength::default();
    let difference = objective.peek(&carrying) - objective.peek(&bare);
    // Discounted by the turn the scenario stands on, like every other term
    // the objective prices — `combat-strength-v3` scales the whole bundle.
    let priced = objective.potion_weight / (1.0 + objective.turn_weight);
    assert!(
        (difference - priced).abs() < 1e-9,
        "one potion held, one shadow price paid: {difference} vs {priced}"
    );
}
