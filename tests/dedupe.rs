//! Action-class dedupe: which copy of a card
//! carries a play is not part of what the play does, so the search branches
//! over gameplay classes, not instances — and the potion shadow price keeps
//! a combat-scoped objective from drinking the belt for free.

use alphaspire::objective::{CombatStrength, Objective};
use alphaspire::search::{action_class, canonical_actions};
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
fn combat_class_comparisons_preserve_the_json_equivalence_and_first_representative() {
    let simulator = fight(&[(10, "CARD.STRIKE_IRONCLAD"), (11, "CARD.STRIKE_IRONCLAD")]);
    let original = simulator.legal_actions()[0].clone();
    assert!(matches!(original, Action::PlayCard { .. }));
    let base = serde_json::to_value(&original).unwrap();
    let mut actions = simulator.legal_actions().to_vec();
    // Every gameplay field remains part of a play's identity, while copy
    // ids and positions do not. Saved JSON also exercises the legacy
    // scrubber's recursion and the distinction between 0.0 and -0.0.
    for (path, values) in [
        ("/card/card_id", vec![serde_json::json!(500)]),
        ("/card/index", vec![serde_json::json!(9)]),
        ("/card/face/damage_bonus", vec![serde_json::json!(7)]),
        ("/card/face/cost_this_turn", vec![serde_json::json!(-1)]),
        (
            "/card/fingerprint/upgrade_level",
            vec![serde_json::json!(1)],
        ),
        (
            "/card/fingerprint/floor_added_to_deck",
            vec![serde_json::json!(5)],
        ),
        (
            "/card/fingerprint/bing_bong_skip_once",
            vec![serde_json::json!(true)],
        ),
        ("/target/combat_id", vec![serde_json::json!(2)]),
        ("/target", vec![serde_json::Value::Null]),
        (
            "/card/fingerprint/properties",
            vec![
                serde_json::json!({"bonus": 7}),
                serde_json::json!({"bonus": 8}),
                serde_json::json!({"bonus": 0.0}),
                serde_json::json!({"bonus": -0.0}),
                serde_json::json!({"cards": [1, 2]}),
                serde_json::json!({"cards": [2, 1]}),
                serde_json::json!({"card_id": 1, "fingerprint": "saved", "index": 0}),
                serde_json::json!({"card_id": 2, "fingerprint": "saved", "index": 9}),
                serde_json::json!({"fingerprint": "saved"}),
            ],
        ),
        (
            "/card/fingerprint/enchantment",
            vec![
                serde_json::json!({"model_id":"ENCHANTMENT.MOMENTUM", "amount":1, "properties":{}}),
                serde_json::json!({"model_id":"ENCHANTMENT.MOMENTUM", "amount":2, "properties":{}}),
                serde_json::json!({"model_id":"ENCHANTMENT.MOMENTUM", "amount":1, "properties":{"cards":[1,2]}}),
                serde_json::json!({"model_id":"ENCHANTMENT.MOMENTUM", "amount":1, "properties":{"cards":[2,1]}}),
            ],
        ),
    ] {
        for value in values {
            let mut action = base.clone();
            *action.pointer_mut(path).unwrap() = value;
            actions.push(serde_json::from_value(action).unwrap());
        }
    }
    actions.extend([Action::EndTurn { turn: 1 }, Action::EndTurn { turn: 2 }]);
    let Action::PlayCard { card, .. } = original else {
        unreachable!()
    };
    let mut other = card.clone();
    other.fingerprint.upgrade_level = 1;
    for cards in [vec![card.clone(), other.clone()], vec![other, card]] {
        actions.push(Action::ChooseCards {
            choice_id: 1.into(),
            cards,
        });
    }
    for left in &actions {
        for right in &actions {
            let expected = if action_class(left) == action_class(right) {
                vec![left.clone()]
            } else {
                vec![left.clone(), right.clone()]
            };
            assert_eq!(canonical_actions(&[left.clone(), right.clone()]), expected);
        }
    }
    let mut seen = std::collections::HashSet::new();
    let expected: Vec<_> = actions
        .iter()
        .filter(|action| seen.insert(action_class(action)))
        .cloned()
        .collect();
    assert_eq!(canonical_actions(&actions), expected);
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
