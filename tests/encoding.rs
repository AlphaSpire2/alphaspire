//! The policy encoding over pinned content: versioned tensors that are a
//! function of what a player sees — indistinguishable decision points encode
//! identically, distinguishable plays encode apart, and the dimensions are
//! the documented constants.

use alphaspire::encoding::{
    ACTION_FEATURES, ACTION_NAMING_TOKENS, ACTION_SURRENDERED_TOKEN, ACTION_TOKENS,
    EVENT_EFFECT_NAMES, MAX_TOKENS, OBSERVATION_SCALARS, PolicyEncoder, TOKEN_FEATURES,
    standard_vocabulary,
};
use alphaspire::plan::ActionPlan;
use sts2_core::UnlockPresetManifest;
use sts2_engine::{Action, BeliefState, Simulator};

const SEED: &str = "NLD6VZXP94";

fn encoder() -> PolicyEncoder {
    let registry = sts2_content::standard_registry();
    PolicyEncoder::new(standard_vocabulary(&registry), &registry)
}

fn fresh_run() -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at(SEED, &character, &preset, 0).unwrap()
}

fn next_action(simulator: &Simulator) -> Action {
    simulator
        .legal_actions()
        .iter()
        .find(|action| matches!(action, Action::ChooseCards { cards, .. } if !cards.is_empty()))
        .or_else(|| simulator.legal_actions().first())
        .cloned()
        .expect("a live run offers an action")
}

fn first_combat() -> Simulator {
    let mut simulator = fresh_run();
    for _ in 0..40 {
        if simulator.state().combat.is_some() {
            return simulator;
        }
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }
    panic!("the seed reaches a combat inside forty decisions");
}

#[test]
fn the_dimensions_are_the_documented_constants() {
    let simulator = first_combat();
    let encoder = encoder();
    let rendered = encoder.encode_observation(&simulator.agent_observation());
    assert_eq!(rendered.scalars.len(), OBSERVATION_SCALARS);
    assert_eq!(rendered.tokens.len(), MAX_TOKENS);
    assert_eq!(rendered.features.len(), MAX_TOKENS * TOKEN_FEATURES);
    for action in simulator.legal_actions() {
        let rendered = encoder.encode_action(&simulator.agent_observation(), action);
        assert_eq!(rendered.tokens.len(), ACTION_TOKENS);
        assert_eq!(rendered.features.len(), ACTION_FEATURES);
    }
    assert!(encoder.vocabulary_size() > 100, "the content registered");
    assert!(encoder.vocabulary_hash().starts_with("sha256:"));
}

#[test]
fn indistinguishable_decision_points_encode_identically() {
    // A belief sample differs from its original only in hidden state, so
    // the encoding — a function of the observation — must not move.
    let original = first_combat();
    let sample = BeliefState::from_simulator(&original)
        .unwrap()
        .sample(21, 3)
        .unwrap();
    let encoder = encoder();
    assert_eq!(
        encoder.encode_observation(&original.agent_observation()),
        encoder.encode_observation(&sample.agent_observation()),
        "the net may see nothing a player cannot"
    );
}

#[test]
fn distinguishable_plays_encode_apart() {
    let simulator = first_combat();
    let encoder = encoder();
    let observation = simulator.agent_observation();
    let mut seen = std::collections::HashSet::new();
    for action in alphaspire::search::canonical_actions(simulator.legal_actions()) {
        let rendered = encoder.encode_action(&observation, &action);
        let key = format!("{:?}|{:?}", rendered.tokens, rendered.features);
        assert!(
            seen.insert(key),
            "two distinct action classes shared an encoding: {action:?}"
        );
    }
}

#[test]
fn the_vocabulary_hash_is_stable_and_order_free() {
    let registry = sts2_content::standard_registry();
    let mut ids = registry.registered_model_ids();
    let forward = PolicyEncoder::new(ids.clone(), &registry);
    ids.reverse();
    let backward = PolicyEncoder::new(ids, &registry);
    assert_eq!(forward.vocabulary_hash(), backward.vocabulary_hash());
    assert_eq!(forward.vocabulary_size(), backward.vocabulary_size());
}

// --- encoding v4: what the batched bump has to be able to tell apart -------

use alphaspire::encoding::{
    CARD_TYPE_NAMES, INTENT_TYPE_NAMES, MAP_TYPE_NAMES, MOVE_HISTORY_DEPTH, TARGET_TYPE_NAMES,
};
use sts2_engine::{DecisionContext, PileName, ScenarioBuilder};

// The token row's block bases, re-derived from the documented widths rather
// than imported: a slot that moves without `POLICY_ENCODING_VERSION` moving
// is what these tests exist to catch.
const ZONES: usize = 14;
const SHARED: usize = ZONES;
const OWNER: usize = SHARED + 7;
const INTENT: usize = OWNER + 9;
const HP: usize = INTENT + INTENT_TYPE_NAMES.len();
const FACE: usize = HP + 2;
const STATIC: usize = FACE + 15;
/// The displayed damage and block, at the tail of the face block.
const FACE_PREVIEW: usize = FACE + 13;
const STATIC_NUMBERS: usize = STATIC + CARD_TYPE_NAMES.len() + TARGET_TYPE_NAMES.len();
const MAP_TYPE: usize = STATIC + CARD_TYPE_NAMES.len() + TARGET_TYPE_NAMES.len() + 4;
const EXTRA: usize = MAP_TYPE + MAP_TYPE_NAMES.len();

/// The action feature vector's blocks, on the same footing.
const ACTION_TARGET: usize = 13;
const ACTION_CARD: usize = ACTION_TARGET + 9;
const ACTION_INDEX: usize = ACTION_CARD + 10;
/// Column, row, ways on, walked, rows ahead, the boss's coordinate, and the
/// destination the map carries no point for.
const ACTION_MAP: usize = ACTION_INDEX + 2;
const ACTION_MAP_TYPE: usize = ACTION_MAP + 7;
const ACTION_PRICE: usize = ACTION_MAP_TYPE + MAP_TYPE_NAMES.len();
const ACTION_KIND: usize = ACTION_PRICE + 4;
const ACTION_STATIC: usize = ACTION_KIND + 6;
const ACTION_STATIC_NUMBERS: usize =
    ACTION_STATIC + CARD_TYPE_NAMES.len() + TARGET_TYPE_NAMES.len();
/// The resolved damage against the aim, the kill fraction, the lethal bit.
const ACTION_PREVIEW: usize = ACTION_STATIC_NUMBERS + 4;
/// Whether the action changes the player's facing.
const ACTION_FLIP: usize = ACTION_PREVIEW + 3;
/// The route the map step opens: the fewest of each point type on any path
/// ahead, then the most, then the longest walk still to come.
const ACTION_ROUTE_LEAST: usize = ACTION_FLIP + 1;
const ACTION_ROUTE_MOST: usize = ACTION_ROUTE_LEAST + MAP_TYPE_NAMES.len();
const ACTION_ROUTE_DEPTH: usize = ACTION_ROUTE_MOST + MAP_TYPE_NAMES.len();
/// Is-a-plan, freed by discard, freed by drink, the freed slot, and two
/// slots of reserve.
const ACTION_PLAN: usize = ACTION_ROUTE_DEPTH + 1;
/// What an event option's body does: a slot per `EventEffect` variant, then
/// the resources it moves and the flag for an amount nothing could read.
const ACTION_EVENT: usize = ACTION_PLAN + 6;
const ACTION_EVENT_NUMBERS: usize = ACTION_EVENT + EVENT_EFFECT_NAMES.len();

#[test]
fn the_layout_blocks_tile_the_feature_row() {
    // The bases above are re-derived from the documented widths, so a block
    // that grows without the row growing — or a row that grows without the
    // version moving — lands here rather than silently reinterpreting a slot
    // every trained checkpoint reads.
    assert_eq!(EXTRA + 8, TOKEN_FEATURES, "the extras close the token row");
    assert_eq!(
        FACE_PREVIEW + 2,
        STATIC,
        "the displayed pair closes the face block"
    );
    assert_eq!(
        ACTION_ROUTE_DEPTH + 1,
        ACTION_PLAN,
        "the route lookahead runs up to the plan block"
    );
    assert_eq!(
        ACTION_PLAN + 6,
        ACTION_EVENT,
        "the plan block runs up to the event body"
    );
    assert_eq!(
        ACTION_EVENT_NUMBERS + 10,
        ACTION_FEATURES,
        "and the event body closes the action row"
    );
    assert_eq!(
        ACTION_NAMING_TOKENS, ACTION_SURRENDERED_TOKEN,
        "the surrendered potion sits past the naming slots"
    );
    assert_eq!(
        ACTION_SURRENDERED_TOKEN + 1,
        ACTION_TOKENS,
        "and closes the token row"
    );
}

/// Whether a one-hot slot is lit. The encoder writes exactly `1.0` and
/// exactly `0.0` into these, but comparing floats strictly is a habit worth
/// not having in a test file.
fn lit(value: f32) -> bool {
    (value - 1.0).abs() < f32::EPSILON
}

/// Whether a scaled feature reads as `expected`.
fn reads(value: f32, expected: f32) -> bool {
    (value - expected).abs() < 1e-6
}

fn id(value: &str) -> sts2_core::ModelId {
    value.parse().unwrap()
}

/// The seeded run walked until `stop` answers.
fn walk_until(mut simulator: Simulator, stop: impl Fn(&Simulator) -> bool) -> Simulator {
    for _ in 0..300 {
        if stop(&simulator) {
            return simulator;
        }
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }
    panic!("the seed reaches the stop inside three hundred decisions");
}

/// One fight against a named monster standing on a named move state.
fn fight_on(monster: &str, state: &str) -> Simulator {
    ScenarioBuilder::new(SEED, id("ENCOUNTER.SHRINKER_BEETLE_WEAK"))
        .player_hp(80, 80)
        .energy(3, 3)
        .turn(1, 1)
        .enemy(1, id(monster), 300, 300, state)
        .card(
            PileName::Hand,
            10,
            sts2_engine::CardFingerprint::base(id("CARD.STRIKE_IRONCLAD")),
            1,
        )
        .build(sts2_content::standard_registry())
        .unwrap()
}

#[test]
fn two_rotation_positions_of_one_boss_encode_apart() {
    // The Insatiable's two THRASH states advertise the same damage but lead
    // to different moves. Intent-line tokens must retain the move ID so the
    // network can distinguish their rotation positions.
    let encoder = encoder();
    let one = fight_on("MONSTER.THE_INSATIABLE", "THRASH_MOVE");
    let two = fight_on("MONSTER.THE_INSATIABLE", "THRASH_MOVE_2");
    let seen = |simulator: &Simulator| {
        let observation = simulator.agent_observation();
        let enemy = observation
            .creatures
            .iter()
            .find(|creature| creature.side != sts2_engine::CombatSide::Player)
            .expect("the fight has its boss");
        (
            enemy.intent.as_ref().unwrap().move_id.clone(),
            enemy.intent.as_ref().unwrap().intents.clone(),
        )
    };
    let (first_move, first_lines) = seen(&one);
    let (second_move, second_lines) = seen(&two);
    assert_ne!(first_move, second_move, "two states");
    assert_eq!(first_lines, second_lines, "advertising the same numbers");
    assert_ne!(
        encoder.encode_observation(&one.agent_observation()),
        encoder.encode_observation(&two.agent_observation()),
        "the rotation position is the whole difference, and it is encoded"
    );
}

#[test]
fn an_attack_with_a_defend_rider_is_not_a_multi_attack() {
    // Crusher's GUARDED_STRIKE advertises an attack line and a defend line.
    // Under the eight-bucket intent hash those two kinds shared bucket six,
    // so the defend was invisible; and the hit count summed every line's
    // repeats, so one blow plus a rider read as two blows.
    let encoder = encoder();
    let simulator = fight_on("MONSTER.CRUSHER", "GUARDED_STRIKE_MOVE");
    let observation = simulator.agent_observation();
    let enemy = observation
        .creatures
        .iter()
        .find(|creature| creature.side != sts2_engine::CombatSide::Player)
        .expect("the fight has its crusher");
    let intent = enemy.intent.as_ref().expect("the crusher shows its move");
    assert_eq!(intent.intents.len(), 2, "an attack line and a defend line");
    let tensors = encoder.encode_observation(&observation);

    // The attack and the defend each light their own slot, and the hit count
    // reads one blow rather than two.
    let creature_row = |tensors: &alphaspire::encoding::ObservationEncoding| {
        let slot = (0..tensors.live_tokens())
            .find(|slot| {
                lit(tensors.features[slot * TOKEN_FEATURES + 8])
                    && !lit(tensors.features[slot * TOKEN_FEATURES + SHARED + 4])
            })
            .expect("an enemy creature row");
        tensors.features[slot * TOKEN_FEATURES..(slot + 1) * TOKEN_FEATURES].to_vec()
    };
    let row = creature_row(&tensors);
    let intent_base = INTENT;
    let attack_slot = INTENT_TYPE_NAMES
        .iter()
        .position(|n| *n == "attack")
        .unwrap();
    let defend_slot = INTENT_TYPE_NAMES
        .iter()
        .position(|n| *n == "defend")
        .unwrap();
    assert!(lit(row[intent_base + attack_slot]), "the blow is shown");
    assert!(lit(row[intent_base + defend_slot]), "so is the guard");
    assert!(
        reads(row[SHARED + 6], 0.1),
        "one blow, not two: hits/10 reads {}",
        row[SHARED + 6]
    );

    // And a real two-hit attack of the same total damage reads two.
    let multi = fight_on("MONSTER.THE_INSATIABLE", "THRASH_MOVE");
    let multi_row = creature_row(&encoder.encode_observation(&multi.agent_observation()));
    assert!(
        reads(multi_row[SHARED + 6], 0.2),
        "a two-hit thrash reads two: {}",
        multi_row[SHARED + 6]
    );
}

#[test]
fn two_map_choices_encode_apart() {
    // Under v3 every `ChooseMap` action encoded bit-identically — no index,
    // no destination, nothing. A map step carries its point now.
    let encoder = encoder();
    let simulator = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::MapNavigation { destinations, .. }
            if destinations.len() > 1)
    });
    let observation = simulator.agent_observation();
    let mut seen = std::collections::HashSet::new();
    let mut steps = 0;
    for action in simulator.legal_actions() {
        if !matches!(action, Action::ChooseMap { .. }) {
            continue;
        }
        steps += 1;
        let rendered = encoder.encode_action(&observation, action);
        assert!(
            seen.insert(format!("{:?}|{:?}", rendered.tokens, rendered.features)),
            "two map steps shared an encoding: {action:?}"
        );
    }
    assert!(steps > 1, "the anchor offers a real choice of paths");
}

#[test]
fn the_map_and_the_deck_are_tokens_out_of_combat() {
    let encoder = encoder();
    let simulator = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::MapNavigation { .. })
    });
    let observation = simulator.agent_observation();
    let tensors = encoder.encode_observation(&observation);
    let zone_count = |zone: usize| {
        (0..tensors.live_tokens())
            .filter(|slot| lit(tensors.features[slot * TOKEN_FEATURES + zone]))
            .count()
    };
    let map = observation.map.as_ref().expect("a run carries its map");
    assert_eq!(zone_count(12), map.points.len(), "one token per map point");
    assert_eq!(
        zone_count(11),
        observation.deck.len()
            + observation
                .deck
                .iter()
                .filter(|card| card.enchantment.is_some())
                .count(),
        "one token per deck card, plus a rider per enchantment"
    );
    // Every map token names something: a token index of zero is the padding
    // row the net masks out of its pool.
    for slot in 0..tensors.live_tokens() {
        assert!(
            tensors.tokens[slot] != 0,
            "token {slot} names nothing (zone one-hot {:?})",
            &tensors.features[slot * TOKEN_FEATURES..slot * TOKEN_FEATURES + ZONES]
        );
    }
    // The boss point wears the act's boss.
    let boss_slot = (0..tensors.live_tokens())
        .find(|slot| {
            let row = &tensors.features[slot * TOKEN_FEATURES..(slot + 1) * TOKEN_FEATURES];
            lit(row[12])
                && lit(row[MAP_TYPE + MAP_TYPE_NAMES.iter().position(|n| *n == "boss").unwrap()])
        })
        .expect("the map has a boss point");
    assert_ne!(tensors.tokens[boss_slot], 0, "the boss is named as itself");
}

#[test]
fn two_reward_screens_encode_apart() {
    // Different reward offers must produce different observations.
    let encoder = encoder();
    let screen = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::Rewards { .. })
    });
    let mut state = screen.state().clone();
    let set = state
        .combat
        .as_mut()
        .and_then(|combat| combat.reward_set.as_mut())
        .or_else(|| state.run.as_mut().and_then(|run| run.reward_set.as_mut()))
        .expect("the screen is open");
    for reward in &mut set.rewards {
        reward.fingerprint.gold_amount += 33;
    }
    let richer = Simulator::from_scenario(state, sts2_content::standard_registry()).unwrap();
    assert_ne!(
        encoder.encode_observation(&screen.agent_observation()),
        encoder.encode_observation(&richer.agent_observation()),
        "a screen offering more gold encodes as a different screen"
    );
}

#[test]
fn a_choice_screens_candidates_reach_the_value_head() {
    // The value head reads the observation trunk alone, so a choice screen's
    // candidates had to become tokens rather than staying inside the policy's
    // per-action encodings. This is the reward screen's card offer, which is
    // the shape the Knowledge Demon's curse picks arrive in.
    let encoder = encoder();
    let screen = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::Rewards { offers, .. }
            if offers.iter().any(|offer| !offer.offered_cards.is_empty()))
    });
    let observation = screen.agent_observation();
    let tensors = encoder.encode_observation(&observation);
    let offers = (0..tensors.live_tokens())
        .filter(|slot| lit(tensors.features[slot * TOKEN_FEATURES + 13]))
        .count();
    let DecisionContext::Rewards { offers: lines, .. } = &observation.decision else {
        unreachable!()
    };
    let cards: usize = lines
        .iter()
        .map(|line| line.offered_cards.len() + usize::from(line.special_card.is_some()))
        .sum();
    assert_eq!(
        offers,
        lines.len() + cards,
        "a token per reward line and a token per card on it"
    );
}

#[test]
fn distinguishable_run_level_plays_encode_apart() {
    // The macro screens the v3 encoding rendered as a family one-hot
    // and an index over eight: two shop items, two rest options, two event
    // options, two treasure relics.
    let encoder = encoder();
    for stop in [
        "shop",
        "rest_site",
        "event",
        "treasure",
        "rewards",
        "map_navigation",
    ] {
        let mut simulator = fresh_run();
        let mut found = false;
        for _ in 0..600 {
            if simulator.state().terminal.is_some() {
                break;
            }
            let kind = match simulator.decision() {
                DecisionContext::Shop { .. } => "shop",
                DecisionContext::RestSite { .. } => "rest_site",
                DecisionContext::Event { .. } => "event",
                DecisionContext::Treasure { .. } => "treasure",
                DecisionContext::Rewards { .. } => "rewards",
                DecisionContext::MapNavigation { .. } => "map_navigation",
                _ => "",
            };
            if kind == stop {
                let observation = simulator.agent_observation();
                let mut seen = std::collections::HashSet::new();
                for action in alphaspire::search::canonical_actions(simulator.legal_actions()) {
                    let rendered = encoder.encode_action(&observation, &action);
                    assert!(
                        seen.insert(format!("{:?}|{:?}", rendered.tokens, rendered.features)),
                        "two {stop} actions shared an encoding: {action:?}"
                    );
                }
                found = true;
                break;
            }
            let action = next_action(&simulator);
            simulator.step_quietly(&action).unwrap();
        }
        assert!(found, "the seed reaches a {stop} screen");
    }
}

#[test]
fn the_intent_types_and_decision_kinds_are_closed_sets() {
    // Nothing is hashed any more, so the names have to be the simulator's
    // own. A content change that renames one shows up here rather than
    // quietly landing in no slot at all.
    let mut simulator = fresh_run();
    let mut kinds = std::collections::HashSet::new();
    let mut intents = std::collections::HashSet::new();
    for _ in 0..600 {
        if simulator.state().terminal.is_some() {
            break;
        }
        let observation = simulator.agent_observation();
        let value = serde_json::to_value(&observation.decision).unwrap();
        kinds.insert(value["kind"].as_str().unwrap().to_owned());
        for creature in &observation.creatures {
            for line in creature.intent.iter().flat_map(|intent| &intent.intents) {
                intents.insert(line.intent_type.clone());
            }
        }
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }
    assert!(kinds.len() >= 5, "the walk met a spread of screens");
    for kind in &kinds {
        assert!(
            alphaspire::encoding::DECISION_KIND_NAMES.contains(&kind.as_str()),
            "decision kind {kind} has no slot"
        );
    }
    for intent in &intents {
        assert!(
            INTENT_TYPE_NAMES.contains(&intent.as_str()),
            "intent type {intent} has no slot"
        );
    }
}

#[test]
fn the_vocabulary_covers_every_move_state_a_monster_can_show() {
    // The move id is the rotation position, and it is a real vocabulary
    // index rather than a hash bucket precisely so that no two states of one
    // monster can collide. That only holds if every state is in the
    // vocabulary.
    let registry = sts2_content::standard_registry();
    let vocabulary: std::collections::BTreeSet<sts2_core::ModelId> =
        alphaspire::encoding::standard_vocabulary(&registry)
            .into_iter()
            .collect();
    let mut checked = 0;
    for id in registry.registered_model_ids() {
        let Some(monster) = registry.monster(&id) else {
            continue;
        };
        for state in &monster.states {
            if !state.is_move() {
                continue;
            }
            let named = sts2_core::ModelId::new("MOVE", state.state_id()).unwrap();
            assert!(
                vocabulary.contains(&named),
                "{id}'s {} is outside the vocabulary",
                state.state_id()
            );
            checked += 1;
        }
    }
    assert!(checked > 200, "the registry's machines were walked");
}

#[test]
fn the_players_own_creature_row_names_something() {
    // A token index of zero is the padding row, and the net's mask
    // (`tokens != 0`) drops such a row out of the pooled observation
    // entirely. A character is not a registered model, so through encoding
    // v3 the player's own creature token was zero and the row carrying the
    // player's side flag, hit points and block never reached the trunk. The
    // vocabulary carries the characters now.
    let encoder = encoder();
    let simulator = first_combat();
    let observation = simulator.agent_observation();
    let tensors = encoder.encode_observation(&observation);
    let player = observation
        .creatures
        .iter()
        .position(|creature| creature.side == sts2_engine::CombatSide::Player)
        .expect("the fight has its player");
    assert_ne!(
        tensors.tokens[player], 0,
        "the player's creature token names the character"
    );
    for slot in 0..tensors.live_tokens() {
        assert_ne!(tensors.tokens[slot], 0, "token {slot} names nothing");
    }
}

#[test]
fn an_act_sample_encodes_as_the_screen_it_shows() {
    // The same fairness property as the combat arm, over the act-v0.5
    // erasure: a sample differs from the original only in the act's hidden
    // remainder, and the encoding now renders the map, the deck and the
    // standing screen's offers — every one of which the erasure pins. If any
    // of them were sampled rather than pinned, this is where it would show.
    let original = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::MapNavigation { .. })
            && simulator
                .state()
                .run
                .as_ref()
                .is_some_and(|run| run.floor >= 4)
    });
    let belief = sts2_engine::ActBeliefState::from_simulator(&original).unwrap();
    let encoder = encoder();
    let seen = encoder.encode_observation(&original.agent_observation());
    for rollout in 0..4 {
        let sample = belief.sample(19, rollout).unwrap();
        assert_eq!(
            encoder.encode_observation(&sample.agent_observation()),
            seen,
            "the net may see nothing a player cannot"
        );
    }
}

// --- encoding v5: the arithmetic, the numbers, and the pointer -------------

/// One fight against a named monster on a named move state, with the player
/// standing at `hp` of eighty.
fn fight_at(monster: &str, state: &str, hp: i32) -> Simulator {
    ScenarioBuilder::new(SEED, id("ENCOUNTER.SHRINKER_BEETLE_WEAK"))
        .player_hp(hp, 80)
        .energy(3, 3)
        .turn(1, 1)
        .enemy(1, id(monster), 300, 300, state)
        .card(
            PileName::Hand,
            10,
            sts2_engine::CardFingerprint::base(id("CARD.STRIKE_IRONCLAD")),
            1,
        )
        .build(sts2_content::standard_registry())
        .unwrap()
}

/// The rows of an encoding, one slice per live token.
fn rows(tensors: &alphaspire::encoding::ObservationEncoding) -> Vec<&[f32]> {
    (0..tensors.live_tokens())
        .map(|slot| &tensors.features[slot * TOKEN_FEATURES..(slot + 1) * TOKEN_FEATURES])
        .collect()
}

#[test]
fn the_encoding_reads_nothing_observation_v9_added() {
    // Observation v9 adds the exact choice-screen family and its non-card
    // alternatives for recorder-compatible script writers. Policy encoding
    // v5 reads the v8 card previews, but still reads neither v9 field: only the
    // candidate cards and selection reach the policy.
    assert_eq!(
        sts2_engine::AGENT_OBSERVATION_VERSION,
        9,
        "a new observation version needs its own blind-spot proof before any \
         checkpoint is re-stamped to it"
    );
    let encoder = encoder();
    let selection = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::ChooseCards { .. })
    });
    let observation = selection.agent_observation();
    let mut blind = observation.clone();
    let DecisionContext::ChooseCards {
        screen,
        alternatives,
        ..
    } = &mut blind.decision
    else {
        panic!("the walk stopped on a card choice");
    };
    *screen = sts2_engine::ChoiceScreen::Unknown;
    alternatives.push(sts2_engine::ChoiceAlternative {
        option_id: "test_only".to_owned(),
        relic: None,
    });
    assert_eq!(
        encoder.encode_observation(&observation),
        encoder.encode_observation(&blind),
        "the observation encoding reads neither the recorder screen nor its alternatives"
    );
    for action in alphaspire::search::canonical_actions(selection.legal_actions()) {
        assert_eq!(
            encoder.encode_action(&observation, &action),
            encoder.encode_action(&blind, &action),
            "the action encoding reads neither recorder-only choice field: {action:?}"
        );
    }
}

#[test]
fn the_owner_one_hot_is_what_an_action_target_gathers() {
    // The scorer is additive in the state and the action vector, so an
    // action reaches the creature it points at only by pooling token rows per
    // owner slot and gathering with the action's target block. That works
    // only if the two blocks are one-hots over the same nine slots.
    let encoder = encoder();
    let simulator = fight_at("MONSTER.CRUSHER", "GUARDED_STRIKE_MOVE", 80);
    let observation = simulator.agent_observation();
    let tensors = encoder.encode_observation(&observation);
    let owner = |row: &[f32]| {
        let lit: Vec<usize> = (0..9).filter(|slot| lit(row[OWNER + slot])).collect();
        assert!(lit.len() <= 1, "an owner block is a one-hot: {lit:?}");
        lit.first().copied()
    };
    for row in rows(&tensors) {
        let on_the_board = lit(row[8]) || lit(row[9]) || lit(row[10]);
        assert_eq!(
            owner(row).is_some(),
            on_the_board,
            "exactly the creature, power and intent rows name an owner"
        );
    }

    // And the slot an action aiming at a creature lights is the slot that
    // creature's own rows carry.
    let aimed = observation
        .creatures
        .iter()
        .position(|creature| creature.side != sts2_engine::CombatSide::Player)
        .expect("the fight has its enemy");
    let creature_rows: Vec<usize> = rows(&tensors)
        .iter()
        .filter(|row| lit(row[8]))
        .filter_map(|row| owner(row))
        .collect();
    assert!(
        creature_rows.contains(&aimed),
        "the enemy's own row names its slot"
    );
    let play = simulator
        .legal_actions()
        .iter()
        .find(|action| {
            matches!(
                action,
                Action::PlayCard {
                    target: Some(_),
                    ..
                }
            )
        })
        .expect("a targeted play is offered")
        .clone();
    let Action::PlayCard {
        target: Some(target),
        ..
    } = &play
    else {
        unreachable!()
    };
    let slot = observation
        .creatures
        .iter()
        .position(|creature| creature.combat_id == target.combat_id)
        .expect("the play aims at a creature on the board");
    let rendered = encoder.encode_action(&observation, &play);
    assert!(
        lit(rendered.features[ACTION_TARGET + slot]),
        "the play lights the slot its target's rows carry"
    );
}

#[test]
fn the_board_threat_scalars_are_the_sum_the_pool_cannot_form() {
    // The trunk means over one bag of tokens, so a sum across creature
    // rows is `mean × n` — a product of two of its own inputs. These are the
    // sums, handed over.
    let encoder = encoder();
    let healthy = fight_at("MONSTER.THE_INSATIABLE", "THRASH_MOVE", 80);
    let observation = healthy.agent_observation();
    let expected: i64 = observation
        .creatures
        .iter()
        .filter(|creature| creature.side != sts2_engine::CombatSide::Player)
        .flat_map(|creature| creature.intent.iter().flat_map(|intent| &intent.intents))
        .filter(|line| line.intent_type == "attack")
        .map(|line| i64::from(line.damage.unwrap_or(0)) * i64::from(line.repeats.unwrap_or(1)))
        .sum();
    assert!(expected > 0, "the thrash advertises a blow");
    let scalars = encoder.encode_observation(&observation).scalars;
    #[allow(
        clippy::cast_precision_loss,
        reason = "the fixture's numbers are small"
    )]
    let incoming = expected as f32 / 50.0;
    assert!(
        reads(scalars[39], incoming),
        "incoming reads {} against {incoming}",
        scalars[39]
    );
    assert!(
        reads(scalars[40], incoming),
        "nothing is blocked, so unblocked is the whole of it"
    );
    assert!(
        reads(scalars[41], incoming * 50.0 / 80.0),
        "the blow as a share of eighty hit points"
    );
    assert!(!lit(scalars[42]), "eighty hit points survive it");
    assert!(reads(scalars[43], 1.0 / 8.0), "one enemy alive");
    assert!(reads(scalars[44], 1.0 / 8.0), "one ally alive");

    // The same board against a player who cannot take it.
    let doomed = fight_at("MONSTER.THE_INSATIABLE", "THRASH_MOVE", 4);
    let scalars = encoder
        .encode_observation(&doomed.agent_observation())
        .scalars;
    assert!(lit(scalars[42]), "four hit points do not");
    assert!(reads(scalars[41], 2.0), "and the share is capped at twice");
}

/// A fight whose hand is exactly the named cards, in the order given, each at
/// the upgrade level beside it.
fn fight_holding(cards: &[(&str, u8)]) -> Simulator {
    let mut builder = ScenarioBuilder::new(SEED, id("ENCOUNTER.SHRINKER_BEETLE_WEAK"))
        .player_hp(80, 80)
        .energy(3, 3)
        .turn(1, 1)
        .enemy(1, id("MONSTER.CRUSHER"), 300, 300, "GUARDED_STRIKE_MOVE");
    for (index, (model, upgrade)) in cards.iter().enumerate() {
        let mut fingerprint = sts2_engine::CardFingerprint::base(id(model));
        fingerprint.upgrade_level = *upgrade;
        builder = builder.card(PileName::Hand, 10 + index as u64, fingerprint, 1);
    }
    builder.build(sts2_content::standard_registry()).unwrap()
}

#[test]
fn a_card_token_carries_what_the_registry_says_it_does() {
    // These are the literal bodies — `attack(6)`, `block(5)`,
    // `attack(8), debuff("POWER.VULNERABLE_POWER", 2)` — and the upgraded
    // form's, because an upgrade replaces the body whole.
    let encoder = encoder();
    let simulator = fight_holding(&[
        ("CARD.STRIKE_IRONCLAD", 0),
        ("CARD.DEFEND_IRONCLAD", 0),
        ("CARD.BASH", 0),
        ("CARD.STRIKE_IRONCLAD", 1),
    ]);
    let observation = simulator.agent_observation();
    let tensors = encoder.encode_observation(&observation);
    let hand: Vec<&[f32]> = rows(&tensors)
        .into_iter()
        .filter(|row| lit(row[0]))
        .collect();
    assert_eq!(hand.len(), 4, "one token per card in hand, no riders");
    let type_slot = |name: &str| STATIC + CARD_TYPE_NAMES.iter().position(|n| *n == name).unwrap();
    let target_slot = |name: &str| {
        STATIC + CARD_TYPE_NAMES.len() + TARGET_TYPE_NAMES.iter().position(|n| *n == name).unwrap()
    };

    assert!(lit(hand[0][type_slot("attack")]), "a Strike is an attack");
    assert!(lit(hand[0][target_slot("any_enemy")]), "aimed at one enemy");
    assert!(reads(hand[0][STATIC_NUMBERS], 6.0 / 50.0), "six damage");
    assert!(reads(hand[0][STATIC_NUMBERS + 1], 0.1), "landing once");
    assert!(reads(hand[0][STATIC_NUMBERS + 2], 0.0), "and no block");

    assert!(lit(hand[1][type_slot("skill")]), "a Defend is a skill");
    assert!(lit(hand[1][target_slot("self")]), "worked on its owner");
    assert!(reads(hand[1][STATIC_NUMBERS + 2], 5.0 / 50.0), "five block");
    assert!(reads(hand[1][STATIC_NUMBERS], 0.0), "and no damage");

    assert!(
        reads(hand[2][STATIC_NUMBERS], 8.0 / 50.0),
        "a Bash hits for eight"
    );
    assert!(
        reads(hand[2][STATIC_NUMBERS + 3], 0.25),
        "and grants the one power the vulnerable rider is"
    );

    assert!(
        reads(hand[3][STATIC_NUMBERS], 9.0 / 50.0),
        "a Strike+ is nine, not the base form's six"
    );

    // And the play carries the same numbers, because the pointer scorer is
    // additive: nothing the card token knows reaches the action that plays it.
    let play = simulator
        .legal_actions()
        .iter()
        .find(|action| {
            matches!(action, Action::PlayCard { card, .. }
                if card.fingerprint.model_id == id("CARD.BASH"))
        })
        .expect("the hand can play its Bash")
        .clone();
    let rendered = encoder.encode_action(&observation, &play);
    assert!(
        lit(rendered.features
            [ACTION_STATIC + CARD_TYPE_NAMES.iter().position(|n| *n == "attack").unwrap()]),
        "the play knows it is an attack"
    );
    assert!(
        reads(rendered.features[ACTION_STATIC_NUMBERS], 8.0 / 50.0),
        "and knows what it hits for"
    );
}

#[test]
fn a_creature_carries_the_moves_it_has_performed() {
    // The Knowledge Demon's curse count and every cooldown and
    // no-repeat weight in the registry read off the moves a creature has
    // performed. A move state is a vocabulary index rather than a number,
    // so the history rides as tokens of its own.
    let encoder = encoder();
    let mut simulator = fight_holding(&[("CARD.STRIKE_IRONCLAD", 0)]);
    for _ in 0..8 {
        let action = simulator
            .legal_actions()
            .iter()
            .find(|action| matches!(action, Action::EndTurn { .. }))
            .or_else(|| simulator.legal_actions().first())
            .cloned()
            .expect("the fight offers an action");
        simulator.step_quietly(&action).unwrap();
        if simulator.state().combat.is_none() {
            break;
        }
    }
    let observation = simulator.agent_observation();
    let performed = observation
        .creatures
        .iter()
        .find(|creature| creature.side != sts2_engine::CombatSide::Player)
        .map(|creature| creature.move_history.recent_performed.len())
        .expect("the fight has its crusher");
    assert!(performed > 0, "the crusher has taken its turns");
    let tensors = encoder.encode_observation(&observation);
    let history: Vec<&[f32]> = rows(&tensors)
        .into_iter()
        .filter(|row| lit(row[10]) && lit(row[EXTRA + 4]))
        .collect();
    assert_eq!(
        history.len(),
        performed.min(MOVE_HISTORY_DEPTH),
        "up to the last three moves, one token each"
    );
    assert!(
        reads(history[0][EXTRA + 3], 1.0),
        "the most recent move reads full recency"
    );
    for row in &history {
        assert!(
            (0..9).any(|slot| lit(row[OWNER + slot])),
            "a history token names the creature it belongs to"
        );
    }
    let player_row = rows(&tensors)
        .into_iter()
        .find(|row| lit(row[8]) && lit(row[SHARED + 4]))
        .expect("the fight has its player");
    let blows = observation
        .creatures
        .iter()
        .find(|creature| creature.side == sts2_engine::CombatSide::Player)
        .map(|creature| creature.unblocked_blows_taken)
        .unwrap();
    assert!(blows > 0, "the crusher has landed on the player");
    #[allow(clippy::cast_precision_loss, reason = "blow counts are small")]
    let expected = blows as f32 / 10.0;
    assert!(
        reads(player_row[EXTRA], expected),
        "the player's row counts the blows block did not stop whole"
    );
}

/// A strengthened player facing one Vulnerable Crusher at `hp`, holding a
/// Strike, a Defend and a Thunderclap, with a second Strike left in the draw
/// pile. The fixture the displayed-number tests read: a preview is only worth
/// encoding where the print and the face disagree, and here every number on
/// the screen has been through a hook pass.
fn strengthened_fight(hp: i32) -> Simulator {
    let mut builder = ScenarioBuilder::new(SEED, id("ENCOUNTER.SHRINKER_BEETLE_WEAK"))
        .player_hp(80, 80)
        .energy(3, 3)
        .turn(1, 1)
        .enemy(1, id("MONSTER.CRUSHER"), hp, 300, "GUARDED_STRIKE_MOVE")
        .power(
            0,
            id("POWER.STRENGTH_POWER"),
            3,
            sts2_engine::PowerType::Buff,
        )
        .power(
            1,
            id("POWER.VULNERABLE_POWER"),
            2,
            sts2_engine::PowerType::Debuff,
        )
        .card(
            PileName::Draw,
            20,
            sts2_engine::CardFingerprint::base(id("CARD.STRIKE_IRONCLAD")),
            1,
        );
    for (index, model) in [
        "CARD.STRIKE_IRONCLAD",
        "CARD.DEFEND_IRONCLAD",
        "CARD.THUNDERCLAP",
    ]
    .iter()
    .enumerate()
    {
        builder = builder.card(
            PileName::Hand,
            10 + index as u64,
            sts2_engine::CardFingerprint::base(id(model)),
            1,
        );
    }
    builder.build(sts2_content::standard_registry()).unwrap()
}

/// The play of `model` aimed at `target`, off the fight's own action list.
fn play_of(simulator: &Simulator, model: &str, target: Option<u64>) -> Action {
    simulator
        .legal_actions()
        .iter()
        .find(|action| match action {
            Action::PlayCard { card, target: aim } => {
                card.fingerprint.model_id == id(model)
                    && aim.as_ref().map(|aim| aim.combat_id.get()) == target
            }
            _ => false,
        })
        .expect("the hand offers the play")
        .clone()
}

#[test]
fn a_hand_card_carries_the_number_its_face_shows() {
    // The game's card preview runs the whole hook pass over every
    // number a card prints, so a Strike in a hand under three strength
    // displays nine. The static block beside it still says six, and the pair
    // is what separates "this card" from "this card here" — alphaspire reads
    // both off the observation and re-derives neither, because a second
    // implementation of a number the engine already prints is a second
    // implementation to drift.
    let encoder = encoder();
    let simulator = strengthened_fight(12);
    let observation = simulator.agent_observation();
    let tensors = encoder.encode_observation(&observation);
    let hand: Vec<&[f32]> = rows(&tensors)
        .into_iter()
        .filter(|row| lit(row[0]))
        .collect();
    assert_eq!(hand.len(), 3, "one token per card in hand");
    assert!(
        reads(hand[0][FACE_PREVIEW], 9.0 / 50.0),
        "a Strike under three strength displays nine"
    );
    assert!(
        reads(hand[0][STATIC_NUMBERS], 6.0 / 50.0),
        "and the registry still says six"
    );
    assert!(reads(hand[0][FACE_PREVIEW + 1], 0.0), "an attack no block");
    assert!(
        reads(hand[1][FACE_PREVIEW + 1], 5.0 / 50.0),
        "a Defend displays its five"
    );
    assert!(reads(hand[1][FACE_PREVIEW], 0.0), "and no damage");

    // The build runs the passes for the hand and nowhere else, so the Strike
    // waiting in the draw pile draws its print and carries no pair.
    for row in rows(&tensors).into_iter().filter(|row| !lit(row[0])) {
        assert!(
            reads(row[FACE_PREVIEW], 0.0) && reads(row[FACE_PREVIEW + 1], 0.0),
            "only a card in hand carries a preview"
        );
    }

    // And a sample written before observation v8 deserializes the list empty,
    // which has to encode as the empty pair rather than as a panic.
    let mut older = observation.clone();
    older.hand_previews.clear();
    let stale = encoder.encode_observation(&older);
    for row in rows(&stale) {
        assert!(reads(row[FACE_PREVIEW], 0.0) && reads(row[FACE_PREVIEW + 1], 0.0));
    }
}

#[test]
fn a_play_carries_what_it_resolves_to_against_its_aim() {
    // The target-specific preview gives damage, the fraction of remaining HP, and
    // whether it kills. Nine on the face, thirteen into a Vulnerable enemy —
    // the aimed pass is not the unaimed one.
    let encoder = encoder();
    let doomed = strengthened_fight(12);
    let observation = doomed.agent_observation();
    let strike = encoder.encode_action(
        &observation,
        &play_of(&doomed, "CARD.STRIKE_IRONCLAD", Some(1)),
    );
    assert!(
        reads(strike.features[ACTION_PREVIEW], 13.0 / 50.0),
        "nine, and half again into the Vulnerable"
    );
    assert!(
        reads(strike.features[ACTION_PREVIEW + 1], 1.0),
        "thirteen over twelve is capped at the whole of it"
    );
    assert!(lit(strike.features[ACTION_PREVIEW + 2]), "and it kills");

    // A sweep names nobody, so the number is the shared one it draws while
    // it is held up, priced against the enemy it comes closest to killing.
    let sweep = encoder.encode_action(&observation, &play_of(&doomed, "CARD.THUNDERCLAP", None));
    assert!(
        reads(sweep.features[ACTION_PREVIEW], 10.0 / 50.0),
        "four, three strength, half again"
    );
    assert!(reads(sweep.features[ACTION_PREVIEW + 1], 10.0 / 12.0));
    assert!(!lit(sweep.features[ACTION_PREVIEW + 2]), "and it does not");

    // A card with no number on its face leaves the block empty.
    let defend = encoder.encode_action(
        &observation,
        &play_of(&doomed, "CARD.DEFEND_IRONCLAD", None),
    );
    assert!((0..3).all(|slot| reads(defend.features[ACTION_PREVIEW + slot], 0.0)));

    // The same play into three hundred hit points is the same blow and a
    // different decision.
    let healthy = strengthened_fight(300);
    let observation = healthy.agent_observation();
    let strike = encoder.encode_action(
        &observation,
        &play_of(&healthy, "CARD.STRIKE_IRONCLAD", Some(1)),
    );
    assert!(reads(strike.features[ACTION_PREVIEW], 13.0 / 50.0));
    assert!(reads(strike.features[ACTION_PREVIEW + 1], 13.0 / 300.0));
    assert!(!lit(strike.features[ACTION_PREVIEW + 2]));
}

/// The crab's two halves, one marked on each side, and a player standing
/// Surrounded with three Strikes in hand.
fn surrounded_fight() -> Simulator {
    let mut builder = ScenarioBuilder::new(SEED, id("ENCOUNTER.SHRINKER_BEETLE_WEAK"))
        .player_hp(80, 80)
        .energy(3, 3)
        .turn(1, 1)
        .enemy(1, id("MONSTER.CRUSHER"), 300, 300, "GUARDED_STRIKE_MOVE")
        .enemy(2, id("MONSTER.ROCKET"), 300, 300, "TARGETING_RETICLE_MOVE")
        // Amount one, counter zero: the facing is the counter, and a reading
        // that took the amount would answer "facing left" for a power that is
        // simply on.
        .power(
            0,
            id("POWER.SURROUNDED_POWER"),
            1,
            sts2_engine::PowerType::Debuff,
        )
        .power(
            1,
            id("POWER.BACK_ATTACK_LEFT_POWER"),
            1,
            sts2_engine::PowerType::Buff,
        )
        .power(
            2,
            id("POWER.BACK_ATTACK_RIGHT_POWER"),
            1,
            sts2_engine::PowerType::Buff,
        );
    for index in 0..3_u64 {
        builder = builder.card(
            PileName::Hand,
            10 + index,
            sts2_engine::CardFingerprint::base(id("CARD.STRIKE_IRONCLAD")),
            1,
        );
    }
    builder.build(sts2_content::standard_registry()).unwrap()
}

#[test]
fn a_play_that_turns_the_player_says_so() {
    // The current facing is already encoded — it is the Surrounded
    // power's counter on the power token — but what a back attack will cost
    // depends on the facing the play *leaves behind*, and no counter can say
    // that about a play that has not happened. The bit is
    // `POWER.SURROUNDED_POWER`'s own rule read forward.
    let encoder = encoder();
    let mut simulator = surrounded_fight();
    let flip = |simulator: &Simulator, at: u64| {
        let observation = simulator.agent_observation();
        encoder
            .encode_action(
                &observation,
                &play_of(simulator, "CARD.STRIKE_IRONCLAD", Some(at)),
            )
            .features[ACTION_FLIP]
    };
    assert!(
        lit(flip(&simulator, 1)),
        "facing right, striking at the half on the left turns the player"
    );
    assert!(!lit(flip(&simulator, 2)), "the half already faced does not");

    // And the engine agrees: the same play moves the counter to one.
    let play = play_of(&simulator, "CARD.STRIKE_IRONCLAD", Some(1));
    simulator.step_quietly(&play).unwrap();
    let facing = simulator
        .agent_observation()
        .creatures
        .iter()
        .find(|creature| creature.side == sts2_engine::CombatSide::Player)
        .and_then(|creature| {
            creature
                .powers
                .iter()
                .find(|power| power.model_id == id("POWER.SURROUNDED_POWER"))
                .map(|power| power.counter)
        })
        .expect("the player stands Surrounded");
    assert_eq!(facing, 1, "the play turned the player");
    assert!(
        !lit(flip(&simulator, 1)),
        "and turning back is the other way"
    );
    assert!(lit(flip(&simulator, 2)));

    // A player who is not surrounded is never turned.
    let plain = strengthened_fight(300);
    let observation = plain.agent_observation();
    let rendered = encoder.encode_action(
        &observation,
        &play_of(&plain, "CARD.STRIKE_IRONCLAD", Some(1)),
    );
    assert!(!lit(rendered.features[ACTION_FLIP]));
}

#[test]
fn a_fight_fits_inside_the_token_budget() {
    // The move-history tokens are three more per creature, and the budget is
    // set for the run-level decisions rather than the combat ones. A fight
    // that truncates loses whatever the encoder pushed last.
    let encoder = encoder();
    let mut simulator = first_combat();
    let mut deepest = 0;
    for _ in 0..200 {
        if simulator.state().combat.is_none() {
            break;
        }
        let tensors = encoder.encode_observation(&simulator.agent_observation());
        deepest = deepest.max(tensors.live_tokens());
        assert!(
            !lit(tensors.scalars[21]),
            "a fight truncated at {} tokens",
            tensors.live_tokens()
        );
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }
    assert!(deepest > 20, "the walk met a real board: {deepest} tokens");
}

// --- encoding v6: what makes one map step rankable against another --------

use sts2_engine::{
    CardFingerprint, ChoiceAlternative, ChoiceIdentity, ChoiceScreen, EnchantmentFingerprint,
    MapCoord, MapPointType, RelicInstance, VisibleMap, VisibleMapPoint,
};

/// Which slot of the route block a map point type reads from.
fn map_type(name: &str) -> usize {
    MAP_TYPE_NAMES
        .iter()
        .position(|entry| *entry == name)
        .expect("the name is one of the point types")
}

fn at(col: u8, row: u8) -> MapCoord {
    MapCoord { col, row }
}

/// A map small enough to read the answer off by eye.
///
/// ```text
///                    (3,3) boss
///        (2,2) rest       (3,2) shop        (4,2) elite
///              (2,1) monster                (4,1) monster
///                    (3,0) ancient
/// ```
///
/// The left step forces no elite and reaches either a rest or a shop; the
/// right step forces the elite. Both are three rooms deep.
fn forked_map(ancient: sts2_core::ModelId, boss: sts2_core::ModelId) -> VisibleMap {
    let point =
        |coord: MapCoord, point_type: MapPointType, children: Vec<MapCoord>| VisibleMapPoint {
            coord,
            point_type,
            children,
            visited: false,
        };
    VisibleMap {
        act: 0,
        width: 7,
        height: 3,
        start: at(3, 0),
        boss_coord: at(3, 3),
        second_boss_coord: None,
        points: vec![
            point(at(3, 0), MapPointType::Ancient, vec![at(2, 1), at(4, 1)]),
            point(at(2, 1), MapPointType::Monster, vec![at(2, 2), at(3, 2)]),
            point(at(4, 1), MapPointType::Monster, vec![at(4, 2)]),
            point(at(2, 2), MapPointType::RestSite, vec![at(3, 3)]),
            point(at(3, 2), MapPointType::Shop, vec![at(3, 3)]),
            point(at(4, 2), MapPointType::Elite, vec![at(3, 3)]),
            point(at(3, 3), MapPointType::Boss, Vec::new()),
        ],
        boss,
        second_boss: None,
        ancient,
    }
}

/// A real map-navigation observation with the hand-built map laid over it, so
/// that everything but the graph is the sight a run actually produces.
fn observation_on_forked_map() -> sts2_engine::AgentObservation {
    let simulator = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::MapNavigation { .. })
    });
    let mut observation = simulator.agent_observation();
    let standing = observation.map.as_ref().expect("a run carries its map");
    observation.map = Some(forked_map(standing.ancient.clone(), standing.boss.clone()));
    observation.map_coord = Some(at(3, 0));
    observation.decision = DecisionContext::MapNavigation {
        current: Some(at(3, 0)),
        destinations: vec![at(2, 1), at(4, 1)],
    };
    observation
}

/// The count a route slot reads back as.
fn route_count(features: &[f32], base: usize, name: &str) -> f32 {
    features[base + map_type(name)] * 8.0
}

#[test]
fn the_route_lookahead_brackets_what_a_branch_forces_and_what_it_offers() {
    let encoder = encoder();
    let observation = observation_on_forked_map();
    let steps = [
        Action::ChooseMap {
            destination: at(2, 1),
        },
        Action::ChooseMap {
            destination: at(4, 1),
        },
    ];
    let rendered = encoder.encode_actions(&observation, &steps);
    // One walk of the map serves the whole screen, and it has to answer what
    // a walk per action would have.
    for (batched, action) in rendered.iter().zip(&steps) {
        assert_eq!(*batched, encoder.encode_action(&observation, action));
    }
    let (left, right) = (&rendered[0].features, &rendered[1].features);

    // Both branches are one monster, then a room, then the boss.
    for branch in [left, right] {
        assert!(reads(
            route_count(branch, ACTION_ROUTE_LEAST, "monster"),
            1.0
        ));
        assert!(reads(
            route_count(branch, ACTION_ROUTE_MOST, "monster"),
            1.0
        ));
        assert!(reads(route_count(branch, ACTION_ROUTE_LEAST, "boss"), 1.0));
        assert!(reads(branch[ACTION_ROUTE_DEPTH], 3.0 / 16.0));
    }

    // The left branch cannot meet an elite and may meet either a rest or a
    // shop, so each of those is offered and neither is forced.
    assert!(reads(route_count(left, ACTION_ROUTE_MOST, "elite"), 0.0));
    for offered in ["rest_site", "shop"] {
        assert!(
            reads(route_count(left, ACTION_ROUTE_LEAST, offered), 0.0),
            "{offered} is on one path of two, so nothing forces it"
        );
        assert!(
            reads(route_count(left, ACTION_ROUTE_MOST, offered), 1.0),
            "{offered} is on one path of two, so the branch offers it"
        );
    }

    // The right branch has one path and it runs through the elite.
    assert!(reads(route_count(right, ACTION_ROUTE_LEAST, "elite"), 1.0));
    assert!(reads(route_count(right, ACTION_ROUTE_MOST, "elite"), 1.0));
    assert!(reads(
        route_count(right, ACTION_ROUTE_MOST, "rest_site"),
        0.0
    ));

    // Which is the whole point: the two steps land on rooms of one type and
    // are told apart by what stands past them.
    assert!(
        lit(left[ACTION_MAP_TYPE + map_type("monster")])
            && lit(right[ACTION_MAP_TYPE + map_type("monster")]),
        "both steps land on a monster room"
    );
    assert_ne!(
        format!("{left:?}"),
        format!("{right:?}"),
        "two monster rooms in different branches are two different options"
    );
}

#[test]
fn the_route_lookahead_survives_a_map_that_loops() {
    // The act's map is a DAG and nothing the build generates loops. A graph
    // that arrives malformed still has to cost a wrong number rather than a
    // hung encoder, so the cycle is answered rather than walked: an edge back
    // into the walk cannot complete a path and contributes nothing.
    let encoder = encoder();
    let mut observation = observation_on_forked_map();
    let map = observation.map.as_mut().expect("the fixture laid a map");
    for point in &mut map.points {
        if point.coord == at(3, 3) {
            point.children = vec![at(3, 0)];
        }
    }
    let rendered = encoder.encode_action(
        &observation,
        &Action::ChooseMap {
            destination: at(4, 1),
        },
    );
    assert!(reads(
        route_count(&rendered.features, ACTION_ROUTE_LEAST, "elite"),
        1.0
    ));
    assert!(
        rendered.features[ACTION_ROUTE_DEPTH] > 0.0,
        "a route settled"
    );
}

#[test]
fn a_map_step_onto_a_coordinate_the_map_does_not_carry_says_so() {
    let encoder = encoder();
    let observation = observation_on_forked_map();
    let known = encoder.encode_action(
        &observation,
        &Action::ChooseMap {
            destination: at(2, 1),
        },
    );
    let unknown = encoder.encode_action(
        &observation,
        &Action::ChooseMap {
            destination: at(6, 9),
        },
    );
    assert!(
        !lit(known.features[ACTION_MAP + 6]),
        "a destination the map carries is not unknown"
    );
    assert!(
        lit(unknown.features[ACTION_MAP + 6]),
        "a destination the map does not carry says so on a flag of its own"
    );
    // Which it has to, because the blocks that would otherwise have said it
    // are empty on every action that names no map point at all.
    assert!(
        (0..MAP_TYPE_NAMES.len()).all(|slot| !lit(unknown.features[ACTION_MAP_TYPE + slot])),
        "nothing is known about the point, so no type is lit"
    );
    assert!(
        (ACTION_ROUTE_LEAST..ACTION_FEATURES).all(|slot| reads(unknown.features[slot], 0.0)),
        "and no route is claimed off it"
    );
}

/// A card-reward screen carrying its own reroll and a relic's button.
fn reward_with_alternatives() -> sts2_engine::AgentObservation {
    let simulator = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::MapNavigation { .. })
    });
    let mut observation = simulator.agent_observation();
    let candidates: Vec<_> = ["CARD.STRIKE_IRONCLAD", "CARD.DEFEND_IRONCLAD"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| sts2_engine::CardHandle {
            card_id: sts2_core::CardInstanceId::from(index as u64 + 1),
            index,
            fingerprint: CardFingerprint::base(id(name)),
            face: sts2_engine::CardFace::default(),
        })
        .collect();
    observation.decision = DecisionContext::ChooseCards {
        choice_id: sts2_core::ChoiceId::from(1),
        identity: ChoiceIdentity::Offered,
        screen: ChoiceScreen::CardReward,
        minimum: 0,
        maximum: 1,
        cancelable: true,
        candidates,
        alternatives: vec![
            ChoiceAlternative {
                option_id: "reroll".to_owned(),
                relic: None,
            },
            ChoiceAlternative {
                option_id: "sacrifice".to_owned(),
                relic: Some(id("RELIC.BURNING_BLOOD")),
            },
        ],
        bundles: Vec::new(),
        picked: Vec::new(),
    };
    observation
}

#[test]
fn the_two_buttons_beside_a_card_reward_encode_apart() {
    let encoder = encoder();
    let observation = reward_with_alternatives();
    let button = |option: &str| {
        encoder.encode_action(
            &observation,
            &Action::ChooseAlternative {
                choice_id: sts2_core::ChoiceId::from(1),
                option_id: option.to_owned(),
            },
        )
    };
    let reroll = button("reroll");
    let sacrifice = button("sacrifice");
    assert!(
        !lit(reroll.features[ACTION_KIND + 1]) && reroll.tokens[0] == 0,
        "the reward's own reroll names no relic"
    );
    assert!(
        lit(sacrifice.features[ACTION_KIND + 1]) && sacrifice.tokens[0] != 0,
        "a relic's button names the relic that put it there"
    );
    assert!(reads(reroll.features[ACTION_INDEX], 0.0));
    assert!(reads(sacrifice.features[ACTION_INDEX], 1.0 / 8.0));
    for both in [&reroll, &sacrifice] {
        assert!(
            reads(both.features[ACTION_CARD + 1], 2.0 / 4.0),
            "each button declines the two cards beside it"
        );
        assert!(reads(both.features[ACTION_PRICE + 3], 2.0 / 4.0));
        assert!(
            lit(both.features[ACTION_KIND + 4]),
            "the screen can be left"
        );
    }
    assert_ne!(format!("{reroll:?}"), format!("{sacrifice:?}"));
}

#[test]
fn a_late_act_macro_observation_does_not_spend_the_token_budget() {
    // Out of combat the encoder pushes the run deck — with a rider for every
    // enchanted card — every point of the act's map, the relics, the belt and
    // whatever the standing screen offers. None of those is capped, so what
    // the budget has to hold is the far end of a run rather than its opening:
    // a macro observation that truncated would silently drop map points,
    // which is precisely what a map step's route lookahead reads.
    let registry = sts2_content::standard_registry();
    let encoder = encoder();
    let mut observation = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::MapNavigation { .. })
    })
    .agent_observation();

    // The widest map the generator lays out, over a sweep of seeds and acts.
    let widest = (0..4)
        .flat_map(|act| {
            [
                "NLD6VZXP94",
                "N3Z9EE2CXB",
                "ZZZZZZZZZZ",
                "QW3RTY8901",
                "ALPHASPIRE1",
            ]
            .map(|seed| sts2_content::standard_act_map(seed, act, 20, true))
        })
        .max_by_key(|map| map.points.len())
        .expect("the sweep generates maps");
    let standing = observation.map.as_ref().expect("a run carries its map");
    observation.map = Some(VisibleMap {
        act: 3,
        width: widest.width,
        height: widest.height,
        start: widest.start,
        boss_coord: widest.boss,
        second_boss_coord: widest.second_boss,
        points: widest
            .points
            .iter()
            .map(|point| VisibleMapPoint {
                coord: point.coord,
                point_type: point.point_type,
                children: point.children.clone(),
                visited: false,
            })
            .collect(),
        boss: standing.boss.clone(),
        second_boss: None,
        ancient: standing.ancient.clone(),
    });

    // A deck at the top of the scalar's own scale, every card enchanted, so
    // that it costs the two tokens an enchanted card costs.
    let of_category = |category: &str| -> Vec<sts2_core::ModelId> {
        registry
            .registered_model_ids()
            .into_iter()
            .filter(|model| model.category() == category)
            .collect()
    };
    let cards = of_category("CARD");
    let enchantments = of_category("ENCHANTMENT");
    observation.deck = cards
        .iter()
        .cycle()
        .take(40)
        .enumerate()
        .map(|(index, model)| {
            let mut print = CardFingerprint::base(model.clone());
            print.enchantment = Some(EnchantmentFingerprint::new(
                enchantments[index % enchantments.len()].clone(),
                3,
            ));
            print
        })
        .collect();
    observation.relics = of_category("RELIC")
        .into_iter()
        .take(25)
        .map(RelicInstance::new)
        .collect();
    observation.potions = vec![Some(of_category("POTION")[0].clone()); 4];

    let tensors = encoder.encode_observation(&observation);
    let live = tensors.live_tokens();
    assert!(
        live > 150,
        "the constructed sight is a late-act one: {live} tokens"
    );
    assert!(
        !lit(tensors.scalars[21]),
        "a late-act macro observation truncated at {live} of {MAX_TOKENS} tokens"
    );
    assert!(
        live < MAX_TOKENS,
        "a late-act macro observation filled {live} of {MAX_TOKENS} token slots"
    );
}

// --- encoding v7: a plan is one scored action -----------------------------

/// A reward line offering one potion.
fn potion_claim(model: &str) -> Action {
    Action::ClaimReward {
        set_id: sts2_core::RewardSetId::from(1),
        reward_index: 0,
        fingerprint: sts2_engine::RewardFingerprint::of("potion").with_model(id(model)),
    }
}

fn discard(slot: usize, model: &str) -> Action {
    Action::DiscardPotion {
        slot,
        model_id: id(model),
    }
}

/// Any live observation: what a claim and a discard encode reads off the
/// action, not off the screen behind it.
fn any_observation() -> sts2_engine::AgentObservation {
    walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::MapNavigation { .. })
    })
    .agent_observation()
}

#[test]
fn the_surrendered_slot_is_empty_on_every_action_that_gives_nothing_up() {
    // The slot names what an action hands over, and stands apart from the
    // pooled naming slots so that a thing given for a thing taken does not
    // pool to the vector of its own reverse. Anything that surrenders nothing
    // leaves it at index 0, which `padding_idx=0` embeds to an exact zero
    // vector.
    //
    // Two kinds of action write it: a potion trade, and an event option whose
    // body gives something up. Neither is reachable by walking a fresh run
    // with the first legal answer, so the walk below is the negative case and
    // the two tests beside it are the positive ones.
    let encoder = encoder();
    let mut simulator = fresh_run();
    for _ in 0..150 {
        if simulator.state().terminal.is_some() {
            break;
        }
        let observation = simulator.agent_observation();
        for action in alphaspire::search::canonical_actions(simulator.legal_actions()) {
            if matches!(action, Action::ChooseEvent { .. }) {
                continue;
            }
            let rendered = encoder.encode_action(&observation, &action);
            assert_eq!(
                rendered.tokens[ACTION_SURRENDERED_TOKEN], 0,
                "{action:?} wrote the surrendered slot"
            );
        }
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }

    // The case that fills every naming slot: a four-card answer names four
    // cards, and must still leave the surrendered slot alone.
    let cards: Vec<_> = [
        "CARD.STRIKE_IRONCLAD",
        "CARD.DEFEND_IRONCLAD",
        "CARD.BASH",
        "CARD.ANGER",
    ]
    .into_iter()
    .enumerate()
    .map(|(index, name)| sts2_engine::CardHandle {
        card_id: sts2_core::CardInstanceId::from(index as u64 + 1),
        index,
        fingerprint: CardFingerprint::base(id(name)),
        face: sts2_engine::CardFace::default(),
    })
    .collect();
    let four = encoder.encode_action(
        &any_observation(),
        &Action::ChooseCards {
            choice_id: sts2_core::ChoiceId::from(1),
            cards,
        },
    );
    assert_eq!(
        four.tokens[..ACTION_NAMING_TOKENS]
            .iter()
            .filter(|token| **token != 0)
            .count(),
        4,
        "the answer names all four cards"
    );
    assert_eq!(
        four.tokens[ACTION_SURRENDERED_TOKEN], 0,
        "and none of them reaches the surrendered slot"
    );
}

#[test]
fn a_trade_and_its_reverse_encode_apart() {
    let encoder = encoder();
    let observation = any_observation();
    let held = "POTION.FIRE_POTION";
    let offered = "POTION.WEAK_POTION";
    let left = encoder.encode_plan(
        &observation,
        &ActionPlan::Trade {
            free: discard(0, held),
            take: potion_claim(offered),
        },
    );
    let right = encoder.encode_plan(
        &observation,
        &ActionPlan::Trade {
            free: discard(0, offered),
            take: potion_claim(held),
        },
    );
    assert_ne!(left, right, "giving up A for B is not giving up B for A");
    assert_eq!(
        left.features, right.features,
        "the two differ in nothing but which potion is where"
    );
    let mut left_pool = left.tokens.clone();
    let mut right_pool = right.tokens.clone();
    left_pool.sort_unstable();
    right_pool.sort_unstable();
    assert_eq!(
        left_pool, right_pool,
        "which a mean over every slot could not see, so the surrendered \
         potion has to sit outside the pooled ones"
    );
}

#[test]
fn a_plan_says_how_the_slot_was_freed_and_a_single_action_says_nothing() {
    let encoder = encoder();
    let observation = any_observation();
    let claim = potion_claim("POTION.WEAK_POTION");
    let bare = encoder.encode_plan(&observation, &ActionPlan::Single(claim.clone()));
    assert!(
        (ACTION_PLAN..ACTION_FEATURES).all(|slot| reads(bare.features[slot], 0.0)),
        "one engine step claims no plan block"
    );
    assert_eq!(
        bare,
        encoder.encode_action(&observation, &claim),
        "and encodes exactly as the action it wraps"
    );

    let thrown = encoder.encode_plan(
        &observation,
        &ActionPlan::Trade {
            free: discard(2, "POTION.FIRE_POTION"),
            take: claim.clone(),
        },
    );
    assert!(lit(thrown.features[ACTION_PLAN]), "it is a plan");
    assert!(lit(thrown.features[ACTION_PLAN + 1]), "freed by discard");
    assert!(
        !lit(thrown.features[ACTION_PLAN + 2]),
        "and not by drinking"
    );
    // Read against the belt the observation carries, never a constant.
    #[allow(clippy::cast_precision_loss, reason = "belt slots are tiny")]
    let belt = observation.potions.len() as f32;
    assert!(reads(thrown.features[ACTION_PLAN + 3], 2.0 / belt));

    let drunk = encoder.encode_plan(
        &observation,
        &ActionPlan::Trade {
            free: Action::UsePotion {
                slot: 2,
                model_id: id("POTION.FIRE_POTION"),
                target: None,
            },
            take: claim,
        },
    );
    assert!(!lit(drunk.features[ACTION_PLAN + 1]), "not thrown away");
    assert!(lit(drunk.features[ACTION_PLAN + 2]), "drunk instead");
    assert_ne!(
        thrown, drunk,
        "and the two ways of making room are two plans"
    );
    assert_eq!(
        thrown.tokens[ACTION_SURRENDERED_TOKEN], drunk.tokens[ACTION_SURRENDERED_TOKEN],
        "both give up the same potion"
    );
    assert!(
        (ACTION_PLAN + 4..ACTION_FEATURES).all(|slot| reads(thrown.features[slot], 0.0)),
        "the reserve is untouched"
    );
}

#[test]
fn the_freed_slot_is_read_against_the_belt_the_player_has() {
    // The belt opens at three, is two under ascension four's Tight Belt, and
    // grows with no fixed bound, so the index cannot be divided by a
    // constant: at two slots a constant of three reads low, and past three it
    // saturates.
    let encoder = encoder();
    let mut observation = any_observation();
    for slots in [2_usize, 3, 5, 10] {
        observation.potions = vec![Some(id("POTION.FIRE_POTION")); slots];
        for slot in 0..slots {
            let rendered = encoder.encode_plan(
                &observation,
                &ActionPlan::Trade {
                    free: discard(slot, "POTION.FIRE_POTION"),
                    take: potion_claim("POTION.WEAK_POTION"),
                },
            );
            #[allow(clippy::cast_precision_loss, reason = "belt slots are tiny")]
            let want = slot as f32 / slots as f32;
            assert!(
                reads(rendered.features[ACTION_PLAN + 3], want),
                "slot {slot} of {slots} reads {} and not {want}",
                rendered.features[ACTION_PLAN + 3]
            );
        }
    }
}

/// A slippery bridge page as the engine builds it: let the rolled card go, or
/// hold on for what the page advertises. There is no third option and no way
/// past it, which is what makes telling the two apart a survival question
/// rather than a preference.
fn bridge_observation(holds: i32) -> sts2_engine::AgentObservation {
    let mut observation = any_observation();
    let hold_on = format!(
        "SLIPPERY_BRIDGE.pages.HOLD_ON_{holds}.options.HOLD_ON_{}",
        holds + 1
    );
    observation.decision = DecisionContext::Event {
        model_id: "EVENT.SLIPPERY_BRIDGE".parse().unwrap(),
        options: vec![
            sts2_engine::EventOption {
                option_id: "SLIPPERY_BRIDGE.pages.INITIAL.options.OVERCOME".to_owned(),
                label: "Overcome".to_owned(),
                effects: vec![sts2_engine::EventEffect::RemoveTheRolledCard {
                    card: Some(sts2_engine::CardFingerprint::base(
                        "CARD.STRIKE_RED".parse().unwrap(),
                    )),
                }],
                is_proceed: false,
                was_chosen: false,
                locked: false,
            },
            sts2_engine::EventOption {
                option_id: hold_on,
                label: "Hold on".to_owned(),
                effects: vec![sts2_engine::EventEffect::HoldOnToTheBridge { cost: 3 + holds }],
                is_proceed: false,
                was_chosen: false,
                locked: false,
            },
        ],
    };
    observation
}

fn bridge_options(
    encoder: &PolicyEncoder,
    observation: &sts2_engine::AgentObservation,
) -> (Vec<f32>, Vec<f32>) {
    let DecisionContext::Event { options, .. } = &observation.decision else {
        panic!("the page is an event");
    };
    let encode = |index: usize| {
        encoder
            .encode_action(
                observation,
                &Action::ChooseEvent {
                    index,
                    option_id: options[index].option_id.clone(),
                },
            )
            .features
    };
    (encode(0), encode(1))
}

#[test]
fn an_event_option_carries_what_taking_it_does() {
    let encoder = encoder();
    let observation = bridge_observation(0);
    let (overcome, hold) = bridge_options(&encoder, &observation);

    // Options sharing an event token must differ in their effect features.
    let differing: Vec<usize> = (0..ACTION_FEATURES)
        .filter(|slot| (overcome[*slot] - hold[*slot]).abs() > f32::EPSILON)
        .filter(|slot| *slot >= ACTION_EVENT)
        .collect();
    assert!(
        differing.len() > 1,
        "the two options must differ in their bodies, not only in their place \
         on the page: {differing:?}"
    );

    let removal = EVENT_EFFECT_NAMES
        .iter()
        .position(|name| *name == "remove_the_rolled_card")
        .unwrap();
    let holding = EVENT_EFFECT_NAMES
        .iter()
        .position(|name| *name == "hold_on_to_the_bridge")
        .unwrap();
    assert!(
        lit(overcome[ACTION_EVENT + removal]),
        "overcome removes a card"
    );
    assert!(!lit(overcome[ACTION_EVENT + holding]));
    assert!(lit(hold[ACTION_EVENT + holding]), "holding on holds on");
    assert!(!lit(hold[ACTION_EVENT + removal]));

    // The removal is a card off the deck and costs nothing; the hold pays.
    assert!(reads(overcome[ACTION_EVENT_NUMBERS + 4], 1.0 / 4.0));
    assert!(reads(overcome[ACTION_EVENT_NUMBERS], 0.0));
    assert!(reads(hold[ACTION_EVENT_NUMBERS], -3.0 / 32.0));
}

#[test]
fn a_hold_on_the_bridge_prices_itself_higher_every_page() {
    // The page ids loop and the effect kinds never change, so the growing
    // cost is the only thing that can tell the fifth hold from the first.
    let encoder = encoder();
    let (_, first) = bridge_options(&encoder, &bridge_observation(0));
    let (_, fifth) = bridge_options(&encoder, &bridge_observation(4));
    assert!(reads(first[ACTION_EVENT_NUMBERS], -3.0 / 32.0));
    assert!(reads(fifth[ACTION_EVENT_NUMBERS], -7.0 / 32.0));
    assert!(
        fifth[ACTION_EVENT_NUMBERS] < first[ACTION_EVENT_NUMBERS],
        "a later hold must read as costing more"
    );
}

/// A relic trader's page as the engine builds it: this relic of yours for
/// that one off the shelf.
fn relic_trade(given: &str, taken: &str) -> sts2_engine::AgentObservation {
    let mut observation = any_observation();
    observation.decision = DecisionContext::Event {
        model_id: "EVENT.RELIC_TRADER".parse().unwrap(),
        options: vec![sts2_engine::EventOption {
            option_id: "RELIC_TRADER.pages.INITIAL.options.TRADE_0".to_owned(),
            label: "Trade".to_owned(),
            effects: vec![sts2_engine::EventEffect::TradeRelic {
                given: given.parse().unwrap(),
                taken: taken.parse().unwrap(),
            }],
            is_proceed: false,
            was_chosen: false,
            locked: false,
        }],
    };
    observation
}

fn only_option(
    encoder: &PolicyEncoder,
    observation: &sts2_engine::AgentObservation,
) -> alphaspire::encoding::ActionEncoding {
    let DecisionContext::Event { options, .. } = &observation.decision else {
        panic!("the page is an event");
    };
    encoder.encode_action(
        observation,
        &Action::ChooseEvent {
            index: 0,
            option_id: options[0].option_id.clone(),
        },
    )
}

#[test]
fn a_relic_trade_and_its_reverse_encode_apart() {
    // The reason the surrendered slot exists, on the screen it was not built
    // for: the naming slots are mean-pooled and a mean is symmetric, so a
    // sword of stone given for a sword of jade would pool to the vector of
    // the trade that hands back what it just took.
    let encoder = encoder();
    let forward = only_option(
        &encoder,
        &relic_trade("RELIC.SWORD_OF_STONE", "RELIC.SWORD_OF_JADE"),
    );
    let backward = only_option(
        &encoder,
        &relic_trade("RELIC.SWORD_OF_JADE", "RELIC.SWORD_OF_STONE"),
    );

    assert_ne!(
        forward.tokens[ACTION_SURRENDERED_TOKEN], 0,
        "a trade names what it gives up"
    );
    assert_ne!(
        forward.tokens[ACTION_SURRENDERED_TOKEN], backward.tokens[ACTION_SURRENDERED_TOKEN],
        "and the two directions give up different relics"
    );
    assert_ne!(
        forward.tokens, backward.tokens,
        "so a trade and its reverse are two actions, not one"
    );
    // The effect kinds and the resources are the same on both, which is
    // exactly why the identities have to carry the difference.
    assert_eq!(forward.features, backward.features);
}
