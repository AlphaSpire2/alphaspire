//! Decision scripts: the interchange artifact of the validation loop.
//!
//! On disk a script is the decision-script profile of `.sts2pgn` Format v1:
//! same grammar as a recording, `Producer` instead of `RecorderVersion`,
//! `Mode CUSTOM` (the one workflow where the game accepts a chosen seed), a
//! config-level `run.start` carrying the expected acts and unlock state, and
//! decision records with full semantic identity. The driver verifies each
//! fingerprint against what the live game offers and refuses on the first
//! mismatch; the recording it makes while driving — never the script — is
//! the conformance artifact.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sts2_engine::{
    Action, ActiveRoom, CardHandle, ChoiceAlternative, ChoiceIdentity, ChoiceScreen,
    DecisionContext, EngineError, ErrorCode, ShopItem, ShopOffer, ShopSlot, Simulator,
};
use sts2_replay::observation::{
    render_card_fingerprint as card_fp, render_reward_fingerprint as reward_fp,
};

/// The producer identity every emitted script carries.
pub const PRODUCER: &str = concat!("alphaspire/", env!("CARGO_PKG_VERSION"));

/// One run's worth of choices, reproducible by construction: the
/// configuration names the run, the analysis seed names the policy's own
/// randomness, and the actions are what it chose.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DecisionScript {
    /// The run seed, in the game's own seed alphabet.
    pub seed: String,
    /// `CHARACTER.*` model id.
    pub character: String,
    pub ascension: u8,
    /// The unlock preset the run was generated against, by preset id; the
    /// emitted profile carries the full expected unlock state instead.
    pub unlock_preset: String,
    /// The analysis seed the generating policy drew its randomness from.
    pub analysis_seed: u64,
    /// What the policy chose, in order.
    pub actions: Vec<Action>,
}

/// Writes the decision-script profile as a run is walked: one record per
/// action, appended with the offer context the simulator is standing on.
#[derive(Clone, Debug)]
pub struct ScriptWriter {
    lines: Vec<String>,
    sequence: u64,
}

impl ScriptWriter {
    /// The header tags and the config-level `run.start`, read off the run the
    /// simulator was generated as.
    pub fn open(simulator: &Simulator, seed: &str, ascension: u8) -> Result<Self, EngineError> {
        let state = simulator.state();
        let run = state
            .run
            .as_ref()
            .ok_or_else(|| EngineError::new(ErrorCode::IllegalAction))?;
        let character = run
            .config
            .players
            .first()
            .map(|player| player.character.entry().to_owned())
            .ok_or_else(|| EngineError::new(ErrorCode::IllegalAction))?;
        let players = serde_json::to_string(&json!([
            {"slot": "p0", "character": character}
        ]))
        .expect("players serialize");
        let players_tag = players.replace('\\', "\\\\").replace('"', "\\\"");
        let lines = vec![
            "[Format \"STS2PGN\"]".to_owned(),
            "[FormatVersion \"1\"]".to_owned(),
            format!(
                "[Profile \"{}\"]",
                sts2_replay::TraceProfile::Script.as_str()
            ),
            format!("[Producer \"{PRODUCER}\"]"),
            format!("[Seed \"{seed}\"]"),
            "[Mode \"CUSTOM\"]".to_owned(),
            format!("[Ascension \"{ascension}\"]"),
            format!("[Players \"{players_tag}\"]"),
            format!("[GameVersion \"{}\"]", sts2_core::PINNED_GAME_VERSION),
            format!("[GameCommit \"{}\"]", sts2_core::PINNED_GAME_COMMIT),
            format!("[ModelIdHash \"{}\"]", sts2_core::PINNED_MODEL_ID_HASH),
            String::new(),
        ];
        let character_model = format!("CHARACTER.{character}");
        let acts: Vec<Value> = run
            .acts
            .iter()
            .enumerate()
            .map(|(index, act)| json!({"index": index, "model_id": act.act_model.to_string()}))
            .collect();
        let start = json!({
            "actor": null,
            "data": {"state": {
                "scope": "run_start.core",
                "run": {"seed": seed, "mode": "custom", "ascension": ascension, "act": 0, "floor": 0},
                "acts": acts,
                "players": [{
                    "slot": "p0",
                    "character_model": character_model,
                    "unlocks": {
                        "epochs": run.unlocks.epochs,
                        "encounters_seen": run.unlocks.encounters_seen,
                        "number_of_runs": run.unlocks.number_of_runs,
                    },
                }],
            }},
        });
        let mut writer = Self { lines, sequence: 0 };
        writer.push("run.start", &start);
        Ok(writer)
    }

    fn push(&mut self, kind: &str, payload: &Value) {
        self.sequence += 1;
        let line = format!(
            "{} {kind} {}",
            self.sequence,
            serde_json::to_string(payload).expect("payloads serialize")
        );
        self.lines.push(line);
    }

    fn push_decision(&mut self, kind: &str, data: &Value) {
        self.push(kind, &json!({"actor": "p0", "data": data}));
    }

    /// Appends the record for one action, read against the decision point the
    /// simulator is standing on — call before stepping the action.
    ///
    /// One record per *engine* action, never per policy decision: an
    /// [`ActionPlan`](crate::plan::ActionPlan) that expands into several
    /// steps writes one record for each, each against the screen that step is
    /// taken on, so a script stays the sequence of clicks a driver replays
    /// into the live game.
    pub fn record(&mut self, simulator: &Simulator, action: &Action) -> Result<(), Unscriptable> {
        let (kind, data) = encode(simulator, action)?;
        self.push_decision(kind, &data);
        Ok(())
    }

    /// The finished script, terminal token `*`: nothing authoritative ended.
    #[must_use]
    pub fn finish(mut self) -> String {
        self.lines.push("*".to_owned());
        self.lines.push(String::new());
        self.lines.join("\n")
    }
}

/// A decision the profile cannot carry: an action read against a decision
/// point that does not offer it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Unscriptable(pub String);

impl std::fmt::Display for Unscriptable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "unscriptable decision: {}", self.0)
    }
}

impl std::error::Error for Unscriptable {}

fn indexed_cards(candidates: &[CardHandle]) -> Vec<Value> {
    candidates
        .iter()
        .map(|candidate| {
            json!({
                "index": candidate.index,
                "fingerprint": card_fp(&candidate.fingerprint),
            })
        })
        .collect()
}

fn indexed_relics(relics: &[sts2_core::ModelId]) -> Vec<Value> {
    relics
        .iter()
        .enumerate()
        .map(|(index, relic)| json!({"index": index, "model_id": relic.to_string()}))
        .collect()
}

fn offered_choice(
    screen: ChoiceScreen,
    candidates: &[CardHandle],
    bundles: &[Vec<usize>],
    alternatives: &[ChoiceAlternative],
) -> Result<Value, Unscriptable> {
    let cards = indexed_cards(candidates);
    let offered = match screen {
        ChoiceScreen::ChooseACard => {
            json!({"context": "choose_a_card", "options": {"options": cards}})
        }
        ChoiceScreen::CardGrid => {
            json!({"context": "card_grid", "options": {"options": cards}})
        }
        ChoiceScreen::RewardGrid => {
            json!({"context": "reward_grid", "options": {"options": cards}})
        }
        ChoiceScreen::CardBundle => {
            let options = bundles
                .iter()
                .enumerate()
                .map(|(index, bundle)| {
                    json!({
                        "index": index,
                        "cards": bundle
                            .iter()
                            .filter_map(|candidate| candidates.get(*candidate))
                            .map(|candidate| card_fp(&candidate.fingerprint))
                            .collect::<Vec<_>>(),
                    })
                })
                .collect::<Vec<_>>();
            json!({"context": "card_bundle", "options": {"options": options}})
        }
        ChoiceScreen::CardReward => {
            let alternatives = std::iter::once(json!({
                "index": candidates.len(),
                "option_id": "skip",
            }))
            .chain(
                alternatives
                    .iter()
                    .enumerate()
                    .map(|(offset, alternative)| {
                        json!({
                            "index": candidates.len() + 1 + offset,
                            "option_id": alternative.option_id,
                        })
                    }),
            )
            .collect::<Vec<_>>();
            json!({
                "context": "card_reward",
                "options": {"cards": cards, "alternatives": alternatives},
            })
        }
        ChoiceScreen::Unknown
        | ChoiceScreen::Hand
        | ChoiceScreen::CombatPile
        | ChoiceScreen::Deck => {
            return Err(Unscriptable(format!(
                "offered cards use unsupported screen {screen:?}"
            )));
        }
    };
    Ok(offered)
}

fn selected_indexes(
    screen: ChoiceScreen,
    candidates: &[CardHandle],
    bundles: &[Vec<usize>],
    cards: &[CardHandle],
) -> Result<Vec<usize>, Unscriptable> {
    match screen {
        ChoiceScreen::CardBundle => {
            let chosen = cards.iter().map(|card| card.index).collect::<Vec<_>>();
            Ok(vec![
                bundles
                    .iter()
                    .position(|bundle| bundle == &chosen)
                    .ok_or_else(|| {
                        Unscriptable("a bundle answer names no offered bundle".to_owned())
                    })?,
            ])
        }
        ChoiceScreen::CardReward if cards.is_empty() => Ok(vec![candidates.len()]),
        ChoiceScreen::ChooseACard
        | ChoiceScreen::CardGrid
        | ChoiceScreen::RewardGrid
        | ChoiceScreen::CardReward => Ok(cards.iter().map(|card| card.index).collect()),
        ChoiceScreen::Unknown
        | ChoiceScreen::Hand
        | ChoiceScreen::CombatPile
        | ChoiceScreen::Deck => Err(Unscriptable(format!(
            "offered cards use unsupported screen {screen:?}"
        ))),
    }
}

/// The `in_combat` a potion record carries, which is whether the game
/// considers a combat in progress and not "a fight is standing". A won
/// fight's state outlives it: combat is marked over before the win is
/// announced, and the `CombatState` stays up behind the rewards screen
/// until the room is left. A drink taken there is an out-of-combat drink,
/// which is what the engine itself reads in `use_potion`.
fn in_a_fight(simulator: &Simulator) -> bool {
    simulator
        .state()
        .combat
        .as_ref()
        .is_some_and(|combat| combat.in_progress)
}

/// The entries a purchase is read against, which are not always a merchant's.
///
/// A merchant is answered on its own screen, so its inventory is the shop
/// decision's. An event that stocks a shelf of its own is left by walking on,
/// so the shelf is bought off the map decision that stands over the finished
/// page — and the game stocks it as an ordinary inventory of ordinary entries,
/// bought through the one purchase wrapper, so both are the same record.
fn purchase_offers(simulator: &Simulator) -> Result<&[ShopOffer], Unscriptable> {
    if let DecisionContext::Shop { offers, .. } = simulator.decision() {
        return Ok(offers);
    }
    match simulator
        .state()
        .run
        .as_ref()
        .and_then(|run| run.active_room.as_ref())
    {
        Some(ActiveRoom::Event { shelf, .. }) if !shelf.is_empty() => Ok(shelf),
        _ => Err(Unscriptable("a purchase without its shop".to_owned())),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per action kind the profile carries"
)]
fn encode(simulator: &Simulator, action: &Action) -> Result<(&'static str, Value), Unscriptable> {
    let record = match action {
        Action::ChooseMap { destination } => (
            "map.choose",
            json!({"col": destination.col, "row": destination.row}),
        ),
        Action::ChooseEvent { index, option_id } => {
            let event_id = match simulator.decision() {
                DecisionContext::Event { model_id, .. } => Some(model_id.to_string()),
                _ => None,
            };
            (
                "event.choose",
                json!({"event_id": event_id, "option_id": option_id, "option_index": index}),
            )
        }
        Action::BuyShopItem {
            offer_index,
            fingerprint,
        } => {
            let model = match fingerprint {
                ShopItem::Card(card) => card.model_id.to_string(),
                ShopItem::Potion(model) | ShopItem::Relic(model) => model.to_string(),
            };
            let entry_type = match fingerprint {
                ShopItem::Card(_) => "card",
                ShopItem::Relic(_) => "relic",
                ShopItem::Potion(_) => "potion",
            };
            let offers = purchase_offers(simulator)?;
            let group = |offer: &ShopOffer| match offer.slot {
                ShopSlot::CharacterCard | ShopSlot::CharacterCardOfType(_) => "character_card",
                ShopSlot::ColorlessCard(_) => "colorless_card",
                ShopSlot::Relic => "relic",
                ShopSlot::Potion => "potion",
            };
            let offer = offers
                .iter()
                .find(|offer| offer.index == *offer_index)
                .ok_or_else(|| Unscriptable("a purchase names no offer".to_owned()))?;
            let entry_group = group(offer);
            let entry_index = offers
                .iter()
                .filter(|other| other.index < offer.index && group(other) == entry_group)
                .count();
            (
                "shop.purchase",
                json!({
                    "inventory_index": offer_index,
                    "entry_index": entry_index,
                    "entry_group": entry_group,
                    "entry_type": entry_type,
                    "model_id": model,
                    "cost": offer.price,
                    "ignore_cost": false,
                }),
            )
        }
        Action::BuyCardRemoval => {
            let DecisionContext::Shop { removal_price, .. } = simulator.decision() else {
                return Err(Unscriptable("a removal without its shop".to_owned()));
            };
            ("shop.remove_card", json!({"cost": removal_price}))
        }
        Action::UsePotion {
            slot,
            model_id,
            target,
        } => (
            "combat.use_potion",
            json!({
                "potion_slot": slot,
                "potion_model": model_id.to_string(),
                "in_combat": in_a_fight(simulator),
                "target_id": target.as_ref().map(|target| target.combat_id.get()),
                "target": target.as_ref().map(|target| json!({
                    "combat_id": target.combat_id.get(),
                    "model_id": target.model_id.to_string(),
                })),
            }),
        ),
        Action::DiscardPotion { slot, model_id } => (
            "combat.discard_potion",
            json!({
                "potion_slot": slot,
                "potion_model": model_id.to_string(),
                "in_combat": in_a_fight(simulator),
            }),
        ),
        Action::RestOption { option, .. } => {
            ("rest.choose", json!({"option_id": option.option_id()}))
        }
        Action::TakeTreasure { relic } => {
            let DecisionContext::Treasure { relics } = simulator.decision() else {
                return Err(Unscriptable("a treasure without its room".to_owned()));
            };
            let relic_index = relics
                .iter()
                .position(|offered| offered == relic)
                .ok_or_else(|| Unscriptable("a treasure names no offered relic".to_owned()))?;
            (
                "treasure.choose_relic",
                json!({
                    "relic_model": relic.to_string(),
                    "relic_index": relic_index,
                    "offered_relics": indexed_relics(relics),
                }),
            )
        }
        Action::UncoverCrystalSphere { x, y, tool } => {
            let DecisionContext::CrystalSphere { divinations } = simulator.decision() else {
                return Err(Unscriptable("a divination without its board".to_owned()));
            };
            (
                "event.crystal_sphere.click",
                json!({
                    "x": x,
                    "y": y,
                    "tool": tool,
                    "remaining_before": divinations,
                }),
            )
        }
        Action::AdvanceAct => ("act.next", json!({})),
        Action::PlayCard { card, target } => (
            "combat.play_card",
            json!({
                "combat_card_index": card.card_id.get(),
                "card_model": card.fingerprint.model_id.to_string(),
                "card": card_fp(&card.fingerprint),
                "target_id": target.as_ref().map(|target| target.combat_id.get()),
                "target": target.as_ref().map(|target| json!({
                    "combat_id": target.combat_id.get(),
                    "model_id": target.model_id.to_string(),
                })),
            }),
        ),
        Action::EndTurn { turn } => ("combat.end_turn", json!({"turn": turn})),
        Action::ChooseCards { choice_id, cards } => {
            let DecisionContext::ChooseCards {
                identity,
                screen,
                candidates,
                alternatives,
                bundles,
                ..
            } = simulator.decision()
            else {
                return Err(Unscriptable("card choice without its screen".to_owned()));
            };
            match identity {
                ChoiceIdentity::DeckCard => (
                    "choice.cards",
                    json!({
                        "choice_id": choice_id.get(),
                        "choice_type": "deck_card",
                        "cards": cards.iter().map(|card| json!({
                            "deck_index": card.index,
                            "fingerprint": card_fp(&card.fingerprint),
                        })).collect::<Vec<_>>(),
                    }),
                ),
                ChoiceIdentity::CombatCard => (
                    "choice.cards",
                    json!({
                        "choice_id": choice_id.get(),
                        "choice_type": "combat_card",
                        "cards": cards.iter().map(|card| json!({
                            "combat_card_index": card.card_id.get(),
                            "fingerprint": card_fp(&card.fingerprint),
                        })).collect::<Vec<_>>(),
                    }),
                ),
                ChoiceIdentity::Offered => {
                    let offered = offered_choice(*screen, candidates, bundles, alternatives)?;
                    let indexes = selected_indexes(*screen, candidates, bundles, cards)?;
                    (
                        "choice.index",
                        json!({
                            "choice_id": choice_id.get(),
                            "choice_type": "index",
                            "indexes": indexes,
                            "offered": offered,
                        }),
                    )
                }
            }
        }
        Action::ChooseAlternative {
            choice_id,
            option_id,
        } => {
            let DecisionContext::ChooseCards {
                screen,
                candidates,
                alternatives,
                bundles,
                ..
            } = simulator.decision()
            else {
                return Err(Unscriptable("an alternative without its screen".to_owned()));
            };
            if *screen != ChoiceScreen::CardReward {
                return Err(Unscriptable(
                    "an alternative is not on a card-reward screen".to_owned(),
                ));
            }
            let offset = alternatives
                .iter()
                .position(|alternative| alternative.option_id == *option_id)
                .ok_or_else(|| {
                    Unscriptable("the screen does not offer that alternative".to_owned())
                })?;
            (
                "choice.index",
                json!({
                    "choice_id": choice_id.get(),
                    "choice_type": "index",
                    "indexes": [candidates.len() + 1 + offset],
                    "offered": offered_choice(*screen, candidates, bundles, alternatives)?,
                }),
            )
        }
        Action::ClaimReward {
            set_id,
            reward_index,
            fingerprint,
        } => (
            "reward.claim",
            json!({
                "set_id": set_id.get(),
                "reward_index": reward_index,
                "reward": reward_fp(fingerprint),
            }),
        ),
        Action::Proceed => proceed_record(
            simulator.decision(),
            simulator
                .state()
                .combat
                .as_ref()
                .is_some_and(|combat| combat.reward_set.is_some()),
        ),
    };
    Ok(record)
}

/// A combat reward screen is the room's terminal rewards screen. Its Proceed
/// button skips anything left in the set and leaves the room in one operation,
/// which the recorder names `room.proceed`. Run-level reward sets are nested
/// loot screens (for example, rewards opened by an event); their Skip button
/// closes only that set and is recorded as `reward.skip`.
fn proceed_record(
    decision: &DecisionContext,
    has_combat_reward_set: bool,
) -> (&'static str, Value) {
    if let DecisionContext::Rewards { set_id, .. } = decision
        && !has_combat_reward_set
    {
        ("reward.skip", json!({"set_id": set_id.get()}))
    } else {
        ("room.proceed", json!({}))
    }
}

#[cfg(test)]
mod tests {
    use super::{encode, indexed_relics, offered_choice, proceed_record, selected_indexes};
    use serde_json::json;
    use sts2_engine::{
        Action, ActiveRoom, CardFingerprint, CardHandle, ChoiceAlternative, ChoiceScreen,
        DecisionContext, PileName, RunPhase, ScenarioBuilder, ShopItem, ShopOffer, ShopSlot,
        Simulator,
    };

    fn card(index: usize, model: &str) -> CardHandle {
        CardHandle {
            index,
            card_id: (index as u64 + 1).into(),
            fingerprint: CardFingerprint::base(model.parse().expect("test model ID")),
            face: sts2_engine::CardFace::default(),
        }
    }

    #[test]
    fn offer_contexts_follow_the_recorder_screen_names() {
        let cards = vec![card(0, "CARD.ONE"), card(1, "CARD.TWO")];
        for (screen, context) in [
            (ChoiceScreen::ChooseACard, "choose_a_card"),
            (ChoiceScreen::CardGrid, "card_grid"),
            (ChoiceScreen::RewardGrid, "reward_grid"),
        ] {
            let offered = offered_choice(screen, &cards, &[], &[]).expect("supported screen");
            assert_eq!(offered["context"], context);
            assert_eq!(
                offered["options"]["options"].as_array().map(Vec::len),
                Some(2)
            );
        }

        let bundle = offered_choice(ChoiceScreen::CardBundle, &cards, &[vec![0, 1]], &[])
            .expect("bundle screen");
        assert_eq!(bundle["context"], "card_bundle");
        assert_eq!(
            bundle["options"]["options"][0]["cards"]
                .as_array()
                .map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn treasure_offers_name_each_relic_by_index_and_model() {
        let relics = vec![
            "RELIC.BLOOD_VIAL".parse().expect("test model ID"),
            "RELIC.TINY_MAILBOX".parse().expect("test model ID"),
        ];
        assert_eq!(
            indexed_relics(&relics),
            vec![
                json!({"index": 0, "model_id": "RELIC.BLOOD_VIAL"}),
                json!({"index": 1, "model_id": "RELIC.TINY_MAILBOX"}),
            ]
        );
    }

    #[test]
    fn offers_use_canonical_card_fingerprints() {
        let mut candidate = card(0, "CARD.SAVED");
        candidate.fingerprint.upgrade_level = 2;
        candidate.fingerprint.floor_added_to_deck = Some(7);
        candidate
            .fingerprint
            .properties
            .insert("CombatsSeen".to_owned(), json!(3));
        let offered = offered_choice(ChoiceScreen::ChooseACard, &[candidate], &[], &[])
            .expect("choose-a-card screen");
        assert_eq!(
            offered["options"]["options"][0]["fingerprint"],
            json!({
                "model_id": "CARD.SAVED",
                "upgrade_level": 2,
                "floor_added_to_deck": 7,
                "enchantment": null,
                "properties": {
                    "bools": null,
                    "card_arrays": null,
                    "cards": null,
                    "int_arrays": null,
                    "ints": [{"name": "CombatsSeen", "value": 3}],
                    "model_ids": null,
                    "strings": null,
                },
            })
        );
    }

    #[test]
    fn card_reward_positions_skip_before_relic_alternatives() {
        let cards = vec![card(0, "CARD.ONE"), card(1, "CARD.TWO")];
        let alternatives = vec![
            ChoiceAlternative {
                option_id: "reroll".to_owned(),
                relic: None,
            },
            ChoiceAlternative {
                option_id: "sacrifice".to_owned(),
                relic: None,
            },
        ];
        let offered = offered_choice(ChoiceScreen::CardReward, &cards, &[], &alternatives)
            .expect("card reward");
        assert_eq!(offered["context"], "card_reward");
        assert_eq!(
            offered["options"]["alternatives"],
            json!([
                {"index": 2, "option_id": "skip"},
                {"index": 3, "option_id": "reroll"},
                {"index": 4, "option_id": "sacrifice"},
            ])
        );
        assert_eq!(
            selected_indexes(ChoiceScreen::CardReward, &cards, &[], &[]).expect("skip index"),
            vec![2]
        );
        assert_eq!(
            selected_indexes(ChoiceScreen::CardGrid, &cards, &[], &[]).expect("grid cancel"),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn discarded_potions_carry_the_live_combat_context() {
        let simulator =
            sts2_content::standard_ironclad_run_at("NLD6VZXP94", 0).expect("seeded run");
        let action = Action::DiscardPotion {
            slot: 0,
            model_id: "POTION.FIRE_POTION".parse().expect("test model ID"),
        };
        let (kind, data) = encode(&simulator, &action).expect("discard is scriptable");
        assert_eq!(kind, "combat.discard_potion");
        assert_eq!(data["in_combat"], false);
    }

    /// The reward screen behind a won fight is not a fight. The game marks
    /// combat over before it announces the win, and the `CombatState` stays
    /// up until the room is left, so a drink taken here is the driver's
    /// out-of-combat drink. Writing "a fight is standing" instead fails
    /// playback on `the potion action's combat context differs`.
    #[test]
    fn a_potion_over_a_won_fight_is_not_drunk_in_combat() {
        let mut fingerprint =
            CardFingerprint::base("CARD.STRIKE_IRONCLAD".parse().expect("static model ID"));
        fingerprint.floor_added_to_deck = Some(1);
        let mut simulator = ScenarioBuilder::new(
            "WON",
            "ENCOUNTER.SHRINKER_BEETLE_WEAK"
                .parse()
                .expect("static model ID"),
        )
        .player_hp(80, 80)
        .energy(9, 9)
        .turn(1, 1)
        .enemy(
            1,
            "MONSTER.SHRINKER_BEETLE".parse().expect("static model ID"),
            1,
            1,
            "SHRINKER_MOVE",
        )
        .card(PileName::Hand, 0, fingerprint, 0)
        .potions(vec![
            Some("POTION.FIRE_POTION".parse().expect("static model ID")),
            None,
            None,
        ])
        .build(sts2_content::standard_registry())
        .expect("the scenario builds");
        let strike = simulator
            .legal_actions()
            .iter()
            .find(|action| matches!(action, Action::PlayCard { .. }))
            .cloned()
            .expect("the strike is offered");
        simulator.step(strike).expect("the strike applies");

        let combat = simulator
            .state()
            .combat
            .as_ref()
            .expect("the fight's state outlives it");
        assert!(
            !combat.in_progress,
            "the fight is over even though its state is still standing"
        );

        let discard = simulator
            .legal_actions()
            .iter()
            .find(|action| matches!(action, Action::DiscardPotion { .. }))
            .cloned()
            .expect("the belt is reachable over the rewards");
        let (kind, data) = encode(&simulator, &discard).expect("discard is scriptable");
        assert_eq!(kind, "combat.discard_potion");
        assert_eq!(data["in_combat"], false);
    }

    /// The run standing in an event that carries no page of its own and
    /// stocks `relics` on a shelf instead, priced at fifty each, with gold
    /// enough for any of them. `EVENT.FAKE_MERCHANT` is that event: its page
    /// is empty, so the map stands over it and the shelf is bought off the
    /// map's own decision.
    fn standing_at_an_event_shelf(relics: &[&str]) -> Simulator {
        let mut simulator =
            sts2_content::standard_ironclad_run_at("NLD6VZXP94", 0).expect("seeded run");
        for _ in 0..300 {
            if matches!(simulator.decision(), DecisionContext::MapNavigation { .. }) {
                break;
            }
            let action = simulator
                .legal_actions()
                .first()
                .cloned()
                .expect("a live run offers an action");
            simulator.step_quietly(&action).expect("the step applies");
        }
        let mut state = simulator.state().clone();
        state.run_player.gold = 500;
        let run = state.run.as_mut().expect("the state is in a run");
        run.phase = RunPhase::Room;
        run.active_room = Some(ActiveRoom::Event {
            model_id: "EVENT.FAKE_MERCHANT".parse().expect("static model ID"),
            options: Vec::new(),
            rng_counter: 0,
            shelf: relics
                .iter()
                .enumerate()
                .map(|(index, relic)| ShopOffer {
                    index,
                    item: ShopItem::Relic(relic.parse().expect("static model ID")),
                    slot: ShopSlot::Relic,
                    base_price: 50,
                    price: 50,
                    sold: false,
                })
                .collect(),
            standing_fight: None,
            rolled_card: None,
            skipped_cards: Vec::new(),
            sphere: None,
        });
        Simulator::from_scenario(state, sts2_content::standard_registry())
            .expect("the state rebuilds")
    }

    /// An event's shelf is stocked as an ordinary merchant inventory and
    /// bought through the one purchase wrapper, so it is recorded as the
    /// purchase it is. Reading it against the shop decision that is not
    /// standing threw the whole run's script away over one relic.
    #[test]
    fn a_shelf_an_event_owns_is_recorded_as_a_purchase() {
        let simulator = standing_at_an_event_shelf(&["RELIC.FAKE_ANCHOR", "RELIC.FAKE_MANGO"]);
        assert!(
            matches!(simulator.decision(), DecisionContext::MapNavigation { .. }),
            "the finished page leaves the map standing over the shelf"
        );
        let buy = simulator
            .legal_actions()
            .iter()
            .filter(|action| matches!(action, Action::BuyShopItem { .. }))
            .nth(1)
            .cloned()
            .expect("the engine offers both entries");

        let (kind, data) = encode(&simulator, &buy).expect("a shelf purchase is scriptable");
        assert_eq!(kind, "shop.purchase");
        assert_eq!(
            data,
            json!({
                "inventory_index": 1,
                "entry_index": 1,
                "entry_group": "relic",
                "entry_type": "relic",
                "model_id": "RELIC.FAKE_MANGO",
                "cost": 50,
                "ignore_cost": false,
            })
        );
    }

    #[test]
    fn proceeding_from_combat_rewards_leaves_the_room() {
        let rewards = DecisionContext::Rewards {
            set_id: 7_u64.into(),
            player_slot: "p0".to_owned(),
            offers: Vec::new(),
        };
        assert_eq!(proceed_record(&rewards, true), ("room.proceed", json!({})));
        assert_eq!(
            proceed_record(&rewards, false),
            ("reward.skip", json!({"set_id": 7}))
        );
        assert_eq!(
            proceed_record(&DecisionContext::RoomProceed, false),
            ("room.proceed", json!({}))
        );
    }
}
