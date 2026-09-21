//! Action plans over pinned content: what a full belt offers, what it
//! expands into, and that a belt with room is untouched.

use alphaspire::plan::{ActionPlan, forced_step, free_claim, is_forced_step, permitted_plans};
use alphaspire::policy::permitted_actions;
use alphaspire::search::canonical_actions;
use sts2_core::UnlockPresetManifest;
use sts2_engine::{
    Action, ActiveRoom, CardFingerprint, DecisionContext, RelicInstance, RestSiteOption,
    RestSiteOptionState, RewardFingerprint, RewardItem, RoomAt, RoomDefinition, ShopItem,
    ShopOffer, ShopSlot, Simulator,
};

const SEED: &str = "NLD6VZXP94";
/// `PotionUsage.CombatOnly`: held, discardable, and never drinkable outside a
/// fight.
const THROWN: &str = "POTION.VULNERABLE_POTION";
/// `PotionUsage.AnyTime`: the belt pours this one at a screen.
const DRUNK: &str = "POTION.BLOOD_POTION";
const PRICE: i32 = 40;
/// Thrown at a merchant rather than drunk: at the fake merchant its body
/// turns him on the player and the fight starts where the screen stood.
const FOUL: &str = "POTION.FOUL_POTION";

fn id(value: &str) -> sts2_core::ModelId {
    value.parse().expect("static model ID")
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

fn rebuilt(simulator: &Simulator, edit: impl FnOnce(&mut sts2_engine::GameState)) -> Simulator {
    let mut state = simulator.state().clone();
    edit(&mut state);
    Simulator::from_scenario(state, sts2_content::standard_registry()).expect("the state rebuilds")
}

/// The belt resized to `slots` and holding the named models from slot zero.
///
/// Resized because the belt is not a constant: it opens at three, is two at
/// ascension four's Tight Belt, and grows with no fixed bound —
/// `RELIC.POTION_BELT` adds two, `RELIC.PHIAL_HOLSTER` one and
/// `RELIC.ALCHEMICAL_COFFER` four.
fn hold(state: &mut sts2_engine::GameState, slots: usize, potions: &[&str]) {
    state.run_player.potions = (0..slots)
        .map(|slot| potions.get(slot).map(|model| id(model)))
        .collect();
}

/// A won fight's reward screen offering one potion, with the belt holding
/// `potions`.
fn potion_reward(slots: usize, potions: &[&str]) -> Simulator {
    let screen = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::Rewards { .. })
    });
    rebuilt(&screen, |state| {
        hold(state, slots, potions);
        let set = state
            .combat
            .as_mut()
            .and_then(|combat| combat.reward_set.as_mut())
            .or_else(|| state.run.as_mut().and_then(|run| run.reward_set.as_mut()))
            .expect("the screen is open");
        set.rewards = vec![RewardItem {
            index: 0,
            fingerprint: RewardFingerprint::of("potion").with_model(id(THROWN)),
            selected: false,
            passed_over: false,
            opened: false,
        }];
    })
}

/// A won fight's reward screen offering exactly `rewards`, with `relics` in
/// the player's bag.
fn reward_screen(rewards: Vec<RewardItem>, relics: &[&str]) -> Simulator {
    let screen = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::Rewards { .. })
    });
    rebuilt(&screen, |state| {
        for relic in relics {
            state.run_player.relics.push(RelicInstance::new(id(relic)));
        }
        let set = state
            .combat
            .as_mut()
            .and_then(|combat| combat.reward_set.as_mut())
            .or_else(|| state.run.as_mut().and_then(|run| run.reward_set.as_mut()))
            .expect("the screen is open");
        set.rewards = rewards;
    })
}

/// One reward of a kind, with nothing on its fingerprint but what `edit`
/// puts there.
fn reward_of(kind: &str, edit: impl FnOnce(&mut RewardFingerprint)) -> RewardItem {
    let mut fingerprint = RewardFingerprint::of(kind);
    edit(&mut fingerprint);
    RewardItem {
        index: 0,
        fingerprint,
        selected: false,
        passed_over: false,
        opened: false,
    }
}

/// A card reward offering `cards`, with the reroll button where `can_reroll`
/// asks for one and `relics` in the bag.
fn card_reward(cards: &[&str], can_reroll: bool, relics: &[&str]) -> Simulator {
    let offered: Vec<CardFingerprint> = cards
        .iter()
        .map(|model| CardFingerprint::base(id(model)))
        .collect();
    reward_screen(
        vec![reward_of("card", |fingerprint| {
            fingerprint.option_count = offered.len();
            fingerprint.offered_cards = offered;
            fingerprint.can_reroll = can_reroll;
        })],
        relics,
    )
}

/// The run standing in a room of `definition`, walked into off the map.
fn entered(definition: RoomDefinition) -> Simulator {
    let map = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::MapNavigation { .. })
    });
    let destination = map
        .legal_actions()
        .iter()
        .find_map(|action| match action {
            Action::ChooseMap { destination } => Some(*destination),
            _ => None,
        })
        .expect("the map offers a step");
    let mut standing = rebuilt(&map, |state| {
        let run = state.run.as_mut().expect("the state is in a run");
        let act = run.current_act;
        run.rooms[act].retain(|room| room.coord != destination);
        run.rooms[act].push(RoomAt {
            coord: destination,
            definition,
        });
    });
    standing
        .step_quietly(&Action::ChooseMap { destination })
        .expect("the walk applies");
    standing
}

/// A rest site standing with exactly `options` on its list.
fn rest_site(options: &[RestSiteOption]) -> Simulator {
    let site = entered(RoomDefinition::RestSite);
    rebuilt(&site, |state| {
        let run = state.run.as_mut().expect("the state is in a run");
        run.active_room = Some(ActiveRoom::RestSite {
            options: options
                .iter()
                .map(|option| RestSiteOptionState {
                    option: *option,
                    standing: true,
                })
                .collect(),
            left_open: false,
        });
    })
}

/// A merchant stocking one affordable potion, with the belt holding
/// `potions`.
fn shop(slots: usize, potions: &[&str]) -> Simulator {
    shop_stocking(slots, potions, 1)
}

/// The same with `offers` potions on the shelf. A merchant stocks exactly
/// three (`stock_merchant_potions_of(spec, offers, 3)`), which is the widest
/// a shop screen gets.
fn shop_stocking(slots: usize, potions: &[&str], offers: usize) -> Simulator {
    shop_shelf(slots, potions, offers, 0)
}

/// The same again with `beside` affordable non-potion entries on the shelf,
/// which is what a merchant really stocks around its potions.
fn shop_shelf(slots: usize, potions: &[&str], offers: usize, beside: usize) -> Simulator {
    let map = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::MapNavigation { .. })
    });
    let destination = map
        .legal_actions()
        .iter()
        .find_map(|action| match action {
            Action::ChooseMap { destination } => Some(*destination),
            _ => None,
        })
        .expect("the map offers a step");
    let mut standing = rebuilt(&map, |state| {
        hold(state, slots, potions);
        state.run_player.gold = 500;
        let run = state.run.as_mut().expect("the state is in a run");
        let act = run.current_act;
        run.rooms[act].retain(|room| room.coord != destination);
        run.rooms[act].push(RoomAt {
            coord: destination,
            definition: RoomDefinition::Shop {
                model_id: None,
                offers: (0..offers)
                    .map(|index| ShopOffer {
                        index,
                        item: ShopItem::Potion(id(THROWN)),
                        slot: ShopSlot::Potion,
                        base_price: PRICE,
                        price: PRICE,
                        sold: false,
                    })
                    .chain((0..beside).map(|entry| ShopOffer {
                        index: offers + entry,
                        item: ShopItem::Relic(id("RELIC.BURNING_BLOOD")),
                        slot: ShopSlot::Relic,
                        base_price: PRICE,
                        price: PRICE,
                        sold: false,
                    }))
                    .collect(),
                removal_price: 75,
            },
        });
    });
    standing
        .step_quietly(&Action::ChooseMap { destination })
        .expect("the walk applies");
    assert!(
        matches!(standing.decision(), DecisionContext::Shop { .. }),
        "the step opens the merchant, not {:?}",
        standing.decision()
    );
    standing
}

/// Everything else a merchant stocks beside its three potions: five
/// character cards, two colorless ones and three relics
/// (`stock_merchant_cards`, `stock_merchant_relics`).
const SHELF_BESIDE_THE_POTIONS: usize = 10;

const FULL: &[&str] = &[THROWN, THROWN, DRUNK];

fn trades(plans: &[ActionPlan]) -> Vec<(&Action, &Action)> {
    plans
        .iter()
        .filter_map(|plan| match plan {
            ActionPlan::Trade { free, take } => Some((free, take)),
            ActionPlan::Single(_) | ActionPlan::Collapse { .. } => None,
        })
        .collect()
}

fn collapses(plans: &[ActionPlan]) -> Vec<(&Action, &Action)> {
    plans
        .iter()
        .filter_map(|plan| match plan {
            ActionPlan::Collapse { open, pick } => Some((open, pick)),
            ActionPlan::Single(_) | ActionPlan::Trade { .. } => None,
        })
        .collect()
}

/// The answers a collapse of `open` offers, in offer order.
fn answers<'a>(plans: &'a [ActionPlan], open: &Action) -> Vec<&'a Action> {
    collapses(plans)
        .into_iter()
        .filter(|(opened, _)| *opened == open)
        .map(|(_, pick)| pick)
        .collect()
}

/// The models a `ChooseCards` answer names, in the order it names them.
fn named(action: &Action) -> Vec<String> {
    match action {
        Action::ChooseCards { cards, .. } => cards
            .iter()
            .map(|handle| handle.fingerprint.model_id.to_string())
            .collect(),
        other => panic!("{other:?} is not a card answer"),
    }
}

fn singles(plans: &[ActionPlan]) -> Vec<Action> {
    plans
        .iter()
        .filter_map(|plan| match plan {
            ActionPlan::Single(action) => Some(action.clone()),
            ActionPlan::Trade { .. } | ActionPlan::Collapse { .. } => None,
        })
        .collect()
}

#[test]
fn a_plans_expansion_is_exactly_its_steps_in_order() {
    let free = Action::DiscardPotion {
        slot: 1,
        model_id: id(THROWN),
    };
    let take = Action::Proceed;
    let single = ActionPlan::Single(take.clone());
    assert_eq!(
        single.steps().collect::<Vec<_>>(),
        vec![&take],
        "one action expands to itself"
    );
    let trade = ActionPlan::Trade {
        free: free.clone(),
        take: take.clone(),
    };
    assert_eq!(
        trade.steps().collect::<Vec<_>>(),
        vec![&free, &take],
        "the slot is freed before the offer is taken"
    );
    assert_eq!(trade.lead(), &free);
    assert_eq!(trade.outcome(), &take);
    assert_eq!(trade.surrender(), Some(&free));
    assert_eq!(single.surrender(), None);

    let open = Action::BuyCardRemoval;
    let collapse = ActionPlan::Collapse {
        open: open.clone(),
        pick: take.clone(),
    };
    assert_eq!(
        collapse.steps().collect::<Vec<_>>(),
        vec![&open, &take],
        "the screen is opened before it is answered"
    );
    assert_eq!(collapse.lead(), &open);
    assert_eq!(collapse.outcome(), &take);
    assert_eq!(collapse.opener(), Some(&open));
    assert_eq!(collapse.surrender(), None, "a collapse gives nothing up");
    assert_eq!(single.opener(), None);
    assert_eq!(trade.opener(), None);
}

#[test]
fn stepping_a_plan_leaves_what_stepping_its_steps_leaves() {
    let plans = permitted_plans(&potion_reward(3, FULL));
    let (free, take) = *trades(&plans).first().expect("a full belt offers a trade");

    let mut expanded = potion_reward(3, FULL);
    for action in (ActionPlan::Trade {
        free: free.clone(),
        take: take.clone(),
    })
    .steps()
    {
        expanded.step_quietly(action).expect("the step applies");
    }

    let mut by_hand = potion_reward(3, FULL);
    by_hand.step_quietly(free).expect("the discard applies");
    by_hand.step_quietly(take).expect("the claim applies");

    assert_eq!(
        expanded.state_key().unwrap(),
        by_hand.state_key().unwrap(),
        "a plan is its steps and nothing else"
    );
    // And the pair is sound without simulating it: freeing a slot costs no
    // gold and cannot make the claim illegal, so the potion really lands.
    assert_eq!(
        expanded.state().run_player.potions.iter().flatten().count(),
        3,
        "the freed slot was refilled by the offer"
    );
}

#[test]
fn a_full_belt_potion_reward_drops_the_bare_claim_and_offers_the_trades() {
    let simulator = potion_reward(3, FULL);
    assert!(
        simulator
            .legal_actions()
            .iter()
            .any(|action| matches!(action, Action::ClaimReward { .. })),
        "the engine still enumerates the claim, because the game does"
    );
    let plans = permitted_plans(&simulator);
    assert!(
        !singles(&plans)
            .iter()
            .any(|action| matches!(action, Action::ClaimReward { .. })),
        "and the policy is not offered the bare claim: {plans:?}"
    );

    let freed: Vec<&Action> = trades(&plans).into_iter().map(|(free, _)| free).collect();
    let discarded: Vec<usize> = freed
        .iter()
        .filter_map(|action| match action {
            Action::DiscardPotion { slot, .. } => Some(*slot),
            _ => None,
        })
        .collect();
    assert_eq!(
        discarded,
        vec![0, 1, 2],
        "one discard-trade per potion held"
    );

    let drunk: Vec<usize> = freed
        .iter()
        .filter_map(|action| match action {
            Action::UsePotion { slot, .. } => Some(*slot),
            _ => None,
        })
        .collect();
    let offered: Vec<usize> = simulator
        .legal_actions()
        .iter()
        .filter_map(|action| match action {
            Action::UsePotion { slot, .. } => Some(*slot),
            _ => None,
        })
        .collect();
    assert_eq!(
        drunk, offered,
        "a drink-trade stands exactly where the engine offers the drink"
    );
    assert_eq!(drunk, vec![2], "and the two combat-only potions offer none");

    for (_, take) in trades(&plans) {
        assert!(
            matches!(take, Action::ClaimReward { .. }),
            "every trade acquires the offer"
        );
    }
}

#[test]
fn a_full_belt_shop_potion_is_enumerated_inert_and_traded_for() {
    let simulator = shop(3, FULL);
    let buy = simulator
        .legal_actions()
        .iter()
        .find(|action| matches!(action, Action::BuyShopItem { .. }))
        .cloned()
        .expect("the engine enumerates the affordable potion");

    let mut clicked = shop(3, FULL);
    let before = clicked.state_key().unwrap();
    clicked.step_quietly(&buy).expect("the click is accepted");
    assert_eq!(
        clicked.state_key().unwrap(),
        before,
        "and settles nothing at all"
    );

    let plans = permitted_plans(&simulator);
    assert!(
        !singles(&plans).contains(&buy),
        "so the policy is not offered it: {plans:?}"
    );
    assert!(
        trades(&plans)
            .iter()
            .all(|(_, take)| *take == &buy && !singles(&plans).contains(take)),
        "it stands only as the second half of a trade"
    );
    assert_eq!(trades(&plans).len(), 4, "three discards and one drink");
}

#[test]
fn a_belt_with_room_offers_no_plans_and_the_list_it_always_offered() {
    for simulator in [
        potion_reward(3, &[THROWN, DRUNK]),
        shop(3, &[THROWN, DRUNK]),
    ] {
        let plans = permitted_plans(&simulator);
        assert!(
            trades(&plans).is_empty(),
            "a slot is free, so nothing is traded for"
        );
        let mut buttons: Vec<Action> = plans.iter().map(|plan| plan.lead().clone()).collect();
        buttons.dedup();
        let offered: Vec<Action> = canonical_actions(&permitted_actions(&simulator))
            .into_iter()
            .filter(|action| !free_claim(action))
            .collect();
        assert_eq!(
            buttons, offered,
            "and every button the screen offers is still on it, in order — \
             a merchant's removal as the run of answers behind it, and the \
             free lines as the harness's own steps"
        );
        let kept: Vec<Action> = simulator
            .legal_actions()
            .iter()
            .filter(|action| !matches!(action, Action::DiscardPotion { .. }))
            .cloned()
            .collect();
        assert_eq!(
            permitted_actions(&simulator).as_ref(),
            kept,
            "with nothing withheld from it but the bare discards"
        );
    }
}

/// A belt of `slots`, every slot holding a potion the belt will not pour
/// outside a fight — so the freeing steps are exactly one discard per slot.
fn thrown_only(slots: usize) -> Vec<&'static str> {
    vec![THROWN; slots]
}

#[test]
fn the_trades_offered_follow_the_belt_the_player_actually_has() {
    // Two at ascension four's Tight Belt, five with a potion belt and a phial
    // holster on top of the opening three. Nothing here may assume three.
    for slots in [2_usize, 3, 5, 10] {
        let held = thrown_only(slots);
        for simulator in [potion_reward(slots, &held), shop(slots, &held)] {
            assert_eq!(
                simulator.state().run_player.potions.len(),
                slots,
                "the belt is the size the test asked for"
            );
            let plans = permitted_plans(&simulator);
            let discarded: Vec<usize> = trades(&plans)
                .into_iter()
                .filter_map(|(free, _)| match free {
                    Action::DiscardPotion { slot, .. } => Some(*slot),
                    _ => None,
                })
                .collect();
            assert_eq!(
                discarded,
                (0..slots).collect::<Vec<_>>(),
                "one discard-trade per slot on a belt of {slots}"
            );
            assert_eq!(
                trades(&plans).len(),
                slots,
                "and nothing drinkable, so no drink-trade beside them"
            );
        }
    }
}

#[test]
fn a_grown_belt_with_a_drinkable_potion_trades_both_ways_out_of_every_slot() {
    let mut held = thrown_only(6);
    held[4] = DRUNK;
    let simulator = potion_reward(6, &held);
    let plans = permitted_plans(&simulator);
    assert_eq!(
        trades(&plans).len(),
        7,
        "six discards and the one drink the belt will pour"
    );
    let drunk: Vec<usize> = trades(&plans)
        .into_iter()
        .filter_map(|(free, _)| match free {
            Action::UsePotion { slot, .. } => Some(*slot),
            _ => None,
        })
        .collect();
    assert_eq!(drunk, vec![4], "out of the slot the drinkable one sits in");
}

/// The widest a full belt can make a shop screen, enumerated rather than
/// estimated.
///
/// Ten slots is the belt's ceiling — three to open, two from
/// `RELIC.POTION_BELT`, one from `RELIC.PHIAL_HOLSTER`, four from
/// `RELIC.ALCHEMICAL_COFFER` — and a merchant stocks three potions beside
/// five character cards, two colorless ones and three relics. Trades are the
/// product of the offers and the ways of emptying a slot, so this is where
/// enumeration is widest. The merchant's removal adds one plan per class of
/// card the deck holds — three on a fresh Ironclad deck — in place of the
/// button that opened it.
///
/// **The all-drinkable case is past the axis the checkpoint prices**
/// (`MAX_ACTIONS`), where a decision is answered with uniform priors and
/// flagged `degraded`. The numbers are pinned here so that any bound put on
/// enumeration is measured against them rather than guessed at.
#[test]
fn the_widest_shop_screen_a_full_belt_can_make() {
    let thrown = shop_shelf(10, &[THROWN; 10], 3, SHELF_BESIDE_THE_POTIONS);
    let drinkable = shop_shelf(10, &[DRUNK; 10], 3, SHELF_BESIDE_THE_POTIONS);

    let plans = permitted_plans(&thrown);
    assert_eq!(
        trades(&plans).len(),
        30,
        "three offers against ten discards"
    );
    assert_eq!(
        plans.len(),
        46,
        "inside the priced axis, the ten bare discards withheld"
    );

    let plans = permitted_plans(&drinkable);
    assert_eq!(
        trades(&plans).len(),
        60,
        "and against ten discards and ten drinks"
    );
    assert_eq!(plans.len(), 86, "which is past it");
    assert!(
        plans.len() > alphaspire::net::MAX_ACTIONS,
        "the number this test exists to keep visible"
    );
}

// --- collapsed screens: an opener and its answer are one decision ---------

/// The three-card offer these tests hand a card reward. Three distinct
/// models, so the canonical list keeps all three.
const OFFER: &[&str] = &[
    "CARD.CLEAVE",
    "CARD.SHRUG_IT_OFF",
    "CARD.POMMEL_STRIKE_IRONCLAD",
];

#[test]
fn a_card_reward_is_claimed_and_answered_as_one_decision() {
    let simulator = card_reward(OFFER, false, &[]);
    let claim = simulator
        .legal_actions()
        .iter()
        .find(|action| matches!(action, Action::ClaimReward { .. }))
        .cloned()
        .expect("the engine enumerates the claim, because the game does");

    let plans = permitted_plans(&simulator);
    assert!(
        !singles(&plans).contains(&claim),
        "the policy is not offered the bare claim: {plans:?}"
    );
    let picks = answers(&plans, &claim);
    assert_eq!(
        picks.iter().map(|pick| named(pick)).collect::<Vec<_>>(),
        OFFER
            .iter()
            .map(|model| vec![(*model).to_owned()])
            .collect::<Vec<_>>(),
        "one plan per card the reward is offering, in offer order"
    );
    assert!(
        picks.iter().all(|pick| !named(pick).is_empty()),
        "and none of them is the skip: declining is Proceed on this screen"
    );
    assert!(
        singles(&plans).contains(&Action::Proceed),
        "which the screen still offers: {plans:?}"
    );
}

#[test]
fn the_reroll_and_the_sacrifice_stand_where_the_relics_put_them() {
    let option_ids = |simulator: &Simulator| -> Vec<String> {
        permitted_plans(simulator)
            .iter()
            .filter_map(|plan| match plan.outcome() {
                Action::ChooseAlternative { option_id, .. } => Some(option_id.clone()),
                _ => None,
            })
            .collect()
    };
    assert_eq!(
        option_ids(&card_reward(OFFER, true, &["RELIC.PAELS_WING"])),
        vec!["reroll".to_owned(), "sacrifice".to_owned()],
        "both buttons lift onto the rewards screen as their own options"
    );
    assert_eq!(
        option_ids(&card_reward(OFFER, true, &[])),
        vec!["reroll".to_owned()],
        "the reroll rides the offer's own can_reroll flag"
    );
    assert_eq!(
        option_ids(&card_reward(OFFER, false, &["RELIC.PAELS_WING"])),
        vec!["sacrifice".to_owned()],
        "and the sacrifice the relic that carries the alternative"
    );
    assert!(
        option_ids(&card_reward(OFFER, false, &[])).is_empty(),
        "neither stands where nothing puts it there"
    );
}

#[test]
fn a_removal_reward_is_claimed_and_answered_as_one_decision() {
    let simulator = reward_screen(vec![reward_of("card_removal", |_| {})], &[]);
    let claim = simulator
        .legal_actions()
        .iter()
        .find(|action| matches!(action, Action::ClaimReward { .. }))
        .cloned()
        .expect("the engine enumerates the claim");

    let plans = permitted_plans(&simulator);
    assert!(
        !singles(&plans).contains(&claim),
        "the bare claim is gone: {plans:?}"
    );
    let picks = answers(&plans, &claim);
    assert_eq!(
        picks.iter().map(|pick| named(pick)).collect::<Vec<_>>(),
        vec![
            vec!["CARD.STRIKE_IRONCLAD".to_owned()],
            vec!["CARD.DEFEND_IRONCLAD".to_owned()],
            vec!["CARD.BASH".to_owned()],
        ],
        "one plan per class of card the starter deck holds"
    );
    assert!(
        picks.iter().all(|pick| !named(pick).is_empty()),
        "and no empty answer among them"
    );
}

#[test]
fn a_removal_reward_over_an_empty_deck_leaves_nothing_where_it_stood() {
    let simulator = rebuilt(
        &reward_screen(vec![reward_of("card_removal", |_| {})], &[]),
        |state| state.run_player.deck.clear(),
    );
    let plans = permitted_plans(&simulator);
    assert!(
        collapses(&plans).is_empty(),
        "a screen with nothing to remove accepts no answer: {plans:?}"
    );
    assert!(
        !singles(&plans)
            .iter()
            .any(|action| matches!(action, Action::ClaimReward { .. })),
        "so the button that only opens it is gone too: {plans:?}"
    );
    assert!(
        !plans.is_empty(),
        "and the screen is still answerable: {plans:?}"
    );
}

#[test]
fn the_merchants_removal_is_bought_and_answered_as_one_decision() {
    let simulator = shop(3, &[THROWN]);
    let plans = permitted_plans(&simulator);
    assert!(
        !singles(&plans).contains(&Action::BuyCardRemoval),
        "the bare purchase is gone: {plans:?}"
    );
    let picks = answers(&plans, &Action::BuyCardRemoval);
    assert_eq!(
        picks.iter().map(|pick| named(pick)).collect::<Vec<_>>(),
        vec![
            vec!["CARD.STRIKE_IRONCLAD".to_owned()],
            vec!["CARD.DEFEND_IRONCLAD".to_owned()],
            vec!["CARD.BASH".to_owned()],
        ],
        "one plan per class of card the deck holds"
    );
}

#[test]
fn the_smith_stands_as_one_arm_and_the_card_as_the_next_decision() {
    let simulator = rest_site(&[RestSiteOption::Heal, RestSiteOption::Smith]);
    let open = Action::RestOption {
        index: 1,
        option: RestSiteOption::Smith,
    };
    let plans = permitted_plans(&simulator);
    assert!(
        singles(&plans).contains(&open),
        "collapsed one-plan-per-card, the smith's mass fragments against the \
         heal's single plan, so it stands as one arm: {plans:?}"
    );
    assert!(
        answers(&plans, &open).is_empty(),
        "and nothing is collapsed behind it"
    );
    assert!(
        singles(&plans).contains(&Action::RestOption {
            index: 0,
            option: RestSiteOption::Heal,
        }),
        "the option beside it is untouched: {plans:?}"
    );

    let mut screen = rest_site(&[RestSiteOption::Heal, RestSiteOption::Smith]);
    screen.step_quietly(&open).expect("the smith opens");
    let standing = permitted_plans(&screen);
    assert_eq!(
        standing
            .iter()
            .map(|plan| named(plan.outcome()))
            .collect::<Vec<_>>(),
        vec![
            vec!["CARD.STRIKE_IRONCLAD".to_owned()],
            vec!["CARD.DEFEND_IRONCLAD".to_owned()],
            vec!["CARD.BASH".to_owned()],
        ],
        "the card is the following decision's own choice, one plan per class \
         of card the smith would upgrade, and backing out is not among them"
    );

    let upgraded = rebuilt(&simulator, |state| {
        for card in &mut state.run_player.deck {
            card.upgrade_level = 1;
        }
    });
    let plans = permitted_plans(&upgraded);
    assert!(
        !singles(&plans).contains(&open),
        "a smith over a fully upgraded deck opens nothing and settles \
         nothing, so the arm is dropped rather than offered as a repeatable \
         step: {plans:?}"
    );
}

#[test]
fn the_cook_keeps_its_two_steps_and_offers_no_empty_answer() {
    let simulator = rest_site(&[RestSiteOption::Heal, RestSiteOption::Cook]);
    let open = Action::RestOption {
        index: 1,
        option: RestSiteOption::Cook,
    };
    let plans = permitted_plans(&simulator);
    assert!(
        singles(&plans).contains(&open),
        "a two-card screen collapses into one plan per pair, which is far \
         past the axis, so the cook stays two steps: {plans:?}"
    );
    assert!(
        answers(&plans, &open).is_empty(),
        "and nothing is collapsed behind it"
    );

    let mut screen = rest_site(&[RestSiteOption::Heal, RestSiteOption::Cook]);
    screen.step_quietly(&open).expect("the cook opens");
    let standing = permitted_plans(&screen);
    assert!(
        standing.iter().all(|plan| named(plan.outcome()).len() == 2),
        "every answer the screen offers takes the two cards it asks for, \
         and backing out is not among them: {standing:?}"
    );
}

#[test]
fn stepping_a_collapse_leaves_what_stepping_its_steps_leaves() {
    let plans = permitted_plans(&card_reward(OFFER, false, &[]));
    let (open, pick) = *collapses(&plans)
        .first()
        .expect("a card reward collapses into its answers");

    let mut expanded = card_reward(OFFER, false, &[]);
    for action in (ActionPlan::Collapse {
        open: open.clone(),
        pick: pick.clone(),
    })
    .steps()
    {
        expanded.step_quietly(action).expect("the step applies");
    }

    let mut by_hand = card_reward(OFFER, false, &[]);
    by_hand.step_quietly(open).expect("the claim applies");
    by_hand.step_quietly(pick).expect("the pick applies");

    assert_eq!(
        expanded.state_key().unwrap(),
        by_hand.state_key().unwrap(),
        "a plan is its steps and nothing else"
    );
    // The answer read off a throwaway copy is the answer the real screen
    // takes: the copy stepped the same action from the same state.
    assert!(
        expanded
            .state()
            .run_player
            .deck
            .iter()
            .any(|card| card.model_id.to_string() == OFFER[0]),
        "and the card the plan named really joined the deck"
    );
}

/// The reward screen of a won fight standing inside the fake merchant's own
/// room, with the belt full and one of its potions a foul one.
///
/// The room is what makes the throw legal: the engine offers a foul potion
/// wherever there is a merchant to throw it at, and the fake merchant is one.
fn potion_reward_at_the_fake_merchant() -> Simulator {
    let screen = potion_reward(2, &[FOUL, DRUNK]);
    rebuilt(&screen, |state| {
        let run = state.run.as_mut().expect("the run stands");
        run.active_room = Some(ActiveRoom::Event {
            model_id: id("EVENT.FAKE_MERCHANT"),
            options: Vec::new(),
            rng_counter: 0,
            shelf: Vec::new(),
            standing_fight: None,
            rolled_card: None,
            skipped_cards: Vec::new(),
            sphere: None,
        });
    })
}

/// The reproducer: the throw is on the belt at the screen, and freeing a slot
/// with it walks off the screen the claim was chosen at.
#[test]
fn the_throw_that_starts_a_fight_is_not_a_freeing_step() {
    let simulator = potion_reward_at_the_fake_merchant();
    let throw = simulator
        .legal_actions()
        .iter()
        .find(
            |action| matches!(action, Action::UsePotion { model_id, .. } if *model_id == id(FOUL)),
        )
        .cloned()
        .expect("the fake merchant is there to throw it at");

    let mut probe = simulator.clone();
    probe.step_quietly(&throw).expect("the throw applies");
    assert!(
        matches!(probe.decision(), DecisionContext::CombatPriority { .. }),
        "the throw started the merchant's fight: {:?}",
        probe.decision()
    );

    for plan in permitted_plans(&simulator) {
        let ActionPlan::Trade { free, take } = &plan else {
            continue;
        };
        assert_ne!(
            free, &throw,
            "no trade frees its slot with a step that walks off the screen: {take:?}"
        );
    }
}

/// And every trade offered anywhere is one the engine will take both steps
/// of: the plan is what the harness steps in order, so a `take` the screen no
/// longer offers is an action the engine refuses.
#[test]
fn every_trade_is_steppable_in_order() {
    for simulator in [
        potion_reward_at_the_fake_merchant(),
        potion_reward(2, &[THROWN, DRUNK]),
        potion_reward(1, &[DRUNK]),
    ] {
        for plan in permitted_plans(&simulator) {
            let ActionPlan::Trade { free, take } = &plan else {
                continue;
            };
            let mut walk = simulator.clone();
            walk.step_quietly(free).expect("the freeing step applies");
            walk.step_quietly(take)
                .expect("and the take the plan named is still offered after it");
        }
    }
}

/// A reward line of `kind` at screen position `index`, with `edit` on its
/// fingerprint.
fn line(index: usize, kind: &str, edit: impl FnOnce(&mut RewardFingerprint)) -> RewardItem {
    let mut item = reward_of(kind, edit);
    item.index = index;
    item
}

/// An elite's screen with every kind of line on it: gold, a potion the belt
/// has room for, a two-card offer and a relic, in that order.
fn full_elite_screen() -> Simulator {
    reward_screen(
        vec![
            line(0, "gold", |fingerprint| fingerprint.gold_amount = 30),
            line(1, "potion", |fingerprint| {
                fingerprint.model_id = Some(id(THROWN));
            }),
            line(2, "card", |fingerprint| {
                fingerprint.option_count = 2;
                fingerprint.offered_cards = vec![
                    CardFingerprint::base(id("CARD.CLEAVE")),
                    CardFingerprint::base(id("CARD.ANGER")),
                ];
            }),
            line(3, "relic", |fingerprint| {
                fingerprint.model_id = Some(id("RELIC.ANCHOR"));
            }),
        ],
        &[],
    )
}

fn claim_of<'a>(simulator: &'a Simulator, kind: &str) -> &'a Action {
    simulator
        .legal_actions()
        .iter()
        .find(|action| {
            matches!(action, Action::ClaimReward { fingerprint, .. } if fingerprint.reward_type == kind)
        })
        .unwrap_or_else(|| panic!("the screen offers a {kind} claim"))
}

#[test]
fn the_free_lines_are_the_harness_steps_and_the_rest_are_the_plans() {
    let screen = full_elite_screen();
    assert!(
        !alphaspire::policy::belt_is_full(&screen),
        "the fresh belt has room"
    );
    // The free lines are exactly gold, the potion and a handed-over card;
    // the card offer and the relic are not free.
    assert!(free_claim(claim_of(&screen, "gold")));
    assert!(free_claim(claim_of(&screen, "potion")));
    assert!(!free_claim(claim_of(&screen, "card")));
    assert!(!free_claim(claim_of(&screen, "relic")));
    assert!(!free_claim(&Action::Proceed));

    // What the policy is asked to score carries none of the free lines:
    // the relic on a plan of its own, one collapse per offered card, and
    // the exit.
    let plans = permitted_plans(&screen);
    assert!(
        plans
            .iter()
            .flat_map(ActionPlan::steps)
            .all(|step| !free_claim(step)),
        "no plan claims a free line: {plans:?}"
    );
    let singles = singles(&plans);
    assert!(
        singles.iter().any(|action| matches!(action, Action::ClaimReward { fingerprint, .. } if fingerprint.reward_type == "relic")),
        "the relic is a plan: {singles:?}"
    );
    assert!(
        singles.contains(&Action::Proceed),
        "walking away is a plan: {singles:?}"
    );
    assert_eq!(
        collapses(&plans).len(),
        2,
        "one collapse per offered card: {plans:?}"
    );
    assert_eq!(plans.len(), 4, "relic, two cards, proceed: {plans:?}");

    // The harness's steps come off the screen in order, one at a time, and
    // stop when nothing free is left; the plans are unchanged throughout.
    let mut screen = screen;
    let gold = forced_step(&screen).expect("gold stands first");
    assert_eq!(&gold, claim_of(&screen, "gold"));
    assert!(is_forced_step(&screen, &gold));
    screen.step_quietly(&gold).unwrap();
    let potion = forced_step(&screen).expect("the potion stands next");
    assert_eq!(&potion, claim_of(&screen, "potion"));
    assert!(
        !is_forced_step(&screen, &gold),
        "a claimed line is not claimed again"
    );
    screen.step_quietly(&potion).unwrap();
    assert_eq!(forced_step(&screen), None, "nothing free is left");
    let after = permitted_plans(&screen);
    assert_eq!(after.len(), 4, "the decision is what it was: {after:?}");
    assert!(
        after
            .iter()
            .flat_map(ActionPlan::steps)
            .all(|step| !free_claim(step))
    );
}

#[test]
fn a_potion_over_a_full_belt_is_no_free_line() {
    let screen = potion_reward(3, &[THROWN, THROWN, THROWN]);
    let claim = claim_of(&screen, "potion");
    assert!(
        free_claim(claim),
        "the claim is free by kind — it is the belt that stops it"
    );
    assert!(
        !is_forced_step(&screen, claim),
        "inert over a full belt, so not the harness's step"
    );
    assert_eq!(forced_step(&screen), None);
    assert!(
        !trades(&permitted_plans(&screen)).is_empty(),
        "the trades stand where the claim would"
    );
}

#[test]
fn a_card_handed_back_is_a_plan_and_leaving_it_is_the_free_removal() {
    // What a thief stole comes back as a line of its own; a deck that has
    // outgrown the card walks past it, so the claim is the policy's.
    let screen = reward_screen(
        vec![line(0, "special_card", |fingerprint| {
            fingerprint.special_card = Some(CardFingerprint::base(id("CARD.STRIKE_IRONCLAD")));
        })],
        &[],
    );
    let claim = claim_of(&screen, "special_card");
    assert!(!free_claim(claim));
    assert!(!is_forced_step(&screen, claim));
    assert_eq!(forced_step(&screen), None);
    let plans = permitted_plans(&screen);
    assert!(
        plans.iter().any(|plan| plan.lead() == claim),
        "the claim is a plan: {plans:?}"
    );
    assert!(
        plans.iter().any(|plan| plan.lead() == &Action::Proceed),
        "and walking past it is the other: {plans:?}"
    );
}

#[test]
fn off_a_reward_screen_the_harness_has_no_step() {
    let start = fresh_run();
    assert!(!matches!(start.decision(), DecisionContext::Rewards { .. }));
    assert_eq!(forced_step(&start), None);
    let shop = shop(3, &[]);
    assert_eq!(forced_step(&shop), None);
}

/// A random walk that checks, at every decision it is asked for, that no
/// free line stands on the screen — and counts the claims the harness made
/// on its behalf.
struct NeverShownAFreeLine {
    inner: alphaspire::policy::UniformRandom,
    reward_screens: usize,
    claims: usize,
}

impl alphaspire::policy::RolloutPolicy for NeverShownAFreeLine {
    fn choose(&mut self, simulator: &Simulator, rng: &mut sts2_rng::MegaRandom) -> Action {
        assert_eq!(
            forced_step(simulator),
            None,
            "asked with a free line standing at {:?}",
            simulator.decision()
        );
        if matches!(simulator.decision(), DecisionContext::Rewards { .. }) {
            self.reward_screens += 1;
        }
        self.inner.choose(simulator, rng)
    }

    fn forced(&mut self, simulator: &Simulator, action: &Action) {
        assert!(
            is_forced_step(simulator, action),
            "told of a claim that is not the harness's: {action:?}"
        );
        assert!(free_claim(action));
        self.claims += 1;
    }
}

#[test]
fn the_harness_claims_every_free_line_before_the_policy_is_asked() {
    let mut policy = NeverShownAFreeLine {
        inner: alphaspire::policy::UniformRandom,
        reward_screens: 0,
        claims: 0,
    };
    let mut rng = sts2_rng::MegaRandom::new(7);
    let mut objective = alphaspire::objective::CombatStrength::default();
    let report = alphaspire::selfplay::play_run(
        SEED,
        &"CHARACTER.IRONCLAD".parse().unwrap(),
        0,
        &mut policy,
        &mut rng,
        &mut objective,
        600,
    )
    .expect("the walk runs");
    assert!(
        policy.reward_screens > 0,
        "the walk stood at a reward screen"
    );
    assert!(policy.claims > 0, "and the harness claimed something there");
    // The claims are engine steps in the run's own record, so a script
    // emitted from it replays them like any other.
    let claimed = report
        .actions
        .iter()
        .filter(|action| free_claim(action))
        .count();
    assert_eq!(
        claimed, policy.claims,
        "every claim is in the run's actions"
    );
}

#[test]
fn the_exit_names_what_it_leaves_on_a_reward_screen() {
    use alphaspire::encoding::{ACTION_NAMING_TOKENS, PolicyEncoder, standard_vocabulary};
    let registry = sts2_content::standard_registry();
    let encoder = PolicyEncoder::new(standard_vocabulary(&registry), &registry);
    let vocabulary = standard_vocabulary(&registry);
    let index_of = |model: &str| -> u32 {
        u32::try_from(
            vocabulary
                .iter()
                .position(|entry| *entry == id(model))
                .unwrap()
                + 1,
        )
        .unwrap()
    };
    let mut screen = full_elite_screen();
    // With everything standing: gold, the potion, the offer and the relic,
    // in screen order, four lines, a relic among them.
    let exit = encoder.encode_action(&screen.agent_observation(), &Action::Proceed);
    let named: Vec<u32> = exit.tokens[..ACTION_NAMING_TOKENS].to_vec();
    assert_eq!(named[1], index_of(THROWN), "the potion by its own model");
    assert_eq!(
        named[3],
        index_of("RELIC.ANCHOR"),
        "the relic by its own model"
    );
    assert!(
        named[0] != 0 && named[2] != 0,
        "gold and the offer by their kinds: {named:?}"
    );
    let plan_base = exit.features.len() - alphaspire::encoding::EVENT_EFFECT_NAMES.len() - 10 - 6;
    assert!(
        (exit.features[plan_base + 4] - 1.0).abs() < 1e-6,
        "four lines stand"
    );
    assert!(
        (exit.features[plan_base + 5] - 1.0).abs() < 1e-6,
        "a relic among them"
    );
    // The harness's claims and the card taken leave the relic alone.
    while let Some(step) = forced_step(&screen) {
        screen.step_quietly(&step).unwrap();
    }
    let plans = permitted_plans(&screen);
    let (open, pick) = collapses(&plans)[0];
    let (open, pick) = (open.clone(), pick.clone());
    screen.step_quietly(&open).unwrap();
    screen.step_quietly(&pick).unwrap();
    let exit = encoder.encode_action(&screen.agent_observation(), &Action::Proceed);
    assert_eq!(
        exit.tokens[0],
        index_of("RELIC.ANCHOR"),
        "the relic is what is left"
    );
    assert_eq!(
        &exit.tokens[1..ACTION_NAMING_TOKENS],
        &[0, 0, 0],
        "and nothing else"
    );
    assert!((exit.features[plan_base + 4] - 0.25).abs() < 1e-6);
    assert!((exit.features[plan_base + 5] - 1.0).abs() < 1e-6);
    // Off a reward screen the exit is bare.
    let bare = encoder.encode_action(&fresh_run().agent_observation(), &Action::Proceed);
    assert!(bare.tokens.iter().all(|token| *token == 0));
    assert!(bare.features[plan_base + 4].abs() < 1e-6);
    assert!(bare.features[plan_base + 5].abs() < 1e-6);
}
