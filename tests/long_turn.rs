//! A turn that could go on forever ends, and its decisions stay cheap.
//!
//! A card made free for the combat that returns to the hand when played can
//! be played without end: every play is legal, every play moves a counter so
//! no state repeats, and under any value that likes what the card gives, no
//! play is worse than ending the turn. The fight's tree outlives each
//! decision, so every simulation of every later decision re-walks the same
//! line and adds an edge to it. Left unbounded the turn never ends and a
//! decision comes to cost minutes.

use alphaspire::objective::Objective;
use alphaspire::policy::RolloutPolicy;
use alphaspire::search::{BeliefSearch, SearchConfig};
use sts2_engine::{Action, CombatSetup, Simulator};
use sts2_rng::MegaRandom;

/// More block is always better, and nothing else is worth anything: the
/// search has every reason to play a free block card again and none to stop.
#[derive(Clone, Default)]
struct Hoard;

impl Objective for Hoard {
    fn reward(&mut self, simulator: &Simulator) -> f64 {
        self.peek(simulator)
    }

    fn peek(&self, simulator: &Simulator) -> f64 {
        let block = simulator.state().combat.as_ref().map_or(0, |combat| {
            combat
                .creatures
                .iter()
                .find(|creature| creature.id == combat.player.creature_id)
                .map_or(0, |creature| creature.block)
        });
        // Inside [0, 1) and rising with every point, so the gap between playing
        // on and ending the turn stays wide against the search's exploration.
        f64::from(block) / (f64::from(block) + 20.0)
    }
}

/// A Regent fight whose opening hand is the whole deck: Particle Wall among
/// Defends, and the potion that makes one card free for the combat.
fn fight() -> Simulator {
    let setup: CombatSetup = serde_json::from_value(serde_json::json!({
        "setup_version": 1,
        "compatibility_id": sts2_core::PINNED_COMPATIBILITY_ID,
        "character": "CHARACTER.REGENT",
        "ascension": 0,
        "act": 0,
        "floor": 3,
        "encounter": "ENCOUNTER.NIBBITS_WEAK",
        "room": "monster",
        "hp": 75,
        "max_hp": 75,
        "gold": 0,
        "max_energy": 3,
        "orb_slots": 0,
        "deck": [
            {"id": "CARD.PARTICLE_WALL"},
            {"id": "CARD.DEFEND_REGENT"},
            {"id": "CARD.DEFEND_REGENT"},
            {"id": "CARD.DEFEND_REGENT"},
            {"id": "CARD.DEFEND_REGENT"}
        ],
        "relics": [],
        "potions": ["POTION.TOUCH_OF_INSANITY"],
        "provenance": {"floor": 3, "origin": "live_run", "seed": "EC02TZ4L10"}
    }))
    .expect("the setup parses");
    sts2_content::combat_from_setup(&setup, "EC02TZ4L10").expect("the setup stands up")
}

fn is_particle_wall(card: &sts2_engine::CardHandle) -> bool {
    card.fingerprint.model_id.to_string() == "CARD.PARTICLE_WALL"
}

#[test]
fn a_free_card_that_returns_to_hand_does_not_hold_the_turn_forever() {
    let mut simulator = fight();
    let potion = simulator
        .legal_actions()
        .iter()
        .find(|action| matches!(action, Action::UsePotion { .. }))
        .expect("the potion can be drunk")
        .clone();
    simulator.step_quietly(&potion).unwrap();
    let pick = simulator
        .legal_actions()
        .iter()
        .find(|action| {
            matches!(action, Action::ChooseCards { cards, .. }
                if cards.len() == 1 && is_particle_wall(&cards[0]))
        })
        .expect("Particle Wall can be chosen")
        .clone();
    simulator.step_quietly(&pick).unwrap();

    let round = simulator.state().combat.as_ref().unwrap().round;
    let config = SearchConfig {
        iterations: 16,
        // A leaf is scored where it stands: the block in hand, not what a
        // random continuation leaves of it.
        rollout_depth: 0,
        temperature: 0.0,
    };
    let mut search = BeliefSearch::new(config, Hoard);
    let mut rng = MegaRandom::new(7);
    let started = std::time::Instant::now();
    let mut plays = 0_usize;
    let mut decisions = 0_usize;
    while simulator
        .state()
        .combat
        .as_ref()
        .is_some_and(|combat| combat.round == round)
        && simulator.state().terminal.is_none()
    {
        let action = search.choose(&simulator, &mut rng);
        if matches!(&action, Action::PlayCard { card, .. } if is_particle_wall(card)) {
            plays += 1;
        }
        simulator.step_quietly(&action).unwrap();
        decisions += 1;
        assert!(decisions < 1_000, "the turn never ended");
    }
    assert!(
        plays > 100,
        "the premise: under this objective the free card is played over and over ({plays} plays)"
    );
    assert_eq!(
        search.degradations().long_turns,
        1,
        "the turn was ended at the ceiling, and counted"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(60),
        "a decision late in the turn costs what an early one does"
    );
}
