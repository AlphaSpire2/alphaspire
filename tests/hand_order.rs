//! The node key folds copy identity and hand order, so two sampled worlds
//! holding the same cards as different copies in a different order share a
//! node. The search must then step each world's own action, never the other
//! world's copy, and never dead-end on a screen it knows.

use alphaspire::env::{Determinizer, NodeKey};
use alphaspire::objective::CombatStrength;
use alphaspire::policy::UniformRandom;
use alphaspire::search::{Mcts, SearchConfig, Uct};
use sts2_engine::{Action, CardFingerprint, EngineError, PileName, ScenarioBuilder, Simulator};
use sts2_rng::MegaRandom;

fn id(value: &str) -> sts2_core::ModelId {
    value.parse().unwrap()
}

/// A beetle fight holding a Strike, a Defend and a Bash, laid out as given.
fn fight() -> Simulator {
    ScenarioBuilder::new("HANDORDER1", id("ENCOUNTER.SHRINKER_BEETLE_WEAK"))
        .player_hp(80, 80)
        .energy(3, 3)
        .turn(1, 1)
        .shuffle_counter(0)
        .enemy(1, id("MONSTER.SHRINKER_BEETLE"), 39, 39, "SHRINKER_MOVE")
        .card(
            PileName::Hand,
            10,
            CardFingerprint::base(id("CARD.STRIKE_IRONCLAD")),
            1,
        )
        .card(
            PileName::Hand,
            11,
            CardFingerprint::base(id("CARD.DEFEND_IRONCLAD")),
            1,
        )
        .card(
            PileName::Hand,
            12,
            CardFingerprint::base(id("CARD.STRIKE_IRONCLAD")),
            1,
        )
        .card(
            PileName::Hand,
            13,
            CardFingerprint::base(id("CARD.BASH")),
            2,
        )
        .card(
            PileName::Draw,
            20,
            CardFingerprint::base(id("CARD.DEFEND_IRONCLAD")),
            1,
        )
        .card(
            PileName::Draw,
            21,
            CardFingerprint::base(id("CARD.STRIKE_IRONCLAD")),
            1,
        )
        .build(sts2_content::standard_registry())
        .unwrap()
}

/// The same fight with the hand reversed and the two Strikes' ids swapped:
/// what a player sees is the same hand.
fn reversed(original: &Simulator) -> Simulator {
    let mut state = original.state().clone();
    let hand = &mut state.combat.as_mut().unwrap().player.piles.hand;
    hand.reverse();
    let strikes: Vec<usize> = hand
        .iter()
        .enumerate()
        .filter(|(_, card)| card.fingerprint.model_id == id("CARD.STRIKE_IRONCLAD"))
        .map(|(index, _)| index)
        .collect();
    let (first, second) = (strikes[0], strikes[1]);
    let swap = hand[first].id;
    hand[first].id = hand[second].id;
    hand[second].id = swap;
    Simulator::from_scenario(state, sts2_content::standard_registry()).unwrap()
}

/// Two worlds, one per parity of the rollout index.
struct TwoWorlds {
    even: Simulator,
    odd: Simulator,
}

impl Determinizer for TwoWorlds {
    fn sample(&mut self, rollout: u64) -> Simulator {
        if rollout.is_multiple_of(2) {
            self.even.clone()
        } else {
            self.odd.clone()
        }
    }

    fn node_key(&self, simulator: &Simulator) -> Result<NodeKey, EngineError> {
        Ok(NodeKey::Observation(simulator.observation_key()?))
    }
}

#[test]
fn worlds_that_hold_the_same_hand_in_another_order_share_the_tree() {
    let even = fight();
    let odd = reversed(&even);
    assert_ne!(even.state_key().unwrap(), odd.state_key().unwrap());
    assert_eq!(
        even.observation_key().unwrap(),
        odd.observation_key().unwrap(),
        "the two worlds are one sight"
    );
    assert_ne!(
        even.legal_actions(),
        odd.legal_actions(),
        "and yet name their plays through different copies"
    );

    let mut determinizer = TwoWorlds {
        even: even.clone(),
        odd,
    };
    let mut mcts = Mcts::new(
        SearchConfig {
            iterations: 96,
            ..SearchConfig::default()
        },
        Uct::default(),
    );
    let mut rollout = UniformRandom;
    let mut rng = MegaRandom::new(7);
    let (chosen, policy) = mcts.decide_pinned(
        &mut determinizer,
        &CombatStrength::default(),
        &mut rollout,
        None,
        &[],
        &mut rng,
    );
    let degraded = mcts.degradations();
    assert_eq!(
        degraded.dead_ends, 0,
        "a world naming another copy is still offered every class: {degraded:?}"
    );
    assert!(
        even.legal_actions().contains(&chosen),
        "the answer is the root world's own action: {chosen:?}"
    );
    // Every class the root offers was walked, in both worlds: the odd world
    // never dead-ended, so the root's edges took all 96 descents between them.
    let visits: u64 = policy
        .candidates
        .iter()
        .map(|candidate| candidate.visits)
        .sum();
    assert_eq!(visits, 96, "every descent walked a root edge");
    assert!(
        policy
            .candidates
            .iter()
            .all(|candidate| candidate.visits > 0),
        "every class was opened: {:?}",
        policy.candidates
    );
    // What the tree stepped was each world's own copy: a Strike played in the
    // odd world is one of the odd world's Strikes, which the even world does
    // not offer under that id.
    let strikes = |simulator: &Simulator| -> Vec<Action> {
        simulator
            .legal_actions()
            .iter()
            .filter(|action| matches!(action, Action::PlayCard { card, .. } if card.fingerprint.model_id == id("CARD.STRIKE_IRONCLAD")))
            .cloned()
            .collect()
    };
    assert!(
        strikes(&even)
            .iter()
            .any(|strike| !determinizer.odd.legal_actions().contains(strike)),
        "the worlds name at least one Strike differently, so the class match was exercised"
    );
}
