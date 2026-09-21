//! Act search preserves the player-visible information boundary, scores
//! progress at the act horizon, and pins standing screens during sampling.
//! Combined with combat search, it plays legal, reproducible runs.

use alphaspire::env::{ActBelief, Belief, Determinizer, NodeKey, act_over};
use alphaspire::objective::{ActBoundary, CombatStrength, Objective};
use alphaspire::policy::{RolloutPolicy, UniformRandom, engine_refuses};
use alphaspire::search::{ActSearch, BeliefSearch, Budget, Gumbel, Mcts, SearchConfig, Uct};
use sts2_core::UnlockPresetManifest;
use sts2_engine::{Action, DecisionContext, RunPhase, RunResult, Simulator};
use sts2_rng::MegaRandom;

const SEED: &str = "NLD6VZXP94";

fn fresh_run() -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at(SEED, &character, &preset, 0).unwrap()
}

/// The first legal action, preferring a non-empty card pick so a skippable
/// reward cannot re-offer itself.
fn next_action(simulator: &Simulator) -> Action {
    simulator
        .legal_actions()
        .iter()
        .find(|action| matches!(action, Action::ChooseCards { cards, .. } if !cards.is_empty()))
        .or_else(|| simulator.legal_actions().first())
        .cloned()
        .expect("a live run offers an action")
}

/// The seeded run walked until `stop` answers, under the simple policy.
fn walk_until(mut simulator: Simulator, stop: impl Fn(&Simulator) -> bool) -> Simulator {
    for _ in 0..200 {
        if stop(&simulator) {
            return simulator;
        }
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }
    panic!("the seed reaches the stop inside two hundred decisions");
}

/// The seeded run walked to a map decision in the middle of act one: the
/// same anchor the engine's own `act-v0.5` contract tests stand on.
fn mid_act() -> Simulator {
    walk_until(fresh_run(), |simulator| {
        simulator.state().combat.is_none()
            && simulator
                .state()
                .run
                .as_ref()
                .is_some_and(|run| run.floor >= 4)
            && matches!(simulator.decision(), DecisionContext::MapNavigation { .. })
    })
}

/// The same state with `mutate` applied and the simulator rebuilt: how the
/// arithmetic tests fabricate a neighboring state that differs in one term.
fn rebuilt(original: &Simulator, mutate: impl FnOnce(&mut sts2_engine::GameState)) -> Simulator {
    let mut state = original.state().clone();
    mutate(&mut state);
    Simulator::from_scenario(state, sts2_content::standard_registry()).unwrap()
}

fn tiny_act() -> SearchConfig {
    SearchConfig {
        iterations: 8,
        rollout_depth: 12,
        temperature: 0.5,
    }
}

fn tiny_combat() -> SearchConfig {
    SearchConfig {
        iterations: 6,
        rollout_depth: 5,
        temperature: 0.5,
    }
}

/// The composed act-mode policy at test budgets, rolling out on the given
/// policies so a test can pick determinism or breadth.
fn composed() -> ActSearch<ActBoundary> {
    ActSearch::new(
        tiny_act(),
        ActBoundary::default(),
        Box::new(UniformRandom),
        BeliefSearch::with_rollout(
            tiny_combat(),
            CombatStrength::default(),
            Box::new(UniformRandom),
        ),
    )
}

#[test]
fn act_belief_keys_by_observation_and_deals_reproducible_fair_worlds() {
    let original = mid_act();
    let mut determinizer = ActBelief::from_simulator(&original, 5).unwrap();
    // The act arm keys by the plain observation now: observation v6 puts a
    // reward screen's offers on the decision, so the aliasing the retired
    // `NodeKey::Offers` digest was patching is gone at the source.
    let NodeKey::Observation(key) = determinizer.node_key(&original).unwrap() else {
        panic!("act belief search keys by observation");
    };
    assert_eq!(key, original.observation_key().unwrap());
    let one = determinizer.sample(2);
    assert_eq!(
        determinizer.node_key(&one).unwrap(),
        NodeKey::Observation(key),
        "a sample keys as the screen it shows"
    );
    let again = determinizer.sample(2);
    assert_eq!(
        one.state_key().unwrap(),
        again.state_key().unwrap(),
        "the same rollout index deals the same world"
    );
    let other = determinizer.sample(3);
    assert_ne!(
        one.state_key().unwrap(),
        other.state_key().unwrap(),
        "another rollout deals another hidden remainder"
    );
    assert_eq!(
        one.observation_key().unwrap(),
        other.observation_key().unwrap(),
        "every world shows the player the same screen"
    );
    assert_eq!(
        one.legal_actions(),
        original.legal_actions(),
        "and offers the real screen's actions"
    );
    assert!(
        !determinizer.deterministic_transitions(),
        "hidden state varies by rollout, so edges cache no keys"
    );
}

#[test]
fn the_act_horizon_is_the_engines_own_transition() {
    let original = mid_act();
    assert!(
        !act_over(&original),
        "a mid-act map decision is inside the horizon"
    );
    let determinizer = ActBelief::from_simulator(&original, 1).unwrap();
    assert!(!determinizer.beyond_horizon(&original));

    // The engine's transition screen, fabricated the way the engine raises
    // it: the phase flips and the decision follows.
    let transition = rebuilt(&original, |state| {
        state.run.as_mut().unwrap().phase = RunPhase::ActTransition;
    });
    assert!(matches!(
        transition.decision(),
        DecisionContext::ActTransition { .. }
    ));
    assert!(
        transition.legal_actions().contains(&Action::AdvanceAct),
        "the transition offers the crossing itself (potion housekeeping may \
         stand beside it)"
    );
    assert!(act_over(&transition), "the transition is the horizon");
    assert!(
        determinizer.beyond_horizon(&transition),
        "a rollout reaching the transition is scored where it stands"
    );
    assert!(
        ActBelief::from_simulator(&transition, 1).is_err(),
        "act-v0.5 refuses to stand at its own horizon"
    );

    // A finished run is inside the horizon's refusal too.
    let dead = rebuilt(&original, |state| {
        state.terminal = Some(RunResult::Defeat);
    });
    assert!(act_over(&dead));
    assert!(ActBelief::from_simulator(&dead, 1).is_err());
}

#[test]
fn a_standing_reward_screen_roots_the_act_search_fairly() {
    // The `fight_over` guard is combat-v0's, not act-v0.5's: the act erasure
    // pins standing screens, so a reward screen roots fairly — every sample
    // offers the real screen's actions, and the search answers with one.
    let original = walk_until(fresh_run(), |simulator| {
        alphaspire::env::fight_over(simulator) && simulator.state().combat.is_some()
    });
    assert!(
        Belief::from_simulator(&original, 3).is_err(),
        "combat-v0 still refuses the victory screen"
    );
    assert!(
        !act_over(&original),
        "a hallway fight's reward screen is not the act's exit"
    );
    let mut determinizer = ActBelief::from_simulator(&original, 3).unwrap();
    for rollout in 0..3 {
        let sample = determinizer.sample(rollout);
        assert_eq!(
            sample.legal_actions(),
            original.legal_actions(),
            "the standing screen is pinned, never resampled"
        );
        assert_eq!(
            sample.observation_key().unwrap(),
            original.observation_key().unwrap()
        );
    }
    let mut mcts = Mcts::new(tiny_act(), Uct::default());
    let objective = ActBoundary::default();
    let mut rollout = UniformRandom;
    let mut rng = MegaRandom::new(5);
    let action = mcts.decide(&mut determinizer, &objective, &mut rollout, &mut rng);
    assert!(
        original.legal_actions().contains(&action),
        "the search answers the real screen with the real screen's action"
    );
}

#[test]
fn an_act_sample_crosses_a_fight_boundary_inside_the_horizon() {
    // The chosen fight-boundary v0: a rollout that reaches a fight entry
    // plays the fight with the rollout policy. So a sampled world walked the
    // way `leaf_value` walks it — step while inside the horizon — enters a
    // fight, stays inside the horizon there, comes out the other side, and
    // keeps crossing floors.
    let mut determinizer = ActBelief::from_simulator(&mid_act(), 7).unwrap();
    let mut sample = determinizer.sample(2);
    let floor = sample.state().run.as_ref().unwrap().floor;
    let mut fought = false;
    let mut fight_finished = false;
    for _ in 0..150 {
        if sample.state().terminal.is_some() || determinizer.beyond_horizon(&sample) {
            break;
        }
        if !alphaspire::env::fight_over(&sample) {
            assert!(
                !determinizer.beyond_horizon(&sample),
                "a fight inside the act is inside the horizon"
            );
            fought = true;
        } else if fought {
            fight_finished = true;
        }
        let action = next_action(&sample);
        sample.step_quietly(&action).unwrap();
    }
    assert!(fought, "the walk crossed into a fight");
    assert!(fight_finished, "and out of it, still inside the sample");
    assert!(
        sample.state().run.as_ref().unwrap().floor >= floor + 3,
        "the walk kept crossing floors past the fight"
    );
}

#[test]
fn act_boundary_arithmetic() {
    let weights = ActBoundary::default();
    let mut objective = ActBoundary::default();
    let original = mid_act();
    let base = objective.peek(&original);
    assert!(base > 0.0, "a live mid-act run is worth something");
    assert!(
        (objective.reward(&original) - base).abs() < f64::EPSILON,
        "the objective is stateless: reward and peek agree"
    );

    // Crossing pays exactly the crossing weight: the fabricated transition
    // differs from the base state in nothing else.
    let crossed = objective.peek(&rebuilt(&original, |state| {
        state.run.as_mut().unwrap().phase = RunPhase::ActTransition;
    }));
    assert!(
        (crossed - base - weights.crossing_weight).abs() < 1e-9,
        "crossing the act pays the crossing weight: {crossed} vs {base}"
    );

    // A floor climbed pays the floor weight.
    let deeper = objective.peek(&rebuilt(&original, |state| {
        state.run.as_mut().unwrap().floor += 1;
    }));
    assert!((deeper - base - weights.floor_weight).abs() < 1e-9);

    // Hit points retained pay proportionally.
    let max_hp = original.state().run_player.max_hp;
    let hurt = objective.peek(&rebuilt(&original, |state| {
        state.run_player.current_hp -= 10;
    }));
    let expected = weights.hp_weight * 10.0 / f64::from(max_hp);
    assert!((base - hurt - expected).abs() < 1e-9);

    // A potion held pays its shadow price.
    let stocked = objective.peek(&rebuilt(&original, |state| {
        let slot = state
            .run_player
            .potions
            .iter_mut()
            .find(|slot| slot.is_none())
            .expect("the belt has an empty slot");
        *slot = Some("POTION.POISON_POTION".parse().unwrap());
    }));
    assert!((stocked - base - weights.potion_weight).abs() < 1e-9);

    // A defeat pays for the floors it climbed and for nothing else — the
    // difference between an objective and a constant on this horizon, where
    // nearly every rollout ends on a dead player.
    let dead = objective.peek(&rebuilt(&original, |state| {
        state.terminal = Some(RunResult::Defeat);
    }));
    let climbed = f64::from(original.state().run.as_ref().unwrap().floor);
    assert!(
        (dead - weights.floor_weight * climbed).abs() < 1e-9,
        "a defeat is worth its climb: {dead}"
    );
    let deeper_death = objective.peek(&rebuilt(&original, |state| {
        state.terminal = Some(RunResult::Defeat);
        state.run.as_mut().unwrap().floor += 4;
    }));
    assert!(
        deeper_death > dead,
        "and a deeper one is worth more: {deeper_death} vs {dead}"
    );
    assert!(
        base > deeper_death,
        "while any live act outranks any dead one: {base} vs {deeper_death}"
    );

    // A leaf standing mid-fight pays the turn price of the fight it is in,
    // as a discount on the crossing terms and never on the climb — so the
    // ordering holds at any turn count and a live act never prices under the
    // dead one it is still ahead of.
    let in_combat = walk_until(fresh_run(), |simulator| simulator.state().combat.is_some());
    let at = |turns: i32| {
        objective.peek(&rebuilt(&in_combat, |state| {
            let turn = &mut state.combat.as_mut().unwrap().player.turn;
            *turn = turn.saturating_add_signed(turns);
        }))
    };
    let climbed = weights.floor_weight * f64::from(in_combat.state().run.as_ref().unwrap().floor);
    assert!(at(0) > at(3), "three turns of fight cost something");
    assert!(at(400) > at(500), "and still do four hundred turns in");
    assert!(
        at(500) > climbed,
        "while no length of fight prices a live act under a dead one"
    );
}

#[test]
fn the_composed_policy_walks_a_run_legally_and_reproducibly() {
    // The engine-refusal guard covers the act arm end to end: every action
    // the composed policy answers a real screen with is one the engine will
    // step — sampled `?` rooms can deal EVENT.TRIAL inside rollouts, and the
    // guard sits in `node_for` and in every rollout policy. And the walk is
    // a function of its seeds alone.
    let walk = || {
        let mut policy = composed();
        let mut rng = MegaRandom::new(11);
        let mut simulator = fresh_run();
        let mut actions = Vec::new();
        for _ in 0..40 {
            if simulator.state().terminal.is_some() {
                break;
            }
            let action = policy.choose(&simulator, &mut rng);
            assert!(
                simulator.legal_actions().contains(&action),
                "the composed policy answers with a legal action: {action:?} \
                 at {:?} offering {:?}",
                simulator.decision(),
                simulator.legal_actions(),
            );
            assert!(
                !engine_refuses(&simulator, &action),
                "and never one the engine enumerates but refuses"
            );
            simulator.step_quietly(&action).unwrap();
            actions.push(action);
        }
        policy.run_ended(&simulator);
        assert!(
            policy.drain_decisions().is_empty(),
            "no recorder was attached, so nothing is recorded"
        );
        assert!(
            policy.drain_macro_decisions().is_empty(),
            "and no macro recorder either"
        );
        (
            actions,
            simulator.state().run.as_ref().map_or(0, |run| run.floor),
        )
    };
    let (actions_one, floor) = walk();
    let (actions_two, _) = walk();
    assert_eq!(actions_one, actions_two, "one seed pair, one line of play");
    assert!(
        floor >= 2,
        "the searched walk genuinely progresses (floor {floor})"
    );
}

#[test]
fn act_mode_keeps_recording_in_combat_samples_and_nothing_else() {
    // Constraint of this chunk: in-combat sample emission works unchanged
    // under the composed policy — the recorder rides the combat search, and
    // the composed policy settles its pending fight at the first
    // out-of-combat decision, where the combat search's own `choose` no
    // longer sees the state. Out-of-combat decisions record nothing until
    // the scheduled encoding bump.
    let mut policy = ActSearch::new(
        tiny_act(),
        ActBoundary::default(),
        Box::new(UniformRandom),
        BeliefSearch::with_rollout(
            tiny_combat(),
            CombatStrength::default(),
            Box::new(UniformRandom),
        )
        .recording(),
    );
    let mut rng = MegaRandom::new(23);
    let mut simulator = fresh_run();
    let mut in_fight_decisions = 0;
    let mut settled_after_fight = false;
    for _ in 0..60 {
        if simulator.state().terminal.is_some() {
            break;
        }
        let fighting = !alphaspire::env::fight_over(&simulator);
        let action = policy.choose(&simulator, &mut rng);
        if fighting {
            in_fight_decisions += 1;
        } else if in_fight_decisions > 0 && !settled_after_fight {
            // The first out-of-combat decision after a fight: the composed
            // settle seam has fired, so the fight's samples are complete
            // and drainable without waiting for the run to end.
            let samples = policy.drain_decisions();
            assert!(
                !samples.is_empty(),
                "the fight's samples settle at the fight's end"
            );
            assert!(
                samples
                    .iter()
                    .all(|sample| sample.encounter.is_some() && !sample.pi.is_empty()),
                "every sample is an in-combat decision with a policy target"
            );
            settled_after_fight = true;
        }
        simulator.step_quietly(&action).unwrap();
    }
    assert!(in_fight_decisions > 0, "the walk fought");
    assert!(settled_after_fight, "and the fight settled mid-run");
}

#[test]
fn act_mode_selfplay_reports_through_the_outcome_tally() {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let batch = alphaspire::selfplay::Batch {
        preset: &preset,
        character: &character,
        ascension: 0,
        analysis_seed: 17,
        seed: None,
        runs: 2,
        max_steps: 40,
        jobs: 2,
        harvest: None,
        force_wins: None,
    };
    let make_policy = || Box::new(composed()) as Box<dyn RolloutPolicy>;
    let make_objective = || Box::new(ActBoundary::default()) as Box<dyn Objective>;
    let mut tally = alphaspire::selfplay::OutcomeTally::default();
    let mut reports = 0;
    alphaspire::selfplay::play_batch(&batch, &make_policy, &make_objective, &mut |_, played| {
        let report = played.expect("an act-mode run completes without panics");
        tally.record(&report);
        reports += 1;
    });
    assert_eq!(reports, 2, "every run reported");
    assert!(
        tally.summary().contains("act 1 cleared"),
        "the tally speaks the project's yardstick: {}",
        tally.summary()
    );
}

/// Control determinizer with the act belief's fight boundary disabled.
struct Unfenced(ActBelief);

impl Determinizer for Unfenced {
    fn sample(&mut self, rollout: u64) -> Simulator {
        self.0.sample(rollout)
    }

    fn node_key(&self, simulator: &Simulator) -> Result<NodeKey, sts2_engine::EngineError> {
        self.0.node_key(simulator)
    }

    fn beyond_horizon(&self, simulator: &Simulator) -> bool {
        self.0.beyond_horizon(simulator)
    }
}

#[test]
fn a_fight_inside_an_act_rollout_is_played_not_searched() {
    // `act-v0.5`'s design is that a rollout which reaches a fight plays it
    // with the cheap rollout policy; the tree branches over the act's own
    // decisions only. Without the fence the tree creeps a ply into the fight
    // per iteration and pays for every crept ply again on every later
    // descent — the cost pathology act-mode batches ran into.
    let original = mid_act();
    assert!(
        !ActBelief::from_simulator(&original, 3)
            .unwrap()
            .plays_out(&original),
        "an out-of-combat decision is the act tree's to search"
    );
    let in_combat = walk_until(fresh_run(), |simulator| {
        simulator.state().combat.is_some() && !alphaspire::env::fight_over(simulator)
    });
    assert!(
        ActBelief::from_simulator(&original, 3)
            .unwrap()
            .plays_out(&in_combat),
        "a live fight is played, not searched"
    );

    let config = SearchConfig {
        iterations: 24,
        rollout_depth: 24,
        temperature: 0.0,
    };
    let search = |mut determinizer: Box<dyn Determinizer>| {
        let mut mcts = Mcts::new(config, Uct::default());
        let mut rollout = UniformRandom;
        let mut rng = MegaRandom::new(31);
        mcts.decide(
            determinizer.as_mut(),
            &ActBoundary::default(),
            &mut rollout,
            &mut rng,
        );
        (mcts.tree_size(), mcts.steps_taken())
    };
    let (fenced_nodes, fenced_steps) =
        search(Box::new(ActBelief::from_simulator(&original, 9).unwrap()));
    let (loose_nodes, loose_steps) = search(Box::new(Unfenced(
        ActBelief::from_simulator(&original, 9).unwrap(),
    )));
    assert!(
        fenced_nodes < loose_nodes,
        "the fenced tree holds only the act's own decisions: {fenced_nodes} vs {loose_nodes}"
    );
    assert!(
        fenced_steps < loose_steps,
        "and re-walks fewer engine steps: {fenced_steps} vs {loose_steps}"
    );
}

#[test]
fn a_spent_budget_downgrades_macro_decisions_and_the_run_walks_on() {
    // The circuit breaker: a run out of macro-search budget answers its
    // out-of-combat decisions with the rollout policy and keeps walking —
    // never truncated, never killed, and still reported.
    let walk = |budget: Budget| {
        let mut policy = composed().within(budget);
        let mut rng = MegaRandom::new(13);
        let mut simulator = fresh_run();
        for _ in 0..40 {
            if simulator.state().terminal.is_some() {
                break;
            }
            let action = policy.choose(&simulator, &mut rng);
            simulator.step_quietly(&action).unwrap();
        }
        policy.run_ended(&simulator);
        (
            policy
                .budget_spent()
                .expect("the act arm reports its spend"),
            simulator.state().run.as_ref().map_or(0, |run| run.floor),
        )
    };

    let (unbudgeted, floor) = walk(Budget::default());
    assert!(Budget::default().is_off(), "no budget is the default");
    assert!(unbudgeted.searched > 0, "an unbudgeted run searches");
    assert_eq!(unbudgeted.downgraded, 0, "and downgrades nothing");
    assert!(unbudgeted.steps > 0, "and spends engine steps doing it");

    let (spent, budgeted_floor) = walk(Budget {
        steps: Some(0),
        seconds: None,
    });
    assert_eq!(spent.searched, 0, "a spent budget searches nothing");
    assert!(spent.downgraded > 0, "every macro decision downgraded");
    assert_eq!(spent.steps, 0, "and the act tree cost nothing");
    assert!(
        budgeted_floor >= floor.min(2),
        "the run still walked: floor {budgeted_floor} against {floor}"
    );
}

#[test]
fn macro_samples_carry_the_act_they_were_searched_in_and_its_horizon_score() {
    // The macro track's whole point: one (x, pi, z) per searched macro
    // decision, z the act-boundary horizon score shared by the act, and the
    // in-combat set untouched beside it in its own channel.
    let mut policy = ActSearch::new(
        tiny_act(),
        ActBoundary::default(),
        Box::new(UniformRandom),
        BeliefSearch::with_rollout(
            tiny_combat(),
            CombatStrength::default(),
            Box::new(UniformRandom),
        )
        .recording(),
    )
    .recording_macro();
    let mut rng = MegaRandom::new(29);
    let mut simulator = fresh_run();
    for _ in 0..60 {
        if simulator.state().terminal.is_some() {
            break;
        }
        let action = policy.choose(&simulator, &mut rng);
        simulator.step_quietly(&action).unwrap();
    }
    assert!(
        policy.drain_macro_decisions().is_empty(),
        "nothing settles before the act's horizon does"
    );
    policy.run_ended(&simulator);
    let macro_samples = policy.drain_macro_decisions();
    let combat_samples = policy.drain_decisions();
    assert!(
        !macro_samples.is_empty(),
        "the walk searched macro decisions"
    );
    assert!(
        !combat_samples.is_empty(),
        "and fought, recording as before"
    );
    let expected = ActBoundary::default().peek(&simulator);
    for sample in &macro_samples {
        assert!(
            sample.encounter.is_none(),
            "a macro decision belongs to no fight"
        );
        assert_eq!(sample.fight, 0, "and to no fight index either");
        assert_eq!(
            sample.act,
            Some(0),
            "it belongs to the act it was searched in"
        );
        assert!(
            !sample.actions.is_empty(),
            "the decision's actions are encoded"
        );
        assert_eq!(sample.pi.len(), sample.actions.len(), "pi is aligned");
        let total: f32 = sample.pi.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "pi is a distribution: {total}");
        #[allow(clippy::cast_possible_truncation, reason = "z is written as f32")]
        let settled = expected as f32;
        assert!(
            (sample.z - settled).abs() < 1e-6,
            "every macro sample of an act shares the act's horizon score"
        );
    }
    assert!(
        combat_samples
            .iter()
            .all(|sample| sample.encounter.is_some() && sample.act.is_none()),
        "the in-combat set is stamped the way it always was"
    );
}

#[test]
fn a_screen_that_re_offers_itself_does_not_circle_the_act_tree_forever() {
    // A potion reward claimed onto a full belt is accepted by the engine and
    // settles nothing: the screen stands exactly where it stood, same
    // observation, same offers. That is a one-step circle in the act tree's
    // own key space, and a tree descent would walk it forever — selection is
    // a pure function of statistics no descent updates until it has
    // finished, so a walk that comes back to a decision point repeats the
    // choice that brought it there.
    //
    // Two guards stand over it, and this pins both: `permitted_actions`
    // withholds the inert step, so no node opens it, and the descent's own
    // key check would score the walk where it stands if anything else closed
    // a loop.
    let standing = walk_until(fresh_run(), |simulator| {
        matches!(simulator.decision(), DecisionContext::Rewards { offers, .. }
            if offers.iter().any(|offer| offer.reward_type == "potion"))
    });
    let full = rebuilt(&standing, |state| {
        let potion: sts2_core::ModelId = "POTION.ASHWATER".parse().unwrap();
        for slot in &mut state.run_player.potions {
            *slot = Some(potion.clone());
        }
    });

    // The engine's own word that the circle is there.
    let key = full.observation_key().unwrap();
    let claim = full
        .legal_actions()
        .iter()
        .find(|action| matches!(action, Action::ClaimReward { .. }))
        .cloned()
        .expect("the standing screen offers a claim");
    let mut stepped = full.clone();
    stepped.step_quietly(&claim).unwrap();
    assert_eq!(
        stepped.observation_key().unwrap(),
        key,
        "claiming a potion onto a full belt lands back on the screen it was claimed from"
    );

    assert!(
        !alphaspire::policy::permitted_actions(&full).contains(&claim),
        "and the policy layer withholds it, so no node of any tree opens it"
    );

    // The search over the same screen still answers legally and stays
    // bounded. The budget is large enough that the root runs out of actions
    // to open, which is when a descent would first have to walk back into a
    // decision point it already stood on.
    let config = SearchConfig {
        iterations: 64,
        rollout_depth: 2,
        temperature: 0.5,
    };
    let mut mcts = Mcts::new(config, Gumbel::default());
    let mut determinizer = ActBelief::from_simulator(&full, 11).expect("an act in progress erases");
    let objective = ActBoundary::default();
    let mut rollout = UniformRandom;
    let mut rng = MegaRandom::new(4);
    let action = mcts.decide(&mut determinizer, &objective, &mut rollout, &mut rng);
    assert!(
        full.legal_actions().contains(&action),
        "the search still answers with a legal action"
    );
    // One rollout plus a short walk down a tree this shallow, per iteration.
    // A descent that circled would not come back at all, so any finite
    // ceiling is the assertion; this one is tight enough to also catch a
    // walk that starts wandering.
    let ceiling = u64::from(config.iterations) * u64::from(config.rollout_depth + 4);
    assert!(
        mcts.steps_taken() <= ceiling,
        "the descent stays bounded: {} steps against {ceiling}",
        mcts.steps_taken()
    );
}
