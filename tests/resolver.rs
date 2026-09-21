//! The step ceiling on a rollout's in-fight resolver.
//!
//! Four properties. A resolver under a step budget stops searching once it
//! has spent it and answers from the checkpoint instead — a downgrade to the
//! cheap answer, never to nothing, so the run walks to its own end and
//! terminates exactly as an unbudgeted one does. The spend is reported, so a
//! batch can say what its fights cost. A budget of zero steps downgrades from
//! the first decision, which is the resolver ladder's bottom rung reached by
//! the ceiling rather than by a flag. And a budgeted run is still a function
//! of its seed pair, because the ceiling counts engine steps rather than
//! seconds — a batch that writes training data may have no other kind.

use std::path::Path;
use std::sync::Arc;

use alphaspire::actor::{MacroActor, Resolver, TierBudget};
use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::{Evaluate, PolicyValueNet};
use alphaspire::objective::CombatStrength;
use alphaspire::policy::RolloutPolicy;
use alphaspire::search::{Budget, BudgetSpend, Gumbel, SearchConfig};
use alphaspire::selfplay::RunReport;
use sts2_rng::MegaRandom;

fn fixture() -> Arc<dyn Evaluate> {
    let registry = sts2_content::standard_registry();
    let encoder = Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ));
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    Arc::new(PolicyValueNet::load(&base, encoder).expect("the fixture checkpoint loads"))
}

/// A searched resolver cheap enough to compose a whole run out of in a debug
/// build, under `budget`.
fn searched(budget: Budget) -> Resolver {
    Resolver::Searched {
        config: SearchConfig {
            iterations: 4,
            rollout_depth: 4,
            ..SearchConfig::default()
        },
        selection: Gumbel::default(),
        budget,
        elite: None,
        boss: None,
    }
}

fn episode(resolver: Resolver) -> RunReport {
    let net = fixture();
    let mut policy: Box<dyn RolloutPolicy> = Box::new(MacroActor::new(
        Arc::clone(&net),
        resolver.build(Arc::clone(&net)),
    ));
    let mut rng = MegaRandom::new(7);
    let mut objective = CombatStrength::default();
    alphaspire::selfplay::play_run(
        "NLD6VZXP94",
        &"CHARACTER.IRONCLAD".parse().unwrap(),
        0,
        policy.as_mut(),
        &mut rng,
        &mut objective,
        4000,
    )
    .expect("the run walks")
}

fn spend(report: &RunReport) -> BudgetSpend {
    report
        .budget
        .expect("a searched resolver reports its spend")
}

#[test]
fn an_unbudgeted_resolver_searches_every_fight_decision() {
    let report = episode(searched(Budget::default()));
    let spent = spend(&report);
    assert!(spent.searched > 0, "the run had fights to search");
    assert_eq!(
        spent.downgraded, 0,
        "and nothing downgraded it: the ceiling is off by default"
    );
    assert!(spent.steps > 0, "the search spent engine steps");
}

#[test]
fn a_resolver_past_its_ceiling_answers_from_the_checkpoint_instead() {
    // The ceiling exists because a batch's wall time is its slowest run and
    // the combat arm had no bound of any kind. What it must not do is stop
    // the run: past the cap the fights are still answered, by the prior the
    // search would have started from.
    // The cap is half of what this run actually spends unbudgeted, not a
    // number chosen once: what a fixture net costs a run is a property of the
    // fixture, and a constant tuned against one of them stops testing the
    // ceiling the moment the fixture is retrained.
    let unbudgeted = episode(searched(Budget::default()));
    let ceiling = spend(&unbudgeted).steps / 2;
    assert!(ceiling > 0, "the unbudgeted run spent steps to halve");
    let capped = episode(searched(Budget {
        steps: Some(ceiling),
        seconds: None,
    }));
    let spent = spend(&capped);
    assert!(
        spent.searched > 0 && spent.downgraded > 0,
        "the cap fell part-way through the run: {spent:?}"
    );
    assert!(
        spent.steps >= ceiling,
        "and it fell where the steps ran out: {spent:?}"
    );
    assert!(
        capped.terminal.is_some(),
        "a downgraded run still walks to its own end"
    );
    assert!(
        spent.steps < spend(&unbudgeted).steps,
        "and it spent less doing so"
    );
}

#[test]
fn a_ceiling_of_zero_steps_is_the_greedy_resolver_reached_by_the_ceiling() {
    // The bottom rung of the ladder, arrived at from above: nothing is ever
    // searched, every fight decision is one forward pass, and the episode is
    // as complete as any other. A downgrade that fell back to nothing — or to
    // the search core's own rollout, which plays fights at uniform random —
    // would be visible here as a run that died in its first fight.
    let capped = episode(searched(Budget {
        steps: Some(0),
        seconds: None,
    }));
    let spent = spend(&capped);
    assert_eq!(spent.searched, 0, "nothing was searched");
    assert_eq!(spent.steps, 0, "so no engine step was spent searching");
    assert!(spent.downgraded > 0, "and every decision was answered");
    assert!(capped.terminal.is_some(), "the episode is complete");
    let greedy = episode(Resolver::Greedy);
    assert_eq!(
        capped.floor, greedy.floor,
        "a resolver capped at zero plays the run the greedy resolver plays"
    );
    assert_eq!(
        capped.macro_decisions, greedy.macro_decisions,
        "decision for decision"
    );
}

#[test]
fn a_budgeted_run_is_still_a_function_of_its_seed_pair() {
    // The ceiling counts engine steps, so what it downgrades does not depend
    // on what else the box was doing. A wall-clock ceiling would make two
    // batches of the same seeds record different episodes, which is why a
    // rollout offers no flag for one.
    let budget = Budget {
        steps: Some(300),
        seconds: None,
    };
    let first = episode(searched(budget));
    let again = episode(searched(budget));
    assert_eq!(
        spend(&first).downgraded,
        spend(&again).downgraded,
        "the cap fell in the same place"
    );
    assert_eq!(
        first.macro_decisions, again.macro_decisions,
        "and the episode is the same episode"
    );
}

#[test]
fn a_tier_budget_is_its_own_tree_under_the_run_s_one_ceiling() {
    // A boss searched deeper than a hallway: the tier's tree spends its own
    // budget, the spend reported is every tree's together, the run's step
    // ceiling still holds over all of them, and the episode stays a function
    // of its seed pair.
    let Resolver::Searched {
        config,
        selection,
        budget,
        ..
    } = searched(Budget::default())
    else {
        unreachable!("the helper builds a searched resolver")
    };
    let tiered = Resolver::Searched {
        config,
        selection,
        budget,
        elite: Some(TierBudget {
            iterations: 8,
            considered: 4,
        }),
        boss: Some(TierBudget {
            iterations: 16,
            considered: 8,
        }),
    };
    let flat = episode(searched(Budget::default()));
    let once = episode(tiered);
    let again = episode(tiered);
    assert!(spend(&once).searched > 0, "the tiered resolver searches");
    assert_eq!(
        alphaspire::selfplay::summarize(&once),
        alphaspire::selfplay::summarize(&again),
        "a tiered episode is a function of its seed pair"
    );
    assert_eq!(spend(&once).steps, spend(&again).steps);
    // The same run at a flat budget spends less than the one that searched
    // its elites and bosses deeper, whenever the walk met one.
    assert!(
        spend(&once).steps >= spend(&flat).steps,
        "tiered {} vs flat {}",
        spend(&once).steps,
        spend(&flat).steps
    );
    let capped = Resolver::Searched {
        config,
        selection,
        budget: Budget {
            steps: Some(40),
            seconds: None,
        },
        elite: Some(TierBudget {
            iterations: 8,
            considered: 4,
        }),
        boss: Some(TierBudget {
            iterations: 16,
            considered: 8,
        }),
    };
    let held = episode(capped);
    assert!(
        spend(&held).downgraded > 0,
        "the ceiling is read over every tree: {:?}",
        spend(&held)
    );
}
