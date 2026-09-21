//! The out-of-combat guard and the heuristic policy, pinned against the
//! actual registered content.

use alphaspire::heuristics::Heuristic;
use alphaspire::objective::{CombatStrength, Objective};
use alphaspire::policy::{
    RolloutPolicy, permitted_actions, poisoned_event_option, spent_event_option,
};
use alphaspire::selfplay::{Batch, RunReport, play_batch};
use sts2_core::UnlockPresetManifest;
use sts2_engine::{EventEffect, EventGeneration, EventOption};

fn option_named<'a>(options: &'a [EventOption], suffix: &str) -> &'a EventOption {
    options
        .iter()
        .find(|option| option.option_id.ends_with(suffix))
        .unwrap_or_else(|| panic!("an option ending in {suffix} stands on the page"))
}

/// EVENT.TRIAL's Reject page, read from the registered content itself:
/// Double-down steps into `EventEffect::Unsupported` and must be poisoned;
/// Accept must not be, so the guard never dead-ends the page.
#[test]
fn the_trial_double_down_line_is_poisoned_and_nothing_beside_it() {
    let registry = sts2_content::standard_registry();
    let trial = registry
        .event(&"EVENT.TRIAL".parse().unwrap())
        .expect("EVENT.TRIAL is registered");
    let EventGeneration::Fixed(page) = &trial.generation else {
        panic!("EVENT.TRIAL lays a fixed page");
    };
    let reject = option_named(page, ".REJECT");
    assert!(
        !poisoned_event_option(&registry, reject),
        "reject only opens a page; the poison sits one page deeper"
    );
    let reject_page: Vec<EventOption> = reject
        .effects
        .iter()
        .find_map(|effect| match effect {
            EventEffect::SetPage(options) => Some(options.clone()),
            _ => None,
        })
        .expect("reject opens a page");
    assert!(
        poisoned_event_option(&registry, option_named(&reject_page, ".DOUBLE_DOWN")),
        "the double-down line steps into Unsupported and is filtered"
    );
    assert!(
        !poisoned_event_option(&registry, option_named(&reject_page, ".ACCEPT")),
        "the page keeps a live line, so filtering cannot dead-end it"
    );
}

/// The relic arm of the guard, against the registered content. One relic
/// keeps `offered_only`'s refused `AfterObtained` default
/// (`sts2-content/src/relics.rs`): the multiplayer-only `MASSIVE_SCROLL`.
/// Every relic a single-player run is offered (Neow's `KALEIDOSCOPE`, Orobas'
/// `TOUCH_OF_OROBAS` and `PRISMATIC_GEM`) overrides the default with a real
/// body and must stay on offer.
#[test]
fn only_relics_refused_on_obtain_poison_the_options_that_obtain_them() {
    let registry = sts2_content::standard_registry();
    let obtain = |relic: &str| EventOption {
        option_id: format!("TEST.options.{relic}"),
        label: relic.to_owned(),
        effects: vec![EventEffect::ObtainRelic(relic.parse().unwrap())],
        is_proceed: false,
        was_chosen: false,
        locked: false,
    };
    assert!(
        poisoned_event_option(&registry, &obtain("RELIC.MASSIVE_SCROLL")),
        "RELIC.MASSIVE_SCROLL is refused on obtain and must be filtered"
    );
    for offered in [
        "RELIC.KALEIDOSCOPE",
        "RELIC.TOUCH_OF_OROBAS",
        "RELIC.PRISMATIC_GEM",
        "RELIC.GOLDEN_PEARL",
    ] {
        assert!(
            registry.relic(&offered.parse().unwrap()).is_some(),
            "{offered} is registered content"
        );
        assert!(
            !poisoned_event_option(&registry, &obtain(offered)),
            "{offered} has a real AfterObtained body and stays on offer"
        );
    }
}

/// At a real run's root — Neow's page — nothing is filtered: the guard is
/// a pass-through everywhere the engine's offers are honest, and it hands
/// back the engine's own slice without copying.
#[test]
fn the_guard_passes_an_honest_screen_through_untouched() {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let simulator = sts2_content::standard_run_on_preset_at("NLD6VZXP94", &character, &preset, 0)
        .expect("the pinned baseline generates");
    let permitted = permitted_actions(&simulator);
    assert_eq!(
        permitted.as_ref(),
        simulator.legal_actions(),
        "an honest page loses nothing to the filter"
    );
    assert!(
        matches!(permitted, std::borrow::Cow::Borrowed(_)),
        "the pass-through allocates nothing"
    );
}

fn walk_heuristic(jobs: usize) -> Vec<(String, String)> {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let batch = Batch {
        preset: &preset,
        character: &character,
        ascension: 0,
        analysis_seed: 401,
        seed: None,
        runs: 4,
        max_steps: 80,
        jobs,
        harvest: None,
        force_wins: None,
    };
    let policy = || Box::new(Heuristic) as Box<dyn RolloutPolicy>;
    let objective = || Box::new(CombatStrength::default()) as Box<dyn Objective>;
    let mut walked = Vec::new();
    play_batch(&batch, &policy, &objective, &mut |index, played| {
        let report: RunReport = played.unwrap();
        assert_eq!(walked.len(), index, "reports arrive in run order");
        walked.push((report.seed, format!("{:?}", report.actions)));
    });
    walked
}

/// The repo invariant the heuristic must keep: a run is a function of its
/// seed pair, so a parallel batch replays the serial one action for action.
#[test]
fn the_heuristic_batch_is_reproducible_across_jobs() {
    assert_eq!(
        walk_heuristic(1),
        walk_heuristic(3),
        "jobs must not change what any run plays"
    );
}

/// The heuristic walks a real run forward on its own: no search on top, a
/// few hundred steps, and the run must genuinely progress the way the
/// first-legal smoke test's runs do.
#[test]
fn the_heuristic_walks_a_run() {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let mut policy = Heuristic;
    let mut rng = sts2_rng::MegaRandom::new(9);
    let mut objective = CombatStrength::default();
    let report = alphaspire::selfplay::play_run_on_preset(
        &preset,
        "NLD6VZXP94",
        &character,
        0,
        &mut policy,
        &mut rng,
        &mut objective,
        400,
        None,
        None,
    )
    .expect("the heuristic never steps an action the engine refuses");
    assert!(
        report.floor >= 2 || report.terminal.is_some(),
        "the run never left the first floor (floor {})",
        report.floor
    );
}

/// An option whose body has already run is not a step worth taking.
///
/// Orobas is the page this matters on: handing over the sea glass obtains a
/// relic and opens a card grid without closing the room, so the option comes
/// back marked chosen and the engine answers a second click by returning
/// without running anything. Left on the list it is a livelock — a greedy
/// batch caught two runs of a hundred there, one spending 4,843 of its 4,906
/// decisions re-clicking that one option — so `permitted_actions` withholds
/// it and every consumer is covered at once.
#[test]
fn an_event_option_that_has_already_run_is_spent() {
    let registry = sts2_content::standard_registry();
    let orobas = registry
        .event(&"EVENT.OROBAS".parse().unwrap())
        .expect("EVENT.OROBAS is registered");
    assert!(
        matches!(orobas.generation, EventGeneration::OrobasRelics { .. }),
        "Orobas lays its page out itself, which is why a spent option survives \
         on it: nothing replaces the page the way SetPage would"
    );

    let sea_glass = EventOption {
        option_id: "OROBAS.pages.INITIAL.options.SEA_GLASS".to_owned(),
        label: "Sea glass".to_owned(),
        effects: vec![EventEffect::ObtainRelic(
            "RELIC.TOUCH_OF_OROBAS".parse().unwrap(),
        )],
        is_proceed: false,
        was_chosen: false,
        locked: false,
    };
    assert!(
        !spent_event_option(&sea_glass),
        "an option nobody has taken is a live line"
    );
    let taken = EventOption {
        was_chosen: true,
        locked: false,
        ..sea_glass.clone()
    };
    assert!(
        spent_event_option(&taken),
        "and the same option, once its body has run, is a step that changes \
         nothing"
    );
    assert!(
        !poisoned_event_option(&registry, &taken),
        "spent and poisoned are different refusals; this one is not about the \
         relic behind the option"
    );
}
