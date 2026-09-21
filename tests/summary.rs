//! The batch summary a rollout writes beside its shards.
//!
//! The properties a summary holds. A run records one fight per fight it entered,
//! with the room and encounter it was fought in and the hit points it cost —
//! none of which survives into a finished report, which is the whole reason
//! the collection happens while the run is played. The last fight of a run
//! that died is the one that killed it. A faulted episode is counted and
//! moves no other number in the file. The per-fight cost table gives every
//! won fight a cell. And every scalar the trainer logs is a number exactly
//! one nesting level down, because that is the depth its reader scans.

use std::path::Path;
use std::sync::Arc;

use alphaspire::actor::{MacroActor, Resolver};
use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::{Evaluate, PolicyValueNet};
use alphaspire::objective::CombatStrength;
use alphaspire::policy::RolloutPolicy;
use alphaspire::selfplay::RunReport;
use alphaspire::summary::{BatchSummary, Outcome, Provenance};
use sts2_rng::MegaRandom;

fn encoder() -> Arc<PolicyEncoder> {
    let registry = sts2_content::standard_registry();
    Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ))
}

/// The fixture checkpoint standing in for both nets: what is under test is
/// the bookkeeping around the run, and a tiny net walks one as honestly as a
/// trained one does.
fn fixture() -> Arc<dyn Evaluate> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    Arc::new(PolicyValueNet::load(&base, encoder()).expect("the fixture checkpoint loads"))
}

/// One episode played to its terminal or to `max_steps`.
fn episode(max_steps: usize) -> RunReport {
    let net = fixture();
    let mut policy: Box<dyn RolloutPolicy> = Box::new(MacroActor::new(
        Arc::clone(&net),
        Resolver::Greedy.build(Arc::clone(&net)),
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
        max_steps,
    )
    .expect("the run walks")
}

fn provenance() -> Provenance {
    Provenance {
        run_net: "ckpts/run".into(),
        combat_net: "ckpts/combat".into(),
        resolver: Resolver::Greedy,
        character: "CHARACTER.IRONCLAD".into(),
        ascension: 0,
        runs: 1,
        analysis_seed: 1,
        max_steps: 4000,
    }
}

#[test]
fn a_run_records_one_fight_for_every_fight_it_entered() {
    let report = episode(4000);
    let metrics = report.metrics.as_ref().expect("the actor collected them");
    assert!(
        !metrics.fights().is_empty(),
        "a run that reached act one fought something"
    );
    for fight in metrics.fights() {
        assert!((1..=3).contains(&fight.act), "the act counts from one");
        assert!(fight.floor > 0, "a fight happens on a floor");
        assert!(
            fight.encounter.is_some(),
            "a fight names the encounter it was, not the room shell around it"
        );
        assert!(fight.max_hp > 0, "the player has a maximum");
        assert!(
            fight.hp_before <= fight.max_hp && fight.hp_after <= fight.max_hp,
            "and neither reading is above it: {fight:?}"
        );
        assert!(
            matches!(fight.outcome, Outcome::Won | Outcome::Lost),
            "a run played to its terminal resolved every fight it started"
        );
    }
}

#[test]
fn the_last_fight_of_a_dead_run_is_the_one_that_killed_it() {
    // This is the diagnostic the whole file exists for: it is what turns
    // "the macro policy is weak" into "the deaths are at bosses in act two",
    // and nothing in a finished report can be read backwards into it.
    let report = episode(4000);
    let death = report.death.as_ref().expect("the seed's run dies");
    let metrics = report.metrics.as_ref().expect("the actor collected them");
    let last = metrics.fights().last().expect("it died in a fight");
    assert_eq!(last.outcome, Outcome::Lost);
    assert_eq!(
        last.encounter,
        death.encounter.as_ref().map(ToString::to_string),
        "the record's encounter is the one the death table names"
    );
    assert_eq!(
        last.floor, report.floor,
        "and it was fought on the floor the run ended on"
    );
    assert!(
        metrics.fights()[..metrics.fights().len() - 1]
            .iter()
            .all(|fight| fight.outcome == Outcome::Won),
        "a run dies once"
    );
}

#[test]
fn a_fight_the_step_cap_interrupts_is_neither_won_nor_lost() {
    // A fight the batch stopped watching is not a fight the player lost, and
    // counting it as one would make the win rate a function of --max-steps.
    let report = episode(60);
    assert_eq!(report.terminal, None, "the cap stopped the run early");
    let metrics = report.metrics.as_ref().expect("the actor collected them");
    let open = metrics
        .fights()
        .iter()
        .filter(|fight| fight.outcome == Outcome::Unresolved)
        .count();
    assert_eq!(
        open, 1,
        "the one fight it was standing in when the cap fell"
    );
    let mut summary = BatchSummary::new(provenance());
    summary.record(&report);
    let json = summary.json();
    assert_eq!(
        json["fights"]["unresolved"], 1,
        "the summary says so rather than folding it into the rate"
    );
    let resolved = f64::from(u32::try_from(json["fights"]["total"].as_u64().unwrap() - 1).unwrap());
    assert!(
        (json["fights"]["win_rate"].as_f64().unwrap()
            - json["fights"]["won"].as_f64().unwrap() / resolved)
            .abs()
            < 1e-9,
        "and the win rate is over the fights that finished: {json}"
    );
}

#[test]
fn a_faulted_episode_is_counted_and_moves_nothing_else() {
    // A faulted run has no honest floor and no honest return — it stopped
    // where the engine tripped, not where the policy took it — so averaging
    // it in would move every number in the file. It is counted and nothing
    // more.
    let report = episode(4000);
    let mut summary = BatchSummary::new(provenance());
    summary.record(&report);
    let clean = summary.json();
    summary.record_fault();
    let after = summary.json();
    assert_eq!(summary.faulted(), 1);
    assert_eq!(after["episodes"]["faulted"], 1);
    assert_eq!(after["episodes"]["played"], clean["episodes"]["played"]);
    assert_eq!(
        after["episodes"]["total"],
        clean["episodes"]["total"].as_u64().unwrap() + 1,
        "the faulted episode is still an episode the batch was asked for"
    );
    for group in ["return", "outcome", "episode_length", "fights"] {
        assert_eq!(
            after[group], clean[group],
            "{group} is unmoved by an episode nobody played"
        );
    }
}

#[test]
fn a_batch_reports_the_rate_it_played_its_episodes_at() {
    // The clock starts where the summary is built, which is where the batch
    // begins, and stops on the last episode handed back.
    let mut summary = BatchSummary::new(provenance());
    let report = episode(400);
    summary.record(&report);
    let json = summary.json();
    let seconds = json["throughput"]["wall_seconds"].as_f64().unwrap();
    let rate = json["throughput"]["runs_per_minute"].as_f64().unwrap();
    assert!(seconds > 0.0, "playing an episode takes time: {json}");
    assert!(rate > 0.0 && rate.is_finite(), "and one episode is a rate");
    assert!(
        (rate * seconds / 60.0 - 1.0).abs() < 1e-9,
        "the rate is re-derivable from the seconds beside it: {json}"
    );
}

#[test]
fn every_won_fight_lands_in_exactly_one_cell_of_the_cost_table() {
    let report = episode(4000);
    let mut summary = BatchSummary::new(provenance());
    summary.record(&report);
    let json = summary.json();
    let counted: u64 = json["hp_lost_by_act_tier"]
        .as_object()
        .expect("the table is an object")
        .iter()
        .filter(|(key, _)| key.ends_with("_n"))
        .map(|(_, value)| value.as_u64().expect("a count"))
        .sum();
    assert_eq!(
        counted,
        json["fights"]["won"].as_u64().unwrap(),
        "no won fight is dropped for want of a column: {json}"
    );
}

#[test]
fn every_scalar_the_trainer_logs_is_a_number_one_level_down() {
    // The trainer walks the file one nesting level deep and logs whatever it
    // finds that is a number. A rate buried two levels down would silently
    // never be logged, and the per-fight records would be walked into if they
    // sat under the group that holds the fight rates.
    let report = episode(4000);
    let mut summary = BatchSummary::new(provenance());
    summary.record(&report);
    let json = summary.json();
    for group in [
        "episodes",
        "throughput",
        "return",
        "return_by_act",
        "outcome",
        "clear_by_act",
        "deaths_by_act",
        "episode_length",
        "fights",
        "hp_lost_by_act_tier",
        "degraded",
        "waste",
        "resolver",
    ] {
        let object = json[group]
            .as_object()
            .unwrap_or_else(|| panic!("{group} is an object"));
        assert!(!object.is_empty(), "{group} carries something");
        for (key, value) in object {
            assert!(
                value.is_number() || (group == "resolver" && key == "kind"),
                "{group}/{key} is a number the trainer can log: {value}"
            );
        }
    }
    // The records are an array at the top level, out of the way of that scan.
    assert!(
        json["fight_records"].is_array(),
        "the per-fight records are their own top-level array"
    );
    assert_eq!(
        u64::try_from(json["fight_records"].as_array().unwrap().len()).unwrap(),
        json["fights"]["total"].as_u64().unwrap(),
        "one record per fight the batch counted"
    );
    // And the batch says what produced it, so a directory of shards is never
    // an orphan.
    assert_eq!(json["provenance"]["run_net"], "ckpts/run");
    assert_eq!(json["provenance"]["combat_net"], "ckpts/combat");
    assert_eq!(json["resolver"]["kind"], "greedy");
}

#[test]
fn the_summary_lands_beside_the_shards_as_one_json_object() {
    let report = episode(400);
    let mut summary = BatchSummary::new(provenance());
    summary.record(&report);
    let directory = std::env::temp_dir().join(format!("alphaspire-summary-{}", std::process::id()));
    let path = directory.join("summary.json");
    summary.write(&path).expect("the summary is written");
    let text = std::fs::read_to_string(&path).expect("and is there");
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("as one JSON object");
    // Compared after a round trip on both sides: the parser reads a float to
    // within an ulp, not exactly, so the in-memory value and the one read
    // back can differ in the last digit for some batches and agree for
    // others.
    let expected: serde_json::Value =
        serde_json::from_str(&summary.json().to_string()).expect("the summary re-parses");
    assert_eq!(parsed, expected);
    assert!(text.ends_with('\n'), "one object, newline-terminated");
    std::fs::remove_dir_all(directory).expect("test output is removed");
}
