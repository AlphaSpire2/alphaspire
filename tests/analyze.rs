//! The analysis of a recorded run: every applied decision becomes a line,
//! the recorded move always has a searched value beside the best one, and
//! the macro lane prices plans and charges the reward to the decision that
//! bought it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use alphaspire::actor::TierBudget;
use alphaspire::analyze::{
    ANALYSIS_FORMAT, Analysis, Budgets, Lane, Nets, PlayedStatus, RunNet, Settings, analyze_trace,
};
use alphaspire::encoding::PolicyEncoder;
use alphaspire::net::PolicyValueNet;
use alphaspire::objective::Coverage;
use alphaspire::policy::UniformRandom;
use sts2_rng::MegaRandom;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn encoder() -> Arc<PolicyEncoder> {
    let registry = sts2_content::standard_registry();
    Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ))
}

fn nets(run: Option<RunNet>) -> Nets {
    let stem = fixtures().join("tiny");
    Nets {
        combat: Arc::new(PolicyValueNet::load(&stem, encoder()).expect("the fixture net loads")),
        combat_stem: stem,
        run,
    }
}

/// Budgets small enough for a test and shaped to make the pin matter: one
/// candidate per root draw, so nothing but the pin puts a low-prior move in.
fn settings(net_only: bool) -> Settings {
    let tiny = TierBudget {
        iterations: 4,
        considered: 1,
    };
    Settings {
        budgets: Budgets {
            hallway: tiny,
            elite: tiny,
            boss: tiny,
            deep: TierBudget {
                iterations: 8,
                considered: 2,
            },
            deep_top: 2,
            deep_cutoff: 0.1,
            min_visits: 1,
        },
        net_only,
        playouts: true,
        lookahead: true,
        full_observation: false,
        analysis_seed: 7,
        return_discount: 1.0,
        threads: 2,
    }
}

/// A script this crate wrote: a random walk of `steps` engine actions on
/// `seed`, on a temporary path.
fn scripted_run(name: &str, seed: &str, steps: usize) -> PathBuf {
    let mut coverage = Coverage::default();
    let mut rng = MegaRandom::new(21);
    let mut policy = UniformRandom;
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let report = alphaspire::selfplay::play_run(
        seed,
        &character,
        0,
        &mut policy,
        &mut rng,
        &mut coverage,
        steps,
    )
    .unwrap();
    let script = report.script.expect("a random walk stays scriptable");
    let directory =
        std::env::temp_dir().join(format!("alphaspire-analyze-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join(format!("{seed}.sts2pgn"));
    std::fs::write(&path, script).unwrap();
    path
}

/// A run-scoped checkpoint made of the combat fixture: same graph, the
/// provenance a PPO export under the elite-and-gold reward carries.
fn run_checkpoint(name: &str) -> RunNet {
    let directory =
        std::env::temp_dir().join(format!("alphaspire-analyze-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let base = directory.join("run");
    std::fs::copy(fixtures().join("tiny.onnx"), base.with_extension("onnx")).unwrap();
    let mut provenance: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("tiny.json")).unwrap())
            .unwrap();
    let object = provenance.as_object_mut().unwrap();
    object.insert("scope".into(), "macro".into());
    object.insert(
        "value_semantics".into(),
        "run-return-v2;elite=0.25;relic=0;gold=0.003".into(),
    );
    std::fs::write(base.with_extension("json"), format!("{provenance}\n")).unwrap();
    RunNet::load(&base, encoder()).expect("the fabricated run checkpoint loads")
}

#[test]
fn a_script_walks_to_its_end_and_every_decision_is_a_line() {
    let trace = scripted_run("walk", "NLD6VZXP94", 200);
    let analysis = analyze_trace(&trace, &nets(None), settings(true)).unwrap();
    let summary = &analysis.summary;
    assert_eq!(summary.analysis_format, ANALYSIS_FORMAT);
    assert!(
        summary.coverage.divergence.is_none(),
        "a script this crate wrote replays: {:?}",
        summary.coverage.divergence
    );
    assert!(
        !summary.coverage.verified,
        "a script carries no observations"
    );
    assert_eq!(summary.coverage.decisions, analysis.lines.len());
    assert!(
        summary.coverage.applied >= analysis.lines.len(),
        "a decision is at least one applied action"
    );
    assert!(
        summary.coverage.claimed > 0,
        "the walk claimed the free lines on its reward screens: {:?}",
        summary.coverage
    );
    assert_eq!(
        summary.coverage.applied,
        summary.coverage.decisions
            + summary.coverage.claimed
            + analysis
                .lines
                .iter()
                .filter(|line| line.played_display.contains("then"))
                .count(),
        "every applied action is a decision, a free claim, or the opener of a collapse"
    );
    assert!(
        analysis.lines.iter().any(|line| line.lane == Lane::Combat)
            && analysis.lines.iter().any(|line| line.lane == Lane::Macro),
        "a walk into a fight has both lanes"
    );
    assert!(
        analysis
            .lines
            .windows(2)
            .all(|pair| pair[0].seq < pair[1].seq),
        "lines follow the trace"
    );
    for line in &analysis.lines {
        let state = serde_json::to_value(&line.state).unwrap();
        assert!(state["draw_pile"].is_array());
        assert!(state["discard_pile"].is_array());
        assert!(state["exhaust_pile"].is_array());
        if line.forced {
            assert!(
                line.candidates.len() < 2,
                "a forced decision offered one plan"
            );
            assert!(line.v.is_none() && line.shallow.is_none());
        } else if line.lane == Lane::Combat {
            assert!(
                line.v.is_some(),
                "seq {}: the net priced the state",
                line.seq
            );
            assert!(line.shallow.is_none(), "net-only searches nothing");
        }
        assert!(!line.played_display.is_empty());
    }
    assert!(!summary.fights.is_empty(), "the walk fought");
    assert!(
        summary.fights.iter().all(|fight| fight.playout.is_none()),
        "net-only plays nothing out"
    );
    // The two files are written and the line file has one line per decision.
    let directory = trace.parent().unwrap().join("out");
    let (summary_path, lines_path) = analysis.write(&directory, "walk").unwrap();
    let written: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(summary_path).unwrap()).unwrap();
    assert_eq!(written["analysis_format"], ANALYSIS_FORMAT);
    assert_eq!(
        std::fs::read_to_string(lines_path).unwrap().lines().count(),
        analysis.lines.len()
    );
}

#[test]
fn the_recorded_move_always_has_a_searched_value() {
    let trace = scripted_run("pinned", "NLD6VZXP94", 200);
    let analysis = analyze_trace(&trace, &nets(None), settings(false)).unwrap();
    let searched: Vec<_> = analysis
        .lines
        .iter()
        .filter(|line| line.shallow.is_some())
        .collect();
    assert!(!searched.is_empty(), "the fights were searched");
    for line in &searched {
        let Some(played) = line.played else {
            continue;
        };
        let table = line.shallow.as_ref().unwrap();
        let candidate = &table.candidates[played];
        // One candidate per draw, so a move the prior ranked second would
        // never be visited without the pin.
        assert!(
            candidate.visits >= 1 && candidate.q_completed.is_some(),
            "seq {}: the recorded move {} was pinned into the search",
            line.seq,
            line.played_display
        );
        assert!(line.delta.is_some(), "seq {}: the gap is priced", line.seq);
        assert!(line.best.is_some());
    }
    let deep: Vec<_> = analysis
        .lines
        .iter()
        .filter(|line| line.deep.is_some())
        .collect();
    assert!(!deep.is_empty(), "the widest gaps were re-searched");
    for line in &deep {
        let table = line.deep.as_ref().unwrap();
        assert_eq!(table.table.budget, settings(false).budgets.deep);
        assert!(table.playout_after_played.is_some());
    }
    for fight in &analysis.summary.fights {
        let playout = fight.playout.as_ref().expect("every fight is played out");
        assert_eq!(playout.budget, settings(false).budgets.hallway);
        assert!(fight.human.turns >= 1 || !fight.human.won);
    }
    // Reproducible from the analysis seed alone.
    let again = analyze_trace(&trace, &nets(None), settings(false)).unwrap();
    let numbers = |analysis: &Analysis| -> Vec<(u64, Option<f64>, Option<f64>)> {
        analysis
            .lines
            .iter()
            .map(|line| {
                (
                    line.seq,
                    line.delta,
                    line.shallow.as_ref().and_then(|table| table.r),
                )
            })
            .collect()
    };
    assert_eq!(numbers(&analysis), numbers(&again));
}

#[test]
fn the_worker_count_changes_nothing_the_analysis_writes() {
    let trace = scripted_run("threads", "NLD6VZXP94", 200);
    let mut alone = settings(false);
    alone.threads = 1;
    let mut pooled = settings(false);
    pooled.threads = 4;
    let alone = analyze_trace(&trace, &nets(None), alone).unwrap();
    let pooled = analyze_trace(&trace, &nets(None), pooled).unwrap();
    assert!(
        alone.lines.iter().any(|line| line.deep.is_some())
            && alone
                .summary
                .fights
                .iter()
                .all(|fight| fight.playout.is_some()),
        "both parallel phases ran"
    );
    assert_eq!(
        serde_json::to_value(&alone.lines).unwrap(),
        serde_json::to_value(&pooled.lines).unwrap(),
        "the deep pass is seeded in trace order, not by worker"
    );
    assert_eq!(
        serde_json::to_value(&alone.summary.fights).unwrap(),
        serde_json::to_value(&pooled.summary.fights).unwrap(),
        "the fight playouts are seeded where the walk reached them"
    );
}

#[test]
fn the_macro_lane_prices_plans_and_charges_the_reward() {
    let trace = scripted_run("macro", "NLD6VZXP94", 200);
    let run = run_checkpoint("macro");
    let analysis = analyze_trace(&trace, &nets(Some(run)), settings(true)).unwrap();
    let priced: Vec<_> = analysis
        .lines
        .iter()
        .filter(|line| line.lane == Lane::Macro && !line.forced)
        .collect();
    assert!(!priced.is_empty());
    for line in &priced {
        assert!(line.v_m.is_some(), "seq {}: the critic priced it", line.seq);
        assert!(
            line.g.is_some(),
            "seq {}: the realised return is settled",
            line.seq
        );
        let mass: f64 = line.candidates.iter().filter_map(|c| c.p).sum();
        assert!(
            (mass - 1.0).abs() < 1e-3,
            "seq {}: a policy sums to one",
            line.seq
        );
        assert!(line.preferred.is_some());
        if matches!(line.played_status, PlayedStatus::OnMenu) {
            assert!(line.played.is_some());
        }
        if line.state.screen == "map_navigation" {
            assert!(
                line.candidates.iter().all(|c| c.v_after.is_none()),
                "a room entered is never looked into"
            );
        }
    }
    assert!(
        priced
            .iter()
            .any(|line| line.candidates.iter().any(|c| c.v_after.is_some())),
        "some screen's outcomes were fully determined"
    );
    let charged: f64 = analysis.lines.iter().filter_map(|line| line.reward).sum();
    let total = analysis
        .summary
        .macro_returns
        .as_ref()
        .unwrap()
        .total_reward;
    assert!(
        (charged - total).abs() < 1e-9,
        "every reward paid was charged to a decision: {charged} against {total}"
    );
}

/// `trace` cut off where its first fight three actions long ends, and with
/// `reload` a quit and a reload spliced into that fight: the map move into
/// it is a save, two actions are played, then `run.resume`, then the fight
/// from its start again. It stops at the fight's end because a reload
/// restarts the reward-set ids the script's later records name.
fn through_a_fight(trace: &Path, reload: bool) -> PathBuf {
    let script = std::fs::read_to_string(trace).unwrap();
    let is_record = |line: &&str| {
        line.split_once(' ')
            .is_some_and(|(head, _)| head.parse::<u64>().is_ok())
    };
    let in_fight = |line: &str| line.split(' ').nth(1).unwrap().starts_with("combat.");
    let head: Vec<&str> = script.lines().take_while(|line| !is_record(line)).collect();
    let records: Vec<&str> = script.lines().filter(is_record).collect();
    let save = (0..records.len() - 3)
        .find(|&index| {
            records[index].contains(" map.choose ")
                && (1..=3).all(|step| in_fight(records[index + step]))
        })
        .expect("the walk entered a fight three actions long");
    let end = (save + 1..records.len())
        .find(|&index| !in_fight(records[index]))
        .unwrap_or(records.len());
    let resume = "0 run.resume {\"actor\":null,\"data\":{}}";
    let mut kept: Vec<&str> = records[..end].to_vec();
    if reload {
        kept.splice(
            save + 3..save + 3,
            std::iter::once(resume).chain(records[save + 1..save + 3].iter().copied()),
        );
    }
    let mut lines: Vec<String> = head.iter().map(|&line| line.to_owned()).collect();
    for (index, record) in kept.iter().enumerate() {
        let (_, rest) = record.split_once(' ').unwrap();
        lines.push(format!("{} {rest}", index + 1));
    }
    let name = if reload {
        "reloaded.sts2pgn"
    } else {
        "plain.sts2pgn"
    };
    let path = trace.with_file_name(name);
    std::fs::write(
        &path,
        lines.join(
            "
",
        ),
    )
    .unwrap();
    path
}

#[test]
fn a_reload_is_analysed_as_if_the_rewound_play_never_happened() {
    let trace = scripted_run("reload", "NLD6VZXP94", 200);
    let plain =
        analyze_trace(&through_a_fight(&trace, false), &nets(None), settings(true)).unwrap();
    let reloaded =
        analyze_trace(&through_a_fight(&trace, true), &nets(None), settings(true)).unwrap();
    let coverage = &reloaded.summary.coverage;
    assert!(coverage.divergence.is_none(), "{:?}", coverage.divergence);
    assert_eq!((coverage.resumes, coverage.rewound), (1, 2));
    assert_eq!(coverage.applied, plain.summary.coverage.applied);
    assert_eq!(coverage.decisions, plain.summary.coverage.decisions);
    let played = |analysis: &Analysis| -> Vec<String> {
        analysis
            .lines
            .iter()
            .map(|line| line.played_display.clone())
            .collect()
    };
    assert_eq!(played(&reloaded), played(&plain));
    assert_eq!(reloaded.summary.fights.len(), plain.summary.fights.len());
    for (left, right) in reloaded.summary.fights.iter().zip(&plain.summary.fights) {
        assert_eq!(left.human.hp_out, right.human.hp_out);
        assert_eq!(left.human.turns, right.human.turns);
    }
}
