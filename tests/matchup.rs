//! The match: identical games in both arms, a paired verdict, and a sign
//! test whose arithmetic is pinned — for the combat gate and for the macro
//! gate beside it, which grades run checkpoints on the same statistics.

use std::path::Path;
use std::sync::Arc;

use alphaspire::actor::Resolver;
use alphaspire::matchup::{
    MacroMatchConfig, MatchConfig, MatchReport, Pair, play_improvement, play_macro_match,
    play_match, sign_test,
};
use alphaspire::net::PolicyValueNet;
use alphaspire::search::{Gumbel, SearchConfig, Selection};
use sts2_core::UnlockPresetManifest;

#[test]
fn the_sign_test_is_the_binomial_tail() {
    assert!((sign_test(0, 0) - 1.0).abs() < 1e-12, "no evidence, p = 1");
    assert!(
        (sign_test(1, 0) - 0.5).abs() < 1e-12,
        "one up out of one is a coin flip"
    );
    assert!(
        (sign_test(2, 0) - 0.25).abs() < 1e-12,
        "two of two: a quarter"
    );
    // Yesterday's wide gate, by hand: 19 ups of 23 non-ties.
    let p = sign_test(19, 4);
    assert!(
        (0.001..0.002).contains(&p),
        "nineteen of twenty-three is about a tenth of a percent: {p}"
    );
    assert!(
        (sign_test(4, 19) - (1.0 - sign_test(20, 3))).abs() < 1e-9,
        "the tail complements its mirror"
    );
}

#[test]
fn a_report_reads_its_pairs() {
    let pair = |baseline, candidate| Pair {
        seed: "SEED".into(),
        baseline,
        candidate,
    };
    let report = MatchReport {
        pairs: vec![pair(5, 9), pair(7, 7), pair(3, 8), pair(9, 4)],
        ..MatchReport::default()
    };
    assert_eq!(
        (report.ups(), report.ties(), report.downs()),
        (2, 1, 1),
        "the ledger"
    );
    let (baseline, candidate) = report.means();
    assert!((baseline - 6.0).abs() < 1e-12 && (candidate - 7.0).abs() < 1e-12);
    assert!(
        !report.candidate_wins(),
        "two of three non-ties is no measured win"
    );
}

#[test]
fn both_arms_of_a_match_play_the_same_games() {
    // A tiny real match: the fixture checkpoint against uniform priors. The
    // fixture net is noise, so no verdict is expected of it — what is pinned
    // is the pairing: every pair carries one seed, played by both arms.
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let selection = || Box::new(Gumbel::default()) as Box<dyn Selection>;
    let config = MatchConfig {
        batch: alphaspire::selfplay::Batch {
            preset: &preset,
            character: &character,
            ascension: 0,
            analysis_seed: 6,
            seed: None,
            runs: 2,
            max_steps: 15,
            jobs: 2,
            harvest: None,
            force_wins: None,
        },
        search: SearchConfig {
            iterations: 6,
            rollout_depth: 4,
            temperature: 0.5,
        },
        selection: &selection,
        rollout: &|| {
            Box::new(alphaspire::policy::UniformRandom)
                as Box<dyn alphaspire::policy::RolloutPolicy>
        },
    };
    let registry = sts2_content::standard_registry();
    let net = Arc::new(
        PolicyValueNet::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny"),
            Arc::new(alphaspire::encoding::PolicyEncoder::new(
                alphaspire::encoding::standard_vocabulary(&registry),
                &registry,
            )),
        )
        .unwrap(),
    );
    let report = play_match(&config, &None, &Some(net));
    assert_eq!(report.pairs.len(), 2, "two games, two pairs");
    assert!(report.dropped.is_empty(), "nothing was lost");
    let expected: Vec<String> = (0..2)
        .map(|index| alphaspire::selfplay::run_seeds(6, index).0)
        .collect();
    let played: Vec<&String> = report.pairs.iter().map(|pair| &pair.seed).collect();
    assert_eq!(
        played,
        expected.iter().collect::<Vec<_>>(),
        "the games are the batch's derived games, in order, in both arms"
    );
}

/// The fixture checkpoint, loaded as the combat net a resolver runs on.
fn fixture() -> Arc<PolicyValueNet> {
    let registry = sts2_content::standard_registry();
    Arc::new(
        PolicyValueNet::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny"),
            Arc::new(alphaspire::encoding::PolicyEncoder::new(
                alphaspire::encoding::standard_vocabulary(&registry),
                &registry,
            )),
        )
        .unwrap(),
    )
}

#[test]
fn a_macro_match_pairs_two_run_checkpoints_over_the_same_games() {
    // The gate for a run policy. Both arms play whole runs — greedily,
    // because that is what a deployed checkpoint does — against one frozen
    // resolver on one frozen combat checkpoint, so a pair's difference is
    // attributable to the run checkpoints and to nothing else.
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let net = fixture();
    let config = MacroMatchConfig {
        batch: alphaspire::selfplay::Batch {
            preset: &preset,
            character: &character,
            ascension: 0,
            analysis_seed: 6,
            seed: None,
            runs: 2,
            max_steps: 60,
            jobs: 2,
            harvest: None,
            force_wins: None,
        },
        combat_net: Arc::clone(&net),
        resolver: Resolver::Greedy,
    };
    let report = play_macro_match(&config, &net, &net);
    assert_eq!(report.pairs.len(), 2, "two games, two pairs");
    assert!(report.dropped.is_empty(), "nothing was lost");
    let expected: Vec<String> = (0..2)
        .map(|index| alphaspire::selfplay::run_seeds(6, index).0)
        .collect();
    let played: Vec<&String> = report.pairs.iter().map(|pair| &pair.seed).collect();
    assert_eq!(
        played,
        expected.iter().collect::<Vec<_>>(),
        "the games are the batch's derived games, in order, in both arms"
    );
}

#[test]
fn a_macro_match_carries_each_arms_degraded_counts() {
    // A promotion decision made over arbitrary picks says so on the same
    // screen as the verdict, which needs the counts to reach the report. The
    // arms here are one checkpoint on both sides, so they degrade the same
    // decisions the same number of times.
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let net = fixture();
    let config = MacroMatchConfig {
        batch: alphaspire::selfplay::Batch {
            preset: &preset,
            character: &character,
            ascension: 0,
            analysis_seed: 6,
            seed: None,
            runs: 2,
            max_steps: 60,
            jobs: 2,
            harvest: None,
            force_wins: None,
        },
        combat_net: Arc::clone(&net),
        resolver: Resolver::Greedy,
    };
    let report = play_macro_match(&config, &net, &net);
    let baseline = report
        .baseline_degradations
        .expect("a macro arm reports what it degraded");
    let candidate = report.candidate_degradations.expect("both arms do");
    assert_eq!(baseline, candidate, "one checkpoint, two identical arms");
}

#[test]
fn a_greedy_macro_arm_against_itself_measures_nothing() {
    // Argmax rather than sampling is what makes this true, and it is the
    // property the gate rests on: the same checkpoint on both sides plays the
    // same run twice, so every pair ties and the verdict is no measured win.
    // A sampled arm would let two copies of one checkpoint differ by their
    // draws, and the gate would report noise as signal.
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let net = fixture();
    let config = MacroMatchConfig {
        batch: alphaspire::selfplay::Batch {
            preset: &preset,
            character: &character,
            ascension: 0,
            analysis_seed: 11,
            seed: None,
            runs: 3,
            max_steps: 60,
            jobs: 3,
            harvest: None,
            force_wins: None,
        },
        combat_net: Arc::clone(&net),
        resolver: Resolver::Greedy,
    };
    let report = play_macro_match(&config, &net, &net);
    assert_eq!(report.ties(), report.pairs.len(), "every pair is a tie");
    assert_eq!((report.ups(), report.downs()), (0, 0));
    assert!(
        !report.candidate_wins(),
        "and no measured win, which is the exit-1 side of the gate"
    );
    assert!(
        (report.p_value() - 1.0).abs() < 1e-12,
        "no informative pairs is no evidence"
    );
}

#[test]
fn the_improvement_gate_pairs_a_checkpoint_against_itself_unsearched() {
    // Expert iteration's premise as a gate: one checkpoint, its fights
    // searched in one arm and read straight off its policy head in the
    // other, over the same games. The fixture net is noise so no verdict is
    // expected of it — what is pinned is that the arms differ in the search
    // alone and still pair seed for seed.
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    let selection = || Box::new(Gumbel::default()) as Box<dyn Selection>;
    let config = MatchConfig {
        batch: alphaspire::selfplay::Batch {
            preset: &preset,
            character: &character,
            ascension: 0,
            analysis_seed: 6,
            seed: None,
            runs: 2,
            max_steps: 15,
            jobs: 2,
            harvest: None,
            force_wins: None,
        },
        search: SearchConfig {
            iterations: 6,
            rollout_depth: 4,
            temperature: 0.5,
        },
        selection: &selection,
        rollout: &|| {
            Box::new(alphaspire::policy::UniformRandom)
                as Box<dyn alphaspire::policy::RolloutPolicy>
        },
    };
    let report = play_improvement(&config, &fixture());
    assert_eq!(report.pairs.len(), 2, "two games, two pairs");
    assert!(report.dropped.is_empty(), "nothing was lost");
    let expected: Vec<String> = (0..2)
        .map(|index| alphaspire::selfplay::run_seeds(6, index).0)
        .collect();
    let played: Vec<&String> = report.pairs.iter().map(|pair| &pair.seed).collect();
    assert_eq!(
        played,
        expected.iter().collect::<Vec<_>>(),
        "student and teacher play the batch's derived games, in order"
    );
}
