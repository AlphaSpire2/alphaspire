//! `match`: one paired gate with a level. Two arms play identical games and
//! the verdict is a paired table, a sign test, and an exit code a promotion
//! script reads — 0 only on a measured candidate win.

use std::path::PathBuf;

use alphaspire::search::SearchConfig;
use clap::{Args, Subcommand};

use super::{
    BatchArgs, ResolverArgs, SelectionPolicy, encoder, fail, playable_character,
    refuse_rollout_policy,
};

/// What is on trial: two combat checkpoints, two run checkpoints, or one
/// checkpoint against itself.
#[derive(Subcommand)]
pub enum Level {
    /// Two combat checkpoints (or one against uniform priors), each
    /// searching its fights over identical games. The promotion gate of the
    /// combat loop
    Combat(CombatArgs),
    /// Two *run* checkpoints, each playing whole runs by argmax against one
    /// frozen combat resolver over identical games. The promotion gate of
    /// the run-level loop
    Macro(MacroArgs),
    /// One checkpoint searched against itself unsearched, over identical
    /// games: exit 0 where the search measurably improves on the net it is
    /// guided by — the premise expert iteration rests on and never checks
    #[command(name = "self")]
    Improvement(SelfArgs),
}

/// The budget a searching arm searches its fights at.
#[derive(Args)]
pub struct GateSearchArgs {
    /// MCTS rollouts per decision
    #[arg(long, default_value_t = 64)]
    iterations: u32,
    /// Maximum depth of each rollout
    #[arg(long, default_value_t = 30)]
    rollout_depth: u32,
    /// Tree policy: UCT, or Gumbel-top-k with sequential halving at the root
    /// and completed-Q selection inside the tree
    #[arg(long, value_enum, default_value = "gumbel")]
    selection: SelectionPolicy,
    /// How many root actions Gumbel spreads its budget over
    #[arg(long, default_value_t = 16)]
    considered: usize,
}

impl GateSearchArgs {
    fn config(&self) -> SearchConfig {
        SearchConfig {
            iterations: self.iterations,
            rollout_depth: self.rollout_depth,
            temperature: 0.5,
        }
    }
}

#[derive(Args)]
pub struct CombatArgs {
    #[command(flatten)]
    batch: BatchArgs,
    #[command(flatten)]
    search: GateSearchArgs,
    /// The checkpoint on trial: the base path of a trainer export
    #[arg(long)]
    candidate_net: PathBuf,
    /// The checkpoint to beat; uniform priors when absent
    #[arg(long)]
    baseline_net: Option<PathBuf>,
}

/// The policy-improvement gate: the checkpoint under test, and the budget
/// the teacher searches at.
///
/// One `--net`, not two. Both arms are the same weights — the question is
/// whether searching them beats reading them — so a second checkpoint would
/// be answering a different question, which is what `match combat` is for.
#[derive(Args)]
pub struct SelfArgs {
    #[command(flatten)]
    batch: BatchArgs,
    /// The budget the *teacher* arm searches at. The budget a generation
    /// writes its labels at is the budget to gate, so it defaults to the
    /// generation default rather than the resolver's
    #[command(flatten)]
    search: GateSearchArgs,
    /// The checkpoint on trial, played both ways
    #[arg(long)]
    net: PathBuf,
}

/// A macro match: the two run checkpoints on trial, and the environment held
/// identical between them.
#[derive(Args)]
pub struct MacroArgs {
    #[command(flatten)]
    batch: BatchArgs,
    #[command(flatten)]
    fights: ResolverArgs,
    /// The run checkpoint on trial: the base path of a trainer export,
    /// refused unless its provenance says `scope: macro` and names a run
    /// reward this build can pay. A greedy arm pays no reward, so the two
    /// arms need not have been trained under the same one
    #[arg(long)]
    candidate_net: PathBuf,
    /// The run checkpoint to beat. Required, unlike the combat gate's: a
    /// combat match's netless arm still searches, so uniform priors are a
    /// real policy to beat, while a greedy macro arm over uniform priors is
    /// "always the first action on the screen" and no promotion decision
    /// should be made against it
    #[arg(long)]
    baseline_net: PathBuf,
    /// The frozen combat checkpoint *both* arms resolve their fights with.
    /// One checkpoint and one resolver for the pair, because the resolver is
    /// the environment: two arms resolving differently would measure the
    /// resolvers and the run checkpoints only incidentally
    #[arg(long)]
    combat_net: PathBuf,
}

/// Plays the gate at its level and speaks the verdict.
pub fn command(level: &Level) -> ! {
    match level {
        Level::Combat(args) => combat(args),
        Level::Macro(args) => r#macro(args),
        Level::Improvement(args) => improvement(args),
    }
}

/// Plays a combat match: two checkpoints (or a checkpoint against uniform
/// priors) over identical games, with a paired verdict.
fn combat(args: &CombatArgs) -> ! {
    let registry = sts2_content::standard_registry();
    let character = playable_character(&registry, &args.batch.character);
    let preset = args.batch.preset();
    let encoder = std::sync::Arc::new(encoder(&registry));
    let load = |base: &PathBuf| {
        let net = alphaspire::net::PolicyValueNet::load(base, std::sync::Arc::clone(&encoder))
            .unwrap_or_else(|error| fail(&error.to_string()));
        if let Some(warning) = net.foreign_to(&character) {
            eprintln!("alphaspire: {}: {warning}", base.display());
        }
        std::sync::Arc::new(net)
    };
    let candidate: alphaspire::matchup::Arm = Some(load(&args.candidate_net));
    let baseline: alphaspire::matchup::Arm = args.baseline_net.as_ref().map(load);
    let selection = || args.search.selection.build(args.search.considered);
    let rollout = || args.batch.policy.build();
    let config = alphaspire::matchup::MatchConfig {
        batch: args.batch.plain(&preset, &character),
        search: args.search.config(),
        selection: &selection,
        rollout: &rollout,
    };
    speak(
        &alphaspire::matchup::play_match(&config, &baseline, &candidate),
        Verdict::MATCH,
    );
}

/// Grades a checkpoint against itself: its fights searched, against its
/// fights read straight off the policy head, over the same seeds.
///
/// Expert iteration's own premise, made checkable. A generation trains the
/// net toward the search's improved policy because the search is supposed to
/// improve on the net; nothing in the loop has ever tested that, and a
/// generation whose teacher was no better than its student is
/// indistinguishable, downstream, from one whose teacher was — the promotion
/// match grades two checkpoints that were both taught by it.
///
/// It shares everything with the combat gate but the arms: the same seeds,
/// the same out-of-combat rollout on both sides, the same paired sign test,
/// and the same exit code, so a generation script gates on it the way it
/// already gates on a promotion.
fn improvement(args: &SelfArgs) -> ! {
    let registry = sts2_content::standard_registry();
    let character = playable_character(&registry, &args.batch.character);
    let preset = args.batch.preset();
    let net =
        alphaspire::net::PolicyValueNet::load(&args.net, std::sync::Arc::new(encoder(&registry)))
            .unwrap_or_else(|error| fail(&error.to_string()));
    if let Some(warning) = net.foreign_to(&character) {
        eprintln!("alphaspire: {}: {warning}", args.net.display());
    }
    let net = std::sync::Arc::new(net);
    let selection = || args.search.selection.build(args.search.considered);
    let rollout = || args.batch.policy.build();
    let config = alphaspire::matchup::MatchConfig {
        batch: args.batch.plain(&preset, &character),
        search: args.search.config(),
        selection: &selection,
        rollout: &rollout,
    };
    speak(
        &alphaspire::matchup::play_improvement(&config, &net),
        Verdict::IMPROVEMENT,
    );
}

/// Grades two *run* checkpoints: each plays whole runs by argmax against one
/// frozen combat resolver, over the same seeds, and the verdict is the one a
/// combat match is graded by.
///
/// A level of `match` rather than a flag, because the two gates take
/// different required flags and `--candidate-net` would otherwise mean two
/// different kinds of checkpoint depending on another flag. What they do
/// share — the paired unit, the sign test, `candidate_wins`, and the exit
/// code a promotion script reads — they share by construction: this calls
/// exactly the statistics the combat gate calls, on arms built the only way
/// that differ.
fn r#macro(args: &MacroArgs) -> ! {
    refuse_rollout_policy(args.batch.policy);
    let resolver = args.fights.settled();
    let registry = sts2_content::standard_registry();
    let character = playable_character(&registry, &args.batch.character);
    let preset = args.batch.preset();
    let encoder = std::sync::Arc::new(encoder(&registry));
    let warn = |base: &PathBuf, net: &alphaspire::net::PolicyValueNet| {
        if let Some(warning) = net.foreign_to(&character) {
            eprintln!("alphaspire: {}: {warning}", base.display());
        }
    };
    let load_run = |base: &PathBuf, flag: &str| {
        let net = alphaspire::reward::checkpoint_terms(base)
            .and_then(|(semantics, _)| {
                alphaspire::net::PolicyValueNet::load_run_priced(
                    base,
                    std::sync::Arc::clone(&encoder),
                    &semantics,
                )
                .map_err(|error| error.to_string())
            })
            .unwrap_or_else(|error| {
                fail(&format!(
                    "{flag} {}: {error}: a macro match grades run checkpoints, whose \
                     critics predict a run's return",
                    base.display()
                ))
            });
        warn(base, &net);
        std::sync::Arc::new(net)
    };
    let candidate = load_run(&args.candidate_net, "--candidate-net");
    let baseline = load_run(&args.baseline_net, "--baseline-net");
    let combat_net = {
        let net = alphaspire::net::PolicyValueNet::load(
            &args.combat_net,
            std::sync::Arc::clone(&encoder),
        )
        .unwrap_or_else(|error| {
            fail(&format!(
                "--combat-net {}: {error}: the resolver plays fights, so it takes the \
                         combat checkpoint",
                args.combat_net.display()
            ))
        });
        warn(&args.combat_net, &net);
        std::sync::Arc::new(net)
    };
    let config = alphaspire::matchup::MacroMatchConfig {
        batch: args.batch.plain(&preset, &character),
        combat_net,
        resolver,
    };
    speak(
        &alphaspire::matchup::play_macro_match(&config, &baseline, &candidate),
        Verdict::MATCH,
    );
}

/// What a paired verdict calls its two arms and its two outcomes. The
/// statistics are one set of statistics; only the words differ, and a gate
/// that printed "candidate wins" for a question nobody asked about a
/// candidate would be read wrong.
#[derive(Clone, Copy)]
struct Verdict {
    baseline: &'static str,
    candidate: &'static str,
    won: &'static str,
    lost: &'static str,
}

impl Verdict {
    const MATCH: Self = Self {
        baseline: "baseline",
        candidate: "candidate",
        won: "candidate wins",
        lost: "no measured win",
    };
    const IMPROVEMENT: Self = Self {
        baseline: "student",
        candidate: "teacher",
        won: "the search improves on the net",
        lost: "no measured improvement: this generation's labels are not a teacher",
    };
}

/// The paired report, spoken, and the exit code a promotion script reads.
fn speak(report: &alphaspire::matchup::MatchReport, words: Verdict) -> ! {
    for line in &report.dropped {
        eprintln!("alphaspire: dropped pair: {line}");
    }
    println!("seed        {:>8} {:>9}", words.baseline, words.candidate);
    for pair in &report.pairs {
        let word = match pair.candidate.cmp(&pair.baseline) {
            std::cmp::Ordering::Greater => "up",
            std::cmp::Ordering::Less => "down",
            std::cmp::Ordering::Equal => "tie",
        };
        println!(
            "{:<11} {:>8} {:>9}  {word}",
            pair.seed, pair.baseline, pair.candidate
        );
    }
    for (label, outcomes, degradations) in [
        (
            words.baseline,
            &report.baseline_outcomes,
            report.baseline_degradations,
        ),
        (
            words.candidate,
            &report.candidate_outcomes,
            report.candidate_degradations,
        ),
    ] {
        println!("{label}: {}", outcomes.summary());
        // A promotion decision made over arbitrary picks should say so on the
        // same screen as the verdict. An arm whose policy counts nothing says
        // nothing here rather than zero.
        if let Some(degradations) = degradations {
            println!("  degraded: {degradations}");
        }
        for line in outcomes.death_table(10) {
            println!("  {line}");
        }
    }
    let (baseline_mean, candidate_mean) = report.means();
    println!(
        "mean floor {baseline_mean:.2} vs {candidate_mean:.2} | {} up {} tie {} down | sign test p = {:.4}",
        report.ups(),
        report.ties(),
        report.downs(),
        report.p_value(),
    );
    if report.candidate_wins() {
        println!("verdict: {}", words.won);
        std::process::exit(0);
    }
    println!("verdict: {}", words.lost);
    std::process::exit(1);
}
