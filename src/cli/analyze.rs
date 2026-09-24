//! `analyze`: recorded runs replayed through the simulator and every
//! decision evaluated — or, without a net, summarized.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use clap::Args;

use super::{die, encoder, fail};

#[derive(Args)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each switch turns one independent stage of the analysis off"
)]
pub struct AnalyzeArgs {
    /// The traces to analyse
    #[arg(required = true, value_name = "TRACE")]
    traces: Vec<PathBuf>,
    /// The combat checkpoint that searches every in-fight decision and plays
    /// the fights out: the base path of a trainer export
    #[arg(long, required_unless_present = "no_net")]
    combat_net: Option<PathBuf>,
    /// The run checkpoint that prices the out-of-combat decisions, reading
    /// the reward it was trained under off its own provenance. Without it
    /// the macro decisions are walked but not priced
    #[arg(long)]
    run_net: Option<PathBuf>,
    /// Load no net and evaluate nothing: replay each trace and write its
    /// outcome, HP curve, room counts and act-end resources as
    /// `<stem>.summary.json` instead of an analysis
    #[arg(
        long,
        conflicts_with_all = [
            "combat_net", "run_net", "resolver_iterations", "resolver_considered",
            "resolver_elite_iterations", "resolver_elite_considered",
            "resolver_boss_iterations", "resolver_boss_considered", "deep_iterations",
            "deep_considered", "deep_top", "deep_cutoff", "min_visits", "net_only",
            "no_playout", "no_macro_lookahead", "full_observation", "analysis_seed",
            "return_discount", "threads",
        ]
    )]
    no_net: bool,
    /// Search iterations per decision, and per playout decision, inside
    /// hallway fights
    #[arg(long, default_value_t = 128, value_name = "N")]
    resolver_iterations: u32,
    /// Root actions the hallway budget is spread over
    #[arg(long, default_value_t = 16, value_name = "N")]
    resolver_considered: usize,
    /// The same two knobs inside elite fights
    #[arg(long, default_value_t = 256, value_name = "N")]
    resolver_elite_iterations: u32,
    #[arg(long, default_value_t = 16, value_name = "N")]
    resolver_elite_considered: usize,
    /// The same two knobs inside boss fights
    #[arg(long, default_value_t = 512, value_name = "N")]
    resolver_boss_iterations: u32,
    #[arg(long, default_value_t = 32, value_name = "N")]
    resolver_boss_considered: usize,
    /// The budget the flagged decisions are re-searched at, and their
    /// play-outs run at
    #[arg(long, default_value_t = 1024, value_name = "N")]
    deep_iterations: u32,
    #[arg(long, default_value_t = 32, value_name = "N")]
    deep_considered: usize,
    /// How many combat decisions, ranked by the first pass's gap between the
    /// best move and the played one, are re-searched
    #[arg(long, default_value_t = 24, value_name = "N")]
    deep_top: usize,
    /// A first-pass gap at or above which a decision is re-searched whatever
    /// its rank
    #[arg(long, default_value_t = 0.10, value_name = "GAP")]
    deep_cutoff: f64,
    /// Visits a candidate needs before it can be named the best move
    #[arg(long, default_value_t = 4, value_name = "N")]
    min_visits: u64,
    /// Skip every search and playout: value heads and policies only
    #[arg(long)]
    net_only: bool,
    /// Search the decisions but do not play the fights out
    #[arg(long)]
    no_playout: bool,
    /// Do not price macro plans by the critic on the states they lead to
    #[arg(long)]
    no_macro_lookahead: bool,
    /// Embed the whole agent observation on every line
    #[arg(long)]
    full_observation: bool,
    /// Seeds every draw the analysis makes: belief worlds, root schedules,
    /// playouts. The game's own seed comes from the trace
    #[arg(long, default_value_t = 1)]
    analysis_seed: u64,
    /// The discount the realised macro return is summed under
    #[arg(long, default_value_t = 1.0, value_name = "GAMMA")]
    return_discount: f64,
    /// Where the output files go (default: beside each trace)
    #[arg(long)]
    out: Option<PathBuf>,
    /// Traces analysed at once
    #[arg(long, default_value_t = 1)]
    jobs: usize,
    /// Workers the fight playouts and the deep pass of one trace are spread
    /// over; the walk itself is sequential, and the output does not depend
    /// on the count
    #[arg(long, default_value_t = 4, value_name = "N")]
    threads: usize,
}

/// The budgets and switches an analysis runs under, refused where a flag
/// names nothing.
fn settings(args: &AnalyzeArgs) -> alphaspire::analyze::Settings {
    use alphaspire::actor::TierBudget;
    use alphaspire::analyze::{Budgets, Settings};
    for (name, value) in [
        ("--resolver-iterations", args.resolver_iterations),
        (
            "--resolver-elite-iterations",
            args.resolver_elite_iterations,
        ),
        ("--resolver-boss-iterations", args.resolver_boss_iterations),
        ("--deep-iterations", args.deep_iterations),
    ] {
        if value == 0 {
            die(&format!("{name} 0 buys no simulation"));
        }
    }
    for (name, value) in [
        ("--resolver-considered", args.resolver_considered),
        (
            "--resolver-elite-considered",
            args.resolver_elite_considered,
        ),
        ("--resolver-boss-considered", args.resolver_boss_considered),
        ("--deep-considered", args.deep_considered),
    ] {
        if value == 0 {
            die(&format!("{name} must be at least one action"));
        }
    }
    if args.threads == 0 {
        die("--threads must be at least one");
    }
    let threads = args.threads;
    Settings {
        budgets: Budgets {
            hallway: TierBudget {
                iterations: args.resolver_iterations,
                considered: args.resolver_considered,
            },
            elite: TierBudget {
                iterations: args.resolver_elite_iterations,
                considered: args.resolver_elite_considered,
            },
            boss: TierBudget {
                iterations: args.resolver_boss_iterations,
                considered: args.resolver_boss_considered,
            },
            deep: TierBudget {
                iterations: args.deep_iterations,
                considered: args.deep_considered,
            },
            deep_top: args.deep_top,
            deep_cutoff: args.deep_cutoff,
            min_visits: args.min_visits,
        },
        net_only: args.net_only,
        playouts: !args.no_playout,
        lookahead: !args.no_macro_lookahead,
        full_observation: args.full_observation,
        analysis_seed: args.analysis_seed,
        return_discount: args.return_discount,
        threads,
    }
}

/// The checkpoints an analysis evaluates with, each through the loader that
/// names the net it is.
fn nets(args: &AnalyzeArgs, combat_net: &Path) -> alphaspire::analyze::Nets {
    let registry = sts2_content::standard_registry();
    let encoder = std::sync::Arc::new(encoder(&registry));
    let combat = alphaspire::net::PolicyValueNet::load(combat_net, std::sync::Arc::clone(&encoder))
        .unwrap_or_else(|error| fail(&format!("--combat-net {}: {error}", combat_net.display())));
    let run = args.run_net.as_ref().map(|stem| {
        alphaspire::analyze::RunNet::load(stem, std::sync::Arc::clone(&encoder))
            .unwrap_or_else(|error| fail(&format!("--run-net {error}")))
    });
    alphaspire::analyze::Nets {
        combat: std::sync::Arc::new(combat),
        combat_stem: combat_net.to_path_buf(),
        run,
    }
}

/// What one trace's work is: its analysis, or without a net its summary.
enum Job {
    Analysis(Box<alphaspire::analyze::Analysis>),
    Summary(Box<alphaspire::trace_summary::TraceSummary>),
}

impl Job {
    fn verification(&self) -> Option<&sts2_replay::ReplayVerification> {
        match self {
            Self::Analysis(analysis) => analysis.summary.coverage.verification.as_ref(),
            Self::Summary(summary) => Some(&summary.verification),
        }
    }
}

/// Where a trace's output goes, and what it is called there.
fn destination(args: &AnalyzeArgs, trace: &Path) -> (PathBuf, String) {
    let stem = trace.file_stem().map_or_else(
        || "trace".to_owned(),
        |stem| stem.to_string_lossy().into_owned(),
    );
    let directory = args.out.clone().unwrap_or_else(|| {
        trace
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    });
    (directory, stem)
}

/// Works each trace and prints one line per trace. Exits 0 when every
/// trace walked to its end, 2 when any analysis stopped short, 3 when one
/// could not be read or written.
///
/// The nets are loaded only where a net is wanted: a summary is fast and
/// net-free, and stays both.
pub fn command(args: &AnalyzeArgs) -> ! {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    if args.jobs == 0 {
        die("--jobs must be at least one");
    }
    let evaluation = (!args.no_net).then(|| {
        let combat_net = args
            .combat_net
            .as_deref()
            .unwrap_or_else(|| die("an analysis takes --combat-net, or --no-net for a summary"));
        (settings(args), nets(args, combat_net))
    });
    let work = |trace: &Path| -> Result<Job, String> {
        match &evaluation {
            Some((settings, nets)) => alphaspire::analyze::analyze_trace(trace, nets, *settings)
                .map(|analysis| Job::Analysis(Box::new(analysis)))
                .map_err(|error| error.to_string()),
            None => alphaspire::trace_summary::summarize_trace(trace)
                .map(|summary| Job::Summary(Box::new(summary)))
                .map_err(|error| error.to_string()),
        }
    };
    // One trace per worker; each trace is sequential, and the nets are
    // shared read-only.
    let next = AtomicUsize::new(0);
    let stopped = AtomicBool::new(false);
    let failed = AtomicBool::new(false);
    let output = std::sync::Mutex::new(std::io::stdout());
    std::thread::scope(|scope| {
        for _ in 0..args.jobs.min(args.traces.len()) {
            scope.spawn(|| {
                while let Some(trace) = args.traces.get(next.fetch_add(1, Ordering::SeqCst)) {
                    let job = match work(trace) {
                        Ok(job) => job,
                        Err(error) => {
                            eprintln!("alphaspire: {error}");
                            failed.store(true, Ordering::SeqCst);
                            continue;
                        }
                    };
                    let (directory, stem) = destination(args, trace);
                    if let Some(verification) = job.verification() {
                        if let Some(warning) = verification.warning {
                            eprintln!("alphaspire: {}: {warning}", trace.display());
                        }
                        if !verification.succeeded() {
                            stopped.store(true, Ordering::SeqCst);
                        }
                    }
                    let line = match job {
                        Job::Analysis(analysis) => {
                            if let Err(error) = analysis.write(&directory, &stem) {
                                eprintln!("alphaspire: {}: {error}", trace.display());
                                failed.store(true, Ordering::SeqCst);
                            }
                            if analysis.summary.coverage.divergence.is_some() {
                                stopped.store(true, Ordering::SeqCst);
                            }
                            analysis.summary_line()
                        }
                        Job::Summary(summary) => {
                            let path = directory.join(format!("{stem}.summary.json"));
                            match write_summary(&directory, &path, &summary) {
                                Ok(()) => {}
                                Err(error) => {
                                    eprintln!("alphaspire: {}: {error}", trace.display());
                                    failed.store(true, Ordering::SeqCst);
                                }
                            }
                            // A summary is what the trace says of itself;
                            // whether the replay verified it is a field of
                            // the summary, not a verdict on the run.
                            format!(
                                "{}: {} {} a{} {} floor {} act {}{} -> {}",
                                trace.display(),
                                summary.source,
                                summary.character,
                                summary.ascension,
                                summary.result,
                                summary.floor_reached,
                                summary.act_reached,
                                if summary.verified {
                                    ""
                                } else {
                                    " (unverified)"
                                },
                                path.display()
                            )
                        }
                    };
                    let mut out = output.lock().expect("stdout is not poisoned");
                    let _ = writeln!(out, "{line}");
                }
            });
        }
    });
    if failed.load(Ordering::SeqCst) {
        std::process::exit(3);
    }
    if stopped.load(Ordering::SeqCst) {
        std::process::exit(2);
    }
    std::process::exit(0);
}

/// The summary as pretty JSON, newline-terminated, in a directory that is
/// made on the way.
fn write_summary(
    directory: &Path,
    path: &Path,
    summary: &alphaspire::trace_summary::TraceSummary,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(summary).map_err(|error| error.to_string())?;
    if !directory.as_os_str().is_empty() {
        std::fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    }
    std::fs::write(path, format!("{json}\n")).map_err(|error| error.to_string())
}
