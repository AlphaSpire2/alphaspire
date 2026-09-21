//! A searched run: belief search inside every fight, and — in act mode —
//! the out-of-combat decisions searched over act-scoped worlds too.

use std::path::PathBuf;

use alphaspire::objective::{ActBoundary, CombatStrength, Objective, SharedCoverage};
use alphaspire::policy::RolloutPolicy;
use alphaspire::search::{ActSearch, BeliefSearch, SearchConfig, TrueStateSearch};
use alphaspire::selfplay;

use super::{RunArgs, SearchMode};
use crate::cli::{
    LostTally, Policy, SelectionPolicy, die, encoder, fail, library_line, playable_character,
};

/// Everything the validation settled about the search, gathered so the
/// harness takes one argument for "and search it like this".
struct Search {
    config: SearchConfig,
    /// The act arm's own budgets: rollouts there cross rooms, not turns.
    act_config: SearchConfig,
    mode: SearchMode,
    selection: SelectionPolicy,
    considered: usize,
    /// The act arm's per-run soft ceiling, off unless a flag set one.
    budget: alphaspire::search::Budget,
    emit_samples: Option<PathBuf>,
    emit_macro_samples: Option<PathBuf>,
    emit_raw: Option<PathBuf>,
    emit_raw_macro: Option<PathBuf>,
    shard_runs: usize,
    /// The counterfactual branches a recording combat search plays out.
    counterfactual: Option<alphaspire::search::Counterfactual>,
    net: Option<PathBuf>,
    /// The act tree's own checkpoint, where one was named.
    macro_net: Option<PathBuf>,
}

/// What the command line settled about the search, or the refusal that
/// ends the process before anything is played.
fn settle(args: &RunArgs) -> Search {
    let emitting = args.emit_samples.is_some() || args.emit_raw.is_some();
    let emitting_macro = args.emit_macro_samples.is_some() || args.emit_raw_macro.is_some();
    if (emitting || args.combat_net.is_some())
        && !matches!(args.mode, SearchMode::Belief | SearchMode::Act)
    {
        die("--emit-samples, --emit-raw and --combat-net require --mode belief or act");
    }
    if (args.act_budget_steps.is_some() || args.act_budget_secs.is_some())
        && !matches!(args.mode, SearchMode::Act)
    {
        die("the --act-budget-* caps require --mode act");
    }
    // Macro emission has two sources, and this is where they part: act mode
    // records the decisions its act tree *searched*, and belief mode records
    // the ones its out-of-combat teacher *answered*. The second is the
    // imitation set — free, since the production combat-generation config
    // already plays every one of those decisions — and it needs a teacher
    // with a distribution to teach, which today means the heuristic policy.
    if emitting_macro {
        match args.mode {
            SearchMode::Act => {}
            SearchMode::Belief if matches!(args.batch.policy, Policy::Heuristic) => {}
            SearchMode::Belief => die(
                "--emit-macro-samples in belief mode clones the out-of-combat \
                 teacher's own scores, and only --policy heuristic has any: \
                 pass it, or use --mode act to record searched decisions",
            ),
            SearchMode::TrueState => die(
                "--emit-macro-samples requires --mode act (searched decisions) \
                 or --mode belief with --policy heuristic (imitation)",
            ),
        }
    }
    if args.macro_net.is_some() && !matches!(args.mode, SearchMode::Act) {
        die("--macro-net guides the act tree, so it requires --mode act");
    }
    if args.macro_net.is_some() && args.combat_net.is_none() {
        // Without a combat checkpoint the act tree's fight-entry leaves
        // would fall back to playing the fight on the rollout policy, which
        // plays fights at uniform random. The nets are the evaluators, or
        // there is no evaluation.
        die(
            "--macro-net needs --combat-net too: the combat checkpoint is what \
             prices a fight-entry leaf, and without it the act tree would \
             value one by playing the fight at random",
        );
    }
    if args.act_budget_secs.is_some() && (emitting_macro || emitting) {
        // A wall-clock cap makes what a run plays depend on what the box
        // was doing, so the samples it writes are no longer a function of
        // the seed pair. Emission takes the step cap.
        die(
            "--act-budget-secs makes a run's play depend on the box, so it \
             cannot write training samples: use --act-budget-steps",
        );
    }
    if args.considered == 0 {
        die("--considered must be at least one action");
    }
    Search {
        config: SearchConfig {
            iterations: args.iterations,
            rollout_depth: args.rollout_depth,
            temperature: args.temperature,
        },
        act_config: SearchConfig {
            iterations: args.act_iterations.unwrap_or(args.iterations),
            rollout_depth: args.act_rollout_depth,
            temperature: args.act_temperature,
        },
        mode: args.mode,
        selection: args.selection,
        considered: args.considered,
        budget: alphaspire::search::Budget {
            steps: args.act_budget_steps,
            seconds: args.act_budget_secs,
        },
        emit_samples: args.emit_samples.clone(),
        emit_macro_samples: args.emit_macro_samples.clone(),
        emit_raw: args.emit_raw.clone(),
        emit_raw_macro: args.emit_raw_macro.clone(),
        shard_runs: args.shard_runs,
        counterfactual: args.counterfactual.build(),
        net: args.combat_net.clone(),
        macro_net: args.macro_net.clone(),
    }
}

/// Plays a batch of searched runs, keeping what the flags asked for:
/// scripts, samples, raw decisions, the fight bank.
#[allow(
    clippy::too_many_lines,
    reason = "one flag per lever the harness offers"
)]
pub fn command(args: &RunArgs, force_wins: Option<&alphaspire::forcewins::ForceWins>) -> ! {
    let search = settle(args);
    let registry = sts2_content::standard_registry();
    let character = playable_character(&registry, &args.batch.character);
    let preset = args.batch.preset();
    let coverage = SharedCoverage::default();
    let encoder = encoder(&registry);
    let recording = search.emit_samples.is_some() || search.emit_raw.is_some();
    let recording_macro = search.emit_macro_samples.is_some() || search.emit_raw_macro.is_some();
    // Share the read-only checkpoint across workers; each worker evaluates
    // its own positions directly.
    let net = search.net.as_ref().map(|base| {
        let net = alphaspire::net::PolicyValueNet::load(base, std::sync::Arc::new(encoder.clone()))
            .unwrap_or_else(|error| fail(&error.to_string()));
        if let Some(warning) = net.foreign_to(&character) {
            eprintln!("alphaspire: {}: {warning}", base.display());
        }
        std::sync::Arc::new(net)
    });
    // The act tree's own checkpoint, loaded through the macro-scoped loader
    // so a combat checkpoint handed to `--macro-net` is refused rather than
    // reinterpreted — and paired with the combat one into the evaluator that
    // owns the fight-entry translation.
    let act_net = search.macro_net.as_ref().map(|base| {
        let macro_net =
            alphaspire::net::PolicyValueNet::load_macro(base, std::sync::Arc::new(encoder.clone()))
                .unwrap_or_else(|error| fail(&error.to_string()));
        if let Some(warning) = macro_net.foreign_to(&character) {
            eprintln!("alphaspire: {}: {warning}", base.display());
        }
        let macro_net = std::sync::Arc::new(macro_net);
        let combat = net
            .clone()
            .unwrap_or_else(|| die("--macro-net needs --combat-net"));
        std::sync::Arc::new(alphaspire::net::ActEvaluator::new(
            macro_net as std::sync::Arc<dyn alphaspire::net::Evaluate>,
            combat as std::sync::Arc<dyn alphaspire::net::Evaluate>,
        ))
    });
    if args.batch.jobs > 1 && matches!(search.mode, SearchMode::TrueState) {
        // Say it plainly rather than refuse: the fuzzer wants the cores, and
        // its scripts are their own reproducers.
        eprintln!(
            "alphaspire: --jobs {} with true-state search: the coverage table \
             steers the search and is paid first-come, so the walks are not \
             reproducible from their seeds",
            args.batch.jobs
        );
    }
    // A policy per run, built here rather than once per batch: a run then
    // depends on its seed pair and nothing else.
    let make_policy = || -> Box<dyn RolloutPolicy> {
        let rollout: Box<dyn RolloutPolicy> = args.batch.policy.build();
        let selection = search.selection.build(search.considered);
        match search.mode {
            SearchMode::TrueState => Box::new(
                TrueStateSearch::with_rollout(search.config, coverage.clone(), rollout)
                    .selecting(selection),
            ),
            // Belief search plays for strength inside each fight — the
            // fair-analysis objective — while the batch still keeps its
            // coverage ledger for the summary line.
            SearchMode::Belief => {
                let mut search_policy =
                    BeliefSearch::with_rollout(search.config, CombatStrength::default(), rollout)
                        .selecting(selection);
                if let Some(net) = &net {
                    search_policy = search_policy
                        .with_net(std::sync::Arc::clone(net)
                            as std::sync::Arc<dyn alphaspire::net::Evaluate>);
                }
                if recording {
                    search_policy = search_policy.recording();
                    if let Some(counterfactual) = search.counterfactual {
                        search_policy = search_policy.branching(counterfactual);
                    }
                }
                // The imitation source: every macro decision the heuristic
                // answers on an ordinary self-play run, recorded with the
                // teacher's own distribution over it.
                if recording_macro {
                    search_policy = search_policy.recording_macro();
                }
                Box::new(search_policy)
            }
            // Act mode composes: the very combat search above keeps every
            // in-fight decision (net guidance and in-combat sample recording
            // unchanged), and out-of-combat decisions are searched over
            // act-scoped worlds under the act-boundary objective. The act
            // tree gets its own checkpoint or none: with --macro-net it runs
            // on macro priors and net-valued leaves, and without one it
            // uses rollouts and uniform priors.
            SearchMode::Act => {
                let mut combat =
                    BeliefSearch::with_rollout(search.config, CombatStrength::default(), rollout)
                        .selecting(selection);
                if let Some(net) = &net {
                    combat = combat
                        .with_net(std::sync::Arc::clone(net)
                            as std::sync::Arc<dyn alphaspire::net::Evaluate>);
                }
                if recording {
                    combat = combat.recording();
                }
                let mut act = ActSearch::new(
                    search.act_config,
                    ActBoundary::default(),
                    args.batch.policy.build(),
                    combat,
                )
                .selecting(search.selection.build(search.considered))
                .within(search.budget);
                if let Some(evaluator) = &act_net {
                    act = act.with_net(std::sync::Arc::clone(evaluator)
                        as std::sync::Arc<dyn alphaspire::net::Evaluate>);
                }
                if recording_macro {
                    act = act.recording_macro();
                }
                Box::new(act)
            }
        }
    };
    let make_objective = || -> Box<dyn Objective> { Box::new(coverage.clone()) };
    if let Some(directory) = &args.out {
        std::fs::create_dir_all(directory).unwrap_or_else(|error| fail(&error.to_string()));
    }
    let batch = selfplay::Batch {
        harvest: args.harvest_fights.as_ref().map(|_| args.harvest_as.into()),
        force_wins,
        ..args.batch.plain(&preset, &character)
    };
    let mut unscriptable = 0_usize;
    let mut lost = LostTally::default();
    let mut outcomes = selfplay::OutcomeTally::default();
    // Streamed, not accumulated: a generation-sized batch's samples do not
    // fit in memory, and never needed to.
    let mut sink = search.emit_samples.as_ref().map(|directory| {
        alphaspire::training::SampleSink::create(
            directory,
            &encoder,
            search.shard_runs,
            std::slice::from_ref(&character),
        )
        .unwrap_or_else(|error| fail(&error.to_string()))
    });
    // The macro sink's own directory and its own header: a different net's
    // data, on a different horizon, under a different `value_semantics`.
    let macro_source = match search.mode {
        SearchMode::Belief => alphaspire::training::SOURCE_HEURISTIC,
        SearchMode::Act | SearchMode::TrueState => alphaspire::training::SOURCE_SEARCH,
    };
    let mut macro_sink = search.emit_macro_samples.as_ref().map(|directory| {
        alphaspire::training::SampleSink::create_macro(
            directory,
            &encoder,
            search.shard_runs,
            macro_source,
            std::slice::from_ref(&character),
        )
        .unwrap_or_else(|error| fail(&error.to_string()))
    });
    let mut raw_sink = search.emit_raw.as_ref().map(|directory| {
        alphaspire::training::DecisionSink::create(
            directory,
            search.shard_runs,
            std::slice::from_ref(&character),
        )
        .unwrap_or_else(|error| fail(&error.to_string()))
    });
    let mut raw_macro_sink = search.emit_raw_macro.as_ref().map(|directory| {
        alphaspire::training::DecisionSink::create_macro(
            directory,
            search.shard_runs,
            macro_source,
            std::slice::from_ref(&character),
        )
        .unwrap_or_else(|error| fail(&error.to_string()))
    });
    // Entries are heavier than samples but shard the same way; a fixed
    // 32 runs per shard keeps library shards in the tens of megabytes.
    let mut library = args.harvest_fights.as_ref().map(|directory| {
        let mut sink = alphaspire::library::LibrarySink::create(directory, 32)
            .unwrap_or_else(|error| fail(&error.to_string()));
        // A forced-win bank names the distribution that shaped it.
        if let Some(config) = force_wins {
            sink.annotate("force_wins", config.echo());
        }
        sink
    });
    selfplay::play_batch(&batch, &make_policy, &make_objective, &mut |_, played| {
        // A lost run is reported loudly and the batch keeps walking: its
        // seed line is the reproducer for whatever fault took it down.
        let mut report = match played {
            Ok(report) => report,
            Err(error) => {
                lost.record(&error);
                eprintln!("alphaspire: {error}");
                return;
            }
        };
        outcomes.record(&report);
        println!("{}", selfplay::summarize(&report));
        if let Err(reason) = &report.script {
            unscriptable += 1;
            eprintln!("alphaspire: {}: {reason}", report.seed);
        }
        if let (Some(directory), Ok(script)) = (&args.out, &report.script) {
            let path = directory.join(format!(
                "{}-{}-a{}.sts2pgn",
                character.entry(),
                report.seed,
                args.batch.ascension
            ));
            std::fs::write(&path, script).unwrap_or_else(|error| fail(&error.to_string()));
        }
        if let Some(sink) = sink.as_mut() {
            sink.write_run(&mut report.decisions)
                .unwrap_or_else(|error| fail(&error.to_string()));
        }
        if let Some(sink) = raw_sink.as_mut() {
            sink.write_run(&mut report.decisions)
                .unwrap_or_else(|error| fail(&error.to_string()));
        }
        if let Some(sink) = macro_sink.as_mut() {
            sink.write_run(&mut report.macro_decisions)
                .unwrap_or_else(|error| fail(&error.to_string()));
        }
        if let Some(sink) = raw_macro_sink.as_mut() {
            sink.write_run(&mut report.macro_decisions)
                .unwrap_or_else(|error| fail(&error.to_string()));
        }
        if let Some(library) = library.as_mut() {
            library
                .write_run(&mut report.fights)
                .unwrap_or_else(|error| fail(&error.to_string()));
        }
    });
    if let (Some(sink), Some(path)) = (sink, &search.emit_samples) {
        let written = sink
            .finish()
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!("training samples: {written} written to {}", path.display());
    }
    if let (Some(sink), Some(path)) = (macro_sink, &search.emit_macro_samples) {
        let written = sink
            .finish()
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!("macro samples: {written} written to {}", path.display());
    }
    if let (Some(sink), Some(path)) = (raw_sink, &search.emit_raw) {
        let written = sink
            .finish()
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!("decisions: {written} written to {}", path.display());
    }
    if let (Some(sink), Some(path)) = (raw_macro_sink, &search.emit_raw_macro) {
        let written = sink
            .finish()
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!("macro decisions: {written} written to {}", path.display());
    }
    if let (Some(library), Some(path)) = (library, &args.harvest_fights) {
        let classes = library
            .finish()
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!("{}", library_line(classes, "banked to", path));
    }
    println!("{}", outcomes.summary());
    let deaths = outcomes.death_table(10);
    if !deaths.is_empty() {
        println!("deaths, deadliest first:");
        for line in &deaths {
            println!("  {line}");
        }
    }
    println!("batch coverage: {} models exercised", coverage.seen_count());
    if let Some(report) = alphaspire::probe::report() {
        print!("{report}");
    }
    // Exit 1 for a batch that walked but lost runs or scripts; die's 2 is a
    // wrong command line, fail's 3 a file that would not open or write.
    if lost.total() > 0 {
        eprintln!("alphaspire: {}", lost.line("runs", args.batch.runs));
    }
    if unscriptable > 0 {
        eprintln!(
            "alphaspire: {unscriptable} of {} runs emitted no script",
            args.batch.runs
        );
    }
    if lost.total() > 0 || unscriptable > 0 {
        std::process::exit(1);
    }
    std::process::exit(0);
}
