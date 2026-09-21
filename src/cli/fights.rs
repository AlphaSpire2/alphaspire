//! `fights`: a fight library played as belief-search fights at an explicit
//! class mix — the data producer — or rewritten between formats.

use std::path::PathBuf;

use alphaspire::objective::CombatStrength;
use alphaspire::policy::{RolloutPolicy, UniformRandom};
use alphaspire::search::{BeliefSearch, SearchConfig};
use clap::{ArgGroup, Args};

use super::{
    CounterfactualArgs, HarvestKind, LostTally, SelectionPolicy, SourceKind, die, encoder, fail,
    joined, library_line,
};

#[derive(Args)]
#[command(group(
    ArgGroup::new("play")
        .multiple(true)
        .conflicts_with("convert")
        .args([
            "from", "fights", "mix", "iterations", "rollout_depth", "selection", "considered",
            "analysis_seed", "max_steps", "emit_samples", "emit_raw", "shard_fights", "net",
            "jobs", "counterfactual", "counterfactual_margin", "counterfactual_max_prior",
            "counterfactual_max_steps", "counterfactual_pass", "counterfactual_pass_rate",
        ])
))]
pub struct FightsArgs {
    /// The fight library directory a harvest wrote
    #[arg(long)]
    library: PathBuf,
    /// Rewrite the library into this directory instead of playing it, each
    /// entry saying its fight as --harvest-as asks: a format-2 bank's exact
    /// states are stood up and projected to their setups, and a format-3
    /// bank keeps or drops its states. A format-2 bank already loads as
    /// state-only entries; this is for the smaller bank, not for loading
    #[arg(long, value_name = "OUT", help_heading = "Rewriting")]
    convert: Option<PathBuf>,
    /// With --convert: what each written entry says its fight as; `state`
    /// and `both` are refused on an entry that carries none
    #[arg(
        long,
        value_enum,
        default_value = "setup",
        requires = "convert",
        help_heading = "Rewriting"
    )]
    harvest_as: HarvestKind,
    /// Which half of each entry the fights stand up from: the setup (one
    /// loadout, many hands — a generation's input) or the exact state (the
    /// very fight that was banked — what a paired evaluation wants; refused
    /// on an entry that carries none)
    #[arg(long, value_enum, default_value = "setup")]
    from: SourceKind,
    /// How many fights to play
    #[arg(long, default_value_t = 1)]
    fights: usize,
    /// The class mix over the bank, e.g. boss=0.3,elite=0.3,hallway=0.4
    /// over the classes hallway, elite, boss, and act2. Weights are
    /// normalized over the classes named; an unnamed class gets nothing;
    /// `all` in place of a weight (boss=all) replays every banked entry of
    /// the class once, off the top of --fights
    #[arg(long, required_unless_present = "convert")]
    mix: Option<String>,
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
    /// Seed for entry selection, world dealing, and the search RNG
    #[arg(long, default_value_t = 1)]
    analysis_seed: u64,
    /// Maximum actions to take in each fight
    #[arg(long, default_value_t = 500)]
    max_steps: usize,
    /// Write expert-iteration training samples into this directory, as
    /// versioned shards beside a manifest
    #[arg(long)]
    emit_samples: Option<PathBuf>,
    /// Write the decisions as recorded, for `alphaspire encode` — see
    /// `run --emit-raw`
    #[arg(long)]
    emit_raw: Option<PathBuf>,
    /// How many fights each sample shard holds
    #[arg(long, default_value_t = 32)]
    shard_fights: usize,
    /// Guide the search with a trained checkpoint: the base path of a
    /// trainer export, `<base>.onnx` beside `<base>.json`
    #[arg(long)]
    net: Option<PathBuf>,
    /// How many fights to play at once. Each playout is a function of its
    /// index, so a parallel batch plays the same fights as a serial one
    #[arg(long, default_value_t = 1)]
    jobs: usize,
    #[command(flatten)]
    counterfactual: CounterfactualArgs,
}

/// Plays the library at the mix asked for, or rewrites it.
pub fn command(args: &FightsArgs) -> ! {
    if let Some(out) = &args.convert {
        let classes = alphaspire::library::convert(&args.library, out, args.harvest_as.into())
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!(
            "{}",
            library_line(
                classes,
                &format!(
                    "written as {} to",
                    alphaspire::library::HarvestAs::from(args.harvest_as).as_str()
                ),
                out
            )
        );
        std::process::exit(0)
    }
    play(args)
}

/// Replays banked fight entries at an explicit tier mix, emitting training
/// samples through the same sink a searched batch uses — the fight
/// library's answer to the tiers self-play starves.
#[allow(
    clippy::too_many_lines,
    reason = "one flag per lever the harness offers"
)]
fn play(args: &FightsArgs) -> ! {
    if args.considered == 0 {
        die("--considered must be at least one action");
    }
    let mix = args
        .mix
        .as_deref()
        .unwrap_or_else(|| die("--mix names the class mix a batch plays"));
    let library = alphaspire::library::FightLibrary::open(&args.library)
        .unwrap_or_else(|error| fail(&error.to_string()));
    let mix = alphaspire::library::Mix::parse(mix).unwrap_or_else(|error| die(&error.to_string()));
    let quotas = mix
        .quotas(args.fights, library.classes())
        .unwrap_or_else(|error| die(&error.to_string()));
    let schedule =
        alphaspire::library::plan(&mix, args.fights, library.classes(), args.analysis_seed)
            .unwrap_or_else(|error| die(&error.to_string()));
    let batch = library
        .load(&schedule, args.from.into())
        .unwrap_or_else(|error| fail(&error.to_string()));
    // Said, never swallowed: a bank harvested from real play may hold a
    // fight this build cannot re-enter, and the batch is the fights that
    // stood up. Whoever reads the lane decides whether the share is small
    // enough to keep.
    if !batch.unusable.is_empty() {
        eprintln!(
            "alphaspire: {} of {} entries the plan drew would not stand up and were skipped",
            batch.unusable.len(),
            schedule.len(),
        );
        for said in batch.unusable.iter().take(3) {
            eprintln!("alphaspire:   {said}");
        }
    }
    let fights = batch.fights;
    // Said loudly rather than refused: a forced-win bank is legitimate
    // generation input — full-search replays of real fights — but it must
    // never quietly become a judging instrument, and the warning is what
    // keeps the provenance in front of whoever reads the two arms.
    if let Some(origins) = library.origins()
        && origins[alphaspire::library::FightOrigin::ForcedWin.index()] > 0
    {
        eprintln!(
            "alphaspire: {} of the bank's {} entries were reached by forced-win walks: \
             fine to train on, not a bank to judge checkpoints with",
            origins[alphaspire::library::FightOrigin::ForcedWin.index()],
            library.entries(),
        );
    }
    // Who the batch is about to play, off the entries the plan drew: a bank
    // may honestly hold several characters, and this is what the shards it
    // writes are stamped with.
    let characters: Vec<sts2_core::ModelId> = fights
        .iter()
        .map(|fight| fight.meta.character.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let registry = sts2_content::standard_registry();
    let encoder = encoder(&registry);
    let net = args.net.as_ref().map(|base| {
        let net = alphaspire::net::PolicyValueNet::load(base, std::sync::Arc::new(encoder.clone()))
            .unwrap_or_else(|error| fail(&error.to_string()));
        for character in &characters {
            if let Some(warning) = net.foreign_to(character) {
                eprintln!("alphaspire: {}: {warning}", base.display());
            }
        }
        std::sync::Arc::new(net)
    });
    let config = SearchConfig {
        iterations: args.iterations,
        rollout_depth: args.rollout_depth,
        temperature: 0.5,
    };
    let emitting = args.emit_samples.is_some() || args.emit_raw.is_some();
    let make_policy = || -> Box<dyn RolloutPolicy> {
        let mut search =
            BeliefSearch::with_rollout(config, CombatStrength::default(), Box::new(UniformRandom))
                .selecting(args.selection.build(args.considered));
        if let Some(net) = &net {
            search = search.with_net(
                std::sync::Arc::clone(net) as std::sync::Arc<dyn alphaspire::net::Evaluate>
            );
        }
        if emitting {
            let mut search = search.recording();
            if let Some(counterfactual) = args.counterfactual.build() {
                search = search.branching(counterfactual);
            }
            Box::new(search)
        } else {
            Box::new(search)
        }
    };
    let mut sink = args.emit_samples.as_ref().map(|directory| {
        alphaspire::training::SampleSink::create(
            directory,
            &encoder,
            args.shard_fights,
            &characters,
        )
        .unwrap_or_else(|error| fail(&error.to_string()))
    });
    let mut raw_sink = args.emit_raw.as_ref().map(|directory| {
        alphaspire::training::DecisionSink::create(directory, args.shard_fights, &characters)
            .unwrap_or_else(|error| fail(&error.to_string()))
    });
    let batch = alphaspire::library::GenerationBatch {
        fights: &fights,
        analysis_seed: args.analysis_seed,
        max_steps: args.max_steps,
        jobs: args.jobs,
    };
    let mut lost = LostTally::default();
    let mut tally = alphaspire::library::FightTally::default();
    alphaspire::library::play_fights(&batch, &make_policy, &mut |_, played| {
        let mut report = match played {
            Ok(report) => report,
            Err(error) => {
                lost.record(&error);
                eprintln!("alphaspire: {error}");
                return;
            }
        };
        tally.record(&report);
        println!("{}", alphaspire::library::summarize(&report));
        if let Some(sink) = sink.as_mut() {
            sink.write_run(&mut report.decisions)
                .unwrap_or_else(|error| fail(&error.to_string()));
        }
        if let Some(sink) = raw_sink.as_mut() {
            sink.write_run(&mut report.decisions)
                .unwrap_or_else(|error| fail(&error.to_string()));
        }
    });
    if let (Some(sink), Some(path)) = (sink, &args.emit_samples) {
        let written = sink
            .finish()
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!("training samples: {written} written to {}", path.display());
    }
    if let (Some(sink), Some(path)) = (raw_sink, &args.emit_raw) {
        let written = sink
            .finish()
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!("decisions: {written} written to {}", path.display());
    }
    println!(
        "fight mix: hallway {} elite {} boss {} act2 {} act3 {} over a bank of {} entries",
        quotas[0],
        quotas[1],
        quotas[2],
        quotas[3],
        quotas[4],
        library.entries(),
    );
    // What the bank says it holds, beside what the plan actually drew out
    // of it: the two disagree when a mix reaches a slice of a mixed bank,
    // and the shards this batch wrote are stamped with the second.
    println!(
        "bank characters: {}",
        joined(
            library
                .characters()
                .map(|ids| ids.iter().map(ToString::to_string).collect::<Vec<String>>())
        )
    );
    println!(
        "played characters: {}",
        joined(Some(
            tally
                .composition()
                .iter()
                .map(|(character, count)| format!("{character} {count}"))
                .collect::<Vec<String>>(),
        ))
    );
    println!("{}", tally.summary());
    let losses = tally.loss_table(10);
    if !losses.is_empty() {
        println!("losses, costliest first:");
        for line in &losses {
            println!("  {line}");
        }
    }
    if let Some(report) = alphaspire::probe::report() {
        print!("{report}");
    }
    if lost.total() > 0 {
        eprintln!("alphaspire: {}", lost.line("fights", args.fights));
        std::process::exit(1);
    }
    std::process::exit(0);
}
