//! A run under a run checkpoint: each run is one PPO episode, its
//! out-of-combat decisions sampled from the checkpoint and its fights
//! answered by the frozen resolver, written as trajectory shards with a
//! batch summary beside them.

use std::path::{Path, PathBuf};

use alphaspire::objective::{CombatStrength, Objective};
use alphaspire::policy::RolloutPolicy;
use alphaspire::selfplay;

use super::{ForceElite, RunArgs};
use crate::cli::{die, encoder, fail, library_line, playable_character, refuse_rollout_policy};

/// Everything the validation block settled about the episodes, gathered so
/// the harness takes one argument for "and play the episodes like this".
///
/// The flags are read and refused before anything is played, and what
/// reaches the harness is a description rather than a command line. The
/// resolver is a description too — a batch builds one policy per run, so it
/// is constructed fresh for every episode and a run stays a function of its
/// index.
struct Rollout<'a> {
    run_net: &'a Path,
    combat_net: &'a Path,
    resolver: alphaspire::actor::Resolver,
    /// Where the batch summary goes: the path named, or `summary.json` in
    /// whichever directory the shards are landing in.
    summary: Option<PathBuf>,
    /// The forced-win config, where the walk skips its fights.
    force_wins: Option<alphaspire::forcewins::ForceWins>,
}

/// What the command line settled about the episodes, or the refusal that
/// ends the process before anything is played.
fn settle(args: &RunArgs, force_wins: Option<alphaspire::forcewins::ForceWins>) -> Rollout<'_> {
    refuse_rollout_policy(args.batch.policy);
    let run_net = args
        .run_net
        .as_deref()
        .unwrap_or_else(|| die("a run under a run checkpoint names one with --run-net"));
    let combat_net = args.combat_net.as_deref().unwrap_or_else(|| {
        die(
            "--run-net needs --combat-net too: the resolver plays the fights, and it \
             plays them with the combat checkpoint",
        )
    });
    // An argmax episode is a checkpoint's play to look at, never data to
    // learn from: PPO's ratio needs the log-probability of an action the
    // policy genuinely drew, and argmax records a certainty it never had.
    if args.greedy && (args.emit_ppo.is_some() || args.emit_raw_ppo.is_some()) {
        die("--greedy plays argmax: its episodes are not on-policy training data");
    }
    if args.force_smith.is_some() && !args.greedy {
        die("--force-smith is an ablation over deployed play: pass --greedy with it");
    }
    if (args.force_relics
        || args.force_skip_cards.is_some()
        || args.force_remove
        || args.force_elite.is_some())
        && !args.greedy
    {
        die("the forcings are ablations over deployed play: pass --greedy with them");
    }
    for (name, mix) in [
        ("--explore-rest", args.explore_rest),
        ("--explore-map", args.explore_map),
    ] {
        if let Some(epsilon) = mix {
            if args.greedy {
                die(&format!(
                    "{name} mixes a sampled distribution and --greedy samples nothing"
                ));
            }
            if !(0.0..=1.0).contains(&epsilon) {
                die(&format!(
                    "{name} is a mixing fraction: pass a value in [0, 1]"
                ));
            }
        }
    }
    Rollout {
        run_net,
        combat_net,
        resolver: args.fights.settled(),
        summary: args.summary.clone().or_else(|| {
            args.emit_ppo
                .as_ref()
                .or(args.emit_raw_ppo.as_ref())
                .map(|directory| directory.join("summary.json"))
        }),
        force_wins,
    }
}

/// The share of a batch that may be lost to faulted runs before the batch
/// itself has failed.
///
/// A faulted run ends one episode and is counted; the batch keeps walking,
/// because the alternative is that one engine fault on run 380 of a thousand
/// throws away every episode before it. But a *rate* is a different fact from
/// an incident: a generation losing one run in a thousand has data, and one
/// losing a fifth of its runs has a broken build, and an unattended loop that
/// exits zero on both would train on the second. The threshold is what
/// separates them, and it is deliberately loose — the runs that survive are
/// still honest episodes, so the cost of continuing is only the runs lost.
const LOSS_TOLERANCE: f64 = 0.05;

/// Plays a batch of PPO episodes: whole runs with their macro decisions
/// sampled from the run checkpoint, their fights answered by the frozen
/// resolver, and their trajectories streamed to the PPO sinks.
#[allow(
    clippy::too_many_lines,
    reason = "one block per checkpoint loaded, sink opened and artifact written"
)]
pub fn command(args: &RunArgs, force_wins: Option<alphaspire::forcewins::ForceWins>) -> ! {
    let rollout = settle(args, force_wins);
    let registry = sts2_content::standard_registry();
    let copied = args.like.as_ref().map(|path| {
        if args.batch.runs != 1 {
            die("--like names one run configuration, so it requires --runs 1");
        }
        if args.batch.seed.is_some() {
            die("--like supplies the game seed, so do not also pass --seed");
        }
        alphaspire::trace_summary::run_spec(path)
            .unwrap_or_else(|error| fail(&format!("--like {}: {error}", path.display())))
    });
    let character = copied.as_ref().map_or_else(
        || playable_character(&registry, &args.batch.character),
        |spec| spec.character.clone(),
    );
    let preset = copied
        .as_ref()
        .map_or_else(|| args.batch.preset(), |spec| spec.preset.clone());
    let ascension = copied
        .as_ref()
        .map_or(args.batch.ascension, |spec| spec.ascension);
    let copied_seed = copied.as_ref().map(|spec| spec.seed.clone());
    let encoder = encoder(&registry);
    // Both checkpoints are loaded once and shared read-only across the
    // batch's workers, exactly as a searched batch's is — and each through
    // the loader that names the net it is: the two files are interchangeable
    // on the wire and nowhere else, so a checkpoint pointed at the wrong flag
    // is refused here rather than answering plausible nonsense for a
    // generation. The reward the runs pay is the checkpoint's own, read off
    // its provenance; reward flags only assert it. So the checkpoint loaded,
    // the shards written and the reward paid all carry the one name.
    let (value_semantics, terms) = alphaspire::reward::checkpoint_terms(rollout.run_net)
        .and_then(|(semantics, terms)| {
            alphaspire::reward::resolve_run_terms(args.stated_reward(), &semantics, terms)
                .map(|terms| (semantics, terms))
        })
        .unwrap_or_else(|error| {
            fail(&format!(
                "--run-net {}: {error}: the actor samples its macro decisions from the PPO \
                 checkpoint (`scope: macro`), whose critic predicts a run's return under the \
                 reward it was trained under",
                rollout.run_net.display()
            ))
        });
    let run_net = std::sync::Arc::new(
        alphaspire::net::PolicyValueNet::load_run_priced(
            rollout.run_net,
            std::sync::Arc::new(encoder.clone()),
            &value_semantics,
        )
        .unwrap_or_else(|error| {
            fail(&format!(
                "--run-net {}: {error}: the actor samples its macro decisions from the \
                 PPO checkpoint, whose critic predicts a run's return under the reward \
                 in play ({value_semantics})",
                rollout.run_net.display()
            ))
        }),
    );
    let combat_net = std::sync::Arc::new(
        alphaspire::net::PolicyValueNet::load(
            rollout.combat_net,
            std::sync::Arc::new(encoder.clone()),
        )
        .unwrap_or_else(|error| {
            fail(&format!(
                "--combat-net {}: {error}: the resolver plays fights, so it takes the \
                 combat checkpoint; a macro-scoped one is a run net and belongs to \
                 --run-net",
                rollout.combat_net.display()
            ))
        }),
    );
    for (flag, net) in [
        (rollout.run_net, &run_net),
        (rollout.combat_net, &combat_net),
    ] {
        if let Some(warning) = net.foreign_to(&character) {
            eprintln!("alphaspire: {}: {warning}", flag.display());
        }
    }
    // A policy per run, so an episode depends on its seed pair and nothing
    // else: its own actor, its own reward, its own resolver, no state shared
    // with any other run of the batch.
    let make_policy = || -> Box<dyn RolloutPolicy> {
        let net = std::sync::Arc::clone(&run_net) as std::sync::Arc<dyn alphaspire::net::Evaluate>;
        let resolver =
            rollout
                .resolver
                .build(std::sync::Arc::clone(&combat_net)
                    as std::sync::Arc<dyn alphaspire::net::Evaluate>);
        // Argmax under --greedy: the same run checkpoint played the way a
        // gate grades it and a deployment would run it, rather than the
        // sampled draw an on-policy ratio needs.
        if args.greedy {
            let mut greedy = alphaspire::actor::MacroGreedy::new(net, resolver);
            if let Some(threshold) = args.force_smith {
                greedy = greedy.forcing_smith(threshold);
                if args.force_smith_random {
                    greedy = greedy.smithing_random_cards();
                }
            }
            if args.force_relics {
                greedy = greedy.forcing_relics();
            }
            if let Some(act) = args.force_skip_cards {
                greedy = greedy.forcing_card_skips(act);
            }
            if let Some(r#where) = args.force_elite {
                greedy = greedy.forcing_elites(match r#where {
                    ForceElite::Everywhere => alphaspire::actor::EliteForcing::Everywhere,
                    ForceElite::OverFights => alphaspire::actor::EliteForcing::OverFights,
                });
            }
            if args.force_remove {
                greedy = greedy.forcing_removals();
                if args.force_remove_random {
                    greedy = greedy.removing_random_cards();
                }
            }
            Box::new(greedy)
        } else {
            let mut actor = alphaspire::actor::MacroActor::new(net, resolver)
                .with_reward_terms(terms.elite, terms.relic, terms.gold)
                .with_gold_scope(terms.scope())
                .with_boss_terms(terms.boss);
            if let Some(epsilon) = args.explore_rest {
                actor = actor.exploring_rest(epsilon);
            }
            if let Some(epsilon) = args.explore_map {
                actor = actor.exploring_map(epsilon);
            }
            Box::new(actor)
        }
    };
    // The harness scores every run with an objective for its own report. An
    // episode is priced by `RunReward` on the trajectory itself, line by
    // line, so nothing here reads that number and the cheapest stateless
    // objective stands in.
    let make_objective = || -> Box<dyn Objective> { Box::new(CombatStrength::default()) };
    let batch = selfplay::Batch {
        ascension,
        seed: copied_seed.as_deref().or(args.batch.seed.as_deref()),
        harvest: args.harvest_fights.as_ref().map(|_| args.harvest_as.into()),
        force_wins: rollout.force_wins.as_ref(),
        ..args.batch.plain(&preset, &character)
    };
    let mut sink = args.emit_ppo.as_ref().map(|directory| {
        alphaspire::training::SampleSink::create_ppo_priced(
            directory,
            &encoder,
            args.shard_runs,
            &value_semantics,
            std::slice::from_ref(&character),
        )
        .unwrap_or_else(|error| fail(&error.to_string()))
    });
    let mut raw_sink = args.emit_raw_ppo.as_ref().map(|directory| {
        alphaspire::training::DecisionSink::create_ppo_priced(
            directory,
            args.shard_runs,
            &value_semantics,
            std::slice::from_ref(&character),
        )
        .unwrap_or_else(|error| fail(&error.to_string()))
    });
    if let Some(directory) = &args.out {
        std::fs::create_dir_all(directory).unwrap_or_else(|error| fail(&error.to_string()));
    }
    let mut library = args.harvest_fights.as_ref().map(|directory| {
        let mut sink = alphaspire::library::LibrarySink::create(directory, 32)
            .unwrap_or_else(|error| fail(&error.to_string()));
        // A forced-win bank names the distribution that shaped it.
        if let Some(config) = rollout.force_wins.as_ref() {
            sink.annotate("force_wins", config.echo());
        }
        sink
    });
    let mut summary = alphaspire::summary::BatchSummary::new(alphaspire::summary::Provenance {
        run_net: rollout.run_net.display().to_string(),
        combat_net: rollout.combat_net.display().to_string(),
        resolver: rollout.resolver,
        character: character.to_string(),
        ascension,
        runs: args.batch.runs,
        analysis_seed: args.batch.analysis_seed,
        max_steps: args.batch.max_steps,
    });
    selfplay::play_batch(
        &batch,
        &make_policy,
        &make_objective,
        &mut |index, played| {
            // A faulted run ends one episode. Its line names the seed that
            // reproduces it, its trajectory reaches no sink — a partial episode
            // has no honest terminal and no honest bootstrap, so it is not
            // training data — and the batch walks on. What it does leave, where
            // scripts are being kept, is the script up to the refused step: the
            // trace that replays the fault in the engine alone.
            let mut report = match played {
                Ok(report) => report,
                Err(error) => {
                    summary.record_fault();
                    eprintln!("alphaspire: {error}");
                    if let (Some(directory), Some(script)) = (&args.out, &error.partial_script) {
                        let path = directory.join(format!(
                            "{}-run{index}-a{}.faulted.sts2pgn",
                            character.entry(),
                            ascension
                        ));
                        std::fs::write(&path, script)
                            .unwrap_or_else(|error| fail(&error.to_string()));
                    }
                    return;
                }
            };
            println!("{}", alphaspire::summary::episode_line(&report));
            summary.record(&report);
            if let Some(directory) = &args.out {
                match &report.script {
                    Ok(script) => {
                        let path = directory.join(format!(
                            "{}-{}-a{}.sts2pgn",
                            character.entry(),
                            report.seed,
                            ascension
                        ));
                        std::fs::write(&path, script)
                            .unwrap_or_else(|error| fail(&error.to_string()));
                    }
                    // Said and walked past: an episode a driver cannot replay is
                    // still an episode, and the batch's exit code is a training
                    // loop's, not a validation batch's.
                    Err(reason) => eprintln!("alphaspire: {}: {reason}", report.seed),
                }
            }
            if let Some(sink) = sink.as_mut() {
                sink.write_run(&mut report.macro_decisions)
                    .unwrap_or_else(|error| fail(&error.to_string()));
            }
            if let Some(sink) = raw_sink.as_mut() {
                sink.write_run(&mut report.macro_decisions)
                    .unwrap_or_else(|error| fail(&error.to_string()));
            }
            if let Some(library) = library.as_mut() {
                library
                    .write_run(&mut report.fights)
                    .unwrap_or_else(|error| fail(&error.to_string()));
            }
        },
    );
    if let (Some(library), Some(path)) = (library, &args.harvest_fights) {
        let classes = library
            .finish()
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!("{}", library_line(classes, "banked to", path));
    }
    if let (Some(sink), Some(path)) = (sink, &args.emit_ppo) {
        let written = sink
            .finish()
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!(
            "trajectory samples: {written} written to {}",
            path.display()
        );
    }
    if let (Some(sink), Some(path)) = (raw_sink, &args.emit_raw_ppo) {
        let written = sink
            .finish()
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!(
            "trajectory decisions: {written} written to {}",
            path.display()
        );
    }
    if let Some(path) = &rollout.summary {
        summary
            .write(path)
            .unwrap_or_else(|error| fail(&error.to_string()));
        println!("batch summary: {}", path.display());
    }
    print!("{}", summary.report());
    if let Some(report) = alphaspire::probe::report() {
        print!("{report}");
    }
    // Exit 1 for a batch that walked but lost more than it can afford to;
    // die's 2 is a wrong command line, fail's 3 a file that would not open or
    // write. A batch under the tolerance exits 0 with its losses counted in
    // the summary, which is what lets an unattended generation continue on an
    // engine fault and stop on a broken build.
    let faulted = summary.faulted();
    if faulted > 0 {
        eprintln!(
            "alphaspire: {faulted} of {} episodes were lost to faults",
            args.batch.runs
        );
    }
    #[allow(clippy::cast_precision_loss, reason = "batch sizes are small")]
    let intolerable = faulted as f64 > LOSS_TOLERANCE * args.batch.runs as f64;
    if intolerable {
        eprintln!(
            "alphaspire: that is more than {:.0}% of the batch",
            100.0 * LOSS_TOLERANCE
        );
        std::process::exit(1);
    }
    std::process::exit(0);
}
