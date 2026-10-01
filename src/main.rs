//! The alphaspire command line: parsed here, dispatched here, and nothing
//! else. Each command owns its arguments and its body under `cli/`.

use std::ffi::OsString;
use std::io;

use clap::{Parser, Subcommand};

mod cli;

#[derive(Parser)]
#[command(
    name = "alphaspire",
    version = include_str!(concat!(env!("OUT_DIR"), "/version.txt")),
    about = "Search and self-play companion to sts2sim"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// A subcommand is a workflow, not a feature: a new flag goes on the
/// command whose workflow it extends, and a one-off instrument is an
/// example under `examples/`, not a command.
#[derive(Subcommand)]
#[allow(
    clippy::large_enum_variant,
    reason = "clap parses one of these once at startup, and a Box is not an Args"
)]
enum Command {
    /// Play whole runs. Under --run-net the out-of-combat decisions are
    /// sampled from a PPO run checkpoint and every fight is answered by the
    /// frozen resolver, each run one episode written as trajectory shards
    /// with a batch summary beside them. Without it the run is searched:
    /// belief search inside fights, and under --mode act the out-of-combat
    /// decisions too, with --macro-net guiding the act tree
    Run(cli::run::RunArgs),
    /// Generate training data from a fight library: banked entries selected
    /// at an explicit class mix, each replayed as its own belief-search
    /// fight. With --convert, rewrite the library between formats instead
    Fights(cli::fights::FightsArgs),
    /// Encode recorded decisions (`--emit-raw`) into training samples under
    /// this build's policy encoding: the shards `--emit-samples` would have
    /// written from the same search, byte for byte. With --vocabulary,
    /// print the encoder's vocabulary instead
    Encode(cli::encode::EncodeArgs),
    /// Play a paired gate: two arms over identical games, a paired table, a
    /// sign test, and an exit code a promotion script reads — 0 only on a
    /// measured candidate win. `combat` grades two combat checkpoints,
    /// `macro` two run checkpoints, `self` one checkpoint against itself
    /// unsearched
    #[command(subcommand)]
    Match(cli::matchup::Level),
    /// Analyse recorded runs: replay each `.sts2pgn` through the simulator
    /// and evaluate every decision the player made — the combat checkpoint
    /// under belief search inside fights, with the recorded move pinned
    /// beside the best one and each fight played out from its entry; the
    /// run checkpoint's critic and policy outside them. Writes one summary
    /// and one line per decision beside each trace; renders nothing. With
    /// --no-net, replay alone and write the trace's outcome, HP curve, room
    /// counts and act-end resources
    Analyze(cli::analyze::AnalyzeArgs),
    /// Print a completion script for the given shell
    Completions { shell: clap_complete::Shell },
    /// A spelling this build no longer answers to, refused with the one
    /// that replaced it.
    #[command(external_subcommand)]
    Retired(Vec<OsString>),
}

fn main() {
    match Cli::parse().command {
        Command::Run(args) => cli::run::command(&args),
        Command::Fights(args) => cli::fights::command(&args),
        Command::Encode(args) => cli::encode::command(&args),
        Command::Match(level) => cli::matchup::command(&level),
        Command::Analyze(args) => cli::analyze::command(&args),
        Command::Completions { shell } => {
            use clap::CommandFactory as _;
            clap_complete::generate(shell, &mut Cli::command(), "alphaspire", &mut io::stdout());
        }
        Command::Retired(words) => {
            use clap::CommandFactory as _;
            let name = words
                .first()
                .map(|word| word.to_string_lossy().into_owned())
                .unwrap_or_default();
            match cli::retired(&name) {
                Some(message) => cli::die(message),
                None => Cli::command()
                    .error(
                        clap::error::ErrorKind::InvalidSubcommand,
                        format!("unrecognized subcommand '{name}'"),
                    )
                    .exit(),
            }
        }
    }
}
