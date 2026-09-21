//! `encode`: everything about this build's sample encoding — recorded
//! decisions turned into training samples, and the vocabulary those
//! samples index into.

use std::path::PathBuf;

use clap::Args;

use super::{die, encoder, fail};

#[derive(Args)]
pub struct EncodeArgs {
    /// The directory a `--emit-raw` wrote
    #[arg(
        long,
        required_unless_present = "vocabulary",
        conflicts_with = "vocabulary"
    )]
    decisions: Option<PathBuf>,
    /// Where the encoded shards and their manifest go
    #[arg(
        long,
        required_unless_present = "vocabulary",
        conflicts_with = "vocabulary"
    )]
    out: Option<PathBuf>,
    /// Print the encoder's vocabulary instead, one model id per line in
    /// index order (index 1 is the first line; index 0 is the padding row).
    /// The vocabulary hash a checkpoint names is the sha256 of exactly this
    /// listing, newline-terminated, so a consumer that maps token indices
    /// back to names — the trainer's sample audit — regenerates its table
    /// from here
    #[arg(long)]
    vocabulary: bool,
}

/// Encodes the set, or prints the vocabulary.
pub fn command(args: &EncodeArgs) -> ! {
    let registry = sts2_content::standard_registry();
    if args.vocabulary {
        for model_id in alphaspire::encoding::standard_vocabulary(&registry) {
            println!("{model_id}");
        }
        std::process::exit(0);
    }
    let (Some(decisions), Some(out)) = (&args.decisions, &args.out) else {
        die("encoding a set takes --decisions and --out");
    };
    let set = alphaspire::training::DecisionSet::open(decisions)
        .unwrap_or_else(|error| fail(&error.to_string()));
    let encoder = encoder(&registry);
    let mut sink = alphaspire::training::SampleSink::create_for(out, &encoder, &set)
        .unwrap_or_else(|error| fail(&error.to_string()));
    set.for_each_run(|mut run| sink.write_run(&mut run))
        .unwrap_or_else(|error| fail(&error.to_string()));
    let written = sink
        .finish()
        .unwrap_or_else(|error| fail(&error.to_string()));
    println!(
        "training samples: {written} written to {} under policy encoding {} ({} decisions \
         over {} runs read from {})",
        out.display(),
        alphaspire::encoding::POLICY_ENCODING_VERSION,
        set.decisions(),
        set.runs(),
        decisions.display(),
    );
    std::process::exit(0);
}
