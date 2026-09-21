//! The command line's shared vocabulary: the argument groups more than one
//! command takes, the enums a flag is spelled in, and the two ways a command
//! ends before it has played anything.
//!
//! One module per command beside this one, each owning its own argument
//! struct; `main.rs` parses and dispatches and holds nothing else.

use alphaspire::policy::{FirstLegal, RolloutPolicy, UniformRandom};
use alphaspire::search::{Gumbel, SearchConfig, Selection, Uct};
use alphaspire::selfplay;
use clap::{Args, ValueEnum};

pub mod analyze;
pub mod encode;
pub mod fights;
pub mod matchup;
pub mod run;

/// What every batch is told: which games to play, as whom, and how many at
/// once. The gates take exactly these; [`run::RunArgs`] adds what a batch
/// keeping its output needs.
#[derive(Args)]
pub struct BatchArgs {
    /// Number of runs to play
    #[arg(long, default_value_t = 1)]
    pub runs: usize,
    /// Policy used to choose actions (or to roll out search)
    #[arg(long, value_enum, default_value = "random")]
    pub policy: Policy,
    /// Character name (for example, `ironclad`) or model ID, checked against
    /// the characters this build plays before any run starts
    #[arg(long, default_value = "CHARACTER.IRONCLAD")]
    pub character: String,
    /// Ascension level
    #[arg(long, default_value_t = 0)]
    pub ascension: u8,
    /// Seed for the policy and search RNG
    #[arg(long, default_value_t = 1)]
    pub analysis_seed: u64,
    /// Maximum actions to take in each run
    #[arg(long, default_value_t = 5000)]
    pub max_steps: usize,
    /// Use a fixed game seed instead of drawing one
    #[arg(long)]
    pub seed: Option<String>,
    /// Unlock preset used to generate runs
    #[arg(long, default_value = sts2_core::FIXTURE_UNLOCK_PRESET)]
    pub preset: String,
    /// Override the preset's prior-run count
    #[arg(long)]
    pub runs_count: Option<u32>,
    /// How many runs to play at once. Each run is built from its own seed
    /// pair, so a parallel batch plays the same games as a serial one
    #[arg(long, default_value_t = 1)]
    pub jobs: usize,
}

impl BatchArgs {
    /// The unlock preset the batch plays under, with the prior-run count
    /// overridden where one was named.
    ///
    /// Generation never reads the count past a first run; the playback
    /// driver compares it exactly against the live profile, so a script
    /// bound for a live game carries the live number.
    pub fn preset(&self) -> sts2_core::UnlockPresetManifest {
        let mut preset = sts2_core::UnlockPresetManifest::with_id(&self.preset)
            .unwrap_or_else(|_| die(&format!("unknown unlock preset {}", self.preset)));
        if let Some(count) = self.runs_count {
            preset.unlocks.number_of_runs = count;
        }
        preset
    }

    /// The batch these flags describe, over a preset and character already
    /// settled, keeping nothing: the gates play each game on both arms, so
    /// neither a script nor a bank could say which arm reached it.
    pub fn plain<'a>(
        &'a self,
        preset: &'a sts2_core::UnlockPresetManifest,
        character: &'a sts2_core::ModelId,
    ) -> selfplay::Batch<'a> {
        selfplay::Batch {
            preset,
            character,
            ascension: self.ascension,
            analysis_seed: self.analysis_seed,
            seed: self.seed.as_deref(),
            runs: self.runs,
            max_steps: self.max_steps,
            jobs: self.jobs,
            harvest: None,
            force_wins: None,
        }
    }
}

/// The counterfactual branches a recording search plays out beside the
/// line it answers with; see `search::Counterfactual`.
#[derive(Args, Clone, Copy)]
pub struct CounterfactualArgs {
    /// Counterfactual branches per fight: at a searched decision, an action
    /// whose completed value stands within --counterfactual-margin of the
    /// answer's while its prior is at most --counterfactual-max-prior is
    /// played out to the fight's end and its decisions recorded with that
    /// outcome, flagged. Only with --emit-samples or --emit-raw (0 = off)
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub counterfactual: usize,
    #[arg(long, default_value_t = 0.05, value_name = "Q")]
    pub counterfactual_margin: f64,
    #[arg(long, default_value_t = 0.1, value_name = "P")]
    pub counterfactual_max_prior: f64,
    /// Decisions a branch may take before it is dropped unrecorded
    #[arg(long, default_value_t = 500, value_name = "N")]
    pub counterfactual_max_steps: usize,
    /// Pass branches per fight: at a searched decision where the turn could
    /// have been ended while an affordable attack would lower an enemy's HP,
    /// the branch ends the turn instead and is played out and recorded the
    /// same way, flagged with kind `pass`. Each qualifying decision branches
    /// with probability --counterfactual-pass-rate (0 = off)
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub counterfactual_pass: usize,
    #[arg(long, default_value_t = 0.125, value_name = "P")]
    pub counterfactual_pass_rate: f64,
}

impl CounterfactualArgs {
    pub fn build(self) -> Option<alphaspire::search::Counterfactual> {
        (self.counterfactual > 0 || self.counterfactual_pass > 0).then_some(
            alphaspire::search::Counterfactual {
                margin: self.counterfactual_margin,
                max_prior: self.counterfactual_max_prior,
                per_fight: self.counterfactual,
                max_steps: self.counterfactual_max_steps,
                pass_per_fight: self.counterfactual_pass,
                pass_rate: self.counterfactual_pass_rate,
            },
        )
    }
}

/// How the decisions inside a fight are answered, as a command line
/// describes it.
///
/// Its own flattened struct because two workflows take it: a run under a
/// run checkpoint, which plays episodes against the resolver, and a macro
/// match, whose two arms must resolve their fights *identically* or measure
/// the resolvers instead of the checkpoints.
#[derive(Args)]
pub struct ResolverArgs {
    /// How in-fight decisions are answered: belief search under Gumbel with
    /// the combat checkpoint at every node, or the checkpoint's own argmax
    /// for one forward pass a decision
    #[arg(long, value_enum, default_value = "searched")]
    pub resolver: ResolverKind,
    /// Search iterations per in-fight decision (default 16). It interacts
    /// with --resolver-considered and the two are tuned together: sixteen
    /// iterations spread over sixteen candidates is about one visit each,
    /// which buys little more than the prior it started from
    #[arg(long)]
    pub resolver_iterations: Option<u32>,
    /// How many root actions the resolver's budget is spread over (default
    /// 16). See --resolver-iterations: at a low iteration budget, fewer
    /// candidates searched deeper and more candidates searched shallowly are
    /// different resolvers, and which is stronger is a measurement
    #[arg(long)]
    pub resolver_considered: Option<usize>,
    /// Engine steps the searched resolver may spend over one run before its
    /// in-fight decisions downgrade to the combat checkpoint's own answer. A
    /// soft cap — the run always walks to its own end — and a reproducible
    /// one, which is the only kind a batch that writes training data may
    /// have: a run under it is still a function of its seed pair. Off by
    /// default, and worth turning on for any unattended batch, since a
    /// batch's wall time is its slowest run
    #[arg(long)]
    pub resolver_budget_steps: Option<u64>,
    /// Search iterations per in-fight decision inside elite fights, where
    /// they differ from --resolver-iterations. A boss is met once an act and
    /// an elite a few times, so the fights that decide a run can be searched
    /// deeper than the hallways at little cost to the batch
    #[arg(long, value_name = "N")]
    pub resolver_elite_iterations: Option<u32>,
    /// Root actions the elite fights' budget is spread over, where it
    /// differs from --resolver-considered
    #[arg(long, value_name = "N")]
    pub resolver_elite_considered: Option<usize>,
    /// Search iterations per in-fight decision inside boss fights, where
    /// they differ from --resolver-iterations. See
    /// --resolver-elite-iterations
    #[arg(long, value_name = "N")]
    pub resolver_boss_iterations: Option<u32>,
    /// Root actions the boss fights' budget is spread over, where it
    /// differs from --resolver-considered
    #[arg(long, value_name = "N")]
    pub resolver_boss_considered: Option<usize>,
}

impl ResolverArgs {
    /// The resolver these flags describe, or the refusal that ends the
    /// process before anything is played.
    ///
    /// Every refusal here is a flag combination that would otherwise be
    /// accepted and quietly do nothing, which is the failure an unattended
    /// batch cannot see: eight hours spent with an idle lever look exactly
    /// like eight hours spent with the lever it was told to pull.
    pub fn settled(&self) -> alphaspire::actor::Resolver {
        let searched = matches!(self.resolver, ResolverKind::Searched);
        if !searched {
            for (flag, named) in [
                ("--resolver-iterations", self.resolver_iterations.is_some()),
                ("--resolver-considered", self.resolver_considered.is_some()),
                (
                    "--resolver-budget-steps",
                    self.resolver_budget_steps.is_some(),
                ),
                (
                    "--resolver-elite-iterations",
                    self.resolver_elite_iterations.is_some(),
                ),
                (
                    "--resolver-elite-considered",
                    self.resolver_elite_considered.is_some(),
                ),
                (
                    "--resolver-boss-iterations",
                    self.resolver_boss_iterations.is_some(),
                ),
                (
                    "--resolver-boss-considered",
                    self.resolver_boss_considered.is_some(),
                ),
            ] {
                if named {
                    die(&format!(
                        "{flag} spends a search budget and --resolver greedy runs no \
                         search: drop it, or pass --resolver searched"
                    ));
                }
            }
            return alphaspire::actor::Resolver::Greedy;
        }
        let iterations = self
            .resolver_iterations
            .unwrap_or(alphaspire::actor::RESOLVER_ITERATIONS);
        let considered = self
            .resolver_considered
            .unwrap_or_else(|| Gumbel::default().considered);
        if considered == 0 {
            die("--resolver-considered must be at least one action");
        }
        if iterations == 0 {
            // Zero simulations leaves the root schedule holding nothing but
            // the prior it sampled its candidates from, which is the greedy
            // resolver reached by an expensive road.
            die(
                "--resolver-iterations 0 buys no simulation, so a searched resolver at \
                 zero is the checkpoint's own answer spelled expensively: pass \
                 --resolver greedy",
            );
        }
        // A tier's own budget inherits whichever of the two knobs it did not
        // name from the base, and is refused at zero for the base's reasons.
        let tier = |name: &str, tier_iterations: Option<u32>, tier_considered: Option<usize>| {
            if tier_iterations.is_none() && tier_considered.is_none() {
                return None;
            }
            let budget = alphaspire::actor::TierBudget {
                iterations: tier_iterations.unwrap_or(iterations),
                considered: tier_considered.unwrap_or(considered),
            };
            if budget.iterations == 0 {
                die(&format!(
                    "--resolver-{name}-iterations 0 buys no simulation: drop the flag to \
                     search {name} fights at the base budget"
                ));
            }
            if budget.considered == 0 {
                die(&format!(
                    "--resolver-{name}-considered must be at least one action"
                ));
            }
            Some(budget)
        };
        alphaspire::actor::Resolver::Searched {
            config: SearchConfig {
                iterations,
                ..SearchConfig::default()
            },
            selection: Gumbel {
                considered,
                ..Gumbel::default()
            },
            elite: tier(
                "elite",
                self.resolver_elite_iterations,
                self.resolver_elite_considered,
            ),
            boss: tier(
                "boss",
                self.resolver_boss_iterations,
                self.resolver_boss_considered,
            ),
            // Steps only. A wall-clock ceiling would make what a fight
            // downgrades depend on what else the box was doing, so the
            // episodes a batch recorded would stop being a function of their
            // seed pairs — and the episodes are the whole point of a rollout.
            budget: alphaspire::search::Budget {
                steps: self.resolver_budget_steps,
                seconds: None,
            },
        }
    }
}

/// Which of the two resolvers answers the decisions inside a fight.
#[derive(Clone, Copy, ValueEnum)]
pub enum ResolverKind {
    /// Belief search under Gumbel: what a generation is expected to run.
    Searched,
    /// The combat checkpoint's own best action, one forward pass a decision.
    Greedy,
}

/// What a harvest writes on each banked entry, as a command line says it.
#[derive(Clone, Copy, ValueEnum)]
pub enum HarvestKind {
    /// The setup: the loadout and the encounter, stood up on the run seed.
    Setup,
    /// The exact erased state of the fight the run stood at.
    State,
    /// Both, side by side.
    Both,
}

impl From<HarvestKind> for alphaspire::library::HarvestAs {
    fn from(kind: HarvestKind) -> Self {
        match kind {
            HarvestKind::Setup => Self::Setup,
            HarvestKind::State => Self::State,
            HarvestKind::Both => Self::Both,
        }
    }
}

/// Which half of a banked entry a load stands its fights up from.
#[derive(Clone, Copy, ValueEnum)]
pub enum SourceKind {
    /// The setup, through the engine's combat start on the entry's run seed.
    Setup,
    /// The exact state; an entry without one is refused, named.
    State,
}

impl From<SourceKind> for alphaspire::library::Source {
    fn from(kind: SourceKind) -> Self {
        match kind {
            SourceKind::Setup => Self::Setup,
            SourceKind::State => Self::State,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum SelectionPolicy {
    Uct,
    Gumbel,
}

impl SelectionPolicy {
    pub fn build(self, considered: usize) -> Box<dyn Selection> {
        match self {
            Self::Uct => Box::new(Uct::default()),
            Self::Gumbel => Box::new(Gumbel {
                considered,
                ..Gumbel::default()
            }),
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Policy {
    First,
    Random,
    /// Hand-written out-of-combat play (card rewards, pathing, events,
    /// rest sites, shops); uniform random inside fights, where search or
    /// chance owns the decision
    Heuristic,
}

impl Policy {
    pub fn build(self) -> Box<dyn RolloutPolicy> {
        match self {
            Self::First => Box::new(FirstLegal),
            Self::Random => Box::new(UniformRandom),
            Self::Heuristic => Box::new(alphaspire::heuristics::Heuristic),
        }
    }
}

/// Refuses a `--policy` on a workflow that has no rollout policy.
///
/// Nothing in a run played by a checkpoint is the rollout policy's: the macro
/// decisions are the run net's and every in-fight one is the resolver's, and
/// a searched resolver with a checkpoint in hand prices its leaves by forward
/// pass rather than by walking one. A flag that names nothing is refused
/// rather than accepted and ignored.
pub fn refuse_rollout_policy(policy: Policy) {
    if !matches!(policy, Policy::Random) {
        die(
            "--policy names nothing here: the run checkpoint answers the macro \
             decisions and the resolver answers the fights",
        );
    }
}

/// The runs or fights a batch lost, panics apart from engine errors: the
/// line the batch ends on says which, because a panic is a bug in this code
/// and an engine error a rule the engine refused, and they are chased in
/// different places.
#[derive(Default)]
pub struct LostTally {
    panics: usize,
    engine_errors: usize,
}

impl LostTally {
    pub fn record(&mut self, error: &selfplay::PlayError) {
        if error.panicked {
            self.panics += 1;
        } else {
            self.engine_errors += 1;
        }
    }

    pub fn total(&self) -> usize {
        self.panics + self.engine_errors
    }

    /// `N of M <unit> were lost: P to panics, E to engine errors`.
    pub fn line(&self, unit: &str, of: usize) -> String {
        format!(
            "{} of {of} {unit} were lost: {} to panics, {} to engine errors",
            self.total(),
            self.panics,
            self.engine_errors
        )
    }
}

/// The encoder this build encodes observations with, over the standard
/// registry: one vocabulary covers all content, so every command builds
/// the same one.
pub fn encoder(registry: &sts2_engine::ContentRegistry) -> alphaspire::encoding::PolicyEncoder {
    alphaspire::encoding::PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(registry),
        registry,
    )
}

/// A stamp read off an artifact, as one line: what it names, or why the
/// line is empty. A stamp absent because it was written before the field
/// existed says something different from one that is present and holds
/// nothing, and an epilogue that quietly printed both as a blank line
/// would be worth less than no line at all.
pub fn joined(names: Option<Vec<String>>) -> String {
    match names {
        None => "not stamped".to_owned(),
        Some(names) if names.is_empty() => "none".to_owned(),
        Some(names) => names.join(", "),
    }
}

/// The character a batch plays, settled before it plays anything.
///
/// A bare character name is the convenient command-line spelling: `silent`
/// resolves to `CHARACTER.SILENT`, case-insensitively. The full model ID is
/// still accepted for existing scripts. [`sts2_core::ModelId`] promises only
/// the shape `CATEGORY.ENTRY`, so `CHARACTER.SILNET` would otherwise fail once
/// inside every run of the batch — an unattended slot spent on a thousand
/// copies of one typo. Checked here against the registry's own character list,
/// it costs one refusal at second zero. `tests/smoke.rs` is what holds that
/// list to the characters a run can actually be started as.
pub fn playable_character(
    registry: &sts2_engine::ContentRegistry,
    name: &str,
) -> sts2_core::ModelId {
    let playable = registry.registered_character_ids();
    playable
        .iter()
        .find(|id| {
            id.entry().eq_ignore_ascii_case(name) || id.to_string().eq_ignore_ascii_case(name)
        })
        .cloned()
        .unwrap_or_else(|| {
            die(&format!(
                "--character {name} is not a character this build plays; the ids are {}",
                playable
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
                    .join(", "),
            ))
        })
}

/// The fight-library epilogue: what a bank holds, by class, and where it
/// went.
pub fn library_line(classes: [usize; 5], verb: &str, path: &std::path::Path) -> String {
    format!(
        "fight library: {} entries (hallway {} elite {} boss {}; act2 {} act3 {}) {verb} {}",
        classes[0] + classes[1] + classes[2],
        classes[0],
        classes[1],
        classes[2],
        classes[3],
        classes[4],
        path.display()
    )
}

/// The command line is wrong: an argument this build cannot parse, a value
/// it does not carry, a combination it refuses. Exit 2 — clap's own code
/// for a bad argument — and nothing was played.
pub fn die(message: &str) -> ! {
    eprintln!("alphaspire: {message}");
    std::process::exit(2);
}

/// The world did not cooperate: a file that would not open, parse, hold
/// what this build reads, or write. Exit 3, distinct from [`die`] so an
/// unattended driver can tell a bad flag at second zero from a shard that
/// failed to land on run 380 — the first wants the command line fixed, the
/// second wants the box looked at. Whatever the batch had written is left
/// unfinished: a sink's manifest is written by `finish`, which this does
/// not reach.
pub fn fail(message: &str) -> ! {
    eprintln!("alphaspire: {message}");
    std::process::exit(3);
}

/// Returns migration guidance for a retired subcommand spelling.
pub fn retired(name: &str) -> Option<&'static str> {
    Some(match name {
        "search" => "`search` is now `run`: the same flags, with `--net` now `--combat-net`",
        "rollout" => "`rollout` is now `run --run-net`: the same flags otherwise",
        "selfplay" => {
            "`selfplay` is retired: `run` plays whole runs, and `--policy heuristic` \
             survives only as the rollout policy inside a searched run"
        }
        "match-macro" => "`match-macro` is now `match macro`",
        "improve" => "`improve` is now `match self`",
        "convert-fights" => {
            "`convert-fights` is now `fights --convert`, naming the directory to write"
        }
        "summarize" => "`summarize` is now `analyze --no-net`",
        "vocabulary" => "`vocabulary` is now `encode --vocabulary`",
        _ => return None,
    })
}
