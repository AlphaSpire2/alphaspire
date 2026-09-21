//! `run`: whole runs, under whichever macro the command line names.
//!
//! One command plays runs. With `--run-net` the out-of-combat decisions are
//! sampled from a PPO run checkpoint and every fight is answered by the
//! frozen resolver; without it the run is searched — belief search inside
//! fights, and under `--mode act` the out-of-combat decisions too. The two
//! bodies share the harness underneath — [`alphaspire::selfplay::play_batch`],
//! the same seed derivation, the same run-order reordering — which is what
//! makes every batch this build plays comparable with every other.

use std::path::PathBuf;

use clap::{ArgGroup, Args, ValueEnum};

use super::{BatchArgs, CounterfactualArgs, HarvestKind, ResolverArgs, SelectionPolicy, die};

mod ppo;
mod searched;

/// A batch of whole runs: what every batch is told, what it keeps, and the
/// flags of whichever macro plays it.
#[derive(Args)]
#[command(group(
    ArgGroup::new("searched")
        .multiple(true)
        .conflicts_with("run_net")
        .args([
            "mode", "iterations", "rollout_depth", "temperature", "selection", "considered",
            "act_iterations", "act_rollout_depth", "act_temperature", "act_budget_steps",
            "act_budget_secs", "emit_samples", "emit_macro_samples", "emit_raw",
            "emit_raw_macro", "macro_net", "counterfactual", "counterfactual_margin",
            "counterfactual_max_prior", "counterfactual_max_steps", "counterfactual_pass",
            "counterfactual_pass_rate",
        ])
))]
#[command(group(
    ArgGroup::new("ppo")
        .multiple(true)
        .requires("run_net")
        .args([
            "like", "greedy", "force_smith", "force_smith_random", "force_skip_cards",
            "force_remove", "force_remove_random", "force_elite", "reward_elite",
            "reward_relic", "reward_gold", "reward_gold_scope", "reward_boss", "explore_rest",
            "explore_map", "emit_ppo", "emit_raw_ppo", "summary", "resolver",
            "resolver_iterations", "resolver_considered", "resolver_budget_steps",
            "resolver_elite_iterations", "resolver_elite_considered",
            "resolver_boss_iterations", "resolver_boss_considered",
        ])
))]
#[allow(
    clippy::struct_excessive_bools,
    reason = "a CLI surface: every independent switch is honestly a bool"
)]
pub struct RunArgs {
    #[command(flatten)]
    pub batch: BatchArgs,
    /// Directory in which to write each run's decision script. A run is
    /// already walked through a `ScriptWriter`, so this keeps what that
    /// wrote rather than replaying anything
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Bank each fight's entry into this library directory — versioned
    /// shards beside a manifest — for replay by `alphaspire fights`. The
    /// bank's distribution is the reaching policy's: a run checkpoint that
    /// climbs banks the act-two and act-three entries a batch dying on floor
    /// nine never reaches
    #[arg(long)]
    pub harvest_fights: Option<PathBuf>,
    /// What each banked entry says its fight as: the setup (the loadout and
    /// the encounter, a few hundred bytes, one loadout under many hands —
    /// what a training bank wants), the exact erased state (tens of
    /// kilobytes, the very fight the run stood at — what paired evaluation
    /// wants), or both
    #[arg(long, value_enum, default_value = "setup")]
    pub harvest_as: HarvestKind,
    /// Forced-win harvesting: play no combat at all — every fight is
    /// harvested at its entry and resolved as won at the price this TOML
    /// config draws (HP loss per tier, act scaling, potion use) — so the
    /// walk banks fights at any depth with no combat competence. Under a
    /// run checkpoint the bank's loadouts are the checkpoint's own decks.
    /// Requires --harvest-fights, refuses every emission flag and --out, and
    /// stamps each entry `forced_win` so no judging instrument mistakes the
    /// bank
    #[arg(long)]
    pub force_wins: Option<PathBuf>,
    /// How many runs each sample shard holds
    #[arg(long, default_value_t = 32)]
    pub shard_runs: usize,
    /// The combat checkpoint: the base path of a trainer export,
    /// `<base>.onnx` beside `<base>.json`. Under --run-net it is what the
    /// resolver plays fights with, frozen for the length of a generation
    /// because a deployed model's behaviour *is* the environment. In a
    /// searched run it guides the combat search inside fights (belief and
    /// act modes) and — with --macro-net — prices the act tree's
    /// fight-entry leaves
    #[arg(long)]
    pub combat_net: Option<PathBuf>,
    /// Play the runs under a PPO run checkpoint: the actor samples its
    /// out-of-combat decisions from it and prices its states with it, and
    /// the resolver answers the fights. The reward the runs pay is read off
    /// the checkpoint's own provenance, which must say `scope: macro` and
    /// name a run reward this build can pay — an act-boundary checkpoint
    /// would otherwise load and make every advantage in the generation
    /// wrong without anything crashing. Requires --combat-net
    #[arg(long, help_heading = "Play under a run checkpoint")]
    pub run_net: Option<PathBuf>,
    /// Act mode: guide the *act tree* with a run-net checkpoint — priors at
    /// every macro node and act-boundary value at every macro leaf. Refused
    /// unless its provenance says `scope: macro` and
    /// `value_semantics: act-boundary-v2`, and requires --combat-net, which
    /// prices the fight-entry leaves. With both, no act-tree leaf is ever
    /// scored by playing a fight at random
    #[arg(long, help_heading = "Searched play")]
    pub macro_net: Option<PathBuf>,
    /// Determinization mode: fair belief analysis over sampled worlds
    /// (combat-scoped, v0), act mode (act-v0.5) — out-of-combat decisions
    /// searched over act-scoped worlds, in-combat decisions kept by the
    /// combat belief search — or clairvoyant coverage fuzzing on the true
    /// state, which is the validation mode and refuses both a checkpoint
    /// and the emission flags
    #[arg(
        long,
        value_enum,
        default_value = "belief",
        help_heading = "Searched play"
    )]
    pub mode: SearchMode,
    /// MCTS rollouts per decision
    #[arg(long, default_value_t = 64, help_heading = "Searched play")]
    pub iterations: u32,
    /// Maximum depth of each rollout
    #[arg(long, default_value_t = 30, help_heading = "Searched play")]
    pub rollout_depth: u32,
    /// Root action sampling temperature (ignored under Gumbel selection,
    /// whose answer is already a sample from the improved policy)
    #[arg(long, default_value_t = 0.5, help_heading = "Searched play")]
    pub temperature: f64,
    /// Tree policy: UCT, or Gumbel-top-k with sequential halving at the root
    /// and completed-Q selection inside the tree
    #[arg(
        long,
        value_enum,
        default_value = "gumbel",
        help_heading = "Searched play"
    )]
    pub selection: SelectionPolicy,
    /// How many root actions Gumbel spreads its budget over
    #[arg(long, default_value_t = 16, help_heading = "Searched play")]
    pub considered: usize,
    /// Act mode: MCTS rollouts per out-of-combat decision (defaults to
    /// --iterations)
    #[arg(long, help_heading = "Searched play")]
    pub act_iterations: Option<u32>,
    /// Act mode: maximum depth of each out-of-combat rollout, priced in
    /// actions across rooms rather than turns of one fight
    #[arg(long, default_value_t = 100, help_heading = "Searched play")]
    pub act_rollout_depth: u32,
    /// Act mode: root sampling temperature over the act tree's visit counts
    /// (ignored under Gumbel selection, as --temperature is). Zero by
    /// default — out-of-combat decisions play for strength, and no training
    /// samples are recorded there, so sampling diversity buys nothing and
    /// costs floors
    #[arg(long, default_value_t = 0.0, help_heading = "Searched play")]
    pub act_temperature: f64,
    /// Act mode: engine steps the act tree may spend over one run before
    /// its macro decisions downgrade to the rollout policy. A soft cap — the
    /// run always walks to its own end — and a reproducible one: a run under
    /// it is still a function of its seed pair, so batches still compare and
    /// shards stay byte-identical across `--jobs`. Off by default
    #[arg(long, help_heading = "Searched play")]
    pub act_budget_steps: Option<u64>,
    /// Act mode: wall-clock seconds one run may spend before its macro
    /// decisions downgrade to the rollout policy. The circuit breaker for a
    /// batch under a hard timeout — and *not* reproducible, since what it
    /// downgrades depends on what else the box is doing, so it is refused
    /// alongside sample emission. Off by default
    #[arg(long, help_heading = "Searched play")]
    pub act_budget_secs: Option<f64>,
    /// Write expert-iteration training samples into this directory, as
    /// versioned shards beside a manifest (belief and act modes only: a
    /// clairvoyant search's policy is a teacher that peeked; act mode
    /// records the in-combat decisions its combat search searched —
    /// out-of-combat ones go to --emit-macro-samples)
    #[arg(long, help_heading = "Searched play")]
    pub emit_samples: Option<PathBuf>,
    /// Act mode: write the *macro* training samples — one per searched
    /// out-of-combat decision, with the act-boundary horizon score as z —
    /// into this directory. Its own directory by construction: the run net
    /// and the combat net are different nets over different horizons, and
    /// mixing their z in one set would train a checkpoint on two meanings of
    /// one number
    #[arg(long, help_heading = "Searched play")]
    pub emit_macro_samples: Option<PathBuf>,
    /// Write every decision --emit-samples would encode *as recorded* — the
    /// observation, the canonical actions, pi and z — as shards `alphaspire
    /// encode` turns into training samples under any build's policy
    /// encoding. The search is paid once; each encoding reads it
    #[arg(long, help_heading = "Searched play")]
    pub emit_raw: Option<PathBuf>,
    /// The same for the macro decisions --emit-macro-samples would encode
    #[arg(long, help_heading = "Searched play")]
    pub emit_raw_macro: Option<PathBuf>,
    #[command(flatten)]
    pub counterfactual: CounterfactualArgs,
    #[command(flatten)]
    pub fights: ResolverArgs,
    /// Write the episodes as encoded training samples into this directory —
    /// versioned shards beside a manifest, under `scope: macro`,
    /// `value_semantics: run-return-v1` and `training_mode: ppo`
    #[arg(long, help_heading = "Play under a run checkpoint")]
    pub emit_ppo: Option<PathBuf>,
    /// Write the same episodes *as recorded* into this directory, for
    /// `alphaspire encode` to read under any build's policy encoding — see
    /// --emit-raw. Both flags may be passed together
    #[arg(long, help_heading = "Play under a run checkpoint")]
    pub emit_raw_ppo: Option<PathBuf>,
    /// Where the batch summary is written (default: `summary.json` inside the
    /// emit directory). It is what the trainer reads its rollout scalars
    /// from, and a batch that emits no shards still writes one where a path
    /// is named
    #[arg(long, help_heading = "Play under a run checkpoint")]
    pub summary: Option<PathBuf>,
    /// Copy the seed, character, ascension and complete unlock profile from
    /// this recording or decision script. The copied configuration overrides
    /// the corresponding batch flags and requires exactly one run
    #[arg(
        long,
        value_name = "TRACE",
        help_heading = "Play under a run checkpoint"
    )]
    pub like: Option<PathBuf>,
    /// Play the run checkpoint by argmax instead of sampling from it: what a
    /// deployed policy does, and what `match macro` grades. Refused with the
    /// emit flags — an argmax trajectory records a certainty the checkpoint
    /// never expressed, and PPO's ratio against it would mean nothing
    #[arg(long, help_heading = "Play under a run checkpoint")]
    pub greedy: bool,
    /// Ablation: at a rest site that offers a smith, take the checkpoint's
    /// best smith plan instead of whatever it preferred, whenever hit
    /// points stand at or above this fraction of maximum (bare flag: zero,
    /// smith everywhere). Only beside --greedy — an override under sampling
    /// is neither the policy's play nor the override's measurement
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "0.0",
        help_heading = "Play under a run checkpoint"
    )]
    pub force_smith: Option<f64>,
    /// With --force-smith: smith a uniformly random upgradable card instead
    /// of the checkpoint's preferred one — the control that separates the
    /// timing of a smith from the choice of its card
    #[arg(
        long,
        requires = "force_smith",
        help_heading = "Play under a run checkpoint"
    )]
    pub force_smith_random: bool,
    /// Ablation: from this act (numbered from one) onward, never answer a
    /// card reward's claim — gold, potions, relics and the exit stay the
    /// policy's own play. Only beside --greedy
    #[arg(long, help_heading = "Play under a run checkpoint")]
    pub force_skip_cards: Option<u32>,
    /// Ablation: buy a card removal wherever a shop offers one the policy
    /// can afford, the checkpoint choosing the card. Only beside --greedy
    #[arg(long, help_heading = "Play under a run checkpoint")]
    pub force_remove: bool,
    /// With --force-remove: remove a uniformly random card instead of the
    /// checkpoint's preferred one — the control that separates buying
    /// removals at all from knowing what to remove
    #[arg(
        long,
        requires = "force_remove",
        help_heading = "Play under a run checkpoint"
    )]
    pub force_remove_random: bool,
    /// Ablation: walk into an elite the map offers as a next room, the
    /// checkpoint choosing among elites where there are several —
    /// `everywhere` (whatever stands beside it) or `over-fights` (only where
    /// every other next room is a fight or an unknown room). Only beside
    /// --greedy
    #[arg(
        long,
        value_name = "WHERE",
        help_heading = "Play under a run checkpoint"
    )]
    pub force_elite: Option<ForceElite>,
    /// What an elite fight won pays on top of the floors-and-outcome
    /// reward. The reward flags are optional: a run always pays the reward
    /// its checkpoint was trained under, read off the provenance. Given,
    /// they state that reward — every term, the unstated ones as zero —
    /// and a statement naming any other reward is refused, so a driver
    /// cannot run under a reward other than the one written in it
    #[arg(
        long,
        value_name = "WEIGHT",
        help_heading = "Play under a run checkpoint"
    )]
    pub reward_elite: Option<f64>,
    /// What each relic gained pays, from any source; optional, as
    /// --reward-elite
    #[arg(
        long,
        value_name = "WEIGHT",
        help_heading = "Play under a run checkpoint"
    )]
    pub reward_relic: Option<f64>,
    /// What each gold gained pays, from any source; spending is not
    /// charged. Optional, as --reward-elite
    #[arg(
        long,
        value_name = "WEIGHT",
        help_heading = "Play under a run checkpoint"
    )]
    pub reward_gold: Option<f64>,
    /// Which gold --reward-gold pays for: `any` (every gold collected) or
    /// `rewards` (only gold claimed off a fight's reward screen — event and
    /// Neow gold is left to the outcome). Optional, as --reward-elite
    #[arg(
        long,
        value_name = "SCOPE",
        help_heading = "Play under a run checkpoint"
    )]
    pub reward_gold_scope: Option<GoldScopeArg>,
    /// What a boss killed pays, by the act it was killed in, as three
    /// slash-separated weights `ACT1/ACT2/ACT3`; the final kill is paid on
    /// top of the victory. Optional, as --reward-elite
    #[arg(
        long,
        value_name = "ACT1/ACT2/ACT3",
        value_parser = parse_boss_weights,
        help_heading = "Play under a run checkpoint"
    )]
    pub reward_boss: Option<alphaspire::reward::BossWeights>,
    /// Hand this fraction of the sampled rest-site distribution to uniform
    /// over the screen's plans, and record the mix as the behaviour policy.
    /// Exploration for the smith the collapsed habit never takes; meaningful
    /// only for sampled play, so refused beside --greedy
    #[arg(long, help_heading = "Play under a run checkpoint")]
    pub explore_rest: Option<f64>,
    /// The same mixing fraction over map-navigation decisions: exploration
    /// for the paths — elite branches above all — a risk-averse habit stops
    /// walking. Refused beside --greedy for the same reason
    #[arg(long, help_heading = "Play under a run checkpoint")]
    pub explore_map: Option<f64>,
}

impl RunArgs {
    /// The reward the flags state, where any of them was given: every term
    /// named, the unstated ones zero. `None` where no reward flag was
    /// given, and the checkpoint's own reward stands unasserted.
    #[must_use]
    pub fn stated_reward(&self) -> Option<alphaspire::reward::RunTerms> {
        let stated = self.reward_elite.is_some()
            || self.reward_relic.is_some()
            || self.reward_gold.is_some()
            || self.reward_gold_scope.is_some()
            || self.reward_boss.is_some();
        stated.then(|| alphaspire::reward::RunTerms {
            elite: self.reward_elite.unwrap_or(0.0),
            relic: self.reward_relic.unwrap_or(0.0),
            gold: self.reward_gold.unwrap_or(0.0),
            gold_rewards_only: self.reward_gold_scope == Some(GoldScopeArg::Rewards),
            boss: self.reward_boss.unwrap_or([0.0; 3]),
        })
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum SearchMode {
    TrueState,
    Belief,
    Act,
}

/// Which gold `--reward-gold` pays for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum GoldScopeArg {
    /// Every gold the run collects.
    Any,
    /// Only gold claimed off a reward screen.
    Rewards,
}

impl From<GoldScopeArg> for alphaspire::reward::GoldScope {
    fn from(scope: GoldScopeArg) -> Self {
        match scope {
            GoldScopeArg::Any => Self::Any,
            GoldScopeArg::Rewards => Self::Rewards,
        }
    }
}

/// `--reward-boss ACT1/ACT2/ACT3`: one finite weight per act.
fn parse_boss_weights(text: &str) -> Result<alphaspire::reward::BossWeights, String> {
    let weights: Vec<f64> = text
        .split('/')
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .ok()
                .filter(|weight| weight.is_finite())
                .ok_or_else(|| format!("`{part}` is not a finite weight"))
        })
        .collect::<Result<_, _>>()?;
    <[f64; 3]>::try_from(weights).map_err(|_| "expected three weights, ACT1/ACT2/ACT3".to_owned())
}

/// Where `--force-elite` applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ForceElite {
    /// Wherever the map offers an elite, whatever stands beside it.
    Everywhere,
    /// Only where every other next room is a fight or an unknown room.
    OverFights,
}

impl RunArgs {
    /// Whether any flag asks the batch to write training data of either
    /// kind.
    fn emitting(&self) -> bool {
        self.emit_samples.is_some()
            || self.emit_macro_samples.is_some()
            || self.emit_raw.is_some()
            || self.emit_raw_macro.is_some()
            || self.emit_ppo.is_some()
            || self.emit_raw_ppo.is_some()
    }

    /// The forced-win config, where the walk skips its fights, refused
    /// beside anything it would make worthless.
    ///
    /// A forced-win walk exists for its bank and for nothing else: no real
    /// game replays a skipped fight, and decisions made where fights cost
    /// only the config's draw must never become training data.
    fn force_wins(&self) -> Option<alphaspire::forcewins::ForceWins> {
        let config = self.force_wins.as_ref().map(|path| {
            alphaspire::forcewins::ForceWins::load(path)
                .unwrap_or_else(|error| die(&format!("--force-wins {error}")))
        })?;
        if self.harvest_fights.is_none() {
            die("--force-wins exists for the bank it leaves, so it requires --harvest-fights");
        }
        if self.out.is_some() {
            die("--force-wins walks are not scriptable, so --out has nothing to keep");
        }
        if self.emitting() {
            die("--force-wins plays no combat and skews every macro decision: emit nothing");
        }
        Some(config)
    }
}

/// Plays the batch under whichever macro the command line named.
pub fn command(args: &RunArgs) -> ! {
    let force_wins = args.force_wins();
    if args.run_net.is_some() {
        ppo::command(args, force_wins)
    } else {
        searched::command(args, force_wins.as_ref())
    }
}
