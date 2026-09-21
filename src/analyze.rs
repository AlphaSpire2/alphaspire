//! Post-game analysis of a recorded run.
//!
//! A `.sts2pgn` trace — a recording of the real game, or a decision script
//! this crate emitted — is replayed through the simulator one record at a
//! time. Before each recorded decision is applied, the state the player
//! stood on is evaluated: inside a fight the combat checkpoint's value and a
//! belief search over the candidates, with the recorded move pinned into the
//! search so it always has a searched value beside the best one; outside a
//! fight the run checkpoint's critic and policy over the permitted plans,
//! with a one-ply lookahead wherever a plan's outcome is fully determined.
//! Every fight is also played out by the combat resolver from the state the
//! player entered it on, in the same world, so what the net would have spent
//! stands beside what the player spent.
//!
//! The evaluators see only what a player could see. The replay is driven on
//! the true state — it has to be, to reproduce the run — but a combat search
//! goes through the belief erasure and a macro pricing reads the agent
//! observation, the same boundary self-play keeps.
//!
//! What comes out is data, not a report: one summary object per trace and
//! one line per decision, under [`ANALYSIS_FORMAT`]. Thresholds, glyphs and
//! layout belong to whatever renders those files.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use sts2_engine::{
    Action, AgentObservation, CardFingerprint, DecisionContext, RunResult, Simulator,
};
use sts2_rng::MegaRandom;

use crate::actor::{Resolver, TierBudget};
use crate::encoding::PolicyEncoder;
use crate::env::fight_over;
use crate::library::Tier;
use crate::net::{Evaluate, PolicyValueNet};
use crate::objective::{CombatStrength, Objective};
use crate::plan::{ActionPlan, is_forced_step, permitted_plans};
use crate::policy::permitted_actions;
use crate::reward::{RunReward, RunTerms};
use crate::search::{BeliefSearch, Budget, Gumbel, RootPolicy, SearchConfig, canonical_actions};

/// The version of the two files this module writes. A field added keeps the
/// number; a field whose meaning changes bumps it.
pub const ANALYSIS_FORMAT: u32 = 1;

/// Engine steps one fight playout may take before it is cut off.
const PLAYOUT_STEP_CAP: usize = 2_000;

/// The search budgets an analysis spends: one per enemy tier for the
/// per-decision search and the fight playouts, and the deeper one the
/// flagged decisions are re-searched at.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Budgets {
    pub hallway: TierBudget,
    pub elite: TierBudget,
    pub boss: TierBudget,
    pub deep: TierBudget,
    /// How many combat decisions, ranked by the first pass's played-versus-
    /// best gap, are re-searched at the deep budget.
    pub deep_top: usize,
    /// A first-pass gap at or above which a decision is re-searched whatever
    /// its rank.
    pub deep_cutoff: f64,
    /// Visits a candidate needs before it can be named the best move.
    pub min_visits: u64,
}

impl Budgets {
    const fn tier(&self, tier: Tier) -> TierBudget {
        match tier {
            Tier::Hallway => self.hallway,
            Tier::Elite => self.elite,
            Tier::Boss => self.boss,
        }
    }
}

/// Everything an analysis is told besides the trace and the nets.
#[derive(Clone, Copy, Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each switch turns one independent stage of the analysis off"
)]
pub struct Settings {
    pub budgets: Budgets,
    /// Skip every search and playout: value heads and policies only.
    pub net_only: bool,
    /// Play each fight out from its entry state with the combat resolver.
    pub playouts: bool,
    /// Price each macro plan by the critic on the state it leads to, where
    /// that state is fully determined by the plan.
    pub lookahead: bool,
    /// Embed the whole agent observation on every line.
    pub full_observation: bool,
    /// Seeds every draw the analysis itself makes: the belief worlds, the
    /// root schedules, the playouts. Never the game's own seed.
    pub analysis_seed: u64,
    /// The discount the realised macro return is summed under.
    pub return_discount: f64,
    /// Workers the fight playouts and the deep pass are spread over. The
    /// walk itself is sequential, and the output does not depend on the
    /// count: every playout and re-search is seeded where the walk reaches
    /// it, whichever worker runs it.
    pub threads: usize,
}

/// A run checkpoint with the reward it was trained under.
pub struct RunNet {
    pub net: Arc<PolicyValueNet>,
    pub stem: PathBuf,
    pub semantics: String,
    pub terms: RunTerms,
}

impl RunNet {
    /// Loads the run checkpoint at `stem`, reading the reward terms off its
    /// own provenance so no flag has to restate them.
    pub fn load(stem: &Path, encoder: Arc<PolicyEncoder>) -> Result<Self, String> {
        let (semantics, terms) = crate::reward::checkpoint_terms(stem)?;
        let net = PolicyValueNet::load_run_priced(stem, encoder, &semantics)
            .map_err(|error| format!("{}: {error}", stem.display()))?;
        Ok(Self {
            net: Arc::new(net),
            stem: stem.to_path_buf(),
            semantics,
            terms,
        })
    }
}

/// The checkpoints an analysis evaluates with.
pub struct Nets {
    pub combat: Arc<PolicyValueNet>,
    pub combat_stem: PathBuf,
    /// The run checkpoint, where one was given; without it the macro
    /// decisions are walked but not priced.
    pub run: Option<RunNet>,
}

/// Why an analysis could not start or could not finish reading its trace.
/// A divergence is not one of these: the walk stops and reports it.
#[derive(Debug)]
pub enum AnalyzeError {
    Io(PathBuf, std::io::Error),
    Trace(String),
}

impl fmt::Display for AnalyzeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, error) => write!(formatter, "{}: {error}", path.display()),
            Self::Trace(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for AnalyzeError {}

// ---------------------------------------------------------------------------
// What is written.

/// Which evaluator a decision belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Lane {
    Combat,
    Macro,
}

/// How the recorded action relates to the candidates that were evaluated.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlayedStatus {
    /// The recorded action is one of the candidates.
    OnMenu,
    /// Legal, but a step the policy layer withholds from its own play, so
    /// it has no candidate and no searched value.
    OffMenu { reason: String },
    /// The engine accepted an action the plan list cannot express.
    Unmatched,
}

/// One evaluated option, as the line names it.
#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub index: usize,
    /// The plan in the engine's own terms.
    pub plan: ActionPlan,
    /// The same plan for a reader.
    pub display: String,
    /// The class the recorded action was matched against.
    pub class: String,
    /// The run policy's probability of this plan. Macro lines only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<f64>,
    /// The critic's value of the state this plan leads to, where that state
    /// is fully determined by the plan. Macro lines only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v_after: Option<f64>,
}

/// What one search learned about one candidate.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct SearchedCandidate {
    pub index: usize,
    pub prior: f64,
    pub visits: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_completed: Option<f64>,
    pub pi: f64,
}

/// The root of one search over a combat decision.
#[derive(Clone, Debug, Serialize)]
pub struct SearchTable {
    pub budget: TierBudget,
    pub r: Option<f64>,
    pub q_spread: Option<f64>,
    /// The candidate with the highest completed value among those visited
    /// at least `min_visits` times.
    pub best: Option<usize>,
    /// `q_completed(best) − q_completed(played)`, where both exist.
    pub delta: Option<f64>,
    pub candidates: Vec<SearchedCandidate>,
}

/// A fight played out by the combat resolver.
#[derive(Clone, Debug, Serialize)]
pub struct Playout {
    pub won: bool,
    pub hp_out: i32,
    pub turns: u32,
    pub potions_used: usize,
    pub budget: TierBudget,
    /// The turn on which the resolver first played something other than
    /// what the recording played, where a recording was there to compare
    /// against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diverged_turn: Option<u32>,
    /// The playout ran into the step cap or an engine refusal before the
    /// fight ended.
    pub cut_off: bool,
}

/// The re-search of a flagged decision, with the fight played on from the
/// recorded move and from the best one.
#[derive(Clone, Debug, Serialize)]
pub struct DeepTable {
    #[serde(flatten)]
    pub table: SearchTable,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub playout_after_played: Option<Playout>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub playout_after_best: Option<Playout>,
}

/// One enemy as the player saw it.
#[derive(Clone, Debug, Serialize)]
pub struct EnemySummary {
    pub name: String,
    pub hp: i32,
    pub max_hp: i32,
    pub block: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
}

/// The state a decision was made in, compactly.
#[derive(Clone, Debug, Serialize)]
pub struct StateSummary {
    pub screen: &'static str,
    pub hp: i32,
    pub max_hp: i32,
    pub block: i32,
    pub energy: i32,
    pub gold: i32,
    pub potions: Vec<String>,
    pub relics: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hand: Vec<String>,
    /// Visible pile contents, retained in the compact output so report
    /// consumers do not need the much larger full observation.
    pub draw_pile: Vec<String>,
    pub discard_pile: Vec<String>,
    pub exhaust_pile: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub enemies: Vec<EnemySummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation: Option<AgentObservation>,
}

/// One decision of the trace.
#[derive(Clone, Debug, Serialize)]
pub struct Line {
    /// The trace record that applied the decision: the coordinate.
    pub seq: u64,
    pub lane: Lane,
    pub floor: u32,
    pub act: usize,
    pub room_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fight: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn: Option<u32>,
    /// Only one plan was permitted; nothing was evaluated.
    pub forced: bool,
    pub played_status: PlayedStatus,
    /// The recorded action's index among the candidates, where it is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub played: Option<usize>,
    /// The recorded action itself, whether or not it is a candidate.
    pub played_action: Action,
    pub played_display: String,
    pub state: StateSummary,
    pub candidates: Vec<Candidate>,
    pub tags: Vec<String>,
    // Combat.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shallow: Option<SearchTable>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deep: Option<DeepTable>,
    /// The operative best move: the deep table's where there is one, the
    /// shallow one's otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r_prev_same_turn: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<f64>,
    // Macro.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v_m: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred: Option<usize>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub class_mass: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub g: Option<f64>,
}

/// One fight of the run: how the player came out of it, and how the
/// resolver did from the same entry.
#[derive(Clone, Debug, Serialize)]
pub struct FightRecord {
    pub index: usize,
    pub floor: u32,
    pub act: usize,
    pub tier: Tier,
    pub encounter: String,
    /// The record whose step entered the fight.
    pub entry_seq: u64,
    pub hp_in: i32,
    pub max_hp: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry_r: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry_v: Option<f64>,
    pub human: FightOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub playout: Option<Playout>,
    pub z: f64,
}

/// How a fight ended for whoever played it.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct FightOutcome {
    pub won: bool,
    pub hp_out: i32,
    pub turns: u32,
    pub potions_used: usize,
}

/// One differing field of a divergence.
#[derive(Clone, Debug, Serialize)]
pub struct Difference {
    pub pointer: String,
    pub expected: Option<serde_json::Value>,
    pub actual: Option<serde_json::Value>,
}

/// Where and why the walk stopped short of the trace's end.
#[derive(Clone, Debug, Serialize)]
pub struct Stop {
    pub code: String,
    pub record: u64,
    pub kind: String,
    pub detail: String,
    pub floor: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_matching_checkpoint: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub differences: Vec<Difference>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub rng_counter_deltas: BTreeMap<String, i64>,
}

/// How much of the trace the analysis covers.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Coverage {
    pub records: usize,
    pub applied: usize,
    pub decisions: usize,
    pub forced: usize,
    /// Free reward lines — gold, a potion there was room for — claimed by
    /// the harness before any policy was asked. Applied,
    /// and paid to the macro decision standing before them, but no
    /// decision and no line.
    pub claimed: usize,
    pub evaluated: usize,
    pub off_menu: usize,
    pub unmatched: usize,
    pub degraded: usize,
    /// The recording's observations were checked and every one agreed.
    /// False for a script, which carries none.
    pub verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub divergence: Option<Stop>,
    /// Reloads the recording walked through, and the applied actions they
    /// rewound: played between the last save and the quit, gone from the
    /// resumed timeline and so from the lines and fights.
    pub resumes: usize,
    pub rewound: usize,
    pub recorded_result: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulated_result: Option<String>,
    pub lookahead_screens: BTreeSet<&'static str>,
}

/// What the trace says about itself.
#[derive(Clone, Debug, Serialize)]
pub struct TraceInfo {
    pub path: PathBuf,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recorder_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub producer: Option<String>,
    pub seed: String,
    pub character: String,
    pub ascension: u8,
    pub mode: String,
    pub floor_reached: u32,
    pub act_reached: usize,
}

/// One checkpoint as the analysis loaded it.
#[derive(Clone, Debug, Serialize)]
pub struct NetInfo {
    pub stem: PathBuf,
    pub value_semantics: String,
}

/// The evaluators and the versions their answers depend on.
#[derive(Clone, Debug, Serialize)]
pub struct NetsInfo {
    pub combat: NetInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<NetInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_reward: Option<RunTerms>,
    pub belief_model: &'static str,
    pub policy_encoding_version: u32,
    pub observation_version: u32,
}

/// The budgets and switches the analysis ran under.
#[derive(Clone, Debug, Serialize)]
pub struct BudgetsInfo {
    #[serde(flatten)]
    pub budgets: Budgets,
    pub analysis_seed: u64,
    pub net_only: bool,
    pub playouts: bool,
    pub lookahead: bool,
}

/// The discount the realised macro return was summed under, and what the
/// whole run paid.
#[derive(Clone, Debug, Serialize)]
pub struct MacroReturns {
    pub discount: f64,
    pub total_reward: f64,
}

/// Wall time by stage.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Timing {
    pub walk_secs: f64,
    pub playout_secs: f64,
    pub deep_secs: f64,
    pub total_secs: f64,
}

/// The two units the numbers are in, spelled out once.
#[derive(Clone, Debug, Serialize)]
pub struct Units {
    pub combat: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#macro: Option<String>,
}

/// The run summary: everything about one trace that is not a decision.
#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    pub analysis_format: u32,
    pub trace: TraceInfo,
    pub nets: NetsInfo,
    pub budgets: BudgetsInfo,
    pub coverage: Coverage,
    pub fights: Vec<FightRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub macro_returns: Option<MacroReturns>,
    pub timing: Timing,
    pub units: Units,
}

/// One trace, analysed.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub summary: Summary,
    pub lines: Vec<Line>,
}

impl Analysis {
    /// Writes `<stem>.analysis.json` and `<stem>.analysis.jsonl` into
    /// `directory`, answering with the two paths.
    pub fn write(&self, directory: &Path, stem: &str) -> std::io::Result<(PathBuf, PathBuf)> {
        std::fs::create_dir_all(directory)?;
        let summary = directory.join(format!("{stem}.analysis.json"));
        let lines = directory.join(format!("{stem}.analysis.jsonl"));
        std::fs::write(&summary, serde_json::to_string_pretty(&self.summary)?)?;
        let mut out = std::io::BufWriter::new(std::fs::File::create(&lines)?);
        for line in &self.lines {
            serde_json::to_writer(&mut out, line)?;
            out.write_all(b"\n")?;
        }
        out.flush()?;
        Ok((summary, lines))
    }

    /// The one-line account of the trace, for the terminal and the batch log.
    #[must_use]
    pub fn summary_line(&self) -> String {
        use std::fmt::Write as _;
        let summary = &self.summary;
        let coverage = &summary.coverage;
        let combat = self
            .lines
            .iter()
            .filter(|line| line.lane == Lane::Combat)
            .count();
        let human: i32 = summary
            .fights
            .iter()
            .map(|fight| fight.hp_in - fight.human.hp_out)
            .sum();
        let mut line = format!(
            "{} act {} floor {} {} | {} decisions ({combat} C / {} M, {} forced) {} fights",
            summary.trace.seed,
            summary.trace.act_reached + 1,
            summary.trace.floor_reached,
            coverage.recorded_result,
            coverage.decisions,
            coverage.decisions - combat,
            coverage.forced,
            summary.fights.len(),
        );
        if summary.fights.iter().any(|fight| fight.playout.is_some()) {
            let net: i32 = summary
                .fights
                .iter()
                .filter_map(|fight| fight.playout.as_ref().map(|p| fight.hp_in - p.hp_out))
                .sum();
            let _ = write!(line, " | hp spent human {human} net {net}");
        }
        if let Some((seq, delta)) = self
            .lines
            .iter()
            .filter_map(|line| line.delta.map(|delta| (line.seq, delta)))
            .max_by(|left, right| left.1.total_cmp(&right.1))
        {
            let _ = write!(line, " | worst Δ {delta:.2} @ seq {seq}");
        }
        let _ = write!(
            line,
            " | analysed {} actions, {}",
            coverage.applied,
            match (&coverage.divergence, coverage.verified) {
                (Some(stop), _) => format!("stopped at record {} ({})", stop.record, stop.code),
                (None, true) => "verified".to_owned(),
                (None, false) => "unverified".to_owned(),
            }
        );
        if coverage.resumes > 0 {
            let _ = write!(
                line,
                " | {} reload{} rewound {} actions",
                coverage.resumes,
                if coverage.resumes == 1 { "" } else { "s" },
                coverage.rewound
            );
        }
        let _ = write!(line, " | {:.0}s", summary.timing.total_secs);
        line
    }
}

// ---------------------------------------------------------------------------
// The walk.

/// Analyses the trace at `path`.
#[allow(
    clippy::too_many_lines,
    reason = "the header read, the record loop and the summary assembly are one sequence"
)]
pub fn analyze_trace(
    path: &Path,
    nets: &Nets,
    settings: Settings,
) -> Result<Analysis, AnalyzeError> {
    let started = Instant::now();
    let bytes = std::fs::read(path).map_err(|error| AnalyzeError::Io(path.to_path_buf(), error))?;
    let sha256 = {
        use sha2::Digest as _;
        format!("sha256:{:x}", sha2::Sha256::digest(&bytes))
    };
    let mut parser = sts2_replay::Parser::new(BufReader::new(std::io::Cursor::new(bytes)))
        .map_err(|error| AnalyzeError::Trace(format!("{}: {error}", path.display())))?;
    check_compatibility(&parser)
        .map_err(|error| AnalyzeError::Trace(format!("{}: {error}", path.display())))?;
    let header = |key: &str| parser.header(key).map(str::to_owned);
    let seed = header("Seed").ok_or_else(|| AnalyzeError::Trace("no Seed header".into()))?;
    let ascension: u8 = header("Ascension")
        .and_then(|text| text.parse().ok())
        .ok_or_else(|| AnalyzeError::Trace("the Ascension header is not a level".into()))?;
    let mode_text = header("Mode").unwrap_or_else(|| "STANDARD".to_owned());
    let mode = if mode_text.eq_ignore_ascii_case("custom") {
        sts2_engine::RunMode::Custom
    } else {
        sts2_engine::RunMode::Standard
    };
    let recorder_version = header("RecorderVersion");
    let producer = header("Producer");
    let recorded_result = header("Result").unwrap_or_else(|| "*".to_owned());

    let mut walk = Walk::new(nets, settings);
    let mut adapter: Option<sts2_replay::ReplayAdapter> = None;
    let mut character = String::new();
    let mut pre: Option<Simulator> = None;
    let mut terminal_token: Option<String> = None;
    for event in &mut parser {
        let event = event.map_err(|error| AnalyzeError::Trace(error.to_string()))?;
        let record = match event {
            sts2_replay::Event::Record(record) => record,
            sts2_replay::Event::Terminal(token) => {
                terminal_token = Some(token);
                break;
            }
        };
        walk.coverage.records += 1;
        if adapter.is_none() {
            let (simulator, played_as) = recorded_run(&seed, ascension, mode, &record)?;
            character = played_as;
            walk.open(&simulator);
            let driver = sts2_replay::ReplayAdapter::new(simulator);
            adapter = Some(if recorder_version.is_some() {
                driver.verifying_observations()
            } else {
                driver
            });
        }
        let driver = adapter.as_mut().expect("the adapter was just opened");
        let role = sts2_replay::record_role(&record.kind);
        if matches!(
            role,
            sts2_replay::TimelineRole::InitiatingAction | sts2_replay::TimelineRole::NestedChoice
        ) && pre.is_none()
        {
            pre = Some(driver.simulator().clone());
        }
        match driver.accept(&record) {
            Ok(sts2_replay::AdapterOutcome::Applied(transition)) => {
                walk.coverage.applied += 1;
                let before = pre.take();
                walk.applied(
                    before,
                    &transition.accepted_action,
                    record.sequence,
                    driver.simulator(),
                );
                if driver.actions_since_save() == 0 {
                    walk.checkpoint();
                }
            }
            // A reload put the run back at its last save: what was played
            // past it is gone from the resumed timeline, and from the walk.
            Ok(sts2_replay::AdapterOutcome::Ignored(_)) if record.kind == "run.resume" => {
                pre = None;
                walk.rewind(driver.resumes(), driver.rewound_actions());
            }
            Ok(sts2_replay::AdapterOutcome::Ignored(_)) => {}
            Err(error) => {
                walk.coverage.divergence = Some(stop_from(&error, driver.simulator()));
                break;
            }
        }
    }
    let Some(mut driver) = adapter else {
        return Err(AnalyzeError::Trace(format!(
            "{}: carries no records to replay",
            path.display()
        )));
    };
    if walk.coverage.divergence.is_none()
        && let Err(error) = driver.verify_pending_observation()
    {
        walk.coverage.divergence = Some(stop_from(&error, driver.simulator()));
    }
    let recorded_result = terminal_token.unwrap_or(recorded_result);
    walk.finish(driver.simulator());
    walk.timing.walk_secs = started.elapsed().as_secs_f64();
    walk.play_out_fights();
    walk.deepen();
    let mut timing = std::mem::take(&mut walk.timing);
    timing.total_secs = started.elapsed().as_secs_f64();

    let final_state = driver.simulator().state();
    let mut coverage = walk.coverage;
    coverage.verified = recorder_version.is_some() && coverage.divergence.is_none();
    coverage.recorded_result = recorded_result;
    coverage.simulated_result = final_state
        .terminal
        .map(|result| format!("{result:?}").to_uppercase());
    coverage.decisions = walk.lines.len();
    let run = final_state.run.as_ref();
    let summary = Summary {
        analysis_format: ANALYSIS_FORMAT,
        trace: TraceInfo {
            path: path.to_path_buf(),
            sha256,
            recorder_version,
            producer,
            seed,
            character,
            ascension,
            mode: mode_text,
            floor_reached: run.map_or(0, |run| run.floor),
            act_reached: run.map_or(0, |run| run.current_act),
        },
        nets: NetsInfo {
            combat: NetInfo {
                stem: nets.combat_stem.clone(),
                value_semantics: crate::objective::VALUE_SEMANTICS.to_owned(),
            },
            run: nets.run.as_ref().map(|run| NetInfo {
                stem: run.stem.clone(),
                value_semantics: run.semantics.clone(),
            }),
            run_reward: nets.run.as_ref().map(|run| run.terms),
            belief_model: sts2_engine::BELIEF_MODEL_VERSION,
            policy_encoding_version: crate::encoding::POLICY_ENCODING_VERSION,
            observation_version: sts2_engine::AGENT_OBSERVATION_VERSION,
        },
        budgets: BudgetsInfo {
            budgets: settings.budgets,
            analysis_seed: settings.analysis_seed,
            net_only: settings.net_only,
            playouts: settings.playouts && !settings.net_only,
            lookahead: settings.lookahead && nets.run.is_some(),
        },
        coverage,
        fights: walk.fights,
        macro_returns: nets.run.as_ref().map(|_| MacroReturns {
            discount: settings.return_discount,
            total_reward: walk.total_reward,
        }),
        timing,
        units: Units {
            combat: format!(
                "{}: won + hp_retained + 0.1·potions, discounted by turns, 0 on defeat",
                crate::objective::VALUE_SEMANTICS
            ),
            r#macro: nets.run.as_ref().map(|run| {
                format!(
                    "{}: discounted return; floor 0.1, victory 10, defeat −5, elite {}, relic {}, gold {}, boss {:?}",
                    run.semantics, run.terms.elite, run.terms.relic, run.terms.gold, run.terms.boss
                )
            }),
        },
    };
    Ok(Analysis {
        summary,
        lines: walk.lines,
    })
}

/// A recording is held to the full compatibility contract. A script carries
/// no recorder, no run id and no mod list, so it is held to the build
/// identity and the run headers it does carry.
pub(crate) fn check_compatibility<R: std::io::BufRead>(
    parser: &sts2_replay::Parser<R>,
) -> Result<(), String> {
    if parser.header("RecorderVersion").is_some() {
        return sts2_replay::validate_compatibility(parser).map_err(|error| error.to_string());
    }
    if parser.header("Producer").is_none() {
        return Err("neither a recording (RecorderVersion) nor a script (Producer)".to_owned());
    }
    let manifest = sts2_core::CompatibilityManifest::pinned()
        .map_err(|error| format!("compatibility manifest: {error}"))?;
    for (key, expected) in [
        ("Format", "STS2PGN".to_owned()),
        ("FormatVersion", "1".to_owned()),
        ("GameVersion", manifest.game.version.clone()),
        ("GameCommit", manifest.game.commit.clone()),
        ("ModelIdHash", manifest.game.model_id_hash.to_string()),
    ] {
        match parser.header(key) {
            Some(found) if found == expected => {}
            Some(found) => return Err(format!("{key} is {found}, this build is {expected}")),
            None => return Err(format!("missing required header {key}")),
        }
    }
    for key in ["Seed", "Ascension", "Players", "Mode"] {
        if parser.header(key).is_none() {
            return Err(format!("missing required header {key}"));
        }
    }
    Ok(())
}

/// Generates the run a trace was played on from its own `run.start`: the
/// seed and level the headers name, the character and the unlock profile
/// the opening state projects. Nothing about the run's later state is read.
pub(crate) fn recorded_run(
    seed: &str,
    ascension: u8,
    mode: sts2_engine::RunMode,
    first: &sts2_replay::Record,
) -> Result<(Simulator, String), AnalyzeError> {
    if first.kind != "run.start" {
        return Err(AnalyzeError::Trace(format!(
            "a trace opens with run.start, not {}",
            first.kind
        )));
    }
    let players = first
        .payload
        .pointer("/data/state/players")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if players != 1 {
        return Err(AnalyzeError::Trace(format!(
            "a trace of {players} players is outside what the simulator replays"
        )));
    }
    let unlocks: sts2_core::UnlockState = first
        .payload
        .pointer("/data/state/players/0/unlocks")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .ok_or_else(|| AnalyzeError::Trace("run.start projects no unlock state".into()))?;
    let preset = sts2_core::UnlockPresetManifest::matching(&unlocks)
        .map_err(|error| AnalyzeError::Trace(error.to_string()))?
        .ok_or_else(|| {
            AnalyzeError::Trace("the recorded profile is not a registered unlock preset".into())
        })?;
    let character = first
        .payload
        .pointer("/data/state/players/0/character_model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| AnalyzeError::Trace("run.start projects no character".into()))?;
    let model: sts2_core::ModelId = character
        .parse()
        .map_err(|_| AnalyzeError::Trace("the character is not a model id".into()))?;
    let simulator = sts2_content::run_on_preset_at(seed, &model, &preset, ascension, mode)
        .map_err(|error| AnalyzeError::Trace(format!("{error:?}")))?;
    Ok((simulator, character.to_owned()))
}

fn stop_from(error: &sts2_replay::AdapterError, simulator: &Simulator) -> Stop {
    let divergence = error.divergence.as_deref();
    Stop {
        code: format!("{:?}", error.code),
        record: error.record_sequence,
        kind: error.record_kind.clone(),
        detail: error.detail.clone(),
        floor: floor_of(simulator),
        last_matching_checkpoint: divergence.and_then(|d| d.last_matching_checkpoint),
        differences: divergence.map_or_else(Vec::new, |d| {
            d.differences
                .iter()
                .take(3)
                .map(|difference| Difference {
                    pointer: difference.pointer.clone(),
                    expected: difference.expected.clone(),
                    actual: difference.actual.clone(),
                })
                .collect()
        }),
        rng_counter_deltas: divergence.map_or_else(BTreeMap::new, |d| d.rng_counter_deltas.clone()),
    }
}

/// A fight the walk is inside.
#[derive(Clone)]
struct OpenFight {
    index: usize,
    entry: Simulator,
    entry_seq: u64,
    tier: Tier,
    hp_in: i32,
    /// Every action the recording took inside it, in order.
    human_actions: Vec<Action>,
    potions_used: usize,
    turns: u32,
    /// The line of its first evaluated decision, where it has one yet.
    first_line: Option<usize>,
}

/// A multi-step plan whose first step has been applied and whose second is
/// the next record.
#[derive(Clone)]
struct PendingPlan {
    line: usize,
    plans: Vec<(usize, ActionPlan)>,
}

/// One belief-search tree over the combat net at `budget`.
fn tree(net: &Arc<PolicyValueNet>, budget: TierBudget) -> BeliefSearch<CombatStrength> {
    BeliefSearch::new(
        SearchConfig {
            iterations: budget.iterations,
            ..SearchConfig::default()
        },
        CombatStrength::default(),
    )
    .selecting(Box::new(Gumbel {
        considered: budget.considered,
        ..Gumbel::default()
    }))
    .with_net(Arc::clone(net) as Arc<dyn Evaluate>)
}

/// The walk's search trees, one per tier budget, built once per trace. The
/// deep pass builds its own, one per re-searched decision.
struct Trees {
    hallway: BeliefSearch<CombatStrength>,
    elite: BeliefSearch<CombatStrength>,
    boss: BeliefSearch<CombatStrength>,
}

impl Trees {
    fn build(net: &Arc<PolicyValueNet>, budgets: Budgets) -> Self {
        Self {
            hallway: tree(net, budgets.hallway),
            elite: tree(net, budgets.elite),
            boss: tree(net, budgets.boss),
        }
    }

    fn for_tier(&mut self, tier: Tier) -> &mut BeliefSearch<CombatStrength> {
        match tier {
            Tier::Hallway => &mut self.hallway,
            Tier::Elite => &mut self.elite,
            Tier::Boss => &mut self.boss,
        }
    }
}

/// A fight playout the walk owes, run once the walk is over.
struct PlayoutJob {
    fight: usize,
    entry: Simulator,
    tier: Tier,
    human: Vec<Action>,
    /// Drawn where the walk reached the fight's end, so the playout's world
    /// is the same whichever worker runs it.
    seed: u64,
}

/// A flagged combat decision the deep pass owes.
struct DeepJob {
    line: usize,
    before: Simulator,
    actions: Vec<Action>,
    played: usize,
    /// The re-search's stream, the playout after the played move, the
    /// playout after the best: drawn in trace order, so no two decisions
    /// share a stream and no worker's timing changes one.
    seeds: [u64; 3],
}

/// The walk's bookkeeping as it stood on the state the build last saved:
/// what a reload puts it back to. A save never lands inside a fight, so the
/// lines, fights and playouts are lengths to cut back to.
#[derive(Clone)]
struct Checkpoint {
    lines: usize,
    fights: usize,
    playout_jobs: usize,
    open_fight: Option<OpenFight>,
    pending: Option<PendingPlan>,
    reward: Option<RunReward>,
    since_macro: f64,
    last_macro: Option<usize>,
    total_reward: f64,
    last_combat: Option<(usize, u32, Option<f64>)>,
    pending_draw: Option<usize>,
    coverage: Coverage,
}

struct Walk<'a> {
    saved: Option<Checkpoint>,
    nets: &'a Nets,
    settings: Settings,
    trees: Option<Trees>,
    rng: MegaRandom,
    lines: Vec<Line>,
    /// The state each combat line was evaluated on, kept for the re-search;
    /// `None` where the line was forced, macro, or not searched.
    pres: Vec<Option<Simulator>>,
    fights: Vec<FightRecord>,
    /// The fight playouts owed, in fight order.
    playout_jobs: Vec<PlayoutJob>,
    open_fight: Option<OpenFight>,
    pending: Option<PendingPlan>,
    reward: Option<RunReward>,
    /// Reward paid since the last macro decision, owed to it.
    since_macro: f64,
    last_macro: Option<usize>,
    total_reward: f64,
    /// The last combat decision's fight, turn and root value: the input of
    /// the intra-turn drop.
    last_combat: Option<(usize, u32, Option<f64>)>,
    /// The draw pile's size at the last combat decision, so the step that
    /// followed can be tagged where it drew.
    pending_draw: Option<usize>,
    coverage: Coverage,
    timing: Timing,
}

impl<'a> Walk<'a> {
    fn new(nets: &'a Nets, settings: Settings) -> Self {
        Self {
            saved: None,
            nets,
            settings,
            trees: (!settings.net_only).then(|| Trees::build(&nets.combat, settings.budgets)),
            rng: MegaRandom::new(settings.analysis_seed),
            lines: Vec::new(),
            pres: Vec::new(),
            fights: Vec::new(),
            playout_jobs: Vec::new(),
            open_fight: None,
            pending: None,
            reward: None,
            since_macro: 0.0,
            last_macro: None,
            total_reward: 0.0,
            last_combat: None,
            pending_draw: None,
            coverage: Coverage::default(),
            timing: Timing::default(),
        }
    }

    /// The run stands on its opening state.
    fn open(&mut self, simulator: &Simulator) {
        self.reward = self.nets.run.as_ref().map(|run| {
            RunReward::starting(simulator)
                .with_terms(run.terms.elite, run.terms.relic, run.terms.gold)
                .with_gold_scope(run.terms.scope())
                .with_boss_terms(run.terms.boss)
        });
        // An unplayed run already has a save to come back to.
        self.checkpoint();
    }

    /// The build just saved: a reload from here on comes back to this.
    fn checkpoint(&mut self) {
        self.saved = Some(Checkpoint {
            lines: self.lines.len(),
            fights: self.fights.len(),
            playout_jobs: self.playout_jobs.len(),
            open_fight: self.open_fight.clone(),
            pending: self.pending.clone(),
            reward: self.reward,
            since_macro: self.since_macro,
            last_macro: self.last_macro,
            total_reward: self.total_reward,
            last_combat: self.last_combat,
            pending_draw: self.pending_draw,
            coverage: self.coverage.clone(),
        });
    }

    /// The recording reloaded its save: everything walked since the last
    /// checkpoint is dropped, as if it had not been played. `resumes` and
    /// `rewound` are the replay's running totals.
    fn rewind(&mut self, resumes: usize, rewound: usize) {
        let Some(saved) = self.saved.clone() else {
            return;
        };
        self.lines.truncate(saved.lines);
        self.pres.truncate(saved.lines);
        self.fights.truncate(saved.fights);
        self.playout_jobs.truncate(saved.playout_jobs);
        self.open_fight = saved.open_fight;
        self.pending = saved.pending;
        self.reward = saved.reward;
        self.since_macro = saved.since_macro;
        self.last_macro = saved.last_macro;
        self.total_reward = saved.total_reward;
        self.last_combat = saved.last_combat;
        self.pending_draw = saved.pending_draw;
        let records = self.coverage.records;
        self.coverage = saved.coverage;
        self.coverage.records = records;
        self.coverage.resumes = resumes;
        self.coverage.rewound = rewound;
        // A plan whose second step was answered before the quit is open
        // again, for the resumed timeline to answer.
        if let Some(pending) = &self.pending {
            let line = &mut self.lines[pending.line];
            line.played = None;
            line.played_status = PlayedStatus::OnMenu;
        }
    }

    /// One recorded action was applied: `before` is the state it was taken
    /// on, `after` the state it produced.
    fn applied(&mut self, before: Option<Simulator>, played: &Action, seq: u64, after: &Simulator) {
        let Some(before) = before else {
            // No state was kept for this record, which the role table said
            // could not apply; the decision is counted but not evaluated.
            self.coverage.unmatched += 1;
            self.after_step(played, seq, after);
            return;
        };
        if let Some(pending) = self.pending.take() {
            let matched = pending
                .plans
                .iter()
                .find(|(_, plan)| {
                    plan.steps()
                        .nth(1)
                        .is_some_and(|step| same_class(step, played))
                })
                .map(|(index, _)| *index);
            if let Some(index) = matched {
                let line = &mut self.lines[pending.line];
                line.played = Some(index);
                line.played_status = PlayedStatus::OnMenu;
                self.after_step(played, seq, after);
                return;
            }
            let line = &mut self.lines[pending.line];
            line.played_status = PlayedStatus::OffMenu {
                reason: "opened a screen and answered it with nothing the plan list named".into(),
            };
            self.coverage.off_menu += 1;
        }
        if fight_over(&before) && is_forced_step(&before, played) {
            // The harness's own step, on a script this crate wrote or a
            // recording where the player took the free line first: nothing
            // was decided, so nothing is priced. The reward it pays still
            // lands on the decision standing before it.
            self.coverage.claimed += 1;
        } else if fight_over(&before) {
            self.macro_decision(&before, played, seq);
        } else {
            self.combat_decision(&before, played, seq);
        }
        self.after_step(played, seq, after);
    }

    /// Bookkeeping on the state a step produced: the reward it paid, the
    /// fight it entered or ended.
    fn after_step(&mut self, played: &Action, seq: u64, after: &Simulator) {
        if let Some(reward) = self.reward.as_mut() {
            let paid = reward.paid(after);
            self.since_macro += paid;
            self.total_reward += paid;
        }
        let live = !fight_over(after);
        if let Some(size) = self.pending_draw.take()
            && live
            && after.agent_observation().draw_pile_size < size
            && let Some(line) = self.lines.last_mut()
        {
            line.tags.push("drew".to_owned());
        }
        if let Some(open) = self.open_fight.as_mut() {
            open.human_actions.push(played.clone());
            open.potions_used += usize::from(matches!(played, Action::UsePotion { .. }));
            if !live {
                self.settle_fight(after);
            }
        } else if live {
            self.open_fight = Some(OpenFight {
                index: self.fights.len(),
                entry: after.clone(),
                entry_seq: seq,
                tier: tier_of(after),
                hp_in: player_hp(after).0,
                human_actions: Vec::new(),
                potions_used: 0,
                turns: 0,
                first_line: None,
            });
        }
    }

    fn settle_fight(&mut self, after: &Simulator) {
        let Some(open) = self.open_fight.take() else {
            return;
        };
        let won = after.state().terminal != Some(RunResult::Defeat);
        let (hp_out, max_hp) = player_hp(after);
        let hp_out = hp_out.max(0);
        let z = CombatStrength::default().peek(after);
        let (entry_r, entry_v) = open.first_line.map_or((None, None), |index| {
            let line = &self.lines[index];
            (line.shallow.as_ref().and_then(|table| table.r), line.v)
        });
        for line in &mut self.lines {
            if line.fight == Some(open.index) {
                line.z = Some(z);
            }
        }
        let entry_state = open.entry.state();
        let floor = entry_state.run.as_ref().map_or(0, |run| run.floor);
        let act = entry_state.run.as_ref().map_or(0, |run| run.current_act);
        let encounter = entry_state
            .combat
            .as_ref()
            .map_or_else(String::new, |combat| combat.encounter_model.to_string());
        if self.settings.playouts && !self.settings.net_only {
            self.playout_jobs.push(PlayoutJob {
                fight: open.index,
                entry: open.entry,
                tier: open.tier,
                human: open.human_actions,
                seed: self.rng.next_u64(),
            });
        }
        self.fights.push(FightRecord {
            index: open.index,
            floor,
            act,
            tier: open.tier,
            encounter,
            entry_seq: open.entry_seq,
            hp_in: open.hp_in,
            max_hp,
            entry_r,
            entry_v,
            human: FightOutcome {
                won,
                hp_out,
                turns: open.turns,
                potions_used: open.potions_used,
            },
            playout: None,
            z,
        });
    }

    /// The walk is over, on `last`.
    fn finish(&mut self, last: &Simulator) {
        if self.open_fight.is_some() {
            self.settle_fight(last);
        }
        if let Some(index) = self.last_macro.take() {
            self.lines[index].reward = Some(self.since_macro);
            self.since_macro = 0.0;
        }
        // The realised return, summed backwards over the macro lines.
        let discount = self.settings.return_discount;
        let mut suffix = 0.0;
        for line in self.lines.iter_mut().rev() {
            if line.lane == Lane::Macro && !line.forced {
                suffix = line.reward.unwrap_or(0.0) + discount * suffix;
                line.g = Some(suffix);
            } else if line.lane == Lane::Macro {
                suffix = line.reward.unwrap_or(0.0) + discount * suffix;
            }
        }
    }

    fn combat_decision(&mut self, before: &Simulator, played: &Action, seq: u64) {
        let observation = before.agent_observation();
        let actions = canonical_actions(&permitted_actions(before));
        let forced = actions.len() < 2;
        let played_index = actions.iter().position(|action| same_class(action, played));
        let status = match played_index {
            Some(_) => PlayedStatus::OnMenu,
            None if crate::policy::withheld(before, played) => PlayedStatus::OffMenu {
                reason: "withheld by the policy layer".into(),
            },
            None => PlayedStatus::Unmatched,
        };
        let fight = self.open_fight.as_ref().map(|open| open.index);
        let turn = before
            .state()
            .combat
            .as_ref()
            .map(|combat| combat.player.turn);
        let mut line = self.line(
            seq,
            Lane::Combat,
            before,
            &observation,
            forced,
            status,
            played_index,
            played,
            actions
                .iter()
                .enumerate()
                .map(|(index, action)| Candidate {
                    index,
                    plan: ActionPlan::Single(action.clone()),
                    display: display_action(action, &observation),
                    class: crate::search::action_class(action),
                    p: None,
                    v_after: None,
                })
                .collect(),
        );
        line.fight = fight;
        line.turn = turn;
        if let Some(open) = self.open_fight.as_mut() {
            open.turns = open.turns.max(turn.unwrap_or(0));
        }
        let mut kept = None;
        if !forced {
            self.coverage.evaluated += 1;
            line.v = Some(self.nets.combat.state_value(before));
            if let Some(trees) = self.trees.as_mut() {
                let tier = tier_of(before);
                let pins: Vec<Action> = played_index.map(|_| played.clone()).into_iter().collect();
                if let Ok(policy) = trees.for_tier(tier).analyze(before, &pins, &mut self.rng) {
                    let table = table_from(
                        &policy,
                        &actions,
                        self.settings.budgets.tier(tier),
                        played_index,
                        self.settings.budgets.min_visits,
                    );
                    line.best = table.best;
                    line.delta = table.delta;
                    line.shallow = Some(table);
                    kept = Some(before.clone());
                }
            }
            if let Some((last_fight, last_turn, r)) = self.last_combat
                && Some(last_fight) == fight
                && Some(last_turn) == turn
            {
                line.r_prev_same_turn = r;
            }
            if let Some(open) = self.open_fight.as_mut()
                && open.first_line.is_none()
            {
                open.first_line = Some(self.lines.len());
            }
            self.pending_draw = Some(observation.draw_pile_size);
        }
        if let (Some(fight), Some(turn)) = (fight, turn) {
            self.last_combat = Some((fight, turn, line.shallow.as_ref().and_then(|table| table.r)));
        }
        self.lines.push(line);
        self.pres.push(kept);
    }

    fn macro_decision(&mut self, before: &Simulator, played: &Action, seq: u64) {
        let observation = before.agent_observation();
        let plans = permitted_plans(before);
        let forced = plans.len() < 2;
        let leads: Vec<(usize, ActionPlan)> = plans
            .iter()
            .enumerate()
            .filter(|(_, plan)| same_class(plan.lead(), played))
            .map(|(index, plan)| (index, plan.clone()))
            .collect();
        let single = leads
            .iter()
            .find(|(_, plan)| matches!(plan, ActionPlan::Single(_)))
            .map(|(index, _)| *index);
        let (played_index, status, pending) = match (single, leads.is_empty()) {
            (Some(index), _) => (Some(index), PlayedStatus::OnMenu, None),
            (None, false) => (None, PlayedStatus::OnMenu, Some(leads)),
            (None, true) if crate::policy::withheld(before, played) => (
                None,
                PlayedStatus::OffMenu {
                    reason: "withheld by the policy layer".into(),
                },
                None,
            ),
            (None, true) => (None, PlayedStatus::Unmatched, None),
        };
        let mut candidates: Vec<Candidate> = plans
            .iter()
            .enumerate()
            .map(|(index, plan)| Candidate {
                index,
                plan: plan.clone(),
                display: display_plan(plan, &observation),
                class: plan_class(plan),
                p: None,
                v_after: None,
            })
            .collect();
        let mut v_m = None;
        let mut preferred = None;
        let mut class_mass = BTreeMap::new();
        let mut degraded = false;
        if !forced && let Some(run) = self.nets.run.as_ref() {
            self.coverage.evaluated += 1;
            let (priors, value, cause) = run.net.plan_priors_and_value(before, &plans).into_parts();
            degraded = cause.is_some();
            v_m = Some(value);
            for (candidate, prior) in candidates.iter_mut().zip(&priors) {
                let p = f64::from(*prior);
                candidate.p = Some(p);
                *class_mass.entry(candidate.class.clone()).or_insert(0.0) += p;
            }
            preferred = priors
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index);
            if self.settings.lookahead {
                for candidate in &mut candidates {
                    candidate.v_after = lookahead(before, &candidate.plan, run.net.as_ref());
                    if candidate.v_after.is_some() {
                        self.coverage
                            .lookahead_screens
                            .insert(screen_name(before.decision()));
                    }
                }
            }
        }
        let mut line = self.line(
            seq,
            Lane::Macro,
            before,
            &observation,
            forced,
            status,
            played_index,
            played,
            candidates,
        );
        line.v_m = v_m;
        line.preferred = preferred;
        line.class_mass = class_mass;
        if degraded {
            line.tags.push("degraded".into());
            self.coverage.degraded += 1;
        }
        // The reward paid since the last macro decision was owed to it; from
        // here on it is owed to this one.
        if self.reward.is_some() {
            if let Some(index) = self.last_macro {
                self.lines[index].reward = Some(self.since_macro);
            }
            self.since_macro = 0.0;
            self.last_macro = Some(self.lines.len());
        }
        if let Some(plans) = pending {
            self.pending = Some(PendingPlan {
                line: self.lines.len(),
                plans,
            });
        }
        self.lines.push(line);
        self.pres.push(None);
    }

    /// The fields every line has.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per field the lanes share"
    )]
    fn line(
        &mut self,
        seq: u64,
        lane: Lane,
        before: &Simulator,
        observation: &AgentObservation,
        forced: bool,
        status: PlayedStatus,
        played: Option<usize>,
        played_action: &Action,
        candidates: Vec<Candidate>,
    ) -> Line {
        if forced {
            self.coverage.forced += 1;
        }
        match &status {
            PlayedStatus::OnMenu => {}
            PlayedStatus::OffMenu { .. } => self.coverage.off_menu += 1,
            PlayedStatus::Unmatched => self.coverage.unmatched += 1,
        }
        let state = before.state();
        let run = state.run.as_ref();
        let mut tags = Vec::new();
        if let PlayedStatus::OffMenu { .. } = &status {
            tags.push("off_menu".to_owned());
        }
        Line {
            seq,
            lane,
            floor: run.map_or(0, |run| run.floor),
            act: run.map_or(0, |run| run.current_act),
            room_type: run
                .and_then(sts2_engine::RunState::standing_room_type)
                .map(|room| format!("{room:?}")),
            fight: None,
            turn: None,
            forced,
            played_status: status,
            played,
            played_action: played_action.clone(),
            played_display: display_action(played_action, observation),
            state: summarize(before, observation, self.settings.full_observation),
            candidates,
            tags,
            v: None,
            shallow: None,
            deep: None,
            best: None,
            delta: None,
            r_prev_same_turn: None,
            z: None,
            v_m: None,
            preferred: None,
            class_mass: BTreeMap::new(),
            reward: None,
            g: None,
        }
    }

    /// The fight playouts the walk owes, run across the workers and filed
    /// in fight order.
    fn play_out_fights(&mut self) {
        let jobs = std::mem::take(&mut self.playout_jobs);
        if jobs.is_empty() {
            return;
        }
        let started = Instant::now();
        let nets = self.nets;
        let budgets = self.settings.budgets;
        let threads = self.settings.threads;
        let play = |index: usize| {
            let job = &jobs[index];
            play_out_with(
                nets,
                budgets.tier(job.tier),
                job.seed,
                job.entry.clone(),
                None,
                &job.human,
            )
        };
        let fights = &mut self.fights;
        crate::selfplay::indexed(jobs.len(), threads, &play, &mut |index, playout| {
            fights[jobs[index].fight].playout = Some(playout);
        });
        self.timing.playout_secs = started.elapsed().as_secs_f64();
    }

    /// The second pass: the flagged combat decisions re-searched at the deep
    /// budget, with the fight played on from the recorded move and the best.
    /// Each decision is searched on a tree of its own, so the decisions run
    /// across the workers and are filed back in trace order.
    fn deepen(&mut self) {
        if self.trees.is_none() {
            return;
        }
        let started = Instant::now();
        let budgets = self.settings.budgets;
        let mut ranked: Vec<(usize, f64)> = self
            .lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| {
                (line.lane == Lane::Combat && self.pres[index].is_some())
                    .then(|| line.delta.map(|delta| (index, delta)))
                    .flatten()
            })
            .collect();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        let selected: Vec<usize> = ranked
            .iter()
            .enumerate()
            .filter(|(rank, (_, delta))| *rank < budgets.deep_top || *delta >= budgets.deep_cutoff)
            .map(|(_, (index, _))| *index)
            .collect();
        let mut jobs = Vec::with_capacity(selected.len());
        for index in selected {
            let Some(before) = self.pres[index].take() else {
                continue;
            };
            let line = &self.lines[index];
            let Some(played) = line.played else {
                continue;
            };
            let actions: Vec<Action> = line
                .candidates
                .iter()
                .map(|candidate| candidate.plan.lead().clone())
                .collect();
            let seeds = [
                self.rng.next_u64(),
                self.rng.next_u64(),
                self.rng.next_u64(),
            ];
            jobs.push(DeepJob {
                line: index,
                before,
                actions,
                played,
                seeds,
            });
        }
        self.pres.clear();
        let nets = self.nets;
        let threads = self.settings.threads;
        let search = |index: usize| -> Option<DeepTable> {
            let job = &jobs[index];
            let played = &job.actions[job.played];
            let mut rng = MegaRandom::new(job.seeds[0]);
            let policy = tree(&nets.combat, budgets.deep)
                .analyze(&job.before, std::slice::from_ref(played), &mut rng)
                .ok()?;
            let table = table_from(
                &policy,
                &job.actions,
                budgets.deep,
                Some(job.played),
                budgets.min_visits,
            );
            let playout_after_played = Some(play_out_with(
                nets,
                budgets.deep,
                job.seeds[1],
                job.before.clone(),
                Some(played),
                &[],
            ));
            let playout_after_best = table.best.filter(|best| *best != job.played).map(|best| {
                play_out_with(
                    nets,
                    budgets.deep,
                    job.seeds[2],
                    job.before.clone(),
                    Some(&job.actions[best]),
                    &[],
                )
            });
            Some(DeepTable {
                table,
                playout_after_played,
                playout_after_best,
            })
        };
        let lines = &mut self.lines;
        crate::selfplay::indexed(jobs.len(), threads, &search, &mut |index, deep| {
            if let Some(deep) = deep {
                let line = &mut lines[jobs[index].line];
                line.best = deep.table.best;
                line.delta = deep.table.delta;
                line.deep = Some(deep);
            }
        });
        self.timing.deep_secs = started.elapsed().as_secs_f64();
    }
}

/// Plays the fight `start` stands in to its end with the combat resolver at
/// `budget`, in the world `start` is: the same draw order and enemy rolls the
/// recording had. `first` is stepped before the resolver is asked; `human`
/// is the recorded action sequence the playout's own is compared against
/// for the turn it first departs.
fn play_out_with(
    nets: &Nets,
    budget: TierBudget,
    seed: u64,
    mut simulator: Simulator,
    first: Option<&Action>,
    human: &[Action],
) -> Playout {
    let resolver = Resolver::Searched {
        config: SearchConfig {
            iterations: budget.iterations,
            ..SearchConfig::default()
        },
        selection: Gumbel {
            considered: budget.considered,
            ..Gumbel::default()
        },
        budget: Budget::default(),
        elite: None,
        boss: None,
    };
    let mut policy = resolver.build(Arc::clone(&nets.combat) as Arc<dyn Evaluate>);
    let mut stream = MegaRandom::new(seed);
    let mut taken = 0_usize;
    let mut potions_used = 0_usize;
    let mut diverged_turn = None;
    let mut cut_off = false;
    let mut turns = turn_of(&simulator);
    if let Some(action) = first {
        if simulator.step_quietly(action).is_err() {
            cut_off = true;
        }
        taken += 1;
    }
    while !cut_off && simulator.state().terminal.is_none() && !fight_over(&simulator) {
        if taken >= PLAYOUT_STEP_CAP || simulator.legal_actions().is_empty() {
            cut_off = true;
            break;
        }
        turns = turn_of(&simulator);
        let action = policy.choose(&simulator, &mut stream);
        if diverged_turn.is_none()
            && !human.is_empty()
            && !human
                .get(taken)
                .is_some_and(|recorded| same_class(recorded, &action))
        {
            diverged_turn = Some(turns);
        }
        if simulator.step_quietly(&action).is_err() {
            cut_off = true;
            break;
        }
        potions_used += usize::from(matches!(action, Action::UsePotion { .. }));
        taken += 1;
    }
    let state = simulator.state();
    Playout {
        won: !cut_off && state.terminal != Some(RunResult::Defeat),
        hp_out: player_hp(&simulator).0.max(0),
        turns: turns.max(turn_of(&simulator)),
        potions_used,
        budget,
        diverged_turn,
        cut_off,
    }
}

/// The critic's value of the state `plan` leads to, or `None` where reaching
/// it drew on the game's random streams — a roll the player could not have
/// seen the result of, which a fair analysis does not read.
fn lookahead(before: &Simulator, plan: &ActionPlan, net: &dyn Evaluate) -> Option<f64> {
    // A room entered reveals what the act generated for it, and the next
    // act's map is generated on the crossing: both are hidden from the
    // player before the step whether or not the step itself rolls.
    if plan
        .steps()
        .any(|step| matches!(step, Action::ChooseMap { .. } | Action::AdvanceAct))
    {
        return None;
    }
    let mut probe = before.clone();
    for step in plan.steps() {
        let transition = probe.step(step.clone()).ok()?;
        if transition.rng_before != transition.rng_after
            || transition.player_rng_before != transition.player_rng_after
        {
            return None;
        }
    }
    Some(net.state_value(&probe))
}

/// Aligns the root's table onto the decision's candidate list and names the
/// best move and the gap to the played one.
fn table_from(
    policy: &RootPolicy,
    actions: &[Action],
    budget: TierBudget,
    played: Option<usize>,
    min_visits: u64,
) -> SearchTable {
    let candidates: Vec<SearchedCandidate> = actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            let edge = policy
                .actions
                .iter()
                .zip(&policy.candidates)
                .find(|((searched, _), _)| same_class(searched, action));
            match edge {
                Some(((_, pi), root)) => SearchedCandidate {
                    index,
                    prior: root.prior,
                    visits: root.visits,
                    q_mean: (root.visits > 0).then_some(root.mean_value),
                    q_completed: Some(root.completed_value),
                    pi: *pi,
                },
                None => SearchedCandidate {
                    index,
                    prior: 0.0,
                    visits: 0,
                    q_mean: None,
                    q_completed: None,
                    pi: 0.0,
                },
            }
        })
        .collect();
    let best = candidates
        .iter()
        .filter(|candidate| candidate.visits >= min_visits.max(1))
        .filter_map(|candidate| candidate.q_completed.map(|q| (candidate.index, q)))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .or_else(|| {
            candidates
                .iter()
                .filter(|candidate| candidate.visits > 0)
                .filter_map(|candidate| candidate.q_completed.map(|q| (candidate.index, q)))
                .max_by(|left, right| left.1.total_cmp(&right.1))
        });
    let delta = match (best, played) {
        (Some((_, best_q)), Some(played)) => candidates[played]
            .q_completed
            .filter(|_| candidates[played].visits > 0)
            .map(|played_q| (best_q - played_q).max(0.0)),
        _ => None,
    };
    SearchTable {
        budget,
        r: policy.root_value,
        q_spread: policy.q_spread,
        best: best.map(|(index, _)| index),
        delta,
        candidates,
    }
}

// ---------------------------------------------------------------------------
// Reading the state.

fn same_class(left: &Action, right: &Action) -> bool {
    crate::search::action_class(left) == crate::search::action_class(right)
}

fn tier_of(simulator: &Simulator) -> Tier {
    Tier::of(
        simulator
            .state()
            .run
            .as_ref()
            .and_then(sts2_engine::RunState::standing_room_type),
    )
}

fn turn_of(simulator: &Simulator) -> u32 {
    simulator
        .state()
        .combat
        .as_ref()
        .map_or(0, |combat| combat.player.turn)
}

fn floor_of(simulator: &Simulator) -> u32 {
    simulator.state().run.as_ref().map_or(0, |run| run.floor)
}

/// The player's hit points and maximum: off their creature inside a fight,
/// off the run outside one.
fn player_hp(simulator: &Simulator) -> (i32, i32) {
    let state = simulator.state();
    state
        .combat
        .as_ref()
        .and_then(|combat| {
            combat
                .creatures
                .iter()
                .find(|creature| creature.id == combat.player.creature_id)
                .map(|creature| (creature.current_hp, creature.max_hp))
        })
        .unwrap_or((state.run_player.current_hp, state.run_player.max_hp))
}

const fn screen_name(decision: &DecisionContext) -> &'static str {
    match decision {
        DecisionContext::MapNavigation { .. } => "map_navigation",
        DecisionContext::Event { .. } => "event",
        DecisionContext::Shop { .. } => "shop",
        DecisionContext::RestSite { .. } => "rest_site",
        DecisionContext::Treasure { .. } => "treasure",
        DecisionContext::CrystalSphere { .. } => "crystal_sphere",
        DecisionContext::ActTransition { .. } => "act_transition",
        DecisionContext::CombatPriority { .. } => "combat_priority",
        DecisionContext::ChooseCards { .. } => "choose_cards",
        DecisionContext::Rewards { .. } => "rewards",
        DecisionContext::RoomProceed => "room_proceed",
        DecisionContext::Terminal { .. } => "terminal",
    }
}

fn summarize(simulator: &Simulator, observation: &AgentObservation, full: bool) -> StateSummary {
    let (hp, max_hp) = player_hp(simulator);
    StateSummary {
        screen: screen_name(&observation.decision),
        hp,
        max_hp,
        block: observation.block.unwrap_or(0),
        energy: observation.energy.unwrap_or(0),
        gold: observation.gold.unwrap_or(0),
        potions: observation
            .potions
            .iter()
            .flatten()
            .map(|potion| short(&potion.to_string()))
            .collect(),
        relics: observation.relics.len(),
        hand: observation
            .hand
            .iter()
            .map(|card| card_name(&card.fingerprint))
            .collect(),
        draw_pile: observation
            .draw_pile
            .iter()
            .map(|card| card_name(&card.fingerprint))
            .collect(),
        discard_pile: observation
            .discard
            .iter()
            .map(|card| card_name(&card.fingerprint))
            .collect(),
        exhaust_pile: observation
            .exhaust
            .iter()
            .map(|card| card_name(&card.fingerprint))
            .collect(),
        enemies: observation
            .creatures
            .iter()
            .filter(|creature| creature.side != sts2_engine::CombatSide::Player)
            .map(|creature| EnemySummary {
                name: format!(
                    "{}#{}",
                    short(&creature.model_id.to_string()),
                    combat_id(&creature.combat_id)
                ),
                hp: creature.current_hp,
                max_hp: creature.max_hp,
                block: creature.block,
                intent: creature.intent.as_ref().map(|intent| {
                    let parts: Vec<String> = intent
                        .intents
                        .iter()
                        .map(|part| {
                            use std::fmt::Write as _;
                            let mut text = part.intent_type.clone();
                            if let Some(damage) = part.damage {
                                let _ = write!(text, " {damage}");
                                if let Some(repeats) = part.repeats.filter(|r| *r > 1) {
                                    let _ = write!(text, "×{repeats}");
                                }
                            }
                            text
                        })
                        .collect();
                    format!("{} [{}]", intent.move_id, parts.join(", "))
                }),
            })
            .collect(),
        observation: full.then(|| observation.clone()),
    }
}

// ---------------------------------------------------------------------------
// Naming things for a reader.

/// A model id without its namespace: `CARD.BASH` reads `BASH`.
fn short(id: &str) -> String {
    id.split_once('.')
        .map_or_else(|| id.to_owned(), |(_, rest)| rest.to_owned())
}

fn card_name(fingerprint: &CardFingerprint) -> String {
    let mut name = short(&fingerprint.model_id.to_string());
    match fingerprint.upgrade_level {
        0 => {}
        1 => name.push('+'),
        level => {
            use std::fmt::Write as _;
            let _ = write!(name, "+{level}");
        }
    }
    name
}

fn target_name(target: &sts2_engine::TargetHandle) -> String {
    format!(
        "{}#{}",
        short(&target.model_id.to_string()),
        combat_id(&target.combat_id)
    )
}

/// A combat id as the trace writes it.
fn combat_id(id: &impl Serialize) -> String {
    serde_json::to_value(id).map_or_else(|_| "?".to_owned(), |value| value.to_string())
}

fn display_action(action: &Action, observation: &AgentObservation) -> String {
    match action {
        Action::ChooseMap { destination } => {
            let kind = observation
                .map
                .as_ref()
                .and_then(|map| {
                    map.points
                        .iter()
                        .find(|point| point.coord == *destination)
                        .map(|point| format!(" {:?}", point.point_type))
                })
                .unwrap_or_default();
            format!("Map ({},{}){kind}", destination.col, destination.row)
        }
        Action::ChooseEvent { index, option_id } => {
            let label = match &observation.decision {
                DecisionContext::Event { options, .. } => options
                    .get(*index)
                    .map(|option| option.label.clone())
                    .filter(|label| !label.is_empty()),
                _ => None,
            };
            format!("Event: {}", label.unwrap_or_else(|| option_id.clone()))
        }
        Action::BuyShopItem { fingerprint, .. } => match fingerprint {
            sts2_engine::ShopItem::Card(card) => format!("Buy {}", card_name(card)),
            sts2_engine::ShopItem::Potion(id) | sts2_engine::ShopItem::Relic(id) => {
                format!("Buy {}", short(&id.to_string()))
            }
        },
        Action::BuyCardRemoval => "Buy card removal".to_owned(),
        Action::UsePotion {
            model_id, target, ..
        } => {
            let mut text = format!("Drink {}", short(&model_id.to_string()));
            if let Some(target) = target {
                use std::fmt::Write as _;
                let _ = write!(text, " → {}", target_name(target));
            }
            text
        }
        Action::DiscardPotion { model_id, .. } => {
            format!("Discard {}", short(&model_id.to_string()))
        }
        Action::RestOption { option, .. } => format!("Rest: {option:?}"),
        Action::TakeTreasure { relic } => format!("Take {}", short(&relic.to_string())),
        Action::UncoverCrystalSphere { x, y, tool } => format!("Sphere ({x},{y}) {tool:?}"),
        Action::AdvanceAct => "Next act".to_owned(),
        Action::PlayCard { card, target } => {
            let mut text = card_name(&card.fingerprint);
            if let Some(target) = target {
                use std::fmt::Write as _;
                let _ = write!(text, " → {}", target_name(target));
            }
            text
        }
        Action::EndTurn { .. } => "End turn".to_owned(),
        Action::ChooseCards { cards, .. } => {
            if cards.is_empty() {
                "Choose nothing".to_owned()
            } else {
                let names: Vec<String> = cards
                    .iter()
                    .map(|card| card_name(&card.fingerprint))
                    .collect();
                format!("Choose {}", names.join(", "))
            }
        }
        Action::ChooseAlternative { option_id, .. } => format!("Option: {option_id}"),
        Action::ClaimReward { fingerprint, .. } => {
            let what = fingerprint.model_id.as_ref().map_or_else(
                || {
                    if fingerprint.gold_amount > 0 {
                        format!("{} gold", fingerprint.gold_amount)
                    } else if !fingerprint.offered_cards.is_empty() {
                        "cards".to_owned()
                    } else {
                        fingerprint.reward_type.clone()
                    }
                },
                |id| short(&id.to_string()),
            );
            format!("Claim {what}")
        }
        Action::Proceed => "Proceed".to_owned(),
    }
}

fn display_plan(plan: &ActionPlan, observation: &AgentObservation) -> String {
    match plan {
        ActionPlan::Single(action) => display_action(action, observation),
        ActionPlan::Trade { free, take } => format!(
            "{}, then {}",
            display_action(free, observation),
            display_action(take, observation)
        ),
        ActionPlan::Collapse { open, pick } => format!(
            "{} → {}",
            display_action(open, observation),
            display_action(pick, observation)
        ),
    }
}

/// The class a macro plan is summed under: coarse enough that the take and
/// the skip of a card reward, or the heal and the smith of a rest site, are
/// each one class however many plans they split into.
fn plan_class(plan: &ActionPlan) -> String {
    match plan {
        ActionPlan::Single(action) => match action {
            Action::RestOption { option, .. } => format!("rest_site:{option:?}"),
            Action::ClaimReward { fingerprint, .. } => {
                format!("rewards:{}", fingerprint.reward_type)
            }
            Action::BuyShopItem { fingerprint, .. } => match fingerprint {
                sts2_engine::ShopItem::Card(_) => "shop:card".to_owned(),
                sts2_engine::ShopItem::Potion(_) => "shop:potion".to_owned(),
                sts2_engine::ShopItem::Relic(_) => "shop:relic".to_owned(),
            },
            Action::ChooseCards { cards, .. } if cards.is_empty() => "choose_cards:none".to_owned(),
            other => other.family().to_owned(),
        },
        ActionPlan::Trade { take, .. } => format!("trade:{}", take.family()),
        ActionPlan::Collapse { open, .. } => format!("{}:pick", open.family()),
    }
}
