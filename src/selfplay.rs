//! The self-play harness: walk runs in the simulator, emit decision scripts,
//! and keep the batch's coverage table.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

use sts2_core::UnlockPresetManifest;
use sts2_engine::{Action, MapPointType, RunPhase, RunResult, RunState, Simulator};
use sts2_rng::{MegaRandom, splitmix64};

use crate::objective::Objective;
use crate::plan::forced_step;
use crate::policy::RolloutPolicy;
use crate::script::{ScriptWriter, Unscriptable};

/// The game's seed alphabet after canonicalization: digits and letters,
/// with `O` and `I` folded into `0` and `1`.
const SEED_ALPHABET: &[u8] = b"0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

/// A fresh seed drawn from the analysis stream.
#[must_use]
pub fn draw_seed(rng: &mut MegaRandom) -> String {
    (0..10)
        .map(|_| {
            let index = usize::try_from(rng.next_u64()).unwrap_or(usize::MAX) % SEED_ALPHABET.len();
            char::from(SEED_ALPHABET[index])
        })
        .collect()
}

/// The seed pair run `index` of a batch plays under: the game seed, and the
/// seed of the analysis stream its policy draws from.
///
/// Derived from the batch's analysis seed and the run's index, never drawn
/// off a stream the policy also consumes. Drawing it there would couple
/// every run to how much search the runs before it had done: two batches
/// differing only in their selection policy would play *different games*
/// from the second run on, which is not an A/B comparison. Deriving the pair
/// makes a run a function of its index alone — so batches compare, and runs
/// may play in any order, or all at once.
#[must_use]
pub fn run_seeds(analysis_seed: u64, index: usize) -> (String, u64) {
    // The golden-ratio fold is the belief sampler's trick, for the same
    // reason: it keeps index 0 and seed 0 from collapsing onto each other.
    let mut mix = analysis_seed
        ^ u64::try_from(index)
            .unwrap_or(u64::MAX)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut draw = MegaRandom::new(splitmix64(&mut mix));
    (draw_seed(&mut draw), splitmix64(&mut mix))
}

/// A run that could not be generated or stepped.
#[derive(Debug)]
pub struct PlayError {
    /// What went wrong, prefixed by the batch with the index and seed that
    /// reproduce it.
    pub message: String,
    /// The decision script as far as the run got, when the fault was an
    /// engine step refusing an action: the trace that replays the fault
    /// directly in the engine, ending at the refused step. Nothing else
    /// carries one.
    pub partial_script: Option<String>,
    /// Whether the fault was a panic caught off the worker, as against an
    /// error the engine or the batch reported: the two are counted apart,
    /// because a panic is a bug in this code and an engine error a rule the
    /// engine refused.
    pub panicked: bool,
}

impl PlayError {
    /// A run lost for the stated reason, with nothing to replay.
    pub fn lost(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            partial_script: None,
            panicked: false,
        }
    }

    /// A run lost to a panic the worker caught.
    pub fn panicked(message: impl Into<String>) -> Self {
        Self {
            panicked: true,
            ..Self::lost(message)
        }
    }

    /// The same fault, with the batch's index and seed in front of it.
    #[must_use]
    pub fn prefixed(self, prefix: &str) -> Self {
        Self {
            message: format!("{prefix}: {}", self.message),
            ..self
        }
    }
}

impl std::fmt::Display for PlayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for PlayError {}

/// What killed a dead run, read off the run's own room record — the engine
/// closes the combat as the player goes down, so the record is what remains.
#[derive(Clone, Debug)]
pub struct Death {
    /// The model of the room that did it: the fight's encounter, or the
    /// event's, or nothing for a room family with no model of its own.
    pub encounter: Option<sts2_core::ModelId>,
    /// What the room turned out to be.
    pub room: Option<MapPointType>,
}

impl Death {
    /// What the run was standing in when it ended.
    fn read(run: &RunState) -> Self {
        let (encounter, room) = standing_room(run);
        Self { encounter, room }
    }
}

/// The room the run is standing in — its model and its family — off the run's
/// own map history.
///
/// The history's last room outranks `current_room_model`: a fight an event
/// pushed is recorded on top of the event, and it is the fight that matters,
/// whether it is the one that killed the run or the one the run is fighting
/// now.
#[must_use]
pub fn standing_room(run: &RunState) -> (Option<sts2_core::ModelId>, Option<MapPointType>) {
    let room = run
        .map_history
        .last()
        .and_then(|act| act.points.last())
        .and_then(|point| point.rooms.last());
    (
        room.and_then(|room| room.model_id.clone()),
        room.map(|room| room.room_type),
    )
}

/// What one walked run came to.
#[derive(Clone, Debug)]
pub struct RunReport {
    pub seed: String,
    pub actions: Vec<Action>,
    pub floor: u32,
    pub act: usize,
    /// Act bosses defeated: standing in act index `n` is `n` cleared, and
    /// the transition screen counts its beaten boss before `AdvanceAct`.
    pub acts_cleared: usize,
    pub terminal: Option<RunResult>,
    /// Where the run died, for a run that died.
    pub death: Option<Death>,
    pub reward: f64,
    /// The emitted decision script, or the reason it could not be emitted.
    pub script: Result<String, Unscriptable>,
    /// Whatever in-combat decisions the policy recorded during this run. A
    /// policy that records nothing answers with nothing.
    pub decisions: Vec<crate::training::Decision>,
    /// The same for macro (out-of-combat) decisions, which are a different
    /// net's data and go to their own shard directory.
    pub macro_decisions: Vec<crate::training::Decision>,
    /// What the run's macro search budget came to, where the policy had one.
    pub budget: Option<crate::search::BudgetSpend>,
    /// What the run's own play recorded about itself — the fights it fought
    /// and what it threw away — where the policy collected it. Observed while
    /// the run was played because a finished report cannot be read backwards
    /// into it. A policy that collects nothing answers with nothing.
    pub metrics: Option<crate::summary::EpisodeMetrics>,
    /// The fights this run entered, banked at their entry decision. Empty
    /// unless the run harvests.
    pub fights: Vec<crate::library::FightEntry>,
}

impl RunReport {
    /// Whether the run got past the act 1 boss — the project's yardstick.
    #[must_use]
    pub fn cleared_act1(&self) -> bool {
        self.acts_cleared >= 1 || matches!(self.terminal, Some(RunResult::Victory))
    }
}

/// Walks one run to its terminal or a step cap, recording the script as it
/// goes. The policy draws only from `rng`, so the walk is reproducible from
/// the seed pair.
pub fn play_run(
    seed: &str,
    character: &sts2_core::ModelId,
    ascension: u8,
    policy: &mut dyn RolloutPolicy,
    rng: &mut MegaRandom,
    objective: &mut dyn Objective,
    max_steps: usize,
) -> Result<RunReport, PlayError> {
    let preset =
        UnlockPresetManifest::pinned().map_err(|error| PlayError::lost(error.to_string()))?;
    play_run_on_preset(
        &preset, seed, character, ascension, policy, rng, objective, max_steps, None, None,
    )
}

/// The same walk on a named profile: what a script bound for a live game
/// needs, since the playback driver verifies the script's unlock state —
/// prior-run count included, exactly — against the profile it will play on.
#[allow(
    clippy::too_many_arguments,
    reason = "one lever per run-defining input"
)]
pub fn play_run_on_preset(
    preset: &UnlockPresetManifest,
    seed: &str,
    character: &sts2_core::ModelId,
    ascension: u8,
    policy: &mut dyn RolloutPolicy,
    rng: &mut MegaRandom,
    objective: &mut dyn Objective,
    max_steps: usize,
    harvest: Option<crate::library::HarvestAs>,
    force_wins: Option<&crate::forcewins::ForceWins>,
) -> Result<RunReport, PlayError> {
    let mut simulator: Simulator =
        sts2_content::standard_run_on_preset_at(seed, character, preset, ascension)
            .map_err(|error| PlayError::lost(format!("{error:?}")))?;
    // A forced-win walk skips its combats, so no real game can take the same
    // actions: the script writer never opens and the report says why.
    let mut writer = if force_wins.is_some() {
        None
    } else {
        Some(
            ScriptWriter::open(&simulator, seed, ascension)
                .map_err(|error| PlayError::lost(format!("{error:?}")))?,
        )
    };
    let mut unscriptable: Option<Unscriptable> = force_wins.is_some().then(|| {
        Unscriptable("a forced-win walk skips its combats, which no real game replays".into())
    });
    // The walk's own stream, one draw off the policy's: what a skipped
    // fight costs must not depend on how many draws the policy has taken,
    // and the policy must not feel the walk's draws.
    let mut god = force_wins.map(|config| (config, MegaRandom::new(rng.next_u64())));
    let mut actions = Vec::new();
    let mut fights = Vec::new();
    let mut fighting = false;
    for _ in 0..max_steps {
        if simulator.state().terminal.is_some() {
            break;
        }
        if let Some(keep) = harvest {
            // A rising edge into a live fight is that fight's entry
            // decision: bank it before the policy moves, said as the batch
            // asked — the setup, the state erased to what a player can
            // see, or both.
            let live = !crate::env::fight_over(&simulator);
            if live && !fighting {
                let mut entry = crate::library::harvest(&simulator, keep)
                    .map_err(|error| PlayError::lost(error.to_string()))?;
                if god.is_some() {
                    entry.meta.origin = crate::library::FightOrigin::ForcedWin;
                }
                fights.push(entry);
            }
            fighting = live;
        }
        // The forced win itself: the fight is resolved where it was banked,
        // at the price the config draws, and the run walks on. An entry
        // decision the engine is still settling — a start-of-combat screen a
        // body waits on — refuses the resolve without moving anything, and
        // the policy answers the screen instead; the resolve lands on a
        // later pass of this loop, before any fight turn is ever taken.
        if let Some((config, god_rng)) = god.as_mut()
            && !crate::env::fight_over(&simulator)
            && force_win(&mut simulator, config, god_rng, &mut actions)
                .map_err(|error| PlayError::lost(format!("{error:?}")))?
        {
            policy.stepped(&simulator);
            continue;
        }
        // An engine dead end on the authoritative run: a live screen the
        // engine enumerates no answer for (`RELIC.BIIIG_HUG`'s over-cap
        // exact-4 pick is the registered shape — see `search::leaf_value`).
        // No policy can answer it and no honest step exists, so the run is
        // reported lost with the fault and screen named — the same loud
        // channel a refused authoritative step already uses — rather than
        // panicking through a policy's own invariant.
        if simulator.legal_actions().is_empty() {
            return Err(PlayError::lost(format!(
                "the engine enumerated no actions on a live run at {:?}",
                simulator.decision()
            )));
        }
        // The reward lines that are free to take are claimed before the
        // policy is asked (see `plan::forced_step`).
        claim_free_lines(
            &mut simulator,
            &mut writer,
            &mut unscriptable,
            policy,
            &mut actions,
        )?;
        // One policy decision, expanded into the engine steps it names. Each
        // is recorded, applied and paid for on its own, so a script stays a
        // sequence of legal engine actions and `stepped` still fires once per
        // step. The cap counts decisions rather than steps, which keeps a
        // plan from being half-applied at the boundary.
        let plan = policy.choose_plan(&simulator, rng);
        for action in plan.steps() {
            apply_step(
                &mut simulator,
                action,
                &mut writer,
                &mut unscriptable,
                policy,
                &mut actions,
            )?;
        }
    }
    policy.run_ended(&simulator);
    let decisions = policy.drain_decisions();
    let macro_decisions = policy.drain_macro_decisions();
    let budget = policy.budget_spent();
    let metrics = policy.drain_run_metrics();
    let state = simulator.state();
    let report = RunReport {
        seed: seed.to_owned(),
        floor: state.run.as_ref().map_or(0, |run| run.floor),
        act: state.run.as_ref().map_or(0, |run| run.current_act),
        acts_cleared: state.run.as_ref().map_or(0, |run| {
            run.current_act + usize::from(matches!(run.phase, RunPhase::ActTransition))
        }),
        terminal: state.terminal,
        death: matches!(state.terminal, Some(RunResult::Defeat))
            .then(|| state.run.as_ref().map(Death::read))
            .flatten(),
        reward: objective.reward(&simulator),
        script: match (writer, unscriptable) {
            (Some(open), None) => Ok(open.finish()),
            (_, Some(reason)) => Err(reason),
            (None, None) => unreachable!("a writer is only dropped with its reason"),
        },
        actions,
        decisions,
        macro_decisions,
        budget,
        metrics,
        fights,
    };
    Ok(report)
}

/// Resolves the standing fight as won at the config's price, maybe
/// spending a potion the way the fight would have. Answers `false` when the
/// engine is still settling the entry — a start-of-combat screen a body
/// waits on refuses the resolve without moving anything, and the caller
/// lets the policy answer the screen — and `true` once the fight fell.
fn force_win(
    simulator: &mut Simulator,
    config: &crate::forcewins::ForceWins,
    god: &mut MegaRandom,
    actions: &mut Vec<Action>,
) -> Result<bool, sts2_engine::EngineError> {
    let state = simulator.state();
    let (Some(combat), Some(run)) = (state.combat.as_ref(), state.run.as_ref()) else {
        return Ok(true);
    };
    let tier = crate::library::Tier::of(run.standing_room_type());
    let act = run.current_act;
    let max_hp = state.run_player.max_hp;
    let current = combat
        .creatures
        .iter()
        .find(|creature| creature.id == combat.player.creature_id)
        .map_or(state.run_player.current_hp, |creature| creature.current_hp);
    // Both draws land before the resolve is attempted, so what a fight
    // costs is a function of the walk's stream alone, never of which
    // screens its entry happened to raise.
    let target = config.hp_after(tier, act, max_hp, current, god);
    let drinks = config.uses_potion(tier, god);
    match simulator.resolve_combat_won(target) {
        Ok(()) => {}
        Err(error) if error.code == sts2_engine::ErrorCode::WrongDecisionContext => {
            return Ok(false);
        }
        Err(error) => return Err(error),
    }
    if drinks {
        let discards: Vec<Action> = simulator
            .legal_actions()
            .iter()
            .filter(|action| matches!(action, Action::DiscardPotion { .. }))
            .cloned()
            .collect();
        if !discards.is_empty() {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "potion belts are far below u64"
            )]
            let pick = (god.next_u64() % discards.len() as u64) as usize;
            let action = discards[pick].clone();
            simulator.step_quietly(&action)?;
            actions.push(action);
        }
    }
    Ok(true)
}

/// One batch of runs: everything every run in it shares.
pub struct Batch<'a> {
    pub preset: &'a UnlockPresetManifest,
    pub character: &'a sts2_core::ModelId,
    pub ascension: u8,
    /// The batch's analysis seed. Every run's own pair is derived from this
    /// and the run's index — see [`run_seeds`].
    pub analysis_seed: u64,
    /// One fixed game seed for every run, where the caller pinned one. The
    /// analysis seed still varies by index, so a pinned seed played `n` times
    /// is `n` different lines of play through one game.
    pub seed: Option<&'a str>,
    pub runs: usize,
    pub max_steps: usize,
    /// How many runs play at once. One is the serial batch.
    pub jobs: usize,
    /// Whether each fight's entry is banked into its run's report, and as
    /// what: the setup, the exact state, or both.
    pub harvest: Option<crate::library::HarvestAs>,
    /// Forced-win harvesting: no combat is played — every fight is resolved
    /// as won at this config's price — so the walk reaches any depth with
    /// no combat competence at all. Pair it with `harvest`; the walk exists
    /// for the bank it leaves behind.
    pub force_wins: Option<&'a crate::forcewins::ForceWins>,
}

/// Plays a batch, `jobs` runs at a time, handing each report to `report` in
/// run order.
///
/// A run whose play panics — an engine invariant tripped by a walk the
/// searches exist to take — arrives as that run's `Err`, naming its index
/// and seed, and every other run still completes. The searches are the
/// fuzzer; a fault they uncover is a find to report with its reproducer,
/// not a reason to lose the batch.
///
/// Every run builds its own policy and its own objective handle. That is what
/// makes the parallel batch honest: a run's play depends on its seed pair
/// alone, never on which runs a worker happened to take first — and never on
/// what the tree of a previous run left in the transposition table. A batch at
/// `jobs = 8` and the same batch at `jobs = 1` play the same games.
///
/// The one thing that does not survive the parallel batch is the *coverage
/// table*: novelty is paid first-come out of a table every run shares, so
/// which run is paid for a first sighting depends on scheduling. That moves
/// the printed reward numbers, and — only in `TrueState` search, where the
/// coverage objective is also what steers the search — the walks themselves.
/// Belief self-play scores its search with a stateless objective and is
/// unaffected.
pub fn play_batch(
    batch: &Batch<'_>,
    policy: &(dyn Fn() -> Box<dyn RolloutPolicy> + Sync),
    objective: &(dyn Fn() -> Box<dyn Objective> + Sync),
    report: &mut dyn FnMut(usize, Result<RunReport, PlayError>),
) {
    indexed(
        batch.runs,
        batch.jobs,
        &|index| {
            let (drawn, analysis) = run_seeds(batch.analysis_seed, index);
            let seed = batch.seed.map_or(drawn, str::to_owned);
            // A panic is a lost run, not a lost batch. The searches exist to
            // walk into engine faults, and a fault's panic arriving as this
            // run's error keeps the other thousand runs — and hands back the
            // one seed that reproduces it.
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut rng = MegaRandom::new(analysis);
                let mut policy = policy();
                let mut objective = objective();
                play_run_on_preset(
                    batch.preset,
                    &seed,
                    batch.character,
                    batch.ascension,
                    policy.as_mut(),
                    &mut rng,
                    objective.as_mut(),
                    batch.max_steps,
                    batch.harvest,
                    batch.force_wins,
                )
            }))
            .unwrap_or_else(|panic| {
                Err(PlayError::panicked(format!(
                    "panicked: {}",
                    panic_message(&panic)
                )))
            })
            .map_err(|error| error.prefixed(&format!("run {index} on {seed}")))
        },
        report,
    );
}

/// An engine fault on the authoritative run, reported with the step it
/// refused, the decision it stood on, and the script as far as it got: the
/// engine's own error names a code and a detail, and neither says which
/// action, on which screen, reached it.
/// More free claims than any reward screen carries lines.
const FORCED_STEP_CAP: usize = 32;

/// The free reward lines standing at `simulator` — gold, a potion the belt
/// has room for — claimed one at a time off the screen
/// each claim leaves, before any policy is asked. Each is recorded and paid
/// for like any step, and no policy is asked about it; the policy is told,
/// so what the screen offered is still counted. Bounded by the screen: a
/// claim the engine accepts without settling its line would loop here
/// forever, and is reported instead.
fn claim_free_lines(
    simulator: &mut Simulator,
    writer: &mut Option<ScriptWriter>,
    unscriptable: &mut Option<Unscriptable>,
    policy: &mut dyn RolloutPolicy,
    actions: &mut Vec<Action>,
) -> Result<(), PlayError> {
    let mut claimed = 0;
    while let Some(action) = forced_step(simulator) {
        if claimed >= FORCED_STEP_CAP {
            return Err(PlayError::lost(format!(
                "a free reward line was claimed {FORCED_STEP_CAP} times and still stands at {:?}",
                simulator.decision()
            )));
        }
        claimed += 1;
        policy.forced(simulator, &action);
        apply_step(simulator, &action, writer, unscriptable, policy, actions)?;
    }
    Ok(())
}

/// One engine step on the authoritative run: recorded where a script is
/// still being written, applied, and told to the policy with the state it
/// produced — which the policy that named (or was spared) the step could not
/// see, and which a reward is a function of. See [`RolloutPolicy::stepped`];
/// every policy that owes nothing there is unaffected.
fn apply_step(
    simulator: &mut Simulator,
    action: &Action,
    writer: &mut Option<ScriptWriter>,
    unscriptable: &mut Option<Unscriptable>,
    policy: &mut dyn RolloutPolicy,
    actions: &mut Vec<Action>,
) -> Result<(), PlayError> {
    if let Some(open) = writer.as_mut()
        && let Err(reason) = open.record(simulator, action)
    {
        *unscriptable = Some(reason);
        *writer = None;
    }
    if let Err(error) = simulator.step_quietly(action) {
        return Err(refused_step(&error, action, simulator, writer.take()));
    }
    policy.stepped(simulator);
    actions.push(action.clone());
    Ok(())
}

fn refused_step(
    error: &sts2_engine::EngineError,
    action: &Action,
    simulator: &Simulator,
    writer: Option<ScriptWriter>,
) -> PlayError {
    PlayError {
        message: format!(
            "{error:?} refusing {action:?} at {:?}",
            simulator.decision()
        ),
        partial_script: writer.map(ScriptWriter::finish),
        panicked: false,
    }
}

/// What a caught panic had to say for itself.
pub(crate) fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(ToString::to_string)
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".to_owned())
}

/// The stack a worker gets, against the platform default of two mebibytes.
///
/// The engine descends its own call stack for a card that plays a card: a
/// draw hook that auto-plays a Strike runs the whole of that Strike's play,
/// including whatever it draws, inside the frame of the draw that produced
/// it. Depth is bounded by the rules — a full hand stops the draw, and a
/// power that plays every Strike drawn caps its own plays where nothing it
/// hits can die — but it is bounded by the rules rather than by anything
/// this crate controls, and it is not bounded small. Overflowing a worker's
/// stack aborts the process, which on an unattended batch loses every run in
/// flight rather than the one line that ran deep.
///
/// Stacks are reserved rather than committed, so the untouched remainder of
/// a worker's is address space and not memory.
const WORKER_STACK: usize = 16 * 1024 * 1024;

/// Fans `count` indexed jobs across `jobs` workers, handing each result to
/// `report` in index order: the pool-and-reorder half every batch this crate
/// plays shares. `play` owns its own failure story; nothing here inspects
/// the results it carries.
pub(crate) fn indexed<T: Send>(
    count: usize,
    jobs: usize,
    play: &(dyn Fn(usize) -> T + Sync),
    report: &mut dyn FnMut(usize, T),
) {
    let next = AtomicUsize::new(0);
    let (sender, receiver) = mpsc::channel();
    let jobs = jobs.clamp(1, count.max(1));
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            let sender = sender.clone();
            let next = &next;
            std::thread::Builder::new()
                .stack_size(WORKER_STACK)
                .spawn_scoped(scope, move || {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= count {
                            break;
                        }
                        if sender.send((index, play(index))).is_err() {
                            break;
                        }
                    }
                })
                .expect("a batch worker thread starts");
        }
        // The workers hold the only senders now, so the loop below ends when
        // the last of them is done.
        drop(sender);
        // Reordering buffer: a report is handed on only once every job before
        // it has been, so what the batch prints does not depend on which
        // worker finished first.
        let mut pending: HashMap<usize, T> = HashMap::new();
        let mut cursor = 0;
        for (index, played) in receiver {
            pending.insert(index, played);
            while let Some(ready) = pending.remove(&cursor) {
                report(cursor, ready);
                cursor += 1;
            }
        }
    });
}

/// One line of the batch summary.
#[must_use]
pub fn summarize(report: &RunReport) -> String {
    let mut line = format!(
        "{} act {} floor {} {} reward {:.1}",
        report.seed,
        report.act,
        report.floor,
        report
            .terminal
            .map_or("unfinished", |terminal| match terminal {
                RunResult::Victory => "VICTORY",
                RunResult::Defeat => "DEFEAT",
            }),
        report.reward,
    );
    if let Some(death) = &report.death {
        let _ = write!(
            line,
            " to {} ({})",
            death
                .encounter
                .as_ref()
                .map_or_else(|| "no room".to_owned(), ToString::to_string),
            room_label(death.room),
        );
    }
    if let Some(budget) = &report.budget
        && budget.searched + budget.downgraded > 0
    {
        let _ = write!(
            line,
            " | macro {} searched {} downgraded over {} steps",
            budget.searched, budget.downgraded, budget.steps,
        );
    }
    if let Err(reason) = &report.script {
        let _ = write!(line, " [unscriptable: {reason}]");
    }
    line
}

/// The room family as the death table names it.
#[must_use]
pub const fn room_label(room: Option<MapPointType>) -> &'static str {
    match room {
        Some(MapPointType::Monster) => "hallway",
        Some(MapPointType::Elite) => "elite",
        Some(MapPointType::Boss) => "boss",
        Some(MapPointType::Ancient) => "event",
        Some(MapPointType::Shop) => "shop",
        Some(MapPointType::RestSite) => "rest site",
        Some(MapPointType::Treasure) => "treasure",
        Some(MapPointType::Unknown) => "unknown",
        None => "nowhere",
    }
}

/// A batch's outcomes in aggregate: the act-1 clear rate the project is
/// chasing, alongside a table of what killed the dead. Feed it every
/// finished run's report; it reads, it never plays.
#[derive(Clone, Debug, Default)]
pub struct OutcomeTally {
    runs: usize,
    act1_clears: usize,
    victories: usize,
    unfinished: usize,
    floors: u64,
    /// Deaths keyed by act index, room family, and killer.
    deaths: crate::tally::Rows<&'static str>,
}

impl OutcomeTally {
    /// Counts one finished run.
    pub fn record(&mut self, report: &RunReport) {
        self.runs += 1;
        self.floors += u64::from(report.floor);
        if report.cleared_act1() {
            self.act1_clears += 1;
        }
        match report.terminal {
            Some(RunResult::Victory) => self.victories += 1,
            Some(RunResult::Defeat) => {
                let (encounter, room) = report
                    .death
                    .as_ref()
                    .map_or((None, None), |death| (death.encounter.as_ref(), death.room));
                self.deaths.record(
                    report.act,
                    room_label(room),
                    encounter.map_or_else(|| "no room".to_owned(), ToString::to_string),
                    report.floor,
                );
            }
            None => self.unfinished += 1,
        }
    }

    /// How many runs cleared act 1.
    #[must_use]
    pub const fn act1_clears(&self) -> usize {
        self.act1_clears
    }

    /// The one-line aggregate: the clear rate, the mean floor, and the runs
    /// that ended some other way.
    #[must_use]
    #[allow(clippy::cast_precision_loss, reason = "batch sizes are small")]
    pub fn summary(&self) -> String {
        let runs = self.runs.max(1) as f64;
        format!(
            "act 1 cleared {}/{} ({:.1}%) | mean floor {:.2} | {} victories | {} unfinished",
            self.act1_clears,
            self.runs,
            100.0 * self.act1_clears as f64 / runs,
            self.floors as f64 / runs,
            self.victories,
            self.unfinished,
        )
    }

    /// The death table, deadliest killer first, at most `limit` rows plus a
    /// remainder line for whatever the limit cut. Each row is one killer in
    /// one room family of one act, with its kill count and floor span.
    #[must_use]
    pub fn death_table(&self, limit: usize) -> Vec<String> {
        self.deaths.table(limit, "deaths")
    }
}
