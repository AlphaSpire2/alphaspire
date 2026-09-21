//! What the environment pays a run for the step it just took.
//!
//! PPO rewards combine floor progress with a terminal outcome term.
//! Optional weights pay for elite wins, relics, gold and boss kills; these
//! default to zero. Boss weights are per act and include the final boss in
//! addition to the victory reward. There is no hit-point shaping.
//!
//! Each reward configuration has its own [`value_semantics`], so a checkpoint
//! cannot silently be used with a different reward definition.
//!
//! This module is the one part of a rollout that reads `simulator.state()`.
//! An observation is what a player can see and an actor may read nothing
//! else; a reward is what the environment pays, and an environment knows the
//! floor and the outcome whether or not the player does.

use std::fmt::Write as _;
use std::path::Path;

use serde::Serialize;
use sts2_engine::{DecisionContext, MapPointType, RunResult, Simulator};

/// Which gold the gold term pays for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GoldScope {
    /// Every gold the run collects, whatever brought it.
    #[default]
    Any,
    /// Only gold claimed off a reward screen — a fight's drop, an elite's,
    /// a boss's. Event and Neow gold is left to the outcome to price: the
    /// worth of 333 gold at floor one depends on where the shops are, and a
    /// term that pays it on the spot cannot see that.
    Rewards,
}

/// The weights of the boss term, one per act: what killing a boss in act
/// one, two and three pays.
pub type BossWeights = [f64; 3];

/// What the value head of a run checkpoint trained under these weights
/// predicts. The base reward keeps the name every existing checkpoint
/// carries; any non-zero extra term names itself, weights included, so two
/// runs of the A/B cannot be read as one another's. Each term added later
/// appends to the name of the terms before it, so a checkpoint trained
/// before a term existed keeps the name it was trained under.
#[must_use]
pub fn value_semantics(
    elite_weight: f64,
    relic_weight: f64,
    gold_weight: f64,
    gold_scope: GoldScope,
    boss_weights: BossWeights,
) -> String {
    let zero = |weight: f64| weight.abs() < f64::EPSILON;
    let no_boss = boss_weights.iter().all(|&weight| zero(weight));
    if zero(elite_weight) && zero(relic_weight) && zero(gold_weight) && no_boss {
        return crate::objective::RUN_VALUE_SEMANTICS.to_owned();
    }
    let mut name = format!("run-return-v2;elite={elite_weight};relic={relic_weight}");
    if !zero(gold_weight) {
        let _ = write!(name, ";gold={gold_weight}");
        if gold_scope == GoldScope::Rewards {
            name.push_str(";gold_scope=rewards");
        }
    }
    if !no_boss {
        let [one, two, three] = boss_weights;
        let _ = write!(name, ";boss={one}/{two}/{three}");
    }
    name
}

/// The optional reward terms a run checkpoint was trained under, read back
/// off its value semantics.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RunTerms {
    pub elite: f64,
    pub relic: f64,
    pub gold: f64,
    pub gold_rewards_only: bool,
    /// What a boss killed pays, by the act it was killed in.
    pub boss: BossWeights,
}

impl RunTerms {
    /// The base reward: floors and outcome, no optional term.
    pub const NONE: Self = Self {
        elite: 0.0,
        relic: 0.0,
        gold: 0.0,
        gold_rewards_only: false,
        boss: [0.0; 3],
    };

    /// Which gold the gold term pays for.
    #[must_use]
    pub const fn scope(self) -> GoldScope {
        if self.gold_rewards_only {
            GoldScope::Rewards
        } else {
            GoldScope::Any
        }
    }

    /// The name a checkpoint trained under these terms carries.
    #[must_use]
    pub fn value_semantics(self) -> String {
        value_semantics(self.elite, self.relic, self.gold, self.scope(), self.boss)
    }
}

/// Parses a run checkpoint's value semantics back into the reward terms
/// that produced it. `None` for a name this build cannot reproduce, which is
/// refused rather than approximated: a critic priced under terms this build
/// cannot pay would have every return around it measured in the wrong
/// units.
#[must_use]
pub fn run_terms(semantics: &str) -> Option<RunTerms> {
    let mut terms = RunTerms::NONE;
    let mut parts = semantics.split(';');
    match parts.next()? {
        crate::objective::RUN_VALUE_SEMANTICS => {}
        "run-return-v2" => {
            for part in parts {
                let (key, value) = part.split_once('=')?;
                match key {
                    "elite" => terms.elite = value.parse().ok()?,
                    "relic" => terms.relic = value.parse().ok()?,
                    "gold" => terms.gold = value.parse().ok()?,
                    "gold_scope" => terms.gold_rewards_only = value == "rewards",
                    "boss" => {
                        let mut weights = value.split('/').map(|weight| weight.parse::<f64>().ok());
                        terms.boss = [weights.next()??, weights.next()??, weights.next()??];
                        if weights.next().is_some() {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
        }
        _ => return None,
    }
    // Only a name this build would itself write is accepted: the round trip
    // is what proves the terms read are the terms meant.
    (terms.value_semantics() == semantics).then_some(terms)
}

/// The reward a run checkpoint was trained under, read off the provenance
/// file beside it: the value semantics it names and the terms that name
/// parses into. Refused where the file is missing, names no semantics, or
/// names a reward this build cannot pay — an act-boundary checkpoint among
/// them.
pub fn checkpoint_terms(stem: &Path) -> Result<(String, RunTerms), String> {
    let provenance = stem.with_extension("json");
    let text = std::fs::read_to_string(&provenance)
        .map_err(|error| format!("{}: {error}", provenance.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("{}: {error}", provenance.display()))?;
    let semantics = json
        .get("value_semantics")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("{}: names no value_semantics", provenance.display()))?
        .to_owned();
    let terms = run_terms(&semantics).ok_or_else(|| {
        format!(
            "{}: value_semantics `{semantics}` is not a run reward this build can pay",
            provenance.display()
        )
    })?;
    Ok((semantics, terms))
}

/// The reward a rollout pays under a checkpoint: always the checkpoint's
/// own. Flags that state a reward do not choose it — they assert it, and a
/// statement that names a different reward is refused. The critic predicts
/// returns in the units it was trained under, so paying anything else
/// would make every advantage in the generation wrong without anything
/// crashing; and a driver whose flags were silently overridden would be
/// running an experiment other than the one written in it.
pub fn resolve_run_terms(
    stated: Option<RunTerms>,
    checkpoint_semantics: &str,
    checkpoint_terms: RunTerms,
) -> Result<RunTerms, String> {
    match stated {
        Some(stated) if stated.value_semantics() != checkpoint_semantics => Err(format!(
            "the reward flags name `{}` but the checkpoint was trained under \
             `{checkpoint_semantics}`; a run pays the checkpoint's own reward, so either \
             drop the flags or state that reward",
            stated.value_semantics()
        )),
        _ => Ok(checkpoint_terms),
    }
}

/// The reward one step pays, read off the state that step left the run
/// standing in.
///
/// Stateful because a floor is a level rather than an increment: what a step
/// pays for progress is the difference between the floor the run stands on
/// now and the floor it stood on when this last paid. One [`RunReward`]
/// therefore belongs to exactly one run, and
/// [`RunReward::starting`] — which reads the baseline off the state the run
/// begins on — is the whole of its reset semantics. There is no `Default`
/// on purpose: a fresh run already stands on floor one, so a zero baseline
/// would pay the first macro decision of every episode for a floor nobody
/// climbed.
///
/// **A run that ends keeps the floors it climbed.** The terminal term is
/// paid on top of the climb, never in place of it, so a death on floor 30
/// returns far more over the episode than a death on floor 3. Zeroing the
/// accrued reward on defeat is the alternative, and it is the one that
/// collapses a generation of episodes onto a single number — the same
/// pathology [`ActBoundary`](crate::objective::ActBoundary) carries its own
/// floor term to avoid, for the same reason.
///
/// The weights are public fields, following
/// [`CombatStrength`](crate::objective::CombatStrength) and `ActBoundary`,
/// so adding a term later is a weight and a semantics bump rather than a
/// rewrite. What they price together is named by
/// [`RUN_VALUE_SEMANTICS`](crate::objective::RUN_VALUE_SEMANTICS).
#[derive(Clone, Copy, Debug)]
pub struct RunReward {
    /// What one floor of progress pays.
    pub floor_weight: f64,
    /// What clearing the run pays, once.
    pub victory_weight: f64,
    /// What being defeated costs, once.
    pub defeat_weight: f64,
    /// What an elite fight won pays. Zero unless asked for.
    pub elite_weight: f64,
    /// What each relic gained pays, from any source. Zero unless asked for.
    pub relic_weight: f64,
    /// What each gold gained pays, from any source; spending is not
    /// charged. Zero unless asked for.
    pub gold_weight: f64,
    /// Which gold the gold term pays for.
    pub gold_scope: GoldScope,
    /// What a boss killed pays, by the act it was killed in. Zero unless
    /// asked for.
    pub boss_weights: BossWeights,
    /// How much gold the run held when this last paid.
    gold: i32,
    /// Whether the run stood inside an elite's fight when this last paid —
    /// the fight ending with the run alive is what the elite term pays on.
    in_elite_fight: bool,
    /// The act of the boss fight the run stood inside when this last paid,
    /// if it stood in one — the fight ending with the run alive is what the
    /// boss term pays on, at that act's weight.
    in_boss_fight: Option<usize>,
    /// How many relics the run held when this last paid.
    relics: usize,
    /// The floor the last payment was measured against.
    floor: u32,
    /// Whether the terminal term has already been paid. A run terminates
    /// once, and the term belongs to the step that terminated it: a state
    /// read twice must not pay twice.
    settled: bool,
}

impl RunReward {
    /// The reward of a run standing at `simulator`, with the default
    /// weights. The floor it stands on is the baseline the first payment is
    /// measured against, so a run that has not moved is owed nothing.
    #[must_use]
    pub fn starting(simulator: &Simulator) -> Self {
        Self {
            floor_weight: 0.1,
            victory_weight: 10.0,
            defeat_weight: -5.0,
            elite_weight: 0.0,
            relic_weight: 0.0,
            gold_weight: 0.0,
            gold_scope: GoldScope::Any,
            boss_weights: [0.0; 3],
            gold: gold_of(simulator),
            in_elite_fight: in_elite_fight(simulator),
            in_boss_fight: boss_fight_act(simulator),
            relics: relics_of(simulator),
            floor: floor_of(simulator),
            settled: false,
        }
    }

    /// The same reward with the optional terms weighted.
    #[must_use]
    pub const fn with_terms(
        mut self,
        elite_weight: f64,
        relic_weight: f64,
        gold_weight: f64,
    ) -> Self {
        self.elite_weight = elite_weight;
        self.relic_weight = relic_weight;
        self.gold_weight = gold_weight;
        self
    }

    /// The same reward paying the gold term for `scope` only.
    #[must_use]
    pub const fn with_gold_scope(mut self, scope: GoldScope) -> Self {
        self.gold_scope = scope;
        self
    }

    /// The same reward paying a boss killed by the act it was killed in.
    #[must_use]
    pub const fn with_boss_terms(mut self, weights: BossWeights) -> Self {
        self.boss_weights = weights;
        self
    }

    /// What a checkpoint trained under this reward predicts.
    #[must_use]
    pub fn value_semantics(&self) -> String {
        value_semantics(
            self.elite_weight,
            self.relic_weight,
            self.gold_weight,
            self.gold_scope,
            self.boss_weights,
        )
    }

    /// What the step that produced this state pays, settling the floor it
    /// was measured against and the terminal term if there is one.
    pub fn paid(&mut self, simulator: &Simulator) -> f64 {
        let terminal = simulator.state().terminal;
        let mut paid = self.priced(floor_of(simulator), terminal);
        // A relic is paid on arrival, whatever brought it; one given up is
        // not refunded, so the count only ever ratchets up here.
        let relics = relics_of(simulator);
        if relics > self.relics {
            #[allow(clippy::cast_precision_loss, reason = "a run holds tens of relics")]
            {
                paid += self.relic_weight * (relics - self.relics) as f64;
            }
        }
        self.relics = relics;
        // Gold is paid on arrival too, whatever brought it, and spending it
        // is not charged: the term is for collecting, and the shop's answer
        // to what it buys is the outcome's to price.
        let gold = gold_of(simulator);
        if gold > self.gold {
            let in_scope = match self.gold_scope {
                GoldScope::Any => true,
                GoldScope::Rewards => {
                    matches!(simulator.decision(), DecisionContext::Rewards { .. })
                }
            };
            if in_scope {
                paid += self.gold_weight * f64::from(gold - self.gold);
            }
        }
        self.gold = gold;
        paid + self.fought(
            in_elite_fight(simulator),
            boss_fight_act(simulator),
            terminal,
        )
    }

    /// The fight terms, from the fights the run stands in directly — whether
    /// it stands in an elite's, and the act of the boss's if it stands in
    /// one — the way [`RunReward::priced`] takes the floor.
    ///
    /// An elite or a boss is paid when its fight ends with the run still
    /// alive: the step that left the fight, read off the state it produced.
    /// A won run leaves its last boss fight the moment it is won, so the
    /// final kill is paid like every other, on top of the victory.
    pub fn fought(
        &mut self,
        in_elite: bool,
        boss_act: Option<usize>,
        terminal: Option<RunResult>,
    ) -> f64 {
        let alive = !matches!(terminal, Some(RunResult::Defeat));
        let mut paid = 0.0;
        if self.in_elite_fight && !in_elite && alive {
            paid += self.elite_weight;
        }
        self.in_elite_fight = in_elite;
        if let Some(act) = self.in_boss_fight
            && boss_act.is_none()
            && alive
        {
            paid += self.boss_weights.get(act).copied().unwrap_or(0.0);
        }
        self.in_boss_fight = boss_act;
        paid
    }

    /// The same payment from the two numbers directly — the floor the run
    /// stands on and how it ended, if it has.
    ///
    /// [`RunReward::paid`] is this read off the authoritative state, and is
    /// what a rollout calls. Anything already holding the pair — a test, a
    /// replayed trajectory — prices it here instead of reconstructing a
    /// simulator to be read from.
    pub fn priced(&mut self, floor: u32, terminal: Option<RunResult>) -> f64 {
        let climbed = floor.saturating_sub(self.floor);
        self.floor = floor.max(self.floor);
        let ended = if self.settled {
            0.0
        } else {
            match terminal {
                Some(RunResult::Victory) => {
                    self.settled = true;
                    self.victory_weight
                }
                Some(RunResult::Defeat) => {
                    self.settled = true;
                    self.defeat_weight
                }
                None => 0.0,
            }
        };
        self.floor_weight * f64::from(climbed) + ended
    }
}

/// The floor the run stands on, or zero where there is no run yet: a state
/// before the climb starts has climbed nothing.
fn floor_of(simulator: &Simulator) -> u32 {
    simulator.state().run.as_ref().map_or(0, |run| run.floor)
}

/// How much gold the run holds.
fn gold_of(simulator: &Simulator) -> i32 {
    simulator.state().run_player.gold
}

/// How many relics the run holds.
fn relics_of(simulator: &Simulator) -> usize {
    simulator.state().run_player.relics.len()
}

/// Whether the run stands inside a fight in an elite room.
fn in_elite_fight(simulator: &Simulator) -> bool {
    let state = simulator.state();
    state.combat.is_some()
        && state.run.as_ref().is_some_and(|run| {
            matches!(
                crate::selfplay::standing_room(run).1,
                Some(MapPointType::Elite)
            )
        })
}

/// The act, counted from zero, of the boss fight the run stands inside, if
/// it stands in one.
fn boss_fight_act(simulator: &Simulator) -> Option<usize> {
    let state = simulator.state();
    let run = state.run.as_ref()?;
    (state.combat.is_some()
        && matches!(
            crate::selfplay::standing_room(run).1,
            Some(MapPointType::Boss)
        ))
    .then_some(run.current_act)
}

/// Whether the run reached a terminal — the episode boundary a trajectory's
/// `done` names.
///
/// Privileged for the same reason the reward is, and kept here beside it so
/// that reading the authoritative state stays one module's job. A run that
/// stopped for any other reason — the harness's step cap, most of all — is
/// *truncated* rather than done, and a truncated episode carries a bootstrap
/// value instead: read as terminal, a cut-off climb would teach the critic
/// that the run ended where the batch ran out of patience.
#[must_use]
pub fn terminated(simulator: &Simulator) -> bool {
    simulator.state().terminal.is_some()
}
