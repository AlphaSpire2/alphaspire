//! What a batch of PPO episodes came to, written as one JSON file beside its
//! shards.
//!
//! The rollout half of the loop's metrics. The trainer owns the
//! metrics sink and reads this file for the scalars it logs beside its own —
//! so the shape is a shallow object of objects, every numeric leaf one level
//! down, and everything that is not a number is there for a human reading the
//! file directly.
//!
//! **Why a clear rate is not enough.** A climb that rests at 76 of 80 hit
//! points scores exactly like one that did not, and a generation whose
//! policy is drifting toward wasted rest sites and skipped card rewards looks
//! identical in the one number a promotion gate reads. The waste counters and
//! the per-fight records are what make that visible, and both are cheap to
//! collect while the run is being played and impossible to reconstruct after
//! it: a finished [`RunReport`] knows the floor the run died on and nothing
//! about the fifteen fights it survived to get there.
//!
//! The collection has no shared state of any kind. [`EpisodeMetrics`] belongs
//! to one run, is fed by the actor playing it, and rides back to the batch on
//! that run's own report — which the harness has already reordered into run
//! order, so a summary is a function of the seed pair like everything else.

use std::path::Path;
use std::time::{Duration, Instant};

use sts2_engine::{
    Action, ChoiceScreen, DecisionContext, GameState, RestSiteOption, RewardFingerprint, RunResult,
    Simulator,
};

use crate::actor::Resolver;
use crate::plan::ActionPlan;
use crate::selfplay::{RunReport, room_label};

/// The fraction of maximum hit points at or above which a rest heal is
/// counted as taken near full.
///
/// A threshold rather than a rule: nothing refuses such a heal, and there are
/// real reasons to take one — a rest site standing between the player and a
/// boss has no later use. What the counter buys is that a policy which rests
/// at nine tenths health as a habit is visible in the batch summary rather
/// than invisible behind an unchanged clear rate.
const NEAR_FULL: f64 = 0.9;

/// The fraction of maximum hit points at or above which a smith is counted
/// as well timed.
///
/// The value of rest play is in *when* the smith happens, not that it
/// happens: forcing it everywhere loses runs, forcing it only up here wins
/// them. A rising smith count below this line is the wrong fix wearing the
/// right counter, so the healthy share is recorded beside the total.
const HEALTHY: f64 = 0.85;

/// The room families the per-fight cost table is cut by, as
/// [`room_label`] names them.
///
/// Room families and nothing finer: which hallway fights are the weak ones is
/// positional inside the simulator, so a difficulty read off a fight ordinal
/// here would be this crate guessing at a rule it cannot see.
const FIGHT_TIERS: [&str; 3] = ["hallway", "elite", "boss"];

/// Where a fight whose room is none of [`FIGHT_TIERS`] is counted.
const OTHER_TIER: &str = "other";

/// The cost table's column for a fight's room family. A fight an event pushed
/// is recorded as a hallway by the engine and lands in that column.
fn fight_tier(room: &'static str) -> &'static str {
    if FIGHT_TIERS.contains(&room) {
        room
    } else {
        OTHER_TIER
    }
}

/// One fight a run entered, as it was seen going in and coming out.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FightRecord {
    /// The act the fight was fought in, counting from one.
    pub act: usize,
    pub floor: u32,
    /// The room family the fight stood in: hallway, elite, boss, or the
    /// event that pushed it.
    pub room: &'static str,
    /// The fight's own model identity, not the room shell around it — two
    /// rooms can spawn the same enemies, and it is the encounter that is
    /// hard.
    pub encounter: Option<String>,
    pub outcome: Outcome,
    pub hp_before: i32,
    pub hp_after: i32,
    pub max_hp: i32,
}

/// How a fight ended for the player.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Won,
    Lost,
    /// The episode stopped — the batch's step cap — while the fight was still
    /// running. Neither a win nor a loss, and counted as neither.
    Unresolved,
}

/// The waste a clear rate is blind to.
///
/// Each counter is something the batch can honestly observe from the outside.
/// The one deliberately absent is the *amount* of healing thrown away against
/// the maximum: the rest screen does not carry the size of its own heal, so
/// how much of it the cap ate is not player-visible, and a number invented
/// from the engine's own arithmetic would be this crate quietly holding a
/// second copy of a rule. What is recorded instead is the fact — the heal
/// ended at full health, so some of it was thrown away — beside the headroom
/// the heal was given, which is the quantity the diagnostic is actually
/// about.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Waste {
    /// Rest-site heals taken.
    pub rest_heals: usize,
    /// Rest-site smiths taken. Not a waste but the heal's alternative: the
    /// pair is one dial, and a batch where this stays at zero while the
    /// heals climb is the collapsed habit the counter exists to show.
    pub rest_smiths: usize,
    /// Of the smiths, the ones taken at or above `HEALTHY` health — the
    /// timing the forced ablation measured the value in.
    pub rest_smiths_healthy: usize,
    /// Of the heals, the ones taken at or above `NEAR_FULL` health.
    pub rest_heals_near_full: usize,
    /// Of those, the ones that ended at full health, so the cap ate whatever
    /// the heal had left over.
    pub rest_heals_capped: usize,
    /// Hit points those heals actually restored.
    pub healing_applied: i64,
    /// Hit points of room they had to restore into — the missing health at
    /// the moment each was taken.
    pub healing_headroom: i64,
    /// Card rewards settled: a card taken, or the claim left standing when
    /// the screen was walked away from.
    pub card_rewards: usize,
    /// Of those, the ones left standing. A skip is not a step of its own —
    /// backing out of the pick screen is withheld from the policy because it
    /// only re-offers the claim — so it is read off the leaving step:
    /// proceeding off a reward screen with a card claim still on it.
    pub card_rewards_skipped: usize,
    /// Gold rewards settled: claimed — by the harness, which takes every
    /// free line before the policy is asked — or left standing when the
    /// screen was walked away from, which only a claim the layer did not
    /// take can leave.
    pub gold_rewards: usize,
    /// Of those, the ones left standing.
    pub gold_rewards_left: usize,
    /// The gold on the screens walked away from: what was left on the table.
    pub gold_left: i64,
    /// Potion rewards settled while the belt had room for them: claimed by
    /// the harness before the policy was asked, or left standing when the
    /// screen was walked away from. A full belt is a different decision —
    /// the claim is inert there and stands only as the take of a trade — and
    /// is not counted.
    pub potion_rewards: usize,
    /// Of those, the ones left standing.
    pub potion_rewards_left: usize,
    /// Relic rewards settled: claimed, or left standing when the screen was
    /// walked away from. A relic on a fight's reward screen is an elite's
    /// drop, so this is the count of elites paid for and not collected.
    pub relic_rewards: usize,
    /// Of those, the ones left standing. Leaving one is right on rare
    /// occasions — a relic whose downside the deck cannot carry — so the
    /// reading is the rate, expected to sit near zero.
    pub relic_rewards_left: usize,
}

/// The trades a run stood in front of and the ones it took, by screen.
///
/// A trade — a held potion given up for an offered one — is the whole reason
/// the policy scores plans, so whether it ever takes one is the readout that
/// says whether that was worth building. Counted per screen because a shop
/// charges gold for the offer and a reward screen does not, and a policy that
/// trades at one and never at the other is saying something.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Trades {
    /// Macro decisions that offered at least one trade.
    pub offered: usize,
    /// Of those, the ones answered by taking one.
    pub taken: usize,
    pub reward_offered: usize,
    pub reward_taken: usize,
    pub shop_offered: usize,
    pub shop_taken: usize,
}

/// The widest action list a macro decision offered, by the screen it stood
/// on.
///
/// The number that decides whether a decision is priced at all: a list past
/// [`MAX_ACTIONS`](crate::net::MAX_ACTIONS) is answered with uniform priors
/// and flagged, and both substitutions widen it: a collapsed button becomes
/// one option per answer its screen accepts, and trade enumeration is
/// multiplicative at a shop — every potion on the shelf against every way of
/// emptying a belt slot. Observed rather than inferred, because neither the
/// belt nor the deck has a fixed size.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Widest {
    pub any: usize,
    pub reward: usize,
    pub shop: usize,
    pub rest: usize,
}

impl Widest {
    /// Counts one macro decision's offer, on the screen it stood on.
    fn saw(&mut self, simulator: &Simulator, offered: usize) {
        self.any = self.any.max(offered);
        match simulator.decision() {
            DecisionContext::Rewards { .. } => self.reward = self.reward.max(offered),
            DecisionContext::Shop { .. } => self.shop = self.shop.max(offered),
            DecisionContext::RestSite { .. } => self.rest = self.rest.max(offered),
            _ => {}
        }
    }

    /// The widest of two runs' worth.
    fn merge(&mut self, other: Self) {
        self.any = self.any.max(other.any);
        self.reward = self.reward.max(other.reward);
        self.shop = self.shop.max(other.shop);
        self.rest = self.rest.max(other.rest);
    }
}

/// A fight that has started and not yet finished.
#[derive(Clone, Debug)]
struct OpenFight {
    act: usize,
    floor: u32,
    room: &'static str,
    encounter: Option<String>,
    hp_before: i32,
    max_hp: i32,
}

/// A rest heal named at a decision and not yet applied by the step.
#[derive(Clone, Copy, Debug)]
struct PendingHeal {
    hp_before: i32,
    max_hp: i32,
}

/// One episode's diagnostics, collected by the policy playing it.
///
/// Fed at two points the harness already provides: the decision the actor is
/// about to take, and the state each accepted step produced. Everything here
/// is a function of those two streams, so nothing reaches into the engine and
/// nothing is shared between runs.
#[derive(Clone, Debug, Default)]
pub struct EpisodeMetrics {
    fights: Vec<FightRecord>,
    waste: Waste,
    trades: Trades,
    widest: Widest,
    open: Option<OpenFight>,
    pending_heal: Option<PendingHeal>,
}

impl EpisodeMetrics {
    /// A free reward line the harness is claiming before the policy is
    /// asked — see [`crate::plan::forced_step`]. Settled and never left, so
    /// the offered counters read it and the left ones cannot.
    pub fn forced(&mut self, simulator: &Simulator, action: &Action) {
        if !matches!(simulator.decision(), DecisionContext::Rewards { .. }) {
            return;
        }
        if let Action::ClaimReward { fingerprint, .. } = action {
            match fingerprint.reward_type.as_str() {
                "gold" => self.waste.gold_rewards += 1,
                "potion" => self.waste.potion_rewards += 1,
                _ => {}
            }
        }
    }

    /// The decision the policy is about to take — everything it was offered
    /// and the plan it picked: where the waste and trade counters read what a
    /// macro screen was answered with.
    pub fn decided(&mut self, simulator: &Simulator, offered: &[ActionPlan], chosen: &ActionPlan) {
        self.widest.saw(simulator, offered.len());
        self.traded(simulator, offered, chosen);
        // Off the lead, not the outcome: a smith is a screen opened and
        // answered, so the plan's outcome is the card pick behind it.
        if matches!(simulator.decision(), DecisionContext::RestSite { .. })
            && matches!(
                chosen.lead(),
                Action::RestOption {
                    option: RestSiteOption::Smith,
                    ..
                }
            )
        {
            self.waste.rest_smiths += 1;
            let (hp, max_hp) = player_hp(simulator.state());
            if max_hp > 0 && f64::from(hp) / f64::from(max_hp) >= HEALTHY {
                self.waste.rest_smiths_healthy += 1;
            }
        }
        match (simulator.decision(), chosen.outcome()) {
            (
                DecisionContext::RestSite { .. },
                Action::RestOption {
                    option: RestSiteOption::Heal,
                    ..
                },
            ) => {
                let (hp, max_hp) = player_hp(simulator.state());
                self.waste.rest_heals += 1;
                if max_hp > 0 && f64::from(hp) / f64::from(max_hp) >= NEAR_FULL {
                    self.waste.rest_heals_near_full += 1;
                }
                self.waste.healing_headroom += i64::from((max_hp - hp).max(0));
                // The step has not landed yet, so how much of the heal the
                // cap ate is read off the state it produces.
                self.pending_heal = Some(PendingHeal {
                    hp_before: hp,
                    max_hp,
                });
            }
            (_, Action::ChooseCards { cards, .. }) if answers_a_card_reward(simulator, chosen) => {
                self.waste.card_rewards += 1;
                // The pick settles the claim. An empty pick would leave the
                // reward standing, but that answer is withheld wherever the
                // claim and the pick are one plan, so the skip a policy can
                // actually take is read off the leaving step below.
                self.waste.card_rewards_skipped += usize::from(cards.is_empty());
            }
            _ => {}
        }
        // Leaving the reward screen is where a skip shows: the only way to
        // decline a card reward is to proceed with its claim still standing,
        // and the same step abandons whatever gold and potions stood beside
        // it. Claiming something else first is not a decision about the card
        // — the screen comes back — so it counts nothing here.
        if matches!(simulator.decision(), DecisionContext::Rewards { .. }) {
            if claims_a_potion_with_room(chosen) {
                self.waste.potion_rewards += 1;
            }
            if chosen.steps().any(|action| {
                matches!(action, Action::ClaimReward { fingerprint, .. } if fingerprint.reward_type == "gold")
            }) {
                self.waste.gold_rewards += 1;
            }
            if chosen.steps().any(|action| {
                matches!(action, Action::ClaimReward { fingerprint, .. } if fingerprint.reward_type == "relic")
            }) {
                self.waste.relic_rewards += 1;
            }
        }
        if matches!(simulator.decision(), DecisionContext::Rewards { .. })
            && matches!(chosen.lead(), Action::Proceed | Action::AdvanceAct)
        {
            if standing_reward(offered, |fingerprint| !fingerprint.offered_cards.is_empty()) {
                self.waste.card_rewards += 1;
                self.waste.card_rewards_skipped += 1;
            }
            let gold: i64 = offered
                .iter()
                .flat_map(ActionPlan::steps)
                .filter_map(|action| match action {
                    Action::ClaimReward { fingerprint, .. }
                        if fingerprint.reward_type == "gold" =>
                    {
                        Some(i64::from(fingerprint.gold_amount))
                    }
                    _ => None,
                })
                .sum();
            if gold > 0 {
                self.waste.gold_rewards += 1;
                self.waste.gold_rewards_left += 1;
                self.waste.gold_left += gold;
            }
            if offered.iter().any(claims_a_potion_with_room) {
                self.waste.potion_rewards += 1;
                self.waste.potion_rewards_left += 1;
            }
            if standing_reward(offered, |fingerprint| fingerprint.reward_type == "relic") {
                self.waste.relic_rewards += 1;
                self.waste.relic_rewards_left += 1;
            }
        }
    }

    /// One macro decision's worth of trade counting.
    fn traded(&mut self, simulator: &Simulator, offered: &[ActionPlan], chosen: &ActionPlan) {
        if !offered.iter().any(ActionPlan::is_trade) {
            return;
        }
        let taken = usize::from(chosen.is_trade());
        self.trades.offered += 1;
        self.trades.taken += taken;
        match simulator.decision() {
            DecisionContext::Rewards { .. } => {
                self.trades.reward_offered += 1;
                self.trades.reward_taken += taken;
            }
            DecisionContext::Shop { .. } => {
                self.trades.shop_offered += 1;
                self.trades.shop_taken += taken;
            }
            // An event's own shelf sells potions through the same purchase:
            // counted in the totals, under neither screen.
            _ => {}
        }
    }

    /// The state an accepted step produced: where a fight's two edges are
    /// seen.
    pub fn stepped(&mut self, simulator: &Simulator) {
        let state = simulator.state();
        if let Some(heal) = self.pending_heal.take() {
            let (hp, _) = player_hp(state);
            self.waste.healing_applied += i64::from((hp - heal.hp_before).max(0));
            if hp >= heal.max_hp {
                self.waste.rest_heals_capped += 1;
            }
        }
        let live = !crate::env::fight_over(simulator);
        match (live, self.open.is_some()) {
            (true, false) => self.open = Some(entered(state)),
            // The fight ended where the engine says it did — on the victory
            // screen the combat is kept alive behind, or on the state the
            // player went down in, which the engine closes the combat for.
            (false, true) => self.close(
                state,
                match state.terminal {
                    Some(RunResult::Defeat) => Outcome::Lost,
                    Some(RunResult::Victory) | None => Outcome::Won,
                },
            ),
            _ => {}
        }
    }

    /// The word that the episode is over, with the state it ended on. A run
    /// the step cap stopped mid-fight closes that fight as
    /// [`Outcome::Unresolved`] rather than dropping it: a fight the batch
    /// stopped watching is not a fight the player lost.
    pub fn ended(&mut self, simulator: &Simulator) {
        if self.open.is_none() {
            return;
        }
        // A fight still open here is one no step ever closed, which means the
        // episode stopped standing inside it. A run that *died* closed its
        // fight on the step that killed it, so this arm is the step cap and
        // nothing else — and a defeat reaching it anyway is still a defeat.
        let outcome = match simulator.state().terminal {
            Some(RunResult::Defeat) => Outcome::Lost,
            Some(RunResult::Victory) | None => Outcome::Unresolved,
        };
        self.close(simulator.state(), outcome);
    }

    /// The fights this episode entered, in the order it entered them.
    #[must_use]
    pub fn fights(&self) -> &[FightRecord] {
        &self.fights
    }

    /// What this episode threw away.
    #[must_use]
    pub const fn waste(&self) -> Waste {
        self.waste
    }

    /// What this episode was offered in trade, and took.
    #[must_use]
    pub const fn trades(&self) -> Trades {
        self.trades
    }

    /// The widest macro decision this episode stood in front of.
    #[must_use]
    pub const fn widest(&self) -> Widest {
        self.widest
    }

    /// Closes the open fight on the state that ended it.
    fn close(&mut self, state: &GameState, outcome: Outcome) {
        let Some(open) = self.open.take() else {
            return;
        };
        let (hp_after, _) = player_hp(state);
        self.fights.push(FightRecord {
            act: open.act,
            floor: open.floor,
            room: open.room,
            encounter: open.encounter,
            outcome,
            hp_before: open.hp_before,
            hp_after,
            max_hp: open.max_hp,
        });
    }
}

/// Whether this plan answers a card reward's pick screen — standing on that
/// screen, or claiming the reward and answering it as one decision.
///
/// The claim and the pick are one plan wherever the reward carries cards, so
/// the screen the answer is given from is the rewards screen and not the
/// pick screen. Reading the plan's own opener is what keeps the counters
/// pointing at the same thing either way.
fn answers_a_card_reward(simulator: &Simulator, plan: &ActionPlan) -> bool {
    match plan.opener() {
        Some(Action::ClaimReward { fingerprint, .. }) => !fingerprint.offered_cards.is_empty(),
        _ => matches!(
            simulator.decision(),
            DecisionContext::ChooseCards {
                screen: ChoiceScreen::CardReward,
                ..
            }
        ),
    }
}

/// Whether this plan claims a potion the belt has room for. Such a claim is
/// a plan of its own; with the belt full it is inert and stands only as the
/// take of a trade, which is a different decision.
fn claims_a_potion_with_room(plan: &ActionPlan) -> bool {
    !plan.is_trade()
        && matches!(
            plan.lead(),
            Action::ClaimReward { fingerprint, .. } if fingerprint.reward_type == "potion"
        )
}

/// Whether a reward line matching `wanted` still stands on the screen: some
/// offered plan claims it, on its own or as the take of a trade.
fn standing_reward(offered: &[ActionPlan], wanted: impl Fn(&RewardFingerprint) -> bool) -> bool {
    offered.iter().flat_map(ActionPlan::steps).any(
        |action| matches!(action, Action::ClaimReward { fingerprint, .. } if wanted(fingerprint)),
    )
}

/// The fight a state has just walked into.
fn entered(state: &GameState) -> OpenFight {
    let (hp_before, max_hp) = player_hp(state);
    let (encounter, room) = state
        .run
        .as_ref()
        .map_or((None, None), crate::selfplay::standing_room);
    OpenFight {
        // Acts are indexed from zero inside the engine and counted from one
        // by everything that reads a summary.
        act: state.run.as_ref().map_or(0, |run| run.current_act) + 1,
        floor: state.run.as_ref().map_or(0, |run| run.floor),
        room: room_label(room),
        // The fight's own model outranks the room's: the encounter the fight
        // stood up is what the combat net was asked about.
        encounter: state
            .combat
            .as_ref()
            .map(|combat| combat.encounter_model.to_string())
            .or_else(|| encounter.map(|id| id.to_string())),
        hp_before,
        max_hp,
    }
}

/// The player's hit points and maximum, wherever they currently live: on
/// their creature inside a fight, on the run outside one. The same reading
/// [`CombatStrength`](crate::objective::CombatStrength) scores a fight by,
/// for the same reason — there is one player, in two places.
fn player_hp(state: &GameState) -> (i32, i32) {
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

/// What produced a batch, so that a summary says what it is a summary of.
#[derive(Clone, Debug)]
pub struct Provenance {
    pub run_net: String,
    pub combat_net: String,
    pub resolver: Resolver,
    pub character: String,
    pub ascension: u8,
    /// Episodes the batch was asked for, which a faulted run makes different
    /// from the episodes it played.
    pub runs: usize,
    pub analysis_seed: u64,
    pub max_steps: usize,
}

/// One episode, reduced to what the batch aggregates over it.
#[derive(Clone, Copy, Debug)]
struct Episode {
    /// The whole episode's undiscounted return.
    total: f64,
    floor: u32,
    /// The act the episode ended in, counting from one.
    act: usize,
    acts_cleared: usize,
    terminal: Option<RunResult>,
    /// Macro decisions the actor recorded.
    decisions: usize,
    degraded: usize,
}

/// A batch's episodes in aggregate. Feed it every finished run's report; it
/// reads, it never plays.
#[derive(Clone, Debug)]
pub struct BatchSummary {
    provenance: Provenance,
    episodes: Vec<Episode>,
    fights: Vec<FightRecord>,
    waste: Waste,
    trades: Trades,
    widest: Widest,
    faulted: usize,
    /// What the batch's resolvers spent, summed over the runs that reported
    /// a spend.
    spend: crate::search::BudgetSpend,
    /// The batch's wall clock: started at construction, stopped on the last
    /// episode handed back. Construct the summary where the rollout begins —
    /// a throughput reading is only as honest as that.
    started: Instant,
    elapsed: Duration,
}

impl BatchSummary {
    #[must_use]
    pub fn new(provenance: Provenance) -> Self {
        Self {
            provenance,
            episodes: Vec::new(),
            fights: Vec::new(),
            waste: Waste::default(),
            trades: Trades::default(),
            widest: Widest::default(),
            faulted: 0,
            spend: crate::search::BudgetSpend::default(),
            started: Instant::now(),
            elapsed: Duration::ZERO,
        }
    }

    /// Counts one episode the batch played to its own end.
    pub fn record(&mut self, report: &RunReport) {
        self.elapsed = self.started.elapsed();
        self.episodes.push(Episode {
            // The undiscounted return of the whole episode is the suffix sum
            // standing on its first line, which is what the actor settles `z`
            // to. An episode that recorded nothing returned nothing.
            total: report
                .macro_decisions
                .first()
                .map_or(0.0, |step| f64::from(step.z)),
            floor: report.floor,
            act: report.act + 1,
            acts_cleared: report.acts_cleared,
            terminal: report.terminal,
            decisions: report.macro_decisions.len(),
            degraded: report
                .macro_decisions
                .iter()
                .filter(|step| step.degraded)
                .count(),
        });
        if let Some(metrics) = &report.metrics {
            self.fights.extend_from_slice(metrics.fights());
            let waste = metrics.waste();
            self.waste.rest_heals += waste.rest_heals;
            self.waste.rest_smiths += waste.rest_smiths;
            self.waste.rest_smiths_healthy += waste.rest_smiths_healthy;
            self.waste.rest_heals_near_full += waste.rest_heals_near_full;
            self.waste.rest_heals_capped += waste.rest_heals_capped;
            self.waste.healing_applied += waste.healing_applied;
            self.waste.healing_headroom += waste.healing_headroom;
            self.waste.card_rewards += waste.card_rewards;
            self.waste.card_rewards_skipped += waste.card_rewards_skipped;
            self.waste.gold_rewards += waste.gold_rewards;
            self.waste.gold_rewards_left += waste.gold_rewards_left;
            self.waste.gold_left += waste.gold_left;
            self.waste.potion_rewards += waste.potion_rewards;
            self.waste.potion_rewards_left += waste.potion_rewards_left;
            self.waste.relic_rewards += waste.relic_rewards;
            self.waste.relic_rewards_left += waste.relic_rewards_left;
            let trades = metrics.trades();
            self.trades.offered += trades.offered;
            self.trades.taken += trades.taken;
            self.trades.reward_offered += trades.reward_offered;
            self.trades.reward_taken += trades.reward_taken;
            self.trades.shop_offered += trades.shop_offered;
            self.trades.shop_taken += trades.shop_taken;
            self.widest.merge(metrics.widest());
        }
        if let Some(spend) = &report.budget {
            self.spend.merge(*spend);
        }
    }

    /// Counts one episode a fault ended. Nothing the episode did is counted:
    /// a partial episode's floor and return are the batch's own accident
    /// rather than the policy's play, and averaging them in would move every
    /// other number in the file. The wall clock still stops here, because the
    /// batch spent that time whatever came of it.
    pub fn record_fault(&mut self) {
        self.elapsed = self.started.elapsed();
        self.faulted += 1;
    }

    /// How many episodes were lost to faults.
    #[must_use]
    pub const fn faulted(&self) -> usize {
        self.faulted
    }

    /// The summary as the trainer reads it.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "batch sizes and floor counts are far below f64 precision"
    )]
    pub fn json(&self) -> serde_json::Value {
        let played = self.episodes.len();
        let terminal = self
            .episodes
            .iter()
            .filter(|episode| episode.terminal.is_some())
            .count();
        let returns = self.per_episode(|episode| episode.total);
        let floors = self.per_episode(|episode| f64::from(episode.floor));
        let lengths = self.per_episode(|episode| episode.decisions as f64);
        let cleared = self.per_episode(|episode| episode.acts_cleared as f64);
        let decisions: usize = self.episodes.iter().map(|episode| episode.decisions).sum();
        let degraded: usize = self.episodes.iter().map(|episode| episode.degraded).sum();
        serde_json::json!({
            "episodes": {
                "total": played + self.faulted,
                "played": played,
                "terminal": terminal,
                "truncated": played - terminal,
                "faulted": self.faulted,
            },
            // Wall-clock rate over every episode the batch attempted, faulted
            // ones included — they cost time too. The elapsed seconds are
            // here so the rate can be re-derived against `episodes.total`.
            "throughput": self.throughput_json(),
            "return": {
                "mean": mean(&returns),
                "median": median(&returns),
            },
            // The mean return of the episodes that *ended* in each act, which
            // is what says whether a death in act two costs what the reward
            // meant it to.
            "return_by_act": self.by_act(|episode| Some(episode.total)),
            "outcome": {
                "clear_rate": self.rate(|episode| {
                    matches!(episode.terminal, Some(RunResult::Victory))
                }),
                "mean_floor": mean(&floors),
                "median_floor": median(&floors),
                // Acts cleared as a count rather than a rate: the depth an
                // average run reaches, which a per-act rate table does not
                // say in one number.
                "mean_acts_cleared": mean(&cleared),
            },
            // Clearing act n is standing past its boss, which is exactly what
            // `acts_cleared` counts — so a victory clears all three.
            "clear_by_act": self.acts(|episode, act| episode.acts_cleared >= act),
            "deaths_by_act": self.deaths(),
            "episode_length": {
                "mean": mean(&lengths),
                "median": median(&lengths),
            },
            "fights": self.fights_json(played),
            // What a fight cost in raw hit points, per act and room family.
            // Every mean carries the count it was taken over: the cells are
            // wildly uneven, and a mean over two fights reads exactly like a
            // mean over two hundred without it.
            "hp_lost_by_act_tier": self.hp_lost_by_act_tier(),
            // Whether macro decisions the checkpoint could not price happen
            // at all is a measurement, and this is the measurement. Such a
            // line carries a uniform policy the checkpoint never produced, so
            // the learner keeps it for the value target and drops it from the
            // surrogate.
            //
            // `decisions`/`of` are the macro half, off the recorded lines.
            // `priced` and the three causes under it count every pricing of
            // the run, in-fight ones included, and `dead_ends`/`cycles` the
            // lines its searches could not walk — off what each run's policy
            // reported. `non_finite_answers` is the process-wide count of
            // numbers read out of a checkpoint that were not numbers: any
            // count at all means the checkpoint is broken.
            //
            // Flat, because the trainer logs the numbers exactly one level
            // down and a group buried below that is a group nobody sees.
            "degraded": {
                "decisions": degraded,
                "of": decisions,
                "fraction": if decisions > 0 { degraded as f64 / decisions as f64 } else { 0.0 },
                "priced": self.spend.degradations.priced_total(),
                "over_axis": self.spend.degradations.over_axis,
                "non_finite": self.spend.degradations.non_finite,
                "out_of_scope": self.spend.degradations.out_of_scope,
                "dead_ends": self.spend.degradations.dead_ends,
                "cycles": self.spend.degradations.cycles,
                "deep": self.spend.degradations.deep,
                "long_turns": self.spend.degradations.long_turns,
                "non_finite_answers": crate::net::non_finite_answers(),
            },
            "waste": self.waste_json(played),
            // Whether the policy ever gives a held potion up for an offered
            // one, which is what scoring a trade as one decision exists to
            // make possible.
            "trades": self.trades_json(),
            // The widest list any macro decision offered, against the axis
            // the checkpoint prices: past it a decision is answered with
            // uniform priors and counted under `degraded`.
            "widest_decision": {
                "any": self.widest.any,
                "reward": self.widest.reward,
                "shop": self.widest.shop,
                "priced_axis": crate::net::MAX_ACTIONS,
            },
            "resolver": resolver_json(self.provenance.resolver, self.spend),
            "provenance": {
                "run_net": self.provenance.run_net,
                "combat_net": self.provenance.combat_net,
                "character": self.provenance.character,
                "ascension": self.provenance.ascension,
                "runs": self.provenance.runs,
                "analysis_seed": self.provenance.analysis_seed,
                "max_steps": self.provenance.max_steps,
            },
            // The array lives at the top level rather than under `fights` so
            // that a reader scanning numeric leaves one level down finds the
            // rates and never walks into a generation's worth of records.
            "fight_records": self.fights,
        })
    }

    /// Writes the summary to `path`, creating the directory it names.
    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        if let Some(directory) = path.parent()
            && !directory.as_os_str().is_empty()
        {
            std::fs::create_dir_all(directory)?;
        }
        std::fs::write(path, format!("{}\n", self.json()))
    }

    /// The headline a batch prints when it is done: what a person watching
    /// the terminal needs, with the file holding the rest.
    #[must_use]
    #[allow(clippy::cast_precision_loss, reason = "batch sizes are small")]
    pub fn report(&self) -> String {
        let json = self.json();
        let played = self.episodes.len();
        let clears: Vec<String> = (1..=3)
            .map(|act| {
                format!(
                    "act {act} {:.1}%",
                    100.0
                        * json["clear_by_act"][format!("act{act}")]
                            .as_f64()
                            .unwrap_or(0.0)
                )
            })
            .collect();
        format!(
            "episodes: {played} played, {} terminal, {} truncated, {} faulted \
             in {:.1}s ({:.1} runs/min)\n\
             return: mean {:.2} median {:.2} | floor: mean {:.2} | length: mean {:.1} decisions\n\
             cleared: {} (mean {:.2} acts) | act-1 hallway hp lost {:.1} over {} won \
             | degraded {} of {} decisions\n\
             elites: {} fought ({:.2} per episode)\n\
             degradations, fights included: {} | non-finite answers {}\n\
             waste: {} rest heals ({} near full, {} capped), {} smiths ({} healthy) | card rewards skipped {} of {}\n\
             left on the table: gold rewards {} of {} ({} gold, {:.1} per run) | potion rewards {} of {} the belt had room for | relic rewards {} of {}\n\
             trades: {} taken of {} offered ({} of {} at rewards, {} of {} at shops)\n\
             widest decision: {} actions ({} at a reward, {} at a shop) against an axis of {}\n",
            json["episodes"]["terminal"],
            json["episodes"]["truncated"],
            self.faulted,
            json["throughput"]["wall_seconds"].as_f64().unwrap_or(0.0),
            json["throughput"]["runs_per_minute"]
                .as_f64()
                .unwrap_or(0.0),
            json["return"]["mean"].as_f64().unwrap_or(0.0),
            json["return"]["median"].as_f64().unwrap_or(0.0),
            json["outcome"]["mean_floor"].as_f64().unwrap_or(0.0),
            json["episode_length"]["mean"].as_f64().unwrap_or(0.0),
            clears.join(", "),
            json["outcome"]["mean_acts_cleared"].as_f64().unwrap_or(0.0),
            json["hp_lost_by_act_tier"]["act1_hallway_mean"]
                .as_f64()
                .unwrap_or(0.0),
            json["hp_lost_by_act_tier"]["act1_hallway_n"],
            json["degraded"]["decisions"],
            json["degraded"]["of"],
            json["fights"]["elites"],
            json["fights"]["elites_per_episode"].as_f64().unwrap_or(0.0),
            self.spend.degradations,
            crate::net::non_finite_answers(),
            self.waste.rest_heals,
            self.waste.rest_heals_near_full,
            self.waste.rest_heals_capped,
            self.waste.rest_smiths,
            self.waste.rest_smiths_healthy,
            self.waste.card_rewards_skipped,
            self.waste.card_rewards,
            self.waste.gold_rewards_left,
            self.waste.gold_rewards,
            self.waste.gold_left,
            json["waste"]["gold_left_per_run"].as_f64().unwrap_or(0.0),
            self.waste.potion_rewards_left,
            self.waste.potion_rewards,
            self.waste.relic_rewards_left,
            self.waste.relic_rewards,
            self.trades.taken,
            self.trades.offered,
            self.trades.reward_taken,
            self.trades.reward_offered,
            self.trades.shop_taken,
            self.trades.shop_offered,
            self.widest.any,
            self.widest.reward,
            self.widest.shop,
            crate::net::MAX_ACTIONS,
        )
    }

    /// One value per played episode, in the order the batch recorded them.
    #[allow(clippy::cast_precision_loss, reason = "batch sizes are small")]
    fn per_episode(&self, value: impl Fn(&Episode) -> f64) -> Vec<f64> {
        self.episodes.iter().map(value).collect()
    }

    /// The batch's wall clock and what it played at.
    #[allow(clippy::cast_precision_loss, reason = "batch sizes are small")]
    fn throughput_json(&self) -> serde_json::Value {
        let seconds = self.elapsed.as_secs_f64();
        let attempted = (self.episodes.len() + self.faulted) as f64;
        serde_json::json!({
            "wall_seconds": seconds,
            "runs_per_minute": if seconds > 0.0 { 60.0 * attempted / seconds } else { 0.0 },
        })
    }

    /// The batch's fights: the pooled rates, and the elites called out of
    /// them.
    ///
    /// Elites separately because they are the fights a macro policy
    /// *chooses*: a route that walks past every one of them clears act one
    /// at a rate a route that fights them cannot, and the pooled fight count
    /// reads the same either way. How many an average climb took on is the
    /// number the routing question is about, so it is counted over played
    /// episodes and a batch that played nothing reports zero rather than
    /// dividing by it. There is deliberately no elite win rate: an elite the
    /// player lost ended the run, so the rate is the death table already
    /// under `deaths_by_act` said backwards.
    #[allow(clippy::cast_precision_loss, reason = "fight counts are small")]
    fn fights_json(&self, played: usize) -> serde_json::Value {
        let won = self
            .fights
            .iter()
            .filter(|fight| fight.outcome == Outcome::Won)
            .count();
        let resolved = self
            .fights
            .iter()
            .filter(|fight| fight.outcome != Outcome::Unresolved)
            .count();
        let elites = self.elites();
        serde_json::json!({
            "total": self.fights.len(),
            "won": won,
            "unresolved": self.fights.len() - resolved,
            "win_rate": if resolved > 0 { won as f64 / resolved as f64 } else { 0.0 },
            "elites": elites,
            "elites_per_episode": if played > 0 {
                elites as f64 / played as f64
            } else {
                0.0
            },
        })
    }

    /// The trade counters with the fraction that makes them readable.
    #[allow(clippy::cast_precision_loss, reason = "counts are small")]
    fn trades_json(&self) -> serde_json::Value {
        let trades = self.trades;
        serde_json::json!({
            "offered": trades.offered,
            "taken": trades.taken,
            "reward_offered": trades.reward_offered,
            "reward_taken": trades.reward_taken,
            "shop_offered": trades.shop_offered,
            "shop_taken": trades.shop_taken,
            "take_fraction": if trades.offered > 0 {
                trades.taken as f64 / trades.offered as f64
            } else {
                0.0
            },
        })
    }

    /// The waste counters with the two fractions that make them readable.
    #[allow(clippy::cast_precision_loss, reason = "counts are small")]
    fn waste_json(&self, played: usize) -> serde_json::Value {
        let waste = self.waste;
        serde_json::json!({
            "rest_heals": waste.rest_heals,
            "rest_smiths": waste.rest_smiths,
            "rest_smiths_healthy": waste.rest_smiths_healthy,
            "rest_heals_near_full": waste.rest_heals_near_full,
            "rest_heals_capped": waste.rest_heals_capped,
            "healing_applied": waste.healing_applied,
            "healing_headroom": waste.healing_headroom,
            "card_rewards": waste.card_rewards,
            "card_rewards_skipped": waste.card_rewards_skipped,
            "card_skip_fraction": if waste.card_rewards > 0 {
                waste.card_rewards_skipped as f64 / waste.card_rewards as f64
            } else {
                0.0
            },
            "gold_rewards": waste.gold_rewards,
            "gold_rewards_left": waste.gold_rewards_left,
            "gold_left": waste.gold_left,
            "gold_left_per_run": if played > 0 {
                waste.gold_left as f64 / played as f64
            } else {
                0.0
            },
            "potion_rewards": waste.potion_rewards,
            "potion_rewards_left": waste.potion_rewards_left,
            "relic_rewards": waste.relic_rewards,
            "relic_rewards_left": waste.relic_rewards_left,
            "relic_left_fraction": if waste.relic_rewards > 0 {
                waste.relic_rewards_left as f64 / waste.relic_rewards as f64
            } else {
                0.0
            },
            "near_full_fraction": if waste.rest_heals > 0 {
                waste.rest_heals_near_full as f64 / waste.rest_heals as f64
            } else {
                0.0
            },
        })
    }

    /// The share of episodes `holds` is true of.
    #[allow(clippy::cast_precision_loss, reason = "batch sizes are small")]
    fn rate(&self, holds: impl Fn(&Episode) -> bool) -> f64 {
        if self.episodes.is_empty() {
            return 0.0;
        }
        self.episodes
            .iter()
            .filter(|episode| holds(episode))
            .count() as f64
            / self.episodes.len() as f64
    }

    /// One entry per act of a three-act climb, keyed `act1` through `act3`.
    /// Per act and never pooled: training across all three at once displaces
    /// encounter-specific competence, and a pooled number is exactly what
    /// hides it.
    fn acts(&self, holds: impl Fn(&Episode, usize) -> bool) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        for act in 1..=3 {
            object.insert(
                format!("act{act}"),
                self.rate(|episode| holds(episode, act)).into(),
            );
        }
        serde_json::Value::Object(object)
    }

    /// The mean of `value` over the episodes that ended in each act.
    fn by_act(&self, value: impl Fn(&Episode) -> Option<f64>) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        for act in 1..=3 {
            let values: Vec<f64> = self
                .episodes
                .iter()
                .filter(|episode| episode.act == act)
                .filter_map(&value)
                .collect();
            object.insert(format!("act{act}"), mean(&values).into());
        }
        serde_json::Value::Object(object)
    }

    /// Hit points lost per fight, keyed `act{n}_{tier}_mean` and
    /// `act{n}_{tier}_n` over every act and [`FIGHT_TIERS`] column.
    ///
    /// Only won fights are averaged. A lost fight ends at zero hit points by
    /// definition, so its cost is the health the player walked in with rather
    /// than what the fight charged, and one of them swamps any cell it enters
    /// — the losses are counted by `deaths_by_act` and `fights`, which is
    /// where they mean something. A cell no fight landed in is a zero over a
    /// count of zero rather than an absent key: a column that comes and goes
    /// between batches is worse for a metrics reader than a stable zero.
    fn hp_lost_by_act_tier(&self) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        for act in 1..=3 {
            for tier in FIGHT_TIERS.into_iter().chain([OTHER_TIER]) {
                let lost: Vec<f64> = self
                    .fights
                    .iter()
                    .filter(|fight| {
                        fight.act == act
                            && fight.outcome == Outcome::Won
                            && fight_tier(fight.room) == tier
                    })
                    .map(|fight| f64::from(fight.hp_before - fight.hp_after))
                    .collect();
                object.insert(format!("act{act}_{tier}_mean"), mean(&lost).into());
                object.insert(format!("act{act}_{tier}_n"), lost.len().into());
            }
        }
        serde_json::Value::Object(object)
    }

    /// The elite fights the batch entered.
    ///
    /// Counted off the room family rather than the encounter, for the reason
    /// [`FIGHT_TIERS`] is: which encounters stand in an elite room is the
    /// simulator's to say, and a list of elite model ids kept here would be
    /// this crate holding a second copy of it.
    fn elites(&self) -> usize {
        self.fights
            .iter()
            .filter(|fight| fight_tier(fight.room) == "elite")
            .count()
    }

    /// Deaths counted by the act they happened in.
    fn deaths(&self) -> serde_json::Value {
        let mut object = serde_json::Map::new();
        for act in 1..=3 {
            let deaths = self
                .episodes
                .iter()
                .filter(|episode| {
                    episode.act == act && matches!(episode.terminal, Some(RunResult::Defeat))
                })
                .count();
            object.insert(format!("act{act}"), deaths.into());
        }
        serde_json::Value::Object(object)
    }
}

/// The resolver's configuration and what it spent, so a batch says what
/// answered its fights.
fn resolver_json(resolver: Resolver, spend: crate::search::BudgetSpend) -> serde_json::Value {
    let mut json = match resolver {
        Resolver::Greedy => serde_json::json!({ "kind": "greedy" }),
        Resolver::Searched {
            config,
            selection,
            budget,
            elite,
            boss,
        } => {
            let mut json = serde_json::json!({
                "kind": "searched",
                "iterations": config.iterations,
                "considered": selection.considered,
                // Zero reads as off, which is what an absent ceiling is: no run
                // of the batch was allowed fewer steps than any other.
                "budget_steps": budget.steps.unwrap_or(0),
            });
            // A tier that spends its own budget names it; one that spends
            // the base budget is absent, so an older summary reads the same.
            for (name, tier) in [("elite", elite), ("boss", boss)] {
                if let Some(tier) = tier {
                    json[name] = serde_json::json!({
                        "iterations": tier.iterations,
                        "considered": tier.considered,
                    });
                }
            }
            json
        }
    };
    let object = json.as_object_mut().expect("the resolver is an object");
    object.insert("decisions_searched".into(), spend.searched.into());
    object.insert("decisions_downgraded".into(), spend.downgraded.into());
    object.insert("steps".into(), spend.steps.into());
    json
}

/// One line for the episode a batch has just finished.
#[must_use]
pub fn episode_line(report: &RunReport) -> String {
    use std::fmt::Write as _;
    let total = report.macro_decisions.first().map_or(0.0, |step| step.z);
    let mut line = format!(
        "{} act {} floor {} {} | return {total:.2} over {} decisions",
        report.seed,
        report.act,
        report.floor,
        report
            .terminal
            .map_or("unfinished", |terminal| match terminal {
                RunResult::Victory => "VICTORY",
                RunResult::Defeat => "DEFEAT",
            }),
        report.macro_decisions.len(),
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
    if let Some(metrics) = &report.metrics {
        let won = metrics
            .fights()
            .iter()
            .filter(|fight| fight.outcome == Outcome::Won)
            .count();
        let _ = write!(line, " | fights {won}/{}", metrics.fights().len());
        let trades = metrics.trades();
        if trades.offered > 0 {
            let _ = write!(line, " | traded {}/{}", trades.taken, trades.offered);
        }
    }
    // What the fights cost, and — the number a ceiling exists to make
    // visible — how many of them the ceiling took away from the search.
    if let Some(spend) = &report.budget
        && spend.searched + spend.downgraded > 0
    {
        let _ = write!(
            line,
            " | resolver {} searched {} downgraded over {} steps",
            spend.searched, spend.downgraded, spend.steps,
        );
    }
    // What the episode answered with something other than what it was asked
    // for: silent on a run that degraded nothing, and named where it did.
    if let Some(spend) = &report.budget
        && spend.degradations.any()
    {
        let _ = write!(line, " | degraded: {}", spend.degradations);
    }
    line
}

/// The arithmetic mean, and zero over nothing: an empty cell of a per-act
/// table is a cell no episode landed in, which is a fact about the batch
/// rather than a missing number.
#[allow(clippy::cast_precision_loss, reason = "batch sizes are small")]
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// The median, taking the lower of the two middles on an even count.
///
/// Reported beside every mean because a return distribution with a large
/// terminal term is not symmetric: one cleared run pulls a mean that no
/// episode of the batch resembles, and the pair is what makes that visible.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) / 2]
}

#[cfg(test)]
mod tests {
    use super::{BatchSummary, Episode, FightRecord, Outcome, Provenance, fight_tier};
    use crate::actor::Resolver;

    fn summary() -> BatchSummary {
        BatchSummary::new(Provenance {
            run_net: "ckpts/run".to_owned(),
            combat_net: "ckpts/combat".to_owned(),
            resolver: Resolver::Greedy,
            character: "CHARACTER.IRONCLAD".to_owned(),
            ascension: 0,
            runs: 0,
            analysis_seed: 1,
            max_steps: 4000,
        })
    }

    /// One fight in a named act and room family, costing `hp_before - hp_after`.
    fn fight(
        act: usize,
        room: &'static str,
        outcome: Outcome,
        hp_before: i32,
        hp_after: i32,
    ) -> FightRecord {
        FightRecord {
            act,
            floor: 1,
            room,
            encounter: Some("ENCOUNTER.TEST".to_owned()),
            outcome,
            hp_before,
            hp_after,
            max_hp: 80,
        }
    }

    /// One played episode that cleared `acts_cleared` acts.
    fn episode(acts_cleared: usize) -> Episode {
        Episode {
            total: 0.0,
            floor: 1,
            act: acts_cleared + 1,
            acts_cleared,
            terminal: None,
            decisions: 1,
            degraded: 0,
        }
    }

    fn number(value: &serde_json::Value) -> f64 {
        value
            .as_f64()
            .unwrap_or_else(|| panic!("{value} is a number"))
    }

    #[test]
    fn every_cell_of_the_cost_table_carries_a_mean_and_the_count_behind_it() {
        let mut summary = summary();
        summary.fights.extend([
            fight(1, "hallway", Outcome::Won, 80, 74),
            fight(1, "hallway", Outcome::Won, 74, 70),
            fight(1, "elite", Outcome::Won, 70, 40),
            fight(2, "boss", Outcome::Won, 60, 35),
        ]);
        let table = summary.json()["hp_lost_by_act_tier"].clone();
        assert!((number(&table["act1_hallway_mean"]) - 5.0).abs() < 1e-9);
        assert_eq!(table["act1_hallway_n"], 2);
        assert!((number(&table["act1_elite_mean"]) - 30.0).abs() < 1e-9);
        assert_eq!(table["act1_elite_n"], 1);
        assert!((number(&table["act2_boss_mean"]) - 25.0).abs() < 1e-9);
        assert_eq!(table["act2_boss_n"], 1);
    }

    #[test]
    fn a_cell_no_fight_landed_in_is_a_stable_zero_over_zero() {
        // Act three is empty for most of training, and a key that comes and
        // goes between batches is worse for a metrics reader than a zero.
        let mut summary = summary();
        summary
            .fights
            .push(fight(1, "hallway", Outcome::Won, 80, 74));
        let table = summary.json()["hp_lost_by_act_tier"].clone();
        for act in 1..=3 {
            for tier in ["hallway", "elite", "boss", "other"] {
                assert!(
                    table[format!("act{act}_{tier}_mean")].is_number()
                        && table[format!("act{act}_{tier}_n")].is_number(),
                    "act{act} {tier} is present whether or not it was fought"
                );
            }
        }
        assert_eq!(table["act3_boss_n"], 0);
        assert_eq!(table["act3_boss_mean"], 0.0);
    }

    #[test]
    fn a_fight_the_player_did_not_win_is_left_out_of_the_mean_it_would_swamp() {
        // A lost fight ends at zero, so its cost is the health the player
        // walked in with; an unresolved one was never finished.
        let mut summary = summary();
        summary.fights.extend([
            fight(1, "elite", Outcome::Won, 80, 60),
            fight(1, "elite", Outcome::Lost, 70, 0),
            fight(1, "elite", Outcome::Unresolved, 70, 55),
        ]);
        let table = summary.json()["hp_lost_by_act_tier"].clone();
        assert_eq!(table["act1_elite_n"], 1);
        assert!((number(&table["act1_elite_mean"]) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn a_batch_that_played_nothing_reports_no_rate_rather_than_dividing_by_zero() {
        let json = summary().json();
        assert_eq!(json["throughput"]["wall_seconds"], 0.0);
        assert_eq!(json["throughput"]["runs_per_minute"], 0.0);
    }

    #[test]
    fn a_room_family_the_table_has_no_column_for_is_counted_rather_than_dropped() {
        assert_eq!(fight_tier("hallway"), "hallway");
        assert_eq!(fight_tier("rest site"), "other");
        let mut summary = summary();
        summary.fights.push(fight(2, "event", Outcome::Won, 50, 41));
        let table = summary.json()["hp_lost_by_act_tier"].clone();
        assert_eq!(table["act2_other_n"], 1);
        assert!((number(&table["act2_other_mean"]) - 9.0).abs() < 1e-9);
    }

    #[test]
    fn elites_are_counted_apart_from_the_pooled_fights() {
        // The routing question: two batches with the same fight count and
        // the same clear rate differ entirely if one of them walked past
        // every elite.
        let mut summary = summary();
        summary.episodes.extend([episode(0), episode(1)]);
        summary.fights.extend([
            fight(1, "hallway", Outcome::Won, 80, 74),
            fight(1, "elite", Outcome::Won, 74, 44),
            fight(2, "elite", Outcome::Lost, 44, 0),
            // However it ended, the route walked into it.
            fight(2, "elite", Outcome::Unresolved, 60, 50),
            fight(2, "boss", Outcome::Won, 60, 35),
        ]);
        let fights = summary.json()["fights"].clone();
        assert_eq!(fights["elites"], 3);
        assert!((number(&fights["elites_per_episode"]) - 1.5).abs() < 1e-9);
        // And the headline says it, in the terminal of whoever is watching
        // the generation go by rather than only in the file.
        assert!(
            summary
                .report()
                .contains("elites: 3 fought (1.50 per episode)"),
            "{}",
            summary.report()
        );
    }

    #[test]
    fn a_batch_that_fought_no_elite_reports_zero_rather_than_dividing_by_it() {
        let mut summary = summary();
        summary
            .fights
            .push(fight(1, "hallway", Outcome::Won, 80, 74));
        let fights = summary.json()["fights"].clone();
        assert_eq!(fights["elites"], 0);
        assert!(number(&fights["elites_per_episode"]).abs() < 1e-9);
    }

    #[test]
    fn acts_cleared_is_averaged_over_the_episodes_that_were_played() {
        let mut summary = summary();
        summary
            .episodes
            .extend([episode(0), episode(1), episode(2)]);
        let outcome = summary.json()["outcome"].clone();
        assert!((number(&outcome["mean_acts_cleared"]) - 1.0).abs() < 1e-9);
    }
}
