//! Training samples: what expert iteration learns from, and how a batch
//! writes them.
//!
//! One [`Decision`] per searched decision — the observation, the canonical
//! actions, the search's root policy over them, and the fight's eventual
//! value — encoded into a [`TrainingSample`] where it is written, or kept
//! as recorded for a later encoding. Decisions are recorded only by belief
//! search: a clairvoyant search's policy is a teacher that peeked, and
//! nothing here may know what a player could not.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::encoding::{ActionEncoding, ObservationEncoding, PolicyEncoder};
use crate::plan::ActionPlan;
use sts2_core::ModelId;
use sts2_engine::AgentObservation;

/// The sample format version. Observations omit padding on disk; readers
/// restore it to `MAX_TOKENS`. Lines carry run/fight provenance and optional
/// act indices. Combat and macro samples have different header scopes and
/// value semantics and use separate shard directories.
///
/// PPO lines also carry `chosen`, `logp`, `value`, `reward`, `done`,
/// `bootstrap` and `degraded`. Their `pi` is the actor's policy and `z` is
/// the undiscounted suffix return; the learner applies its own discount.
/// These fields are omitted from non-PPO samples.
///
/// In format 5, `pi`, `chosen` and `logp` range over [`ActionPlan`]s, which
/// can contain multiple engine steps. Combat plans are single actions.
/// Format-2 lines remain readable with absent encounter IDs and zero
/// run/fight stamps. The optional header character list describes the
/// source data; see `stamp_characters`.
pub const SAMPLE_FORMAT: u32 = 5;

/// The header `scope` of a shard directory holding in-combat samples: what
/// the combat net trains on.
pub const SCOPE_COMBAT: &str = "combat";

/// The header `scope` of a shard directory holding macro samples: what the
/// run net trains on.
pub const SCOPE_MACRO: &str = "macro";

/// Where a macro shard's decisions came from, stamped in its header beside
/// [`SCOPE_MACRO`] and following exactly the same idiom: `scope` says which
/// net the data belongs to, `source` says which teacher answered it.
///
/// Two teachers write macro samples, and a set trained on one is not a set
/// trained on the other:
///
/// - [`SOURCE_HEURISTIC`] — the hand-written macro policy answered, and π is
///   a softmax of its own action scores. Cheap: every belief-mode self-play
///   run already plays ~50 of these, so a combat generation's batch emits a
///   macro set for free. Its ceiling is the teacher's.
/// - [`SOURCE_SEARCH`] — act-mode MCTS answered, and π is the root's
///   improved policy. Expensive, and the only source that can exceed the
///   heuristic.
///
/// One source per shard directory, because one batch has one teacher. A
/// header key rather than a per-line one for the same reason `scope` is a
/// header key, and — since combat headers are untouched — a combat shard
/// stays byte-identical to what this build always wrote.
pub const SOURCE_HEURISTIC: &str = "heuristic";

/// See [`SOURCE_HEURISTIC`].
pub const SOURCE_SEARCH: &str = "search";

/// The header `training_mode` of a shard directory holding PPO trajectories:
/// what a run-level rollout writes, and the key the trainer dispatches its
/// loss on.
///
/// A third header key rather than a third `scope` or a third
/// `value_semantics`, because it answers a third question. `scope` says which
/// net the data belongs to and `value_semantics` says what that net's value
/// head predicts; both are already `macro` and
/// [`RUN_VALUE_SEMANTICS`](crate::objective::RUN_VALUE_SEMANTICS) for a PPO
/// set, and neither says whether a line teaches a distribution or an action
/// actually taken. That difference is the whole difference between a
/// cross-entropy fit and an importance-ratio surrogate, so it gets a key of
/// its own — and a set without one is a set the PPO loss must refuse rather
/// than reinterpret.
///
/// Absent on every other set, exactly as `source` is: expert-iteration shards
/// stay byte-identical to what this build always wrote.
pub const TRAINING_MODE_PPO: &str = "ppo";

/// One decision, as the trainer sees it. `pi` is aligned with `actions` and
/// sums to one; `z` is the objective's score of the fight this decision was
/// part of, read where the fight ended.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingSample {
    pub observation: ObservationEncoding,
    pub actions: Vec<ActionEncoding>,
    pub pi: Vec<f32>,
    pub z: f32,
    /// The encounter the decision was searched in — the fight's own model
    /// identity, not the room shell around it. Absent only on a line
    /// written before format 3.
    #[serde(default)]
    pub encounter: Option<sts2_core::ModelId>,
    /// Which run of the batch recorded the sample — stamped by the sink,
    /// whose write order is run order — and which fight of that run it was
    /// searched in, counted by the recorder.
    #[serde(default)]
    pub run: usize,
    #[serde(default)]
    pub fight: usize,
    /// Which act of the run a *macro* sample was searched in — the act whose
    /// boundary settled its `z`, the way `fight` names the fight that
    /// settled a combat sample's. Absent on every in-combat line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub act: Option<usize>,
    /// Which action of `actions` the actor took.
    ///
    /// This and the six keys below are the trajectory half of a line, and
    /// they appear together or not at all. A searched or imitated line names
    /// no single action — its `pi` is the whole of what it teaches — while a
    /// PPO line is one action actually drawn from a distribution, which is
    /// what an importance ratio can be formed against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen: Option<usize>,
    /// The natural log of the actor's probability of `chosen`, as the actor
    /// that took it computed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logp: Option<f32>,
    /// Uniform exploration mass mixed into the actor's policy on this row.
    /// PPO records zero explicitly when no exploration is applied. Older
    /// trajectories and non-PPO samples omit this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exploration_epsilon: Option<f32>,
    /// The critic's value of this state at rollout time, exactly as the
    /// checkpoint emitted it — unscaled, because the learner scales its own
    /// targets and rescaling here would apply that twice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f32>,
    /// What the environment paid for the step out of this state: everything
    /// accrued between this decision and the next one the actor made, so a
    /// decision that walked into a room is paid for the whole fight inside
    /// it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<f32>,
    /// Whether that step ended the episode. True only on a terminal — a run
    /// won or lost — and false on a truncation, which is a climb the batch
    /// stopped watching rather than a climb that ended.
    #[serde(default, skip_serializing_if = "is_false")]
    pub done: bool,
    /// The critic's value of the state a *truncated* episode stopped on,
    /// present on that episode's last line and on no other. It is what keeps
    /// the learner from reading a step cap as a death.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap: Option<f32>,
    /// Whether `pi` is the uniform fallback rather than a policy the
    /// checkpoint produced — a decision offering more actions than the net's
    /// action axis prices. Honest degradation for a search, which loses
    /// guidance on one node; dishonest for a learner, which would form a
    /// ratio against a log-probability no net ever emitted. Flagged so the
    /// surrogate can drop the row while the value target keeps it.
    #[serde(default, skip_serializing_if = "is_false")]
    pub degraded: bool,
    /// Whether this decision lies on a counterfactual branch: a line the
    /// search rated close to its answer but the prior all but refused,
    /// played out from that action to the fight's end and scored there.
    /// A branch shares its parent's `run` and `fight` — it is the same
    /// fight's family for a held-out split — and its `z` is its own.
    #[serde(default, skip_serializing_if = "is_false")]
    pub counterfactual: bool,
    /// How far apart the search's completed values were at this decision,
    /// `max − min`, read in the objective's own units before the transform
    /// that wrote `pi` normalized them away.
    ///
    /// The evidence `pi` rests on. The improved-policy transform divides
    /// this out — a decision whose best action wins by a thousandth and one
    /// whose best action wins outright are written as the same label — so
    /// without it nothing downstream can tell a confident label from a
    /// confidently guessed one. A learner is free to weight rows by it.
    ///
    /// Absent on a macro line, which no tree stood at, and on every line
    /// written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q_spread: Option<f32>,
    /// The search's own value of this decision's state: the root's mixed
    /// value over its completed Q, in the objective's units — the same
    /// scale as `z`, and the line's `value_semantics` names what both mean.
    ///
    /// Where `z` is one number smeared over every decision of the fight,
    /// this one is per-decision: it moves when a play moves the state,
    /// which is what a bootstrapped value target needs. It is also the
    /// search's estimate rather than the horizon's answer, so a learner
    /// that blends it into the value target is trading variance for the
    /// net's own bias — recorded, like `q_spread`, so that trade is made
    /// at training time rather than at search time.
    ///
    /// Absent on a macro-imitation line (no tree stood there), on a PPO
    /// line (whose `value` is the critic's read, a different quantity),
    /// and on every line written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_value: Option<f32>,
}

/// Whether a flag is off, and so absent from the wire: `false` is what a
/// line without the key means, and writing it would put seven redundant
/// bytes on every in-combat line ever recorded.
#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde hands skip_serializing_if the field by reference"
)]
fn is_false(flag: &bool) -> bool {
    !*flag
}

/// One decision as it was recorded, before any encoding: what the player
/// saw, the canonical actions on offer, the policy aligned onto them, and —
/// once its horizon settled it — its value.
///
/// A [`TrainingSample`] is a function of exactly this and a
/// [`PolicyEncoder`], applied at write time by [`SampleSink`] or later by
/// `alphaspire encode` over what a [`DecisionSink`] kept. The search is the
/// expensive half of a generation and the encoding the cheap one; recorded
/// this way, one search pass reads through any encoding this build or a
/// later one carries, and a new encoding no longer costs a new search.
///
/// The stamps and the trajectory keys are [`TrainingSample`]'s, documented
/// there and carried here unchanged so that a recorded set encodes into
/// exactly the samples the batch would have written.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Decision {
    pub observation: AgentObservation,
    /// What the decision offered, as the policy scored it. Almost every
    /// entry is one engine action; a trade is two.
    pub actions: Vec<ActionPlan>,
    pub pi: Vec<f32>,
    pub z: f32,
    #[serde(default)]
    pub encounter: Option<sts2_core::ModelId>,
    #[serde(default)]
    pub run: usize,
    #[serde(default)]
    pub fight: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub act: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logp: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exploration_epsilon: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<f32>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub done: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap: Option<f32>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub degraded: bool,
    /// Whether this decision lies on a counterfactual branch — see
    /// [`TrainingSample::counterfactual`].
    #[serde(default, skip_serializing_if = "is_false")]
    pub counterfactual: bool,
    /// Which rule took the branch this decision lies on — `rival` or
    /// `pass` — so a recorded set can be split by it. Absent off a branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counterfactual_kind: Option<String>,
    /// The evidence `pi` rests on: `max − min` over the search's completed
    /// values at this decision, before the transform normalized them.
    /// Documented on [`TrainingSample`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q_spread: Option<f32>,
    /// The root's own mixed value at this decision, in the objective's
    /// units. Documented on [`TrainingSample`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_value: Option<f32>,
}

impl Decision {
    /// The decision as the trainer sees it under `encoder`.
    #[must_use]
    pub fn encode(&self, encoder: &PolicyEncoder) -> TrainingSample {
        TrainingSample {
            observation: encoder.encode_observation(&self.observation),
            actions: encoder.encode_plans(&self.observation, &self.actions),
            pi: self.pi.clone(),
            z: self.z,
            encounter: self.encounter.clone(),
            run: self.run,
            fight: self.fight,
            act: self.act,
            chosen: self.chosen,
            logp: self.logp,
            exploration_epsilon: self.exploration_epsilon,
            value: self.value,
            reward: self.reward,
            done: self.done,
            bootstrap: self.bootstrap,
            degraded: self.degraded,
            counterfactual: self.counterfactual,
            q_spread: self.q_spread,
            root_value: self.root_value,
        }
    }
}

/// The recorded-decision file format: what a [`DecisionSink`] writes and
/// `alphaspire encode` reads. Pinned to the observation version, the belief
/// model and the game build — everything a decision's meaning depends on —
/// and to nothing about any encoding, which is the point.
///
/// Version 2 records an [`ActionPlan`] per offer
/// rather than a bare action. A single-action plan is written exactly as the
/// action was, so every line a fight records is unchanged; a trade is a shape
/// a version 1 reader has no case for, which is what the bump says.
pub const DECISION_FORMAT: u32 = 2;

/// Every version and dimension the samples depend on. Each shard opens with
/// this line, so one shard alone is loadable and self-validating.
fn header(
    encoder: &PolicyEncoder,
    scope: &str,
    value_semantics: &str,
    source: Option<&str>,
    training_mode: Option<&str>,
    characters: &[ModelId],
) -> serde_json::Value {
    let mut header = serde_json::json!({
        "format": SAMPLE_FORMAT,
        "scope": scope,
        "policy_encoding_version": crate::encoding::POLICY_ENCODING_VERSION,
        "observation_version": sts2_engine::AGENT_OBSERVATION_VERSION,
        "belief_model_version": sts2_engine::BELIEF_MODEL_VERSION,
        "value_semantics": value_semantics,
        "vocabulary_hash": encoder.vocabulary_hash(),
        "vocabulary_size": encoder.vocabulary_size(),
        "observation_scalars": crate::encoding::OBSERVATION_SCALARS,
        "max_tokens": crate::encoding::MAX_TOKENS,
        "token_features": crate::encoding::TOKEN_FEATURES,
        "action_tokens": crate::encoding::ACTION_TOKENS,
        "action_features": crate::encoding::ACTION_FEATURES,
    });
    stamp_source(&mut header, source);
    stamp_training_mode(&mut header, training_mode);
    stamp_characters(&mut header, characters);
    header
}

/// Which teacher answered, where one did — and, for the heuristic, the
/// temperature its scores were read at.
fn stamp_source(header: &mut serde_json::Value, source: Option<&str>) {
    if let Some(source) = source {
        let object = header.as_object_mut().expect("the header is an object");
        object.insert("source".into(), source.into());
        if source == SOURCE_HEURISTIC {
            object.insert(
                "macro_policy_temperature".into(),
                crate::heuristics::MACRO_POLICY_TEMPERATURE.into(),
            );
        }
    }
}

/// Which learner the set is written for, where it is written for a particular
/// one. See [`TRAINING_MODE_PPO`].
fn stamp_training_mode(header: &mut serde_json::Value, training_mode: Option<&str>) {
    if let Some(mode) = training_mode {
        header
            .as_object_mut()
            .expect("the header is an object")
            .insert("training_mode".into(), mode.into());
    }
}

/// Which characters the decisions came from, where the batch knows.
///
/// A list rather than one id, because a set can honestly hold several: a
/// bank replay draws whatever the bank holds, and merging two characters'
/// sets is a deliberate thing to do (one vocabulary covers all content, so
/// a checkpoint trained on one character loads for any). Descriptive, not a
/// pin — nothing refuses a set for its composition; the stamp is what makes
/// a mis-pointed directory or a mixed bank visible after the fact.
///
/// Omitted rather than written empty when the batch does not know, so a set
/// encoded from decisions recorded before the stamp existed is byte-identical
/// to what this build always wrote.
fn stamp_characters(header: &mut serde_json::Value, characters: &[ModelId]) {
    if characters.is_empty() {
        return;
    }
    let mut ids: Vec<String> = characters.iter().map(ToString::to_string).collect();
    ids.sort();
    ids.dedup();
    header
        .as_object_mut()
        .expect("the header is an object")
        .insert("characters".into(), ids.into());
}

/// Every pin a recorded decision depends on, and none an encoding adds.
/// `runs_per_shard` rides along so `alphaspire encode` can shard its output
/// exactly as the batch would have.
fn decision_header(
    scope: &str,
    value_semantics: &str,
    source: Option<&str>,
    training_mode: Option<&str>,
    runs_per_shard: usize,
    characters: &[ModelId],
) -> serde_json::Value {
    let mut header = serde_json::json!({
        "format": DECISION_FORMAT,
        "scope": scope,
        "observation_version": sts2_engine::AGENT_OBSERVATION_VERSION,
        "belief_model_version": sts2_engine::BELIEF_MODEL_VERSION,
        "compatibility_id": sts2_core::PINNED_COMPATIBILITY_ID,
        "value_semantics": value_semantics,
        "runs_per_shard": runs_per_shard,
    });
    stamp_source(&mut header, source);
    stamp_training_mode(&mut header, training_mode);
    stamp_characters(&mut header, characters);
    header
}

/// A directory of shards, one file per `runs_per_shard` runs, that both
/// sinks write through.
///
/// Streaming rather than accumulating is the point. A generation-sized batch
/// is on the order of 10^5 samples; held in memory until the last run
/// finished, that is gigabytes resident before a byte reaches disk. Here each
/// run's lines are written as they arrive and the file rolls every
/// `runs_per_shard` runs, so the batch's memory is one run's worth.
///
/// Shard boundaries are run boundaries, and runs arrive in run order, so
/// which lines land in which shard is a function of the run index — a batch
/// at any `--jobs` writes byte-identical shards.
struct Shards {
    directory: PathBuf,
    header: serde_json::Value,
    runs_per_shard: usize,
    /// The shard file stem, and the manifest's word for a line.
    prefix: &'static str,
    unit: &'static str,
    open: Option<std::io::BufWriter<std::fs::File>>,
    runs_in_shard: usize,
    lines_in_shard: usize,
    /// The manifest entries of the shards closed so far.
    closed: Vec<serde_json::Value>,
    runs: usize,
    total: usize,
}

impl Shards {
    fn open(
        directory: &Path,
        header: serde_json::Value,
        runs_per_shard: usize,
        prefix: &'static str,
        unit: &'static str,
    ) -> std::io::Result<Self> {
        std::fs::create_dir_all(directory)?;
        Ok(Self {
            directory: directory.to_path_buf(),
            header,
            runs_per_shard: runs_per_shard.max(1),
            prefix,
            unit,
            open: None,
            runs_in_shard: 0,
            lines_in_shard: 0,
            closed: Vec::new(),
            runs: 0,
            total: 0,
        })
    }

    /// The index the next run written will carry.
    const fn next_run(&self) -> usize {
        self.runs
    }

    /// One run's lines. A run that recorded nothing still counts against
    /// the shard's run budget, so the boundaries stay where the run index
    /// puts them.
    fn write_run<T: serde::Serialize>(&mut self, lines: &[T]) -> std::io::Result<()> {
        if self.runs_in_shard >= self.runs_per_shard {
            self.close_shard()?;
        }
        if self.open.is_none() {
            let path = self.directory.join(self.shard_name(self.closed.len()));
            let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
            writeln!(writer, "{}", self.header)?;
            self.open = Some(writer);
        }
        let writer = self.open.as_mut().expect("the shard is open");
        self.lines_in_shard += lines.len();
        self.total += lines.len();
        for line in lines {
            writeln!(writer, "{}", serde_json::to_string(line)?)?;
        }
        self.runs_in_shard += 1;
        self.runs += 1;
        Ok(())
    }

    /// Closes the last shard and writes the manifest: the shard list with a
    /// count each, which is the set's integrity check.
    fn finish(mut self) -> std::io::Result<usize> {
        self.close_shard()?;
        let mut manifest = self.header.clone();
        let object = manifest.as_object_mut().expect("the header is an object");
        object.insert("runs".into(), self.runs.into());
        object.insert(self.unit.into(), self.total.into());
        object.insert("shards".into(), serde_json::Value::Array(self.closed));
        std::fs::write(
            self.directory.join("manifest.json"),
            format!("{manifest}\n"),
        )?;
        Ok(self.total)
    }

    fn close_shard(&mut self) -> std::io::Result<()> {
        let Some(mut writer) = self.open.take() else {
            return Ok(());
        };
        writer.flush()?;
        self.closed.push(serde_json::json!({
            "file": self.shard_name(self.closed.len()),
            self.unit: self.lines_in_shard,
            "runs": self.runs_in_shard,
        }));
        self.runs_in_shard = 0;
        self.lines_in_shard = 0;
        Ok(())
    }

    fn shard_name(&self, index: usize) -> String {
        format!("{}-{index:05}.jsonl", self.prefix)
    }
}

/// A batch's training samples, streamed to a directory of shards: each
/// run's decisions encoded under this build's policy encoding as they
/// arrive. See `Shards` for why streaming, and [`Decision`] for the other
/// way to keep them.
pub struct SampleSink {
    shards: Shards,
    encoder: PolicyEncoder,
}

impl SampleSink {
    /// Opens `directory` for in-combat samples, creating it if it is not
    /// there. `characters` is who the decisions will come from — see
    /// `stamp_characters`.
    pub fn create(
        directory: &Path,
        encoder: &PolicyEncoder,
        runs_per_shard: usize,
        characters: &[ModelId],
    ) -> std::io::Result<Self> {
        Self::open(
            directory,
            encoder,
            runs_per_shard,
            SCOPE_COMBAT,
            crate::objective::VALUE_SEMANTICS,
            None,
            None,
            characters,
        )
    }

    /// The same, for macro samples — the out-of-combat decisions a teacher
    /// answered. A separate directory by construction: the run net and the
    /// combat net are different nets over different horizons, and their `z`
    /// are not the same quantity — which is exactly what the header's
    /// `value_semantics` says. `source` names which teacher answered
    /// ([`SOURCE_HEURISTIC`] or [`SOURCE_SEARCH`]).
    pub fn create_macro(
        directory: &Path,
        encoder: &PolicyEncoder,
        runs_per_shard: usize,
        source: &str,
        characters: &[ModelId],
    ) -> std::io::Result<Self> {
        Self::open(
            directory,
            encoder,
            runs_per_shard,
            SCOPE_MACRO,
            crate::objective::ACT_VALUE_SEMANTICS,
            Some(source),
            None,
            characters,
        )
    }

    /// The sink a run-level PPO rollout writes its trajectories through:
    /// macro-scoped like [`SampleSink::create_macro`], and marked
    /// [`TRAINING_MODE_PPO`] because what it holds is an on-policy trajectory
    /// rather than a teacher's distribution.
    ///
    /// Three header keys make a PPO set what it is —
    /// `scope: macro`, `value_semantics: run-return-v1`, and
    /// `training_mode: ppo` — and the trainer refuses a set missing any of
    /// them. That is the same refusal discipline the loaders keep: a set is
    /// never reinterpreted into a loss it was not recorded for, because an
    /// expert-iteration set fed to the surrogate would form ratios against
    /// log-probabilities nothing ever sampled, and a PPO set fed to
    /// cross-entropy would train toward one-hot actions that were only ever
    /// draws.
    ///
    /// No `source`: a PPO line's teacher is the checkpoint that acted, which
    /// is neither of the two teachers `source` names, and naming it would
    /// invite a reader to pool sets that must not be pooled.
    pub fn create_ppo(
        directory: &Path,
        encoder: &PolicyEncoder,
        runs_per_shard: usize,
        characters: &[ModelId],
    ) -> std::io::Result<Self> {
        Self::create_ppo_priced(
            directory,
            encoder,
            runs_per_shard,
            crate::objective::RUN_VALUE_SEMANTICS,
            characters,
        )
    }

    /// [`SampleSink::create_ppo`] under a named reward: `value_semantics` is
    /// what [`crate::reward::value_semantics`] returns for the weights the
    /// batch was paid at, and is what the trainer's checkpoint will carry.
    pub fn create_ppo_priced(
        directory: &Path,
        encoder: &PolicyEncoder,
        runs_per_shard: usize,
        value_semantics: &str,
        characters: &[ModelId],
    ) -> std::io::Result<Self> {
        Self::open(
            directory,
            encoder,
            runs_per_shard,
            SCOPE_MACRO,
            value_semantics,
            None,
            Some(TRAINING_MODE_PPO),
            characters,
        )
    }

    /// The sink that encodes a recorded set: the set's own scope, value
    /// semantics, source, characters and shard size, so what it writes is
    /// what the batch that recorded the set would have written under this
    /// encoding.
    pub fn create_for(
        directory: &Path,
        encoder: &PolicyEncoder,
        decisions: &DecisionSet,
    ) -> std::io::Result<Self> {
        Self::open(
            directory,
            encoder,
            decisions.runs_per_shard(),
            decisions.scope(),
            decisions.value_semantics(),
            decisions.source(),
            decisions.training_mode(),
            &decisions.characters(),
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per header field the caller settles"
    )]
    fn open(
        directory: &Path,
        encoder: &PolicyEncoder,
        runs_per_shard: usize,
        scope: &str,
        value_semantics: &str,
        source: Option<&str>,
        training_mode: Option<&str>,
        characters: &[ModelId],
    ) -> std::io::Result<Self> {
        Ok(Self {
            shards: Shards::open(
                directory,
                header(
                    encoder,
                    scope,
                    value_semantics,
                    source,
                    training_mode,
                    characters,
                ),
                runs_per_shard,
                "samples",
                "samples",
            )?,
            encoder: encoder.clone(),
        })
    }

    /// One run's decisions, stamped with the run index they arrived under —
    /// write order is run order by the batch's contract, so the sink's own
    /// count is the index — and encoded.
    pub fn write_run(&mut self, decisions: &mut [Decision]) -> std::io::Result<()> {
        let run = self.shards.next_run();
        let samples: Vec<TrainingSample> = decisions
            .iter_mut()
            .map(|decision| {
                decision.run = run;
                decision.encode(&self.encoder)
            })
            .collect();
        self.shards.write_run(&samples)
    }

    /// Closes the last shard and writes the manifest; the count is what was
    /// written.
    pub fn finish(self) -> std::io::Result<usize> {
        self.shards.finish()
    }
}

/// A batch's decisions kept as recorded, streamed to a directory of shards
/// beside a manifest — the same sharding as [`SampleSink`], the same run
/// stamps, no encoding. `alphaspire encode` turns the directory into the
/// samples any build's [`SampleSink`] would have written from the same
/// search.
pub struct DecisionSink {
    shards: Shards,
}

impl DecisionSink {
    /// Opens `directory` for in-combat decisions.
    pub fn create(
        directory: &Path,
        runs_per_shard: usize,
        characters: &[ModelId],
    ) -> std::io::Result<Self> {
        Self::open(
            directory,
            runs_per_shard,
            SCOPE_COMBAT,
            crate::objective::VALUE_SEMANTICS,
            None,
            None,
            characters,
        )
    }

    /// The same for macro decisions, under the same separation
    /// [`SampleSink::create_macro`] keeps.
    pub fn create_macro(
        directory: &Path,
        runs_per_shard: usize,
        source: &str,
        characters: &[ModelId],
    ) -> std::io::Result<Self> {
        Self::open(
            directory,
            runs_per_shard,
            SCOPE_MACRO,
            crate::objective::ACT_VALUE_SEMANTICS,
            Some(source),
            None,
            characters,
        )
    }

    /// The same for a PPO rollout's trajectories, under the header
    /// [`SampleSink::create_ppo`] describes. `alphaspire encode` carries all
    /// three keys from here into the encoded set, so a rollout may record
    /// once and encode under whatever encoding the learner is on.
    pub fn create_ppo(
        directory: &Path,
        runs_per_shard: usize,
        characters: &[ModelId],
    ) -> std::io::Result<Self> {
        Self::create_ppo_priced(
            directory,
            runs_per_shard,
            crate::objective::RUN_VALUE_SEMANTICS,
            characters,
        )
    }

    /// [`DecisionSink::create_ppo`] under a named reward, as
    /// [`SampleSink::create_ppo_priced`].
    pub fn create_ppo_priced(
        directory: &Path,
        runs_per_shard: usize,
        value_semantics: &str,
        characters: &[ModelId],
    ) -> std::io::Result<Self> {
        Self::open(
            directory,
            runs_per_shard,
            SCOPE_MACRO,
            value_semantics,
            None,
            Some(TRAINING_MODE_PPO),
            characters,
        )
    }

    fn open(
        directory: &Path,
        runs_per_shard: usize,
        scope: &str,
        value_semantics: &str,
        source: Option<&str>,
        training_mode: Option<&str>,
        characters: &[ModelId],
    ) -> std::io::Result<Self> {
        let runs_per_shard = runs_per_shard.max(1);
        Ok(Self {
            shards: Shards::open(
                directory,
                decision_header(
                    scope,
                    value_semantics,
                    source,
                    training_mode,
                    runs_per_shard,
                    characters,
                ),
                runs_per_shard,
                "decisions",
                "decisions",
            )?,
        })
    }

    /// One run's decisions, stamped with their run index.
    pub fn write_run(&mut self, decisions: &mut [Decision]) -> std::io::Result<()> {
        let run = self.shards.next_run();
        for decision in decisions.iter_mut() {
            decision.run = run;
        }
        self.shards.write_run(decisions)
    }

    /// Closes the last shard and writes the manifest; the count is what was
    /// written.
    pub fn finish(self) -> std::io::Result<usize> {
        self.shards.finish()
    }
}

/// A directory a [`DecisionSink`] wrote, opened for encoding.
///
/// The manifest's pins — format, observation version, belief model, game
/// build — must all be this build's. A set whose decisions this build would
/// read differently is refused, never reinterpreted: an observation from
/// another version is not this encoder's input, whatever it parses as.
pub struct DecisionSet {
    directory: PathBuf,
    manifest: serde_json::Value,
}

impl DecisionSet {
    pub fn open(directory: &Path) -> std::io::Result<Self> {
        let manifest_path = directory.join("manifest.json");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
        let pins = serde_json::json!({
            "format": DECISION_FORMAT,
            "observation_version": sts2_engine::AGENT_OBSERVATION_VERSION,
            "belief_model_version": sts2_engine::BELIEF_MODEL_VERSION,
            "compatibility_id": sts2_core::PINNED_COMPATIBILITY_ID,
        });
        for (field, want) in pins.as_object().expect("the pins are an object") {
            let found = manifest.get(field).unwrap_or(&serde_json::Value::Null);
            if found != want {
                return Err(std::io::Error::other(format!(
                    "{}: {field} is {found}, this build wants {want}",
                    manifest_path.display()
                )));
            }
        }
        for field in [
            "scope",
            "value_semantics",
            "runs_per_shard",
            "runs",
            "shards",
        ] {
            if manifest.get(field).is_none() {
                return Err(std::io::Error::other(format!(
                    "{}: no {field}",
                    manifest_path.display()
                )));
            }
        }
        Ok(Self {
            directory: directory.to_path_buf(),
            manifest,
        })
    }

    #[must_use]
    pub fn scope(&self) -> &str {
        self.manifest["scope"].as_str().unwrap_or(SCOPE_COMBAT)
    }

    #[must_use]
    pub fn value_semantics(&self) -> &str {
        self.manifest["value_semantics"]
            .as_str()
            .unwrap_or_default()
    }

    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.manifest
            .get("source")
            .and_then(serde_json::Value::as_str)
    }

    /// Which learner the set was recorded for, where it was recorded for a
    /// particular one — [`TRAINING_MODE_PPO`] for a rollout's trajectories,
    /// nothing for an expert-iteration set. Carried into the encoded set so
    /// that encoding a recorded set never launders a trajectory into
    /// something the PPO loss would refuse.
    #[must_use]
    pub fn training_mode(&self) -> Option<&str> {
        self.manifest
            .get("training_mode")
            .and_then(serde_json::Value::as_str)
    }

    /// Which characters the decisions came from, empty where the set does
    /// not say — a set recorded before the header carried them. An id the
    /// header holds that this build cannot read is dropped rather than
    /// refused: the field describes the set, it does not gate it.
    #[must_use]
    pub fn characters(&self) -> Vec<ModelId> {
        self.manifest
            .get("characters")
            .and_then(serde_json::Value::as_array)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| id.as_str()?.parse().ok())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[must_use]
    pub fn runs_per_shard(&self) -> usize {
        usize::try_from(self.manifest["runs_per_shard"].as_u64().unwrap_or(1)).unwrap_or(1)
    }

    /// How many runs the batch wrote, recorded or not.
    #[must_use]
    pub fn runs(&self) -> usize {
        usize::try_from(self.manifest["runs"].as_u64().unwrap_or(0)).unwrap_or(0)
    }

    /// How many decisions the set holds.
    #[must_use]
    pub fn decisions(&self) -> usize {
        usize::try_from(self.manifest["decisions"].as_u64().unwrap_or(0)).unwrap_or(0)
    }

    /// Hands `visit` every run of the batch in run order — an empty
    /// vector for a run that recorded nothing, so a sink fed from here
    /// puts its shard boundaries exactly where the batch did.
    pub fn for_each_run(
        &self,
        mut visit: impl FnMut(Vec<Decision>) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        let shards = self.manifest["shards"]
            .as_array()
            .ok_or_else(|| std::io::Error::other("the manifest lists no shards"))?;
        // The run `pending` belongs to; lines arrive in run order, so a
        // stamp past it closes the run and opens (empty) runs up to its own.
        let mut next_run = 0;
        let mut pending: Vec<Decision> = Vec::new();
        for shard in shards {
            let file = shard["file"]
                .as_str()
                .ok_or_else(|| std::io::Error::other("a shard entry names no file"))?;
            let text = std::fs::read_to_string(self.directory.join(file))?;
            for line in text.lines().skip(1) {
                let decision: Decision = serde_json::from_str(line)?;
                if decision.run < next_run {
                    return Err(std::io::Error::other(format!(
                        "{file}: run {} after run {next_run}: not a sink's run order",
                        decision.run
                    )));
                }
                while decision.run > next_run {
                    visit(std::mem::take(&mut pending))?;
                    next_run += 1;
                }
                pending.push(decision);
            }
        }
        let runs = self.runs().max(next_run + usize::from(!pending.is_empty()));
        while next_run < runs {
            visit(std::mem::take(&mut pending))?;
            next_run += 1;
        }
        Ok(())
    }
}
