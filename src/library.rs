//! The fight library: fight entries banked during self-play, and training
//! data generated from the bank at an explicit mix.
//!
//! Bosses are met once per act and early-generation runs die in act one, so
//! the natural self-play distribution starves exactly the highest-stakes
//! fights. The library attacks the imbalance directly: every fight a run
//! enters is banked at its entry decision as an erased [`BeliefState`]
//! snapshot, and generation replays banked entries as standalone
//! belief-search fights at whatever mix the data plan asks for — including
//! `all` lanes that replay every banked entry of the scarcest classes.
//!
//! An entry says its fight two ways, and a harvest chooses which to write.
//! The *setup* — the engine's [`CombatSetup`]: who walks in, carrying what,
//! through which door — is a few hundred bytes any producer can write, and
//! the engine stands it up through its own combat start on the run's seed:
//! one loadout, many hands, and a bank that outlives every internal the
//! engine cares to change. The *state* — the erased [`BeliefState`] snapshot
//! of the very fight the run stood at, its canonical world projected
//! portable — is tens of kilobytes and the one thing a setup cannot be:
//! exact, hand and enemy rolls and stream positions included. A training
//! bank carries setups alone; a holdout bank for paired evaluation carries
//! both, so two nets meet identical worlds. A consumer says which it needs,
//! and is refused, entry named, where it asks for a state that is not there.
//!
//! Fair by construction either way: a setup carries nothing a player could
//! not see, and a state is the canonical world of an erasure, so re-erasing
//! it on load recovers the very belief that was banked. Where a loadout
//! came from is the entry's origin — harvested from a run's own reached
//! state, from a forced-win walk, from a replayed human run, or folded from
//! a human record and never reached at all — and the manifest counts each.
//!
//! The format follows the sample shards: a directory of JSONL shards
//! beside a `manifest.json`, every shard opening with the full version
//! header. A library written under another belief model or another game
//! build is refused, never reinterpreted.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sts2_core::{ModelId, UnlockPresetManifest};
use sts2_engine::{
    BeliefState, CombatSetup, ContentRegistry, MapPointType, PortableState, ProjectionScope,
    RunResult, Simulator,
};
use sts2_rng::{MegaRandom, splitmix64};

use crate::env::fight_over;
use crate::objective::{CombatStrength, Objective};
use crate::policy::RolloutPolicy;
use crate::selfplay::{PlayError, indexed, panic_message, run_seeds};

/// The library file format: what an entry line holds and how the manifest
/// counts it. Format 2 added the entry's origin, the act-three class, and
/// the split between the header fields a load depends on and the ones that
/// only describe. Format 3 says the fight as a [`CombatSetup`] and makes
/// the exact state optional beside it, counted by the manifest. A format-2
/// entry is a format-3 entry carrying only a state, so a format-2 bank
/// opens and loads as one whose every entry is state-only; [`convert`]
/// rewrites it as setups where that is wanted.
pub const LIBRARY_FORMAT: u32 = 3;

/// The format a bank written before the setup existed carries: every entry
/// an exact state and nothing else, and no count of either on its manifest.
const PREVIOUS_FORMAT: u32 = 2;

/// The header fields a load actually depends on. The rest of the header —
/// the observation version above all — describes the build that wrote the
/// bank without gating it: an entry stores no observation, so a bank
/// outlives an observation bump the way a checkpoint cannot.
const ENFORCED: [&str; 3] = ["format", "belief_model_version", "compatibility_id"];

/// A library that could not be written, opened, or replayed, and why.
#[derive(Debug)]
pub struct LibraryError(pub String);

impl std::fmt::Display for LibraryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl std::error::Error for LibraryError {}

/// The room tier a fight was met in. Everything that is not an elite or
/// boss room — monster rooms, unknowns that turned into one, the fight an
/// event pushed — is a hallway fight.
#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Hallway,
    Elite,
    Boss,
}

impl Tier {
    /// Every tier, in the canonical order counts are kept in.
    pub const ALL: [Self; 3] = [Self::Hallway, Self::Elite, Self::Boss];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hallway => "hallway",
            Self::Elite => "elite",
            Self::Boss => "boss",
        }
    }

    /// Which tier the room a fight stands in belongs to.
    #[must_use]
    pub fn of(room: Option<MapPointType>) -> Self {
        match room {
            Some(MapPointType::Boss) => Self::Boss,
            Some(MapPointType::Elite) => Self::Elite,
            _ => Self::Hallway,
        }
    }

    /// A tier's slot in the class counts: the first three classes are the
    /// tiers, in the same order.
    const fn index(self) -> usize {
        match self {
            Self::Hallway => 0,
            Self::Elite => 1,
            Self::Boss => 2,
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.as_str())
    }
}

/// A class of banked entries the mix can name: the three room tiers, and
/// the act overlays for the fights deep runs are scarcest in. The tiers
/// partition the bank; `act2` and `act3` overlap them (and each other), so
/// an act-three boss can be drawn by three lanes — the overlap replays the
/// scarcest entries more, which is the mix's whole purpose.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Class {
    Hallway,
    Elite,
    Boss,
    /// Entries whose fight was entered in the second act or beyond (the
    /// engine counts acts from zero).
    ActTwo,
    /// Entries whose fight was entered in the third act or beyond.
    ActThree,
}

impl Class {
    /// Every class, in the canonical order quotas and schedules walk.
    pub const ALL: [Self; 5] = [
        Self::Hallway,
        Self::Elite,
        Self::Boss,
        Self::ActTwo,
        Self::ActThree,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hallway => "hallway",
            Self::Elite => "elite",
            Self::Boss => "boss",
            Self::ActTwo => "act2",
            Self::ActThree => "act3",
        }
    }

    /// This class's slot in every `[usize; 5]` the library counts with.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Hallway => 0,
            Self::Elite => 1,
            Self::Boss => 2,
            Self::ActTwo => 3,
            Self::ActThree => 4,
        }
    }

    /// Whether a banked entry belongs to this class. An entry's within-class
    /// index — how a plan names it — is the count of earlier bank entries
    /// this answers yes for.
    #[must_use]
    pub fn holds(self, meta: &FightMeta) -> bool {
        match self {
            Self::Hallway => meta.tier == Tier::Hallway,
            Self::Elite => meta.tier == Tier::Elite,
            Self::Boss => meta.tier == Tier::Boss,
            Self::ActTwo => meta.act >= 1,
            Self::ActThree => meta.act >= 2,
        }
    }
}

impl std::fmt::Display for Class {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.as_str())
    }
}

impl std::str::FromStr for Class {
    type Err = LibraryError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|class| class.as_str() == name)
            .ok_or_else(|| {
                LibraryError(format!(
                    "{name:?} is not a class: hallway, elite, boss, act2, act3"
                ))
            })
    }
}

/// How a banked entry's fight was reached: by a run playing its combats,
/// or by a forced-win walk that skipped them. A forced-win entry is a real
/// engine-initialized fight with a loadout grown through real reward
/// screens, but the walk that reached it paid no combat cost beyond the
/// injected one — so the origin rides every entry, the manifest counts it,
/// and a judging instrument can refuse a bank that carries any.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FightOrigin {
    #[default]
    Natural,
    ForcedWin,
    /// Reached by replaying a human's recorded run to this fight's entry —
    /// an exact state a person stood at, with their deck behind it.
    HumanRun,
    /// Folded from a human's run record and never reached at all: the
    /// loadout the record says the person carried onto the floor, said as
    /// a setup. Exactly the deck the human had, with no walk that had to
    /// agree with the record to get there — and no exact state, ever.
    HumanRecord,
}

/// How many origins there are: the width of every count the manifest keeps
/// per origin.
pub const ORIGINS: usize = 4;

impl FightOrigin {
    /// Every origin, in the canonical order the manifest counts them.
    pub const ALL: [Self; ORIGINS] = [
        Self::Natural,
        Self::ForcedWin,
        Self::HumanRun,
        Self::HumanRecord,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::ForcedWin => "forced_win",
            Self::HumanRun => "human_run",
            Self::HumanRecord => "human_record",
        }
    }

    /// This origin's slot in the `[usize; ORIGINS]` the manifest counts
    /// with.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Natural => 0,
            Self::ForcedWin => 1,
            Self::HumanRun => 2,
            Self::HumanRecord => 3,
        }
    }
}

/// What a harvest writes on each entry: the setup alone, the exact state
/// alone, or both. A training bank wants the setup — a few hundred bytes
/// the generation reads as one loadout under many hands — and never the
/// state it would not read; a holdout bank for paired evaluation wants both.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HarvestAs {
    #[default]
    Setup,
    State,
    Both,
}

impl HarvestAs {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::State => "state",
            Self::Both => "both",
        }
    }

    const fn writes_setup(self) -> bool {
        matches!(self, Self::Setup | Self::Both)
    }

    const fn writes_state(self) -> bool {
        matches!(self, Self::State | Self::Both)
    }
}

impl std::str::FromStr for HarvestAs {
    type Err = LibraryError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "setup" => Ok(Self::Setup),
            "state" => Ok(Self::State),
            "both" => Ok(Self::Both),
            other => Err(LibraryError(format!(
                "a harvest writes setup, state or both, not {other}"
            ))),
        }
    }
}

/// Which half of an entry a load stands its fights up from.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Source {
    /// The setup, stood up through the engine's combat start on the run
    /// seed the entry names — one loadout, many hands. An entry that
    /// carries only a state is stood up and projected to reach its setup.
    #[default]
    Setup,
    /// The exact state, re-erased into the very belief that was banked. An
    /// entry without one is refused, named.
    State,
}

impl Source {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::State => "state",
        }
    }
}

impl std::str::FromStr for Source {
    type Err = LibraryError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "setup" => Ok(Self::Setup),
            "state" => Ok(Self::State),
            other => Err(LibraryError(format!(
                "a load stands fights up from the setup or the state, not {other}"
            ))),
        }
    }
}

/// Where a banked entry came from: enough to stratify selection, read the
/// bank at a glance, and name the run that reached it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FightMeta {
    pub encounter: ModelId,
    pub tier: Tier,
    pub act: usize,
    pub floor: u32,
    pub character: ModelId,
    pub ascension: u8,
    /// The seed of the run this fight was reached in, with the run's index
    /// in its harvesting batch and the fight's index within the run — the
    /// same (run, fight) pair format 3 samples carry, so a banked entry and
    /// the samples its fight taught join exactly.
    pub seed: String,
    pub run: usize,
    pub fight: usize,
    /// The player's hit points at the entry decision.
    pub entry_hp: i32,
    /// How many cards the deck carried in.
    pub deck: usize,
    /// How the fight was reached. Absent on an entry written before the
    /// field existed, which reads as naturally.
    #[serde(default)]
    pub origin: FightOrigin,
}

/// One banked fight entry: the metadata, and the fight said one or both
/// ways — the setup, and the erased snapshot dressed as its canonical world
/// so it serializes through the simulator's portable state and re-erases
/// on load. At least one is always there; which ones is the harvest's
/// choice, and the manifest counts each.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FightEntry {
    pub meta: FightMeta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup: Option<CombatSetup>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<PortableState>,
}

impl FightEntry {
    /// The setup this entry says its fight as, projecting the exact state
    /// where the harvest kept only that. `registry` is the content the
    /// entry's character sees at its ascension.
    fn setup(&self, registry: &Arc<ContentRegistry>) -> Result<CombatSetup, LibraryError> {
        if let Some(setup) = &self.setup {
            return Ok(setup.clone());
        }
        let state = self
            .state
            .clone()
            .ok_or_else(|| LibraryError("an entry says its fight at least one way".into()))?;
        let world = Simulator::from_portable_state(state, Arc::clone(registry))
            .map_err(|error| LibraryError(format!("{error:?}")))?;
        world
            .combat_setup()
            .ok_or_else(|| LibraryError("a banked state stands inside a fight".into()))
    }
}

/// Banks the fight a simulator stands at the entry of, said as `keep` asks:
/// the setup the run projects, the state erased to what a player can see,
/// or both, beside where it was met. Harvested from real reached states
/// only — this is the one constructor, and it takes a live simulator, so a
/// loadout that never occurred enters the bank only through a fold that
/// says so by its origin.
pub fn harvest(simulator: &Simulator, keep: HarvestAs) -> Result<FightEntry, LibraryError> {
    let state = simulator.state();
    let combat = state
        .combat
        .as_ref()
        .ok_or_else(|| LibraryError("only a fight is banked".into()))?;
    let run = state
        .run
        .as_ref()
        .ok_or_else(|| LibraryError("only a fight inside a run is banked".into()))?;
    let character = run
        .config
        .players
        .iter()
        .find(|player| player.slot == state.run_player.slot)
        .map(|player| player.character.clone())
        .ok_or_else(|| LibraryError("the run names no player for its slot".into()))?;
    // In a fight the player's hit points live on their creature.
    let entry_hp = combat
        .creatures
        .iter()
        .find(|creature| creature.id == combat.player.creature_id)
        .map_or(state.run_player.current_hp, |creature| creature.current_hp);
    let meta = FightMeta {
        encounter: combat.encounter_model.clone(),
        tier: Tier::of(run.standing_room_type()),
        act: run.current_act,
        floor: run.floor,
        character,
        ascension: run.config.ascension,
        seed: run.config.seed.clone(),
        // Both stamped by the sink, which knows the batch's run order and
        // receives a run's fights in the order they were entered.
        run: 0,
        fight: 0,
        entry_hp,
        deck: state.run_player.deck.len(),
        // The harvest reads a reached state and cannot know how the walk
        // reached it; a forced-win driver overrides this after harvesting.
        origin: FightOrigin::Natural,
    };
    let setup = keep
        .writes_setup()
        .then(|| {
            simulator
                .combat_setup()
                .ok_or_else(|| LibraryError("only a live fight is banked".into()))
        })
        .transpose()?;
    let state = keep
        .writes_state()
        .then(|| {
            let belief = BeliefState::from_simulator(simulator)
                .map_err(|error| LibraryError(format!("{error:?}")))?;
            let world = belief
                .sample(0, 0)
                .map_err(|error| LibraryError(format!("{error:?}")))?;
            world
                .project(ProjectionScope::CombatCore)
                .map_err(|error| LibraryError(format!("{error:?}")))
        })
        .transpose()?;
    Ok(FightEntry { meta, setup, state })
}

/// Every version the banked entries depend on. Each shard opens with this
/// line, so one shard alone is loadable and self-validating.
fn header() -> serde_json::Value {
    serde_json::json!({
        "format": LIBRARY_FORMAT,
        "belief_model_version": sts2_engine::BELIEF_MODEL_VERSION,
        "observation_version": sts2_engine::AGENT_OBSERVATION_VERSION,
        "compatibility_id": sts2_core::PINNED_COMPATIBILITY_ID,
    })
}

/// A batch's harvested fights, streamed to a directory of shards: the same
/// shape as [`crate::training::SampleSink`], for the same reasons. Entries
/// reach disk as their runs finish, shard boundaries are run boundaries,
/// and runs arrive in run order — so a batch at any `--jobs` writes
/// byte-identical shards, and the sink's own run count is each entry's run
/// index.
pub struct LibrarySink {
    directory: PathBuf,
    header: serde_json::Value,
    runs_per_shard: usize,
    open: Option<std::io::BufWriter<std::fs::File>>,
    runs_in_shard: usize,
    entries_in_shard: usize,
    shards: Vec<serde_json::Value>,
    runs: usize,
    classes: [usize; 5],
    origins: [usize; ORIGINS],
    /// How many entries carry a setup, and how many an exact state. A
    /// consumer that needs the state reads the count before it asks.
    with_setup: usize,
    with_state: usize,
    /// Who the banked entries were reached by. The bank may honestly hold
    /// several characters, so the manifest describes the composition rather
    /// than pinning one. Written even when it is empty, unlike a sample
    /// header's stamp: a sink that banked the fights knows who reached
    /// them, so an empty list means an empty bank and never "did not
    /// record".
    characters: BTreeSet<ModelId>,
    /// Extra manifest fields the driver wants remembered — a forced-win
    /// batch records the config that shaped its walks here, so a bank
    /// names the distribution it was harvested under.
    annotations: serde_json::Map<String, serde_json::Value>,
}

impl LibrarySink {
    /// Opens `directory`, creating it if it is not there.
    pub fn create(directory: &Path, runs_per_shard: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(directory)?;
        Ok(Self {
            directory: directory.to_path_buf(),
            header: header(),
            runs_per_shard: runs_per_shard.max(1),
            open: None,
            runs_in_shard: 0,
            entries_in_shard: 0,
            shards: Vec::new(),
            runs: 0,
            classes: [0; 5],
            origins: [0; ORIGINS],
            with_setup: 0,
            with_state: 0,
            characters: BTreeSet::new(),
            annotations: serde_json::Map::new(),
        })
    }

    /// Remembers an extra manifest field, written beside the counts at
    /// finish. A key the header already owns is refused by the write there
    /// being last; annotate before finishing and pick keys of your own.
    pub fn annotate(&mut self, key: &str, value: serde_json::Value) {
        self.annotations.insert(key.to_owned(), value);
    }

    /// One run's fights, in the order the run entered them, stamped with
    /// the (run, fight) indices they arrived under. A run that entered no
    /// fight still counts against the shard's run budget, so the boundaries
    /// stay where the run index puts them.
    pub fn write_run(&mut self, fights: &mut [FightEntry]) -> std::io::Result<()> {
        if self.runs_in_shard >= self.runs_per_shard {
            self.close_shard()?;
        }
        if self.open.is_none() {
            let path = self.directory.join(Self::shard_name(self.shards.len()));
            let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
            writeln!(writer, "{}", self.header)?;
            self.open = Some(writer);
        }
        let writer = self.open.as_mut().expect("the shard is open");
        self.entries_in_shard += fights.len();
        for (index, fight) in fights.iter_mut().enumerate() {
            fight.meta.run = self.runs;
            fight.meta.fight = index;
            for class in Class::ALL {
                if class.holds(&fight.meta) {
                    self.classes[class.index()] += 1;
                }
            }
            self.origins[fight.meta.origin.index()] += 1;
            if fight.setup.is_none() && fight.state.is_none() {
                return Err(std::io::Error::other(
                    "an entry says its fight at least one way",
                ));
            }
            self.with_setup += usize::from(fight.setup.is_some());
            self.with_state += usize::from(fight.state.is_some());
            self.characters.insert(fight.meta.character.clone());
            writeln!(writer, "{}", serde_json::to_string(fight)?)?;
        }
        self.runs_in_shard += 1;
        self.runs += 1;
        Ok(())
    }

    /// Closes the last shard and writes the manifest, answering with the
    /// per-class entry counts — the numbers a harvest summary line wants.
    pub fn finish(mut self) -> std::io::Result<[usize; 5]> {
        self.close_shard()?;
        let mut manifest = self.header.clone();
        let object = manifest.as_object_mut().expect("the header is an object");
        for (key, value) in self.annotations {
            object.insert(key, value);
        }
        object.insert("runs".into(), self.runs.into());
        object.insert(
            "entries".into(),
            Tier::ALL
                .iter()
                .map(|tier| self.classes[tier.index()])
                .sum::<usize>()
                .into(),
        );
        object.insert(
            "classes".into(),
            serde_json::Value::Object(
                Class::ALL
                    .into_iter()
                    .map(|class| {
                        (
                            class.as_str().to_owned(),
                            serde_json::Value::from(self.classes[class.index()]),
                        )
                    })
                    .collect(),
            ),
        );
        object.insert(
            "origins".into(),
            serde_json::Value::Object(
                FightOrigin::ALL
                    .into_iter()
                    .map(|origin| {
                        (
                            origin.as_str().to_owned(),
                            serde_json::Value::from(self.origins[origin.index()]),
                        )
                    })
                    .collect(),
            ),
        );
        object.insert("with_setup".into(), self.with_setup.into());
        object.insert("with_state".into(), self.with_state.into());
        object.insert(
            "characters".into(),
            self.characters
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<String>>()
                .into(),
        );
        object.insert("shards".into(), serde_json::Value::Array(self.shards));
        std::fs::write(
            self.directory.join("manifest.json"),
            format!("{manifest}\n"),
        )?;
        Ok(self.classes)
    }

    fn close_shard(&mut self) -> std::io::Result<()> {
        let Some(mut writer) = self.open.take() else {
            return Ok(());
        };
        writer.flush()?;
        self.shards.push(serde_json::json!({
            "file": Self::shard_name(self.shards.len()),
            "entries": self.entries_in_shard,
            "runs": self.runs_in_shard,
        }));
        self.runs_in_shard = 0;
        self.entries_in_shard = 0;
        Ok(())
    }

    fn shard_name(index: usize) -> String {
        format!("entries-{index:05}.jsonl")
    }
}

/// An opened library: the manifest verified against this build, the entries
/// still on disk until a plan names the ones generation needs.
pub struct FightLibrary {
    directory: PathBuf,
    header: serde_json::Value,
    shards: Vec<String>,
    classes: [usize; 5],
    origins: Option<[usize; ORIGINS]>,
    characters: Option<Vec<ModelId>>,
    with_setup: usize,
    with_state: usize,
}

impl FightLibrary {
    /// Opens a library directory and verifies its manifest: the format, the
    /// belief model, and the game build must all match this one. A bank
    /// whose erasure or content this build cannot honor is refused, never
    /// reinterpreted. The observation version is on the manifest but not on
    /// the gate: an entry stores no observation, so a bank written under an
    /// earlier observation shape still loads.
    pub fn open(directory: &Path) -> Result<Self, LibraryError> {
        let manifest_path = directory.join("manifest.json");
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&manifest_path)
                .map_err(|error| LibraryError(format!("{}: {error}", manifest_path.display())))?,
        )
        .map_err(|error| LibraryError(format!("{}: {error}", manifest_path.display())))?;
        let mut expected = header();
        // The previous format's entries are this format's state-only
        // entries, so the bank is read as it is; its shards carry its own
        // format number, and that is what they are checked against.
        let previous = manifest.get("format").and_then(serde_json::Value::as_u64)
            == Some(u64::from(PREVIOUS_FORMAT));
        if previous {
            expected["format"] = serde_json::json!(PREVIOUS_FORMAT);
        }
        for field in ENFORCED {
            let want = &expected[field];
            let found = manifest.get(field).unwrap_or(&serde_json::Value::Null);
            if found != want {
                return Err(LibraryError(format!(
                    "library {field} is {found}, this build wants {want}"
                )));
            }
        }
        let count = |key: &str| -> Option<usize> {
            manifest
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|count| usize::try_from(count).expect("entry counts fit usize"))
        };
        // A manifest written before the counts existed says, by its format,
        // that every entry is a state and none a setup.
        let entries = count("entries").unwrap_or(0);
        let with_setup = count("with_setup").unwrap_or(if previous { 0 } else { entries });
        let with_state = count("with_state").unwrap_or(if previous { entries } else { 0 });
        let mut classes = [0; 5];
        for class in Class::ALL {
            classes[class.index()] = usize::try_from(
                manifest["classes"][class.as_str()]
                    .as_u64()
                    .ok_or_else(|| {
                        LibraryError(format!("the manifest counts no {class} entries"))
                    })?,
            )
            .expect("entry counts fit usize");
        }
        let shards = manifest["shards"]
            .as_array()
            .ok_or_else(|| LibraryError("the manifest lists no shards".into()))?
            .iter()
            .map(|shard| {
                shard["file"]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| LibraryError("a shard entry names no file".into()))
            })
            .collect::<Result<Vec<String>, LibraryError>>()?;
        let characters = manifest
            .get("characters")
            .and_then(serde_json::Value::as_array)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| id.as_str()?.parse().ok())
                    .collect()
            });
        let origins = manifest.get("origins").map(|counts| {
            FightOrigin::ALL.map(|origin| {
                usize::try_from(counts[origin.as_str()].as_u64().unwrap_or(0))
                    .expect("origin counts fit usize")
            })
        });
        Ok(Self {
            directory: directory.to_path_buf(),
            header: expected,
            shards,
            classes,
            origins,
            characters,
            with_setup,
            with_state,
        })
    }

    /// How many entries the bank holds per origin, in canonical origin
    /// order, where the manifest says — `None` for a bank written before it
    /// did. A judging instrument reads this to refuse a bank that any
    /// forced-win walk fed.
    #[must_use]
    pub const fn origins(&self) -> Option<[usize; ORIGINS]> {
        self.origins
    }

    /// How many entries carry a setup.
    #[must_use]
    pub const fn with_setup(&self) -> usize {
        self.with_setup
    }

    /// How many entries carry an exact state — the ones a load from the
    /// state can stand up. A consumer that needs every entry exact reads
    /// this against [`Self::entries`] before it asks.
    #[must_use]
    pub const fn with_state(&self) -> usize {
        self.with_state
    }

    /// Which characters the bank was harvested from, where the manifest
    /// says — `None` for a bank written before it did. Descriptive: a
    /// heterogeneous bank is legal, since every entry stands up against its
    /// own character's registry, and the header is what makes the mixture
    /// visible.
    #[must_use]
    pub fn characters(&self) -> Option<&[ModelId]> {
        self.characters.as_deref()
    }

    /// How many entries the bank holds per class, in canonical class order.
    #[must_use]
    pub const fn classes(&self) -> [usize; 5] {
        self.classes
    }

    /// How many entries the bank holds in all. The tiers partition the
    /// bank; the `act2` overlay does not add to the total.
    #[must_use]
    pub fn entries(&self) -> usize {
        Tier::ALL
            .iter()
            .map(|tier| self.classes[tier.index()])
            .sum()
    }

    /// Loads the entries a plan selected, answered in plan order — one
    /// handle per playout, shared where playouts replay the same entry.
    ///
    /// A plan names an entry as the k-th of its class in bank order, so
    /// this is one streaming pass over the shards: every line's metadata is
    /// read, only selected lines are parsed in full, and each selected
    /// entry is stood up — from its setup, through the engine's combat
    /// start on the run seed the entry names, or from its exact state — and
    /// erased into the belief the playouts deal worlds from.
    ///
    /// An entry the engine will not re-enter is one fight the batch does not
    /// get, not a batch the caller does not get: a bank is harvested in bulk
    /// from real play and may hold a fight this build cannot stand up, and
    /// refusing the whole plan over one of them throws away every other
    /// fight in it. Those entries are skipped and named in
    /// [`LoadedBatch::unusable`], and what to do about a bank losing too many
    /// of them is the caller's to decide. A bank that does not carry the half
    /// `source` names is still refused outright, on the first entry: that is
    /// a question about the bank and not about one fight.
    pub fn load(
        &self,
        plan: &[(Class, usize)],
        source: Source,
    ) -> Result<LoadedBatch, LibraryError> {
        /// A shard line read for its metadata alone, the state skipped.
        #[derive(serde::Deserialize)]
        struct Probe {
            meta: FightMeta,
        }
        let needed: BTreeSet<(Class, usize)> = plan.iter().copied().collect();
        let mut found: BTreeMap<(Class, usize), Arc<LoadedFight>> = BTreeMap::new();
        let mut registries: BTreeMap<(ModelId, u8), Arc<ContentRegistry>> = BTreeMap::new();
        let preset = UnlockPresetManifest::pinned()
            .map_err(|error| LibraryError(format!("unlock preset: {error}")))?;
        let mut counts = [0_usize; 5];
        let mut unusable: Vec<String> = Vec::new();
        for shard in &self.shards {
            let path = self.directory.join(shard);
            let text = std::fs::read_to_string(&path)
                .map_err(|error| LibraryError(format!("{}: {error}", path.display())))?;
            let mut lines = text.lines();
            let opened: serde_json::Value = lines
                .next()
                .map(serde_json::from_str)
                .transpose()
                .map_err(|error| LibraryError(format!("{}: {error}", path.display())))?
                .ok_or_else(|| LibraryError(format!("{}: an empty shard", path.display())))?;
            if ENFORCED
                .iter()
                .any(|field| opened.get(*field) != self.header.get(*field))
            {
                return Err(LibraryError(format!(
                    "{}: shard header differs from the manifest",
                    path.display()
                )));
            }
            for line in lines {
                let probe: Probe = serde_json::from_str(line)
                    .map_err(|error| LibraryError(format!("{}: {error}", path.display())))?;
                // The entry's name in every class that holds it. An entry
                // several selected classes name is parsed once and shared.
                let mut keys = Vec::new();
                for class in Class::ALL {
                    if class.holds(&probe.meta) {
                        let key = (class, counts[class.index()]);
                        counts[class.index()] += 1;
                        if needed.contains(&key) {
                            keys.push(key);
                        }
                    }
                }
                if keys.is_empty() {
                    continue;
                }
                let entry: FightEntry = serde_json::from_str(line)
                    .map_err(|error| LibraryError(format!("{}: {error}", path.display())))?;
                let registry = registries
                    .entry((entry.meta.character.clone(), entry.meta.ascension))
                    .or_insert_with(|| {
                        sts2_content::registry_at(&entry.meta.character, entry.meta.ascension)
                    });
                let world = match stand_up(&entry, source, &preset, registry) {
                    Ok(world) => world,
                    Err(StandUpFailure::Bank(error)) => return Err(error),
                    Err(StandUpFailure::Entry(error)) => {
                        unusable.push(error.to_string());
                        continue;
                    }
                };
                let belief = BeliefState::from_simulator(&world)
                    .map_err(|error| LibraryError(format!("{error:?}")))?;
                let fight = Arc::new(LoadedFight {
                    meta: entry.meta,
                    belief,
                });
                for key in keys {
                    found.insert(key, Arc::clone(&fight));
                }
            }
        }
        if counts != self.classes {
            return Err(LibraryError(format!(
                "the manifest counts {:?} entries by class, the shards hold {counts:?}",
                self.classes
            )));
        }
        // Past the count check above, the shards hold exactly what the
        // manifest says they do, and `plan` draws its keys from those same
        // counts — so a key with nothing behind it is an entry that would
        // not stand up, already named in `unusable`, and not a bank whose
        // manifest lies about itself.
        Ok(LoadedBatch {
            fights: plan
                .iter()
                .filter_map(|key| found.get(key).cloned())
                .collect(),
            unusable,
        })
    }
}

/// What a plan drew: the fights that stood up, and what was said about the
/// entries that would not.
pub struct LoadedBatch {
    pub fights: Vec<Arc<LoadedFight>>,
    pub unusable: Vec<String>,
}

/// Why an entry did not stand up, and therefore whose problem it is.
///
/// A bank that does not carry the half the caller asked for cannot serve
/// this batch at all and says so on the first entry — asking a setup-only
/// bank for exact states is a question about the bank, not about one fight.
/// An entry the engine will not re-enter is one fight, and the batch goes on
/// without it.
enum StandUpFailure {
    Bank(LibraryError),
    Entry(LibraryError),
}

/// An entry stood up from the half `source` names: the setup through the
/// engine's own combat start on the entry's run seed, or the exact state.
fn stand_up(
    entry: &FightEntry,
    source: Source,
    preset: &UnlockPresetManifest,
    registry: &Arc<ContentRegistry>,
) -> Result<Simulator, StandUpFailure> {
    let named = |what: &str| {
        format!(
            "entry {} of run {} (seed {}, floor {}) {what}",
            entry.meta.fight, entry.meta.run, entry.meta.seed, entry.meta.floor
        )
    };
    match source {
        Source::State => {
            let state = entry.state.clone().ok_or_else(|| {
                StandUpFailure::Bank(LibraryError(named("carries no exact state")))
            })?;
            Simulator::from_portable_state(state, Arc::clone(registry)).map_err(|error| {
                StandUpFailure::Entry(LibraryError(format!("{}: {error:?}", named("state"))))
            })
        }
        Source::Setup => {
            let setup = entry.setup(registry).map_err(StandUpFailure::Bank)?;
            sts2_content::combat_from_setup_with(
                &setup,
                &entry.meta.seed,
                preset,
                Some(Arc::clone(registry)),
            )
            .map_err(|error| {
                StandUpFailure::Entry(LibraryError(format!("{}: {error:?}", named("setup"))))
            })
        }
    }
}

/// The manifest of a bank [`convert`] can read — the previous format or this
/// one, on this belief model and build — and the shard files it lists.
fn convertible_manifest(from: &Path) -> Result<(serde_json::Value, Vec<String>), LibraryError> {
    let manifest_path = from.join("manifest.json");
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .map_err(|error| LibraryError(format!("{}: {error}", manifest_path.display())))?,
    )
    .map_err(|error| LibraryError(format!("{}: {error}", manifest_path.display())))?;
    let format = manifest.get("format").and_then(serde_json::Value::as_u64);
    if format != Some(u64::from(PREVIOUS_FORMAT)) && format != Some(u64::from(LIBRARY_FORMAT)) {
        return Err(LibraryError(format!(
            "library format is {}, this build converts {PREVIOUS_FORMAT} and {LIBRARY_FORMAT}",
            format.map_or("unsaid".to_owned(), |format| format.to_string())
        )));
    }
    let expected = header();
    for field in &ENFORCED[1..] {
        if manifest.get(*field) != expected.get(*field) {
            return Err(LibraryError(format!(
                "library {field} is {}, this build wants {}",
                manifest[*field], expected[*field]
            )));
        }
    }
    let shards = manifest["shards"]
        .as_array()
        .ok_or_else(|| LibraryError("the manifest lists no shards".into()))?
        .iter()
        .map(|shard| {
            shard["file"]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| LibraryError("a shard entry names no file".into()))
        })
        .collect::<Result<Vec<String>, LibraryError>>()?;
    Ok((manifest, shards))
}

/// Rewrites a bank into this build's format, each entry saying its fight as
/// `keep` asks: a format-2 bank's exact states are stood up and projected
/// to reach their setups, and a format-3 bank has its states dropped or
/// kept. Runs are regrouped as the source grouped them, so the (run, fight)
/// stamps survive where every run left at least one entry. Answers with the
/// per-class counts written.
pub fn convert(from: &Path, to: &Path, keep: HarvestAs) -> Result<[usize; 5], LibraryError> {
    let (manifest, shards) = convertible_manifest(from)?;
    let runs_per_shard = manifest["shards"]
        .as_array()
        .and_then(|shards| shards.first())
        .and_then(|shard| shard["runs"].as_u64())
        .and_then(|runs| usize::try_from(runs).ok())
        .unwrap_or(32);
    let mut sink = LibrarySink::create(to, runs_per_shard)
        .map_err(|error| LibraryError(format!("{}: {error}", to.display())))?;
    for (key, value) in manifest.as_object().into_iter().flatten() {
        // What the source bank said about itself beyond the counts it kept
        // — a forced-win config, a provenance note — is carried across.
        if !matches!(
            key.as_str(),
            "format"
                | "belief_model_version"
                | "observation_version"
                | "compatibility_id"
                | "runs"
                | "entries"
                | "classes"
                | "origins"
                | "characters"
                | "shards"
                | "with_setup"
                | "with_state"
        ) {
            sink.annotate(key, value.clone());
        }
    }
    let mut registries: BTreeMap<(ModelId, u8), Arc<ContentRegistry>> = BTreeMap::new();
    let mut run: Vec<FightEntry> = Vec::new();
    let mut current: Option<usize> = None;
    let mut flush = |run: &mut Vec<FightEntry>| -> Result<(), LibraryError> {
        if run.is_empty() {
            return Ok(());
        }
        sink.write_run(run)
            .map_err(|error| LibraryError(format!("{}: {error}", to.display())))?;
        run.clear();
        Ok(())
    };
    for shard in &shards {
        let path = from.join(shard);
        let text = std::fs::read_to_string(&path)
            .map_err(|error| LibraryError(format!("{}: {error}", path.display())))?;
        for line in text.lines().skip(1) {
            let entry: FightEntry = serde_json::from_str(line)
                .map_err(|error| LibraryError(format!("{}: {error}", path.display())))?;
            if current != Some(entry.meta.run) {
                flush(&mut run)?;
                current = Some(entry.meta.run);
            }
            let registry = registries
                .entry((entry.meta.character.clone(), entry.meta.ascension))
                .or_insert_with(|| {
                    sts2_content::registry_at(&entry.meta.character, entry.meta.ascension)
                });
            let setup = keep
                .writes_setup()
                .then(|| entry.setup(registry))
                .transpose()?;
            let state = if keep.writes_state() {
                Some(entry.state.clone().ok_or_else(|| {
                    LibraryError(format!(
                        "entry {} of run {} carries no exact state to keep",
                        entry.meta.fight, entry.meta.run
                    ))
                })?)
            } else {
                None
            };
            run.push(FightEntry {
                meta: entry.meta,
                setup,
                state,
            });
        }
    }
    flush(&mut run)?;
    sink.finish()
        .map_err(|error| LibraryError(format!("{}: {error}", to.display())))
}

/// A banked entry stood back up: the belief it was banked from, ready to
/// deal worlds, with its metadata for the summary line.
pub struct LoadedFight {
    pub meta: FightMeta,
    pub belief: BeliefState,
}

/// One lane of the mix: a relative weight over the fights the absolute
/// lanes leave, or every banked entry of the class, once each.
#[derive(Clone, Copy, Debug)]
enum Share {
    Weight(f64),
    All,
}

/// An explicit class mix, parsed from `boss=0.3,elite=0.3,hallway=0.4`.
/// Weights are relative — normalized over the lanes named — and a class
/// not named gets nothing. `all` in place of a weight replays every banked
/// entry of the class exactly once, taken off the top of the fight count
/// before the weighted lanes split the rest: `boss=all,act2=all,hallway=1`
/// covers the bank's scarcest entries whole, however few fights buy it.
#[derive(Clone, Copy, Debug)]
pub struct Mix {
    lanes: [Option<Share>; 5],
}

impl Mix {
    pub fn parse(text: &str) -> Result<Self, LibraryError> {
        let mut lanes = [None; 5];
        for part in text.split(',') {
            let (name, value) = part
                .split_once('=')
                .ok_or_else(|| LibraryError(format!("{part:?} is not a class=weight pair")))?;
            let class: Class = name.trim().parse()?;
            let share = match value.trim() {
                "all" => Share::All,
                weight => {
                    let weight: f64 = weight
                        .parse()
                        .map_err(|_| LibraryError(format!("{value:?} is not a weight")))?;
                    if !(weight > 0.0 && weight.is_finite()) {
                        return Err(LibraryError(format!("a {class} weight must be positive")));
                    }
                    Share::Weight(weight)
                }
            };
            if lanes[class.index()].is_some() {
                return Err(LibraryError(format!("{class} is named twice")));
            }
            lanes[class.index()] = Some(share);
        }
        Ok(Self { lanes })
    }

    /// How many of `fights` each class is owed, in canonical class order,
    /// against a bank holding `classes` entries per class. The `all` lanes
    /// take their bank counts off the top; the weighted lanes split what
    /// remains by largest-remainder apportionment — exact, deterministic,
    /// and summing to `fights`.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "fight counts are far below f64 precision"
    )]
    pub fn quotas(&self, fights: usize, classes: [usize; 5]) -> Result<[usize; 5], LibraryError> {
        let mut quotas = [0_usize; 5];
        let mut weights = [0.0_f64; 5];
        for class in Class::ALL {
            match self.lanes[class.index()] {
                Some(Share::All) => {
                    if classes[class.index()] == 0 {
                        return Err(LibraryError(format!(
                            "the mix asks for every {class} entry but the library banks none"
                        )));
                    }
                    quotas[class.index()] = classes[class.index()];
                }
                Some(Share::Weight(weight)) => weights[class.index()] = weight,
                None => {}
            }
        }
        let spoken_for: usize = quotas.iter().sum();
        let Some(remaining) = fights.checked_sub(spoken_for) else {
            return Err(LibraryError(format!(
                "the mix's all lanes name {spoken_for} entries but only {fights} fights are asked \
                 for"
            )));
        };
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            if remaining > 0 {
                return Err(LibraryError(format!(
                    "the mix names no weighted lane for the remaining {remaining} fights"
                )));
            }
            return Ok(quotas);
        }
        let shares = weights.map(|weight| weight / total * remaining as f64);
        let mut owed = remaining;
        for (quota, share) in quotas.iter_mut().zip(shares) {
            *quota += share.floor() as usize;
            owed -= share.floor() as usize;
        }
        let mut order = [0, 1, 2, 3, 4];
        order.sort_by(|&a, &b| {
            (shares[b] - shares[b].floor())
                .partial_cmp(&(shares[a] - shares[a].floor()))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for index in order {
            if owed == 0 {
                break;
            }
            quotas[index] += 1;
            owed -= 1;
        }
        Ok(quotas)
    }
}

/// The generation schedule: which banked entry each fight playout replays,
/// named as (class, index-within-class in bank order).
///
/// Quotas come from the mix. Inside a class the bank order is shuffled once
/// under the analysis seed — so a short schedule spreads over the bank
/// instead of replaying the harvest's first runs — and the playouts walk
/// the shuffled order in cycles, keeping any entry's replay count within
/// one of any other's; an `all` lane's quota is exactly one cycle, every
/// banked entry once. Deterministic in (class counts, mix, fights, seed),
/// which is half of what keeps the emitted shards byte-identical.
pub fn plan(
    mix: &Mix,
    fights: usize,
    classes: [usize; 5],
    analysis_seed: u64,
) -> Result<Vec<(Class, usize)>, LibraryError> {
    let quotas = mix.quotas(fights, classes)?;
    let mut schedule = Vec::with_capacity(fights);
    for class in Class::ALL {
        let quota = quotas[class.index()];
        if quota == 0 {
            continue;
        }
        let held = classes[class.index()];
        if held == 0 {
            return Err(LibraryError(format!(
                "the mix asks for {class} fights but the library banks none"
            )));
        }
        let mut order: Vec<usize> = (0..held).collect();
        // The belief sampler's golden-ratio fold, here giving each class its
        // own shuffle stream off the one analysis seed.
        let mut fold =
            analysis_seed ^ (class.index() as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = MegaRandom::new(splitmix64(&mut fold));
        for hole in (1..order.len()).rev() {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "bank sizes are far below u64"
            )]
            let pick = (rng.next_u64() % (hole as u64 + 1)) as usize;
            order.swap(hole, pick);
        }
        for playout in 0..quota {
            schedule.push((class, order[playout % held]));
        }
    }
    Ok(schedule)
}

/// One generation batch: every fight playout it will run, plan-aligned —
/// playout `i` replays `fights[i]`.
pub struct GenerationBatch<'a> {
    pub fights: &'a [Arc<LoadedFight>],
    pub analysis_seed: u64,
    pub max_steps: usize,
    /// How many fights play at once. One is the serial batch.
    pub jobs: usize,
}

/// How one fight playout ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FightOutcome {
    Won,
    Lost,
    /// The step cap fell before the fight did.
    Unfinished,
}

/// What one fight playout came to.
pub struct FightReport {
    pub meta: FightMeta,
    pub outcome: FightOutcome,
    /// The objective's score where the playout ended — the very `z` its
    /// decisions carry.
    pub value: f64,
    pub decisions: Vec<crate::training::Decision>,
}

/// Plays a generation batch, `jobs` playouts at a time, handing each report
/// to `report` in playout order.
///
/// Playout `i` deals its world with `sample(analysis_seed, i)` — the bank's
/// replay contract — and seeds its search stream from the same index fold
/// the run batch uses, so a playout is a function of its index and the
/// loaded plan alone: `--jobs` stays a throughput lever, and the emitted
/// shards stay byte-identical. A playout whose play panics is lost loudly,
/// with its entry and index as the reproducer, and the batch keeps walking.
pub fn play_fights(
    batch: &GenerationBatch<'_>,
    policy: &(dyn Fn() -> Box<dyn RolloutPolicy> + Sync),
    report: &mut dyn FnMut(usize, Result<FightReport, PlayError>),
) {
    indexed(
        batch.fights.len(),
        batch.jobs,
        &|index| {
            let fight = &batch.fights[index];
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut policy = policy();
                play_fight(
                    fight,
                    batch.analysis_seed,
                    index,
                    policy.as_mut(),
                    batch.max_steps,
                )
            }))
            .unwrap_or_else(|panic| {
                Err(PlayError::panicked(format!(
                    "panicked: {}",
                    panic_message(&panic)
                )))
            })
            .map_err(|error| {
                error.prefixed(&format!(
                    "fight {index} of {} from {} run {}",
                    fight.meta.encounter, fight.meta.seed, fight.meta.run
                ))
            })
        },
        report,
    );
}

/// One playout: a world dealt from the banked belief, walked under the
/// policy to the fight's own end — where the belief horizon stops — and
/// scored there, exactly as a fight inside a full run would be.
fn play_fight(
    fight: &LoadedFight,
    analysis_seed: u64,
    index: usize,
    policy: &mut dyn RolloutPolicy,
    max_steps: usize,
) -> Result<FightReport, PlayError> {
    let mut simulator = fight
        .belief
        .sample(analysis_seed, index as u64)
        .map_err(|error| PlayError::lost(format!("{error:?}")))?;
    let (_, policy_seed) = run_seeds(analysis_seed, index);
    let mut rng = MegaRandom::new(policy_seed);
    let objective = CombatStrength::default();
    for _ in 0..max_steps {
        if simulator.state().terminal.is_some() || fight_over(&simulator) {
            break;
        }
        let action = policy.choose(&simulator, &mut rng);
        simulator
            .step_quietly(&action)
            .map_err(|error| PlayError::lost(format!("{error:?}")))?;
    }
    policy.run_ended(&simulator);
    let decisions = policy.drain_decisions();
    let outcome = if simulator.state().terminal == Some(RunResult::Defeat) {
        FightOutcome::Lost
    } else if fight_over(&simulator) {
        FightOutcome::Won
    } else {
        FightOutcome::Unfinished
    };
    Ok(FightReport {
        meta: fight.meta.clone(),
        outcome,
        value: objective.peek(&simulator),
        decisions,
    })
}

/// One line of the generation summary.
#[must_use]
pub fn summarize(report: &FightReport) -> String {
    format!(
        "{} {} act {} floor {} {} value {:.2} ({} decisions)",
        report.meta.encounter,
        report.meta.tier,
        report.meta.act,
        report.meta.floor,
        match report.outcome {
            FightOutcome::Won => "WON",
            FightOutcome::Lost => "LOST",
            FightOutcome::Unfinished => "unfinished",
        },
        report.value,
        report.decisions.len(),
    )
}

/// A generation batch's fight outcomes in aggregate: the win rate and the
/// mean value, alongside a table of what the losses went to and a reading of
/// which characters the batch actually played. Feed it every finished
/// playout's report; it reads, it never plays.
///
/// The counterpart of [`crate::selfplay::OutcomeTally`], which counts runs.
/// A bank replay has no floors to average and no act to clear, so what it
/// aggregates is the fight.
#[derive(Clone, Debug, Default)]
pub struct FightTally {
    fights: usize,
    won: usize,
    lost: usize,
    unfinished: usize,
    value: f64,
    /// Losses keyed by act index, tier, and encounter.
    losses: crate::tally::Rows<Tier>,
    characters: BTreeMap<ModelId, usize>,
}

impl FightTally {
    /// Counts one finished playout.
    pub fn record(&mut self, report: &FightReport) {
        self.fights += 1;
        self.value += report.value;
        *self
            .characters
            .entry(report.meta.character.clone())
            .or_default() += 1;
        match report.outcome {
            FightOutcome::Won => self.won += 1,
            FightOutcome::Unfinished => self.unfinished += 1,
            FightOutcome::Lost => {
                self.lost += 1;
                self.losses.record(
                    report.meta.act,
                    report.meta.tier,
                    report.meta.encounter.to_string(),
                    report.meta.floor,
                );
            }
        }
    }

    /// The one-line aggregate: the win rate, the mean value, and the
    /// playouts that ended some other way.
    #[must_use]
    #[allow(clippy::cast_precision_loss, reason = "batch sizes are small")]
    pub fn summary(&self) -> String {
        let fights = self.fights.max(1) as f64;
        format!(
            "fights won {}/{} ({:.1}%) | mean value {:.2} | {} lost | {} unfinished",
            self.won,
            self.fights,
            100.0 * self.won as f64 / fights,
            self.value / fights,
            self.lost,
            self.unfinished,
        )
    }

    /// The loss table, costliest encounter first, at most `limit` rows plus
    /// a remainder line for whatever the limit cut. Each row is one
    /// encounter of one tier of one act, with its loss count and floor span.
    #[must_use]
    pub fn loss_table(&self, limit: usize) -> Vec<String> {
        self.losses.table(limit, "losses")
    }

    /// Who the batch played, with a playout count each: the composition of
    /// the data it just wrote.
    #[must_use]
    pub const fn composition(&self) -> &BTreeMap<ModelId, usize> {
        &self.characters
    }
}
