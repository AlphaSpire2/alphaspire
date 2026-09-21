//! The determinization boundary the search core stands on.
//!
//! The search core never touches the authoritative simulator directly, only
//! simulators handed to it by its determinizer. This keeps hidden state out
//! of belief search.
//! Three modes implement it: [`TrueState`] for clairvoyant coverage fuzzing,
//! [`Belief`] for fair analysis over worlds sampled from what a player can
//! see inside one fight, and [`ActBelief`] for the same fairness across the
//! whole current act.

use sts2_engine::{
    ActBeliefState, Action, BeliefState, DecisionContext, EngineError, ObservationKey, Simulator,
    StateKey,
};

/// Whether the fight this simulator stands in is over.
///
/// Not `combat.is_none()`: the engine keeps the combat alive behind the
/// victory screen — `has_ended` is its own word that the fight is done, and
/// `rewards_open` outlives it while the room waits to be left. `combat-v0`
/// erases nothing at the run level, so a reward roll is *sampled* in a
/// possible world rather than seen: two samples of one reward screen offer
/// different gold. The fight is where the erasure stops being honest, so the
/// fight is where the horizon goes.
#[must_use]
pub fn fight_over(simulator: &Simulator) -> bool {
    simulator
        .state()
        .combat
        .as_ref()
        .is_none_or(|combat| combat.has_ended || combat.rewards_open)
}

/// Whether this simulator stands at the end of its act — the horizon of the
/// `act-v0.5` belief model.
///
/// The act ends where the engine raises its own transition
/// ([`DecisionContext::ActTransition`]) — before the next act's map is
/// generated or read — and a run that ended (either way) ended inside it.
/// One wrinkle the transition check alone would miss: the engine offers
/// [`Action::AdvanceAct`] straight off the boss's own exit screens too — the
/// boss's reward set and its room-proceed — and stepping it from *there*
/// crosses immediately, no transition screen between. Crossing inside a
/// sample regenerates the next act's map from the analysis rng, a world
/// neither true nor fairly sampled (the next act's pools were carried as
/// they stand), so any state the crossing is one step from is the horizon:
/// where `AdvanceAct` is on offer, the sample is scored where it stands.
/// The cost is small and accepted: the boss's own reward claims go
/// unsearched, scored just before the belt inflates.
#[must_use]
pub fn act_over(simulator: &Simulator) -> bool {
    simulator.state().terminal.is_some()
        || matches!(simulator.decision(), DecisionContext::ActTransition { .. })
        || simulator
            .legal_actions()
            .iter()
            .any(|action| matches!(action, Action::AdvanceAct))
}

/// What the tree and transposition table key a decision point by.
///
/// The variants never collide in one table: exact keys belong to
/// [`TrueState`] search, observation keys to both belief arms — and keying
/// a belief node by exact state would let a search distinguish states no
/// player could.
///
/// An observation key separates two differently-rolled reward screens by
/// itself, because the observation carries what a screen offers; nothing
/// about the offers needs joining to the key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NodeKey {
    /// The simulator's canonical full-state key.
    Exact(StateKey),
    /// The canonical key over what a policy can see.
    Observation(ObservationKey),
}

/// A source of simulators to run rollouts on.
///
/// The same rollout index yields the same determinization; different indices
/// may differ wherever the mode hides state. `TrueState` hides nothing, so
/// every index yields the same world.
pub trait Determinizer {
    /// A simulator standing at the decision point the search is analyzing,
    /// with hidden state fixed for the length of one rollout.
    fn sample(&mut self, rollout: u64) -> Simulator;

    /// The key for the decision point a sampled simulator is standing on.
    fn node_key(&self, simulator: &Simulator) -> Result<NodeKey, EngineError>;

    /// Whether one action from one keyed state always reaches one keyed
    /// state. True for `TrueState`, where every sample is the same world, so
    /// a search may cache the key an edge reaches instead of recomputing it
    /// every descent. False for any belief mode: hidden state varies by
    /// rollout, and the same action can land on different keys.
    fn deterministic_transitions(&self) -> bool {
        false
    }

    /// Whether a sampled world has walked past what this mode samples. A
    /// search treats such a state as a leaf: it is scored where it stands and
    /// never stepped further, because beyond the horizon the sample's hidden
    /// state was carried rather than sampled and reading it would be
    /// clairvoyant. `TrueState` samples everything, so nothing is beyond it.
    fn beyond_horizon(&self, _simulator: &Simulator) -> bool {
        false
    }

    /// Whether a sampled world stands on a decision this mode *plays* rather
    /// than *branches over*. Inside the horizon — so unlike
    /// [`beyond_horizon`](Determinizer::beyond_horizon) the walk continues —
    /// but below the tree: the search stops opening nodes there and hands the
    /// rest of the line to its rollout policy.
    ///
    /// The act mode is what needs it. `act-v0.5`'s design is that a rollout
    /// which reaches a fight *plays* it with the cheap rollout policy under
    /// the depth cap, and only out-of-combat decisions are searched (the
    /// reasoning is in `ActSearch`'s own documentation). Without a fence the
    /// tree creeps into the fight anyway — one ply per iteration — and every
    /// in-fight ply it opens costs an observation key, a canonical action
    /// list, and a re-stepped prefix on every later iteration, spent on
    /// decisions the act tree is not the right searcher for. Modes that
    /// branch over everything they can reach answer `false` and nothing
    /// changes.
    fn plays_out(&self, _simulator: &Simulator) -> bool {
        false
    }
}

/// The validation mode: samples are clones of the authentic full state —
/// deck order, unrevealed map, every stream position. Clairvoyant by design;
/// a coverage fuzzer, not a player, whose output is still a legal sequence
/// of game inputs.
#[derive(Clone, Debug)]
pub struct TrueState {
    simulator: Simulator,
}

impl TrueState {
    #[must_use]
    pub fn new(simulator: Simulator) -> Self {
        Self { simulator }
    }

    /// Advances the authentic state by one chosen action, re-rooting the
    /// determinizer on the decision point that follows.
    pub fn advance(&mut self, action: &sts2_engine::Action) -> Result<(), EngineError> {
        self.simulator.step_quietly(action)
    }

    /// The authentic state, read-only: what a driver script is emitted from.
    #[must_use]
    pub const fn simulator(&self) -> &Simulator {
        &self.simulator
    }
}

impl Determinizer for TrueState {
    fn sample(&mut self, _rollout: u64) -> Simulator {
        self.simulator.clone()
    }

    fn node_key(&self, simulator: &Simulator) -> Result<NodeKey, EngineError> {
        Ok(NodeKey::Exact(simulator.state_key()?))
    }

    fn deterministic_transitions(&self) -> bool {
        true
    }
}

/// Fair analysis: samples hidden state
/// consistent with `agent_observation` and visible history, keys nodes by
/// what a player can see, and is labeled analysis, not a fidelity claim.
///
/// The erasure and the sampling live in the simulator's [`BeliefState`]
/// (model `combat-v0`): the authoritative simulator is erased to
/// player-visible state *before* this determinizer is built, so what the
/// search holds cannot be clairvoyant. The model is combat-scoped, and the
/// end of the fight is the horizon: a rollout that leaves the combat is
/// scored where it stands.
#[derive(Clone, Debug)]
pub struct Belief {
    belief: BeliefState,
    analysis_seed: u64,
}

impl Belief {
    /// Erases a combat decision point to what a player can see. Refused
    /// outside a fight — and on the victory screen, which is past it — where
    /// `combat-v0` has nothing honest to sample.
    pub fn from_simulator(simulator: &Simulator, analysis_seed: u64) -> Result<Self, EngineError> {
        if fight_over(simulator) {
            return Err(EngineError::new(sts2_engine::ErrorCode::InvalidConfig)
                .with_detail("belief", "combat-v0 samples a fight in progress"));
        }
        Ok(Self {
            belief: BeliefState::from_simulator(simulator)?,
            analysis_seed,
        })
    }
}

impl Determinizer for Belief {
    fn sample(&mut self, rollout: u64) -> Simulator {
        self.belief
            .sample(self.analysis_seed, rollout)
            .expect("an erased combat samples")
    }

    fn node_key(&self, simulator: &Simulator) -> Result<NodeKey, EngineError> {
        Ok(NodeKey::Observation(simulator.observation_key()?))
    }

    fn beyond_horizon(&self, simulator: &Simulator) -> bool {
        fight_over(simulator)
    }
}

/// Act-scoped analysis using [`ActBeliefState`] (model `act-v0.5`).
/// Samples hidden state through [`act_over`], including future rooms,
/// rewards, and encounters.
///
/// The current reward screen, shop inventory, or event page is preserved
/// in every sample. Future rewards are sampled during rollouts, and
/// observation keys include the offers to distinguish reward screens.
///
/// Search covers pathing, card picks, events, rest sites, and shops.
/// Combat remains within the act horizon but is handled by the rollout
/// policy or evaluated by the combat network in `search.rs`.
#[derive(Clone, Debug)]
pub struct ActBelief {
    belief: ActBeliefState,
    analysis_seed: u64,
}

impl ActBelief {
    /// Erases a run decision point to what a player can see, act-scoped.
    /// Refused at [`act_over`] — the transition screen, the boss's exit
    /// screens, a finished run — where the act's remainder is spent and a
    /// sample would have nothing of the act left to be uncertain about; the
    /// real screen is answered directly instead. Also refused (by the
    /// engine) outside a run, where no act bounds a sample.
    pub fn from_simulator(simulator: &Simulator, analysis_seed: u64) -> Result<Self, EngineError> {
        if act_over(simulator) {
            return Err(EngineError::new(sts2_engine::ErrorCode::InvalidConfig)
                .with_detail("belief", "act-v0.5 samples an act in progress"));
        }
        Ok(Self {
            belief: ActBeliefState::from_simulator(simulator)?,
            analysis_seed,
        })
    }
}

impl Determinizer for ActBelief {
    fn sample(&mut self, rollout: u64) -> Simulator {
        self.belief
            .sample(self.analysis_seed, rollout)
            .expect("an erased act samples")
    }

    fn node_key(&self, simulator: &Simulator) -> Result<NodeKey, EngineError> {
        Ok(NodeKey::Observation(simulator.observation_key()?))
    }

    fn beyond_horizon(&self, simulator: &Simulator) -> bool {
        act_over(simulator)
    }

    /// A live fight is played, not searched: the act tree's nodes are the
    /// act's own decisions — pathing, card picks, events, rest sites, shops,
    /// standing reward screens — and everything between them is rollout.
    fn plays_out(&self, simulator: &Simulator) -> bool {
        !fight_over(simulator)
    }
}
