//! Search and self-play companion to `sts2sim`, named after Stockfish.
//!
//! Alphaspire consumes the simulator's public API — `legal_actions`, `step`,
//! `snapshot`/`restore`, `agent_observation` — and never reaches inside it.
//! Its first job is the validation loop: policies play runs in the simulator,
//! emit decision scripts, a driver executes them in the real game while the
//! recorder writes the trace, and the trace replays back through the
//! simulator's own comparator. The real game defines the expected behavior.

pub mod actor;
pub mod analyze;
pub mod encoding;
pub mod env;
pub mod forcewins;
pub mod heuristics;
pub mod library;
pub mod matchup;
pub mod net;
pub mod objective;
pub mod plan;
pub mod policy;
pub mod probe;
pub mod reward;
pub mod script;
pub mod search;
pub mod selfplay;
pub mod summary;
mod tally;
pub mod trace_summary;
pub mod training;
