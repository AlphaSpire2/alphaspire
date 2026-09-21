//! Fast, network-free summaries of recording and decision-script traces.

use std::collections::BTreeMap;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sts2_engine::{GameState, MapPointType, RunMode, RunResult, Simulator};

use crate::analyze::{AnalyzeError, check_compatibility, recorded_run};

pub const TRACE_SUMMARY_FORMAT: u32 = 1;

/// The run-defining fields read from a trace's headers and opening state.
#[derive(Clone, Debug)]
pub struct RunSpec {
    pub seed: String,
    pub character: sts2_core::ModelId,
    pub ascension: u8,
    pub preset: sts2_core::UnlockPresetManifest,
}

/// One settled HP sample, keeping the last state reached on each floor.
#[derive(Clone, Debug, Serialize)]
pub struct HpPoint {
    pub floor: u32,
    pub act: usize,
    pub hp: i32,
    pub max_hp: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub room: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encounter: Option<String>,
}

/// High-level progress and resources at the end of one act's observed play.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ActSummary {
    pub act: usize,
    pub floors_visited: usize,
    pub fights: usize,
    pub hallway_fights: usize,
    pub elites: usize,
    pub bosses: usize,
    pub rest_sites: usize,
    pub shops: usize,
    pub events: usize,
    pub treasures: usize,
    pub hp_start: i32,
    pub hp_end: i32,
    pub max_hp_end: i32,
    pub deck_size_end: usize,
    pub relics_end: usize,
    pub gold_end: i32,
}

/// Everything a comparison of runs needs from one run, without evaluating it.
#[derive(Clone, Debug, Serialize)]
pub struct TraceSummary {
    pub trace_summary_format: u32,
    pub path: PathBuf,
    pub source: &'static str,
    pub seed: String,
    pub character: String,
    pub ascension: u8,
    pub result: &'static str,
    pub floor_reached: u32,
    pub act_reached: usize,
    pub verified: bool,
    pub hp_curve: Vec<HpPoint>,
    pub acts: Vec<ActSummary>,
}

/// Reads enough of a trace to reproduce its starting world exactly.
pub fn run_spec(path: &Path) -> Result<RunSpec, AnalyzeError> {
    let bytes = std::fs::read(path).map_err(|error| AnalyzeError::Io(path.to_path_buf(), error))?;
    let mut parser = sts2_replay::Parser::new(BufReader::new(std::io::Cursor::new(bytes)))
        .map_err(|error| AnalyzeError::Trace(format!("{}: {error}", path.display())))?;
    check_compatibility(&parser)
        .map_err(|error| AnalyzeError::Trace(format!("{}: {error}", path.display())))?;
    let seed = parser
        .header("Seed")
        .ok_or_else(|| AnalyzeError::Trace("no Seed header".into()))?
        .to_owned();
    let ascension = parser
        .header("Ascension")
        .and_then(|text| text.parse().ok())
        .ok_or_else(|| AnalyzeError::Trace("the Ascension header is not a level".into()))?;
    let mode = mode(parser.header("Mode"));
    let first = loop {
        match parser
            .next()
            .ok_or_else(|| AnalyzeError::Trace("the trace carries no records".into()))?
            .map_err(|error| AnalyzeError::Trace(error.to_string()))?
        {
            sts2_replay::Event::Record(record) => break record,
            sts2_replay::Event::Terminal(_) => {}
        }
    };
    let (simulator, character) = recorded_run(&seed, ascension, mode, &first)?;
    let model = character
        .parse()
        .map_err(|_| AnalyzeError::Trace("the character is not a model id".into()))?;
    let unlocks = simulator
        .state()
        .run
        .as_ref()
        .ok_or_else(|| AnalyzeError::Trace("run.start generated no run".into()))?
        .unlocks
        .clone();
    let preset = sts2_core::UnlockPresetManifest::matching(&unlocks)
        .map_err(|error| AnalyzeError::Trace(error.to_string()))?
        .ok_or_else(|| AnalyzeError::Trace("the trace has no registered unlock preset".into()))?;
    Ok(RunSpec {
        seed,
        character: model,
        ascension,
        preset,
    })
}

/// Replays a trace once and extracts run-level facts without loading a net.
pub fn summarize_trace(path: &Path) -> Result<TraceSummary, AnalyzeError> {
    let bytes = std::fs::read(path).map_err(|error| AnalyzeError::Io(path.to_path_buf(), error))?;
    let mut parser = sts2_replay::Parser::new(BufReader::new(std::io::Cursor::new(bytes)))
        .map_err(|error| AnalyzeError::Trace(format!("{}: {error}", path.display())))?;
    check_compatibility(&parser)
        .map_err(|error| AnalyzeError::Trace(format!("{}: {error}", path.display())))?;
    let header = |key: &str| parser.header(key).map(str::to_owned);
    let seed = header("Seed").ok_or_else(|| AnalyzeError::Trace("no Seed header".into()))?;
    let ascension = header("Ascension")
        .and_then(|text| text.parse().ok())
        .ok_or_else(|| AnalyzeError::Trace("the Ascension header is not a level".into()))?;
    let run_mode = mode(parser.header("Mode"));
    let is_recording = parser.header("RecorderVersion").is_some();
    let header_result = header("Result");

    let mut adapter: Option<sts2_replay::ReplayAdapter> = None;
    let mut character = String::new();
    let mut curve = BTreeMap::<u32, HpPoint>::new();
    let mut act_resources = BTreeMap::<usize, ActSummary>::new();
    let mut terminal_token = None;
    for event in &mut parser {
        let event = event.map_err(|error| AnalyzeError::Trace(error.to_string()))?;
        let record = match event {
            sts2_replay::Event::Record(record) => record,
            sts2_replay::Event::Terminal(token) => {
                terminal_token = Some(token);
                break;
            }
        };
        if adapter.is_none() {
            let (simulator, played_as) = recorded_run(&seed, ascension, run_mode, &record)?;
            character = played_as;
            observe(&simulator, &mut curve, &mut act_resources);
            let driver = sts2_replay::ReplayAdapter::new(simulator);
            adapter = Some(if is_recording {
                driver.verifying_observations()
            } else {
                driver
            });
        }
        let driver = adapter.as_mut().expect("the adapter was just opened");
        match driver.accept(&record) {
            Ok(sts2_replay::AdapterOutcome::Applied(_)) => {
                observe(driver.simulator(), &mut curve, &mut act_resources);
            }
            // A reload puts the run back at its last save; whatever was
            // sampled past that point is gone from the resumed timeline.
            Ok(sts2_replay::AdapterOutcome::Ignored(_)) if record.kind == "run.resume" => {
                if let Some(run) = driver.simulator().state().run.as_ref() {
                    curve.split_off(&(run.floor + 1));
                    act_resources.split_off(&(run.current_act + 2));
                }
                observe(driver.simulator(), &mut curve, &mut act_resources);
            }
            Ok(sts2_replay::AdapterOutcome::Ignored(_)) => {}
            Err(error) => {
                return Err(AnalyzeError::Trace(format!("{}: {error}", path.display())));
            }
        }
    }
    let Some(mut driver) = adapter else {
        return Err(AnalyzeError::Trace(format!(
            "{}: carries no records to replay",
            path.display()
        )));
    };
    driver
        .verify_pending_observation()
        .map_err(|error| AnalyzeError::Trace(format!("{}: {error}", path.display())))?;
    observe(driver.simulator(), &mut curve, &mut act_resources);
    let state = driver.simulator().state();
    add_room_counts(state, &mut act_resources);
    let run = state
        .run
        .as_ref()
        .ok_or_else(|| AnalyzeError::Trace("the replay ended without a run".into()))?;
    let result = match state.terminal {
        Some(RunResult::Victory) => "victory",
        Some(RunResult::Defeat) => "defeat",
        None => match terminal_token.as_deref().or(header_result.as_deref()) {
            Some(value) if value.eq_ignore_ascii_case("victory") => "victory",
            Some(value) if value.eq_ignore_ascii_case("defeat") => "defeat",
            _ => "unfinished",
        },
    };
    Ok(TraceSummary {
        trace_summary_format: TRACE_SUMMARY_FORMAT,
        path: path.to_path_buf(),
        source: if is_recording { "recording" } else { "script" },
        seed,
        character,
        ascension,
        result,
        floor_reached: run.floor,
        act_reached: run.current_act + 1,
        verified: is_recording,
        hp_curve: curve.into_values().collect(),
        acts: act_resources.into_values().collect(),
    })
}

fn mode(value: Option<&str>) -> RunMode {
    if value.is_some_and(|text| text.eq_ignore_ascii_case("custom")) {
        RunMode::Custom
    } else {
        RunMode::Standard
    }
}

fn observe(
    simulator: &Simulator,
    curve: &mut BTreeMap<u32, HpPoint>,
    acts: &mut BTreeMap<usize, ActSummary>,
) {
    let state = simulator.state();
    let Some(run) = state.run.as_ref() else {
        return;
    };
    let (hp, max_hp) = player_hp(state);
    let act = run.current_act + 1;
    let room_type = run.standing_room_type();
    let encounter = crate::selfplay::standing_room(run)
        .0
        .map(|model| model.to_string());
    curve.insert(
        run.floor,
        HpPoint {
            floor: run.floor,
            act,
            hp: hp.max(0),
            max_hp,
            room: room_type.map(|room| crate::selfplay::room_label(Some(room))),
            encounter,
        },
    );
    let entry = acts.entry(act).or_insert_with(|| ActSummary {
        act,
        hp_start: hp.max(0),
        ..ActSummary::default()
    });
    entry.hp_end = hp.max(0);
    entry.max_hp_end = max_hp;
    entry.deck_size_end = state.run_player.deck.len();
    entry.relics_end = state.run_player.relics.len();
    entry.gold_end = state.run_player.gold;
}

fn add_room_counts(state: &GameState, acts: &mut BTreeMap<usize, ActSummary>) {
    let Some(run) = state.run.as_ref() else {
        return;
    };
    for history in &run.map_history {
        let act = history.act + 1;
        let summary = acts.entry(act).or_insert_with(|| ActSummary {
            act,
            ..ActSummary::default()
        });
        summary.floors_visited = history.points.len();
        for room in history.points.iter().flat_map(|point| &point.rooms) {
            match room.room_type {
                MapPointType::Monster => {
                    summary.fights += 1;
                    summary.hallway_fights += 1;
                }
                MapPointType::Elite => {
                    summary.fights += 1;
                    summary.elites += 1;
                }
                MapPointType::Boss => {
                    summary.fights += 1;
                    summary.bosses += 1;
                }
                MapPointType::RestSite => summary.rest_sites += 1,
                MapPointType::Shop => summary.shops += 1,
                MapPointType::Ancient | MapPointType::Unknown => summary.events += 1,
                MapPointType::Treasure => summary.treasures += 1,
            }
        }
    }
}

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
