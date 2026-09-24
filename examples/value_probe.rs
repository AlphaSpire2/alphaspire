//! Prices the child states of one recorded combat decision with combat
//! checkpoints' value heads, and re-runs that decision's belief search under
//! several analysis seeds.
//!
//! ```text
//! value_probe <trace.sts2pgn> <seq> <net-stem>... -- <line>...
//! ```
//!
//! A line is a comma-separated list of matchers walked from the decision:
//! `end_turn`, or a substring of a play-card action's JSON such as
//! `CARD.BLUDGEON`. Each line is walked in the true world and in
//! `PROBE_WORLDS` sampled belief worlds (default 64); the value head is read
//! at every step. `PROBE_ITERS` / `PROBE_CONSIDERED` / `PROBE_SEEDS` shape
//! the searches (default 1024 / 32 / 5).
//!
//! A diagnostic, not a product: it is exempt from the crate's pedantic
//! lints so its one long walk can stay in reading order.
#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::format_push_string,
    clippy::items_after_statements,
    clippy::must_use_candidate
)]

use std::path::Path;
use std::sync::Arc;

use alphaspire::encoding::PolicyEncoder;
use alphaspire::env::{Belief, Determinizer, fight_over};
use alphaspire::net::{Evaluate, PolicyValueNet};
use alphaspire::objective::CombatStrength;
use alphaspire::search::{BeliefSearch, Gumbel, SearchConfig};
use sts2_engine::{Action, Simulator};
use sts2_rng::MegaRandom;

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let split = args
        .iter()
        .position(|arg| arg == "--")
        .unwrap_or(args.len());
    let trace = Path::new(&args[1]);
    let seq: u64 = args[2].parse().expect("a record sequence");
    let stems = &args[3..split];
    let lines: Vec<Vec<String>> = args[split + 1..]
        .iter()
        .map(|line| line.split(',').map(str::to_owned).collect())
        .collect();
    let worlds = env_or("PROBE_WORLDS", 64);
    let iterations = env_or("PROBE_ITERS", 1024);
    let considered = usize::try_from(env_or("PROBE_CONSIDERED", 32)).expect("small");
    let seeds = env_or("PROBE_SEEDS", 5);

    let root = state_at(trace, seq);
    let registry = sts2_content::standard_registry();
    let encoder = Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ));
    let nets: Vec<(String, Arc<PolicyValueNet>)> = stems
        .iter()
        .map(|stem| {
            let net = PolicyValueNet::load(Path::new(stem), Arc::clone(&encoder))
                .unwrap_or_else(|error| panic!("{stem}: {error}"));
            (stem.clone(), Arc::new(net))
        })
        .collect();

    println!("== seq {seq}: {}", describe(&root));
    for (name, net) in &nets {
        println!("   v[{name}] = {:.4}", net.state_value(&root));
    }
    println!("== legal actions");
    for (index, action) in root.legal_actions().iter().enumerate() {
        println!("   {index}: {}", short(action));
    }

    for line in &lines {
        println!("== line {}", line.join(" > "));
        let mut sim = root.clone();
        for step in line {
            let Some(action) = find(&sim, step) else {
                println!("   (no legal action matches {step})");
                break;
            };
            let label = short(&action);
            if sim.step_quietly(&action).is_err() {
                println!("   {label}: refused");
                break;
            }
            let values: Vec<String> = nets
                .iter()
                .map(|(name, net)| format!("v[{name}]={:.4}", net.state_value(&sim)))
                .collect();
            println!(
                "   true world after {label}: {} | {}",
                describe(&sim),
                values.join(" ")
            );
        }
        // The same line in sampled worlds: what the tree's leaves see.
        let mut belief = Belief::from_simulator(&root, 1).expect("a fight samples");
        let mut sums = vec![0.0; nets.len()];
        let mut lows = vec![f64::INFINITY; nets.len()];
        let mut highs = vec![f64::NEG_INFINITY; nets.len()];
        let mut hp_sum = 0.0;
        let mut enemy_sum = 0.0;
        let mut ended = 0_u64;
        let mut completed = 0_u64;
        let mut keys = std::collections::HashSet::new();
        for world in 0..worlds {
            let mut sim = belief.sample(world);
            let mut whole = true;
            for step in line {
                let Some(action) = find(&sim, step) else {
                    whole = false;
                    break;
                };
                if sim.step_quietly(&action).is_err() {
                    whole = false;
                    break;
                }
            }
            if !whole {
                continue;
            }
            completed += 1;
            if let Ok(key) = sim.observation_key() {
                keys.insert(format!("{key:?}"));
            }
            if sim.state().terminal.is_some() || fight_over(&sim) {
                ended += 1;
            }
            let (hp, enemy) = hp_pair(&sim);
            hp_sum += f64::from(hp);
            enemy_sum += f64::from(enemy);
            for (index, (_, net)) in nets.iter().enumerate() {
                let value = net.state_value(&sim);
                sums[index] += value;
                lows[index] = lows[index].min(value);
                highs[index] = highs[index].max(value);
            }
        }
        if completed > 0 {
            #[allow(clippy::cast_precision_loss, reason = "a world count")]
            let n = completed as f64;
            println!(
                "   {completed}/{worlds} sampled worlds walked the line ({ended} ended the fight; {} distinct positions); mean hp {:.1}, mean enemy hp {:.1}",
                keys.len(),
                hp_sum / n,
                enemy_sum / n
            );
            for (index, (name, _)) in nets.iter().enumerate() {
                println!(
                    "   sampled v[{name}]: mean {:.4}  min {:.4}  max {:.4}",
                    sums[index] / n,
                    lows[index],
                    highs[index]
                );
            }
        }
    }

    if let Ok(list) = std::env::var("PROBE_DEPTH_WORLDS") {
        for (_, net) in &nets {
            for worlds in list.split(',').filter_map(|w| w.parse::<u64>().ok()) {
                for seed in 1..=seeds {
                    let level = std::env::var("PROBE_KEY_LEVEL")
                        .ok()
                        .and_then(|text| text.parse().ok());
                    depth_profile(
                        &root,
                        net,
                        u32::try_from(iterations).expect("a budget"),
                        considered,
                        worlds,
                        seed,
                        level,
                    );
                }
            }
        }
        return;
    }
    if std::env::var("PROBE_HANDS").is_ok() {
        // Distinct next hands after ending the turn, as card multisets.
        let mut belief = Belief::from_simulator(&root, 1).expect("a fight samples");
        let mut hands: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        let mut keys = std::collections::HashSet::new();
        let mut levels: Vec<std::collections::HashSet<String>> =
            (0..5).map(|_| std::collections::HashSet::new()).collect();
        let mut fields: std::collections::BTreeMap<String, std::collections::HashSet<String>> =
            std::collections::BTreeMap::new();
        let end = find(&root, "end_turn").expect("end turn is legal");
        let mut draw_size = None;
        for world in 0..worlds {
            let mut sim = belief.sample(world);
            if draw_size.is_none() {
                let piles = &sim.state().combat.as_ref().expect("in combat").player.piles;
                draw_size = Some((piles.draw.len(), piles.discard.len(), piles.hand.len()));
            }
            sim.step_quietly(&end).expect("end turn steps");
            if let Ok(key) = sim.observation_key() {
                keys.insert(format!("{key:?}"));
            }
            for level in 1..=4u8 {
                levels[usize::from(level)].insert(canonical_json(&sim, level));
            }
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&canonical_json(&sim, 4))
                && let Some(map) = value.as_object()
            {
                for (field, content) in map {
                    fields
                        .entry(field.clone())
                        .or_default()
                        .insert(content.to_string());
                }
            }
            let piles = &sim.state().combat.as_ref().expect("in combat").player.piles;
            let mut names: Vec<String> = piles
                .hand
                .iter()
                .map(|card| {
                    format!(
                        "{}{}",
                        card.fingerprint
                            .model_id
                            .to_string()
                            .trim_start_matches("CARD."),
                        if card.fingerprint.upgrade_level > 0 {
                            "+"
                        } else {
                            ""
                        }
                    )
                })
                .collect();
            names.sort();
            *hands.entry(names.join(",")).or_insert(0) += 1;
        }
        println!(
            "== next hands over {worlds} worlds: draw/discard/hand before end turn {:?}; {} distinct hands as multisets; {} distinct observation keys",
            draw_size,
            hands.len(),
            keys.len()
        );
        println!(
            "   distinct keys by canonical level: L1 ids scrubbed {} | L2 + piles sorted {} | L3 + history dropped {} | L4 + decision dropped {}",
            levels[1].len(),
            levels[2].len(),
            levels[3].len(),
            levels[4].len()
        );
        let varying: Vec<String> = fields
            .iter()
            .filter(|(_, set)| set.len() > 1)
            .map(|(field, set)| format!("{field}:{}", set.len()))
            .collect();
        println!(
            "   fields with more than one distinct value at L4: {}",
            varying.join(" ")
        );
        for (field, set) in &fields {
            if set.len() > 1 && set.len() <= 300 {
                let mut samples: Vec<&String> = set.iter().collect();
                samples.sort();
                let shown: Vec<String> = samples
                    .iter()
                    .take(2)
                    .map(|text| text.chars().take(400).collect())
                    .collect();
                println!("   {field}: e.g. {}", shown.join("  ||  "));
            }
        }
        let mut common: Vec<_> = hands.iter().collect();
        common.sort_by(|a, b| b.1.cmp(a.1));
        for (hand, n) in common.iter().take(8) {
            println!("   {n:4}  {hand}");
        }
        return;
    }
    if let Ok(list) = std::env::var("PROBE_FEW_WORLDS") {
        for (name, net) in &nets {
            for worlds in list.split(',').filter_map(|w| w.parse::<u64>().ok()) {
                for seed in 1..=seeds {
                    let policy = search_few_worlds(
                        &root,
                        net,
                        u32::try_from(iterations).expect("a budget"),
                        considered,
                        worlds,
                        seed,
                    );
                    let mut rows: Vec<_> = policy
                        .actions
                        .iter()
                        .zip(&policy.candidates)
                        .map(|((action, pi), candidate)| {
                            (candidate.completed_value, short(action), *pi, *candidate)
                        })
                        .collect();
                    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    let top: Vec<String> = rows
                        .iter()
                        .take(4)
                        .map(|(_, label, pi, c)| {
                            format!(
                                "{label} Q {:.3} n {} pi {:.2}",
                                c.completed_value, c.visits, pi
                            )
                        })
                        .collect();
                    println!(
                        "== few-worlds [{name}] {iterations}/{considered} worlds {worlds} seed {seed}: {}",
                        top.join(" | ")
                    );
                }
            }
        }
        return;
    }
    for (name, net) in &nets {
        for seed in 1..=seeds {
            let mut tree = BeliefSearch::new(
                SearchConfig {
                    iterations: u32::try_from(iterations).expect("a budget"),
                    ..SearchConfig::default()
                },
                CombatStrength::default(),
            )
            .selecting(Box::new(Gumbel {
                considered,
                ..Gumbel::default()
            }))
            .with_net(Arc::clone(net) as Arc<dyn Evaluate>);
            let mut rng = MegaRandom::new(seed);
            let policy = tree
                .analyze(&root, &[], &mut rng)
                .expect("a fight searches");
            println!(
                "== search [{name}] {iterations}/{considered} seed {seed}: root_value {:?} q_spread {:?}",
                policy.root_value, policy.q_spread
            );
            let mut rows: Vec<_> = policy
                .actions
                .iter()
                .zip(&policy.candidates)
                .map(|((action, pi), candidate)| {
                    (candidate.completed_value, short(action), *pi, *candidate)
                })
                .collect();
            rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (_, label, pi, candidate) in rows {
                println!(
                    "   {label:36} prior {:.4} visits {:4} mean {:.4} completed {:.4} pi {:.3}",
                    candidate.prior,
                    candidate.visits,
                    candidate.mean_value,
                    candidate.completed_value,
                    pi
                );
            }
        }
    }
}

/// The first legal action a matcher names: `end_turn`, or a substring of a
/// play-card / use-potion action's JSON.
fn find(simulator: &Simulator, matcher: &str) -> Option<Action> {
    simulator
        .legal_actions()
        .iter()
        .find(|action| {
            if matcher == "end_turn" {
                return matches!(action, Action::EndTurn { .. });
            }
            let json = serde_json::to_string(action).expect("an action serializes");
            (json.contains("\"play_card\"") || json.contains("\"use_potion\""))
                && json.contains(matcher)
        })
        .cloned()
}

fn short(action: &Action) -> String {
    let json = serde_json::to_value(action).expect("an action serializes");
    match action {
        Action::EndTurn { .. } => "End turn".to_owned(),
        Action::PlayCard { .. } => {
            let id = json["card"]["fingerprint"]["model_id"]
                .as_str()
                .unwrap_or("?")
                .trim_start_matches("CARD.")
                .to_owned();
            let upgrade = json["card"]["fingerprint"]["upgrade_level"]
                .as_u64()
                .unwrap_or(0);
            let target = json["target"]["model_id"]
                .as_str()
                .map_or(String::new(), |t| {
                    format!(" -> {}", t.trim_start_matches("MONSTER."))
                });
            format!("{id}{}{target}", if upgrade > 0 { "+" } else { "" })
        }
        Action::UsePotion { model_id, .. } => format!("Drink {model_id}"),
        Action::DiscardPotion { model_id, .. } => format!("Discard {model_id}"),
        other => {
            let text = serde_json::to_string(other).expect("an action serializes");
            text.chars().take(60).collect()
        }
    }
}

fn hp_pair(simulator: &Simulator) -> (i32, i32) {
    let state = simulator.state();
    let Some(combat) = state.combat.as_ref() else {
        return (state.run_player.current_hp, 0);
    };
    let player = combat
        .creatures
        .iter()
        .find(|creature| creature.id == combat.player.creature_id)
        .map_or(0, |creature| creature.current_hp);
    let enemies = combat
        .creatures
        .iter()
        .filter(|creature| creature.id != combat.player.creature_id)
        .map(|creature| creature.current_hp.max(0))
        .sum();
    (player, enemies)
}

fn describe(simulator: &Simulator) -> String {
    let state = simulator.state();
    let Some(combat) = state.combat.as_ref() else {
        return format!(
            "out of combat, hp {}/{}, terminal {:?}",
            state.run_player.current_hp, state.run_player.max_hp, state.terminal
        );
    };
    let mut text = String::new();
    for creature in &combat.creatures {
        if creature.id == combat.player.creature_id {
            text.push_str(&format!(
                "turn {} hp {}/{} blk {} E {} | ",
                combat.player.turn,
                creature.current_hp,
                creature.max_hp,
                creature.block,
                combat.player.energy
            ));
        }
    }
    for creature in &combat.creatures {
        if creature.id != combat.player.creature_id {
            let powers: Vec<String> = creature
                .powers
                .iter()
                .map(|power| {
                    format!(
                        "{}={}",
                        power.model_id.to_string().trim_start_matches("POWER."),
                        power.amount
                    )
                })
                .collect();
            text.push_str(&format!(
                "{} {}/{} blk {} {} [{}] ",
                creature.model_id.to_string().trim_start_matches("MONSTER."),
                creature.current_hp,
                creature.max_hp,
                creature.block,
                creature.move_state,
                powers.join(",")
            ));
        }
    }
    if fight_over(simulator) {
        text.push_str("(fight over)");
    }
    text
}

/// The simulator standing before the record `seq` of `trace` was applied.
fn state_at(trace: &Path, seq: u64) -> Simulator {
    let file = std::fs::File::open(trace).expect("the trace opens");
    let input = sts2_replay::decode_input(file).expect("the trace input opens");
    let mut parser = sts2_replay::Parser::new(input).expect("the trace parses");
    let header = |key: &str| parser.header(key).map(str::to_owned);
    let seed = header("Seed").expect("a Seed header");
    let ascension: u8 = header("Ascension")
        .and_then(|text| text.parse().ok())
        .expect("an Ascension header");
    let mode = if header("Mode").is_some_and(|text| text.eq_ignore_ascii_case("custom")) {
        sts2_engine::RunMode::Custom
    } else {
        sts2_engine::RunMode::Standard
    };
    let verifying = header("RecorderVersion").is_some();
    let mut adapter: Option<sts2_replay::ReplayAdapter> = None;
    for event in &mut parser {
        let record = match event.expect("a record reads") {
            sts2_replay::Event::Record(record) => record,
            sts2_replay::Event::Terminal(_) => break,
        };
        if adapter.is_none() {
            let simulator = recorded_run(&seed, ascension, mode, &record);
            let driver = sts2_replay::ReplayAdapter::new(simulator);
            adapter = Some(if verifying {
                driver.verifying_observations()
            } else {
                driver
            });
        }
        let driver = adapter.as_mut().expect("opened");
        let role = sts2_replay::record_role(&record.kind);
        if record.sequence == seq
            && matches!(
                role,
                sts2_replay::TimelineRole::InitiatingAction
                    | sts2_replay::TimelineRole::NestedChoice
            )
        {
            return driver.simulator().clone();
        }
        if let Err(error) = driver.accept(&record) {
            panic!("record {} diverged: {error:?}", record.sequence);
        }
    }
    panic!("record {seq} is not a decision in {}", trace.display());
}

fn recorded_run(
    seed: &str,
    ascension: u8,
    mode: sts2_engine::RunMode,
    first: &sts2_replay::Record,
) -> Simulator {
    assert_eq!(first.kind, "run.start", "a trace opens with run.start");
    let unlocks: sts2_core::UnlockState = first
        .payload
        .pointer("/data/state/players/0/unlocks")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .expect("run.start projects an unlock state");
    let preset = sts2_core::UnlockPresetManifest::matching(&unlocks)
        .expect("the manifest reads")
        .expect("a registered unlock preset");
    let character = first
        .payload
        .pointer("/data/state/players/0/character_model")
        .and_then(serde_json::Value::as_str)
        .expect("run.start projects a character");
    let model: sts2_core::ModelId = character.parse().expect("a model id");
    sts2_content::run_on_preset_at(seed, &model, &preset, ascension, mode).expect("the run opens")
}

/// A belief that hands the search only `worlds` distinct samples, so visits
/// past a turn boundary land on positions the tree has already opened.
struct FewWorlds {
    inner: Belief,
    worlds: u64,
}

impl Determinizer for FewWorlds {
    fn sample(&mut self, rollout: u64) -> Simulator {
        self.inner.sample(rollout % self.worlds)
    }

    fn node_key(
        &self,
        simulator: &Simulator,
    ) -> Result<alphaspire::env::NodeKey, sts2_engine::EngineError> {
        self.inner.node_key(simulator)
    }

    fn beyond_horizon(&self, simulator: &Simulator) -> bool {
        self.inner.beyond_horizon(simulator)
    }
}

/// The root table of a search over `worlds` sampled worlds.
pub fn search_few_worlds(
    root: &Simulator,
    net: &Arc<PolicyValueNet>,
    iterations: u32,
    considered: usize,
    worlds: u64,
    seed: u64,
) -> alphaspire::search::RootPolicy {
    let mut mcts = alphaspire::search::Mcts::new(
        SearchConfig {
            iterations,
            ..SearchConfig::default()
        },
        Gumbel {
            considered,
            ..Gumbel::default()
        },
    );
    let mut determinizer = FewWorlds {
        inner: Belief::from_simulator(root, seed).expect("a fight samples"),
        worlds,
    };
    let mut rollout = alphaspire::policy::UniformRandom;
    let mut rng = MegaRandom::new(seed);
    let (_, policy) = mcts.decide_pinned(
        &mut determinizer,
        &CombatStrength::default(),
        &mut rollout,
        Some(net.as_ref() as &dyn Evaluate),
        &[],
        &mut rng,
    );
    policy
}

fn turn_of(simulator: &Simulator) -> u32 {
    simulator
        .state()
        .combat
        .as_ref()
        .map_or(0, |combat| combat.player.turn)
}

/// A net that records the turn of every position it is asked to price.
struct Counting {
    inner: Arc<PolicyValueNet>,
    turns: std::sync::Mutex<std::collections::BTreeMap<u32, u64>>,
}

impl Counting {
    fn note(&self, simulator: &Simulator) {
        *self
            .turns
            .lock()
            .expect("counter")
            .entry(turn_of(simulator))
            .or_insert(0) += 1;
    }
}

impl Evaluate for Counting {
    fn priors_and_value(
        &self,
        simulator: &Simulator,
        actions: &[Action],
    ) -> alphaspire::net::Priced {
        self.note(simulator);
        self.inner.priors_and_value(simulator, actions)
    }

    fn plan_priors_and_value(
        &self,
        simulator: &Simulator,
        plans: &[alphaspire::plan::ActionPlan],
    ) -> alphaspire::net::Priced {
        self.note(simulator);
        self.inner.plan_priors_and_value(simulator, plans)
    }

    fn state_value(&self, simulator: &Simulator) -> f64 {
        self.note(simulator);
        self.inner.state_value(simulator)
    }
}

/// A belief over `worlds` samples that also records, per iteration, the
/// deepest turn any keyed node stood on.
struct Walked {
    inner: Belief,
    worlds: u64,
    level: Option<u8>,
    current: u32,
    started: bool,
    deepest: std::collections::BTreeMap<u32, u64>,
}

impl Walked {
    fn flush(&mut self) {
        if self.started {
            *self.deepest.entry(self.current).or_insert(0) += 1;
        }
        self.current = 0;
        self.started = true;
    }
}

impl Determinizer for Walked {
    fn sample(&mut self, rollout: u64) -> Simulator {
        self.flush();
        self.inner.sample(rollout % self.worlds)
    }

    fn node_key(
        &self,
        simulator: &Simulator,
    ) -> Result<alphaspire::env::NodeKey, sts2_engine::EngineError> {
        // Keyed positions are nodes the walk stood on; the deepest per
        // iteration is tracked through interior mutability below.
        DEEPEST.with(|cell| {
            let turn = turn_of(simulator);
            let mut deepest = cell.borrow_mut();
            if turn > *deepest {
                *deepest = turn;
            }
        });
        match self.level {
            Some(level) => Ok(canonical_key(simulator, level)),
            None => self.inner.node_key(simulator),
        }
    }

    fn beyond_horizon(&self, simulator: &Simulator) -> bool {
        self.inner.beyond_horizon(simulator)
    }
}

thread_local! {
    static DEEPEST: std::cell::RefCell<u32> = const { std::cell::RefCell::new(0) };
}

/// Runs one search over `worlds` samples and prints where its leaves landed
/// and how deep its iterations walked, by turn.
pub fn depth_profile(
    root: &Simulator,
    net: &Arc<PolicyValueNet>,
    iterations: u32,
    considered: usize,
    worlds: u64,
    seed: u64,
    level: Option<u8>,
) {
    let counting = Arc::new(Counting {
        inner: Arc::clone(net),
        turns: std::sync::Mutex::new(std::collections::BTreeMap::new()),
    });
    let mut mcts = alphaspire::search::Mcts::new(
        SearchConfig {
            iterations,
            ..SearchConfig::default()
        },
        Gumbel {
            considered,
            ..Gumbel::default()
        },
    );
    let mut determinizer = Walked {
        inner: Belief::from_simulator(root, seed).expect("a fight samples"),
        worlds,
        level,
        current: 0,
        started: false,
        deepest: std::collections::BTreeMap::new(),
    };
    let mut rollout = alphaspire::policy::UniformRandom;
    let mut rng = MegaRandom::new(seed);
    // Per-iteration depth: `sample` opens an iteration, so read the
    // thread-local high-water mark at each open and at the end.
    let mut deepest: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    struct Hooked<'a> {
        inner: &'a mut Walked,
        deepest: &'a mut std::collections::BTreeMap<u32, u64>,
        open: bool,
    }
    impl Determinizer for Hooked<'_> {
        fn sample(&mut self, rollout: u64) -> Simulator {
            if self.open {
                let turn = DEEPEST.with(|cell| *cell.borrow());
                *self.deepest.entry(turn).or_insert(0) += 1;
            }
            DEEPEST.with(|cell| *cell.borrow_mut() = 0);
            self.open = true;
            self.inner.sample(rollout)
        }
        fn node_key(
            &self,
            simulator: &Simulator,
        ) -> Result<alphaspire::env::NodeKey, sts2_engine::EngineError> {
            self.inner.node_key(simulator)
        }
        fn beyond_horizon(&self, simulator: &Simulator) -> bool {
            self.inner.beyond_horizon(simulator)
        }
    }
    let mut hooked = Hooked {
        inner: &mut determinizer,
        deepest: &mut deepest,
        open: false,
    };
    let (_, policy) = mcts.decide_pinned(
        &mut hooked,
        &CombatStrength::default(),
        &mut rollout,
        Some(counting.as_ref() as &dyn Evaluate),
        &[],
        &mut rng,
    );
    if hooked.open {
        let turn = DEEPEST.with(|cell| *cell.borrow());
        *hooked.deepest.entry(turn).or_insert(0) += 1;
    }
    let leaves = counting.turns.lock().expect("counter").clone();
    let total: u64 = leaves.values().sum();
    let leaf_text: Vec<String> = leaves
        .iter()
        .map(|(turn, n)| format!("t{turn}:{n}"))
        .collect();
    let iter_total: u64 = deepest.values().sum();
    let iter_text: Vec<String> = deepest
        .iter()
        .map(|(turn, n)| format!("t{turn}:{n}"))
        .collect();
    let mut rows: Vec<_> = policy
        .actions
        .iter()
        .zip(&policy.candidates)
        .map(|((action, pi), candidate)| {
            (candidate.completed_value, short(action), *pi, *candidate)
        })
        .collect();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<String> = rows
        .iter()
        .take(3)
        .map(|(_, label, pi, c)| format!("{label} Q {:.3} pi {:.2}", c.completed_value, pi))
        .collect();
    let degraded = mcts.degradations();
    println!(
        "== depth {iterations}/{considered} worlds {worlds} key {} seed {seed}: nodes {} dead_ends {} | net calls by turn ({total}): {} | iterations by deepest turn ({iter_total}): {} | {}",
        level.map_or("raw".to_owned(), |level| format!("L{level}")),
        mcts.tree_size(),
        degraded.dead_ends,
        leaf_text.join(" "),
        iter_text.join(" "),
        top.join(" | ")
    );
}

/// The observation as JSON, canonicalized to `level`: 0 raw; 1 copy identity
/// (card ids, hand indices) scrubbed; 2 plus the ordered piles sorted; 3 plus
/// the visible history dropped; 4 plus the decision context dropped.
fn canonical_json(simulator: &Simulator, level: u8) -> String {
    let mut value =
        serde_json::to_value(simulator.agent_observation()).expect("an observation serializes");
    if level >= 1 {
        scrub_ids(&mut value);
    }
    if level >= 2
        && let Some(map) = value.as_object_mut()
    {
        for pile in ["hand", "hand_previews", "discard", "exhaust", "play"] {
            if let Some(cards) = map.get_mut(pile).and_then(serde_json::Value::as_array_mut) {
                cards.sort_by_key(std::string::ToString::to_string);
            }
        }
    }
    if level >= 3
        && let Some(map) = value.as_object_mut()
    {
        map.remove("visible_history");
    }
    if level >= 4
        && let Some(map) = value.as_object_mut()
    {
        map.remove("decision");
    }
    value.to_string()
}

fn scrub_ids(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            if map.contains_key("card_id") {
                map.remove("card_id");
                if map.contains_key("fingerprint") {
                    map.remove("index");
                }
            }
            for entry in map.values_mut() {
                scrub_ids(entry);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                scrub_ids(item);
            }
        }
        _ => {}
    }
}

fn canonical_key(simulator: &Simulator, level: u8) -> alphaspire::env::NodeKey {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(canonical_json(simulator, level).as_bytes());
    alphaspire::env::NodeKey::Observation(sts2_engine::ObservationKey(digest.into()))
}
