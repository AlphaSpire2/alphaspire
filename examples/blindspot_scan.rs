//! Finds the positions where a value head prices passing the turn above
//! attacking first — a one-turn-out blindness — across a whole fight bank, and says which fights hold one.
//!
//! ```text
//! blindspot_scan <library> <net-stem>...
//! ```
//!
//! Every fight the plan draws (the same `plan` as `alphaspire fights`, so
//! row `i` here is line `i` of a `fights` log run with the same mix, count
//! and seed) is walked under the first checkpoint's greedy policy. At each
//! decision where the turn can be ended and an attack is affordable, the
//! decision is erased to what the player sees, `SCAN_WORLDS` belief worlds
//! are dealt, and two kinds of line are walked in each: `end turn` alone,
//! and `attack, end turn` for every distinct affordable attack. Every
//! checkpoint's value head reads the position each line reaches (a fight
//! that ended is scored by the objective, as a search leaf is). A checkpoint
//! whose mean value after passing exceeds its mean after the best attack has
//! the position inverted.
//!
//! `SCAN_MIX` / `SCAN_FIGHTS` / `SCAN_SEED` mirror the `fights` flags
//! (default `elite=all,boss=all`, every such entry, seed 1); `SCAN_WORLDS`
//! (16) and `SCAN_JOBS` (cores) size the work; `SCAN_OUT` names the JSONL
//! the rows go to (default `blindspot-rows.jsonl`).
//!
//! `SCAN_PLAYOUT=<file>` switches to ground truth: each `fight decision`
//! line of the file names a probed decision, which is re-walked to and then
//! played out from under the first checkpoint's belief search
//! (`SCAN_ITERS`/`SCAN_CONSIDERED`, default 64/16), `SCAN_REPEATS` (4)
//! times per line — the pass, and the best attack followed by free play —
//! from the true dealt world. The rows carry each line's mean objective
//! and win rate beside the value reads, so an inversion the outcomes
//! contradict is a blind spot and one they confirm is a rule the head knew.
//!
//! A diagnostic, not a product: exempt from the crate's pedantic lints.
#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::format_push_string,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::obfuscated_if_else
)]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use alphaspire::encoding::PolicyEncoder;
use alphaspire::env::{Belief, Determinizer, fight_over};
use alphaspire::library::{Class, FightLibrary, LoadedFight, Mix, plan};
use alphaspire::net::{Evaluate, PolicyValueNet};
use alphaspire::objective::{CombatStrength, Objective};
use alphaspire::policy::{RolloutPolicy, UniformRandom, permitted_actions};
use alphaspire::search::{BeliefSearch, Gumbel, SearchConfig};
use sts2_engine::{Action, CardType, Simulator};
use sts2_rng::MegaRandom;

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

/// One checkpoint's read of one probed decision.
#[derive(serde::Serialize)]
struct Read {
    v_pass: f64,
    best_attack: String,
    v_best: f64,
    /// `v_pass - v_best`: positive is inverted.
    margin: f64,
    inverted: bool,
    /// Every attack line's mean, by label.
    attacks: BTreeMap<String, f64>,
    /// Mean player and enemy HP where the pass line and the best attack
    /// line end: whether the attack changed anything the head could see.
    hp_after_pass: f64,
    enemy_hp_after_pass: f64,
    hp_after_best: f64,
    enemy_hp_after_best: f64,
    /// Fraction of worlds where the enemies' moves and powers at the best
    /// attack line's end differ from the pass line's: whether the attack
    /// changed what the enemy is about to do, not only its HP.
    intent_changed: f64,
}

#[derive(serde::Serialize)]
struct Row {
    fight: usize,
    meta: serde_json::Value,
    decision: usize,
    turn: u32,
    hp: i32,
    max_hp: i32,
    enemy_hp: i32,
    energy: i32,
    enemy_block: i32,
    enemy_powers: Vec<String>,
    hand: Vec<String>,
    /// What the walking checkpoint's greedy policy then played.
    walked: String,
    walk_passed: bool,
    worlds: u64,
    reads: BTreeMap<String, Read>,
    #[serde(skip_serializing_if = "Option::is_none")]
    playout: Option<Playout>,
}

/// Both lines played out from the true world under the first checkpoint's search.
#[derive(serde::Serialize)]
struct Playout {
    iterations: u32,
    considered: usize,
    repeats: u64,
    attack: String,
    pass_value: f64,
    pass_won: f64,
    attack_value: f64,
    attack_won: f64,
    /// Mean turn the fight ended on, per line, and how many times per
    /// playout the search ended a turn with an attack still affordable.
    pass_turns: f64,
    attack_turns: f64,
    pass_passes: f64,
    attack_passes: f64,
    /// What one search at the root itself chose, and whether that was the pass.
    root_choice: String,
    root_passed: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: blindspot_scan <library> <net-stem>...");
        std::process::exit(2);
    }
    let mix_text = std::env::var("SCAN_MIX").unwrap_or_else(|_| "elite=all,boss=all".to_owned());
    let seed = env_or("SCAN_SEED", 1);
    let worlds = env_or("SCAN_WORLDS", 16);
    let jobs = usize::try_from(env_or(
        "SCAN_JOBS",
        std::thread::available_parallelism().map_or(1, |n| n.get() as u64),
    ))
    .expect("a job count");
    let out = std::env::var("SCAN_OUT").unwrap_or_else(|_| "blindspot-rows.jsonl".to_owned());
    // fight -> decisions to play out; empty when scanning.
    let mut wanted: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    if let Ok(list) = std::env::var("SCAN_PLAYOUT") {
        for line in std::fs::read_to_string(&list)
            .expect("the playout list reads")
            .lines()
        {
            let mut parts = line.split_whitespace();
            if let (Some(f), Some(d)) = (parts.next(), parts.next())
                && let (Ok(f), Ok(d)) = (f.parse(), d.parse())
            {
                wanted.entry(f).or_default().push(d);
            }
        }
    }
    let playout = wanted
        .is_empty()
        .then_some(None)
        .unwrap_or(Some(PlayoutConfig {
            iterations: u32::try_from(env_or("SCAN_ITERS", 64)).expect("a budget"),
            considered: usize::try_from(env_or("SCAN_CONSIDERED", 16)).expect("a width"),
            repeats: env_or("SCAN_REPEATS", 4),
        }));

    let library = FightLibrary::open(Path::new(&args[1])).unwrap_or_else(|error| panic!("{error}"));
    let mix = Mix::parse(&mix_text).unwrap_or_else(|error| panic!("{error}"));
    let fights = std::env::var("SCAN_FIGHTS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or_else(|| {
            // Every entry of each `all` lane once, which is the only count an all-lane mix accepts.
            mix_text
                .split(',')
                .filter_map(|term| term.split_once('='))
                .filter(|(_, share)| *share == "all")
                .filter_map(|(class, _)| class.parse::<Class>().ok())
                .map(|class| library.classes()[class.index()])
                .sum()
        });
    let schedule =
        plan(&mix, fights, library.classes(), seed).unwrap_or_else(|error| panic!("{error}"));
    let loaded = library
        .load(&schedule, alphaspire::library::Source::State)
        .unwrap_or_else(|error| panic!("{error}"))
        .fights;

    let registry = sts2_content::standard_registry();
    let encoder = Arc::new(PolicyEncoder::new(
        alphaspire::encoding::standard_vocabulary(&registry),
        &registry,
    ));
    let nets: Vec<(String, Arc<PolicyValueNet>)> = args[2..]
        .iter()
        .map(|stem| {
            let net = PolicyValueNet::load(Path::new(stem), Arc::clone(&encoder))
                .unwrap_or_else(|error| panic!("{stem}: {error}"));
            let name = Path::new(stem)
                .file_name()
                .map_or(stem.clone(), |n| n.to_string_lossy().into_owned());
            (name, Arc::new(net))
        })
        .collect();
    eprintln!(
        "blindspot_scan: {} fights ({mix_text}, seed {seed}), {worlds} worlds, {} checkpoints, walking under {}",
        loaded.len(),
        nets.len(),
        nets[0].0
    );

    let targets: Vec<usize> = if wanted.is_empty() {
        (0..loaded.len()).collect()
    } else {
        wanted
            .keys()
            .copied()
            .filter(|f| *f < loaded.len())
            .collect()
    };
    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<Row>> = Mutex::new(Vec::new());
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            scope.spawn(|| {
                loop {
                    let slot = next.fetch_add(1, Ordering::Relaxed);
                    if slot >= targets.len() {
                        break;
                    }
                    let index = targets[slot];
                    let fight = &loaded[index];
                    let only = wanted.get(&index).map(Vec::as_slice);
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        scan_fight(
                            fight,
                            index,
                            seed,
                            worlds,
                            &nets,
                            &registry,
                            only,
                            playout.as_ref(),
                        )
                    }));
                    match result {
                        Ok(mut found) => rows.lock().expect("rows").append(&mut found),
                        Err(_) => eprintln!(
                            "blindspot_scan: fight {index} of {} from {} run {} panicked; skipped",
                            fight.meta.encounter, fight.meta.seed, fight.meta.run
                        ),
                    }
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_multiple_of(200) || (playout.is_some() && n.is_multiple_of(20)) {
                        eprintln!("blindspot_scan: {n}/{} fights", targets.len());
                    }
                }
            });
        }
    });

    let mut rows = rows.into_inner().expect("rows");
    rows.sort_by_key(|row| (row.fight, row.decision));
    let mut file = std::io::BufWriter::new(std::fs::File::create(&out).expect("the output opens"));
    for row in &rows {
        serde_json::to_writer(&mut file, row).expect("a row serializes");
        file.write_all(b"\n").expect("a row writes");
    }
    file.flush().expect("the output flushes");

    // Summary: per checkpoint, inversions over decisions and over fights, by tier.
    println!(
        "probed {} decisions over {} fights; rows in {out}",
        rows.len(),
        targets.len()
    );
    if playout.is_some() {
        let played: Vec<&Playout> = rows.iter().filter_map(|row| row.playout.as_ref()).collect();
        let n = played.len().max(1) as f64;
        println!(
            "played out {} decisions: pass line mean value {:.3} won {:.1}%; attack line mean value {:.3} won {:.1}%; attack better in {} decisions, pass better in {}",
            played.len(),
            played.iter().map(|p| p.pass_value).sum::<f64>() / n,
            100.0 * played.iter().map(|p| p.pass_won).sum::<f64>() / n,
            played.iter().map(|p| p.attack_value).sum::<f64>() / n,
            100.0 * played.iter().map(|p| p.attack_won).sum::<f64>() / n,
            played
                .iter()
                .filter(|p| p.attack_value > p.pass_value)
                .count(),
            played
                .iter()
                .filter(|p| p.pass_value > p.attack_value)
                .count()
        );
    }
    for (name, _) in &nets {
        let mut inverted = 0usize;
        let mut by_tier: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let mut by_turn: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
        let mut fights_hit = std::collections::BTreeSet::new();
        let mut margins = Vec::new();
        for row in &rows {
            let read = &row.reads[name];
            let tier = row.meta["tier"].as_str().unwrap_or("?").to_owned();
            let t = by_tier.entry(tier).or_default();
            let u = by_turn.entry(row.turn.min(6)).or_default();
            t.0 += 1;
            u.0 += 1;
            if read.inverted {
                inverted += 1;
                t.1 += 1;
                u.1 += 1;
                fights_hit.insert(row.fight);
                margins.push(read.margin);
            }
        }
        margins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = margins.get(margins.len() / 2).copied().unwrap_or(0.0);
        let tiers: Vec<String> = by_tier
            .iter()
            .map(|(tier, (n, k))| format!("{tier} {k}/{n}"))
            .collect();
        let turns: Vec<String> = by_turn
            .iter()
            .map(|(turn, (n, k))| format!("t{turn}{} {k}/{n}", if *turn == 6 { "+" } else { "" }))
            .collect();
        println!(
            "[{name}] inverted {inverted}/{} decisions ({:.1}%) in {} fights; median inverted margin {median:.3}; by tier {}; by turn {}",
            rows.len(),
            100.0 * inverted as f64 / rows.len().max(1) as f64,
            fights_hit.len(),
            tiers.join(", "),
            turns.join(", ")
        );
    }
}

/// Walks one banked fight under the first checkpoint's greedy policy and
/// probes every decision that could end the turn with an attack in hand.
struct PlayoutConfig {
    iterations: u32,
    considered: usize,
    repeats: u64,
}

fn scan_fight(
    fight: &LoadedFight,
    index: usize,
    seed: u64,
    worlds: u64,
    nets: &[(String, Arc<PolicyValueNet>)],
    registry: &sts2_engine::ContentRegistry,
    only: Option<&[usize]>,
    playout: Option<&PlayoutConfig>,
) -> Vec<Row> {
    let mut simulator = fight
        .belief
        .sample(seed, index as u64)
        .expect("a banked entry samples");
    let walker = &nets[0].1;
    let objective = CombatStrength::default();
    let meta = serde_json::to_value(&fight.meta).expect("meta serializes");
    let mut rows = Vec::new();
    for decision in 0..2000 {
        if simulator.state().terminal.is_some() || fight_over(&simulator) {
            break;
        }
        let actions = permitted_actions(&simulator).into_owned();
        if actions.is_empty() {
            break;
        }
        let priced = walker.priors_and_value(&simulator, &actions);
        let (priors, _, _) = priced.into_parts();
        let choice = priors
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        let chosen = actions[choice].clone();

        let end = actions
            .iter()
            .find(|action| matches!(action, Action::EndTurn { .. }))
            .cloned();
        let mut attacks: Vec<(String, Action)> = Vec::new();
        for action in &actions {
            if let Action::PlayCard { card, target } = action
                && registry.card_type_of(&card.fingerprint) == Some(CardType::Attack)
            {
                let label = format!(
                    "{}{}{}",
                    card.fingerprint
                        .model_id
                        .to_string()
                        .trim_start_matches("CARD."),
                    if card.fingerprint.upgrade_level > 0 {
                        "+"
                    } else {
                        ""
                    },
                    target
                        .as_ref()
                        .map_or(String::new(), |t| format!("@{:?}", t.combat_id))
                );
                if !attacks.iter().any(|(l, _)| *l == label) {
                    attacks.push((label, action.clone()));
                }
            }
        }
        let listed = only.is_none_or(|list| list.contains(&decision));
        if let (Some(end), false, true) = (end, attacks.is_empty(), listed)
            && let Some(row) = probe(&simulator, &end, &attacks, seed, worlds, nets, &objective)
        {
            let (hp, max_hp, enemy_hp, energy, turn, hand, enemy_block, enemy_powers) =
                describe(&simulator);
            rows.push(Row {
                fight: index,
                meta: meta.clone(),
                decision,
                turn,
                hp,
                max_hp,
                enemy_hp,
                energy,
                enemy_block,
                enemy_powers,
                hand,
                walked: short(&chosen),
                walk_passed: matches!(chosen, Action::EndTurn { .. }),
                worlds: row.0,
                playout: playout.map(|config| {
                    let best = &row.1[&nets[0].0].best_attack;
                    let attack = attacks
                        .iter()
                        .find(|(label, _)| label == best)
                        .map(|(_, action)| action.clone())
                        .expect("the best attack is one of the lines");
                    play_out(
                        &simulator, &end, &attack, best, config, &nets[0].1, &objective, registry,
                        index, decision,
                    )
                }),
                reads: row.1,
            });
        }
        if simulator.step_quietly(&chosen).is_err() {
            break;
        }
    }
    rows
}

/// The lines, over sampled worlds, and every checkpoint's mean read of
/// where each ends. `None` when no world walked the pass line whole.
fn probe(
    root: &Simulator,
    end: &Action,
    attacks: &[(String, Action)],
    seed: u64,
    worlds: u64,
    nets: &[(String, Arc<PolicyValueNet>)],
    objective: &CombatStrength,
) -> Option<(u64, BTreeMap<String, Read>)> {
    let mut belief = Belief::from_simulator(root, seed).ok()?;
    // sums[line][net]; line 0 is the pass.
    let mut sums = vec![vec![0.0; nets.len()]; attacks.len() + 1];
    let mut counts = vec![0u64; attacks.len() + 1];
    let mut hp_sums = vec![(0.0_f64, 0.0_f64); attacks.len() + 1];
    let mut intent_diffs = vec![0.0_f64; attacks.len() + 1];
    for world in 0..worlds {
        let dealt = belief.sample(world);
        let mut pass_intent = None;
        // The pass alone.
        {
            let mut sim = dealt.clone();
            if sim.step_quietly(end).is_ok() {
                counts[0] += 1;
                pass_intent = Some(intents(&sim));
                let (hp, enemy) = hp_pair(&sim);
                hp_sums[0].0 += f64::from(hp);
                hp_sums[0].1 += f64::from(enemy);
                for (n, (_, net)) in nets.iter().enumerate() {
                    sums[0][n] += leaf_value(&sim, net.as_ref(), objective);
                }
            }
        }
        for (a, (_, attack)) in attacks.iter().enumerate() {
            let mut sim = dealt.clone();
            if sim.step_quietly(attack).is_err() {
                continue;
            }
            if sim.state().terminal.is_none() && !fight_over(&sim) {
                // A card that opened a choice leaves no turn to end; that line is not walked.
                let Some(next_end) = permitted_actions(&sim)
                    .iter()
                    .find(|action| matches!(action, Action::EndTurn { .. }))
                    .cloned()
                else {
                    continue;
                };
                if sim.step_quietly(&next_end).is_err() {
                    continue;
                }
            }
            counts[a + 1] += 1;
            if pass_intent
                .as_ref()
                .is_some_and(|pass| *pass != intents(&sim))
            {
                intent_diffs[a + 1] += 1.0;
            }
            let (hp, enemy) = hp_pair(&sim);
            hp_sums[a + 1].0 += f64::from(hp);
            hp_sums[a + 1].1 += f64::from(enemy);
            for (n, (_, net)) in nets.iter().enumerate() {
                sums[a + 1][n] += leaf_value(&sim, net.as_ref(), objective);
            }
        }
    }
    if counts[0] == 0 || counts[1..].iter().all(|&c| c == 0) {
        return None;
    }
    let mut reads = BTreeMap::new();
    for (n, (name, _)) in nets.iter().enumerate() {
        let v_pass = sums[0][n] / counts[0] as f64;
        let mut means = BTreeMap::new();
        let mut best: Option<(String, f64, usize)> = None;
        for (a, (label, _)) in attacks.iter().enumerate() {
            if counts[a + 1] == 0 {
                continue;
            }
            let mean = sums[a + 1][n] / counts[a + 1] as f64;
            means.insert(label.clone(), mean);
            if best.as_ref().is_none_or(|(_, b, _)| mean > *b) {
                best = Some((label.clone(), mean, a + 1));
            }
        }
        let (best_attack, v_best, line) = best.expect("an attack line was walked");
        let n_pass = counts[0] as f64;
        let n_best = counts[line] as f64;
        reads.insert(
            name.clone(),
            Read {
                v_pass,
                best_attack,
                v_best,
                margin: v_pass - v_best,
                inverted: v_pass > v_best,
                attacks: means,
                hp_after_pass: hp_sums[0].0 / n_pass,
                enemy_hp_after_pass: hp_sums[0].1 / n_pass,
                hp_after_best: hp_sums[line].0 / n_best,
                enemy_hp_after_best: hp_sums[line].1 / n_best,
                intent_changed: intent_diffs[line] / n_best,
            },
        );
    }
    Some((counts[0], reads))
}

/// Both lines from the true world, `repeats` times each under the search:
/// the pass, and the attack followed by whatever the search then plays.
fn play_out(
    root: &Simulator,
    end: &Action,
    attack: &Action,
    label: &str,
    config: &PlayoutConfig,
    net: &Arc<PolicyValueNet>,
    objective: &CombatStrength,
    registry: &sts2_engine::ContentRegistry,
    fight: usize,
    decision: usize,
) -> Playout {
    let make_search = || {
        BeliefSearch::with_rollout(
            SearchConfig {
                iterations: config.iterations,
                ..SearchConfig::default()
            },
            CombatStrength::default(),
            Box::new(UniformRandom),
        )
        .selecting(Box::new(Gumbel {
            considered: config.considered,
            ..Gumbel::default()
        }))
        .with_net(Arc::clone(net) as Arc<dyn Evaluate>)
    };
    let root_choice = {
        let mut search = make_search();
        let mut rng = MegaRandom::new((fight as u64) << 40 ^ (decision as u64) << 20 ^ 0xF00);
        search.choose(root, &mut rng)
    };
    let mut totals = [(0.0_f64, 0.0_f64); 2];
    let mut lengths = [(0.0_f64, 0.0_f64); 2];
    for (line, first) in [end, attack].into_iter().enumerate() {
        for repeat in 0..config.repeats {
            let mut simulator = root.clone();
            if simulator.step_quietly(first).is_err() {
                continue;
            }
            let mut search = make_search();
            let mut rng = MegaRandom::new(
                (fight as u64) << 40 ^ (decision as u64) << 20 ^ (line as u64) << 10 ^ repeat,
            );
            let mut passes = 0.0;
            for _ in 0..2000 {
                if simulator.state().terminal.is_some() || fight_over(&simulator) {
                    break;
                }
                let action = search.choose(&simulator, &mut rng);
                if matches!(action, Action::EndTurn { .. })
                    && permitted_actions(&simulator).iter().any(|a| {
                        matches!(a, Action::PlayCard { card, .. }
                            if registry.card_type_of(&card.fingerprint) == Some(CardType::Attack))
                    })
                {
                    passes += 1.0;
                }
                if simulator.step_quietly(&action).is_err() {
                    break;
                }
            }
            lengths[line].0 += f64::from(turn_of(&simulator));
            lengths[line].1 += passes;
            totals[line].0 += objective.peek(&simulator);
            totals[line].1 += f64::from(u8::from(
                simulator.state().terminal != Some(sts2_engine::RunResult::Defeat)
                    && fight_over(&simulator),
            ));
        }
    }
    let n = config.repeats as f64;
    Playout {
        iterations: config.iterations,
        considered: config.considered,
        repeats: config.repeats,
        attack: label.to_owned(),
        pass_value: totals[0].0 / n,
        pass_won: totals[0].1 / n,
        attack_value: totals[1].0 / n,
        attack_won: totals[1].1 / n,
        pass_turns: lengths[0].0 / n,
        attack_turns: lengths[1].0 / n,
        pass_passes: lengths[0].1 / n,
        attack_passes: lengths[1].1 / n,
        root_passed: matches!(root_choice, Action::EndTurn { .. }),
        root_choice: short(&root_choice),
    }
}

fn turn_of(simulator: &Simulator) -> u32 {
    simulator
        .state()
        .combat
        .as_ref()
        .map_or(0, |combat| combat.player.turn)
}

/// What a search leaf would carry: the objective's score once the fight is
/// over, the value head's read otherwise.
fn leaf_value(simulator: &Simulator, net: &PolicyValueNet, objective: &CombatStrength) -> f64 {
    if simulator.state().terminal.is_some() || fight_over(simulator) {
        objective.peek(simulator)
    } else {
        net.state_value(simulator)
    }
}

fn describe(simulator: &Simulator) -> (i32, i32, i32, i32, u32, Vec<String>, i32, Vec<String>) {
    let state = simulator.state();
    let Some(combat) = state.combat.as_ref() else {
        return (
            state.run_player.current_hp,
            state.run_player.max_hp,
            0,
            0,
            0,
            Vec::new(),
            0,
            Vec::new(),
        );
    };
    let enemy_block = combat
        .creatures
        .iter()
        .filter(|creature| creature.id != combat.player.creature_id && creature.current_hp > 0)
        .map(|creature| creature.block)
        .sum();
    let enemy_powers = combat
        .creatures
        .iter()
        .filter(|creature| creature.id != combat.player.creature_id && creature.current_hp > 0)
        .flat_map(|creature| {
            creature.powers.iter().map(|power| {
                format!(
                    "{}={}",
                    power.model_id.to_string().trim_start_matches("POWER."),
                    power.amount
                )
            })
        })
        .collect();
    let (hp, max_hp) = combat
        .creatures
        .iter()
        .find(|creature| creature.id == combat.player.creature_id)
        .map_or((0, 0), |creature| (creature.current_hp, creature.max_hp));
    let enemy_hp = combat
        .creatures
        .iter()
        .filter(|creature| creature.id != combat.player.creature_id)
        .map(|creature| creature.current_hp.max(0))
        .sum();
    let hand = combat
        .player
        .piles
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
    (
        hp,
        max_hp,
        enemy_hp,
        combat.player.energy,
        combat.player.turn,
        hand,
        enemy_block,
        enemy_powers,
    )
}

/// The living enemies' move states and powers, in order: what the player
/// is shown about the coming enemy turn.
fn intents(simulator: &Simulator) -> Vec<String> {
    simulator
        .state()
        .combat
        .as_ref()
        .map_or(Vec::new(), |combat| {
            combat
                .creatures
                .iter()
                .filter(|creature| {
                    creature.id != combat.player.creature_id && creature.current_hp > 0
                })
                .map(|creature| {
                    let powers: Vec<String> = creature
                        .powers
                        .iter()
                        .map(|power| format!("{}={}", power.model_id, power.amount))
                        .collect();
                    format!("{:?}|{}", creature.move_state, powers.join(","))
                })
                .collect()
        })
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

fn short(action: &Action) -> String {
    match action {
        Action::EndTurn { .. } => "End turn".to_owned(),
        Action::PlayCard { card, target } => format!(
            "{}{}{}",
            card.fingerprint
                .model_id
                .to_string()
                .trim_start_matches("CARD."),
            if card.fingerprint.upgrade_level > 0 {
                "+"
            } else {
                ""
            },
            target
                .as_ref()
                .map_or(String::new(), |t| format!("@{:?}", t.combat_id))
        ),
        Action::UsePotion { model_id, .. } => {
            format!(
                "Drink {}",
                model_id.to_string().trim_start_matches("POTION.")
            )
        }
        other => serde_json::to_string(other).unwrap_or_else(|_| "?".to_owned()),
    }
}
