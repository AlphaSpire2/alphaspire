//! The value head's response to enemy HP alone: at a banked fight's
//! decision, the turn is ended in one dealt world, and every enemy HP from
//! its current value down to 1 is written into that position and read by
//! each checkpoint. Nothing else in the state changes, so the curve is the
//! head's own shape along one input.
//!
//! ```text
//! hp_sweep <library> <fight> <decision> <net-stem>...
//! ```
//!
//! `fight` and `decision` are the blind-spot scan's row indices under the
//! same `SCAN_MIX` / `SCAN_SEED` (defaults `elite=all,boss=all`, 1); the
//! walk to the decision is the scan's greedy walk under the first
//! checkpoint. `SWEEP_STEP` (default 1) is the HP stride.
//!
//! A diagnostic, not a product: exempt from the crate's pedantic lints.
#![allow(clippy::pedantic, clippy::too_many_lines, clippy::cast_precision_loss)]

use std::path::Path;
use std::sync::Arc;

use alphaspire::encoding::PolicyEncoder;
use alphaspire::env::{Belief, Determinizer, fight_over};
use alphaspire::library::{Class, FightLibrary, Mix, plan};
use alphaspire::net::{Evaluate, PolicyValueNet};
use alphaspire::policy::permitted_actions;
use sts2_engine::{Action, Simulator};

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: hp_sweep <library> <fight> <decision> <net-stem>...");
        std::process::exit(2);
    }
    let mix_text = std::env::var("SCAN_MIX").unwrap_or_else(|_| "elite=all,boss=all".to_owned());
    let seed = env_or("SCAN_SEED", 1);
    let step = env_or("SWEEP_STEP", 1).max(1) as i32;
    let fight: usize = args[2].parse().expect("a fight index");
    let decision: usize = args[3].parse().expect("a decision index");

    let library = FightLibrary::open(Path::new(&args[1])).unwrap_or_else(|error| panic!("{error}"));
    let mix = Mix::parse(&mix_text).unwrap_or_else(|error| panic!("{error}"));
    let fights: usize = mix_text
        .split(',')
        .filter_map(|term| term.split_once('='))
        .filter(|(_, share)| *share == "all")
        .filter_map(|(class, _)| class.parse::<Class>().ok())
        .map(|class| library.classes()[class.index()])
        .sum();
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
    let nets: Vec<(String, PolicyValueNet)> = args[4..]
        .iter()
        .map(|stem| {
            let net = PolicyValueNet::load(Path::new(stem), Arc::clone(&encoder))
                .unwrap_or_else(|error| panic!("{stem}: {error}"));
            let name = Path::new(stem)
                .file_name()
                .map_or(stem.clone(), |n| n.to_string_lossy().into_owned());
            (name, net)
        })
        .collect();

    // The scan's walk to the decision, under the first checkpoint's greedy policy.
    let entry = &loaded[fight];
    let mut simulator = entry
        .belief
        .sample(seed, fight as u64)
        .expect("a banked entry samples");
    for _ in 0..decision {
        let actions = permitted_actions(&simulator).into_owned();
        let (priors, _, _) = nets[0]
            .1
            .priors_and_value(&simulator, &actions)
            .into_parts();
        let choice = priors
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        simulator
            .step_quietly(&actions[choice])
            .expect("the scan's walk replays");
    }
    let end = permitted_actions(&simulator)
        .iter()
        .find(|action| matches!(action, Action::EndTurn { .. }))
        .cloned()
        .expect("the turn can be ended");
    let mut belief = Belief::from_simulator(&simulator, seed).expect("a fight samples");
    let mut world = belief.sample(0);
    world.step_quietly(&end).expect("the pass steps");
    if world.state().terminal.is_some() || fight_over(&world) {
        println!("the fight ended on the pass in world 0; nothing to sweep");
        return;
    }
    println!(
        "== {} fight {fight} decision {decision}: after the pass, {}",
        entry.meta.encounter,
        describe(&world)
    );
    let combat = world.state().combat.as_ref().expect("in combat");
    let enemies: Vec<(usize, i32)> = combat
        .creatures
        .iter()
        .enumerate()
        .filter(|(_, creature)| creature.id != combat.player.creature_id && creature.current_hp > 0)
        .map(|(index, creature)| (index, creature.current_hp))
        .collect();
    let names: Vec<String> = nets.iter().map(|(name, _)| name.clone()).collect();
    println!("enemy_hp_total\t{}", names.join("\t"));
    let total: i32 = enemies.iter().map(|(_, hp)| hp).sum();
    // Lower every living enemy in proportion, one HP of total at a time.
    let mut removed = 0;
    while removed < total - enemies.len() as i32 {
        let mut state = world.state().clone();
        let combat = state.combat.as_mut().expect("in combat");
        let mut left = removed;
        for (index, hp) in &enemies {
            let take = left.min(hp - 1);
            combat.creatures[*index].current_hp = hp - take;
            left -= take;
        }
        let sim = Simulator::from_scenario(state, Arc::clone(&registry)).expect("a state rebuilds");
        let values: Vec<String> = nets
            .iter()
            .map(|(_, net)| format!("{:.4}", net.state_value(&sim)))
            .collect();
        println!("{}\t{}", total - removed, values.join("\t"));
        removed += step;
    }
}

fn describe(simulator: &Simulator) -> String {
    let state = simulator.state();
    let Some(combat) = state.combat.as_ref() else {
        return "out of combat".to_owned();
    };
    let mut text = String::new();
    for creature in &combat.creatures {
        if creature.id == combat.player.creature_id {
            text.push_str(&format!(
                "turn {} hp {}/{} E {} | ",
                combat.player.turn, creature.current_hp, creature.max_hp, combat.player.energy
            ));
        } else if creature.current_hp > 0 {
            text.push_str(&format!(
                "{} {}/{} {} ",
                creature.model_id.to_string().trim_start_matches("MONSTER."),
                creature.current_hp,
                creature.max_hp,
                creature.move_state
            ));
        }
    }
    text
}
