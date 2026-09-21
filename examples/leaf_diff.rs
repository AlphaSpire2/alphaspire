//! What differs between the two leaves a belief search weighs when it
//! considers passing: the position after ending the turn, and the position
//! after playing one attack and then ending it, in the same dealt world.
//! Prints every observation field that differs and each checkpoint's value
//! at both leaves.
//!
//! ```text
//! leaf_diff <library> <fight> <decision> <attack-substring> <net-stem>...
//! ```
//!
//! Indices are the blind-spot scan's under the same `SCAN_MIX` /
//! `SCAN_SEED`; the walk is the scan's greedy walk under the first
//! checkpoint. `SCAN_WORLD` (default 0) picks the dealt world.
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
use sts2_engine::Action;

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!("usage: leaf_diff <library> <fight> <decision> <attack-substring> <net-stem>...");
        std::process::exit(2);
    }
    let mix_text = std::env::var("SCAN_MIX").unwrap_or_else(|_| "elite=all,boss=all".to_owned());
    let seed = env_or("SCAN_SEED", 1);
    let world_index = env_or("SCAN_WORLD", 0);
    let fight: usize = args[2].parse().expect("a fight index");
    let decision: usize = args[3].parse().expect("a decision index");
    let matcher = &args[4];

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
    let nets: Vec<(String, PolicyValueNet)> = args[5..]
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
    let legal = permitted_actions(&simulator).into_owned();
    let end = legal
        .iter()
        .find(|action| matches!(action, Action::EndTurn { .. }))
        .cloned()
        .expect("the turn can be ended");
    let attack = legal
        .iter()
        .find(|action| {
            matches!(action, Action::PlayCard { .. })
                && serde_json::to_string(action)
                    .expect("serializes")
                    .contains(matcher.as_str())
        })
        .cloned()
        .expect("an attack matches");
    let mut belief = Belief::from_simulator(&simulator, seed).expect("a fight samples");
    let dealt = belief.sample(world_index);

    let mut pass = dealt.clone();
    pass.step_quietly(&end).expect("the pass steps");
    let mut hit = dealt.clone();
    hit.step_quietly(&attack).expect("the attack steps");
    let after = permitted_actions(&hit)
        .iter()
        .find(|action| matches!(action, Action::EndTurn { .. }))
        .cloned()
        .expect("the turn can still be ended");
    hit.step_quietly(&after)
        .expect("the pass after the attack steps");
    for (label, sim) in [("pass", &pass), ("attack, pass", &hit)] {
        let over = sim.state().terminal.is_some() || fight_over(sim);
        let values: Vec<String> = nets
            .iter()
            .map(|(name, net)| format!("{name}={:.4}", net.state_value(sim)))
            .collect();
        println!("== {label}: fight over {over}; {}", values.join(" "));
    }
    let a = serde_json::to_value(pass.agent_observation()).expect("serializes");
    let b = serde_json::to_value(hit.agent_observation()).expect("serializes");
    println!("== observation fields that differ (pass -> attack,pass):");
    diff("", &a, &b);
}

fn diff(path: &str, a: &serde_json::Value, b: &serde_json::Value) {
    use serde_json::Value;
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
            keys.sort();
            keys.dedup();
            for key in keys {
                let sub = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                diff(
                    &sub,
                    x.get(key).unwrap_or(&Value::Null),
                    y.get(key).unwrap_or(&Value::Null),
                );
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (index, (p, q)) in x.iter().zip(y).enumerate() {
                diff(&format!("{path}[{index}]"), p, q);
            }
        }
        _ => {
            if a != b {
                let show = |v: &Value| {
                    let text = v.to_string();
                    if text.len() > 160 {
                        format!("{}…({} chars)", &text[..160], text.len())
                    } else {
                        text
                    }
                };
                println!("   {path}: {} -> {}", show(a), show(b));
            }
        }
    }
}
