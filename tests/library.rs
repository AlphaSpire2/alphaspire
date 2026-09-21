//! The fight library's contracts: a banked entry stands back up as the very
//! belief it was banked from, the mix apportions exactly and selects
//! deterministically — `all` lanes covering a class whole — generation
//! writes byte-identical stamped shards at any job count, and a bank from
//! another model or build is refused.

use std::path::PathBuf;

use alphaspire::encoding::{PolicyEncoder, standard_vocabulary};
use alphaspire::library::{
    Class, FightLibrary, GenerationBatch, HarvestAs, LibrarySink, Mix, Source, Tier, harvest, plan,
    play_fights,
};
use alphaspire::objective::CombatStrength;
use alphaspire::policy::{RolloutPolicy, UniformRandom};
use alphaspire::search::{BeliefSearch, SearchConfig};
use alphaspire::training::SampleSink;
use sts2_core::UnlockPresetManifest;
use sts2_engine::{BeliefState, Simulator};

fn fresh_run(seed: &str) -> Simulator {
    let preset = UnlockPresetManifest::pinned().unwrap();
    let character = "CHARACTER.IRONCLAD".parse().unwrap();
    sts2_content::standard_run_on_preset_at(seed, &character, &preset, 0).unwrap()
}

/// The first legal action, preferring a non-empty card pick so a skippable
/// reward cannot re-offer itself.
fn next_action(simulator: &Simulator) -> sts2_engine::Action {
    simulator
        .legal_actions()
        .iter()
        .find(
            |action| matches!(action, sts2_engine::Action::ChooseCards { cards, .. } if !cards.is_empty()),
        )
        .or_else(|| simulator.legal_actions().first())
        .cloned()
        .expect("a live run offers an action")
}

/// The seeded run walked to its first in-combat decision.
fn first_combat(seed: &str) -> Simulator {
    let mut simulator = fresh_run(seed);
    for _ in 0..40 {
        if simulator.state().combat.is_some() {
            return simulator;
        }
        let action = next_action(&simulator);
        simulator.step_quietly(&action).unwrap();
    }
    panic!("the seed reaches a combat inside forty decisions");
}

/// A library of one banked entry — the fixture seed's first fight — written
/// to a fresh directory named for the test.
fn banked(name: &str) -> (PathBuf, Simulator) {
    let original = first_combat("NLD6VZXP94");
    let directory = std::env::temp_dir().join(format!("alphaspire-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let mut sink = LibrarySink::create(&directory, 2).unwrap();
    let mut fights = vec![harvest(&original, HarvestAs::Both).unwrap()];
    sink.write_run(&mut fights).unwrap();
    assert_eq!(
        (fights[0].meta.run, fights[0].meta.fight),
        (0, 0),
        "the sink stamps run and fight order"
    );
    let classes = sink.finish().unwrap();
    assert_eq!(
        classes,
        [1, 0, 0, 0, 0],
        "a run's first fight is an act-one hallway fight"
    );
    (directory, original)
}

#[test]
fn a_banked_entry_stands_back_up_as_the_belief_it_was() {
    let (directory, original) = banked("library-roundtrip");
    let library = FightLibrary::open(&directory).unwrap();
    assert_eq!(library.classes(), [1, 0, 0, 0, 0]);
    assert_eq!(library.entries(), 1);
    assert_eq!((library.with_setup(), library.with_state()), (1, 1));
    let loaded = library
        .load(&[(Class::Hallway, 0)], Source::State)
        .unwrap()
        .fights;
    let fight = &loaded[0];

    let state = original.state();
    let combat = state.combat.as_ref().unwrap();
    assert_eq!(fight.meta.encounter, combat.encounter_model);
    assert_eq!(fight.meta.seed, "NLD6VZXP94");
    assert_eq!(fight.meta.ascension, 0);
    assert_eq!(fight.meta.tier, Tier::Hallway);
    assert_eq!(fight.meta.deck, state.run_player.deck.len());
    assert_eq!(fight.meta.entry_hp, state.run_player.current_hp);
    assert_eq!(
        fight.meta.floor,
        state.run.as_ref().map_or(0, |run| run.floor)
    );

    // The disk round trip changes nothing: a world dealt from the loaded
    // entry is byte-for-byte the world the original's own erasure deals,
    // and it agrees with the original decision on everything visible.
    let fresh = BeliefState::from_simulator(&original).unwrap();
    let ours = fight.belief.sample(9, 1).unwrap();
    let theirs = fresh.sample(9, 1).unwrap();
    assert_eq!(
        ours.state_key().unwrap(),
        theirs.state_key().unwrap(),
        "erasure is a projection, and the file round trip preserves it"
    );
    assert_eq!(
        ours.agent_observation(),
        original.agent_observation(),
        "a loaded entry shows the player the very screen they stood on"
    );
    assert_eq!(
        ours.observation_key().unwrap(),
        original.observation_key().unwrap()
    );
    assert_eq!(ours.legal_actions(), original.legal_actions());

    // Stood up from the setup instead, the entry is the same loadout through
    // the same door on the same run seed — the same enemies at the same
    // health, the same deck, relics and hit points — under a hand the
    // engine's own combat start dealt rather than the one the run drew.
    let from_setup = library
        .load(&[(Class::Hallway, 0)], Source::Setup)
        .unwrap()
        .fights;
    let world = from_setup[0].belief.sample(9, 1).unwrap();
    let combat = world.state().combat.as_ref().unwrap();
    assert_eq!(
        combat.encounter_model,
        state.combat.as_ref().unwrap().encounter_model
    );
    let enemies = |simulator: &Simulator| -> Vec<(String, i32)> {
        simulator
            .state()
            .combat
            .as_ref()
            .unwrap()
            .creatures
            .iter()
            .filter(|creature| creature.side == sts2_engine::CombatSide::Enemy)
            .map(|creature| (creature.model_id.to_string(), creature.max_hp))
            .collect()
    };
    assert_eq!(enemies(&world), enemies(&original));
    assert_eq!(
        world.state().run_player.deck.len(),
        state.run_player.deck.len()
    );
    assert_eq!(
        world.state().run_player.current_hp,
        state.run_player.current_hp
    );
    assert_eq!(
        world
            .combat_setup()
            .map(|setup| (setup.deck, setup.relics, setup.potions)),
        original
            .combat_setup()
            .map(|setup| (setup.deck, setup.relics, setup.potions))
    );
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn a_training_bank_carries_setups_alone_and_says_so() {
    let original = first_combat("NLD6VZXP94");
    let directory =
        std::env::temp_dir().join(format!("alphaspire-library-setups-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let mut sink = LibrarySink::create(&directory, 2).unwrap();
    let mut fights = vec![harvest(&original, HarvestAs::Setup).unwrap()];
    assert!(fights[0].setup.is_some() && fights[0].state.is_none());
    sink.write_run(&mut fights).unwrap();
    sink.finish().unwrap();
    let line = std::fs::read_to_string(directory.join("entries-00000.jsonl")).unwrap();
    let entry = line.lines().nth(1).unwrap();
    assert!(
        entry.len() < 4_000,
        "a setup-only entry is a few hundred bytes, not tens of kilobytes: {}",
        entry.len()
    );
    assert!(
        !entry.contains("\"state\""),
        "an unsaid state is not written as null"
    );

    let library = FightLibrary::open(&directory).unwrap();
    assert_eq!((library.with_setup(), library.with_state()), (1, 0));
    let loaded = library
        .load(&[(Class::Hallway, 0)], Source::Setup)
        .unwrap()
        .fights;
    assert_eq!(
        loaded[0].meta.encounter,
        original.state().combat.as_ref().unwrap().encounter_model
    );
    // A consumer that needs the exact state is refused, entry named.
    let message = library
        .load(&[(Class::Hallway, 0)], Source::State)
        .err()
        .expect("a setup-only entry has no exact state to stand up")
        .to_string();
    assert!(
        message.contains("no exact state") && message.contains("NLD6VZXP94"),
        "{message}"
    );
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn an_entry_this_build_cannot_stand_up_costs_its_own_fight_and_no_other() {
    // A bank is harvested in bulk from real play and may hold a fight this
    // build cannot re-enter — an encounter it does not know. That entry is
    // one fight lost, and every other entry in the plan still plays. The
    // encounter here is named so as not to exist, rather than borrowing a
    // real gap: the contract under test is the skip, not the gap.
    let directory = std::env::temp_dir().join(format!(
        "alphaspire-library-unusable-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    let mut sink = LibrarySink::create(&directory, 2).unwrap();
    for seed in ["NLD6VZXP94", "GA08KNEDWE"] {
        let mut fights = vec![harvest(&first_combat(seed), HarvestAs::Setup).unwrap()];
        sink.write_run(&mut fights).unwrap();
    }
    sink.finish().unwrap();

    let shard = directory.join("entries-00000.jsonl");
    let text = std::fs::read_to_string(&shard).unwrap();
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let spoiled = lines[1].clone();
    let encounter = spoiled
        .split("\"encounter\":\"")
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .expect("a banked entry names its encounter")
        .to_owned();
    lines[1] = spoiled.replace(&encounter, "ENCOUNTER.NOT_AN_ENCOUNTER_THIS_BUILD_KNOWS");
    std::fs::write(&shard, lines.join("\n") + "\n").unwrap();

    let library = FightLibrary::open(&directory).unwrap();
    let batch = library
        .load(&[(Class::Hallway, 0), (Class::Hallway, 1)], Source::Setup)
        .expect("one unusable entry is not a refused batch");
    assert_eq!(
        batch.fights.len(),
        1,
        "the entry that stands up is still played"
    );
    assert_eq!(batch.unusable.len(), 1, "{:?}", batch.unusable);
    assert!(
        batch.unusable[0].contains("setup") && batch.unusable[0].contains("entry"),
        "the skipped entry is named: {}",
        batch.unusable[0]
    );
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn a_bank_of_the_previous_format_loads_as_state_only_entries_and_converts() {
    // A format-2 entry is `{meta, state}` and nothing else: written here by
    // hand off a state-only harvest, under the previous format's header and
    // with no count of setups or states on its manifest.
    let original = first_combat("NLD6VZXP94");
    let old = std::env::temp_dir().join(format!("alphaspire-library-old-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&old);
    let mut sink = LibrarySink::create(&old, 2).unwrap();
    let mut fights = vec![harvest(&original, HarvestAs::State).unwrap()];
    sink.write_run(&mut fights).unwrap();
    sink.finish().unwrap();
    for name in ["manifest.json", "entries-00000.jsonl"] {
        let path = old.join(name);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, text.replacen("\"format\":3", "\"format\":2", 1)).unwrap();
    }
    let manifest_path = old.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("with_setup");
    manifest.as_object_mut().unwrap().remove("with_state");
    std::fs::write(&manifest_path, format!("{manifest}\n")).unwrap();

    // It opens as it is: every entry a state, none a setup, and both halves
    // stand up — the state as the very fight, the setup by projecting it.
    let library =
        FightLibrary::open(&old).expect("the previous format's entries are this format's");
    assert_eq!((library.with_setup(), library.with_state()), (0, 1));
    let exact = library
        .load(&[(Class::Hallway, 0)], Source::State)
        .unwrap()
        .fights;
    assert_eq!(
        exact[0].belief.sample(3, 3).unwrap().state_key().unwrap(),
        BeliefState::from_simulator(&original)
            .unwrap()
            .sample(3, 3)
            .unwrap()
            .state_key()
            .unwrap()
    );
    let projected = library
        .load(&[(Class::Hallway, 0)], Source::Setup)
        .unwrap()
        .fights;
    assert_eq!(
        projected[0].meta.encounter,
        original.state().combat.as_ref().unwrap().encounter_model
    );

    let new = old.with_file_name(format!("alphaspire-library-new-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&new);
    let classes = alphaspire::library::convert(&old, &new, HarvestAs::Setup).unwrap();
    assert_eq!(classes, [1, 0, 0, 0, 0]);
    let library = FightLibrary::open(&new).unwrap();
    assert_eq!((library.with_setup(), library.with_state()), (1, 0));
    let loaded = library
        .load(&[(Class::Hallway, 0)], Source::Setup)
        .unwrap()
        .fights;
    assert_eq!(
        loaded[0].meta.encounter,
        original.state().combat.as_ref().unwrap().encounter_model
    );
    assert_eq!((loaded[0].meta.run, loaded[0].meta.fight), (0, 0));

    // Kept whole, the exact state rides beside the setup it projected to.
    let both = old.with_file_name(format!("alphaspire-library-both-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&both);
    alphaspire::library::convert(&old, &both, HarvestAs::Both).unwrap();
    let library = FightLibrary::open(&both).unwrap();
    assert_eq!((library.with_setup(), library.with_state()), (1, 1));
    let exact = library
        .load(&[(Class::Hallway, 0)], Source::State)
        .unwrap()
        .fights;
    assert_eq!(
        exact[0].belief.sample(3, 3).unwrap().state_key().unwrap(),
        BeliefState::from_simulator(&original)
            .unwrap()
            .sample(3, 3)
            .unwrap()
            .state_key()
            .unwrap()
    );
    for directory in [old, new, both] {
        std::fs::remove_dir_all(&directory).unwrap();
    }
}

#[test]
fn the_mix_apportions_exactly_and_the_schedule_is_deterministic() {
    let bank = [50, 20, 10, 8, 3];
    let mix = Mix::parse("boss=0.3,elite=0.3,hallway=0.4").unwrap();
    assert_eq!(
        mix.quotas(10, bank).unwrap(),
        [4, 3, 3, 0, 0],
        "exact shares apportion exactly"
    );
    assert_eq!(
        mix.quotas(7, bank).unwrap(),
        [3, 2, 2, 0, 0],
        "the largest remainder takes the leftover"
    );
    assert_eq!(
        Mix::parse("boss=1").unwrap().quotas(5, bank).unwrap(),
        [0, 0, 5, 0, 0],
        "an unnamed class gets nothing"
    );
    assert_eq!(
        Mix::parse("boss=all,act2=all,hallway=1")
            .unwrap()
            .quotas(30, bank)
            .unwrap(),
        [12, 0, 10, 8, 0],
        "the all lanes cover their banks whole and the weights split the rest"
    );
    assert_eq!(
        Mix::parse("act3=all,boss=all,hallway=1")
            .unwrap()
            .quotas(20, bank)
            .unwrap(),
        [7, 0, 10, 0, 3],
        "the act-three lane covers its bank whole like any other all lane"
    );
    for (mix, fights, classes, named) in [
        // An all lane over an empty class is a starved ask, not a no-op.
        ("act2=all", 5, [5, 0, 0, 0, 0], "act2"),
        ("act3=all", 5, [5, 0, 0, 2, 0], "act3"),
        // An all lane the fight count cannot hold.
        ("boss=all", 3, [0, 0, 5, 0, 0], "3 fights"),
        // Fights left over with no weighted lane to spend them.
        ("boss=all", 9, [0, 0, 5, 0, 0], "remaining 4 fights"),
    ] {
        let message = Mix::parse(mix)
            .unwrap()
            .quotas(fights, classes)
            .unwrap_err();
        assert!(
            message.to_string().contains(named),
            "{mix} refuses naming {named}: {message}"
        );
    }

    let mix = Mix::parse("boss=0.3,elite=0.3,hallway=0.4").unwrap();
    let held = [5, 2, 1, 0, 0];
    let schedule = plan(&mix, 10, held, 7).unwrap();
    assert_eq!(schedule.len(), 10);
    for class in Class::ALL {
        let of_class: Vec<usize> = schedule
            .iter()
            .filter(|(c, _)| *c == class)
            .map(|(_, index)| *index)
            .collect();
        assert_eq!(
            of_class.len(),
            mix.quotas(10, held).unwrap()[class.index()],
            "{class} plays its quota"
        );
        let held = held[class.index()];
        assert!(of_class.iter().all(|&index| index < held));
        // Cycling the shuffled bank keeps any entry's replay count within
        // one of any other's.
        for entry in 0..held {
            let replays = of_class.iter().filter(|&&index| index == entry).count();
            assert!(
                replays == of_class.len() / held || replays == of_class.len().div_ceil(held),
                "{class} entry {entry} replayed {replays} times"
            );
        }
    }
    assert_eq!(
        schedule,
        plan(&mix, 10, held, 7).unwrap(),
        "one seed, one schedule"
    );
    assert_ne!(
        schedule,
        plan(&mix, 10, held, 8).unwrap(),
        "another seed selects other entries"
    );

    let starved = plan(&Mix::parse("boss=1").unwrap(), 3, [4, 2, 0, 0, 0], 1);
    let message = starved.unwrap_err().to_string();
    assert!(
        message.contains("boss"),
        "an empty class refuses by name: {message}"
    );
}

#[test]
fn generation_writes_byte_identical_stamped_shards_at_any_job_count() {
    let (directory, _) = banked("library-generation");
    let library = FightLibrary::open(&directory).unwrap();
    let mix = Mix::parse("hallway=1").unwrap();
    let schedule = plan(&mix, 3, library.classes(), 5).unwrap();
    let fights = library.load(&schedule, Source::Setup).unwrap().fights;
    let registry = sts2_content::standard_registry();
    let encoder = PolicyEncoder::new(standard_vocabulary(&registry), &registry);

    let emit = |jobs: usize, out: &str| -> PathBuf {
        let samples = std::env::temp_dir().join(format!("alphaspire-{out}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&samples);
        let make_policy = || -> Box<dyn RolloutPolicy> {
            Box::new(
                BeliefSearch::with_rollout(
                    SearchConfig {
                        iterations: 4,
                        rollout_depth: 8,
                        temperature: 0.5,
                    },
                    CombatStrength::default(),
                    Box::new(UniformRandom),
                )
                .recording(),
            )
        };
        let mut sink = SampleSink::create(&samples, &encoder, 2, &[]).unwrap();
        play_fights(
            &GenerationBatch {
                fights: &fights,
                analysis_seed: 5,
                max_steps: 300,
                jobs,
            },
            &make_policy,
            &mut |index, played| {
                let mut report = played.unwrap();
                assert_eq!(report.meta.tier, Tier::Hallway);
                assert!(
                    !report.decisions.is_empty(),
                    "playout {index} searched decisions"
                );
                // A playout is one fight, so every sample carries the banked
                // encounter and fight zero; the sink stamps the playout
                // index as the run.
                assert!(report.decisions.iter().all(|sample| {
                    sample.encounter.as_ref() == Some(&report.meta.encounter) && sample.fight == 0
                }));
                sink.write_run(&mut report.decisions).unwrap();
                assert!(report.decisions.iter().all(|sample| sample.run == index));
            },
        );
        assert!(sink.finish().unwrap() > 0);
        samples
    };
    let serial = emit(1, "fights-serial");
    let parallel = emit(2, "fights-parallel");
    for name in [
        "manifest.json",
        "samples-00000.jsonl",
        "samples-00001.jsonl",
    ] {
        assert_eq!(
            std::fs::read(serial.join(name)).unwrap(),
            std::fs::read(parallel.join(name)).unwrap(),
            "{name} is a function of (library, mix, seed), not of --jobs"
        );
    }
    std::fs::remove_dir_all(&directory).unwrap();
    std::fs::remove_dir_all(&serial).unwrap();
    std::fs::remove_dir_all(&parallel).unwrap();
}

#[test]
fn a_bank_from_another_model_or_build_is_refused() {
    let (directory, _) = banked("library-refusal");
    let manifest_path = directory.join("manifest.json");
    let pristine = std::fs::read_to_string(&manifest_path).unwrap();

    let doctor = |field: &str, value: &str| {
        let mut manifest: serde_json::Value = serde_json::from_str(&pristine).unwrap();
        manifest[field] = serde_json::Value::String(value.to_owned());
        std::fs::write(&manifest_path, format!("{manifest}\n")).unwrap();
        let message = match FightLibrary::open(&directory) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a doctored {field} opened"),
        };
        assert!(
            message.contains(field),
            "the refusal names the field: {message}"
        );
    };
    doctor("belief_model_version", "combat-v9");
    doctor("compatibility_id", "sts2-v9.999.9-00000000-models-0");

    // A format-1 bank is refused by number, never reinterpreted.
    let mut manifest: serde_json::Value = serde_json::from_str(&pristine).unwrap();
    manifest["format"] = serde_json::json!(1);
    std::fs::write(&manifest_path, format!("{manifest}\n")).unwrap();
    let message = match FightLibrary::open(&directory) {
        Err(error) => error.to_string(),
        Ok(_) => panic!("a format-1 bank opened"),
    };
    assert!(message.contains("format"), "{message}");

    // The observation version describes without gating: an entry stores no
    // observation, so a bank written under another observation shape opens
    // and loads. The shard headers carry the old number too, so this also
    // proves the shard check reads only the enforced fields.
    let mut manifest: serde_json::Value = serde_json::from_str(&pristine).unwrap();
    manifest["observation_version"] = serde_json::json!(1);
    std::fs::write(&manifest_path, format!("{manifest}\n")).unwrap();
    let library =
        FightLibrary::open(&directory).expect("an observation bump does not orphan a bank");
    library.load(&[(Class::Hallway, 0)], Source::State).unwrap();

    // The pristine manifest still opens: the refusals above were the
    // doctored fields' own.
    std::fs::write(&manifest_path, &pristine).unwrap();
    FightLibrary::open(&directory).unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn the_bank_counts_how_its_fights_were_reached() {
    use alphaspire::library::FightOrigin;

    let original = first_combat("NLD6VZXP94");
    let directory =
        std::env::temp_dir().join(format!("alphaspire-library-origins-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let mut sink = LibrarySink::create(&directory, 2).unwrap();
    sink.annotate("force_wins", serde_json::json!({"min_hp": 1}));
    let mut entry = harvest(&original, HarvestAs::Both).unwrap();
    assert_eq!(
        entry.meta.origin,
        FightOrigin::Natural,
        "a harvest cannot know how the walk reached it"
    );
    entry.meta.origin = FightOrigin::ForcedWin;
    sink.write_run(&mut [entry]).unwrap();
    sink.finish().unwrap();

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(directory.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(
        manifest["origins"],
        serde_json::json!({"natural": 0, "forced_win": 1, "human_run": 0, "human_record": 0}),
        "the manifest counts the walk that fed the bank"
    );
    assert_eq!(
        manifest["force_wins"],
        serde_json::json!({"min_hp": 1}),
        "an annotation lands beside the counts"
    );

    let library = FightLibrary::open(&directory).unwrap();
    assert_eq!(library.origins(), Some([0, 1, 0, 0]));
    let loaded = library
        .load(&[(Class::Hallway, 0)], Source::State)
        .unwrap()
        .fights;
    assert_eq!(loaded[0].meta.origin, FightOrigin::ForcedWin);
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn the_bank_says_who_it_was_harvested_from() {
    // The composition is descriptive, not a pin: a heterogeneous bank is
    // legal, since every entry stands up against its own character's
    // registry, and the manifest is what makes the mixture visible. So a
    // bank written before the manifest carried it opens exactly as before
    // and answers that it does not say.
    let (directory, _) = banked("library-characters");
    let manifest_path = directory.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert_eq!(
        manifest["characters"],
        serde_json::json!(["CHARACTER.IRONCLAD"])
    );
    assert_eq!(
        FightLibrary::open(&directory).unwrap().characters(),
        Some(["CHARACTER.IRONCLAD".parse().unwrap()].as_slice())
    );

    manifest.as_object_mut().unwrap().remove("characters");
    std::fs::write(&manifest_path, format!("{manifest}\n")).unwrap();
    let library = FightLibrary::open(&directory).expect("an unstamped bank still opens");
    assert!(library.characters().is_none());
    assert_eq!(library.entries(), 1, "and still holds what it held");
    library.load(&[(Class::Hallway, 0)], Source::State).unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
}

/// A finished playout, as the tally reads one: everything else on a report
/// is what the tally ignores.
fn played(
    encounter: &str,
    tier: Tier,
    act: usize,
    floor: u32,
    character: &str,
    outcome: alphaspire::library::FightOutcome,
    value: f64,
) -> alphaspire::library::FightReport {
    alphaspire::library::FightReport {
        meta: alphaspire::library::FightMeta {
            encounter: encounter.parse().unwrap(),
            tier,
            act,
            floor,
            character: character.parse().unwrap(),
            ascension: 0,
            seed: "NLD6VZXP94".into(),
            run: 0,
            fight: 0,
            entry_hp: 60,
            deck: 10,
            origin: alphaspire::library::FightOrigin::Natural,
        },
        outcome,
        value,
        decisions: Vec::new(),
    }
}

#[test]
fn the_fight_tally_counts_wins_and_names_the_costliest_encounters() {
    use alphaspire::library::FightOutcome::{Lost, Unfinished, Won};

    let mut tally = alphaspire::library::FightTally::default();
    for (encounter, tier, act, floor, character, outcome, value) in [
        (
            "ENCOUNTER.A",
            Tier::Hallway,
            0,
            4,
            "CHARACTER.SILENT",
            Won,
            2.0,
        ),
        (
            "ENCOUNTER.B",
            Tier::Boss,
            0,
            9,
            "CHARACTER.SILENT",
            Lost,
            0.0,
        ),
        (
            "ENCOUNTER.B",
            Tier::Boss,
            0,
            12,
            "CHARACTER.SILENT",
            Lost,
            0.0,
        ),
        (
            "ENCOUNTER.C",
            Tier::Elite,
            1,
            6,
            "CHARACTER.REGENT",
            Lost,
            1.0,
        ),
        (
            "ENCOUNTER.D",
            Tier::Hallway,
            1,
            7,
            "CHARACTER.REGENT",
            Unfinished,
            0.5,
        ),
    ] {
        tally.record(&played(
            encounter, tier, act, floor, character, outcome, value,
        ));
    }

    let summary = tally.summary();
    assert!(
        summary.contains("fights won 1/5 (20.0%)") && summary.contains("mean value 0.70"),
        "{summary}"
    );
    assert!(
        summary.contains("3 lost") && summary.contains("1 unfinished"),
        "{summary}"
    );

    // Costliest first, and the floor span of the rows that were paid more
    // than once.
    let table = tally.loss_table(10);
    assert_eq!(
        table,
        [
            "   2  ENCOUNTER.B (boss, act 0, floors 9-12)",
            "   1  ENCOUNTER.C (elite, act 1, floor 6)",
        ]
    );
    // Cut to one row, and the remainder line says what the cut hid.
    let cut = tally.loss_table(1);
    assert_eq!(cut.len(), 2, "{cut:?}");
    assert_eq!(cut[1], "      … and 1 losses over 1 more rows");

    let composition = tally.composition();
    assert_eq!(composition.len(), 2, "a bank may hold several characters");
    assert_eq!(composition[&"CHARACTER.SILENT".parse().unwrap()], 3);
    assert_eq!(composition[&"CHARACTER.REGENT".parse().unwrap()], 2);
}
