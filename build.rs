mod build_support;

use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo manifest directory"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"));

    // This intentionally absent file makes Cargo refresh provenance on every
    // build, including new untracked files and edits to a patched dependency.
    // Watching HEAD or the index alone misses unstaged source changes.
    println!(
        "cargo:rerun-if-changed={}",
        out.join("refresh-provenance").display()
    );

    // Metadata resolves the actual dependency, including Cargo path patches.
    // Offline/locked prevents this nested query from fetching or changing the
    // dependency graph. The outer Cargo invocation has already resolved it.
    let output = Command::new(env::var_os("CARGO").expect("Cargo executable"))
        .current_dir(&root)
        .args([
            "metadata",
            "--offline",
            "--locked",
            "--format-version",
            "1",
            "--filter-platform",
        ])
        .arg(env::var("TARGET").expect("Cargo target"))
        .output()
        .expect("run cargo metadata for build provenance");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata = serde_json::from_slice(&output.stdout).expect("Cargo metadata JSON");
    let simulator =
        build_support::simulator_identity(&metadata).expect("resolved simulator identity");
    let own = build_support::revision(&root, None);
    let version = format!(
        "{} ({own})\nsts2sim {simulator}",
        env::var("CARGO_PKG_VERSION").expect("package version")
    );
    fs::write(out.join("version.txt"), version).expect("write build provenance");
}
