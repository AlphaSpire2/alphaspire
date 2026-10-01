#[path = "../build_support.rs"]
mod build_support;

use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Value, json};

struct Checkout(PathBuf);

impl Checkout {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "alphaspire-provenance-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        let checkout = Self(path);
        checkout.git(&["init", "--quiet"]);
        fs::write(checkout.0.join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::write(checkout.0.join("source.txt"), "original\n").unwrap();
        checkout.git(&["add", "."]);
        checkout.git(&["commit", "--quiet", "-m", "fixture"]);
        checkout
    }

    fn git(&self, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.0)
            .args([
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgSign=false",
                "-c",
            ])
            .arg(format!(
                "core.hooksPath={}",
                self.0.join("no-hooks").display()
            ))
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }
}

impl Drop for Checkout {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn metadata(checkout: &Checkout) -> Value {
    json!({
        "packages": [
            {"id": "sim", "name": "sts2-core", "version": "0.7.0", "source": null,
             "manifest_path": checkout.0.join("Cargo.toml")},
            // An unrelated version in the graph must not win over the root's dependency.
            {"id": "other", "name": "sts2-core", "version": "0.1.0", "source": null,
             "manifest_path": "unused/Cargo.toml"}
        ],
        "resolve": {"root": "app", "nodes": [{"id": "app", "deps": [{"pkg": "sim"}]}]}
    })
}

#[test]
fn resolved_path_patch_reports_its_version_and_actual_checkout() {
    let checkout = Checkout::new();
    let head = checkout.git(&["rev-parse", "HEAD"]);
    assert_eq!(
        build_support::simulator_identity(&metadata(&checkout)).unwrap(),
        format!("0.7.0 ({head})")
    );
}

#[test]
fn git_sources_report_the_checkout_and_detect_unstaged_staged_and_untracked_changes() {
    let checkout = Checkout::new();
    let head = checkout.git(&["rev-parse", "HEAD"]);
    let source = format!("git+https://github.com/AlphaSpire2/sts2sim?rev={head}#{head}");
    assert_eq!(build_support::revision(&checkout.0, Some(&source)), head);
    fs::write(checkout.0.join(".cargo-ok"), "").unwrap();
    assert_eq!(build_support::revision(&checkout.0, Some(&source)), head);
    assert_eq!(
        build_support::revision(&checkout.0, None),
        format!("{head} dirty")
    );
    fs::write(checkout.0.join("source.txt"), "modified\n").unwrap();
    assert_eq!(
        build_support::revision(&checkout.0, Some(&source)),
        format!("{head} dirty")
    );
    checkout.git(&["add", "source.txt"]);
    assert_eq!(
        build_support::revision(&checkout.0, Some(&source)),
        format!("{head} dirty")
    );
    checkout.git(&["commit", "--quiet", "-m", "change"]);
    let head = checkout.git(&["rev-parse", "HEAD"]);
    fs::write(checkout.0.join("new-source.txt"), "new\n").unwrap();
    assert_eq!(
        build_support::revision(&checkout.0, Some(&source)),
        format!("{head} dirty")
    );
}

#[test]
fn archives_do_not_borrow_the_enclosing_repository_identity() {
    let checkout = Checkout::new();
    let archive = checkout.0.join("archive");
    fs::create_dir(&archive).unwrap();
    fs::write(archive.join("Cargo.toml"), "[workspace]\n").unwrap();
    assert_eq!(
        build_support::revision(&archive, None),
        "unknown; dirty unknown"
    );
    assert_eq!(
        build_support::revision(
            &archive,
            Some("git+https://github.com/AlphaSpire2/sts2sim#abc123")
        ),
        "abc123; dirty unknown"
    );
}

#[test]
fn mixed_simulator_versions_are_rejected_instead_of_mislabeled() {
    let checkout = Checkout::new();
    let mut metadata = metadata(&checkout);
    metadata["resolve"]["nodes"][0]["deps"]
        .as_array_mut()
        .unwrap()
        .push(json!({"pkg": "other"}));
    assert!(
        build_support::simulator_identity(&metadata)
            .unwrap_err()
            .contains("inconsistent identities")
    );
}
