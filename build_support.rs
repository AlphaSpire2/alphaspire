//! Build-time provenance, also exercised by integration tests.

use std::{path::Path, process::Command};

use serde_json::Value;

fn git(directory: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// Report the checkout actually compiled, with explicit unknowns for archives
/// or machines without Git. Never mistake an enclosing repository for it.
pub fn revision(directory: &Path, source: Option<&str>) -> String {
    if git(directory, &["ls-files", "--error-unmatch", "Cargo.toml"]).is_some()
        && let Some(head) = git(directory, &["rev-parse", "HEAD"])
    {
        return match git(
            directory,
            &["status", "--porcelain", "--untracked-files=normal"],
        ) {
            Some(status)
                if status.lines().all(|line| {
                    // Cargo writes this checkout marker after fetching Git
                    // dependencies. It is not a source modification.
                    line == "?? .cargo-ok"
                        && source.is_some_and(|source| source.starts_with("git+"))
                }) =>
            {
                head
            }
            Some(_) => format!("{head} dirty"),
            None => format!("{head}; dirty unknown"),
        };
    }
    let revision = source
        .filter(|source| source.starts_with("git+"))
        .and_then(|source| source.rsplit_once('#'))
        .map_or("unknown", |(_, revision)| revision);
    format!("{revision}; dirty unknown")
}

/// Read only the simulator crates linked directly into Alphaspire, rather
/// than selecting an arbitrary package from Cargo's complete package list.
pub fn simulator_identity(metadata: &Value) -> Result<String, String> {
    let packages = metadata["packages"].as_array().ok_or("missing packages")?;
    let root = metadata["resolve"]["root"]
        .as_str()
        .ok_or("missing root package")?;
    let node = metadata["resolve"]["nodes"]
        .as_array()
        .ok_or("missing resolved nodes")?
        .iter()
        .find(|node| node["id"] == root)
        .ok_or("missing root node")?;
    let mut identity = None;
    for dependency in node["deps"].as_array().ok_or("missing root dependencies")? {
        let package = packages
            .iter()
            .find(|package| package["id"] == dependency["pkg"])
            .ok_or("missing resolved package")?;
        if !package["name"]
            .as_str()
            .is_some_and(|name| name.starts_with("sts2-"))
        {
            continue;
        }
        let version = package["version"]
            .as_str()
            .ok_or("missing simulator version")?;
        let manifest = package["manifest_path"]
            .as_str()
            .ok_or("missing simulator manifest")?;
        let directory = Path::new(manifest)
            .parent()
            .ok_or("missing simulator directory")?;
        let revision = revision(directory, package["source"].as_str());
        let current = format!("{version} ({revision})");
        if let Some(previous) = &identity {
            if previous != &current {
                return Err(format!(
                    "simulator crates have inconsistent identities: {previous} and {current}"
                ));
            }
        } else {
            identity = Some(current);
        }
    }
    identity.ok_or_else(|| "no resolved simulator dependencies".to_owned())
}
