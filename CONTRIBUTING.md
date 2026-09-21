# Contributing

Alphaspire's simulator dependency, `sts2sim`, is private. Building, running,
and testing Alphaspire requires authorized access to it. The public workflow
checks formatting only and does not grant access to simulator source.

Documentation changes and issue reports are welcome without simulator access.
For bugs, include the Alphaspire revision, Rust version, operating system,
command, seed, expected result, and actual result. Remove credentials, local
paths, and private recordings or datasets before posting. A short reproducer
is more useful than a complete training run.

## Development

Use Rust 1.88 or newer. Configure Git authentication for the private dependency
outside this repository; never put credentials in Cargo files or Git URLs.
Cargo can use an existing Git credential setup with
`CARGO_NET_GIT_FETCH_WITH_CLI=true`.

When developing against a sibling simulator checkout, copy
`.cargo/config.example.toml` to `.cargo/config.toml` and adjust the paths.
The local config is ignored. Release verification must use the pinned Git
dependency without this override; do not commit a lockfile regenerated with
local simulator paths. Local overrides may require omitting `--locked` while
developing; restore the committed lockfile before preparing a pull request.

Before submitting a code change, run these checks with simulator access:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets --all-features
cargo doc --locked --no-deps
```

Check compatibility with the minimum supported toolchain as well as stable.
State which checks you ran and explain any unavailable checks in the pull
request. Keep changes focused and add regression coverage for behavior changes.
Keep generated datasets, recordings, checkpoints, and build outputs out of
pull requests; the small checked-in test fixtures are intentional.

For dependency changes, also run `cargo audit` and
`cargo deny --locked check licenses` with those tools installed. `deny.toml`
records the accepted dependency licenses. Review newly introduced licenses
before changing that list.

Contributions are made under the repository's AGPL-3.0-or-later license.
