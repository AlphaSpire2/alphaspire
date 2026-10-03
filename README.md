# alphaspire

AlphaZero inspired analysis engine for Slay the Spire 2, built on
sts2sim. Alphaspire can generate or analyze
[sts2pgn](https://steamcommunity.com/sharedfiles/filedetails/?id=3790144369) files.
Compatible with patch `v0.107.1` single-player, ascension 0-10, with no
gameplay-affecting mods.

## Download a release

The [GitHub releases](https://github.com/AlphaSpire2/alphaspire/releases) provide
platform archives with the executable. Models are separate ZIP downloads and
can contain checkpoints for multiple characters and policy types. Extract the
binary archive and follow its included README. To use trained policies, extract
a model ZIP separately, keeping each checkpoint's JSON and ONNX files together.
Running a packaged release does not require Rust or a simulator checkout.

## Contributing

The best ways to contribute are to
[open an issue](https://github.com/AlphaSpire2/alphaspire/issues) with a bug report
or suggestion, and submit your sts2pgn replays to
[alphaspire.dev](https://alphaspire.dev/).

## Usage

Pass checkpoint paths without extensions. Each checkpoint needs its `.json` and
`.onnx` files together.

### Generate runs

```sh
alphaspire run --runs 10 --jobs 4 --character ironclad --ascension 1 --greedy \
  --run-net ckpts/ironclad-macro-example \
  --combat-net ckpts/ironclad-combat-example \
  --resolver-iterations 64 \
  --out runs --summary runs/summary.json
```

The macro checkpoint makes decisions outside combat; the combat checkpoint
guides the in-fight search. `--out` saves each run as a `.sts2pgn` decision
script.

### Resolver iterations

`--resolver-iterations N` sets the search iterations per combat
decision, you can think of it like depth in a chess engine.
For example, add `--resolver-iterations 128` to search longer;
`--resolver-considered` controls how many candidate actions
share that budget (default: 16).

| Iterations | Use case |
| ---------- | -------- |
| 16         | macro training where speed is important       |
| 64         | model benchmarking and general run generation |
| 128-1028   | deep run analysis or max effort benchmark     |

### Analyze a run

Pass a recorded or generated `.sts2pgn` trace. Replace
`runs/your-run.sts2pgn` with the path to your trace:

```sh
alphaspire analyze runs/your-run.sts2pgn \
  --combat-net ckpts/ironclad-combat-example \
  --run-net ckpts/ironclad-macro-example \
  --out analysis
```

This writes `<trace>.analysis.json` with the run summary and
`<trace>.analysis.jsonl` with one line per decision. Omit `--run-net` to
evaluate combat decisions only. For a quick replay summary without models,
use:

```sh
alphaspire analyze runs/your-run.sts2pgn --no-net --out analysis
```

That writes `<trace>.summary.json`. See `alphaspire run --help` and
`alphaspire analyze --help` for search budgets, seeds, and other options.

## License

Alphaspire is licensed under AGPL-3.0-or-later. See [LICENSE](LICENSE).
