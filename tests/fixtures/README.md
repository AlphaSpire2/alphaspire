# Regression checkpoint

`tiny.onnx` and `tiny.json` are a small combat checkpoint used to exercise
model loading, inference, search, and sample generation. This is a regression
fixture, not a recommended playing policy or a benchmark of playing strength.
The fixture is covered by the repository's AGPL-3.0-or-later license.

The recorded provenance in `tiny.json` describes 1,110 samples from a v8 smoke
batch of 12 heuristic-macro runs at 24 search iterations, two training epochs,
seed 1, embedding width 4, and hidden width 8. Its trainer revision is
`1eeda403f16a2898f30d73ce6c1c8b1adbcd7d16`. The current metadata names encoding
and observation version 9. The character field is intentionally absent to
test compatibility with checkpoints written before character provenance.

The training samples and a complete regeneration recipe are not bundled.
Keep the model and metadata together when updating the fixture and run the
full test suite after replacing them.

SHA-256 of `tiny.onnx`:
`cb4ae34368e4a0f5e3a991f206ae14a89044f20d69806b0cfb8594e5331e920b`.
