//! The learned prior: an ONNX policy/value checkpoint behind the search.
//!
//! A trainer exports `<base>.onnx`
//! beside `<base>.json`, the provenance file naming every version the
//! checkpoint depends on. Loading verifies all of them against this build —
//! a checkpoint trained on another encoding, another observation shape, or
//! another vocabulary is refused, never reinterpreted. The one thing it
//! reads without enforcing is which characters trained it, which is a
//! warning rather than a refusal because loading across characters is how a
//! new character is bootstrapped.
//!
//! Inference is tract, pure Rust, CPU, deterministic: the same checkpoint
//! and the same position always price the same priors, which keeps a
//! net-guided search reproducible from (run seed, analysis seed, config,
//! checkpoint).
//!
//! Two things happen to the graph between the file and the forward pass,
//! both exact. The plan is built with the token and action axes symbolic
//! and runs at the shape a position actually has — the tokens it filled
//! and the actions it offered — instead of the padded ceiling the trainer
//! exports at; the net masks padding out of every reduction, so the answer
//! is the same function of the position either way. And every embedding
//! lookup (a `Gather` of a constant table by a graph input) is lifted out
//! of the graph and done here, because copying rows out of a table is
//! something this side does in nanoseconds and tract's general-purpose
//! indexing does in most of the forward's time. The lifted lookups feed
//! the plan through extra float inputs appended after the six the trainer
//! declares.

use std::path::Path;
use std::sync::Arc;

use sts2_engine::Simulator;
use tract_onnx::pb;
use tract_onnx::prelude::tract_data::internal::{anyhow, bail};
use tract_onnx::prelude::*;

use crate::encoding::{
    ACTION_FEATURES, ACTION_TOKENS, ActionEncoding, OBSERVATION_SCALARS, POLICY_ENCODING_VERSION,
    PolicyEncoder, TOKEN_FEATURES,
};

/// The most actions one evaluation prices: the width the checkpoint was
/// trained against. The plan runs at the width a decision actually has, so
/// nothing here pads to it, but a decision offering more is refused loudly
/// rather than priced by a net that never saw one.
pub const MAX_ACTIONS: usize = 64;

/// Production inference evaluates one position per call. Workers call the
/// shared evaluator directly without waiting for a cross-worker batch.
pub const INFER_BATCH: usize = 1;

/// Why priors are not the checkpoint's own work. One cause per pricing, and
/// they are ordered by how much they cost: an over-axis decision loses the
/// checkpoint's ranking, a non-finite row means the checkpoint is broken.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Degraded {
    /// The decision offered more actions than [`MAX_ACTIONS`], so the whole
    /// list is priced uniform off the value head alone.
    OverAxis,
    /// The checkpoint answered this position with something that was not a
    /// number; `settled` read it as zero, which softmaxes to uniform.
    NonFinite,
    /// The evaluator holds no checkpoint that prices this kind of state and
    /// answers uniform rather than inventing a ranking.
    OutOfScope,
}

/// A priced decision: priors aligned onto the actions asked about, the
/// state's value, and what stood in for the checkpoint where anything did.
///
/// The three come apart together and only together
/// ([`Priced::into_parts`]): there is no way to read the priors without being
/// handed their status. A caller cannot re-derive that status for itself —
/// what degrades a pricing differs by evaluator, and only the evaluator knows
/// which of its own conditions it hit.
#[derive(Clone, Debug)]
pub struct Priced {
    priors: Vec<f32>,
    value: f64,
    degraded: Option<Degraded>,
}

impl Priced {
    /// The checkpoint's own priors and value.
    #[must_use]
    pub const fn real(priors: Vec<f32>, value: f64) -> Self {
        Self {
            priors,
            value,
            degraded: None,
        }
    }

    /// Priors standing in for a checkpoint that did not price the decision,
    /// under the cause that made them necessary.
    #[must_use]
    pub const fn fallback(priors: Vec<f32>, value: f64, cause: Degraded) -> Self {
        Self {
            priors,
            value,
            degraded: Some(cause),
        }
    }

    /// The priors, the value, and the cause where the priors are a fallback.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f32>, f64, Option<Degraded>) {
        (self.priors, self.value, self.degraded)
    }

    /// What stood in for the checkpoint, where anything did.
    #[must_use]
    pub const fn degraded(&self) -> Option<Degraded> {
        self.degraded
    }
}

/// What the search asks of a checkpoint: priors over a decision's canonical
/// actions with the state's value, or the value alone. `PolicyValueNet`
/// answers directly; an inference server answers by batching many askers
/// into one forward pass. The search cannot tell the difference, which is
/// the point.
pub trait Evaluate: Send + Sync {
    /// Prior probabilities over `actions` and the state's value, priced off
    /// what the player sees, with whether they are the checkpoint's own.
    fn priors_and_value(&self, simulator: &Simulator, actions: &[sts2_engine::Action]) -> Priced;

    /// Prior probabilities over `plans` and the state's value.
    ///
    /// The default prices a plan by what it acquires, which is all an
    /// evaluator with no plan block to read can say about one. A checkpoint
    /// trained at [`POLICY_ENCODING_VERSION`] 7 or later reads the plan's own
    /// block and overrides it.
    fn plan_priors_and_value(
        &self,
        simulator: &Simulator,
        plans: &[crate::plan::ActionPlan],
    ) -> Priced {
        let actions: Vec<sts2_engine::Action> =
            plans.iter().map(|plan| plan.outcome().clone()).collect();
        self.priors_and_value(simulator, &actions)
    }

    /// The state's value alone.
    fn state_value(&self, simulator: &Simulator) -> f64;
}

/// A checkpoint that could not be loaded, and why.
#[derive(Debug)]
pub struct NetError(pub String);

impl std::fmt::Display for NetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl std::error::Error for NetError {}

type Plan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// The inputs the trainer declares, in order: scalars, tokens, token
/// features, action tokens, action features, action mask. The lifted
/// lookups' rows follow them.
const DECLARED_INPUTS: usize = 6;

/// One embedding lookup lifted out of the graph: which declared input
/// holds the ids, and the table the graph looked them up in.
struct Lifted {
    indices_input: usize,
    table: tract_ndarray::Array2<f32>,
}

impl Lifted {
    /// The rows `indices` name, in the indices' shape with the table's
    /// width appended: what the graph's `Gather` would have produced.
    fn rows(&self, indices: &Tensor) -> Tensor {
        let ids = indices.as_slice::<i64>().expect("token ids are i64");
        let width = self.table.ncols();
        let mut shape: TVec<usize> = indices.shape().into();
        shape.push(width);
        let mut rows = Tensor::zero::<f32>(&shape).expect("a dense f32 tensor");
        let flat = rows.as_slice_mut::<f32>().expect("just made it f32");
        let table = self
            .table
            .as_slice()
            .expect("the table is contiguous row-major");
        for (id, row) in ids.iter().zip(flat.chunks_exact_mut(width)) {
            // The encoder only emits ids the vocabulary the checkpoint was
            // verified against holds, so an id off the table is a bug in
            // the encoder, not a position to price.
            let start = usize::try_from(*id).expect("a token id is not negative") * width;
            row.copy_from_slice(&table[start..start + width]);
        }
        rows
    }
}

/// Takes every embedding lookup out of `proto`'s graph and hands back what
/// the host must do in its place.
///
/// A lookup is a `Gather` along axis 0 of a two-dimensional float
/// initializer by the ids of a declared graph input. Each becomes a float
/// input named `<ids>_rows`, appended after the declared inputs in the
/// order the graph listed the lookups, and everything that read the
/// gather's output reads the new input instead. A `Gather` shaped any
/// other way — by ids computed inside the graph, along another axis, of
/// something that is not a constant table — is left where it is; the plan
/// still runs it, just not as fast.
fn lift_embedding_lookups(
    proto: &mut pb::ModelProto,
    model_dir: Option<&str>,
) -> TractResult<Vec<Lifted>> {
    let graph = proto
        .graph
        .as_mut()
        .ok_or_else(|| anyhow!("the model holds no graph"))?;
    let declared: Vec<String> = graph.input.iter().map(|input| input.name.clone()).collect();
    let mut lifted = Vec::new();
    let mut renamed: Vec<(String, String)> = Vec::new();
    let mut kept = Vec::with_capacity(graph.node.len());
    for node in graph.node.drain(..) {
        let Some(lookup) = embedding_lookup(&node, &declared, &graph.initializer) else {
            kept.push(node);
            continue;
        };
        let (indices_input, table) = lookup;
        let table = tract_onnx::tensor::load_tensor(
            &tract_onnx::data_resolver::MmapDataResolver,
            table,
            model_dir,
        )?
        .to_array_view::<f32>()?
        .to_owned()
        .into_dimensionality::<tract_ndarray::Ix2>()?;
        let rows_name = format!("{}_rows", declared[indices_input]);
        if declared.contains(&rows_name) || renamed.iter().any(|(_, to)| *to == rows_name) {
            bail!("the graph already declares an input named {rows_name}");
        }
        let mut rows_shape = graph.input[indices_input]
            .r#type
            .as_ref()
            .and_then(|kind| match &kind.value {
                Some(pb::type_proto::Value::TensorType(tensor)) => tensor.shape.clone(),
                None => None,
            })
            .unwrap_or_default();
        rows_shape.dim.push(pb::tensor_shape_proto::Dimension {
            denotation: String::new(),
            value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(
                i64::try_from(table.ncols()).expect("an embedding width fits i64"),
            )),
        });
        graph.input.push(pb::ValueInfoProto {
            name: rows_name.clone(),
            r#type: Some(pb::TypeProto {
                denotation: String::new(),
                value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                    elem_type: pb::tensor_proto::DataType::Float as i32,
                    shape: Some(rows_shape),
                })),
            }),
            doc_string: String::new(),
        });
        renamed.push((node.output[0].clone(), rows_name));
        lifted.push(Lifted {
            indices_input,
            table,
        });
    }
    for name in kept
        .iter_mut()
        .flat_map(|node| node.input.iter_mut())
        .chain(graph.output.iter_mut().map(|output| &mut output.name))
    {
        if let Some((_, to)) = renamed.iter().find(|(from, _)| from == name) {
            to.clone_into(name);
        }
    }
    graph.node = kept;
    // A table nothing reads any more is not carried into the plan.
    let read: std::collections::HashSet<&str> = graph
        .node
        .iter()
        .flat_map(|node| node.input.iter())
        .chain(graph.input.iter().map(|input| &input.name))
        .map(String::as_str)
        .collect();
    let stale: Vec<String> = graph
        .initializer
        .iter()
        .filter(|table| !read.contains(table.name.as_str()))
        .map(|table| table.name.clone())
        .collect();
    graph
        .initializer
        .retain(|table| !stale.contains(&table.name));
    Ok(lifted)
}

/// Whether `node` is an embedding lookup [`lift_embedding_lookups`] takes:
/// which declared input holds its ids and which initializer is its table.
fn embedding_lookup<'a>(
    node: &pb::NodeProto,
    declared: &[String],
    initializers: &'a [pb::TensorProto],
) -> Option<(usize, &'a pb::TensorProto)> {
    let along_rows = node
        .attribute
        .iter()
        .all(|attribute| attribute.name != "axis" || attribute.i == 0);
    if node.op_type != "Gather" || !node.domain.is_empty() || !along_rows {
        return None;
    }
    let [table, ids] = node.input.as_slice() else {
        return None;
    };
    if node.output.len() != 1 {
        return None;
    }
    let indices_input = declared.iter().position(|name| name == ids)?;
    let table = initializers.iter().find(|tensor| tensor.name == *table)?;
    let is_float = table.data_type == pb::tensor_proto::DataType::Float as i32;
    (is_float && table.dims.len() == 2).then_some((indices_input, table))
}

/// A loaded, provenance-verified policy/value checkpoint, with the encoder
/// it prices positions through.
pub struct PolicyValueNet {
    plan: Plan,
    /// The embedding lookups taken out of the graph, in the order their
    /// rows are fed to the plan after the declared inputs.
    lifted: Vec<Lifted>,
    encoder: Arc<PolicyEncoder>,
    batch: usize,
    /// Whose data trained it, as its provenance names them. Empty where the
    /// provenance is silent — see [`PolicyValueNet::foreign_to`].
    characters: Vec<String>,
}

impl PolicyValueNet {
    /// Loads a *combat* checkpoint: `<base>.onnx` verified against
    /// `<base>.json` and this build — the policy encoding version, the
    /// observation version, the belief model version, and the vocabulary
    /// hash of `encoder` must all match, and the checkpoint must be the
    /// combat net (`scope: combat`, `value_semantics: combat-strength-v3`).
    pub fn load(base: &Path, encoder: Arc<PolicyEncoder>) -> Result<Self, NetError> {
        Self::load_with_batch(base, encoder, INFER_BATCH)
    }

    /// Loads a *macro* checkpoint: everything [`PolicyValueNet::load`]
    /// verifies, with the scope and the value semantics of the run net
    /// instead — `scope: macro`, `value_semantics: act-boundary-v2`.
    ///
    /// The two loaders exist because the two checkpoints are
    /// interchangeable on the wire and nowhere else. They share an encoder,
    /// a vocabulary, a graph shape and a file layout, so a macro checkpoint
    /// passed to `--net` would load, price fights with a head trained on
    /// acts, and answer plausible nonsense forever. Every other version this
    /// module checks is refused the same way and for the same reason: a
    /// checkpoint is never reinterpreted, only refused.
    ///
    /// A checkpoint whose provenance names no scope reads as `combat` —
    /// which is what the trainer itself defaults an unstamped header to, and
    /// what every checkpoint written before the macro track existed
    /// actually is.
    pub fn load_macro(base: &Path, encoder: Arc<PolicyEncoder>) -> Result<Self, NetError> {
        Self::load_scoped(
            base,
            encoder,
            INFER_BATCH,
            crate::training::SCOPE_MACRO,
            crate::objective::ACT_VALUE_SEMANTICS,
        )
    }

    /// Loads a *run* checkpoint — the PPO net: everything
    /// [`PolicyValueNet::load`] verifies, at the run net's scope and value
    /// semantics (`scope: macro`,
    /// `value_semantics: run-return-v1`).
    ///
    /// It shares [`PolicyValueNet::load_macro`]'s scope and differs only in
    /// what the value head was trained to predict, which is precisely why it
    /// needs a loader of its own. An act-boundary critic scores one act and
    /// answers inside a fixed band; a PPO critic scores a whole run under
    /// [`RunReward`](crate::reward::RunReward) and answers on a scale that
    /// grows with the climb and goes negative on a defeat. Nothing about the
    /// file distinguishes them — same graph, same vocabulary, same layout —
    /// so a PPO checkpoint passed to `--macro-net` would load and let the
    /// act tree min-max PPO returns as act-boundary scores, and an
    /// act-boundary checkpoint used as a PPO critic would make every
    /// advantage in a generation wrong in a way nothing crashes on. Refusing
    /// is the only honest answer: a checkpoint is never reinterpreted.
    pub fn load_run(base: &Path, encoder: Arc<PolicyEncoder>) -> Result<Self, NetError> {
        Self::load_run_priced(base, encoder, crate::objective::RUN_VALUE_SEMANTICS)
    }

    /// [`PolicyValueNet::load_run`] for a run checkpoint trained under a
    /// named reward: `value_semantics` is what
    /// [`crate::reward::value_semantics`] returns for the weights in play,
    /// and a checkpoint carrying any other name is refused.
    pub fn load_run_priced(
        base: &Path,
        encoder: Arc<PolicyEncoder>,
        value_semantics: &str,
    ) -> Result<Self, NetError> {
        Self::load_scoped(
            base,
            encoder,
            INFER_BATCH,
            crate::training::SCOPE_MACRO,
            value_semantics,
        )
    }

    /// The same load at another fixed batch shape. Probe plumbing: every
    /// production path runs at [`INFER_BATCH`], because one shape everywhere
    /// is what keeps batch composition out of the math.
    pub fn load_with_batch(
        base: &Path,
        encoder: Arc<PolicyEncoder>,
        batch: usize,
    ) -> Result<Self, NetError> {
        Self::load_scoped(
            base,
            encoder,
            batch,
            crate::training::SCOPE_COMBAT,
            crate::objective::VALUE_SEMANTICS,
        )
    }

    /// Everything `<base>.json` must agree with this build on, checked
    /// before a byte of the graph is read — and, answered back, the one
    /// field it may disagree on: the characters that trained it.
    fn verify_provenance(
        base: &Path,
        encoder: &PolicyEncoder,
        scope: &str,
        value_semantics: &str,
    ) -> Result<Vec<String>, NetError> {
        let provenance_path = base.with_extension("json");
        let provenance: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&provenance_path)
                .map_err(|error| NetError(format!("{}: {error}", provenance_path.display())))?,
        )
        .map_err(|error| NetError(format!("{}: {error}", provenance_path.display())))?;
        let check = |field: &str, expected: &str| -> Result<(), NetError> {
            let found = provenance
                .get(field)
                .map(|value| match value {
                    serde_json::Value::String(text) => text.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_default();
            if found == expected {
                Ok(())
            } else {
                Err(NetError(format!(
                    "checkpoint {field} is {found:?}, this build wants {expected:?}"
                )))
            }
        };
        check(
            "policy_encoding_version",
            &POLICY_ENCODING_VERSION.to_string(),
        )?;
        check(
            "observation_version",
            &sts2_engine::AGENT_OBSERVATION_VERSION.to_string(),
        )?;
        check("belief_model_version", sts2_engine::BELIEF_MODEL_VERSION)?;
        check("vocabulary_hash", encoder.vocabulary_hash())?;
        // Absent reads as the combat net, exactly as the trainer's own
        // default does: every checkpoint written before the macro track
        // existed is one.
        let found_scope = provenance
            .get("scope")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(crate::training::SCOPE_COMBAT);
        if found_scope != scope {
            return Err(NetError(format!(
                "checkpoint scope is {found_scope:?}, this loader wants \
                 {scope:?}: the combat net and the run net are different \
                 nets over different horizons"
            )));
        }
        check("value_semantics", value_semantics)?;
        // Read, never checked here: a mismatch is a warning the caller
        // raises, not a refusal. See `foreign_to`.
        let characters = provenance
            .get("characters")
            .and_then(serde_json::Value::as_array)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| id.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        Ok(characters)
    }

    fn load_scoped(
        base: &Path,
        encoder: Arc<PolicyEncoder>,
        batch: usize,
        scope: &str,
        value_semantics: &str,
    ) -> Result<Self, NetError> {
        let characters = Self::verify_provenance(base, &encoder, scope, value_semantics)?;

        let onnx_path = base.with_extension("onnx");
        let (plan, lifted) = Self::plan_for(&onnx_path, batch)
            .map_err(|error| NetError(format!("{}: {error:?}", onnx_path.display())))?;
        Ok(Self {
            plan,
            lifted,
            encoder,
            batch,
            characters,
        })
    }

    /// The graph as the file holds it, its embedding lookups lifted out,
    /// planned at a fixed batch and symbolic token (`t`) and action (`a`)
    /// axes so one plan prices every shape a position comes in.
    fn plan_for(onnx_path: &Path, batch: usize) -> TractResult<(Plan, Vec<Lifted>)> {
        let onnx = tract_onnx::onnx();
        let mut proto = onnx.proto_model_for_path(onnx_path)?;
        let model_dir = onnx_path.parent().and_then(Path::to_str);
        let lifted = lift_embedding_lookups(&mut proto, model_dir)?;
        let tract_onnx::model::ParseResult {
            mut model,
            unresolved_inputs,
            ..
        } = onnx.parse(&proto, model_dir)?;
        if !unresolved_inputs.is_empty() {
            bail!("the graph reads inputs nothing declares: {unresolved_inputs:?}");
        }
        let (batch, t, a) = (
            batch.to_dim(),
            model.symbols.sym("t").to_dim(),
            model.symbols.sym("a").to_dim(),
        );
        let declared: [(DatumType, TVec<TDim>); DECLARED_INPUTS] = [
            (
                f32::datum_type(),
                tvec![batch.clone(), OBSERVATION_SCALARS.to_dim()],
            ),
            (i64::datum_type(), tvec![batch.clone(), t.clone()]),
            (
                f32::datum_type(),
                tvec![batch.clone(), t, TOKEN_FEATURES.to_dim()],
            ),
            (
                i64::datum_type(),
                tvec![batch.clone(), a.clone(), ACTION_TOKENS.to_dim()],
            ),
            (
                f32::datum_type(),
                tvec![batch.clone(), a.clone(), ACTION_FEATURES.to_dim()],
            ),
            (f32::datum_type(), tvec![batch, a]),
        ];
        for (slot, (datum, shape)) in declared.iter().enumerate() {
            model.set_input_fact(slot, InferenceFact::dt_shape(*datum, shape.clone()))?;
        }
        for (offset, lift) in lifted.iter().enumerate() {
            let mut shape = declared[lift.indices_input].1.clone();
            shape.push(lift.table.ncols().to_dim());
            model.set_input_fact(
                DECLARED_INPUTS + offset,
                InferenceFact::dt_shape(f32::datum_type(), shape),
            )?;
        }
        let plan = model.into_optimized()?.into_runnable()?;
        Ok((plan, lifted))
    }

    /// Which characters' data trained the checkpoint, as its provenance
    /// names them. Empty for a checkpoint written before provenance carried
    /// the field, which reads as "does not say", not as "trained on
    /// nothing".
    #[must_use]
    pub fn characters(&self) -> &[String] {
        &self.characters
    }

    /// Returns a warning if the checkpoint's recorded training characters
    /// exclude `character`. Returns `None` if training characters are unknown.
    ///
    /// Checkpoints can be used across characters because they share a
    /// vocabulary. This allows training a new character with learned priors.
    #[must_use]
    pub fn foreign_to(&self, character: &sts2_core::ModelId) -> Option<String> {
        let played = character.to_string();
        if self.characters.is_empty() || self.characters.contains(&played) {
            return None;
        }
        Some(format!(
            "the checkpoint was trained on {} and is about to play {played}: \
             cross-character priors are weaker than a checkpoint of the \
             character's own, and a mis-pointed --net looks exactly like this",
            self.characters.join(", "),
        ))
    }

    /// The encoder this checkpoint prices through.
    #[must_use]
    pub fn encoder(&self) -> &PolicyEncoder {
        &self.encoder
    }

    /// The fixed batch shape this checkpoint's plan runs at.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch
    }

    /// How many of the graph's embedding lookups this side does for it.
    /// A production export has three; how many lift depends on whether the
    /// export indexes each by a declared input or by ids it computes.
    #[must_use]
    pub fn lifted_lookups(&self) -> usize {
        self.lifted.len()
    }

    /// The encoder as a handle a server or a worker can carry off.
    #[must_use]
    pub fn encoder_handle(&self) -> Arc<PolicyEncoder> {
        Arc::clone(&self.encoder)
    }

    /// One forward pass over up to [`INFER_BATCH`] encoded positions,
    /// answering with each row's raw logits and value. The batch is padded
    /// to the fixed size with zero rows and the padding discarded, so a
    /// request's answer never depends on what else was queued beside it.
    /// The logits come back at the batch's action width — the widest
    /// decision in it — and a row reads only as many as it asked about.
    #[must_use]
    pub fn run_batch(&self, requests: &[EncodedRequest]) -> Vec<(Vec<f32>, f64)> {
        self.priced_batch(requests)
            .into_iter()
            .map(|row| (row.logits, row.value))
            .collect()
    }

    /// The same forward pass, keeping what [`run_batch`](Self::run_batch)
    /// drops: whether the checkpoint answered the row with a number that was
    /// not one.
    ///
    /// The plan runs at the shape the batch needs and no larger: as many
    /// token slots as its fullest observation filled, as many action rows
    /// as its widest decision offered, and at least one of each so the
    /// graph always has an axis to reduce over. A value-only request with
    /// no actions runs one masked-out action row.
    fn priced_batch(&self, requests: &[EncodedRequest]) -> Vec<Row> {
        assert!(
            requests.len() <= self.batch,
            "a batch holds {} requests, the plan runs at most {}",
            requests.len(),
            self.batch
        );
        let mut live_tokens = 1;
        let mut width = 1;
        for request in requests {
            crate::probe::tally(
                crate::probe::Tally::LiveTokens,
                request.observation.live_tokens() as u64,
            );
            crate::probe::tally(
                crate::probe::Tally::PricedActions,
                request.actions.len() as u64,
            );
            assert!(
                request.actions.len() <= MAX_ACTIONS,
                "a decision offered {} actions, the net prices at most {MAX_ACTIONS}",
                request.actions.len()
            );
            live_tokens = live_tokens.max(request.observation.live_tokens());
            width = width.max(request.actions.len());
        }
        let mut scalars = tract_ndarray::Array2::<f32>::zeros((self.batch, OBSERVATION_SCALARS));
        let mut tokens = tract_ndarray::Array2::<i64>::zeros((self.batch, live_tokens));
        let mut token_features =
            tract_ndarray::Array3::<f32>::zeros((self.batch, live_tokens, TOKEN_FEATURES));
        let mut action_tokens =
            tract_ndarray::Array3::<i64>::zeros((self.batch, width, ACTION_TOKENS));
        let mut action_features =
            tract_ndarray::Array3::<f32>::zeros((self.batch, width, ACTION_FEATURES));
        let mut action_mask = tract_ndarray::Array2::<f32>::zeros((self.batch, width));
        for (row, request) in requests.iter().enumerate() {
            let observation = &request.observation;
            for (column, &scalar) in observation.scalars.iter().enumerate() {
                scalars[(row, column)] = scalar;
            }
            // Live tokens fill the encoding from slot 0; past them is the
            // padding the plan no longer runs over.
            let live = observation.live_tokens();
            for (column, &token) in observation.tokens.iter().take(live).enumerate() {
                tokens[(row, column)] = i64::from(token);
            }
            for (column, &feature) in observation
                .features
                .iter()
                .take(live * TOKEN_FEATURES)
                .enumerate()
            {
                token_features[(row, column / TOKEN_FEATURES, column % TOKEN_FEATURES)] = feature;
            }
            for (slot, action) in request.actions.iter().enumerate() {
                for (position, &token) in action.tokens.iter().enumerate() {
                    action_tokens[(row, slot, position)] = i64::from(token);
                }
                for (position, &feature) in action.features.iter().enumerate() {
                    action_features[(row, slot, position)] = feature;
                }
                action_mask[(row, slot)] = 1.0;
            }
        }
        let mut inputs: TVec<TValue> = tvec!(
            Tensor::from(scalars).into(),
            Tensor::from(tokens).into(),
            Tensor::from(token_features).into(),
            Tensor::from(action_tokens).into(),
            Tensor::from(action_features).into(),
            Tensor::from(action_mask).into(),
        );
        for lift in &self.lifted {
            inputs.push(lift.rows(&inputs[lift.indices_input]).into());
        }
        let outputs = self
            .plan
            .run(inputs)
            .expect("a verified checkpoint prices a position");
        let logits = outputs[0]
            .to_array_view::<f32>()
            .expect("logits are f32")
            .into_dimensionality::<tract_ndarray::Ix2>()
            .expect("logits are [batch, actions]");
        let values = outputs[1].as_slice::<f32>().expect("the value is f32");
        let widths = requests.iter().map(|request| request.actions.len());
        settle_rows(logits, values, widths)
    }
}

/// The plan's two outputs read back as one [`Row`] per request, each
/// flagged where the checkpoint answered it with a number that was not
/// one. `widths` says how many logits each request asked about; past
/// those the row is the batch's padding and says nothing about the
/// checkpoint, so it is settled but not read.
fn settle_rows(
    logits: tract_ndarray::ArrayView2<'_, f32>,
    values: &[f32],
    widths: impl Iterator<Item = usize>,
) -> Vec<Row> {
    widths
        .enumerate()
        .map(|(row, width)| {
            let answered = logits.index_axis(tract_ndarray::Axis(0), row);
            let non_finite = !values[row].is_finite()
                || answered.iter().take(width).any(|logit| !logit.is_finite());
            Row {
                logits: answered.iter().map(|&logit| settled(logit, 0.0)).collect(),
                value: f64::from(settled(values[row], 0.0)),
                non_finite,
            }
        })
        .collect()
}

/// One row of a forward pass: the logits over the padded action axis, the
/// value, and whether the checkpoint answered either with a number that was
/// not one.
struct Row {
    logits: Vec<f32>,
    value: f64,
    non_finite: bool,
}

/// A number a checkpoint answered with, or `fallback` where it answered with
/// no number at all.
///
/// Same discipline as the over-wide decision above: a checkpoint degrades
/// honestly or is refused, and is never reinterpreted. A `NaN` or an
/// infinity out of a value head or a logit row would be reinterpreted by
/// every consumer downstream in a different way — `total_cmp` sorts a `NaN`
/// above every real score, a softmax over one answers uniform, a min-max
/// normalization spreads it to every sibling — so the search's behaviour
/// would depend on which comparison saw it first. Zero is the neutral
/// answer in both units the heads speak: a flat logit, and a value the
/// min-max cannot distinguish from an ordinary low one. It is announced
/// once per process, loudly, because a checkpoint that produces one is a
/// training bug that wants finding, not a condition to live with.
///
/// The warning is once and the count is always: one line however many rows
/// a `--jobs 24` batch settles, and [`non_finite_answers`] says how many
/// that was.
fn settled(value: f32, fallback: f32) -> f32 {
    static ANNOUNCED: std::sync::Once = std::sync::Once::new();
    if value.is_finite() {
        return value;
    }
    NON_FINITE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ANNOUNCED.call_once(|| {
        eprintln!(
            "warning: a checkpoint answered with {value}, not a number; \
             reading it as {fallback}. The checkpoint is broken — retrain or \
             re-export it. Further occurrences are counted, not printed."
        );
    });
    fallback
}

/// Numbers this process has read out of a checkpoint that were not numbers,
/// logits and values together. Process-wide because the checkpoints are
/// shared read-only across a batch's workers, so the count is the batch's.
static NON_FINITE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many non-finite numbers `settled` has read as zero, over the whole
/// process. Any count at all means the checkpoint is broken.
#[must_use]
pub fn non_finite_answers() -> u64 {
    NON_FINITE.load(std::sync::atomic::Ordering::Relaxed)
}

/// One encoded position for [`PolicyValueNet::run_batch`]: what a worker
/// hands an inference server, encoding already done on the worker's core.
pub struct EncodedRequest {
    pub observation: crate::encoding::ObservationEncoding,
    pub actions: Vec<ActionEncoding>,
}

/// A request built from what the player sees.
#[must_use]
pub fn encode_request(
    encoder: &PolicyEncoder,
    simulator: &Simulator,
    actions: &[sts2_engine::Action],
) -> EncodedRequest {
    let observation = simulator.agent_observation();
    EncodedRequest {
        actions: encoder.encode_actions(&observation, actions),
        observation: encoder.encode_observation(&observation),
    }
}

/// The same, over plans: a trade's row is its acquisition's block plus what
/// it gives up.
#[must_use]
pub fn encode_plan_request(
    encoder: &PolicyEncoder,
    simulator: &Simulator,
    plans: &[crate::plan::ActionPlan],
) -> EncodedRequest {
    let observation = simulator.agent_observation();
    EncodedRequest {
        actions: encoder.encode_plans(&observation, plans),
        observation: encoder.encode_observation(&observation),
    }
}

impl PolicyValueNet {
    /// Uniform priors over `width` actions with the value head's read of the
    /// position, under the cause that made the checkpoint's own ranking
    /// unavailable.
    #[allow(
        clippy::cast_precision_loss,
        reason = "action counts are far below f32 precision"
    )]
    fn uniform(&self, simulator: &Simulator, width: usize, cause: Degraded) -> Priced {
        Priced::fallback(
            vec![1.0 / width as f32; width],
            self.state_value(simulator),
            cause,
        )
    }

    /// One request answered: the softmax over the rows the decision filled,
    /// flagged where the checkpoint's answer held a number that was not one.
    fn priced(&self, request: &EncodedRequest, width: usize) -> Priced {
        let row = self
            .priced_batch(std::slice::from_ref(request))
            .pop()
            .expect("one request, one answer");
        let priors = softmax(&row.logits[..width]);
        if row.non_finite {
            Priced::fallback(priors, row.value, Degraded::NonFinite)
        } else {
            Priced::real(priors, row.value)
        }
    }
}

impl Evaluate for PolicyValueNet {
    fn priors_and_value(&self, simulator: &Simulator, actions: &[sts2_engine::Action]) -> Priced {
        // The checkpoint prices at most MAX_ACTIONS rows, and a decision
        // wider than that gets uniform priors and the value head's read of
        // the position instead. The search loses its guidance on that one
        // node, and says so.
        if actions.len() > MAX_ACTIONS {
            return self.uniform(simulator, actions.len(), Degraded::OverAxis);
        }
        let request = encode_request(&self.encoder, simulator, actions);
        self.priced(&request, actions.len())
    }

    fn plan_priors_and_value(
        &self,
        simulator: &Simulator,
        plans: &[crate::plan::ActionPlan],
    ) -> Priced {
        // Same ceiling and the same degradation as the action path above.
        if plans.len() > MAX_ACTIONS {
            return self.uniform(simulator, plans.len(), Degraded::OverAxis);
        }
        let request = encode_plan_request(&self.encoder, simulator, plans);
        self.priced(&request, plans.len())
    }

    fn state_value(&self, simulator: &Simulator) -> f64 {
        let request = encode_request(&self.encoder, simulator, &[]);
        self.run_batch(std::slice::from_ref(&request))
            .pop()
            .expect("one request, one answer")
            .1
    }
}

/// A numerically settled softmax.
pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = logits.iter().map(|&logit| (logit - peak).exp()).collect();
    let total: f32 = weights.iter().sum();
    if total > 0.0 {
        weights.iter().map(|weight| weight / total).collect()
    } else {
        #[allow(
            clippy::cast_precision_loss,
            reason = "action counts are far below f32 precision"
        )]
        {
            vec![1.0 / logits.len() as f32; logits.len()]
        }
    }
}

/// The act tree's evaluator: two checkpoints behind one `Evaluate` seam, so
/// the search core learns nothing new and every act-tree leaf costs one
/// forward pass instead of a hundred-and-fifty-step random walk.
///
/// **Which net answers where.** The act tree's nodes are exactly its macro
/// decisions — `ActBelief::plays_out` fences live fights out of the tree —
/// so:
///
/// - *Macro nodes* (priors, and the value of a leaf standing on one) are the
///   macro checkpoint's. Its value head was trained on
///   [`ACT_VALUE_SEMANTICS`](crate::objective::ACT_VALUE_SEMANTICS) targets,
///   which is the very quantity this search min-maxes, so there is nothing
///   to translate: the number goes in as it comes out.
/// - *Fight entries* — the leaf where a tree path or a rollout walks into a
///   live fight — are the combat checkpoint's, translated. This is the piece
///   the act-v0.5 design deferred ("two incomparable scales inside one
///   min-max") and the one the backlog's queued item 2 asked for.
///
/// **The translation, and what is honest about it.** The combat head answers
/// with one scalar: the expected
/// [`CombatStrength`](crate::objective::CombatStrength) of the fight's exit,
/// `(win + hp_retained + 0.1·potions) / (1 + 0.02·turns)`. Two
/// quantities the act horizon needs are tangled inside it, and a single
/// scalar cannot separate them — *this is the caveat, stated plainly: what
/// follows is a reading of one number, not a measured `P(survive)`.* The
/// reading:
///
/// - **Survival** `s = clamp(v / win_weight, 0, 1)`. A head that scores the
///   fight below the win weight alone is saying it may not walk out; at or
///   above it, walking out is what the head is confident of.
/// - **Health after** `h = clamp(v − win_weight, 0, 1)`. Whatever the head
///   scores *above* the win weight is, by the objective's own arithmetic,
///   the health it expects to be left holding. Exact only for a fight that
///   ends on turn zero; under `combat-strength-v3`'s discount a fight that
///   runs long reads back low, so both quantities are floors on the truth
///   rather than estimates of it — conservative in the direction that makes
///   the search respect fights, which is the safe side of this reading.
///
/// Those two feed [`ActBoundary`](crate::objective::ActBoundary)'s own
/// formula at this state, over its own weights — no new numbers are
/// introduced:
///
/// ```text
/// through = crossing_weight + hp_weight·h + potion_weight·potions + floor_weight·floor
/// died    = floor_weight·floor                     (ActBoundary's defeat pricing, exact)
/// value   = s·through + (1 − s)·died
/// ```
///
/// The `crossing_weight` term is deliberate and is what keeps the two leaf
/// kinds on one scale: the macro net predicts the act's *eventual* boundary
/// score, so a fight-entry leaf must predict one too, not score the state it
/// stands on. Without it every arm that walked into a fight would be priced
/// a full crossing below every arm that did not, and the search would learn
/// to avoid fights — the exact pathology a naive splice produces. The price
/// of including it is a documented optimism: the surviving branch assumes
/// the *rest* of the act is crossed, which the combat head has no opinion
/// about. It is a constant offset across the arms of one decision, so it
/// shifts values without reordering them, and the min-max normalizes it out.
pub struct ActEvaluator {
    macro_net: Arc<dyn Evaluate>,
    combat_net: Arc<dyn Evaluate>,
    boundary: crate::objective::ActBoundary,
    /// [`CombatStrength`](crate::objective::CombatStrength)'s own win
    /// weight: the pivot the combat scalar is read against.
    combat_win_weight: f64,
}

impl ActEvaluator {
    /// The act tree's evaluator over a macro checkpoint and a combat one.
    #[must_use]
    pub fn new(macro_net: Arc<dyn Evaluate>, combat_net: Arc<dyn Evaluate>) -> Self {
        Self {
            macro_net,
            combat_net,
            boundary: crate::objective::ActBoundary::default(),
            combat_win_weight: crate::objective::CombatStrength::default().win_weight,
        }
    }

    /// One combat-scalar reading, in act-boundary units, at the state the
    /// fight is entered on. See this type's own documentation for what the
    /// arithmetic assumes.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "floors and potion belts are far below f64 precision"
    )]
    pub fn translate(&self, combat_value: f64, simulator: &Simulator) -> f64 {
        let state = simulator.state();
        let floor = f64::from(state.run.as_ref().map_or(0, |run| run.floor));
        let potions = state.run_player.potions.iter().flatten().count() as f64;
        let survival = (combat_value / self.combat_win_weight).clamp(0.0, 1.0);
        let health = (combat_value - self.combat_win_weight).clamp(0.0, 1.0);
        let boundary = self.boundary;
        let through = boundary.crossing_weight
            + boundary.hp_weight * health
            + boundary.potion_weight * potions
            + boundary.floor_weight * floor;
        let died = boundary.floor_weight * floor;
        (survival * through + (1.0 - survival) * died).max(0.0)
    }
}

impl ActEvaluator {
    /// The in-fight answer: uniform priors over `width` and the translated
    /// combat value.
    ///
    /// Defensive — the act tree's fence means no node is ever opened inside
    /// a fight. What reaches here would be priced off in-fight play the act
    /// tree is not choosing between, so it is priced off nothing and counted
    /// as [`Degraded::OutOfScope`].
    #[allow(
        clippy::cast_precision_loss,
        reason = "action counts are far below f32 precision"
    )]
    fn out_of_scope(&self, simulator: &Simulator, width: usize) -> Priced {
        let uniform = if width == 0 {
            Vec::new()
        } else {
            vec![1.0 / width as f32; width]
        };
        Priced::fallback(uniform, self.state_value(simulator), Degraded::OutOfScope)
    }
}

impl Evaluate for ActEvaluator {
    fn priors_and_value(&self, simulator: &Simulator, actions: &[sts2_engine::Action]) -> Priced {
        if crate::env::fight_over(simulator) {
            return self.macro_net.priors_and_value(simulator, actions);
        }
        self.out_of_scope(simulator, actions.len())
    }

    fn plan_priors_and_value(
        &self,
        simulator: &Simulator,
        plans: &[crate::plan::ActionPlan],
    ) -> Priced {
        // Plans stand only at macro screens, which is exactly where the
        // macro net answers; the in-fight arm is the same defensive uniform.
        if crate::env::fight_over(simulator) {
            return self.macro_net.plan_priors_and_value(simulator, plans);
        }
        self.out_of_scope(simulator, plans.len())
    }

    fn state_value(&self, simulator: &Simulator) -> f64 {
        if crate::env::fight_over(simulator) {
            // A macro state: already act-boundary units, nothing to
            // translate.
            self.macro_net.state_value(simulator)
        } else {
            self.translate(self.combat_net.state_value(simulator), simulator)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkpoint that answers with something that is not a number degrades
    /// the pricing rather than passing a settled zero off as its own work,
    /// and every such number is counted however few are printed.
    ///
    /// The answer is injected at the seam between the plan and the rows it
    /// becomes. Whether a `NaN` fed to a real graph comes out the other end
    /// is the backend's business — tract's kernels read `max(NaN, 0)` as
    /// zero on some plan shapes and not others — and the property here is
    /// what happens once one does.
    #[test]
    fn a_non_finite_answer_is_flagged_and_counted() {
        let logits = tract_ndarray::arr2(&[[0.5, 0.5, 0.0], [f32::NAN, f32::NAN, 0.0]]);
        let values = [0.25, f32::INFINITY];
        let before = non_finite_answers();
        let rows = settle_rows(logits.view(), &values, [2, 2].into_iter());
        assert!(!rows[0].non_finite, "an ordinary row is the net's own");
        assert!(rows[1].non_finite, "a poisoned row is flagged");
        // The counter is the process's, so a test beside this one may add
        // to it; the three settled here are its floor.
        assert!(
            non_finite_answers() - before >= 3,
            "every number settled is counted"
        );
        assert!(
            rows[1].value.is_finite(),
            "the value is settled, not passed on"
        );
        let priors = softmax(&rows[1].logits[..2]);
        assert!(
            priors.iter().all(|prior| (prior - 0.5).abs() < 1e-6),
            "a flat row softmaxes to uniform: {priors:?}"
        );
    }

    /// The padding past a row's own width is the batch's, not the
    /// checkpoint's, and is never read as an answer.
    #[test]
    fn padding_past_the_width_is_not_an_answer() {
        let logits = tract_ndarray::arr2(&[[0.5, f32::NAN]]);
        let rows = settle_rows(logits.view(), &[0.0], [1].into_iter());
        assert!(!rows[0].non_finite);
    }
}
