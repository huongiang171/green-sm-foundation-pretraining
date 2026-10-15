# Green SM Foundation Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one reusable Green SM Foundation checkpoint on the linked cross-service corpus, with masked-field reconstruction and next-action prediction, then produce a reproducible frozen-embedding artifact for both Food Recommendation and Ad Optimization.

**Architecture:** Member 3 adds an isolated `src/pretraining` package around the existing `FoundationModel`; it does not reimplement tokenization, adapters, Transformer, downstream heads, or the public evaluator. Each optimizer step has two non-leaking views of the same history: a bidirectional, corrupted view for masked-field reconstruction and an uncorrupted causal view for next-action prediction. `UserEmbeddingFusion` consumes `long_term` and `short_term` so next-action loss updates the `UserIdAdapter`; checkpointing saves both the Foundation weights and fusion weights.

**Tech Stack:** Python 3.10+, PyTorch, PyArrow/Parquet, JSON configuration, pytest. No GPU is required for unit tests; the pilot/full training commands target Kaggle GPU.

**Spec:** `docs/superpowers/specs/2026-10-15-green-sm-simulated-corpus-design.md`; data handoff and readiness gates: `docs/superpowers/plans/2026-10-15-green-sm-data-readiness-pilot.md`.

## Read This First: Execution Gate

Do **not** run a real pretraining job from the current `main` branch. The generated pilot contains real geo fields, but the current runtime is still `FoundationBatch` v2: it has no `geo_values` or `geo_missing_mask`, and `AdapterRegistry` does not route `GeoAdapter`. Before Task 4's integration run, Member 1 and Member 2 must merge this v3 contract:

1. `FoundationBatch` and `foundation_collate` expose aligned `geo_values: FloatTensor[B, L, 2]` and `geo_missing_mask: BoolTensor[B, L]`; the tokenizer emits a `geo` modality token containing its geo-cell categorical ID and normalized coordinate target.
2. `FoundationModelConfig` receives `geo_modality_id`, and `AdapterRegistry` routes those positions through `GeoAdapter` while retaining key/modality/source metadata.
3. The selected vocabulary artifact contains `source_to_id["sequence"]`, `modality_to_id["numeric"]`, and `modality_to_id["geo"]`; max token length is 256 in the data and model configs.

Tasks 1–3 can be written against that published interface once it is agreed. Task 4 must make the absence or mismatch of the contract an actionable error; it must not silently drop geo loss or fall back to ID-only training.

## File Structure

```text
configs/
  pretrain_green_sm_pilot.json       # committed 500–1,000 step Kaggle pilot recipe
  pretrain_green_sm_full.json        # committed full-corpus recipe; no local paths
src/models/
  objectives.py                      # Member 3: fusion, heads, typed loss breakdown
src/pretraining/
  __init__.py
  config.py                          # strict JSON config dataclasses and validation
  masking.py                         # immutable, deterministic multi-modal corruption
  gates.py                           # data/model/v3 readiness preflight
  artifacts.py                       # run/checkpoint manifests and SHA-256 helpers
  runner.py                          # vocabulary, streaming batches, train/eval/checkpoint loop
  __main__.py                        # `python -m src.pretraining --config ...`
tests/pretraining/
  __init__.py
  conftest.py                        # tiny v3 batch, fake report/root, CPU config helpers
  test_config_and_gates.py
  test_masking.py
  test_objectives.py
  test_runner.py
docs/
  kaggle_foundation_pretraining_runbook.md
```

`src/foundation_input/**` and `src/models/adapters.py` are upstream ownership. This plan consumes their v3 interface after it lands; it does not recreate it in Member 3's branch. `src/downstream/**`, `src/baselines/**`, and public evaluation code remain out of scope.

## Locked Training Decisions

- **Corpora:** train the linked and paired-control roots separately with identical architecture, seed, schedule, batch size, and tokenizer policy. Each root owns a vocabulary/scaler artifact fitted only from its own train split, as required by the governing corpus spec. The linked checkpoint is the only artifact handed to Food and Ads; control is an ablation artifact only.
- **No leakage:** fit each vocabulary/scaler only on its own `train` split; never read validation/test while fitting. For a given root, next-action target group remains excluded by `encode_sample` before the causal forward pass.
- **Two forwards:** masked reconstruction uses `foundation(masked_batch, attention_mode="bidirectional")`. Next-action uses `foundation(original_batch, attention_mode="next_action")`; this is necessary because the two objectives have different visibility rules.
- **Masking:** choose eligible semantic tokens with deterministic seed `(run_seed, global_step, sample position)` and `mask_probability=0.15`; never select PAD/CLS/SEP. Replace selected categorical/text IDs with `MASK`. For selected numeric and geo positions also replace the continuous inputs with neutral zero and mark the model-input missing flag; preserve originals only in a separate immutable target object. Do not use BERT's “leave unchanged” branch for continuous features.
- **Loss:** masked categorical/text/geo-cell values use cross entropy; observed numeric uses SmoothL1 on normalized raw value; numeric missingness and geo missingness use BCE-with-logits; valid geo coordinates use SmoothL1. Every component is normalized by its own eligible count; a zero-count component is an autograd-safe zero and is logged as zero count. Next-action is CE over `next_action_type_ids`, filtered strictly by `next_action_valid_mask`.
- **Fusion:** `z_user = LayerNorm(Linear([long_term ; short_term]))`, with output width exactly `d_model`. A next-action head consumes `z_user`; the gradient regression test is mandatory. The downstream frozen feature is this `z_user`, while raw long/short vectors are retained for diagnosis.
- **Evaluation:** validation runs only on `validation` at fixed step intervals. Test may run once, after checkpoint selection, through an explicit CLI flag; it is never used for early stopping or hyperparameter choice. Member 3 logs pretraining losses only and calls no private downstream evaluator.

## Global Constraints

- Use the linked pilot root `data/processed/green_sm_sim_pilot_v1/green_sm_sim_linked_v1/foundation` and paired-control root `data/processed/green_sm_sim_pilot_v1/green_sm_sim_control_paired_v1/foundation` only through `load_manifest`, `iter_normalized_samples`, and `FoundationPretrainDataset`.
- Require both `reports/data_quality_report.json` files to have `readiness_passed=true`, matching generator config digest/target-world fingerprint, and matching logical opportunity manifests before starting either paired run.
- Require `upstream_v3_geo_contract_ready=true` in the readiness reports **and** inspect the runtime batch fields. A report alone is insufficient.
- Keep `DatasetConfig(max_tokens=256, max_event_count=30, next_action_prediction=True)` aligned with `FoundationModelConfig(max_length=256)`; reject a conflicting config before allocating a model.
- Derive all vocabulary cardinalities plus sequence/numeric/geo routing IDs from the saved vocabulary artifact. Do not hard-code integer IDs or infer them from a batch.
- Preserve the input contract's source namespace. Do not join or remap users across roots; user-ID hashing/bucketing remains entirely inside `UserIdAdapter`.
- Write generated models, logs, copied vocabulary, reports, and manifests below the configured external `output_dir`; do not commit checkpoints, Parquet data, or Kaggle secrets.
- Use `python -m pytest`, not bare `pytest`; use a repository-local `--basetemp .pytest-tmp/pretraining` when needed on this Windows workspace.

## Review Focus

1. **Unavailable or partial v3 geo input:** the preflight must reject v2, a missing geo modality, or a report/runtime disagreement rather than train an apparently successful incomplete model. Pinned by Task 1's `test_preflight_rejects_missing_geo_contract`.
2. **Masked-value leakage:** special/padded tokens must never be selected, and numeric/geo raw values must be neutralized in the model input while remaining recoverable only as labels. Pinned by Task 2's `test_masking_neutralizes_continuous_targets_without_mutating_source_batch`.
3. **Invalid next-action rows:** `next_action_valid_mask=False` rows must contribute exactly zero CE and must not make a batch non-finite. Pinned by Task 3's `test_next_action_loss_ignores_invalid_rows`.
4. **Vocabulary/test leakage:** each corpus vocabulary must be fitted only from its own train split; validation/test must never be read while fitting. Pinned by Task 4's `test_build_run_inputs_never_iterates_validation_or_test_when_fitting_vocab`.
5. **Wrong checkpoint used downstream:** restore must reject changed vocab, corpus manifest, model configuration, or incomplete checkpoint state before returning a frozen embedder. Pinned by Task 5's `test_load_frozen_embedder_rejects_manifest_mismatch`.

---

### Task 1: Define the strict run configuration and fail-closed v3 preflight

**Files:**
- Create: `src/pretraining/__init__.py`
- Create: `src/pretraining/config.py`
- Create: `src/pretraining/gates.py`
- Create: `tests/pretraining/__init__.py`
- Create: `tests/pretraining/conftest.py`
- Create: `tests/pretraining/test_config_and_gates.py`
- Create: `configs/pretrain_green_sm_pilot.json`
- Create: `configs/pretrain_green_sm_full.json`

**Interfaces:**
- Consumes: `FoundationBatch` v3 type metadata and both readiness reports. Vocabulary routing IDs are validated in Task 4, after the shared artifact exists.
- Produces:

```python
@dataclass(frozen=True)
class PretrainingConfig: ...

def load_pretraining_config(path: Path) -> PretrainingConfig: ...

@dataclass(frozen=True)
class PreflightResult:
    linked_manifest_sha256: str
    control_manifest_sha256: str
    shared_generator_config_digest: str
    shared_target_world_fingerprint: str

def preflight_green_sm(config: PretrainingConfig) -> PreflightResult: ...
def validate_v3_batch_contract(batch: FoundationBatch, routing: RoutingIds) -> None: ...
```

- [ ] **Step 1: Write failing config/preflight tests**

```python
def test_config_rejects_length_mismatch(tmp_path: Path) -> None:
    path = write_config(tmp_path, model_max_length=128, dataset_max_tokens=256)
    with pytest.raises(ValueError, match="max_tokens.*max_length"):
        load_pretraining_config(path)

def test_preflight_rejects_missing_geo_contract(v2_batch, ready_reports, config) -> None:
    with pytest.raises(RuntimeError, match="geo_values.*geo_missing_mask"):
        validate_v3_batch_contract(v2_batch, ROUTING)
```

Also cover unknown JSON keys, zero/negative training values, a false readiness flag, unequal paired fingerprints, and non-256 max length.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/pretraining/test_config_and_gates.py -v --basetemp .pytest-tmp/pretraining-config`

Expected: FAIL because the pretraining package does not exist.

- [ ] **Step 3: Implement strict JSON configuration and preflight**

`PretrainingConfig` must contain `linked_root`, `control_root`, `output_dir`, seed, model dimensions, dataset limits, optimizer schedule, masking settings, objective weights, and artifact names. Reject unknown/missing fields and non-finite numeric settings. `preflight_green_sm` must checksum both manifests, load both reports, validate the paired identity fields, and reject a runtime codebase that does not declare the v3 geo fields. `validate_v3_batch_contract` separately validates tensor dtype/shape after Task 4 collates the first real batch. Neither function may create an output directory or fit vocabulary.

Use committed paths relative to repository root in the pilot/full JSON files; Kaggle overrides only `output_dir` and `device` through explicit CLI arguments, never by editing the committed recipe.

- [ ] **Step 4: Verify the gate and commit**

Run: `python -m pytest tests/pretraining/test_config_and_gates.py -q --basetemp .pytest-tmp/pretraining-config`

Expected: PASS; the current v2 checkout remains demonstrably rejected by the v3 preflight test.

```powershell
git add src/pretraining tests/pretraining/test_config_and_gates.py configs/pretrain_green_sm_pilot.json configs/pretrain_green_sm_full.json
git commit -m "feat: add Green SM pretraining config gate"
```

### Task 2: Build deterministic, immutable multimodal masking

**Files:**
- Create: `src/pretraining/masking.py`
- Create: `tests/pretraining/test_masking.py`

**Interfaces:**
- Consumes: a v3 `FoundationBatch`, special token IDs, modality IDs from `PretrainingConfig`, and `(seed, global_step)`.
- Produces:

```python
@dataclass(frozen=True)
class MaskedFieldTargets:
    selected_mask: torch.BoolTensor              # [B, L]
    original_input_ids: torch.LongTensor         # [B, L]
    original_numeric_values: torch.FloatTensor   # [B, L]
    original_numeric_missing: torch.BoolTensor   # [B, L]
    original_geo_values: torch.FloatTensor       # [B, L, 2]
    original_geo_missing: torch.BoolTensor       # [B, L]

@dataclass(frozen=True)
class CorruptedBatch:
    batch: FoundationBatch
    targets: MaskedFieldTargets

def corrupt_for_masked_reconstruction(
    batch: FoundationBatch, *, routing: RoutingIds, probability: float,
    seed: int, global_step: int,
) -> CorruptedBatch: ...
```

- [ ] **Step 1: Write failing masking tests**

```python
def test_masking_neutralizes_continuous_targets_without_mutating_source_batch(v3_batch) -> None:
    corrupted = corrupt_for_masked_reconstruction(
        v3_batch, routing=ROUTING, probability=1.0, seed=7, global_step=3
    )
    assert not corrupted.targets.selected_mask[0, 0]  # CLS
    assert not corrupted.targets.selected_mask[0, 3]  # SEP
    assert corrupted.batch.numeric_values[0, NUMERIC_POS] == 0.0
    assert corrupted.batch.numeric_missing_mask[0, NUMERIC_POS]
    torch.testing.assert_close(corrupted.targets.original_numeric_values, v3_batch.numeric_values)
    torch.testing.assert_close(v3_batch.input_ids, ORIGINAL_INPUT_IDS)
```

Add deterministic same-seed assertions, different-step variation, PAD protection, geo coordinate neutralization, and `probability=0`/`1` boundaries.

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `python -m pytest tests/pretraining/test_masking.py -v --basetemp .pytest-tmp/pretraining-masking`

Expected: FAIL because `corrupt_for_masked_reconstruction` is absent.

- [ ] **Step 3: Implement copy-on-write corruption**

Select only `attention_mask=True` positions whose `input_id` is not PAD/CLS/SEP and whose modality is categorical, text, numeric, or geo. Clone every tensor that is changed and preserve targets in `MaskedFieldTargets`; do not mutate `targets` from the source batch. Set selected IDs to `SPECIAL_TOKEN_IDS["MASK"]`. For numeric/geo, neutralize continuous inputs exactly as in the locked decision and retain the original missing state for BCE labels. Raise if a selected modality lacks the v3 continuous field it requires.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tests/pretraining/test_masking.py -q --basetemp .pytest-tmp/pretraining-masking`

Expected: PASS with no in-place mutation of the input batch.

```powershell
git add src/pretraining/masking.py tests/pretraining/test_masking.py
git commit -m "feat: add deterministic multimodal masking"
```

### Task 3: Implement fusion, prediction heads, and correctly masked losses

**Files:**
- Modify: `src/models/objectives.py`
- Create: `tests/pretraining/test_objectives.py`

**Interfaces:**
- Consumes: `FoundationOutput`, `MaskedFieldTargets`, uncorrupted `FoundationBatch.targets`, vocabulary cardinalities, and routing IDs from Tasks 1–2.
- Produces:

```python
class UserEmbeddingFusion(nn.Module):
    def forward(self, long_term: torch.Tensor, short_term: torch.Tensor) -> torch.Tensor: ...

class PretrainingHeads(nn.Module):
    def forward_masked(self, hidden_states: torch.Tensor) -> MaskedPredictions: ...
    def forward_next_action(self, z_user: torch.Tensor) -> torch.Tensor: ...

@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    masked_categorical: torch.Tensor
    numeric_value: torch.Tensor
    numeric_missing: torch.Tensor
    geo_coordinate: torch.Tensor
    geo_missing: torch.Tensor
    next_action: torch.Tensor
    counts: Mapping[str, int]

def compute_pretraining_loss(...) -> LossBreakdown: ...
```

- [ ] **Step 1: Write failing objective tests**

```python
def test_next_action_loss_ignores_invalid_rows(tiny_outputs, targets) -> None:
    targets["next_action_valid_mask"] = torch.tensor([True, False])
    loss = compute_pretraining_loss(...)
    assert loss.counts["next_action"] == 1
    assert torch.isfinite(loss.total)

def test_next_action_backpropagates_to_user_id_adapter(tiny_model, v3_batch) -> None:
    output = tiny_model(v3_batch, attention_mode="next_action")
    z_user = fusion(output.long_term, output.short_term)
    loss = F.cross_entropy(heads.forward_next_action(z_user), valid_labels)
    loss.backward()
    assert tiny_model.user_adapter.embedding.weight.grad.abs().sum() > 0
```

Also assert categorical loss ignores non-selected and PAD positions; observed/missing numeric targets route to different components; coordinate loss excludes original-missing geo; every zero-count component is finite autograd-safe zero; and all fused/output widths equal `d_model`.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/pretraining/test_objectives.py -v --basetemp .pytest-tmp/pretraining-objectives`

Expected: FAIL because the heads and loss interface are absent.

- [ ] **Step 3: Implement the heads and weighted reduction**

Use one value-vocabulary CE head for all selected categorical/text/geo-cell IDs, one scalar numeric-value head, one numeric-missingness head, one two-coordinate geo head, one geo-missingness head, and one value-vocabulary next-action head. `UserEmbeddingFusion` is `Linear(2*d_model, d_model)` followed by `LayerNorm`; do not concatenate it directly into a downstream-specific width. Filter each loss at its exact eligible mask and divide by that component's count before applying its config weight. Filter next action only by `batch.targets["next_action_valid_mask"]`; label `0` is never treated as a valid class.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tests/pretraining/test_objectives.py -q --basetemp .pytest-tmp/pretraining-objectives`

Expected: PASS, including the non-zero `UserIdAdapter` gradient regression test.

```powershell
git add src/models/objectives.py tests/pretraining/test_objectives.py
git commit -m "feat: add foundation pretraining objectives"
```

### Task 4: Assemble train-only vocabulary, streaming batches, and one safe optimization step

**Files:**
- Create: `src/pretraining/runner.py`
- Modify: `tests/pretraining/conftest.py`
- Create: `tests/pretraining/test_runner.py`

**Interfaces:**
- Consumes: `PretrainingConfig`, `PreflightResult`, `FoundationPretrainDataset`, `foundation_collate`, `semantic_tokens`, `fit_vocabulary`, Task 2 corruption, and Task 3 modules.
- Produces:

```python
@dataclass(frozen=True)
class CorpusRunInputs:
    vocabulary: VocabularyArtifacts
    model_config: FoundationModelConfig
    train: Iterable[EncodedSample]
    validation: Iterable[EncodedSample]

def build_corpus_run_inputs(config: PretrainingConfig, preflight: PreflightResult, *, corpus: Literal["linked", "control"]) -> CorpusRunInputs: ...
def train_one_step(state: TrainingState, samples: list[EncodedSample], *, global_step: int) -> LossBreakdown: ...
def evaluate_split(state: TrainingState, split: str, *, max_batches: int | None = None) -> Mapping[str, float]: ...
```

- [ ] **Step 1: Write failing data/step tests**

```python
def test_build_run_inputs_never_iterates_validation_or_test_when_fitting_vocab(monkeypatch, config) -> None:
    seen_splits = []
    monkeypatch.setattr(runner, "iter_normalized_samples", recording_loader(seen_splits))
    runner.build_run_inputs(config, preflight)
    assert "validation" not in seen_splits
    assert "test" not in seen_splits

def test_train_one_step_updates_model_and_reports_finite_components(tiny_state, encoded_samples) -> None:
    before = clone_trainable_parameters(tiny_state)
    losses = train_one_step(tiny_state, encoded_samples, global_step=0)
    assert torch.isfinite(losses.total)
    assert parameters_changed(before, tiny_state)
```

Add a test that each corpus only consumes its own train samples while fitting vocabulary; test unknown `sequence`/`numeric`/`geo` routing IDs fail before model creation; test `DatasetConfig(next_action_prediction=True, max_tokens=256, max_event_count=30)` is used; test validation runs without optimizer mutation; test a batch with zero valid next-action rows still trains reconstruction safely.

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `python -m pytest tests/pretraining/test_runner.py -v --basetemp .pytest-tmp/pretraining-runner`

Expected: FAIL because `runner` is absent.

- [ ] **Step 3: Implement bounded-memory input construction and step execution**

For each corpus, fit a vocabulary by streaming `semantic_tokens` from only that corpus's **train** iterator into `fit_vocabulary`; do not materialize canonical samples or token records. Store that corpus artifact under its configured output directory before constructing its dataset. Build model cardinalities/routing IDs exclusively from that artifact, then call `validate_v3_batch_contract` on the first collated batch. Batch with a bounded deterministic shuffle buffer; mixed-source batches remain forbidden. A training step performs the two forwards defined above, combines Task 3 losses, verifies all gradients/parameters are finite before `optimizer.step()`, and returns scalar losses/counts. Validation uses `torch.no_grad()`, fixed masking seed, and no optimizer/scheduler mutation.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tests/pretraining/test_runner.py -q --basetemp .pytest-tmp/pretraining-runner`

Expected: PASS; the input test proves no validation/test samples were used to fit preprocessing metadata.

```powershell
git add src/pretraining/runner.py tests/pretraining/conftest.py tests/pretraining/test_runner.py
git commit -m "feat: add streaming foundation pretraining step"
```

### Task 5: Add auditable checkpointing, restore, and frozen embedding export

**Files:**
- Create: `src/pretraining/artifacts.py`
- Modify: `src/pretraining/runner.py`
- Modify: `tests/pretraining/test_runner.py`

**Interfaces:**
- Consumes: model/fusion/head/optimizer/scheduler state, `PreflightResult`, config digest, vocabulary SHA-256, and validation metrics.
- Produces:

```python
def save_checkpoint(state: TrainingState, *, kind: Literal["last", "best"], output_dir: Path) -> Path: ...
def load_frozen_embedder(checkpoint_path: Path, *, expected: CheckpointIdentity) -> FrozenFoundationEmbedder: ...

class FrozenFoundationEmbedder(nn.Module):
    def embed_users(self, batch: FoundationBatch) -> dict[str, torch.Tensor]: ...
    # returns long_term, short_term, and z_user; all parameters require_grad=False
```

- [ ] **Step 1: Write failing artifact tests**

```python
def test_load_frozen_embedder_rejects_manifest_mismatch(tmp_path, trained_state, identity) -> None:
    checkpoint = save_checkpoint(trained_state, kind="best", output_dir=tmp_path)
    with pytest.raises(ValueError, match="linked_manifest_sha256"):
        load_frozen_embedder(checkpoint, expected=replace(identity, linked_manifest_sha256="wrong"))

def test_loaded_embedder_is_frozen_and_returns_shared_feature(trained_state, v3_batch, tmp_path) -> None:
    embedder = load_frozen_embedder(save_checkpoint(...), expected=identity)
    output = embedder.embed_users(v3_batch)
    assert set(output) == {"long_term", "short_term", "z_user"}
    assert output["z_user"].shape[-1] == trained_state.model.config.d_model
    assert not any(parameter.requires_grad for parameter in embedder.parameters())
```

Also assert a checkpoint contains no absolute Kaggle path or secret, stores global step/RNG/optimizer/scheduler for exact resume, persists component loss history and counts, and fails on changed config digest or vocabulary SHA.

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `python -m pytest tests/pretraining/test_runner.py -v --basetemp .pytest-tmp/pretraining-artifacts`

Expected: FAIL because checkpoint identity and frozen export are absent.

- [ ] **Step 3: Implement checkpoint and manifest pair**

Write `last.pt`/`best.pt` atomically and adjacent JSON `checkpoint_manifest.json`. The JSON must record checkpoint SHA-256, run/config digest, both corpus manifest SHAs, shared vocabulary SHA, model/routing dimensions, step/epoch, seed, objective weights, data rejection count, validation metric used for selection, and software timestamp. `best` is selected by lowest total validation loss, with global step as deterministic tie-breaker. `load_frozen_embedder` verifies every expected identity field before loading, sets `eval()`, sets every parameter `requires_grad_(False)`, and exposes `z_user` for both downstream tasks.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tests/pretraining/test_runner.py -q --basetemp .pytest-tmp/pretraining-artifacts`

Expected: PASS with a restored frozen `z_user` artifact and fail-closed identity checks.

```powershell
git add src/pretraining/artifacts.py src/pretraining/runner.py tests/pretraining/test_runner.py
git commit -m "feat: add auditable foundation checkpoints"
```

### Task 6: Expose the CLI, pilot protocol, and reproducible Kaggle handoff

**Files:**
- Create: `src/pretraining/__main__.py`
- Modify: `src/pretraining/runner.py`
- Create: `docs/kaggle_foundation_pretraining_runbook.md`
- Modify: `tests/pretraining/test_runner.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces:

```text
python -m src.pretraining --config configs/pretrain_green_sm_pilot.json --device cuda --output-dir /kaggle/working/green-sm-pilot
python -m src.pretraining --config ... --resume /kaggle/working/.../last.pt
python -m src.pretraining --config ... --evaluate-test --checkpoint .../best.pt
```

- [ ] **Step 1: Write failing CLI smoke tests**

```python
def test_cli_runs_one_cpu_step_and_writes_manifest(tmp_path, patched_tiny_roots) -> None:
    result = subprocess.run([... "--max-steps", "1", "--device", "cpu"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "checkpoint_manifest.json").is_file()

def test_cli_refuses_to_run_when_preflight_fails(tmp_path, v2_roots) -> None:
    result = subprocess.run([...], capture_output=True, text=True)
    assert result.returncode != 0
    assert "geo_values" in result.stderr
```

- [ ] **Step 2: Run the smoke tests and confirm they fail**

Run: `python -m pytest tests/pretraining/test_runner.py -v --basetemp .pytest-tmp/pretraining-cli`

Expected: FAIL because the module CLI is absent.

- [ ] **Step 3: Implement CLI and write the runbook**

The CLI must print config digest, linked/control manifest SHAs, routing IDs, device, parameter count, each logged component loss/count, and final checkpoint SHA. It may override only `output_dir`, `device`, `max_steps`, and `resume`; all other overrides require a copied JSON config, preserving reproducibility. It must execute linked pilot first, then a separate control pilot with the same architecture/config digest but its own train-fitted vocabulary. The runbook must give Kaggle upload/mount paths, pilot resource settings (128 d-model, 4 layers, batch 32, max 1,000 steps), success criteria (finite losses/gradients, lower trailing total-loss mean than initial mean, non-zero user-adapter gradient, manifests for both roots), and the explicit rule that full generation/training starts only after the pilot review.

- [ ] **Step 4: Run final verification and commit**

Run:

```powershell
python -m pytest tests/pretraining tests/models -q --basetemp .pytest-tmp/pretraining-final
python -m pytest tests/foundation_input -q --basetemp .pytest-tmp/pretraining-foundation-input
python -m src.pretraining --help
```

Expected: all scoped tests pass and CLI help exits 0. If whole-repository pytest still stops at the pre-existing `tests/test_imports.py` relative-import collection failure, report it as unrelated and do not modify it in this branch.

```powershell
git add src/pretraining/__main__.py src/pretraining/runner.py docs/kaggle_foundation_pretraining_runbook.md tests/pretraining/test_runner.py
git commit -m "feat: add foundation pretraining CLI"
```

## Final Handoff Checklist

- [ ] Run Task 6's scoped verification after the upstream v3 contract is merged.
- [ ] Run exactly one linked and one paired-control Kaggle pilot; compare their manifests to prove matching configuration and separate corpus/vocabulary SHAs.
- [ ] Hand Member 4 the linked `best.pt`, `checkpoint_manifest.json`, copied `vocabulary.json`, and `FrozenFoundationEmbedder` loader API. Member 4 freezes it and uses only `z_user` as the shared user feature.
- [ ] Hand Member 3's pilot loss trace and manifest to the baseline/evaluation team; do not claim downstream quality from pretraining loss alone.
- [ ] Do not generate the full corpus or start full pretraining until the team reviews the pilot's loss, gradient, geo-loss counts, checkpoint identity, and downstream compatibility.

## Plan Self-Review

- **Spec coverage:** paired linked/control runs, masked reconstruction, next-action prediction, valid geo regression, long/short fusion, frozen downstream reuse, train-only preprocessing, time-respecting validation, checkpoint identity, and pilot gates are each assigned to Tasks 1–6.
- **Deliberate boundary:** the geo tokenizer/batch/adapter changes are not silently duplicated here; they are a named Member 1/2 prerequisite and Task 1/6 make their absence a hard failure.
- **Type consistency:** Task 2 produces `CorruptedBatch`/`MaskedFieldTargets`; Task 3 consumes them for `LossBreakdown`; Task 4 consumes `LossBreakdown`; Task 5 serializes the `TrainingState` and exposes `FrozenFoundationEmbedder`; Task 6 only invokes those public interfaces.
- **Review-focus coverage:** each of the five risky conditions is mapped to an explicit named test in its owning task.
