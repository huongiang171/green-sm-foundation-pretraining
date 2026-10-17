# Green SM Food Candidate Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and demonstrate a frozen-Foundation Food ranker that scores every immutable Food candidate supplied for a Green SM request, while preserving the shared evaluator/output contract.

**Architecture:** The task consumes the checksum-bound `green_sm_food_opportunities_v1` artifact and resolves each `foundation_sample_id` to the causal Foundation snapshot that generated it. A frozen linked-corpus checkpoint returns long- and short-term user embeddings; a new FoodRanker fuses them with item, merchant, category, price, and location features to produce one uncalibrated ranking score per candidate. Training is listwise over each request's complete candidate set; validation/test outputs are label-free and passed to the common output writer.

**Tech Stack:** Python 3.10+, PyTorch, PyArrow/Parquet, PyYAML, existing `FoundationPretrainDataset`, `FrozenFoundationEmbedder`, and `OutputContractV1`.

**Spec:** `docs/superpowers/specs/2026-10-15-green-sm-simulated-corpus-design.md`

## Global Constraints

- Use `green_sm_sim_linked_v2/food_opportunities`, not `data/downstream1_meituan_data`; Meituan users, vocabulary, and Foundation corpus do not match the Green SM checkpoint.
- Resolve each request only through its exact `foundation_sample_id`, user ID, split, and cutoff timestamp; any mismatch is a hard error before training or inference.
- The candidate set is handed off by Data: never resample, add, remove, deduplicate silently, or create candidates in downstream code.
- Foundation is frozen by default and must load only after checkpoint SHA, selected-checkpoint manifest, vocabulary SHA, and Foundation-manifest linkage pass verification.
- Fit candidate categorical vocabularies and numeric normalizers on Food **train** only; validation/test unseen category/cell values map to `UNK`, while high-cardinality item and merchant IDs use deterministic field-namespaced hashing.
- Food scores are finite, real-valued ranking scores; unlike Ads, they are not probabilities and must not be clipped or sigmoid-transformed for export.
- Prediction parquet is label-free and exactly matches the supplied candidate universe. Train/validation require listwise labels; test prediction must neither validate nor consume a label. Labels may appear only in a separate local demo artifact when that private/offline field is present.
- Every output writer call must isolate validation and test artifacts: write to `<run_dir>/predictions/<split>/` and reject a non-empty split directory. The common writer must also use split-specific manifest/report names so a direct caller cannot overwrite another split.
- Baseline/Evaluation owns Recall/NDCG/MRR computation. This work must call its published evaluator command only after contract validation; it must not reimplement metrics.
- Keep all data/checkpoint/output paths in YAML; no artifact path is hard-coded in Python. The same code must work for pilot and full corpus versions.
- Use TDD for every behavior change, `python -m pytest --basetemp .pytest-tmp/<task>`, and `python -m ruff check` before each commit. Do not reformat unrelated code.

## Review Focus

- A Food row whose candidate becomes available after its cutoff must be rejected before snapshot loading (Task 2 regression test).
- A repeated `(request_id, candidate_id)` or a request with no positive/only positive candidates must be rejected rather than silently changing the ranking set (Task 2 regression tests).
- A snapshot with the right ID but a different user, split, or cutoff must never be embedded (Task 3 regression tests).
- Validation/test-only category, geo-cell, or price outside train statistics must not expand training vocabularies or yield non-finite feature tensors (Task 2 regression test).
- An output with a missing/extra candidate, a duplicate key, NaN/Inf score, or a Food score outside `[0,1]` must be handled correctly: reject the first three but accept the last (Task 1 and Task 5 regression tests).
- Padding must never become an exported candidate and test prediction must succeed if its handoff omits labels (Task 2 and Task 5 regression tests).

---

## File structure and ownership

- `src/contracts/output.py` — shared exact-candidate writer/validator extended from Ads-only to task-aware Food and Ads rules.
- `src/downstream/food_recommendation.py` — Food handoff validation, causal snapshot resolution, train-only candidate feature preparation, batches, ranker, loss, training/inference CLI, and demo JSON writer.
- `configs/food_recommendation.yaml` — immutable-path recipe for either pilot or full Green SM artifact; concrete checksums are filled from the selected artifact manifests, never placeholders at run time.
- `docs/kaggle_food_recommendation_runbook.md` — Kaggle commands to mount data/checkpoint, train, export, invoke the evaluator, and inspect a demo request.
- `tests/contracts/test_output.py` — Food-specific shared-contract tests while retaining existing Ads behavior.
- `tests/downstream/test_food_recommendation.py` — Food data, model, frozen-checkpoint boundary, CLI, and demo regression tests.

### Task 1: Make Output Contract v1 task-aware without breaking Ads

**Files:**
- Modify: `src/contracts/output.py`
- Modify: `tests/contracts/test_output.py`

**Interfaces:**
- Consumes: `ExpectedCandidate`, `PredictionRow`, a verified candidate universe, and run metadata.
- Produces: `VerifiedOpportunityIndex(..., task: Literal["food_recommendation", "ad_optimization"], ...)` and `OutputContractV1.write_predictions(...)` for both tasks.
- Later tasks use: Food writer behavior: required metadata contains `food_manifest_sha256`, `score_semantics == "ranking"`, and finite scores have no `[0,1]` restriction.

- [ ] **Step 1: Write failing Food contract tests**

```python
def test_food_writer_accepts_unbounded_finite_ranking_scores(tmp_path: Path) -> None:
    index = VerifiedOpportunityIndex(
        [ExpectedCandidate("r1", "item1", "u1", "2026-09-01T08:00:00+07:00", "test")],
        task="food_recommendation", data_version="green_sm_sim_pilot_v2",
        source_checksums={"food_manifest_sha256": "a" * 64},
    )
    paths = OutputContractV1.write_predictions(
        [PredictionRow("r1", "item1", 3.75, "test")], expected_candidates=index,
        output_dir=tmp_path, split="test", run_metadata=_food_metadata(),
    )
    assert paths.predictions.is_file()

def test_food_writer_rejects_missing_candidate_and_nan_score(tmp_path: Path) -> None:
    ...
```

Also retain/extend an Ads test asserting Ads still rejects score `1.01`, and add a test that writes validation then test into the same requested root and proves their prediction, manifest, and report paths are distinct and intact.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/contracts/test_output.py --basetemp .pytest-tmp/food-output-red -q`

Expected: FAIL because `VerifiedOpportunityIndex` currently accepts only `ad_optimization` and the writer only accepts probability semantics.

- [ ] **Step 3: Implement task-specific validation in `src/contracts/output.py`**

Use one supported-task map rather than a Food-only fork. Keep the common primary key, exact candidate-set check, deterministic write order, injected request metadata, atomic Parquet/JSON writes, and label-free schema unchanged. Require these task-specific metadata/score rules:

```python
TASK_RULES = {
    "food_recommendation": {"manifest_key": "food_manifest_sha256", "score_semantics": "ranking"},
    "ad_optimization": {"manifest_key": "ads_manifest_sha256", "score_semantics": "probability"},
}
```

Apply the `[0, 1]` score range only to Ads. Require finite scores for both tasks. Derive `manifest_path` and `report_path` from `split` (for example, `run_manifest.validation.json` and `validation_report.validation.json`) so writing one split cannot overwrite the other even when callers share an output root.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest tests/contracts/test_output.py --basetemp .pytest-tmp/food-output-green -q`

Expected: PASS; Food accepts `3.75`, rejects NaN/missing/extra/duplicate candidates, Ads keeps its probability guard, and validation/test metadata remain separate.

- [ ] **Step 5: Commit the shared contract change**

```bash
git add src/contracts/output.py tests/contracts/test_output.py
git commit -m "feat: support food ranking output contract"
```

### Task 2: Add strict Food opportunity loading and train-only candidate features

**Files:**
- Create: `src/downstream/food_recommendation.py`
- Create: `tests/downstream/test_food_recommendation.py`

**Interfaces:**
- Consumes: `food_opportunities.{train,validation,test}.parquet` whose declared schema is `green_sm_food_opportunities_v1`.
- Produces: `FoodOpportunity`, `validate_food_opportunities`, `load_green_sm_food_opportunities`, `FoodFeatureArtifacts`, and `build_food_batches`.
- Later tasks use: a request-major `FoodBatch` containing `FoundationBatch`, candidate feature tensors, `positive_mask`, `candidate_mask`, and original opportunities in deterministic request/candidate order.

- [ ] **Step 1: Write failing data-contract tests**

```python
def test_food_loader_maps_the_real_green_sm_parquet_schema_to_a_typed_opportunity(tmp_path: Path) -> None:
    ...
    row = load_green_sm_food_opportunities(tmp_path, split="train")[0]
    assert row.item_id == "food_item_411"
    assert row.merchant_id == "merchant_091"
    assert row.cutoff_timestamp == 1_788_224_410  # UTC epoch seconds from Arrow Asia/Ho_Chi_Minh timestamp

def test_food_validation_rejects_future_candidate_and_duplicate_candidate() -> None:
    with pytest.raises(ValueError, match="candidate_available_timestamp.*cutoff_timestamp"):
        validate_food_opportunities([_food_opportunity(candidate_available_timestamp=101)], "train")
    with pytest.raises(ValueError, match="duplicate candidate.*request-1"):
        validate_food_opportunities([_food_opportunity(), _food_opportunity(is_positive=0)], "train")

def test_candidate_features_are_fit_only_on_train_and_keep_eval_oov_finite() -> None:
    artifacts = FoodFeatureArtifacts.fit([_food_opportunity(category="noodles", geo_cell="cell_01", price=50_000)])
    batch = build_food_batches([_food_opportunity(split="validation", category="new", geo_cell="cell_99", price=99_000)], ...)
    assert batch.category_ids.tolist() == [[1]]
    assert torch.isfinite(batch.price_values).all()

def test_test_inference_does_not_require_or_read_is_positive() -> None:
    rows = [_food_opportunity(split="test", is_positive=None)]
    assert validate_food_opportunities(rows, expected_split="test", require_labels=False)

def test_food_batch_never_promotes_padding_to_a_candidate() -> None:
    batch = build_food_batches(_variable_length_requests(), ...)[0]
    assert not batch.positive_mask[~batch.candidate_mask].any()

def test_feature_artifacts_round_trip_preserves_train_normalizers_exactly(tmp_path: Path) -> None:
    artifacts = FoodFeatureArtifacts.fit(_train_rows())
    artifacts.save(tmp_path / "feature_artifacts.json")
    assert FoodFeatureArtifacts.load(tmp_path / "feature_artifacts.json") == artifacts
```

The fixture set must include 20 candidates and two positives for at least one request, plus a variable-length request fixture to test `candidate_mask`; do not hard-code real data's `20` or `2` into the loader.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/downstream/test_food_recommendation.py --basetemp .pytest-tmp/food-data-red -q`

Expected: FAIL because the Food module does not exist.

- [ ] **Step 3: Implement the typed Food handoff and feature artifacts**

Implement these public types/functions in `src/downstream/food_recommendation.py`:

```python
@dataclass(frozen=True)
class FoodOpportunity:
    request_id: str
    user_id: str
    foundation_sample_id: str
    cutoff_timestamp: int  # UTC epoch seconds, converted only by _timestamp(...)
    candidate_id: str
    split: Literal["train", "validation", "test"]
    candidate_available_timestamp: int  # UTC epoch seconds
    item_id: str
    merchant_id: str
    category: str
    geo_cell: str
    latitude: float
    longitude: float
    item_price: float
    is_positive: int | None
    is_hard_negative: bool

class FoodFeatureArtifacts:
    @classmethod
    def fit(cls, rows: Sequence[FoodOpportunity]) -> "FoodFeatureArtifacts": ...
    def encode(self, rows: Sequence[FoodOpportunity]) -> "FoodCandidateFeatures": ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> "FoodFeatureArtifacts": ...

def validate_food_opportunities(
    rows: Sequence[FoodOpportunity], expected_split: Literal["train", "validation", "test"], *,
    require_labels: bool
) -> tuple[FoodOpportunity, ...]: ...

def load_green_sm_food_opportunities(
    root: str | Path, *, split: Literal["train", "validation", "test"]
) -> tuple[FoodOpportunity, ...]: ...
```

The loader must map the artifact's actual fields exactly: `candidate_id -> item_id`, `candidate_category -> category`, `candidate_geo_cell -> geo_cell`, `candidate_lat/candidate_lon -> latitude/longitude`, and Arrow timezone-aware `cutoff_timestamp`/`candidate_available_timestamp -> UTC epoch seconds` through one `_timestamp(value, field=...)` helper. Fixtures must write those actual Parquet field names and timezone-aware Arrow timestamps.

For every split, validation enforces non-empty IDs, exact split, finite coordinates/price, availability at or before cutoff, exactly one user/sample/cutoff per request, and unique candidate IDs. With `require_labels=True` (train/validation), additionally require binary labels plus at least one positive and one negative per request. With `require_labels=False` (test prediction), do not access `is_positive` at all. Fit `<PAD>=0`, `<UNK>=1` categorical vocabularies for `category` and `geo_cell`; create a Food-local `stable_food_hash_id(field_name, value, bucket_count)` with namespace `green-sm-food-v1`, rather than importing Ads' `taobao-ads-v1` helper. Normalize price and coordinates with train-only mean/std, using scale `1.0` when a train feature has zero standard deviation.

- [ ] **Step 4: Implement `FoodBatch` and `build_food_batches`**

```python
@dataclass
class FoodBatch:
    foundation_batch: FoundationBatch
    item_hash_ids: torch.LongTensor
    merchant_hash_ids: torch.LongTensor
    category_ids: torch.LongTensor
    geo_cell_ids: torch.LongTensor
    price_values: torch.FloatTensor
    geo_values: torch.FloatTensor
    positive_mask: torch.BoolTensor
    candidate_mask: torch.BoolTensor
    opportunities: tuple[tuple[FoodOpportunity, ...], ...]

def build_food_batches(
    rows: Sequence[FoodOpportunity], *, encoded_samples_by_id: Mapping[str, EncodedSample],
    feature_artifacts: FoodFeatureArtifacts, batch_size: int, item_hash_buckets: int,
    merchant_hash_buckets: int
) -> tuple[FoodBatch, ...]: ...
```

Group by `request_id`, sort requests and candidates deterministically, collate only one matching Foundation sample per request, and pad candidates with zero IDs/value tensors plus `candidate_mask=False`. Require `positive_mask <= candidate_mask`; for label-free test batches populate an all-false `positive_mask` which is never sent to listwise loss. Add an assertion that flattening for export iterates original opportunities only, never padded columns.

- [ ] **Step 5: Run Food data/batch tests to verify they pass**

Run: `python -m pytest tests/downstream/test_food_recommendation.py --basetemp .pytest-tmp/food-data-green -q`

Expected: PASS for real-schema loading, UTC timestamp conversion, integrity validation, train-only vocabulary/normalization, label-free test validation, and padded request-major batches.

- [ ] **Step 6: Commit the data boundary**

```bash
git add src/downstream/food_recommendation.py tests/downstream/test_food_recommendation.py
git commit -m "feat: add Green SM Food opportunity batches"
```

### Task 3: Resolve only causal Foundation snapshots for Food

**Files:**
- Modify: `src/downstream/food_recommendation.py`
- Modify: `tests/downstream/test_food_recommendation.py`

**Interfaces:**
- Consumes: typed Food rows, immutable Green SM Foundation root, Foundation vocabulary artifact, and tokenizer contract from `load_green_sm_tokenizer_config`.
- Produces: `resolve_green_sm_food_samples(...) -> dict[str, EncodedSample]`.
- Later tasks use: the returned map as the only allowable input to `build_food_batches`.

- [ ] **Step 1: Write failing causal-join tests**

```python
def test_food_snapshot_resolution_rejects_user_cutoff_or_split_mismatch(monkeypatch) -> None:
    canonical = _canonical(sample_id="s1", user_id="different-user", split="train", cutoff=100)
    ...
    with pytest.raises(ValueError, match="s1.*user_id"):
        resolve_green_sm_food_samples([_food_opportunity(foundation_sample_id="s1")], ...)

def test_food_snapshot_resolution_returns_only_the_exact_matching_sample(monkeypatch) -> None:
    ...
    assert resolved == {"s1": encoded}
```

- [ ] **Step 2: Run the causal-join tests to verify they fail**

Run: `python -m pytest tests/downstream/test_food_recommendation.py --basetemp .pytest-tmp/food-causal-red -q`

Expected: FAIL because `resolve_green_sm_food_samples` is not implemented.

- [ ] **Step 3: Implement causal snapshot resolution**

```python
def resolve_green_sm_food_samples(
    rows: Sequence[FoodOpportunity], *, foundation_root: str | Path,
    vocabulary_path: str | Path, split: Literal["train", "validation", "test"], max_tokens: int
) -> dict[str, EncodedSample]: ...
```

Follow the Green SM Ads bridge pattern: load `VocabularyArtifacts`, load the published tokenizer config, iterate normalized canonical samples with `validate=True`, and re-encode only referenced samples. Reject missing sample IDs, duplicate/inconsistent request linkage, user mismatch, cutoff mismatch, or split mismatch.

- [ ] **Step 4: Run the causal-join tests to verify they pass**

Run: `python -m pytest tests/downstream/test_food_recommendation.py --basetemp .pytest-tmp/food-causal-green -q`

Expected: PASS; no candidate may be paired with a different causal snapshot.

- [ ] **Step 5: Commit the causal join**

```bash
git add src/downstream/food_recommendation.py tests/downstream/test_food_recommendation.py
git commit -m "feat: resolve causal Food foundation snapshots"
```

### Task 4: Implement frozen-Foundation FoodRanker and multi-positive listwise loss

**Files:**
- Modify: `src/downstream/food_recommendation.py`
- Modify: `tests/downstream/test_food_recommendation.py`

**Interfaces:**
- Consumes: `FoodBatch`, `FrozenFoundationEmbedder.embed_users`, and `d_model` read from the verified checkpoint.
- Produces: `FoodRanker.forward(...) -> torch.Tensor` with shape `[B, C]`, `masked_multi_positive_listwise_loss`, and `frozen_foundation_food_logits`.

- [ ] **Step 1: Write failing model/loss tests**

```python
def test_food_ranker_returns_finite_score_for_each_real_or_padded_candidate() -> None:
    ranker = FoodRanker(..., d_model=4, dropout=0.0)
    scores = ranker(torch.ones(2, 4), torch.zeros(2, 4), **_food_features())
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores).all()

def test_multi_positive_listwise_loss_prefers_higher_positive_scores() -> None:
    good = masked_multi_positive_listwise_loss(
        torch.tensor([[3.0, 2.0, -2.0]]), torch.tensor([[True, True, False]]), torch.tensor([[True, True, True]])
    )
    bad = masked_multi_positive_listwise_loss(
        torch.tensor([[-2.0, -1.0, 3.0]]), torch.tensor([[True, True, False]]), torch.tensor([[True, True, True]])
    )
    assert good < bad

def test_frozen_foundation_parameters_receive_no_gradient() -> None:
    ...
    assert all(parameter.grad is None for parameter in embedder.parameters())
    assert any(parameter.grad is not None for parameter in ranker.parameters())
```

- [ ] **Step 2: Run the model/loss tests to verify they fail**

Run: `python -m pytest tests/downstream/test_food_recommendation.py --basetemp .pytest-tmp/food-model-red -q`

Expected: FAIL because `FoodRanker` and the listwise loss do not exist.

- [ ] **Step 3: Implement `FoodRanker` and its loss**

```python
class FoodRanker(nn.Module):
    def __init__(self, *, d_model: int, item_hash_buckets: int, merchant_hash_buckets: int,
                 num_categories: int, num_geo_cells: int, dropout: float) -> None: ...
    def forward(self, long_term: torch.Tensor, short_term: torch.Tensor, *,
                item_hash_ids: torch.LongTensor, merchant_hash_ids: torch.LongTensor,
                category_ids: torch.LongTensor, geo_cell_ids: torch.LongTensor,
                price_values: torch.FloatTensor, geo_values: torch.FloatTensor) -> torch.Tensor: ...

def masked_multi_positive_listwise_loss(
    scores: torch.Tensor, positive_mask: torch.BoolTensor, candidate_mask: torch.BoolTensor
) -> tuple[torch.Tensor, int]: ...
```

Project `[long_term, short_term]` to a user vector. Embed item/merchant/category/geo-cell, project normalized price and normalized `(lat, lon)` independently, fuse candidate features, then score `(user, candidate)` pairs with an MLP. Implement loss as mean over requests of `logsumexp(scores[valid]) - logsumexp(scores[positive])`; validate masks and require every request to contain both valid positive and valid negative candidates.

- [ ] **Step 4: Implement frozen scoring boundary and rerun tests**

```python
def frozen_foundation_food_logits(
    embedder: FrozenFoundationEmbedder, ranker: FoodRanker, batch: FoodBatch
) -> torch.Tensor: ...
```

Implement the boundary exactly as `with torch.no_grad(): embeddings = embedder.embed_users(batch.foundation_batch)`, followed by `ranker(embeddings["long_term"], embeddings["short_term"], ...)` **outside** the context. `FrozenFoundationEmbedder` remains parameter-frozen; the ranker forward graph must remain differentiable during training. Run: `python -m pytest tests/downstream/test_food_recommendation.py --basetemp .pytest-tmp/food-model-green -q`

Expected: PASS; score dimensions/values and loss ordering are correct, and Foundation remains frozen.

- [ ] **Step 5: Commit ranker and loss**

```bash
git add src/downstream/food_recommendation.py tests/downstream/test_food_recommendation.py
git commit -m "feat: add frozen Foundation Food ranker"
```

### Task 5: Build checksum-gated Food training, prediction export, and CLI

**Files:**
- Modify: `src/downstream/food_recommendation.py`
- Create: `configs/food_recommendation.yaml`
- Modify: `tests/downstream/test_food_recommendation.py`
- Modify: `tests/contracts/test_output.py`

**Interfaces:**
- Consumes: Food/foundation manifests, selected Foundation checkpoint manifest, `FoodFeatureArtifacts`, and `FoodRanker`.
- Produces: `python -m src.downstream.food_recommendation --config ... --mode {validate,train,predict,demo}` plus atomic ranker checkpoints, run manifests, label-free predictions, and a separate demo JSON.

- [ ] **Step 1: Write failing config/preflight/export tests**

```python
def test_food_preflight_rejects_wrong_food_manifest_checksum(tmp_path: Path) -> None: ...

def test_food_preflight_rejects_checkpoint_with_wrong_foundation_identity(tmp_path: Path) -> None: ...

def test_food_prediction_export_has_exact_candidates_and_unbounded_ranking_scores(tmp_path: Path) -> None: ...

def test_food_demo_contains_top_five_and_labels_but_prediction_parquet_does_not(tmp_path: Path) -> None: ...

def test_validation_and_test_exports_do_not_overwrite_each_other(tmp_path: Path) -> None: ...

def test_predict_rejects_food_run_manifest_or_feature_artifact_sha_mismatch(tmp_path: Path) -> None: ...
```

Use miniature Parquet/manifests and a mocked tiny frozen embedder; do not make unit tests depend on Kaggle or a real pretraining run.

- [ ] **Step 2: Run the config/preflight/export tests to verify they fail**

Run: `python -m pytest tests/downstream/test_food_recommendation.py tests/contracts/test_output.py --basetemp .pytest-tmp/food-cli-red -q`

Expected: FAIL because the run config, preflight, train/predict entry points, and Food exporter are absent.

- [ ] **Step 3: Implement immutable config and preflight**

Add `load_food_recommendation_config(path)` and `validate_food_data_handoff(config)`. Require this YAML structure:

```yaml
seed: 42
task: food_recommendation
data:
  data_version: green_sm_sim_pilot_v2
  opportunities_root: /kaggle/input/.../food_opportunities
  manifest_path: /kaggle/input/.../food_opportunities/manifest.json
  foundation_root: /kaggle/input/.../foundation
  foundation_manifest_path: /kaggle/input/.../foundation/manifest.json
  vocabulary_path: /kaggle/input/.../linked/vocabulary.json
  expected_food_manifest_sha256: <manifest sha256>
  expected_foundation_manifest_sha256: <foundation manifest sha256>
  max_tokens: 256
checkpoint:
  path: /kaggle/input/.../linked/best.pt
  manifest_path: /kaggle/input/.../linked/checkpoint_manifest.json
  expected_sha256: <checkpoint sha256>
ranker:
  run_dir: /kaggle/working/artifacts/food/<food-run-id>
  expected_run_manifest_sha256: <required for predict/demo; absent only for a new train>
freeze_foundation: true
model:
  dropout: 0.1
  item_hash_buckets: 65536
  merchant_hash_buckets: 16384
training:
  device: cuda
  batch_size: 16
  epochs: 20
  learning_rate: 0.0003
  weight_decay: 0.00001
  artifact_dir: /kaggle/working/artifacts/food
```

Preflight verifies every declared SHA, Food schema version, split file SHA, Food-to-Foundation manifest link, `checkpoint_manifest.json` identity, selected checkpoint SHA, and `freeze_foundation is True`. It must build the `CheckpointIdentity` from verified manifests rather than accepting manually copied identity values. For `predict` and `demo`, resolve the Food ranker only from `ranker.run_dir/run_manifest.json`; verify the configured run-manifest SHA, its `best.pt` SHA, `feature_artifacts.json` SHA, saved `FoodRanker` dimensions/hash-bucket config, and its recorded Foundation checkpoint identity. Do not infer a sibling `best.pt` path.

- [ ] **Step 4: Implement training and deterministic artifact lifecycle**

Implement `train_food_ranker(...)` to fit feature artifacts only on train, train only ranker parameters, validate listwise loss every epoch, and save a new non-empty-safe run directory containing `feature_artifacts.json`, `best.pt`, `last.pt`, `train_metrics.jsonl`, and `run_manifest.json`. Save `FoodFeatureArtifacts` atomically and include its SHA, the exact `FoodRanker` constructor settings, the relative names/SHA values of `best.pt` and `last.pt`, Foundation checkpoint identity, code revision, manifest SHAs, config digest, and split row/request counts in the Food run manifest. Select best checkpoint by validation listwise loss; do not compute Recall/NDCG/MRR here. Seed Python/NumPy/PyTorch.

- [ ] **Step 5: Implement prediction, evaluator boundary, and demo**

Implement `predict_food_split(...)` for validation/test only. Load the saved FoodRanker and feature artifacts through the verified Food run manifest plus the verified frozen Foundation checkpoint, score exactly supplied candidates, construct a `VerifiedOpportunityIndex(task="food_recommendation", ...)`, then write via `OutputContractV1` to `<ranker.run_dir>/predictions/<split>/`. Test batches must use `require_labels=False`; validation can use labels only for internal training loss, never in the exported table. Invoke a configured Baseline evaluator command only after writer success; preserve its exit code and never calculate its metrics locally.

Implement `write_food_demo(...)` separately: choose a requested test `request_id` or deterministic first test request, sort its 20 candidates by score descending/candidate ID ascending, and write the top five with candidate ID, merchant ID, category, price, geo-cell, score, rank, and offline `is_positive`. It is not an evaluator submission.

- [ ] **Step 6: Add the CLI and verify all Food paths**

```python
def main(argv: Sequence[str] | None = None) -> int: ...
```

Expose `--mode validate`, `--mode train`, `--mode predict --split {validation,test}`, and `--mode demo --request-id <id>`. `predict`/`demo` must require ranker run metadata via config or explicit CLI overrides; `train` creates a fresh run directory. Validate before constructing a model. Map data/config/checkpoint/output errors to non-zero exit codes with clear task/run context.

Run: `python -m pytest tests/downstream/test_food_recommendation.py tests/contracts/test_output.py --basetemp .pytest-tmp/food-cli-green -q`

Expected: PASS; train smoke creates a reloadable Food checkpoint, output has exact label-free candidate rows, and demo is isolated from the evaluator file.

- [ ] **Step 7: Commit the runnable Food task**

```bash
git add src/downstream/food_recommendation.py configs/food_recommendation.yaml \
  tests/downstream/test_food_recommendation.py tests/contracts/test_output.py
git commit -m "feat: train Green SM Food candidate ranker"
```

### Task 6: Document the pilot-to-full Kaggle workflow and final verification

**Files:**
- Create: `docs/kaggle_food_recommendation_runbook.md`
- Modify: `README.md` only if it already has a downstream-command index; otherwise leave it unchanged.

**Interfaces:**
- Consumes: a completed linked Foundation checkpoint and the YAML recipe from Task 5.
- Produces: reproducible commands for validating, training, exporting, evaluator invocation, demo inspection, and switching from pilot to full artifact manifests.

- [ ] **Step 1: Write a failing CLI smoke test for documented commands**

Add a test in `tests/downstream/test_food_recommendation.py` that invokes the exact `--mode validate` command shape against a fixture config and expects a JSON `data_handoff_valid` result.

- [ ] **Step 2: Run it to verify failure, then implement/fix the command path**

Run: `python -m pytest tests/downstream/test_food_recommendation.py::test_food_cli_validate_data_checks_checksum_bound_handoff --basetemp .pytest-tmp/food-runbook -q`

Expected before implementation: FAIL; after Task 5 implementation: PASS.

- [ ] **Step 3: Write the runbook**

Include, in this order: mount/copy immutable linked pilot corpus and selected checkpoint; verify SHA/manifest; run `validate`; run `train`; export validation and test predictions; hand validated outputs to the Baseline evaluator; write/read one demo JSON; archive the four output manifests/checkpoints. State that changing to full corpus changes YAML data paths/checksums/version and creates a new `run_id`, but not Food code.

- [ ] **Step 4: Run final verification**

Run:

```bash
python -m pytest tests/contracts/test_output.py tests/downstream/test_food_recommendation.py --basetemp .pytest-tmp/food-final-contract -q
python -m pytest tests/pretraining tests/models tests/foundation_input --basetemp .pytest-tmp/food-final-foundation -q
python -m ruff check src/contracts/output.py src/downstream/food_recommendation.py tests/contracts/test_output.py tests/downstream/test_food_recommendation.py
```

Expected: all selected tests pass and Ruff reports no violations. On Kaggle after pilot pretraining completes, also run the runbook's `validate` and a two-epoch `train` smoke before the full Food run.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/kaggle_food_recommendation_runbook.md tests/downstream/test_food_recommendation.py
git commit -m "docs: add Green SM Food ranking runbook"
```

## Self-review

- Spec coverage: the plan uses linked Foundation snapshots, frozen long/short embeddings, candidate item/merchant/category/price/location metadata, exact temporal candidate sets, multi-positive ranking, label-free outputs, baseline-owned metrics, and a pilot/full configuration boundary. The linked/control and target-service-only experiments remain experiment configurations after this implementation; they do not require a separate code path.
- Input safety: each Review Focus case is assigned to Task 1, 2, 3, or 5 tests. Full-corpus paths/checksums are configuration-only, so the implementation does not encode pilot row counts.
- Type consistency: Task 2 establishes `FoodOpportunity`, `FoodFeatureArtifacts`, and `FoodBatch`; Task 3 supplies the `EncodedSample` map required by Task 2 batching; Task 4 consumes `FoodBatch`; Task 5 uses all preceding interfaces.
- Scope: this does not modify the simulator, Foundation encoder/pretraining code, Baseline metrics, or Ads ranker behavior. The necessary task-aware extension in shared Output Contract is deliberately tested against existing Ads behavior.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-16-green-sm-food-ranking.md`. Please review the plan. I recommend **Native** execution because the six tasks have tightly coupled interfaces and all changes are isolated to Food plus the shared output contract; a single implementation session can keep the contracts coherent. Does the plan capture what you want, and which approach should we use?
