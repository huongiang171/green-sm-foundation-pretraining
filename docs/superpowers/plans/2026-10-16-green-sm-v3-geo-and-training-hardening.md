# Green SM v3 Geo Contract and Training Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the simulated Green SM linked/control corpora trainable end-to-end by carrying every declared location and service context through the v3 input contract, routing it through the Foundation encoder, and producing reliable pretraining/downstream handoff artifacts.

**Architecture:** A registered location becomes one `geo` token: its categorical value is the `geo_cell`, while its train-normalized `[lat, lon]` pair travels in aligned v3 tensors. `AdapterRegistry` combines normal ID/pair/key features for the cell and role with `GeoAdapter` coordinate features before the shared Transformer. The pretraining runner uses the corpus-declared tokenizer policy rather than the OmniBehavior defaults, then exposes versioned checkpoints and prediction artifacts through stable public modules.

**Tech Stack:** Python 3.11, PyTorch, PyArrow/Parquet, pytest, JSON/YAML, existing `omnibehavior_pretraining_v2` storage format.

**Spec:** `docs/superpowers/specs/2026-10-15-green-sm-simulated-corpus-design.md`

## Global Constraints

- Do not overwrite an existing simulated corpus; `GreenSmConfig.from_json()` only parses, while `build_dataset()` alone rejects a non-empty `output_root`.
- Keep on-disk Foundation storage schema `omnibehavior_pretraining_v2`; the runtime batch contract is v3.
- Corpus reports are immutable descriptions of emitted data. They declare `data_geo_contract_version: 3`; only preflight evaluates whether the currently checked-out runtime can consume it.
- The only geo roles are `profile.home_location`, `profile.work_location`, `context.pickup_location`, `context.dropoff_location`, `context.merchant_location`, `context.delivery_location`, and `context.ad_context_location`.
- A geo token uses `geo_cell` for vocabulary lookup and carries coordinates separately; it must never become two ordinary numeric tokens.
- Fit categorical vocabularies and numeric/geo normalizers from the Foundation **train** split only; persist the result with the vocabulary/checkpoint.
- Linked and control use the same model, tokenizer policy, seed family, train budget, and evaluation schedule.
- Do not bypass the baseline/evaluator team: downstream code writes predictions and reports, but does not implement a substitute official metric evaluator.

## Review Focus

- Malformed or unregistered `*_location` objects must fail during tokenization, rather than silently becoming categorical strings. Covered in Task 2.
- Green SM history contexts must retain only the declared Ride/Food/Delivery/Ads fields and never an arbitrary future/label field. Covered in Task 2.
- A valid geo value and its `geo_cell` must each affect model features; changes to either signal must be observable. Covered in Task 4.
- Non-geo, padding, `CLS`, and `SEP` positions must have `[0.0, 0.0]` plus `geo_missing_mask=False`; a missing geo token alone has `geo_missing_mask=True`. Covered in Task 3.
- Resume, periodic validation, best-checkpoint selection, and downstream loading must preserve checkpoint identity and never select an unverified artifact. Covered in Tasks 6 and 7.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/data/green_sm/build.py` | Publish the corpus-declared Green SM tokenizer/geo metadata alongside the existing manifest. |
| `src/data/green_sm/config.py` | Pure simulation configuration parsing; no output-directory mutation guard. |
| `src/data/green_sm/validate.py` | Report the immutable geo contract declared by the emitted corpus, never the currently imported runtime class. |
| `src/foundation_input/types.py` | Versioned semantic-token, encoded-record, and `FoundationBatch` v3 tensors. |
| `src/foundation_input/tokenizer.py` | Green-SM-aware context allowlist, location validation, semantic geo tokens, and geo normalization propagation. |
| `src/foundation_input/vocab.py` | Train-only geo normalizer and persisted tokenizer/geo contract. |
| `src/foundation_input/dataset.py` | Collate aligned geo tensors without changing event-group/truncation behavior. |
| `src/pretraining/green_sm_contract.py` (new) | Load and validate `preprocessing_config.json`, then construct the exact `TokenizerConfig` for one corpus. |
| `src/models/adapters.py` | Route geo tokens through `GeoAdapter` while retaining categorical cell/role features. |
| `src/models/encoder.py` | Require and validate `geo_modality_id` in the model configuration. |
| `src/pretraining/gates.py` | Fail closed on incomplete data, batch, tokenizer, and model geo contracts. |
| `src/pretraining/runner.py` | Use the corpus tokenizer contract, periodic validation, best/last checkpoint handling, and run manifest. |
| `src/pretraining/artifacts.py` | Versioned checkpoint manifests and verified frozen embedder loading. |
| `src/training/checkpoint.py` (new) | Stable public Member-3 checkpoint loader for downstream tasks. |
| `src/contracts/output.py` (new) | Stable, opportunity-index-validated prediction/run-manifest writer shared by Food and Ads. |
| `src/downstream/ad_optimization.py` | Keep the existing Taobao handoff isolated; consume the public checkpoint API for its unit-level preflight only. |
| `src/downstream/green_sm_ad_optimization.py` (new) | Green-SM-specific Ads opportunity-to-FoundationBatch integration and frozen-foundation training/inference. |
| `tests/data/`, `tests/foundation_input/`, `tests/models/`, `tests/pretraining/`, `tests/downstream/` | Regression, contract, and end-to-end smoke coverage. |

### Task 1: Publish one authoritative Green SM preprocessing contract

**Ownership:** Data team

**Files:**
- Modify: `src/data/green_sm/build.py: _foundation_config()`
- Modify: `src/data/green_sm/config.py: GreenSmConfig.from_json()`
- Modify: `src/data/green_sm/validate.py: _root_report()`
- Modify: `tests/data/test_green_sm_build.py`
- Modify: `tests/data/test_green_sm_config.py`
- Modify: `tests/data/test_green_sm_validate.py`

**Consumes:** the existing fixed 64-cell public catalogue and the seven-role `geo_field_registry`.

**Produces:** `preprocessing_config.json` containing `tokenizer_contract.context_allowlist`, `profile_allowlist`, `geo_field_registry`, coordinate tolerance, a public deterministic `geo_catalogue` mapping each cell to its exact coordinates, and its fingerprint; an immutable report that describes the corpus-declared data contract.

- [ ] **Step 1: Write failing tests for pure config loading and published geo metadata**

  Assert that `GreenSmConfig.from_json("configs/green_sm_sim_pilot.json")` succeeds even after the configured output directory exists and is non-empty; assert `build_dataset()` still rejects that directory. Build a fixture corpus and assert its preprocessing config contains the exact seven fields, an explicit profile allowlist, four Green SM scenario allowlists, coordinate tolerance, all 64 public cells, and a stable catalogue fingerprint. Assert that re-running validation under a runtime with different `FoundationBatch` fields does not change the already-emitted report.

- [ ] **Step 2: Run the focused data tests and verify the current failures**

  Run: `python -m pytest tests/data/test_green_sm_build.py tests/data/test_green_sm_config.py tests/data/test_green_sm_validate.py -q`

  Expected: the config reuse test fails at `GreenSmConfig.from_json`, and the preprocessing-contract assertions fail because the fields are not published.

- [ ] **Step 3: Make configuration parsing pure and emit the public tokenizer contract**

  Remove the non-empty-output-directory check from `GreenSmConfig.from_json`; retain the guard only in `build_dataset`. Replace the static v1 `ROOT_NAMES` constant with a deterministic function derived from `data_version`, so `green_sm_sim_pilot_v2` emits `green_sm_sim_linked_v2` and `green_sm_sim_control_paired_v2`. Extend `_foundation_config()` with a `tokenizer_contract` object:

  ```python
  {
      "context_allowlist": {
          "Ride": ["pickup_location", "dropoff_location"],
          "Food": ["merchant_location", "food_category"],
          "Delivery": ["delivery_location", "delivery_category"],
          "Ads": ["ad_context_location", "placement"],
      },
      "profile_allowlist": ["home_location", "work_location", "budget_bucket", "preferred_service", "active_days_so_far"],
      "geo_field_registry": [...seven exact fully qualified keys...],
      "coordinate_tolerance": 1e-6,
      "geo_catalogue": {"cell_00": {"lat": ..., "lon": ...}, ...},
      "geo_catalogue_fingerprint": "...",
  }
  ```

  This is public static catalogue metadata, not latent user state or a label.

- [ ] **Step 4: Make report readiness precise**

  Replace the runtime-dependent `upstream_v3_geo_contract_ready` report field with `data_geo_contract_version: 3`, `geo_catalogue_fingerprint`, and the emitted preprocessing-config SHA. Read those values from the corpus files, not imported `FoundationBatch` code. Update preflight to require this data declaration and independently inspect runtime batch/model/adapter capabilities. Do not claim model-routing readiness in a data-only report.

- [ ] **Step 5: Run data tests and regenerate the pilot reports only after the contract merge**

  Run: `python -m pytest tests/data/test_green_sm_build.py tests/data/test_green_sm_config.py tests/data/test_green_sm_validate.py -q`

  Expected: PASS. Then run the documented non-overwriting validation/rebuild workflow to create a new immutable pilot root (for example `green_sm_sim_pilot_v2`), never modifying `green_sm_sim_pilot_v1`.

- [ ] **Step 6: Commit the data contract**

  ```powershell
  git add src/data/green_sm tests/data
  git commit -m "feat: publish Green SM v3 geo preprocessing contract"
  ```

### Task 2: Turn Green SM locations and service contexts into validated semantic tokens

**Ownership:** Member 1

**Files:**
- Modify: `src/foundation_input/types.py: KeyValueToken, TokenRecord`
- Modify: `src/foundation_input/tokenizer.py: TokenizerConfig, _modality(), _append_token(), semantic_tokens(), tokenize_sample()`
- Modify: `src/foundation_input/vocab.py: _tokenizer_contract(), VocabularyArtifacts, fit_vocabulary()`
- Create: `src/pretraining/green_sm_contract.py`
- Modify: `tests/foundation_input/test_tokenizer.py`
- Create: `tests/pretraining/test_green_sm_contract.py`

**Consumes:** Task 1's persisted `tokenizer_contract` and Foundation train stream.

**Produces:** `KeyValueToken(value=<geo_cell>, modality="geo", geo_coordinates=(lat, lon)|None)`; `TokenRecord.geo_value` and `TokenRecord.geo_missing`; a round-trippable train-only geo normalizer; `load_green_sm_tokenizer_config(root) -> TokenizerConfig`.

- [ ] **Step 1: Write failing tokenization and vocabulary tests**

  Create a canonical Green SM sample containing all four scenarios and the two profile locations. Assert all seven registered role keys emit exactly one `geo` token and that `Food` retains `merchant_location` and `food_category`, `Ride` retains pickup/dropoff, `Delivery` retains location/category, and `Ads` retains location/placement. Assert an unallowlisted profile field is excluded. Assert an unregistered `context.other_location`, an unknown `geo_cell`, non-finite coordinates, a cell/coordinate mismatch beyond the persisted tolerance, and a registered `*_location` supplied as a scalar/list/malformed mapping each raise `ValueError` naming the field.

  Fit a vocabulary from train-only geo tokens; assert validation/test coordinates are normalized using persisted train statistics and `geo_cell` is encoded through its own categorical namespace. Save/load the vocabulary and assert its tokenizer/geo contract remains equal.

- [ ] **Step 2: Run the new focused tests and verify failure**

  Run: `python -m pytest tests/foundation_input/test_tokenizer.py tests/pretraining/test_green_sm_contract.py -q`

  Expected: FAIL because the default tokenizer drops Green SM context and treats locations as ordinary dictionaries.

- [ ] **Step 3: Extend the token data model without breaking v2 callers**

  Add optional, defaulted fields to `KeyValueToken`:

  ```python
  geo_coordinates: tuple[float, float] | None = None
  ```

  Add defaulted `TokenRecord.geo_value: tuple[float, float] = (0.0, 0.0)` and `geo_missing: bool = False`. A geo token's `.value` is its `geo_cell` string or `None`; never leave a raw dictionary as its vocabulary value.

- [ ] **Step 4: Implement strict geo parsing and Green SM context policy**

  Extend `TokenizerConfig` with immutable `profile_allowlist`, `geo_field_registry`, `geo_catalogue`, and `coordinate_tolerance`. In `_append_token`, if the fully-qualified field is registered, accept `None` as an explicit missing geo value; otherwise require exactly `lat`, `lon`, and `geo_cell`, finite floats, a known cell, and coordinates equal to that public catalogue cell within the persisted tolerance. Reject every unregistered `*_location` value regardless of whether it is a mapping, scalar, or list. Set modality `geo` and preserve the cell ID/coordinates as specified above. In `semantic_tokens()`, emit only profile fields in `profile_allowlist` and context fields in the scenario allowlist.

  `semantic_tokens()` must use only the supplied allowlist. `load_green_sm_tokenizer_config()` reads the Task-1 metadata, checks version/fingerprint/schema, and constructs this config; it must not fall back to `TokenizerConfig()` defaults.

- [ ] **Step 5: Add train-only geo normalisation to `VocabularyArtifacts`**

  During `fit_vocabulary()`, collect valid geo coordinates only from incoming train tokens, calculate one deterministic mean/standard-deviation pair per coordinate dimension, and persist them as `geo_normalization`. Define the zero-variance policy explicitly as `scale=1.0` for that coordinate dimension, yielding normalized zero for the observed constant train value. Add `normalize_geo(token) -> tuple[tuple[float, float], bool]`, returning `[0.0, 0.0], True` only for an explicit missing geo token. Include profile allowlist, geo field registry, catalogue fingerprint, tolerance, and zero-variance policy in `_tokenizer_contract()` so an incompatible tokenizer cannot reuse a vocabulary.

- [ ] **Step 6: Run tokenization and vocabulary tests**

  Run: `python -m pytest tests/foundation_input/test_tokenizer.py tests/pretraining/test_green_sm_contract.py -q`

  Expected: PASS, including malformed-location failures and train-only statistic checks.

- [ ] **Step 7: Commit the semantic input contract**

  ```powershell
  git add src/foundation_input src/pretraining/green_sm_contract.py tests/foundation_input tests/pretraining/test_green_sm_contract.py
  git commit -m "feat: tokenize Green SM locations as v3 geo tokens"
  ```

### Task 3: Collate and validate `FoundationBatch` v3 geo tensors

**Ownership:** Member 1

**Files:**
- Modify: `src/foundation_input/types.py: FoundationBatch`
- Modify: `src/foundation_input/dataset.py: foundation_collate()`
- Modify: `src/pretraining/gates.py: validate_v3_batch_contract()`
- Modify: `tests/foundation_input/test_dataset.py`
- Modify: `tests/models/conftest.py`
- Modify: `tests/models/test_adapters.py`
- Modify: `tests/models/test_encoder.py`
- Modify: `tests/pretraining/test_config_and_gates.py`
- Modify: `tests/pretraining/test_runner.py`

**Consumes:** Task 2's geo fields in `TokenRecord`.

**Produces:** a batch with `geo_values: FloatTensor[B, L, 2]` and `geo_missing_mask: BoolTensor[B, L]` aligned with every input token.

- [ ] **Step 1: Write failing batch-contract tests**

  Collate a batch containing valid geo, missing geo, numeric, categorical, CLS, SEP, and padding positions. Assert the valid geo row has its normalized coordinates; the missing geo row is `[0.0, 0.0]` and `True`; every non-geo/special/padding row is `[0.0, 0.0]` and `False`. Add rejection tests for wrong rank, dtype, non-finite coordinates, and a missing flag on a non-geo modality.

- [ ] **Step 2: Run focused tests and verify failure**

  Run: `python -m pytest tests/foundation_input/test_dataset.py tests/models tests/pretraining/test_config_and_gates.py tests/pretraining/test_runner.py -q --basetemp .pytest-tmp/task-3-red`

  Expected: FAIL because `FoundationBatch` and collation have no geo fields.

- [ ] **Step 3: Add the aligned v3 tensors and preserve existing layout**

  Add required `geo_values` and `geo_missing_mask` fields to `FoundationBatch`. In `foundation_collate()`, build/pad `[L, 2]` geo rows in the same CLS → tabular → SEP → sequence ordering used by all existing fields. Do not alter group IDs, next-action labels, truncation, or stable user-ID hashing.

- [ ] **Step 4: Strengthen `validate_v3_batch_contract()`**

  Validate shapes, `float32`/`bool` dtypes, finiteness, and missing-mask semantics against `routing.geo`: only a geo position may have `geo_missing_mask=True`; a non-geo position must retain the zero sentinel. Keep this validation at the pretraining boundary as well as model-level validation.

- [ ] **Step 5: Run batch and preflight tests**

  Run: `python -m pytest tests/foundation_input/test_dataset.py tests/models tests/pretraining/test_config_and_gates.py tests/pretraining/test_runner.py -q --basetemp .pytest-tmp/task-3-green`

  Expected: PASS.

- [ ] **Step 6: Commit the v3 batch contract**

  ```powershell
  git add src/foundation_input/types.py src/foundation_input/dataset.py src/pretraining/gates.py tests/foundation_input tests/models tests/pretraining
  git commit -m "feat: propagate geo tensors through FoundationBatch v3"
  ```

### Task 4: Route geo tokens through the Foundation encoder

**Ownership:** Member 2

**Files:**
- Modify: `src/models/adapters.py: GeoAdapter, AdapterRegistry`
- Modify: `src/models/encoder.py: FoundationModelConfig`
- Modify: `tests/models/test_adapters.py`
- Modify: `tests/models/test_encoder.py`
- Modify: `tests/pretraining/test_config_and_gates.py`

**Consumes:** Task 3's complete batch contract and the vocabulary geo modality ID.

**Produces:** a model configuration with `geo_modality_id`; token features where every geo token uses both continuous coordinates and categorical cell/role context.

- [ ] **Step 1: Write failing routing tests**

  Create a valid v3 batch with a single geo position. Assert that changing only its normalized coordinates changes that position's adapter feature; changing only its `input_id`/geo-cell changes it too. Assert numeric values do not affect geo positions, geo tensors do not affect categorical/numeric positions, valid output remains `[B, L, d_model]`, and malformed geo tensor shape fails before Transformer execution.

- [ ] **Step 2: Run the model tests and verify failure**

  Run: `python -m pytest tests/models/test_adapters.py tests/models/test_encoder.py tests/pretraining/test_config_and_gates.py -q --basetemp .pytest-tmp/task-4-red`

  Expected: FAIL because `AdapterRegistry` never reads geo tensors and `FoundationModelConfig` has no geo routing ID.

- [ ] **Step 3: Make geo a required model routing field**

  Add `geo_modality_id: int | None` to `FoundationModelConfig`. Require it from vocabulary metadata, validate `0 < geo_modality_id < modality_vocab_size`, and reject equality with `numeric_modality_id`. Update every model construction helper and test fixture to pass the vocabulary-derived ID; remove the current conditional `if "geo_modality_id" in ...` compatibility branches from `runner.py`.

- [ ] **Step 4: Integrate `GeoAdapter` in `AdapterRegistry.forward()`**

  Instantiate `self.geo_adapter = GeoAdapter(d_model=config.d_model)`. Add `_geo_tensors()` with the same shape/dtype/finiteness checks as the batch gate. Compute features as:

  ```python
  id_features = value_embedding(input_ids) + pair_embedding(pair_ids)
  numeric_features = numeric_mlp([value, missing])
  geo_features = id_features + geo_adapter(geo_values, geo_missing_mask)
  ```

  Select `numeric_features` for numeric tokens, `geo_features` for geo tokens, and `id_features` otherwise; then add existing key/modality/source features, run the per-modality projection, and add position/time-gap features for sequence tokens. This intentionally preserves `geo_cell`, source, and role key while making continuous location available.

- [ ] **Step 5: Make runtime preflight verify routing, not merely field names**

  In `preflight_green_sm()`, require the complete v3 batch fields, required geo model config, and an explicit `AdapterRegistry.supports_geo_routing` capability constant or equivalent tested public predicate. The data report stays data-only; preflight is the runtime truth.

- [ ] **Step 6: Run model, gate, and objective tests**

  Run: `python -m pytest tests/models tests/pretraining/test_config_and_gates.py tests/pretraining/test_masking.py tests/pretraining/test_objectives.py -q --basetemp .pytest-tmp/task-4-green`

  Expected: PASS. Existing geo masking/reconstruction tests must now use a real collated v3 batch instead of hand-added attributes.

- [ ] **Step 7: Commit the adapter/encoder integration**

  ```powershell
  git add src/models src/pretraining/gates.py src/pretraining/runner.py tests/models tests/pretraining
  git commit -m "feat: route Green SM geo tokens through foundation adapters"
  ```

### Task 5: Wire the actual corpus contract into pretraining and regenerate immutable pilot data

**Ownership:** Member 3 with Data team support

**Files:**
- Modify: `src/pretraining/runner.py: build_corpus_run_inputs(), _routing_from_vocab()`
- Modify: `src/pretraining/gates.py: preflight_green_sm()`
- Modify: `src/pretraining/__main__.py` if it duplicates corpus setup
- Modify: `tests/pretraining/test_runner.py`
- Modify: `tests/data/test_green_sm_validate.py`

**Consumes:** Tasks 1–4.

**Produces:** a real Green SM build path that fits its vocabulary on train data with `geo` modality, passes preflight, and performs a one/two-step CPU smoke run.

- [ ] **Step 1: Write the failing corpus-integration test**

  Build a small Green SM fixture root, call `build_corpus_run_inputs()`, and assert the vocabulary includes `geo`, only the declared Green SM profile/context keys, train-only geo normalization metadata, and a model config whose `geo_modality_id` equals the persisted vocabulary routing ID. Collate one train sample and assert `validate_v3_batch_contract()` and `model.encode()` pass.

- [ ] **Step 2: Run the pretraining integration test and verify failure**

  Run: `python -m pytest tests/pretraining/test_runner.py tests/pretraining/test_config_and_gates.py -q --basetemp .pytest-tmp/task-5-red`

  Expected: FAIL because `build_corpus_run_inputs()` creates a default OmniBehavior tokenizer rather than loading `preprocessing_config.json`.

- [ ] **Step 3: Use `load_green_sm_tokenizer_config()` in the runner**

  For each corpus root, load and validate the declared preprocessing contract before fitting vocabulary. Fit with the returned tokenizer only from the train stream. Require `geo` in `_routing_from_vocab()` and remove all legacy conditional routing. Ensure linked/control tokenizer policy and public catalogue fingerprints match in `preflight_green_sm()`.

- [ ] **Step 4: Regenerate and validate a new immutable pilot**

  Create `configs/green_sm_sim_pilot_v2.json` rather than editing v1, with `data_version: "green_sm_sim_pilot_v2"` and `output_root: "data/processed/green_sm_sim_pilot_v2"`. It must emit the child roots `green_sm_sim_linked_v2` and `green_sm_sim_control_paired_v2`. Create `configs/pretrain_green_sm_pilot_v2.json` pointing exactly at their `/foundation` roots; retain the v1 recipes/artifacts as historical records. Generate v2, run its data validator, and confirm that `reports/data_quality_report.json` declares `data_geo_contract_version: 3` and `python -m src.pretraining --config configs/pretrain_green_sm_pilot_v2.json --max-steps 1 --corpus linked --device cpu` reaches training instead of failing at the geo preflight gate.

- [ ] **Step 5: Run the integration suite**

  Run: `python -m pytest tests/data/test_green_sm_validate.py tests/foundation_input tests/models tests/pretraining -q --basetemp .pytest-tmp/task-5-green`

  Expected: PASS.

- [ ] **Step 6: Commit corpus wiring separately from generated data**

  ```powershell
  git add src/pretraining configs tests/pretraining tests/data
  git commit -m "feat: pretrain from the declared Green SM v3 contract"
  ```

  Add the new small pilot only if the repository policy permits simulated artifacts; never stage Meituan, Taobao, or OmniBehavior datasets.

### Task 6: Make training evaluation and checkpoint selection real

**Ownership:** Member 3

**Files:**
- Modify: `src/pretraining/config.py: TrainingSettings`
- Modify: `src/pretraining/__main__.py`
- Create: `configs/pretrain_green_sm_pilot_v2.json`
- Modify: `configs/pretrain_green_sm_full.json`
- Modify: `src/pretraining/runner.py: run_corpus(), evaluate_split()`
- Modify: `src/pretraining/artifacts.py: save_checkpoint()`
- Modify: `tests/pretraining/test_runner.py`
- Create: `tests/pretraining/test_artifacts.py` if missing

**Consumes:** Task 5's trainable v3 corpus.

**Produces:** an immutable run directory keyed by `run_id`, periodic bounded validation, `last.pt`, selected `best.pt`, separate manifests, and a final `run_manifest.json` with objective trace/evaluation history and complete reproducibility state.

- [ ] **Step 1: Write failing scheduling tests**

  With a deterministic tiny iterable and `max_steps=3`, `eval_every_steps=1`, `checkpoint_every_steps=2`, assert evaluation occurs at steps 1, 2, and 3; `last.pt` is saved at step 2 and final step 3; `best.pt` corresponds to the lowest validation loss; all manifests name the correct file and SHA. Assert a fresh run refuses a non-empty `output_dir/run_id`, while resume accepts only the same run directory, selected manifest, and `CheckpointIdentity`; it appends rather than overwrites historical trace entries. Assert Python, NumPy, PyTorch CPU, and CUDA-if-available RNG states round-trip.

- [ ] **Step 2: Run the runner/artifact tests and verify failure**

  Run: `python -m pytest tests/pretraining/test_runner.py tests/pretraining/test_artifacts.py -q --basetemp .pytest-tmp/task-6-red`

  Expected: FAIL because the existing loop ignores both schedule fields and always writes only a final `last.pt`.

- [ ] **Step 3: Add an explicit validation-budget setting and run identity**

  Add `validation_max_batches: int` plus a non-empty slug-safe `run_id` to `PretrainingConfig`, validate both, and include them in each v2 JSON recipe. CLI may override `output_dir` and `device`, but changing `run_id` requires a copied recipe or explicit override that is serialized as the effective config and digest. Use the configured training batch size in evaluation, bounded by `validation_max_batches`, so the Kaggle pilot does not evaluate 24,000 singleton batches every 100 steps.

- [ ] **Step 4: Implement schedule-aware training state persistence**

  Create `output_dir/run_id/corpus`; a fresh run refuses an existing non-empty directory. Resume must target that exact directory, read its immutable effective config and existing run manifest, verify identity/config/vocabulary compatibility before loading any weights, and retain all earlier trace entries. Treat externally reported steps as one-based. After every update, atomically rewrite the trace with the appended loss record; when a schedule boundary or final step occurs, run validation and atomically append `{step, validation_total}`. At each checkpoint boundary and final step write `last.pt`; write `best.pt` only when validation improves. Change `save_checkpoint()` to write `last_manifest.json`/`best_manifest.json` and a final selected `checkpoint_manifest.json` whose `selected_kind` is `best` when present.

  At run start seed Python `random`, NumPy, PyTorch CPU, and CUDA when available; save/restore all four RNG states plus determinism/cudnn settings inside checkpoint payload. Write `run_manifest.json` containing run ID, effective config digest, corpus identity, vocabulary SHA, train trace path, evaluations, selected checkpoint manifest, device, effective step count, RNG/determinism metadata, and software version/timestamp.

- [ ] **Step 5: Run pretraining scheduler tests**

  Run: `python -m pytest tests/pretraining/test_runner.py tests/pretraining/test_artifacts.py -q --basetemp .pytest-tmp/task-6-green`

  Expected: PASS.

- [ ] **Step 6: Commit run reliability changes**

  ```powershell
  git add src/pretraining configs/pretrain_green_sm_pilot_v2.json configs/pretrain_green_sm_full.json tests/pretraining
  git commit -m "feat: evaluate and checkpoint Green SM pretraining periodically"
  ```

### Task 7: Publish stable checkpoint and opportunity-validated output contracts

**Ownership:** Member 3 for checkpoint API; Member 4 for Ads integration

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/checkpoint.py`
- Create: `src/contracts/__init__.py`
- Create: `src/contracts/output.py`
- Modify: `src/pretraining/artifacts.py`
- Modify: `src/downstream/ad_optimization.py`
- Modify: `configs/ad_optimization.yaml`
- Modify: `tests/downstream/test_ad_optimization.py`
- Create: `tests/contracts/test_output.py`

**Consumes:** Task 6's selected manifest/checkpoint and the existing `FrozenFoundationEmbedder`.

**Produces:** `load_foundation_checkpoint()` with integrity/identity verification; `OutputContractV1` that validates a complete prediction set against checksum-verified opportunities before writing data-free predictions and run metadata; unit-level Ads preflight that imports real public APIs rather than deliberately missing modules.

- [ ] **Step 1: Write failing checkpoint/output tests**

  Save a tiny checkpoint, then load it through `src.training.checkpoint.load_foundation_checkpoint()` using its selected manifest. Assert an SHA/identity/format mismatch fails, every embedder parameter has `requires_grad=False`, and `embed_users()` returns finite long/short embeddings. Construct a checksum-verified opportunity index, then assert duplicate/missing/unexpected `(request_id, candidate_id)`, mismatched `user_id`/cutoff/split, non-finite score, Ads probability outside `[0,1]`, invalid run metadata, or a row containing label fields is rejected. Assert valid rows create a fully aligned Parquet prediction file plus `run_manifest.json` and validation report.

- [ ] **Step 2: Run focused tests and verify failure**

  Run: `python -m pytest tests/downstream/test_ad_optimization.py tests/contracts/test_output.py -q --basetemp .pytest-tmp/task-7-red`

  Expected: FAIL at the intentionally absent `src.training.checkpoint` and `src.contracts.output` imports.

- [ ] **Step 3: Define the public checkpoint loader**

  `load_foundation_checkpoint(checkpoint_path, manifest_path, *, map_location="cpu") -> FrozenFoundationEmbedder` must verify the manifest schema/version, filename, SHA-256, and serialized `CheckpointIdentity` before delegating to the existing artifact loader. Do not let downstream code deserialize a checkpoint directly.

- [ ] **Step 4: Define `OutputContractV1`**

  Implement a small writer/validator with:

  ```python
  OutputContractV1.write_predictions(
      rows: Iterable[PredictionRow], *, expected_candidates: VerifiedOpportunityIndex,
      output_dir: Path, split: Literal["validation", "test"], run_metadata: Mapping[str, object]
  ) -> OutputPaths
  ```

  `VerifiedOpportunityIndex` is constructed only after the opportunity manifest, every split Parquet checksum, data version, task, candidate set, user ID, and cutoff fields have been validated. `PredictionRow` supplies `request_id`, `candidate_id`, `score`, optional `raw_logit`, and `split`; labels are forbidden. The writer requires exactly one row for every verified candidate and injects the fixed `user_id`/cutoff metadata from that index. The emitted Parquet schema contains `task`, `data_version`, `run_id`, `split`, `request_id`, `user_id`, `cutoff_timestamp`, `candidate_id`, `score`, and optional `raw_logit`; it contains no label. For `task="ad_optimization"` with `score_semantics="probability"`, enforce `0 <= score <= 1`. `run_metadata` must include selected-checkpoint SHA, vocabulary SHA, Foundation/Ads manifest SHA, config digest, seed, run ID, score semantics, candidate/request counts, and source checksums.

- [ ] **Step 5: Replace the Ads placeholder handoff with real preflight**

  Update `validate_member3_handoff()` to use the public loader and selected manifest rather than a placeholder SHA string. Keep `freeze_foundation: true` mandatory. This task must not pretend that the Taobao `AdOpportunity` schema can consume a Green SM checkpoint: it only proves public import, manifest verification, and frozen-embedder behavior with a `FoundationBatch` fixture. Keep the current Taobao-specific handoff schema isolated; real Green SM Ads train/inference is Task 8.

- [ ] **Step 6: Run downstream and public-contract tests**

  Run: `python -m pytest tests/contracts/test_output.py tests/downstream/test_ad_optimization.py -q --basetemp .pytest-tmp/task-7-green`

  Expected: PASS, including a two-batch fixture-level CPU embedder smoke test with Foundation weights unchanged.

- [ ] **Step 7: Commit the handoff API and isolated Ads preflight**

  ```powershell
  git add src/training src/contracts src/pretraining/artifacts.py src/downstream/ad_optimization.py configs/ad_optimization.yaml tests/contracts tests/downstream
  git commit -m "feat: publish frozen foundation checkpoint and prediction contracts"
  ```

### Task 8: Build the real Green SM Ads downstream bridge

**Ownership:** Member 4

**Files:**
- Create: `src/downstream/green_sm_ad_optimization.py`
- Create: `configs/green_sm_ad_optimization_pilot_v2.yaml`
- Create: `tests/downstream/test_green_sm_ad_optimization.py`
- Modify: `docs/kaggle_ad_optimization_runbook.md`

**Consumes:** the selected linked Green SM checkpoint/vocabulary from Task 6, Task 7 public contracts, and `green_sm_sim_pilot_v2/green_sm_sim_linked_v2/ads_opportunities`.

**Produces:** a genuine frozen-Foundation Green SM Ads pipeline, not an attempted reuse of Taobao-specific code.

- [ ] **Step 1: Write failing Green SM Ads integration tests**

  Load Green SM Ads candidate rows and their referenced causal Foundation snapshots. Assert every opportunity resolves exactly one same-user/same-cutoff `EncodedSample`; collated batches pass the v3 geo gate; the verified frozen linked checkpoint returns embeddings; only the ranker receives gradients; and validation output has exactly every verified candidate once.

- [ ] **Step 2: Run the integration test and verify failure**

  Run: `python -m pytest tests/downstream/test_green_sm_ad_optimization.py -q --basetemp .pytest-tmp/task-8-red`

  Expected: FAIL because no Green-SM-specific opportunity adapter/trainer exists.

- [ ] **Step 3: Implement the Green SM Ads adapter and runner**

  Parse the published Green SM Ads fields (`request_id`, `user_id`, `foundation_sample_id`, ISO cutoff, candidate/campaign/placement/category/geo fields, `clicked`) without coercing them to Taobao types. Resolve sample IDs through the persisted Green SM vocabulary/tokenizer, build v3 batches, call only `load_foundation_checkpoint(...).embed_users()`, and detach/freeze its output. Train `AdRanker`-equivalent candidate layers on train only; validate/predict through `VerifiedOpportunityIndex` and `OutputContractV1`. Do not import labels into validation/test prediction rows.

- [ ] **Step 4: Run the real two-batch CPU smoke integration**

  Run: `python -m pytest tests/downstream/test_green_sm_ad_optimization.py -q --basetemp .pytest-tmp/task-8-green`

  Expected: PASS, including unchanged Foundation weights and complete validation prediction coverage.

- [ ] **Step 5: Commit the Green SM Ads bridge**

  ```powershell
  git add src/downstream/green_sm_ad_optimization.py configs/green_sm_ad_optimization_pilot_v2.yaml tests/downstream docs/kaggle_ad_optimization_runbook.md
  git commit -m "feat: train frozen foundation embeddings for Green SM ads"
  ```

### Task 9: Full verification and hygiene gate

**Ownership:** integration owner

**Files:**
- Modify only as required by failures from prior tasks.
- Update: `docs/kaggle_ad_optimization_runbook.md` and the Green SM pretraining runbook with actual v2 pilot paths, selected-manifest names, and CLI commands.

**Consumes:** Tasks 1–8.

**Produces:** evidence that the repository is locally contract-ready; Kaggle receives one selected Foundation artifact and immutable datasets.

- [ ] **Step 1: Run focused quality checks**

  Run:

  ```powershell
  python -m pytest tests/data tests/foundation_input tests/models tests/pretraining tests/contracts tests/downstream -q --basetemp .pytest-tmp/green-sm-v3
  python -m ruff check src tests
  ```

  Expected: all scoped tests pass and Ruff reports zero violations. If the pre-existing `tests/test_imports.py` collection issue remains, document it separately rather than hiding it with this change.

- [ ] **Step 2: Run an actual CPU smoke train on the regenerated pilot**

  Run:

  ```powershell
  python -m src.pretraining --config configs/pretrain_green_sm_pilot_v2.json --max-steps 2 --corpus linked --device cpu
  ```

  Expected: preflight passes; `output_dir/run_id/linked` has `last.pt`, a selected checkpoint manifest, vocabulary, atomically written loss trace, evaluation history, and run manifest. Confirm `geo` appears in the persisted vocabulary and at least one geo objective count is non-zero.

- [ ] **Step 3: Validate downstream preflight without training a production model**

  Mount/copy the selected checkpoint, vocabulary, and immutable v2 pilot/root paths into the Green SM Ads downstream config. Run its `--preflight`, then its two-batch `--smoke-train`. Confirm frozen Foundation parameter checks pass and a validation prediction artifact covers the checksum-verified candidate set through `OutputContractV1`.

- [ ] **Step 4: Review generated artifacts before any push**

  Verify `git status --short`; stage only source, configuration, docs, tests, and intentionally versioned small simulated pilot artifacts. Do not stage raw Meituan, Taobao, OmniBehavior, or an entire Kaggle output/checkpoint directory.

- [ ] **Step 5: Commit documentation/hygiene only if changed**

  ```powershell
  git add docs
  git commit -m "docs: update Green SM pretraining and downstream handoff runbooks"
  ```

## Merge and Execution Order

1. Merge Task 1 first; it is the source of truth for Green SM service context and public geo catalogue.
2. Tasks 2 and 3 are one input-contract chain and should merge before Task 4. Member 1 can prepare Task 3 tests while Task 2 is under review, but must rebase on its interface.
3. Task 4 is Member 2's adapter work; it must use the exact fields and normalizer from Tasks 2–3, not add a competing geo representation.
4. Task 5 connects the real data to the real model. Regenerate a new pilot only here, after the contract is merged.
5. Task 6 makes the train run auditable; Task 7 consumes its selected checkpoint. They must not be merged in reverse order.
6. Task 8 is the first genuine Green SM Ads integration; do not reinterpret a Taobao smoke test as this evidence.
7. Task 9 is the only point where Kaggle training is authorized. A data-quality report alone is insufficient.

## Self-Review

- **Spec coverage:** location as cell plus coordinates, train-only normalization, exact geo roles, service context retention, v3 tensors, adapter fusion, geo objectives, temporal linked/control parity, and frozen checkpoint reuse each map to Tasks 1–7.
- **External validation coverage:** immutable data reporting and config reuse are Task 1; geo preflight is Tasks 3–5; unused evaluation/checkpoint settings plus reproducible resume are Task 6; missing checkpoint/output public modules and full candidate validation are Task 7; the missing Green SM Ads adapter is Task 8; Ruff/scoped smoke verification is Task 9.
- **Type consistency:** `KeyValueToken.geo_coordinates` flows to `TokenRecord.geo_value`, then `FoundationBatch.geo_values`, then `AdapterRegistry.geo_adapter`; `geo_modality_id` always comes from persisted vocabulary routing.
- **Deliberate boundary:** this plan does not invent a location graph, external geocoder, or private metric evaluator. Green SM-specific Ads schema migration is held until data team publishes its final Ads artifact contract.
