# Foundation Input Contract Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the input contract needed by the foundation model: expose vocabulary cardinalities, make numeric/missing semantics explicit, and carry normalized numeric values with a missing mask through `FoundationBatch`.

**Architecture:** The team has selected the full numeric extension. Keep `numeric` as the canonical modality name because the tokenizer, vocabulary, tests, and model configuration already use it. Treat numeric missingness as a separate signal: numeric tokens always use `modality="numeric"`, valid values are transformed by a train-fitted per-key policy, and missing/non-finite values become `numeric_value=0.0` with `numeric_missing_mask=True`. Non-numeric missing fields remain explicit `modality="missing"` tokens. `FoundationBatch` v2 therefore requires aligned raw numeric tensors; the input adapter emits them and Person 2 consumes them through the numeric MLP route.

**Tech Stack:** Python 3.13, dataclasses, PyTorch tensors, pytest, Ruff, JSON vocabulary metadata.

**Spec:** `docs/foundation_input_contract.md`, `docs/parallel_vibe_implementation_plan.md`

## Global Constraints

- Vocabulary, numeric boundaries, numeric transform policies, and normalization statistics are fit only from the selected dataset's logical `train` split.
- Validation and test are transform-only; they must not mutate vocabulary or normalization metadata.
- Special IDs remain `PAD=0`, `CLS=1`, `SEP=2`, `MASK=3`, `UNK=4`.
- `numeric` is the only canonical code-level spelling; `numerical` is prose-only and must not appear as a modality value or field name.
- Numeric missing routing comes from an explicit per-dataset `(source, key)` field registry, not only from observing non-null numeric values in train.
- The initial registry is derived from `data/README_pretrain_data.md`, `data/README_Meituan.md`, and the selected dataset's `preprocessing_config.json`, then verified against actual Parquet values before fitting. Storage metadata such as `source_action_count` is not a business numeric feature.
- Missing numeric values are represented by `numeric_value=0.0` plus `numeric_missing_mask=True`; zero is a valid observed numeric value and must have mask `False`.
- `FoundationBatch.numeric_values` has shape `[B, L]`, dtype `float32`; `FoundationBatch.numeric_missing_mask` has shape `[B, L]`, dtype `bool`.
- Padding and non-numeric tokens have `numeric_values=0.0` and `numeric_missing_mask=False`.
- Geo fields are out of scope until the data contract supplies latitude/longitude columns.

## Architecture decision

- **Selected: full numeric extension.** Implement Tasks 1–3. `FoundationBatch` v2 requires `numeric_values` and `numeric_missing_mask`, and Person 2 must update `AdapterRegistry` before the shared model tests are considered green.
- **Text policy:** `TokenizerConfig.include_action_content=False` by default, so the noisy Customer Service conversation field is excluded from the MVP. An ablation may set it to `True`; then flatten list-valued text into deterministic role-tagged text (`role: content` per message, stable message order), apply the configured cap, and fit a separate vocabulary/checkpoint. Never tokenize Python's list/dict representation. The default `max_text_chars=256` is acceptable for the optional full-text run because it bounds memory and keeps one field to one token; use deterministic head+tail truncation so the beginning and latest message are both retained.
- **Geo:** never add geo tensors from this plan alone. Add `geo_values` and `geo_missing_mask` only after the Data team confirms latitude/longitude in the selected pretrain manifest and supplies train-only coordinate normalization metadata.

## Current `/data` Compatibility Notes

- Both current roots use `omnibehavior_pretraining_v2` with `snapshots/*.parquet` and `events/*.parquet`; the event payload is JSON in `context_json` and `action_json`, not a single generic `value` column.
- OmniBehavior preview data contains eight numeric profile count fields plus the string `as_of_time`; the README's phrase "9 numerical" is therefore a documentation inconsistency that must not become a ninth numeric token.
- Observed OmniBehavior numeric examples are `context.duration`, `context.item_price`, `action.comment_cnt`, `action.conversion_cnt`, `action.play_count`, `action.play_duration`, `action.played_loop_cnt`, and `action.show_cnt`. Boolean fields remain categorical; text/list fields need their declared text/list policy.
- OmniBehavior also contains `action.content` as a list of `{role, content}` messages. The tokenizer must flatten this deterministically or exclude it by policy; converting the Python list representation directly into one token is not an acceptable stable text encoding.
- Observed Meituan numeric examples are the shared profile counts, `context.item_price`, `action.click_cnt`, `action.item_cnt`, and `action.order_amt`. `action.order_price_interval` remains categorical by source semantics.
- `training_targets_stored=false` and `mask_policy=null` in both preprocessing configs. Mask rate and next-action targets are training-time derived objectives, not labels supplied by the data.
- The current pretrain configs have no latitude/longitude fields and `timezone=null`; date-only timestamps and uncertain ordering must remain explicit. Geo and fabricated timezone features are out of scope.
- OmniBehavior has 492,801 uncertain-order histories, 788,723 histories truncated to 30 observations, and 68,411 histories with a partial eligibility group in its data report. Next-action coverage must therefore be measured, not assumed.
- Meituan's current manifest contains only `train` snapshots, one snapshot per user at `2021-03-21 23:59:59`. Do not random-split those rows and call the result temporal validation/test. A valid Meituan pretraining evaluation requires Data team regeneration with multiple time cutoffs and a manifest containing explicit `train`/`validation`/`test` snapshots; until then, test transform-only behavior on OmniBehavior validation/test and run Meituan only as a train/compatibility smoke test.

## Adjacent downstream contract blockers

- The current collator uses zero-valued next-action targets when no eligible target exists. Before Person 3 implements the objective, add `targets["next_action_valid_mask"]` and require at least two complete sequence groups so the input context is non-empty after removing the target group.
- `next_action` targets must use `event.event_type`; `action.type` is a behavior field and is not a success/label column according to the data preprocessing limitations.

## Review Focus

- A numeric column containing both `0.0` and missing values must distinguish them through the mask; test this in Task 2.
- A numeric key missing in validation/test must retain numeric routing if it appeared as numeric in train; test this in Task 2.
- A custom text hash bucket count and a single-value numeric boundary must produce sufficient value cardinality; test this in Task 1.
- A serialized vocabulary must preserve cardinalities and normalization statistics exactly; test this in Task 1.
- Padded, categorical, text, and missing tokens must not accidentally enter the numeric adapter; test this in Task 3.

### Task 1: Vocabulary metadata and cardinalities

**Files:**
- Modify: `src/foundation_input/vocab.py:45-140, 143-201`
- Modify: `tests/foundation_input/test_vocab.py`
- Modify: `docs/foundation_input_contract.md:32-36`
- Modify: `docs/parallel_vibe_implementation_plan.md:150-180, 208-220`
- Read as data sources: `data/README_pretrain_data.md`, `data/README_Meituan.md`, `data/pretrain_data/meituan_data/preprocessing_config.json`

**Interfaces:**
- Produces `VocabularyArtifacts.model_cardinalities() -> dict[str, int]` with `value_vocab_size`, `key_vocab_size`, `modality_vocab_size`, `source_vocab_size`, `time_gap_vocab_size`, and `text_hash_buckets`. The first five are model dimension inputs; `text_hash_buckets` is artifact metadata and must not be passed blindly to `FoundationModelConfig` unless that config explicitly adds the field.
- Produces `VocabularyArtifacts.numeric_normalization: dict[str, tuple[float, float]]`, mapping a numeric namespace to `(center, scale)` with `scale >= 1e-6`; constant columns use `scale=1.0`.
- Produces `VocabularyArtifacts.numeric_field_registry: tuple[tuple[str, str], ...]` from the explicit tokenizer/data configuration, so transform-time `None` values retain numeric routing even if a field is all-missing in a particular train slice.
- Persists the tokenizer time-gap boundaries or an equivalent `time_gap_vocab_size` in the vocabulary metadata so `model_cardinalities()` is reproducible after reload.

- [ ] **Step 1: Write failing vocabulary tests.**

  Add tests for:

  ```python
  cardinalities = artifacts.model_cardinalities()
  assert cardinalities["value_vocab_size"] > max(SPECIAL_TOKEN_IDS.values())
  assert cardinalities["key_vocab_size"] >= max(artifacts.key_to_id.values(), default=0) + 1
  assert artifacts.numeric_field_registry == (("tabular", "profile.score"),)
  assert artifacts.numeric_normalization["omnibehavior_v2:tabular:numeric:score"] == (2.0, 1.0)
  ```

  Also test JSON save/load equality for `model_cardinalities`, `numeric_field_registry`, `numeric_normalization`, and the time-gap cardinality.

- [ ] **Step 2: Run the focused tests and confirm they fail.**

  Run `python -m pytest tests/foundation_input/test_vocab.py -q`.
  Expected result: failures because the method and metadata fields do not exist.

- [ ] **Step 3: Implement the metadata.**

  Extend `VocabularyArtifacts` with `numeric_normalization`, `numeric_field_registry`, and the time-gap metadata. Use `log1p` followed by z-score for declared non-negative count, duration, amount, and price fields; use z-score without `log1p` for signed values or score-like fields. Fit center/scale from finite train values only, use `scale=1.0` for a constant column, and serialize the transform name per numeric namespace. Do not silently apply a different policy at transform time. Serialize and restore all fields in `to_metadata()` and `load()`.

  Implement `model_cardinalities()` by taking the maximum possible encoded ID, including special IDs, categorical IDs, numeric bucket IDs, and text hash IDs, then returning `max_id + 1` for each ID namespace. Include `time_gap_vocab_size` from the persisted time-gap metadata and `text_hash_buckets` for model/config diagnostics. Do not return the number of dictionary entries when IDs are sparse.

- [ ] **Step 4: Run the focused tests and lint.**

  Run `python -m pytest tests/foundation_input/test_vocab.py -q` and `python -m ruff check src/foundation_input/vocab.py tests/foundation_input/test_vocab.py`.
  Expected result: all focused tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit the self-contained vocabulary change.**

  Run `git add src/foundation_input/vocab.py tests/foundation_input/test_vocab.py docs/foundation_input_contract.md docs/parallel_vibe_implementation_plan.md` and commit with `feat: expose vocabulary model cardinalities`.

### Task 2: Numeric token semantics and encoded numeric payload

**Files:**
- Modify: `src/foundation_input/tokenizer.py:74-115, 200-248`
- Modify: `src/foundation_input/types.py:33-44`
- Modify: `tests/foundation_input/test_tokenizer.py`
- Modify: `tests/foundation_input/test_contract.py`

**Interfaces:**
- `TokenizerConfig.numeric_fields: frozenset[tuple[str, str]]` is the source-of-truth registry, using pairs such as `("tabular", "profile.score")` and `("sequence", "action.play_duration")`.
- `TokenizerConfig.include_action_content` defaults to `False`; its value is persisted in vocabulary metadata and must match during transform.
- For the current OmniBehavior-compatible pretrain schema, seed the registry with the documented profile count fields (`profile.active_days_30d`, `profile.event_count_30d`, `profile.video_browsing_event_count_30d`, `profile.live_streaming_event_count_30d`, `profile.e_commerce_event_count_30d`, `profile.advertisement_event_count_30d`, `profile.search_behavior_event_count_30d`, `profile.customer_service_event_count_30d`) and documented numeric event fields only after Parquet inspection confirms their runtime types. Keep `order_price_interval` categorical.
- `TokenRecord` gains `numeric_value: float = 0.0` and `numeric_missing: bool = False`.
- `tokenize_sample()` emits a normalized numeric payload for `modality="numeric"` records.
- A missing token whose `(source, key)` is in the explicit numeric field registry is reclassified as numeric for transform, preserving numeric routing while keeping its raw value as `None`.

- [ ] **Step 1: Write failing tokenizer tests.**

  Fit a vocabulary on a train sample where the declared numeric field `profile.score` has values `0.0` and `2.0`. Tokenize a transform sample containing `score=0.0` and another containing `score=None`. Assert:

  ```python
  assert observed.numeric_missing is False
  assert observed.numeric_value == pytest.approx(-1.0)
  assert missing.numeric_missing is True
  assert missing.numeric_value == 0.0
  assert missing.token.modality == "numeric"
  ```

  Add a categorical `None` assertion that it remains `modality="missing"` and does not set the numeric flag.

- [ ] **Step 2: Run the focused tests and confirm they fail.**

  Run `python -m pytest tests/foundation_input/test_tokenizer.py tests/foundation_input/test_contract.py -q`.
  Expected result: failures because `TokenRecord` has no numeric payload and missing numeric routing is not defined.

- [ ] **Step 3: Implement numeric normalization and missing semantics.**

  Add a small helper in `tokenizer.py` that maps a numeric token to `(normalized_value, missing)` using the vocabulary's train-fitted field registry, transform policy, and statistics. Treat `None`, non-finite values, and failed numeric conversion as missing. Reclassify missing tokens only when their `(source, key)` is declared numeric; never infer numeric routing from `None` alone. Leave categorical/text missing tokens unchanged.

  Keep `encode_value()` behavior backward-compatible: bucket IDs remain available for masked-value objectives, while the new continuous payload is the source for the numeric adapter.

- [ ] **Step 4: Run tests and inspect edge cases.**

  Run `python -m pytest tests/foundation_input/test_tokenizer.py tests/foundation_input/test_contract.py -q` and verify that zero, negative values, constant columns, missing values, and non-finite values behave as specified.

- [ ] **Step 5: Commit the self-contained tokenizer change.**

  Run `git add src/foundation_input/tokenizer.py src/foundation_input/types.py tests/foundation_input/test_tokenizer.py tests/foundation_input/test_contract.py` and commit with `feat: add normalized numeric token payloads`.

### Task 3: FoundationBatch v2 tensors and cross-team model handoff

**Files:**
- Modify: `src/foundation_input/types.py:97-109`
- Modify: `src/foundation_input/dataset.py:161-300`
- Modify: `tests/foundation_input/test_dataset.py`
- Modify: `tests/models/conftest.py` and model tests that construct `FoundationBatch`
- Modify: `docs/foundation_input_contract.md:38-48`
- Coordinate with: `src/models/adapters.py`, `src/models/encoder.py`

**Interfaces:**
- `FoundationBatch` v2 and `foundation_collate()` return `numeric_values` and `numeric_missing_mask` aligned one-to-one with `input_ids`; both fields are required, not optional.
- Person 2's `AdapterRegistry` consumes the two tensors for numeric routing and preserves the existing ID path for categorical/text/missing tokens.

- [ ] **Step 1: Write failing collator and contract tests for the selected full numeric contract.**

  Add a test that collates one numeric, one categorical, one missing, one sequence, and one padded token, then asserts shapes, dtypes, normalized value, and masks:

  ```python
  assert batch.numeric_values.shape == batch.input_ids.shape
  assert batch.numeric_values.dtype == torch.float32
  assert batch.numeric_missing_mask.dtype == torch.bool
  assert batch.numeric_missing_mask[0, numeric_index]
  assert not batch.numeric_missing_mask[0, categorical_index]
  assert batch.numeric_values[0, pad_index] == 0.0
  ```

  Update model fixtures in the test changeset to include zero-filled numeric tensors so old model tests still exercise the same ID contract.

- [ ] **Step 2: Run focused tests and confirm they fail.**

  Run `python -m pytest tests/foundation_input/test_dataset.py tests/models -q`.
  Expected result: input-contract failures until the new tensors are emitted; if Torch is unavailable, record the environment failure and run the dataset-only test separately.

- [ ] **Step 3: Implement aligned tensor emission.**

  In `foundation_collate`, append `record.numeric_value` and `record.numeric_missing` for real records, and append `0.0`/`False` for `[CLS]`, `[SEP]`, tabular nonnumeric records, target-only records, and padding. Pad all new rows to the same `max_length` as the existing ID tensors and construct them with `backend.float32` and `backend.bool`.

- [ ] **Step 4: Hand off the model-side consumption to Person 2.**

  Person 2 updates `AdapterRegistry` with a numeric projection that consumes `batch.numeric_values` and `batch.numeric_missing_mask`, masks nonnumeric positions, and aligns the result to `d_model`. The model must not infer numeric values from `input_ids`; bucket IDs remain objective metadata.

  Person 2 also adds model tests for zero-versus-missing distinction and confirms `FoundationModelConfig` is constructed from an explicit filtered mapping of supported `*_vocab_size` values; `text_hash_buckets` remains metadata unless the config is intentionally extended.

- [ ] **Step 5: Run the complete input contract checks.**

  Run `python -m pytest tests/foundation_input -q`, `python -m ruff check src/foundation_input tests/foundation_input`, and the real-data smoke check on 20 samples each from `data/pretrain_data/OmniBehavior_data` and `data/pretrain_data/meituan_data`.
  Expected result: every sampled batch has aligned numeric tensors, and sequence retention remains intact.

- [ ] **Step 6: Commit the cross-team contract change.**

  Commit the input-side files with `feat: extend foundation batch with numeric features` only after Person 2 has updated their constructors and adapter tests in the coordinated branch/PR.

## Self-review and scope boundary

- `VocabularyArtifacts.model_cardinalities()` is covered by Task 1.
- Naming is explicitly resolved to `numeric`; `numerical` is removed from code-level modality/field names.
- Missing numeric semantics are covered by Task 2 and represented independently from the numeric value.
- `numeric_values` and `numeric_missing_mask` are covered by Task 3.
- Geo remains excluded until latitude/longitude is present in the data contract.
- This plan intentionally does not add user-ID vocabulary cardinalities, because OOV user IDs already use the stable-hash path and the model uses `num_user_buckets`.
