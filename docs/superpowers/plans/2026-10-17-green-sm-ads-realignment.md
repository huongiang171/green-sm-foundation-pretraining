# Green SM Ads Realignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Taobao-specific Ads route with a checksum-gated, frozen-Foundation Green SM full-corpus Ads CTR runner.

**Architecture:** `src/downstream/ad_optimization.py` becomes the sole canonical Green SM Ads module. It reads native Green SM opportunities, resolves snapshots through the exact linked Foundation vocabulary, uses `checkpoint_manifest.json` to select the Foundation artifact, freezes the embedder, and trains only an Ads ranker. `OutputContractV1` exports one label-free probability per validation/test candidate.

**Tech Stack:** Python 3.13, PyTorch, PyArrow, PyYAML, pytest, Ruff, Kaggle T4.

**Spec:** `docs/superpowers/specs/2026-10-17-green-sm-ads-realignment-design.md`

## Global Constraints

- Use only `green_sm_sim_full_v1/green_sm_sim_linked_v1`; do not read Taobao data at runtime.
- Require `green_sm_ads_opportunities_v1`, a passing linked data-quality report, immutable Parquet SHA-256s, and an exact linked Foundation-manifest binding.
- Resolve Foundation checkpoint filename/SHA from `checkpoint_manifest.json`; require matching `vocabulary.json`.
- Freeze Foundation; train only the ranker with masked candidate-weighted BCE over all real training candidates.
- Preserve all validation/test candidates; export probability scores through `OutputContractV1`, never labels.
- Retain tracked `data/taobao_data/*.csv` and `data/README_taobao.md`; remove only the declared Taobao ETL/runtime files.
- Do not stage datasets, checkpoints, Kaggle working artifacts, or raw Taobao CSVs.

## Review Focus

- A mismatched or placeholder checkpoint-selection manifest must fail in validate mode before any model/data loader is created.
- A checkpoint selected as `last` rather than `best` must still load when its manifest/checksum are valid.
- Flipping test `clicked` values must not change inference scores or introduce `clicked` into output artifacts.
- Candidate padding, no-click requests, and unknown category/placement/geo values must remain valid and deterministic.
- A stale Taobao import/reference must be absent from executable Ads code/config/runbook after migration.

---

### Task 1: Define Green SM Ads contract and checkpoint-selection tests

**Files:**
- Modify: `tests/downstream/test_ad_optimization.py`
- Delete: `tests/downstream/test_green_sm_ad_optimization.py`

**Interfaces:**
- Produces: executable tests for `GreenSmAdsOpportunity`, `validate_ads_data_handoff()`, `resolve_selected_foundation_handoff()`, and label-free prediction construction.
- Consumes: existing Green SM Ads Parquet schema and public Foundation checkpoint manifest format.

- [ ] Write failing tests that construct native Green SM rows and assert required geo fields, timestamp types, no-click requests, per-request consistency, and train-only `UNK` vocab behavior.
- [ ] Write failing tests for a `checkpoint_manifest.json` selecting both `best.pt` and `last.pt`, a bad/placeholder SHA, mismatched linked manifest, and vocabulary SHA mismatch.
- [ ] Write a failing test proving `frozen_foundation_ads_logits()` leaves Foundation gradients `None` while ranker gradients are populated.
- [ ] Write a failing inference test that flips `clicked` after batch construction, requires identical scores, and asserts exported Parquet lacks `clicked`.
- [ ] Run: `python -m pytest tests/downstream/test_ad_optimization.py -q --basetemp .pytest-tmp/ads-task-1`
- [ ] Commit: `test: define Green SM Ads downstream contract`

### Task 2: Replace Taobao-shaped Ads primitives with native Green SM implementation

**Files:**
- Modify: `src/downstream/ad_optimization.py`
- Modify: `tests/downstream/test_ad_optimization.py`
- Delete: `src/downstream/green_sm_ad_optimization.py`

**Interfaces:**
- Produces: `GreenSmAdsOpportunity`, `AdsFeatureArtifacts`, `AdsBatch`, `AdsRanker`, `load_green_sm_ads_opportunities()`, `build_ads_batches()`, `frozen_foundation_ads_logits()`, and `expected_ads_candidates()`.
- Consumes: native fields `candidate_category`, `placement`, `candidate_geo_cell`, `candidate_lat`, `candidate_lon`, `campaign_id`, and `clicked`.

- [ ] Implement a native immutable opportunity record; validate finite lat/lon, non-empty geo cell, availability/cutoff ordering, split, and candidate uniqueness without aliases such as `cate_id` or `pid`.
- [ ] Implement train-only category/placement/geo vocabularies with PAD/UNK, `green-sm-ads-v1` hash namespaces for candidate/campaign IDs, request-major padding, and masked BCE.
- [ ] Implement `AdsRanker` with fused long/short user vectors and candidate/campaign/category/placement/geo embeddings; derive `d_model` from the loaded Foundation model.
- [ ] Implement `frozen_foundation_ads_logits()` with only `embedder.embed_users()` under `torch.no_grad()`; run the ranker outside that context.
- [ ] Implement OutputContract candidate/prediction helpers that flatten only real candidate-mask cells and emit sigmoid probability plus optional raw logit.
- [ ] Run: `python -m pytest tests/downstream/test_ad_optimization.py -q --basetemp .pytest-tmp/ads-task-2`
- [ ] Commit: `feat: add native Green SM Ads ranker primitives`

### Task 3: Add checksum-gated train, reload, and prediction lifecycle

**Files:**
- Modify: `src/downstream/ad_optimization.py`
- Modify: `tests/downstream/test_ad_optimization.py`
- Create: `configs/ad_optimization.yaml`

**Interfaces:**
- Produces: `main()`, `run_ads_training()`, `run_ads_prediction()`, `train_ads_ranker()`, and a full-corpus Kaggle configuration.
- Consumes: Task 2 primitives, `OutputContractV1`, `load_foundation_checkpoint()`, and the selected checkpoint package.

- [ ] Add failing handoff tests for linked full data-quality report, every Ads file checksum, Foundation-manifest link, selected manifest, selected checkpoint identity, and vocabulary SHA.
- [ ] Add failing lifecycle tests: train writes feature artifacts/last/best/metrics/run manifest atomically; reload rejects changed run/data/checkpoint identity; prediction writes separate validation/test directories with exact candidate sets.
- [ ] Implement config parsing and validate mode. Require `data_version == green_sm_sim_full_v1`, `freeze_foundation: true`, corpus `linked`, and non-placeholder checksums in every CLI mode.
- [ ] Resolve checkpoint path from configured `checkpoint.package_root/checkpoint_manifest.json`; verify the selected file and then call the public Foundation loader with that manifest.
- [ ] Implement training/validation BCE lifecycle and reloadable Ads artifacts, mirroring Food's artifact discipline but retaining Ads probability semantics.
- [ ] Implement `--mode validate|train|predict`, with `--split validation|test` for prediction; do not implement an evaluator owned by the Baseline team.
- [ ] Set config paths to the full linked Green SM corpus and require the yet-to-be-published full checkpoint package through explicit placeholders.
- [ ] Run: `python -m pytest tests/downstream/test_ad_optimization.py tests/contracts/test_output.py -q --basetemp .pytest-tmp/ads-task-3`
- [ ] Commit: `feat: add Green SM Ads training lifecycle`

### Task 4: Replace operational handoff materials

**Files:**
- Modify: `docs/kaggle_ad_optimization_runbook.md`
- Create: `notebooks/ad_optimization_kaggle.ipynb`
- Modify: `configs/ad_optimization.yaml`

**Interfaces:**
- Produces: an executable Kaggle workflow that mounts only the full Green SM corpus and selected Foundation package, trains the ranker, verifies artifacts, and exports validation/test predictions.
- Consumes: Task 3 CLI and artifact locations.

- [ ] Create a notebook that discovers the full linked corpus and checkpoint package, computes config SHA inputs at runtime, refuses missing/placeholder artifacts, and checks CUDA before train.
- [ ] Add train, prediction, checkpoint/run-manifest verification, and zip/export cells; never overwrite a non-empty run directory.
- [ ] Rewrite the runbook for the same commands and paths, including the handoff to Baseline: prediction Parquet, split manifest/report, Ads run manifest, and selected Foundation manifest.
- [ ] Run: `python -c "import json; json.load(open('notebooks/ad_optimization_kaggle.ipynb', encoding='utf-8')); print('valid')"`
- [ ] Commit: `docs: add Green SM Ads Kaggle workflow`

### Task 5: Remove abandoned Taobao ETL/runtime surface and audit migration

**Files:**
- Delete: `configs/taobao_ads_etl.json`
- Delete: `src/data/taobao_ads/`
- Delete: `tests/data/test_taobao_ads_etl.py`
- Delete: `docs/kaggle_taobao_ads_etl_runbook.md`
- Delete: `docs/taobao_ads_data_contract.md`
- Delete: `docs/superpowers/plans/2026-10-12-ad-optimization-kaggle.md`
- Delete: `docs/superpowers/plans/2026-10-13-taobao-ads-foundation-etl.md`
- Delete: `.pytest-tmp-ads/`
- Delete: `.pytest-tmp-ads-review/`
- Retain: `data/taobao_data/*.csv`, `data/README_taobao.md`

**Interfaces:**
- Produces: no executable Taobao Ads import, configuration, test, or runbook; all Ads entry points use Green SM.

- [ ] Verify each deletion target is uncommitted/generated as expected, then remove only the listed paths.
- [ ] Audit executable scope with `rg -i 'taobao|taobao_ads|taobao_display_ads' src tests configs docs notebooks`; retain only the explicit raw-data README/historical records outside executable scope.
- [ ] Run: `python -m pytest tests/downstream/test_ad_optimization.py tests/contracts/test_output.py -q --basetemp .pytest-tmp/ads-task-5` and `python -m ruff check src/downstream tests/downstream configs`.
- [ ] Commit: `chore: remove abandoned Taobao Ads pipeline`

### Task 6: Full-corpus smoke validation and final audit

**Files:**
- Modify: `tests/downstream/test_ad_optimization.py`
- Modify: `docs/kaggle_ad_optimization_runbook.md`

**Interfaces:**
- Produces: a deterministic CPU smoke path with fixtures and a documented post-pretraining Kaggle command sequence.

- [ ] Add a test asserting full-corpus config rejects missing full checkpoint artifacts rather than falling back to pilot/Taobao paths.
- [ ] Add a small CPU fixture smoke test that runs one train epoch and validation/test export with a fake frozen embedder, then reloads the ranker manifest.
- [ ] Run focused suite and `python -m ruff check src/downstream tests/downstream`.
- [ ] Run full suite: `python -m pytest -q --basetemp .pytest-tmp/full-suite`.
- [ ] Commit: `test: harden Green SM Ads full-corpus handoff`
