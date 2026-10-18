# Food PoC Accelerated Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, leakage-safe Food downstream PoC that trains quickly enough for Kaggle by deriving a 20% train/validation candidate universe from the immutable full linked corpus, retaining the full test universe, and computing each frozen Foundation embedding once per run.

**Architecture:** The full Green SM corpus and full-linked Foundation checkpoint remain immutable. A deterministic derived opportunity artifact contains complete request groups for 20% of train and validation, but copies the complete test split. The Food runner validates the derived manifest’s parent linkage, caches long/short embeddings in memory once, trains only the ranker, persists per-epoch metrics/checkpoints, and streams epoch progress. The Kaggle notebook attaches both full corpus and derived PoC data, discovers only immediate input roots, and runs the new accelerated configuration.

**Tech Stack:** Python 3.10+, PyArrow, PyTorch, pytest, Kaggle T4 GPU.

**Spec:** This document records the user-approved deadline strategy from the conversation: full pretrained Foundation; 20% train/validation, full test; no future information or label leakage.

## Global Constraints

- Never mutate `green_sm_sim_full_v1`, the full checkpoint, or their manifests.
- Sampling unit is a complete `request_id`; all candidate rows are kept together.
- Use seed `20261018`, a 20% fraction, and stratify by `(positive_count, primary_positive_category)`.
- The derived manifest keeps the original `schema_version` and `foundation_manifest_sha256`, retains `data_version=green_sm_sim_full_v1`, and records parent checksums plus a `derivation` object.
- Train and validation use the derived files; test is byte-for-byte the full source test parquet.
- Cache identity includes Food manifest SHA, Foundation checkpoint SHA, vocabulary SHA, split, max tokens, and d_model. Cached tensors contain no labels.
- Emit an epoch line and atomically refresh `train_metrics.jsonl`, `last.pt`, and `best.pt` at every epoch.
- The Kaggle notebook must not recursively scan `/kaggle/input`.

## Review Focus

- A selected request must include every source candidate and preserve its source order; Task 1 tests this.
- No train/validation/test request may appear in a different split; Task 1 tests source split and parent SHA binding.
- A changed label must not affect selection except through the declared, audit-only stratification fields; Task 1 records the selector inputs and output request IDs.
- Cache reuse with a mismatched checkpoint/vocabulary/manifest identity must fail rather than silently reuse embeddings; Task 2 tests this.
- Test inference must still use all 60,000 full-test Food candidate rows; Task 3 verifies the generated Kaggle config points test to the derived manifest whose test checksum equals the parent checksum.

---

### Task 1: Build and validate the derived Food PoC opportunity artifact

**Files:**
- Create: `src/data/green_sm/poc.py`
- Create: `tests/data/test_green_sm_food_poc.py`
- Modify: `src/data/green_sm/__init__.py`

**Interfaces:**
- Produces `build_food_poc_subset(source_root, destination_root, *, fraction=0.2, seed=20261018) -> dict[str, object]` and CLI `python -m src.data.green_sm.poc`.
- Consumes the full Food opportunity manifest and emits `food_opportunities/manifest.json` plus three parquet files.

- [ ] Write failing tests for complete-request sampling, 20% request counts per train/validation stratum, unchanged full test SHA, and rejection of non-empty output directories.
- [ ] Run `python -m pytest tests/data/test_green_sm_food_poc.py -q --basetemp .pytest-tmp/food-poc-red` and verify the API is absent.
- [ ] Implement manifest verification, deterministic stratified request selection, atomic parquet/manifest writing, and an auditable `derivation` block.
- [ ] Run the focused test green, generate `data/processed/green_sm_food_poc20_v1` locally from the existing full corpus, then validate row/request counts and checksums.

### Task 2: Cache frozen Food embeddings and publish epoch progress

**Files:**
- Modify: `src/downstream/food_recommendation.py`
- Modify: `tests/downstream/test_food_recommendation.py`

**Interfaces:**
- Produces an internal cached-batch representation containing CPU long/short tensors and original candidate tensors, plus `build_food_embedding_cache(...)`.
- `train_food_ranker(...)` consumes cached train/validation batches and writes one JSONL metric/checkpoint per completed epoch.

- [ ] Write failing tests proving `embed_users()` is called once per batch across multiple epochs, ranker gradients remain non-null, cache identity mismatch rejects, and metrics file contains one line after each epoch.
- [ ] Run the focused tests and confirm they fail because embedding work is repeated per epoch and metrics are deferred.
- [ ] Implement in-memory caching after verified batch construction; keep Foundation inside `torch.no_grad()`, keep ranker outside it, transfer cached tensors to the selected device per ranker batch, and print `Food epoch <n>/<N>` with `flush=True`.
- [ ] Run focused Food tests and existing downstream tests.

### Task 3: Ship a deadline-safe Food Kaggle workflow

**Files:**
- Modify: `configs/food_recommendation_full.yaml`
- Modify: `notebooks/food_recommendation_kaggle.ipynb`
- Create: `configs/food_recommendation_poc20.yaml`
- Create: `tests/downstream/test_food_poc_notebook.py`

**Interfaces:**
- The notebook clones the implementation branch, locates attached datasets only with `Path('/kaggle/input').iterdir()`, validates the full Foundation/checkpoint and PoC opportunity artifact, writes runtime YAML, trains 6 epochs, predicts validation/test, and zips the run directory.

- [ ] Write failing notebook/config tests for no `rglob`, PoC manifest discovery, 6 epochs, and use of the full test parquet checksum.
- [ ] Implement the YAML and notebook. The notebook may use the full corpus only for Foundation snapshots and the derived PoC dataset only for Food opportunities.
- [ ] Run notebook structural tests, then run `--mode validate` locally against the derived artifact.

### Task 4: Verify and hand off

**Files:**
- Modify: `docs/superpowers/plans/2026-10-18-food-poc-accelerated.md`

- [ ] Run `python -m pytest tests/data/test_green_sm_food_poc.py tests/downstream/test_food_recommendation.py tests/downstream/test_food_poc_notebook.py -q --basetemp .pytest-tmp/food-poc-full`.
- [ ] Run `python -m ruff check src/data/green_sm/poc.py src/downstream/food_recommendation.py tests/data/test_green_sm_food_poc.py tests/downstream/test_food_poc_notebook.py`.
- [ ] Record generated artifact counts/checksums and the measured local cache-build timing. Commit code/docs but leave generated data ignored; package the derived data separately for Kaggle upload.

## Execution Notes

- The user explicitly requested immediate implementation; execute natively in the current workspace rather than waiting for a separate execution-mode decision.
- Estimated Kaggle T4 timing after caching: cache build roughly 10–30 minutes, 6 ranker epochs roughly 5–20 minutes, validation/test export roughly 10–25 minutes. Exact time depends on mounted-input I/O; this is a target estimate, not a guarantee.
