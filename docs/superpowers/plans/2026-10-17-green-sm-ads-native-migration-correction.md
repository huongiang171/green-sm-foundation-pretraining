# Green SM Native Ads Migration Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the published bridge-based Ads prototype with a native Green SM Ads CTR runner whose validation, training, reload, and export paths satisfy the full-corpus handoff contract.

**Architecture:** The canonical `src/downstream/ad_optimization.py` will own native Green SM data records, train-fitted feature artifacts, batching, ranker, lifecycle, and CLI. It must consume only the linked full corpus and resolve the frozen Foundation package via `checkpoint_manifest.json`; `src/downstream/green_sm_ad_optimization.py` is transitional code and is deleted only after native tests pass. Training can read labels, whereas validation/test export is label-free and goes exclusively through `OutputContractV1`.

**Tech Stack:** Python 3.13, PyTorch, PyArrow, PyYAML, pytest, Ruff, Kaggle T4.

**Spec:** `docs/superpowers/specs/2026-10-17-green-sm-ads-realignment-design.md`

## Global Constraints

- Use only `green_sm_sim_full_v1/green_sm_sim_linked_v1`; no Taobao file, schema alias, or runtime import is permitted in the executable Ads path.
- At `validate`, `train`, and `predict`, require data version `green_sm_sim_full_v1`, a passing data-quality report, immutable opportunity checksums, and the exact linked Foundation manifest identity.
- Resolve the selected Foundation artifact exclusively from `checkpoint.package_root/checkpoint_manifest.json`; verify the selected checkpoint SHA and `vocabulary.json` SHA before loading it.
- Require the selected checkpoint `corpus == "linked"`; reject missing, placeholder, or mismatched configured checksum values before loaders or models are built.
- Foundation stays frozen. Only `embedder.embed_users()` runs under `torch.no_grad()`; ranker forward, BCE, and backward run outside it.
- Train candidate-weighted BCE over every real train candidate, including no-click requests; do not move or resample the published time-based splits.
- Candidate latitude/longitude are contract-validated only. Ranker v1 uses `candidate_geo_cell`, not a second coordinate feature.
- Export every validation/test candidate via `OutputContractV1` with a probability in `[0,1]`; never include `clicked` in feature tensors for test prediction or in output artifacts.
- Retain tracked `data/taobao_data/*.csv` and `data/README_taobao.md`; delete only obsolete Taobao ETL/runtime materials listed in Task 5.
- Do not stage datasets, checkpoints, Kaggle working outputs, `.ai-log/`, or pytest temp directories.

## Review Focus

- **Selection manifest:** `selected_kind: "last"` must work exactly like `"best"` when its selected file and identities match; a bad/placeholder SHA must fail in `validate` before model construction. Covered by Task 1.
- **No label leakage:** changing test `clicked` values must leave feature tensors and scores unchanged, and Parquet output must not contain `clicked`. Covered by Tasks 2 and 4.
- **Real candidate shape:** padding is never exported or trained; no-click requests and deterministic category/placement/geo `UNK` cases remain valid. Covered by Task 2.
- **Foundation freeze:** every Foundation gradient is `None`, while at least one Ads-ranker gradient exists after a train step. Covered by Task 3.
- **Migration residue:** after removal, no `taobao`, `cate_id`, or `pid` reference remains in executable Ads source, tests, config, Kaggle notebook, or runbook. Covered by Task 5.

---

## File Map

- `src/downstream/ad_optimization.py`: sole public Green SM Ads module: native records, handoff checks, features, batches, ranker, training/reload/predict CLI.
- `tests/downstream/test_ad_optimization.py`: Green SM-native unit and lifecycle regression suite.
- `tests/downstream/test_green_sm_ads_native.py`: small native-schema regressions already started locally; merge its cases into the canonical suite, then delete this temporary test file.
- `configs/ad_optimization.yaml`: explicit full-corpus/Kaggle configuration with non-placeholder selected-checkpoint identity requirements.
- `notebooks/ad_optimization_kaggle.ipynb`: Kaggle execution sequence that first validates, then trains and exports only verified artifacts.
- `docs/kaggle_ad_optimization_runbook.md`: human equivalent of the notebook and Baseline handoff description.
- `src/downstream/green_sm_ad_optimization.py`: legacy bridge to remove after the native suite passes.

### Task 1: Lock the native handoff contract in regression tests

**Files:**
- Modify: `tests/downstream/test_ad_optimization.py`
- Modify: `tests/downstream/test_green_sm_ads_native.py`

**Interfaces:**
- Produces: tests defining `GreenSmAdsOpportunity.from_mapping(row, *, require_label)`, `resolve_selected_foundation_handoff(config)`, and `validate_ads_data_handoff(config)` behavior.
- Consumes: Green SM Ads fields `candidate_category`, `placement`, `candidate_geo_cell`, `candidate_lat`, `candidate_lon`, and official checkpoint-selection manifest fields.

- [ ] **Step 1: Write failing handoff tests.**

  Cover: correct native row; absent `cate_id`/`pid`; non-empty geo cell and finite lat/lon; timezone-aware cutoff/availability order; request-level user/snapshot/cutoff consistency; a no-click request; and request-local candidate uniqueness.

- [ ] **Step 2: Write failing selection-manifest tests.**

  Build minimal package fixtures where `checkpoint_manifest.json` selects `best.pt` and then `last.pt`. Assert both resolve only when selected filename SHA, linked Foundation-manifest SHA, vocabulary SHA, configured selected SHA, and `corpus == "linked"` all agree. Assert placeholders or mismatches fail from `validate_ads_data_handoff()` before any model loader is invoked.

- [ ] **Step 3: Run the focused test to verify red state.**

  Run: `python -m pytest tests/downstream/test_ad_optimization.py tests/downstream/test_green_sm_ads_native.py -q --basetemp .pytest-tmp/ads-correction-task-1`

  Expected: failure because old Taobao-shaped records/bridge cannot satisfy native field and selection assertions.

- [ ] **Step 4: Add only native record and selection-validation interfaces needed for the tests.**

  `GreenSmAdsOpportunity.from_mapping()` must accept a mapping and `require_label: bool`; when false it must neither require nor retain `clicked`. `resolve_selected_foundation_handoff()` must return the selected checkpoint path plus verified identity fields, never infer `best.pt`.

- [ ] **Step 5: Run the focused tests to verify green state.**

  Run: `python -m pytest tests/downstream/test_ad_optimization.py tests/downstream/test_green_sm_ads_native.py -q --basetemp .pytest-tmp/ads-correction-task-1-green`

  Expected: PASS.

- [ ] **Step 6: Commit the contract boundary.**

  ```powershell
  git add src/downstream/ad_optimization.py tests/downstream/test_ad_optimization.py tests/downstream/test_green_sm_ads_native.py
  git commit -m "test: define native Green SM Ads handoff contract"
  ```

### Task 2: Rebuild candidate features and batching without Taobao aliases

**Files:**
- Modify: `src/downstream/ad_optimization.py`
- Modify: `tests/downstream/test_ad_optimization.py`
- Delete: `tests/downstream/test_green_sm_ads_native.py`

**Interfaces:**
- Produces: `AdsFeatureArtifacts.fit(rows)`, `load_green_sm_ads_opportunities(root, split, *, require_labels)`, `build_ads_batches(rows, snapshots, artifacts, *, require_labels)`, and a native `AdsBatch` whose labels are optional.
- Consumes: verified native opportunities from Task 1 and `EncodedSample` snapshots selected from the linked vocabulary.

- [ ] **Step 1: Write failing feature/batch tests.**

  Assert train-only category/placement/geo vocabulary fits PAD and UNK; `candidate_id`/`campaign_id` use exact `green-sm-ads-v1` hashing; batch dimensions are request-major; `candidate_mask` excludes padding; and a test batch built with `require_labels=False` has no label tensor.

- [ ] **Step 2: Write the label-leakage regression.**

  Construct two otherwise identical test row fixtures whose `clicked` values differ. Build prediction features with `require_labels=False` and assert equal tensors, expected candidates, and scores. Assert every `PredictionRow`/Parquet column is label-free.

- [ ] **Step 3: Run to verify red state.**

  Run: `python -m pytest tests/downstream/test_ad_optimization.py -q --basetemp .pytest-tmp/ads-correction-task-2`

  Expected: failure because the legacy adapter maps native rows into `cate_id`/`pid` and propagates click labels.

- [ ] **Step 4: Implement the native feature path.**

  Keep candidate category, placement, and geo cell as separate train-fitted vocabularies with PAD/UNK; hash candidate and campaign IDs only. Validate latitude/longitude but do not encode them. Make `AdsBatch.labels: torch.Tensor | None`, and prohibit a BCE call without labels.

- [ ] **Step 5: Move the three local `test_green_sm_ads_native.py` cases into the canonical suite and remove that temporary file.**

  The native schema, finite geo, and hash-namespace tests belong beside the feature implementations they specify.

- [ ] **Step 6: Run to verify green state.**

  Run: `python -m pytest tests/downstream/test_ad_optimization.py -q --basetemp .pytest-tmp/ads-correction-task-2-green`

  Expected: PASS.

- [ ] **Step 7: Commit native primitives.**

  ```powershell
  git add src/downstream/ad_optimization.py tests/downstream/test_ad_optimization.py
  git rm tests/downstream/test_green_sm_ads_native.py
  git commit -m "feat: build native Green SM Ads features and batches"
  ```

### Task 3: Implement frozen-Foundation ranker and verified train lifecycle

**Files:**
- Modify: `src/downstream/ad_optimization.py`
- Modify: `tests/downstream/test_ad_optimization.py`
- Modify: `configs/ad_optimization.yaml`

**Interfaces:**
- Produces: `AdsRanker`, `frozen_foundation_ads_logits(embedder, batch, ranker)`, `train_ads_ranker(...)`, and `run_ads_training(config)`.
- Consumes: Task 1 handoff and Task 2 batch/artifact interfaces; public `src.training.checkpoint.load_foundation_checkpoint` and `OutputContractV1`.

- [ ] **Step 1: Write failing model/lifecycle tests.**

  Assert ranker takes Foundation long/short vectors plus native candidate features and emits one logit per batch slot. After one backward pass, all Foundation gradients are `None` and at least one ranker gradient is non-`None`. Assert one CPU fixture epoch writes a new `artifact_dir/ad_optimization/<run_id>/` containing feature artifacts, `last.pt`, `best.pt`, metrics JSONL, and an identity-bound run manifest.

- [ ] **Step 2: Write failing validation-gate tests.**

  Starting from full config fixtures, fail `--mode validate` for non-full `data_version`, failed quality report, missing selected checkpoint, placeholder SHA, selected `corpus != linked`, and any selected-checkpoint/Foundation-manifest/vocabulary mismatch. Assert loader/model creation is not reached.

- [ ] **Step 3: Run to verify red state.**

  Run: `python -m pytest tests/downstream/test_ad_optimization.py -q --basetemp .pytest-tmp/ads-correction-task-3`

  Expected: failure because the current prototype's bridge is broad/no-native and validate does not fully gate selected identity.

- [ ] **Step 4: Implement the ranker and training path.**

  Derive Foundation width from the loaded embedder; fuse long/short user vectors with candidate/campaign hashed embeddings and category/placement/geo embeddings. Put `embedder.embed_users()` alone in `torch.no_grad()`. Optimize only `AdsRanker` via masked BCE over real candidates; choose best by validation BCE.

- [ ] **Step 5: Implement auditable run creation and reload contract.**

  Refuse an existing run directory. Bind the Ads manifest SHA, Foundation manifest SHA, selected Foundation checkpoint SHA, vocabulary SHA, config digest, seed, request counts, artifacts SHA, and model settings into the Ads run manifest. Reload must compare every binding before scoring.

- [ ] **Step 6: Make config and CLI strict.**

  Implement `python -m src.downstream.ad_optimization --config <yaml> --mode validate|train|predict`. In every mode call full handoff validation first. Keep full Kaggle paths explicit in `configs/ad_optimization.yaml`; it must contain actual hashes only when the package is available, otherwise validation fails instead of falling back.

- [ ] **Step 7: Run to verify green state.**

  Run: `python -m pytest tests/downstream/test_ad_optimization.py tests/contracts/test_output.py -q --basetemp .pytest-tmp/ads-correction-task-3-green`

  Expected: PASS.

- [ ] **Step 8: Commit lifecycle.**

  ```powershell
  git add src/downstream/ad_optimization.py tests/downstream/test_ad_optimization.py configs/ad_optimization.yaml
  git commit -m "feat: add verified Green SM Ads training lifecycle"
  ```

### Task 4: Add reloadable label-free prediction/export behavior

**Files:**
- Modify: `src/downstream/ad_optimization.py`
- Modify: `tests/downstream/test_ad_optimization.py`
- Modify: `tests/contracts/test_output.py`

**Interfaces:**
- Produces: `run_ads_prediction(config, *, split: Literal["validation", "test"]) -> OutputPaths`.
- Consumes: Task 3 Ads run manifest and checkpoint plus `OutputContractV1`.

- [ ] **Step 1: Write failing export/reload tests.**

  Train a tiny fixture ranker, reload it, and predict both validation and test. Assert separate `predictions/validation/` and `predictions/test/` directories, all real candidates exactly once, scores in `[0, 1]`, no padding rows, and a verified split manifest/report each.

- [ ] **Step 2: Extend the leakage test through actual Parquet export.**

  Flip only test labels in source input, run prediction twice, and compare request/candidate/score rows. Assert neither expected-candidate construction nor output schema includes `clicked`.

- [ ] **Step 3: Run to verify red state.**

  Run: `python -m pytest tests/downstream/test_ad_optimization.py tests/contracts/test_output.py -q --basetemp .pytest-tmp/ads-correction-task-4`

  Expected: failure until prediction uses native optional-label batches and split-specific `OutputContractV1` exports.

- [ ] **Step 4: Implement verified prediction.**

  Reload and verify Ads and Foundation identities. Build validation/test prediction batches with `require_labels=False`; flatten only `candidate_mask` positions; transform logits with sigmoid; and hand rows plus verified candidate index to `OutputContractV1`.

- [ ] **Step 5: Run to verify green state.**

  Run: `python -m pytest tests/downstream/test_ad_optimization.py tests/contracts/test_output.py -q --basetemp .pytest-tmp/ads-correction-task-4-green`

  Expected: PASS.

- [ ] **Step 6: Commit prediction/export.**

  ```powershell
  git add src/downstream/ad_optimization.py tests/downstream/test_ad_optimization.py tests/contracts/test_output.py
  git commit -m "feat: export label-free Green SM Ads predictions"
  ```

### Task 5: Complete the migration, operations handoff, and cleanup

**Files:**
- Delete: `src/downstream/green_sm_ad_optimization.py`
- Delete: `tests/downstream/test_green_sm_ad_optimization.py`
- Delete if present: `configs/taobao_ads_etl.json`, `src/data/taobao_ads/`, `tests/data/test_taobao_ads_etl.py`, `docs/kaggle_taobao_ads_etl_runbook.md`, `docs/taobao_ads_data_contract.md`, `docs/superpowers/plans/2026-10-12-ad-optimization-kaggle.md`, `docs/superpowers/plans/2026-10-13-taobao-ads-foundation-etl.md`
- Modify: `notebooks/ad_optimization_kaggle.ipynb`
- Modify: `docs/kaggle_ad_optimization_runbook.md`
- Modify: `configs/ad_optimization.yaml`

**Interfaces:**
- Produces: one native Ads CLI and a Kaggle runbook/notebook whose actions are validate, train, verify, predict-validation, predict-test, and export.
- Consumes: Tasks 1–4 only; no bridge or Taobao ETL types.

- [ ] **Step 1: Prove native suite passes before deleting the bridge.**

  Run: `python -m pytest tests/downstream/test_ad_optimization.py tests/contracts/test_output.py -q --basetemp .pytest-tmp/ads-correction-before-delete`

  Expected: PASS.

- [ ] **Step 2: Delete the bridge and obsolete runtime materials.**

  Remove only the paths listed above if they exist. Do not delete `data/taobao_data/*.csv`, `data/README_taobao.md`, user-authored historical documents outside the listed scope, input datasets, or generated Kaggle artifacts.

- [ ] **Step 3: Replace the Kaggle notebook and runbook.**

  Notebook cells must discover exactly one full linked Green SM root and checkpoint package, assert CUDA, populate config identity values from the package, run `--mode validate`, then train, verify, and export validation/test output. All outputs remain under `/kaggle/working`; notebook must not hard-code `best.pt` outside the selection manifest. The runbook documents the same order and gives the Baseline team the two prediction directories plus their manifests/reports.

- [ ] **Step 4: Audit executable references.**

  Run: `rg -n -i 'taobao|taobao_ads|cate_id|pid' src/downstream tests/downstream configs/ad_optimization.yaml notebooks/ad_optimization_kaggle.ipynb docs/kaggle_ad_optimization_runbook.md`

  Expected: no matches. If an allowed historical/raw record is intentionally retained, audit it separately and do not weaken this executable-scope command.

- [ ] **Step 5: Run final verification.**

  Run:

  ```powershell
  python -m pytest tests/downstream/test_ad_optimization.py tests/contracts/test_output.py -q --basetemp .pytest-tmp/ads-correction-final
  python -m ruff check src/downstream tests/downstream configs
  python -c "import json; json.load(open('notebooks/ad_optimization_kaggle.ipynb', encoding='utf-8')); print('valid notebook JSON')"
  ```

  Expected: all commands succeed.

- [ ] **Step 6: Commit final migration.**

  ```powershell
  git add -A src/downstream tests/downstream tests/contracts configs docs/kaggle_ad_optimization_runbook.md notebooks/ad_optimization_kaggle.ipynb
  git commit -m "feat: complete native Green SM Ads migration"
  ```

## Plan Self-Review

- **Spec coverage:** Tasks 1 and 3 implement the full handoff gate; Task 2 owns native data and anti-leak input boundaries; Task 3 owns frozen training and audited artifacts; Task 4 owns label-free output; Task 5 owns migration deletion, Kaggle operation, and audit.
- **Type consistency:** all later tasks consume `GreenSmAdsOpportunity`, `AdsBatch`, and resolved selection objects introduced earlier; no later task relies on the bridge.
- **Review focus coverage:** every focus item is attached to a named task and an executable regression test.
- **Scope discipline:** raw Taobao CSV/README remain untouched; the plan does not add a baseline evaluator or a production-uplift claim.
