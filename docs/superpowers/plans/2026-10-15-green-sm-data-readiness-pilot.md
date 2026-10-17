# Green SM Data Readiness and Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate deterministic paired linked/control Green SM corpora, validate that they satisfy explicit pretraining-data quality gates, and publish a 2% pilot artifact before generating the full corpus.

**Architecture:** A deterministic simulator first creates one target world containing users, catalogues, temporal decision cutoffs, candidates, and labels. It derives two Foundation-history views from that world: a linked view that retains informative cross-service episodes and a paired-control view that resamples other-service observations without changing logical downstream targets. A validator reads the emitted Parquet artifacts through the real Foundation loader and writes a machine-readable readiness report.

**Tech Stack:** Python 3, dataclasses, `random.Random`, PyArrow Parquet, JSON configuration, pytest, existing `src.foundation_input` manifest/loader/validation APIs.

**Spec:** `docs/superpowers/specs/2026-10-15-green-sm-simulated-corpus-design.md`

## Global Constraints

- Generate no real user, location, address, map, or external-geocoder data.
- Use one deterministic city with 64 cells, 12,000 full-corpus users, 42 days, and `Asia/Ho_Chi_Minh`.
- Use temporal splits: days 1–28 train, days 29–35 validation, days 36–42 test.
- Full corpus targets: 96,000 / 24,000 / 24,000 Foundation snapshots and the Food/Ads request counts in the spec.
- Pilot fraction is exactly `0.02`: 240 users, 1,920 / 480 / 480 snapshots; Food 360 / 60 / 60 requests; Ads 480 / 80 / 80 requests.
- Do not overwrite an existing output root; write immutable manifests and SHA-256 checksums.
- Final artifacts require the upstream FoundationBatch v3 geo contract, `max_tokens=max_length=256`, and real `GeoAdapter` routing. Until then, fixtures may use the published v3 JSON contract only.
- Fit vocabulary/scaler metadata from the logical train split only; validation and test are transform-only.
- Linked and paired-control downstream logical rows must be identical after excluding the Foundation-manifest checksum field.

## Review Focus

- Same timestamp ordering: request-target events must never enter their own snapshot, including when other events share the cutoff timestamp; test in Task 4.
- Event-group boundaries: a generated group with multiple semantic fields must remain complete after history selection and must not exceed seven tokens; test in Task 4.
- Train/test leakage through catalogue or normalization: test-day fields and coordinate statistics must not change train vocabulary/scaler metadata; test in Tasks 4 and 5.
- Paired control integrity: a control row must retain every target, candidate, label, user, and cutoff but remove predictive other-service episode linkage; test in Tasks 3 and 5.
- Cold-start validity: test cold users must have no train snapshot/event and must not resolve to a learned train user-ID row; test in Task 2 and report in Task 5.

---

## Quantitative gates to add to the design specification

These are implementation defaults for this project. They are deliberately configuration values, not hidden constants.

- `min_train_event_rows_per_event_type: 5_000` in a full run; scale by pilot fraction with a floor of `100`.
- `max_train_event_type_share: 0.45`.
- `min_train_valid_next_action_rate: 0.75`.
- `min_train_maskable_tokens`: categorical `100_000`, numeric `20_000`, text `10_000`, geo `50_000`; scale by pilot fraction with a floor of `100` per modality.
- `min_valid_geo_rate: 0.60` among emitted geo tokens.
- `max_non_user_oov_rate_validation_test: 0.01` for closed catalogue categorical fields; free text and user IDs are reported separately.
- `cold_start_test_user_rate: 0.15`; cold users first become active on or after day 29 and have no train snapshots.
- `min_food_hard_negatives_per_request: 10`.
- Food candidate positive rate must be in `[0.05, 0.20]`; Ads candidate click rate must be in `[0.02, 0.15]`; Ads ranking-eligible request rate must be at least `0.30`.
- Paired-control audit: adding other-service control features to a fixed validation probe may improve AUC by at most `0.01` above the declared base-feature probe.

### Task 1: Add the quantitative readiness contract and Green SM configuration

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-green-sm-simulated-corpus-design.md`
- Create: `configs/green_sm_sim_pilot.json`
- Create: `configs/green_sm_sim_full.json`
- Create: `src/data/green_sm/config.py`
- Test: `tests/data/test_green_sm_config.py`

**Interfaces:**
- Produces: `GreenSmConfig.from_json(path: str | Path) -> GreenSmConfig`.
- Produces: `ReadinessThresholds` dataclass with `for_fraction(fraction: float) -> ReadinessThresholds`.
- Consumed by: every later task.

- [ ] **Step 1: Add the quantitative gate list above and the cold-start cohort to the specification**

- [ ] **Step 2: Write failing tests for config parsing and pilot scaling**

```python
def test_pilot_scales_counts_but_keeps_proportions(tmp_path: Path) -> None:
    config = GreenSmConfig.from_json(_write_pilot_config(tmp_path))
    assert config.user_count == 240
    assert config.snapshot_counts == {"train": 1920, "validation": 480, "test": 480}
    assert config.thresholds.min_train_event_rows_per_event_type == 100
    assert config.thresholds.min_ads_ranking_eligible_rate == 0.30
```

- [ ] **Step 3: Run the config test to verify it fails**

Run: `pytest tests/data/test_green_sm_config.py -v`

Expected: FAIL because `GreenSmConfig` does not exist.

- [ ] **Step 4: Implement `GreenSmConfig`, `ReadinessThresholds`, and immutable JSON configs**

`GreenSmConfig` must include seed, mode, user count, output root, 42-day split boundaries, snapshot/request targets, `max_history_events=30`, `max_tokens=256`, and all quality thresholds. Reject a pilot fraction other than `0.02`, an invalid split total, an output root that already contains files, or a threshold outside `[0, 1]` where applicable.

- [ ] **Step 5: Run the config test to verify it passes**

Run: `pytest tests/data/test_green_sm_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-10-15-green-sm-simulated-corpus-design.md configs/green_sm_sim_pilot.json configs/green_sm_sim_full.json src/data/green_sm/config.py tests/data/test_green_sm_config.py
git commit -m "docs: define Green SM data readiness gates"
```

### Task 2: Create deterministic catalogue, user cohorts, and target world

**Files:**
- Create: `src/data/green_sm/world.py`
- Test: `tests/data/test_green_sm_world.py`

**Interfaces:**
- Consumes: `GreenSmConfig`.
- Produces: `build_target_world(config: GreenSmConfig) -> TargetWorld`.
- `TargetWorld` exposes `catalogue`, `users`, `food_requests`, and `ads_requests`.
- Every `LogicalRequest` contains `request_id`, `user_id`, `cutoff_time`, `split`, `candidates`, and labels; it contains no Foundation-manifest field.

- [ ] **Step 1: Write failing deterministic-world and cold-start cohort tests**

```python
def test_target_world_is_seed_deterministic_and_contains_cold_test_users() -> None:
    first = build_target_world(_pilot_config(seed=7))
    second = build_target_world(_pilot_config(seed=7))
    assert first.fingerprint() == second.fingerprint()
    assert len(first.cold_start_user_ids) == 36
    assert all(user.first_active_day >= 29 for user in first.cold_start_users())
```

- [ ] **Step 2: Run the world test to verify it fails**

Run: `pytest tests/data/test_green_sm_world.py -v`

Expected: FAIL because `build_target_world` does not exist.

- [ ] **Step 3: Implement `build_target_world(config: GreenSmConfig) -> TargetWorld`**

Use per-entity deterministic seeds derived from `config.seed`, never global RNG state. Create the fixed city/catalogue before day 1. Allocate 15% of users to the cold-start cohort and assign their first activity no earlier than validation day 1. Generate logical Food and Ads requests once; candidate IDs, candidate features, labels, user IDs, cutoffs, and splits are shared by both output views. Generate Food hard negatives according to the configured minimum.

- [ ] **Step 4: Add distribution tests for exact request counts, unique candidates, hard negatives, label-rate ranges, and no latent truth in public objects**

- [ ] **Step 5: Run world tests to verify they pass**

Run: `pytest tests/data/test_green_sm_world.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data/green_sm/world.py tests/data/test_green_sm_world.py
git commit -m "feat: create deterministic Green SM target world"
```

### Task 3: Derive linked and paired-control event histories

**Files:**
- Create: `src/data/green_sm/histories.py`
- Test: `tests/data/test_green_sm_histories.py`

**Interfaces:**
- Consumes: `TargetWorld` from Task 2.
- Produces: `build_history_views(world: TargetWorld) -> tuple[HistoryView, HistoryView]` where the order is `(linked, paired_control)`.
- `HistoryView.events_for(user_id: str)` returns chronological event groups, each with a stable `source_record_id`.

- [ ] **Step 1: Write failing paired-view equality and signal-removal tests**

```python
def test_control_preserves_targets_but_removes_episode_linkage() -> None:
    world = build_target_world(_pilot_config())
    linked, control = build_history_views(world)
    assert linked.logical_request_digest() == control.logical_request_digest()
    assert linked.events_for("u000001") != control.events_for("u000001")
    assert control.has_target_episode_marker() is False
```

- [ ] **Step 2: Run the histories test to verify it fails**

Run: `pytest tests/data/test_green_sm_histories.py -v`

Expected: FAIL because `build_history_views` does not exist.

- [ ] **Step 3: Implement linked event generation and paired-control resampling**

Generate Ride, Food, Delivery, and Ads event groups from the shared target world. In the linked view, pre-cutoff non-target-service observations may reveal the target episode only through approved location/time/service behavior. In control, preserve each user’s target-service history and resample other-service groups inside `(split, active-hour band, geo-cell, service-coverage)` strata without accessing label values at transform time. Keep group size at most seven semantic fields and record every group’s source index.

- [ ] **Step 4: Add tests for chronological order, identical target-world digest, complete groups, cold users having no train groups, and no request event in its own history**

- [ ] **Step 5: Run histories tests to verify they pass**

Run: `pytest tests/data/test_green_sm_histories.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data/green_sm/histories.py tests/data/test_green_sm_histories.py
git commit -m "feat: derive paired Green SM history views"
```

### Task 4: Emit immutable Foundation and downstream Parquet artifacts

**Files:**
- Create: `src/data/green_sm/build.py`
- Create: `src/data/green_sm/__init__.py`
- Create: `src/data/green_sm/__main__.py`
- Test: `tests/data/test_green_sm_build.py`

**Interfaces:**
- Consumes: `GreenSmConfig`, `TargetWorld`, and `HistoryView`.
- Produces: `build_dataset(config: GreenSmConfig) -> dict[str, object]`.
- Produces two roots named `green_sm_sim_linked_v1` and `green_sm_sim_control_paired_v1`, each containing Foundation, Food, Ads, manifests, and reports.

- [ ] **Step 1: Write failing build tests using the real Foundation manifest loader**

```python
def test_pilot_build_loads_every_split_and_preserves_paired_requests(tmp_path: Path) -> None:
    result = build_dataset(_pilot_config(output_root=tmp_path / "out"))
    for root in result["foundation_roots"].values():
        assert list(iter_normalized_samples(root, "train"))
        assert list(iter_normalized_samples(root, "validation"))
        assert list(iter_normalized_samples(root, "test"))
    assert result["paired_logical_rows_equal"] is True
```

- [ ] **Step 2: Run the build test to verify it fails**

Run: `pytest tests/data/test_green_sm_build.py -v`

Expected: FAIL because the Green SM builder does not exist.

- [ ] **Step 3: Implement Parquet emission, snapshots, manifests, checksums, and CLI**

Mirror the immutable-output and checksum conventions in `src/data/taobao_ads/build.py`. Write one chronological events shard and snapshots shard per user. Every snapshot must reference an exact contiguous history slice, use the right split boundary, and retain only pre-cutoff events. Emit paired Food/Ads opportunity files with the same logical request values and different Foundation-manifest checksums. The CLI command is `python -m src.data.green_sm --config <path>` and must build then validate.

- [ ] **Step 4: Add tests for refusing output overwrite, deterministic re-build fingerprint, max-256-token sample budget, candidate availability, snapshot cutoff equality, and test-only user absence from train**

- [ ] **Step 5: Run build tests to verify they pass**

Run: `pytest tests/data/test_green_sm_build.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data/green_sm tests/data/test_green_sm_build.py
git commit -m "feat: emit paired Green SM foundation artifacts"
```

### Task 5: Implement quality audits and readiness report

**Files:**
- Create: `src/data/green_sm/validate.py`
- Test: `tests/data/test_green_sm_validate.py`

**Interfaces:**
- Consumes: `GreenSmConfig` and roots emitted by Task 4.
- Produces: `validate_outputs(config: GreenSmConfig) -> dict[str, object]`.
- Produces: `reports/data_quality_report.json` and `reports/causal_audit_report.json` in each root.

- [ ] **Step 1: Write failing tests for each quantitative gate**

```python
def test_validator_rejects_control_with_predictive_other_service_history(tmp_path: Path) -> None:
    config = _build_valid_pilot(tmp_path)
    _inject_control_only_label_marker(config)
    with pytest.raises(ValueError, match="paired-control conditional independence"):
        validate_outputs(config)
```

- [ ] **Step 2: Run validator tests to verify they fail**

Run: `pytest tests/data/test_green_sm_validate.py -v`

Expected: FAIL because `validate_outputs` does not exist.

- [ ] **Step 3: Implement `validate_outputs(config: GreenSmConfig) -> dict[str, object]`**

Use the real Foundation loader for every split, then calculate all Task 1 thresholds: event/action support, class concentration, valid next-action proportion, maskable token counts per modality, geo validity, closed-vocabulary OOV rate, cold-start cohort size, Food hard-negative count, Food/Ads label rates, Ads ranking-eligible rate, paired logical-row equality, and paired-control conditional audit. Fail closed on every unmet threshold and persist all observed values with seed, revision, checksums, and config digest.

- [ ] **Step 4: Add tests for each failure class plus a complete valid-pilot report**

- [ ] **Step 5: Run validator tests to verify they pass**

Run: `pytest tests/data/test_green_sm_validate.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data/green_sm/validate.py tests/data/test_green_sm_validate.py
git commit -m "feat: validate Green SM pretraining readiness"
```

### Task 6: Run and archive the 2% data pilot

**Files:**
- Create: `docs/green_sm_pilot_runbook.md`
- Modify: `docs/superpowers/specs/2026-10-15-green-sm-simulated-corpus-design.md`
- Test: `tests/data/test_green_sm_build.py`

**Interfaces:**
- Consumes: `configs/green_sm_sim_pilot.json` and the CLI from Task 4.
- Produces: a versioned pilot manifest and readiness reports outside Git.
- Handoff to Member 3: the pilot root, config digest, vocabulary artifact after train-only fitting, and report JSONs.

- [ ] **Step 1: Write a failing integration test that invokes the pilot CLI and reads its report**

- [ ] **Step 2: Run it to verify it fails before the CLI/report implementation is complete**

Run: `pytest tests/data/test_green_sm_build.py -k pilot_cli -v`

Expected: FAIL until Tasks 4–5 are complete.

- [ ] **Step 3: Write the runbook and execute the pilot build locally or on Kaggle CPU**

The runbook must state that this is a data/contract pilot, not a performance run. Archive its immutable data version, manifest SHA-256 values, config digest, and readiness report. Do not publish private latent-truth files.

- [ ] **Step 4: Run the integration test and full data test suite**

Run: `pytest tests/data -v`

Expected: PASS.

- [ ] **Step 5: Commit source, configs, docs, and tests; do not commit generated Parquet data**

```bash
git add src/data/green_sm tests/data configs/green_sm_sim_pilot.json configs/green_sm_sim_full.json docs/green_sm_pilot_runbook.md docs/superpowers/specs/2026-10-15-green-sm-simulated-corpus-design.md
git commit -m "feat: add validated Green SM simulation pilot"
```

## Member 3 handoff gate: pretraining smoke run

This is intentionally outside the Data-team implementation plan because the repository currently has no pretraining objective or runner. Member 3 may start this only after the v3 geo contract and pilot readiness report pass.

1. Fit vocabulary/scalers from pilot train samples only and persist their metadata.
2. Run the planned objective for 500–1,000 optimizer steps with the pilot data.
3. Require finite total/component losses, no NaN/Inf gradients, a lower trailing mean loss than initial mean loss, and non-zero `UserIdAdapter.embedding.weight` gradient on valid batches.
4. Record checkpoint SHA-256, exact config, loss trace, and batch rejection count.
5. If any condition fails, diagnose the failing contract/objective before generating the full corpus.

## Self-review

- Spec coverage: Tasks 1–5 implement the paired corpus, temporal splits, geo-ready rows, manifests, candidate artifacts, leakage checks, and readiness audits. Task 6 creates the required pilot. The upstream v3 geo batch and Member 3 objective are explicitly gated rather than silently assumed.
- Step scan: each task has one public interface, failing tests, a bounded implementation, verification, and commit point.
- Type consistency: `GreenSmConfig` drives world, histories, build, and validation; `TargetWorld` is produced once and shared by both views; `HistoryView` is only an intermediate to artifact emission.
- Review focus: each listed risk has a named owning task/test.
- Proportion: the plan defines interfaces and acceptance values without reproducing simulator implementation code.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-15-green-sm-data-readiness-pilot.md`. Please review the plan. I recommend **Native** execution because the tasks share exact schema and deterministic-generation interfaces, and the current environment does not permit agent delegation. Does the plan capture what you want?
