# Shared Green SM Downstream Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only, reproducible evaluator and comparison CLI for label-free Food and Ads predictions on the immutable Green SM linked corpus.

**Architecture:** Artifact ingestion is separate from metrics. A loader verifies existing Foundation `OutputContractV1` or a baseline `evaluation_input_v1.json`, then exposes one canonical prediction view. Pure metric functions receive only this verified view and evaluator-only labels; reports record policy and source identity, so invalid comparisons fail before a table is rendered.

**Tech Stack:** Python 3.10+, PyArrow, scikit-learn 1.7.2, pytest, argparse; CPU only.

**Spec:** `docs/superpowers/specs/2026-10-17-shared-downstream-evaluation-design.md`

## Global Constraints

- Evaluate only `validation` or `test` against the immutable linked-corpus Food/Ads opportunity table.
- Reject label fields (`clicked`, `is_positive`, `label`) in prediction Parquet; reject duplicate, missing, and extra candidates.
- Candidate equality requires `request_id`, `candidate_id`, `user_id`, `split`, and equal timezone-aware UTC cutoff instant.
- Policy values are exactly `green_sm_evaluator_v1`, `green_sm_metric_policy_v1`, `score_desc__candidate_id_asc`, and `parse_to_utc_instant`.
- Food uses binary `is_positive`; Ads uses binary `clicked`, all rows for pointwise metrics, and positive-click requests only for ranking metrics.
- Do not modify Food/Ads training, checkpoints, corpus generation, or existing Foundation prediction exports merely to evaluate them.
- Baselines must declare non-applicable Foundation provenance; they must never create fake SHA values.
- Metric code and the evaluation notebook run on CPU only.

## Review Focus

- An offset-equivalent timestamp passes, while a naive or different instant fails (Task 3).
- A baseline with a forged/stale candidate manifest or split SHA fails before labels are read (Task 3).
- Tied scores rank by ascending Unicode-code-point `candidate_id`, never input order (Task 2).
- Single-class Ads emits JSON `null` AUC/AP plus a reason, but preserves LogLoss/Brier (Tasks 2 and 4).
- Reports differing in policy, evaluator, task, split, or candidate identity are rejected by comparison (Task 5).

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Pin the common metric implementation. |
| `src/evaluation/policy.py` | Policy/version/score-semantics constants. |
| `src/evaluation/types.py` | Canonical prediction, provenance, identity, and report types. |
| `src/evaluation/metrics.py` | Pure Food and Ads metric functions. |
| `src/evaluation/artifacts.py` | Verify Foundation V1/baseline artifacts and immutable opportunities. |
| `src/evaluation/runner.py` | Join verified data to labels and atomically write reports. |
| `src/evaluation/compare.py` | Strict report comparison gate. |
| `src/evaluation/__main__.py` | `evaluate` and `compare` CLI. |
| `tests/evaluation/` | Unit, artifact, runner, comparison, and CLI tests. |
| `docs/evaluation_input_v1.md` | Baseline prediction/sidecar contract. |
| `notebooks/downstream_evaluation_kaggle.ipynb` | CPU-only Kaggle runbook. |

### Task 1: Establish the versioned policy and shared types

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/evaluation/__init__.py`
- Create: `src/evaluation/policy.py`
- Create: `src/evaluation/types.py`
- Create: `tests/evaluation/test_policy.py`

**Interfaces:**
- Produces `EVALUATOR_VERSION`, `METRIC_POLICY`, and `MetricPolicy.identity() -> dict[str, str]`.
- Produces frozen dataclasses `CanonicalPrediction`, `SourceIdentity`, `ModelProvenance`, `VerifiedEvaluationInput`, and `EvaluationReport`.
- Later tasks must import these types rather than use anonymous dictionaries at module boundaries.

- [ ] **Step 1: Write the failing policy test**

```python
def test_metric_policy_identity_is_complete() -> None:
    assert METRIC_POLICY.identity() == {
        "evaluator_version": "green_sm_evaluator_v1",
        "metric_policy_version": "green_sm_metric_policy_v1",
        "tie_break_policy": "score_desc__candidate_id_asc",
        "timestamp_policy": "parse_to_utc_instant",
    }
```

Also test that Food requires `ranking` score semantics and Ads requires `probability`.

- [ ] **Step 2: Verify the test fails**

Run: `python -m pytest tests/evaluation/test_policy.py -v --basetemp .pytest-tmp/policy`

Expected: FAIL because policy/types modules do not exist.

- [ ] **Step 3: Implement the policy/type boundary**

Add `scikit-learn==1.7.2` to runtime dependencies. Implement exact policy constants and task-score semantics in `policy.py`. Implement the frozen dataclasses in `types.py`; `ModelProvenance` must represent baseline Foundation identity as `{"applicable": false}`, not a dummy checksum.

- [ ] **Step 4: Verify policy tests pass**

Run: `python -m pytest tests/evaluation/test_policy.py -v --basetemp .pytest-tmp/policy`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/evaluation/__init__.py src/evaluation/policy.py src/evaluation/types.py tests/evaluation/test_policy.py
git commit -m "feat: define versioned downstream evaluation policy"
```

### Task 2: Implement deterministic Food and Ads metrics

**Files:**
- Modify: `src/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Consumes verified `ScoredLabel` records (`request_id`, `candidate_id`, `score`, binary label).
- Produces `rank_candidates(rows) -> list[ScoredLabel]`, `evaluate_food_requests(rows) -> MetricResult`, and `evaluate_ads(rows) -> MetricResult`.
- `MetricResult` contains JSON-safe `metrics`, `eligibility`, and `undefined_metrics`.

- [ ] **Step 1: Write failing Food tests**

Use one request with two positives at ranks 1 and 3; assert exact `Recall@3=1.0`, `Recall@10=1.0`, `HitRate@1=1.0`, `MRR=1.0`, and binary-gain NDCG. Add equal scores received in reverse candidate-ID order and assert ascending candidate ID wins. Add an all-negative request and assert it is excluded with `no_positive`.

- [ ] **Step 2: Write failing Ads tests**

Use known labels/probabilities to assert finite ROC-AUC, non-interpolated AP, epsilon-clipped LogLoss, and Brier. Verify a zero-click request remains in pointwise counts but not ranking counts. For a single-class split, assert AUC/AP are `None`, their reasons are `single_class_labels`, and LogLoss/Brier exist.

- [ ] **Step 3: Verify tests fail**

Run: `python -m pytest tests/evaluation/test_metrics.py -v --basetemp .pytest-tmp/metrics`

Expected: FAIL because metric interfaces do not exist.

- [ ] **Step 4: Implement pure metric functions**

Rank only with `(-score, candidate_id)`. Food uses binary gain, `Recall@K = positives_top_k / positives_all`, DCG discount `1 / log2(rank + 1)`, and reciprocal rank of the first positive. Ads validates labels and probabilities, calls `sklearn.metrics.roc_auc_score`/non-interpolated `average_precision_score` only when two classes exist, clips with `1e-15` for LogLoss, and reports Brier on validated probability scores. Do not return NaN.

- [ ] **Step 5: Verify tests pass**

Run: `python -m pytest tests/evaluation/test_metrics.py -v --basetemp .pytest-tmp/metrics`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat: add deterministic Food and Ads evaluation metrics"
```

### Task 3: Verify source and prediction artifacts before label access

**Files:**
- Create: `src/evaluation/artifacts.py`
- Create: `docs/evaluation_input_v1.md`
- Create: `tests/evaluation/conftest.py`
- Create: `tests/evaluation/test_artifacts.py`

**Interfaces:**
- Produces `load_verified_predictions(task: str, split: Literal["validation", "test"], prediction_dir: Path, opportunities_root: Path) -> VerifiedEvaluationInput`.
- The return contains canonical label-free predictions, evaluator-only labels, `SourceIdentity`, and `ModelProvenance`.
- Task 4 must use this function as its sole source of labels.

- [ ] **Step 1: Write minimal immutable fixture builders and failing tests**

Create tiny Food and Ads opportunity Parquets plus `manifest.json` with actual per-split SHA-256 values. Build helpers for valid Foundation V1 output and valid baseline output. Test a Foundation load succeeds and canonicalizes an offset timestamp to UTC. Add cases for stale manifest/split SHA, label column, duplicate/missing/extra pair, wrong `user_id`, wrong split, naive/different cutoff, and a baseline that supplies a fake Foundation identity.

- [ ] **Step 2: Verify artifact tests fail**

Run: `python -m pytest tests/evaluation/test_artifacts.py -v --basetemp .pytest-tmp/artifacts`

Expected: FAIL because the loader does not exist.

- [ ] **Step 3: Implement `artifacts.py` and the baseline contract document**

Implement strict JSON parsing, SHA-256 verification, timezone-aware UTC canonicalization, and schema validation. Read the source manifest and verify its selected split file before exposing labels. Then require exact per-row equality of request ID, candidate ID, user ID, split, and canonical cutoff instant.

Foundation path requires `predictions.<split>.parquet`, `run_manifest.<split>.json`, and a passed `validation_report.<split>.json`; map V1 run metadata into `ModelProvenance(kind="foundation")`. Baseline path requires the same prediction filename and `evaluation_input_v1.json` with task/split/data/run ID, real source SHA values, correct score semantics, `kind="baseline"`, and `foundation_identity={"applicable": false}`.

Document that schema and sidecar in `docs/evaluation_input_v1.md`; labels are forbidden in the baseline prediction file.

- [ ] **Step 4: Verify artifact tests pass**

Run: `python -m pytest tests/evaluation/test_artifacts.py -v --basetemp .pytest-tmp/artifacts`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/artifacts.py docs/evaluation_input_v1.md tests/evaluation/conftest.py tests/evaluation/test_artifacts.py
git commit -m "feat: verify downstream evaluation input artifacts"
```

### Task 4: Produce self-contained evaluation reports

**Files:**
- Create: `src/evaluation/runner.py`
- Create: `tests/evaluation/test_runner.py`

**Interfaces:**
- Produces `evaluate_prediction_directory(...) -> EvaluationReport`.
- Produces `write_evaluation_report(report: EvaluationReport, output_path: Path, *, overwrite: bool = False) -> Path`.
- Task 5 consumes only persisted reports from this task.

- [ ] **Step 1: Write failing runner tests**

For Food, assert report metric names, policy fields, `source_request_count`, `complete_candidate_request_count`, `ranking_eligible_request_count`, and reasoned exclusions. For Ads, assert every candidate row is counted, plus `ranking_eligible_request_count` and `zero_click_excluded_request_count`. Confirm JSON serialization preserves `null` AUC/AP and `undefined_metrics`.

- [ ] **Step 2: Verify runner tests fail**

Run: `python -m pytest tests/evaluation/test_runner.py -v --basetemp .pytest-tmp/runner`

Expected: FAIL because runner module does not exist.

- [ ] **Step 3: Implement report orchestration**

Load once through Task 3, dispatch metrics by task, and combine policy, source identity, provenance, eligibility, metrics, and undefined reasons. Atomically write `evaluation_report.<task>.<split>.json`; reject a pre-existing path unless `overwrite=True` was passed explicitly.

- [ ] **Step 4: Verify runner tests pass**

Run: `python -m pytest tests/evaluation/test_runner.py -v --basetemp .pytest-tmp/runner`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/runner.py tests/evaluation/test_runner.py
git commit -m "feat: generate versioned downstream evaluation reports"
```

### Task 5: Add strict comparison and command-line interface

**Files:**
- Create: `src/evaluation/compare.py`
- Create: `src/evaluation/__main__.py`
- Create: `tests/evaluation/test_compare.py`
- Create: `tests/evaluation/test_cli.py`

**Interfaces:**
- `compare_reports(reports: Sequence[EvaluationReport]) -> dict[str, object]`.
- CLI commands:

```text
python -m src.evaluation evaluate --task TASK --split SPLIT --prediction-dir PATH --opportunities-root PATH --output PATH
python -m src.evaluation compare --report PATH --report PATH --output PATH
```

- [ ] **Step 1: Write failing comparison/CLI tests**

Create Foundation and baseline reports with same identity but different provenance; assert both run IDs and model kinds appear. Parametrize one mismatch each for evaluator version, metric-policy version, tie-break, timestamp policy, task, split, data version, candidate manifest SHA, and candidate split SHA; each must raise an error naming the field. Add subprocess CLI tests that evaluate fixture predictions and compare two reports.

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/evaluation/test_compare.py tests/evaluation/test_cli.py -v --basetemp .pytest-tmp/cli`

Expected: FAIL because comparison module and CLI do not exist.

- [ ] **Step 3: Implement compare and CLI**

Comparison accepts at least two reports, enforces every listed identity field, and retains provenance plus all metrics by run. `evaluate` prints the report path and exits nonzero on invalid inputs. `compare` atomically writes JSON and fails on incompatible reports.

- [ ] **Step 4: Verify tests pass**

Run: `python -m pytest tests/evaluation/test_compare.py tests/evaluation/test_cli.py -v --basetemp .pytest-tmp/cli`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/compare.py src/evaluation/__main__.py tests/evaluation/test_compare.py tests/evaluation/test_cli.py
git commit -m "feat: add downstream evaluation and comparison CLI"
```

### Task 6: Add the CPU-only Kaggle runbook and run repository verification

**Files:**
- Create: `notebooks/downstream_evaluation_kaggle.ipynb`
- Modify: `README.md`
- Modify: `tests/evaluation/test_cli.py`

**Interfaces:**
- The notebook consumes completed prediction artifacts and one mounted linked corpus.
- It writes only evaluation/comparison JSON artifacts to `/kaggle/working`; it does no training or metric implementation.

- [ ] **Step 1: Add a failing notebook-contract test**

Parse the notebook JSON and assert its code cells: clone/install the repo, locate mounted linked corpus roots, invoke the two committed `python -m src.evaluation evaluate` commands, invoke `compare`, and archive JSON. Assert no cell imports scikit-learn or defines metric formulas.

- [ ] **Step 2: Verify the test fails**

Run: `python -m pytest tests/evaluation/test_cli.py -v --basetemp .pytest-tmp/notebook`

Expected: FAIL because the notebook does not exist.

- [ ] **Step 3: Create the notebook and README section**

Use input-root discovery instead of user-specific Kaggle paths. Keep Food and Ads output directories separate. Add README commands exactly matching Task 5 and state that Foundation artifacts use V1 sidecars while baselines require `evaluation_input_v1.json`.

- [ ] **Step 4: Run targeted and complete verification**

Run:

```bash
python -m pytest tests/evaluation -v --basetemp .pytest-tmp/evaluation
python -m pytest -q --basetemp .pytest-tmp/all
python -m ruff check src/evaluation tests/evaluation
```

Expected: all tests PASS and Ruff reports no violations in new evaluator code/tests.

- [ ] **Step 5: Commit**

```bash
git add notebooks/downstream_evaluation_kaggle.ipynb README.md tests/evaluation/test_cli.py
git commit -m "docs: add Kaggle downstream evaluation runbook"
```

## Self-Review

- **Spec coverage:** Tasks 1–2 implement policy and formulas; Task 3 implements provenance/checksum/candidate/timestamp boundaries; Task 4 reports eligibility and undefined metrics; Task 5 blocks invalid comparisons; Task 6 supplies CPU-only Kaggle usage.
- **Step scan:** Each task follows failing test, focused failure, implementation, passing test, and commit.
- **Type consistency:** Types flow one way: policy/types → metrics/artifacts → runner → compare/CLI.
- **Review Focus:** Timestamp and provenance are tested in Task 3; ties/single-class Ads in Tasks 2/4; comparison compatibility in Task 5.
- **Proportion:** The plan fixes interfaces, values, file paths, tests, and commits while leaving implementation bodies to the executor.

