# Shared-signal Foundation PoC runbook

This is a controlled synthetic proof of signal transfer between Food and Ads.
It does **not** establish a production uplift.  The generated labels and the
causal history policy are deliberately coupled so that the experiment can test
whether the Foundation pipeline preserves a known, non-leaking cross-service
signal.

## 1. Build, validate, and package the public corpus

Run from the repository root.  The build refuses a non-empty output root, so
use a new root or delete only a disposable previous PoC output deliberately.

```powershell
python -m src.data.green_sm --config configs/green_sm_shared_signal_poc.json
python scripts/package_shared_signal_poc.py pack `
  data/processed/green_sm_shared_signal_poc_v1 `
  artifacts/green_sm_shared_signal_poc_v1.tar.gz.b64
```

The payload contains only the paired linked/control public roots.  It excludes
the generator's `private_audit` state; do not attach an unpacked corpus with
thousands of files to Kaggle.

## 2. Kaggle execution order

1. Upload `green_sm_shared_signal_poc_v1.tar.gz.b64` as one private Kaggle
   dataset.
2. Import `notebooks/shared_signal_poc_pretrain_kaggle.ipynb`, attach that
   payload, enable one GPU, and run it.  Save only its
   `green_sm_shared_signal_poc_checkpoint.tar.gz` output as the checkpoint
   dataset.
3. Import the Food and Ads notebooks separately.  Attach exactly the corpus
   payload and the selected-checkpoint package.  Each notebook derives runtime
   paths and SHA-256 values from the mounted manifests and emits a compressed
   run artifact containing the ranker, manifest, and label-free predictions.
4. Evaluate validation predictions while selecting a checkpoint/epoch.  Run
   test once, only after that selection is fixed.  Do not use test metrics to
   tune training, candidate features, epoch count, or the Foundation objective.

The downstream notebooks first validate manifest and checkpoint lineage.  A
failure there means the input assets are mixed or stale; do not bypass that
check by replacing a SHA in the YAML manually.

## 3. Required comparison

Use the same public opportunities, candidate universe, causal cutoff, split,
ranker hyperparameters, seeds where applicable, and evaluator policy for all
conditions.

| Condition | Foundation encoder given to Food/Ads | Purpose |
|---|---|---|
| **pretrained mixed** | Selected frozen checkpoint pretrained on the linked mixed Food+Ads history with the next-semantic objective | Measures the complete proposed system. |
| **random frozen** | The same architecture and seed family, but weights are never optimized; freeze it before ranker training | Separates the value of learned Foundation representations from ranker capacity. |
| **cross-service ablation** | Frozen checkpoint pretrained after removing cross-service episode tokens, while retaining same-service/noise events | Tests whether cross-service context, rather than merely more history tokens, causes the gain. |

The current notebooks execute the **pretrained mixed** condition.  The random
frozen and cross-service ablation checkpoints must be produced as separate,
checksum-bound runs before comparison; never fake either condition by changing
downstream scores or reusing the mixed checkpoint.  Report Food ranking and
Ads CTR metrics for `all`, `warm`, and `cold` request groups, with confidence
intervals or repeated seeds where the time budget permits.

## 4. What to inspect before interpreting a result

- Data quality report: exact 70/20/10 episode mix, exact 15% cold validation
  and test requests, 12 Food categories, no item/campaign ID leakage in history.
- Pretraining run manifest: validation total, `validation_next_action`, and
  `validation_next_semantic`; retain the selected checkpoint manifest.
- Downstream run manifest: selected Foundation checkpoint SHA and both
  manifest SHAs must match the evaluated prediction metadata.
- Evaluation report: candidate-universe checksum and eligibility counts must
  match across conditions.  A metric delta without this identity match is not
  comparable.

## 5. CPU smoke scope

The test suite builds a reduced PoC corpus and exercises the immutable
contracts.  It is a regression/smoke check, not a quality experiment.  Use a
GPU only for the Kaggle training runs; CPU smoke results must not be presented
as final ranking or CTR metrics.
