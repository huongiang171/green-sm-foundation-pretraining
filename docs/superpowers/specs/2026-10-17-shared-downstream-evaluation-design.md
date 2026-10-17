# Shared Green SM Downstream Evaluation Design

## Intent

Produce one reproducible, CPU-only evaluator that scores every Food or Ads
prediction artifact against the immutable Green SM **linked-corpus** opportunity
table. Its reports must be directly comparable across Baseline and Foundation
runs. The evaluator is separate from downstream training: models export
label-free scores; only the evaluator reads `is_positive` or `clicked` after
verifying the candidate universe.

This design establishes a versioned metric policy before any metric is
implemented:

- `evaluator_version = "green_sm_evaluator_v1"`
- `metric_policy_version = "green_sm_metric_policy_v1"`
- `tie_break_policy = "score_desc__candidate_id_asc"`
- `timestamp_policy = "parse_to_utc_instant"`

Those four identities are mandatory fields of every evaluation report and are
part of the comparison gate.

## Inputs, provenance, and validation boundary

The evaluator accepts one task (`food_recommendation` or `ad_optimization`),
one split (`validation` or `test`), a prediction directory, and the matching
opportunities root. It verifies the immutable opportunity `manifest.json` and
the selected split Parquet checksum before it reads any labels.

### Foundation artifacts

Existing Foundation Food/Ads outputs remain valid input: the evaluator reads
their `OutputContractV1` files (`predictions.<split>.parquet`,
`run_manifest.<split>.json`, and `validation_report.<split>.json`). It checks
the contract pass flag, task, split, data version, score semantics, finite
scores, Foundation manifest binding, and the exact candidate universe keyed by
`(request_id, candidate_id)`. The producer's prediction Parquet must contain
no label column.

`cutoff_timestamp` is **not** compared as a raw string. Both the source and
prediction timestamp are parsed as timezone-aware values and canonicalized to
the same UTC instant. A naive, unparsable, or unequal instant is a hard error.

### Baseline artifacts

`OutputContractV1` currently requires Foundation checkpoint/vocabulary hashes,
which a non-Foundation baseline legitimately does not have. Baselines must not
invent placeholder hashes. Instead, the evaluator accepts an explicit
`evaluation_input_v1.json` sidecar alongside
`predictions.<split>.parquet`. That Parquet has the same required label-free
schema as an `OutputContractV1` prediction file:
`task`, `data_version`, `run_id`, `split`, `request_id`, `user_id`,
`cutoff_timestamp`, `candidate_id`, and finite `score`; `raw_logit` is
optional. It must not contain `clicked`, `is_positive`, or any label field.
The sidecar declares:

```json
{
  "schema_version": "evaluation_input_v1",
  "model_provenance": {
    "kind": "baseline",
    "baseline_id": "lightgbm_v1",
    "foundation_identity": {"applicable": false}
  },
  "task": "food_recommendation",
  "data_version": "green_sm_sim_full_v1",
  "split": "test",
  "run_id": "baseline-food-full-v1-r1",
  "candidate_manifest_sha256": "...",
  "candidate_split_sha256": "...",
  "score_semantics": "ranking"
}
```

The baseline sidecar's candidate hashes must equal the actual immutable
opportunity manifest and selected split-Parquet hashes. For **both** Foundation
and baseline inputs, every output row is matched one-to-one against the source
candidate universe and must have equal `request_id`, `candidate_id`, `user_id`,
`split`, and canonical UTC cutoff instant. Duplicate, missing, extra, or
mismatched rows are hard errors.

For Foundation artifacts, the evaluator derives the equivalent canonical
provenance from the existing V1 run manifest (`kind = "foundation"`) rather
than requiring models already trained to be retrained or re-exported. This is
an evaluator input adaptation, not a change to the downstream model contract.

In either path, the resulting canonical input identity includes task, data
version, split, score semantics, candidate opportunity-manifest SHA, split
Parquet SHA, and provenance. Baseline and Foundation can therefore be compared
without pretending they use the same type of model.

## Versioned metric policy

All request ranking uses descending score and, for equal scores, ascending
`candidate_id` using Python's Unicode code-point lexicographic order. This same
deterministic order is used for Food and Ads.

The evaluator adds and pins `scikit-learn` in the project dependency lock for
this policy. It uses its documented `roc_auc_score` and non-interpolated
`average_precision_score` behavior, including score ties; the exact installed
version is included in each report as `metric_library_version`. This avoids two
implementations silently applying different AUC/AP tie conventions.

### Food ranking

Food labels are binary: `relevance = 1` when `is_positive == 1`, otherwise
`0`. For a request with candidate labels `r_1, ..., r_n` after applying the
fixed ranking order:

- `Recall@K = sum(r_i for i <= K) / sum(r_i for all i)`. A request with no
  positive is excluded from Food ranking metrics and counted by reason.
- `DCG@K = sum((2^r_i - 1) / log2(i + 1) for i = 1..K)`.
  `NDCG@K = DCG@K / IDCG@K`, where `IDCG@K` is the DCG of the same labels
  sorted by relevance. Thus a multi-positive request receives credit for every
  positive placed in the top K.
- `HitRate@1 = 1` iff the first candidate is positive, otherwise `0`.
- `MRR = 1 / min(i where r_i == 1)`; equivalently, it uses the first ranked
  positive, not an arbitrary positive.

Report `NDCG@5`, `NDCG@10`, `Recall@3`, `Recall@10`, `HitRate@1`, and `MRR`.
The official comparison fields are `NDCG@10`, `Recall@10`, and `MRR`.

Food eligibility counts must distinguish total source requests, requests with
a valid complete candidate set, ranking-eligible requests (at least one
positive), and exclusions grouped by reason (`no_positive`, invalid candidate
group, or contract/input rejection). A contract rejection is a hard failure,
not a silently excluded request.

### Ads CTR and ranking

Ads labels are binary: `clicked` must be `0` or `1`. Across **all candidate
rows**, including requests that have no clicked candidate, report:

- `ROC-AUC` using rank-based binary AUC;
- `PR-AUC` defined as non-interpolated average precision (the standard
  `average_precision_score` definition);
- `LogLoss = mean(-(y*log(p) + (1-y)*log(1-p)))`, with
  `p = clip(score, 1e-15, 1 - 1e-15)`;
- Brier score from the validated probability score in `[0, 1]`.

If a split contains one label class only, ROC-AUC and PR-AUC are reported as
`null` with `undefined_reason = "single_class_labels"`; the evaluator must not
invent a numerical score or crash. LogLoss and Brier remain defined.

For the demo-only ranking view, report `NDCG@3` and `Recall@3` on the subset of
requests with at least one click, using the same binary-gain formulas and
tie-break rule above. Requests without a click remain in every pointwise CTR
metric but are explicitly excluded from the ranking subset. The report lists
all candidate rows, total requests, ranking-eligible requests, and zero-click
excluded requests. These offline metrics are not evidence of causal ad-policy
uplift.

## Outputs and comparison

The evaluator writes a self-contained JSON report. It contains the canonical
input identity, evaluator/metric/tie-break/timestamp policy versions, source
row and request eligibility counts, undefined-metric reasons, and metric
values. It also includes the opportunity manifest SHA as
`candidate_manifest_sha256` and selected split Parquet SHA as
`candidate_split_sha256`. Undefined numeric metrics serialize as JSON `null`;
their reason appears under a stable `undefined_metrics` object keyed by metric
name.

The comparison command combines already-verified reports only when every one
of the following values is identical: `evaluator_version`,
`metric_policy_version`, `tie_break_policy`, `timestamp_policy`, task, split,
data version, `candidate_manifest_sha256`, and `candidate_split_sha256`.
Otherwise it rejects the comparison rather than producing an unfair
Baseline-versus-Foundation table. The comparison output retains each run's
model provenance so the slide makes clear which run is a baseline and which
uses Foundation.

## Operating model

Metric implementation and tests live in `src/evaluation/` and run without a
GPU. A small CPU-only Kaggle notebook mounts the corpus and completed Food/Ads
artifacts, runs the committed evaluator CLI, and archives reports. No metric
logic belongs in either downstream notebook.

## Constraints and non-goals

- Keep `clicked` and `is_positive` out of prediction features and prediction
  Parquet outputs; they are evaluator-only labels.
- Do not change Food/Ads model training, checkpoint, or corpus generation
  logic merely to evaluate existing Foundation artifacts.
- Do not report accuracy for Ads, and do not calculate Ads ranking metrics on
  zero-click requests.
- Use the same evaluator binary and temporal split for each valid comparison.
- The evaluator proves reproducible offline ranking/CTR evaluation only; it
  does not claim real-world Food conversion or Ads uplift.
