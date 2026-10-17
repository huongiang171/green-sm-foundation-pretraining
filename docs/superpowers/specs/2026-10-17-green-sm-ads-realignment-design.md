# Green SM Ads Downstream Realignment Design

## Intent

Replace the abandoned Taobao-based Ads route with one executable Ads
downstream task that consumes the simulated Green SM full corpus.  The task
must show whether a frozen, linked-corpus Green SM Foundation checkpoint adds
useful user representations to candidate click-through-rate (CTR) scoring.

## Scope

The only source dataset for Ads is:

`green_sm_sim_full_v1/green_sm_sim_linked_v1`.

The task reads its immutable `ads_opportunities/` and `foundation/` roots,
then resolves the checksum-verified Foundation checkpoint generated from that
same linked foundation root.  `checkpoint_manifest.json`, not a hard-coded
`best.pt` path, is the source of truth: it supplies `selected_kind`,
`checkpoint`, checkpoint SHA-256, checkpoint identity, and vocabulary SHA.
The checkpoint handoff package must also contain its matching
`vocabulary.json`; Ads verifies its SHA-256 against the selected manifest
before it encodes any snapshot.  The Foundation embedder is always frozen.
Only the Ads ranker is trainable.

Taobao is out of scope for the executable Ads pipeline.  The uncommitted
Taobao ETL, its tests, configs, runbooks, data-contract documents, historical
Taobao ETL plan, and temporary pytest directories are removed rather than
adapted.  The former Taobao config and downstream code are replaced so they
cannot accidentally be selected on Kaggle.  The already tracked
`data/taobao_data/*.csv` and `data/README_taobao.md` are deliberately outside
this deletion scope until their owner explicitly approves their removal;
neither may be read by the new Ads runtime.

Migration is staged rather than a blind deletion.  First, the canonical
`src/downstream/ad_optimization.py` is rebuilt around native Green SM field
names and a `green-sm-ads-v1` hash namespace.  Its unit/integration tests must
pass while the legacy bridge still exists.  Only then is
`src/downstream/green_sm_ad_optimization.py` deleted, eliminating its current
dependency on Taobao-shaped types and its incorrect broad `no_grad` wrapper.

## Data and learning contract

Each Green SM Ads candidate row supplies `request_id`, `user_id`,
`foundation_sample_id`, cutoff and availability timestamps, `candidate_id`,
`campaign_id`, `candidate_category`, `placement`, `candidate_geo_cell`,
candidate latitude/longitude, split, and `clicked`.

The runner validates the `green_sm_ads_opportunities_v1` manifest, all split
Parquet checksums, the linked Foundation-manifest checksum, candidate-set
integrity, request-level consistency, and snapshot identity before model
construction.  A full-data run is rejected at `validate`, `train`, and
`predict` unless the data-quality report passes, the configured data version
is `green_sm_sim_full_v1`, the selected checkpoint manifest exists, its
`corpus` is `linked`, and its linked-manifest/vocabulary identities equal the
actual handoff files.  Placeholder SHA values are never accepted by any mode.
Train, validation, and test are the already published time-based splits; no
rows are resampled or moved between them.

The ranker predicts a CTR probability independently for every real candidate
in a request.  It trains binary cross-entropy over every non-padding candidate
in the train split, including requests with no click.  Best checkpoint
selection uses validation BCE.  The runner exports every validation/test
candidate without its `clicked` label, allowing the Baseline team to compute
its own probability and ranking metrics on its approved eligible subset.

## Model

For each request snapshot, the frozen Foundation provides `long_term` and
`short_term`, each with the Foundation `d_model` width.  The Ads ranker fuses
them with candidate/campaign hash embeddings plus train-fitted categorical
embeddings for category, placement, and geo cell.  Unknown evaluation
categories map to `UNK`; candidate/campaign IDs retain deterministic hash
features.  The candidate latitude/longitude columns remain contract-validated
but are not separate ranker features in v1: `candidate_geo_cell` gives a
stable location representation without adding a duplicate coordinate MLP.

The Foundation embedding call alone is inside `torch.no_grad()`.  The ranker
forward, BCE loss, and backward pass remain outside it; every Foundation
parameter must have no gradient while at least one ranker parameter must have
a gradient in the regression test.

## Artifacts and public handoff

Every training run uses a new, empty
`artifact_dir/ad_optimization/<run_id>/` directory.  It contains train-fitted
feature artifacts, `last.pt`, `best.pt`, JSONL epoch metrics, and a run
manifest with SHA-256 values and all provenance needed to reload it.  The
manifest binds the Foundation checkpoint SHA, vocabulary SHA, Ads manifest
SHA, Foundation manifest SHA, config digest, seed, model settings, and
request counts.

Prediction reloads and verifies both the Foundation and Ads manifests before
scoring.  It verifies that the chosen Ads ranker was trained from the same
Foundation checkpoint and immutable Ads/Foundation handoff.  Prediction files
live in distinct `predictions/validation/` and `predictions/test/` directories
and are written exclusively via `OutputContractV1` with probability scores in
`[0, 1]`.  The test-inference path must not read `clicked` when constructing
feature tensors, expected candidates, prediction rows, or Parquet output; a
regression test changes test labels while requiring identical scores and a
label-free export schema.

## CLI and Kaggle usage

`python -m src.downstream.ad_optimization --config <yaml> --mode validate`
checks only data/checkpoint bindings.  `--mode train` trains the ranker; and
`--mode predict --split validation|test` reloads an existing ranker run and
exports a label-free prediction file.  The YAML has full-corpus Kaggle paths
as placeholders until the full Foundation `best.pt` and its manifest are
published; no placeholder checkpoint may start training.

The Kaggle notebook/runbook attaches only the full Green SM corpus and the
published Foundation checkpoint dataset.  It never attaches raw Taobao files
or creates outputs inside `/kaggle/input`.

## Migration and verification order

1. Add Green-SM-native regression tests for manifest/checksum binding, causal
   snapshot resolution, no-click requests, selected-checkpoint resolution,
   frozen-Foundation gradients, and label-free test export.
2. Move the reusable model, batching, artifact, CLI, and output logic into
   canonical `src/downstream/ad_optimization.py` using native field names;
   switch to the Green SM full config and Kaggle runbook.
3. Run focused Green SM tests, then remove the legacy bridge and every
   uncommitted Taobao ETL runtime file in the declared scope.
4. Run a repository reference audit that permits only the intentionally
   retained raw-data README/CSV paths and historical design record, followed
   by the focused test suite and Ruff.
5. Only a clean migration may proceed to full-corpus `validate`, Kaggle
   train, and validation/test prediction export.

## Non-goals

- No Taobao data ingestion, conversion, or joint Taobao/Green-SM pretraining.
- No Foundation fine-tuning during Ads training.
- No labels in validation/test prediction exports.
- No duplicate implementation of the Baseline team evaluator or metrics.
- No claim that a simulator result demonstrates production CTR uplift.
