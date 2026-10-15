# Green SM Simulated Corpus Design

## Intent

Build a reproducible synthetic Green SM corpus that can both run the existing
foundation-model pipeline and test its central claim: a user representation
pretrained on a mixed Ride, Food, Delivery, and Ads timeline should improve
Food Recommendation and Ad Optimization when cross-service behavior contains
real predictive information.

The corpus follows the architecture described in Grab's *User foundation
models*: unordered tabular profile plus ordered, multimodal event history;
typed modality adapters; a shared Transformer; long-term user-ID and
short-term sequence embeddings; masked-field reconstruction and next-action
pretraining objectives. It is a controlled simulation for engineering and
hypothesis testing, not evidence of production performance.

## Decisions already approved

- Version one uses one synthetic city with 64 geo-cells, not multiple cities.
- Location is an MVP modality, not a later enhancement.
- A location is represented by both a continuous coordinate and a shared
  categorical geo-cell; all services reference the same location catalogue.
- The simulator provides a linked corpus and a paired control corpus. Both
  expose the exact same logical downstream requests, candidates, labels, user
  IDs, and cutoffs; only their cross-service histories differ.
  A single-service-history condition is an input ablation over the linked
  corpus, not a third copied dataset.
- Food and Ads use the same checkpoint pretrained on the linked corpus.

## Goals

1. Produce a Foundation dataset accepted by the repository's
   `omnibehavior_pretraining_v2` loader after the geo-contract extension.
2. Produce temporal Food and Ads candidate/request artifacts that refer to
   exact, causal Foundation snapshots by `foundation_sample_id`.
3. Make location a meaningful cross-service signal without leaking labels or
   future outcomes into a prediction input.
4. Make the claim falsifiable with matched control and ablation experiments.
5. Keep the full run practical on Kaggle; local machines are for validation
   and smoke runs only.

## Non-goals

- Mimicking real Green SM, Grab, merchant, driver, or customer records.
- Raw personal locations, maps, external geocoders, real addresses, images,
  LLM text encoders, Ray, or distributed training.
- Claiming the simulator proves real-world generalisation or patent novelty.
- Creating a location graph, road network, or separate city-generalisation
  benchmark in v1.

## Corpus scale and temporal protocol

`green_sm_sim_linked_v1` and `green_sm_sim_control_paired_v1` have the same
public schema, catalogue, user IDs, request cutoffs, split boundaries, Food/Ads
candidate sets, and labels. They may differ only in the generated
cross-service-history observations needed to form the linked or control view.

- 12,000 users; one city with 8 zones x 8 geo-cells.
- 42 consecutive simulated days in timezone `Asia/Ho_Chi_Minh`.
- Train: days 1-28; validation: days 29-35; test: days 36-42.
- Target scale: about 55 event groups per user, about 660,000 event-table
  rows total.
- Foundation pretraining snapshots are not limited to downstream decision
  cutoffs: 96,000 train, 24,000 validation, and 24,000 test snapshots. This
  supplies about 8 / 2 / 2 distinct temporal views per user. Downstream
  requests reference a causal subset of these snapshots.
- Food requests: 18,000 / 3,000 / 3,000 across train / validation / test,
  20 candidates per request (about 480,000 candidate rows).
- Ads requests: 24,000 / 4,000 / 4,000 across train / validation / test,
  10 exposed candidates per request (about 320,000 candidate rows).

Those are targets, not permission to silently emit fewer rows. The generator
must either meet all counts or fail with a report explaining the shortfall.

### Quantitative readiness gates

The implementation stores these as configuration, never as hidden constants.
For a full run the generator requires at least 5,000 train event rows per
event type; no event type may exceed 45% of train rows; and at least 75% of
train snapshots must have a valid next-action target. The train minimum counts
for maskable categorical, numeric, text, and geo fields are respectively
100,000, 20,000, 10,000, and 50,000. Pilot counts scale by 2%, with a floor
of 100 per modality/event type.

At least 60% of emitted geo tokens must be valid, and closed-catalogue
validation/test OOV rate may not exceed 1%. Exactly 15% of users are a cold
cohort: their first active day is at least day 29 and they have no train
event or snapshot. Food requests require at least ten hard negatives; Food
positive rate must be 5–20%; Ads click rate must be 2–15%; and at least 30%
of Ads requests must have a click for ranking evaluation. The paired-control
audit allows at most 0.01 AUC gain from other-service control features on a
fixed validation probe.

## Synthetic world

### Fixed catalogue, available before day 1

The city has 64 cells labelled `cell_00` through `cell_63`, each assigned one
of `residential`, `office`, `mall`, `airport`, `university`, `nightlife`,
`hospital`, or `transit` zone types. The catalogue contains 160 Food
merchants, 480 Food items, 120 Delivery pickup/drop-off POIs, and 120 Ad
campaigns. Every catalogue entity has an ID, category, owning geo-cell,
coordinate inside that cell, and `available_at <= day_1_start`.

The catalogue is public model input where applicable. It must not contain a
user's future conversion, click, label, hidden preference, or split-specific
availability value.

### Latent user state, never model input

Each user receives a deterministic seed-derived latent state:

- home and work cells; commute regularity and active-hour pattern;
- Food cuisine and price preferences;
- spending/budget tendency; delivery convenience preference;
- promotion sensitivity and ad-affinity; and
- service propensity for Ride, Food, Delivery, and Ads.

The generator may use that state to create behaviour, but writes it only to
`private_audit/latent_truth.parquet`, excluded from vocabulary fitting,
Foundation input, candidates, downstream features, Git, and model mounts.

At least 70% of users must be active in at least three services, at least 30%
in all four, and at least 20% must have sparse Food/Ads history but meaningful
Ride or Delivery activity. The validation report must publish these coverage
rates.

### Linked behavior mechanism

The linked corpus creates cross-service signal through causal, pre-cutoff
state only. A Ride drop-off or Delivery destination in a mall/office/nightlife
cell increases the near-future likelihood of Food browsing or ordering in the
same or neighbouring cell. Cuisine, price, time of day, and user taste choose
the Food item. Ad exposure/click likelihood depends on candidate category,
placement, current cell, latent ad preference, and recent behaviour such as a
Food search or Ride destination. The later Food choice/click is never written
into the earlier history.

The paired control is not a second downstream task. The generator first
creates one target world containing the user, request cutoff, candidate set,
and Food/Ads label. It then emits two Foundation-history views for that exact
target world:

- the linked view retains the pre-cutoff cross-service observations generated
  from the episode state; and
- the control view retains the profile and target-service history but resamples
  the other-service observations within declared time/cell/user-activity
  strata so they do not reveal the target episode.

Thus Food/Ads labels and candidate sets are byte-for-byte equivalent after
excluding the Foundation-manifest linkage field. In the control view, a target
must be conditionally independent of other-service history given the declared
profile, target-service history, time, request cell, and candidate features.
The causal audit report must test this property with a documented
permutation/conditional-prediction check.

## Foundation storage and semantic input

The on-disk Foundation schema remains `omnibehavior_pretraining_v2` so the
existing manifest, events, and snapshots loader remains the storage base. The
new `preprocessing_config.json` records `foundation_batch_contract_version: 3`
and the geo-field registry. Each user has one complete chronological events
shard and one snapshots shard:

```text
<root>/
  manifest.json
  preprocessing_config.json
  events/<stable-shard-id>.parquet
  snapshots/<stable-shard-id>.parquet
  reports/data_quality_report.json
  reports/causal_audit_report.json
```

Each event retains all v2 required columns. Its `context_json` contains only
allowlisted business fields and its `action_json` contains only event outcomes
known at that event's timestamp. An event group is one raw action observation;
it is never split during snapshot construction. The simulator and Foundation
run use `max_tokens=max_length=256`; `configs/pretrain.yaml` must be changed
from 128 before the final training run. A generated event must create at most
seven semantic tokens, profile tokens are capped at 20, and the generator must
assert the exact resulting sample length is at most 256.

Required scenarios and event types are:

- `Ride`: `ride_search`, `ride_book`, `ride_complete`;
- `Food`: `food_search`, `merchant_view`, `item_view`, `food_order`;
- `Delivery`: `delivery_quote`, `delivery_book`, `delivery_complete`;
- `Ads`: `ad_impression`, `ad_click`.

Every event includes `event.scenario`, `event.event_type`, service-appropriate
entity/category fields, and only applicable numeric values such as fare,
distance, delivery fee, item price, or bid. Profile includes only as-of
aggregates and stable attributes, including activity counts, preferred service,
budget bucket, and optional `home_location` / `work_location` geo fields.

## Location contract required before generation

Location is not encoded as two independent numeric tokens. Each location is
one semantic geo token whose key identifies the role:

- `profile.home_location`, `profile.work_location`;
- `context.pickup_location`, `context.dropoff_location`;
- `context.merchant_location`, `context.delivery_location`; and
- `context.ad_context_location`.

The raw JSON value has exactly `lat`, `lon`, and `geo_cell`. The tokenizer
marks these registered keys as modality `geo`, uses `geo_cell` as the
vocabulary value ID, and carries the coordinate pair separately. For a missing
geo value it emits an explicit geo token with `UNK` cell ID, finite placeholder
coordinates `[0.0, 0.0]`, and a true geo-missing flag. It must reject malformed
objects, non-finite coordinates, an unknown role key, or a cell/coordinate
combination outside the fixed city catalogue.

`FoundationBatch` v3 adds aligned tensors:

```python
geo_values: torch.FloatTensor       # [B, L, 2], train-normalized lat/lon
geo_missing_mask: torch.BoolTensor  # [B, L], True only for a geo token with no location
```

Non-geo special, tabular, sequence, and padding positions contain
`[0.0, 0.0]` and `False`. The geo adapter combines an MLP over valid normalized
coordinates with the normal value/pair embeddings for `geo_cell` and the
existing key embedding for role. It projects the result through the same
modality alignment mechanism to `d_model`. This keeps geo-cell, coordinate,
and location role available without adding a redundant `geo_role_ids` tensor.

All numeric and coordinate normalisation statistics, token allowlists, geo
registry, catalogue fingerprint, and vocabulary IDs are fit from/derived on
the Foundation train split only and persisted with the vocabulary/checkpoint.

## Downstream artifacts

Food and Ads each have independent immutable roots, linked by checksum to the
corresponding linked or paired-control Foundation manifest.

```text
green_sm_sim_linked_v1/
  foundation/
  food_opportunities/
    manifest.json
    food_opportunities.{train,validation,test}.parquet
  ads_opportunities/
    manifest.json
    ad_opportunities.{train,validation,test}.parquet
```

Each logical downstream target is emitted into a linked root and a paired
control root. Corresponding rows have the same request/candidate/label values
and Foundation sample IDs; their manifests point to their respective
Foundation roots and therefore have different Foundation-manifest checksums.
The data-quality report compares the two roots after removing this linkage
field and fails if any logical target differs.

Each Food candidate row contains `request_id`, `user_id`,
`foundation_sample_id`, `cutoff_timestamp`, `candidate_id`, `is_positive`,
`split`, `candidate_available_timestamp`, item/merchant category, price, and
candidate location. Each request has exactly 20 unique candidates and at
least one positive, including documented hard negatives matched by category,
price band, and/or location. Each Ads candidate row contains the analogous request,
user, snapshot, cutoff, candidate/ad/campaign/placement fields, availability
time, ad location, and binary `clicked`; each request has exactly 10 unique
exposed candidates and may have zero clicks. The report must publish the
fraction of Ads requests with at least one click and fail if it is below the
declared ranking-eligible minimum. For both tasks it must report label rate
and label entropy by split so the simulator cannot silently produce a trivial
target.

The Data team hands labels to the baseline/evaluator; neither downstream
inference nor its public prediction file includes labels. Every request must
join one snapshot for the same user and exact cutoff. Candidate availability,
profile as-of time, and every history event are at or before cutoff. An Ads
impression is the decision at cutoff; its click label is strictly after cutoff.
Neither the impression nor click of that request may be in that request's
Foundation history, although either may appear in later snapshots once
causally eligible.

## Pretraining and evaluation protocol

The same Foundation architecture is pretrained separately on the linked and
control Foundation roots, each with its own train-fitted vocabulary and
checkpoint. The runs use identical model dimensions, tokenizer policy,
optimizer, batch size, random-seed family, total optimizer steps, and
evaluation schedule. It uses masked field reconstruction plus next-action
key/value prediction. Geo reconstruction applies a coordinate regression loss
only at masked, valid geo positions; the cell remains a categorical
reconstruction target. Invalid next-action rows remain available to other
losses but are excluded through `next_action_valid_mask`.

The pretraining implementation must define a user-level representation
`z_user = Fuse(long_term, short_term)` and feed it into at least the
next-action prediction head. This makes the User ID Adapter part of the
computational graph of a supervised pretraining loss. A regression test must
verify non-zero gradient for `UserIdAdapter.embedding.weight` on a valid
batch. Masked-token heads may continue to consume token-level hidden states.

The primary downstream experiment uses the linked checkpoint, frozen, with
both `long_term` and `short_term` embeddings. The required comparison matrix
uses identical temporal validation/test requests and candidate sets:

1. baseline without foundation embedding;
2. linked checkpoint with full mixed history;
3. linked checkpoint with only target-service history at inference;
4. paired-control checkpoint with full mixed history; and
5. linked full-history result reported separately for sparse Food/Ads users.

The Baseline/Evaluation team owns metrics and the evaluator. Data and
Foundation teams may validate row integrity but must not implement a private
replacement evaluator.

## Acceptance gates

Generation fails rather than silently repairing data when any gate fails:

- required event/snapshot/candidate schema, checksum, manifest linkage, or
  declared row count differs;
- paired linked/control logical request, candidate, or label values differ
  after excluding their Foundation-manifest linkage field;
- a snapshot history is not a contiguous chronological event-shard slice;
- an event, profile, candidate availability, or exposure is after cutoff;
- a next-action target appears in its own input history;
- an event group is incomplete or too large for the documented token budget;
- coordinates are invalid, outside the city bounds, or inconsistent with their
  declared cell; or
- training-only vocabularies/scalers/catalogue availability use validation or
  test data.

The final reports must also prove the declared paired-control conditional
independence audit, Food/Ads label prevalence and entropy, hard-negative mix,
ranking-eligible Ads request rate, and equality of linked/control training
budgets.

The final reports must include data volumes, split counts, token-group-size
distribution, geo missing rate, service coverage, candidate-set integrity,
cross-service transition rates for linked versus control, and the exact seed,
generator revision, and all artifact checksums.

## Ownership and order

This is two coupled projects, not one PR:

1. Foundation input/model work: implement the versioned geo contract,
   tokenizer/collator propagation, LocationAdapter routing, and its tests.
   This is owned jointly by Members 1 and 2.
2. Data work: implement deterministic catalogue, one target world, paired
   linked/control behavior views, v2 Foundation files, Food/Ads opportunities,
   manifests, and validation.
   This is owned by the Data team.

The geo contract and the 256-token training configuration must merge before
the Data team declares generated artifacts trainable. The simulator can be
developed in parallel against JSON fixtures and the published contract; final
data validation and real training wait for the merged contract. Member 3 owns
the pretraining objectives, `Fuse(long_term, short_term)` route, gradient
regression test, and checkpoint manifest; these must merge before pretraining.
Downstream training is subsequent Member 3/4 work, not part of the simulator
implementation PR.
