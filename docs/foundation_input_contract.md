# Foundation Input Contract

## Dataset selection

The canonical foundation pretraining run uses `data/pretrain_data/OmniBehavior_data`, as specified by `data/README_pretrain_data.md`. The Meituan-compatible run uses `data/pretrain_data/meituan_data`, which is a Meituan TRD-derived dataset converted to the same `omnibehavior_pretraining_v2` storage schema.

Each process receives one `dataset_root` and one logical split. The adapter does not concatenate sources or join users across roots. Dataset identity is part of the vocabulary namespace.

## Canonical sample

`CanonicalSample` contains:

- `sample_id`, `dataset_name`, `split`, `user_id`;
- `cutoff_time` and `cutoff_inclusive`;
- `profile` parsed from `profile_json`;
- ordered `EventRecord` values reconstructed from `history_event_ids`;
- `history_order_known`, `history_truncated`, and `history_group_truncated`.
- `history_start`/`history_stop` are checked against the contiguous event-shard slice;
  the requested split and cutoff coordinate are checked against preprocessing split bounds.

An `EventRecord` preserves `event_id`, `source_record_id`, source/action indices, event and eligibility timestamps, timestamp precision, scenario, event type, and parsed context/action dictionaries.

## Semantic tokens

- Profile fields become unordered `source="tabular"` tokens with keys such as `profile.event_count_30d`.
- Event fields become ordered `source="sequence"` tokens with keys such as `event.scenario`, `event.event_type`, `context.product_id`, and `action.click_cnt`.
- Context fields are restricted to the dataset's `context_allowlist`.
- Tokens preserve `event_group_id=source_record_id` for coordinated masking.
- `as_of_time`, event IDs, source indices, eligibility flags, and other storage metadata are not business-value tokens.
- Date-only/uncertain timestamps remain uncertain; no timezone or sub-day precision is invented.

## Vocabulary and numeric values

Vocabulary, numeric boundaries, and normalization metadata are fit only from the selected dataset's logical `train` split through `fit_vocabulary_from_samples`. Validation and test are transform-only. The serialized vocabulary records its fit split, schema version, and optional source fingerprint. OOV values map to `UNK`.

`order_price_interval` and other source buckets remain categorical values. They are never converted into exact monetary values.

## Batch layout

```text
[CLS] [tabular tokens...] [SEP] [event tokens...] [PAD...]
```

Special IDs are fixed: `PAD=0`, `CLS=1`, `SEP=2`, `MASK=3`, `UNK=4`.

`max_history_events=30` is an event-group limit. `max_tokens` is a separate token limit and truncation happens at event-group boundaries where possible.

The collator returns `FoundationBatch` and includes `targets["event_group_ids"]` as provenance metadata. When `DatasetConfig.next_action_prediction=True`, the final retained event group is moved to an explicit target-only view (`next_action_records`), excluded from input, and exposed through `next_action_type_ids` and `next_action_group_ids`. This creates a training objective view; it does not claim that stored data contains labels or masks.
