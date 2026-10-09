---
license: cc-by-nc-sa-4.0
language:
- en
pretty_name: OmniBehavior user pretraining observations
tags:
- user-behavior
- pretraining
- parquet
configs:
- config_name: events
  data_files:
  - split: rows
    path: events/*.parquet
- config_name: snapshots
  data_files:
  - split: rows
    path: snapshots/*.parquet
---

# OmniBehavior user pretraining observations

Derived from [OmniBehavior](https://huggingface.co/datasets/jiawei-ucas/OmniBehavior), source revision
`28479ffdca40b8657018719b0763e56f2981d254`. Source attribution: Jiawei Chen et al.,
[Towards Real-world Human Behavior Simulation](https://arxiv.org/abs/2604.08362),
2026. This derivative retains the source **CC BY-NC-SA 4.0** license; see `LICENSE`.
Use this release for non-commercial research under that license.

This release contains **186 users**, **804,435 raw records**,
**891,565 action observations** and **792,481 snapshots**.
14 source users are missing; IDs are listed in `data_report.json`.
Schema version: `omnibehavior_pretraining_v2`. Source profile prose is excluded.

| Logical split | Snapshots |
| --- | ---: |
| Train, September–October 2025 | 531,852 |
| Validation, November 1–15, 2025 | 131,582 |
| Test, November 16–30, 2025 | 129,047 |

## Storage and reading

`events/` stores one observation per raw action, including original context and
negative states. `snapshots/` stores profiles and references to at most 30 events.
The two Hub configurations expose physical tables using a split named `rows`.
**`rows` includes all logical train/validation/test data; it is not a training split.**
Use the bundled reader to reconstruct input records and select the logical split:

```python
import sys
from huggingface_hub import snapshot_download

dataset_dir = snapshot_download("hongduc05/pretrain_gsm_usr", repo_type="dataset")
sys.path.insert(0, dataset_dir)
from pretraining_dataset import iter_pretraining_samples

for sample in iter_pretraining_samples(dataset_dir, split="train"):
    # Prepare masks/targets and batches in the training pipeline.
    print(sample["sample_id"])
    break
```

The reader requires Python 3.10+ and `pyarrow` (generated with version
`24.0.0`). See `pretraining_data_format.md` for the complete format,
`preprocessing_config.json` for the allowlist and `preview.json` for ten reconstructed samples.
Fit vocabulary, buckets and normalization only from the logical training split.

## Temporal and labeling limitations

- Source timezone and actual log availability are unknown.
- Date-only records remain date-only and enter snapshots only when their day ends.
- Interval metadata preserves uncertain order. Array order does not establish next-action order.
- A history may contain part of a shared timestamp or source-record group.
- Negative action flags remain negative. `action.type` is not a success label.
- The reader filters context using an allowlist; full context is still stored in the events table.
- Related fields and shared context require coordinated reconstruction masking.
- No training masks, next-action labels or validated CVR labels are supplied.

All local shard checksums, index references, cutoffs and profiles were verified
before upload. The uploader verifies remote file sizes and content hashes at the uploaded commit.
