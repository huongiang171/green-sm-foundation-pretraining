# Green SM 2% data-readiness pilot

This pilot is a **data and contract validation run**, not a model-performance
run. It creates a deterministic linked corpus and a paired-control corpus from
the same logical Food and Ads targets, then verifies causality, split
boundaries, checksums, candidate sets, class rates, cold-start coverage, and
paired-control integrity.

## Preconditions

- Run from the repository root with the project Python environment.
- Keep `data/processed/green_sm_sim_pilot_v1/` absent or empty. The builder
  refuses overwrite so every generated corpus is immutable.
- The repository's Foundation input must merge the v3 geo contract,
  `max_tokens=max_length=256`, and real `GeoAdapter` routing before treating
  this data as trainable. The present pilot report records that as an upstream
  blocker; it does not hide it.

## Build and validate

```powershell
python -m src.data.green_sm --config configs/green_sm_sim_pilot.json
```

The command creates two roots:

- `data/processed/green_sm_sim_pilot_v1/green_sm_sim_linked_v1/`
- `data/processed/green_sm_sim_pilot_v1/green_sm_sim_control_paired_v1/`

Each root has a `foundation/` v2-storage manifest and shards, plus
`food_opportunities/`, `ads_opportunities/`, and `reports/`. The two
opportunity datasets have identical request IDs, users, cutoffs, candidate
sets, and labels. Their manifests intentionally point at different Foundation
manifest checksums.

## Archive checklist

Do not commit generated Parquet data. Archive the whole immutable output root
outside Git (for example as a Kaggle Dataset version) together with:

1. `foundation/manifest.json` and its SHA-256 from each root;
2. each Food/Ads opportunity manifest and SHA-256;
3. `reports/data_quality_report.json` and `reports/causal_audit_report.json`;
4. the JSON configuration and the `generator_config_digest` recorded in the
   report; and
5. the Git commit SHA used to generate it.

If any report gate fails, discard that output version and diagnose the data
contract before retrying with a new empty output root.

## Member 3 handoff

Only after the upstream v3 geo contract is merged and the archived pilot has
`readiness_passed: true`, hand Member 3 the two Foundation roots, opportunity
roots, manifest checksums, config digest, and reports. Member 3 then fits the
vocabulary/scalers from **train only**, runs the planned 500–1,000-step
pretraining smoke run, and records loss/gradient/checkpoint evidence before a
full corpus is generated.
