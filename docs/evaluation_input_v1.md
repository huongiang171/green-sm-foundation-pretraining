# Evaluation Input V1 for Baselines

Non-Foundation baselines export label-free `predictions.<split>.parquet` plus
`evaluation_input_v1.json` in the same directory. Prediction rows require
`task`, `data_version`, `run_id`, `split`, `request_id`, `user_id`,
`cutoff_timestamp`, `candidate_id`, and finite `score`; `raw_logit` is optional.
They must not contain `clicked`, `is_positive`, `label`, features, or embeddings.

The sidecar declares the task, split, run ID, score semantics, and SHA-256 of
the immutable opportunity `manifest.json` and its selected split Parquet. Its
provenance must be `{"kind":"baseline", ..., "foundation_identity":{"applicable":false}}`.
Do not create placeholder Foundation hashes. Food uses `ranking` scores; Ads
uses probability scores in `[0, 1]`.
