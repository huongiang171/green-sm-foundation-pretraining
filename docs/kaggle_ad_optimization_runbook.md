# Green SM Ads Optimization on Kaggle

Attach only these private, read-only Kaggle datasets:

- `green-sm-sim-full`, containing `green_sm_sim_full_v1`;
- `green-sm-full-v1-checkpoint`, containing the published selected Foundation checkpoint package.

Open `notebooks/ad_optimization_kaggle.ipynb`, enable a GPU, then use **Save & Run All**.
The notebook clones `feat/foundation-encoder`, computes all input checksums at runtime, and refuses a missing or mismatched selected checkpoint, vocabulary, manifest, or full-corpus quality report.

The run creates `/kaggle/working/artifacts/ad_optimization/ads-full-v1-r1/` with:

- `best.pt` and `last.pt` for the Ads ranker;
- `feature_artifacts.json`, `train_metrics.jsonl`, and `run_manifest.json`;
- label-free `predictions/validation/` and `predictions/test/` OutputContract artifacts.

Download `green_sm_ads_full_v1_r1.zip` from Kaggle Output. Give the Baseline team the two prediction directories, the Ads run manifest, and the Foundation checkpoint selection manifest. Do not attach or use Taobao data for this workflow.
