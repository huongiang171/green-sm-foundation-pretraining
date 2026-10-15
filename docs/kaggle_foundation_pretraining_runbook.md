# Green SM Foundation Pretraining on Kaggle

Upload the repository and the selected immutable Green SM corpus to Kaggle. Enable a GPU, then run:

```bash
python -m src.pretraining --config configs/pretrain_green_sm_pilot.json --device cuda --output-dir /kaggle/working/green-sm-pilot
```

Run this only after `FoundationBatch` v3, geo tokenization, and `GeoAdapter` routing have merged. The current checkout deliberately fails preflight instead of training without geo.

The pilot runs 1,000 steps per corpus with `d_model=128`, four Transformer layers, four heads, and batch size 32. It writes each corpus's own train-fitted `vocabulary.json`, `loss_trace.json`, `last.pt`, and `checkpoint_manifest.json` under the chosen output directory.

Accept the pilot only when both linked and control runs have finite losses/gradients, non-zero `UserIdAdapter` gradient, a lower trailing mean loss than initial mean loss, valid geo-loss counts, and complete manifests. Use the linked checkpoint only for Food and Ads downstream work; control is the ablation.

Do not start the full-corpus recipe until the pilot artifact review is complete.
