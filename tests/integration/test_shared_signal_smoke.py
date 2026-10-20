"""A small real-corpus checkpoint-to-Food/Ads integration smoke test."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml

from src.data.green_sm.build import build_dataset
from src.data.green_sm.config import GreenSmConfig
from src.data.green_sm.validate import validate_outputs
from src.downstream.ad_optimization import run_ads_prediction, run_ads_training
from src.downstream.food_recommendation import run_food_prediction, run_food_training
from src.pretraining.config import load_pretraining_config
from src.pretraining.gates import preflight_green_sm
from src.pretraining.runner import run_corpus


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_shared_signal_corpus_through_frozen_food_and_ads(tmp_path: Path) -> None:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        base = GreenSmConfig.from_json("configs/green_sm_shared_signal_poc.json")
        thresholds = replace(
            base.thresholds, min_train_event_rows_per_event_type=1,
            max_train_event_type_share=1.0, min_train_valid_next_action_rate=0.0,
            min_train_maskable_tokens={"categorical": 1, "numeric": 1, "text": 1, "geo": 1},
            min_valid_geo_rate=0.0, max_non_user_oov_rate_validation_test=1.0,
        )
        data_config = replace(
            base, output_root=tmp_path / "data", user_count=96, thresholds=thresholds,
            snapshot_counts={"train": 80, "validation": 40, "test": 40},
            food_request_counts={"train": 40, "validation": 20, "test": 20},
            ads_request_counts={"train": 40, "validation": 20, "test": 20},
        )
        build_dataset(data_config)
        validate_outputs(data_config)
        linked = data_config.output_root / "green_sm_shared_signal_poc_linked_v1"
        control = data_config.output_root / "green_sm_shared_signal_poc_control_paired_v1"
        recipe = load_pretraining_config("configs/pretrain_green_sm_shared_signal_poc.json")
        recipe = replace(
            recipe, linked_root=linked / "foundation", control_root=control / "foundation",
            output_dir=tmp_path / "pretrain", run_id="smoke",
            model=replace(recipe.model, d_model=8, num_heads=2, num_layers=1,
                          num_user_buckets=128, categorical_pair_buckets=257),
            training=replace(recipe.training, batch_size=4, max_steps=2, eval_every_steps=2,
                             checkpoint_every_steps=2, validation_max_batches=1),
        )
        run_corpus(recipe, preflight_green_sm(recipe), corpus="linked", output_dir=recipe.output_dir,
                   device="cpu", max_steps=2)
        checkpoint = recipe.output_dir / recipe.run_id / "linked"
        selected = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
        trace = json.loads((checkpoint / "loss_trace.json").read_text())
        assert trace[-1]["counts"]["next_semantic"] > 0
        foundation_manifest = linked / "foundation" / "manifest.json"
        for task, train, predict in (("food", run_food_training, run_food_prediction),
                                     ("ads", run_ads_training, run_ads_prediction)):
            template = "food_recommendation" if task == "food" else "ad_optimization"
            config = yaml.safe_load(Path(f"configs/{template}_shared_signal_poc.yaml").read_text())
            opportunities = linked / f"{task}_opportunities"
            manifest = opportunities / "manifest.json"
            config["data"].update({
                "opportunities_root": str(opportunities), "manifest_path": str(manifest),
                "foundation_root": str(linked / "foundation"),
                "foundation_manifest_path": str(foundation_manifest),
                f"expected_{task}_manifest_sha256": sha(manifest),
                "expected_foundation_manifest_sha256": sha(foundation_manifest),
            })
            if task == "food":
                config["data"]["vocabulary_path"] = str(checkpoint / "vocabulary.json")
            else:
                config["data"]["quality_report_path"] = str(linked / "reports" / "data_quality_report.json")
            config["checkpoint"].update({"package_root": str(checkpoint),
                                          "expected_selected_checkpoint_sha256": selected["checkpoint_sha256"]})
            run_dir = tmp_path / f"{task}_run"
            config["ranker"]["run_dir"] = str(run_dir)
            config["training"].update({"device": "cpu", "epochs": 1, "batch_size": 20,
                                       "artifact_dir": str(run_dir)})
            train(config)
            config["ranker"]["expected_run_manifest_sha256"] = sha(run_dir / "run_manifest.json")
            paths = predict(config, split="test")
            columns = pq.read_table(paths.predictions).column_names
            assert "score" in columns and "clicked" not in columns and "is_positive" not in columns
    finally:
        torch.set_num_threads(old_threads)
