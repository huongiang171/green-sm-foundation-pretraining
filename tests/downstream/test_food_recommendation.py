"""Contract tests for the Green SM Food-ranking data boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.foundation_input.types import EncodedSample


def _food_opportunity(**overrides: object):
    from src.downstream.food_recommendation import FoodOpportunity

    values: dict[str, object] = {
        "request_id": "request-1",
        "user_id": "user-1",
        "foundation_sample_id": "foundation-1",
        "cutoff_timestamp": 100,
        "candidate_id": "candidate-1",
        "split": "train",
        "candidate_available_timestamp": 100,
        "item_id": "candidate-1",
        "merchant_id": "merchant-1",
        "category": "noodles",
        "geo_cell": "cell-01",
        "latitude": 10.0,
        "longitude": 106.0,
        "item_price": 50_000.0,
        "is_positive": 1,
        "is_hard_negative": False,
    }
    values.update(overrides)
    return FoodOpportunity(**values)


def _train_rows():
    return [
        _food_opportunity(candidate_id="candidate-1", item_id="candidate-1", is_positive=1),
        _food_opportunity(candidate_id="candidate-2", item_id="candidate-2", is_positive=0),
    ]


def _encoded_samples_by_id() -> dict[str, EncodedSample]:
    return {
        sample_id: EncodedSample(
            sample_id=sample_id,
            dataset_name="green_sm",
            user_id="user-1",
            records=(),
        )
        for sample_id in ("foundation-1", "foundation-2")
    }


def _variable_length_requests():
    return [
        _food_opportunity(request_id="request-1", candidate_id="candidate-1", item_id="candidate-1", is_positive=1),
        _food_opportunity(request_id="request-1", candidate_id="candidate-2", item_id="candidate-2", is_positive=0),
        _food_opportunity(
            request_id="request-2",
            foundation_sample_id="foundation-2",
            candidate_id="candidate-3",
            item_id="candidate-3",
            is_positive=1,
        ),
        _food_opportunity(
            request_id="request-2",
            foundation_sample_id="foundation-2",
            candidate_id="candidate-4",
            item_id="candidate-4",
            is_positive=0,
        ),
        _food_opportunity(
            request_id="request-2",
            foundation_sample_id="foundation-2",
            candidate_id="candidate-5",
            item_id="candidate-5",
            is_positive=0,
        ),
    ]


def test_food_loader_maps_the_real_green_sm_parquet_schema_to_typed_opportunity(
    tmp_path: Path,
) -> None:
    from src.downstream.food_recommendation import load_green_sm_food_opportunities

    cutoff = datetime(2026, 9, 1, 11, 50, 10, tzinfo=timezone.utc)
    available = datetime(2026, 9, 1, 11, 49, 0, tzinfo=timezone.utc)
    row = {
        "request_id": "food:train:000000",
        "user_id": "u000187",
        "foundation_sample_id": "sample-187",
        "cutoff_timestamp": cutoff,
        "candidate_id": "food_item_411",
        "split": "train",
        "candidate_available_timestamp": available,
        "candidate_category": "noodles",
        "candidate_geo_cell": "hcm-17",
        "candidate_lat": 10.77,
        "candidate_lon": 106.69,
        "is_positive": 1,
        "is_hard_negative": False,
        "item_price": 85000,
        "merchant_id": "merchant_091",
    }
    negative = {
        **row,
        "candidate_id": "food_item_412",
        "is_positive": 0,
        "is_hard_negative": True,
    }
    pq.write_table(
        pa.Table.from_pylist([row, negative]),
        tmp_path / "food_opportunities.train.parquet",
    )

    loaded = load_green_sm_food_opportunities(tmp_path, split="train")

    assert loaded[0].item_id == "food_item_411"
    assert loaded[0].merchant_id == "merchant_091"
    assert loaded[0].cutoff_timestamp == int(cutoff.timestamp())
    assert loaded[0].candidate_available_timestamp == int(available.timestamp())


def test_food_validation_rejects_future_candidate_and_duplicate_candidate() -> None:
    from src.downstream.food_recommendation import validate_food_opportunities

    with pytest.raises(ValueError, match="candidate_available_timestamp.*cutoff_timestamp"):
        validate_food_opportunities(
            [_food_opportunity(candidate_available_timestamp=101)],
            expected_split="train",
            require_labels=True,
        )
    with pytest.raises(ValueError, match="duplicate candidate.*request-1"):
        validate_food_opportunities(
            [
                _food_opportunity(),
                _food_opportunity(candidate_id="candidate-1", item_id="candidate-1", is_positive=0),
            ],
            expected_split="train",
            require_labels=True,
        )


def test_candidate_features_are_fit_only_on_train_and_keep_eval_oov_finite() -> None:
    from src.downstream.food_recommendation import FoodFeatureArtifacts, build_food_batches

    artifacts = FoodFeatureArtifacts.fit(_train_rows())
    evaluation_row = _food_opportunity(
        split="validation",
        category="new-category",
        geo_cell="new-cell",
        candidate_id="candidate-3",
        item_id="candidate-3",
        item_price=99_000.0,
    )

    batch = build_food_batches(
        [
            evaluation_row,
            _food_opportunity(
                split="validation",
                candidate_id="candidate-4",
                item_id="candidate-4",
                is_positive=0,
            ),
        ],
        encoded_samples_by_id=_encoded_samples_by_id(),
        feature_artifacts=artifacts,
        batch_size=1,
        item_hash_buckets=17,
        merchant_hash_buckets=13,
        require_labels=True,
    )[0]

    assert batch.category_ids.tolist() == [[1, 2]]
    assert batch.geo_cell_ids.tolist() == [[1, 2]]
    assert torch.isfinite(batch.price_values).all()
    assert torch.isfinite(batch.geo_values).all()


def test_test_inference_does_not_require_or_read_is_positive() -> None:
    from src.downstream.food_recommendation import validate_food_opportunities

    rows = [_food_opportunity(split="test", is_positive=None)]

    assert validate_food_opportunities(rows, expected_split="test", require_labels=False) == tuple(rows)


def test_food_batch_never_promotes_padding_to_a_candidate() -> None:
    from src.downstream.food_recommendation import FoodFeatureArtifacts, build_food_batches

    rows = _variable_length_requests()
    batch = build_food_batches(
        rows,
        encoded_samples_by_id=_encoded_samples_by_id(),
        feature_artifacts=FoodFeatureArtifacts.fit(rows),
        batch_size=2,
        item_hash_buckets=17,
        merchant_hash_buckets=13,
        require_labels=True,
    )[0]

    assert not batch.positive_mask[~batch.candidate_mask].any()
    assert batch.candidate_mask.tolist() == [[True, True, False], [True, True, True]]
    assert tuple(item.candidate_id for item in batch.opportunities[0]) == (
        "candidate-1",
        "candidate-2",
    )


def test_feature_artifacts_round_trip_preserves_train_normalizers_exactly(tmp_path: Path) -> None:
    from src.downstream.food_recommendation import FoodFeatureArtifacts

    artifacts = FoodFeatureArtifacts.fit(_train_rows())
    path = tmp_path / "feature_artifacts.json"
    artifacts.save(path)

    assert FoodFeatureArtifacts.load(path) == artifacts


def _labelled_food_request(*, foundation_sample_id: str = "s1"):
    return [
        _food_opportunity(foundation_sample_id=foundation_sample_id, is_positive=1),
        _food_opportunity(
            foundation_sample_id=foundation_sample_id,
            candidate_id="candidate-2",
            item_id="candidate-2",
            is_positive=0,
        ),
    ]


def _canonical(*, sample_id: str = "s1", user_id: str = "user-1", split: str = "train"):
    return SimpleNamespace(
        sample_id=sample_id,
        user_id=user_id,
        split=split,
        cutoff_time=datetime.fromtimestamp(100, tz=timezone.utc),
    )


def _patch_food_snapshot_dependencies(monkeypatch: pytest.MonkeyPatch, canonicals, encoded) -> None:
    import src.downstream.food_recommendation as food

    monkeypatch.setattr(food, "load_green_sm_tokenizer_config", lambda _root: "tokenizer")
    monkeypatch.setattr(food.VocabularyArtifacts, "load", lambda _path: "vocabulary")
    monkeypatch.setattr(food, "iter_normalized_samples", lambda *_args, **_kwargs: iter(canonicals))
    monkeypatch.setattr(food, "encode_sample", lambda *_args: encoded)


def test_food_snapshot_resolution_rejects_user_cutoff_or_split_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.downstream.food_recommendation import resolve_green_sm_food_samples

    _patch_food_snapshot_dependencies(
        monkeypatch,
        [_canonical(user_id="different-user")],
        _encoded_samples_by_id()["foundation-1"],
    )

    with pytest.raises(ValueError, match="s1.*user_id"):
        resolve_green_sm_food_samples(
            _labelled_food_request(foundation_sample_id="s1"),
            foundation_root="unused",
            vocabulary_path="unused",
            split="train",
            max_tokens=32,
        )


def test_food_snapshot_resolution_returns_only_exact_matching_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.downstream.food_recommendation import resolve_green_sm_food_samples

    encoded = _encoded_samples_by_id()["foundation-1"]
    _patch_food_snapshot_dependencies(
        monkeypatch,
        [_canonical(), _canonical(sample_id="unreferenced")],
        encoded,
    )

    resolved = resolve_green_sm_food_samples(
        _labelled_food_request(foundation_sample_id="s1"),
        foundation_root="unused",
        vocabulary_path="unused",
        split="train",
        max_tokens=32,
    )

    assert resolved == {"s1": encoded}


def _food_features() -> dict[str, torch.Tensor]:
    return {
        "item_hash_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "merchant_hash_ids": torch.tensor([[1, 2, 0], [2, 3, 4]]),
        "category_ids": torch.tensor([[1, 2, 0], [2, 3, 4]]),
        "geo_cell_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "price_values": torch.tensor([[0.1, -0.2, 0.0], [1.0, 0.5, -1.0]]),
        "geo_values": torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]], [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]]
        ),
    }


def test_food_ranker_returns_finite_score_for_each_real_or_padded_candidate() -> None:
    from src.downstream.food_recommendation import FoodRanker

    ranker = FoodRanker(
        d_model=4,
        item_hash_buckets=11,
        merchant_hash_buckets=13,
        num_categories=6,
        num_geo_cells=7,
        dropout=0.0,
    )

    scores = ranker(torch.ones(2, 4), torch.zeros(2, 4), **_food_features())

    assert scores.shape == (2, 3)
    assert torch.isfinite(scores).all()


def test_multi_positive_listwise_loss_prefers_higher_positive_scores() -> None:
    from src.downstream.food_recommendation import masked_multi_positive_listwise_loss

    good, _ = masked_multi_positive_listwise_loss(
        torch.tensor([[3.0, 2.0, -2.0]]),
        torch.tensor([[True, True, False]]),
        torch.tensor([[True, True, True]]),
    )
    bad, _ = masked_multi_positive_listwise_loss(
        torch.tensor([[-2.0, -1.0, 3.0]]),
        torch.tensor([[True, True, False]]),
        torch.tensor([[True, True, True]]),
    )

    assert good < bad


class _StubFoundation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def embed_users(self, _foundation_batch: object) -> dict[str, torch.Tensor]:
        return {
            "long_term": self.scale * torch.ones(1, 4),
            "short_term": self.scale * torch.zeros(1, 4),
        }


def test_frozen_foundation_parameters_receive_no_gradient() -> None:
    from src.downstream.food_recommendation import FoodRanker, frozen_foundation_food_logits

    embedder = _StubFoundation()
    ranker = FoodRanker(
        d_model=4,
        item_hash_buckets=11,
        merchant_hash_buckets=13,
        num_categories=6,
        num_geo_cells=7,
        dropout=0.0,
    )
    features = {name: value[:1] for name, value in _food_features().items()}
    batch = SimpleNamespace(foundation_batch=object(), **features)

    frozen_foundation_food_logits(embedder, ranker, batch).sum().backward()

    assert all(parameter.grad is None for parameter in embedder.parameters())
    assert any(parameter.grad is not None for parameter in ranker.parameters())


def test_food_preflight_rejects_wrong_food_manifest_checksum(tmp_path: Path) -> None:
    from src.downstream.food_recommendation import (
        FoodConfigurationError,
        validate_food_data_handoff,
    )

    food_manifest = tmp_path / "food-manifest.json"
    food_manifest.write_text("{}", encoding="utf-8")
    config = {
        "task": "food_recommendation",
        "freeze_foundation": True,
        "data": {
            "opportunities_root": str(tmp_path),
            "manifest_path": str(food_manifest),
            "foundation_manifest_path": str(tmp_path / "foundation-manifest.json"),
            "expected_food_manifest_sha256": "0" * 64,
            "expected_foundation_manifest_sha256": "1" * 64,
        },
    }

    with pytest.raises(FoodConfigurationError, match="Food manifest SHA-256 mismatch"):
        validate_food_data_handoff(config)


def test_food_prediction_export_has_exact_candidates_and_unbounded_ranking_scores(
    tmp_path: Path,
) -> None:
    from src.contracts.output import OutputContractV1
    from src.downstream.food_recommendation import (
        FoodFeatureArtifacts,
        build_food_batches,
        expected_food_candidates,
        food_prediction_rows_from_batch,
    )

    rows = [
        _food_opportunity(split="validation", candidate_id="candidate-1", item_id="candidate-1"),
        _food_opportunity(
            split="validation", candidate_id="candidate-2", item_id="candidate-2", is_positive=0
        ),
    ]
    batch = build_food_batches(
        rows,
        encoded_samples_by_id=_encoded_samples_by_id(),
        feature_artifacts=FoodFeatureArtifacts.fit(_train_rows()),
        batch_size=1,
        item_hash_buckets=17,
        merchant_hash_buckets=13,
        require_labels=True,
    )[0]
    paths = OutputContractV1.write_predictions(
        food_prediction_rows_from_batch(batch, torch.tensor([[3.5, -2.0]]), split="validation"),
        expected_candidates=expected_food_candidates(
            rows, data_version="test-v1", source_checksums={"food_manifest_sha256": "abc"}
        ),
        output_dir=tmp_path / "predictions" / "validation",
        split="validation",
        run_metadata={
            "selected_checkpoint_sha256": "checkpoint",
            "vocabulary_sha256": "vocabulary",
            "foundation_manifest_sha256": "foundation",
            "food_manifest_sha256": "food",
            "config_digest": "config",
            "seed": 42,
            "run_id": "food-run",
            "score_semantics": "ranking",
        },
    )

    exported = pq.read_table(paths.predictions).to_pylist()
    assert [row["score"] for row in exported] == [3.5, -2.0]
    assert "is_positive" not in exported[0]


def test_food_cli_validate_data_checks_checksum_bound_handoff(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.downstream.food_recommendation as food

    monkeypatch.setattr(food, "load_food_recommendation_config", lambda _path: {"task": "food_recommendation"})
    monkeypatch.setattr(
        food,
        "validate_food_data_handoff",
        lambda _config: {"food_manifest_sha256": "food", "data_version": "pilot"},
    )

    assert food.main(["--config", "fixture.yaml", "--mode", "validate"]) == 0
    assert '"status": "data_handoff_valid"' in capsys.readouterr().out
