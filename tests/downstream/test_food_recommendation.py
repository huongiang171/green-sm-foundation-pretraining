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
