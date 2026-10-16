from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.downstream import ad_optimization
from src.downstream.ad_optimization import (
    AdOpportunity,
    AdRanker,
    build_ad_batches,
    encode_train_category,
    fit_train_vocabulary,
    load_encoded_foundation_samples_for_ads,
    masked_binary_cross_entropy,
    score_probabilities,
    stable_hash_id,
    summarize_temporal_oov,
    validate_opportunities,
)
from src.foundation_input.types import CanonicalSample, EncodedSample


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_handoff_config(tmp_path: Path) -> Path:
    """Create a minimal checksum-bound Data handoff, encoded as YAML-compatible JSON."""

    ads_root = tmp_path / "ads"
    foundation_root = tmp_path / "foundation"
    ads_root.mkdir()
    foundation_root.mkdir()
    files: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        path = ads_root / f"ad_opportunities.{split}.parquet"
        path.write_bytes(f"{split}-fixture".encode())
        files[split] = {"file": path.name, "rows": 1, "sha256": _sha256(path)}
    foundation_manifest = foundation_root / "manifest.json"
    foundation_manifest.write_text(
        json.dumps({"schema_version": "omnibehavior_pretraining_v2", "status": "complete"}), encoding="utf-8"
    )
    ads_manifest = ads_root / "manifest.json"
    ads_manifest.write_text(
        json.dumps(
            {
                "schema_version": "taobao_ads_opportunities_v1",
                "data_version": "fixture_v1",
                "foundation_manifest_sha256": _sha256(foundation_manifest),
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "task": "ad_optimization",
        "data": {
            "opportunities_root": str(ads_root),
            "manifest_path": str(ads_manifest),
            "foundation_root": str(foundation_root),
            "foundation_manifest_path": str(foundation_manifest),
            "expected_data_version": "fixture_v1",
            "expected_ads_manifest_sha256": _sha256(ads_manifest),
            "expected_foundation_manifest_sha256": _sha256(foundation_manifest),
        },
        "checkpoint": {
            "path": str(tmp_path / "foundation.pt"),
            "expected_sha256": "supplied-by-member-3-checkpoint-manifest",
        },
        "freeze_foundation": True,
    }
    config_path = tmp_path / "ad.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _opportunity(**overrides: object) -> AdOpportunity:
    row: dict[str, object] = {
        "request_id": "request-1",
        "user_id": "user-1",
        "foundation_sample_id": "sample-1",
        "cutoff_timestamp": 100,
        "candidate_id": "ad-1",
        "clicked": 0,
        "split": "train",
        "exposure_timestamp": 100,
        "candidate_available_timestamp": 100,
        "pid": "slot-1",
        "cate_id": "category-1",
        "campaign_id": "campaign-1",
    }
    row.update(overrides)
    return AdOpportunity.from_mapping(row)


def _encoded(sample_id: str = "sample-1", user_id: str = "user-1") -> EncodedSample:
    return EncodedSample(sample_id=sample_id, dataset_name="taobao_display_ads", user_id=user_id, records=())


def test_train_category_vocabulary_maps_validation_only_value_to_unk() -> None:
    vocabulary = fit_train_vocabulary([_opportunity(cate_id="food")], field="cate_id")

    assert vocabulary == {"<PAD>": 0, "<UNK>": 1, "food": 2}
    assert encode_train_category("new-category", vocabulary) == 1


def test_candidate_hash_is_stable_for_validation_only_candidate() -> None:
    first = stable_hash_id("candidate_id", "new-ad", bucket_count=65_536)

    assert first == stable_hash_id("candidate_id", "new-ad", bucket_count=65_536)
    assert 2 <= first < 65_536


def test_validation_rejects_duplicate_candidate_within_request() -> None:
    rows = [_opportunity(), _opportunity(clicked=1)]

    with pytest.raises(ValueError, match="duplicate candidate.*request-1"):
        validate_opportunities(rows, expected_split="train")


def test_validation_rejects_post_cutoff_exposure() -> None:
    with pytest.raises(ValueError, match="exposure_timestamp.*cutoff_timestamp"):
        _opportunity(exposure_timestamp=101)


def test_temporal_oov_report_never_fits_evaluation_values() -> None:
    report = summarize_temporal_oov(
        [_opportunity(candidate_id="old", campaign_id="old-campaign")],
        [_opportunity(split="validation", candidate_id="new", campaign_id="new-campaign")],
    )

    assert report["candidate_id"]["unseen_rows"] == 1
    assert report["campaign_id"]["unseen_rate"] == 1.0


def test_build_batches_keeps_no_click_and_empty_history_sample() -> None:
    rows = [
        _opportunity(candidate_id="ad-1", clicked=0),
        _opportunity(candidate_id="ad-2", clicked=1),
        _opportunity(
            request_id="request-2",
            foundation_sample_id="sample-2",
            candidate_id="ad-3",
            clicked=0,
        ),
    ]
    category_vocabulary = fit_train_vocabulary(rows, field="cate_id")
    placement_vocabulary = fit_train_vocabulary(rows, field="pid")

    batches = build_ad_batches(
        rows,
        encoded_samples_by_id={"sample-1": _encoded(), "sample-2": _encoded("sample-2")},
        category_vocabulary=category_vocabulary,
        placement_vocabulary=placement_vocabulary,
        candidate_hash_buckets=17,
        campaign_hash_buckets=17,
        batch_size=2,
    )

    assert len(batches) == 1
    assert batches[0].candidate_mask.tolist() == [[True, True], [True, False]]
    assert batches[0].clicked.tolist() == [[0.0, 1.0], [0.0, 0.0]]
    assert batches[0].foundation_batch.input_ids.shape[0] == 2


def test_ad_ranker_returns_one_logit_per_real_or_padded_candidate() -> None:
    model = AdRanker(
        candidate_buckets=8,
        campaign_buckets=8,
        num_categories=5,
        num_placements=4,
        d_model=4,
        dropout=0.0,
    )
    logits = model(
        torch.ones(2, 4),
        torch.zeros(2, 4),
        candidate_hash_ids=torch.tensor([[2, 3], [4, 0]]),
        campaign_hash_ids=torch.tensor([[2, 3], [4, 0]]),
        category_ids=torch.tensor([[2, 2], [3, 0]]),
        placement_ids=torch.tensor([[2, 2], [2, 0]]),
    )

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_masked_ctr_loss_accepts_no_click_and_multi_click_requests() -> None:
    logits = torch.tensor([[2.0, -1.0], [0.5, -0.5]], requires_grad=True)
    clicked = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    mask = torch.tensor([[True, True], [True, True]])

    loss, count = masked_binary_cross_entropy(logits, clicked, mask)
    loss.backward()

    assert count == 4
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_changed_foundation_embedding_changes_only_its_request_score() -> None:
    torch.manual_seed(0)
    model = AdRanker(
        candidate_buckets=8,
        campaign_buckets=8,
        num_categories=5,
        num_placements=4,
        d_model=4,
        dropout=0.0,
    ).eval()
    features = {
        "candidate_hash_ids": torch.tensor([[2], [3]]),
        "campaign_hash_ids": torch.tensor([[2], [3]]),
        "category_ids": torch.tensor([[2], [3]]),
        "placement_ids": torch.tensor([[2], [2]]),
    }
    original = model(torch.zeros(2, 4), torch.zeros(2, 4), **features)
    changed = model(torch.tensor([[9.0] * 4, [0.0] * 4]), torch.zeros(2, 4), **features)

    assert not torch.equal(original[0], changed[0])
    torch.testing.assert_close(original[1], changed[1])


def test_score_probabilities_zeroes_padding_only() -> None:
    probabilities = score_probabilities(
        torch.tensor([[0.0, 2.0]]), torch.tensor([[True, False]])
    )

    assert probabilities.tolist() == [[0.5, 0.0]]


def test_foundation_resolution_rejects_user_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    canonical = CanonicalSample(
        sample_id="sample-1",
        dataset_name="taobao_display_ads",
        split="train",
        user_id="different-user",
        cutoff_time=datetime.fromtimestamp(100, tz=timezone.utc),
        cutoff_inclusive=False,
        history_order_known=True,
        history_truncated=False,
        history_group_truncated=False,
        profile={},
        events=(),
    )
    monkeypatch.setattr(ad_optimization.VocabularyArtifacts, "load", lambda _: object())
    monkeypatch.setattr(
        ad_optimization,
        "FoundationPretrainDataset",
        lambda *args, **kwargs: SimpleNamespace(tokenizer_config=object(), dataset_config=object()),
    )
    monkeypatch.setattr(ad_optimization, "iter_normalized_samples", lambda *args, **kwargs: [canonical])

    with pytest.raises(ValueError, match="sample-1.*user_id"):
        load_encoded_foundation_samples_for_ads(
            [_opportunity()],
            foundation_root="unused",
            vocabulary_path="unused",
            split="train",
            max_tokens=256,
        )


def test_foundation_resolution_returns_matching_encoded_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    canonical = CanonicalSample(
        sample_id="sample-1",
        dataset_name="taobao_display_ads",
        split="train",
        user_id="user-1",
        cutoff_time=datetime.fromtimestamp(100, tz=timezone.utc),
        cutoff_inclusive=False,
        history_order_known=True,
        history_truncated=False,
        history_group_truncated=False,
        profile={},
        events=(),
    )
    encoded = _encoded()
    monkeypatch.setattr(ad_optimization.VocabularyArtifacts, "load", lambda _: object())
    monkeypatch.setattr(
        ad_optimization,
        "FoundationPretrainDataset",
        lambda *args, **kwargs: SimpleNamespace(tokenizer_config=object(), dataset_config=object()),
    )
    monkeypatch.setattr(ad_optimization, "iter_normalized_samples", lambda *args, **kwargs: [canonical])
    monkeypatch.setattr(ad_optimization, "encode_sample", lambda *args: encoded)

    resolved = load_encoded_foundation_samples_for_ads(
        [_opportunity()], foundation_root="unused", vocabulary_path="unused", split="train", max_tokens=256
    )

    assert resolved == {"sample-1": encoded}


def test_cli_validate_data_checks_checksum_bound_handoff(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write_handoff_config(tmp_path)

    exit_code = ad_optimization.main(["--config", str(config_path), "--validate-data"])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "data_handoff_valid"
    assert report["data_version"] == "fixture_v1"


def test_cli_preflight_blocks_placeholder_member3_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write_handoff_config(tmp_path)

    exit_code = ad_optimization.main(["--config", str(config_path), "--preflight"])

    assert exit_code == 3
    assert "Member 3 checkpoint manifest has not been supplied" in capsys.readouterr().err


def test_cli_rejects_tampered_ads_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write_handoff_config(tmp_path)
    (tmp_path / "ads" / "ad_opportunities.train.parquet").write_bytes(b"tampered")

    exit_code = ad_optimization.main(["--config", str(config_path), "--validate-data"])

    assert exit_code == 2
    assert "Ads train parquet SHA-256 mismatch" in capsys.readouterr().err
