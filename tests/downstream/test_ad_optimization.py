from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
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


def _green_sm_opportunity_row(**overrides: object) -> dict[str, object]:
    """Literal native Green SM fixture; no Taobao compatibility aliases."""

    timestamp = datetime(2026, 10, 1, tzinfo=timezone.utc)
    row: dict[str, object] = {
        "request_id": "ads:train:1",
        "user_id": "u001",
        "foundation_sample_id": "green-sm:sample:1",
        "cutoff_timestamp": timestamp,
        "candidate_id": "campaign-creative-1",
        "campaign_id": "campaign-1",
        "candidate_category": "food",
        "placement": "home",
        "candidate_geo_cell": "HCM_Q1",
        "candidate_lat": 10.7769,
        "candidate_lon": 106.7009,
        "candidate_available_timestamp": timestamp,
        "exposure_timestamp": timestamp,
        "split": "train",
        "clicked": 1,
    }
    row.update(overrides)
    return row


def test_native_green_sm_opportunity_requires_native_geo_and_hides_label_for_inference() -> None:
    """Removing native schema support must fail rather than silently routing aliases."""

    row_type = ad_optimization.GreenSmAdsOpportunity
    training_row = row_type.from_mapping(_green_sm_opportunity_row(), require_label=True)
    inference_row = row_type.from_mapping(_green_sm_opportunity_row(clicked=0), require_label=False)

    assert training_row.candidate_category == "food"
    assert training_row.candidate_geo_cell == "HCM_Q1"
    assert training_row.clicked == 1
    assert inference_row.clicked is None
    assert not hasattr(training_row, "cate_id")
    assert not hasattr(training_row, "pid")


def test_native_green_sm_opportunity_rejects_non_finite_location() -> None:
    """Dropping numeric location validation must be observable at the native boundary."""

    row_type = ad_optimization.GreenSmAdsOpportunity
    with pytest.raises(ValueError, match="candidate_lat"):
        row_type.from_mapping(_green_sm_opportunity_row(candidate_lat=float("nan")), require_label=True)


def test_native_feature_artifacts_fit_train_only_category_placement_and_geo_vocabularies() -> None:
    """Fitting validation values would leak future catalog metadata into the ranker."""

    row_type = ad_optimization.GreenSmAdsOpportunity
    artifacts_type = ad_optimization.AdsFeatureArtifacts
    train_rows = (
        row_type.from_mapping(_green_sm_opportunity_row(), require_label=True),
        row_type.from_mapping(
            _green_sm_opportunity_row(
                request_id="ads:train:2",
                candidate_id="campaign-creative-2",
                candidate_category="mobility",
                placement="search",
                candidate_geo_cell="HCM_Q3",
            ),
            require_label=True,
        ),
    )

    artifacts = artifacts_type.fit(train_rows)

    assert artifacts.category["<PAD>"] == 0
    assert artifacts.placement["<UNK>"] == 1
    assert artifacts.geo_cell["HCM_Q1"] == 2
    assert artifacts.encode("category", "food") != artifacts.encode("category", "future-only")
    assert artifacts.encode("geo_cell", "future-only") == 1


def test_native_prediction_batch_omits_labels_and_preserves_real_candidate_mask() -> None:
    """Prediction must score the same candidate tensor even when labels are unavailable."""

    row_type = ad_optimization.GreenSmAdsOpportunity
    train_rows = (
        row_type.from_mapping(_green_sm_opportunity_row(), require_label=True),
        row_type.from_mapping(
            _green_sm_opportunity_row(
                request_id="ads:train:2", foundation_sample_id="green-sm:sample:2",
                candidate_id="campaign-creative-2", campaign_id="campaign-2",
            ),
            require_label=True,
        ),
    )
    test_rows = tuple(
        row_type.from_mapping(
            _green_sm_opportunity_row(
                request_id="ads:test:1", split="test", foundation_sample_id="green-sm:sample:3",
                candidate_id=f"campaign-creative-{index}", clicked=index % 2,
            ),
            require_label=False,
        )
        for index in (3, 4)
    )
    artifacts = ad_optimization.AdsFeatureArtifacts.fit(train_rows)

    batches = ad_optimization.build_green_sm_ads_batches(
        test_rows,
        encoded_samples_by_id={"green-sm:sample:3": _encoded("green-sm:sample:3", "u001")},
        artifacts=artifacts,
        candidate_hash_buckets=97,
        campaign_hash_buckets=97,
        batch_size=1,
    )

    assert batches[0].labels is None
    assert batches[0].candidate_mask.tolist() == [[True, True]]
    assert batches[0].geo_cell_ids.tolist()[0][0] == artifacts.encode("geo_cell", "HCM_Q1")
    assert batches[0].candidate_hash_ids.tolist()[0][0] == ad_optimization.stable_green_sm_hash_id(
        "candidate_id", "campaign-creative-3", bucket_count=97
    )


def test_native_loader_does_not_retain_clicked_for_test_inference(tmp_path: Path) -> None:
    """An inference loader reading clicked would allow accidental test-label leakage."""

    root = tmp_path / "ads_opportunities"
    root.mkdir()
    row = _green_sm_opportunity_row(split="test", clicked=1)
    pq.write_table(pa.Table.from_pylist([row]), root / "ads_opportunities.test.parquet")

    rows = ad_optimization.load_green_sm_ads_opportunities(root, split="test", require_labels=False)

    assert len(rows) == 1
    assert rows[0].clicked is None


def test_frozen_foundation_native_logits_keep_foundation_gradients_empty() -> None:
    """Removing the narrow no-grad boundary must make this train-safety test fail."""

    class FixtureEmbedder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def embed_users(self, _batch: object) -> dict[str, torch.Tensor]:
            value = self.weight.expand(1, 4)
            return {"long_term": value, "short_term": value}

    ranker = ad_optimization.AdsRanker(
        candidate_buckets=17,
        campaign_buckets=17,
        num_categories=3,
        num_placements=3,
        num_geo_cells=3,
        d_model=4,
        dropout=0.0,
    )
    batch = SimpleNamespace(
        foundation_batch=object(),
        candidate_hash_ids=torch.tensor([[2]]),
        campaign_hash_ids=torch.tensor([[2]]),
        category_ids=torch.tensor([[2]]),
        placement_ids=torch.tensor([[2]]),
        geo_cell_ids=torch.tensor([[2]]),
        candidate_mask=torch.tensor([[True]]),
        labels=torch.tensor([[1.0]]),
    )
    embedder = FixtureEmbedder()

    logits = ad_optimization.frozen_foundation_ads_logits(embedder, ranker, batch)
    loss, count = ad_optimization.masked_binary_cross_entropy(logits, batch.labels, batch.candidate_mask)
    loss.backward()

    assert count == 1
    assert embedder.weight.grad is None
    assert any(parameter.grad is not None for parameter in ranker.parameters())


def test_train_ads_ranker_optimizes_only_native_ranker_parameters() -> None:
    """A training loop that updates Foundation defeats the frozen downstream contract."""

    class FixtureEmbedder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def embed_users(self, _batch: object) -> dict[str, torch.Tensor]:
            value = self.weight.expand(1, 4)
            return {"long_term": value, "short_term": value}

    ranker = ad_optimization.AdsRanker(
        candidate_buckets=17, campaign_buckets=17, num_categories=3,
        num_placements=3, num_geo_cells=3, d_model=4, dropout=0.0,
    )
    batch = SimpleNamespace(
        foundation_batch=object(), candidate_hash_ids=torch.tensor([[2]]),
        campaign_hash_ids=torch.tensor([[2]]), category_ids=torch.tensor([[2]]),
        placement_ids=torch.tensor([[2]]), geo_cell_ids=torch.tensor([[2]]),
        candidate_mask=torch.tensor([[True]]), labels=torch.tensor([[1.0]]),
    )
    embedder = FixtureEmbedder()

    metrics = ad_optimization.train_ads_ranker(
        embedder, ranker, train_batches=(batch,), validation_batches=(batch,),
        device=torch.device("cpu"), epochs=1, learning_rate=0.01, weight_decay=0.0,
    )

    assert metrics[0]["epoch"] == 1
    assert metrics[0]["validation_bce"] >= 0.0
    assert embedder.weight.grad is None


def test_native_prediction_export_is_identical_when_test_click_labels_change(tmp_path: Path) -> None:
    """Adding clicked to test features or export would violate the public handoff."""

    row_type = ad_optimization.GreenSmAdsOpportunity
    first = row_type.from_mapping(
        _green_sm_opportunity_row(split="test", clicked=0), require_label=False
    )
    changed_label = row_type.from_mapping(
        _green_sm_opportunity_row(split="test", clicked=1), require_label=False
    )
    batch = SimpleNamespace(
        candidate_mask=torch.tensor([[True]]), opportunities=((first,),),
    )
    changed_batch = SimpleNamespace(
        candidate_mask=torch.tensor([[True]]), opportunities=((changed_label,),),
    )
    logits = torch.tensor([[0.4]])

    rows = ad_optimization.native_prediction_rows_from_batch(batch, logits, split="test")
    changed_rows = ad_optimization.native_prediction_rows_from_batch(changed_batch, logits, split="test")
    paths = ad_optimization.OutputContractV1.write_predictions(
        rows,
        expected_candidates=ad_optimization.expected_green_sm_ads_candidates(
            (first,), data_version="green_sm_sim_full_v1", source_checksums={"ads_manifest_sha256": "a" * 64}
        ),
        output_dir=tmp_path,
        split="test",
        run_metadata={
            "selected_checkpoint_sha256": "c" * 64,
            "vocabulary_sha256": "v" * 64,
            "foundation_manifest_sha256": "f" * 64,
            "ads_manifest_sha256": "a" * 64,
            "config_digest": "d" * 64,
            "seed": 7,
            "run_id": "fixture",
            "score_semantics": "probability",
        },
    )

    assert rows == changed_rows
    assert "clicked" not in pq.read_table(paths.predictions).column_names


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


def test_selected_foundation_handoff_uses_checkpoint_manifest_not_best_filename(tmp_path: Path) -> None:
    """Ads must follow the published selected checkpoint, including a valid `last` selection."""

    from src.downstream.ad_optimization import resolve_selected_foundation_handoff

    package = tmp_path / "foundation-package"
    package.mkdir()
    checkpoint = package / "last.pt"
    checkpoint.write_bytes(b"selected-last-checkpoint")
    vocabulary = package / "vocabulary.json"
    vocabulary.write_text('{"schema_version":"fixture"}\n', encoding="utf-8")
    manifest = package / "checkpoint_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "green_sm_foundation_checkpoint_v1",
                "checkpoint_format_version": 1,
                "selected_kind": "last",
                "checkpoint": "last.pt",
                "checkpoint_sha256": _sha256(checkpoint),
                "vocabulary_sha256": _sha256(vocabulary),
                "linked_manifest_sha256": "linked-foundation",
                "control_manifest_sha256": "control-foundation",
                "config_digest": "config",
                "corpus": "linked",
            }
        ),
        encoding="utf-8",
    )

    handoff = resolve_selected_foundation_handoff(package)

    assert handoff.checkpoint_path == checkpoint
    assert handoff.selected_kind == "last"
    assert handoff.vocabulary_path == vocabulary


def _write_green_sm_handoff_config(tmp_path: Path, *, selected_kind: str = "last") -> dict[str, object]:
    """Build a complete small linked handoff whose expected values are hand-derived."""

    ads_root = tmp_path / "ads"
    foundation_root = tmp_path / "foundation"
    report_root = tmp_path / "reports"
    package_root = tmp_path / "foundation-package"
    for path in (ads_root, foundation_root, report_root, package_root):
        path.mkdir()
    foundation_manifest = foundation_root / "manifest.json"
    foundation_manifest.write_text(
        json.dumps({"schema_version": "green_sm_foundation_v3", "status": "complete"}),
        encoding="utf-8",
    )
    files: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        path = ads_root / f"ads_opportunities.{split}.parquet"
        path.write_bytes(f"native-{split}".encode())
        files[split] = {"file": path.name, "rows": 1, "sha256": _sha256(path)}
    ads_manifest = ads_root / "manifest.json"
    ads_manifest.write_text(
        json.dumps(
            {
                "schema_version": "green_sm_ads_opportunities_v1",
                "data_version": "green_sm_sim_full_v1",
                "foundation_manifest_sha256": _sha256(foundation_manifest),
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    (report_root / "data_quality_report.json").write_text(
        json.dumps(
            {"readiness_passed": True, "foundation_manifest_sha256": _sha256(foundation_manifest)}
        ),
        encoding="utf-8",
    )
    selected = package_root / f"{selected_kind}.pt"
    selected.write_bytes(f"selected-{selected_kind}".encode())
    vocabulary = package_root / "vocabulary.json"
    vocabulary.write_text('{"schema_version":"fixture"}\n', encoding="utf-8")
    (package_root / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "green_sm_foundation_checkpoint_v1",
                "checkpoint_format_version": 1,
                "selected_kind": selected_kind,
                "checkpoint": selected.name,
                "checkpoint_sha256": _sha256(selected),
                "vocabulary_sha256": _sha256(vocabulary),
                "linked_manifest_sha256": _sha256(foundation_manifest),
                "control_manifest_sha256": "c" * 64,
                "config_digest": "d" * 64,
                "corpus": "linked",
            }
        ),
        encoding="utf-8",
    )
    return {
        "task": "ad_optimization",
        "freeze_foundation": True,
        "data": {
            "data_version": "green_sm_sim_full_v1",
            "corpus": "linked",
            "opportunities_root": str(ads_root),
            "manifest_path": str(ads_manifest),
            "foundation_root": str(foundation_root),
            "foundation_manifest_path": str(foundation_manifest),
            "quality_report_path": str(report_root / "data_quality_report.json"),
            "expected_ads_manifest_sha256": _sha256(ads_manifest),
            "expected_foundation_manifest_sha256": _sha256(foundation_manifest),
        },
        "checkpoint": {
            "package_root": str(package_root),
            "expected_selected_checkpoint_sha256": _sha256(selected),
        },
    }


def test_green_sm_validate_binds_the_manifest_selected_last_checkpoint(tmp_path: Path) -> None:
    """Changing selected checkpoint identity must stop validate before a model can load."""

    report = ad_optimization.validate_green_sm_ads_data_handoff(
        _write_green_sm_handoff_config(tmp_path, selected_kind="last")
    )

    assert report["selected_kind"] == "last"
    assert report["selected_checkpoint_sha256"] == _sha256(tmp_path / "foundation-package" / "last.pt")


def test_green_sm_validate_rejects_placeholder_selected_checkpoint_sha(tmp_path: Path) -> None:
    """A placeholder must never let a full-corpus Kaggle run reach model construction."""

    config = _write_green_sm_handoff_config(tmp_path)
    checkpoint = dict(config["checkpoint"])
    checkpoint["expected_selected_checkpoint_sha256"] = "replace-with-selected-foundation-checkpoint-sha256"
    config["checkpoint"] = checkpoint

    with pytest.raises(ValueError, match="must be a SHA-256"):
        ad_optimization.validate_green_sm_ads_data_handoff(config)


def test_green_sm_ads_handoff_rejects_nonpassing_quality_report(tmp_path: Path) -> None:
    from src.downstream.ad_optimization import validate_green_sm_ads_data_handoff

    ads_root = tmp_path / "ads"
    foundation_root = tmp_path / "foundation"
    report_root = tmp_path / "reports"
    ads_root.mkdir(); foundation_root.mkdir(); report_root.mkdir()
    foundation_manifest = foundation_root / "manifest.json"
    foundation_manifest.write_text('{"schema_version":"omnibehavior_pretraining_v2","status":"complete"}', encoding="utf-8")
    files = {}
    for split in ("train", "validation", "test"):
        path = ads_root / f"ads_opportunities.{split}.parquet"
        path.write_bytes(split.encode())
        files[split] = {"file": path.name, "sha256": _sha256(path), "rows": 1}
    ads_manifest = ads_root / "manifest.json"
    ads_manifest.write_text(json.dumps({
        "schema_version": "green_sm_ads_opportunities_v1",
        "foundation_manifest_sha256": _sha256(foundation_manifest), "files": files,
    }), encoding="utf-8")
    report = report_root / "data_quality_report.json"
    report.write_text(json.dumps({"readiness_passed": False, "foundation_manifest_sha256": _sha256(foundation_manifest)}), encoding="utf-8")

    with pytest.raises(ValueError, match="quality report"):
        validate_green_sm_ads_data_handoff({
            "task": "ad_optimization", "freeze_foundation": True,
            "data": {
                "data_version": "green_sm_sim_full_v1", "corpus": "linked",
                "opportunities_root": str(ads_root), "manifest_path": str(ads_manifest),
                "foundation_root": str(foundation_root), "foundation_manifest_path": str(foundation_manifest),
                "quality_report_path": str(report),
                "expected_ads_manifest_sha256": _sha256(ads_manifest),
                "expected_foundation_manifest_sha256": _sha256(foundation_manifest),
            },
        })


def test_cli_validate_uses_green_sm_handoff(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The production CLI exposes a Green SM validate mode, not Taobao flags."""

    monkeypatch.setattr(ad_optimization, "load_ad_optimization_config", lambda _: {"task": "ad_optimization"})
    monkeypatch.setattr(
        ad_optimization,
        "validate_green_sm_ads_data_handoff",
        lambda _: {"data_version": "green_sm_sim_full_v1", "ads_manifest_sha256": "a" * 64},
    )

    exit_code = ad_optimization.main(["--config", "unused.yaml", "--mode", "validate"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ads_data_handoff_valid"
