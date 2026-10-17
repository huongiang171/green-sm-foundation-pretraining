from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from src.downstream.green_sm_ad_optimization import load_green_sm_ads_opportunities


def test_green_sm_ads_bridge_reads_its_own_schema_without_taobao_field_assumptions(tmp_path) -> None:
    timestamp = datetime(2026, 10, 1, tzinfo=timezone.utc)
    pq.write_table(
        pa.Table.from_pylist([{
            "request_id": "r1", "user_id": "u1", "foundation_sample_id": "foundation:r1",
            "cutoff_timestamp": timestamp, "candidate_id": "campaign1", "split": "validation",
            "candidate_available_timestamp": timestamp, "candidate_category": "food", "clicked": 1,
            "campaign_id": "campaign1", "placement": "home", "exposure_timestamp": timestamp,
        }]),
        tmp_path / "ads_opportunities.validation.parquet",
    )

    rows = load_green_sm_ads_opportunities(tmp_path, split="validation")

    assert rows[0].cate_id == "food"
    assert rows[0].pid == "home"
    assert rows[0].foundation_sample_id == "foundation:r1"


def test_frozen_foundation_keeps_ranker_gradients_enabled() -> None:
    from types import SimpleNamespace

    import torch

    from src.downstream.ad_optimization import AdRanker
    from src.downstream.green_sm_ad_optimization import frozen_foundation_logits

    class StubFoundation(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def embed_users(self, _batch):
            return {
                "long_term": self.scale * torch.ones(1, 4),
                "short_term": self.scale * torch.zeros(1, 4),
            }

    embedder = StubFoundation()
    ranker = AdRanker(
        candidate_buckets=8, campaign_buckets=8, num_categories=3,
        num_placements=3, d_model=4, dropout=0.0,
    )
    batch = SimpleNamespace(
        foundation_batch=object(),
        candidate_hash_ids=torch.tensor([[2]]), campaign_hash_ids=torch.tensor([[2]]),
        category_ids=torch.tensor([[2]]), placement_ids=torch.tensor([[2]]),
    )

    frozen_foundation_logits(embedder, ranker, batch).sum().backward()

    assert embedder.scale.grad is None
    assert any(parameter.grad is not None for parameter in ranker.parameters())
