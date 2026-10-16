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
