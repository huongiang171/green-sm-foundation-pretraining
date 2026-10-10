"""Step 3: Generate manifest.json, preprocessing_config.json, preview.json, and data_report.json.

Creates all required metadata files in pretrain_data/meituan_data/ conforming
to the OmniBehavior v2 specification.
"""

import concurrent.futures
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meituan_analysis.common.config import (
    PRETRAIN_OUTPUT_DIR,
    PRETRAIN_EVENTS_DIR,
    PRETRAIN_SNAPSHOTS_DIR,
    SCHEMA_VERSION,
    MAX_HISTORY_EVENTS,
    ARTIFACTS_DIR,
)
from meituan_analysis.common.utils import (
    get_logger,
    ensure_dir,
    dump_json,
    compute_file_sha256,
)

logger = get_logger("03_generate_manifest")

def file_info(filepath: Path) -> dict:
    """Computes file bytes and sha256."""
    size = filepath.stat().st_size
    h = compute_file_sha256(filepath)
    return {"bytes": size, "sha256": h}

def build_preprocessing_config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "language": "zh",
        "timezone": None,
        "max_history_events": MAX_HISTORY_EVENTS,
        "event_unit": "one raw action observation",
        "profile_text_used": False,
        "context_allowlist": {
            "Advertisement": ["product", "industry_main", "industry_sub", "caption", "ocr_text", "asr_text"],
            "Live Streaming": ["live_category", "live_title", "is_shop_live", "live_cover_content"],
            "Video Browsing": ["caption", "photo_type", "duration", "ocr", "asr"],
            "E-commerce": [
                "product_source", "product_id", "item_title", "item_desc", "item_price",
                "external_item_title", "category_level1_name", "category_level2_name",
                "category_level3_name", "industry_level1_name", "industry_level2_name",
                "industry_level3_name"
            ],
            "Customer Service": ["ticket_category", "product_name", "item_price", "product_category_info"],
            "Search Behavior": []
        },
        "scenario_profile_fields": {
            "Advertisement": "advertisement_event_count_30d",
            "Live Streaming": "live_streaming_event_count_30d",
            "Video Browsing": "video_browsing_event_count_30d",
            "E-commerce": "e_commerce_event_count_30d",
            "Customer Service": "customer_service_event_count_30d",
            "Search Behavior": "search_behavior_event_count_30d"
        },
        "profile_window": "(cutoff - 30 days, cutoff] in (eligibility_time, inclusive) coordinates",
        "active_day_basis": "calendar date of source timestamp; counts observations including negative states",
        "history_selection": "last 30 sorted by eligibility_time, eligibility_inclusive, source_record_index, action_index",
        "snapshot_cadence": "one per eligibility coordinate; day closing precedes an exact midnight observation",
        "splits": {
            "start_inclusive": "2021-03-01T00:00:00",
            "train_end_exclusive": "2021-03-22T00:00:00",
            "day_closing_at_boundary": "belongs to preceding split"
        },
        "compression": {
            "codec": "zstd",
            "level": 3,
            "rows_per_write_batch": 4096
        },
        "training_targets_stored": False,
        "mask_policy": None,
        "vocabulary_fit": False
    }

def process_single_user_meta(uid: str) -> dict:
    ev_path = PRETRAIN_EVENTS_DIR / f"{uid}.parquet"
    sn_path = PRETRAIN_SNAPSHOTS_DIR / f"{uid}.parquet"

    ev_info = file_info(ev_path)
    sn_info = file_info(sn_path)

    ev_table = pq.read_table(ev_path, columns=["row_index"])
    sn_table = pq.read_table(sn_path, columns=["split"])

    ev_rows = len(ev_table)
    sn_rows = len(sn_table)

    return {
        "user_id": uid,
        "events_file": f"events/{uid}.parquet",
        "snapshots_file": f"snapshots/{uid}.parquet",
        "event_rows": ev_rows,
        "snapshot_rows": sn_rows,
        "snapshots_by_split": {
            "train": sn_rows
        },
        "events_bytes": ev_info["bytes"],
        "snapshots_bytes": sn_info["bytes"],
        "events_sha256": ev_info["sha256"],
        "snapshots_sha256": sn_info["sha256"],
    }

def build_preview(sample_uids: List[str]) -> List[dict]:
    previews = []
    for uid in sample_uids:
        ev_table = pq.read_table(PRETRAIN_EVENTS_DIR / f"{uid}.parquet")
        sn_table = pq.read_table(PRETRAIN_SNAPSHOTS_DIR / f"{uid}.parquet")

        ev_dict = ev_table.to_pydict()
        sn_dict = sn_table.to_pydict()

        sample_id = sn_dict["sample_id"][0]
        cutoff_time = sn_dict["cutoff_time"][0].isoformat()
        cutoff_inclusive = sn_dict["cutoff_inclusive"][0]
        h_start = sn_dict["history_start"][0]
        h_stop = sn_dict["history_stop"][0]
        h_order_known = sn_dict["history_order_known"][0]
        h_truncated = sn_dict["history_truncated"][0]
        h_group_truncated = sn_dict["history_group_truncated"][0]
        profile = json.loads(sn_dict["profile_json"][0])

        history_events = []
        for i in range(h_start, h_stop):
            history_events.append({
                "event_id": ev_dict["event_id"][i],
                "source_record_id": ev_dict["source_record_id"][i],
                "action_index": ev_dict["action_index"][i],
                "event_time": ev_dict["event_time"][i],
                "time_precision": ev_dict["time_precision"][i],
                "scenario": ev_dict["scenario"][i],
                "event_type": ev_dict["event_type"][i],
                "context": json.loads(ev_dict["context_json"][i]),
                "action": json.loads(ev_dict["action_json"][i]),
            })

        previews.append({
            "sample_id": sample_id,
            "input": {
                "user_id": uid,
                "cutoff_time": cutoff_time,
                "cutoff_inclusive": cutoff_inclusive,
                "timezone": None,
                "history_order_known": h_order_known,
                "history_truncated": h_truncated,
                "history_group_truncated": h_group_truncated,
                "profile": profile,
                "history": history_events,
            }
        })
    return previews

def main():
    logger.info("=== STEP 3: GENERATE MANIFEST & PRETRAIN CONFIG ===")
    start_time = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    # 1. preprocessing_config.json
    config_data = build_preprocessing_config()
    config_file = PRETRAIN_OUTPUT_DIR / "preprocessing_config.json"
    dump_json(config_data, config_file)
    logger.info(f"Saved preprocessing_config.json")

    # 2. preview.json
    pretrain_users_file = ARTIFACTS_DIR / "pretrain_users.txt"
    with pretrain_users_file.open("r", encoding="utf-8") as f:
        all_uids = [line.strip() for line in f if line.strip()]

    sample_uids = all_uids[:5]
    preview_data = build_preview(sample_uids)
    preview_file = PRETRAIN_OUTPUT_DIR / "preview.json"
    dump_json(preview_data, preview_file)
    logger.info(f"Saved preview.json with {len(preview_data)} reconstructed samples.")

    # 3. Process all users for manifest
    logger.info(f"Processing metadata for all {len(all_uids):,} users...")
    max_workers = min(12, (os.cpu_count() or 4) * 2)

    user_meta_list = []
    total_events = 0
    total_snapshots = 0
    total_event_bytes = 0
    total_snapshot_bytes = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_user_meta, uid): uid for uid in all_uids}
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            res = fut.result()
            user_meta_list.append(res)
            total_events += res["event_rows"]
            total_snapshots += res["snapshot_rows"]
            total_event_bytes += res["events_bytes"]
            total_snapshot_bytes += res["snapshots_bytes"]

            if idx % 2000 == 0 or idx == len(all_uids):
                logger.info(f"Hashed {idx:,}/{len(all_uids):,} users...")

    # Sort users deterministically by user_id
    user_meta_list.sort(key=lambda x: x["user_id"])

    # 4. data_report.json
    event_counts = [u["event_rows"] for u in user_meta_list]
    report_data = {
        "dataset_name": "Meituan TRD OmniBehavior v2 Pretrain Dataset",
        "generated_at_utc": started_at,
        "total_users": len(all_uids),
        "total_events": total_events,
        "total_snapshots": total_snapshots,
        "total_event_bytes": total_event_bytes,
        "total_snapshot_bytes": total_snapshot_bytes,
        "avg_events_per_user": float(total_events / len(all_uids)),
        "min_events_per_user": int(min(event_counts)),
        "max_events_per_user": int(max(event_counts)),
        "p50_events_per_user": float(sorted(event_counts)[len(event_counts) // 2]),
    }
    report_file = PRETRAIN_OUTPUT_DIR / "data_report.json"
    dump_json(report_data, report_file)
    logger.info(f"Saved data_report.json")

    completed_at = datetime.now(timezone.utc).isoformat()

    # 5. manifest.json
    manifest_data = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "started_at_utc": started_at,
        "source_dataset": "Meituan TRD (Takeout Recommendation Dataset)",
        "license": "Research and Non-Commercial Use Only",
        "pyarrow_version": "24.0.0",
        "config_sha256": compute_file_sha256(config_file),
        "preview_sha256": compute_file_sha256(preview_file),
        "report_sha256": compute_file_sha256(report_file),
        "total_events": total_events,
        "total_snapshots": total_snapshots,
        "total_users": len(all_uids),
        "completed_at_utc": completed_at,
        "shards": user_meta_list,
        "users": user_meta_list,
    }
    manifest_file = PRETRAIN_OUTPUT_DIR / "manifest.json"
    dump_json(manifest_data, manifest_file)
    logger.info(f"Saved manifest.json ({manifest_file.stat().st_size:,} bytes).")

    elapsed = time.time() - start_time
    logger.info(f"✓ Step 3 completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
