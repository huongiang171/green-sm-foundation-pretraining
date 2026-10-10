"""Step 2: Build Pretrain Dataset according to OmniBehavior v2 schema.

Converts 20,000 sampled users' interaction history up to 2021-03-21T23:59:59
into:
- pretrain_data/meituan_data/events/{user_id}.parquet (compressed with zstd level 3)
- pretrain_data/meituan_data/snapshots/{user_id}.parquet (compressed with zstd level 3)
"""

import concurrent.futures
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meituan_analysis.common.config import (
    MEITUAN_RAW_DIR,
    ARTIFACTS_DIR,
    PRETRAIN_OUTPUT_DIR,
    PRETRAIN_EVENTS_DIR,
    PRETRAIN_SNAPSHOTS_DIR,
    PRETRAIN_CUTOFF_TIME,
    PRETRAIN_SPLIT,
    MAX_HISTORY_EVENTS,
    SCHEMA_VERSION,
    DEFAULT_SCENARIO,
    DEFAULT_PRODUCT_SOURCE,
    DEFAULT_INDUSTRY_LEVEL1,
    PRICE_INTERVAL_TO_MIDPOINT,
)
from meituan_analysis.common.utils import (
    get_logger,
    ensure_dir,
    epoch_to_datetime,
    epoch_to_iso,
    clean_tag,
    safe_float,
    dump_json,
)

logger = get_logger("02_build_pretrain_data")

# Cutoff timestamp in epoch seconds: 2021-03-21 23:59:59 (Asia/Shanghai or local)
# 2021-03-21 23:59:59 datetime
CUTOFF_DT = datetime.fromisoformat(PRETRAIN_CUTOFF_TIME)
CUTOFF_EPOCH = int(CUTOFF_DT.timestamp())

_METADATA = {"schema_version": SCHEMA_VERSION}

EVENT_SCHEMA = pa.schema([
    ("row_index", pa.int32()),
    ("event_id", pa.string()),
    ("user_id", pa.string()),
    ("source_record_id", pa.string()),
    ("source_record_index", pa.int32()),
    ("action_index", pa.int32()),
    ("source_action_count", pa.int32()),
    ("source_timestamp", pa.string()),
    ("event_time", pa.string()),
    ("time_precision", pa.string()),
    ("time_lower_bound", pa.timestamp("us")),
    ("time_upper_bound", pa.timestamp("us")),
    ("upper_bound_exclusive", pa.bool_()),
    ("eligibility_time", pa.timestamp("us")),
    ("eligibility_inclusive", pa.bool_()),
    ("scenario", pa.string()),
    ("event_type", pa.string()),
    ("context_json", pa.large_string()),
    ("action_json", pa.large_string()),
], metadata=_METADATA)

SNAPSHOT_SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("user_id", pa.string()),
    ("cutoff_time", pa.timestamp("us")),
    ("cutoff_inclusive", pa.bool_()),
    ("split", pa.string()),
    ("history_start", pa.int32()),
    ("history_stop", pa.int32()),
    ("history_event_ids", pa.list_(pa.field("element", pa.string()))),
    ("history_order_known", pa.bool_()),
    ("history_truncated", pa.bool_()),
    ("history_group_truncated", pa.bool_()),
    ("profile_json", pa.string()),
], metadata=_METADATA)


def load_pois() -> Dict[str, dict]:
    """Loads POI master metadata into memory."""
    logger.info("Loading pois.txt...")
    pois_file = MEITUAN_RAW_DIR / "pois.txt"
    poi_dict = {}
    with pois_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        p_id_idx = header.index("wm_poi_id")
        p_name_idx = header.index("wm_poi_name")
        tag2_idx = header.index("primary_second_tag_name")
        tag3_idx = header.index("primary_third_tag_name")
        tag1_idx = header.index("primary_first_tag_name")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            poi_id = parts[p_id_idx]
            poi_dict[poi_id] = {
                "wm_poi_name": parts[p_name_idx],
                "primary_first_tag_name": clean_tag(parts[tag1_idx]),
                "primary_second_tag_name": clean_tag(parts[tag2_idx]),
                "primary_third_tag_name": clean_tag(parts[tag3_idx]),
            }
    logger.info(f"Loaded {len(poi_dict):,} POIs into memory.")
    return poi_dict


def process_user_batch(
    users_batch: List[str],
    user_orders: Dict[str, list],
    order_sessions: Dict[str, str],
    poi_dict: Dict[str, dict],
    events_dir: Path,
    snapshots_dir: Path,
) -> Tuple[int, int]:
    """Processes a batch of users and writes their Parquet files."""
    total_events_written = 0
    total_snapshots_written = 0

    for uid in users_batch:
        orders = user_orders.get(uid, [])
        if not orders:
            continue

        # Sort orders strictly by order_timestamp ascending
        orders.sort(key=lambda o: (o["order_timestamp"], o["wm_order_id"]))

        events_data = []
        unique_dates = set()

        for src_idx, order in enumerate(orders):
            oid = order["wm_order_id"]
            order_ts = order["order_timestamp"]
            order_dt = epoch_to_datetime(order_ts)
            dt_str = order["dt"]
            unique_dates.add(dt_str)

            price_interval = order["order_price_interval"]
            price_midpoint = PRICE_INTERVAL_TO_MIDPOINT.get(price_interval, 35.0)

            # Check clicks in session
            clicks_str = order_sessions.get(oid, "NULL")
            clicks = [c for c in clicks_str.split("#") if c and c != "NULL"] if clicks_str != "NULL" else []
            num_clicks = len(clicks)
            source_action_count = num_clicks + 1

            # 1. Click events
            for clk_idx, clk_poi_id in enumerate(clicks):
                # Nominal step: 30s before order
                clk_ts = order_ts - (num_clicks - clk_idx) * 30
                clk_dt = epoch_to_datetime(clk_ts)
                clk_iso = epoch_to_iso(clk_ts)

                poi_meta = poi_dict.get(clk_poi_id, {
                    "wm_poi_name": f"poi_{clk_poi_id}",
                    "primary_first_tag_name": "unknown",
                    "primary_second_tag_name": "unknown",
                    "primary_third_tag_name": "unknown",
                })

                context = {
                    "product_source": DEFAULT_PRODUCT_SOURCE,
                    "product_id": str(clk_poi_id),
                    "item_title": poi_meta["wm_poi_name"],
                    "item_desc": "",
                    "item_price": float(price_midpoint),
                    "external_item_title": "",
                    "category_level1_name": poi_meta["primary_first_tag_name"],
                    "category_level2_name": poi_meta["primary_second_tag_name"],
                    "category_level3_name": poi_meta["primary_third_tag_name"],
                    "industry_level1_name": DEFAULT_INDUSTRY_LEVEL1,
                    "industry_level2_name": poi_meta["primary_first_tag_name"],
                    "industry_level3_name": poi_meta["primary_second_tag_name"],
                }

                action = {
                    "type": "click",
                    "click_cnt": 1,
                }

                events_data.append({
                    "event_id": f"{uid}:{src_idx}:{clk_idx}",
                    "user_id": uid,
                    "source_record_id": f"{uid}:{src_idx}",
                    "source_record_index": src_idx,
                    "action_index": clk_idx,
                    "source_action_count": source_action_count,
                    "source_timestamp": clk_iso,
                    "event_time": clk_iso,
                    "time_precision": "second",
                    "time_lower_bound": clk_dt,
                    "time_upper_bound": clk_dt,
                    "upper_bound_exclusive": False,
                    "eligibility_time": clk_dt,
                    "eligibility_inclusive": True,
                    "scenario": DEFAULT_SCENARIO,
                    "event_type": "click",
                    "context_json": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                    "action_json": json.dumps(action, ensure_ascii=False, separators=(",", ":")),
                })

            # 2. Purchase event
            order_iso = epoch_to_iso(order_ts)
            order_poi_id = order["wm_poi_id"]
            poi_meta = poi_dict.get(order_poi_id, {
                "wm_poi_name": f"poi_{order_poi_id}",
                "primary_first_tag_name": "unknown",
                "primary_second_tag_name": "unknown",
                "primary_third_tag_name": "unknown",
            })

            context = {
                "product_source": DEFAULT_PRODUCT_SOURCE,
                "product_id": str(order_poi_id),
                "item_title": poi_meta["wm_poi_name"],
                "item_desc": "",
                "item_price": float(price_midpoint),
                "external_item_title": "",
                "category_level1_name": poi_meta["primary_first_tag_name"],
                "category_level2_name": poi_meta["primary_second_tag_name"],
                "category_level3_name": poi_meta["primary_third_tag_name"],
                "industry_level1_name": DEFAULT_INDUSTRY_LEVEL1,
                "industry_level2_name": poi_meta["primary_first_tag_name"],
                "industry_level3_name": poi_meta["primary_second_tag_name"],
            }

            action = {
                "type": "purchase",
                "is_pay": True,
                "order_price_interval": price_interval,
                "order_amt": float(price_midpoint),
                "item_cnt": 1,
            }

            events_data.append({
                "event_id": f"{uid}:{src_idx}:{num_clicks}",
                "user_id": uid,
                "source_record_id": f"{uid}:{src_idx}",
                "source_record_index": src_idx,
                "action_index": num_clicks,
                "source_action_count": source_action_count,
                "source_timestamp": order_iso,
                "event_time": order_iso,
                "time_precision": "second",
                "time_lower_bound": order_dt,
                "time_upper_bound": order_dt,
                "upper_bound_exclusive": False,
                "eligibility_time": order_dt,
                "eligibility_inclusive": True,
                "scenario": DEFAULT_SCENARIO,
                "event_type": "purchase",
                "context_json": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                "action_json": json.dumps(action, ensure_ascii=False, separators=(",", ":")),
            })

        # Filter strictly <= CUTOFF_DT
        events_data = [e for e in events_data if e["time_upper_bound"] <= CUTOFF_DT]
        if not events_data:
            continue

        num_events = len(events_data)
        # Assign row_index
        for r_idx, ev in enumerate(events_data):
            ev["row_index"] = r_idx

        # Build PyArrow Table for Events
        ev_arrays = [
            pa.array([e["row_index"] for e in events_data], type=pa.int32()),
            pa.array([e["event_id"] for e in events_data], type=pa.string()),
            pa.array([e["user_id"] for e in events_data], type=pa.string()),
            pa.array([e["source_record_id"] for e in events_data], type=pa.string()),
            pa.array([e["source_record_index"] for e in events_data], type=pa.int32()),
            pa.array([e["action_index"] for e in events_data], type=pa.int32()),
            pa.array([e["source_action_count"] for e in events_data], type=pa.int32()),
            pa.array([e["source_timestamp"] for e in events_data], type=pa.string()),
            pa.array([e["event_time"] for e in events_data], type=pa.string()),
            pa.array([e["time_precision"] for e in events_data], type=pa.string()),
            pa.array([e["time_lower_bound"] for e in events_data], type=pa.timestamp("us")),
            pa.array([e["time_upper_bound"] for e in events_data], type=pa.timestamp("us")),
            pa.array([e["upper_bound_exclusive"] for e in events_data], type=pa.bool_()),
            pa.array([e["eligibility_time"] for e in events_data], type=pa.timestamp("us")),
            pa.array([e["eligibility_inclusive"] for e in events_data], type=pa.bool_()),
            pa.array([e["scenario"] for e in events_data], type=pa.string()),
            pa.array([e["event_type"] for e in events_data], type=pa.string()),
            pa.array([e["context_json"] for e in events_data], type=pa.large_string()),
            pa.array([e["action_json"] for e in events_data], type=pa.large_string()),
        ]
        events_table = pa.Table.from_arrays(ev_arrays, schema=EVENT_SCHEMA)
        events_file = events_dir / f"{uid}.parquet"
        pq.write_table(events_table, events_file, compression="zstd", compression_level=3)
        total_events_written += num_events

        # Snapshot construction
        if num_events > MAX_HISTORY_EVENTS:
            hist_start = num_events - MAX_HISTORY_EVENTS
            hist_stop = num_events
            hist_truncated = True
            hist_event_ids = [e["event_id"] for e in events_data[hist_start:hist_stop]]
        else:
            hist_start = 0
            hist_stop = num_events
            hist_truncated = False
            hist_event_ids = [e["event_id"] for e in events_data]

        profile = {
            "as_of_time": PRETRAIN_CUTOFF_TIME,
            "active_days_30d": len(unique_dates),
            "event_count_30d": num_events,
            "e_commerce_event_count_30d": num_events,
            "video_browsing_event_count_30d": 0,
            "live_streaming_event_count_30d": 0,
            "advertisement_event_count_30d": 0,
            "search_behavior_event_count_30d": 0,
            "customer_service_event_count_30d": 0,
        }

        snap_arrays = [
            pa.array([f"{uid}:{PRETRAIN_CUTOFF_TIME}:closed"], type=pa.string()),
            pa.array([uid], type=pa.string()),
            pa.array([CUTOFF_DT], type=pa.timestamp("us")),
            pa.array([True], type=pa.bool_()),
            pa.array([PRETRAIN_SPLIT], type=pa.string()),
            pa.array([hist_start], type=pa.int32()),
            pa.array([hist_stop], type=pa.int32()),
            pa.array([hist_event_ids], type=pa.list_(pa.field("element", pa.string()))),
            pa.array([True], type=pa.bool_()),
            pa.array([hist_truncated], type=pa.bool_()),
            pa.array([False], type=pa.bool_()),
            pa.array([json.dumps(profile, ensure_ascii=False, separators=(",", ":"))], type=pa.string()),
        ]
        snap_table = pa.Table.from_arrays(snap_arrays, schema=SNAPSHOT_SCHEMA)
        snap_file = snapshots_dir / f"{uid}.parquet"
        pq.write_table(snap_table, snap_file, compression="zstd", compression_level=3)
        total_snapshots_written += 1

    return total_events_written, total_snapshots_written


def main():
    logger.info("=== STEP 2: BUILD PRETRAIN DATASET ===")
    start_time = time.time()

    ensure_dir(PRETRAIN_EVENTS_DIR)
    ensure_dir(PRETRAIN_SNAPSHOTS_DIR)

    # 1. Load POIs
    poi_dict = load_pois()

    # 2. Load Sampled Pretrain Users
    pretrain_users_file = ARTIFACTS_DIR / "pretrain_users.txt"
    with pretrain_users_file.open("r", encoding="utf-8") as f:
        pretrain_users = [line.strip() for line in f if line.strip()]
    pretrain_users_set = set(pretrain_users)
    logger.info(f"Loaded {len(pretrain_users):,} pretrain users.")

    # 3. Read orders_train.txt for pretrain users
    logger.info("Filtering orders_train.txt for pretrain users...")
    train_orders_file = MEITUAN_RAW_DIR / "orders_train.txt"
    user_orders = defaultdict(list)
    needed_order_ids = set()

    with train_orders_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        u_idx = header.index("user_id")
        o_idx = header.index("wm_order_id")
        p_idx = header.index("wm_poi_id")
        pr_idx = header.index("order_price_interval")
        ts_idx = header.index("order_timestamp")
        dt_idx = header.index("dt")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            uid = parts[u_idx]
            if uid in pretrain_users_set:
                oid = parts[o_idx]
                ts = int(parts[ts_idx])
                # Check <= CUTOFF_EPOCH
                if ts <= CUTOFF_EPOCH:
                    user_orders[uid].append({
                        "wm_order_id": oid,
                        "wm_poi_id": parts[p_idx],
                        "order_price_interval": parts[pr_idx],
                        "order_timestamp": ts,
                        "dt": parts[dt_idx],
                    })
                    needed_order_ids.add(oid)

    logger.info(f"Retained {len(needed_order_ids):,} orders across {len(user_orders):,} users.")

    # 4. Read orders_poi_session.txt for needed order IDs
    logger.info("Reading orders_poi_session.txt for needed orders...")
    session_file = MEITUAN_RAW_DIR / "orders_poi_session.txt"
    order_sessions = {}

    with session_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        o_idx = header.index("wm_order_id")
        c_idx = header.index("clicks")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            oid = parts[o_idx]
            if oid in needed_order_ids:
                order_sessions[oid] = parts[c_idx]

    logger.info(f"Loaded {len(order_sessions):,} session records.")

    # 5. Process in parallel chunks
    chunk_size = 500
    chunks = [pretrain_users[i:i + chunk_size] for i in range(0, len(pretrain_users), chunk_size)]
    logger.info(f"Processing {len(pretrain_users):,} users across {len(chunks)} batches using ThreadPoolExecutor...")

    total_events = 0
    total_snapshots = 0

    max_workers = min(8, (os.cpu_count() or 4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_user_batch,
                chunk,
                user_orders,
                order_sessions,
                poi_dict,
                PRETRAIN_EVENTS_DIR,
                PRETRAIN_SNAPSHOTS_DIR,
            )
            for chunk in chunks
        ]

        for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            ev_count, snap_count = fut.result()
            total_events += ev_count
            total_snapshots += snap_count
            if idx % 10 == 0 or idx == len(chunks):
                logger.info(f"Completed {idx}/{len(chunks)} batches. Total events so far: {total_events:,}")

    elapsed = time.time() - start_time
    logger.info(
        f"✓ Done! Total events: {total_events:,}, Total snapshots: {total_snapshots:,} in {elapsed:.2f} seconds."
    )

    summary = {
        "total_users_processed": len(pretrain_users),
        "total_events_written": total_events,
        "total_snapshots_written": total_snapshots,
        "elapsed_seconds": elapsed,
        "events_dir": str(PRETRAIN_EVENTS_DIR),
        "snapshots_dir": str(PRETRAIN_SNAPSHOTS_DIR),
    }
    dump_json(summary, ARTIFACTS_DIR / "02_pretrain_build_summary.json")
    logger.info("Step 2 completed successfully!")


if __name__ == "__main__":
    main()
