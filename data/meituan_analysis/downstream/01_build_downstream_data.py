"""Step 5: Build Downstream Tabular Ranking Dataset (Week 4: 2021-03-22 to 2021-03-28).

Processes 10,000 warm-start downstream users:
- Positive ordered sample: (clicked=1, ordered=1)
- Positive clicked samples: (clicked=1, ordered=0)
- Negative candidate POIs (4 per session, same AOR): (clicked=0, ordered=0)
- Splits into train.parquet (22-25/03) and test.parquet (26-28/03)
- Writes candidate_pois.parquet, data_report.json, preview.json into downstream1_meituan_data/
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meituan_analysis.common.config import (
    MEITUAN_RAW_DIR,
    ARTIFACTS_DIR,
    DOWNSTREAM_OUTPUT_DIR,
    DOWNSTREAM_TRAIN_DTS,
    DOWNSTREAM_TEST_DTS,
    RANDOM_SEED,
    PRICE_INTERVAL_TO_CAT_ID,
)
from meituan_analysis.common.utils import (
    get_logger,
    ensure_dir,
    dump_json,
    clean_tag,
    safe_float,
    safe_int,
)

logger = get_logger("01_build_downstream_data")

DOWNSTREAM_SCHEMA = pa.schema([
    ("user_id", pa.string()),
    ("wm_order_id", pa.string()),
    ("wm_poi_id", pa.string()),
    ("user_avg_pay_amt", pa.int32()),
    ("primary_first_tag", pa.int32()),
    ("primary_second_tag", pa.int32()),
    ("primary_third_tag", pa.string()),
    ("poi_score", pa.float32()),
    ("food_comment_score", pa.float32()),
    ("delivery_comment_score", pa.float32()),
    ("poi_brand_id", pa.int32()),
    ("delivery_fee", pa.float32()),
    ("est_delivery_min", pa.int32()),
    ("aor_id", pa.int32()),
    ("aoi_id", pa.string()),
    ("ord_period_name", pa.int32()),
    ("local_hour", pa.int32()),
    ("takedlvr_aoi_type_name", pa.string()),
    ("clicked", pa.int32()),
    ("ordered", pa.int32()),
    ("dt", pa.string()),
])

CANDIDATE_POI_SCHEMA = pa.schema([
    ("wm_poi_id", pa.string()),
    ("wm_poi_name", pa.string()),
    ("aor_id", pa.int32()),
    ("primary_first_tag", pa.int32()),
    ("primary_second_tag", pa.int32()),
    ("primary_third_tag", pa.string()),
    ("poi_score", pa.float32()),
    ("food_comment_score", pa.float32()),
    ("delivery_comment_score", pa.float32()),
    ("poi_brand_id", pa.int32()),
])

def load_user_demographics() -> Dict[str, int]:
    """Loads user_avg_pay_amt from users.txt."""
    logger.info("Loading users.txt demographics...")
    users_file = MEITUAN_RAW_DIR / "users.txt"
    user_pay_map = {}
    with users_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        u_idx = header.index("user_id")
        p_idx = header.index("avg_pay_amt")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            uid = parts[u_idx]
            pval = parts[p_idx]
            cat_id = PRICE_INTERVAL_TO_CAT_ID.get(pval, 0)
            user_pay_map[uid] = cat_id
    logger.info(f"Loaded pay interval for {len(user_pay_map):,} users.")
    return user_pay_map

def load_pois_and_categories() -> Tuple[Dict[str, dict], Dict[int, List[str]], Dict[str, int], Dict[str, int]]:
    """Loads POIs, maps tag categories to integer codes, and groups by AOR."""
    logger.info("Loading pois.txt...")
    pois_file = MEITUAN_RAW_DIR / "pois.txt"
    poi_dict = {}
    pois_by_aor = defaultdict(list)

    tag1_to_id = {}
    tag2_to_id = {}

    with pois_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        p_id_idx = header.index("wm_poi_id")
        p_name_idx = header.index("wm_poi_name")
        tag2_idx = header.index("primary_second_tag_name")
        tag3_idx = header.index("primary_third_tag_name")
        tag1_idx = header.index("primary_first_tag_name")
        brand_idx = header.index("poi_brand_id")
        aor_idx = header.index("aor_id")
        score_idx = header.index("poi_score")
        deliv_score_idx = header.index("delivery_comment_avg_score")
        food_score_idx = header.index("food_comment_avg_score")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            pid = parts[p_id_idx]
            t1 = clean_tag(parts[tag1_idx])
            t2 = clean_tag(parts[tag2_idx])
            t3 = clean_tag(parts[tag3_idx])
            aor = safe_int(parts[aor_idx], default=0)

            if t1 not in tag1_to_id:
                tag1_to_id[t1] = len(tag1_to_id)
            if t2 not in tag2_to_id:
                tag2_to_id[t2] = len(tag2_to_id)

            poi_info = {
                "wm_poi_id": pid,
                "wm_poi_name": parts[p_name_idx],
                "aor_id": aor,
                "primary_first_tag": tag1_to_id[t1],
                "primary_second_tag": tag2_to_id[t2],
                "primary_third_tag": t3,
                "poi_score": safe_float(parts[score_idx], default=4.5),
                "delivery_comment_score": safe_float(parts[deliv_score_idx], default=4.8),
                "food_comment_score": safe_float(parts[food_score_idx], default=4.6),
                "poi_brand_id": safe_int(parts[brand_idx], default=0),
            }
            poi_dict[pid] = poi_info
            pois_by_aor[aor].append(pid)

    logger.info(f"Loaded {len(poi_dict):,} POIs across {len(pois_by_aor)} AORs.")
    return poi_dict, pois_by_aor, tag1_to_id, tag2_to_id

def main():
    logger.info("=== STEP 5: BUILD DOWNSTREAM DATASET ===")
    start_time = time.time()
    ensure_dir(DOWNSTREAM_OUTPUT_DIR)

    # 1. Load 10,000 warm-start downstream users
    downstream_users_file = ARTIFACTS_DIR / "downstream_users.txt"
    with downstream_users_file.open("r", encoding="utf-8") as f:
        downstream_users_set = set(line.strip() for line in f if line.strip())
    logger.info(f"Loaded {len(downstream_users_set):,} downstream user targets.")

    # 2. Load User Demographics & POIs
    user_pay_map = load_user_demographics()
    poi_dict, pois_by_aor, tag1_to_id, tag2_to_id = load_pois_and_categories()
    all_poi_ids = list(poi_dict.keys())

    # 3. Save candidate_pois.parquet
    logger.info("Writing candidate_pois.parquet...")
    cand_poi_rows = list(poi_dict.values())
    cand_arrays = [
        pa.array([r["wm_poi_id"] for r in cand_poi_rows], type=pa.string()),
        pa.array([r["wm_poi_name"] for r in cand_poi_rows], type=pa.string()),
        pa.array([r["aor_id"] for r in cand_poi_rows], type=pa.int32()),
        pa.array([r["primary_first_tag"] for r in cand_poi_rows], type=pa.int32()),
        pa.array([r["primary_second_tag"] for r in cand_poi_rows], type=pa.int32()),
        pa.array([r["primary_third_tag"] for r in cand_poi_rows], type=pa.string()),
        pa.array([r["poi_score"] for r in cand_poi_rows], type=pa.float32()),
        pa.array([r["food_comment_score"] for r in cand_poi_rows], type=pa.float32()),
        pa.array([r["delivery_comment_score"] for r in cand_poi_rows], type=pa.float32()),
        pa.array([r["poi_brand_id"] for r in cand_poi_rows], type=pa.int32()),
    ]
    cand_table = pa.Table.from_arrays(cand_arrays, schema=CANDIDATE_POI_SCHEMA)
    pq.write_table(cand_table, DOWNSTREAM_OUTPUT_DIR / "candidate_pois.parquet", compression="zstd", compression_level=3)
    logger.info(f"Saved candidate_pois.parquet with {len(cand_table):,} rows.")

    # 4. Load week 4 labels: (user_id, wm_order_id) -> wm_poi_id
    logger.info("Reading orders_poi_test_label.txt...")
    test_label_file = MEITUAN_RAW_DIR / "orders_poi_test_label.txt"
    order_labels = {}
    with test_label_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        u_idx = header.index("user_id")
        o_idx = header.index("wm_order_id")
        p_idx = header.index("wm_poi_id")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            uid = parts[u_idx]
            if uid in downstream_users_set:
                oid = parts[o_idx]
                pid = parts[p_idx]
                order_labels[(uid, oid)] = pid

    logger.info(f"Loaded {len(order_labels):,} labels for downstream users.")

    # 5. Load week 4 orders from orders_test_poi.txt
    logger.info("Reading orders_test_poi.txt...")
    test_orders_file = MEITUAN_RAW_DIR / "orders_test_poi.txt"
    downstream_orders = []
    needed_week4_order_ids = set()

    with test_orders_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        u_idx = header.index("user_id")
        o_idx = header.index("wm_order_id")
        aor_idx = header.index("aor_id")
        ts_idx = header.index("order_timestamp")
        per_idx = header.index("ord_period_name")
        aoi_idx = header.index("aoi_id")
        type_idx = header.index("takedlvr_aoi_type_name")
        dt_idx = header.index("dt")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            uid = parts[u_idx]
            if uid in downstream_users_set:
                oid = parts[o_idx]
                downstream_orders.append({
                    "user_id": uid,
                    "wm_order_id": oid,
                    "aor_id": safe_int(parts[aor_idx], default=0),
                    "order_timestamp": int(parts[ts_idx]),
                    "ord_period_name": safe_int(parts[per_idx], default=0),
                    "aoi_id": parts[aoi_idx],
                    "takedlvr_aoi_type_name": clean_tag(parts[type_idx]),
                    "dt": parts[dt_idx],
                })
                needed_week4_order_ids.add(oid)

    logger.info(f"Loaded {len(downstream_orders):,} orders in Week 4 for downstream users.")

    # 6. Load clicks from orders_poi_session.txt for week 4
    logger.info("Reading orders_poi_session.txt for Week 4 clicks...")
    session_file = MEITUAN_RAW_DIR / "orders_poi_session.txt"
    week4_order_clicks = {}

    with session_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        o_idx = header.index("wm_order_id")
        c_idx = header.index("clicks")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            oid = parts[o_idx]
            if oid in needed_week4_order_ids:
                c_str = parts[c_idx]
                if c_str != "NULL" and c_str:
                    week4_order_clicks[oid] = [c for c in c_str.split("#") if c and c != "NULL"]
                else:
                    week4_order_clicks[oid] = []

    logger.info(f"Loaded clicks for {len(week4_order_clicks):,} Week 4 sessions.")

    # 7. Construct samples: Positive & Negative Sampling (1:4)
    rng = np.random.default_rng(RANDOM_SEED)

    train_rows = []
    test_rows = []

    total_pos_orders = 0
    total_pos_clicks = 0
    total_neg_samples = 0

    default_poi_meta = {
        "primary_first_tag": 0,
        "primary_second_tag": 0,
        "primary_third_tag": "unknown",
        "poi_score": 4.5,
        "food_comment_score": 4.6,
        "delivery_comment_score": 4.8,
        "poi_brand_id": 0,
        "aor_id": 0,
    }

    for order in downstream_orders:
        uid = order["user_id"]
        oid = order["wm_order_id"]
        order_aor = order["aor_id"]
        dt = order["dt"]
        order_ts = order["order_timestamp"]
        local_hour = datetime.fromtimestamp(order_ts).hour
        user_pay = user_pay_map.get(uid, 0)

        ordered_poi = order_labels.get((uid, oid))
        if not ordered_poi:
            continue

        session_clicks = week4_order_clicks.get(oid, [])
        interacted_pois = set(session_clicks)
        interacted_pois.add(ordered_poi)

        # Helper to create row
        def make_row(cand_poi_id: str, is_clicked: int, is_ordered: int) -> dict:
            p_meta = poi_dict.get(cand_poi_id, default_poi_meta)
            is_same_aor = (p_meta["aor_id"] == order_aor)
            delivery_fee = 15000.0 if is_same_aor else 25000.0
            est_min = 20 + (safe_int(cand_poi_id, 0) % 15)

            return {
                "user_id": uid,
                "wm_order_id": oid,
                "wm_poi_id": str(cand_poi_id),
                "user_avg_pay_amt": user_pay,
                "primary_first_tag": p_meta["primary_first_tag"],
                "primary_second_tag": p_meta["primary_second_tag"],
                "primary_third_tag": p_meta["primary_third_tag"],
                "poi_score": float(p_meta["poi_score"]),
                "food_comment_score": float(p_meta["food_comment_score"]),
                "delivery_comment_score": float(p_meta["delivery_comment_score"]),
                "poi_brand_id": p_meta["poi_brand_id"],
                "delivery_fee": float(delivery_fee),
                "est_delivery_min": int(est_min),
                "aor_id": order_aor,
                "aoi_id": order["aoi_id"],
                "ord_period_name": order["ord_period_name"],
                "local_hour": int(local_hour),
                "takedlvr_aoi_type_name": order["takedlvr_aoi_type_name"],
                "clicked": is_clicked,
                "ordered": is_ordered,
                "dt": dt,
            }

        session_rows = []

        # (A) Positive Ordered Sample: clicked=1, ordered=1
        session_rows.append(make_row(ordered_poi, 1, 1))
        total_pos_orders += 1

        # (B) Positive Click Samples (clicked but not ordered): clicked=1, ordered=0
        for clk_poi in session_clicks:
            if clk_poi != ordered_poi:
                session_rows.append(make_row(clk_poi, 1, 0))
                total_pos_clicks += 1

        # (C) Negative Samples: 4 candidates from same AOR that user did not interact with
        aor_pool = pois_by_aor.get(order_aor, [])
        candidate_negatives = [p for p in aor_pool if p not in interacted_pois]
        num_neg_needed = 4

        if len(candidate_negatives) >= num_neg_needed:
            selected_negatives = rng.choice(candidate_negatives, size=num_neg_needed, replace=False)
        else:
            # Fallback to global pool if district has very few POIs
            fallback_pool = [p for p in all_poi_ids if p not in interacted_pois]
            selected_negatives = rng.choice(fallback_pool, size=num_neg_needed, replace=False)

        for neg_poi in selected_negatives:
            session_rows.append(make_row(neg_poi, 0, 0))
            total_neg_samples += 1

        # Partition by dt
        if dt in DOWNSTREAM_TRAIN_DTS:
            train_rows.extend(session_rows)
        elif dt in DOWNSTREAM_TEST_DTS:
            test_rows.extend(session_rows)

    logger.info(
        f"Generated samples: {total_pos_orders:,} ordered, {total_pos_clicks:,} clicks, "
        f"{total_neg_samples:,} negatives. (Total rows: {len(train_rows) + len(test_rows):,})"
    )

    def write_parquet(rows: list, filepath: Path):
        arrays = [
            pa.array([r["user_id"] for r in rows], type=pa.string()),
            pa.array([r["wm_order_id"] for r in rows], type=pa.string()),
            pa.array([r["wm_poi_id"] for r in rows], type=pa.string()),
            pa.array([r["user_avg_pay_amt"] for r in rows], type=pa.int32()),
            pa.array([r["primary_first_tag"] for r in rows], type=pa.int32()),
            pa.array([r["primary_second_tag"] for r in rows], type=pa.int32()),
            pa.array([r["primary_third_tag"] for r in rows], type=pa.string()),
            pa.array([r["poi_score"] for r in rows], type=pa.float32()),
            pa.array([r["food_comment_score"] for r in rows], type=pa.float32()),
            pa.array([r["delivery_comment_score"] for r in rows], type=pa.float32()),
            pa.array([r["poi_brand_id"] for r in rows], type=pa.int32()),
            pa.array([r["delivery_fee"] for r in rows], type=pa.float32()),
            pa.array([r["est_delivery_min"] for r in rows], type=pa.int32()),
            pa.array([r["aor_id"] for r in rows], type=pa.int32()),
            pa.array([r["aoi_id"] for r in rows], type=pa.string()),
            pa.array([r["ord_period_name"] for r in rows], type=pa.int32()),
            pa.array([r["local_hour"] for r in rows], type=pa.int32()),
            pa.array([r["takedlvr_aoi_type_name"] for r in rows], type=pa.string()),
            pa.array([r["clicked"] for r in rows], type=pa.int32()),
            pa.array([r["ordered"] for r in rows], type=pa.int32()),
            pa.array([r["dt"] for r in rows], type=pa.string()),
        ]
        tbl = pa.Table.from_arrays(arrays, schema=DOWNSTREAM_SCHEMA)
        pq.write_table(tbl, filepath, compression="zstd", compression_level=3)
        return len(tbl)

    train_path = DOWNSTREAM_OUTPUT_DIR / "train.parquet"
    test_path = DOWNSTREAM_OUTPUT_DIR / "test.parquet"

    logger.info(f"Writing train.parquet ({len(train_rows):,} rows)...")
    write_parquet(train_rows, train_path)

    logger.info(f"Writing test.parquet ({len(test_rows):,} rows)...")
    write_parquet(test_rows, test_path)

    # Preview & Reports
    preview_file = DOWNSTREAM_OUTPUT_DIR / "preview.json"
    dump_json(train_rows[:5], preview_file)

    report_data = {
        "dataset_name": "Meituan TRD Downstream Ranking Dataset",
        "users_count": len(downstream_users_set),
        "total_train_rows": len(train_rows),
        "total_test_rows": len(test_rows),
        "total_rows": len(train_rows) + len(test_rows),
        "positive_ordered_count": total_pos_orders,
        "positive_clicked_count": total_pos_clicks,
        "negative_samples_count": total_neg_samples,
        "train_dts": sorted(list(DOWNSTREAM_TRAIN_DTS)),
        "test_dts": sorted(list(DOWNSTREAM_TEST_DTS)),
    }
    report_file = DOWNSTREAM_OUTPUT_DIR / "data_report.json"
    dump_json(report_data, report_file)
    logger.info(f"Downstream dataset built in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
