"""Step 1: Sample and filter users for Pretrain and Downstream tasks.

Logic:
1. Map order_id -> user_id and count orders in 21 days (orders_train.txt).
2. Count click events from orders_poi_session.txt for train period orders.
3. Calculate total events per user (orders + clicks).
4. Filter users with >= 3 events (Inactivity Filter).
5. Sample 20,000 users for Pretrain using seed=42.
6. Identify users from the 20k who have orders in Week 4 (orders_test_poi.txt).
7. Sample 10,000 warm-start users for Downstream using seed=42.
8. Persist pretrain_users.txt, downstream_users.txt, and sampling metrics.
"""

import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meituan_analysis.common.config import (
    MEITUAN_RAW_DIR,
    ARTIFACTS_DIR,
    RANDOM_SEED,
    PRETRAIN_USER_COUNT,
    DOWNSTREAM_USER_COUNT,
)
from meituan_analysis.common.utils import get_logger, ensure_dir, dump_json

logger = get_logger("01_sample_and_filter")

def main():
    logger.info("=== STEP 1: SAMPLING & FILTERING USERS ===")
    ensure_dir(ARTIFACTS_DIR)

    # 1. Read orders_train.txt
    logger.info("Reading orders_train.txt...")
    train_orders_file = MEITUAN_RAW_DIR / "orders_train.txt"
    order_to_user = {}
    user_order_counts = defaultdict(int)
    all_train_users = set()

    with train_orders_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        u_idx = header.index("user_id")
        o_idx = header.index("wm_order_id")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            uid = parts[u_idx]
            oid = parts[o_idx]
            order_to_user[oid] = uid
            user_order_counts[uid] += 1
            all_train_users.add(uid)

    logger.info(f"Loaded {len(order_to_user):,} train orders across {len(all_train_users):,} users.")

    # 2. Read orders_poi_session.txt for clicks in train period
    logger.info("Reading orders_poi_session.txt...")
    session_file = MEITUAN_RAW_DIR / "orders_poi_session.txt"
    user_click_counts = defaultdict(int)

    with session_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        o_idx = header.index("wm_order_id")
        c_idx = header.index("clicks")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            oid = parts[o_idx]
            clicks_str = parts[c_idx]
            if oid in order_to_user:
                uid = order_to_user[oid]
                if clicks_str != "NULL" and clicks_str:
                    clicks_list = clicks_str.split("#")
                    user_click_counts[uid] += len(clicks_list)

    # 3. Calculate total events and apply Inactivity Filter (>= 3 events)
    eligible_users = []
    for uid in sorted(all_train_users):
        total_events = user_order_counts[uid] + user_click_counts[uid]
        if total_events >= 3:
            eligible_users.append(uid)

    logger.info(
        f"Eligible users (>= 3 events in train period): {len(eligible_users):,} / {len(all_train_users):,} "
        f"({len(eligible_users) / len(all_train_users) * 100:.2f}%)"
    )

    # 4. Sample 20,000 users for Pretrain
    rng = np.random.default_rng(RANDOM_SEED)
    eligible_users_arr = np.array(eligible_users)
    pretrain_users_arr = rng.choice(eligible_users_arr, size=PRETRAIN_USER_COUNT, replace=False)
    pretrain_users_set = set(pretrain_users_arr)
    logger.info(f"Sampled {len(pretrain_users_set):,} users for Pretrain Pool.")

    # 5. Read orders_test_poi.txt to find users with orders in Week 4
    logger.info("Reading orders_test_poi.txt to find Week 4 active users...")
    test_orders_file = MEITUAN_RAW_DIR / "orders_test_poi.txt"
    week4_users = set()

    with test_orders_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        u_idx = header.index("user_id")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            week4_users.add(parts[u_idx])

    logger.info(f"Total unique users in Week 4: {len(week4_users):,}")

    # 6. Warm-start intersection (Pretrain 20k users who also exist in Week 4)
    warmstart_candidates = sorted(list(pretrain_users_set.intersection(week4_users)))
    logger.info(
        f"Warm-start candidate users (intersection of 20k pretrain & week 4): {len(warmstart_candidates):,}"
    )

    if len(warmstart_candidates) < DOWNSTREAM_USER_COUNT:
        raise ValueError(
            f"Not enough warmstart candidates ({len(warmstart_candidates)}) to sample {DOWNSTREAM_USER_COUNT} users!"
        )

    # 7. Sample 10,000 warm-start downstream users
    downstream_users_arr = rng.choice(np.array(warmstart_candidates), size=DOWNSTREAM_USER_COUNT, replace=False)
    downstream_users_set = set(downstream_users_arr)
    logger.info(f"Sampled {len(downstream_users_set):,} warm-start users for Downstream.")

    # 8. Save user IDs to files
    pretrain_users_file = ARTIFACTS_DIR / "pretrain_users.txt"
    with pretrain_users_file.open("w", encoding="utf-8") as f:
        for uid in sorted(pretrain_users_set):
            f.write(f"{uid}\n")

    downstream_users_file = ARTIFACTS_DIR / "downstream_users.txt"
    with downstream_users_file.open("w", encoding="utf-8") as f:
        for uid in sorted(downstream_users_set):
            f.write(f"{uid}\n")

    report = {
        "random_seed": RANDOM_SEED,
        "total_train_users": len(all_train_users),
        "eligible_train_users_ge_3": len(eligible_users),
        "sampled_pretrain_users": len(pretrain_users_set),
        "total_week4_users": len(week4_users),
        "warmstart_candidate_pool": len(warmstart_candidates),
        "sampled_downstream_users": len(downstream_users_set),
        "downstream_is_subset_of_pretrain": downstream_users_set.issubset(pretrain_users_set),
    }

    report_file = ARTIFACTS_DIR / "01_sampling_report.json"
    dump_json(report, report_file)
    logger.info(f"Sampling report saved to: {report_file}")
    logger.info(f"Pretrain users saved to: {pretrain_users_file}")
    logger.info(f"Downstream users saved to: {downstream_users_file}")
    logger.info("Step 1 completed successfully!")

if __name__ == "__main__":
    main()
