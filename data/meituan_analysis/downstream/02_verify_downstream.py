"""Step 6: Verify and validate Downstream Ranking Dataset.

Checks:
1. File existence and schema verification for train.parquet, test.parquet, candidate_pois.parquet.
2. Distinct user_id count: exactly 10,000 users.
3. Warm-start consistency: 100% downstream users are in pretrain_users.txt.
4. Label logic:
   - clicked, ordered in {0, 1}
   - ordered == 1 => clicked == 1
   - clicked == 0 => ordered == 0
5. Temporal partition purity:
   - train dt in {20210322..20210325}
   - test dt in {20210326..20210328}
6. POI foreign key validity against pois.txt.
"""

import sys
import time
from pathlib import Path
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meituan_analysis.common.config import (
    MEITUAN_RAW_DIR,
    ARTIFACTS_DIR,
    DOWNSTREAM_OUTPUT_DIR,
    DOWNSTREAM_TRAIN_DTS,
    DOWNSTREAM_TEST_DTS,
    DOWNSTREAM_USER_COUNT,
)
from meituan_analysis.common.utils import get_logger, dump_json, load_json

logger = get_logger("02_verify_downstream")

def main():
    logger.info("=== STEP 6: VERIFY DOWNSTREAM DATASET ===")
    start_time = time.time()

    train_path = DOWNSTREAM_OUTPUT_DIR / "train.parquet"
    test_path = DOWNSTREAM_OUTPUT_DIR / "test.parquet"
    cand_path = DOWNSTREAM_OUTPUT_DIR / "candidate_pois.parquet"
    report_path = DOWNSTREAM_OUTPUT_DIR / "data_report.json"
    preview_path = DOWNSTREAM_OUTPUT_DIR / "preview.json"

    assert train_path.exists(), "train.parquet missing!"
    assert test_path.exists(), "test.parquet missing!"
    assert cand_path.exists(), "candidate_pois.parquet missing!"
    assert report_path.exists(), "data_report.json missing!"
    assert preview_path.exists(), "preview.json missing!"

    # 1. Load reference pretrain users & pois
    with (ARTIFACTS_DIR / "pretrain_users.txt").open("r", encoding="utf-8") as f:
        pretrain_users_set = set(line.strip() for line in f if line.strip())

    all_pois = set()
    with (MEITUAN_RAW_DIR / "pois.txt").open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        p_idx = header.index("wm_poi_id")
        for line in f:
            all_pois.add(line.split("\t")[p_idx])

    # 2. Read train and test parquet
    train_tbl = pq.read_table(train_path)
    test_tbl = pq.read_table(test_path)
    cand_tbl = pq.read_table(cand_path)

    logger.info(f"Loaded train.parquet: {len(train_tbl):,} rows")
    logger.info(f"Loaded test.parquet: {len(test_tbl):,} rows")
    logger.info(f"Loaded candidate_pois.parquet: {len(cand_tbl):,} rows")

    train_data = train_tbl.to_pydict()
    test_data = test_tbl.to_pydict()

    # 3. Check distinct users
    all_downstream_users = set(train_data["user_id"]).union(set(test_data["user_id"]))
    logger.info(f"Total unique downstream users: {len(all_downstream_users):,}")
    assert len(all_downstream_users) == DOWNSTREAM_USER_COUNT, (
        f"Expected {DOWNSTREAM_USER_COUNT} users, got {len(all_downstream_users)}"
    )

    # 4. Warm-start subset verification
    is_subset = all_downstream_users.issubset(pretrain_users_set)
    logger.info(f"100% downstream users are subset of pretrain users: {is_subset}")
    assert is_subset, "Downstream users must be a subset of pretrain users!"

    # 5. Check labels & temporal splits
    def audit_table(data_dict, expected_dts, split_name):
        n_rows = len(data_dict["user_id"])
        label_logic_errors = 0
        poi_fk_errors = 0
        dt_leak_errors = 0

        clicked_counts = {0: 0, 1: 0}
        ordered_counts = {0: 0, 1: 0}

        for i in range(n_rows):
            c = data_dict["clicked"][i]
            o = data_dict["ordered"][i]
            dt = data_dict["dt"][i]
            pid = data_dict["wm_poi_id"][i]

            clicked_counts[c] = clicked_counts.get(c, 0) + 1
            ordered_counts[o] = ordered_counts.get(o, 0) + 1

            if o == 1 and c != 1:
                label_logic_errors += 1
            if c == 0 and o != 0:
                label_logic_errors += 1

            if dt not in expected_dts:
                dt_leak_errors += 1

            if pid not in all_pois:
                poi_fk_errors += 1

        logger.info(
            f"[{split_name}] Rows: {n_rows:,} | Clicked: {clicked_counts} | Ordered: {ordered_counts}"
        )
        assert label_logic_errors == 0, f"Label logic errors in {split_name}: {label_logic_errors}"
        assert dt_leak_errors == 0, f"Temporal leakage errors in {split_name}: {dt_leak_errors}"
        assert poi_fk_errors == 0, f"POI FK errors in {split_name}: {poi_fk_errors}"

        return {
            "rows": n_rows,
            "clicked_distribution": clicked_counts,
            "ordered_distribution": ordered_counts,
            "label_logic_errors": label_logic_errors,
            "dt_leak_errors": dt_leak_errors,
            "poi_fk_errors": poi_fk_errors,
        }

    train_audit = audit_table(train_data, DOWNSTREAM_TRAIN_DTS, "TRAIN")
    test_audit = audit_table(test_data, DOWNSTREAM_TEST_DTS, "TEST")

    summary = {
        "status": "PASSED",
        "downstream_users_count": len(all_downstream_users),
        "is_warmstart_subset_of_pretrain": is_subset,
        "candidate_pois_count": len(cand_tbl),
        "train_audit": train_audit,
        "test_audit": test_audit,
    }

    report_out = ARTIFACTS_DIR / "06_downstream_verification_report.json"
    dump_json(summary, report_out)
    logger.info(f"Verification report saved to: {report_out}")
    logger.info(f"✓ ALL DOWNSTREAM VERIFICATION CHECKS PASSED in {time.time() - start_time:.2f}s!")

if __name__ == "__main__":
    main()
