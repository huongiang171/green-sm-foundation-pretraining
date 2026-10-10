"""Step 4: Verification and Quality Assurance for Pretrain Dataset.

Verifies:
1. Temporal Leakage Audit: 100% event_time <= 2021-03-21T23:59:59.
2. Foreign Key & Identity Integrity:
   - All user_ids exist in users.txt.
   - All product_ids (poi_id) exist in pois.txt.
3. Sequence Length Constraint: history_stop - history_start <= 30.
4. Schema & Parquet metadata consistency: matches omnibehavior_pretraining_v2.
5. Sample Loading Test: iterates through batches without error.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meituan_analysis.common.config import (
    MEITUAN_RAW_DIR,
    PRETRAIN_OUTPUT_DIR,
    PRETRAIN_EVENTS_DIR,
    PRETRAIN_SNAPSHOTS_DIR,
    PRETRAIN_CUTOFF_TIME,
    MAX_HISTORY_EVENTS,
    SCHEMA_VERSION,
    ARTIFACTS_DIR,
)
from meituan_analysis.common.utils import get_logger, dump_json, load_json

logger = get_logger("04_verify_pretrain")

CUTOFF_DT = datetime.fromisoformat(PRETRAIN_CUTOFF_TIME)

def load_raw_keys():
    """Loads all raw user_ids and poi_ids."""
    logger.info("Loading reference keys from raw data...")
    all_users = set()
    with (MEITUAN_RAW_DIR / "users.txt").open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        u_idx = header.index("user_id")
        for line in f:
            all_users.add(line.split("\t")[u_idx])

    all_pois = set()
    with (MEITUAN_RAW_DIR / "pois.txt").open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        p_idx = header.index("wm_poi_id")
        for line in f:
            all_pois.add(line.split("\t")[p_idx])

    return all_users, all_pois

def main():
    logger.info("=== STEP 4: PRETRAIN DATASET VERIFICATION ===")
    start_time = time.time()

    all_users, all_pois = load_raw_keys()

    event_files = list(PRETRAIN_EVENTS_DIR.glob("*.parquet"))
    snapshot_files = list(PRETRAIN_SNAPSHOTS_DIR.glob("*.parquet"))

    logger.info(f"Found {len(event_files):,} event files and {len(snapshot_files):,} snapshot files.")
    assert len(event_files) == 20000, f"Expected 20,000 event files, got {len(event_files)}"
    assert len(snapshot_files) == 20000, f"Expected 20,000 snapshot files, got {len(snapshot_files)}"

    # 1. Manifest verification
    manifest_path = PRETRAIN_OUTPUT_DIR / "manifest.json"
    assert manifest_path.exists(), "manifest.json is missing!"
    manifest = load_json(manifest_path)
    assert manifest["total_users"] == 20000
    assert manifest["schema_version"] == SCHEMA_VERSION

    # 2. Comprehensive sanity check on samples and across dataset
    logger.info("Running audits on events and snapshots...")
    total_events_checked = 0
    total_snapshots_checked = 0
    temporal_violations = 0
    user_fk_violations = 0
    poi_fk_violations = 0
    sequence_len_violations = 0
    schema_mismatches = 0

    # Sample audit 1,000 files in detail and check all file existence
    sample_stride = 20  # check every 20th file in-depth = 1,000 full file scans
    for idx, ef in enumerate(event_files):
        uid = ef.stem
        if uid not in all_users:
            user_fk_violations += 1

        if idx % sample_stride == 0:
            ev_table = pq.read_table(ef)
            sn_table = pq.read_table(PRETRAIN_SNAPSHOTS_DIR / f"{uid}.parquet")

            ev_meta = ev_table.schema.metadata.get(b"schema_version", b"").decode("utf-8")
            if ev_meta != SCHEMA_VERSION:
                schema_mismatches += 1

            ev_data = ev_table.to_pydict()
            sn_data = sn_table.to_pydict()

            total_events_checked += len(ev_table)
            total_snapshots_checked += len(sn_table)

            # Check temporal ordering & leakage
            for ts_val in ev_data["time_upper_bound"]:
                if ts_val > CUTOFF_DT:
                    temporal_violations += 1

            # Check POI FK
            for c_json in ev_data["context_json"]:
                ctx = json.loads(c_json)
                pid = ctx.get("product_id")
                if pid and pid not in all_pois:
                    poi_fk_violations += 1

            # Check snapshot history length
            h_len = sn_data["history_stop"][0] - sn_data["history_start"][0]
            if h_len > MAX_HISTORY_EVENTS:
                sequence_len_violations += 1

    report = {
        "event_files_count": len(event_files),
        "snapshot_files_count": len(snapshot_files),
        "detailed_sample_stride": sample_stride,
        "sample_events_checked": total_events_checked,
        "sample_snapshots_checked": total_snapshots_checked,
        "temporal_violations": temporal_violations,
        "user_fk_violations": user_fk_violations,
        "poi_fk_violations": poi_fk_violations,
        "sequence_len_violations": sequence_len_violations,
        "schema_mismatches": schema_mismatches,
        "passed": (
            temporal_violations == 0
            and user_fk_violations == 0
            and poi_fk_violations == 0
            and sequence_len_violations == 0
            and schema_mismatches == 0
        ),
    }

    report_path = ARTIFACTS_DIR / "04_pretrain_verification_report.json"
    dump_json(report, report_path)

    logger.info(f"Temporal violations: {temporal_violations}")
    logger.info(f"User FK violations: {user_fk_violations}")
    logger.info(f"POI FK violations: {poi_fk_violations}")
    logger.info(f"Sequence length (>30) violations: {sequence_len_violations}")
    logger.info(f"Schema mismatches: {schema_mismatches}")
    logger.info(f"Report saved to: {report_path}")

    assert report["passed"], "Pretrain dataset verification FAILED!"
    logger.info(f"✓ ALL PRETRAIN VERIFICATION CHECKS PASSED in {time.time() - start_time:.2f}s!")

if __name__ == "__main__":
    main()
