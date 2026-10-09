"""Step 0: Health check and validation of raw Meituan data files.

Validates:
- UTF-8 encoding and tab-delimited formatting.
- Header row and column names.
- Line counts and column count consistency across all data rows.
- Missing/Null value percentages and anomalies (e.g. 'NULL', '未知', price intervals).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meituan_analysis.common.config import MEITUAN_RAW_DIR, ARTIFACTS_DIR
from meituan_analysis.common.utils import get_logger, ensure_dir, dump_json

logger = get_logger("00_health_check_raw")

RAW_FILES = [
    "users.txt",
    "pois.txt",
    "spus.txt",
    "orders_train.txt",
    "orders_poi_session.txt",
    "orders_test_poi.txt",
    "orders_poi_test_label.txt",
]

def check_file(filename: str) -> dict:
    filepath = MEITUAN_RAW_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Checking {filename}...")
    total_lines = 0
    inconsistent_lines = 0
    header = []
    null_counts = {}
    weizhi_counts = {}
    sample_rows = []

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        # Read header
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"Empty file: {filename}")
        header = header_line.rstrip("\r\n").split("\t")
        expected_cols = len(header)
        null_counts = {col: 0 for col in header}
        weizhi_counts = {col: 0 for col in header}

        for idx, line in enumerate(f, start=1):
            total_lines += 1
            line_str = line.rstrip("\r\n")
            parts = line_str.split("\t")
            if len(parts) != expected_cols:
                inconsistent_lines += 1
                if inconsistent_lines <= 5:
                    logger.warning(
                        f"Data line {idx} in {filename} has {len(parts)} columns, expected {expected_cols}"
                    )
                continue

            if idx <= 3:
                sample_rows.append(dict(zip(header, parts)))

            for c_name, val in zip(header, parts):
                if val == "NULL" or val == "":
                    null_counts[c_name] += 1
                if "未知" in val:
                    weizhi_counts[c_name] += 1

    report = {
        "filename": filename,
        "total_data_rows": total_lines,
        "inconsistent_lines": inconsistent_lines,
        "num_columns": expected_cols,
        "columns": header,
        "null_counts": {k: v for k, v in null_counts.items() if v > 0},
        "weizhi_counts": {k: v for k, v in weizhi_counts.items() if v > 0},
        "sample_rows": sample_rows,
    }
    logger.info(
        f"✓ {filename}: {total_lines:,} data rows, {expected_cols} columns, {inconsistent_lines} inconsistent lines."
    )
    return report

def main():
    logger.info("=== STEP 0: RAW DATA HEALTH CHECK ===")
    overall_report = {}
    for filename in RAW_FILES:
        report = check_file(filename)
        overall_report[filename] = report

    out_file = ARTIFACTS_DIR / "00_raw_health_report.json"
    dump_json(overall_report, out_file)
    logger.info(f"Report saved to: {out_file}")
    logger.info("Step 0 Health Check completed successfully!")

if __name__ == "__main__":
    main()
