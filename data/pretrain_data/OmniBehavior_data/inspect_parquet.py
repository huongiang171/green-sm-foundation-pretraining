"""Script kiem tra va phan tich truc tiep cac file Parquet trong events/ va snapshots/.

Doc truc tiep bang pyarrow.parquet ma khong can qua generator make_sample,
giup kiem tra nhanh schema, metadata, toc do doc va kich thuoc luu tru tren dia.

Cach su dung:
    python inspect_parquet.py --overview
    python inspect_parquet.py --user user_001 --table events --limit 5
    python inspect_parquet.py --user user_001 --table snapshots --limit 3
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
import pyarrow.parquet as pq

# Thiet lap UTF-8 output cho console tren Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent
EVENTS_DIR = DATA_DIR / "events"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"


def print_overview():
    """Thong ke tong quan ve so file, dung luong va so dong trong parquet."""
    print("=" * 80)
    print("TONG QUAN HE THONG FILE PARQUET (events/ va snapshots/)")
    print("=" * 80)

    event_files = sorted(list(EVENTS_DIR.glob("*.parquet")))
    snapshot_files = sorted(list(SNAPSHOTS_DIR.glob("*.parquet")))

    print(f"Thu muc Events    : {len(event_files)} files parquet")
    print(f"Thu muc Snapshots : {len(snapshot_files)} files parquet")

    # Tinh dung luong
    total_events_bytes = sum(f.stat().st_size for f in event_files)
    total_snapshots_bytes = sum(f.stat().st_size for f in snapshot_files)
    print(f"Dung luong Events   : {total_events_bytes / (1024*1024):.2f} MB")
    print(f"Dung luong Snapshots: {total_snapshots_bytes / (1024*1024):.2f} MB")
    print(f"Tong dung luong     : {(total_events_bytes + total_snapshots_bytes) / (1024*1024):.2f} MB")

    # Doc metadata mau tu user_001
    sample_file = EVENTS_DIR / "user_001.parquet"
    if sample_file.exists():
        meta = pq.read_metadata(sample_file)
        print(f"\n[Events Parquet - Mau user_001.parquet]")
        print(f"  • So dong (rows)    : {meta.num_rows}")
        print(f"  • So cot (columns) : {meta.num_columns}")
        print(f"  • Compression      : {meta.row_group(0).column(0).compression}")
        print("  • Schema cac cot:")
        schema = pq.read_schema(sample_file)
        for name, type_ in zip(schema.names, schema.types):
            print(f"      - {name:<22}: {type_}")

    sample_snap = SNAPSHOTS_DIR / "user_001.parquet"
    if sample_snap.exists():
        meta_s = pq.read_metadata(sample_snap)
        print(f"\n[Snapshots Parquet - Mau user_001.parquet]")
        print(f"  • So dong (rows)    : {meta_s.num_rows}")
        print(f"  • So cot (columns) : {meta_s.num_columns}")
        print(f"  • Compression      : {meta_s.row_group(0).column(0).compression}")
        print("  • Schema cac cot:")
        schema_s = pq.read_schema(sample_snap)
        for name, type_ in zip(schema_s.names, schema_s.types):
            print(f"      - {name:<22}: {type_}")


def inspect_user_table(user_id: str, table_type: str, limit: int = 5):
    """In cac dong du lieu chi tiet cua mot user tu file parquet."""
    dir_path = EVENTS_DIR if table_type == "events" else SNAPSHOTS_DIR
    file_path = dir_path / f"{user_id}.parquet"

    if not file_path.exists():
        print(f"[LOI] Khong tim thay file: {file_path}")
        return

    print("=" * 80)
    print(f"CHI TIET BANG '{table_type.upper()}' CUA USER: {user_id}")
    print(f"File: {file_path.name} | Kich thuoc: {file_path.stat().st_size / 1024:.1f} KB")
    print("=" * 80)

    table = pq.read_table(file_path)
    total_rows = len(table)
    print(f"Tong so ban ghi (rows): {total_rows}")
    print(f"Hien thi {min(limit, total_rows)} ban ghi dau tien:\n")

    pydict = table.slice(0, limit).to_pylist()
    for idx, row in enumerate(pydict, 1):
        print(f"--- [Dong #{idx}] ---")
        for k, v in row.items():
            if k in ("context_json", "action_json", "profile_json") and isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    print(f"  {k:<22}: {json.dumps(parsed, ensure_ascii=False)[:120]}...")
                except Exception:
                    print(f"  {k:<22}: {str(v)[:100]}")
            else:
                print(f"  {k:<22}: {v}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Kiem tra truc tiep file Parquet trong bo du lieu")
    parser.add_argument("--overview", action="store_true", help="Hien thi thong tin tong quan ve toan bo thu muc")
    parser.add_argument("--user", type=str, default="user_001", help="User ID can kiem tra (mac dinh: user_001)")
    parser.add_argument("--table", choices=["events", "snapshots"], default="events", help="Bang muon xem: events hoac snapshots")
    parser.add_argument("--limit", type=int, default=3, help="So dong muon xem chi tiet (mac dinh: 3)")

    args = parser.parse_args()

    if args.overview or len(sys.argv) == 1:
        print_overview()
    else:
        inspect_user_table(args.user, args.table, args.limit)


if __name__ == "__main__":
    main()
