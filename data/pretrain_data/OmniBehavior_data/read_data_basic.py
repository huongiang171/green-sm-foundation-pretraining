"""Script doc du lieu mau co ban tu bo du lieu OmniBehavior Pretraining v2.

Su dung module bundled pretraining_dataset.py de tai cac ban ghi snapshot da duoc
tai tao day du theo cac logical split (train, validation, test).

Cach su dung:
    python read_data_basic.py --split train --num-samples 2
    python read_data_basic.py --split test --num-samples 1 --user-id user_001
    python read_data_basic.py --num-samples 5 --export-json samples_demo.json
"""

import argparse
import io
import json
import sys
from pathlib import Path

# Thiet lap UTF-8 output cho console tren Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from pretraining_dataset import iter_pretraining_samples, SPLITS


def print_sample_details(idx: int, sample: dict):
    """In chi tiet mot ban ghi snapshot voi dinh dang de doc."""
    sample_id = sample.get("sample_id", "N/A")
    inp = sample.get("input", {})
    user_id = inp.get("user_id", "N/A")
    cutoff_time = inp.get("cutoff_time", "N/A")
    cutoff_inclusive = inp.get("cutoff_inclusive", False)
    history_order_known = inp.get("history_order_known", False)
    history_truncated = inp.get("history_truncated", False)
    profile = inp.get("profile", {})
    history = inp.get("history", [])

    print(f"\n{'='*80}")
    print(f"[SAMPLE #{idx+1}] ID: {sample_id}")
    print(f"{'='*80}")
    print(f"User ID               : {user_id}")
    print(f"Cutoff Time           : {cutoff_time} (inclusive: {cutoff_inclusive})")
    print(f"History Order Known   : {history_order_known} | Truncated (>30): {history_truncated}")
    print(f"So luong events       : {len(history)} / 30 (max)")

    print("\n--- Ho so nguoi dung 30 ngay (Profile) ---")
    print(f"   * as_of_time                      : {profile.get('as_of_time')}")
    print(f"   * active_days_30d                 : {profile.get('active_days_30d')}")
    print(f"   * event_count_30d (tong cong)     : {profile.get('event_count_30d')}")
    print(f"   * video_browsing_event_count_30d  : {profile.get('video_browsing_event_count_30d')}")
    print(f"   * live_streaming_event_count_30d  : {profile.get('live_streaming_event_count_30d')}")
    print(f"   * e_commerce_event_count_30d      : {profile.get('e_commerce_event_count_30d')}")
    print(f"   * advertisement_event_count_30d   : {profile.get('advertisement_event_count_30d')}")
    print(f"   * search_behavior_event_count_30d : {profile.get('search_behavior_event_count_30d')}")
    print(f"   * customer_service_event_count_30d: {profile.get('customer_service_event_count_30d')}")

    print(f"\n--- Chuoi lich su tuong tac gan nhat ({len(history)} events) ---")
    if not history:
        print("   (Lich su trong)")
    else:
        events_to_show = history if len(history) <= 6 else (history[:3] + [{"__ellipsis__": True}] + history[-3:])
        for e_idx, event in enumerate(events_to_show):
            if event.get("__ellipsis__"):
                print(f"   ... [An {len(history) - 6} events o giua] ...")
                continue
            e_id = event.get("event_id")
            time_str = event.get("event_time")
            scenario = event.get("scenario")
            e_type = event.get("event_type")
            ctx = event.get("context")
            action = event.get("action")
            
            ctx_summary = []
            if ctx:
                for k, v in ctx.items():
                    if v is not None and v != "":
                        v_str = str(v)
                        if len(v_str) > 30:
                            v_str = v_str[:27] + "..."
                        ctx_summary.append(f"{k}='{v_str}'")
            ctx_str = ", ".join(ctx_summary) if ctx_summary else "none"

            print(f"   [{e_idx+1:02d}] {time_str} | {scenario:<15} | {e_type:<12}")
            print(f"        └─ ID     : {e_id}")
            print(f"        └─ Context: {ctx_str}")
            print(f"        └─ Action : {json.dumps(action, ensure_ascii=False)}")


def main():
    parser = argparse.ArgumentParser(description="Doc du lieu mau tu bo du lieu tien huan luyen OmniBehavior")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="train",
        help="Logical split can doc: train (mac dinh), validation, test hoac all",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="So luong mau snapshot muon in ra (mac dinh: 2)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Loc theo user_id cu the (vi du: user_001)",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Duong dan file JSON de xuat du lieu cac mau da trich xuat (optional)",
    )

    args = parser.parse_args()
    split_arg = None if args.split == "all" else args.split

    print(f"==> Bat dau doc du lieu tu: {DATA_DIR}")
    print(f"==> Split: {args.split.upper()} | So mau yeu cau: {args.num_samples}")
    if args.user_id:
        print(f"==> Bo loc user: {args.user_id}")

    samples = []
    count = 0
    iterator = iter_pretraining_samples(DATA_DIR, split=split_arg)

    for sample in iterator:
        uid = sample.get("input", {}).get("user_id")
        if args.user_id and uid != args.user_id:
            continue

        print_sample_details(count, sample)
        samples.append(sample)
        count += 1
        if count >= args.num_samples:
            break

    print(f"\n[OK] Da doc thanh cong {len(samples)} mau.")

    if args.export_json:
        out_path = Path(args.export_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"[OK] Da luu {len(samples)} mau vao file: {out_path.resolve()}")


if __name__ == "__main__":
    main()
