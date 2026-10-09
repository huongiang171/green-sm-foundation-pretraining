"""Vi du xay dung PyTorch Dataset va DataLoader cho bo du lieu OmniBehavior Pretrain v2.

Chuyen doi cac ban ghi snapshot thanh cac tensor san sang dua vao mo hinh User Foundation Model:
1. Tabular Profile features: Vector numerical dac trung 8 chieu duoc chuan hoa.
2. History Event Sequence: Chuoi toi da 30 events voi event_type_ids, scenario_ids, attention_mask.
3. User ID: Mapping user thanh integer index cho User ID Adapter.

Cach chay thu nghiem:
    python torch_dataset_loader.py --split train --batch-size 4 --limit 20
"""

import argparse
import io
import json
import math
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

from pretraining_dataset import iter_pretraining_samples

try:
    import torch
    from torch.utils.data import IterableDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Danh muc 6 Scenario va cac Event Type pho bien
SCENARIO_VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1,
    "Video Browsing": 2,
    "Live Streaming": 3,
    "E-commerce": 4,
    "Search Behavior": 5,
    "Advertisement": 6,
    "Customer Service": 7,
}

EVENT_TYPE_VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1,
    "watch": 2,
    "like": 3,
    "comment": 4,
    "purchase": 5,
    "cart": 6,
    "search": 7,
    "show": 8,
    "follow": 9,
    "conversion": 10,
    "collect": 11,
    "click": 12,
    "click_cart": 13,
    "share": 14,
    "dislike": 15,
    "send_gift": 16,
    "activation": 17,
    "dialogue": 18,
    "download": 19,
    "unfollow": 20,
    "join_group": 21,
    "submit": 22,
    "report": 23,
}

MAX_SEQ_LEN = 30


def process_sample_to_features(sample: dict) -> dict:
    """Bien doi mot ban ghi snapshot tho thanh cac feature san sang tao tensor."""
    inp = sample.get("input", {})
    user_id_str = inp.get("user_id", "user_000")
    profile = inp.get("profile", {})
    history = inp.get("history", [])

    # Trich xuat numerical user profile features (8 chieu)
    # Su dung log1p de giam skewness cua cac bien dem
    profile_features = [
        math.log1p(float(profile.get("active_days_30d", 0))),
        math.log1p(float(profile.get("event_count_30d", 0))),
        math.log1p(float(profile.get("video_browsing_event_count_30d", 0))),
        math.log1p(float(profile.get("live_streaming_event_count_30d", 0))),
        math.log1p(float(profile.get("e_commerce_event_count_30d", 0))),
        math.log1p(float(profile.get("advertisement_event_count_30d", 0))),
        math.log1p(float(profile.get("search_behavior_event_count_30d", 0))),
        math.log1p(float(profile.get("customer_service_event_count_30d", 0))),
    ]

    # Trich xuat chuoi sequence events (padding toi da MAX_SEQ_LEN = 30)
    seq_len = min(len(history), MAX_SEQ_LEN)
    recent_events = history[-seq_len:] if history else []

    scenario_ids = [0] * MAX_SEQ_LEN
    event_type_ids = [0] * MAX_SEQ_LEN
    attention_mask = [0] * MAX_SEQ_LEN

    for idx, ev in enumerate(recent_events):
        sc_name = ev.get("scenario", "")
        et_name = ev.get("event_type", "")
        scenario_ids[idx] = SCENARIO_VOCAB.get(sc_name, SCENARIO_VOCAB["<UNK>"])
        event_type_ids[idx] = EVENT_TYPE_VOCAB.get(et_name, EVENT_TYPE_VOCAB["<UNK>"])
        attention_mask[idx] = 1

    return {
        "sample_id": sample.get("sample_id", ""),
        "user_id": user_id_str,
        "profile_vector": profile_features,
        "scenario_ids": scenario_ids,
        "event_type_ids": event_type_ids,
        "attention_mask": attention_mask,
        "seq_len": seq_len,
    }


if HAS_TORCH:
    class OmniBehaviorIterableDataset(IterableDataset):
        """Streaming Dataset cho phep duyet bo du lieu lon ma khong can load het vao RAM."""
        def __init__(self, dataset_dir: Path | str, split: str = "train", limit: int | None = None):
            self.dataset_dir = Path(dataset_dir)
            self.split = split
            self.limit = limit

        def __iter__(self):
            count = 0
            for sample in iter_pretraining_samples(self.dataset_dir, split=self.split):
                feat = process_sample_to_features(sample)
                yield {
                    "sample_id": feat["sample_id"],
                    "user_id": feat["user_id"],
                    "profile_vector": torch.tensor(feat["profile_vector"], dtype=torch.float32),
                    "scenario_ids": torch.tensor(feat["scenario_ids"], dtype=torch.long),
                    "event_type_ids": torch.tensor(feat["event_type_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(feat["attention_mask"], dtype=torch.float32),
                    "seq_len": torch.tensor(feat["seq_len"], dtype=torch.long),
                }
                count += 1
                if self.limit is not None and count >= self.limit:
                    break


def main():
    parser = argparse.ArgumentParser(description="Chay thu PyTorch DataLoader tren tap OmniBehavior Pretrain")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train", help="Logical split")
    parser.add_argument("--batch-size", type=int, default=4, help="Kich thuoc batch (mac dinh: 4)")
    parser.add_argument("--limit", type=int, default=8, help="Gioi han so mau chay thu (mac dinh: 8)")
    args = parser.parse_args()

    print("=" * 80)
    print("DEMO PYTORCH DATA LOADER CHO USER FOUNDATION MODEL PRETRAIN")
    print("=" * 80)

    if not HAS_TORCH:
        print("[CANH BAO] Moi truong hien tai chua cai pytorch. Demo bang Python thuong:")
        count = 0
        for sample in iter_pretraining_samples(DATA_DIR, split=args.split):
            feat = process_sample_to_features(sample)
            print(f"\nMau #{count+1} - ID: {feat['sample_id']}")
            print(f"  • User ID        : {feat['user_id']}")
            print(f"  • Profile Vector : {feat['profile_vector']}")
            print(f"  • Scenario IDs   : {feat['scenario_ids'][:feat['seq_len']]}")
            print(f"  • Event Type IDs : {feat['event_type_ids'][:feat['seq_len']]}")
            print(f"  • Mask Sum       : {sum(feat['attention_mask'])}")
            count += 1
            if count >= args.limit:
                break
        return

    print(f"Dang khoi tao Streaming Dataset tu '{DATA_DIR}' (split={args.split})...")
    ds = OmniBehaviorIterableDataset(DATA_DIR, split=args.split, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size)

    print(f"Bat dau duyet cac Batch voi batch_size = {args.batch_size}:\n")
    for b_idx, batch in enumerate(loader, 1):
        print(f"📦 [Batch #{b_idx}]")
        print(f"   • profile_vector.shape : {list(batch['profile_vector'].shape)}  -> [Batch, 8]")
        print(f"   • scenario_ids.shape   : {list(batch['scenario_ids'].shape)}  -> [Batch, 30]")
        print(f"   • event_type_ids.shape : {list(batch['event_type_ids'].shape)}  -> [Batch, 30]")
        print(f"   • attention_mask.shape : {list(batch['attention_mask'].shape)}  -> [Batch, 30]")
        print(f"   • Sample IDs trong batch: {batch['sample_id'][:2]} ...")
        print(f"   • seq_lens trong batch  : {batch['seq_len'].tolist()}")
        print("-" * 60)

    print("\n[OK] Chay DataLoader thanh cong!")


if __name__ == "__main__":
    main()
