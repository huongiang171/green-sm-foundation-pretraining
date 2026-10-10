"""Utility functions for data loading, parsing, serialization, and hashing."""

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def epoch_to_iso(epoch_seconds: int) -> str:
    """Converts unix epoch seconds to local ISO-8601 string: YYYY-MM-DDTHH:MM:SS."""
    dt = datetime.fromtimestamp(epoch_seconds)
    return dt.isoformat(timespec="seconds")

def epoch_to_datetime(epoch_seconds: int) -> datetime:
    """Converts unix epoch seconds to datetime object."""
    return datetime.fromtimestamp(epoch_seconds)

def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()

def dump_json(obj: Any, filepath: Path, indent: int = 2) -> None:
    """Dumps an object to a formatted JSON file."""
    ensure_dir(filepath.parent)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def load_json(filepath: Path) -> Any:
    """Loads an object from a JSON file."""
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)

def clean_tag(val: Optional[str]) -> str:
    """Cleans Chinese tag strings (e.g. 未知 -> unknown)."""
    if val is None or val == "" or val == "NULL" or val == "未知":
        return "unknown"
    return str(val).strip()

def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts value to float."""
    if val is None or val == "" or val == "NULL":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val: Any, default: int = 0) -> int:
    """Safely converts value to int."""
    if val is None or val == "" or val == "NULL":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default
