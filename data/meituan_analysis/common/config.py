"""Common configuration and paths for Meituan data processing pipeline."""

from pathlib import Path

# Base Paths
WORKSPACE_ROOT = Path("d:/Meituan")
MEITUAN_RAW_DIR = WORKSPACE_ROOT / "meituan_data"
ANALYSIS_DIR = WORKSPACE_ROOT / "meituan_analysis"
ARTIFACTS_DIR = ANALYSIS_DIR / "artifacts"

# Target Output Directories
PRETRAIN_OUTPUT_DIR = WORKSPACE_ROOT / "pretrain_data" / "meituan_data"
PRETRAIN_EVENTS_DIR = PRETRAIN_OUTPUT_DIR / "events"
PRETRAIN_SNAPSHOTS_DIR = PRETRAIN_OUTPUT_DIR / "snapshots"

DOWNSTREAM_OUTPUT_DIR = WORKSPACE_ROOT / "downstream1_meituan_data"

# Random Seed
RANDOM_SEED = 42

# Sampling Targets
PRETRAIN_USER_COUNT = 20_000
DOWNSTREAM_USER_COUNT = 10_000

# Temporal Cutoffs
# Pretrain: 2021-03-01 to 2021-03-21
PRETRAIN_START_TIME = "2021-03-01T00:00:00"
PRETRAIN_CUTOFF_TIME = "2021-03-21T23:59:59"
PRETRAIN_SPLIT = "train"

# Downstream: 2021-03-22 to 2021-03-28
# Train split: 20210322 to 20210325
# Test split:  20210326 to 20210328
DOWNSTREAM_TRAIN_DTS = {"20210322", "20210323", "20210324", "20210325"}
DOWNSTREAM_TEST_DTS = {"20210326", "20210327", "20210328"}

# Model / Pretrain Constraints
MAX_HISTORY_EVENTS = 30
SCHEMA_VERSION = "omnibehavior_pretraining_v2"
DEFAULT_SCENARIO = "E-commerce"
DEFAULT_PRODUCT_SOURCE = "meituan_takeout"
DEFAULT_INDUSTRY_LEVEL1 = "Food Delivery"

# Price Interval Mappings
PRICE_INTERVAL_TO_CAT_ID = {
    "<29": 0,
    "[29,36)": 1,
    "[36,49)": 2,
    "[49,65)": 3,
    ">=65": 4,
}

PRICE_INTERVAL_TO_MIDPOINT = {
    "<29": 20.0,
    "[29,36)": 32.5,
    "[36,49)": 42.5,
    "[49,65)": 57.0,
    ">=65": 75.0,
}
