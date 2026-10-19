"""Configuration and explicit readiness thresholds for Green SM simulation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")
PILOT_FRACTION = 0.02
TOTAL_DAYS = 42


def _require_split_counts(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(SPLITS):
        raise ValueError(f"{field} must contain exactly {list(SPLITS)}")
    parsed = {split: int(value[split]) for split in SPLITS}
    if any(count <= 0 for count in parsed.values()):
        raise ValueError(f"{field} counts must be positive")
    return parsed


def _unit_interval(value: object, *, field: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return parsed


@dataclass(frozen=True)
class ReadinessThresholds:
    """Full-corpus thresholds; ``for_fraction`` derives a pilot equivalent."""

    min_train_event_rows_per_event_type: int
    max_train_event_type_share: float
    min_train_valid_next_action_rate: float
    min_train_maskable_tokens: Mapping[str, int]
    min_valid_geo_rate: float
    max_non_user_oov_rate_validation_test: float
    cold_start_test_user_rate: float
    min_food_hard_negatives_per_request: int
    food_positive_rate_range: tuple[float, float]
    ads_click_rate_range: tuple[float, float]
    min_ads_ranking_eligible_rate: float
    max_control_auc_lift: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ReadinessThresholds:
        required = {
            "min_train_event_rows_per_event_type",
            "max_train_event_type_share",
            "min_train_valid_next_action_rate",
            "min_train_maskable_tokens",
            "min_valid_geo_rate",
            "max_non_user_oov_rate_validation_test",
            "cold_start_test_user_rate",
            "min_food_hard_negatives_per_request",
            "food_positive_rate_range",
            "ads_click_rate_range",
            "min_ads_ranking_eligible_rate",
            "max_control_auc_lift",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"thresholds missing required fields: {missing}")
        maskable = payload["min_train_maskable_tokens"]
        expected_modalities = {"categorical", "numeric", "text", "geo"}
        if not isinstance(maskable, Mapping) or set(maskable) != expected_modalities:
            raise ValueError("min_train_maskable_tokens must define categorical, numeric, text, and geo")
        maskable_counts = {key: int(value) for key, value in maskable.items()}
        if any(value <= 0 for value in maskable_counts.values()):
            raise ValueError("min_train_maskable_tokens values must be positive")

        def parse_range(field: str) -> tuple[float, float]:
            values = payload[field]
            if not isinstance(values, list | tuple) or len(values) != 2:
                raise ValueError(f"{field} must contain [minimum, maximum]")
            minimum = _unit_interval(values[0], field=f"{field}[0]")
            maximum = _unit_interval(values[1], field=f"{field}[1]")
            if minimum > maximum:
                raise ValueError(f"{field} minimum must not exceed maximum")
            return minimum, maximum

        event_rows = int(payload["min_train_event_rows_per_event_type"])
        hard_negatives = int(payload["min_food_hard_negatives_per_request"])
        if event_rows <= 0 or hard_negatives <= 0:
            raise ValueError("count thresholds must be positive")
        return cls(
            min_train_event_rows_per_event_type=event_rows,
            max_train_event_type_share=_unit_interval(
                payload["max_train_event_type_share"], field="max_train_event_type_share"
            ),
            min_train_valid_next_action_rate=_unit_interval(
                payload["min_train_valid_next_action_rate"], field="min_train_valid_next_action_rate"
            ),
            min_train_maskable_tokens=maskable_counts,
            min_valid_geo_rate=_unit_interval(payload["min_valid_geo_rate"], field="min_valid_geo_rate"),
            max_non_user_oov_rate_validation_test=_unit_interval(
                payload["max_non_user_oov_rate_validation_test"],
                field="max_non_user_oov_rate_validation_test",
            ),
            cold_start_test_user_rate=_unit_interval(
                payload["cold_start_test_user_rate"], field="cold_start_test_user_rate"
            ),
            min_food_hard_negatives_per_request=hard_negatives,
            food_positive_rate_range=parse_range("food_positive_rate_range"),
            ads_click_rate_range=parse_range("ads_click_rate_range"),
            min_ads_ranking_eligible_rate=_unit_interval(
                payload["min_ads_ranking_eligible_rate"], field="min_ads_ranking_eligible_rate"
            ),
            max_control_auc_lift=_unit_interval(
                payload["max_control_auc_lift"], field="max_control_auc_lift"
            ),
        )

    def for_fraction(self, fraction: float) -> ReadinessThresholds:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be within (0, 1]")
        return ReadinessThresholds(
            min_train_event_rows_per_event_type=max(
                100, round(self.min_train_event_rows_per_event_type * fraction)
            ),
            max_train_event_type_share=self.max_train_event_type_share,
            min_train_valid_next_action_rate=self.min_train_valid_next_action_rate,
            min_train_maskable_tokens={
                modality: max(100, round(count * fraction))
                for modality, count in self.min_train_maskable_tokens.items()
            },
            min_valid_geo_rate=self.min_valid_geo_rate,
            max_non_user_oov_rate_validation_test=self.max_non_user_oov_rate_validation_test,
            cold_start_test_user_rate=self.cold_start_test_user_rate,
            min_food_hard_negatives_per_request=self.min_food_hard_negatives_per_request,
            food_positive_rate_range=self.food_positive_rate_range,
            ads_click_rate_range=self.ads_click_rate_range,
            min_ads_ranking_eligible_rate=self.min_ads_ranking_eligible_rate,
            max_control_auc_lift=self.max_control_auc_lift,
        )


@dataclass(frozen=True)
class GreenSmConfig:
    """All deterministic inputs for one immutable Green SM corpus version."""

    data_version: str
    mode: str
    seed: int
    output_root: Path
    pilot_fraction: float
    user_count: int
    timezone: str
    total_days: int
    split_days: Mapping[str, int]
    snapshot_counts: Mapping[str, int]
    food_request_counts: Mapping[str, int]
    ads_request_counts: Mapping[str, int]
    max_history_events: int
    max_tokens: int
    behavior_policy: str
    thresholds: ReadinessThresholds

    @classmethod
    def from_json(cls, path: str | Path) -> GreenSmConfig:
        source = Path(path)
        payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
        required = {
            "data_version",
            "mode",
            "seed",
            "output_root",
            "pilot_fraction",
            "user_count",
            "timezone",
            "total_days",
            "split_days",
            "snapshot_counts",
            "food_request_counts",
            "ads_request_counts",
            "max_history_events",
            "max_tokens",
            "thresholds",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Green SM config missing required fields: {missing}")
        mode = str(payload["mode"])
        behavior_policy = str(payload.get("behavior_policy", "legacy_v1"))
        if behavior_policy not in {"legacy_v1", "shared_signal_poc_v1"}:
            raise ValueError("behavior_policy must be 'legacy_v1' or 'shared_signal_poc_v1'")
        pilot_fraction = float(payload["pilot_fraction"])
        if mode not in {"pilot", "full"}:
            raise ValueError("mode must be 'pilot' or 'full'")
        if mode == "pilot" and pilot_fraction != PILOT_FRACTION:
            raise ValueError(f"pilot_fraction must equal {PILOT_FRACTION} for pilot mode")
        if mode == "full" and pilot_fraction != 1.0:
            raise ValueError("pilot_fraction must equal 1.0 for full mode")
        total_days = int(payload["total_days"])
        split_days = _require_split_counts(payload["split_days"], field="split_days")
        if total_days != TOTAL_DAYS or sum(split_days.values()) != total_days:
            raise ValueError("split_days must sum to the 42 simulated days")
        output_root = Path(str(payload["output_root"]))
        user_count = int(payload["user_count"])
        max_history_events = int(payload["max_history_events"])
        max_tokens = int(payload["max_tokens"])
        if user_count <= 0 or max_history_events <= 0 or max_tokens <= 0:
            raise ValueError("user_count, max_history_events, and max_tokens must be positive")
        if str(payload["timezone"]) != "Asia/Ho_Chi_Minh":
            raise ValueError("timezone must be Asia/Ho_Chi_Minh")
        thresholds = ReadinessThresholds.from_mapping(payload["thresholds"])
        return cls(
            data_version=str(payload["data_version"]),
            mode=mode,
            seed=int(payload["seed"]),
            output_root=output_root,
            pilot_fraction=pilot_fraction,
            user_count=user_count,
            timezone="Asia/Ho_Chi_Minh",
            total_days=total_days,
            split_days=split_days,
            snapshot_counts=_require_split_counts(payload["snapshot_counts"], field="snapshot_counts"),
            food_request_counts=_require_split_counts(
                payload["food_request_counts"], field="food_request_counts"
            ),
            ads_request_counts=_require_split_counts(
                payload["ads_request_counts"], field="ads_request_counts"
            ),
            max_history_events=max_history_events,
            max_tokens=max_tokens,
            behavior_policy=behavior_policy,
            thresholds=thresholds.for_fraction(pilot_fraction),
        )

    @property
    def config_digest(self) -> str:
        canonical = {
            "data_version": self.data_version,
            "mode": self.mode,
            "seed": self.seed,
            "output_root": str(self.output_root),
            "pilot_fraction": self.pilot_fraction,
            "user_count": self.user_count,
            "timezone": self.timezone,
            "total_days": self.total_days,
            "split_days": dict(self.split_days),
            "snapshot_counts": dict(self.snapshot_counts),
            "food_request_counts": dict(self.food_request_counts),
            "ads_request_counts": dict(self.ads_request_counts),
            "max_history_events": self.max_history_events,
            "max_tokens": self.max_tokens,
            "behavior_policy": self.behavior_policy,
            "thresholds": {
                "min_train_event_rows_per_event_type": self.thresholds.min_train_event_rows_per_event_type,
                "max_train_event_type_share": self.thresholds.max_train_event_type_share,
                "min_train_valid_next_action_rate": self.thresholds.min_train_valid_next_action_rate,
                "min_train_maskable_tokens": dict(self.thresholds.min_train_maskable_tokens),
                "min_valid_geo_rate": self.thresholds.min_valid_geo_rate,
                "max_non_user_oov_rate_validation_test": self.thresholds.max_non_user_oov_rate_validation_test,
                "cold_start_test_user_rate": self.thresholds.cold_start_test_user_rate,
                "min_food_hard_negatives_per_request": self.thresholds.min_food_hard_negatives_per_request,
                "food_positive_rate_range": self.thresholds.food_positive_rate_range,
                "ads_click_rate_range": self.thresholds.ads_click_rate_range,
                "min_ads_ranking_eligible_rate": self.thresholds.min_ads_ranking_eligible_rate,
                "max_control_auc_lift": self.thresholds.max_control_auc_lift,
            },
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
