"""Deterministic, public target-world construction for the Green SM simulator."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

from .config import GreenSmConfig


TZ = ZoneInfo("Asia/Ho_Chi_Minh")
SIMULATION_START = datetime(2026, 9, 1, tzinfo=TZ)
ZONE_TYPES = (
    "residential",
    "office",
    "mall",
    "airport",
    "university",
    "nightlife",
    "hospital",
    "transit",
)
FOOD_CATEGORIES = (
    "noodles",
    "rice",
    "coffee",
    "tea",
    "bakery",
    "vegetarian",
    "seafood",
    "grill",
    "dessert",
    "western",
    "healthy",
    "snacks",
)


def _rng(seed: int, *parts: object) -> random.Random:
    text = ":".join([str(seed), *(str(part) for part in parts)])
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return random.Random(value)


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported value in public payload: {type(value)!r}")


@dataclass(frozen=True)
class GeoCell:
    cell_id: str
    zone_type: str
    lat: float
    lon: float

    def public_payload(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "zone_type": self.zone_type,
            "lat": self.lat,
            "lon": self.lon,
        }


@dataclass(frozen=True)
class CatalogueEntity:
    entity_id: str
    category: str
    geo_cell: str
    lat: float
    lon: float
    available_at: datetime
    price: int | None = None
    merchant_id: str | None = None

    def public_payload(self) -> dict[str, object]:
        result: dict[str, object] = {
            "entity_id": self.entity_id,
            "category": self.category,
            "geo_cell": self.geo_cell,
            "lat": self.lat,
            "lon": self.lon,
            "available_at": self.available_at,
        }
        if self.price is not None:
            result["price"] = self.price
        if self.merchant_id is not None:
            result["merchant_id"] = self.merchant_id
        return result


@dataclass(frozen=True)
class Catalogue:
    cells: tuple[GeoCell, ...]
    food_merchants: tuple[CatalogueEntity, ...]
    food_items: tuple[CatalogueEntity, ...]
    delivery_pois: tuple[CatalogueEntity, ...]
    ad_campaigns: tuple[CatalogueEntity, ...]

    def cell(self, cell_id: str) -> GeoCell:
        return next(cell for cell in self.cells if cell.cell_id == cell_id)

    def public_payload(self) -> dict[str, object]:
        return {
            "cells": [cell.public_payload() for cell in self.cells],
            "food_merchants": [entry.public_payload() for entry in self.food_merchants],
            "food_items": [entry.public_payload() for entry in self.food_items],
            "delivery_pois": [entry.public_payload() for entry in self.delivery_pois],
            "ad_campaigns": [entry.public_payload() for entry in self.ad_campaigns],
        }


@dataclass(frozen=True)
class SimulatedUser:
    user_id: str
    home_cell: str
    work_cell: str
    first_active_day: int
    budget_bucket: str
    preferred_service: str

    def public_payload(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "home_cell": self.home_cell,
            "work_cell": self.work_cell,
            "first_active_day": self.first_active_day,
            "budget_bucket": self.budget_bucket,
            "preferred_service": self.preferred_service,
        }


@dataclass(frozen=True)
class LogicalCandidate:
    candidate_id: str
    category: str
    geo_cell: str
    lat: float
    lon: float
    available_at: datetime
    label: int
    is_hard_negative: bool = False
    price: int | None = None
    merchant_id: str | None = None
    placement: str | None = None

    @property
    def is_positive(self) -> bool:
        return bool(self.label)

    def public_payload(self) -> dict[str, object]:
        result: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "category": self.category,
            "geo_cell": self.geo_cell,
            "lat": self.lat,
            "lon": self.lon,
            "available_at": self.available_at,
            "label": self.label,
            "is_hard_negative": self.is_hard_negative,
        }
        if self.price is not None:
            result["price"] = self.price
        if self.merchant_id is not None:
            result["merchant_id"] = self.merchant_id
        if self.placement is not None:
            result["placement"] = self.placement
        return result


@dataclass(frozen=True)
class LogicalRequest:
    request_id: str
    user_id: str
    cutoff_time: datetime
    split: str
    service: str
    request_cell: str
    candidates: tuple[LogicalCandidate, ...]

    def public_payload(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "cutoff_time": self.cutoff_time,
            "split": self.split,
            "service": self.service,
            "request_cell": self.request_cell,
            "candidates": [candidate.public_payload() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class TargetWorld:
    catalogue: Catalogue
    users: tuple[SimulatedUser, ...]
    food_requests: tuple[LogicalRequest, ...]
    ads_requests: tuple[LogicalRequest, ...]
    cold_start_user_ids: frozenset[str]

    def all_requests(self) -> tuple[LogicalRequest, ...]:
        return self.food_requests + self.ads_requests

    def cold_start_users(self) -> tuple[SimulatedUser, ...]:
        return tuple(user for user in self.users if user.user_id in self.cold_start_user_ids)

    def public_payload(self) -> dict[str, object]:
        return {
            "catalogue": self.catalogue.public_payload(),
            "users": [user.public_payload() for user in self.users],
            "food_requests": [request.public_payload() for request in self.food_requests],
            "ads_requests": [request.public_payload() for request in self.ads_requests],
            "cold_start_user_ids": sorted(self.cold_start_user_ids),
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.public_payload(), default=_json_default, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def food_positive_rate(self) -> float:
        labels = [candidate.label for request in self.food_requests for candidate in request.candidates]
        return sum(labels) / len(labels)

    def ads_click_rate(self) -> float:
        labels = [candidate.label for request in self.ads_requests for candidate in request.candidates]
        return sum(labels) / len(labels)

    def ads_ranking_eligible_rate(self) -> float:
        return sum(any(candidate.label for candidate in request.candidates) for request in self.ads_requests) / len(
            self.ads_requests
        )


def _coordinate(cell_index: int, *, offset: int) -> tuple[float, float]:
    row, column = divmod(cell_index, 8)
    return 10.70 + row * 0.0125 + offset * 0.0001, 106.60 + column * 0.0125 + offset * 0.0001


def _build_catalogue(config: GreenSmConfig) -> Catalogue:
    available_at = SIMULATION_START - timedelta(seconds=1)
    cells = tuple(
        GeoCell(
            cell_id=f"cell_{index:02d}",
            zone_type=ZONE_TYPES[index % len(ZONE_TYPES)],
            lat=_coordinate(index, offset=0)[0],
            lon=_coordinate(index, offset=0)[1],
        )
        for index in range(64)
    )
    merchants: list[CatalogueEntity] = []
    for index in range(160):
        cell_index = index % 64
        lat, lon = _coordinate(cell_index, offset=(index % 9) + 1)
        merchants.append(
            CatalogueEntity(
                entity_id=f"merchant_{index:03d}",
                category=FOOD_CATEGORIES[index % len(FOOD_CATEGORIES)],
                geo_cell=f"cell_{cell_index:02d}",
                lat=lat,
                lon=lon,
                available_at=available_at,
            )
        )
    items = tuple(
        CatalogueEntity(
            entity_id=f"food_item_{index:03d}",
            category=merchants[index % len(merchants)].category,
            geo_cell=merchants[index % len(merchants)].geo_cell,
            lat=merchants[index % len(merchants)].lat,
            lon=merchants[index % len(merchants)].lon,
            available_at=available_at,
            price=35_000 + (index % 9) * 15_000,
            merchant_id=merchants[index % len(merchants)].entity_id,
        )
        for index in range(480)
    )
    delivery_pois = tuple(
        CatalogueEntity(
            entity_id=f"delivery_poi_{index:03d}",
            category="pickup" if index % 2 == 0 else "dropoff",
            geo_cell=f"cell_{index % 64:02d}",
            lat=_coordinate(index % 64, offset=(index % 7) + 1)[0],
            lon=_coordinate(index % 64, offset=(index % 7) + 1)[1],
            available_at=available_at,
        )
        for index in range(120)
    )
    campaigns = tuple(
        CatalogueEntity(
            entity_id=f"campaign_{index:03d}",
            category=FOOD_CATEGORIES[index % len(FOOD_CATEGORIES)],
            geo_cell=f"cell_{index % 64:02d}",
            lat=_coordinate(index % 64, offset=(index % 5) + 1)[0],
            lon=_coordinate(index % 64, offset=(index % 5) + 1)[1],
            available_at=available_at,
        )
        for index in range(120)
    )
    return Catalogue(cells=cells, food_merchants=tuple(merchants), food_items=items, delivery_pois=delivery_pois, ad_campaigns=campaigns)


def _build_users(config: GreenSmConfig) -> tuple[tuple[SimulatedUser, ...], frozenset[str]]:
    cold_count = round(config.user_count * config.thresholds.cold_start_test_user_rate)
    cold_indices = set(range(config.user_count - cold_count + 1, config.user_count + 1))
    users: list[SimulatedUser] = []
    for index in range(1, config.user_count + 1):
        rng = _rng(config.seed, "user", index)
        is_cold = index in cold_indices
        users.append(
            SimulatedUser(
                user_id=f"u{index:06d}",
                home_cell=f"cell_{rng.randrange(64):02d}",
                work_cell=f"cell_{rng.randrange(64):02d}",
                first_active_day=29 + rng.randrange(14) if is_cold else 1 + rng.randrange(8),
                budget_bucket=("low", "medium", "high")[rng.randrange(3)],
                preferred_service=("Ride", "Food", "Delivery", "Ads")[rng.randrange(4)],
            )
        )
    return tuple(users), frozenset(f"u{index:06d}" for index in cold_indices)


def _split_day_bounds(split: str) -> tuple[int, int]:
    return {"train": (1, 28), "validation": (29, 35), "test": (36, 42)}[split]


def _request_time(split: str, ordinal: int, count: int, *, service: str) -> tuple[int, datetime]:
    day_start, day_end = _split_day_bounds(split)
    day_count = day_end - day_start + 1
    day = day_start + ordinal % day_count
    minute = ordinal // day_count
    second = 10 if service == "Food" else 30
    cutoff = SIMULATION_START + timedelta(days=day - 1, hours=8, minutes=minute, seconds=second)
    return day, cutoff


def _eligible_users(users: Iterable[SimulatedUser], day: int) -> tuple[SimulatedUser, ...]:
    return tuple(user for user in users if user.first_active_day <= day)


def _request_user(
    *,
    users: tuple[SimulatedUser, ...],
    cold_user_ids: frozenset[str],
    split: str,
    ordinal: int,
    day: int,
    seed: int,
    service: str,
) -> SimulatedUser:
    cold = tuple(user for user in users if user.user_id in cold_user_ids and user.first_active_day <= day)
    if split != "train" and ordinal < len(cold_user_ids) and cold:
        return cold[ordinal % len(cold)]
    eligible = _eligible_users(users, day)
    return eligible[_rng(seed, "request-user", service, split, ordinal).randrange(len(eligible))]


def _candidate_from_entity(entity: CatalogueEntity, *, label: int, hard_negative: bool, placement: str | None = None) -> LogicalCandidate:
    return LogicalCandidate(
        candidate_id=entity.entity_id,
        category=entity.category,
        geo_cell=entity.geo_cell,
        lat=entity.lat,
        lon=entity.lon,
        available_at=entity.available_at,
        label=label,
        is_hard_negative=hard_negative,
        price=entity.price,
        merchant_id=entity.merchant_id,
        placement=placement,
    )


def _food_request(
    config: GreenSmConfig,
    catalogue: Catalogue,
    users: tuple[SimulatedUser, ...],
    cold_ids: frozenset[str],
    split: str,
    ordinal: int,
    count: int,
) -> LogicalRequest:
    day, cutoff = _request_time(split, ordinal, count, service="Food")
    user = _request_user(users=users, cold_user_ids=cold_ids, split=split, ordinal=ordinal, day=day, seed=config.seed, service="Food")
    rng = _rng(config.seed, "food-request", split, ordinal)
    preferred_category = FOOD_CATEGORIES[int(user.user_id[-3:]) % len(FOOD_CATEGORIES)]
    matching = [item for item in catalogue.food_items if item.category == preferred_category]
    positives = rng.sample(matching, 2)
    positive_ids = {entry.entity_id for entry in positives}
    hard_negative_count = config.thresholds.min_food_hard_negatives_per_request
    hard_negatives = rng.sample(
        [item for item in matching if item.entity_id not in positive_ids], hard_negative_count
    )
    used_ids = positive_ids | {entry.entity_id for entry in hard_negatives}
    selected = positives + hard_negatives + rng.sample(
        [item for item in catalogue.food_items if item.entity_id not in used_ids],
        20 - len(positives) - len(hard_negatives),
    )
    # The first two labels give a stable 10% candidate-level positive rate;
    # the next ten share their category and are explicit hard negatives.
    candidates = tuple(
        _candidate_from_entity(
            entity,
            label=1 if index < 2 else 0,
            hard_negative=2 <= index < 2 + hard_negative_count,
        )
        for index, entity in enumerate(selected)
    )
    return LogicalRequest(
        request_id=f"food:{split}:{ordinal:06d}",
        user_id=user.user_id,
        cutoff_time=cutoff,
        split=split,
        service="Food",
        request_cell=user.work_cell if cutoff.hour < 17 else user.home_cell,
        candidates=candidates,
    )


def _ads_request(
    config: GreenSmConfig,
    catalogue: Catalogue,
    users: tuple[SimulatedUser, ...],
    cold_ids: frozenset[str],
    split: str,
    ordinal: int,
    count: int,
) -> LogicalRequest:
    day, cutoff = _request_time(split, ordinal, count, service="Ads")
    user = _request_user(users=users, cold_user_ids=cold_ids, split=split, ordinal=ordinal, day=day, seed=config.seed, service="Ads")
    rng = _rng(config.seed, "ads-request", split, ordinal)
    selected = rng.sample(list(catalogue.ad_campaigns), 10)
    has_click = ordinal % 2 == 0
    candidates = tuple(
        _candidate_from_entity(
            entity,
            label=1 if has_click and index == 0 else 0,
            hard_negative=False,
            placement=("home", "search", "checkout")[ordinal % 3],
        )
        for index, entity in enumerate(selected)
    )
    return LogicalRequest(
        request_id=f"ads:{split}:{ordinal:06d}",
        user_id=user.user_id,
        cutoff_time=cutoff,
        split=split,
        service="Ads",
        request_cell=user.work_cell if cutoff.hour < 17 else user.home_cell,
        candidates=candidates,
    )


def build_target_world(config: GreenSmConfig) -> TargetWorld:
    """Create public logical targets once; history views must never mutate them."""

    catalogue = _build_catalogue(config)
    users, cold_ids = _build_users(config)
    food_requests = tuple(
        _food_request(config, catalogue, users, cold_ids, split, ordinal, config.food_request_counts[split])
        for split in ("train", "validation", "test")
        for ordinal in range(config.food_request_counts[split])
    )
    ads_requests = tuple(
        _ads_request(config, catalogue, users, cold_ids, split, ordinal, config.ads_request_counts[split])
        for split in ("train", "validation", "test")
        for ordinal in range(config.ads_request_counts[split])
    )
    return TargetWorld(
        catalogue=catalogue,
        users=users,
        food_requests=food_requests,
        ads_requests=ads_requests,
        cold_start_user_ids=cold_ids,
    )
