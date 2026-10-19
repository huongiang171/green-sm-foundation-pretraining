"""Deterministic, public target-world construction for the Green SM simulator."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
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
class UserAffinity:
    """Private, seed-derived state used only while constructing the target world."""

    primary_food_category: str
    secondary_food_category: str
    merchant_family: int
    preferred_price_band: int
    preferred_geo_cell: str
    preferred_placement: str
    promotion_sensitivity: float


@dataclass(frozen=True)
class RequestEpisode:
    """Private causal provenance; deliberately absent from serialized target rows."""

    kind: str
    direction: str
    source_record_id: str


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
    user_affinities: Mapping[str, UserAffinity]
    episodes: Mapping[str, RequestEpisode]

    def all_requests(self) -> tuple[LogicalRequest, ...]:
        return self.food_requests + self.ads_requests

    def cold_start_users(self) -> tuple[SimulatedUser, ...]:
        return tuple(user for user in self.users if user.user_id in self.cold_start_user_ids)

    def affinity_for(self, user_id: str) -> UserAffinity:
        return self.user_affinities[user_id]

    def episode_for(self, request_id: str) -> RequestEpisode:
        return self.episodes[request_id]

    def episode_mix_counts(self) -> dict[str, int]:
        return {
            kind: sum(episode.kind == kind for episode in self.episodes.values())
            for kind in ("cross_service", "same_service", "noise")
        }

    def episode_direction_counts(self) -> dict[str, int]:
        return {
            direction: sum(episode.direction == direction for episode in self.episodes.values())
            for direction in ("food_to_ads", "ads_to_food")
        }

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
                first_active_day=(29 if config.behavior_policy == "shared_signal_poc_v1" else 29 + rng.randrange(14))
                if is_cold
                else 1 + rng.randrange(8),
                budget_bucket=("low", "medium", "high")[rng.randrange(3)],
                preferred_service=("Ride", "Food", "Delivery", "Ads")[rng.randrange(4)],
            )
        )
    return tuple(users), frozenset(f"u{index:06d}" for index in cold_indices)


def _build_affinities(config: GreenSmConfig, users: Iterable[SimulatedUser]) -> dict[str, UserAffinity]:
    affinities: dict[str, UserAffinity] = {}
    for user in users:
        rng = _rng(config.seed, "shared-affinity", user.user_id)
        primary_index = rng.randrange(len(FOOD_CATEGORIES))
        secondary_index = (primary_index + 1 + rng.randrange(len(FOOD_CATEGORIES) - 1)) % len(
            FOOD_CATEGORIES
        )
        primary_families = [index % 8 for index in range(160) if index % len(FOOD_CATEGORIES) == primary_index]
        affinities[user.user_id] = UserAffinity(
            primary_food_category=FOOD_CATEGORIES[primary_index],
            secondary_food_category=FOOD_CATEGORIES[secondary_index],
            merchant_family=primary_families[rng.randrange(len(primary_families))],
            preferred_price_band=rng.randrange(9),
            preferred_geo_cell=user.home_cell if rng.random() < 0.5 else user.work_cell,
            preferred_placement=("home", "search", "checkout")[rng.randrange(3)],
            promotion_sensitivity=round(0.35 + rng.random() * 0.60, 4),
        )
    return affinities


def _merchant_family(merchant_id: str | None) -> int:
    if merchant_id is None:
        return -1
    return int(merchant_id.rsplit("_", 1)[-1]) % 8


def _food_compatibility(item: CatalogueEntity, affinity: UserAffinity) -> int:
    score = 0
    if item.category == affinity.primary_food_category:
        score += 8
    elif item.category == affinity.secondary_food_category:
        score += 4
    if _merchant_family(item.merchant_id) == affinity.merchant_family:
        score += 3
    if item.geo_cell == affinity.preferred_geo_cell:
        score += 2
    if item.price is not None and (item.price - 35_000) // 15_000 == affinity.preferred_price_band:
        score += 2
    return score


def _ads_compatibility(campaign: CatalogueEntity, affinity: UserAffinity, placement: str) -> int:
    score = 0
    if campaign.category == affinity.primary_food_category:
        score += 7
    elif campaign.category == affinity.secondary_food_category:
        score += 3
    if campaign.geo_cell == affinity.preferred_geo_cell:
        score += 2
    if placement == affinity.preferred_placement:
        score += 2
    return score


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
    config: GreenSmConfig,
    users: tuple[SimulatedUser, ...],
    cold_user_ids: frozenset[str],
    split: str,
    ordinal: int,
    day: int,
    seed: int,
    service: str,
    request_count: int,
) -> SimulatedUser:
    cold = tuple(user for user in users if user.user_id in cold_user_ids and user.first_active_day <= day)
    if config.behavior_policy == "shared_signal_poc_v1" and split != "train":
        cold_request_count = round(request_count * config.thresholds.cold_start_test_user_rate)
        cold_ordinals = sorted(
            range(request_count),
            key=lambda value: _rng(config.seed, "cold-request", service, split, value).random(),
        )[:cold_request_count]
        if ordinal in cold_ordinals:
            return cold[_rng(config.seed, "cold-request-user", service, split, ordinal).randrange(len(cold))]
        eligible = tuple(
            user
            for user in _eligible_users(users, day)
            if user.user_id not in cold_user_ids
        )
        return eligible[_rng(config.seed, "request-user", service, split, ordinal).randrange(len(eligible))]
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
    user = _request_user(config=config, users=users, cold_user_ids=cold_ids, split=split, ordinal=ordinal, day=day, seed=config.seed, service="Food", request_count=count)
    rng = _rng(config.seed, "food-request", split, ordinal)
    if config.behavior_policy == "shared_signal_poc_v1":
        affinity = _build_affinities(config, (user,))[user.user_id]
        scored = sorted(
            catalogue.food_items,
            key=lambda item: (_food_compatibility(item, affinity), item.entity_id),
            reverse=True,
        )
        positives = scored[:2]
        positive_ids = {item.entity_id for item in positives}
        hard_pool = [
            item
            for item in catalogue.food_items
            if item.entity_id not in positive_ids
            and item.category in {affinity.primary_food_category, affinity.secondary_food_category}
            and (
                _merchant_family(item.merchant_id) != affinity.merchant_family
                or item.geo_cell != affinity.preferred_geo_cell
                or (item.price is not None and (item.price - 35_000) // 15_000 != affinity.preferred_price_band)
            )
        ]
        hard_negative_count = config.thresholds.min_food_hard_negatives_per_request
        hard_negatives = hard_pool[:hard_negative_count]
        if len(hard_negatives) != hard_negative_count:
            raise ValueError("shared_signal_poc_v1 catalogue cannot supply required hard negatives")
        used_ids = positive_ids | {item.entity_id for item in hard_negatives}
        remaining = [item for item in catalogue.food_items if item.entity_id not in used_ids]
        selected = positives + hard_negatives + rng.sample(
            remaining, 20 - len(positives) - len(hard_negatives)
        )
        rng.shuffle(selected)
        return LogicalRequest(
            request_id=f"food:{split}:{ordinal:06d}",
            user_id=user.user_id,
            cutoff_time=cutoff,
            split=split,
            service="Food",
            request_cell=user.work_cell if cutoff.hour < 17 else user.home_cell,
            candidates=tuple(
                _candidate_from_entity(
                    item,
                    label=int(item.entity_id in positive_ids),
                    hard_negative=item.entity_id in {entry.entity_id for entry in hard_negatives},
                )
                for item in selected
            ),
        )
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
    user = _request_user(config=config, users=users, cold_user_ids=cold_ids, split=split, ordinal=ordinal, day=day, seed=config.seed, service="Ads", request_count=count)
    rng = _rng(config.seed, "ads-request", split, ordinal)
    if config.behavior_policy == "shared_signal_poc_v1":
        affinity = _build_affinities(config, (user,))[user.user_id]
        placement = ("home", "search", "checkout")[ordinal % 3]
        scored = sorted(
            catalogue.ad_campaigns,
            key=lambda campaign: (_ads_compatibility(campaign, affinity, placement), campaign.entity_id),
            reverse=True,
        )
        positive = scored[0]
        selected = [positive, *rng.sample(scored[1:], 9)]
        rng.shuffle(selected)
        return LogicalRequest(
            request_id=f"ads:{split}:{ordinal:06d}",
            user_id=user.user_id,
            cutoff_time=cutoff,
            split=split,
            service="Ads",
            request_cell=user.work_cell if cutoff.hour < 17 else user.home_cell,
            candidates=tuple(
                _candidate_from_entity(
                    campaign,
                    label=int(campaign.entity_id == positive.entity_id),
                    hard_negative=False,
                    placement=placement,
                )
                for campaign in selected
            ),
        )
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
    affinities = _build_affinities(config, users) if config.behavior_policy == "shared_signal_poc_v1" else {}
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
    episodes: dict[str, RequestEpisode] = {}
    if config.behavior_policy == "shared_signal_poc_v1":
        for requests, direction in ((food_requests, "ads_to_food"), (ads_requests, "food_to_ads")):
            count = len(requests)
            cross_count = round(count * 0.70)
            same_count = round(count * 0.20)
            for index, request in enumerate(requests):
                kind = "cross_service" if index < cross_count else "same_service" if index < cross_count + same_count else "noise"
                episodes[request.request_id] = RequestEpisode(
                    kind=kind,
                    direction=direction if kind == "cross_service" else request.service.lower(),
                    source_record_id=f"episode:{request.request_id}",
                )
    return TargetWorld(
        catalogue=catalogue,
        users=users,
        food_requests=food_requests,
        ads_requests=ads_requests,
        cold_start_user_ids=cold_ids,
        user_affinities=affinities,
        episodes=episodes,
    )
