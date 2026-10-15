"""Causal linked and paired-control history views over one target world."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping

from .world import SIMULATION_START, Catalogue, LogicalRequest, SimulatedUser, TargetWorld, _rng


def _digest_text(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def _split_for_time(event_time: datetime) -> str:
    day = (event_time.date() - SIMULATION_START.date()).days + 1
    if day <= 28:
        return "train"
    if day <= 35:
        return "validation"
    return "test"


@dataclass(frozen=True)
class EventGroup:
    """One complete semantic observation; it is never split by history selection."""

    source_record_id: str
    source_record_index: int
    user_id: str
    event_time: datetime
    split: str
    scenario: str
    event_type: str
    context: Mapping[str, object]
    action: Mapping[str, object]
    linked_episode: bool = False

    @property
    def semantic_field_count(self) -> int:
        return len(self.context) + len(self.action)

    def public_payload(self) -> dict[str, object]:
        return {
            "source_record_id": self.source_record_id,
            "source_record_index": self.source_record_index,
            "user_id": self.user_id,
            "event_time": self.event_time.isoformat(),
            "split": self.split,
            "scenario": self.scenario,
            "event_type": self.event_type,
            "context": dict(self.context),
            "action": dict(self.action),
        }


@dataclass(frozen=True)
class HistoryView:
    name: str
    world: TargetWorld
    _events_by_user: Mapping[str, tuple[EventGroup, ...]]

    def events_for(self, user_id: str) -> tuple[EventGroup, ...]:
        return self._events_by_user.get(user_id, ())

    def events_before(self, user_id: str, cutoff_time: datetime) -> tuple[EventGroup, ...]:
        return tuple(group for group in self.events_for(user_id) if group.event_time < cutoff_time)

    def logical_request_digest(self) -> str:
        return self.world.fingerprint()

    def has_target_episode_marker(self) -> bool:
        return any(
            group.linked_episode
            for groups in self._events_by_user.values()
            for group in groups
        )


def _geo(catalogue: Catalogue, cell_id: str) -> dict[str, object]:
    cell = catalogue.cell(cell_id)
    return {"lat": cell.lat, "lon": cell.lon, "geo_cell": cell.cell_id}


def _event_fields(
    *,
    catalogue: Catalogue,
    scenario: str,
    event_type: str,
    cell_id: str,
    ordinal: int,
) -> tuple[dict[str, object], dict[str, object]]:
    category = ("noodles", "rice", "coffee", "tea", "bakery", "vegetarian")[ordinal % 6]
    if scenario == "Ride":
        return (
            {"pickup_location": _geo(catalogue, cell_id), "dropoff_location": _geo(catalogue, cell_id)},
            {"fare": 25_000 + (ordinal % 8) * 5_000, "distance_km": 1.5 + (ordinal % 12) * 0.8},
        )
    if scenario == "Food":
        return (
            {"merchant_location": _geo(catalogue, cell_id), "food_category": category},
            {"query": f"{category} nearby", "item_price": 35_000 + (ordinal % 9) * 15_000},
        )
    if scenario == "Delivery":
        return (
            {"delivery_location": _geo(catalogue, cell_id), "delivery_category": category},
            {"delivery_fee": 12_000 + (ordinal % 6) * 2_000, "distance_km": 1.0 + (ordinal % 10) * 0.6},
        )
    return (
        {"ad_context_location": _geo(catalogue, cell_id), "placement": ("home", "search", "checkout")[ordinal % 3]},
        {"campaign_id": f"campaign_{ordinal % 120:03d}", "bid": 2_000 + (ordinal % 7) * 500},
    )


_TEMPLATES = (
    ("Ride", "ride_search"),
    ("Ride", "ride_book"),
    ("Ride", "ride_complete"),
    ("Food", "food_search"),
    ("Food", "merchant_view"),
    ("Food", "item_view"),
    ("Food", "food_order"),
    ("Delivery", "delivery_quote"),
    ("Delivery", "delivery_book"),
    ("Delivery", "delivery_complete"),
    ("Ads", "ad_impression"),
    ("Ads", "ad_click"),
)


def _generic_groups(world: TargetWorld, user: SimulatedUser, *, view: str) -> list[EventGroup]:
    groups: list[EventGroup] = []
    user_number = int(user.user_id[1:])
    for day in range(user.first_active_day, 43):
        count = 1 + int(_rng(user_number, "generic-count", day).random() < 0.5)
        for slot in range(count):
            scenario, event_type = _TEMPLATES[(user_number + day + slot) % len(_TEMPLATES)]
            event_time = SIMULATION_START + timedelta(days=day - 1, hours=7, minutes=slot * 11 + user_number % 11)
            linked_cell = user.work_cell if day % 2 else user.home_cell
            # A paired control retains same-service observations and resamples the
            # other-service record identity/content without using target labels.
            record_id = f"generic:{user.user_id}:{day:02d}:{slot}"
            cell_id = linked_cell
            ordinal = day + slot + user_number
            if view == "control" and scenario in {"Ride", "Delivery"}:
                record_id = f"control:{_digest_text(user.user_id, day, slot, scenario)}"
                ordinal += 37
            context, action = _event_fields(
                catalogue=world.catalogue,
                scenario=scenario,
                event_type=event_type,
                cell_id=cell_id,
                ordinal=ordinal,
            )
            groups.append(
                EventGroup(
                    source_record_id=record_id,
                    source_record_index=0,
                    user_id=user.user_id,
                    event_time=event_time,
                    split=_split_for_time(event_time),
                    scenario=scenario,
                    event_type=event_type,
                    context=context,
                    action=action,
                )
            )
    return groups


def _episode_group(world: TargetWorld, request: LogicalRequest, *, view: str) -> EventGroup:
    if request.service == "Food":
        scenario, event_type, before = "Ride", "ride_complete", timedelta(minutes=25)
    else:
        scenario, event_type, before = "Food", "food_search", timedelta(minutes=15)
    ordinal = int(_digest_text(request.request_id), 16) % 100_000
    # The control remains within the same request-cell/time stratum, but uses
    # independent content and a distinct source record. It never accesses a
    # candidate label while resampling.
    if view == "control":
        ordinal = int(_digest_text("control", request.user_id, request.cutoff_time.isoformat()), 16) % 100_000
        source_record_id = f"control:{_digest_text(request.user_id, request.cutoff_time.isoformat(), scenario)}"
    else:
        source_record_id = f"episode:{scenario.lower()}:{_digest_text(request.user_id, request.cutoff_time.isoformat())}"
    event_time = request.cutoff_time - before
    context, action = _event_fields(
        catalogue=world.catalogue,
        scenario=scenario,
        event_type=event_type,
        cell_id=request.request_cell,
        ordinal=ordinal,
    )
    return EventGroup(
        source_record_id=source_record_id,
        source_record_index=0,
        user_id=request.user_id,
        event_time=event_time,
        split=request.split,
        scenario=scenario,
        event_type=event_type,
        context=context,
        action=action,
        linked_episode=view == "linked",
    )


def _view(world: TargetWorld, name: str) -> HistoryView:
    events_by_user: dict[str, list[EventGroup]] = {
        user.user_id: _generic_groups(world, user, view=name) for user in world.users
    }
    for request in world.all_requests():
        events_by_user[request.user_id].append(_episode_group(world, request, view=name))
    normalized: dict[str, tuple[EventGroup, ...]] = {}
    for user_id, groups in events_by_user.items():
        ordered = sorted(groups, key=lambda group: (group.event_time, group.source_record_id))
        normalized[user_id] = tuple(
            EventGroup(
                source_record_id=group.source_record_id,
                source_record_index=index,
                user_id=group.user_id,
                event_time=group.event_time,
                split=group.split,
                scenario=group.scenario,
                event_type=group.event_type,
                context=group.context,
                action=group.action,
                linked_episode=group.linked_episode,
            )
            for index, group in enumerate(ordered)
        )
    return HistoryView(name=name, world=world, _events_by_user=normalized)


def build_history_views(world: TargetWorld) -> tuple[HistoryView, HistoryView]:
    """Return linked then paired-control views over the exact same logical targets."""

    return _view(world, "linked"), _view(world, "control")
