from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.data.green_sm.config import GreenSmConfig
from src.data.green_sm.histories import build_history_views
from src.data.green_sm.world import build_target_world


def _pilot_config(tmp_path: Path) -> GreenSmConfig:
    return replace(GreenSmConfig.from_json("configs/green_sm_sim_pilot.json"), output_root=tmp_path / "out")


def test_control_preserves_targets_but_removes_episode_linkage(tmp_path: Path) -> None:
    world = build_target_world(_pilot_config(tmp_path))
    linked, control = build_history_views(world)

    assert linked.logical_request_digest() == control.logical_request_digest()
    assert linked.events_for("u000001") != control.events_for("u000001")
    assert linked.has_target_episode_marker() is True
    assert control.has_target_episode_marker() is False


def test_history_views_are_chronological_complete_and_causal_for_cold_users(tmp_path: Path) -> None:
    world = build_target_world(_pilot_config(tmp_path))
    linked, control = build_history_views(world)

    for view in (linked, control):
        for user in world.users:
            groups = view.events_for(user.user_id)
            assert list(groups) == sorted(groups, key=lambda group: (group.event_time, group.source_record_id))
            assert all(1 <= group.semantic_field_count <= 7 for group in groups)
            if user.user_id in world.cold_start_user_ids:
                assert all(group.split != "train" for group in groups)
        for request in world.all_requests():
            prior = view.events_before(request.user_id, request.cutoff_time)
            assert all(group.event_time < request.cutoff_time for group in prior)
            assert all(request.request_id not in group.source_record_id for group in prior)


def test_shared_signal_poc_emits_causal_cross_service_events_and_control_removes_them(tmp_path: Path) -> None:
    base = GreenSmConfig.from_json("configs/green_sm_shared_signal_poc.json")
    config = replace(
        base,
        output_root=tmp_path / "out",
        user_count=96,
        snapshot_counts={"train": 40, "validation": 20, "test": 20},
        food_request_counts={"train": 20, "validation": 10, "test": 10},
        ads_request_counts={"train": 20, "validation": 10, "test": 10},
    )
    world = build_target_world(config)
    linked, control = build_history_views(world)
    request = next(
        request
        for request in world.food_requests
        if world.episode_for(request.request_id).kind == "cross_service"
    )
    episode = world.episode_for(request.request_id)

    linked_prior = linked.events_before(request.user_id, request.cutoff_time)
    control_prior = control.events_before(request.user_id, request.cutoff_time)
    assert episode.direction == "ads_to_food"
    assert any(
        group.source_record_id == episode.source_record_id
        and group.scenario == "Ads"
        and group.event_time < request.cutoff_time
        for group in linked_prior
    )
    assert not any(group.source_record_id == episode.source_record_id for group in control_prior)
    assert all("food_item_" not in repr(group.public_payload()) for group in linked_prior)
