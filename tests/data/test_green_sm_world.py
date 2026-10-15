from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.data.green_sm.config import GreenSmConfig
from src.data.green_sm.world import build_target_world


def _pilot_config(tmp_path: Path, *, seed: int = 7) -> GreenSmConfig:
    base = GreenSmConfig.from_json("configs/green_sm_sim_pilot.json")
    return replace(base, seed=seed, output_root=tmp_path / "out")


def test_target_world_is_seed_deterministic_and_contains_cold_test_users(tmp_path: Path) -> None:
    first = build_target_world(_pilot_config(tmp_path, seed=7))
    second = build_target_world(_pilot_config(tmp_path, seed=7))

    assert first.fingerprint() == second.fingerprint()
    assert len(first.cold_start_user_ids) == 36
    assert all(user.first_active_day >= 29 for user in first.cold_start_users())


def test_target_world_has_exact_requests_public_candidates_and_valid_label_rates(tmp_path: Path) -> None:
    world = build_target_world(_pilot_config(tmp_path))

    assert {
        split: sum(request.split == split for request in world.food_requests)
        for split in ("train", "validation", "test")
    } == {"train": 360, "validation": 60, "test": 60}
    assert {
        split: sum(request.split == split for request in world.ads_requests)
        for split in ("train", "validation", "test")
    } == {"train": 480, "validation": 80, "test": 80}
    assert all(len(request.candidates) == 20 for request in world.food_requests)
    assert all(len({candidate.candidate_id for candidate in request.candidates}) == 20 for request in world.food_requests)
    assert all(sum(candidate.is_hard_negative for candidate in request.candidates) >= 10 for request in world.food_requests)
    assert all(
        candidate.category == request.candidates[0].category
        for request in world.food_requests
        for candidate in request.candidates
        if candidate.is_hard_negative
    )
    assert all(len(request.candidates) == 10 for request in world.ads_requests)
    assert all(len({candidate.candidate_id for candidate in request.candidates}) == 10 for request in world.ads_requests)
    assert 0.05 <= world.food_positive_rate() <= 0.20
    assert 0.02 <= world.ads_click_rate() <= 0.15
    assert world.ads_ranking_eligible_rate() >= 0.30
    assert "latent" not in repr(world.public_payload()).lower()
    assert all("foundation_manifest" not in request.public_payload() for request in world.all_requests())
