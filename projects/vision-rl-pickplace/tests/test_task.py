"""The pick-and-place success rule and goal sampler: pure functions, no simulator."""

import dataclasses

import numpy as np
import pybullet as p
import pytest

from pickplace.task import (
    PlaceOutcome, PlacementState, ReleaseTracker, cube_bottom_height, cube_tilt, evaluate_placement,
    goal_region, sample_place_goal, spawn_region,
)

PICK = np.array([0.45, -0.10])
GOAL = np.array([0.58, 0.05])


@pytest.fixture
def placed(cfg):
    """A successful placement: every failure test changes exactly one field."""
    return PlacementState(
        was_lifted=True, release_height=0.005, steps_without_contact=cfg.place.release_steps,
        object_position=np.r_[GOAL + [0.01, 0.0], cfg.object.size / 2], speed=0.0, tilt=0.0,
        pick_xy=PICK, goal_xy=GOAL,
    )


def _eval(cfg, state, **place_overrides):
    place = dataclasses.replace(cfg.place, **place_overrides)
    return evaluate_placement(state, place, spawn_region(cfg))


# ------------------------------------------------------------------ goal sampler
@pytest.mark.parametrize("seed", range(20))
def test_goals_far_enough_and_inside_bounds(cfg, seed):
    rng = np.random.default_rng(seed)
    region = spawn_region(cfg)
    pick = rng.uniform(region.low, region.high)
    goal = sample_place_goal(rng, pick, cfg)
    assert np.linalg.norm(goal - pick) >= cfg.place.min_distance + cfg.place.goal_tolerance
    assert goal_region(cfg).contains(goal)


def test_goal_within_tolerance_is_always_far_enough_and_in_workspace(cfg):
    """Landing anywhere within tolerance of a goal satisfies min_distance and the workspace rule."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        pick = rng.uniform(spawn_region(cfg).low, spawn_region(cfg).high)
        goal = sample_place_goal(rng, pick, cfg)
        angle = rng.uniform(0, 2 * np.pi)
        landed = goal + cfg.place.goal_tolerance * np.array([np.cos(angle), np.sin(angle)])
        assert np.linalg.norm(landed - pick) >= cfg.place.min_distance - 1e-9
        assert spawn_region(cfg).contains(landed)


def test_goal_same_for_same_seed(cfg):
    a = sample_place_goal(np.random.default_rng(42), PICK, cfg)
    b = sample_place_goal(np.random.default_rng(42), PICK, cfg)
    np.testing.assert_array_equal(a, b)


def test_edge_goals_lie_near_the_edge(cfg):
    rng = np.random.default_rng(1)
    region = goal_region(cfg)
    for _ in range(20):
        goal = sample_place_goal(rng, PICK, cfg, edge_goals=True)
        assert np.min(np.r_[goal - region.low, region.high - goal]) <= cfg.place.edge_band


def test_goal_sampler_gives_up_loudly(cfg):
    impossible = dataclasses.replace(cfg, place=dataclasses.replace(cfg.place, min_distance=5.0))
    with pytest.raises(RuntimeError, match="no goal"):
        sample_place_goal(np.random.default_rng(0), PICK, impossible)


# ------------------------------------------------------------------ geometry
def test_tilt_treats_every_face_as_upright():
    assert cube_tilt(p.getQuaternionFromEuler([0, 0, 0.7])) == pytest.approx(0, abs=1e-6)
    assert cube_tilt(p.getQuaternionFromEuler([np.pi / 2, 0, 0])) == pytest.approx(0, abs=1e-6)  # rolled onto a side
    assert cube_tilt(p.getQuaternionFromEuler([np.pi / 4, 0, 0])) == pytest.approx(np.pi / 4, abs=1e-6)  # on an edge
    assert cube_tilt(p.getQuaternionFromEuler([0.2, 0, 0])) == pytest.approx(0.2, abs=1e-6)


def test_bottom_height_of_flat_and_tilted_cube():
    flat = p.getQuaternionFromEuler([0, 0, 0.3])
    assert cube_bottom_height([0, 0, 0.02], flat, 0.04) == pytest.approx(0.0, abs=1e-9)
    edge = p.getQuaternionFromEuler([np.pi / 4, 0, 0])
    assert cube_bottom_height([0, 0, 0.02], edge, 0.04) == pytest.approx(0.02 - 0.02 * np.sqrt(2), abs=1e-9)


# ------------------------------------------------------------------ release tracker
def test_release_needs_consecutive_steps_without_contact():
    t = ReleaseTracker(release_steps=3)
    t.update(True, 0.05)
    t.update(False, 0.004)          # let go 4 mm above the floor
    t.update(False, 0.0)
    assert not t.released
    t.update(False, 0.0)
    assert t.released and t.release_height == pytest.approx(0.004)


def test_regrasp_resets_the_release():
    t = ReleaseTracker(release_steps=1)
    t.update(False, 0.1)
    assert t.released
    t.update(True, 0.1)
    assert not t.released and t.release_height is None


def test_release_tracker_rejects_bad_steps():
    with pytest.raises(ValueError):
        ReleaseTracker(0)


# ------------------------------------------------------------------ one test per outcome
def test_success(cfg, placed):
    assert _eval(cfg, placed) is PlaceOutcome.SUCCESS


def test_pushed_not_lifted(cfg, placed):
    assert _eval(cfg, dataclasses.replace(placed, was_lifted=False)) is PlaceOutcome.NOT_LIFTED


def test_still_held(cfg, placed):
    assert _eval(cfg, dataclasses.replace(placed, release_height=None, steps_without_contact=0)) \
        is PlaceOutcome.NOT_RELEASED
    assert _eval(cfg, dataclasses.replace(placed, steps_without_contact=cfg.place.release_steps - 1)) \
        is PlaceOutcome.NOT_RELEASED


def test_thrown_or_dropped(cfg, placed):
    state = dataclasses.replace(placed, release_height=cfg.place.max_release_height + 0.01)
    assert _eval(cfg, state) is PlaceOutcome.RELEASED_TOO_HIGH


def test_still_moving(cfg, placed):
    assert _eval(cfg, dataclasses.replace(placed, speed=0.05)) is PlaceOutcome.NOT_AT_REST


def test_toppled(cfg, placed):
    assert _eval(cfg, dataclasses.replace(placed, tilt=np.pi / 4)) is PlaceOutcome.TOPPLED


def test_left_workspace(cfg, placed):
    state = dataclasses.replace(placed, object_position=np.array([0.9, 0.0, 0.02]), goal_xy=np.array([0.9, 0.0]))
    assert _eval(cfg, state) is PlaceOutcome.LEFT_WORKSPACE


def test_too_close_to_pick_spot(cfg, placed):
    state = dataclasses.replace(placed, object_position=np.r_[PICK + [0.05, 0.0], 0.02])
    assert _eval(cfg, state) is PlaceOutcome.TOO_CLOSE


def test_missed_goal(cfg, placed):
    state = dataclasses.replace(placed, object_position=np.r_[GOAL + [0.0, -0.05], 0.02])
    assert _eval(cfg, state) is PlaceOutcome.MISSED_GOAL


def test_goal_mode_any_needs_only_min_distance(cfg, placed):
    state = dataclasses.replace(placed, object_position=np.r_[GOAL + [0.0, -0.05], 0.02])
    assert _eval(cfg, state, goal_mode="any") is PlaceOutcome.SUCCESS
