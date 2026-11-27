"""Pick-and-place task definition: goal sampling and the placement success rule.

One rule for every controller. The scripted baseline and the RL environment
both call :func:`evaluate_placement`, so their success rates measure the same
thing. Everything here is pure (no simulator), so each failure case has its
own unit test.

A placement succeeds when the cube
    1. was lifted at least ``task.success_lift`` while held   (not pushed or slid)
    2. was let go with its bottom <= ``max_release_height``   (not thrown or dropped)
    3. has had no finger contact for ``release_steps`` steps  (not still held)
    4. is at rest and upright                                 (not toppled)
    5. is inside the spawn region                             (in camera view, reachable)
    6. is >= ``min_distance`` from where it was picked        (the core requirement)
    7. in goal_mode "target": is within ``goal_tolerance`` of the goal
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import Config, PlaceConfig

MAX_GOAL_TRIES = 100


class PlaceOutcome(str, Enum):
    SUCCESS = "success"
    NOT_LIFTED = "not_lifted"
    NOT_RELEASED = "not_released"
    RELEASED_TOO_HIGH = "released_too_high"
    NOT_AT_REST = "not_at_rest"
    TOPPLED = "toppled"
    LEFT_WORKSPACE = "left_workspace"
    TOO_CLOSE = "too_close"
    MISSED_GOAL = "missed_goal"


# Outcomes that cannot be undone within an episode. The RL env terminates on them.
TERMINAL_FAILURES = frozenset({PlaceOutcome.RELEASED_TOO_HIGH, PlaceOutcome.TOPPLED, PlaceOutcome.LEFT_WORKSPACE})


@dataclass(frozen=True)
class GoalRegion:
    """Axis-aligned rectangle of allowed goal positions (xy, metres)."""

    low: np.ndarray
    high: np.ndarray

    def contains(self, xy: np.ndarray) -> bool:
        return bool(np.all(xy >= self.low) and np.all(xy <= self.high))


def spawn_region(cfg: Config) -> GoalRegion:
    ws = cfg.workspace
    return GoalRegion(np.array([ws.object_x[0], ws.object_y[0]]), np.array([ws.object_x[1], ws.object_y[1]]))


def goal_region(cfg: Config) -> GoalRegion:
    """Spawn region shrunk by the goal tolerance: a cube within tolerance stays inside the spawn region."""
    region, tol = spawn_region(cfg), cfg.place.goal_tolerance
    return GoalRegion(region.low + tol, region.high - tol)


def sample_place_goal(rng: np.random.Generator, pick_xy: np.ndarray, cfg: Config,
                      edge_goals: bool = False) -> np.ndarray:
    """Random goal at least ``min_distance + goal_tolerance`` from ``pick_xy``.

    Call it right after the cube position is drawn from the same generator, so a
    seed gives the same pick and place spots to every controller. With
    ``edge_goals`` only goals within ``place.edge_band`` of the region's edge are
    accepted (out-of-distribution test).
    """
    pl = cfg.place
    region = goal_region(cfg)
    min_dist = pl.min_distance + pl.goal_tolerance
    pick_xy = np.asarray(pick_xy, float)[:2]
    for _ in range(MAX_GOAL_TRIES):
        goal = rng.uniform(region.low, region.high)
        if np.linalg.norm(goal - pick_xy) < min_dist:
            continue
        if edge_goals and np.min(np.r_[goal - region.low, region.high - goal]) > pl.edge_band:
            continue
        return goal
    raise RuntimeError(f"no goal >= {min_dist:.3f} m from {pick_xy} found in {MAX_GOAL_TRIES} tries")


# --------------------------------------------------------------------- geometry
def rotation_matrix(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def cube_tilt(quat_xyzw) -> float:
    """Angle between world up and the nearest cube face normal.

    A cube looks the same on every face, so a cube that rolled onto another face
    is upright (tilt 0); one resting on an edge is at 45 degrees.
    """
    up_components = np.abs(rotation_matrix(quat_xyzw)[2])   # each cube axis' z-component
    return float(np.arccos(np.clip(up_components.max(), -1.0, 1.0)))


def cube_bottom_height(position, quat_xyzw, size: float) -> float:
    """Height of the cube's lowest corner above the floor."""
    half_extent_z = 0.5 * size * np.abs(rotation_matrix(quat_xyzw)[2]).sum()
    return float(position[2] - half_extent_z)


# ---------------------------------------------------------------------- release
class ReleaseTracker:
    """Detects a release: no finger contact for ``release_steps`` consecutive steps.

    Records the cube's bottom height at the first step without contact, which is
    the height it was let go from. Any contact resets it, so re-grasping and
    placing again is judged on the latest release only.
    """

    def __init__(self, release_steps: int):
        if release_steps < 1:
            raise ValueError("release_steps must be >= 1")
        self.release_steps = release_steps
        self.reset()

    def reset(self) -> None:
        self.steps_without_contact = 0
        self.release_height: float | None = None

    def update(self, in_contact: bool, bottom_height: float) -> None:
        if in_contact:
            self.reset()
            return
        if self.steps_without_contact == 0:
            self.release_height = float(bottom_height)
        self.steps_without_contact += 1

    @property
    def released(self) -> bool:
        return self.release_height is not None and self.steps_without_contact >= self.release_steps


# ------------------------------------------------------------------- evaluation
@dataclass(frozen=True)
class PlacementState:
    """Everything the success rule needs, measured from true simulator state."""

    was_lifted: bool
    release_height: float | None      # cube bottom when contact was lost; None if never released
    steps_without_contact: int
    object_position: np.ndarray       # cube centre, world frame
    speed: float                      # linear speed, m/s
    tilt: float                       # radians, see cube_tilt
    pick_xy: np.ndarray
    goal_xy: np.ndarray


def evaluate_placement(state: PlacementState, place: PlaceConfig, region: GoalRegion) -> PlaceOutcome:
    """Apply the success rule; the first unmet condition is the reported outcome."""
    if not state.was_lifted:
        return PlaceOutcome.NOT_LIFTED
    if state.release_height is None or state.steps_without_contact < place.release_steps:
        return PlaceOutcome.NOT_RELEASED
    if state.release_height > place.max_release_height:
        return PlaceOutcome.RELEASED_TOO_HIGH
    if state.speed >= place.rest_speed:
        return PlaceOutcome.NOT_AT_REST
    if state.tilt >= place.max_tilt:
        return PlaceOutcome.TOPPLED
    xy = np.asarray(state.object_position, float)[:2]
    if not region.contains(xy):
        return PlaceOutcome.LEFT_WORKSPACE
    if np.linalg.norm(xy - np.asarray(state.pick_xy, float)[:2]) < place.min_distance:
        return PlaceOutcome.TOO_CLOSE
    if place.goal_mode == "target" and np.linalg.norm(xy - np.asarray(state.goal_xy, float)[:2]) > place.goal_tolerance:
        return PlaceOutcome.MISSED_GOAL
    return PlaceOutcome.SUCCESS


__all__ = [
    "GoalRegion", "PlaceOutcome", "PlacementState", "ReleaseTracker", "TERMINAL_FAILURES",
    "cube_bottom_height", "cube_tilt", "evaluate_placement", "goal_region", "sample_place_goal", "spawn_region",
]
