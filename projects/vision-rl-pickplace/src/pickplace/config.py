"""Typed, validated configuration loaded from YAML.

Unknown keys raise immediately so a typo in a config file fails loudly
instead of silently falling back to a default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import yaml

from .rewards import place_failure_threshold

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True)
class SimConfig:
    time_step: float
    control_substeps: int
    gravity: float


@dataclass(frozen=True)
class RobotConfig:
    urdf: str
    base_position: list[float]
    home_joints: list[float]
    arm_max_force: float
    gripper_max_force: float
    gripper_open_width: float


@dataclass(frozen=True)
class ObjectConfig:
    size: float
    mass: float
    color_rgba: list[float]
    lateral_friction: float
    yaw_range: list[float]


@dataclass(frozen=True)
class WorkspaceConfig:
    object_x: list[float]
    object_y: list[float]
    ee_low: list[float]
    ee_high: list[float]


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fov_deg: float
    near: float
    far: float
    eye: list[float]
    target: list[float]
    up: list[float]


@dataclass(frozen=True)
class PerceptionConfig:
    hsv_ranges: list[list[list[int]]]
    min_area_px: int
    top_face_tolerance: float
    min_area_ratio: float
    morph_kernel: int


@dataclass(frozen=True)
class TaskConfig:
    success_lift: float


@dataclass(frozen=True)
class GraspConfig:
    pregrasp_height: float
    lift_height: float
    grasp_z_offset: float
    position_tolerance: float
    max_move_steps: int
    gripper_settle_steps: int


@dataclass(frozen=True)
class RewardConfig:
    distance_weight: float
    grasp_bonus: float
    lift_weight: float
    success_bonus: float
    step_penalty: float


@dataclass(frozen=True)
class SACConfig:
    learning_rate: float
    buffer_size: int
    batch_size: int
    gamma: float
    tau: float
    learning_starts: int
    net_arch: list[int]


@dataclass(frozen=True)
class RLConfig:
    algorithm: str
    total_timesteps: int
    max_episode_steps: int
    action_scale: float
    obs_source: str
    object_yaw_range: list[float]
    reward: RewardConfig
    sac: SACConfig
    seed: int


@dataclass(frozen=True)
class PlaceRewardConfig:
    shaping: str
    step_penalty: float
    reach_weight: float
    grasp_bonus: float
    lift_weight: float
    goal_weight: float
    lower_weight: float
    lower_radius: float
    distance_ref: float
    lower_ref: float
    success_bonus: float
    fail_penalty: float


@dataclass(frozen=True)
class CurriculumConfig:
    start_grasped_prob: float


@dataclass(frozen=True)
class PlaceRLConfig:
    max_episode_steps: int
    total_timesteps: int
    gamma: float
    reward: PlaceRewardConfig
    curriculum: CurriculumConfig


@dataclass(frozen=True)
class PlaceConfig:
    goal_mode: str
    min_distance: float
    goal_tolerance: float
    max_release_height: float
    max_tilt: float
    rest_speed: float
    release_steps: int
    edge_band: float
    hover_height: float
    place_z_offset: float
    carry_step: float
    settle_steps: int
    rl: PlaceRLConfig


@dataclass(frozen=True)
class Config:
    sim: SimConfig
    robot: RobotConfig
    object: ObjectConfig
    workspace: WorkspaceConfig
    camera: CameraConfig
    perception: PerceptionConfig
    task: TaskConfig
    grasp: GraspConfig
    rl: RLConfig
    place: PlaceConfig

    def replace(self, **sections: Any) -> "Config":
        """Return a copy with whole sections swapped (configs are immutable)."""
        return dataclasses.replace(self, **sections)


class ConfigError(ValueError):
    """Raised when a config file is missing keys, has unknown keys or bad values."""


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    missing = names - set(data)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    if missing:
        raise ConfigError(f"{path}: missing keys {sorted(missing)}")
    kwargs = {}
    for name in names:
        hint = hints[name]
        value = data[name]
        kwargs[name] = _build(hint, value, f"{path}.{name}") if is_dataclass(hint) else value
    return cls(**kwargs)


def _validate(cfg: Config) -> None:
    def check(cond: bool, msg: str) -> None:
        if not cond:
            raise ConfigError(msg)

    check(len(cfg.robot.home_joints) == 7, "robot.home_joints must have 7 values")
    check(cfg.sim.control_substeps >= 1, "sim.control_substeps must be >= 1")
    for axis in ("object_x", "object_y"):
        lo, hi = getattr(cfg.workspace, axis)
        check(lo < hi, f"workspace.{axis} must be [low, high] with low < high")
    check(
        all(lo < hi for lo, hi in zip(cfg.workspace.ee_low, cfg.workspace.ee_high)),
        "workspace.ee_low must be below workspace.ee_high on every axis",
    )
    check(0 < cfg.camera.near < cfg.camera.far, "camera: need 0 < near < far")
    check(0 < cfg.camera.fov_deg < 180, "camera.fov_deg must be in (0, 180)")
    check(0 < cfg.perception.min_area_ratio <= 1, "perception.min_area_ratio must be in (0, 1]")
    check(cfg.rl.obs_source in ("ground_truth", "perception"), "rl.obs_source invalid")
    check(cfg.rl.algorithm in ("SAC", "PPO"), "rl.algorithm must be SAC or PPO")
    check(cfg.rl.action_scale > 0, "rl.action_scale must be positive")
    check(0 < cfg.task.success_lift < cfg.grasp.lift_height,
          "task.success_lift must be positive and below grasp.lift_height, "
          "or the scripted baseline could never succeed")
    w = cfg.rl.reward
    max_shaped = w.grasp_bonus + w.lift_weight * cfg.task.success_lift
    check(
        w.step_penalty > max_shaped,
        f"rl.reward.step_penalty ({w.step_penalty}) must exceed the max per-step shaping "
        f"({max_shaped:.2f}); otherwise the agent learns to hover instead of finishing",
    )
    _validate_place(cfg, check)


def _validate_place(cfg: Config, check) -> None:
    pl, ws = cfg.place, cfg.workspace
    check(pl.goal_mode in ("target", "any"), "place.goal_mode must be target or any")
    check(pl.min_distance > 0, "place.min_distance must be positive")
    check(0 < pl.goal_tolerance < pl.min_distance,
          "place.goal_tolerance must be positive and below place.min_distance")
    # Goals are sampled at least goal_tolerance inside the spawn region, so any cube
    # within tolerance of its goal is inside the region too.
    gx = ws.object_x[1] - ws.object_x[0] - 2 * pl.goal_tolerance
    gy = ws.object_y[1] - ws.object_y[0] - 2 * pl.goal_tolerance
    check(gx > 0 and gy > 0, "place.goal_tolerance leaves no room for goals inside the spawn region")
    # From a pick spot at the region centre, the farthest goal is a corner: half the diagonal.
    check(0.5 * float(np.hypot(gx, gy)) > pl.min_distance + pl.goal_tolerance,
          "a goal at place.min_distance + place.goal_tolerance must fit inside the spawn region "
          "from every pick spot")
    check(0 < pl.edge_band < 0.5 * min(gx, gy), "place.edge_band must be positive and below half the goal region")
    check(0 < pl.max_release_height < pl.hover_height, "place.max_release_height must be in (0, hover_height)")
    check(0 <= pl.place_z_offset < pl.max_release_height,
          "place.place_z_offset must be below place.max_release_height, or the baseline drops the cube")
    check(0 < pl.max_tilt < np.pi / 4, "place.max_tilt must be in (0, pi/4)")
    check(pl.rest_speed > 0, "place.rest_speed must be positive")
    check(pl.release_steps >= 1, "place.release_steps must be >= 1")
    check(pl.carry_step > 0, "place.carry_step must be positive")
    check(pl.settle_steps >= 0, "place.settle_steps must be >= 0")
    check(cfg.workspace.ee_low[2] < pl.hover_height < cfg.grasp.lift_height,
          "place.hover_height must be inside the workspace and below grasp.lift_height")

    rl = pl.rl
    w = rl.reward
    check(rl.max_episode_steps >= 1 and rl.total_timesteps >= 1, "place.rl step counts must be positive")
    check(0 < rl.gamma < 1, "place.rl.gamma must be in (0, 1)")
    check(0 <= rl.curriculum.start_grasped_prob <= 1, "place.rl.curriculum.start_grasped_prob must be in [0, 1]")
    check(w.shaping in ("dense", "potential"), "place.rl.reward.shaping must be dense or potential")
    check(min(w.reach_weight, w.grasp_bonus, w.lift_weight, w.goal_weight, w.lower_weight,
              w.success_bonus, w.fail_penalty) >= 0, "place.rl.reward weights must be non-negative")
    check(min(w.lower_radius, w.distance_ref, w.lower_ref) > 0,
          "place.rl.reward.lower_radius, distance_ref and lower_ref must be positive")
    # Same rule as pick-and-lift: standing still must cost reward. Dense: every step pays
    # -step_penalty - (phi_max - phi) <= -step_penalty. Potential: standing still pays
    # -step_penalty + (gamma - 1) * phi <= -step_penalty, since phi >= 0.
    check(w.step_penalty > 0,
          "place.rl.reward.step_penalty must be positive; otherwise the agent learns to hover instead of finishing")
    # Terminal failures (dropped, toppled, left the workspace) need a lifted cube. From any
    # lifted state, failing must cost more than never succeeding, discounted.
    threshold = place_failure_threshold(w, rl.gamma, cfg.task.success_lift)
    check(
        w.fail_penalty > threshold,
        f"place.rl.reward.fail_penalty ({w.fail_penalty}) must exceed {threshold:.1f}, the discounted cost "
        f"of never succeeding from a lifted state; otherwise ending the episode by failing beats trying",
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a YAML config. Defaults to ``config/default.yaml``."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh)
    cfg = _build(Config, data, "config")
    _validate(cfg)
    return cfg


__all__ = ["Config", "ConfigError", "PlaceConfig", "load_config", "DEFAULT_CONFIG_PATH"]
