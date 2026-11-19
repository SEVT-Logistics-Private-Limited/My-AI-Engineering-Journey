"""Typed, validated configuration loaded from YAML.

Unknown keys raise immediately so a typo in a config file fails loudly
instead of silently falling back to a default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

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


__all__ = ["Config", "ConfigError", "load_config", "DEFAULT_CONFIG_PATH"]
