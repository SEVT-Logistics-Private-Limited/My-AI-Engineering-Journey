"""Gymnasium environment for learning pick-and-lift with the Panda.

Observation (10,), all in metres in the robot base frame:
    [ee_xyz(3), gripper_width(1), object_xyz(3), object_xyz - ee_xyz(3)]

Action (4,) in [-1, 1]:
    [dx, dy, dz, gripper]  - dx..dz scaled by ``rl.action_scale``;
                             gripper > 0 closes, <= 0 opens.

Cartesian deltas + IK keep the action space low-dimensional and the policy
independent of the arm's joint configuration; learning joint-space control
from scratch is far less sample-efficient.

Reward (dense, staged):
    - step_penalty                             always (see below)
    - distance_weight * ||ee - object||        always (reach)
    + grasp_bonus                              while both fingers touch the object
    + lift_weight * clip(lift, 0, success_lift) while grasped (lift)
    + success_bonus                            once, and the episode terminates

The step penalty is larger than the best possible shaped reward per step.
Without it the first training run reward-hacked: hovering with the cube held
just below the success height collected grasp/lift bonuses forever, worth
more (discounted) than the one-time success bonus.

``obs_source`` chooses where object_xyz comes from:
    ground_truth  simulator state (fast; used for training)
    perception    camera + detector + tracker (slow; tests robustness to estimation error)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .camera import RGBDCamera, render_presentation
from .config import Config, load_config
from .kinematics import PandaController
from .perception import ColorObjectDetector, ObjectTracker
from .sim import Scene

OBS_DIM = 10


class PickLiftEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}

    def __init__(self, cfg: Config | None = None, obs_source: str | None = None,
                 render_mode: str | None = None, gui: bool = False):
        super().__init__()
        self.cfg = cfg or load_config()
        self.obs_source = obs_source or self.cfg.rl.obs_source
        if self.obs_source not in ("ground_truth", "perception"):
            raise ValueError(f"unknown obs_source {self.obs_source!r}")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        self.scene = Scene(self.cfg, gui=gui)
        self.controller = PandaController(self.scene)
        self.camera = RGBDCamera(self.cfg.camera, self.scene.client)
        self.detector = ColorObjectDetector(self.cfg.perception)
        self.tracker = ObjectTracker(self.cfg.perception.min_area_px)

        ws = self.cfg.workspace
        self._ee_low = np.array(ws.ee_low)
        self._ee_high = np.array(ws.ee_high)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        obs_low = np.concatenate([self._ee_low, [0.0], self._ee_low, self._ee_low - self._ee_high]) - 0.1
        obs_high = np.concatenate([self._ee_high, [0.1], self._ee_high, self._ee_high - self._ee_low]) + 0.1
        self.observation_space = spaces.Box(obs_low.astype(np.float32), obs_high.astype(np.float32), dtype=np.float32)

        self._target = np.zeros(3)
        self._object_z0 = 0.0
        self._steps = 0
        self._object_estimate = np.zeros(3)

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        rl = self.cfg.rl
        self.scene.reset_robot()
        self.scene.randomize_object(self.np_random, yaw_range=rl.object_yaw_range)
        self.scene.step(10)  # let the cube settle on the floor
        self._object_z0 = float(self.scene.object_state().position[2])

        # Start above a random point in the workspace so the policy must search.
        ws = self.cfg.workspace
        start = np.array([
            self.np_random.uniform(*ws.object_x),
            self.np_random.uniform(*ws.object_y),
            self.np_random.uniform(0.15, 0.25),
        ])
        # Perception needs a clear view before the arm moves into the frame.
        self.tracker.reset()
        if self.obs_source == "perception":
            self._update_estimate()
        self.controller.set_gripper(closed=False)
        self.controller.move_to(start, tolerance=0.01, max_steps=80)
        self._target = self.controller.ee_position()
        self._steps = 0
        return self._observe(), self._info()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (4,):
            raise ValueError(f"action must have shape (4,), got {action.shape}")
        rl = self.cfg.rl

        # Integrate on the commanded target, not the measured pose, so small
        # actions are not swallowed by tracking error.
        self._target = np.clip(self._target + action[:3] * rl.action_scale, self._ee_low, self._ee_high)
        self.controller.set_gripper(closed=bool(action[3] > 0))
        self.controller.command_pose(self._target)
        self.scene.step(self.cfg.sim.control_substeps)
        self._steps += 1

        obs = self._observe()
        reward, success = self._reward()
        terminated = success
        truncated = self._steps >= rl.max_episode_steps
        return obs, reward, terminated, truncated, self._info(success)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self.render_frame()

    def close(self):
        self.scene.close()

    # ------------------------------------------------------------- helpers
    def render_frame(self, width: int = 480, height: int = 360) -> np.ndarray:
        return render_presentation(self.scene.client, width, height)

    def _update_estimate(self) -> None:
        frame = self.camera.capture()
        det = self.detector.detect(frame.rgb, frame.depth, self.camera.intrinsics, self.camera.T_world_cam)
        est = self.tracker.update(det)
        if est is not None:
            # Detector reports the top face; the policy was trained on the centre.
            self._object_estimate = est.position - np.array([0, 0, self.cfg.object.size / 2])

    def _object_position(self) -> np.ndarray:
        if self.obs_source == "ground_truth":
            return self.scene.object_state().position
        self._update_estimate()
        return self._object_estimate

    def _observe(self) -> np.ndarray:
        ee = self.controller.ee_position()
        obj = self._object_position()
        width = self.controller.gripper_width()
        return np.concatenate([ee, [width], obj, obj - ee]).astype(np.float32)

    def _reward(self) -> tuple[float, bool]:
        # Reward always uses true state: it is a training signal, not an input.
        w = self.cfg.rl.reward
        ee = self.controller.ee_position()
        obj = self.scene.object_state().position
        grasped = self.scene.is_grasped()
        lift = float(obj[2] - self._object_z0)

        reward = -w.step_penalty - w.distance_weight * float(np.linalg.norm(ee - obj))
        if grasped:
            reward += w.grasp_bonus + w.lift_weight * float(np.clip(lift, 0.0, self.cfg.rl.success_lift))
        success = grasped and lift >= self.cfg.rl.success_lift
        if success:
            reward += w.success_bonus
        return reward, success

    def _info(self, success: bool = False) -> dict[str, Any]:
        obj = self.scene.object_state().position
        return {
            "is_success": success,
            "lift": float(obj[2] - self._object_z0),
            "grasped": self.scene.is_grasped(),
            "object_position": obj.copy(),
        }


def register() -> None:
    """Register ``PickLift-v0`` with Gymnasium (idempotent)."""
    if "PickLift-v0" not in gym.registry:
        gym.register(id="PickLift-v0", entry_point="pickplace.env:PickLiftEnv")
