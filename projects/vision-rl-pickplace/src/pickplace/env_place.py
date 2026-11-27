"""Gymnasium environment for learning pick-and-place with the Panda (``PickPlace-v0``).

Same scene, camera, estimator, action space and controller as ``PickLiftEnv``;
the task is to set the cube down at a goal spot on the floor.

Observation (17,), metres in the robot base frame:
    [ee_xyz(3), gripper_width(1), object_xyz(3), object_xyz - ee_xyz(3),
     goal_xyz(3), goal_xyz - object_xyz(3), lifted(1)]

    ``lifted`` says whether the cube has been lifted ``task.success_lift`` yet.
    Success depends on that history; without the flag the policy could not
    tell a cube that was carried to the goal from one that was pushed there.

Action (4,): unchanged, [dx, dy, dz, gripper].

Reward (``place.rl.reward``, math in ``rewards.py``), from a staged potential
phi(s) >= 0 that rises through reach -> grasp -> lift -> carry -> lower:
    dense (default)   -step_penalty - (phi_max - phi(s'))     every step
    potential         -step_penalty + gamma*phi(s') - phi(s)  every step
    + success_bonus, episode ends     task.evaluate_placement == SUCCESS
    - fail_penalty, episode ends      released too high, toppled, left the workspace

Dense: every step costs at least step_penalty, so hovering at the goal or
looping (re-lifting, putting the cube down and picking it up) never beats
finishing; once lifted, phi ignores the grasp and the lift height, so neither
pays anything. Being further along costs less on every following step, so value
differences scale with 1 / (1 - gamma): the same signal strength that let
pick-and-lift train in 25k steps (before the lift, the dense reward equals
PickLiftEnv's up to a constant). Potential shaping is policy-invariant and
sums to at most zero around any loop, but a value difference between two
states is only their difference in phi; it reached 0% validation success after
60k steps and is kept as an option for comparison.

The success rule needs the cube lifted first, so pushing it to the goal never
succeeds, and a release from above ``place.max_release_height`` ends the
episode with the fail penalty, so throwing does not pay either. Config
validation requires the fail penalty to exceed the discounted cost of never
succeeding from a lifted state, so failing on purpose to end the episode early
never beats trying.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import Config
from .env import PickLiftEnv
from .pipeline import PickPipeline
from .rewards import place_potential, place_step_reward
from .task import (
    TERMINAL_FAILURES, PlaceOutcome, PlacementState, ReleaseTracker, evaluate_placement,
    sample_place_goal, spawn_region,
)

OBS_DIM = 17


class PickPlaceEnv(PickLiftEnv):
    _COUPLED_FINGERS = True

    def __init__(self, cfg: Config | None = None, obs_source: str | None = None,
                 render_mode: str | None = None, gui: bool = False, curriculum: bool = False,
                 edge_goals: bool = False, object_yaw_range: list[float] | None = None):
        """
        curriculum        training only: start some episodes with the cube already picked by the
                          scripted ``pick()``, held above the pick spot or just above the goal
                          (``place.rl.curriculum``)
        edge_goals        out-of-distribution test: goals near the edge of the goal region
        object_yaw_range  override ``rl.object_yaw_range`` (out-of-distribution test)
        """
        super().__init__(cfg, obs_source=obs_source, render_mode=render_mode, gui=gui)
        if curriculum and self.obs_source != "ground_truth":
            raise ValueError("the curriculum is for training, which uses obs_source='ground_truth'")
        self.curriculum = curriculum
        self.edge_goals = edge_goals
        self.object_yaw_range = object_yaw_range
        self._region = spawn_region(self.cfg)
        self._release = ReleaseTracker(self.cfg.place.release_steps)
        self._picker = PickPipeline(self.scene, self.camera, controller=self.controller)

        ws = self.cfg.workspace
        lo, hi = self._ee_low, self._ee_high
        span = hi - lo
        base = self.observation_space
        obs_low = np.concatenate([base.low, lo - 0.1, -span - 0.1, [0.0]])
        obs_high = np.concatenate([base.high, hi + 0.1, span + 0.1, [1.0]])
        self.observation_space = spaces.Box(obs_low.astype(np.float32), obs_high.astype(np.float32), dtype=np.float32)

        self._goal = np.array([np.mean(ws.object_x), np.mean(ws.object_y), self.cfg.object.size / 2])
        self._pick_xy = self._goal[:2].copy()
        self._lifted = False          # true state: drives reward and success
        self._lifted_obs = False      # what the policy sees (estimator-based under perception)
        self._phi = 0.0
        self._outcome = PlaceOutcome.NOT_LIFTED

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        gym.Env.reset(self, seed=seed)
        self._reset_object(self.object_yaw_range)
        self._pick_xy = self.scene.object_state().position[:2].copy()
        # Drawn right after the cube, from the same generator: every controller
        # gets the same pick and place spots for a seed.
        goal_xy = sample_place_goal(self.np_random, self._pick_xy, self.cfg, edge_goals=self.edge_goals)
        self._goal = np.r_[goal_xy, self.cfg.object.size / 2]
        self.scene.add_goal_marker(goal_xy)

        self._release.reset()
        self._lifted = self._lifted_obs = False
        cur = self.cfg.place.rl.curriculum
        u = self.np_random.random() if self.curriculum else 1.0
        if u < cur.start_above_goal_prob:
            self._reset_grasped(above_goal=True)
        elif u < cur.start_above_goal_prob + cur.start_grasped_prob:
            self._reset_grasped()
        else:
            self._reset_arm()
        self._update_task_state()
        self._phi = self._potential()
        return self._observe(), self._place_info()

    def step(self, action: np.ndarray):
        self._apply_action(action)
        self._update_task_state()
        obs = self._observe()

        w = self.cfg.place.rl.reward
        phi = self._potential()
        reward = place_step_reward(w, self.cfg.place.rl.gamma, self.cfg.task.success_lift, phi, self._phi)
        self._phi = phi
        success = self._outcome is PlaceOutcome.SUCCESS
        failed = self._outcome in TERMINAL_FAILURES
        if success:
            reward += w.success_bonus
        elif failed:
            reward -= w.fail_penalty
        terminated = success or failed
        truncated = self._steps >= self.cfg.place.rl.max_episode_steps
        return obs, float(reward), terminated, truncated, self._place_info()

    # ------------------------------------------------------------- helpers
    def _reset_grasped(self, above_goal: bool = False) -> None:
        """Curriculum start: the scripted baseline picks the cube and holds it at lift height.

        With ``above_goal`` the hand then carries it to just above the goal (within
        the goal tolerance in xy, cube bottom 3-13 cm above the floor), so the policy
        meets the lowering, the release and the success reward early.
        """
        self.estimator.reset()
        self._object_estimate = self._object_prior.copy()
        result = self._picker.pick()
        self._gripper_closed = result.success
        self.controller.set_gripper(closed=self._gripper_closed)
        if result.success:
            self._update_task_state()            # latch "lifted" before lowering towards the goal
        if result.success and above_goal:
            offset = self.np_random.uniform(-1.0, 1.0, size=2) * self.cfg.place.goal_tolerance
            hand_z = self.cfg.object.size / 2 + self.np_random.uniform(0.03, 0.13)
            self.controller.move_linear(np.r_[self._goal[:2] + offset, hand_z])
        self._target = self.controller.ee_position()
        self._steps = 0

    def _update_task_state(self) -> None:
        """Latch the lifted flag, track the release and apply the shared success rule."""
        scene = self.scene
        obj = scene.object_state()
        grasped = scene.is_grasped()
        lift = float(obj.position[2] - self._object_z0)
        if grasped and lift >= self.cfg.task.success_lift:
            self._lifted = True
        if self.obs_source == "ground_truth":
            self._lifted_obs = self._lifted
        self._release.update(any(scene.finger_contacts()), scene.object_bottom_height())
        self._outcome = evaluate_placement(self._placement_state(), self.cfg.place, self._region)

    def _placement_state(self) -> PlacementState:
        scene = self.scene
        return PlacementState(
            was_lifted=self._lifted,
            release_height=self._release.release_height,
            steps_without_contact=self._release.steps_without_contact,
            object_position=scene.object_state().position,
            speed=float(np.linalg.norm(scene.object_velocity())),
            tilt=scene.object_tilt(),
            pick_xy=self._pick_xy,
            goal_xy=self._goal[:2],
        )

    def _potential(self) -> float:
        # True state, like the reward of PickLiftEnv: a training signal, not an input.
        obj = self.scene.object_state().position
        return place_potential(
            self.cfg.place.rl.reward, self.cfg.task.success_lift,
            lifted=self._lifted, grasped=self.scene.is_grasped(),
            reach_dist=float(np.linalg.norm(self.controller.ee_position() - obj)),
            lift=float(obj[2] - self._object_z0),
            goal_dist=float(np.linalg.norm(obj[:2] - self._goal[:2])),
            bottom_height=self.scene.object_bottom_height(),
        )

    def _observe(self) -> np.ndarray:
        base = super()._observe()       # under perception, this refreshes the estimate
        obj = base[4:7].astype(np.float64)
        # Under perception the flag comes from the estimator, not the simulator:
        # held (attached) and estimated at least success_lift above the floor.
        if (self.obs_source == "perception" and self.estimator.attached
                and obj[2] - self._object_prior[2] >= self.cfg.task.success_lift):
            self._lifted_obs = True
        return np.concatenate([base, self._goal, self._goal - obj, [float(self._lifted_obs)]]).astype(np.float32)

    def _place_info(self) -> dict[str, Any]:
        obj = self.scene.object_state().position
        return {
            "is_success": self._outcome is PlaceOutcome.SUCCESS,
            "outcome": self._outcome.value,
            "lifted": self._lifted,
            "grasped": self.scene.is_grasped(),
            "object_position": obj.copy(),
            "goal": self._goal.copy(),
            "pick_xy": self._pick_xy.copy(),
            "distance_to_goal": float(np.linalg.norm(obj[:2] - self._goal[:2])),
            "distance_moved": float(np.linalg.norm(obj[:2] - self._pick_xy)),
            "tilt": self.scene.object_tilt(),
            "estimate_error": (
                float(np.linalg.norm(self._object_estimate - obj)) if self.obs_source == "perception" else 0.0
            ),
        }


def register() -> None:
    """Register ``PickPlace-v0`` with Gymnasium (idempotent)."""
    if "PickPlace-v0" not in gym.registry:
        gym.register(id="PickPlace-v0", entry_point="pickplace.env_place:PickPlaceEnv")
