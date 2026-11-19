"""Cartesian control of the Panda via PyBullet's damped-least-squares IK.

Two levels of interface:
    * ``command_pose``  - one non-blocking control tick (used by the RL env)
    * ``move_to``       - closed-loop until the end-effector converges (used by the baseline)
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from .config import Config
from .sim import ARM_JOINTS, EE_LINK, FINGER_JOINTS, Scene

# Top-down grasp: gripper z-axis pointing at the floor.
_DOWN_ROLL, _DOWN_PITCH = np.pi, 0.0
# With the hand pointing down, joint 7 rotates the gripper about world z.
_WRIST_JOINT = 6
_WRIST_NEUTRAL = 0.785


def top_down_orientation(yaw: float) -> tuple[float, float, float, float]:
    """Gripper pointing down. At yaw 0 the fingers close along world y."""
    return p.getQuaternionFromEuler([_DOWN_ROLL, _DOWN_PITCH, yaw])


def orientation_error(q_actual, q_target) -> float:
    """Angle in radians between two quaternions."""
    dot = abs(float(np.dot(q_actual, q_target)))
    return float(2 * np.arccos(min(1.0, dot)))


class PandaController:
    ORIENTATION_TOL = np.deg2rad(3.0)

    def __init__(self, scene: Scene):
        self.scene = scene
        self.cfg: Config = scene.cfg
        self.client = scene.client
        self.robot = scene.robot_id
        lower, upper = [], []
        for j in ARM_JOINTS:
            info = p.getJointInfo(self.robot, j, physicsClientId=self.client)
            lower.append(info[8])
            upper.append(info[9])
        self._lower, self._upper = lower, upper
        self._ranges = [u - l for l, u in zip(lower, upper)]
        # Null-space bias towards an elbow-up, wrist-down posture so IK
        # solutions stay consistent from one call to the next.
        self._rest = [0.0, 0.2, 0.0, -2.2, 0.0, 2.4, 0.785]
        self._gripper_target = self.cfg.robot.gripper_open_width / 2

    # ----------------------------------------------------------------- state
    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=True, physicsClientId=self.client)
        return np.array(state[4]), np.array(state[5])

    def ee_position(self) -> np.ndarray:
        return self.ee_pose()[0]

    def gripper_width(self) -> float:
        return float(sum(p.getJointState(self.robot, j, physicsClientId=self.client)[0] for j in FINGER_JOINTS))

    def joint_positions(self) -> np.ndarray:
        return np.array([p.getJointState(self.robot, j, physicsClientId=self.client)[0] for j in ARM_JOINTS])

    # ------------------------------------------------------------------- IK
    def solve_ik(self, position: np.ndarray, orientation, yaw: float = 0.0) -> np.ndarray:
        # The null-space rest pose would otherwise pull the wrist back to
        # neutral and the solver would only partly achieve the requested yaw.
        rest = list(self._rest)
        rest[_WRIST_JOINT] = _WRIST_NEUTRAL - yaw
        q = p.calculateInverseKinematics(
            self.robot, EE_LINK, list(position), list(orientation),
            lowerLimits=self._lower, upperLimits=self._upper,
            jointRanges=self._ranges, restPoses=rest,
            maxNumIterations=100, residualThreshold=1e-4,
            physicsClientId=self.client,
        )
        return np.array(q[:7])

    # -------------------------------------------------------------- control
    def command_pose(self, position: np.ndarray, yaw: float = 0.0) -> None:
        """Set joint targets for one control tick (does not step physics)."""
        q = self.solve_ik(position, top_down_orientation(yaw), yaw)
        p.setJointMotorControlArray(
            self.robot, ARM_JOINTS, p.POSITION_CONTROL, targetPositions=q,
            forces=[self.cfg.robot.arm_max_force] * 7, physicsClientId=self.client,
        )
        self._apply_gripper()

    def set_gripper(self, closed: bool) -> None:
        self._gripper_target = 0.0 if closed else self.cfg.robot.gripper_open_width / 2
        self._apply_gripper()

    def _apply_gripper(self) -> None:
        p.setJointMotorControlArray(
            self.robot, FINGER_JOINTS, p.POSITION_CONTROL,
            targetPositions=[self._gripper_target] * 2,
            forces=[self.cfg.robot.gripper_max_force] * 2, physicsClientId=self.client,
        )

    def move_to(self, position: np.ndarray, yaw: float = 0.0, tolerance: float | None = None,
                max_steps: int | None = None, on_step=None) -> bool:
        """Drive the end-effector to ``position``; return True if it converged."""
        tol = self.cfg.grasp.position_tolerance if tolerance is None else tolerance
        max_steps = self.cfg.grasp.max_move_steps if max_steps is None else max_steps
        position = np.asarray(position, float)
        target_orn = top_down_orientation(yaw)
        for _ in range(max_steps):
            self.command_pose(position, yaw)
            self.scene.step(self.cfg.sim.control_substeps)
            if on_step is not None:
                on_step()
            pos, orn = self.ee_pose()
            if np.linalg.norm(pos - position) < tol and orientation_error(orn, target_orn) < self.ORIENTATION_TOL:
                return True
        return False

    def actuate_gripper(self, closed: bool, steps: int | None = None, on_step=None) -> None:
        steps = self.cfg.grasp.gripper_settle_steps if steps is None else steps
        self.set_gripper(closed)
        for _ in range(steps):
            self.scene.step(self.cfg.sim.control_substeps)
            if on_step is not None:
                on_step()
