"""PyBullet scene: ground plane, Franka Panda and a target cube.

Each ``Scene`` owns its own physics client, so several scenes (for example
vectorised RL environments) can run in one process without interfering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data

from .config import Config

# Panda link/joint indices in pybullet_data's franka_panda/panda.urdf.
ARM_JOINTS = tuple(range(7))
FINGER_JOINTS = (9, 10)
EE_LINK = 11  # panda_grasptarget: the point midway between the fingertips


@dataclass(frozen=True)
class ObjectState:
    position: np.ndarray
    orientation: np.ndarray  # quaternion xyzw
    yaw: float


class Scene:
    def __init__(self, cfg: Config, gui: bool = False):
        self.cfg = cfg
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("failed to connect to the PyBullet physics server")
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, cfg.sim.gravity, physicsClientId=self.client)
        p.setTimeStep(cfg.sim.time_step, physicsClientId=self.client)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(1.1, 50, -35, [0.5, 0, 0.1], physicsClientId=self.client)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot_id = p.loadURDF(
            cfg.robot.urdf, cfg.robot.base_position, useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.client,
        )
        self.object_id = self._create_cube()
        self._configure_contacts()
        self.reset_robot()

    # ------------------------------------------------------------------ setup
    def _create_cube(self) -> int:
        half = self.cfg.object.size / 2
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3, physicsClientId=self.client)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half] * 3, rgbaColor=self.cfg.object.color_rgba,
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            self.cfg.object.mass, col, vis, basePosition=[0.5, 0, half],
            physicsClientId=self.client,
        )

    def _configure_contacts(self) -> None:
        friction = self.cfg.object.lateral_friction
        p.changeDynamics(
            self.object_id, -1, lateralFriction=friction, spinningFriction=0.001,
            rollingFriction=0.001, physicsClientId=self.client,
        )
        for j in FINGER_JOINTS:
            p.changeDynamics(
                self.robot_id, j, lateralFriction=friction, spinningFriction=0.001,
                physicsClientId=self.client,
            )

    # ------------------------------------------------------------------ reset
    def reset_robot(self) -> None:
        for j, q in zip(ARM_JOINTS, self.cfg.robot.home_joints):
            p.resetJointState(self.robot_id, j, q, physicsClientId=self.client)
        half_open = self.cfg.robot.gripper_open_width / 2
        for j in FINGER_JOINTS:
            p.resetJointState(self.robot_id, j, half_open, physicsClientId=self.client)
        # Hold the reset pose until a controller takes over.
        p.setJointMotorControlArray(
            self.robot_id, ARM_JOINTS, p.POSITION_CONTROL,
            targetPositions=self.cfg.robot.home_joints,
            forces=[self.cfg.robot.arm_max_force] * 7, physicsClientId=self.client,
        )
        p.setJointMotorControlArray(
            self.robot_id, FINGER_JOINTS, p.POSITION_CONTROL,
            targetPositions=[half_open] * 2,
            forces=[self.cfg.robot.gripper_max_force] * 2, physicsClientId=self.client,
        )

    def place_object(self, xy: np.ndarray, yaw: float) -> None:
        half = self.cfg.object.size / 2
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(
            self.object_id, [xy[0], xy[1], half], quat, physicsClientId=self.client,
        )
        p.resetBaseVelocity(self.object_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

    def randomize_object(self, rng: np.random.Generator, yaw_range: list[float] | None = None) -> ObjectState:
        ws = self.cfg.workspace
        yaw_range = self.cfg.object.yaw_range if yaw_range is None else yaw_range
        xy = np.array([rng.uniform(*ws.object_x), rng.uniform(*ws.object_y)])
        yaw = float(rng.uniform(*yaw_range))
        self.place_object(xy, yaw)
        return self.object_state()

    # ------------------------------------------------------------------ state
    def object_state(self) -> ObjectState:
        pos, orn = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return ObjectState(np.array(pos), np.array(orn), float(yaw))

    def finger_contacts(self) -> tuple[bool, bool]:
        """Whether each finger is touching the object."""
        return tuple(
            bool(p.getContactPoints(self.robot_id, self.object_id, linkIndexA=j, physicsClientId=self.client))
            for j in FINGER_JOINTS
        )

    def is_grasped(self) -> bool:
        return all(self.finger_contacts())

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            p.stepSimulation(physicsClientId=self.client)

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def __enter__(self) -> "Scene":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
