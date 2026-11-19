"""Classical baseline: Camera -> Detection -> 3D localisation -> IK -> grasp.

Every stage is explicit and reports why it failed, which is what makes the
baseline useful as a reference for the learned policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .camera import RGBDCamera
from .kinematics import PandaController
from .perception import ColorObjectDetector, Detection
from .sim import Scene


class Outcome(str, Enum):
    SUCCESS = "success"
    NOT_DETECTED = "not_detected"
    OUT_OF_WORKSPACE = "out_of_workspace"
    UNREACHABLE = "unreachable"
    GRASP_FAILED = "grasp_failed"


@dataclass
class PickResult:
    outcome: Outcome
    detection: Detection | None = None
    lift_height: float = 0.0

    @property
    def success(self) -> bool:
        return self.outcome is Outcome.SUCCESS


class PickPipeline:
    def __init__(self, scene: Scene, camera: RGBDCamera, on_step: Callable[[], None] | None = None):
        self.scene = scene
        self.cfg = scene.cfg
        self.camera = camera
        self.detector = ColorObjectDetector(self.cfg.perception)
        self.controller = PandaController(scene)
        self.on_step = on_step

    def perceive(self) -> Detection | None:
        frame = self.camera.capture()
        return self.detector.detect(frame.rgb, frame.depth, self.camera.intrinsics, self.camera.T_world_cam)

    def _in_workspace(self, xyz: np.ndarray) -> bool:
        ws = self.cfg.workspace
        return bool(np.all(xyz >= np.array(ws.ee_low)) and np.all(xyz <= np.array(ws.ee_high)))

    def run(self) -> PickResult:
        g = self.cfg.grasp
        detection = self.perceive()
        if detection is None:
            return PickResult(Outcome.NOT_DETECTED)

        # The detector sees the top face; the grasp targets the cube's centre.
        centre = detection.position - np.array([0, 0, self.cfg.object.size / 2])
        grasp = centre + np.array([0, 0, g.grasp_z_offset])
        pregrasp = centre + np.array([0, 0, g.pregrasp_height])
        lift = centre + np.array([0, 0, g.lift_height])
        if not all(self._in_workspace(pt) for pt in (grasp, pregrasp, lift)):
            return PickResult(Outcome.OUT_OF_WORKSPACE, detection)

        z0 = self.scene.object_state().position[2]
        yaw = detection.yaw
        ctl, cb = self.controller, self.on_step

        ctl.actuate_gripper(closed=False, steps=5, on_step=cb)
        if not ctl.move_to(pregrasp, yaw, tolerance=0.01, on_step=cb):
            return PickResult(Outcome.UNREACHABLE, detection)
        if not ctl.move_to(grasp, yaw, on_step=cb):
            return PickResult(Outcome.UNREACHABLE, detection)
        ctl.actuate_gripper(closed=True, on_step=cb)
        ctl.move_to(lift, yaw, tolerance=0.01, on_step=cb)
        ctl.actuate_gripper(closed=True, steps=10, on_step=cb)  # hold: a slipping grasp shows here

        lifted = self.scene.object_state().position[2] - z0
        outcome = Outcome.SUCCESS if lifted >= g.success_lift and self.scene.is_grasped() else Outcome.GRASP_FAILED
        return PickResult(outcome, detection, float(lifted))
