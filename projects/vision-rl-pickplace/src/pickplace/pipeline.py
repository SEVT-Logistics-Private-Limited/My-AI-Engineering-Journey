"""Classical baseline: Camera -> Detection -> 3D localisation -> IK -> grasp.

``PickPipeline`` picks and lifts; ``PickPlacePipeline`` then carries the cube,
sets it down at a goal spot, retreats and checks the result with the camera.

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
from .task import (
    PlaceOutcome, PlacementState, ReleaseTracker, evaluate_placement, spawn_region,
)


class Outcome(str, Enum):
    SUCCESS = "success"
    NOT_DETECTED = "not_detected"
    OUT_OF_WORKSPACE = "out_of_workspace"
    UNREACHABLE = "unreachable"
    GRASP_FAILED = "grasp_failed"
    # Pick-and-place stages
    PLACE_UNREACHABLE = "place_unreachable"
    DROPPED_IN_TRANSIT = "dropped_in_transit"
    # Placement check (same values as task.PlaceOutcome)
    NOT_LIFTED = PlaceOutcome.NOT_LIFTED.value
    NOT_RELEASED = PlaceOutcome.NOT_RELEASED.value
    RELEASED_TOO_HIGH = PlaceOutcome.RELEASED_TOO_HIGH.value
    NOT_AT_REST = PlaceOutcome.NOT_AT_REST.value
    TOPPLED = PlaceOutcome.TOPPLED.value
    LEFT_WORKSPACE = PlaceOutcome.LEFT_WORKSPACE.value
    TOO_CLOSE = PlaceOutcome.TOO_CLOSE.value
    MISSED_GOAL = PlaceOutcome.MISSED_GOAL.value


@dataclass
class PickResult:
    outcome: Outcome
    detection: Detection | None = None
    lift_height: float = 0.0

    @property
    def success(self) -> bool:
        return self.outcome is Outcome.SUCCESS


class PickPipeline:
    def __init__(self, scene: Scene, camera: RGBDCamera, on_step: Callable[[], None] | None = None,
                 controller: PandaController | None = None):
        self.scene = scene
        self.cfg = scene.cfg
        self.camera = camera
        self.detector = ColorObjectDetector(self.cfg.perception)
        # Callers that already drive the arm (the RL env's curriculum reset) pass
        # theirs, so the gripper state stays consistent.
        self.controller = controller or PandaController(scene)
        self.on_step = on_step

    def perceive(self) -> Detection | None:
        frame = self.camera.capture()
        return self.detector.detect(frame.rgb, frame.depth, self.camera.intrinsics, self.camera.T_world_cam)

    def _in_workspace(self, xyz: np.ndarray) -> bool:
        ws = self.cfg.workspace
        return bool(np.all(xyz >= np.array(ws.ee_low)) and np.all(xyz <= np.array(ws.ee_high)))

    def run(self) -> PickResult:
        return self.pick()

    def pick(self) -> PickResult:
        """Detect, grasp and lift. Ends holding the cube at lift height on success."""
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
        # Same success rule as the RL environment: held, and lifted at least task.success_lift.
        held = self.scene.is_grasped()
        outcome = Outcome.SUCCESS if held and lifted >= self.cfg.task.success_lift else Outcome.GRASP_FAILED
        return PickResult(outcome, detection, float(lifted))


@dataclass
class PickPlaceResult:
    outcome: Outcome
    pick: PickResult
    goal_xy: np.ndarray
    pick_xy: np.ndarray
    final_position: np.ndarray | None = None
    distance_to_goal: float = float("nan")    # metres, cube centre (xy) to goal
    distance_moved: float = float("nan")      # metres, cube centre (xy) to pick spot
    tilt: float = float("nan")                # radians
    release_height: float | None = None
    # Phase 3: the camera's view of the result, taken with the arm back home.
    camera_position: np.ndarray | None = None
    camera_error: float = float("nan")        # metres, detected vs true cube centre (xy)
    camera_goal_error: float = float("nan")   # metres, detected cube centre (xy) vs goal

    @property
    def success(self) -> bool:
        return self.outcome is Outcome.SUCCESS


class PickPlacePipeline(PickPipeline):
    """Pick, carry, place, retreat, check.

    Expects a ``Scene`` built with ``coupled_fingers=True``: with independent
    finger motors the held cube drifts between the fingers by up to 2 cm while
    carried, which showed as a systematic 18 mm placement error.

    Control uses only what a real robot has (the camera detection made before
    the arm moves, joint encoders, the commanded grasp). Simulator state is read
    only to score the outcome with the shared rule in ``task.py``.
    """

    def __init__(self, scene: Scene, camera: RGBDCamera, on_step: Callable[[], None] | None = None,
                 controller: PandaController | None = None):
        if not scene.coupled_fingers:
            raise ValueError("PickPlacePipeline needs Scene(..., coupled_fingers=True)")
        super().__init__(scene, camera, on_step, controller)

    def run(self, goal_xy: np.ndarray) -> PickPlaceResult:  # type: ignore[override]
        goal_xy = np.asarray(goal_xy, float)[:2]
        pick_xy = self.scene.object_state().position[:2].copy()
        picked = self.pick()
        result = PickPlaceResult(picked.outcome, picked, goal_xy, pick_xy)
        if not picked.success:
            return result
        # Keep the grasp yaw; the placed cube's rotation does not matter.
        return self.place(goal_xy, result, yaw=picked.detection.yaw)

    def place(self, goal_xy: np.ndarray, result: PickPlaceResult, yaw: float = 0.0) -> PickPlaceResult:
        """Carry the held cube to ``goal_xy``, set it down, retreat and check.

        Starts from whatever hold the arm is in, so any pick can precede it: the
        scripted ``pick()`` or a learned policy (``evaluate.py --controller hybrid``).
        """
        cfg, pl = self.cfg, self.cfg.place
        ctl, cb, scene = self.controller, self.on_step, self.scene
        goal_xy = np.asarray(goal_xy, float)[:2]
        tracker = ReleaseTracker(pl.release_steps)

        def track() -> None:
            tracker.update(any(scene.finger_contacts()), scene.object_bottom_height())
            if cb is not None:
                cb()

        # Carry no lower than the hover height (a learned pick may stop just above
        # the success height). The hand holds the cube centre grasp_z_offset below the grasp point.
        carry_z = max(ctl.ee_position()[2], pl.hover_height)
        release_z = cfg.object.size / 2 + pl.place_z_offset - cfg.grasp.grasp_z_offset
        waypoints = [np.r_[goal_xy, z] for z in (carry_z, pl.hover_height, release_z)]
        if not all(self._in_workspace(pt) for pt in waypoints):
            result.outcome = Outcome.PLACE_UNREACHABLE
            return result

        def grasp_lost() -> bool:
            return not scene.is_grasped()

        # Carry at lift height, lower to hover height, then to the release height.
        for target in waypoints:
            reached = ctl.move_linear(target, yaw, on_step=track, should_abort=grasp_lost)
            if grasp_lost():
                result.outcome = Outcome.DROPPED_IN_TRANSIT
                return self._score(result, tracker)
            if not reached:
                result.outcome = Outcome.PLACE_UNREACHABLE
                return result

        ctl.actuate_gripper(closed=False, on_step=track)
        # Straight up first so the fingers do not drag the cube, then home (out of view).
        ctl.move_linear(np.r_[ctl.ee_position()[:2], pl.hover_height], yaw, on_step=track)
        ctl.move_home(on_step=track)
        for _ in range(pl.settle_steps):
            scene.step(cfg.sim.control_substeps)
            track()
        self._score(result, tracker)
        self._camera_check(result)
        return result

    def _score(self, result: PickPlaceResult, tracker: ReleaseTracker) -> PickPlaceResult:
        scene = self.scene
        obj = scene.object_state()
        state = PlacementState(
            was_lifted=True,              # place() runs only after a successful pick()
            release_height=tracker.release_height,
            steps_without_contact=tracker.steps_without_contact,
            object_position=obj.position,
            speed=float(np.linalg.norm(scene.object_velocity())),
            tilt=scene.object_tilt(),
            pick_xy=result.pick_xy,
            goal_xy=result.goal_xy,
        )
        placed = evaluate_placement(state, self.cfg.place, spawn_region(self.cfg))
        if result.outcome is Outcome.SUCCESS:      # stage failures (e.g. dropped) take precedence
            result.outcome = Outcome(placed.value)
        result.final_position = obj.position.copy()
        result.distance_to_goal = float(np.linalg.norm(obj.position[:2] - result.goal_xy))
        result.distance_moved = float(np.linalg.norm(obj.position[:2] - result.pick_xy))
        result.tilt = state.tilt
        result.release_height = tracker.release_height
        return result

    def _camera_check(self, result: PickPlaceResult) -> None:
        """One photo with the arm home: the robot checks its own result."""
        detection = self.perceive()
        if detection is None or result.final_position is None:
            return
        centre = detection.position - np.array([0, 0, self.cfg.object.size / 2])
        result.camera_position = centre
        result.camera_error = float(np.linalg.norm(centre[:2] - result.final_position[:2]))
        result.camera_goal_error = float(np.linalg.norm(centre[:2] - result.goal_xy))
