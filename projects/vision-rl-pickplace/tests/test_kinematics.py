import numpy as np
import pybullet as p
import pytest

from pickplace.camera import RGBDCamera
from pickplace.kinematics import PandaController
from pickplace.pipeline import Outcome, PickPipeline


@pytest.mark.parametrize("target,yaw", [([0.5, 0.1, 0.15], 0.0), ([0.6, -0.15, 0.1], 0.6), ([0.45, 0.0, 0.25], -0.7)])
def test_ik_reaches_cartesian_targets(scene, target, yaw):
    ctl = PandaController(scene)
    assert ctl.move_to(target, yaw, max_steps=200)
    pos, orn = ctl.ee_pose()
    assert np.linalg.norm(pos - target) < 0.006
    assert abs(p.getEulerFromQuaternion(orn)[2] - yaw) < np.deg2rad(4)


def test_gripper_opens_and_closes(cfg, scene):
    ctl = PandaController(scene)
    ctl.actuate_gripper(closed=True)
    assert ctl.gripper_width() < 0.005
    ctl.actuate_gripper(closed=False)
    assert ctl.gripper_width() == pytest.approx(cfg.robot.gripper_open_width, abs=0.005)


def test_baseline_picks_random_objects(scene):
    pipe = PickPipeline(scene, RGBDCamera(scene.cfg.camera, scene.client))
    rng = np.random.default_rng(7)
    outcomes = []
    for _ in range(5):
        scene.reset_robot()
        scene.randomize_object(rng)
        scene.step(20)
        outcomes.append(pipe.run().outcome)
    assert outcomes.count(Outcome.SUCCESS) == 5, outcomes


def test_baseline_reports_missing_object(scene):
    scene.place_object(np.array([5.0, 5.0]), 0.0)
    pipe = PickPipeline(scene, RGBDCamera(scene.cfg.camera, scene.client))
    assert pipe.run().outcome is Outcome.NOT_DETECTED
