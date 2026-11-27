import numpy as np
import pytest

from pickplace.camera import RGBDCamera
from pickplace.kinematics import PandaController
from pickplace.perception import ColorObjectDetector
from pickplace.pipeline import Outcome, PickPlacePipeline
from pickplace.sim import Scene
from pickplace.task import PlaceOutcome, sample_place_goal


@pytest.fixture
def coupled_scene(cfg):
    s = Scene(cfg, coupled_fingers=True)
    yield s
    s.close()


def _episode(scene, seed):
    rng = np.random.default_rng(seed)
    scene.reset()
    state = scene.randomize_object(rng)
    goal = sample_place_goal(rng, state.position[:2], scene.cfg)
    scene.add_goal_marker(goal)
    scene.step(20)
    return goal


def test_baseline_places_on_fixed_seeds(cfg, coupled_scene):
    pipe = PickPlacePipeline(coupled_scene, RGBDCamera(cfg.camera, coupled_scene.client))
    for seed in (20000, 20001, 20002):
        goal = _episode(coupled_scene, seed)
        r = pipe.run(goal)
        assert r.outcome is Outcome.SUCCESS, (seed, r.outcome)
        assert r.distance_to_goal < cfg.place.goal_tolerance
        assert r.distance_moved >= cfg.place.min_distance
        assert r.release_height <= cfg.place.max_release_height
        # The camera check, taken with the arm home, agrees with the simulator.
        assert r.camera_error < 0.005


def test_unreachable_goal_is_reported(cfg, coupled_scene):
    pipe = PickPlacePipeline(coupled_scene, RGBDCamera(cfg.camera, coupled_scene.client))
    _episode(coupled_scene, 20000)
    r = pipe.run(np.array([1.5, 0.0]))
    assert r.outcome is Outcome.PLACE_UNREACHABLE


def test_place_pipeline_needs_coupled_fingers(cfg, scene):
    with pytest.raises(ValueError, match="coupled_fingers"):
        PickPlacePipeline(scene, RGBDCamera(cfg.camera, scene.client))


def test_move_linear_limits_step_size(cfg, scene):
    ctl = PandaController(scene)
    assert ctl.move_to([0.5, 0.0, 0.2])
    positions = []
    assert ctl.move_linear([0.5, 0.15, 0.2], max_step=0.01, on_step=lambda: positions.append(ctl.ee_position()))
    steps = np.linalg.norm(np.diff(np.array(positions), axis=0), axis=1)
    assert steps.max() < 0.02          # tracking lag allows a little over max_step, never a jump
    assert np.linalg.norm(ctl.ee_position() - [0.5, 0.15, 0.2]) < cfg.grasp.position_tolerance


def test_move_linear_aborts(scene):
    ctl = PandaController(scene)
    calls = []
    assert not ctl.move_linear([0.5, 0.1, 0.2], should_abort=lambda: calls.append(1) or len(calls) >= 3)
    assert len(calls) == 3


def test_goal_marker_is_invisible_to_the_detector_and_to_physics(cfg, scene):
    """The green marker under the cube changes neither detection nor the cube's resting pose."""
    cam = RGBDCamera(cfg.camera, scene.client)
    det = ColorObjectDetector(cfg.perception)
    xy = np.array([0.6, 0.1])

    def measure(with_marker: bool):
        scene.reset()
        scene.place_object(xy, 0.3)
        if with_marker:
            scene.add_goal_marker(xy)
        scene.step(30)
        frame = cam.capture()
        return det.detect(frame.rgb, frame.depth, cam.intrinsics, cam.T_world_cam), scene.object_state()

    (d0, s0), (d1, s1) = measure(False), measure(True)
    np.testing.assert_allclose(d1.position, d0.position, atol=1e-6)
    np.testing.assert_allclose(s1.position, s0.position, atol=1e-9)


def test_lift_scene_fingers_stay_independent(cfg, scene):
    """Pick-and-lift keeps the original gripper model (its results must not change)."""
    assert not scene.coupled_fingers


def test_pipeline_outcomes_cover_every_placement_outcome():
    for outcome in PlaceOutcome:
        assert Outcome(outcome.value).value == outcome.value
