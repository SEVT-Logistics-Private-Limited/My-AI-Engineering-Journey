import numpy as np
import pytest

from pickplace.camera import Intrinsics, RGBDCamera
from pickplace.perception import ColorObjectDetector, Detection, ObjectStateEstimator, deproject
from pickplace.utils import transform_points, wrap_to_quarter_turn


def test_deproject_principal_point():
    K = Intrinsics(fx=300, fy=300, cx=160, cy=120, width=320, height=240)
    np.testing.assert_allclose(deproject(160, 120, 0.7, K), [0, 0, 0.7])
    np.testing.assert_allclose(deproject(460, 120, 1.0, K), [1.0, 0, 1.0])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_deproject_rejects_invalid_depth(bad):
    K = Intrinsics(300, 300, 160, 120, 320, 240)
    with pytest.raises(ValueError):
        deproject(10, 10, bad, K)


def test_camera_extrinsics_project_world_origin_to_image_centre(cfg, scene):
    cam = RGBDCamera(cfg.camera, scene.client)
    target = np.array(cfg.camera.target)
    p_cam = transform_points(np.linalg.inv(cam.T_world_cam), target)
    K = cam.intrinsics
    u, v = K.fx * p_cam[0] / p_cam[2] + K.cx, K.fy * p_cam[1] / p_cam[2] + K.cy
    assert abs(u - K.cx) < 1e-6 and abs(v - K.cy) < 1e-6


def test_metric_depth_matches_floor_distance(cfg, scene):
    scene.place_object(np.array([5.0, 5.0]), 0.0)  # move the cube out of view
    frame = RGBDCamera(cfg.camera, scene.client).capture()
    centre = frame.depth[cfg.camera.height // 2, cfg.camera.width // 2]
    assert centre == pytest.approx(cfg.camera.eye[2], abs=2e-3)


def test_no_detection_on_empty_scene(cfg, scene):
    scene.place_object(np.array([5.0, 5.0]), 0.0)
    cam = RGBDCamera(cfg.camera, scene.client)
    f = cam.capture()
    assert ColorObjectDetector(cfg.perception).detect(f.rgb, f.depth, cam.intrinsics, cam.T_world_cam) is None


def test_localisation_accuracy(cfg, scene):
    cam = RGBDCamera(cfg.camera, scene.client)
    det = ColorObjectDetector(cfg.perception)
    rng = np.random.default_rng(42)
    for _ in range(15):
        scene.randomize_object(rng)
        scene.step(30)
        truth = scene.object_state()
        f = cam.capture()
        d = det.detect(f.rgb, f.depth, cam.intrinsics, cam.T_world_cam)
        assert d is not None
        top = truth.position + [0, 0, cfg.object.size / 2]
        assert np.linalg.norm(d.position - top) < 0.005          # < 5 mm
        assert abs(wrap_to_quarter_turn(d.yaw - truth.yaw)) < np.deg2rad(6)


SIZE = 0.04


def _det(xyz_top, area):
    return Detection((0.0, 0.0), np.asarray(xyz_top, float), 0.0, area, np.zeros((1, 1), bool))


def _estimator():
    return ObjectStateEstimator(object_size=SIZE, min_area_px=10, min_area_ratio=0.8)


def test_estimator_converts_top_face_to_centre_and_holds_when_unseen():
    est = _estimator()
    far_ee, open_w = np.array([0.5, 0.0, 0.3]), 0.08
    pos = est.update(_det([0.5, 0.1, SIZE], 300), far_ee, open_w, gripper_closed=False)
    np.testing.assert_allclose(pos, [0.5, 0.1, SIZE / 2])
    held = est.update(None, far_ee, open_w, gripper_closed=False)
    np.testing.assert_allclose(held, pos)
    assert est.frames_since_detection == 1


def test_estimator_rejects_partially_occluded_detection():
    est = _estimator()
    ee = np.array([0.5, 0.0, 0.3])
    est.update(_det([0.5, 0.1, SIZE], 300), ee, 0.08, False)
    pos = est.update(_det([0.52, 0.1, SIZE], 150), ee, 0.08, False)   # half the cube hidden
    np.testing.assert_allclose(pos, [0.5, 0.1, SIZE / 2])
    pos = est.update(_det([0.51, 0.1, SIZE], 280), ee, 0.08, False)   # mostly visible again
    np.testing.assert_allclose(pos, [0.51, 0.1, SIZE / 2])


def test_estimator_follows_end_effector_while_holding_and_drops_on_release():
    est = _estimator()
    est.update(_det([0.5, 0.1, SIZE], 300), np.array([0.5, 0.0, 0.3]), 0.08, False)
    grasp_ee = np.array([0.505, 0.1, 0.02])
    est.update(None, grasp_ee, gripper_width=SIZE, gripper_closed=True)   # first contact
    assert not est.attached                                               # not yet confirmed
    est.update(None, grasp_ee, gripper_width=SIZE, gripper_closed=True)   # fingers settled
    assert est.attached
    lifted = est.update(None, grasp_ee + [0.0, 0.0, 0.1], SIZE, True)
    np.testing.assert_allclose(lifted, [0.5, 0.1, SIZE / 2 + 0.1])     # rigid offset kept
    released = est.update(None, grasp_ee + [0.0, 0.0, 0.1], 0.08, False)
    assert not est.attached
    np.testing.assert_allclose(released, [0.5, 0.1, SIZE / 2])         # falls back onto the table


@pytest.mark.parametrize("ee,widths", [
    ([0.5, 0.1, 0.25], [SIZE, SIZE]),                  # closed on something far from the object
    ([0.5, 0.1, 0.02], [0.002, 0.002]),                # at the object, closed on nothing
    ([0.5, 0.1, 0.02], [0.65 * SIZE, 0.65 * SIZE]),    # pinching a corner: not a stable grasp
    ([0.5, 0.1, 0.02], [0.9 * SIZE, 1.1 * SIZE]),      # fingers still moving: slipping
])
def test_estimator_does_not_attach_without_evidence_of_a_grasp(ee, widths):
    est = _estimator()
    est.update(_det([0.5, 0.1, SIZE], 300), np.array([0.5, 0.0, 0.3]), 0.08, False)
    for w in widths:
        est.update(None, np.array(ee), w, gripper_closed=True)
    assert not est.attached


def test_estimator_validates_ratio():
    with pytest.raises(ValueError):
        ObjectStateEstimator(SIZE, 10, 0.0)


def test_estimator_attaches_to_a_cube_held_across_its_diagonal():
    est = _estimator()
    est.update(_det([0.5, 0.1, SIZE], 300), np.array([0.5, 0.0, 0.3]), 0.08, False)
    for _ in range(2):
        est.update(None, np.array([0.5, 0.1, 0.03]), gripper_width=SIZE * np.sqrt(2), gripper_closed=True)
    assert est.attached
