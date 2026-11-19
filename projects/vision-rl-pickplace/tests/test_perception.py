import numpy as np
import pytest

from pickplace.camera import Intrinsics, RGBDCamera
from pickplace.perception import ColorObjectDetector, Detection, ObjectTracker, deproject
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


def test_tracker_holds_last_estimate():
    tracker = ObjectTracker(min_area_px=10)
    good = Detection((0, 0), np.ones(3), 0.0, 100, np.zeros((1, 1), bool))
    assert tracker.update(good) is good
    assert tracker.update(None) is good           # occluded: keep last
    small = Detection((0, 0), np.zeros(3), 0.0, 5, np.zeros((1, 1), bool))
    assert tracker.update(small) is good          # partial view: keep last
    assert tracker.frames_since_update == 2
