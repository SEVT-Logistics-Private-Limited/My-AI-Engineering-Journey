"""Phase 1 milestone: the camera finds the cube and prints its 3D position.

    python scripts/detect_object.py --trials 10
    python scripts/detect_object.py --save-debug assets/detection.png
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from pickplace.camera import RGBDCamera
from pickplace.config import load_config
from pickplace.perception import ColorObjectDetector
from pickplace.sim import Scene
from pickplace.utils import wrap_to_quarter_turn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-debug", default=None, help="write an annotated RGB/depth image of the last trial")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    with Scene(cfg) as scene:
        camera = RGBDCamera(cfg.camera, scene.client)
        detector = ColorObjectDetector(cfg.perception)
        print(f"intrinsics: fx={camera.intrinsics.fx:.1f} fy={camera.intrinsics.fy:.1f} "
              f"cx={camera.intrinsics.cx:.1f} cy={camera.intrinsics.cy:.1f}")
        print(f"{'trial':>5} {'estimated xyz (m)':>28} {'true xyz (m)':>28} {'err mm':>7} {'yaw err deg':>11}")
        errors = []
        for i in range(args.trials):
            scene.randomize_object(rng)
            scene.step(30)
            truth = scene.object_state()
            top = truth.position + np.array([0, 0, cfg.object.size / 2])
            frame = camera.capture()
            det = detector.detect(frame.rgb, frame.depth, camera.intrinsics, camera.T_world_cam)
            if det is None:
                print(f"{i:>5} {'not detected':>28}")
                continue
            err = np.linalg.norm(det.position - top) * 1000
            yaw_err = np.rad2deg(abs(wrap_to_quarter_turn(det.yaw - truth.yaw)))
            errors.append(err)
            print(f"{i:>5} {np.array2string(det.position, precision=4):>28} "
                  f"{np.array2string(top, precision=4):>28} {err:7.2f} {yaw_err:11.2f}")
        if errors:
            print(f"mean error {np.mean(errors):.2f} mm, max {np.max(errors):.2f} mm, "
                  f"detected {len(errors)}/{args.trials}")

        if args.save_debug and det is not None:
            vis = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            u, v = (int(round(c)) for c in det.pixel)
            contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
            cv2.drawMarker(vis, (u, v), (255, 255, 0), cv2.MARKER_CROSS, 12, 1)
            label = "({:.3f}, {:.3f}, {:.3f}) m".format(*det.position)
            cv2.putText(vis, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            d = frame.depth
            depth_vis = cv2.applyColorMap(
                cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_VIRIDIS,
            )
            cv2.imwrite(args.save_debug, np.hstack([vis, depth_vis]))
            print(f"saved {args.save_debug}")


if __name__ == "__main__":
    main()
