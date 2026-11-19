"""Virtual RGB-D camera with explicit pinhole intrinsics and extrinsics.

The perception module only consumes ``(rgb, depth_m, intrinsics, T_world_cam)``.
That is the same interface a real RealSense/ZED driver would provide, so the
perception code transfers unchanged to hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from .config import CameraConfig
from .utils import look_at_rotation, make_transform


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: np.ndarray          # (H, W) float32, metres along the optical axis
    segmentation: np.ndarray   # (H, W) int32 body ids; simulation-only ground truth


class RGBDCamera:
    def __init__(self, cfg: CameraConfig, client: int):
        self.cfg = cfg
        self.client = client
        aspect = cfg.width / cfg.height
        self._view = p.computeViewMatrix(cfg.eye, cfg.target, cfg.up)
        self._proj = p.computeProjectionMatrixFOV(cfg.fov_deg, aspect, cfg.near, cfg.far)

        # PyBullet's fov is vertical: fy = (H/2) / tan(fov/2). Square pixels -> fx = fy.
        fy = (cfg.height / 2) / np.tan(np.deg2rad(cfg.fov_deg) / 2)
        self.intrinsics = Intrinsics(
            fx=fy, fy=fy, cx=cfg.width / 2, cy=cfg.height / 2,
            width=cfg.width, height=cfg.height,
        )
        R = look_at_rotation(np.array(cfg.eye), np.array(cfg.target), np.array(cfg.up))
        self.T_world_cam = make_transform(R, np.array(cfg.eye, dtype=float))

    def capture(self) -> RGBDFrame:
        cfg = self.cfg
        _, _, rgba, zbuf, seg = p.getCameraImage(
            cfg.width, cfg.height, self._view, self._proj,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.client,
        )
        rgba = np.reshape(rgba, (cfg.height, cfg.width, 4)).astype(np.uint8)
        zbuf = np.reshape(zbuf, (cfg.height, cfg.width)).astype(np.float64)
        seg = np.reshape(seg, (cfg.height, cfg.width)).astype(np.int32)
        return RGBDFrame(rgb=rgba[..., :3].copy(), depth=self._linearize(zbuf), segmentation=seg)

    def _linearize(self, zbuf: np.ndarray) -> np.ndarray:
        """Convert the OpenGL non-linear depth buffer to metric depth."""
        n, f = self.cfg.near, self.cfg.far
        return (f * n / (f - (f - n) * zbuf)).astype(np.float32)


def render_presentation(client: int, width: int = 480, height: int = 360) -> np.ndarray:
    """Angled 'showcase' view for videos; independent of the perception camera."""
    view = p.computeViewMatrixFromYawPitchRoll([0.35, 0.0, 0.25], 1.45, 50, -25, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, width / height, 0.05, 5.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=client,
    )
    return np.reshape(rgba, (height, width, 4))[..., :3].astype(np.uint8)
