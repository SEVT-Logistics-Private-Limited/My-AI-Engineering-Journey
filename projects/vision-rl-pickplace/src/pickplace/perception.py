"""Perception: RGB-D frame -> 3D object pose estimate.

Pipeline
    1. HSV colour threshold -> binary mask (red wraps around hue 0, so two ranges)
    2. Morphological opening to remove speckle
    3. Largest contour above an area threshold -> object blob
    4. Centroid pixel (u, v) + median depth over the blob -> 3D point in the camera frame
    5. Camera frame -> world frame via the camera extrinsics
    6. Yaw from the blob's minimum-area rectangle, measured in the world frame

Only the camera-agnostic ``(rgb, depth, intrinsics, T_world_cam)`` is used, so
the module is unchanged on a real RGB-D sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Intrinsics
from .config import PerceptionConfig
from .utils import transform_points, wrap_to_quarter_turn


@dataclass(frozen=True)
class Detection:
    pixel: tuple[float, float]       # (u, v) centroid
    position: np.ndarray             # top-surface point in world frame
    yaw: float                       # world-frame yaw, wrapped to [-pi/4, pi/4)
    area_px: int
    mask: np.ndarray                 # (H, W) bool blob mask


def deproject(u: float, v: float, depth: float, K: Intrinsics) -> np.ndarray:
    """Pixel + metric depth -> 3D point in the OpenCV camera frame."""
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError(f"invalid depth {depth!r}")
    x = (u - K.cx) * depth / K.fx
    y = (v - K.cy) * depth / K.fy
    return np.array([x, y, depth])


def pixel_to_world(u: float, v: float, depth: float, K: Intrinsics, T_world_cam: np.ndarray) -> np.ndarray:
    return transform_points(T_world_cam, deproject(u, v, depth, K))


class ColorObjectDetector:
    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg
        self._ranges = [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in cfg.hsv_ranges]
        k = max(1, cfg.morph_kernel)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = np.zeros(rgb.shape[:2], np.uint8)
        for lo, hi in self._ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

    def detect(
        self, rgb: np.ndarray, depth: np.ndarray, K: Intrinsics, T_world_cam: np.ndarray,
    ) -> Detection | None:
        """Return the largest matching object, or ``None`` if nothing is visible."""
        mask = self.segment(rgb)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.cfg.min_area_px:
            return None

        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [contour], -1, 255, thickness=cv2.FILLED)
        blob_mask = (blob > 0) & (mask > 0) & np.isfinite(depth) & (depth > 0)
        if not blob_mask.any():
            return None

        # Perspective makes the side faces visible and drags the centroid away
        # from the object's axis. Keep only the surface nearest the camera.
        z_near = float(np.percentile(depth[blob_mask], 5))
        top_mask = blob_mask & (depth <= z_near + self.cfg.top_face_tolerance)
        z = float(np.median(depth[top_mask]))
        vs, us = np.nonzero(top_mask)
        # +0.5: pixel i spans [i, i+1), so its centre is at i + 0.5.
        u, v = float(us.mean()) + 0.5, float(vs.mean()) + 0.5

        position = pixel_to_world(u, v, z, K, T_world_cam)
        top_contours, _ = cv2.findContours(
            top_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        yaw = self._estimate_yaw(max(top_contours, key=cv2.contourArea), z, K, T_world_cam)
        return Detection((u, v), position, yaw, int(area), blob_mask)

    @staticmethod
    def _estimate_yaw(contour: np.ndarray, z: float, K: Intrinsics, T_world_cam: np.ndarray) -> float:
        """Deproject one edge of the min-area rectangle and measure it in the world.

        Measuring in the world frame avoids hard-coding how the camera is
        mounted relative to the robot base.
        """
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        a = pixel_to_world(*box[0], z, K, T_world_cam)
        b = pixel_to_world(*box[1], z, K, T_world_cam)
        return wrap_to_quarter_turn(float(np.arctan2(b[1] - a[1], b[0] - a[0])))


class ObjectTracker:
    """Minimal state estimator: hold the last confident detection.

    An overhead camera loses the object when the gripper covers it, so the
    controller keeps acting on the last good estimate instead of on noise.
    """

    def __init__(self, min_area_px: int):
        self.min_area_px = min_area_px
        self.last: Detection | None = None
        self.frames_since_update = 0

    def reset(self) -> None:
        self.last, self.frames_since_update = None, 0

    def update(self, detection: Detection | None) -> Detection | None:
        if detection is not None and detection.area_px >= self.min_area_px:
            self.last, self.frames_since_update = detection, 0
        else:
            self.frames_since_update += 1
        return self.last
