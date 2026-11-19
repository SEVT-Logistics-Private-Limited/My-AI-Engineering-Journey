"""Perception: RGB-D frame -> 3D object pose estimate.

Pipeline
    1. HSV colour threshold -> binary mask (red wraps around hue 0, so two ranges)
    2. Morphological opening to remove speckle
    3. Largest contour above an area threshold -> object blob
    4. Centroid pixel (u, v) + median depth over the blob -> 3D point in the camera frame
    5. Camera frame -> world frame via the camera extrinsics
    6. Yaw from the blob's minimum-area rectangle, measured in the world frame
    7. ``ObjectStateEstimator`` fuses detections with gripper proprioception so
       the estimate survives occlusion by the arm and moves with a grasped object

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
    top_mask: np.ndarray | None = None  # (H, W) bool pixels kept by the top-face depth gate


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
        return Detection((u, v), position, yaw, int(area), blob_mask, top_mask)

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


class ObjectStateEstimator:
    """Estimate the object's centre by fusing camera detections with gripper proprioception.

    A lone overhead camera fails in two ways once the arm is in view. Both were
    measured: fed raw detections, the trained policy fell from 100% to 21% success.

    * Partial occlusion. The gripper hides part of the cube, and the visible
      sliver gives a biased position. A detection is trusted only if its blob is
      at least ``min_area_ratio`` of the largest (unoccluded) blob seen so far.
    * Grasp. A held object is hidden by the hand and moves with it. Once the
      fingers have settled on the object (closed, width in the hold band and
      steady for two consecutive frames), the object is treated as rigidly
      attached to the end-effector and its estimate follows forward kinematics.
      A grasp that is still slipping or closing never looks settled. On release
      the object is assumed to come to rest on the table.

    Everything used here (camera, joint encoders, gripper width) exists on a
    real robot; no simulator state is read.
    """

    # Gripper width band, as a fraction of the object size, that means "holding it":
    # from a face-to-face grasp (1.0, with margin) to a hold across the face
    # diagonal (sqrt(2)). Narrower means fingers on an edge or corner, which is not stable.
    HOLD_WIDTH_RANGE = (0.75, 1.5)
    # Fingers that move less than this between frames count as settled on the object.
    WIDTH_SETTLE_TOL = 0.002

    def __init__(self, object_size: float, min_area_px: int, min_area_ratio: float):
        if not 0.0 < min_area_ratio <= 1.0:
            raise ValueError("min_area_ratio must be in (0, 1]")
        self.object_size = object_size
        self.min_area_px = min_area_px
        self.min_area_ratio = min_area_ratio
        self.reset()

    def reset(self) -> None:
        self.position: np.ndarray | None = None   # object centre, world frame
        self.reference_area = 0
        self.attached = False
        self.frames_since_detection = 0
        self._offset = np.zeros(3)
        self._contact_offset: np.ndarray | None = None
        self._prev_in_band = False
        self._prev_width = 0.0

    def update(self, detection: Detection | None, ee_position: np.ndarray,
               gripper_width: float, gripper_closed: bool) -> np.ndarray | None:
        """Fold in one camera frame and the current gripper state; return the estimate."""
        lo, hi = (r * self.object_size for r in self.HOLD_WIDTH_RANGE)
        in_band = gripper_closed and lo <= gripper_width <= hi
        if in_band and not self._prev_in_band:
            # First contact: record where the object sits relative to the hand,
            # before it starts moving with it.
            near = self.position is not None and np.linalg.norm(ee_position - self.position) < self.object_size
            self._contact_offset = self.position - ee_position if near else None
        settled = (in_band and self._prev_in_band
                   and abs(gripper_width - self._prev_width) < self.WIDTH_SETTLE_TOL)
        self._prev_in_band, self._prev_width = in_band, gripper_width

        if self.attached and not in_band:
            self.attached = False
            self.position = self.position.copy()
            self.position[2] = self.object_size / 2      # released: comes to rest on the table
        elif not self.attached and settled and self._contact_offset is not None:
            self.attached = True
            self._offset = self._contact_offset

        if self.attached:
            self.position = ee_position + self._offset
            return self.position

        if self._is_confident(detection):
            self.position = detection.position - np.array([0.0, 0.0, self.object_size / 2])
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
        return self.position

    def _is_confident(self, detection: Detection | None) -> bool:
        if detection is None or detection.area_px < self.min_area_px:
            return False
        # The largest blob seen is the best available measure of the unoccluded area.
        self.reference_area = max(self.reference_area, detection.area_px)
        return detection.area_px >= self.min_area_ratio * self.reference_area
