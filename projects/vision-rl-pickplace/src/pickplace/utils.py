"""Small geometry helpers shared by perception, control and the RL env."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def wrap_to_quarter_turn(angle: float) -> float:
    """Wrap to [-pi/4, pi/4): a cube looks the same every 90 degrees.

    Choosing the smallest equivalent yaw keeps the wrist rotation minimal.
    """
    return float((angle + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and a 3-vector."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) or (3,) array of points."""
    pts = np.atleast_2d(points)
    out = pts @ T[:3, :3].T + T[:3, 3]
    return out[0] if np.ndim(points) == 1 else out


def look_at_rotation(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation of an OpenCV-convention camera (x right, y down, z forward) in world.

    Columns are the camera axes expressed in the world frame.
    """
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    norm = np.linalg.norm(x)
    if norm < 1e-9:
        raise ValueError("camera 'up' vector is parallel to the viewing direction")
    x /= norm
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
