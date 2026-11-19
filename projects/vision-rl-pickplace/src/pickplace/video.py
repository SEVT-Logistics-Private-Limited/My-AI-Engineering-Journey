"""Write rendered frames to a GIF or MP4 file."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

SUPPORTED_SUFFIXES = (".gif", ".mp4")


def check_video_path(path: str | Path) -> Path:
    """Validate the output format up front, before minutes of evaluation are spent."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported video format {path.suffix!r}; use one of {SUPPORTED_SUFFIXES}")
    return path


def save_video(frames: Sequence[np.ndarray], path: str | Path, fps: int) -> Path:
    """Save RGB frames of shape (H, W, 3) as a looping GIF or an H.264 MP4."""
    path = check_video_path(path)
    if len(frames) == 0:
        raise ValueError("no frames to save")
    if fps <= 0:
        raise ValueError("fps must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
    else:
        # H.264 in yuv420p needs even frame sizes; a block size of 8 avoids the
        # default 16-pixel padding, which would distort 480x360 frames.
        imageio.mimsave(path, frames, fps=fps, codec="libx264", macro_block_size=8)
    return path
