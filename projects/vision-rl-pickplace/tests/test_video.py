import imageio.v2 as imageio
import numpy as np
import pytest

from pickplace.video import check_video_path, save_video

H, W, N = 48, 64, 12


def _frames():
    """A moving bright square, so consecutive frames really differ."""
    frames = []
    for i in range(N):
        f = np.zeros((H, W, 3), np.uint8)
        f[10:30, 4 * i:4 * i + 16] = (230, 60, 50)
        frames.append(f)
    return frames


def test_gif_round_trip(tmp_path):
    path = save_video(_frames(), tmp_path / "out.gif", fps=24)
    read = imageio.mimread(path)
    assert len(read) == N
    assert read[0].shape[:2] == (H, W)


def test_mp4_round_trip(tmp_path):
    path = save_video(_frames(), tmp_path / "sub" / "out.mp4", fps=24)   # parent dir is created
    with imageio.get_reader(path) as reader:
        assert reader.get_meta_data()["fps"] == pytest.approx(24)
        read = [frame for frame in reader]
    assert len(read) == N
    assert read[0].shape == (H, W, 3)


@pytest.mark.parametrize("name", ["out.avi", "out.png", "out"])
def test_unsupported_format_is_rejected_before_any_work(name):
    with pytest.raises(ValueError, match="unsupported video format"):
        check_video_path(name)


def test_empty_frames_rejected(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        save_video([], tmp_path / "out.gif", fps=24)
