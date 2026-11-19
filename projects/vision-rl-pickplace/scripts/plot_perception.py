"""Render the perception figure for the README (light and dark variants).

    python scripts/plot_perception.py --out-dir assets

Places the cube off the optical axis, where perspective makes its side faces
visible, runs the real detector on one RGB-D frame, and shows three zoomed
panels: the camera image with the detection, the segmentation (top face kept,
side faces rejected by the depth gate), and the height map from depth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from viz_theme import SEQUENTIAL, THEMES  # noqa: E402  (scripts/ is on sys.path when run directly)

from pickplace.camera import RGBDCamera  # noqa: E402
from pickplace.config import load_config  # noqa: E402
from pickplace.perception import ColorObjectDetector  # noqa: E402
from pickplace.sim import Scene  # noqa: E402
from pickplace.utils import transform_points  # noqa: E402

CUBE_XY, CUBE_YAW = (0.62, 0.12), 0.35   # off-axis and rotated: shows side faces and yaw
CROP_HALF = 26                           # pixels either side of the detection


def capture(cfg):
    with Scene(cfg) as scene:
        camera = RGBDCamera(cfg.camera, scene.client)
        scene.place_object(np.array(CUBE_XY), CUBE_YAW)
        scene.step(30)
        truth = scene.object_state()
        frame = camera.capture()
        det = ColorObjectDetector(cfg.perception).detect(
            frame.rgb, frame.depth, camera.intrinsics, camera.T_world_cam)
        if det is None:
            raise SystemExit("cube not detected; check the camera and perception config")
        return frame, det, truth, camera


def project(point, camera) -> np.ndarray:
    """World point -> image coordinates in imshow's convention (pixel i centred at i)."""
    p = transform_points(np.linalg.inv(camera.T_world_cam), point)
    K = camera.intrinsics
    return np.array([K.fx * p[0] / p[2] + K.cx - 0.5, K.fy * p[1] / p[2] + K.cy - 0.5])


def render(cfg, frame, det, truth, camera, theme: str, out: Path) -> None:
    t = THEMES[theme]
    u0, v0 = (int(round(c - 0.5)) - CROP_HALF for c in det.pixel)
    crop = (slice(v0, v0 + 2 * CROP_HALF), slice(u0, u0 + 2 * CROP_HALF))
    extent = (u0 - 0.5, u0 + 2 * CROP_HALF - 0.5, v0 + 2 * CROP_HALF - 0.5, v0 - 0.5)

    top_true = truth.position + np.array([0, 0, cfg.object.size / 2])
    err_mm = np.linalg.norm(det.position - top_true) * 1000
    centre = project(det.position, camera)
    # Orientation axis through the centre, +-1.5 cm: stays inside the 4 cm top face.
    axis = 0.015 * np.array([np.cos(det.yaw), np.sin(det.yaw), 0])
    ends = [project(det.position - axis, camera), project(det.position + axis, camera)]

    # A dedicated colorbar column keeps the three image panels the same size and aligned.
    fig, axs = plt.subplots(1, 4, figsize=(10, 3.9), dpi=160,
                            gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
    axes, cax = axs[:3], axs[3]
    fig.patch.set_facecolor(t["surface"])
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 1. Camera image with the detection.
    ax = axes[0]
    ax.imshow(frame.rgb[crop], extent=extent, interpolation="nearest")
    ax.contour(np.arange(u0, u0 + 2 * CROP_HALF), np.arange(v0, v0 + 2 * CROP_HALF),
               det.top_mask[crop].astype(float), levels=[0.5], colors="white", linewidths=1.6)
    ax.plot(*zip(*ends), color="white", linewidth=1.6, solid_capstyle="round")
    ax.plot(*centre, marker="+", color="white", markersize=11, markeredgewidth=1.8)
    ax.set_title("Camera image + detection", loc="left", color=t["ink"], fontsize=11, pad=8)
    x, y, z = det.position
    ax.set_xlabel(f"estimate ({x:.3f}, {y:.3f}, {z:.3f}) m · error {err_mm:.1f} mm",
                  color=t["ink2"], fontsize=8.5, labelpad=8)

    # 2. Segmentation: what the depth gate keeps.
    ax = axes[1]
    seg = np.empty(det.mask[crop].shape + (3,))
    seg[:] = to_rgb(t["surface"])
    seg[det.mask[crop] & ~det.top_mask[crop]] = to_rgb(t["muted"])
    seg[det.top_mask[crop]] = to_rgb(t["series"])
    ax.imshow(seg, extent=extent, interpolation="nearest")
    for spine in ax.spines.values():          # frame the panel: its background is the surface
        spine.set_visible(True)
        spine.set_color(t["grid"])
    ax.set_title("Segmentation", loc="left", color=t["ink"], fontsize=11, pad=8)
    ax.set_xlabel("blue: top face, used · gray: side faces, rejected",
                  color=t["ink2"], fontsize=8.5, labelpad=8)

    # 3. Height above the floor from depth (the camera looks straight down).
    ax = axes[2]
    height_cm = (cfg.camera.eye[2] - frame.depth[crop]) * 100
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL[theme])
    im = ax.imshow(height_cm, extent=extent, interpolation="nearest", cmap=cmap, vmin=0, vmax=4.5)
    ax.set_title("Height from depth", loc="left", color=t["ink"], fontsize=11, pad=8)
    ax.set_xlabel("height above the floor", color=t["ink2"], fontsize=8.5, labelpad=8)
    bar = fig.colorbar(im, cax=cax, ticks=[0, 2, 4])
    bar.ax.set_yticklabels(["0 cm", "2 cm", "4 cm"])
    bar.ax.tick_params(colors=t["muted"], labelsize=8.5, length=0)
    bar.outline.set_visible(False)

    fig.tight_layout(w_pad=2.5)
    # Match the colorbar to the height-map panel's vertical extent.
    img, bar_box = axes[2].get_position(), cax.get_position()
    cax.set_position([bar_box.x0, img.y0, bar_box.width, img.height])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print(f"{out}: localisation error {err_mm:.1f} mm, "
          f"{int(det.top_mask.sum())} of {int(det.mask.sum())} blob pixels on the top face")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default="assets")
    args = parser.parse_args()
    cfg = load_config(args.config)
    frame, det, truth, camera = capture(cfg)
    out_dir = Path(args.out_dir)
    render(cfg, frame, det, truth, camera, "light", out_dir / "perception.png")
    render(cfg, frame, det, truth, camera, "dark", out_dir / "perception_dark.png")


if __name__ == "__main__":
    main()
