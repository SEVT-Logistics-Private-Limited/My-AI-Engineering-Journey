"""Plot the validation curve of a training run (light and dark variants + CSV).

    python scripts/plot_training.py --run runs/sac_final
    python scripts/plot_training.py --run runs/sac_final --out-dir assets
    python scripts/plot_training.py --run runs/sac_place_main --task place   # training_curve_place*.png

Reads <run>/eval/evaluations.npz, written by the EvalCallback in train_rl.py,
and writes training_curve.png, training_curve_dark.png and training_curve.csv.
Two panels on separate axes (success rate, episode length): the two measures
have different units, so they never share a y-axis.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402
from viz_theme import THEMES  # noqa: E402  (scripts/ is on sys.path when run directly)


@dataclass(frozen=True)
class TaskStyle:
    goal: str           # what a successful validation episode does
    time_limit: int     # episode length that means "no success"
    x_tick: int         # training-step spacing of x-axis ticks
    prefix: str         # output file stem


TASKS = {
    "lift": TaskStyle("lift the cube", 100, 20_000, "training_curve"),
    "place": TaskStyle("place the cube at the goal", 200, 50_000, "training_curve_place"),
}


@dataclass(frozen=True)
class Curve:
    steps: np.ndarray
    success: np.ndarray       # fraction of validation episodes that succeeded
    reward: np.ndarray        # mean validation return
    length: np.ndarray        # mean validation episode length (steps)
    n_episodes: int

    @property
    def checkpoint_index(self) -> int:
        # EvalCallback saves on a strictly better mean reward: the first maximum.
        return int(np.argmax(self.reward))


def load_curve(run: Path) -> Curve:
    path = run / "eval" / "evaluations.npz"
    if not path.is_file():
        raise SystemExit(f"no evaluation log at {path} (train with scripts/train_rl.py first)")
    data = np.load(path)
    if "successes" not in data:
        raise SystemExit(f"{path} has no success flags; the env must report info['is_success']")
    return Curve(
        steps=data["timesteps"], success=data["successes"].mean(axis=1),
        reward=data["results"].mean(axis=1), length=data["ep_lengths"].mean(axis=1),
        n_episodes=int(data["results"].shape[1]),
    )


def _style_axes(ax, t: dict, x_max: float, x_tick: int) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(1)
    ax.grid(axis="y", color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0, pad=6)
    # Explicit range: set_xlim(0, None) before plotting would freeze the default (0, 1).
    ax.set_xlim(0, x_max * 1.04)
    ax.xaxis.set_major_locator(MultipleLocator(x_tick))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:.0f}K"))
    ax.set_xlabel("Training steps", color=t["muted"], fontsize=9, labelpad=6)


def _plot_series(ax, x, y, t: dict) -> None:
    ax.fill_between(x, y, 0, color=t["series"], alpha=0.10, linewidth=0)
    ax.plot(x, y, color=t["series"], linewidth=2, solid_joinstyle="round", solid_capstyle="round",
            marker="o", markersize=6, markerfacecolor=t["series"],
            markeredgecolor=t["surface"], markeredgewidth=1.4, zorder=3)


def _mark_checkpoint(ax, step: float, top: float, t: dict) -> None:
    ax.axvline(step, color=t["axis"], linewidth=1, zorder=1)
    ax.annotate("saved checkpoint", (step, top), xytext=(4, 0), textcoords="offset points",
                ha="left", va="top", fontsize=8.5, color=t["muted"])


def render(curve: Curve, theme: str, out: Path, task: TaskStyle = TASKS["lift"]) -> None:
    t = THEMES[theme]
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ckpt = curve.steps[curve.checkpoint_index]

    # Panel 1: success rate.
    _style_axes(ax_s, t, float(curve.steps[-1]), task.x_tick)
    _plot_series(ax_s, curve.steps, curve.success * 100, t)
    ax_s.set_ylim(0, 114)
    ax_s.yaxis.set_major_locator(MultipleLocator(25))
    ax_s.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%" if v <= 100 else ""))
    ax_s.set_title("Validation success rate", loc="left", color=t["ink"], fontsize=11, pad=18)
    ax_s.text(0, 1.03, f"share of {curve.n_episodes} validation episodes that {task.goal}",
              transform=ax_s.transAxes, fontsize=8.5, color=t["ink2"])
    first_full = int(np.argmax(curve.success >= 1.0)) if (curve.success >= 1.0).any() else None
    if first_full is not None:
        # Above the 100% line, left of the point: the rising segment never reaches there.
        ax_s.annotate(f"100% at {curve.steps[first_full] / 1000:.0f}K",
                      (curve.steps[first_full], 100), xytext=(-6, 5), textcoords="offset points",
                      ha="right", va="bottom", fontsize=8.5, color=t["ink2"])
    _mark_checkpoint(ax_s, ckpt, 60, t)

    # Panel 2: episode length (time limit = no success).
    limit = task.time_limit
    _style_axes(ax_l, t, float(curve.steps[-1]), task.x_tick)
    _plot_series(ax_l, curve.steps, curve.length, t)
    ax_l.set_ylim(0, limit * 1.14)
    ax_l.yaxis.set_major_locator(MultipleLocator(limit / 4))
    ax_l.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}" if v <= limit else ""))
    ax_l.set_title("Steps to finish an episode", loc="left", color=t["ink"], fontsize=11, pad=18)
    ax_l.text(0, 1.03, f"mean over the same episodes; {limit} = time limit, no success",
              transform=ax_l.transAxes, fontsize=8.5, color=t["ink2"])
    ax_l.annotate(f"{curve.length[-1]:.1f} steps", (curve.steps[-1], curve.length[-1]),
                  xytext=(0, 10), textcoords="offset points", ha="right", va="bottom",
                  fontsize=8.5, color=t["ink2"])
    _mark_checkpoint(ax_l, ckpt, 0.6 * limit, t)

    fig.tight_layout(w_pad=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def write_csv(curve: Curve, out: Path) -> None:
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timesteps", "success_rate", "mean_reward", "mean_episode_length", "episodes", "saved_checkpoint"])
        for i, step in enumerate(curve.steps):
            w.writerow([int(step), f"{curve.success[i]:.2f}", f"{curve.reward[i]:.2f}",
                        f"{curve.length[i]:.1f}", curve.n_episodes, int(i == curve.checkpoint_index)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="training run directory (from train_rl.py --out)")
    parser.add_argument("--out-dir", default="assets")
    parser.add_argument("--task", choices=tuple(TASKS), default="lift")
    args = parser.parse_args()

    task = TASKS[args.task]
    curve = load_curve(Path(args.run))
    out_dir = Path(args.out_dir)
    names = (f"{task.prefix}.png", f"{task.prefix}_dark.png", f"{task.prefix}.csv")
    render(curve, "light", out_dir / names[0], task)
    render(curve, "dark", out_dir / names[1], task)
    write_csv(curve, out_dir / names[2])
    print(f"saved checkpoint at {int(curve.steps[curve.checkpoint_index])} steps; "
          f"wrote {', '.join(names)} to {out_dir}")


if __name__ == "__main__":
    main()
