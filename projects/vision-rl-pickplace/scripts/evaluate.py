"""Evaluate a controller and optionally record a video.

    # Trained RL policy, object position from ground truth (as in training)
    python scripts/evaluate.py --controller rl --model models/sac_picklift.zip --episodes 50

    # Same policy, object position from the camera pipeline (perception gap)
    python scripts/evaluate.py --controller rl --model models/sac_picklift.zip --obs-source perception

    # Classical Vision + IK baseline, recording a GIF
    python scripts/evaluate.py --controller baseline --episodes 3 --video assets/baseline.gif
    python scripts/evaluate.py --controller rl --episodes 3 --video runs/policy.mp4

    # Pick-and-place (--task place): same controllers, plus out-of-distribution checks
    python scripts/evaluate.py --task place --controller baseline --episodes 100
    python scripts/evaluate.py --task place --controller rl --obs-source perception
    python scripts/evaluate.py --task place --controller rl --edge-goals --random-yaw

Both controllers see the same seeded object placements, so their numbers are comparable.
Default seeds (30000+) are the held-out test set used for the README results.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np

from pickplace.camera import RGBDCamera, render_presentation
from pickplace.config import load_config
from pickplace.env import PickLiftEnv
from pickplace.env_place import PickPlaceEnv
from pickplace.pipeline import PickPipeline, PickPlacePipeline
from pickplace.sim import Scene
from pickplace.task import sample_place_goal
from pickplace.video import check_video_path, save_video


class FrameRecorder:
    """Collects every ``stride``-th frame from the showcase camera."""

    def __init__(self, client: int, enabled: bool, stride: int = 1):
        self.client, self.enabled, self.stride = client, enabled, stride
        self.frames: list[np.ndarray] = []
        self._tick = 0

    def __call__(self) -> None:
        if self.enabled and self._tick % self.stride == 0:
            self.frames.append(render_presentation(self.client))
        self._tick += 1

    def save(self, path: str, fps: int) -> None:
        if not self.frames:
            return
        save_video(self.frames, path, fps)
        print(f"saved {len(self.frames)} frames -> {path}")


def evaluate_rl(cfg, model_path: str, obs_source: str, episodes: int, seed: int, video: str | None) -> list[bool]:
    from stable_baselines3 import PPO, SAC  # lazy: the baseline needs no torch

    algo = SAC if cfg.rl.algorithm == "SAC" else PPO
    model = algo.load(model_path, device="cpu")
    env = PickLiftEnv(cfg, obs_source=obs_source)
    rec = FrameRecorder(env.scene.client, video is not None)
    results = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        rec()
        done, steps, info = False, 0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            rec()
            steps += 1
            done = terminated or truncated
        if video and info["is_success"]:
            for _ in range(8):  # brief hold so the lift is visible
                env.step(np.array([0, 0, 0, 1.0]))
                rec()
        results.append(bool(info["is_success"]))
        print(f"episode {ep:3d}: {'success' if info['is_success'] else 'fail':<8} steps={steps:3d} lift={info['lift']:.3f} m")
    if video:
        rec.save(video, fps=24)
    env.close()
    return results


def evaluate_baseline(cfg, episodes: int, seed: int, video: str | None) -> list[bool]:
    results = []
    with Scene(cfg) as scene:
        rec = FrameRecorder(scene.client, video is not None)
        pipeline = PickPipeline(scene, RGBDCamera(cfg.camera, scene.client), on_step=rec)
        for ep in range(episodes):
            # Match the RL env's placement distribution for a fair comparison.
            rng = np.random.default_rng(seed + ep)
            scene.reset()
            scene.randomize_object(rng)
            scene.step(20)
            result = pipeline.run()
            results.append(result.success)
            print(f"episode {ep:3d}: {result.outcome.value:<16} lift={result.lift_height:.3f} m")
        if video:
            rec.save(video, fps=24)
    return results


# ---------------------------------------------------------------- pick-and-place
DEFAULT_MODELS = {"lift": "models/sac_picklift.zip", "place": "models/sac_pickplace.zip"}


def _place_record(ep: int, outcome: str, info: dict, seconds: float, steps: int | None = None) -> dict:
    rec = {"episode": ep, "outcome": outcome, "seconds": seconds, "steps": steps, **info}
    cam = rec.get("camera_error", float("nan"))
    print(f"episode {ep:3d}: {outcome:<18} goal_err={1000 * rec['distance_to_goal']:6.1f} mm  "
          f"moved={100 * rec['distance_moved']:5.1f} cm  tilt={np.rad2deg(rec['tilt']):4.1f} deg"
          + (f"  steps={steps:3d}" if steps is not None else "")
          + (f"  camera_err={1000 * cam:.1f} mm" if np.isfinite(cam) else ""))
    return rec


def evaluate_place_rl(cfg, model_path: str, obs_source: str, episodes: int, seed: int, video: str | None,
                      edge_goals: bool, random_yaw: bool) -> list[dict]:
    from stable_baselines3 import PPO, SAC  # lazy: the baseline needs no torch

    algo = SAC if cfg.rl.algorithm == "SAC" else PPO
    model = algo.load(model_path, device="cpu")
    env = PickPlaceEnv(cfg, obs_source=obs_source, edge_goals=edge_goals,
                       object_yaw_range=cfg.object.yaw_range if random_yaw else None)
    rec = FrameRecorder(env.scene.client, video is not None)
    records = []
    for ep in range(episodes):
        t0 = time.time()
        obs, info = env.reset(seed=seed + ep)
        rec()
        done, steps = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            rec()
            steps += 1
            done = terminated or truncated
        if video and info["is_success"]:
            for _ in range(12):  # brief pause on the placed cube, hand moving away
                env.step(np.array([0, 0, 1.0, -1.0]))
                rec()
        keep = ("distance_to_goal", "distance_moved", "tilt")
        records.append(_place_record(ep, info["outcome"], {k: info[k] for k in keep}, time.time() - t0, steps))
    if video:
        rec.save(video, fps=24)
    env.close()
    return records


def evaluate_place_baseline(cfg, episodes: int, seed: int, video: str | None, edge_goals: bool) -> list[dict]:
    records = []
    with Scene(cfg, coupled_fingers=True) as scene:
        rec = FrameRecorder(scene.client, video is not None)
        pipeline = PickPlacePipeline(scene, RGBDCamera(cfg.camera, scene.client), on_step=rec)
        for ep in range(episodes):
            t0 = time.time()
            # Same draws, in the same order, as PickPlaceEnv.reset: same pick and place spots per seed.
            rng = np.random.default_rng(seed + ep)
            scene.reset()
            state = scene.randomize_object(rng)
            goal = sample_place_goal(rng, state.position[:2], cfg, edge_goals=edge_goals)
            scene.add_goal_marker(goal)
            scene.step(20)
            r = pipeline.run(goal)
            info = {"distance_to_goal": r.distance_to_goal, "distance_moved": r.distance_moved,
                    "tilt": r.tilt, "camera_error": r.camera_error}
            records.append(_place_record(ep, r.outcome.value, info, time.time() - t0))
        if video:
            rec.save(video, fps=24)
    return records


def summarize_place(records: list[dict]) -> None:
    n = len(records)
    ok = [r for r in records if r["outcome"] == "success"]
    print(f"\noutcomes ({n} episodes):")
    for outcome, count in collections.Counter(r["outcome"] for r in records).most_common():
        print(f"  {outcome:<18} {count:4d}  ({100 * count / n:.0f}%)")

    def stat(key: str, scale: float, unit: str, rows: list[dict]) -> str:
        vals = np.array([r[key] for r in rows], float)
        vals = vals[np.isfinite(vals)] * scale
        if vals.size == 0:
            return "-"
        return f"mean {vals.mean():.1f} {unit} · median {np.median(vals):.1f} · max {vals.max():.1f}"

    rows = ok or records
    print(f"metrics over {'successful' if ok else 'all'} episodes:")
    print(f"  distance to goal      {stat('distance_to_goal', 1000, 'mm', rows)}")
    print(f"  distance moved        {stat('distance_moved', 100, 'cm', rows)}")
    print(f"  tilt                  {stat('tilt', float(np.rad2deg(1)), 'deg', rows)}")
    if any(np.isfinite(r.get("camera_error", np.nan)) for r in rows):
        print(f"  camera check error    {stat('camera_error', 1000, 'mm', rows)}")
    print(f"  time per episode      {stat('seconds', 1, 's', records)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("lift", "place"), default="lift")
    parser.add_argument("--controller", choices=("rl", "baseline"), default="rl")
    parser.add_argument("--model", default=None,
                        help="policy file (default: models/sac_picklift.zip, or models/sac_pickplace.zip for --task place)")
    parser.add_argument("--obs-source", choices=("ground_truth", "perception"), default="ground_truth")
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    # Seeds 30000+ are the test set: never used for training, checkpoint selection
    # or debugging (10000+ and 20000+ served as development sets).
    parser.add_argument("--seed", type=int, default=30_000, help="first evaluation seed")
    parser.add_argument("--video", default=None, help="output .gif or .mp4")
    parser.add_argument("--min-success", type=float, default=None,
                        help="exit non-zero if the success rate (0-1) is below this; used as a CI regression gate")
    parser.add_argument("--edge-goals", action="store_true",
                        help="place only: goals near the edge of the goal region (out of distribution)")
    parser.add_argument("--random-yaw", action="store_true",
                        help="place, rl only: random cube yaw from object.yaw_range (out of distribution)")
    args = parser.parse_args()
    if args.task == "lift" and (args.edge_goals or args.random_yaw):
        parser.error("--edge-goals and --random-yaw need --task place")
    args.model = args.model or DEFAULT_MODELS[args.task]
    if args.min_success is not None and not 0.0 <= args.min_success <= 1.0:
        parser.error("--min-success must be between 0 and 1")
    if args.video is not None:
        try:
            check_video_path(args.video)
        except ValueError as exc:
            parser.error(str(exc))

    cfg = load_config(args.config)
    start = time.time()
    if args.controller == "rl" and not Path(args.model).is_file():
        raise SystemExit(f"model not found: {args.model} (train one with scripts/train_rl.py --task {args.task})")
    if args.task == "place":
        if args.controller == "rl":
            records = evaluate_place_rl(cfg, args.model, args.obs_source, args.episodes, args.seed, args.video,
                                        args.edge_goals, args.random_yaw)
            label = f"place rl[{args.obs_source}]"
        else:
            records = evaluate_place_baseline(cfg, args.episodes, args.seed, args.video, args.edge_goals)
            label = "place baseline[perception]"
        summarize_place(records)
        results = [r["outcome"] == "success" for r in records]
    elif args.controller == "rl":
        results = evaluate_rl(cfg, args.model, args.obs_source, args.episodes, args.seed, args.video)
        label = f"rl[{args.obs_source}]"
    else:
        results = evaluate_baseline(cfg, args.episodes, args.seed, args.video)
        label = "baseline[perception]"
    rate = float(np.mean(results))
    print(f"\n{label}: success {sum(results)}/{len(results)} ({100 * rate:.1f}%) in {time.time() - start:.1f} s")
    if args.min_success is not None and rate < args.min_success:
        raise SystemExit(f"FAIL: success rate {rate:.2f} is below the required {args.min_success:.2f}")


if __name__ == "__main__":
    main()
