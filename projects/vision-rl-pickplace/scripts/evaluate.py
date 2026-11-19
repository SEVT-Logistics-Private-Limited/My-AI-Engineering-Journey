"""Evaluate a controller and optionally record a video.

    # Trained RL policy, object position from ground truth (as in training)
    python scripts/evaluate.py --controller rl --model models/sac_picklift.zip --episodes 50

    # Same policy, object position from the camera pipeline (perception gap)
    python scripts/evaluate.py --controller rl --model models/sac_picklift.zip --obs-source perception

    # Classical Vision + IK baseline, recording a GIF
    python scripts/evaluate.py --controller baseline --episodes 3 --video assets/baseline.gif

Both controllers see the same seeded object placements, so their numbers are comparable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pickplace.camera import RGBDCamera, render_presentation
from pickplace.config import load_config
from pickplace.env import PickLiftEnv
from pickplace.pipeline import PickPipeline
from pickplace.sim import Scene


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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".gif"):
            imageio.mimsave(path, self.frames, duration=1000 / fps, loop=0)
        else:
            imageio.mimsave(path, self.frames, fps=fps)
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
            scene.reset_robot()
            scene.randomize_object(rng)
            scene.step(20)
            result = pipeline.run()
            results.append(result.success)
            print(f"episode {ep:3d}: {result.outcome.value:<16} lift={result.lift_height:.3f} m")
        if video:
            rec.save(video, fps=24)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", choices=("rl", "baseline"), default="rl")
    parser.add_argument("--model", default="models/sac_picklift.zip")
    parser.add_argument("--obs-source", choices=("ground_truth", "perception"), default="ground_truth")
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000, help="held-out seeds, disjoint from training")
    parser.add_argument("--video", default=None, help="output .gif or .mp4")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = time.time()
    if args.controller == "rl":
        if not Path(args.model).is_file():
            raise SystemExit(f"model not found: {args.model} (train one with scripts/train_rl.py)")
        results = evaluate_rl(cfg, args.model, args.obs_source, args.episodes, args.seed, args.video)
        label = f"rl[{args.obs_source}]"
    else:
        results = evaluate_baseline(cfg, args.episodes, args.seed, args.video)
        label = "baseline[perception]"
    rate = 100 * np.mean(results)
    print(f"\n{label}: success {sum(results)}/{len(results)} ({rate:.1f}%) in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
