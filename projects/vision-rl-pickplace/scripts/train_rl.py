"""Train a pick-and-lift or pick-and-place policy with Stable-Baselines3.

    python scripts/train_rl.py                       # pick-and-lift, SAC, settings from config/default.yaml
    python scripts/train_rl.py --task place          # pick-and-place (place.rl settings, curriculum)
    python scripts/train_rl.py --timesteps 20000     # quick smoke run
    tensorboard --logdir runs/                       # monitor reward / success rate

Outputs (under --out, default runs/<algo>_<timestamp>/):
    model.zip          final policy
    best/best_model.zip  best policy by evaluation success
    tb/                TensorBoard logs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from pickplace.config import load_config
from pickplace.env import PickLiftEnv
from pickplace.env_place import PickPlaceEnv


def make_env(cfg, seed: int, task: str = "lift", curriculum: bool = False) -> Monitor:
    if task == "place":
        # The curriculum (episodes that start with the cube in hand) is for training
        # only; validation always starts from scratch.
        env = PickPlaceEnv(cfg, obs_source="ground_truth", curriculum=curriculum)
    else:
        env = PickLiftEnv(cfg, obs_source="ground_truth")
    env.reset(seed=seed)
    return Monitor(env, info_keywords=("is_success",))


def build_model(cfg, env, tb_dir: Path, task: str = "lift"):
    rl = cfg.rl
    # Pick-and-place is a longer task: it uses its own discount (place.rl.gamma).
    gamma = cfg.place.rl.gamma if task == "place" else rl.sac.gamma
    if rl.algorithm == "SAC":
        s = rl.sac
        return SAC(
            "MlpPolicy", env, learning_rate=s.learning_rate, buffer_size=s.buffer_size,
            batch_size=s.batch_size, gamma=gamma, tau=s.tau, learning_starts=s.learning_starts,
            policy_kwargs={"net_arch": list(s.net_arch)}, tensorboard_log=str(tb_dir),
            seed=rl.seed, verbose=1, device="cpu",
        )
    return PPO("MlpPolicy", env, gamma=gamma, tensorboard_log=str(tb_dir),
               seed=rl.seed, verbose=1, device="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("lift", "place"), default="lift")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override rl.total_timesteps (lift) or place.rl.total_timesteps (place)")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--eval-freq", type=int, default=5000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    default_steps = cfg.place.rl.total_timesteps if args.task == "place" else cfg.rl.total_timesteps
    timesteps = args.timesteps or default_steps
    task_tag = "_place" if args.task == "place" else ""
    out = Path(args.out or f"runs/{cfg.rl.algorithm.lower()}{task_tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg, cfg.rl.seed, args.task, curriculum=True)
    eval_env = make_env(cfg, cfg.rl.seed + 1000, args.task, curriculum=False)
    model = build_model(cfg, env, out / "tb", args.task)
    callback = EvalCallback(
        eval_env, n_eval_episodes=20, eval_freq=args.eval_freq, deterministic=True,
        best_model_save_path=str(out / "best"), log_path=str(out / "eval"),
    )

    start = time.time()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    model.save(out / "model")
    print(f"done: {timesteps} steps in {(time.time() - start) / 60:.1f} min -> {out}")
    if args.task == "place":
        print(f"deploy with: cp {out / 'best' / 'best_model.zip'} models/sac_pickplace.zip")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
