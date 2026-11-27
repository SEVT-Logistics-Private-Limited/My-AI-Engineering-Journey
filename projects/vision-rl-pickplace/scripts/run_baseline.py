"""Classical Vision + IK baseline, no learning: pick-and-lift or pick-and-place.

    python scripts/run_baseline.py --episodes 50                  # pick-and-lift, headless benchmark
    python scripts/run_baseline.py --episodes 3 --gui             # watch it in the PyBullet window
    python scripts/run_baseline.py --task place --episodes 50     # pick-and-place
"""

from __future__ import annotations

import argparse
import collections
import time

import numpy as np

from pickplace.camera import RGBDCamera
from pickplace.config import load_config
from pickplace.pipeline import PickPipeline, PickPlacePipeline
from pickplace.sim import Scene
from pickplace.task import sample_place_goal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("lift", "place"), default="lift")
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true", help="open the PyBullet GUI (needs a display)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    outcomes: collections.Counter[str] = collections.Counter()
    place = args.task == "place"
    with Scene(cfg, gui=args.gui, coupled_fingers=place) as scene:
        camera = RGBDCamera(cfg.camera, scene.client)
        on_step = (lambda: time.sleep(cfg.sim.time_step * cfg.sim.control_substeps)) if args.gui else None
        pipeline = (PickPlacePipeline if place else PickPipeline)(scene, camera, on_step=on_step)
        start = time.time()
        for ep in range(args.episodes):
            scene.reset()
            state = scene.randomize_object(rng)
            if place:
                goal = sample_place_goal(rng, state.position[:2], cfg)
                scene.add_goal_marker(goal)
            scene.step(20)
            if place:
                r = pipeline.run(goal)
                outcomes[r.outcome.value] += 1
                print(f"episode {ep:3d}: {r.outcome.value:<18} goal_err={1000 * r.distance_to_goal:6.1f} mm  "
                      f"moved={100 * r.distance_moved:5.1f} cm")
                continue
            result = pipeline.run()
            outcomes[result.outcome.value] += 1
            est = "-" if result.detection is None else np.array2string(result.detection.position, precision=3)
            print(f"episode {ep:3d}: {result.outcome.value:<16} detected at {est}  lift={result.lift_height:.3f} m")
        elapsed = time.time() - start

    n = args.episodes
    print(f"\nsuccess rate: {outcomes['success']}/{n} ({100 * outcomes['success'] / n:.1f}%)  "
          f"[{elapsed / n:.2f} s/episode]")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome:<16} {count}")


if __name__ == "__main__":
    main()
