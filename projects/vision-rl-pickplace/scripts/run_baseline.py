"""Phase 2 milestone: classical Vision + IK pick-and-lift, no learning.

    python scripts/run_baseline.py --episodes 50          # headless benchmark
    python scripts/run_baseline.py --episodes 3 --gui     # watch it in the PyBullet window
"""

from __future__ import annotations

import argparse
import collections
import time

import numpy as np

from pickplace.camera import RGBDCamera
from pickplace.config import load_config
from pickplace.pipeline import PickPipeline
from pickplace.sim import Scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true", help="open the PyBullet GUI (needs a display)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    outcomes: collections.Counter[str] = collections.Counter()
    with Scene(cfg, gui=args.gui) as scene:
        camera = RGBDCamera(cfg.camera, scene.client)
        on_step = (lambda: time.sleep(cfg.sim.time_step * cfg.sim.control_substeps)) if args.gui else None
        pipeline = PickPipeline(scene, camera, on_step=on_step)
        start = time.time()
        for ep in range(args.episodes):
            scene.reset()
            scene.randomize_object(rng)
            scene.step(20)
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
