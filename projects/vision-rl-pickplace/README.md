<div align="center">

# Vision-RL Pick-and-Lift

**Computer vision and robot learning for robotic grasping.** A simulated Franka Panda robot arm finds a cube with a depth camera, keeps tracking it even when its own arm hides it, and picks it up with a control policy learned by reinforcement learning on a standard CPU.

[![CI](https://github.com/SEVT-Logistics-Private-Limited/My-AI-Engineering-Journey/actions/workflows/vision-rl-pickplace.yml/badge.svg?branch=main)](https://github.com/SEVT-Logistics-Private-Limited/My-AI-Engineering-Journey/actions/workflows/vision-rl-pickplace.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-2563eb)
![OpenCV](https://img.shields.io/badge/vision-OpenCV-5c3ee8)
![PyBullet](https://img.shields.io/badge/sim-PyBullet-16a34a)
![SAC](https://img.shields.io/badge/RL-SAC%20%C2%B7%20Stable--Baselines3-7c3aed)
![Gymnasium](https://img.shields.io/badge/env-Gymnasium-0f766e)

<img src="assets/rl_policy.gif" width="420" alt="Trained SAC policy, acting on camera estimates, picking up the cube"> <img src="assets/baseline.gif" width="420" alt="Classical Vision + IK baseline picking up the cube">

*Left: learned SAC policy acting on camera estimates. Right: classical Vision + IK baseline. Real time, 24 fps.*

**99%** pick success from camera input · **1.8 mm** 3D localisation error · **18 min** training on a 4-core CPU · **47** tests in CI

</div>

---

> [!NOTE]
> **Status: research prototype.** Developed and evaluated in a physics simulator (PyBullet); not yet tested on a physical robot. The path to hardware is in the [roadmap](#limitations-and-roadmap).

## Executive summary

| Area | Summary |
|---|---|
| **Challenge** | To pick up an object, a robot must see it, locate it precisely and grasp it reliably, including when its own arm hides the object from the camera. |
| **Solution** | A complete perception-to-action software stack, built and tested in simulation: computer vision locates the object from a depth camera, a state estimator keeps tracking it through occlusion, and a control policy trained by reinforcement learning picks it up. |
| **Results** | 99% pick success from camera input on 100 held-out test episodes; 1.8 mm localisation error; 100% on object orientations never seen in training. |
| **Efficiency** | Trained in 18 minutes on a standard 4-core CPU, with no GPU and no robot hardware. |
| **Why it matters** | Picking is a core step in warehouse and logistics automation. This prototype demonstrates, without any robot hardware, the software such systems need: perception, tracking, learned control and rigorous testing. |
| **Quality** | Every result is reproducible from its seed; 47 automated tests and continuous-integration gates on the trained policy guard against regressions. |

## Contents

[Results](#results) · [Pick-and-place](#pick-and-place) · [How it works](#how-it-works) · [Engineering challenges](#engineering-challenges) · [Skills demonstrated](#skills-demonstrated) · [Key design decisions](#key-design-decisions) · [Quickstart](#quickstart) · [Testing and CI](#testing-and-ci) · [Project structure](#project-structure) · [Limitations and roadmap](#limitations-and-roadmap) · [Topics covered](#topics-covered-in-this-project) · [Resources](#resources)

## Results

All numbers come from **100 held-out test episodes** (seeds 30000–30099), each with the cube at a random position in a 25 × 40 cm workspace. These seeds were evaluated once, after development was frozen. Every controller is scored by the same rule: an episode succeeds when the cube is held and lifted at least 5 cm.

| Controller | Object position from | Success rate |
|---|---|---|
| Vision + IK baseline | Camera | **100%** (100/100) |
| SAC policy | Simulator ground truth (as trained) | **99%** (99/100) |
| SAC policy | **Camera + state estimator** | **99%** (99/100) |
| SAC policy, cube rotated ±45° (unseen in training) | Simulator ground truth | **100%** (100/100) |

| Perception (100 placements vs ground truth) | Result |
|---|---|
| 3D localisation error, top face | 1.8 mm mean · 2.8 mm max |
| Yaw error | 0.5° mean · 3.8° max |
| Detection rate, unoccluded | 100% |

Reproduce any row with `scripts/evaluate.py`, for example `python scripts/evaluate.py --controller rl --obs-source perception --episodes 100` for the camera-driven policy.

**Training:** SAC, 60,000 environment steps, 18 minutes on a 4-core CPU. Validation success first reaches 100% at 25,000 steps; the deployed policy is the best validation checkpoint, at 40,000 steps.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/training_curve_dark.png">
  <img src="assets/training_curve.png" width="100%" alt="Two line charts against training steps. Validation success rate: 0% up to 15K steps, 85% at 20K, 100% from 25K with one dip to 95% at 35K. Steps to finish an episode: 100, the time limit, up to 15K, then about 12 steps from 25K onward. A vertical line marks the saved checkpoint at 40K.">
</picture>

<details>
<summary>Training curve data (20 validation episodes per point)</summary>

| Training steps | Success rate | Mean episode length | Mean return |
|---|---|---|---|
| 5,000 | 0% | 100.0 | -114.20 |
| 10,000 | 0% | 100.0 | -84.34 |
| 15,000 | 0% | 100.0 | -84.02 |
| 20,000 | 85% | 26.6 | -17.41 |
| 25,000 | 100% | 12.8 | -1.51 |
| 30,000 | 100% | 14.3 | -2.58 |
| 35,000 | 95% | 16.1 | -4.71 |
| 40,000 (saved) | 100% | 11.3 | -0.29 |
| 45,000 | 100% | 11.8 | -0.71 |
| 50,000 | 100% | 11.7 | -0.78 |
| 55,000 | 100% | 11.4 | -0.32 |
| 60,000 | 100% | 11.7 | -0.60 |

Regenerate the chart from any run with `python scripts/plot_training.py --run runs/<name>`. The values are also in [`assets/training_curve.csv`](assets/training_curve.csv).

</details>

<details>
<summary>Evaluation protocol</summary>

| Seeds | Used for |
|---|---|
| 0 (one stream) | Training episodes |
| 1000 (one stream) | Validation during training: selects the saved checkpoint |
| 10000+, 20000+ | Development: debugging perception and determinism |
| **30000–30099** | **Test: the results above, evaluated once** |

Every episode is a pure function of its seed (see [Engineering challenges](#engineering-challenges), item 4), so `python scripts/evaluate.py` reproduces these numbers on the same PyBullet build. Both test failures are analysed under [Limitations](#limitations-and-roadmap).

</details>

## Pick-and-place

The second task: pick the cube up, carry it, and put it back down on the floor at a goal spot (the green disc) at least 10 cm from where it was picked. It reuses the scene, camera, state estimator and controller. Pick-and-lift above is unchanged: its regression outputs (tests, baseline, both policy evaluations) were checked to be identical per episode after the change.

<img src="assets/baseline_place.gif" width="420" alt="Classical Vision + IK baseline picking up the cube, carrying it to the green goal disc and setting it down">

*Classical Vision + IK baseline. Real time, 24 fps.*

### Success rule

One implementation ([`task.py`](src/pickplace/task.py)) scores every controller. Each condition closes a shortcut a learning agent would otherwise take:

| Condition | Default | Without it, the agent could |
|---|---|---|
| Lifted first | ≥ 5 cm while held (the pick-and-lift rule) | push or slide the cube to the goal |
| Released low | cube bottom ≤ 2 cm above the floor when the fingers let go | throw or drop it |
| Released | no finger contact for 3 control steps | hold it at the goal |
| At rest and upright | speed < 1 cm/s, tilt < 15° from the nearest face | leave it rolling or tipped over |
| On target | within 3 cm of the goal | put it anywhere |
| Moved | ≥ 10 cm from the pick spot | put it back where it was |
| In the workspace | inside the 25 × 40 cm spawn region (camera view, arm reach) | knock it out of view |

Goals are random per episode, at least 13 cm (10 cm + 3 cm tolerance) from the pick spot and 3 cm inside the spawn region, so landing anywhere within tolerance satisfies the last two rows by construction (tested). A goal is drawn right after the cube's position from the same seeded generator, so the baseline and the policy get the same pick and place spots for each seed. `place.goal_mode: any` drops the goal tolerance and keeps only the 10 cm rule.

### Results

**Pending.** The pick-and-place policy is still training; results for every controller on the held-out test seeds (30000–30099) will be added together, after development is frozen, following the same protocol as pick-and-lift.

### How the baseline places

```mermaid
flowchart LR
    A["Camera<br/>locate the cube"] --> B["Pick<br/>grasp, lift"] --> C["Carry<br/>≤ 1 cm per step,<br/>grasp checked"] --> D["Place<br/>lower; release with the<br/>cube 5 mm above the floor"] --> E["Retreat<br/>straight up,<br/>then home"] --> F["Camera check<br/>where did it land?"]
```

The arm carries at lift height in steps of at most 1 cm per control tick (`move_to` jumps straight to its final target, fast enough to shake the cube loose), checks the grasp at every step, lowers to 12 cm and then until the cube bottom is 5 mm above the floor, opens, and retreats straight up so the fingers do not drag the cube. With the arm back home and out of view, one photo checks where the cube landed: the robot verifies its own result. Every stage reports why it failed: `place_unreachable`, `dropped_in_transit`, then the success-rule outcomes (`released_too_high`, `toppled`, `missed_goal`, and so on).

### How the policy is trained

| Aspect | Detail |
|---|---|
| **Environment** | `PickPlace-v0` ([`env_place.py`](src/pickplace/env_place.py)): same scene, camera, estimator and 4-D action as pick-and-lift |
| **Observation (17)** | The pick-and-lift observation (10) · goal position (3) · goal − object (3) · a "lifted yet" flag (1), because success depends on history the current state does not show |
| **Reward** | −0.1 per step · progress terms as potential differences γΦ(s′) − Φ(s): reach → grasp → lift, then carry → lower near the goal · +25 on success · −25, and the episode ends, if the cube is dropped from above 2 cm, topples or leaves the workspace |
| **Curriculum** | Half the training episodes start with the cube already in hand (the scripted pick as the reset); evaluation always starts from scratch |
| **Discount** | γ = 0.98: the task takes ~30–80 steps, and γ = 0.95 looks only about 20 ahead |

Written as potential differences, no progress bonus can be collected twice: around any loop of states the shaping sums to (γ − 1)·ΣΦ ≤ 0, so hovering at the goal, lifting again and again, or putting the cube down and picking it up all lose reward, by construction. Each of these is a unit test, as are pushing (never lifted, never a success) and throwing (a terminal failure). One more trap is checked by config validation: with a terminal penalty smaller than a full episode of step penalties, failing on purpose to end the episode early would beat trying.

## How it works

```mermaid
flowchart LR
    PER["Perception<br/>RGB-D camera, OpenCV<br/>3D localisation"] --> EST["State estimation<br/>vision + proprioception"]
    EST --> DEC["Robot learning<br/>SAC policy"]
    DEC --> CTRL["Control<br/>inverse kinematics"]
    CTRL --> ROB["Franka Panda<br/>PyBullet, 240 Hz"]
    ROB -. "camera images" .-> PER
    ROB -. "joint encoders,<br/>gripper width" .-> EST
```

The loop runs at 24 Hz. The classical baseline shares the perception and the controller but needs no state estimator: it takes one photo before the arm moves, while nothing hides the cube, then runs a fixed grasp sequence. Both are scored by the same success rule: the cube is held and lifted at least 5 cm. For pick-and-place, the baseline continues with carry → place → camera check ([diagram](#how-the-baseline-places)).

### 1. Perception: computer vision

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/perception_dark.png">
  <img src="assets/perception.png" width="100%" alt="Three zoomed views of one camera frame. Left: the camera image with the detected top face outlined, the centroid and the orientation axis, and the estimate (0.618, 0.120, 0.040) m with 1.8 mm error. Middle: the segmentation, with the top face in blue and the side faces, rejected by the depth gate, in gray. Right: the height map from depth, with the cube 4 cm above the floor.">
</picture>

*One RGB-D frame through the real detector, zoomed on the cube. Generated by `scripts/plot_perception.py`.*

- **Camera:** an overhead RGB-D camera (320×240, 40° field of view) with explicit pinhole intrinsics `K` and extrinsics `T_world_cam`. The module consumes only `(rgb, depth, K, T_world_cam)`, the same interface a RealSense or ZED driver provides, so it transfers to hardware unchanged.
- **Segmentation:** HSV thresholding with two hue ranges (red wraps around 0), a morphological opening, and the largest contour above an area threshold.
- **Perspective-bias correction:** the camera sees the cube's side faces as well as its top, which drags the centroid off-centre. Keeping only pixels within 1 cm of the nearest surface cut the mean error from 3.3 mm to 1.8 mm, and the max from 6.1 mm to 2.8 mm.
- **3D localisation and orientation:** the centroid and median depth are deprojected through `K⁻¹` and transformed to the world frame. Yaw comes from the minimum-area rectangle, measured in the world frame so no camera-mounting convention is hard-coded.

### 2. State estimation: vision + proprioception

Once the arm moves into view, the overhead camera alone is unreliable: the gripper partly hides the cube, and a grasped cube is invisible. `ObjectStateEstimator` combines the camera with signals any real arm has:

- **Occlusion gating:** a detection is trusted only if its blob covers at least 80% of the largest unoccluded blob seen.
- **Grasp-aware tracking:** once the fingers settle on the cube (width between face-to-face and the face diagonal, still for two frames), the estimate follows the hand through forward kinematics.

### 3. Robot learning: reinforcement learning

| Aspect | Detail |
|---|---|
| **Algorithm** | Soft Actor-Critic (Stable-Baselines3), MLP 256×256, γ = 0.95 |
| **Observation (10)** | End-effector position · gripper width · object position · object − end-effector |
| **Action (4)** | Δx, Δy, Δz (up to 3 cm per step) · gripper open/close |
| **Reward** | −1 per step · −distance to the object · +0.25 while both fingers touch it · +10 × lift height (capped at 5 cm) · +10 on success |
| **Episode** | Success = held and lifted ≥ 5 cm, the same rule the baseline is scored by; time limit 100 steps (about 4 s at 24 Hz) |

The policy trains on simulator state, since rendering and processing a camera frame makes each step about 30 times slower (roughly 50 ms instead of 1.5 ms). It is then evaluated on camera estimates, which measures the perception gap instead of hiding it. A scripted expert solves the same environment through the same action interface (tested), which separates "is the task solvable?" from "did the agent learn?"

### 4. Control: inverse kinematics

PyBullet's damped-least-squares inverse kinematics, with a null-space bias that lets the wrist reach the commanded yaw, drives the end-effector to Cartesian targets. The baseline runs pre-grasp → grasp → lift; the policy issues small Cartesian steps.

## Engineering challenges

What broke during development, how each issue was diagnosed, and the measured effect of the fix. Every fix is covered by a test.

| # | Problem | Root cause | Fix | Result |
|---|---|---|---|---|
| 1 | Evaluation reward rose, but success stayed at 0% | **Reward hacking.** Holding the cube just below the success height earned grasp and lift bonuses every step, worth ~15 discounted, more than the one-time +10 success bonus. | A step penalty larger than the best per-step shaping reward, enforced by config validation | 0% → 100% validation success by 25k steps |
| 2 | 100% success with ground truth, **21%** with the camera | The estimate stayed on the floor while the cube was lifted, and partly hidden detections (42 of 330 px) were accepted | Camera + proprioception state estimator | 21% → 98% on development seeds |
| 3 | Estimate dragged 6 cm away during a failed grasp | **False attachment.** A corner pinch counted as a grasp. | Hold band from cube geometry, plus a finger-settle check | 98% → 100% on development seeds |
| 4 | Same seed, different outcome | **Non-deterministic resets.** Bullet keeps solver and contact caches across resets; `saveState`/`restoreState` did not clear them (verified). Trajectories diverged by up to 6 cm. | Rebuild the physics world each episode (60 ms) | Bit-identical rollouts per seed |
| 5 | Wrist reached only ~55% of the commanded rotation | The IK null-space rest pose pinned the wrist joint, and convergence checked position only | Bias the rest pose by the target yaw; check orientation too | Yaw error < 3° |
| 6 | Pick-and-place: every cube landed ~18 mm off target, always on the gripper's closing axis | **Uncoupled fingers.** The URDF drives the two fingers with independent motors; holding the cube, both saturate at equal force, so nothing keeps the pair centred and the cube drifted up to 2 cm while carried. Re-aiming the hand from finger kinematics helped (to ~7 mm) but a second correction near the floor dragged the cube (12 mm). | Couple the fingers with a gear constraint, as on the real Franka Hand; pick-and-place only, so pick-and-lift results stay bit-identical | 18 mm → 3 mm mean goal error, 100% placement on development seeds |

Complexity was added only where measurements justified it. One estimator rule never fired in 100 development episodes, so it was removed.

## Skills demonstrated

| Area | In this project |
|---|---|
| **Computer vision** | RGB-D perception · HSV segmentation and contour analysis (OpenCV) · pinhole camera model with explicit intrinsics and extrinsics · depth-based 3D deprojection · perspective-bias correction · orientation estimation |
| **State estimation** | Fusing vision with proprioception (forward kinematics, gripper width) · occlusion handling · grasp detection from finger geometry |
| **Robot learning** | Gymnasium environment design · reward shaping and diagnosing reward hacking · potential-based shaping and curriculum learning · SAC with Stable-Baselines3 · validation-based checkpoint selection · held-out and out-of-distribution evaluation |
| **Robotics** | Franka Panda in PyBullet · damped-least-squares inverse kinematics with null-space control · Cartesian control · top-down grasp planning |
| **ML engineering** | Seed-deterministic simulation · experiment protocol with data splits · pytest · GitHub Actions with model regression gates · typed, validated YAML config |

## Key design decisions

| Decision | Why |
|---|---|
| **Hardware-agnostic perception interface** | Consuming only `(rgb, depth, K, T_world_cam)` keeps the vision code unchanged on a real sensor. |
| **Cartesian action space through IK** | A 4-D action is far more sample-efficient than 9-D joint control, which is what makes CPU-only training practical. |
| **Train on state, evaluate on perception** | Fast training, plus an honest measurement of the perception gap. |
| **Reward computed from true state** | The reward is a training signal, not an input, so it should not inherit perception noise. |
| **Fresh physics world every episode** | Results depend only on the seed, never on episode order. |
| **Validated configuration** | Unknown or missing keys, and reward settings that allow hovering, fail at load time. |
| **One success rule per task, shared by every controller** | `task.py` scores the scripted baseline and the RL policy identically, and has no simulator inside, so every failure case has its own unit test. |
| **Potential-based progress rewards for pick-and-place** | A bonus written as γΦ(s′) − Φ(s) sums to at most zero around any loop, which removes hovering, re-lifting and put-down-pick-up hacks by construction rather than by patching each one. |
| **Additive extension** | Pick-and-place reuses the scene, camera, estimator and controller, and the pick-and-lift regression outputs (tests, baseline, both policy evaluations) were checked to be identical per episode after the change. |

## Quickstart

Requires Python 3.10+. No GPU needed.

```bash
git clone https://github.com/SEVT-Logistics-Private-Limited/My-AI-Engineering-Journey.git
cd My-AI-Engineering-Journey/projects/vision-rl-pickplace
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install -r requirements.txt
```

```bash
python scripts/detect_object.py --trials 10          # camera → 3D position, with error vs ground truth
python scripts/run_baseline.py --episodes 20         # classical Vision + IK pick-and-lift
python scripts/evaluate.py --controller rl           # trained policy on the test seeds
python scripts/evaluate.py --controller rl --obs-source perception   # ... acting on camera estimates
python scripts/train_rl.py                           # retrain from scratch (~18 min on 4 CPU cores)
python scripts/plot_training.py --run runs/<name>    # training curve → assets/
python scripts/plot_perception.py                    # perception figure → assets/
pytest -q                                            # 116 tests, ~20 s
```

Pick-and-place uses the same scripts with `--task place`:

```bash
python scripts/run_baseline.py --task place --episodes 20                        # Vision + IK pick-and-place
python scripts/evaluate.py --task place --controller baseline --episodes 100      # test seeds, failure breakdown
python scripts/evaluate.py --task place --controller rl --obs-source perception   # trained policy on camera input
python scripts/evaluate.py --task place --controller rl --edge-goals --random-yaw # out-of-distribution checks
python scripts/train_rl.py --task place                                           # retrain (curriculum, γ = 0.98)
python scripts/plot_training.py --run runs/<name> --task place                    # training_curve_place → assets/
```

Useful flags: `--gui` on `run_baseline.py` opens the PyBullet viewer (with `--task place` too); `--video out.gif` or `--video out.mp4` on `evaluate.py` records the episodes; `--min-success 0.9` makes `evaluate.py` fail below a success rate.

**Verified from a clean install (November 2026)** on a 4-core CPU with Python 3.13, PyTorch 2.15 (CPU), Stable-Baselines3 2.9.2, Gymnasium 1.4.0 and PyBullet 3.2.7 built from source:

- all 47 tests pass;
- every success rate in [Results](#results) reproduces exactly, including which seed fails;
- retraining from scratch took 21 minutes, matched the validation curve at every point, and produced a policy bit-identical to the committed `models/sac_picklift.zip`;
- perception error varies slightly with the PyBullet build: five seed sets of 100 placements gave 1.6–1.7 mm mean and under 3 mm max error.

## Testing and CI

- **116 tests:** config validation, camera geometry, perception accuracy, the state estimator, inverse kinematics, the baseline pipelines, both Gymnasium environments, determinism, video and figures. For pick-and-place: one test per placement outcome on hand-built states, goal sampling, solvability through the action API, and one test per reward hack (hovering at the goal, pushing, throwing, re-lifting, putting down and picking up again).
- **GitHub Actions** on Python 3.10 and 3.12, for every change to this project: the full test suite, a baseline smoke run, a **pick-and-place baseline gate** (at least 90% on 10 episodes), and **regression gates on the committed policy** (at least 90% success with ground truth and 80% with the camera), so a change that breaks the trained model fails CI.

## Project structure

```text
vision-rl-pickplace/
├── config/default.yaml        Simulation, camera, perception, grasp and RL settings
├── src/pickplace/
│   ├── camera.py              RGB-D camera with explicit intrinsics/extrinsics
│   ├── perception.py          Segmentation, 3D deprojection, yaw, state estimator
│   ├── kinematics.py          Inverse kinematics and Cartesian control
│   ├── pipeline.py            Classical baselines: Vision → IK → grasp (→ carry → place → camera check)
│   ├── task.py                Pick-and-place goal sampler and success rule (pure, shared)
│   ├── env.py                 Gymnasium environment (PickLift-v0)
│   ├── env_place.py           Gymnasium environment (PickPlace-v0)
│   ├── sim.py                 PyBullet scene with deterministic resets
│   ├── config.py              Typed, validated configuration
│   ├── video.py               GIF / MP4 writer
│   └── utils.py               Geometry helpers
├── scripts/
│   ├── detect_object.py       Perception demo and accuracy check
│   ├── run_baseline.py        Classical pipeline benchmark
│   ├── train_rl.py            SAC training (PPO also supported); --task lift | place
│   ├── evaluate.py            Evaluation, video recording, CI gate
│   ├── plot_training.py       Training-curve chart and CSV
│   ├── plot_perception.py     Perception figure
│   └── viz_theme.py           Shared figure colours
├── models/sac_picklift.zip    Trained policy (best validation checkpoint)
├── tests/                     116 tests
└── assets/                    Demo GIFs, perception figure, training curve
```

## Limitations and roadmap

**Known limitations**

- **Single, uniquely coloured object under fixed lighting.** HSV segmentation is the weakest link for real-world use.
- **Hidden displacement.** If the hand knocks the cube while the arm fully hides it, the estimate cannot recover until the cube is visible again. This caused the camera-mode test failure (seed 30056).
- **Rare policy stall.** In the ground-truth test failure (seed 30023), the policy held a tilted cube but hovered instead of lifting it.
- **No wrist rotation in the policy.** Rotated cubes still succeed because the 8 cm gripper opening exceeds the cube's 5.7 cm diagonal; elongated objects would need a yaw action.
- **Simulation only.** No domain randomisation yet, so sim-to-real transfer is untested. Determinism holds per PyBullet build.
- **Pick-and-place is checked only once the arm is out of view.** The baseline's camera check runs with the arm back home; a cube that bounces or tips while the hand hides it is the same hidden-displacement case as above. With camera input, the RL policy's arm is still above the cube right after release, so its estimate relies on the "released cubes rest where they were let go" assumption.
- **The goal is a point, not a region,** and there is a single object; placing next to or on top of other objects is not modelled.

**Roadmap**

1. Domain randomisation (friction, mass, lighting, camera pose) as the first step toward hardware.
2. Learned detection or instance segmentation, then point-cloud grasp poses for cluttered scenes.
3. A wrist camera or second viewpoint, to recover from hidden displacement.
4. Visual RL from pixels (an image encoder with augmentation, e.g. DrQ) to learn perception end to end.
5. ~~Extend pick-and-lift to full pick-and-place (carry the cube to a goal position)~~ **Done:** see [Pick-and-place](#pick-and-place). Still open: Hindsight Experience Replay, which needs a dictionary observation and a reward that can be recomputed for any goal; the "was lifted" condition does not fit that format cleanly.

## Topics covered in this project

This project applies topics explained in [My AI Engineering Journey](../../README.md). Each topic page has an **"In our project"** section with the details.

| # | Topic | Used | How it is used here |
|---|---|---|---|
| 11 | [VLA](../../docs/11-vla.md#vision-rl-pick-and-lift) | Partly | Vision and action are built: camera → learned policy → arm. There is no language part: the task is fixed in code, not given as an instruction |
| 13 | [ROS](../../docs/13-ros.md#vision-rl-pick-and-lift) | Not yet | Runs as one Python program in simulation; each part maps to a ROS 2 node, the planned step toward a real robot |
| 21 | [Reinforcement Learning](../../docs/21-reinforcement-learning.md#vision-rl-pick-and-lift) | Yes | The core of the project: a SAC policy learns to reach, grasp and lift the cube by trial and error, guided by rewards and penalties |

## Built with

[PyBullet](https://pybullet.org) physics and its Franka Panda model · [OpenCV](https://opencv.org) · [Gymnasium](https://gymnasium.farama.org) · [Stable-Baselines3](https://stable-baselines3.readthedocs.io) with [PyTorch](https://pytorch.org) · [matplotlib](https://matplotlib.org)

## Resources

| Resource | Purpose |
|---|---|
| [Trained policy](models/sac_picklift.zip) | The deployed SAC checkpoint (best validation score, at 40,000 steps) |
| [Training curve data](assets/training_curve.csv) | The validation results behind the training chart |
| [Configuration](config/default.yaml) | Every setting: camera, workspace, success rule, rewards and training |
| [CI workflow](../../.github/workflows/vision-rl-pickplace.yml) | Tests and policy regression gates, run on every change |
| [VLA](../../docs/11-vla.md) · [ROS](../../docs/13-ros.md) · [Reinforcement Learning](../../docs/21-reinforcement-learning.md) · [Image](../../docs/04-image.md) · [Image Resolution](../../docs/05-image-resolution.md) | Related concepts, explained in this repository |

## Author and contact

**Rishi Sasanala**, sole author · [rishisasanala@gmail.com](mailto:rishisasanala@gmail.com) · part of [My AI Engineering Journey](../../README.md)
