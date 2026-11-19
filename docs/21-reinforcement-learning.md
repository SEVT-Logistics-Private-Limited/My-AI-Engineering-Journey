# 21. Reinforcement Learning

> Reinforcement learning is a type of ML where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties for its actions.

## Introduction

The other types of ML learn from a fixed dataset. Reinforcement learning is different: the model **learns by doing**, through trial and error.

## What is it?

**Reinforcement Learning (RL)** is a type of ML where an **agent** learns how to **make decisions** by **interacting with an environment** and **receiving rewards or penalties for its actions**.

| Term | Meaning | In the bicycle example |
|---|---|---|
| **Agent** | The learner that makes decisions | The kid |
| **Environment** | The world the agent acts in | The bicycle and the road |
| **Action** | What the agent does | Pedalling, steering, balancing |
| **Reward** | Feedback for a good action | Staying upright and moving forward |
| **Penalty** | Feedback for a bad action | Falling down |

### Example: a kid learning to ride a bicycle

Imagine a kid learning to ride a bicycle. Nobody can give them a table of correct answers. They try, wobble and fall (penalty), try again, and stay balanced a little longer (reward). After many attempts they learn which actions keep them upright.

## Why is it needed?

Some problems have no dataset of correct answers. The right move depends on the situation, and the result of an action is only known after trying it. Examples: playing a game, controlling a robot, driving. Reinforcement learning handles these by learning from the results of its own actions.

## How does it work?

1. The agent takes an action.
2. The environment responds.
3. The agent receives a reward or a penalty.
4. The agent learns from that feedback.
5. It chooses a better action next time, and the loop repeats.

Over many rounds, the agent learns the actions that earn the most reward.

## Flow diagram

```text
Agent → Takes action → Environment → Reward / Penalty
  ↑                                        ↓
Chooses a better action  ←────────────  Learns
```

## Real-world use cases

- Game-playing AI (chess, Go, video games)
- Robots learning to walk or pick up objects
- Self-driving cars
- Recommendation systems that learn from what users click

## In our project

### Vision-RL Pick-and-Lift

**Used. This is how our robot learns to pick up a cube.** Nobody programs its movements: a simulated robot arm practises by trial and error and learns which moves earn the most reward. [Project page →](../projects/vision-rl-pickplace/README.md)

| RL term | In our robot |
|---|---|
| **Agent** | A small neural network (the policy: two layers of 256 neurons), trained with the SAC algorithm |
| **Environment** | `PickLift-v0`: a simulated Franka Panda robot arm and a 4 cm cube, in the PyBullet physics simulator |
| **Observation** | 10 numbers: where the hand is, how wide the gripper is open, where the cube is, and where the cube is relative to the hand |
| **Action** | 4 numbers: move the hand up to 3 cm in any direction (x, y, z), and open or close the gripper |
| **Reward** | +0.25 each step both fingers touch the cube · +10 × the lift height each step it is held (up to +0.5) · +10 once it is held 5 cm up, which ends the attempt |
| **Penalty** | −1 every step, and a penalty equal to the distance between the hand and the cube, so slow or wandering attempts score less |

```text
New attempt: the cube and the hand start at random positions
        ↓
The policy reads 10 numbers (hand, gripper, cube)
        ↓
It chooses a small hand move (up to 3 cm) and opens or closes the gripper
        ↓
The simulator moves the arm for 1/24 of a second
        ↓
Reward or penalty for that step
        ↓
SAC improves the policy from stored experience
        ↓
Repeat until the cube is lifted 5 cm or 100 steps pass, then start a new attempt
```

Details:
- **Algorithm:** Soft Actor-Critic (SAC) from Stable-Baselines3. It stores past steps in a replay buffer and learns from them many times over, so it needs fewer attempts. That is what makes training on a CPU practical.
- **Training:** 60,000 steps, about 18 minutes on a standard 4-core CPU, with no GPU.
- **Keeping the best version:** every 5,000 steps, the policy is tested on 20 validation attempts it never trained on, and the best version is saved. Validation success first reached 100% at 25,000 steps; the saved policy is from 40,000 steps.
- **Reward hacking, found and fixed:** the first training run found a loophole. Holding the cube just below the 5 cm success height earned the grasp and lift rewards on every step, which added up to more than the one-time +10 for finishing, so the robot hovered instead of finishing. A −1 penalty per step, larger than the most any step can earn (0.75), made finishing always the better choice. The configuration now rejects reward settings that would allow hovering.
- **Fair testing:** training, validation and testing use different random seeds, like the train/validation/test split of a dataset. Final results come from 100 test attempts (seeds 30000–30099), run once, after development stopped.
- **Trained on exact positions, tested with the camera:** training uses the simulator's exact cube position, because processing camera images makes each step about 30 times slower. Testing with the camera's estimate shows the cost of imperfect vision instead of hiding it.
- **Results:** 99% success on the 100 test attempts, both with exact positions and with the camera; 100% when the cube is rotated in ways never seen in training. The reference is a hand-written controller with no learning, scored by the same rule: it reaches 100%.

Code: [`env.py`](../projects/vision-rl-pickplace/src/pickplace/env.py) (environment and reward), [`train_rl.py`](../projects/vision-rl-pickplace/scripts/train_rl.py) (training), [`evaluate.py`](../projects/vision-rl-pickplace/scripts/evaluate.py) (testing), [`default.yaml`](../projects/vision-rl-pickplace/config/default.yaml) (`rl:` settings)

Related: how the robot connects to [VLA](11-vla.md#vision-rl-pick-and-lift) (partly) and [ROS](13-ros.md#vision-rl-pick-and-lift) (the next step).

### Telugu Voice AI Assistant

**Not used yet.** Our voice assistant answers questions; it doesn't learn from rewards and penalties.

---

**Previous:** [← Semi-Supervised Learning](20-semi-supervised-learning.md) · [Home](../README.md) · **Next:** [Linear Regression →](22-linear-regression.md)
