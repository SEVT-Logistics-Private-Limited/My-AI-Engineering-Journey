# 11. VLA (Vision Language Action)

> A VLA helps AI-powered robots understand their surroundings, follow instructions and perform physical actions. It sees, understands and acts.

## Introduction

A [VLM](10-vlm.md) can describe what it sees, but it can't do anything about it. A VLA goes one step further: it **acts**.

## What is it?

**VLA (Vision Language Action)** helps **AI-powered robots** to:
- understand their surroundings
- follow instructions
- perform physical actions

> **Sees → Understands → Acts**
>
> Vision + Language + Action

## Why is it needed?

Robots need to turn an instruction in human language into real movement in the physical world.

## How does it work?

Example: you tell a robot **"Pick up the red bottle."**

The VLA:
1. **Sees** the bottle using a camera
2. **Understands** the instruction
3. **Determines** the required movement
4. **Generates** the action

### VLM vs VLA

| VLM | VLA |
|---|---|
| Vision Language Model | Vision Language Action |
| Understands images and language | Connects vision and language to action |
| Describes images and answers questions | Helps control physical actions |
| Image question answering | Robotic manipulation |
| Shown a red bottle: "I can see a red bottle on the table." | Sees the red bottle, understands "pick it up", generates actions and picks it up |
| Only says what it sees; no action | Sees, understands and acts |
| **VLM = sees and understands** | **VLA = sees, understands and acts** |

## Flow diagram

```text
Instruction: "Pick up the red bottle"
              ↓
Sees the bottle (camera)
              ↓
Understands the instruction
              ↓
Determines the required movement
              ↓
Generates the action
              ↓
Robot picks up the bottle
```

## Real-world use cases

- Warehouse robots picking items
- Household robots
- Robotic arms in factories

## In our project

### Vision-RL Pick-and-Place

**Partly used: our robot has the Vision and Action parts, but not the Language part.** It finds a cube with a camera, picks it up and sets it down at a goal spot elsewhere on the floor, but its task ("put the red cube on the green spot") is built into the code, not given as an instruction. So it is a vision-to-action system, not a full VLA. [Project page →](../projects/vision-rl-pickplace/README.md)

| VLA step | In our robot | Built? |
|---|---|---|
| **Sees** the object with a camera | An overhead depth camera. Computer vision (OpenCV) finds the red cube and works out its 3D position, to within 1.8 mm on average | Yes |
| **Understands** the instruction | There is no instruction. The goal is fixed in code: the computer vision looks for red, the reward teaches the policy to pick it up, and the program sets the goal spot | No |
| **Determines** the required movement | A state estimator keeps track of the cube, even when the arm hides it; a policy trained with [reinforcement learning](21-reinforcement-learning.md#vision-rl-pick-and-place) chooses each hand move of the pick, and a planned carry takes the cube to the goal | Yes |
| **Generates** the action | Each hand move is converted into commands for the arm's 7 joints (inverse kinematics), plus open or close for the gripper | Yes |

How it differs from a real VLA model, such as Google DeepMind's RT-2 or the open-source OpenVLA:

| | Real VLA model | Our robot |
|---|---|---|
| **Input** | Camera image + an instruction in words | Camera image (colour and depth) only |
| **Design** | One large neural network: image and words in, actions out | Separate parts: computer vision → state estimator → small learned pick policy → planned carry and place → inverse kinematics |
| **Training** | Large collections of robot demonstrations, on GPUs | 18 minutes of trial and error in simulation, on a standard CPU |
| **New task** | Give a new instruction: "pick up the blue cup" | Change the code; a different kind of task also needs retraining |
| **Finding faults** | Hard: one model does everything | Easier: each part is measured and tested on its own |

```text
Real VLA:
"Put the red cube on the green spot" + camera image → one large model → arm actions

Our robot (the goal "put the red cube on the green spot" is built in):
Camera image → computer vision → state estimator → learned pick policy → planned carry and place → inverse kinematics → arm actions
```

Why we built it this way: separate parts train in minutes on a CPU, and each one can be measured on its own (for example, the 1.8 mm vision error). A VLA model needs far more data and computing power.

Next steps toward a full VLA:
1. **Add language:** accept an instruction such as "put the blue cube by the wall", typed or spoken (through speech-to-text, as our voice assistant does), and turn it into a goal for the robot.
2. **Give the goal to the policy:** train one goal-conditioned policy that follows many instructions, instead of one fixed task.
3. **Learn vision too:** replace the colour detector with a learned image model (item 4 of the project [roadmap](../projects/vision-rl-pickplace/README.md#limitations-and-roadmap)), then compare against an open VLA model fine-tuned on demonstrations from our simulator.

### Telugu Voice AI Assistant

**Not used yet.** The voice assistant is not a robot: it answers questions but doesn't perform physical actions. It does have a language part (Telugu speech and text), which is the part our robot is missing. Together, the two projects cover vision, language and action, but not yet in one system.

---

**Previous:** [← VLM](10-vlm.md) · [Home](../README.md) · **Next:** [LLM →](12-llm.md)
