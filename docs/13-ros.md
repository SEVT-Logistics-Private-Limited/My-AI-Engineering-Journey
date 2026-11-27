# 13. ROS (Robot Operating System)

> ROS is a robotics software framework that provides tools and communication mechanisms to build and control robots.

## Introduction

A robot is many parts working together: cameras, sensors, arms, a microphone. Something has to connect them so they can talk to each other. That's ROS.

## What is it?

**ROS (Robot Operating System)** is a **robotics software framework** that provides **tools and communication mechanisms** to build and control robots.

## Why is it needed?

Imagine you're building a robot. It has:
- a camera
- sensors
- arms
- a microphone

All of these **need to communicate with each other**. ROS helps **connect, communicate between and coordinate** these components.

## How does it work?

ROS acts like the robot's nervous system. Each part (camera, sensor, arm) sends and receives messages through ROS, so information from the camera can reach the AI brain, and commands from the AI brain can reach the arms.

### How it fits with the other AI models

| Name | What it does |
|---|---|
| NLP | Understands human language |
| LLM | Understands and generates language using a trained model |
| VLM | Understands images and text, and generates text responses |
| VLA | Connects vision, language and physical actions |
| **ROS** | **Connects and coordinates robot software and hardware** |

## Flow diagram

```text
Camera  ─┐
Sensors ─┤
Arms    ─┼──→  ROS  ←──→  AI brain (VLA)
Mic     ─┘
        (connects, communicates, coordinates)
```

```text
Camera sees the bottle → ROS → VLA decides "pick it up" → ROS → Arm moves
```

## Real-world use cases

- Research and university robots
- Warehouse and delivery robots
- Drones
- Industrial robotic arms

## In our project

### Vision-RL Pick-and-Place

**Not used yet.** Our robot runs in a simulator as one Python program, so its parts talk through ordinary function calls and don't need ROS. The parts are already split the way ROS splits a robot into nodes, which makes ROS the natural next step when the project moves to a real robot. [Project page →](../projects/vision-rl-pickplace/README.md)

How each part would map to ROS 2:

| Part of our robot | Today, in simulation | With ROS 2, on a real robot |
|---|---|---|
| **Camera** | `RGBDCamera` renders colour and depth images | A camera driver node (for example, for an Intel RealSense) publishes the images and the camera's settings (`sensor_msgs/Image`, `sensor_msgs/CameraInfo`) |
| **Computer vision** | `ColorObjectDetector` finds the cube and its 3D position | A perception node publishes the cube's position (`geometry_msgs/PoseStamped`) |
| **State estimator** | `ObjectStateEstimator` keeps tracking the cube when the arm hides it | An estimator node combines the cube position with the arm's joint readings (`/joint_states`) |
| **Learned policy** | The SAC policy chooses each hand move of the pick, 24 times a second | A policy node publishes hand moves (`geometry_msgs/TwistStamped`) and gripper commands |
| **Carry and place** | `PickPlacePipeline` carries the held cube to the goal in small straight-line steps, sets it down and checks the result with the camera | A place node sends the carry and place targets to MoveIt |
| **Arm control** | `PandaController` turns hand moves into joint commands (inverse kinematics) | MoveIt Servo and the arm's driver move the real motors |

```text
Today: one program, parts call each other directly
Camera → Computer vision → State estimator → Pick policy → Carry and place → Inverse kinematics → Simulated arm

With ROS 2: separate nodes exchange messages
Camera node
    ↓ colour + depth images
Perception node
    ↓ cube position
Estimator node  ←── /joint_states (arm sensors)
    ↓ tracked cube position
Policy node
    ↓ hand moves for the pick, 24 times a second
Place node
    ↓ carry and place targets, once the cube is lifted
MoveIt + arm driver
    ↓
Real robot arm
```

Why not ROS yet:
- **No hardware yet.** In simulation, one program is simpler and faster. ROS earns its place when real cameras and motors need connecting.
- **Exactly repeatable results.** One program steps the simulator in lockstep, so the same seed always gives the same result. Separate ROS nodes run on their own timing, which makes exact repeats harder.
- **Simple setup and testing.** The project installs with `pip` and runs its 122 automated tests on every change, with no ROS installation.

Already ROS-ready:
- The computer vision needs only colour, depth, the camera's intrinsics (`K`), and where the camera is and which way it points (`T_world_cam`): exactly what a ROS camera driver provides (`Image`, `CameraInfo` and a `tf` transform).
- Each part has one job and a small interface, so each can become one node.
- The state estimator uses only signals every real arm reports (joint angles and gripper width), which ROS publishes on `/joint_states`.

Next step: wrap each part as a ROS 2 node, then run the same pipeline on a real arm, after the domain-randomisation step in the project [roadmap](../projects/vision-rl-pickplace/README.md#limitations-and-roadmap).

### Telugu Voice AI Assistant

**Not used yet.** ROS is for robots, and our voice assistant is software only.

---

**Previous:** [← LLM](12-llm.md) · [Home](../README.md) · **Next:** [RAG →](14-rag.md)
