# Roadmap

[Home](README.md)

Status as of November 2026. Project plans come from each project's own roadmap: [Vision-RL Pick-and-Lift](projects/vision-rl-pickplace/README.md#limitations-and-roadmap) and the [Yodha implementation repository](https://github.com/SEVT-Logistics-Private-Limited/tiny-llm-jetson#roadmap).

## Done

| Area | Work |
|---|---|
| **Yodha** | Telugu voice assistant v10.1, in production on Azure: Telugu-first retrieval over the vehicle manuals, three-model LLM fallback, token streaming with sentence-by-sentence speech, sign-in and rate limiting, tests before every deployment |
| **Telugu language model** | A 25M-parameter GPT-style model trained from scratch on 50,000 Telugu sentences, plus a conversational fine-tune |
| **GPU scaling study** | The same model trained at sizes from 984 parameters to 3.07 billion, to find how large a model a Jetson Orin Nano and an A100 can train |
| **Vision-RL Pick-and-Lift** | Simulation prototype: camera perception, state estimation and a SAC policy with 99% success from the camera on 100 held-out tests; pick-and-place extension at 100% on held-out tests (learned pick + scripted place); 122 tests in CI |
| **Engineering notes** | All 34 topics in the [notes](docs/README.md) |

## In progress

| Area | Work |
|---|---|
| **Edge compression for Yodha** | A Telugu LoRA fine-tune of a pretrained 3B model, then ternary quantisation-aware training, so the assistant can run offline ([code](https://github.com/SEVT-Logistics-Private-Limited/tiny-llm-jetson/tree/main/compression)) |

## Planned

### Projects

| Project | Next steps |
|---|---|
| **Yodha** | A retrieval evaluation set, with the knowledge base defined as infrastructure-as-code · replace the hosted LLM with the compressed Telugu model · run offline on a Jetson Orin Nano |
| **Vision-RL Pick-and-Lift** | Domain randomisation (friction, mass, lighting, camera pose) · learned detection for cluttered scenes · a wrist camera · visual RL from pixels · an end-to-end learned pick-and-place policy (goal-conditioned, Hindsight Experience Replay) · ROS 2 nodes, then a real arm |

### Learning topics

| Area | Topics |
|---|---|
| **Audio** | How sound becomes numbers · Speech to text and text to speech |
| **Computer basics** | Digital logic design (DLD) · Operating system and kernel · High-level vs low-level languages · Embedded systems |
| **Programming** | Python · C and C++ · Java |
| **AI** | Open models (DeepSeek, Llama, Qwen) · Tools and skills for LLMs · AI agents and automation · RPA (Robotic Process Automation) |
