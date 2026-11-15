<div align="center">

# My AI Engineering Journey

**A structured guide to AI engineering, from computer fundamentals to LLMs, RAG and robotics, applied in a live production project.**

![Topics](https://img.shields.io/badge/topics-33-2563eb)
![Applied in project](https://img.shields.io/badge/applied%20in%20project-13%20of%2033-16a34a)
![Deployed on](https://img.shields.io/badge/deployed%20on-Azure-0078d4)

Yodha AI · SEVT Logistics Private Limited

[Topics](#topics) · [Featured Project](#featured-project) · [Learning Path](#learning-path) · [Roadmap](ROADMAP.md)

</div>

---

## Overview

Each topic is a short, consistent page: a clear explanation, a flow diagram, and how the concept is applied in our live **Telugu Voice AI Assistant**.

| At a glance | Detail |
|---|---|
| **Topics** | 33, grouped into 8 areas |
| **Applied in our project** | 13 of the 33 topics |
| **Page format** | Introduction · Definition · Rationale · Mechanism · Flow diagram · Use cases · Application in our project |
| **Featured project** | [Telugu Voice AI Assistant](projects/telugu-voice-assistant/README.md), deployed on Azure Container Apps |
| **Author** | Rishi Sasanala |

## Learning path

Topics are designed to be read in order. Each builds on the previous one.

```text
Hardware                       CPU → GPU
                                     ↓
Images and video               Image → Image Resolution → Video → Frame → FPS
                                     ↓
Language, vision and robotics  NLP → VLM → VLA → LLM → ROS
                                     ↓
Tech stack                     RAG → API → FastAPI
                                     ↓
Machine learning foundations   ML → Supervised → Unsupervised → Semi-Supervised → Reinforcement
                                     ↓
Machine learning algorithms    Linear Regression → Logistic Regression → Decision Tree → Random Forest
                               → Gradient Boosting → KNN → Naive Bayes → K-Means
                                     ↓
Deep learning                  DL → Transformers
                                     ↓
LLMs                           Prompt Engineering → Hallucination → Tokenization
```

## Topics

### Hardware

Every AI model runs on a processor. This section covers the two that matter most: the CPU, which works step by step, and the GPU, which does many calculations at once and makes training practical.

| # | Topic | Summary | Topics inside |
|---|---|---|---|
| 1 | [CPU](docs/01-cpu.md) | Step-by-step processing, binary, and how far LLM training can go on a CPU | Step-by-step (sequential) processing · Binary: the language of the CPU · Training an LLM on a CPU: how far you can go |
| 2 | [GPU](docs/02-gpu.md) | Parallel processing, number formats, and how far we trained on a GPU | Parallel processing · Number formats (bits per number) · How far you can train on a GPU · GPU memory usage |

### Images and video

Computers cannot see pictures, only numbers. This section shows how an image is stored as a grid of pixels, where each pixel holds red, green and blue values, and how resolution decides how much detail an image keeps. It then builds up to video: a sequence of images (frames) shown at a fixed speed (FPS). These basics explain how AI models are able to "see".

| # | Topic | Summary | Topics inside |
|---|---|---|---|
| 3 | [Image](docs/03-image.md) | Pixels and RGB values | Pixels · RGB colour values · How a computer processes an image |
| 4 | [Image Resolution](docs/04-image-resolution.md) | Width × height in pixels | Width × height · Megapixels (MP) · Resolution vs memory and GPU needs |
| 5 | [Video](docs/05-video.md) | Images played continuously | Video as a sequence of images · How a video is stored in a computer |
| 6 | [Frame](docs/06-frame.md) | One image inside a video | Image vs frame · Frames as the building blocks of video |
| 7 | [FPS](docs/07-fps.md) | Frames shown per second | Frames per second · 30 / 60 / 120 FPS · How much data one second of video holds |

### Language, vision and robotics

The main families of AI models: ones that understand language, ones that also see, and ones that also act. The section ends with ROS, the software that connects a robot's parts to those models.

| # | Topic | Summary | Topics inside |
|---|---|---|---|
| 8 | [NLP](docs/08-nlp.md) | Branch of AI that understands human language | What NLP is · Understanding a request: intent, action and required information |
| 9 | [VLM](docs/09-vlm.md) | Sees and understands (image + text) | Image + text understanding · How a VLM reads an image |
| 10 | [VLA](docs/10-vla.md) | Sees, understands and acts | Vision → language → action · VLM vs VLA |
| 11 | [LLM](docs/11-llm.md) | Trained on huge text to understand and generate language | GPT-style models · How an LLM reads text · How it generates an answer · Temperature |
| 12 | [ROS](docs/12-ros.md) | Connects and coordinates a robot's parts | What ROS does · How it fits with the other AI models |

### Tech stack

The building blocks used to turn a model into a working product: grounding answers in documents, and serving them over an API.

| # | Topic | Summary | Topics inside |
|---|---|---|---|
| 13 | [RAG](docs/13-rag.md) | Retrieve documents, then let the LLM answer | RAG vs fine-tuning · Document ingestion · Chunking · Embedding · Retrieval · Evaluating a RAG system |
| 14 | [API](docs/14-api.md) | How two software systems talk: request → response | Request → response · Types of APIs · REST and its methods · SOAP · REST vs SOAP |
| 15 | [FastAPI](docs/15-fastapi.md) | Building APIs in Python | Python functions as endpoints · Automatic `/docs` · Input validation · Request flow |

### Machine learning foundations

What machine learning is, and the four ways a model can learn, depending on what kind of data and feedback it gets.

| # | Topic | Summary | Topics inside |
|---|---|---|---|
| 16 | [ML](docs/16-ml.md) | Systems that learn patterns from data | AI vs ML · Rules vs learning from data · Types of ML (supervised, unsupervised, semi-supervised, reinforcement) |
| 17 | [Supervised Learning](docs/17-supervised-learning.md) | Learns from inputs with known outputs: classification and regression | Inputs + known target outputs · House-price example · Classification · Regression |
| 18 | [Unsupervised Learning](docs/18-unsupervised-learning.md) | Finds patterns in data without known outputs | No known target outputs · Customer-data example · Clustering · Anomaly detection · Supervised vs unsupervised |
| 19 | [Semi-Supervised Learning](docs/19-semi-supervised-learning.md) | Learns from a little labelled data plus a lot of unlabelled data | Labelled vs unlabelled data · 1000-emails example · How labels spread to unlabelled data · Supervised vs semi-supervised vs unsupervised |
| 20 | [Reinforcement Learning](docs/20-reinforcement-learning.md) | An agent learns decisions from rewards and penalties | Agent · Environment · Action · Reward and penalty · Bicycle example · The learning loop |

### Machine learning algorithms

The previous section covered the types of learning. Each page here is one classic algorithm: seven supervised ones, then K-Means, which is unsupervised.

| # | Topic | Summary | Topics inside |
|---|---|---|---|
| 21 | [Linear Regression](docs/21-linear-regression.md) | Predicts a continuous number by fitting a straight line | Regression · Experience → salary example · Fitting a line · Predicting a new value |
| 22 | [Logistic Regression](docs/22-logistic-regression.md) | Predicts the probability of a class, then classifies | Classification · 1000-emails spam example · Features and learned weights · Probability → decision |
| 23 | [Decision Tree](docs/23-decision-tree.md) | Reaches a decision by asking a series of questions | Series of questions · Play outside or stay inside example · Learning from historical examples |
| 24 | [Random Forest](docs/24-random-forest.md) | Many independent Decision Trees combined | Multiple trees · Voting (classification) · Averaging (regression) · Independent trees |
| 25 | [Gradient Boosting / XGBoost](docs/25-gradient-boosting.md) | Models built one after another, each fixing earlier mistakes | Sequential models · 25 × 4 maths example · Random Forest vs boosting · XGBoost |
| 26 | [KNN](docs/26-knn.md) | Predicts from the nearest existing examples | Nearest examples · Movie example · What K means |
| 27 | [Naive Bayes](docs/27-naive-bayes.md) | Uses probability to pick the most likely category | Probability per category · Email-words example · Naive Bayes vs Logistic Regression |
| 28 | [K-Means](docs/28-k-means.md) | Groups similar data points into K clusters | Clustering · Supermarket-customers example · Low / medium / high spenders · K-Means vs KNN |

### Deep learning

Neural networks with many layers, how they are trained, and the Transformer architecture that modern LLMs are built on.

| # | Topic | Summary | Topics inside |
|---|---|---|---|
| 29 | [DL](docs/29-dl.md) | Multi-layer neural networks; epochs, batches, training | Neural networks · Epoch · Batch · Batch size · The training loop |
| 30 | [Transformers](docs/30-transformers.md) | The architecture behind LLMs: attention, Q/K/V, masking | Architecture · Embeddings · Encoder vs decoder · Self-attention · Query, Key, Value (Q/K/V) · Multi-head attention · Masking |

### LLMs

Working with large language models in practice: how to ask, why they sometimes make things up, and how text is turned into tokens.

| # | Topic | Summary | Topics inside |
|---|---|---|---|
| 31 | [Prompt Engineering](docs/31-prompt-engineering.md) | Zero/one/few-shot, chain of thought, role prompting | Zero-shot · One-shot · Few-shot · Chain of thought · Role prompting |
| 32 | [Hallucination](docs/32-hallucination.md) | When an LLM makes things up, and how to reduce it | What hallucination is · Why it happens · How to reduce it (RAG, prompts, validation, temperature, citations) |
| 33 | [Tokenization](docs/33-tokenization.md) | Text → tokens → token IDs | Token · Tokenization · Token IDs · Why tokens matter (limits and cost) |

## Featured project

**Telugu Voice AI Assistant.** A voice assistant that listens in Telugu, grounds its answers in vehicle manuals using Retrieval-Augmented Generation (RAG), and replies in spoken Telugu.

```text
User speaks (Telugu)
        ↓
Speech to text
        ↓
FastAPI server receives the question (REST API on Azure)
        ↓
Translate the query (Telugu → English)
        ↓
RAG: retrieve the matching section from ChromaDB (vehicle manuals)
        ↓
LLM (Gemini) generates the answer using the manual section
        ↓
Text to speech (FastAPI: POST /tts)
        ↓
User hears the answer
```

| Resource | Link |
|---|---|
| **Project overview** | [Telugu Voice AI Assistant](projects/telugu-voice-assistant/README.md) |
| **Topic-by-topic mapping** | [Topics covered in this project](projects/telugu-voice-assistant/README.md#topics-covered-in-this-project) |
| **Live application** | [Open the assistant](https://telugu-voice-app.gentletree-f4002f64.southindia.azurecontainerapps.io/assistant) |
| **Source code** | [tiny-llm-jetson](https://github.com/SEVT-Logistics-Private-Limited/tiny-llm-jetson) |

## Repository structure

```text
.
├── docs/         33 topic pages, numbered in reading order
├── projects/     Telugu Voice AI Assistant write-up and screenshots
├── assets/       Images used in the topic pages
└── ROADMAP.md    Covered and planned topics
```
