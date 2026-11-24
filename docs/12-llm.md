# 12. LLM (Large Language Model)

> An LLM is an AI model trained on a huge amount of text data to understand and generate human language.

## Introduction

ChatGPT, Claude and Gemini are all LLMs. They can answer questions, write and summarise, and all of them work by predicting text one token at a time.

## What is it?

An **LLM (Large Language Model)** is an AI model (a neural network) **trained on a very large amount of text data** to learn patterns in language, the relationships between tokens, and to **predict the next token** in a sequence.

**In simple words:** it learns patterns from text, and uses those patterns to understand and generate language.

Examples: models from families such as **GPT, Claude, Gemini**.

### GPT-style models

**GPT** stands for **Generative Pre-trained Transformer**.

GPT-style models are generally **decoder-only** [Transformer](31-transformers.md) models designed for **autoregressive generation**: the model generates the answer **token by token**.

Examples: GPT-3, GPT-4, GPT-5, Llama, Qwen.

## Why is it needed?

A single LLM can answer questions, explain topics, write, summarise, translate and hold a conversation.

## How does it work?

### How does an LLM read text?

1. **Text input:** "I love AI"
2. **Tokenization:** ["I", "love", "AI"]
3. **Token IDs:** [40, 2031, 1589]
4. **Embeddings:** a numerical vector representing each token
5. **Transformer** processes the embeddings

### How does it generate an answer?

Predict the next token → add it to the context → predict the next token → repeat.

### Temperature

**Temperature** controls the **randomness of token selection** during generation. It affects the generation (sampling) process.
- **Low temperature:** focused, predictable answers
- **High temperature:** more random and creative answers

## Flow diagram

How an LLM reads text:

```text
Text input: "I love AI"
        ↓
Tokenization: ["I", "love", "AI"]
        ↓
Token IDs: [40, 2031, 1589]
        ↓
Embeddings (numbers for each token)
        ↓
Transformer processes the embeddings
```

How a GPT-style model generates an answer:

```text
Input tokens
     ↓
Transformer
     ↓
Predict next token
     ↓
Add token to context
     ↓
Predict next token
     ↓
Repeat
```

## Real-world use cases

- Chat assistants (ChatGPT, Claude, Gemini)
- Writing, summarising and translating
- Coding help
- Voice assistants

## In our project

**Used, in two ways.**

### 1. Gemini powers the live assistant

The LLM does three jobs in our voice assistant:

| Job | Example |
|---|---|
| **Answer questions** | Generates Yodha's Telugu answer |
| **Translate** | Telugu question → English for the manual search |
| **Speech to text (backup)** | Transcribes Telugu audio when the browser can't |

**Fallback chain:** if one model fails or returns nothing, the next one is tried automatically. A transient server error gets one retry first, and the chain can be changed through the `LLM_MODELS` environment variable without a code change.

```text
gemini-3.6-flash → (fails) → gemini-3.8-flash → (fails) → gemini-3.5-flash-lite → (all fail) → polite Telugu apology
```

- Answers are limited to **500 output tokens** (`max_output_tokens`) to keep them short and fast.
- The **last 8 messages** of the conversation are sent each time, so the assistant remembers context.

### 2. We built our own GPT-style models from scratch

The same decoder-only architecture, written from scratch in PyTorch, was trained in two settings:

| Model | Parameters | Training text | Trained on |
|---|---|---|---|
| **Telugu model** | **25M** | 50,000 real Telugu sentences (2.9 million characters) | Colab A100 |
| Scaling study, rungs 1–3 | 984 → 990K → 151M | One English sentence repeated 200 times (13-character vocabulary) | Jetson Orin Nano |
| Scaling study, rungs 4–8 | 399M → 3.07B | The same sentence | Colab A100 |

The **Telugu model** is the real language model: it learned to generate Telugu text. The **scaling study** answers a different question, how large a model each GPU can train, so it uses a tiny corpus on purpose: it is a memory experiment, not a test of language ability. The memory results are in [GPU](02-gpu.md#how-far-did-we-train-on-each-gpu).

Each one is a decoder-only model that generates text **token by token**, exactly as described above. The goal is to eventually replace Gemini with our own Telugu model.

Code: [`voice-assistant/api.py`](https://github.com/SEVT-Logistics-Private-Limited/tiny-llm-jetson/blob/main/voice-assistant/api.py) (`_llm_call_with_fallback`, `telugu_llm`), [`tiny_llm.py`](https://github.com/SEVT-Logistics-Private-Limited/tiny-llm-jetson/blob/main/research/scaling/tiny_llm.py), [`tiny_llm_telugu.py`](https://github.com/SEVT-Logistics-Private-Limited/tiny-llm-jetson/blob/main/research/telugu/tiny_llm_telugu.py)

---

**Previous:** [← VLA](11-vla.md) · [Home](../README.md) · **Next:** [ROS →](13-ros.md)
