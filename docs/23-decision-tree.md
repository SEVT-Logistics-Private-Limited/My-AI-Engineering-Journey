# 23. Decision Tree

> A Decision Tree is a supervised learning algorithm that makes predictions by asking a series of questions and following the answers to reach a decision.

## Introduction

People make many decisions by asking themselves a few simple questions in order. A Decision Tree does exactly the same thing, with questions it has learned from data.

## What is it?

A **Decision Tree (DT)** is a [supervised learning](17-supervised-learning.md) algorithm that **makes predictions by asking a series of questions and following the answers to reach a decision**.

### Example: play outside or stay inside?

Imagine you want to play a game and have to decide whether to play outside or stay inside. You ask yourself:

1. Is it raining?
2. Is it sunny?
3. Is it too cold?

Based on the answers, you make the decision. That is exactly the idea behind a Decision Tree.

## Why is it needed?

- The decision can be read and explained step by step, unlike many other models.
- It works for both classification (play / don't play) and regression (predict a number).
- It handles different kinds of inputs (yes/no, categories, numbers) without much preparation.

## How does it work?

1. You give the model some historical examples (the weather on past days, and whether you played outside).
2. The model studies them and learns the patterns: which questions separate the outcomes best.
3. It arranges those questions as a tree: the most useful question at the top, follow-up questions below.
4. For a new case, it starts at the top, follows the answers down the branches, and the end of the path is the prediction.

## Flow diagram

```text
              Is it raining?
              /            \
           Yes              No
            ↓                ↓
       Stay inside     Is it too cold?
                        /          \
                     Yes            No
                      ↓              ↓
                 Stay inside    Play outside
```

## Real-world use cases

- Loan approval (income, credit history, existing loans)
- Medical diagnosis from symptoms
- Deciding which customers get an offer
- Troubleshooting guides

## In our project

**Not used yet.** We don't train a Decision Tree on historical examples.

---

**Previous:** [← Logistic Regression](22-logistic-regression.md) · [Home](../README.md) · **Next:** [Random Forest →](24-random-forest.md)
