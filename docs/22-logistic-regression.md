# 22. Logistic Regression

> Logistic Regression is a supervised learning classification algorithm that predicts the probability of a class and uses that probability to make a classification decision.

## Introduction

[Linear Regression](21-linear-regression.md) predicts a number. Logistic Regression, despite its name, is used for **classification**: deciding which category something belongs to.

## What is it?

**Logistic Regression (LR)** is a [supervised learning](17-supervised-learning.md) **classification** algorithm. It **predicts the probability of a class** and **uses that probability to make the classification decision**.

### Example: 1000 emails

- You have 1000 emails, each marked spam or not spam.
- Your model studies this historical data and learns the patterns associated with **spam / not spam**.
- When a new mail comes in, the model determines spam / not spam based on those patterns.

This is where Logistic Regression helps.

## Why is it needed?

- Many decisions are yes/no: spam or not, fraud or genuine, pass or fail.
- A probability is more useful than a bare answer: "95% spam" and "55% spam" can be handled differently.
- It is simple, fast and easy to explain.

## How does it work?

1. Collect examples with known categories (emails marked spam / not spam).
2. The model looks at the available features of each example, for instance:
   - number of suspicious words
   - links
   - sender information
3. During training it learns a **weight** for each feature: how strongly that feature points towards spam.
4. For a new email, it combines the features and their weights into a **probability** between 0 and 1.
5. The probability is turned into a decision, usually: above 0.5 → spam, otherwise not spam.

## Flow diagram

```text
New email
        ↓
Features (suspicious words, links, sender info)
        ↓
Learned weights
        ↓
Probability of spam (e.g. 0.92)
        ↓
Decision: Spam / Not spam
```

## Real-world use cases

- Spam detection
- Fraud detection (fraud / genuine)
- Medical screening (disease / no disease)
- Predicting whether a customer will leave

## In our project

**Not used yet.** We don't train a classifier on labelled examples.

---

**Previous:** [← Linear Regression](21-linear-regression.md) · [Home](../README.md) · **Next:** [Decision Tree →](23-decision-tree.md)
