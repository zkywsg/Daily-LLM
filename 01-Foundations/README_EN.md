[English](README_EN.md) | [中文](README.md)

# Phase 01: Foundations

> This phase is about learning how to turn a real problem into something measurable, trainable, and debuggable, not just how to call a model.

## Why This Phase Matters

Many people jump straight into Transformers, RAG, or agents, then get blocked by more basic questions: How should the target be defined? Which metric actually matters? Why does offline performance look good while production results degrade? Where did leakage happen?

Phase 01 builds that missing judgment first.

The entire phase follows one running case study: a consumer finance platform wants to predict whether a user will default within the next 90 days. You will first build a trustworthy classical ML baseline, then move into deep learning to understand both why neural networks are powerful and why they are not always the right answer.

## Who This Phase Is For

- Software engineers moving into AI or ML
- LLM application builders who want stronger modeling intuition
- Beginners who can use libraries but do not yet understand why models work or fail

## What You Will Get

- A practical framework for turning business goals into samples, features, labels, losses, and metrics
- A clearer sense of when to use logistic regression, decision trees, SVMs, K-Means, and PCA
- An operational understanding of forward propagation, backpropagation, and the training loop
- The ability to distinguish real signal from misleading offline results
- A smoother transition into CNNs, sequence models, and Transformers

## Running Case Study

### Credit Risk: 90-Day Default Prediction

Assume you work on a consumer finance platform. Every day, the system needs to decide:

- which users can receive a higher credit line,
- which users require stricter review,
- which users should trigger early warning workflows.

If high-risk users are misclassified as low-risk, the company absorbs bad debt. If low-risk users are flagged as high-risk, the business loses good customers and revenue.
That makes credit risk a strong teaching case for fundamentals: the objective is concrete, the cost of mistakes is real, and the metric choice cannot be superficial.

## Learning Map

| Module | What You Will Learn | Why It Matters |
| --- | --- | --- |
| [Machine Learning Foundations](machine-learning/README_EN.md) | Supervised learning, unsupervised learning, evaluation, generalization, feature engineering, baseline modeling | Builds the decision framework for turning problems into models |
| [Deep Learning Basics](deep-learning-basics/README_EN.md) | MLPs, activations, losses, backpropagation, optimizers, training stability | Explains why representation learning matters and why training succeeds or fails |

Recommended order:

1. Finish the machine learning chapter first to build the problem-definition and evaluation framework.
2. Then move to deep learning to understand how neural networks extend beyond linear decision boundaries.
3. After that, continue to Phase 02 for more specialized architectures and training patterns.

## Highlights

- **Decision-making over algorithm listing**: each chapter explains when to choose a method, not just what it is.
- **Minimal math with maximum payoff**: only the derivations that improve understanding stay.
- **A real business frame, not toy examples**: the same credit-risk case is used to discuss metrics, tradeoffs, and error analysis.
- **Engineering-first explanations**: data splits, leakage, imbalance, instability, and debugging order are all part of the story.

## Common Mistakes Before This Phase

- Assuming a more complex model is automatically better
- Looking only at accuracy while ignoring recall, AUC, and business cost
- Accidentally leaking future information into training data
- Memorizing conclusions without understanding why the model behaves that way

## How To Study This Phase

- Treat the two chapters as one continuous story rather than two unrelated topics.
- When you encounter a metric, equation, or training loop, map it back to the case study and ask what risk it helps reduce.
- If your long-term goal is LLM engineering, the value of this phase is judgment, not nostalgia for simpler models.

## Next Step

After this phase, move on to [CNN Architectures](../02-Neural-Networks/cnn-architectures/README_EN.md).
At that point, you will not just know that networks can be deeper. You will already understand why stronger inductive bias matters and why training stability stays central as systems scale.
