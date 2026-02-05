# Hyperparameter Search and AutoML

**[English](README.md) | [中文](README_CN.md)**

## Table of Contents

1. [Background](#1-background)
2. [Core Concepts](#2-core-concepts)
3. [Mathematical Principles](#3-mathematical-principles)
4. [Code Implementation](#4-code-implementation)
5. [Experimental Comparison](#5-experimental-comparison)
6. [Best Practices and Common Pitfalls](#6-best-practices-and-common-pitfalls)
7. [Summary](#7-summary)

---

## 1. Background

### 1.1 Importance of Hyperparameters

Hyperparameters significantly affect model performance:
- Learning rate: Too large leads to non-convergence, too small leads to slow convergence
- Batch size: Affects generalization and training speed
- Model architecture: Determines expressive power

### 1.2 Problems with Manual Tuning

- Time-consuming: Requires multiple experiments
- Experience-dependent: Requires domain knowledge
- Local optima: Easy to get stuck in local optima

---

## 2. Core Concepts

### 2.1 Search Strategies

| Strategy | Principle | Advantages | Disadvantages |
|-----------|------------|--------------|----------------|
| **Grid** | Exhaustive search of all combinations | Comprehensive | Curse of dimensionality |
| **Random** | Random sampling | Efficient | May miss optimal |
| **Bayesian** | Model based on history | Efficient and precise | Complex |
| **Early Stop** | Terminate poor experiments early | Saves resources | Possible misjudgment |

### 2.2 Key Hyperparameters

- **Learning rate**: Most important
- **Batch size**: Affects stability and speed
- **Regularization**: dropout, weight decay
- **Network architecture**: Number of layers, width

---

## 3. Mathematical Principles

### 3.1 Objective

$$
x^* = \arg\max_{x \in \mathcal{X}} f(x)
$$

Where:
- $x$: Hyperparameter configuration
- $f(x)$: Validation set performance

### 3.2 Gaussian Process

Model $f$ using Gaussian process:

$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$

Acquisition function (Expected Improvement):

$$
\text{EI}(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)]
$$

---

## 4. Code Implementation

### 4.1 Optuna Example

```python
import optuna
from transformers import TrainingArguments, Trainer

def objective(trial):
    """Objective function"""
    # Define search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 1, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    # Create training arguments
    args = TrainingArguments(
        output_dir="./tune",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch"
    )

    # Training
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=eval_data)
    trainer.train()

    # Return validation metric (Optuna defaults to minimization, so take negative value)
    eval_result = trainer.evaluate()
    return -eval_result["eval_loss"]

# Create study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Best parameters
print(f"Best trial: {study.best_trial.value}")
print(f"Best params: {study.best_trial.params}")
```

### 4.2 Ray Tune Example

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_func(config):
    """Training function"""
    # Use hyperparameters from config for training
    args = TrainingArguments(
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"]
    )

    trainer = Trainer(model=model, args=args, ...)
    trainer.train()

    # Report results
    eval_result = trainer.evaluate()
    tune.report(loss=eval_result["eval_loss"])

# Search space
search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([4, 8, 16]),
    "epochs": tune.choice([1, 2, 3])
}

# Early stopping scheduler
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1
)

# Run search
analysis = tune.run(
    train_func,
    config=search_space,
    num_samples=20,
    scheduler=scheduler
)

# Best configuration
best_config = analysis.get_best_config(metric="loss", mode="min")
print(best_config)
```

---

## 5. Experimental Comparison

### 5.1 Efficiency Comparison

| Strategy | Trial Count | Time to Find Optimal | Resource Consumption |
|-----------|--------------|----------------------|---------------------|
| **Grid (100 points)** | 100 | 100 | High |
| **Random** | 50 | ~30 | Medium |
| **Bayesian** | 30 | ~20 | Low |
| **Bayesian + EarlyStop** | 20 | ~15 | Lowest |

### 5.2 Effect Comparison

Assuming true optimal Loss = 1.5:

| Strategy | Best Loss | Gap |
|-----------|-----------|------|
| Manual | 1.8 | +20% |
| Grid | 1.6 | +6.7% |
| Random | 1.55 | +3.3% |
| Bayesian | 1.52 | +1.3% |

---

## 6. Best Practices and Common Pitfalls

### 6.1 Best Practices

1. **From coarse to fine**: First search a wide range, then fine-tune
2. **Log scale**: Use log scale for learning rate, etc.
3. **Early stopping**: Terminate poor experiments early
4. **Random seeds**: Fix seeds to ensure reproducibility
5. **Parallelization**: Utilize multiple GPUs for parallel search

### 6.2 Common Pitfalls

1. **Overfitting validation set**: Multiple searches lead to overfitting
2. **Ignoring computational cost**: Search is more expensive than training
3. **Inappropriate search space**: Range too large or too small
4. **Not considering interactions**: Hyperparameters have interactions

---

## 7. Summary

Hyperparameter search is an important means to improve model performance:

1. **Strategy selection**: Bayesian > Random > Grid
2. **Key hyperparameters**: Learning rate is most important
3. **Early stopping**: Saves computational resources
4. **Tools**: Optuna, Ray Tune

**Recommended workflow**:
1. Wide-range random search (10-20 times)
2. Fine-tuned Bayesian optimization (20-30 times)
3. Train final model with best parameters
