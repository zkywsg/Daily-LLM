[English](README_EN.md) | [中文](README.md)

# Deep Learning Basics

> This chapter addresses a practical question: when handcrafted features and linear boundaries stop being enough for a risk problem, why can neural networks learn stronger representations, and why does training sometimes work and sometimes fail?

## Learning Goals

By the end of this chapter, you should be able to answer:

1. Why can “multiple layers + nonlinearity” express patterns that linear models cannot?
2. What is backpropagation actually computing, and how does it connect to the training loop?
3. When training becomes unstable or loss stops improving, what should you check first?

## 1. Why Classical ML Starts to Plateau

Continue the same credit-risk case study from the previous chapter.
Logistic regression can be excellent on structured data, but it starts to struggle when the signal depends on richer interactions:

- a high credit line is not necessarily dangerous, but “high line + sudden utilization spike + recent income volatility” might be,
- one late payment alone may not matter much, but “when it happened + how spending changed + how repayment behavior evolved” can matter a lot,
- you can manually engineer some crosses, but you cannot enumerate every meaningful combination.

This is where representation learning becomes valuable. The model is no longer only learning a decision boundary. It is also learning features that are better aligned with the task.

## 2. From Linear Models to Neural Networks

A single neuron is still just a linear transformation followed by a nonlinear activation:

$$
a = \phi(w^\top x + b)
$$

If you stack only linear layers, the whole network collapses into one linear transformation.
The real jump in expressiveness comes from inserting nonlinearities between layers.

A multilayer perceptron (MLP) can be written as:

$$
h^{(1)} = \phi(W^{(1)}x+b^{(1)}), \quad
h^{(2)} = \phi(W^{(2)}h^{(1)}+b^{(2)}), \quad
\hat{y}=W^{(3)}h^{(2)}+b^{(3)}
$$

The progression is:

1. the first layer combines raw features into local patterns,
2. the next layer recombines those patterns into higher-order interactions,
3. the output layer uses that learned representation to make the final prediction.

Deep networks are stronger not because “more layers” is magic, but because they can progressively rewrite the representation itself.

## 3. Forward Propagation: How the Model Produces a Prediction

Forward propagation means computing from input to output through the network:

1. features enter the first layer,
2. each layer applies a linear transformation and then an activation,
3. the last layer produces logits or predicted values,
4. a loss function measures the gap between prediction and truth.

For binary credit-risk prediction, a common form is:

$$
z = W^{(L)}h^{(L-1)} + b^{(L)}, \quad
\hat{p} = \sigma(z)
$$

Here $\hat{p}$ can be interpreted as the predicted default probability.

Forward propagation gives the answer. Loss functions and backpropagation explain how the model learns to improve that answer.

## 4. Loss Functions Define What “Good” Means

Common pairings:

- regression: linear output + MSE,
- binary classification: sigmoid + BCE,
- multi-class classification: logits + cross-entropy.

Binary cross-entropy takes the form:

$$
L = -\big(y\log(\hat{p}) + (1-y)\log(1-\hat{p})\big)
$$

If the true user is risky but the model assigns a low default probability, the loss becomes large.
Everything the optimizer does afterward is an attempt to reduce that loss.

## 5. What Backpropagation Is Actually Doing

Backpropagation is not a mysterious extra algorithm. It is the chain rule applied systematically over a computation graph.

For a two-layer network:

$$
\frac{\partial L}{\partial W^{(2)}}=
\frac{\partial L}{\partial z^{(2)}}\frac{\partial z^{(2)}}{\partial W^{(2)}},
\quad
\frac{\partial L}{\partial W^{(1)}}=
\frac{\partial L}{\partial z^{(2)}}
\frac{\partial z^{(2)}}{\partial h^{(1)}}
\frac{\partial h^{(1)}}{\partial z^{(1)}}
\frac{\partial z^{(1)}}{\partial W^{(1)}}
$$

Intuitively it does two things:

1. measure how wrong the final prediction is,
2. assign that error back through the network to each parameter.

This maps directly to code:

- `loss = criterion(logits, y)` defines the error,
- `loss.backward()` propagates that error backward and computes gradients,
- `optimizer.step()` updates the parameters using those gradients.

Backpropagation is best understood as responsibility assignment: which parameter contributed to the final mistake, and by how much?

## 6. Choosing Activation Functions

| Function | Common use | Strength | Risk |
| --- | --- | --- | --- |
| ReLU | Default hidden-layer choice | Simple, stable, fast | Can create dead neurons |
| Leaky ReLU | Alternative to ReLU | Keeps a small gradient on negative values | Adds a hyperparameter |
| Sigmoid | Binary output layer | Easy probability interpretation | Saturation leads to tiny gradients |
| Tanh | Common in older sequence models | Zero-centered | Also saturates |
| Softmax | Multi-class output layer | Produces normalized class probabilities | Needs numerical care |

Use ReLU-family activations in hidden layers by default and choose the output activation according to the task.
If you place sigmoid throughout a deep hidden stack, vanishing gradients become much more likely.

## 7. Optimizers and the Training Loop

| Optimizer | Update intuition | Typical use |
| --- | --- | --- |
| SGD | Take one step along the current gradient | Strong baseline, large-scale training |
| Momentum | Add inertia to gradients | Reduce oscillation |
| Adam | Adapt learning rates per parameter | Fast start for small and medium models |
| AdamW | Adam with decoupled weight decay | Common modern default |

Standard training loop:

```python
for epoch in range(num_epochs):
    model.train()
    logits = model(x_batch)
    loss = criterion(logits, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The logic behind it matters:

- `forward` produces predictions,
- `criterion` defines deviation,
- `backward` computes responsibility,
- `step` actually moves the parameters.

The `zero_grad -> backward -> step` order is the skeleton of training.

## 8. Training Stability: Why Models Suddenly Stop Learning

### 8.1 Initialization

- Xavier/Glorot is often used with Tanh or Sigmoid,
- He initialization is often used with ReLU.

If initialization is too large, activations may explode. If it is too small, gradients may fade out.

### 8.2 Normalization

Batch normalization often takes the form:

$$
\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

Its value is not cosmetic. It often makes training more stable and learning rates easier to manage.

### 8.3 Regularization

- Dropout randomly suppresses neurons to reduce co-adaptation,
- weight decay penalizes large weights,
- early stopping prevents prolonged overfitting.

### 8.4 Debugging Order for Unstable Training

1. verify the data and labels,
2. check whether the learning rate is too high or too low,
3. inspect initialization,
4. confirm input scaling and normalization,
5. tune regularization strength.

When training fails, inspect the input and learning rate before reaching for more elaborate tricks.

## 9. Running Case Study Upgrade: From Logistic Regression to an MLP

### 9.1 Why Try an MLP

Stay with the same default-prediction task.
If you suspect risk emerges from higher-order behavior interactions, an MLP can learn richer feature combinations automatically.

That still does not mean deep learning is always better.
On purely tabular data, classical ML and tree-based models are often extremely strong. Whether an MLP is worth using depends on data scale, feature type, maintenance cost, and the size of the gain.

### 9.2 Minimal PyTorch Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RiskMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = RiskMLP(in_dim=num_features)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(num_epochs):
    logits = model(x_batch)
    loss = criterion(logits, y_batch.float())

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### 9.3 How To Compare It with Logistic Regression

Do not ask only which model has lower loss. Compare:

- whether AUC and Recall improve in a meaningful way,
- whether the gain is stable on validation and out-of-time data,
- whether training and deployment cost are justified,
- whether interpretability still meets business and compliance needs.

This is the core of engineering judgment: performance is only one dimension.

## 10. Troubleshooting and Common Mistakes

- Symptom 1: loss does not decrease.
  Check first: labels, learning rate, input scaling.
- Symptom 2: training looks good but validation degrades.
  Check first: overfitting, regularization, training duration.
- Symptom 3: exploding gradients or NaNs.
  Check first: learning rate, initialization, numerical stability, gradient clipping.
- Mistake 1: deeper always means better.
  Fix: capacity must match data size and task complexity.
- Mistake 2: Adam is always best.
  Fix: final judgment belongs to validation behavior and target metrics.
- Mistake 3: deep learning automatically beats classical ML.
  Fix: on structured tabular tasks, that is often false.

## 11. What You Should Remember

- The core value of deep learning is representation learning, not just more parameters.
- Backpropagation assigns the final error back to each parameter through the chain rule.
- Training success is often determined first by data quality, learning rate, initialization, and normalization.
- In real systems, the choice to use deep learning must be justified by gain, stability, and interpretability together.

## Next Step

This chapter gives you the basic neural network intuition, but the MLP is only the beginning.
Next you will see why image tasks benefit from CNN inductive bias and why sequence tasks need architectures that better model order and dependency.

- Continue to [CNN Architectures](../../02-Neural-Networks/cnn-architectures/README_EN.md)
- Or move to [Sequence Models](../../02-Neural-Networks/sequence-models/README_EN.md)

---

**Previous**: [Machine Learning Foundations](../machine-learning/README_EN.md) | **Next**: [CNN Architectures](../../02-Neural-Networks/cnn-architectures/README_EN.md)
