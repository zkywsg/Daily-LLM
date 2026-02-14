# Deep Learning Basics

**[English](README_EN.md) | [中文](README.md)**

## Overview

Deep Learning uses neural networks with multiple layers to learn hierarchical representations of data. This guide covers the fundamentals of neural networks, activation functions, and training techniques.

## Neural Network Fundamentals

### 1. Perceptron to Multi-Layer Networks

**Single Perceptron**:
```
output = activation(w · x + b)
```

**Multi-Layer Perceptron (MLP)**:
- Input layer → Hidden layer(s) → Output layer
- Non-linear activation enables universal approximation

### 2. Forward Propagation

```
Layer 1: z¹ = W¹x + b¹, a¹ = σ(z¹)
Layer 2: z² = W²a¹ + b², a² = σ(z²)
Output: ŷ = a²
```

### 3. Activation Functions

| Function | Formula | Range | Properties |
|----------|---------|-------|------------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | (0, 1) | Smooth, vanishing gradient |
| **Tanh** | $\tanh(x)$ | (-1, 1) | Zero-centered |
| **ReLU** | $\max(0, x)$ | [0, ∞) | Computationally efficient |
| **Leaky ReLU** | $\max(\alpha x, x)$ | (-∞, ∞) | Mitigates dying ReLU |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | (0, 1) | Multi-class output |

## Backpropagation

### Chain Rule Application

```
∂L/∂W² = ∂L/∂a² · ∂a²/∂z² · ∂z²/∂W²
∂L/∂W¹ = ∂L/∂a² · ∂a²/∂z² · ∂z²/∂a¹ · ∂a¹/∂z¹ · ∂z¹/∂W¹
```

### Training Loop

```python
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
```

## Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# Training setup
model = NeuralNetwork(784, 256, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
```

## Key Concepts

### 1. Weight Initialization
- **Xavier/Glorot**: $\mathcal{U}(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}})$
- **He**: $\mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})$ for ReLU

### 2. Normalization
- **Batch Normalization**: Normalize activations
  - $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$
  - Stabilizes training, allows higher learning rates

### 3. Regularization
- **Dropout**: Randomly zero neurons during training
- **L2 Regularization**: Weight decay $\lambda \sum w^2$
- **Early Stopping**: Halt when validation loss plateaus

## Optimization Algorithms

| Algorithm | Update Rule | Characteristics |
|-----------|-------------|-----------------|
| **SGD** | $\theta = \theta - \alpha \nabla J$ | Simple, may oscillate |
| **Momentum** | $v = \beta v + \nabla J$ | Accelerates convergence |
| **Adam** | Adaptive LR per parameter | Most popular, robust |
| **AdamW** | Adam + weight decay decoupling | Better regularization |

## Practical Tips

1. **Learning Rate Scheduling**: Decay over time
2. **Gradient Clipping**: Prevent exploding gradients
3. **Batch Size**: Trade-off between speed and stability
4. **Monitoring**: Track train/validation loss curves

## Common Architectures

- **MLP**: Universal function approximator
- **CNN**: Spatial hierarchies (see CNN Architectures)
- **RNN**: Sequential data (see Sequence Models)
- **Transformer**: Attention-based (see Transformers)

---

**Previous**: [Machine Learning](../machine-learning/README.md) | **Next**: [CNN Architectures](../../02-Neural-Networks/cnn-architectures/README.md)
