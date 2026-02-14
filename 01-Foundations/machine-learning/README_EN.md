# Machine Learning Foundations

**[English](README_EN.md) | [中文](README.md)**

## Overview

Machine Learning (ML) enables computers to learn patterns from data without explicit programming. This guide covers fundamental algorithms, mathematical foundations, and practical implementations.

## Core Concepts

### 1. Supervised Learning

**Definition**: Learning from labeled data (input-output pairs)

**Key Algorithms**:
- **Linear Regression**: Predict continuous values
  - Formula: $y = wx + b$
  - Loss: MSE = $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$
  
- **Logistic Regression**: Binary classification
  - Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - Loss: Cross-Entropy
  
- **Decision Trees**: Rule-based learning
  - Split criteria: Gini impurity, Information gain
  
- **SVM**: Maximum margin classifier
  - Kernel trick for non-linear boundaries

### 2. Unsupervised Learning

**Definition**: Finding patterns in unlabeled data

**Key Algorithms**:
- **K-Means Clustering**: Partition into k clusters
  - Objective: Minimize within-cluster variance
  
- **PCA**: Dimensionality reduction
  - Find principal components via eigen decomposition
  
- **Gaussian Mixture Models**: Probabilistic clustering

### 3. Key Concepts

| Concept | Description | Formula |
|---------|-------------|---------|
| **Bias-Variance Tradeoff** | Model complexity balance | $Error = Bias^2 + Variance + Noise$ |
| **Overfitting** | Model memorizes training data | Regularization solution |
| **Cross-Validation** | Robust performance estimation | k-fold CV |
| **Gradient Descent** | Optimization algorithm | $\theta = \theta - \alpha \nabla J(\theta)$ |

## Implementation Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate data
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse:.4f}")
print(f"Coefficients: w={model.coef_[0][0]:.2f}, b={model.intercept_[0]:.2f}")
```

## Best Practices

1. **Feature Engineering**: Critical for model performance
2. **Data Preprocessing**: Normalization, handling missing values
3. **Model Selection**: Cross-validation for robust evaluation
4. **Regularization**: L1/L2 to prevent overfitting

## Mathematical Foundations

### Linear Algebra
- Vectors, matrices, operations
- Eigenvalues and eigenvectors
- Matrix decomposition (SVD, QR)

### Probability & Statistics
- Random variables, distributions
- Bayes' theorem: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
- Maximum Likelihood Estimation

### Optimization
- Convex optimization
- Lagrange multipliers
- Convergence analysis

## Further Reading

- "Pattern Recognition and Machine Learning" - Bishop
- "The Elements of Statistical Learning" - Hastie, Tibshirani
- scikit-learn documentation

---

**Next**: [Deep Learning Basics](../deep-learning-basics/README.md)
