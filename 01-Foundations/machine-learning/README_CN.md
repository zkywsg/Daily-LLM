[English](README.md) | [中文](README_CN.md)

# 机器学习基础

## 概述

机器学习（Machine Learning, ML）使计算机能够在无需显式编程的情况下从数据中学习模式。本指南涵盖基础算法、数学基础和实际实现。

## 核心概念

### 1. 监督学习

**定义**：从标记数据（输入-输出对）中学习

**核心算法**：

- **线性回归**：预测连续值
  - 公式：$y = wx + b$
  - 损失：MSE = $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$

- **逻辑回归**：二分类
  - Sigmoid：$\sigma(z) = \frac{1}{1 + e^{-z}}$
  - 损失：交叉熵

- **决策树**：基于规则的学习
  - 分裂准则：基尼不纯度、信息增益

- **支持向量机（SVM）**：最大间隔分类器
  - 核技巧用于非线性边界

### 2. 无监督学习

**定义**：在未标记数据中发现模式

**核心算法**：

- **K-Means 聚类**：分割成 k 个簇
  - 目标：最小化簇内方差

- **主成分分析（PCA）**：降维
  - 通过特征分解找到主成分

- **高斯混合模型**：概率聚类

### 3. 关键概念

| 概念 | 描述 | 公式 |
|------|------|------|
| **偏差-方差权衡** | 模型复杂度的平衡 | $Error = Bias^2 + Variance + Noise$ |
| **过拟合** | 模型记忆训练数据 | 正则化解决方案 |
| **交叉验证** | 鲁棒的性能估计 | k 折交叉验证 |
| **梯度下降** | 优化算法 | $\theta = \theta - \alpha \nabla J(\theta)$ |

## 实现示例

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成数据
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练
model = LinearRegression()
model.fit(X_train, y_train)

# 评估
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse:.4f}")
print(f"Coefficients: w={model.coef_[0][0]:.2f}, b={model.intercept_[0]:.2f}")
```

## 最佳实践

1. **特征工程**：对模型性能至关重要
2. **数据预处理**：归一化、处理缺失值
3. **模型选择**：使用交叉验证进行鲁棒评估
4. **正则化**：L1/L2 防止过拟合

## 数学基础

### 线性代数
- 向量、矩阵、运算
- 特征值和特征向量
- 矩阵分解（SVD、QR）

### 概率与统计
- 随机变量、分布
- 贝叶斯定理：$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
- 最大似然估计

### 优化
- 凸优化
- 拉格朗日乘子
- 收敛性分析

## 进阶阅读

- "Pattern Recognition and Machine Learning" - Bishop
- "The Elements of Statistical Learning" - Hastie, Tibshirani
- scikit-learn 文档

---

**下一步**：[深度学习基础](../deep-learning-basics/README_CN.md)
