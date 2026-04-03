# 为什么深度学习离不开矩阵乘法？—— 线性代数基础

## 这个问题从哪来

> 上一章我们构建了神经网络：输入 x 经过权重矩阵 W 变换，加上偏置 b，再经过激活函数。这个 `y = Wx + b` 看起来简单，但它到底在做什么？为什么用矩阵而不是用标量？
> 答案是：矩阵乘法是"批量线性变换"的数学语言。神经网络的每一层，本质上都在对数据做一次线性变换 + 非线性扭曲。不理解矩阵在做什么，就只能把神经网络当黑盒用。

## 学习目标

完成本章后，你应能回答：

1. 矩阵乘法的几何意义是什么？
2. 为什么注意力机制里需要转置（QK^T）？
3. SVD 分解在深度学习中有什么用？

---

## 1. 直觉

矩阵就是一个"变换规则"。

想象平面上有一组点，矩阵乘法可以把它们**旋转、缩放、拉伸或投影**。2×2 矩阵作用在 2 维向量上，就像一个函数作用在数字上——输入一个向量，输出另一个向量。

深度学习中的每一层 $y = Wx + b$，就是用矩阵 $W$ 对输入 $x$ 做一次空间变换，再加上偏置 $b$ 平移原点。多层叠加后，原本混在一起的数据就被"拧"到了可分的状态。

> 你要记住：矩阵不是一堆数字的排列，它是空间变换的压缩表示。理解了这一点，矩阵乘法的规则就不再是死记硬背。

---

## 2. 机制

### 2.1 向量与矩阵基础

**向量**：有方向和大小的对象。在深度学习中，一个词的 embedding、一个样本的特征，都是向量。

**矩阵乘法**：行看列规则。$(m,n) \times (n,p) \rightarrow (m,p)$——中间维度必须匹配。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])    # (2, 2)
B = np.array([[5, 6], [7, 8]])    # (2, 2)
C = A @ B                          # (2, 2) 矩阵乘法
D = A * B                          # (2, 2) 逐元素乘法（完全不同！）

print(f"A @ B = \n{C}")
print(f"A * B = \n{D}")
```

> **常见错误**：`*` 是逐元素乘法，`@` 或 `np.matmul` 才是矩阵乘法。两者结果完全不同！

**转置**：$A^T$ 把行列互换。$(AB)^T = B^T A^T$——转置后乘法顺序要反转。

**单位矩阵** $I$：任何矩阵乘以 $I$ 不变（矩阵乘法中的"1"）。

### 2.2 矩阵乘法 = 线性变换

矩阵乘法的几何直觉：

| 2×2 矩阵 | 效果 |
|----------|------|
| $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ | 旋转 θ 度 |
| $\begin{pmatrix} 2 & 0 \\ 0 & 0.5 \end{pmatrix}$ | x 方向拉伸 2 倍，y 方向压缩一半 |
| $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ | 投影到 x 轴（丢失 y 信息） |

**神经网络中每一层的本质**：
$$y = \sigma(Wx + b)$$

- $Wx$：线性变换（旋转 + 缩放 + 投影）
- $+ b$：平移
- $\sigma$：非线性扭曲

全连接层就是矩阵乘法。卷积的底层也可以用矩阵运算实现（im2col 把局部区域展开成行，卷积变成一次矩阵乘法）。

### 2.3 范数

范数是"向量大小"的度量。

| 范数 | 公式 | 直觉 | 深度学习中的应用 |
|------|------|------|----------------|
| L1 | $\|x\|_1 = \sum \|x_i\|$ | 曼哈顿距离 | LASSO 正则化（稀疏性） |
| L2 | $\|x\|_2 = \sqrt{\sum x_i^2}$ | 欧几里得距离 | 权重衰减、梯度裁剪 |
| L∞ | $\|x\|_\infty = \max \|x_i\|$ | 最大绝对值 | 按值梯度裁剪 |

**余弦相似度**：衡量方向而非大小。

$$\text{cos\_sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$$

值域 [-1, 1]：1 表示同向，0 表示正交，-1 表示反向。

> 在对比学习（CLIP、SimCLR）中，余弦相似度是衡量样本相似性的标准方式。
> → 详见 [损失函数](../loss-functions/README.md)

### 2.4 特征值与 SVD

**特征值/特征向量**的直觉：矩阵 $A$ 作用在向量 $v$ 上，如果结果只是 $v$ 的缩放（$Av = \lambda v$），那么 $v$ 就是特征向量，$\lambda$ 是特征值——"这个方向被放大/缩了多少倍"。

**SVD（奇异值分解）**：$A = U\Sigma V^T$

直觉：把任意矩阵拆成"旋转 → 缩放 → 旋转"三步。

- $V^T$：先旋转到特殊坐标系
- $\Sigma$：在各方向上缩放（奇异值）
- $U$：再旋转到目标坐标系

应用：
- **降维**：只保留前 k 个奇异值，近似原矩阵，丢失的信息最少
- **推荐系统**：协同过滤中用 SVD 做矩阵补全
- **LoRA 预告**：LoRA 的低秩分解思想本质上就是 SVD 的近似——只学最重要的方向
  → 详见 [数值精度](../numerical-precision/README.md)

### 2.5 广播机制

NumPy/PyTorch 的广播规则：从右往左对齐维度，尾部维度必须相同或为 1。

```
(3, 1) + (1, 4) → (3, 4)  ✓ 尾部 1 和 4 中，1 可以广播
(3,)   + (4,)   → 报错     ✗ 尾部 3 ≠ 4
(3, 1) + (4,)   → (3, 4)  ✓ (4,) 从右对齐到 (1, 4)，再广播
```

常见场景：
- batch 数据加偏置：`(B, D) + (D,)` → `(B, D)`
- 注意力 mask：`(B, 1, T) + (1, T, T)` → `(B, T, T)`

```python
import numpy as np

# batch 加偏置
x = np.random.randn(32, 64)    # batch=32, dim=64
b = np.random.randn(64)        # 偏置
result = x + b                  # (32, 64) + (64,) → (32, 64)

# 注意力 mask
mask = np.tril(np.ones((8, 8)))  # (8, 8) 下三角
batch_mask = mask[np.newaxis, :, :]  # (1, 8, 8)
```

---

## 3. 渐进式实现

**Step 1 · 纯 NumPy 手写矩阵乘法**

```python
import numpy as np
import time

def matmul_naive(A, B):
    """三重循环矩阵乘法，仅用于理解算法"""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

A = np.random.randn(100, 200)
B = np.random.randn(200, 150)

start = time.time()
C1 = matmul_naive(A, B)
t1 = time.time() - start

start = time.time()
C2 = A @ B
t2 = time.time() - start

print(f"三重循环: {t1:.3f}s")
print(f"NumPy @:  {t2:.6f}s")
print(f"结果一致: {np.allclose(C1, C2)}")
# NumPy 比三重循环快几千到几万倍（向量化 + BLAS 优化）
```

**Step 2 · 线性变换可视化**

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成单位圆上的一组点
theta = np.linspace(0, 2 * np.pi, 100)
points = np.array([np.cos(theta), np.sin(theta)])  # (2, 100)

# 定义变换矩阵
transforms = {
    "旋转 45°": np.array([[0.707, -0.707], [0.707, 0.707]]),
    "缩放":     np.array([[2.0, 0], [0, 0.5]]),
    "剪切":     np.array([[1, 0.5], [0, 1]]),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].plot(points[0], points[1], 'b-')
axes[0].set_title("原始单位圆")
axes[0].set_aspect('equal')
axes[0].grid(True)

for ax, (name, T) in zip(axes[1:], transforms.items()):
    transformed = T @ points
    ax.plot(transformed[0], transformed[1], 'r-')
    ax.set_title(name)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

plt.tight_layout()
plt.savefig("linear_transforms.png", dpi=150)
plt.show()
```

**Step 3 · SVD 降维示例**

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载灰度图（或用随机矩阵代替）
img = np.random.rand(64, 64)  # 替换为真实图片时用 Image.open().convert('L')
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# 用前 k 个奇异值重建
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, k in enumerate([1, 5, 10, 20, 50]):
    reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    axes[i].imshow(reconstructed, cmap='gray')
    axes[i].set_title(f'k={k} (保留 {S[:k].sum()/S.sum():.0%} 能量)')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("svd_reconstruction.png", dpi=150)
plt.show()
# k 越大，保留信息越多。LoRA 的思路类似：只学最重要的方向。
```

---

## 4. 工程陷阱（按严重度排序）

1. **矩阵乘法 vs 逐元素乘法混淆**（最常见）
   现象：`A * B`（逐元素）和 `A @ B`（矩阵乘法）搞混，维度对不上或结果错误。
   处置：`*` 是逐元素，`@` 或 `torch.matmul` 是矩阵乘法，永远不要混淆。

2. **维度不匹配**
   现象：`(B, D)` 和 `(D, K)` 相乘搞反顺序。
   处置：记住"中间维度必须相同"，`(m,n) @ (n,p)` → `(m,p)`。

3. **转置忘记导致形状错误**
   现象：注意力计算中 QK^T 写成 QK，维度报错。
   处置：注意力里相似度计算一定是 `Q @ K.T`，转置不可省略。
   → 详见 [注意力机制](../attention-primer/README.md)

4. **广播机制理解错误**
   现象：以为 `(3,) + (4,)` 能广播，实际报错。
   处置：广播从右往左对齐，尾部维度必须相同或为 1。

---

## 演进笔记

> **线性代数的遗产**：深度学习的几乎所有操作都可以归结为矩阵乘法——全连接层是矩阵乘法，卷积可以展开成矩阵乘法，注意力也是矩阵乘法（QK^T）。GPU 之所以适合深度学习，正是因为它能极快地做矩阵运算。
>
> SVD 和低秩分解的思想后来直接启发了 LoRA：不用微调全部参数，只在低秩子空间中学习变化量，参数量减少 1000 倍。
>
> **留下的新问题**：我们已经有了"把输入变换成分数"的能力（线性变换），但这些分数还不是概率——怎么把任意实数变成合法的概率分布？这引出了 Softmax。

→ 下一章：[Softmax 与概率分布 — 为什么多分类不能直接比大小？](../softmax/README.md)

---

**上一章**：[深度学习基础](../deep-learning-basics/README.md) | **下一章**：[Softmax 与概率分布](../softmax/README.md)
