[English](README_EN.md) | [中文](README.md)

# Why Can't Deep Learning Do Without Matrix Multiplication? — Linear Algebra Basics

## Where This Problem Comes From

> In the previous chapter we built a neural network: input x goes through weight matrix W, adds bias b, and then passes through an activation function. This `y = Wx + b` looks simple, but what is it actually doing? Why use a matrix instead of a scalar?
> The answer is: matrix multiplication is the mathematical language of "batch linear transformation." Every layer of a neural network is essentially applying a linear transformation + nonlinear warping to the data. Without understanding what matrices are doing, you can only treat neural networks as black boxes.

## Learning Objectives

After completing this chapter, you should be able to answer:

1. What is the geometric meaning of matrix multiplication?
2. Why does the attention mechanism need a transpose (QK^T)?
3. What is SVD decomposition useful for in deep learning?

---

## 1. Intuition

A matrix is a "transformation rule."

Imagine a set of points on a plane. Matrix multiplication can **rotate, scale, stretch, or project** them. A 2×2 matrix acting on a 2D vector is just like a function acting on a number — input a vector, output another vector.

In deep learning, every layer $y = Wx + b$ uses matrix $W$ to apply a spatial transformation to input $x$, plus bias $b$ to shift the origin. After stacking many layers, data that was originally mixed together gets "twisted" into a separable state.

> Key takeaway: a matrix is not just an arrangement of numbers; it is a compressed representation of a spatial transformation. Once you understand this, the rules of matrix multiplication are no longer rote memorization.

---

## 2. Mechanics

### 2.1 Vectors and Matrices Basics

**Vector**: an object with both direction and magnitude. In deep learning, a word's embedding or a sample's features are both vectors.

**Matrix multiplication**: row-by-column rule. $(m,n) \times (n,p) \rightarrow (m,p)$ — the inner dimensions must match.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])    # (2, 2)
B = np.array([[5, 6], [7, 8]])    # (2, 2)
C = A @ B                          # (2, 2) matrix multiplication
D = A * B                          # (2, 2) element-wise multiplication (completely different!)

print(f"A @ B = \n{C}")
print(f"A * B = \n{D}")
```

> **Common mistake**: `*` is element-wise multiplication, `@` or `np.matmul` is matrix multiplication. The results are completely different!

**Transpose**: $A^T$ swaps rows and columns. $(AB)^T = B^T A^T$ — the order reverses after transposition.

**Identity matrix** $I$: any matrix multiplied by $I$ stays unchanged (the "1" of matrix multiplication).

### 2.2 Matrix Multiplication = Linear Transformation

Geometric intuition of matrix multiplication:

| 2×2 matrix | Effect |
|------------|--------|
| $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ | Rotate by θ degrees |
| $\begin{pmatrix} 2 & 0 \\ 0 & 0.5 \end{pmatrix}$ | Stretch x by 2×, compress y by half |
| $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ | Project onto the x-axis (loses y information) |

**The essence of every neural network layer**:
$$y = \sigma(Wx + b)$$

- $Wx$: linear transformation (rotation + scaling + projection)
- $+ b$: translation
- $\sigma$: nonlinear warping

A fully connected layer is matrix multiplication. The underlying operation of convolution can also be expressed with matrix operations (im2col unfolds local regions into rows, turning convolution into a single matrix multiplication).

### 2.3 Norms

Norms are measures of "vector size."

| Norm | Formula | Intuition | Application in deep learning |
|------|---------|-----------|-----------------------------|
| L1 | $\|x\|_1 = \sum \|x_i\|$ | Manhattan distance | LASSO regularization (sparsity) |
| L2 | $\|x\|_2 = \sqrt{\sum x_i^2}$ | Euclidean distance | Weight decay, gradient clipping |
| L∞ | $\|x\|_\infty = \max \|x_i\|$ | Maximum absolute value | Clip-by-value gradient clipping |

**Cosine similarity**: measures direction rather than magnitude.

$$\text{cos\_sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$$

Range [-1, 1]: 1 means same direction, 0 means orthogonal, -1 means opposite.

> In contrastive learning (CLIP, SimCLR), cosine similarity is the standard way to measure sample similarity.
> → See [Loss Functions](../loss-functions/README_EN.md)

### 2.4 Eigenvalues and SVD

**Eigenvalue/eigenvector** intuition: when matrix $A$ acts on vector $v$, if the result is just a scaled version of $v$ ($Av = \lambda v$), then $v$ is an eigenvector and $\lambda$ is an eigenvalue — "this direction is magnified or shrunk by how much."

**SVD (Singular Value Decomposition)**: $A = U\Sigma V^T$

Intuition: break any matrix into "rotation → scaling → rotation" three steps.

- $V^T$: first rotate to a special coordinate system
- $\Sigma$: scale along each direction (singular values)
- $U$: then rotate to the target coordinate system

Applications:
- **Dimensionality reduction**: keep only the top k singular values to approximate the original matrix with minimal information loss
- **Recommendation systems**: matrix completion via SVD in collaborative filtering
- **LoRA preview**: LoRA's low-rank decomposition idea is essentially an SVD approximation — learning only the most important directions
  → See [Numerical Precision](../numerical-precision/README.md)

### 2.5 Broadcasting

NumPy/PyTorch broadcasting rules: align dimensions from right to left, trailing dimensions must be the same or 1.

```
(3, 1) + (1, 4) → (3, 4)  ✓ trailing 1 and 4: 1 can broadcast
(3,)   + (4,)   → error   ✗ trailing 3 ≠ 4
(3, 1) + (4,)   → (3, 4)  ✓ (4,) right-aligns to (1, 4), then broadcasts
```

Common scenarios:
- Adding bias to batch data: `(B, D) + (D,)` → `(B, D)`
- Attention mask: `(B, 1, T) + (1, T, T)` → `(B, T, T)`

```python
import numpy as np

# batch bias addition
x = np.random.randn(32, 64)    # batch=32, dim=64
b = np.random.randn(64)        # bias
result = x + b                  # (32, 64) + (64,) → (32, 64)

# attention mask
mask = np.tril(np.ones((8, 8)))  # (8, 8) lower triangle
batch_mask = mask[np.newaxis, :, :]  # (1, 8, 8)
```

---

## 3. Progressive Implementation

**Step 1 · Pure NumPy handwritten matrix multiplication**

```python
import numpy as np
import time

def matmul_naive(A, B):
    """Triple-loop matrix multiplication, for understanding the algorithm only"""
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

print(f"Triple loop: {t1:.3f}s")
print(f"NumPy @:     {t2:.6f}s")
print(f"Results match: {np.allclose(C1, C2)}")
# NumPy is thousands to tens of thousands of times faster (vectorization + BLAS optimization)
```

**Step 2 · Linear transformation visualization**

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate points on the unit circle
theta = np.linspace(0, 2 * np.pi, 100)
points = np.array([np.cos(theta), np.sin(theta)])  # (2, 100)

# Define transformation matrices
transforms = {
    "Rotate 45°": np.array([[0.707, -0.707], [0.707, 0.707]]),
    "Scale":     np.array([[2.0, 0], [0, 0.5]]),
    "Shear":     np.array([[1, 0.5], [0, 1]]),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].plot(points[0], points[1], 'b-')
axes[0].set_title("Original unit circle")
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

**Step 3 · SVD dimensionality reduction example**

```python
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image (or use random matrix as substitute)
img = np.random.rand(64, 64)  # replace with real image via Image.open().convert('L')
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# Reconstruct with top k singular values
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, k in enumerate([1, 5, 10, 20, 50]):
    reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    axes[i].imshow(reconstructed, cmap='gray')
    axes[i].set_title(f'k={k} (retaining {S[:k].sum()/S.sum():.0%} energy)')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("svd_reconstruction.png", dpi=150)
plt.show()
# Larger k retains more information. LoRA uses a similar idea: learn only the most important directions.
```

---

## 4. Engineering Pitfalls (Sorted by Severity)

1. **Confusing matrix multiplication with element-wise multiplication** (most common)
   Symptom: mixing up `A * B` (element-wise) and `A @ B` (matrix multiplication), causing dimension mismatches or wrong results.
   Fix: `*` is element-wise, `@` or `torch.matmul` is matrix multiplication — never confuse them.

2. **Dimension mismatch**
   Symptom: multiplying `(B, D)` and `(D, K)` in the wrong order.
   Fix: remember "inner dimensions must match," `(m,n) @ (n,p)` → `(m,p)`.

3. **Missing transpose causes shape errors**
   Symptom: writing QK instead of QK^T in attention computation, causing a dimension error.
   Fix: similarity in attention is always `Q @ K.T`; the transpose cannot be omitted.
   → See [Attention Mechanisms](../attention-primer/README.md)

4. **Misunderstanding broadcasting**
   Symptom: thinking `(3,) + (4,)` will broadcast, but it errors.
   Fix: broadcasting aligns from right to left; trailing dimensions must be the same or 1.

---

## Evolution Notes

> **The legacy of linear algebra**: almost every operation in deep learning can be reduced to matrix multiplication — fully connected layers are matrix multiplication, convolutions can be unfolded into matrix multiplication, and attention is also matrix multiplication (QK^T). The reason GPUs are so well-suited to deep learning is precisely that they can perform matrix operations extremely fast.
>
> The idea of SVD and low-rank decomposition directly inspired LoRA: instead of fine-tuning all parameters, learn changes only in a low-rank subspace, reducing parameter count by 1000×.
>
> **New problems left behind**: we now have the ability to "transform inputs into scores" (linear transformation), but those scores are not yet probabilities — how do we turn arbitrary real numbers into a valid probability distribution? This leads to Softmax.

→ Next: [Softmax & Probability Distributions — Why Can't We Just Compare Raw Scores in Multi-Class Classification?](../softmax/README_EN.md)

---

**Previous**: [Deep Learning Basics](../deep-learning-basics/README_EN.md) | **Next**: [Softmax & Probability Distributions](../softmax/README_EN.md)
