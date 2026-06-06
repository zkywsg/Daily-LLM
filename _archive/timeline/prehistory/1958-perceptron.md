# 1958 · Perceptron 感知机

> 神经网络的第一形态。

## 为什么放在前史区

Rosenblatt 提出的感知机是后来所有神经元结构的祖先 —— 现代 Transformer 里每一个 Linear 层
本质上都是「加权求和 + 非线性」的扩展。它不是「被前代逼出来」，而是开启神经网络这条路线本身。

## 它做了什么

- 一个神经元 = `输出 = step(w · x + b)`
- 给出了基于梯度下降的「感知机学习规则」
- 证明在线性可分情形下能找到正确分类的权重

## 为什么后来停了 20 年

Minsky & Papert 1969 的《Perceptrons》指出单层感知机无法解决 XOR 等线性不可分问题。
这本书的影响（部分被高估）让神经网络研究在 1970s-1980s 进入"AI 冬天"，直到 1986 反向传播
让多层网络变得可训练才回暖。

## 引向哪里

- 多层感知机（MLP）→ CNN → Transformer，结构在演化，但「加权求和 + 非线性 + 学习权重」的内核没变
- 现代深度学习的"激活函数"（ReLU/GELU/...）就是 step 函数的可微替代

## 继续学

- [foundations/deep-learning/deep-learning-basics/](../../foundations/deep-learning/deep-learning-basics/) —— 神经元到 MLP
- [foundations/deep-learning/activation-functions/](../../foundations/deep-learning/activation-functions/) —— 非线性的演化史
