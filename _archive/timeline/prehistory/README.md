# 深度学习前史

> 2012 之前，重要的不是年份，是基石。

主时间线从 2012 AlexNet 起，因为深度学习正式的「问题 → 突破 → 新问题」叙事是从那一年开始的。
但在那之前，有四个里程碑构成了后来一切的基础。它们**不是**「被前一代逼出来的突破」，而是
后来一切深度学习的奠基石——所以独立放在前史区，明确标注「非主线、是基石」。

## 四个里程碑

| 年份 | 里程碑 | 为什么重要 | 继续学 |
|---|---|---|---|
| 1948 | [Shannon 信息论](./1948-shannon.md) | 信息可量化的数学语言，交叉熵/互信息/KL 散度都源自这里 | [概率与信息论](../../foundations/math/probability-information-theory/) |
| 1958 | [Perceptron 感知机](./1958-perceptron.md) | 神经网络的第一形态，确立"加权求和 + 非线性"最小单元 | [深度学习基础](../../foundations/deep-learning/deep-learning-basics/) |
| 1986 | [Backpropagation 反向传播](./1986-backprop.md) | 让多层网络真正可训练的链式求导算法 | [反向传播](../../foundations/deep-learning/backpropagation/) |
| 1997 | [LSTM 长短期记忆](./1997-lstm.md) | 门控记忆机制，Transformer 之前的最佳序列模型 | [编码器-解码器](../../foundations/structures/encoder-decoder/) |

## 如何阅读

- 不必按年份顺序读 —— 每篇都自成一体，挑你感兴趣的进
- 想系统看「这些数学最后变成了什么深度学习概念」，跟着每篇底部的"继续学"链接进入 `foundations/`
- 想看正式时间线（2012 起），回到 [../README.md](../README.md)
