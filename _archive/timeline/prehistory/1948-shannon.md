# 1948 · Shannon 信息论

> 在深度学习之前奠定「信息可量化」的数学语言。

## 为什么放在前史区

它不是被前一代深度学习方法逼出来的突破，而是奠基石之一 —— 交叉熵、互信息、KL 散度都源自这里。
深度学习的损失函数、表示学习的目标、变分推断的下界，全部建立在 Shannon 给出的这套语言之上。

## 它做了什么

- **信息熵 H(X)** —— 用不确定性度量信息量，给"压缩极限"一个数学定义
- **互信息 I(X; Y)** —— 量化两个变量之间共享的信息
- **信道容量 C** —— 噪声信道上可靠传输的最大速率

## 引向哪里

- 损失函数：交叉熵直接来自信息熵；KL 散度是相对熵的别名
- 表示学习：互信息最大化（InfoMax）是自监督学习的核心思路之一
- 生成模型：VAE 的 ELBO 本质是「最大化数据似然，最小化后验与先验的 KL」

## 继续学

- [foundations/math/probability-information-theory/](../../foundations/math/probability-information-theory/) —— 信息论的完整工具箱
- [foundations/deep-learning/loss-functions/](../../foundations/deep-learning/loss-functions/) —— 交叉熵在分类任务中的角色
