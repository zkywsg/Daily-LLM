# 超参数搜索与AutoML

[English](README.md) | [中文](README_CN.md)

## 目录

1. [背景](#1-背景)
2. [核心概念](#2-核心概念)
3. [数学原理](#3-数学原理)
4. [代码实现](#4-代码实现)
5. [实验对比](#5-实验对比)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景

### 1.1 超参数的重要性

超参数显著影响模型性能：
- 学习率: 过大不收敛，过小收敛慢
- 批量大小: 影响泛化和训练速度
- 模型结构: 决定表达能力

### 1.2 手工调参问题

- 耗时: 需要多次实验
- 经验依赖: 需要领域知识
- 局部最优: 容易陷入局部最优

---

## 2. 核心概念

### 2.1 搜索策略

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **网格搜索** | 遍历所有组合 | 全面 | 维度灾难 |
| **随机搜索** | 随机采样 | 高效 | 可能错过最优 |
| **贝叶斯优化** | 基于历史建模 | 高效精准 | 复杂 |
| **早停策略** | 提前终止差实验 | 节省资源 | 可能误判 |

### 2.2 关键超参数

- **学习率**: 最重要
- **批量大小**: 影响稳定性和速度
- **正则化**: dropout, weight decay
- **网络结构**: 层数、宽度

---

## 3. 数学原理

### 3.1 目标

$$
x^* = \arg\max_{x \in \mathcal{X}} f(x)
$$

其中:
- $x$: 超参数配置
- $f(x)$: 验证集性能

### 3.2 高斯过程

用高斯过程建模 $f$:

$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$

采集函数 (期望提升):

$$
\text{EI}(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)]
$$

---

## 4. 代码实现

### 4.1 Optuna示例

```python
import optuna
from transformers import TrainingArguments, Trainer

def objective(trial):
    """目标函数"""
    # 定义搜索空间
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 1, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    # 创建训练参数
    args = TrainingArguments(
        output_dir="./tune",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch"
    )

    # 训练
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=eval_data)
    trainer.train()

    # 返回验证指标 (Optuna默认最小化，所以取负值)
    eval_result = trainer.evaluate()
    return -eval_result["eval_loss"]

# 创建study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# 最佳参数
print(f"Best trial: {study.best_trial.value}")
print(f"Best params: {study.best_trial.params}")
```

### 4.2 Ray Tune示例

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_func(config):
    """训练函数"""
    # 使用config中的超参数训练
    args = TrainingArguments(
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"]
    )

    trainer = Trainer(model=model, args=args, ...)
    trainer.train()

    # 报告结果
    eval_result = trainer.evaluate()
    tune.report(loss=eval_result["eval_loss"])

# 搜索空间
search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([4, 8, 16]),
    "epochs": tune.choice([1, 2, 3])
}

# 早停调度器
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1
)

# 运行搜索
analysis = tune.run(
    train_func,
    config=search_space,
    num_samples=20,
    scheduler=scheduler
)

# 最佳配置
best_config = analysis.get_best_config(metric="loss", mode="min")
print(best_config)
```

---

## 5. 实验对比

### 5.1 效率对比

| 策略 | 试验次数 | 找到最优所需 | 资源消耗 |
|------|---------|-------------|---------|
| **网格搜索 (100点)** | 100 | 100 | 高 |
| **随机搜索** | 50 | ~30 | 中 |
| **贝叶斯优化** | 30 | ~20 | 低 |
| **贝叶斯优化+早停** | 20 | ~15 | 最低 |

### 5.2 效果对比

假设真实最优Loss=1.5:

| 策略 | 最佳Loss | 差距 |
|------|---------|------|
| 手工调参 | 1.8 | +20% |
| 网格搜索 | 1.6 | +6.7% |
| 随机搜索 | 1.55 | +3.3% |
| 贝叶斯优化 | 1.52 | +1.3% |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **从粗到细**: 先大范围搜索，再精细调整
2. **对数尺度**: 学习率等用对数尺度
3. **早停**: 差实验提前终止
4. **随机种子**: 固定种子确保可复现
5. **并行**: 利用多GPU并行搜索

### 6.2 常见陷阱

1. **过拟合验证集**: 多次搜索导致过拟合
2. **忽视计算成本**: 搜索比训练还贵
3. **搜索空间不当**: 范围过大或过小
4. **不考虑交互**: 超参数间有交互

---

## 7. 总结

超参数搜索是提升模型性能的重要手段：

1. **策略选择**: 贝叶斯优化 > 随机搜索 > 网格搜索
2. **关键超参**: 学习率最重要
3. **早停**: 节省计算资源
4. **工具**: Optuna、Ray Tune

**推荐流程**:
1. 大范围随机搜索 (10-20次)
2. 精细贝叶斯优化 (20-30次)
3. 最佳参数训练最终模型
