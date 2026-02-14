# 实验与可重复性（Experimentation and Reproducibility）

本章面向 LLM 系统研发的实验与可重复性（Reproducibility）实践，覆盖从配置管理（Configuration Management）到版本控制（Version Control）、实验追踪（Experiment Tracking）、随机性管理（Random Seed Control）、环境隔离（Environment Isolation）与实验对比分析（Experiment Comparison）。目标是让同一份代码、同一份数据、同一套环境在多次运行、多人协作、不同硬件上都能得到一致或可解释的结果。

---

## 1. 背景：为什么可重复性重要

在 LLM 系统中，训练与评估涉及海量数据、复杂依赖、随机性与长链路流水线。可重复性（Reproducibility）不仅是科研诚信，更是工程可信度与迭代效率的基石：

- **研发可信**：可重复结果让结论可验证，避免“只在某台机器上有效”。
- **调试效率**：可重复运行是定位回归问题（Regression）与性能退化的前提。
- **跨团队协作**：一致的环境与配置让多人协作可对齐与可复现。
- **监管与审计**：模型发布需可追溯训练数据、参数、代码版本与产出模型。

LLM 训练和评测的不确定性来自多个层面：数据采样、初始化权重、并行计算的非确定性、硬件差异、依赖库版本差异。工程上需要把这些不确定性“可控化”，做到可重复或可解释。

在企业级 LLM 系统中，可重复性还直接影响：

- **成本控制**：无法复现会导致重复训练，浪费算力预算。
- **模型治理（Model Governance）**：需要追溯模型从训练到部署的完整链路。
- **上线风险**：回归测试不稳定会增加灰度发布风险。
- **知识沉淀**：可追踪实验让“经验”转化为可复用资产。

### 1.1 可重复性的工程成本

可重复性并非“免费”，需要投入工程成本，但长期收益显著：

- 初期成本：建立配置管理与追踪系统、改造训练脚本
- 持续成本：维护数据与模型版本、更新文档与清单
- 长期收益：调试效率提升、研发协作一致、回归风险降低

实践中应将可重复性纳入研发流程，而非作为“额外工作”。

### 1.2 可重复性与可解释性

当实验结果无法完全一致时，至少需要做到“可解释”（Explainability）：

- 差异来源清晰（随机性、数据漂移或环境变化）
- 变化幅度可量化（统计区间或阈值）
- 可通过对比分析定位影响因素

这意味着团队不仅要追求“重复”，也要追求“解释”。

---

## 2. 核心概念：实验追踪、版本控制与可追溯性

### 2.1 实验追踪（Experiment Tracking）

实验追踪记录每次运行的配置、代码版本、数据版本、指标与产出，典型工具包括 **MLflow** 与 **Weights & Biases (WandB)**。完整追踪应覆盖：

- 配置（Config）：超参数、路径、模型结构、训练策略
- 代码（Code）：Git commit、分支、差异
- 数据（Data）：数据版本、数据过滤逻辑、统计摘要
- 模型（Model）：权重、训练曲线、评估指标
- 环境（Env）：Python 版本、依赖包版本、硬件信息

### 2.2 配置管理（Configuration Management）

配置管理将实验参数与代码逻辑解耦，推荐使用 **YAML** 或 **Hydra**。配置的原则：

- 单一事实来源（Single Source of Truth）
- 结构化与可继承（Structured + Inheritance）
- 可追踪与可序列化（Trackable + Serializable）

配置管理还应考虑：

- **参数空间定义**：明确哪些参数可调，哪些必须固定
- **配置校验**：防止非法组合导致不可复现
- **配置归档**：将每次运行的配置写入 run 目录

### 2.3 版本控制（Version Control）

版本控制不仅包括代码（Code），还包括数据（Data）与模型（Model）。常用策略：

- 代码版本：Git + 语义化标签（Tag）
- 数据版本：DVC、Git LFS、数据快照（Snapshot）
- 模型版本：MLflow Model Registry / WandB Artifacts

### 2.4 可追溯性（Traceability）

可追溯性要求从实验结果能回溯到所有关键输入：数据、配置、环境、代码。理想状态下，任何一次实验结果都可以通过一个“实验 ID（Run ID）”完全重建。

### 2.5 数据谱系（Data Lineage）

数据谱系（Data Lineage）描述数据从原始来源到训练输入的加工路径。LLM 数据通常经历清洗、去重、过滤、抽样、标注等多个步骤，若不记录每个步骤的参数与版本，将无法复现数据集。

建议记录：

- 原始数据来源（URL/版本/时间戳）
- 清洗脚本与参数
- 过滤规则与阈值
- 采样方法与随机种子
- 产出数据的哈希摘要

### 2.6 模型谱系（Model Lineage）

模型谱系要求记录从基础模型到微调模型的所有依赖，包括：

- 基础模型版本（如 `llama2-7b`）
- 微调数据版本
- 微调配置与训练日志
- 量化/蒸馏/剪枝过程的细节

### 2.7 配置版本化策略

配置本身也需要版本控制，建议遵循：

- 所有配置文件纳入 Git
- 每次实验运行时将配置快照复制到 run 目录
- 关键配置（训练策略、数据过滤规则）做语义化 tag

推荐建立 `configs/` 目录的变更日志（Changelog），记录重大参数变更原因。

### 2.8 实验元数据标准

建立统一元数据规范（Metadata Schema）便于自动化分析，常见字段：

- `run_id`、`experiment_name`
- `git_commit`、`branch`
- `data_version`、`data_hash`
- `model_name`、`model_version`
- `seed`、`hardware`
- `metrics`、`artifacts`

可以将元数据以 JSON 或 YAML 保存，并作为实验追踪系统的附加信息。

### 2.9 实验登记表（Experiment Registry）

实验登记表是团队协作的重要组件，建议包含：

- 实验名称与目标
- 负责人
- 数据版本与模型版本
- 预期指标与评测方案
- 风险点与依赖

登记表可放在共享文档或数据库中，并与实验追踪系统连接。

### 2.10 资产管理（Artifacts Management）

LLM 项目会产生大量工件（Artifacts），包括：

- 训练日志
- 模型权重
- 中间 checkpoint
- 可视化图表

这些工件应统一存储与命名，避免散落在个人目录中。

### 2.11 命名规范与实验编码

一致的命名规范是可追溯性的基础。建议为实验、数据、模型定义统一编码规则：

- 实验：`exp_<日期>_<目标>_<版本>`
- 数据：`data_<来源>_<版本>`
- 模型：`model_<架构>_<版本>`

例如：`exp_2026-02-01_repro_v1`，`data_corpus_v1.2.0`，`model_llama2_v1`。

### 2.12 实验文档化

每次重要实验应生成简要文档，包含：

- 目标与假设
- 方法与配置
- 数据与模型版本
- 结论与后续计划

文档可以与 run 绑定，作为可重复性记录的一部分。

### 2.13 配置管理反模式

常见反模式会显著降低可重复性：

- 配置散落在代码中，无法集中管理
- 运行时修改配置但不记录
- 配置文件存在“隐式默认值”，导致不同人运行结果不同

避免反模式的核心是将配置显式化并记录。

### 2.14 元数据字段字典

| 字段 | 说明 |
| --- | --- |
| run_id | 实验唯一标识 |
| experiment | 实验分组 |
| git_commit | 代码版本 |
| data_version | 数据版本 |
| model_version | 模型版本 |
| seed | 随机种子 |
| hardware | 硬件信息 |
| metrics | 关键指标 |

### 2.15 审计与合规需求

在受监管场景下，需满足审计与合规要求：

- 可追溯训练数据来源
- 可追溯模型训练配置
- 可追溯评测结果与报告

实验追踪系统应支持导出审计报告。

---

## 3. 数学原理：随机性控制与统计检验

### 3.1 随机性与可重复性

随机性（Randomness）来自初始化、数据采样、Dropout、并行执行等。可重复性需要在可控范围内固定随机性。设随机变量 $X$ 表示实验结果指标（如准确率），其期望与方差为：

\[
\mathbb{E}[X] = \mu, \quad \mathrm{Var}(X) = \sigma^2
\]

固定随机种子（Seed）是控制随机变量生成过程的必要条件，但**不能保证绝对一致**，尤其在 GPU 并行与非确定性算子存在时。

在并行训练中，不同线程调度顺序可能导致浮点累积误差不同。即使随机种子相同，浮点运算顺序差异仍可能导致结果漂移。

### 3.2 统计稳定性与重复实验

单次实验不足以判断模型优劣，需要多次重复实验并进行统计检验。常见做法：

- 多次运行，取均值与方差
- 使用 t 检验（t-test）或 bootstrap 估计置信区间

两组实验结果 $X$ 与 $Y$ 的均值差检验：

\[
t = \frac{\bar{X} - \bar{Y}}{\sqrt{\frac{s_X^2}{n_X} + \frac{s_Y^2}{n_Y}}}
\]

当 $p$ 值小于显著性水平（如 0.05）时，认为差异显著。

**Bootstrap 置信区间示意：**

\[
\hat{\theta}^* = \frac{1}{B} \sum_{b=1}^{B} \theta_b, \quad CI_{95\%} = [\theta_{0.025}, \theta_{0.975}]
\]

### 3.3 重现性（Reproducibility）与再现性（Replicability）

- **Reproducibility**：在相同数据与代码条件下重现结果
- **Replicability**：在不同实现或数据条件下再现结论

工程目标至少保证 Reproducibility，科研目标同时追求 Replicability。

### 3.4 随机性传播与控制边界

在复杂系统中，随机性传播（Randomness Propagation）可能影响下游指标。一个常见策略是限定“随机边界”：

- 数据采样阶段使用固定种子
- 训练阶段使用确定性算子（deterministic ops）
- 评估阶段固定评测集与推理参数

若无法完全确定性，可通过多次运行统计稳定性。

### 3.5 数值精度与硬件差异

数值精度（Numerical Precision）是可重复性的重要来源之一。不同硬件或不同混合精度策略可能导致细微差异：

- FP32 与 FP16 的累积误差不同
- BF16 在部分硬件上有不同的舍入策略
- 不同 GPU 架构（如 V100 vs A100）会带来数值差异

对于敏感指标，需要明确：

- 训练精度（FP32/FP16/BF16）
- 是否使用自动混合精度（AMP）
- 梯度缩放策略

建议在实验追踪中记录精度相关配置，并在对比实验中确保一致。

### 3.6 置信区间与统计功效

在多次实验中，除了均值，还应报告置信区间（Confidence Interval）：

\[
CI = \bar{X} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}
\]

其中 $z_{\alpha/2}$ 是标准正态分布分位数。统计功效（Power）用于衡量检测差异的能力，样本量不足会导致结论不可靠。对于 LLM 评测，应尽可能提高样本量或使用 bootstrap。

### 3.7 方差来源拆解

方差来自多个来源：

- 数据采样方差
- 初始化方差
- 并行计算方差

可以通过控制变量实验来拆解方差来源。例如固定模型初始化，仅改变数据采样，评估采样方差贡献。

---

## 4. 代码实现：追踪工具、配置管理与种子控制

### 4.1 配置管理（YAML + Hydra）

**YAML 基本配置示例：**

```yaml
# configs/train.yaml
project: llm-exp
seed: 42
data:
  path: data/corpus_v1
  version: v1.2.0
model:
  name: llama2
  hidden_size: 4096
training:
  batch_size: 16
  lr: 3e-4
  epochs: 3
tracking:
  tool: mlflow
  experiment: exp_repro
```

**Hydra 组合与覆盖示例：**

```yaml
# configs/config.yaml
defaults:
  - model: llama2
  - training: base
  - data: corpus_v1
  - _self_

seed: 42
project: llm-exp
```

```yaml
# configs/model/llama2.yaml
name: llama2
hidden_size: 4096
```

```yaml
# configs/training/base.yaml
batch_size: 16
lr: 3e-4
epochs: 3
```

```python
# train.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 中文注释：打印配置，确保可追踪
    print(cfg)
    # 中文注释：这里接入训练逻辑
    pass

if __name__ == "__main__":
    main()
```

**命令行覆盖：**

```bash
python train.py training.lr=1e-4 data.version=v1.2.1 seed=123
```

Hydra 会自动输出运行目录与配置快照，极大增强可追踪性。

**Hydra 多运行（multirun）示例：**

```bash
python train.py -m training.lr=1e-4,3e-4,5e-4 training.batch_size=8,16
```

### 4.1.1 配置校验与结构化配置

使用 dataclass 或 pydantic 定义结构化配置，保证类型一致性：

```python
# config_schema.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 16
    lr: float = 3e-4
    epochs: int = 3

@dataclass
class ModelConfig:
    name: str = "llama2"
    hidden_size: int = 4096
```

### 4.1.2 配置管理策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 单文件 YAML | 简单直观 | 扩展性差 | 小型实验 |
| 分层 YAML | 组合灵活 | 结构复杂 | 中型项目 |
| Hydra + schema | 强校验、可组合 | 学习成本 | 大型系统 |

对于 LLM 项目，推荐采用 Hydra + 结构化配置，确保参数一致性。

### 4.2 实验追踪（MLflow / WandB）

### 4.2 实验追踪（MLflow / WandB）

**MLflow 示例：**

```python
# mlflow_train.py
import mlflow
import mlflow.sklearn

def train_model(params):
    # 中文注释：这里用简单模型示意
    model = {"dummy": True}
    metrics = {"acc": 0.85, "loss": 0.35}
    return model, metrics

def main():
    mlflow.set_experiment("exp_repro")
    with mlflow.start_run():
        params = {"lr": 3e-4, "batch_size": 16}
        mlflow.log_params(params)
        model, metrics = train_model(params)
        mlflow.log_metrics(metrics)
        # 中文注释：记录模型与环境
        mlflow.set_tag("git_commit", "<commit_hash>")
        mlflow.set_tag("data_version", "v1.2.0")
        mlflow.log_dict({"model": model}, "model.json")

if __name__ == "__main__":
    main()
```

**MLflow 记录工件（Artifacts）示例：**

```python
# mlflow_artifact.py
import mlflow

def main():
    mlflow.set_experiment("exp_repro")
    with mlflow.start_run():
        # 中文注释：保存配置文件
        mlflow.log_artifact("configs/train.yaml")
        # 中文注释：保存训练日志
        mlflow.log_artifact("logs/train.log")

if __name__ == "__main__":
    main()
```

**WandB 示例：**

```python
# wandb_train.py
import wandb

def train_model(config):
    # 中文注释：模拟训练过程
    wandb.log({"loss": 0.35, "acc": 0.85})

def main():
    wandb.init(project="llm-exp", config={"lr": 3e-4, "batch_size": 16})
    train_model(wandb.config)
    wandb.finish()

if __name__ == "__main__":
    main()
```

**WandB Sweep 示例（参数搜索）：**

```yaml
# sweep.yaml
method: grid
parameters:
  lr:
    values: [1e-4, 3e-4, 5e-4]
  batch_size:
    values: [8, 16]
```

```bash
wandb sweep sweep.yaml
```

**关键点：**

- 记录 Git commit、数据版本、配置快照
- 输出模型权重与评估指标
- 支持对比多个 runs（Run Comparison）

**实验追踪字段建议：**

- `run_id`：唯一标识
- `experiment`：实验分组
- `config_hash`：配置摘要
- `data_hash`：数据摘要
- `model_hash`：模型摘要
- `seed`：随机种子
- `hardware`：GPU/CPU 型号

### 4.2.1 实验追踪与配置绑定

在实验启动时，自动保存配置并绑定 run：

```python
# track_config.py
import json
import mlflow

def main():
    cfg = {"lr": 3e-4, "batch_size": 16}
    mlflow.set_experiment("exp_repro")
    with mlflow.start_run():
        mlflow.log_params(cfg)
        mlflow.log_dict(cfg, "config.json")

if __name__ == "__main__":
    main()
```

### 4.3 版本控制（代码、数据、模型）

#### 代码版本（Git）

- commit 固定语义
- tag 绑定发布版本
- 记录实验 run 与 commit 关联

**Git LFS（大文件）示例：**

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "track model weights"
```

#### 数据版本（DVC / Git LFS）

**DVC 示例：**

```bash
dvc init
dvc add data/corpus_v1
git add data/corpus_v1.dvc .gitignore
git commit -m "track corpus v1"
```

**数据清单（Manifest）示例：**

```python
# build_manifest.py
import hashlib
import json
from pathlib import Path

def sha256_file(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = Path("data/corpus_v1")
    manifest = {str(p): sha256_file(p) for p in root.rglob("*") if p.is_file()}
    with open("data/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
```

**数据版本标记：**

```bash
git tag -a data-v1.2.0 -m "dataset v1.2.0"
```

**DVC 远程存储示例：**

```bash
dvc remote add -d storage s3://my-bucket/dvc
dvc push
```

### 4.3.1 版本控制策略对比

| 对象 | 推荐工具 | 说明 |
| --- | --- | --- |
| 代码 | Git | 版本管理核心 |
| 数据 | DVC / LakeFS | 数据快照、版本回滚 |
| 模型 | MLflow / WandB Artifacts | 模型注册与回滚 |

不同对象的版本控制应统一命名和标记，避免信息孤岛。

#### 模型版本（Model Registry）

在 MLflow Model Registry 或 WandB Artifacts 中注册模型：

- `model:v1` 绑定训练参数、指标、数据版本
- 支持回滚与灰度发布

### 4.4 随机种子管理（Seed Control）

**Python / NumPy / PyTorch / TensorFlow / JAX：**

```python
# seed.py
import os
import random
import numpy as np

def set_seed(seed: int):
    # 中文注释：Python 内置随机
    random.seed(seed)
    # 中文注释：NumPy 随机
    np.random.seed(seed)
    # 中文注释：环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)

try:
    import torch
    def set_torch_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except Exception:
    def set_torch_seed(seed: int):
        pass

try:
    import tensorflow as tf
    def set_tf_seed(seed: int):
        tf.random.set_seed(seed)
except Exception:
    def set_tf_seed(seed: int):
        pass

try:
    import jax
    def set_jax_seed(seed: int):
        # 中文注释：JAX 使用 PRNGKey
        return jax.random.PRNGKey(seed)
except Exception:
    def set_jax_seed(seed: int):
        return None

def main():
    seed = 42
    set_seed(seed)
    set_torch_seed(seed)
    set_tf_seed(seed)
    key = set_jax_seed(seed)
    print("seed set", seed, key)

if __name__ == "__main__":
    main()
```

**注意：**

- GPU 并行可能仍有非确定性
- 某些算子在不同硬件上结果有微小差异
- 固定随机种子是必要条件，但不充分

**PyTorch 强制确定性算子：**

```python
# torch_deterministic.py
import torch

def enable_deterministic():
    # 中文注释：强制使用确定性算法
    torch.use_deterministic_algorithms(True)
    # 中文注释：禁用 cuDNN benchmark
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    enable_deterministic()
```

### 4.5 环境隔离（Docker / Conda / Poetry）

**Dockerfile 示例：**

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
CMD ["python", "train.py"]
```

**Conda 环境：**

```yaml
# environment.yml
name: llm-exp
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy=1.26
  - pip
  - pip:
      - torch==2.1.0
      - transformers==4.35.0
```

**Poetry 示例：**

```toml
# pyproject.toml (节选)
[tool.poetry]
name = "llm-exp"
version = "0.1.0"
description = "Experiment reproducibility"

[tool.poetry.dependencies]
python = ">=3.10,<3.12"
numpy = "^1.26"
```

通过固定依赖版本与环境描述文件，确保环境可重建。

### 4.5.1 环境隔离策略对比

| 工具 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| Docker | 环境一致性强 | GPU 配置复杂 | 生产/跨机 |
| Conda | 依赖管理方便 | 环境漂移可能 | 研究/实验 |
| Poetry | Python 依赖锁定 | 对系统依赖弱 | 轻量项目 |

### 4.6 环境快照与硬件信息记录

记录硬件与系统信息可帮助解释小范围漂移：

```bash
python -c "import platform; print(platform.platform())"
nvidia-smi
```

建议将硬件信息记录到实验追踪系统的 tag 中。

### 4.7 数据处理流水线可重复性

LLM 数据处理通常包含多个步骤（去重、清洗、过滤、抽样、标注），任何一步变化都会影响最终结果。建议：

- 每一步处理脚本独立版本化
- 记录每一步输入输出数据统计
- 固定采样的随机种子与分片规则

**数据处理流水线示例：**

```python
# pipeline.py
import json
import random

def load_raw(path):
    # 中文注释：读取原始数据
    with open(path, "r") as f:
        return [line.strip() for line in f]

def filter_data(data, min_len=10):
    # 中文注释：过滤过短样本
    return [x for x in data if len(x) >= min_len]

def sample_data(data, n=1000, seed=42):
    # 中文注释：固定随机种子采样
    random.seed(seed)
    return random.sample(data, n)

def save(path, data):
    with open(path, "w") as f:
        for x in data:
            f.write(x + "\n")

def main():
    raw = load_raw("data/raw.txt")
    filtered = filter_data(raw, min_len=10)
    sampled = sample_data(filtered, n=1000, seed=42)
    save("data/processed.txt", sampled)
    # 中文注释：保存数据统计
    stats = {"raw": len(raw), "filtered": len(filtered), "sampled": len(sampled)}
    with open("data/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
```

### 4.8 评测可重复性（Evaluation Reproducibility）

评测可重复性不仅关注模型训练，还应固定评测输入、提示词（Prompt）、解码参数：

- prompt 版本化（prompt template version）
- 解码参数固定（temperature, top_p, max_tokens）
- 评测集版本与抽样策略固定

**评测配置示例：**

```yaml
# eval.yaml
prompt:
  template: "You are a helpful assistant."
  version: v2
decoding:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 256
data:
  eval_set: eval_v1
  seed: 123
```

### 4.9 分布式训练的可重复性

分布式训练引入通信顺序与同步策略差异，建议：

- 固定通信后端版本（NCCL / GLOO）
- 固定 gradient accumulation 策略
- 记录 world size 与 rank 配置

**分布式配置示例：**

```yaml
# dist.yaml
distributed:
  backend: nccl
  world_size: 8
  gradient_accumulation_steps: 4
```

### 4.10 实验目录结构建议

统一目录结构可以减少协作成本：

```
project/
  configs/
  data/
  logs/
  runs/
  scripts/
  models/
```

建议每次实验在 `runs/<run_id>/` 下保存：

- `config.yaml` 配置快照
- `metrics.json` 指标
- `artifacts/` 模型与日志

### 4.11 统一日志格式

采用结构化日志（JSON log）便于检索：

```python
# logger.py
import json
import time

def log_event(event, payload):
    record = {"ts": time.time(), "event": event, "payload": payload}
    print(json.dumps(record))

if __name__ == "__main__":
    log_event("train_start", {"lr": 3e-4})
```

### 4.12 复现实验最小包（Minimal Repro Package）

对于关键结论，建议生成“最小复现包”，包含：

- 数据子集（最小可复现样本）
- 精简脚本（可独立运行）
- 依赖文件（environment.yml / requirements.txt）
- 说明文档（README）

这可以显著降低他人复现成本。

### 4.13 端到端可重复流程示例

以下示例展示从数据准备、训练到评测的端到端可重复流程：

```bash
# step1: 准备环境
conda env create -f environment.yml
conda activate llm-exp

# step2: 同步数据
dvc pull

# step3: 运行训练
python train.py seed=42 data.version=v1.2.0 training.lr=3e-4

# step4: 运行评测
python eval.py --config eval.yaml
```

**关键约束说明：**

- 每一步的输入输出都必须记录版本
- 训练与评测使用相同的配置系统
- 输出模型必须注册到模型仓库

### 4.14 评测数据与指标标准化

评测可重复性需要固定指标计算方式：

- 评测脚本版本化
- 指标公式透明化
- 后处理步骤固定

**指标配置示例：**

```yaml
# metrics.yaml
metrics:
  - name: accuracy
    type: classification
  - name: bleu
    type: nlp
    params:
      n_gram: 4
```

### 4.15 Prompt 版本化与模板管理

Prompt 本身是模型输入，应像代码一样版本化。推荐：

- Prompt 存储在独立目录 `prompts/`
- 每次更改记录版本号
- 在评测记录中绑定 prompt 版本

**Prompt 文件示例：**

```text
# prompts/v2.txt
You are a helpful assistant. Answer concisely.
```

### 4.16 训练过程快照（Checkpoint）管理

训练过程中的 checkpoint 是复现关键：

- 固定保存频率
- 记录 checkpoint 对应的 step 与指标
- 记录 checkpoint 文件哈希

**Checkpoint 记录示例：**

```json
{
  "checkpoint": "ckpt_1000.pt",
  "step": 1000,
  "loss": 0.45,
  "hash": "<sha256>"
}
```

### 4.17 多环境配置（dev/staging/prod）

不同环境可能需要不同的资源配置，建议采用分层配置：

```yaml
# configs/env/dev.yaml
hardware:
  gpu: "A100"
  nodes: 1
```

```yaml
# configs/env/prod.yaml
hardware:
  gpu: "A100"
  nodes: 8
```

通过 Hydra 的 `defaults` 组合即可快速切换环境。

### 4.18 模型卡（Model Card）与可重复性

模型卡（Model Card）是模型发布的标准化说明文档，可用于记录复现信息：

```markdown
# Model Card
- Model: llama2-7b
- Data Version: v1.2.0
- Training Config: config.yaml
- Eval Config: eval.yaml
- Metrics: acc=0.85
```

建议将模型卡作为模型仓库中的必备文件。

### 4.19 配置哈希与签名

为了确保配置未被篡改，可对配置文件生成哈希：

```bash
sha256sum configs/train.yaml > config.sha
```

在实验追踪中记录 `config_hash`，保证同一配置唯一可识别。

### 4.20 依赖锁定与环境导出

对于 Python 项目，可使用 `pip freeze` 输出依赖锁定文件：

```bash
pip freeze > requirements.lock.txt
```

在实验追踪中记录该文件，确保依赖一致。

---

## 5. 实验对比：一致性检查与分析工具

### 5.1 实验对比基础

实验对比需要至少三类信息：

- 配置差异（Config Diff）
- 数据差异（Data Diff）
- 代码差异（Code Diff）

**实验对比表格示例：**

| Run ID | 数据版本 | 模型配置 | lr | batch | acc | loss |
| --- | --- | --- | --- | --- | --- | --- |
| run_001 | v1.2.0 | llama2-7b | 3e-4 | 16 | 0.85 | 0.35 |
| run_002 | v1.2.0 | llama2-7b | 1e-4 | 16 | 0.84 | 0.37 |
| run_003 | v1.2.1 | llama2-7b | 3e-4 | 16 | 0.86 | 0.34 |

**配置对比示例：**

```python
# diff_config.py
import yaml
from deepdiff import DeepDiff

def load(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg1 = load("run1/config.yaml")
    cfg2 = load("run2/config.yaml")
    diff = DeepDiff(cfg1, cfg2, ignore_order=True)
    print(diff)

if __name__ == "__main__":
    main()
```

### 5.2 实验对比统计分析

假设两组实验结果 $X$ 与 $Y$，多次运行后得到均值与方差：

\[
\bar{X} = \frac{1}{n} \sum_{i=1}^n x_i, \quad s_X^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{X})^2
\]

可用 t-test 或 bootstrap 评估差异显著性：

```python
# ttest_demo.py
import numpy as np
from scipy import stats

def main():
    # 中文注释：两组实验结果
    x = np.array([0.82, 0.83, 0.81, 0.84])
    y = np.array([0.80, 0.79, 0.81, 0.80])
    t, p = stats.ttest_ind(x, y, equal_var=False)
    print("t=", t, "p=", p)

if __name__ == "__main__":
    main()
```

**Bootstrap 示例：**

```python
# bootstrap_demo.py
import numpy as np

def bootstrap_ci(values, n=1000, alpha=0.05):
    means = []
    for _ in range(n):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    low = np.percentile(means, 100 * alpha / 2)
    high = np.percentile(means, 100 * (1 - alpha / 2))
    return low, high

def main():
    values = np.array([0.82, 0.83, 0.81, 0.84])
    low, high = bootstrap_ci(values)
    print("CI=", low, high)

if __name__ == "__main__":
    main()
```

### 5.3 可视化对比

推荐使用 MLflow/WandB 内置对比面板，也可导出 CSV 自行绘制：

```python
# plot_compare.py
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("runs.csv")
    # 中文注释：对比不同 run 的 loss 曲线
    for run_id, g in df.groupby("run_id"):
        plt.plot(g["step"], g["loss"], label=run_id)
    plt.legend()
    plt.savefig("compare.png")

if __name__ == "__main__":
    main()
```

### 5.4 回归检测与阈值策略

回归检测需要定义可接受的性能浮动区间：

- 绝对阈值：如 top-1 accuracy 下降不超过 0.2%
- 相对阈值：如 loss 增加不超过 1%

可结合历史基线构建自动报警机制。

### 5.4 一致性检查（Repro Check）

常用一致性指标：

- 指标差异阈值（如 top-1 accuracy 差 < 0.1%）
- 参数分布相似度（例如权重层统计）
- 训练曲线形状一致性

可以用哈希校验模型权重与数据快照：

```bash
sha256sum model.pt
sha256sum data/corpus_v1/*
```

### 5.5 端到端复现实验脚本

```bash
# reproduce.sh
set -e

# 中文注释：创建环境
conda env create -f environment.yml
conda activate llm-exp

# 中文注释：同步数据版本
dvc pull

# 中文注释：运行训练
python train.py seed=42 training.lr=3e-4
```

### 5.6 消融实验（Ablation Study）模板

消融实验用于验证某个组件的有效性，建议使用统一模板记录：

- baseline：完整模型
- ablation_1：移除某模块
- ablation_2：替换参数策略

**消融记录示例：**

| Run ID | Variant | 修改点 | acc | loss | 备注 |
| --- | --- | --- | --- | --- | --- |
| run_010 | baseline | 无 | 0.85 | 0.35 | 完整模型 |
| run_011 | no_dropout | 去除 dropout | 0.83 | 0.38 | 稳定性下降 |
| run_012 | lr_decay | 替换学习率策略 | 0.86 | 0.34 | 小幅提升 |

### 5.7 实验报告（Repro Report）模板

推荐为关键实验生成可重复性报告：

```markdown
# Repro Report

## 基础信息
- run_id: run_001
- git_commit: abc123
- data_version: v1.2.0
- seed: 42
- hardware: A100-80G

## 配置摘要
- model: llama2-7b
- lr: 3e-4
- batch_size: 16
- epochs: 3

## 结果
- acc: 0.85
- loss: 0.35

## 复现说明
- 使用 reproduce.sh
- 依赖环境 environment.yml
```

### 5.8 自动化对比流水线

通过简单脚本实现实验对比自动化：

```python
# compare_runs.py
import json

def load_run(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    run_a = load_run("runs/run_a.json")
    run_b = load_run("runs/run_b.json")
    print("acc diff:", run_a["acc"] - run_b["acc"])
    print("loss diff:", run_a["loss"] - run_b["loss"])

if __name__ == "__main__":
    main()
```

### 5.9 实验对比工具栈

常见对比工具与用途：

| 工具 | 角色 | 说明 |
| --- | --- | --- |
| MLflow | 追踪 + 对比 | 对比指标、配置、工件 |
| WandB | 追踪 + 面板 | run 对比与可视化 |
| DVC | 数据差异 | 对比数据版本与管道 |
| Git | 代码差异 | 关联 commit 与 PR |

建议将对比结果导出成统一报告，便于审核与分享。

### 5.10 回归测试与基线管理

回归测试保证性能不退化，关键步骤：

- 选定基线 run 作为比较对象
- 定义指标阈值与统计规则
- 失败时自动阻断合并或发布

**基线配置示例：**

```yaml
# baseline.yaml
baseline_run: run_001
metric_thresholds:
  acc: -0.002
  loss: 0.005
```

### 5.11 结果审计与签名

对于重要模型，建议进行结果签名与审计：

- 结果摘要 hash
- 训练过程日志 hash
- 模型权重 hash

**签名示例：**

```bash
sha256sum metrics.json > metrics.sha
sha256sum model.pt > model.sha
```

### 5.12 Prompt 回归测试

对话模型的行为会因 prompt 变化而产生巨大差异。建议建立 prompt 回归测试集：

- 固定一组代表性 prompt
- 对每次模型更新进行自动化评测
- 记录输出差异并人工抽查

**Prompt 回归配置示例：**

```yaml
# prompt_regression.yaml
prompts:
  - id: p1
    text: "Summarize the following text"
  - id: p2
    text: "Explain the algorithm in simple terms"
```

### 5.13 大模型评测指标一致性

LLM 评测常用指标包括 BLEU、ROUGE、MMLU、TruthfulQA 等。需要确保：

- 指标实现版本固定
- 评测集版本固定
- 评分脚本可复现

建议将评分脚本与指标配置文件一起纳入版本控制。

### 5.14 对比报告模板

对比报告用于总结不同实验的差异与结论：

```markdown
# Experiment Comparison Report

## 运行摘要
- run_a: run_001
- run_b: run_002

## 配置差异
- lr: 3e-4 -> 1e-4
- batch_size: 16 -> 16

## 数据差异
- data_version: v1.2.0 -> v1.2.0

## 结果对比
- acc: 0.85 -> 0.84
- loss: 0.35 -> 0.37

## 结论
- lr 降低导致性能下降
```

### 5.15 数据漂移检测（Data Drift）

数据漂移会导致评测结果不可比，建议定期检测数据分布差异。常用指标包括 KL 散度（Kullback-Leibler Divergence）：

\[
D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
\]

**漂移检测示例：**

```python
# drift_check.py
import numpy as np

def kl_div(p, q):
    p = np.asarray(p) + 1e-12
    q = np.asarray(q) + 1e-12
    return np.sum(p * np.log(p / q))

def main():
    p = np.array([0.4, 0.6])
    q = np.array([0.5, 0.5])
    print("kl=", kl_div(p, q))

if __name__ == "__main__":
    main()
```

### 5.16 指标解释与报告规范

指标报告应包含数值、统计区间与运行次数，例如：

- acc=0.85 ± 0.01 (n=5)
- loss=0.35 ± 0.02 (n=5)

在报告中明确统计方式，避免误解。

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

- **实验标准化**：所有实验必须使用统一的入口脚本与配置
- **自动记录**：自动记录 Git commit、配置、数据版本
- **最小化随机性**：固定随机种子，避免非确定性算子
- **环境可重建**：使用 Docker/Conda/Poetry 记录依赖
- **数据可追溯**：每次数据处理生成快照与哈希
- **版本统一命名**：数据、模型、代码统一命名规则
- **失败也是数据**：失败实验记录原因与日志
- **实验注册表**：统一登记实验与产出模型

### 6.2 常见陷阱

- 只固定了 Python 随机种子，忽略 GPU 与第三方库
- 忽视依赖库的 minor 版本差异
- 数据预处理不记录配置与参数
- 缺少实验失败记录（失败也应可追踪）
- 不记录硬件信息（GPU 型号、驱动版本）
- 分布式训练未固定通信后端版本
- 忽略随机数据增强的种子

### 6.3 复现等级（Reproducibility Levels）

可将复现能力分为三个等级：

1. **Level 1：脚本复现**
   - 同机同环境运行脚本可得相同结果
2. **Level 2：环境复现**
   - 通过环境文件或容器，在不同机器上复现
3. **Level 3：跨平台复现**
   - 跨硬件、跨系统仍保持结果稳定或可解释

LLM 项目至少应达到 Level 2，关键实验争取 Level 3。

### 6.4 常见问题排查（Troubleshooting）

- **结果漂移**：检查随机种子、数据版本、依赖版本
- **指标突然下降**：检查数据管道是否更新、训练配置是否变更
- **评测不稳定**：检查 prompt 模板、评测集抽样策略
- **GPU 差异**：检查混合精度与 cuDNN 设置

### 6.5 成熟度模型（Maturity Model）

| 阶段 | 特征 | 改进重点 |
| --- | --- | --- |
| 初级 | 手动记录 | 建立配置与追踪系统 |
| 中级 | 自动记录 | 统一数据与模型版本 |
| 高级 | 全自动流水线 | 统计检验与回归检测 |

### 6.6 LLM 特有注意点

- **Prompt 版本化**：Prompt 作为输入，必须版本化与追踪
- **解码策略**：温度与采样策略直接影响结果
- **对齐数据**：RLHF/对齐数据需记录清晰
- **模型压缩**：量化/蒸馏需要记录过程与配置

### 6.7 复现失败案例分析

**案例 1：评测漂移**

现象：同一模型在不同机器上评测结果差异 1%。

原因：评测脚本默认使用随机采样，未固定 seed；同时 prompt 模板在不同分支有差异。

解决方案：

- 固定评测采样种子
- 统一 prompt 版本，加入 Git 管理
- 将评测配置写入 `eval.yaml` 并记录到 run

**案例 2：训练结果回归**

现象：某次训练 loss 明显变高。

原因：依赖包 minor 版本更新，引入默认行为变化；同时数据清洗脚本参数变动未记录。

解决方案：

- 固定依赖版本，使用 lock 文件
- 数据清洗参数写入配置并追踪

### 6.8 复现策略与流程建议

复现流程应形成团队标准操作流程（SOP）：

- 新实验必须登记（experiment registry）
- 每次 run 必须生成配置快照
- 每个实验必须关联 Git commit
- 关键实验必须生成复现报告

建立 SOP 后，可将复现工作自动化到 CI 中。

### 6.9 复现清单（Extended Checklist）

以下清单适用于关键实验的最终验收：

- **配置一致**：训练、评测、数据处理配置完整存档
- **数据一致**：数据版本与哈希匹配，数据处理脚本可重放
- **代码一致**：Git commit 与依赖锁定文件一致
- **模型一致**：模型权重哈希一致，注册到模型仓库
- **环境一致**：Python、依赖包、系统库与驱动版本一致
- **随机性一致**：随机种子与确定性设置记录完整
- **评测一致**：prompt 模板与解码参数固定
- **指标一致**：指标实现版本一致，评分脚本可复现

对于跨团队交付，建议在交付前执行一次“复现实验”，验证在新环境中重跑可得到一致结果。

### 6.10 团队协作规范

团队协作中建议明确责任分工：

- 负责人保证实验登记与配置完整
- 数据负责人保证数据版本与处理脚本可追溯
- 模型负责人保证模型注册与评测报告完整

通过明确角色和标准，避免“信息断层”导致不可复现。

---

## 7. 总结

实验与可重复性（Experimentation and Reproducibility）是 LLM 系统工程的核心能力。通过配置管理（YAML / Hydra）、实验追踪（MLflow / WandB）、版本控制（Git / DVC / Model Registry）、随机种子控制与环境隔离（Docker / Conda / Poetry），可以显著提升实验可信度与团队协作效率。同时，多次实验与统计检验保证了结论的稳健性。工程上应将可重复性作为“默认要求”，并把追踪与版本控制融入开发流程，才能在大规模实验中避免“不可复现”的陷阱。

进一步建议：

- 将可重复性指标纳入项目 KPI
- 对关键实验强制生成复现报告
- 在 CI 中加入复现测试与回归检测

可重复性不是一次性工作，而是持续性的工程能力建设。

当系统规模扩大时，建议设立专门的复现负责人或平台团队，持续维护实验追踪与版本控制基础设施。
这能显著降低跨团队交付成本。
对于外部发布，还可提升对外可信度。
建议在关键里程碑前执行一次全链路复现。
确保审计与评测结果可对齐。

---

## 参考清单（Checklist）

- [ ] 每次实验生成唯一 Run ID 并记录
- [ ] 记录 Git commit 与分支
- [ ] 记录数据版本与数据哈希
- [ ] 固定随机种子并记录
- [ ] 固定环境与依赖版本
- [ ] 记录评估指标与模型产出
- [ ] 多次实验对比并进行统计检验
