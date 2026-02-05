# Experimentation and Reproducibility

**[English](README.md) | [中文](README_CN.md)**

This chapter covers experimentation and reproducibility practices for LLM system development, ranging from Configuration Management to Version Control, Experiment Tracking, Random Seed Control, Environment Isolation, and Experiment Comparison. The goal is to ensure that the same code, same data, and same environment produce consistent or explainable results across multiple runs, multiple team members, and different hardware.

---

## 1. Background: Why is Reproducibility Important?

In LLM systems, training and evaluation involve massive data, complex dependencies, randomness, and long pipeline chains. Reproducibility is not only scientific integrity but also the cornerstone of engineering credibility and iteration efficiency:

- **Engineering Credibility**: Reproducible results make conclusions verifiable, avoiding "only works on certain machines."
- **Debugging Efficiency**: Reproducible runs are the prerequisite for locating regression issues and performance degradation.
- **Cross-team Collaboration**: Consistent environments and configurations enable multi-person collaboration and reproducibility.
- **Regulation and Audit**: Model releases require traceability of training data, parameters, code versions, and output models.

Uncertainties in LLM training and evaluation come from multiple levels: data sampling, initialization weights, non-determinism in parallel computing, hardware differences, and dependency library version differences. Engineering needs to make these uncertainties "controllable" to achieve reproducibility or explainability.

In enterprise-level LLM systems, reproducibility also directly impacts:

- **Cost Control**: Inability to reproduce leads to duplicate training, wasting compute budget.
- **Model Governance**: Need to trace the complete chain from model training to deployment.
- **Launch Risk**: Unstable regression testing increases canary deployment risk.
- **Knowledge Accumulation**: Traceable experiments turn "experience" into reusable assets.

### 1.1 Engineering Cost of Reproducibility

Reproducibility is not "free," requiring engineering investment, but long-term benefits are significant:

- Initial cost: Establish configuration management and tracking systems, refactor training scripts
- Ongoing cost: Maintain data and model versions, update documentation and checklists
- Long-term benefits: Improved debugging efficiency, consistent R&D collaboration, reduced regression risk

In practice, reproducibility should be integrated into the R&D process rather than treated as "extra work."

### 1.2 Reproducibility and Explainability

When experimental results cannot be completely consistent, at minimum achieve "explainability":

- Clear sources of differences (randomness, data drift, or environment changes)
- Quantifiable change magnitude (statistical intervals or thresholds)
- Ability to locate influencing factors through comparative analysis

This means teams should not only pursue "repetition" but also "explanation."

---

## 2. Core Concepts: Experiment Tracking, Version Control, and Traceability

### 2.1 Experiment Tracking

Experiment tracking records configuration, code version, data version, metrics, and outputs for each run. Typical tools include **MLflow** and **Weights & Biases (WandB)**. Complete tracking should cover:

- Configuration (Config): Hyperparameters, paths, model architecture, training strategy
- Code (Code): Git commit, branch, diff
- Data (Data): Data version, data filtering logic, statistical summary
- Model (Model): Weights, training curves, evaluation metrics
- Environment (Env): Python version, dependency package versions, hardware information

### 2.2 Configuration Management

Configuration management decouples experiment parameters from code logic. **YAML** or **Hydra** is recommended. Configuration principles:

- Single Source of Truth
- Structured + Inheritance
- Trackable + Serializable

Configuration management should also consider:

- **Parameter space definition**: Clearly define which parameters are tunable and which must be fixed
- **Configuration validation**: Prevent illegal combinations that cause irreproducibility
- **Configuration archiving**: Write configuration of each run to the run directory

### 2.3 Version Control

Version control includes not only code (Code), but also data (Data) and models (Model). Common strategies:

- Code version: Git + semantic tags (Tag)
- Data version: DVC, Git LFS, data snapshots
- Model version: MLflow Model Registry / WandB Artifacts

### 2.4 Traceability

Traceability requires that experimental results can trace back to all key inputs: data, configuration, environment, code. Ideally, any experimental result can be fully reconstructed through a "run ID."

### 2.5 Data Lineage

Data lineage describes the processing path of data from original sources to training input. LLM data typically goes through multiple steps: cleaning, deduplication, filtering, sampling, and annotation. If each step's parameters and versions are not recorded, the dataset cannot be reproduced.

Recommend recording:

- Original data sources (URL/version/timestamp)
- Cleaning scripts and parameters
- Filtering rules and thresholds
- Sampling methods and random seeds
- Hash summary of output data

### 2.6 Model Lineage

Model lineage requires recording all dependencies from base model to fine-tuned model, including:

- Base model version (e.g., `llama2-7b`)
- Fine-tuning data version
- Fine-tuning configuration and training logs
- Details of quantization/distillation/pruning processes

### 2.7 Configuration Versioning Strategy

Configuration itself needs version control. Recommended to follow:

- All configuration files committed to Git
- Copy configuration snapshot to run directory on each experiment run
- Semantic tag for key configurations (training strategy, data filtering rules)

Recommend establishing change log (Changelog) for `configs/` directory, recording reasons for major parameter changes.

### 2.8 Experiment Metadata Standards

Establish unified metadata schema (Metadata Schema) for automated analysis. Common fields:

- `run_id`, `experiment_name`
- `git_commit`, `branch`
- `data_version`, `data_hash`
- `model_name`, `model_version`
- `seed`, `hardware`
- `metrics`, `artifacts`

Metadata can be saved as JSON or YAML and attached as additional information in experiment tracking system.

### 2.9 Experiment Registry

Experiment registry is an important component for team collaboration. Recommended to include:

- Experiment name and goals
- Owner
- Data version and model version
- Expected metrics and evaluation plan
- Risk points and dependencies

Registry can be placed in shared documents or databases and connected to experiment tracking system.

### 2.10 Artifact Management

LLM projects generate many artifacts, including:

- Training logs
- Model weights
- Intermediate checkpoints
- Visualization charts

These artifacts should be stored and named uniformly, avoiding scattered personal directories.

### 2.11 Naming Conventions and Experiment Encoding

Consistent naming conventions are the foundation of traceability. Recommended to define unified encoding rules for experiments, data, and models:

- Experiment: `exp_<date>_<goal>_<version>`
- Data: `data_<source>_<version>`
- Model: `model_<architecture>_<version>`

For example: `exp_2026-02-01_repro_v1`, `data_corpus_v1.2.0`, `model_llama2_v1`.

### 2.12 Experiment Documentation

Each important experiment should generate brief documentation, including:

- Goals and hypotheses
- Methods and configurations
- Data and model versions
- Conclusions and follow-up plans

Documentation can be bound to the run as part of reproducibility records.

### 2.13 Configuration Management Anti-patterns

Common anti-patterns significantly reduce reproducibility:

- Configuration scattered in code, unable to be centrally managed
- Modify configuration at runtime without recording
- Configuration files have "implicit default values," causing different results for different people

Core to avoiding anti-patterns is making configuration explicit and recording it.

### 2.14 Metadata Field Dictionary

| Field | Description |
| --- | --- |
| run_id | Unique experiment identifier |
| experiment | Experiment grouping |
| git_commit | Code version |
| data_version | Data version |
| model_version | Model version |
| seed | Random seed |
| hardware | Hardware information |
| metrics | Key metrics |

### 2.15 Audit and Compliance Requirements

In regulated scenarios, need to meet audit and compliance requirements:

- Traceable training data sources
- Traceable model training configurations
- Traceable evaluation results and reports

Experiment tracking system should support exporting audit reports.

---

## 3. Mathematical Principles: Randomness Control and Statistical Testing

### 3.1 Randomness and Reproducibility

Randomness comes from initialization, data sampling, Dropout, parallel execution, etc. Reproducibility requires controlling randomness within a controllable range. Let random variable $X$ represent experimental result metric (e.g., accuracy), its expectation and variance are:

\[
\mathbb{E}[X] = \mu, \quad \mathrm{Var}(X) = \sigma^2
\]

Fixing random seed is a necessary condition for controlling random variable generation process, but **cannot guarantee absolute consistency**, especially when GPU parallelism and non-deterministic operators exist.

In parallel training, different thread scheduling orders may lead to different floating-point accumulation errors. Even with same random seed, differences in floating-point operation order may still cause result drift.

### 3.2 Statistical Stability and Repeated Experiments

Single experiment is insufficient to judge model superiority. Need multiple repeated experiments and statistical testing. Common practices:

- Multiple runs, take mean and variance
- Use t-test or bootstrap to estimate confidence intervals

For two sets of experimental results $X$ and $Y$, the mean difference test:

\[
t = \frac{\bar{X} - \bar{Y}}{\sqrt{\frac{s_X^2}{n_X} + \frac{s_Y^2}{n_Y}}}
\]

When $p$-value is less than significance level (e.g., 0.05), the difference is considered significant.

**Bootstrap confidence interval example**:

\[
\hat{\theta}^* = \frac{1}{B} \sum_{b=1}^{B} \theta_b, \quad CI_{95\%} = [\theta_{0.025}, \theta_{0.975}]
\]

### 3.3 Reproducibility and Replicability

- **Reproducibility**: Reproduce results under same data and code conditions
- **Replicability**: Reproduce conclusions under different implementation or data conditions

Engineering goal at minimum ensures Reproducibility, scientific goal pursues Replicability simultaneously.

### 3.4 Randomness Propagation and Control Boundaries

In complex systems, randomness propagation may affect downstream metrics. A common strategy is to define "random boundaries":

- Use fixed seeds in data sampling stage
- Use deterministic operators in training stage
- Fixed evaluation sets and inference parameters in evaluation stage

If cannot be completely deterministic, statistical stability can be assessed through multiple runs.

### 3.5 Numerical Precision and Hardware Differences

Numerical precision is an important source of reproducibility. Different hardware or different mixed precision strategies may cause subtle differences:

- FP32 and FP16 have different accumulation errors
- BF16 has different rounding strategies on some hardware
- Different GPU architectures (e.g., V100 vs A100) bring numerical differences

For sensitive metrics, need to clearly define:

- Training precision (FP32/FP16/BF16)
- Whether to use automatic mixed precision (AMP)
- Gradient scaling strategy

Recommend recording precision-related configurations in experiment tracking and ensuring consistency in comparative experiments.

### 3.6 Confidence Intervals and Statistical Power

In multiple experiments, besides mean, confidence intervals should also be reported:

\[
CI = \bar{X} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}
\]

Where $z_{\alpha/2}$ is the standard normal distribution quantile. Statistical power measures ability to detect differences. Insufficient sample size leads to unreliable conclusions. For LLM evaluation, maximize sample size or use bootstrap as much as possible.

### 3.7 Variance Source Decomposition

Variance comes from multiple sources:

- Data sampling variance
- Initialization variance
- Parallel computing variance

Can decompose variance sources through control variable experiments. For example, fix model initialization, only change data sampling, evaluate sampling variance contribution.

---

## 4. Code Implementation: Tracking Tools, Configuration Management, and Seed Control

### 4.1 Configuration Management (YAML + Hydra)

**YAML basic configuration example**:

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

**Hydra combination and override example**:

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
    # Print configuration to ensure traceability
    print(cfg)
    # Insert training logic here
    pass

if __name__ == "__main__":
    main()
```

**Command line override**:

```bash
python train.py training.lr=1e-4 data.version=v1.2.1 seed=123
```

Hydra automatically outputs run directory and configuration snapshot, greatly enhancing traceability.

**Hydra multirun example**:

```bash
python train.py -m training.lr=1e-4,3e-4,5e-4 training.batch_size=8,16
```

### 4.1.1 Configuration Validation and Structured Configuration

Use dataclass or pydantic to define structured configuration, ensuring type consistency:

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

### 4.1.2 Configuration Management Strategy Comparison

| Strategy | Advantages | Disadvantages | Use Case |
| --- | --- | --- | --- |
| Single file YAML | Simple and intuitive | Poor scalability | Small experiments |
| Layered YAML | Flexible combination | Complex structure | Medium projects |
| Hydra + schema | Strong validation, composable | Learning cost | Large systems |

For LLM projects, recommend using Hydra + structured configuration to ensure parameter consistency.

### 4.2 Experiment Tracking (MLflow / WandB)

**MLflow example**:

```python
# mlflow_train.py
import mlflow
import mlflow.sklearn

def train_model(params):
    # Simple model example
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
        # Record model and environment
        mlflow.set_tag("git_commit", "<commit_hash>")
        mlflow.set_tag("data_version", "v1.2.0")
        mlflow.log_dict({"model": model}, "model.json")

if __name__ == "__main__":
    main()
```

**MLflow artifact recording example**:

```python
# mlflow_artifact.py
import mlflow

def main():
    mlflow.set_experiment("exp_repro")
    with mlflow.start_run():
        # Save configuration file
        mlflow.log_artifact("configs/train.yaml")
        # Save training logs
        mlflow.log_artifact("logs/train.log")

if __name__ == "__main__":
    main()
```

**WandB example**:

```python
# wandb_train.py
import wandb

def train_model(config):
    # Simulate training process
    wandb.log({"loss": 0.35, "acc": 0.85})

def main():
    wandb.init(project="llm-exp", config={"lr": 3e-4, "batch_size": 16})
    train_model(wandb.config)
    wandb.finish()

if __name__ == "__main__":
    main()
```

**WandB Sweep example (parameter search)**:

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

**Key points**:

- Record Git commit, data version, configuration snapshot
- Output model weights and evaluation metrics
- Support comparing multiple runs

**Experiment tracking field recommendations**:

- `run_id`: Unique identifier
- `experiment`: Experiment grouping
- `config_hash`: Configuration digest
- `data_hash`: Data digest
- `model_hash`: Model digest
- `seed`: Random seed
- `hardware`: GPU/CPU model

### 4.2.1 Experiment Tracking and Configuration Binding

When starting experiment, automatically save configuration and bind to run:

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

### 4.3 Version Control (code, data, models)

#### Code Version (Git)

- Commit fixed semantics
- Tag binds release version
- Record experiment run and commit association

**Git LFS (large files) example**:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "track model weights"
```

#### Data Version (DVC / Git LFS)

**DVC example**:

```bash
dvc init
dvc add data/corpus_v1
git add data/corpus_v1.dvc .gitignore
git commit -m "track corpus v1"
```

**Data manifest example**:

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

**Data version tag**:

```bash
git tag -a data-v1.2.0 -m "dataset v1.2.0"
```

**DVC remote storage example**:

```bash
dvc remote add -d storage s3://my-bucket/dvc
dvc push
```

### 4.3.1 Version Control Strategy Comparison

| Object | Recommended Tools | Description |
| --- | --- | --- |
| Code | Git | Version management core |
| Data | DVC / LakeFS | Data snapshots, version rollback |
| Model | MLflow / WandB Artifacts | Model registration and rollback |

Different objects' version control should use unified naming and tags, avoiding information silos.

#### Model Version (Model Registry)

Register model in MLflow Model Registry or WandB Artifacts:

- `model:v1` binds training parameters, metrics, data version
- Supports rollback and canary deployment

### 4.4 Random Seed Management (Seed Control)

**Python / NumPy / PyTorch / TensorFlow / JAX**:

```python
# seed.py
import os
import random
import numpy as np

def set_seed(seed: int):
    # Python built-in random
    random.seed(seed)
    # NumPy random
    np.random.seed(seed)
    # Environment variable
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
        # JAX uses PRNGKey
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

**Note**:

- GPU parallelism may still be non-deterministic
- Some operators have subtle differences on different hardware
- Fixing random seed is necessary but not sufficient

**PyTorch forced deterministic operators**:

```python
# torch_deterministic.py
import torch

def enable_deterministic():
    # Force deterministic algorithms
    torch.use_deterministic_algorithms(True)
    # Disable cuDNN benchmark
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    enable_deterministic()
```

### 4.5 Environment Isolation (Docker / Conda / Poetry)

**Dockerfile example**:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
CMD ["python", "train.py"]
```

**Conda environment**:

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

**Poetry example**:

```toml
# pyproject.toml (excerpt)
[tool.poetry]
name = "llm-exp"
version = "0.1.0"
description = "Experiment reproducibility"

[tool.poetry.dependencies]
python = ">=3.10,<3.12"
numpy = "^1.26"
```

By fixing dependency versions and environment description files, ensure environment can be rebuilt.

### 4.5.1 Environment Isolation Strategy Comparison

| Tool | Advantages | Disadvantages | Use Case |
| --- | --- | --- | --- |
| Docker | Strong environment consistency | Complex GPU configuration | Production/cross-machine |
| Conda | Convenient dependency management | Possible environment drift | Research/experiments |
| Poetry | Python dependency locking | Weak on system dependencies | Lightweight projects |

### 4.6 Environment Snapshots and Hardware Information Recording

Recording hardware and system information can help explain small-range drift:

```bash
python -c "import platform; print(platform.platform())"
nvidia-smi
```

Recommend recording hardware information to experiment tracking system tags.

### 4.7 Data Processing Pipeline Reproducibility

LLM data processing typically includes multiple steps (deduplication, cleaning, filtering, sampling, annotation), any change affects final results. Recommend:

- Independently version each processing script
- Record input/output data statistics for each step
- Fix sampling random seeds and sharding rules

**Data processing pipeline example**:

```python
# pipeline.py
import json
import random

def load_raw(path):
    # Read raw data
    with open(path, "r") as f:
        return [line.strip() for line in f]

def filter_data(data, min_len=10):
    # Filter too short samples
    return [x for x in data if len(x) >= min_len]

def sample_data(data, n=1000, seed=42):
    # Fix random seed for sampling
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
    # Save data statistics
    stats = {"raw": len(raw), "filtered": len(filtered), "sampled": len(sampled)}
    with open("data/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
```

### 4.8 Evaluation Reproducibility

Evaluation reproducibility should not only focus on model training, but also fix evaluation inputs, prompts, decoding parameters:

- Prompt versioning (prompt template version)
- Fixed decoding parameters (temperature, top_p, max_tokens)
- Fixed evaluation set version and sampling strategy

**Evaluation configuration example**:

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

### 4.9 Distributed Training Reproducibility

Distributed training introduces communication order and synchronization strategy differences. Recommend:

- Fix communication backend version (NCCL / GLOO)
- Fix gradient accumulation strategy
- Record world size and rank configuration

**Distributed configuration example**:

```yaml
# dist.yaml
distributed:
  backend: nccl
  world_size: 8
  gradient_accumulation_steps: 4
```

### 4.10 Recommended Experiment Directory Structure

Unified directory structure reduces collaboration cost:

```
project/
  configs/
  data/
  logs/
  runs/
  scripts/
  models/
```

Recommend saving in `runs/<run_id>/` for each experiment:

- `config.yaml` Configuration snapshot
- `metrics.json` Metrics
- `artifacts/` Models and logs

### 4.11 Unified Log Format

Use structured logs (JSON logs) for easy retrieval:

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

### 4.12 Minimal Reproduction Package

For key conclusions, recommend generating "minimal reproduction package," including:

- Data subset (minimal reproducible sample)
- Simplified scripts (can run independently)
- Dependency files (environment.yml / requirements.txt)
- Documentation (README)

This can significantly reduce others' reproduction cost.

### 4.13 End-to-End Reproducible Process Example

The following example shows end-to-end reproducible process from data preparation, training to evaluation:

```bash
# step1: Prepare environment
conda env create -f environment.yml
conda activate llm-exp

# step2: Sync data
dvc pull

# step3: Run training
python train.py seed=42 data.version=v1.2.0 training.lr=3e-4

# step4: Run evaluation
python eval.py --config eval.yaml
```

**Key constraint notes**:

- Each step's input/output must record version
- Training and evaluation use same configuration system
- Output model must be registered to model registry

### 4.14 Evaluation Data and Metrics Standardization

Evaluation reproducibility requires fixing metric calculation methods:

- Version evaluation scripts
- Transparent metric formulas
- Fixed post-processing steps

**Metrics configuration example**:

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

### 4.15 Prompt Versioning and Template Management

Prompt itself is model input, should be versioned like code. Recommend:

- Prompt stored in independent directory `prompts/`
- Record version number on each change
- Bind prompt version in evaluation records

**Prompt file example**:

```text
# prompts/v2.txt
You are a helpful assistant. Answer concisely.
```

### 4.16 Training Process Snapshot (Checkpoint) Management

Checkpoints during training are key to reproduction:

- Fix save frequency
- Record checkpoint corresponding step and metrics
- Record checkpoint file hash

**Checkpoint recording example**:

```json
{
  "checkpoint": "ckpt_1000.pt",
  "step": 1000,
  "loss": 0.45,
  "hash": "<sha256>"
}
```

### 4.17 Multi-Environment Configuration (dev/staging/prod)

Different environments may need different resource configurations. Recommend using layered configuration:

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

Quickly switch environments through Hydra `defaults` combination.

### 4.18 Model Card and Reproducibility

Model card is standardized documentation for model release, can be used to record reproducibility information:

```markdown
# Model Card
- Model: llama2-7b
- Data Version: v1.2.0
- Training Config: config.yaml
- Eval Config: eval.yaml
- Metrics: acc=0.85
```

Recommend making model card a required file in model registry.

### 4.19 Configuration Hash and Signature

To ensure configuration not tampered, can generate hash for configuration files:

```bash
sha256sum configs/train.yaml > config.sha
```

Record `config_hash` in experiment tracking, ensuring same configuration uniquely identifiable.

### 4.20 Dependency Locking and Environment Export

For Python projects, can use `pip freeze` to output dependency lock file:

```bash
pip freeze > requirements.lock.txt
```

Record this file in experiment tracking, ensuring dependency consistency.

---

## 5. Experimental Comparison: Consistency Check and Analysis Tools

### 5.1 Experimental Comparison Basics

Experimental comparison requires at least three types of information:

- Configuration difference (Config Diff)
- Data difference (Data Diff)
- Code difference (Code Diff)

**Experimental comparison table example**:

| Run ID | Data Version | Model Config | lr | batch | acc | loss |
| --- | --- | --- | --- | --- | --- | --- |
| run_001 | v1.2.0 | llama2-7b | 3e-4 | 16 | 0.85 | 0.35 |
| run_002 | v1.2.0 | llama2-7b | 1e-4 | 16 | 0.84 | 0.37 |
| run_003 | v1.2.1 | llama2-7b | 3e-4 | 16 | 0.86 | 0.34 |

**Configuration comparison example**:

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

### 5.2 Experimental Comparison Statistical Analysis

Assume two sets of experimental results $X$ and $Y$, after multiple runs get mean and variance:

\[
\bar{X} = \frac{1}{n} \sum_{i=1}^n x_i, \quad s_X^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{X})^2
\]

Can use t-test or bootstrap to evaluate significance of differences:

```python
# ttest_demo.py
import numpy as np
from scipy import stats

def main():
    # Two sets of experimental results
    x = np.array([0.82, 0.83, 0.81, 0.84])
    y = np.array([0.80, 0.79, 0.81, 0.80])
    t, p = stats.ttest_ind(x, y, equal_var=False)
    print("t=", t, "p=", p)

if __name__ == "__main__":
    main()
```

**Bootstrap example**:

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

### 5.3 Visualization Comparison

Recommend using MLflow/WandB built-in comparison panels, or export CSV and plot yourself:

```python
# plot_compare.py
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("runs.csv")
    # Compare loss curves of different runs
    for run_id, g in df.groupby("run_id"):
        plt.plot(g["step"], g["loss"], label=run_id)
    plt.legend()
    plt.savefig("compare.png")

if __name__ == "__main__":
    main()
```

### 5.4 Regression Detection and Threshold Strategy

Regression detection needs to define acceptable performance float range:

- Absolute threshold: e.g., top-1 accuracy drop not exceeding 0.2%
- Relative threshold: e.g., loss increase not exceeding 1%

Can combine historical baseline to build automatic alert mechanism.

### 5.5 Consistency Check (Repro Check)

Common consistency metrics:

- Metric difference threshold (e.g., top-1 accuracy diff < 0.1%)
- Parameter distribution similarity (e.g., weight layer statistics)
- Training curve shape consistency

Can use hash to verify model weights and data snapshots:

```bash
sha256sum model.pt
sha256sum data/corpus_v1/*
```

### 5.6 End-to-End Reproduction Experiment Script

```bash
# reproduce.sh
set -e

# Create environment
conda env create -f environment.yml
conda activate llm-exp

# Sync data version
dvc pull

# Run training
python train.py seed=42 training.lr=3e-4
```

### 5.7 Ablation Study Template

Ablation study used to verify effectiveness of certain components. Recommend using unified template to record:

- baseline: complete model
- ablation_1: remove certain module
- ablation_2: replace parameter strategy

**Ablation record example**:

| Run ID | Variant | Modification | acc | loss | Notes |
| --- | --- | --- | --- | --- | --- |
| run_010 | baseline | None | 0.85 | 0.35 | Complete model |
| run_011 | no_dropout | Remove dropout | 0.83 | 0.38 | Stability decreased |
| run_012 | lr_decay | Replace LR strategy | 0.86 | 0.34 | Slight improvement |

### 5.8 Experiment Report (Repro Report) Template

Recommend generating reproducibility report for key experiments:

```markdown
# Repro Report

## Basic Information
- run_id: run_001
- git_commit: abc123
- data_version: v1.2.0
- seed: 42
- hardware: A100-80G

## Configuration Summary
- model: llama2-7b
- lr: 3e-4
- batch_size: 16
- epochs: 3

## Results
- acc: 0.85
- loss: 0.35

## Reproduction Notes
- Use reproduce.sh
- Depends on environment environment.yml
```

### 5.9 Automated Comparison Pipeline

Automate experiment comparison through simple scripts:

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

### 5.10 Experimental Comparison Tool Stack

Common comparison tools and uses:

| Tool | Role | Description |
| --- | --- | --- |
| MLflow | Tracking + Comparison | Compare metrics, configurations, artifacts |
| WandB | Tracking + Dashboard | Run comparison and visualization |
| DVC | Data difference | Compare data versions and pipelines |
| Git | Code difference | Associate commits and PRs |

Recommend exporting comparison results into unified report for review and sharing.

### 5.11 Regression Testing and Baseline Management

Regression testing ensures performance doesn't degrade. Key steps:

- Select baseline run as comparison object
- Define metric thresholds and statistical rules
- Auto-block merge or release on failure

**Baseline configuration example**:

```yaml
# baseline.yaml
baseline_run: run_001
metric_thresholds:
  acc: -0.002
  loss: 0.005
```

### 5.12 Result Audit and Signature

For important models, recommend result signing and auditing:

- Result digest hash
- Training process log hash
- Model weight hash

**Signing example**:

```bash
sha256sum metrics.json > metrics.sha
sha256sum model.pt > model.sha
```

### 5.13 Prompt Regression Testing

Conversational model behavior can change significantly due to prompt variations. Recommend establishing prompt regression test set:

- Fix representative prompts
- Automated evaluation for each model update
- Record output differences and manual sampling

**Prompt regression configuration example**:

```yaml
# prompt_regression.yaml
prompts:
  - id: p1
    text: "Summarize following text"
  - id: p2
    text: "Explain algorithm in simple terms"
```

### 5.14 LLM Evaluation Metric Consistency

LLM evaluation commonly used metrics include BLEU, ROUGE, MMLU, TruthfulQA, etc. Need to ensure:

- Metric implementation version fixed
- Evaluation set version fixed
- Scoring scripts reproducible

Recommend including scoring scripts and metric configuration files in version control together.

### 5.15 Comparison Report Template

Comparison report used to summarize differences and conclusions between different experiments:

```markdown
# Experiment Comparison Report

## Run Summary
- run_a: run_001
- run_b: run_002

## Configuration Differences
- lr: 3e-4 -> 1e-4
- batch_size: 16 -> 16

## Data Differences
- data_version: v1.2.0 -> v1.2.0

## Result Comparison
- acc: 0.85 -> 0.84
- loss: 0.35 -> 0.37

## Conclusion
- LR reduction leads to performance degradation
```

### 5.16 Data Drift Detection

Data drift leads to incomparable evaluation results. Recommend regularly detecting data distribution differences. Common metrics include KL divergence (Kullback-Leibler Divergence):

\[
D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
\]

**Drift detection example**:

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

### 5.17 Metric Interpretation and Reporting Standards

Metric reports should include values, statistical intervals, and number of runs, for example:

- acc=0.85 ± 0.01 (n=5)
- loss=0.35 ± 0.02 (n=5)

Clearly state statistical methods in reports to avoid misunderstanding.

---

## 6. Best Practices and Common Pitfalls

### 6.1 Best Practices

- **Experiment standardization**: All experiments must use unified entry script and configuration
- **Automatic recording**: Automatically record Git commit, configuration, data version
- **Minimize randomness**: Fix random seeds, avoid non-deterministic operators
- **Rebuildable environment**: Use Docker/Conda/Poetry to record dependencies
- **Traceable data**: Generate snapshots and hashes for each data processing
- **Unified naming**: Unified naming rules for data, models, and code
- **Failures are also data**: Record reasons and logs for failed experiments
- **Experiment registry**: Unified registration of experiments and output models

### 6.2 Common Pitfalls

- Only fixed Python random seed, ignoring GPU and third-party libraries
- Ignored dependency library minor version differences
- Data preprocessing didn't record configuration and parameters
- Missing experiment failure records (failures should also be traceable)
- Didn't record hardware information (GPU model, driver version)
- Distributed training didn't fix communication backend version
- Ignored random data augmentation seeds

### 6.3 Reproducibility Levels

Can divide reproducibility capability into three levels:

1. **Level 1: Script reproduction**
   - Running script on same machine and same environment yields same results
2. **Level 2: Environment reproduction**
   - Reproduce on different machines through environment files or containers
3. **Level 3: Cross-platform reproduction**
   - Maintain result stability or explainability across hardware and systems

LLM projects should at least achieve Level 2, key experiments strive for Level 3.

### 6.4 Common Problem Troubleshooting

- **Result drift**: Check random seeds, data versions, dependency versions
- **Sudden metric decline**: Check if data pipeline updated, training configuration changed
- **Evaluation instability**: Check prompt templates, evaluation set sampling strategies
- **GPU differences**: Check mixed precision and cuDNN settings

### 6.5 Maturity Model

| Stage | Features | Improvement Focus |
| --- | --- | --- |
| Elementary | Manual recording | Establish configuration and tracking systems |
| Intermediate | Automatic recording | Unify data and model versions |
| Advanced | Full automation pipeline | Statistical testing and regression detection |

### 6.6 LLM-Specific Points

- **Prompt versioning**: Prompt as input must be versioned and tracked
- **Decoding strategy**: Temperature and sampling strategy directly affect results
- **Alignment data**: RLHF/alignment data need clear recording
- **Model compression**: Quantization/distillation needs to record process and configuration

### 6.7 Reproduction Failure Case Analysis

**Case 1: Evaluation Drift**

Phenomenon: Same model has 1% difference in evaluation results on different machines.

Cause: Evaluation script defaults to random sampling, didn't fix seed; also prompt templates have differences on different branches.

Solution:

- Fix evaluation sampling seed
- Unify prompt version, add to Git management
- Write evaluation configuration to `eval.yaml` and record to run

**Case 2: Training Result Regression**

Phenomenon: Certain training loss is significantly higher.

Cause: Dependency package minor version update, introduced default behavior changes; also data cleaning script parameters changed without recording.

Solution:

- Fix dependency versions, use lock file
- Write data cleaning parameters to configuration and track

### 6.8 Reproduction Strategy and Process Recommendations

Reproduction process should form team standard operating procedure (SOP):

- New experiments must be registered (experiment registry)
- Each run must generate configuration snapshot
- Each experiment must be associated with Git commit
- Key experiments must generate reproduction report

After establishing SOP, can automate reproduction work into CI.

### 6.9 Reproduction Checklist (Extended Checklist)

Following checklist applies to final acceptance of key experiments:

- **Configuration consistent**: Training, evaluation, data processing configurations fully archived
- **Data consistent**: Data version and hash match, data processing scripts replayable
- **Code consistent**: Git commit matches dependency lock file
- **Model consistent**: Model weight hash matches, registered to model registry
- **Environment consistent**: Python, dependency packages, system libraries and driver versions consistent
- **Randomness consistent**: Random seeds and deterministic settings fully recorded
- **Evaluation consistent**: Prompt template and decoding parameters fixed
- **Metrics consistent**: Metric implementation version consistent, scoring scripts reproducible

For cross-team delivery, recommend executing "reproduction experiment" before delivery, verifying re-running in new environment yields consistent results.

### 6.10 Team Collaboration Standards

In team collaboration, recommend clearly defining responsibility division:

- Owner ensures experiment registration and configuration completeness
- Data owner ensures data version and processing scripts are traceable
- Model owner ensures model registration and evaluation report completeness

Through clear roles and standards, avoid "information silos" causing irreproducibility.

---

## 7. Summary

Experimentation and Reproducibility is a core capability of LLM system engineering. Through configuration management (YAML / Hydra), experiment tracking (MLflow / WandB), version control (Git / DVC / Model Registry), random seed control and environment isolation (Docker / Conda / Poetry), can significantly improve experiment credibility and team collaboration efficiency. Simultaneously, multiple experiments and statistical testing ensure conclusion robustness. In engineering, reproducibility should be taken as "default requirement," and tracking and version control integrated into development process, to avoid "irreproducible" traps in large-scale experiments.

Further recommendations:

- Include reproducibility metrics in project KPIs
- Force generating reproduction report for key experiments
- Add reproduction testing and regression detection to CI

Reproducibility is not one-time work, but continuous engineering capability construction.

When system scale expands, recommend establishing dedicated reproduction owner or platform team, continuously maintaining experiment tracking and version control infrastructure.
This can significantly reduce cross-team delivery cost.
For external releases, can also improve external credibility.
Recommend executing end-to-end reproduction before key milestones.
Ensure audit and evaluation results can be aligned.

---

## Reference Checklist (Checklist)

- [ ] Generate unique Run ID and record for each experiment
- [ ] Record Git commit and branch
- [ ] Record data version and data hash
- [ ] Fix random seed and record
- [ ] Fix environment and dependency versions
- [ ] Record evaluation metrics and model outputs
- [ ] Compare multiple experiments and perform statistical testing
