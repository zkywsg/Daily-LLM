[English](README.md) | [中文](README_CN.md)
# CI/CD for ML (MLOps CI/CD)

## 目录

1. [背景 (Why CI/CD for ML?)](#1-背景-why-cicd-for-ml)
2. [核心概念 (Pipelines, Automation, Testing)](#2-核心-concepts-pipelines-automation-testing)
3. [数学原理 (Deployment Frequency, Lead Time)](#3-数学原理-deployment-frequency-lead-time)
4. [代码实现 (GitHub Actions Pipeline)](#4-代码实现-github-actions-pipeline)
5. [实验对比 (Manual vs Automated)](#5-实验对比-manual-vs-automated)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why CI/CD for ML?)

### 1.1 ML部署的挑战

传统软件 vs ML系统：
- **数据依赖**: 模型依赖数据版本
- **实验追踪**: 需要记录实验参数
- **模型验证**: 需要特殊测试
- **环境复杂**: GPU、依赖库等

### 1.2 CI/CD价值

- **自动化**: 减少人工操作
- **一致性**: 确保环境一致
- **快速迭代**: 加速模型上线
- **质量保证**: 自动测试保障

---

## 2. 核心概念 (Pipelines, Automation, Testing)

### 2.1 ML流水线阶段

```
代码提交 → 数据验证 → 训练 → 评估 → 注册 → 部署 → 监控
```

### 2.2 持续集成 (CI)

- 代码提交触发
- 自动测试
- 构建镜像

### 2.3 持续部署 (CD)

- 自动部署到Staging
- 人工审批生产
- 蓝绿/金丝雀发布

---

## 3. 数学原理 (Deployment Frequency, Lead Time)

### 3.1 DORA指标

- **部署频率**: 多久部署一次
- **变更前置时间**: 从提交到部署
- **恢复时间**: 失败恢复时间
- **变更失败率**: 导致失败的变更比例

### 3.2 效率提升

$$
\text{Time Saved} = \text{Manual Time} - \text{Automated Time}
$$

---

## 4. 代码实现 (GitHub Actions Pipeline)

### 4.1 ML训练流水线

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Validate data
        run: |
          python scripts/validate_data.py --data-path data/
      
      - name: Check data drift
        run: |
          python scripts/check_drift.py --baseline data/baseline --current data/current

  train:
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Train model
        run: |
          python train.py \
            --epochs 10 \
            --batch-size 32 \
            --output models/
      
      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: models/

  evaluate:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Download model
        uses: actions/download-artifact@v3
        with:
          name: trained-model
          path: models/
      
      - name: Evaluate model
        run: |
          python evaluate.py \
            --model models/model.pkl \
            --test-data data/test.csv \
            --output results/
      
      - name: Compare with baseline
        run: |
          python scripts/compare_models.py \
            --baseline models/baseline.pkl \
            --new models/model.pkl \
            --threshold 0.05
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-results
          path: results/

  register-model:
    needs: evaluate
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Register to MLflow
        run: |
          python scripts/register_model.py \
            --run-id ${{ github.run_id }} \
            --model-name "production-model"
```

### 4.2 部署流水线

```yaml
# .github/workflows/deploy.yml
name: Deploy Model

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to deploy'
        required: true
      environment:
        description: 'Environment (staging/prod)'
        required: true
        default: 'staging'

jobs:
  deploy-staging:
    if: github.event.inputs.environment == 'staging'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Staging
        run: |
          python scripts/deploy.py \
            --version ${{ github.event.inputs.model_version }} \
            --env staging \
            --endpoint https://staging.api.example.com
      
      - name: Run smoke tests
        run: |
          python tests/smoke_test.py --env staging

  deploy-production:
    if: github.event.inputs.environment == 'prod'
    runs-on: ubuntu-latest
    environment: production  # 需要人工审批
    steps:
      - name: Deploy to Production
        run: |
          python scripts/deploy.py \
            --version ${{ github.event.inputs.model_version }} \
            --env prod \
            --endpoint https://api.example.com
      
      - name: Run smoke tests
        run: |
          python tests/smoke_test.py --env prod
      
      - name: Notify team
        uses: slack/notify-action@v1
        with:
          message: "Model ${{ github.event.inputs.model_version }} deployed to production"
```

### 4.3 Python部署脚本

```python
# scripts/deploy.py
import argparse
import mlflow
from mlflow.tracking import MlflowClient

def deploy_model(version: str, env: str, endpoint: str):
    """部署模型"""
    client = MlflowClient()
    
    # 获取模型
    model_uri = f"models:/production-model/{version}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # 部署到目标环境 (简化示例)
    if env == "staging":
        deploy_to_k8s(model, namespace="staging", endpoint=endpoint)
    else:
        deploy_to_k8s(model, namespace="production", endpoint=endpoint)
    
    # 更新模型阶段
    client.transition_model_version_stage(
        name="production-model",
        version=version,
        stage="Production" if env == "prod" else "Staging"
    )
    
    print(f"✓ Model {version} deployed to {env}")

def deploy_to_k8s(model, namespace: str, endpoint: str):
    """部署到Kubernetes (示例)"""
    # 实际实现使用Kubernetes API或Helm
    print(f"Deploying to {namespace} at {endpoint}")
    # ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--env", required=True, choices=["staging", "prod"])
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    
    deploy_model(args.version, args.env, args.endpoint)
```

---

## 5. 实验对比 (Manual vs Automated)

### 5.1 效率对比

| 阶段 | 手动 | 自动化 | 提升 |
|------|------|--------|------|
| **数据验证** | 30分钟 | 5分钟 | 6x |
| **模型训练** | 4小时 | 4小时 | - |
| **评估测试** | 1小时 | 15分钟 | 4x |
| **部署上线** | 2小时 | 10分钟 | 12x |
| **总计** | 7.5小时 | 4.5小时 | 1.7x |

### 5.2 质量对比

| 指标 | 手动 | 自动化 |
|------|------|--------|
| **部署频率** | 1/周 | 5/周 |
| **失败率** | 15% | 3% |
| **恢复时间** | 4小时 | 15分钟 |
| **一致性** | 低 | 高 |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **版本控制**: 代码、数据、模型版本管理
2. **环境隔离**: 开发/测试/生产环境分离
3. **自动化测试**: 单元测试、集成测试、冒烟测试
4. **渐进部署**: 蓝绿部署、金丝雀发布
5. **回滚机制**: 快速回滚到上一版本
6. **监控告警**: 部署后实时监控

### 6.2 CI/CD流水线图

```
代码提交
    ↓
数据验证 → 失败: 告警
    ↓
模型训练
    ↓
自动评估 → 失败: 停止
    ↓
注册模型
    ↓
部署Staging → 失败: 停止
    ↓
人工审批 (生产)
    ↓
部署Production
    ↓
监控验证 → 异常: 自动回滚
    ↓
完成
```

---

## 7. 总结

CI/CD是MLOps的核心，实现自动化、可靠的模型交付：

1. **自动化**: 减少人工，加快速度
2. **质量保证**: 自动测试保障
3. **可追溯**: 完整的版本历史
4. **快速回滚**: 问题及时恢复

**推荐工具链**:
- GitHub Actions / GitLab CI
- MLflow / Weights & Biases
- Docker / Kubernetes
- Prometheus / Grafana

**关键指标**:
- 部署频率: 每日多次
- 前置时间: < 1小时
- 恢复时间: < 15分钟
- 失败率: < 5%
