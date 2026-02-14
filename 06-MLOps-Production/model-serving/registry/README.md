[English](README_EN.md) | [中文](README.md)
# 模型注册与版本管理 (Model Registry and Versioning)

## 目录

1. [背景 (Why Model Registry?)](#1-背景-why-model-registry)
2. [核心概念 (Versioning, Staging, Lifecycle)](#2-核心概念-versioning-staging-lifecycle)
3. [数学原理 (Model Cards, Metrics)](#3-数学原理-model-cards-metrics)
4. [代码实现 (MLflow Integration)](#4-代码实现-mlflow-integration)
5. [实验对比 (With vs Without Registry)](#5-实验对比-with-vs-without-registry)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Model Registry?)

### 1.1 模型管理的挑战

- **版本混乱**: 多个实验版本难以追踪
- **部署风险**: 不确定哪个版本最适合生产
- **回滚困难**: 新版本出问题无法快速回退
- **协作问题**: 团队成员难以找到正确模型

### 1.2 模型注册表的价值

- **版本控制**: 清晰的版本历史
- **阶段管理**: 开发→测试→生产
- **血缘追踪**: 记录数据来源和参数
- **协作**: 团队共享模型资产

---

## 2. 核心概念 (Versioning, Staging, Lifecycle)

### 2.1 模型版本

```
v1.0.0 → v1.1.0 (改进) → v2.0.0 (重大更新)
```

### 2.2 模型阶段

| 阶段 | 说明 | 使用场景 |
|------|------|---------|
| **None** | 未注册 | 实验阶段 |
| **Staging** | 测试中 | 验证阶段 |
| **Production** | 生产中 | 线上服务 |
| **Archived** | 已归档 | 历史版本 |

### 2.3 模型卡片 (Model Card)

记录模型元数据：
- 训练数据
- 模型架构
- 性能指标
- 限制和偏见

---

## 3. 数学原理 (Model Cards, Metrics)

### 3.1 模型版本化

语义化版本: MAJOR.MINOR.PATCH

### 3.2 指标追踪

$$
\text{Model Score} = \sum_{i} w_i \cdot \text{metric}_i
$$

---

## 4. 代码实现 (MLflow Integration)

### 4.1 MLflow模型注册

```python
import mlflow
from mlflow.tracking import MlflowClient

# 设置跟踪URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my_experiment")

# 训练并记录
with mlflow.start_run() as run:
    # 记录参数
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("batch_size", 32)
    
    # 训练
    model = train_model()
    
    # 评估
    metrics = evaluate(model)
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("f1", metrics["f1"])
    
    # 保存模型
    mlflow.pytorch.log_model(model, "model")
    
    run_id = run.info.run_id

# 注册模型
client = MlflowClient()
model_name = "my_model"

# 创建模型 (如果不存在)
try:
    client.create_registered_model(model_name)
except:
    pass

# 注册版本
result = client.create_model_version(
    name=model_name,
    source=f"runs:/{run_id}/model",
    run_id=run_id
)

version = result.version
print(f"Model registered as version {version}")

# 转移到Staging
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Staging"
)

# 设置标签
client.set_model_version_tag(
    name=model_name,
    version=version,
    key="validation_status",
    value="passed"
)
```

### 4.2 加载特定版本

```python
import mlflow.pyfunc

# 加载生产版本
model = mlflow.pyfunc.load_model("models:/my_model/Production")

# 或加载特定版本
model = mlflow.pyfunc.load_model("models:/my_model/3")

# 预测
predictions = model.predict(input_data)
```

### 4.3 模型比较

```python
def compare_model_versions(model_name, versions):
    """比较不同版本的模型"""
    client = MlflowClient()
    
    results = []
    for v in versions:
        version = client.get_model_version(model_name, v)
        run = client.get_run(version.run_id)
        
        results.append({
            "version": v,
            "accuracy": run.data.metrics.get("accuracy"),
            "f1": run.data.metrics.get("f1"),
            "stage": version.current_stage
        })
    
    return pd.DataFrame(results)

# 比较
comparison = compare_model_versions("my_model", [1, 2, 3])
print(comparison)
```

---

## 5. 实验对比 (With vs Without Registry)

### 5.1 效率对比

| 场景 | 无注册表 | 有注册表 | 提升 |
|------|---------|---------|------|
| **找到正确模型** | 30分钟 | 1分钟 | 30x |
| **部署到生产** | 2小时 | 10分钟 | 12x |
| **回滚版本** | 4小时 | 5分钟 | 48x |
| **团队协作** | 混乱 | 有序 | - |

### 5.2 风险降低

| 风险 | 无注册表 | 有注册表 |
|------|---------|---------|
| **部署错误版本** | 高 | 低 |
| **无法回滚** | 经常 | 不会 |
| **模型丢失** | 可能 | 不会 |
| **合规审计** | 困难 | 容易 |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **命名规范**: 模型名称清晰有意义
2. **版本标签**: 添加有意义的标签
3. **模型卡片**: 记录完整的模型信息
4. **自动过渡**: 通过CI/CD自动推进阶段
5. **定期清理**: 归档旧版本

### 6.2 工作流程

```
实验 → 注册 → Staging → 验证 → Production → 监控 → (回滚或升级)
```

---

## 7. 总结

模型注册是MLOps的核心组件：

1. **版本管理**: 清晰的版本历史
2. **阶段控制**: 开发→测试→生产
3. **血缘追踪**: 数据来源和参数
4. **协作**: 团队共享资产

**推荐工具**:
- MLflow (开源)
- Weights & Biases
- DVC
- 自研 (大型组织)

**关键实践**:
- 所有生产模型必须注册
- 使用语义化版本
- 维护模型卡片
- 自动化部署流程
