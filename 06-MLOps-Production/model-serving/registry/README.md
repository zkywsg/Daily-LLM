# Model Registry and Versioning

**[English](README.md) | [中文](README_CN.md)**

## Table of Contents

1. [Background](#1-background)
2. [Core Concepts](#2-core-concepts)
3. [Mathematical Principles](#3-mathematical-principles)
4. [Code Implementation](#4-code-implementation)
5. [Experimental Comparison](#5-experimental-comparison)
6. [Best Practices and Common Pitfalls](#6-best-practices-and-common-pitfalls)
7. [Summary](#7-summary)

---

## 1. Background

### 1.1 Model Management Challenges

- **Version Chaos**: Multiple experiment versions are hard to track
- **Deployment Risk**: Uncertain which version is most suitable for production
- **Rollback Difficulty**: Cannot quickly roll back when new version has problems
- **Collaboration Issues**: Team members have trouble finding correct models

### 1.2 Value of Model Registry

- **Version Control**: Clear version history
- **Stage Management**: Development → Testing → Production
- **Lineage Tracking**: Record data sources and parameters
- **Collaboration**: Team shares model assets

---

## 2. Core Concepts

### 2.1 Model Versions

```
v1.0.0 → v1.1.0 (improvement) → v2.0.0 (major update)
```

### 2.2 Model Stages

| Stage | Description | Use Case |
|-------|-------------|----------|
| **None** | Not registered | Experiment phase |
| **Staging** | Testing | Validation phase |
| **Production** | In production | Online serving |
| **Archived** | Archived | Historical versions |

### 2.3 Model Cards

Record model metadata:
- Training data
- Model architecture
- Performance metrics
- Limitations and biases

---

## 3. Mathematical Principles

### 3.1 Model Versioning

Semantic versioning: MAJOR.MINOR.PATCH

### 3.2 Metrics Tracking

$$
\text{Model Score} = \sum_{i} w_i \cdot \text{metric}_i
$$

---

## 4. Code Implementation

### 4.1 MLflow Model Registration

```python
import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my_experiment")

# Train and log
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("batch_size", 32)

    # Train
    model = train_model()

    # Evaluate
    metrics = evaluate(model)
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("f1", metrics["f1"])

    # Save model
    mlflow.pytorch.log_model(model, "model")

    run_id = run.info.run_id

# Register model
client = MlflowClient()
model_name = "my_model"

# Create model (if not exists)
try:
    client.create_registered_model(model_name)
except:
    pass

# Register version
result = client.create_model_version(
    name=model_name,
    source=f"runs:/{run_id}/model",
    run_id=run_id
)

version = result.version
print(f"Model registered as version {version}")

# Transition to Staging
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Staging"
)

# Set tags
client.set_model_version_tag(
    name=model_name,
    version=version,
    key="validation_status",
    value="passed"
)
```

### 4.2 Load Specific Version

```python
import mlflow.pyfunc

# Load production version
model = mlflow.pyfunc.load_model("models:/my_model/Production")

# Or load specific version
model = mlflow.pyfunc.load_model("models:/my_model/3")

# Predict
predictions = model.predict(input_data)
```

### 4.3 Model Comparison

```python
def compare_model_versions(model_name, versions):
    """Compare different model versions"""
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

# Compare
comparison = compare_model_versions("my_model", [1, 2, 3])
print(comparison)
```

---

## 5. Experimental Comparison

### 5.1 Efficiency Comparison

| Scenario | Without Registry | With Registry | Improvement |
|----------|------------------|---------------|-------------|
| **Find correct model** | 30 minutes | 1 minute | 30x |
| **Deploy to production** | 2 hours | 10 minutes | 12x |
| **Rollback version** | 4 hours | 5 minutes | 48x |
| **Team collaboration** | Chaotic | Orderly | - |

### 5.2 Risk Reduction

| Risk | Without Registry | With Registry |
|------|------------------|---------------|
| **Deploy wrong version** | High | Low |
| **Cannot rollback** | Frequent | Won't happen |
| **Model lost** | Possible | Won't happen |
| **Compliance audit** | Difficult | Easy |

---

## 6. Best Practices and Common Pitfalls

### 6.1 Best Practices

1. **Naming conventions**: Clear and meaningful model names
2. **Version tags**: Add meaningful tags
3. **Model cards**: Record complete model information
4. **Automatic transition**: Automatically advance stages through CI/CD
5. **Regular cleanup**: Archive old versions

### 6.2 Workflow

```
Experiment → Register → Staging → Validate → Production → Monitor → (Rollback or Upgrade)
```

---

## 7. Summary

Model registry is a core component of MLOps:

1. **Version management**: Clear version history
2. **Stage control**: Development → Testing → Production
3. **Lineage tracking**: Data sources and parameters
4. **Collaboration**: Team shares assets

**Recommended Tools**:
- MLflow (open source)
- Weights & Biases
- DVC
- Self-developed (large organizations)

**Key Practices**:
- All production models must be registered
- Use semantic versioning
- Maintain model cards
- Automate deployment pipeline
