[English](README.md) | [中文](README_CN.md)
# 监控与漂移检测 (Monitoring and Drift Detection)

## 目录

1. [背景 (Why Monitor?)](#1-背景-why-monitor)
2. [核心概念 (Metrics, Drift Types, Alerts)](#2-核心概念-metrics-drift-types-alerts)
3. [数学原理 (Drift Detection, Statistical Tests)](#3-数学原理-drift-detection-statistical-tests)
4. [代码实现 (Monitoring System)](#4-代码实现-monitoring-system)
5. [实验对比 (Detection Methods)](#5-实验对比-detection-methods)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Monitor?)

### 1.1 为什么需要监控？

生产环境模型面临：
- **数据漂移**: 输入分布变化
- **概念漂移**: 输入输出关系变化
- **性能下降**: 准确率随时间降低
- **异常行为**: 模型输出异常

### 1.2 监控的价值

- **及时发现问题**: 在影响用户前发现
- **根因分析**: 快速定位问题来源
- **预防性维护**: 主动干预
- **合规审计**: 满足监管要求

---

## 2. 核心概念 (Metrics, Drift Types, Alerts)

### 2.1 监控指标

| 类型 | 指标 | 说明 |
|------|------|------|
| **系统** | 延迟、错误率 | 服务健康 |
| **数据** | 输入分布 | 数据漂移 |
| **模型** | 预测分布 | 输出漂移 |
| **业务** | 用户满意度 | 业务影响 |

### 2.2 漂移类型

- **数据漂移 (Data Drift)**: P(X)变化
- **概念漂移 (Concept Drift)**: P(Y|X)变化
- **标签漂移 (Label Drift)**: P(Y)变化

### 2.3 告警级别

- **P0 (Critical)**: 立即处理
- **P1 (High)**: 4小时内处理
- **P2 (Medium)**: 24小时内处理
- **P3 (Low)**: 跟踪观察

---

## 3. 数学原理 (Drift Detection, Statistical Tests)

### 3.1 KL散度

$$
D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
$$

### 3.2 KS检验

$$
D_{n,m} = \sup_{x} |F_{1,n}(x) - F_{2,m}(x)|
$$

### 3.3 PSI (Population Stability Index)

$$
PSI = \sum_{i} (A_i - E_i) \times \ln(\frac{A_i}{E_i})
$$

- PSI < 0.1: 无漂移
- 0.1 ≤ PSI < 0.25: 轻微漂移
- PSI ≥ 0.25: 显著漂移

---

## 4. 代码实现 (Monitoring System)

### 4.1 漂移检测器

```python
import numpy as np
from scipy import stats

class DriftDetector:
    """漂移检测器"""
    
    def __init__(self, reference_data: np.ndarray):
        self.reference = reference_data
        self.reference_dist = self._estimate_distribution(reference_data)
    
    def detect_ks(self, current_data: np.ndarray, threshold: float = 0.05) -> Dict:
        """KS检验检测漂移"""
        statistic, p_value = stats.ks_2samp(self.reference, current_data)
        
        return {
            "drift_detected": p_value < threshold,
            "statistic": statistic,
            "p_value": p_value,
            "method": "KS"
        }
    
    def detect_psi(self, current_data: np.ndarray, buckets: int = 10) -> Dict:
        """PSI检测漂移"""
        # 分桶
        min_val = min(self.reference.min(), current_data.min())
        max_val = max(self.reference.max(), current_data.max())
        bins = np.linspace(min_val, max_val, buckets + 1)
        
        # 计算分布
        ref_counts, _ = np.histogram(self.reference, bins=bins)
        cur_counts, _ = np.histogram(current_data, bins=bins)
        
        # 转换为比例
        ref_pct = ref_counts / len(self.reference)
        cur_pct = cur_counts / len(current_data)
        
        # 计算PSI
        psi = 0
        for i in range(buckets):
            if ref_pct[i] > 0 and cur_pct[i] > 0:
                psi += (cur_pct[i] - ref_pct[i]) * np.log(cur_pct[i] / ref_pct[i])
        
        return {
            "drift_detected": psi > 0.25,
            "psi": psi,
            "severity": "high" if psi > 0.25 else "medium" if psi > 0.1 else "low",
            "method": "PSI"
        }
    
    def detect_kl(self, current_data: np.ndarray) -> Dict:
        """KL散度检测"""
        # 估计当前分布
        current_dist = self._estimate_distribution(current_data)
        
        # 计算KL散度
        kl_div = 0
        for key in self.reference_dist:
            p = self.reference_dist[key]
            q = current_dist.get(key, 1e-10)
            kl_div += p * np.log(p / q)
        
        return {
            "drift_detected": kl_div > 0.1,
            "kl_divergence": kl_div,
            "method": "KL"
        }

# 使用
detector = DriftDetector(reference_data=np.random.normal(0, 1, 1000))

# 检测正常数据
result_normal = detector.detect_ks(np.random.normal(0, 1, 500))
print(f"Normal: {result_normal}")

# 检测漂移数据
result_drift = detector.detect_ks(np.random.normal(2, 1.5, 500))
print(f"Drift: {result_drift}")
```

### 4.2 监控系统

```python
class ModelMonitor:
    """模型监控系统"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics = []
        self.alerts = []
    
    def log_prediction(self, input_data: Dict, output: str, latency: float):
        """记录预测"""
        self.metrics.append({
            "timestamp": time.time(),
            "input_length": len(str(input_data)),
            "output_length": len(output),
            "latency": latency
        })
    
    def check_alerts(self) -> List[Dict]:
        """检查告警"""
        alerts = []
        
        # 延迟告警
        recent_latencies = [m["latency"] for m in self.metrics[-100:]]
        if recent_latencies:
            p95 = np.percentile(recent_latencies, 95)
            if p95 > 2.0:  # 2秒阈值
                alerts.append({
                    "level": "P1",
                    "metric": "latency_p95",
                    "value": p95,
                    "message": f"P95 latency {p95:.2f}s exceeds threshold"
                })
        
        # 错误率告警 (模拟)
        # ...
        
        return alerts

# 使用
monitor = ModelMonitor("llm-v1")
monitor.log_prediction({"query": "hello"}, "hi there", 0.5)
alerts = monitor.check_alerts()
```

---

## 5. 实验对比 (Detection Methods)

### 5.1 检测方法对比

| 方法 | 灵敏度 | 计算成本 | 适用场景 |
|------|--------|---------|---------|
| **KS检验** | 中 | 低 | 连续变量 |
| **PSI** | 高 | 中 | 评分模型 |
| **KL散度** | 高 | 中 | 分布对比 |
| **Wasserstein** | 高 | 高 | 分布距离 |

### 5.2 检测延迟

| 窗口大小 | 检测延迟 | 误报率 |
|---------|---------|--------|
| 100 | 5分钟 | 高 |
| 1000 | 30分钟 | 中 |
| 10000 | 3小时 | 低 |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **多维度监控**: 数据、模型、业务指标
2. **适当窗口**: 窗口太小误报高，太大延迟高
3. **基线选择**: 使用稳定时期的分布作为基线
4. **告警分级**: 不同严重程度不同处理
5. **自动响应**: 检测到漂移自动触发重训练

### 6.2 监控仪表板

```
┌─────────────────────────────────────────────┐
│ 模型监控仪表板 - llm-v1                      │
├─────────────────────────────────────────────┤
│ 延迟: P50=200ms, P95=800ms, P99=1500ms      │
│ 错误率: 0.5%                                │
│ QPS: 120                                    │
├─────────────────────────────────────────────┤
│ 数据漂移: PSI=0.05 (正常)                   │
│ 概念漂移: 未检测到                          │
├─────────────────────────────────────────────┤
│ 告警:                                       │
│ [P2] 输入长度均值增加20%                    │
└─────────────────────────────────────────────┘
```

---

## 7. 总结

监控和漂移检测是生产模型运维的关键：

1. **漂移检测**: KS、PSI、KL方法
2. **实时监控**: 延迟、错误率、QPS
3. **自动告警**: 分级响应
4. **预防维护**: 主动干预

**推荐实践**:
- 每日检查PSI
- 实时监控P95延迟
- 每周基线对比
- 月度模型重评估
