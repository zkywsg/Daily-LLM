[English](README.md) | [中文](README_CN.md)
# 成本优化与容量规划 (Cost Optimization and Capacity Planning)

## 目录

1. [背景 (Cost Challenges)](#1-背景-cost-challenges)
2. [核心概念 (Cost Drivers, Optimization Strategies)](#2-核心-concepts-cost-drivers-optimization-strategies)
3. [数学原理 (Cost Models, ROI)](#3-数学原理-cost-models-roi)
4. [代码实现 (Cost Tracking)](#4-代码实现-cost-tracking)
5. [实验对比 (Before vs After)](#5-实验对比-before-vs-after)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Cost Challenges)

### 1.1 LLM成本构成

| 成本项 | 占比 | 说明 |
|--------|------|------|
| **GPU实例** | 60% | 推理和训练 |
| **存储** | 15% | 模型和数据 |
| **网络** | 10% | 数据传输 |
| **API调用** | 15% | 外部模型 |

### 1.2 成本驱动因素

- **模型大小**: 大模型需要更多资源
- **请求量**: 流量决定实例数
- **延迟要求**: 低延迟需要高性能硬件
- **可用性要求**: 高可用需要冗余

---

## 2. 核心概念 (Cost Drivers, Optimization Strategies)

### 2.1 优化策略

| 策略 | 节省 | 影响 |
|------|------|------|
| **量化** | 50-75% | 轻微质量下降 |
| **缓存** | 30-50% | 命中率依赖 |
| **批处理** | 20-40% | 延迟增加 |
| **Spot实例** | 60-90% | 可用性风险 |
| **混合部署** | 30-50% | 架构复杂 |

### 2.2 容量规划

**垂直扩展**: 升级单机配置
**水平扩展**: 增加实例数量
**自动扩缩**: 按需调整

---

## 3. 数学原理 (Cost Models, ROI)

### 3.1 单查询成本

$$
C_{query} = C_{compute} + C_{storage} + C_{network} + C_{api}
$$

### 3.2 总成本

$$
C_{total} = N_{queries} \times C_{query} + C_{fixed}
$$

### 3.3 成本效益

$$
ROI = \frac{\text{Revenue} - \text{Cost}}{\text{Cost}} \times 100\%
$$

---

## 4. 代码实现 (Cost Tracking)

### 4.1 成本追踪器

```python
import time

class CostTracker:
    """成本追踪器"""
    
    # 定价 (示例)
    PRICING = {
        "gpu_a100": 2.5,  # $/小时
        "gpu_v100": 1.2,
        "gpu_t4": 0.35,
        "api_gpt4": 0.03,  # $/1K tokens
        "api_gpt35": 0.002,
        "storage": 0.1  # $/GB/月
    }
    
    def __init__(self):
        self.usage = {
            "gpu_hours": {},
            "api_tokens": {},
            "storage_gb": 0
        }
    
    def log_gpu_usage(self, gpu_type: str, hours: float):
        """记录GPU使用"""
        if gpu_type not in self.usage["gpu_hours"]:
            self.usage["gpu_hours"][gpu_type] = 0
        self.usage["gpu_hours"][gpu_type] += hours
    
    def log_api_usage(self, api_type: str, tokens: int):
        """记录API使用"""
        if api_type not in self.usage["api_tokens"]:
            self.usage["api_tokens"][api_type] = 0
        self.usage["api_tokens"][api_type] += tokens
    
    def calculate_cost(self) -> Dict:
        """计算成本"""
        cost = {
            "gpu": 0,
            "api": 0,
            "storage": 0,
            "total": 0
        }
        
        # GPU成本
        for gpu_type, hours in self.usage["gpu_hours"].items():
            rate = self.PRICING.get(gpu_type, 1.0)
            cost["gpu"] += hours * rate
        
        # API成本
        for api_type, tokens in self.usage["api_tokens"].items():
            rate = self.PRICING.get(api_type, 0.01)
            cost["api"] += (tokens / 1000) * rate
        
        # 存储成本
        cost["storage"] = self.usage["storage_gb"] * self.PRICING["storage"]
        
        cost["total"] = cost["gpu"] + cost["api"] + cost["storage"]
        
        return cost

# 使用
tracker = CostTracker()
tracker.log_gpu_usage("gpu_a100", 24)  # 24小时A100
tracker.log_api_usage("api_gpt4", 50000)  # 50K tokens

cost = tracker.calculate_cost()
print(f"总成本: ${cost['total']:.2f}")
print(f"  GPU: ${cost['gpu']:.2f}")
print(f"  API: ${cost['api']:.2f}")
```

### 4.2 成本优化建议

```python
def generate_cost_report(tracker: CostTracker) -> str:
    """生成成本报告和优化建议"""
    cost = tracker.calculate_cost()
    
    report = f"""
成本分析报告
============
总成本: ${cost['total']:.2f}

成本构成:
- GPU: ${cost['gpu']:.2f} ({cost['gpu']/cost['total']*100:.1f}%)
- API: ${cost['api']:.2f} ({cost['api']/cost['total']*100:.1f}%)
- 存储: ${cost['storage']:.2f} ({cost['storage']/cost['total']*100:.1f}%)

优化建议:
"""
    
    # GPU优化建议
    if cost['gpu'] / cost['total'] > 0.5:
        report += """
1. GPU成本占比过高 (>50%):
   - 考虑使用INT8量化，可节省50% GPU成本
   - 评估是否可使用 cheaper GPU (T4 vs A100)
   - 启用动态批处理提高利用率
   - 使用Spot实例节省60-90%成本
"""
    
    # API优化建议
    if cost['api'] / cost['total'] > 0.3:
        report += """
2. API成本占比过高 (>30%):
   - 缓存常用查询，减少API调用
   - 评估使用开源模型替代
   - 批处理API请求
"""
    
    return report

# 使用
report = generate_cost_report(tracker)
print(report)
```

---

## 5. 实验对比 (Before vs After)

### 5.1 优化效果

| 优化项 | 优化前 | 优化后 | 节省 |
|--------|--------|--------|------|
| **量化INT8** | $100/天 | $50/天 | 50% |
| **启用缓存** | $80/天 | $48/天 | 40% |
| **批处理** | $100/天 | $70/天 | 30% |
| **Spot实例** | $100/天 | $30/天 | 70% |

### 5.2 组合优化

| 策略组合 | 成本 | 质量影响 |
|---------|------|---------|
| 无优化 | $100 | 100% |
| INT8 + 缓存 | $35 | 97% |
| INT8 + 缓存 + Spot | $20 | 97% |
| 蒸馏 + INT4 | $15 | 92% |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **成本预算**: 设定月度/季度预算
2. **实时监控**: 每日检查成本趋势
3. **混合策略**: 不同场景用不同策略
4. **容量预测**: 提前规划扩容
5. **定期评估**: 每月评估成本效益

### 6.2 容量规划公式

$$
\text{Instances} = \frac{\text{QPS} \times \text{Latency}}{\text{Batch Size} \times \text{Utilization Target}}
$$

---

## 7. 总结

成本控制是LLM应用可持续发展的关键：

1. **量化压缩**: 降低计算成本
2. **缓存策略**: 减少重复计算
3. **批处理**: 提高吞吐量
4. **容量规划**: 避免过度配置

**成本优化检查清单**:
- [ ] 启用INT8量化
- [ ] 配置查询缓存
- [ ] 使用动态批处理
- [ ] 评估Spot实例
- [ ] 监控每日成本
- [ ] 设置预算告警

**目标**: 在保证质量的前提下，成本降低50-70%。
