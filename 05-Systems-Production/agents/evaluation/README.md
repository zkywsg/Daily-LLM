**[English](README_EN.md) | [中文](README.md)**

# Agent评估 (Agent Evaluation)

## 目录

1. [背景 (Why Agent Evaluation?)](#1-背景-why-agent-evaluation)
2. [核心概念 (Metrics, Benchmarks)](#2-核心概念-metrics-benchmarks)
3. [数学原理 (Success Rate, Trajectory Scoring)](#3-数学原理-success-rate-trajectory-scoring)
4. [代码实现 (Evaluation Framework)](#4-代码实现-evaluation-framework)
5. [实验对比 (Evaluation Methods)](#5-实验对比-evaluation-methods)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Agent Evaluation?)

### 1.1 评估挑战

Agent比简单LLM更难评估：
- **多步推理**: 中间步骤影响最终结果
- **工具调用**: 需要验证工具使用正确性
- **自主性**: 路径不固定，难以对比
- **长时运行**: 评估成本高

### 1.2 评估维度

| 维度 | 说明 | 指标 |
|------|------|------|
| **任务完成** | 是否达成目标 | 成功率 |
| **效率** | 完成任务的速度 | 步数、时间 |
| **正确性** | 答案的准确性 | 准确率 |
| **成本** | Token和API消耗 | 费用 |
| **安全性** | 是否有害操作 | 安全分数 |

---

## 2. 核心概念 (Metrics, Benchmarks)

### 2.1 评估指标

#### 2.1.1 任务成功率

$$
\text{Success Rate} = \frac{\text{成功任务数}}{\text{总任务数}}
$$

#### 2.1.2 轨迹效率

$$
\text{Efficiency} = \frac{\text{最优步数}}{\text{实际步数}}
$$

#### 2.1.3 工具使用正确率

$$
\text{Tool Accuracy} = \frac{\text{正确工具调用}}{\text{总工具调用}}
$$

### 2.2 基准测试

| 基准 | 类型 | 任务数 | 特点 |
|------|------|--------|------|
| **WebArena** | 网页操作 | 812 | 真实网站 |
| **AgentBench** | 多环境 | 1000+ | 代码/OS/DB |
| **ToolBench** | 工具使用 | 16000+ | API调用 |
| **SWE-bench** | 代码修复 | 2000+ | GitHub Issues |

---

## 3. 数学原理 (Success Rate, Trajectory Scoring)

### 3.1 轨迹评分

$$
\text{Score}(\tau) = \sum_{t=1}^{T} \gamma^{t-1} R(s_t, a_t)
$$

其中:
- $\tau$: 轨迹
- $R$: 每步奖励
- $\gamma$: 折扣因子

### 3.2 人类对齐评分

$$
\text{Human Score} = \frac{1}{N} \sum_{i=1}^{N} \text{Rating}_i
$$

---

## 4. 代码实现 (Evaluation Framework)

### 4.1 基础评估框架

```python
from typing import List, Dict, Any
import json

class AgentEvaluator:
    """Agent评估器"""
    
    def __init__(self):
        self.results = []
    
    def evaluate_task(self, task: Dict, agent_response: Any, trajectory: List[Dict]) -> Dict:
        """评估单个任务"""
        result = {
            "task_id": task["id"],
            "success": False,
            "metrics": {}
        }
        
        # 1. 检查任务完成
        result["success"] = self._check_success(task, agent_response)
        
        # 2. 计算步数效率
        result["metrics"]["steps"] = len(trajectory)
        result["metrics"]["optimal_steps"] = task.get("optimal_steps", len(trajectory))
        result["metrics"]["efficiency"] = result["metrics"]["optimal_steps"] / max(result["metrics"]["steps"], 1)
        
        # 3. 工具使用分析
        tool_calls = [step for step in trajectory if step.get("type") == "tool_call"]
        correct_tools = sum(1 for t in tool_calls if self._is_correct_tool(t, task))
        result["metrics"]["tool_accuracy"] = correct_tools / max(len(tool_calls), 1)
        
        # 4. 成本计算
        result["metrics"]["token_count"] = sum(step.get("tokens", 0) for step in trajectory)
        
        self.results.append(result)
        return result
    
    def _check_success(self, task: Dict, response: Any) -> bool:
        """检查任务是否成功"""
        # 根据任务类型检查
        if task["type"] == "exact_match":
            return response == task["expected"]
        elif task["type"] == "contains":
            return task["expected"] in str(response)
        elif task["type"] == "custom":
            # 使用自定义验证函数
            return task["validator"](response)
        return False
    
    def _is_correct_tool(self, tool_call: Dict, task: Dict) -> bool:
        """检查工具调用是否正确"""
        expected_tools = task.get("expected_tools", [])
        return tool_call.get("tool") in expected_tools
    
    def get_summary(self) -> Dict:
        """获取评估摘要"""
        if not self.results:
            return {}
        
        total = len(self.results)
        success_count = sum(1 for r in self.results if r["success"])
        
        return {
            "total_tasks": total,
            "success_count": success_count,
            "success_rate": success_count / total,
            "avg_steps": sum(r["metrics"]["steps"] for r in self.results) / total,
            "avg_efficiency": sum(r["metrics"]["efficiency"] for r in self.results) / total,
            "avg_tool_accuracy": sum(r["metrics"]["tool_accuracy"] for r in self.results) / total,
            "total_tokens": sum(r["metrics"]["token_count"] for r in self.results)
        }

# 使用示例
evaluator = AgentEvaluator()

# 模拟任务
task = {
    "id": "task_1",
    "type": "exact_match",
    "expected": "42",
    "optimal_steps": 3,
    "expected_tools": ["calculator"]
}

# 模拟Agent响应
response = "42"
trajectory = [
    {"type": "thought", "content": "I need to calculate"},
    {"type": "tool_call", "tool": "calculator", "tokens": 50},
    {"type": "answer", "content": "42", "tokens": 10}
]

result = evaluator.evaluate_task(task, response, trajectory)
summary = evaluator.get_summary()
print(f"成功率: {summary['success_rate']:.2%}")
```

### 4.2 轨迹可视化

```python
import matplotlib.pyplot as plt

class TrajectoryVisualizer:
    """轨迹可视化"""
    
    def plot_trajectory(self, trajectory: List[Dict], title: str = "Agent Trajectory"):
        """绘制Agent执行轨迹"""
        steps = range(len(trajectory))
        types = [step.get("type", "unknown") for step in trajectory]
        
        # 颜色映射
        color_map = {
            "thought": "blue",
            "tool_call": "green",
            "observation": "orange",
            "answer": "red"
        }
        
        colors = [color_map.get(t, "gray") for t in types]
        
        plt.figure(figsize=(12, 4))
        plt.bar(steps, [1]*len(steps), color=colors, width=0.8)
        plt.xlabel("Step")
        plt.title(title)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[k], label=k) for k in color_map]
        plt.legend(handles=legend_elements, loc="upper right")
        
        plt.savefig("trajectory.png")
        plt.close()
```

---

## 5. 实验对比 (Evaluation Methods)

### 5.1 自动评估 vs 人工评估

| 方法 | 成本 | 速度 | 一致性 | 准确性 |
|------|------|------|--------|--------|
| **自动** | 低 | 快 | 高 | 中 |
| **人工** | 高 | 慢 | 中 | 高 |
| **LLM Judge** | 中 | 中 | 中 | 高 |

### 5.2 评估指标对比

| 指标 | 易测量 | 有意义 | 适用场景 |
|------|--------|--------|---------|
| **成功率** | ✅ | ✅ | 所有任务 |
| **步数效率** | ✅ | ✅ | 多步任务 |
| **工具准确率** | ✅ | ✅ | 工具使用 |
| **人类评分** | ❌ | ✅ | 主观质量 |
| **成本** | ✅ | ✅ | 生产环境 |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **多维度评估**: 不只关注成功率
2. **基准对比**: 与基线Agent对比
3. **A/B测试**: 线上对比不同版本
4. **错误分析**: 分析失败案例
5. **持续监控**: 生产环境持续评估

### 6.2 常见陷阱

1. **过度拟合**: 针对特定基准优化
2. **指标虚荣**: 追求表面指标忽视质量
3. **忽视成本**: 只看成功率不看成本
4. **静态评估**: 不随数据更新

### 6.3 评估检查清单

```markdown
- [ ] 任务成功率
- [ ] 步数效率
- [ ] 工具使用正确性
- [ ] 成本分析
- [ ] 安全性检查
- [ ] 人工抽样验证
- [ ] 与基线对比
- [ ] 错误案例分析
```

---

## 7. 总结

Agent评估是确保系统质量的关键：

1. **核心指标**: 成功率、效率、成本、安全
2. **评估方法**: 自动、人工、LLM-as-Judge
3. **基准测试**: WebArena、AgentBench、ToolBench
4. **持续改进**: 基于评估结果优化

**关键原则**:
- 多维度综合评估
- 成本与效果平衡
- 持续监控迭代
- 错误驱动改进

**推荐流程**:
1. 离线基准测试
2. 人工抽样验证
3. A/B测试上线
4. 生产监控告警
