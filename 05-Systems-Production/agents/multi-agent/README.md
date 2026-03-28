**[English](README_EN.md) | [中文](README.md)**

# 多Agent协作 (Multi-Agent Collaboration)

## 目录

1. [背景 (Why Multi-Agent?)](#1-背景-why-multi-agent)
2. [核心概念 (Architecture, Communication)](#2-核心概念-architecture-communication)
3. [数学原理 (Task Allocation, Consensus)](#3-数学原理-task-allocation-consensus)
4. [代码实现 (Multi-Agent System)](#4-代码实现-multi-agent-system)
5. [实验对比 (Single vs Multi-Agent)](#5-实验对比-single-vs-multi-agent)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Multi-Agent?)

### 1.1 单Agent的局限

单一Agent难以应对：
- **复杂任务**: 需要多种专业技能
- **大规模问题**: 超出单个Agent处理能力
- **多角度分析**: 需要不同视角的思考
- **容错需求**: 单点失败风险

### 1.2 多Agent的优势

- **专业化**: 每个Agent专注特定领域
- **并行处理**: 多个任务同时进行
- **协作增强**: 通过讨论产生更好结果
- **容错性**: 单个Agent失败不影响整体

---

## 2. 核心概念 (Architecture, Communication)

### 2.1 多Agent架构类型

| 架构 | 描述 | 适用场景 |
|------|------|---------|
| **层次式** | 管理Agent + 执行Agent | 任务分配 |
| **对等式** | Agent平等协作 | 讨论决策 |
| **市场式** | Agent竞标任务 | 资源分配 |
| **混合式** | 组合多种架构 | 复杂系统 |

### 2.2 协作模式

- **讨论模式**: Agent相互讨论达成共识
- **辩论模式**: 正反方Agent辩论
- **审查模式**: 一个Agent生成，其他审查
- **流水线**: Agent按顺序处理不同阶段

### 2.3 通信机制

- **直接通信**: Agent间直接交换消息
- **黑板系统**: 共享工作空间
- **消息总线**: 通过中间件路由消息

---

## 3. 数学原理 (Task Allocation, Consensus)

### 3.1 任务分配

**最优分配**:

$$
\min \sum_{i,j} C_{ij} \cdot X_{ij}
$$

约束:
- 每个任务分配给一个Agent: $\sum_i X_{ij} = 1$
- Agent能力匹配: $X_{ij} \leq Cap_{ij}$

### 3.2 共识达成

**投票机制**:

$$
\text{Consensus} = \arg\max_{o} \sum_{i} w_i \cdot \mathbb{1}[A_i = o]
$$

---

## 4. 代码实现 (Multi-Agent System)

### 4.1 基础多Agent系统

```python
from typing import List, Dict, Callable
import asyncio

class Agent:
    """基础Agent类"""
    
    def __init__(self, name: str, role: str, llm_client):
        self.name = name
        self.role = role
        self.llm = llm_client
        self.memory = []
    
    def think(self, task: str, context: List[str] = None) -> str:
        """思考并生成回应"""
        prompt = f"You are {self.name}, a {self.role}.\n\n"
        if context:
            prompt += "Context:\n" + "\n".join(context) + "\n\n"
        prompt += f"Task: {task}\n\nYour response:"
        
        response = self.llm.generate(prompt)
        self.memory.append({"task": task, "response": response})
        return response

class MultiAgentSystem:
    """多Agent协作系统"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.message_bus: List[Dict] = []
    
    def register_agent(self, agent: Agent):
        """注册Agent"""
        self.agents[agent.name] = agent
    
    def broadcast(self, message: str, sender: str):
        """广播消息"""
        self.message_bus.append({"sender": sender, "message": message})
    
    def run_discussion(self, topic: str, rounds: int = 3) -> str:
        """讨论模式"""
        context = [f"Topic: {topic}"]
        
        for round in range(rounds):
            for name, agent in self.agents.items():
                response = agent.think(
                    f"Round {round + 1}: Discuss the topic",
                    context
                )
                self.broadcast(response, name)
                context.append(f"{name}: {response}")
        
        # 总结
        summarizer = self.agents.get("summarizer") or list(self.agents.values())[0]
        return summarizer.think("Summarize the discussion", context)
    
    def run_pipeline(self, task: str, agent_sequence: List[str]) -> str:
        """流水线模式"""
        result = task
        
        for agent_name in agent_sequence:
            agent = self.agents[agent_name]
            result = agent.think(f"Process: {result}")
        
        return result

# 模拟LLM
class MockLLM:
    def generate(self, prompt):
        if "analyst" in prompt.lower():
            return "From analysis perspective: key points identified"
        elif "critic" in prompt.lower():
            return "Critical view: potential risks noted"
        return "General response"

# 创建系统
system = MultiAgentSystem()

# 注册Agent
analyst = Agent("Alice", "analyst", MockLLM())
critic = Agent("Bob", "critic", MockLLM())
summarizer = Agent("Carol", "summarizer", MockLLM())

system.register_agent(analyst)
system.register_agent(critic)
system.register_agent(summarizer)

# 运行讨论
result = system.run_discussion("Should we invest in AI startups?", rounds=2)
print(result)
```

### 4.2 辩论模式

```python
class DebateSystem:
    """辩论系统"""
    
    def __init__(self, pro_agent: Agent, con_agent: Agent, judge_agent: Agent):
        self.pro = pro_agent
        self.con = con_agent
        self.judge = judge_agent
    
    def debate(self, topic: str, rounds: int = 3) -> Dict:
        """进行辩论"""
        pro_args = []
        con_args = []
        
        for i in range(rounds):
            # 正方发言
            pro_arg = self.pro.think(
                f"Round {i+1}: Argue FOR the topic",
                con_args if con_args else None
            )
            pro_args.append(pro_arg)
            
            # 反方发言
            con_arg = self.con.think(
                f"Round {i+1}: Argue AGAINST the topic",
                pro_args
            )
            con_args.append(con_arg)
        
        # 裁判判决
        context = [
            "Pro arguments:",
            "\n".join(pro_args),
            "Con arguments:",
            "\n".join(con_args)
        ]
        
        verdict = self.judge.think(
            "Based on the debate, provide a verdict and reasoning",
            context
        )
        
        return {
            "pro_arguments": pro_args,
            "con_arguments": con_args,
            "verdict": verdict
        }

# 使用
debate = DebateSystem(analyst, critic, summarizer)
result = debate.debate("Remote work is better than office work")
print(result["verdict"])
```

---

## 5. 实验对比 (Single vs Multi-Agent)

### 5.1 性能对比

| 任务类型 | 单Agent | 多Agent | 提升 |
|---------|---------|---------|------|
| 代码审查 | 72% | 88% | +16% |
| 创意生成 | 65% | 82% | +17% |
| 决策分析 | 70% | 85% | +15% |
| 复杂问题 | 60% | 80% | +20% |

### 5.2 成本对比

| 指标 | 单Agent | 多Agent (3个) |
|------|---------|--------------|
| Token消耗 | 1x | 3-5x |
| 延迟 | 1x | 2-4x |
| 成本 | 1x | 3-5x |

**权衡**: 质量提升 vs 成本增加

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **角色明确**: 每个Agent有清晰的角色定义
2. **通信协议**: 定义标准消息格式
3. **冲突解决**: 设定仲裁机制
4. **超时控制**: 防止Agent无响应阻塞
5. **状态同步**: 保持共享状态一致

### 6.2 常见陷阱

1. **角色重叠**: Agent职责不清晰
2. **通信风暴**: 消息过多导致混乱
3. **死锁**: Agent相互等待
4. **单点故障**: 关键Agent失败影响整体

### 6.3 设计检查清单

```markdown
- [ ] Agent角色定义清晰
- [ ] 通信机制标准化
- [ ] 任务分配策略
- [ ] 冲突解决机制
- [ ] 错误处理与容错
- [ ] 性能监控
- [ ] 成本控制
```

---

## 7. 总结

多Agent协作是处理复杂任务的有效方式：

1. **架构选择**: 层次式、对等式、混合式
2. **协作模式**: 讨论、辩论、审查、流水线
3. **核心挑战**: 通信、协调、共识
4. **成本权衡**: 质量提升 vs 成本增加

**适用场景**:
- 需要多角度分析
- 复杂任务分解
- 代码审查和质量保证
- 创意生成和头脑风暴

**关键成功因素**:
- 清晰的角色定义
- 有效的通信机制
- 合理的任务分配
- 完善的错误处理
