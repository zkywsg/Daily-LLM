[English](README_EN.md) | [中文](README.md)
# 推理服务架构 (Serving Architecture)

## 目录

1. [背景 (Serving Challenges)](#1-背景-serving-challenges)
2. [核心概念 (Batching, Caching, Scaling)](#2-核心-concepts-batching-caching-scaling)
3. [数学原理 (Latency, Throughput, Queuing)](#3-数学原理-latency-throughput-queuing)
4. [代码实现 (Serving System)](#4-代码实现-serving-system)
5. [实验对比 (Architecture Comparison)](#5-实验对比-architecture-comparison)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Serving Challenges)

### 1.1 推理服务的挑战

- **高并发**: 大量用户同时请求
- **低延迟**: 用户体验要求高
- **成本**: GPU资源昂贵
- **稳定性**: 99.9%+可用性要求

### 1.2 关键指标

| 指标 | 目标 | 说明 |
|------|------|------|
| **P50延迟** | <500ms | 50%请求响应时间 |
| **P99延迟** | <2000ms | 99%请求响应时间 |
| **吞吐量** | >100 QPS | 每秒查询数 |
| **可用性** | 99.9% | 年停机<8.7小时 |

---

## 2. 核心概念 (Batching, Caching, Scaling)

### 2.1 动态批处理 (Dynamic Batching)

将多个请求合并批量处理：
- 提高GPU利用率
- 降低平均延迟
- 需要等待时间权衡

### 2.2 缓存策略

- **Prompt Cache**: 缓存相同前缀
- **KV Cache**: 缓存注意力Key-Value
- **Result Cache**: 缓存完全相同的请求

### 2.3 扩展策略

- **水平扩展**: 增加实例数
- **垂直扩展**: 升级单实例
- **自动扩缩**: 根据负载调整

---

## 3. 数学原理 (Latency, Throughput, Queuing)

### 3.1 延迟分解

$$
\text{Total Latency} = T_{\text{queue}} + T_{\text{prefill}} + T_{\text{decode}} + T_{\text{network}}
$$

### 3.2 Little定律

$$
L = \lambda \cdot W
$$

其中:
- $L$: 系统中平均请求数
- $\lambda$: 到达率
- $W$: 平均等待时间

---

## 4. 代码实现 (Serving System)

### 4.1 批处理服务

```python
import asyncio
from typing import List, Dict
import time

class BatchInferenceServer:
    """批处理推理服务"""
    
    def __init__(self, model, max_batch_size=8, max_wait_time=0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = asyncio.Queue()
        self.running = False
    
    async def start(self):
        """启动批处理循环"""
        self.running = True
        while self.running:
            batch = await self._collect_batch()
            if batch:
                await self._process_batch(batch)
    
    async def _collect_batch(self) -> List[Dict]:
        """收集批次"""
        batch = []
        start_time = time.time()
        
        while len(batch) < self.max_batch_size:
            remaining_time = self.max_wait_time - (time.time() - start_time)
            
            if remaining_time <= 0 and batch:
                break
            
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=max(remaining_time, 0.001)
                )
                batch.append(request)
            except asyncio.TimeoutError:
                if batch:
                    break
        
        return batch
    
    async def _process_batch(self, batch: List[Dict]):
        """处理批次"""
        # 合并输入
        inputs = [r["input"] for r in batch]
        
        # 批处理推理
        results = self.model.generate_batch(inputs)
        
        # 分发结果
        for request, result in zip(batch, results):
            request["future"].set_result(result)
    
    async def predict(self, input_data: str) -> str:
        """预测接口"""
        future = asyncio.Future()
        await self.request_queue.put({
            "input": input_data,
            "future": future
        })
        return await future

# 使用
server = BatchInferenceServer(model)
asyncio.create_task(server.start())

# 请求
result = await server.predict("Hello, how are you?")
```

### 4.2 vLLM集成

```python
from vllm import LLM, SamplingParams

# 初始化vLLM
llm = LLM(
    model="meta-llama/Llama-2-7b",
    tensor_parallel_size=2,  # 2卡并行
    gpu_memory_utilization=0.9
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# 批处理生成
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is"
]

outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output.outputs[0].text}")
```

---

## 5. 实验对比 (Architecture Comparison)

### 5.1 批处理效果

| 批大小 | 吞吐量 | P50延迟 | P99延迟 |
|--------|--------|---------|---------|
| 1 | 20 QPS | 50ms | 100ms |
| 4 | 60 QPS | 80ms | 150ms |
| 8 | 100 QPS | 120ms | 200ms |
| 16 | 150 QPS | 180ms | 350ms |

### 5.2 架构对比

| 架构 | 吞吐量 | 延迟 | 复杂度 |
|------|--------|------|--------|
| **单实例** | 低 | 低 | 低 |
| **批处理** | 高 | 中 | 中 |
| **多实例** | 高 | 低 | 高 |
| **vLLM** | 很高 | 低 | 中 |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **动态批处理**: 根据负载调整批大小
2. **KV Cache**: 启用PagedAttention
3. **预加载**: 提前加载热点模型
4. **限流**: 防止过载
5. **监控**: 实时跟踪延迟和错误

### 6.2 服务架构图

```
用户请求 → Load Balancer → [Inference Instance 1]
                           [Inference Instance 2]
                           [Inference Instance N]
                           
每个实例:
Request Queue → Batching → Model Inference → Response
                    ↓
               KV Cache Manager
```

---

## 7. 总结

推理服务架构是模型上线的关键：

1. **批处理**: 提高吞吐量
2. **缓存**: 降低延迟
3. **扩展**: 支持高并发
4. **监控**: 确保稳定性

**推荐方案**:
- 小规模: vLLM单实例
- 中规模: vLLM + Load Balancer
- 大规模: Kubernetes + vLLM + Auto-scaling

**关键配置**:
- 批大小: 8-16
- KV Cache: 启用
- 量化: INT8 (成本敏感)
