# Serving Architecture

**[English](README_EN.md) | [中文](README.md)**

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

### 1.1 Serving Challenges

- **High Concurrency**: Large number of users requesting simultaneously
- **Low Latency**: High user experience requirements
- **Cost**: GPU resources are expensive
- **Stability**: 99.9%+ availability requirements

### 1.2 Key Metrics

| Metric | Target | Description |
|--------|---------|-------------|
| **P50 Latency** | <500ms | 50% request response time |
| **P99 Latency** | <2000ms | 99% request response time |
| **Throughput** | >100 QPS | Queries per second |
| **Availability** | 99.9% | Yearly downtime <8.7 hours |

---

## 2. Core Concepts

### 2.1 Dynamic Batching

Merge multiple requests for batch processing:
- Increase GPU utilization
- Reduce average latency
- Trade-off with wait time

### 2.2 Caching Strategies

- **Prompt Cache**: Cache identical prefixes
- **KV Cache**: Cache attention Key-Value
- **Result Cache**: Cache completely identical requests

### 2.3 Scaling Strategies

- **Horizontal Scaling**: Increase number of instances
- **Vertical Scaling**: Upgrade single instance
- **Auto-scaling**: Adjust based on load

---

## 3. Mathematical Principles

### 3.1 Latency Decomposition

$$
\text{Total Latency} = T_{\text{queue}} + T_{\text{prefill}} + T_{\text{decode}} + T_{\text{network}}
$$

### 3.2 Little's Law

$$
L = \lambda \cdot W
$$

Where:
- $L$: Average number of requests in system
- $\lambda$: Arrival rate
- $W$: Average wait time

---

## 4. Code Implementation

### 4.1 Batching Service

```python
import asyncio
from typing import List, Dict
import time

class BatchInferenceServer:
    """Batch inference service"""

    def __init__(self, model, max_batch_size=8, max_wait_time=0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = asyncio.Queue()
        self.running = False

    async def start(self):
        """Start batching loop"""
        self.running = True
        while self.running:
            batch = await self._collect_batch()
            if batch:
                await self._process_batch(batch)

    async def _collect_batch(self) -> List[Dict]:
        """Collect batch"""
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
        """Process batch"""
        # Merge inputs
        inputs = [r["input"] for r in batch]

        # Batch inference
        results = self.model.generate_batch(inputs)

        # Distribute results
        for request, result in zip(batch, results):
            request["future"].set_result(result)

    async def predict(self, input_data: str) -> str:
        """Prediction interface"""
        future = asyncio.Future()
        await self.request_queue.put({
            "input": input_data,
            "future": future
        })
        return await future

# Usage
server = BatchInferenceServer(model)
asyncio.create_task(server.start())

# Request
result = await server.predict("Hello, how are you?")
```

### 4.2 vLLM Integration

```python
from vllm import LLM, SamplingParams

# Initialize vLLM
llm = LLM(
    model="meta-llama/Llama-2-7b",
    tensor_parallel_size=2,  # 2-card parallelism
    gpu_memory_utilization=0.9
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# Batch generation
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

## 5. Experimental Comparison

### 5.1 Batching Effects

| Batch Size | Throughput | P50 Latency | P99 Latency |
|-----------|-------------|--------------|--------------|
| 1 | 20 QPS | 50ms | 100ms |
| 4 | 60 QPS | 80ms | 150ms |
| 8 | 100 QPS | 120ms | 200ms |
| 16 | 150 QPS | 180ms | 350ms |

### 5.2 Architecture Comparison

| Architecture | Throughput | Latency | Complexity |
|-------------|-------------|-----------|------------|
| **Single Instance** | Low | Low | Low |
| **Batching** | High | Medium | Medium |
| **Multi-instance** | High | Low | High |
| **vLLM** | Very High | Low | Medium |

---

## 6. Best Practices and Common Pitfalls

### 6.1 Best Practices

1. **Dynamic batching**: Adjust batch size based on load
2. **KV Cache**: Enable PagedAttention
3. **Preloading**: Load hot models in advance
4. **Rate limiting**: Prevent overload
5. **Monitoring**: Track latency and errors in real time

### 6.2 Service Architecture Diagram

```
User Request → Load Balancer → [Inference Instance 1]
                           [Inference Instance 2]
                           [Inference Instance N]

Each instance:
Request Queue → Batching → Model Inference → Response
                    ↓
                KV Cache Manager
```

---

## 7. Summary

Serving architecture is key to model deployment:

1. **Batching**: Increase throughput
2. **Caching**: Reduce latency
3. **Scaling**: Support high concurrency
4. **Monitoring**: Ensure stability

**Recommended Solutions**:
- Small scale: vLLM single instance
- Medium scale: vLLM + Load Balancer
- Large scale: Kubernetes + vLLM + Auto-scaling

**Key Configuration**:
- Batch size: 8-16
- KV Cache: Enabled
- Quantization: INT8 (cost-sensitive)
