# 评估与回归测试 (Evaluation and Regression Testing)

## 目录

1. [背景 (Why Regression Testing?)](#1-背景-why-regression-testing)
2. [核心概念 (Test Suites, Benchmarks, A/B)](#2-核心概念-test-suites-benchmarks-ab)
3. [数学原理 (Statistical Significance, Confidence)](#3-数学原理-statistical-significance-confidence)
4. [代码实现 (Testing Framework)](#4-代码实现-testing-framework)
5. [实验对比 (With vs Without Testing)](#5-实验对比-with-vs-without-testing)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Regression Testing?)

### 1.1 模型回归问题

新版本模型可能：
- **性能下降**: 在某些任务上变差
- **行为改变**: 输出不一致
- **偏见增加**: 公平性恶化
- **安全漏洞**: 产生有害内容

### 1.2 回归测试的价值

- **质量保障**: 确保新版本不破坏旧功能
- **快速反馈**: 及早发现问题
- **信心建立**: 放心部署新版本
- **基准追踪**: 监控长期趋势

---

## 2. 核心概念 (Test Suites, Benchmarks, A/B)

### 2.1 测试套件

**单元测试**: 单个功能测试
**集成测试**: 端到端流程测试
**回归测试**: 与基线对比

### 2.2 评估维度

| 维度 | 测试内容 | 工具 |
|------|---------|------|
| **功能** | 输出正确性 | 断言 |
| **性能** | 延迟/吞吐 | Benchmark |
| **质量** | 准确性 | 数据集 |
| **安全** | 有害输出 | 分类器 |

### 2.3 A/B测试

线上对比新旧版本：
- 流量分割
- 指标收集
- 统计显著性检验

---

## 3. 数学原理 (Statistical Significance, Confidence)

### 3.1 假设检验

$$
H_0: \mu_{new} = \mu_{baseline}
$$

检验统计量:

$$
z = \frac{\bar{x}_{new} - \bar{x}_{baseline}}{\sqrt{\frac{s_{new}^2}{n_{new}} + \frac{s_{baseline}^2}{n_{baseline}}}}
$$

### 3.2 置信区间

$$
CI = \bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}
$$

---

## 4. 代码实现 (Testing Framework)

### 4.1 回归测试框架

```python
import json
from typing import List, Dict
import difflib

class ModelRegressionTester:
    """模型回归测试器"""
    
    def __init__(self, baseline_model, new_model, test_cases: List[Dict]):
        self.baseline = baseline_model
        self.new_model = new_model
        self.test_cases = test_cases
    
    def run_tests(self) -> Dict:
        """执行回归测试"""
        results = {
            "passed": 0,
            "failed": 0,
            "regressions": [],
            "improvements": []
        }
        
        for case in self.test_cases:
            input_data = case["input"]
            expected = case.get("expected")
            
            # 获取两个模型的输出
            baseline_output = self.baseline.generate(input_data)
            new_output = self.new_model.generate(input_data)
            
            # 比较
            similarity = self._calculate_similarity(baseline_output, new_output)
            
            # 评估
            if expected:
                # 有预期答案，检查正确性
                baseline_correct = self._check_correctness(baseline_output, expected)
                new_correct = self._check_correctness(new_output, expected)
                
                if baseline_correct and not new_correct:
                    results["regressions"].append({
                        "case": case["id"],
                        "reason": "New model wrong, baseline correct"
                    })
                    results["failed"] += 1
                elif not baseline_correct and new_correct:
                    results["improvements"].append({
                        "case": case["id"],
                        "reason": "New model correct, baseline wrong"
                    })
                    results["passed"] += 1
                else:
                    results["passed"] += 1
            else:
                # 无预期答案，检查一致性
                if similarity < 0.8:  # 差异过大
                    results["regressions"].append({
                        "case": case["id"],
                        "similarity": similarity,
                        "baseline": baseline_output[:100],
                        "new": new_output[:100]
                    })
                    results["failed"] += 1
                else:
                    results["passed"] += 1
        
        return results
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def _check_correctness(self, output: str, expected: str) -> bool:
        """检查输出是否正确"""
        return expected.lower() in output.lower()

# 使用
baseline = load_model("v1.0")
new_model = load_model("v1.1")
test_cases = json.load(open("test_suite.json"))

tester = ModelRegressionTester(baseline, new_model, test_cases)
results = tester.run_tests()

print(f"Passed: {results['passed']}/{len(test_cases)}")
print(f"Regressions: {len(results['regressions'])}")
```

### 4.2 Benchmark测试

```python
class ModelBenchmark:
    """模型基准测试"""
    
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
    
    def run_benchmark(self) -> Dict:
        """运行基准测试"""
        results = {
            "accuracy": [],
            "latency": [],
            "perplexity": []
        }
        
        for sample in self.dataset:
            # 准确率
            prediction = self.model.predict(sample["input"])
            correct = prediction == sample["label"]
            results["accuracy"].append(correct)
            
            # 延迟
            start = time.time()
            self.model.generate(sample["input"])
            latency = time.time() - start
            results["latency"].append(latency)
        
        # 汇总
        return {
            "accuracy": sum(results["accuracy"]) / len(results["accuracy"]),
            "avg_latency": sum(results["latency"]) / len(results["latency"]),
            "p95_latency": sorted(results["latency"])[int(len(results["latency"]) * 0.95)]
        }

# 使用
benchmark = ModelBenchmark(model, test_dataset)
metrics = benchmark.run_benchmark()
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

---

## 5. 实验对比 (With vs Without Testing)

### 5.1 问题发现时间

| 阶段 | 无测试 | 有测试 | 提升 |
|------|--------|--------|------|
| **发现问题** | 生产环境 | 测试阶段 | 提前10x |
| **修复成本** | 高 | 低 | -80% |
| **用户影响** | 大 | 无 | -100% |

### 5.2 部署信心

| 指标 | 无测试 | 有测试 |
|------|--------|--------|
| **部署频率** | 低 | 高 |
| **回滚率** | 15% | 2% |
| **信心指数** | 低 | 高 |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **自动化**: CI/CD集成测试
2. **全面覆盖**: 功能、性能、安全
3. **基线对比**: 与上一版本对比
4. **阈值设置**: 定义可接受的回归范围
5. **持续监控**: 生产环境持续测试

### 6.2 测试金字塔

```
       /\
      /  \  集成测试 (20%)
     /____\
    /      \  单元测试 (70%)
   /________\
  /          \  回归测试 (10%)
 /____________\
```

---

## 7. 总结

回归测试是模型质量的守门员：

1. **测试套件**: 覆盖功能、性能、安全
2. **基线对比**: 与生产版本对比
3. **统计检验**: 确保差异显著
4. **自动化**: CI/CD集成

**推荐流程**:
1. 开发阶段: 单元测试
2. 合并前: 集成测试
3. 部署前: 回归测试
4. 生产后: 持续监控

**关键指标**:
- 准确率不下降 > 2%
- 延迟不增加 > 10%
- 安全通过率 > 99%
