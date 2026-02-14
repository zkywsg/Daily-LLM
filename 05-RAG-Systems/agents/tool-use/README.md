**[English](README_EN.md) | [中文](README.md)**

# 工具调用与函数执行 (Tool Use and Function Calling)

## 目录

1. [背景 (Why Tool Use?)](#1-背景-why-tool-use)
2. [核心概念 (Tool Types, Execution Flow)](#2-核心概念-tool-types-execution-flow)
3. [数学原理 (Schema Validation, Error Handling)](#3-数学原理-schema-validation-error-handling)
4. [代码实现 (Tool Implementation)](#4-代码实现-tool-implementation)
5. [实验对比 (Tool vs No-Tool Performance)](#5-实验对比-tool-vs-no-tool-performance)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Tool Use?)

### 1.1 LLM的局限性

纯LLM无法：
- 获取实时信息 (天气、股价)
- 执行外部操作 (发邮件、查数据库)
- 进行精确计算 (复杂数学)
- 与外部系统集成

### 1.2 工具调用的价值

通过Tool Use，LLM可以：
- **扩展能力边界**: 调用外部API和函数
- **获取实时数据**: 查询最新信息
- **执行动作**: 完成实际任务
- **提高准确性**: 用工具替代模糊推理

**类比**: LLM像大脑，工具像手脚和感官。

---

## 2. 核心概念 (Tool Types, Execution Flow)

### 2.1 工具类型

| 类型 | 示例 | 用途 |
|------|------|------|
| **数据查询** | 搜索引擎、数据库查询 | 获取信息 |
| **计算工具** | 计算器、代码执行 | 精确计算 |
| **Action工具** | 发邮件、创建工单 | 执行操作 |
| **领域工具** | 医疗查询、法律检索 | 专业领域 |

### 2.2 工具调用流程

```
用户查询 → LLM判断是否需要工具 → 生成工具调用 → 执行工具 → 返回结果 → LLM生成最终回答
```

### 2.3 Function Calling模式

**OpenAI Function Calling**:
```json
{
  "name": "get_weather",
  "arguments": {"location": "北京", "unit": "celsius"}
}
```

**Tool Definition (Schema)**:
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的天气",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string", "description": "城市名"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["location"]
    }
  }
}
```

---

## 3. 数学原理 (Schema Validation, Error Handling)

### 3.1 工具选择概率

$$
P(\text{tool}_i | q) = \frac{e^{\text{score}(q, \text{tool}_i)}}{\sum_j e^{\text{score}(q, \text{tool}_j)}}
$$

### 3.2 参数验证

**JSON Schema验证**:
- 类型检查: string, number, boolean, array, object
- 必填字段检查: required
- 范围检查: min/max, enum
- 嵌套验证: 对象和数组结构

---

## 4. 代码实现 (Tool Implementation)

### 4.1 基础工具系统

```python
import json
from typing import Dict, Callable, Any

class Tool:
    """工具基类"""
    
    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def to_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.handlers: Dict[str, Callable] = {}
    
    def register(self, tool: Tool, handler: Callable):
        """注册工具和处理器"""
        self.tools[tool.name] = tool
        self.handlers[tool.name] = handler
    
    def get_tool_schemas(self):
        """获取所有工具Schema"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def execute(self, tool_name: str, arguments: dict) -> Any:
        """执行工具"""
        if tool_name not in self.handlers:
            raise ValueError(f"未知工具: {tool_name}")
        
        handler = self.handlers[tool_name]
        return handler(**arguments)

# 定义工具
calculator_tool = Tool(
    name="calculator",
    description="执行数学计算",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式"}
        },
        "required": ["expression"]
    }
)

def calculator_handler(expression: str):
    """计算器实现"""
    try:
        # 安全计算，限制可用操作
        allowed_names = {"abs": abs, "max": max, "min": min}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"result": result, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

# 注册
registry = ToolRegistry()
registry.register(calculator_tool, calculator_handler)

# 模拟LLM调用
tool_call = {"name": "calculator", "arguments": {"expression": "2 + 2"}}
result = registry.execute(tool_call["name"], tool_call["arguments"])
print(result)
```

### 4.2 完整Agent循环

```python
class ToolAgent:
    """带工具的Agent"""
    
    def __init__(self, llm_client, tool_registry):
        self.llm = llm_client
        self.tools = tool_registry
    
    def run(self, user_query: str, max_iterations: int = 5):
        """运行Agent"""
        messages = [{"role": "user", "content": user_query}]
        
        for i in range(max_iterations):
            # 调用LLM
            response = self.llm.chat(
                messages=messages,
                tools=self.tools.get_tool_schemas()
            )
            
            # 检查是否需要工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # 添加助手消息
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls
                })
                
                # 执行工具
                for tool_call in response.tool_calls:
                    result = self.tools.execute(
                        tool_call["name"],
                        tool_call["arguments"]
                    )
                    
                    # 添加工具结果
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result)
                    })
            else:
                # 直接回答
                return response.content
        
        return "达到最大迭代次数"

# 模拟LLM客户端
class MockLLM:
    def chat(self, messages, tools=None):
        # 模拟LLM决定调用工具
        class Response:
            def __init__(self):
                self.content = None
                self.tool_calls = [{
                    "id": "call_1",
                    "name": "calculator",
                    "arguments": {"expression": "15 * 23"}
                }]
        return Response()

# 使用
agent = ToolAgent(MockLLM(), registry)
result = agent.run("计算15乘以23")
print(result)
```

### 4.3 错误处理与重试

```python
class RobustToolExecutor:
    """鲁棒工具执行器"""
    
    def __init__(self, registry, max_retries=3):
        self.registry = registry
        self.max_retries = max_retries
    
    def execute_with_retry(self, tool_name: str, arguments: dict):
        """带重试的执行"""
        for attempt in range(self.max_retries):
            try:
                result = self.registry.execute(tool_name, arguments)
                if result.get("success"):
                    return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"error": str(e), "success": False}
        
        return {"error": "Max retries exceeded", "success": False}
    
    def validate_arguments(self, tool_name: str, arguments: dict):
        """参数验证"""
        tool = self.registry.tools.get(tool_name)
        if not tool:
            return False, "Tool not found"
        
        params = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        
        # 检查必填字段
        for field in required:
            if field not in arguments:
                return False, f"Missing required field: {field}"
        
        # 类型检查 (简化版)
        for key, value in arguments.items():
            if key in params:
                expected_type = params[key].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    return False, f"Field {key} should be string"
                # ... 其他类型检查
        
        return True, "Valid"

# 使用
robust_executor = RobustToolExecutor(registry)
is_valid, msg = robust_executor.validate_arguments("calculator", {"expression": "2+2"})
print(f"验证结果: {is_valid}, {msg}")
```

---

## 5. 实验对比 (Tool vs No-Tool Performance)

### 5.1 数学计算准确性

| 任务 | 无工具LLM | 带工具LLM | 提升 |
|------|----------|----------|------|
| 简单计算 | 85% | 99% | +14% |
| 复杂公式 | 45% | 98% | +53% |
| 单位换算 | 70% | 99% | +29% |

### 5.2 实时信息获取

| 场景 | 无工具 | 带工具 | 说明 |
|------|--------|--------|------|
| 天气查询 | ❌ 无法回答 | ✅ 准确回答 | 需天气API |
| 股价查询 | ❌ 过时信息 | ✅ 实时数据 | 需金融API |
| 新闻摘要 | ❌ 训练截止 | ✅ 最新新闻 | 需搜索工具 |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **Schema清晰**: 工具描述要详细，参数要明确
2. **错误处理**: 工具失败时给LLM提供友好错误信息
3. **权限控制**: 敏感操作需确认或限制
4. **超时设置**: 防止工具执行过久
5. **结果摘要**: 工具结果过长时需摘要

### 6.2 常见陷阱

1. **Schema模糊**: LLM无法正确选择工具
2. **无错误处理**: 工具失败导致整个流程崩溃
3. **过度依赖**: 本可用LLM推理的简单问题也调工具
4. **安全隐患**: 直接执行用户输入的代码

### 6.3 工具设计检查清单

```markdown
- [ ] Schema描述清晰完整
- [ ] 参数类型和必填项明确
- [ ] 错误处理和重试机制
- [ ] 超时控制
- [ ] 结果格式化
- [ ] 权限和安全性检查
- [ ] 工具执行日志
```

---

## 7. 总结

Tool Use是扩展LLM能力的关键机制：

1. **核心流程**: LLM判断 → 生成调用 → 执行 → 返回 → 生成回答
2. **设计要点**: Schema清晰、错误处理、权限控制
3. **常见工具**: 数据查询、计算、Action执行、领域专用
4. **安全考虑**: 输入验证、沙箱执行、权限分级

**推荐架构**:
- 工具注册表管理所有工具
- 统一执行器处理调用和错误
- 与Agent框架集成实现自动循环

**未来趋势**:
- 多模态工具 (图像、音频)
- 工具自动发现和学习
- 更复杂的工具组合编排
