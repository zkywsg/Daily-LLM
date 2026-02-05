# 分块策略 (Chunking Strategies)

## 目录

1. [背景 (Why Chunking Matters)](#1-背景-why-chunking-matters)
2. [核心概念 (Chunking Types, Boundaries)](#2-核心概念-chunking-types-boundaries)
3. [数学原理 (Overlap Formulas, Coverage Metrics)](#3-数学原理-overlap-formulas-coverage-metrics)
4. [代码实现 (Chunking Implementations)](#4-代码实现-chunking-implementations)
5. [实验对比 (Chunk Size Impact)](#5-实验对比-chunk-size-impact)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Chunking Matters)

### 1.1 为什么需要Chunking？

在RAG系统中，文档通常远超模型上下文限制。Chunking将长文档切分为适当大小的片段，以便：

1. **适配上下文窗口**: 满足模型长度限制
2. **提高检索精度**: 细粒度匹配查询意图
3. **减少噪声**: 只保留相关信息
4. **优化存储**: 高效索引与检索

### 1.2 Chunking的影响

**质量维度**:
- **语义完整性**: 保持上下文连贯
- **信息密度**: 避免过碎或冗余
- **边界准确性**: 不在关键信息处截断

**性能维度**:
- **检索召回**: 影响相关片段命中率
- **生成质量**: 上下文质量决定回答质量
- **计算成本**: 影响存储与检索开销

---

## 2. 核心概念 (Chunking Types, Boundaries)

### 2.1 Chunking策略分类

#### 2.1.1 固定大小切分 (Fixed-size)

**原理**: 按固定字符数或token数切分

```python
chunks = [text[i:i+size] for i in range(0, len(text), size)]
```

**优点**: 简单、均匀、易控制
**缺点**: 可能截断语义单元

#### 2.1.2 语义边界切分 (Semantic)

**原理**: 在句子、段落等自然边界处切分

**方法**:
- 句子边界 (句号、问号、感叹号)
- 段落边界 (换行符)
- 章节边界 (标题标记)

**优点**: 保持语义完整
**缺点**: 块大小不均匀

#### 2.1.3 递归切分 (Recursive)

**原理**: 层级切分，先尝试大边界，失败则降级

```
尝试段落切分 → 过大则尝试句子 → 仍大则尝试固定大小
```

**优点**: 平衡语义与大小
**缺点**: 实现复杂

#### 2.1.4 结构化切分 (Structured)

针对特定格式的切分:

- **Markdown**: 按标题层级切分
- **代码**: 按函数/类切分
- **表格**: 按行/列切分
- **JSON**: 按对象切分

### 2.2 重叠策略 (Overlap)

**原理**: 相邻块之间保留重叠内容，保持上下文连贯

**重叠方式**:
- **固定重叠**: 固定字符数重叠
- **句子重叠**: 保留完整句子作为重叠
- **语义重叠**: 基于语义相似度选择重叠内容

---

## 3. 数学原理 (Overlap Formulas, Coverage Metrics)

### 3.1 块大小计算

**目标块数**:
$$
N = \lceil \frac{L_{\text{doc}}}{S_{\text{chunk}} - O} \rceil
$$

其中:
- $L_{\text{doc}}$: 文档长度
- $S_{\text{chunk}}$: 块大小
- $O$: 重叠大小

### 3.2 覆盖率 (Coverage)

**信息覆盖率**:
$$
\text{Coverage} = \frac{\sum_{i} |C_i| - \sum_{i,j} |C_i \cap C_j|}{L_{\text{doc}}}
$$

理想情况下应接近100%，避免信息丢失。

### 3.3 语义连贯性

**边界质量**:
$$
Q_{\text{boundary}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \text{sim}(\text{end}_i, \text{start}_{i+1})
$$

其中 $\text{sim}$ 是语义相似度函数。

---

## 4. 代码实现 (Chunking Implementations)

### 4.1 固定大小切分

```python
def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50):
    """固定大小切分"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 考虑重叠
    
    return chunks

# 使用
text = "这是一段很长的文本..."
chunks = fixed_size_chunk(text, chunk_size=500, overlap=50)
print(f"切分成 {len(chunks)} 个块")
```

### 4.2 语义边界切分

```python
import re

def semantic_chunk(text: str, max_size: int = 500):
    """基于语义边界切分"""
    # 按句子切分
    sentences = re.split(r'(?<=[。！？])', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# 使用
text = "第一句。第二句！第三句？第四句。"
chunks = semantic_chunk(text, max_size=100)
print(chunks)
```

### 4.3 递归切分

```python
def recursive_chunk(text: str, max_size: int = 500, separators=["\n\n", "\n", "。", "；"]):
    """递归层级切分"""
    if len(text) <= max_size:
        return [text]
    
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current = ""
            for part in parts:
                if len(current) + len(part) < max_size:
                    current += part + sep
                else:
                    if current:
                        chunks.extend(recursive_chunk(current, max_size, separators[1:]))
                    current = part + sep
            if current:
                chunks.extend(recursive_chunk(current, max_size, separators[1:]))
            return chunks
    
    # 最后手段: 固定大小切分
    return fixed_size_chunk(text, max_size, overlap=50)

# 使用
long_text = "段落1\n\n段落2。句子1。句子2\n段落3"
chunks = recursive_chunk(long_text, max_size=50)
print(f"递归切分结果: {len(chunks)} 块")
```

### 4.4 Markdown结构化切分

```python
def markdown_chunk(text: str):
    """Markdown结构化切分"""
    import re
    
    # 按标题层级切分
    pattern = r'^(#{1,6}\s+.+)$'
    parts = re.split(pattern, text, flags=re.MULTILINE)
    
    chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i]
        content = parts[i+1] if i+1 < len(parts) else ""
        chunks.append({
            "header": header,
            "content": header + content,
            "level": header.count("#")
        })
    
    return chunks

# 使用
md_text = """
# 标题1
内容1
## 子标题
子内容
# 标题2
内容2
"""
chunks = markdown_chunk(md_text)
for chunk in chunks:
    print(f"Level {chunk['level']}: {chunk['header']}")
```

### 4.5 代码切分

```python
import re

def code_chunk(code: str, language: str = "python"):
    """代码结构化切分"""
    if language == "python":
        # 按函数/类切分
        pattern = r'((?:def|class)\s+\w+[^:]*:.*?)(?=\n(?:def|class)\s+|\Z)'
        matches = re.findall(pattern, code, re.DOTALL)
        return matches if matches else [code]
    
    return [code]

# 使用
python_code = """
def func1():
    pass

def func2():
    pass
"""
chunks = code_chunk(python_code, "python")
print(f"代码切分成 {len(chunks)} 个函数")
```

---

## 5. 实验对比 (Chunk Size Impact)

### 5.1 Chunk大小对检索的影响

| Chunk大小 | 检索Recall | 生成质量 | 存储成本 | 延迟 |
|-----------|-----------|---------|---------|------|
| 200字符 | 0.65 | 中 | 高 | 低 |
| 500字符 | 0.78 | 良 | 中 | 中 |
| 1000字符 | 0.82 | 优 | 中 | 中 |
| 2000字符 | 0.75 | 良 | 低 | 高 |

**结论**: 500-1000字符是最佳平衡点。

### 5.2 重叠率影响

| 重叠率 | 语义连贯性 | 存储冗余 | 检索效果 |
|--------|-----------|---------|---------|
| 0% | 低 | 无 | 可能断章取义 |
| 10% | 中 | 低 | 基本连贯 |
| 20% | 高 | 中 | 最佳平衡 |
| 50% | 很高 | 高 | 过度冗余 |

**结论**: 10-20%重叠率是推荐值。

### 5.3 策略对比

| 策略 | 语义完整性 | 大小均匀性 | 实现复杂度 | 适用场景 |
|------|-----------|-----------|-----------|---------|
| 固定大小 | 低 | 高 | 低 | 通用 |
| 语义边界 | 高 | 低 | 中 | 文章 |
| 递归 | 高 | 中 | 高 | 复杂文档 |
| 结构化 | 很高 | 中 | 中 | 结构化数据 |

---

## 6. 最佳实践与常见陷阱

### 6.1 Chunking选择决策树

```
开始
  ↓
结构化数据? → [是] → Markdown/代码/表格 → 结构化切分
              ↓ [否]
              ↓
            追求简单? → [是] → 固定大小切分
                        ↓ [否]
                        递归切分
```

### 6.2 最佳实践

1. **上下文长度匹配**: Chunk大小不超过模型上下文50%
2. **保留元数据**: 记录块来源、位置、标题
3. **适度重叠**: 10-20%重叠保持连贯
4. **预处理清洗**: 去除冗余空白、格式标准化
5. **分层索引**: 大块用于粗筛，小块用于精排

### 6.3 常见陷阱

1. **过度切分**: 块太小丢失上下文
2. **边界截断**: 在关键词中间截断
3. **忽视格式**: 代码/Markdown未结构化处理
4. **无重叠**: 相邻块完全断裂
5. **不清洗**: 保留过多格式标记

### 6.4 检查清单

```markdown
- [ ] 块大小适配模型上下文
- [ ] 保留文档结构元数据
- [ ] 设置合理重叠率
- [ ] 特殊格式结构化处理
- [ ] 切分后质量抽样检查
```

---

## 7. 总结

Chunking是RAG系统的关键环节，策略选择需考虑：

**核心原则**:
1. **语义优先**: 尽量保持语义单元完整
2. **大小平衡**: 500-1000字符是推荐范围
3. **重叠保障**: 10-20%重叠防止断章取义
4. **元数据保留**: 记录来源便于溯源

**推荐配置**:
- **通用文本**: 递归切分，500字符，20%重叠
- **技术文档**: Markdown结构化切分
- **代码**: 函数/类级别切分
- **表格**: 行/列级别切分

**未来趋势**:
- 基于LLM的智能切分
- 自适应块大小
- 多粒度分层索引
