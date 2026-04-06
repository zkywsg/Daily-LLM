# Phase 02 · Language Transformers 结构与规范对齐设计方案

**日期**: 2026-04-06
**范围**: `02-Language-Transformers/` 中英双语 README、`attention-mechanisms/src/` 代码文件、`attention-mechanisms/notebook.ipynb`

---

## 一、目标

本次工作只处理 `02-Language-Transformers/` 的结构与规范一致性，不扩写知识深度，不新增超出当前章节范围的新内容。

核心目标：

1. 让 02 章中文 README 全部符合仓库统一模板。
2. 让英文 README 与中文 README 的章节骨架严格镜像。
3. 让 `attention-mechanisms` 下的 Python 文件符合仓库代码规范。
4. 让 `attention-mechanisms/notebook.ipynb` 符合 10-cell 固定结构。
5. 保持 02 章与 `00-Timeline/README.md` 的现有时间线节点同步，不引入删漏。

## 二、非目标

1. 不补充新的论文、历史节点或延伸知识点。
2. 不修改算法行为或测试语义。
3. 不扩大到 `03-Scale-Multimodal/` 或其他章节。
4. 不对 `00-Timeline/README.md` 做大规模重写；只有在 02 章节点发生必要联动时才做最小修正。
5. 不把本次规范整改变成内容重写项目。

## 三、文件范围

### 必做

- `02-Language-Transformers/README.md`
- `02-Language-Transformers/README_EN.md`
- `02-Language-Transformers/recurrent-networks/README.md`
- `02-Language-Transformers/recurrent-networks/README_EN.md`
- `02-Language-Transformers/attention-mechanisms/README.md`
- `02-Language-Transformers/attention-mechanisms/README_EN.md`
- `02-Language-Transformers/transformer-architecture/README.md`
- `02-Language-Transformers/transformer-architecture/README_EN.md`
- `02-Language-Transformers/pretrained-models/README.md`
- `02-Language-Transformers/pretrained-models/README_EN.md`
- `02-Language-Transformers/attention-mechanisms/src/attention.py`
- `02-Language-Transformers/attention-mechanisms/src/test_attention.py`
- `02-Language-Transformers/attention-mechanisms/notebook.ipynb`

### 按联动结果决定

- `00-Timeline/README.md`

## 四、README 结构设计

### 4.1 中文 README

所有中文模块页统一采用以下结构：

1. 标题：`# 为什么 [旧方法] 不够用了？—— [技术名]`
2. `## 这个问题从哪来`
3. `## 学习目标`
4. `## 1. 直觉`
5. `## 2. 机制`
6. `## 3. 工程陷阱`
7. `## 演进笔记`
8. `---`
9. `**上一章** | **下一章**`

签名元素要求：

1. 每个模块必须有 `这个问题从哪来`
2. 每个模块必须包含 `> 你要记住：...`，且全篇不超过 3 次
3. 每个模块必须以 `演进笔记` 收束，并链接到后续模块

### 4.2 英文 README

英文 README 不强行逐字套用中文标题，但必须与中文页保持完全一致的章节骨架：

1. 标题位置一致
2. 历史来源段一致
3. 学习目标数量一致
4. `Intuition / Mechanism / Engineering Pitfalls / Evolution Notes` 顺序一致
5. `Remember this:` 的提示数量与中文页对应
6. 导航位置与链接对应

### 4.3 总览页

`02-Language-Transformers/README.md` 与 `README_EN.md` 不再维持当前仅目录式写法，而是调整为符合阶段入口页风格的模块页：

1. 保留阶段总览和子模块导航功能
2. 增加 `这个问题从哪来`、`学习目标`、`直觉`、`机制`、`工程陷阱`、`演进笔记`
3. 保留时间线节点表，但放入统一结构中

## 五、模块级整改约束

### 5.1 recurrent-networks

当前结构已较接近模板，本次只做规范对齐：

1. 检查 `你要记住` 数量与格式
2. 将渐进式实现补齐为 4 个 Step
3. 校正中英文对应章节顺序
4. 不扩写额外理论内容

### 5.2 attention-mechanisms

本次重点处理：

1. Mermaid 图统一为仓库规定配色与灰色连线
2. Step 标题与注释统一成“解决什么问题”
3. 英文版改为与中文版同样的结构骨架
4. 保留现有内容重点，不新增延伸专题

### 5.3 transformer-architecture

这是本轮整改的最高优先级文件，原因是当前偏离模板最明显。

必须完成：

1. 改成规范标题
2. 补 `这个问题从哪来`
3. 将普通提示语改成规范的 `> 你要记住：...`
4. 用 Mermaid 图替换当前纯文本结构图
5. 增补 `演进笔记`
6. 调整整篇顺序，使其回到仓库统一模板

### 5.4 pretrained-models

本次主要做结构整理，不改变已有内容边界：

1. 保持 BERT / GPT / T5 三家族主线
2. 将渐进式实现整理为统一的 4-step 形式
3. 统一中英文章节骨架
4. 保持与下一阶段入口的演进衔接

## 六、代码文件规范设计

### 6.1 `attention.py`

本次只修规范，不改行为：

1. 文件头改为单行格式：`"""模块名 · 路径 · 核心内容 · 依赖"""`
2. 修正错误路径 `03-NLP-Transformers`
3. 为核心函数补三行注释头
4. 检查 docstring 中参数 shape 标注是否完整
5. 保留 `torch.manual_seed(42)` 的统一用法

### 6.2 `test_attention.py`

本次只修规范，不改测试逻辑：

1. 文件头改为单行格式
2. 修正错误路径
3. 保留现有测试语义
4. 只做必要的注释和格式调整

## 七、Notebook 结构设计

`attention-mechanisms/notebook.ipynb` 必须整理为固定 10 cells：

1. `[MD]` 标题 + 一句话验证目标
2. `[Code]` 导入 + `seed(42)` + 环境检查
3. `[MD]` `## 直觉（含公式）`
4. `[Code]` Step 1 最小实现
5. `[Code]` Step 2 边界处理
6. `[Code]` Step 3 完整实现
7. `[MD]` `## 验证`
8. `[Code]` shape 验证 / 期望输出检查
9. `[MD]` `## 可视化`
10. `[Code]` matplotlib 图

本次不新增训练逻辑，不把 notebook 扩成实验脚本。

## 八、时间线联动设计

根据 `AGENTS.md` 中“时间线与模块双向同步”的要求，本次需要核对而不是预设重写：

1. 检查 `02-Language-Transformers/README.md` 的时间线节点是否保留现有项目口径
2. 检查重排后是否删漏 `Word2Vec / GloVe / FastText / Transformer / ELMo / GPT-1 / GPT-2 / T5 / RoBERTa / ALBERT / DistilBERT`
3. 如果 02 总览页节点没有变化，则 `00-Timeline/README.md` 不做修改

## 九、实施顺序

建议按以下顺序执行：

1. 重构 `transformer-architecture` 中英 README
2. 重构 `02-Language-Transformers` 总览页中英 README
3. 对齐 `attention-mechanisms` 与 `pretrained-models` 中英 README
4. 补齐 `recurrent-networks` 的 Step 结构与英文骨架一致性
5. 修正 `attention.py` 与 `test_attention.py` 的代码规范
6. 重排 `attention-mechanisms/notebook.ipynb` 为 10-cell 结构
7. 最后做时间线节点回查

## 十、验证方案

实施完成后，需要按以下方式验证：

1. 逐个 README 检查模板章节是否齐全
2. 逐个 README 检查签名元素是否齐全且次数合规
3. 检查中英文页面的章节顺序是否一致
4. 检查 Mermaid 图是否采用 `graph TD` 与统一配色
5. 检查 Python 文件头、三行注释头、shape 标注是否符合规范
6. 检查 notebook 是否为 10 cells 且职责顺序正确
7. 运行 `attention-mechanisms/src/test_attention.py` 对应测试，确认规范整改未破坏现有行为

## 十一、风险与边界

1. 总览页改成模板页后，diff 会明显变大，但这是结构统一的必要代价。
2. 英文页从自由写法改为镜像结构时，可读性会略受模板约束，但这是本次工作的明确目标。
3. notebook 的 cell 重排可能改变执行顺序体验，因此只做最小职责调整，不扩展示例内容。
4. 若实施中发现某个时间线节点必须联动，将只做最小改动，不借机扩充时间线正文。
