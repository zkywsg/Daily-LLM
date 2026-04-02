# CNN Architectures Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the Chinese and English CNN chapter into a problem-driven narrative that matches the approved spec and the repository's README style.

**Architecture:** Keep the existing module boundaries and rewrite in place. The Chinese README becomes the source-of-truth structure for the problem chain, then the English README is rewritten with the same section order, conclusions, and transitions; finally, the Phase 01 overview is adjusted only if its one-line CNN description no longer matches the new chapter positioning.

**Tech Stack:** Markdown, Mermaid, repository style rules in `STYLE.md`, git diff/status for verification

---

## File Map

- Modify: `01-Visual-Intelligence/cnn-architectures/README.md`
  - Rewrite title framing, learning goals, intuition, mechanisms, architecture evolution, engineering pitfalls, and evolution note.
- Modify: `01-Visual-Intelligence/cnn-architectures/README_EN.md`
  - Mirror the Chinese structure with accurate English terminology rather than line-by-line translation.
- Modify: `01-Visual-Intelligence/README.md`
  - Adjust the CNN bullet only if the old summary no longer matches the rewritten chapter.
- Reference: `STYLE.md`
  - Source of truth for README structure, signature elements, Mermaid palette, and footer style.
- Reference: `docs/superpowers/specs/2026-04-02-cnn-architectures-design.md`
  - Approved design to implement exactly.

### Task 1: Re-check the current chapter and lock the rewrite scope

**Files:**
- Reference: `docs/superpowers/specs/2026-04-02-cnn-architectures-design.md`
- Reference: `STYLE.md`
- Reference: `01-Visual-Intelligence/cnn-architectures/README.md`
- Reference: `01-Visual-Intelligence/cnn-architectures/README_EN.md`
- Reference: `01-Visual-Intelligence/README.md`

- [ ] **Step 1: Re-open the approved spec and style guide**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-04-02-cnn-architectures-design.md
sed -n '1,220p' STYLE.md
```

Expected: The plan executor can point to the required section order, `你要记住` constraints, `演进笔记`, and the limit that the chapter ends with a bridge to attention/ViT rather than a full later-CNN survey.

- [ ] **Step 2: Snapshot the current files before editing**

Run:

```bash
sed -n '1,260p' 01-Visual-Intelligence/cnn-architectures/README.md
sed -n '1,260p' 01-Visual-Intelligence/cnn-architectures/README_EN.md
sed -n '1,120p' 01-Visual-Intelligence/README.md
```

Expected: You can identify which sections will be rewritten in place, which long code blocks will be removed or shortened, and whether the Phase 01 overview line needs to change.

- [ ] **Step 3: Write a temporary rewrite checklist in your notes**

Use this checklist while editing:

```text
1. Open with image spatial structure, not benchmark-first framing
2. Replace learning goals with the approved three questions
3. Rebuild section 2 as:
   - convolution and feature maps
   - receptive field and downsampling
   - residual connections
   - local-to-global limitation
4. Rewrite architecture evolution as "bottleneck -> response"
5. Keep only minimal convolution / conv block / residual block code
6. End with CNN strengths + boundary -> attention / ViT bridge
7. Mirror the same structure in README_EN.md
8. Touch 01-Visual-Intelligence/README.md only if summary drift exists
```

Expected: The executor has a concrete, non-ambiguous checklist before touching Markdown.

- [ ] **Step 4: Commit the scope-check checkpoint**

```bash
git add docs/superpowers/plans/2026-04-02-cnn-architectures.md
git commit -m "plan: define cnn chapter rewrite"
```

Expected: A clean checkpoint exists for the approved implementation plan before content edits begin.

### Task 2: Rewrite the Chinese CNN chapter around the problem chain

**Files:**
- Modify: `01-Visual-Intelligence/cnn-architectures/README.md`

- [ ] **Step 1: Replace the title framing, opening problem statement, and learning goals**

Update the top of `01-Visual-Intelligence/cnn-architectures/README.md` so it starts with this structure:

```markdown
# 为什么图像不能直接交给全连接网络？—— CNN 架构演进（2012–2017）

## 这个问题从哪来

> 2012 年之前，视觉识别系统大量依赖 SIFT、HOG 等手工特征，图像先被设计成特征，再交给分类器。即便使用多层感知机，把图像直接压平成向量也会带来参数爆炸，并破坏像素的空间邻域关系。AlexNet 的突破不只是“更深”，而是第一次大规模证明：把局部性和参数共享写进网络结构，才能让模型真正利用图像这种数据的形状。

## 学习目标

完成本章后，你应能回答：

1. 卷积的局部连接、参数共享、层级特征分别解决了什么问题？
2. 为什么 VGG、GoogLeNet、ResNet、DenseNet、SE-Net 会按这样的顺序出现？
3. CNN 为什么在视觉任务中强大，又为什么最终要与注意力机制汇流？
```

Expected: The chapter no longer opens as a generic CNN overview; it opens from the mismatch between image structure and fully connected modeling.

- [ ] **Step 2: Rewrite the intuition section into a three-step argument**

Replace the current intuition block with content following this shape:

```markdown
## 1. 直觉

图像不是“很多数字的集合”，而是带有二维邻域关系的信号：相邻像素通常共同构成边缘、纹理和局部形状，远处像素的关联则往往需要更高层语义才能建立。

如果把一张图像直接压平成向量，全连接层会同时犯两个错误：一是参数量随输入尺寸急剧膨胀；二是模型看不见“谁和谁本来是邻居”。对模型而言，左上角的像素和右下角的像素只是两个普通维度，没有结构差别。

卷积层的做法是反过来：先承认图像的局部性，只让一个小卷积核看局部窗口，再把同一组参数复用到整张图上。这样模型学到的就不是“某个固定坐标上的模式”，而是“无论出现在何处都成立的局部模式”。

> 你要记住：CNN 的本质不是“层更多”，而是“把空间结构写进模型假设里”。
```

Expected: The intuition section explains why convolution exists before discussing any architecture names.

- [ ] **Step 3: Rebuild the mechanism section with the approved four subsections**

Edit section 2 so the headings appear exactly as:

```markdown
## 2. 机制

### 2.1 卷积与特征图
### 2.2 感受野与下采样
### 2.3 残差连接
### 2.4 从局部到全局的限制
```

And make sure the content under them covers:

```text
2.1 formula + kernel/stride/padding/channel semantics + feature map as response map
2.2 stacked 3x3 rationale + pooling/stride tradeoffs + brief effective receptive field note
2.3 degradation vs overfitting + shortcut + identity mapping intuition
2.4 why CNN approximates global context gradually but does not natively mix long-range dependencies
```

Expected: The mechanics no longer jump straight from convolution basics into architecture history; they build the conceptual constraints that make later architectures necessary.

- [ ] **Step 4: Replace the architecture table with bottleneck-driven evolution prose**

Rewrite the evolution section so the chapter explicitly uses the pattern below:

```markdown
## 3. 架构演进：每一代都在修上一代的问题

### 3.1 AlexNet：先把 CNN 跑通
### 3.2 VGG：把“深而整齐”做成范式
### 3.3 GoogLeNet：回应 VGG 的计算负担
### 3.4 ResNet：解决“越深越难训练”
### 3.5 DenseNet / SE-Net：在复用与重标定上继续打磨
```

Each subsection must answer:

```text
- What bottleneck did the previous generation leave behind?
- What structural idea answered it?
- What new tradeoff remained afterward?
```

Expected: A reader can explain why the sequence of architectures exists without memorizing a flat chronology table.

- [ ] **Step 5: Shrink the code examples to the three approved blocks**

Keep only these code responsibilities in the Chinese chapter:

```text
1. minimal Conv2d shape check
2. standard Conv + BN + ReLU + downsampling block
3. minimal residual block with shortcut alignment
```

Delete or heavily compress the large end-to-end TinyResNet training example so the chapter does not read like a model-building tutorial.

Expected: The code supports the architecture explanation instead of dominating the second half of the chapter.

- [ ] **Step 6: Rewrite the engineering pitfalls and evolution note**

Ensure the Chinese ending contains this structure:

```markdown
## 4. 工程陷阱

1. shortcut 未对齐
2. 过早下采样
3. 只堆层数，不看感受野设计
4. BN / activation / residual 顺序混乱

## 演进笔记

> **这一技术的遗产**：...
>
> 这正是后续视觉模型逐步引入注意力机制、并最终走向 ViT 的原因。
```

Also keep the footer links in repository style:

```markdown
---
**上一章**: [训练与优化](../training/README.md) | **下一章**: [序列模型](../sequence-models/README.md)
```

Expected: The chapter ends by framing attention as the next modeling move, not by expanding into a new CNN survey.

- [ ] **Step 7: Run a targeted content verification on the Chinese README**

Run:

```bash
rg -n "^## 这个问题从哪来|^## 学习目标|^## 1\\. 直觉|^## 2\\. 机制|^## 3\\. 架构演进|^## 4\\. 工程陷阱|^## 演进笔记|你要记住|注意力|ViT" 01-Visual-Intelligence/cnn-architectures/README.md
```

Expected: All required top-level sections, one approved `你要记住` line in intuition, and the attention/ViT bridge are present.

- [ ] **Step 8: Commit the Chinese rewrite**

```bash
git add 01-Visual-Intelligence/cnn-architectures/README.md
git commit -m "docs: rewrite cnn chapter in chinese"
```

Expected: The Chinese rewrite is isolated in its own commit.

### Task 3: Rewrite the English CNN chapter to match the approved structure

**Files:**
- Modify: `01-Visual-Intelligence/cnn-architectures/README_EN.md`
- Reference: `01-Visual-Intelligence/cnn-architectures/README.md`

- [ ] **Step 1: Replace the English title, opening, and goals with structure-aligned content**

Update the top of `01-Visual-Intelligence/cnn-architectures/README_EN.md` to follow this pattern:

```markdown
# Why Can't Images Be Modeled Well by Fully Connected Networks? — CNN Architecture Evolution (2012–2017)

## Where This Problem Came From

> Before AlexNet, visual recognition relied heavily on hand-crafted features such as SIFT and HOG. Flattening an image into a vector made MLPs expensive and structurally blind to spatial locality. The real breakthrough of CNNs was not depth alone, but the decision to encode locality and parameter sharing directly into the architecture.

## Learning Goals

After this chapter, you should be able to answer:

1. What problems are solved by local connectivity, parameter sharing, and hierarchical feature learning?
2. Why did VGG, GoogLeNet, ResNet, DenseNet, and SE-Net appear in that order?
3. Why are CNNs powerful in vision, and why do they eventually need to converge with attention-based modeling?
```

Expected: The English chapter opens from the same architectural problem as the Chinese one, not from generic benchmark history.

- [ ] **Step 2: Mirror the same section order and conceptual constraints**

Ensure the English README contains these top-level sections in the same order:

```markdown
## 1. Intuition
## 2. Mechanism
## 3. Architecture Evolution
## 4. Engineering Pitfalls
## Evolution Notes
```

And the mechanism section uses these subsections:

```markdown
### 2.1 Convolution and Feature Maps
### 2.2 Receptive Fields and Downsampling
### 2.3 Residual Connections
### 2.4 The Limit from Local to Global Modeling
```

Expected: The English version is structurally isomorphic to the Chinese one.

- [ ] **Step 3: Write natural English instead of line-by-line translation**

Use English terminology like the following where appropriate:

```text
inductive bias
spatial locality
feature hierarchy
response map
identity shortcut
long-range dependency
global context
channel recalibration
```

Keep the architecture sequence and takeaways aligned with Chinese, but prefer fluent technical English over literal sentence matching.

Expected: The English README reads like an original technical chapter, not a translation artifact.

- [ ] **Step 4: Keep the same reduced code scope in English**

Make sure the English code examples also stop at:

```text
1. minimal convolution shape verification
2. standard convolution block
3. minimal residual block
```

Do not leave the large full TinyResNet example in English if it was removed or compressed in Chinese.

Expected: Both language versions make the same scoping decision about code.

- [ ] **Step 5: End with the same attention/ViT bridge and footer**

The English ending should include the same conclusion:

```markdown
## Evolution Notes

> **What this technique left behind:** ...
>
> This is why later vision models gradually introduced attention and eventually converged toward ViT-style token mixing.

---
**Previous**: [Training and Optimization](../training/README_EN.md) | **Next**: [Sequence Models](../sequence-models/README_EN.md)
```

Expected: The English README lands on the same boundary condition and next-step framing as the Chinese one.

- [ ] **Step 6: Run a targeted content verification on the English README**

Run:

```bash
rg -n "^## Where This Problem Came From|^## Learning Goals|^## 1\\. Intuition|^## 2\\. Mechanism|^## 3\\. Architecture Evolution|^## 4\\. Engineering Pitfalls|^## Evolution Notes|attention|ViT|inductive bias" 01-Visual-Intelligence/cnn-architectures/README_EN.md
```

Expected: The English README contains the full mirrored section skeleton and the final bridge to attention/ViT.

- [ ] **Step 7: Commit the English rewrite**

```bash
git add 01-Visual-Intelligence/cnn-architectures/README_EN.md
git commit -m "docs: rewrite cnn chapter in english"
```

Expected: The English rewrite is isolated and reviewable on its own.

### Task 4: Align the Phase 01 overview only if summary drift exists

**Files:**
- Modify: `01-Visual-Intelligence/README.md`
- Reference: `01-Visual-Intelligence/cnn-architectures/README.md`

- [ ] **Step 1: Compare the overview bullet against the rewritten chapter**

Run:

```bash
sed -n '1,80p' 01-Visual-Intelligence/README.md
sed -n '1,120p' 01-Visual-Intelligence/cnn-architectures/README.md
```

Expected: You can decide whether the current summary still matches the new chapter emphasis on problem-driven CNN evolution and the bridge to attention.

- [ ] **Step 2: If needed, replace the CNN summary line with a tighter problem-driven description**

If the current bullet feels too list-like, replace just the descriptive lines under `### [CNN 架构]` with something like:

```markdown
### [CNN 架构](cnn-architectures/README.md)
从“图像为什么不能直接交给全连接层”出发，沿着 AlexNet → ResNet 的问题链理解经典 CNN 演进，并收束到注意力出现前的局部建模边界。
- 卷积、感受野与下采样
- 深度、计算量与信息流动的权衡
- 经典 CNN 演进如何一步步逼近注意力时代
```

If the existing overview still matches, make no edit.

Expected: The Phase 01 overview reflects the rewritten chapter without expanding the scope of this task.

- [ ] **Step 3: Verify links and headings remain intact**

Run:

```bash
rg -n "CNN 架构|cnn-architectures/README.md|sequence-models/README.md" 01-Visual-Intelligence/README.md 01-Visual-Intelligence/cnn-architectures/README.md 01-Visual-Intelligence/cnn-architectures/README_EN.md
```

Expected: Internal links still point to the same module paths and the Phase 01 overview still reads cleanly.

- [ ] **Step 4: Commit the overview alignment if changed**

```bash
git add 01-Visual-Intelligence/README.md
git commit -m "docs: align visual intelligence overview with cnn rewrite"
```

Expected: Only the overview summary change, if any, is captured here.

### Task 5: Final verification and handoff

**Files:**
- Verify: `01-Visual-Intelligence/cnn-architectures/README.md`
- Verify: `01-Visual-Intelligence/cnn-architectures/README_EN.md`
- Verify: `01-Visual-Intelligence/README.md`

- [ ] **Step 1: Run a final diff review on the rewritten files**

Run:

```bash
git diff -- 01-Visual-Intelligence/cnn-architectures/README.md
git diff -- 01-Visual-Intelligence/cnn-architectures/README_EN.md
git diff -- 01-Visual-Intelligence/README.md
```

Expected: The diff shows a structural rewrite, reduced oversized code samples, and no unrelated file changes.

- [ ] **Step 2: Run a final structure check against the spec**

Run:

```bash
rg -n "^## 这个问题从哪来|^## 学习目标|^## 1\\. 直觉|^## 2\\. 机制|^## 3\\. 架构演进|^## 4\\. 工程陷阱|^## 演进笔记" 01-Visual-Intelligence/cnn-architectures/README.md
rg -n "^## Where This Problem Came From|^## Learning Goals|^## 1\\. Intuition|^## 2\\. Mechanism|^## 3\\. Architecture Evolution|^## 4\\. Engineering Pitfalls|^## Evolution Notes" 01-Visual-Intelligence/cnn-architectures/README_EN.md
```

Expected: Both versions expose the agreed skeleton with no missing top-level sections.

- [ ] **Step 3: Run a repository status check**

Run:

```bash
git status --short
```

Expected: Only the intended chapter files and any explicitly accepted overview adjustment appear as modified.

- [ ] **Step 4: Commit the final verified rewrite**

```bash
git add 01-Visual-Intelligence/cnn-architectures/README.md 01-Visual-Intelligence/cnn-architectures/README_EN.md 01-Visual-Intelligence/README.md
git commit -m "docs: restructure cnn architecture chapters"
```

Expected: The final state is committed after verification, with the rewrite reviewable as documentation-only work.

## Self-Review

### Spec coverage

- Opening reframed around image spatial structure: covered in Task 2 Step 1 and Task 3 Step 1.
- Learning goals replaced with the approved three questions: covered in Task 2 Step 1 and Task 3 Step 1.
- Mechanism section rebuilt into four subsections: covered in Task 2 Step 3 and Task 3 Step 2.
- Architecture evolution rewritten as bottleneck-driven narrative: covered in Task 2 Step 4 and Task 3 Step 3.
- Oversized code examples reduced to three essential blocks: covered in Task 2 Step 5 and Task 3 Step 4.
- Engineering pitfalls narrowed to four chapter-specific issues: covered in Task 2 Step 6.
- Attention/ViT bridge at the end of both chapters: covered in Task 2 Step 6 and Task 3 Step 5.
- Optional Phase 01 overview alignment: covered in Task 4.

### Placeholder scan

Reviewed the plan for unresolved placeholders and vague action text. None remain.

### Type and naming consistency

- Chinese top-level headings consistently use `这个问题从哪来 / 学习目标 / 1. 直觉 / 2. 机制 / 3. 架构演进 / 4. 工程陷阱 / 演进笔记`.
- English top-level headings consistently use `Where This Problem Came From / Learning Goals / 1. Intuition / 2. Mechanism / 3. Architecture Evolution / 4. Engineering Pitfalls / Evolution Notes`.
- The reduced code scope is consistently defined as minimal convolution, standard conv block, and minimal residual block.
