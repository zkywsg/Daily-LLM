# GNN 家族金标本交互页 · 设计

**日期**:2026-07-22
**作者**:通过 brainstorming 共同确定
**状态**:Design Approved,等待写实施计划

---

## 1. 背景与动机

第 17 个家族(`17-graph-neural-networks`,GCN/GraphSAGE/GAT/GIN/Graphormer)的 markdown 正本已全部完成并验证通过(见 `docs/superpowers/plans/2026-07-22-graph-neural-networks-family.md`),按计划留待后续单独轮次补金标本交互页。本轮就是补这 5 个节点的交互页。

## 2. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 覆盖范围 | 5 个节点全部做(GCN、GraphSAGE、GAT、GIN、Graphormer) |
| 2 | 复用结构 | 完全复用现有金标本模式:hero + 3 个 Stage(对应机制一/二/三)+ footer(关键代码/性能数据/影响后续),参照 `web/src/components/node/golden/mixtral/` 的文件结构 |
| 3 | 数据 | 每节点 `lib/data.ts` 提供确定性(非随机、非真训练)demo 数据/函数,`lib/prose.ts` 复用现有 `extractProse` 正则提取逻辑(H2/H3 标题匹配 previousWork/intuition/mechanism1-3/synergy/keyCode/performance/aftermath) |
| 4 | 分支策略 | 直接在 `master` 分支做(与前两轮 markdown 正本一致) |

## 3. 每个节点的文件结构(与 mixtral 完全一致的模式)

```
web/src/components/node/golden/{slug}/
  NodePage{Name}.tsx           # hero + 3 Stage + footer
  NodePage{Name}.module.css    # 复用现有 Stage.module.css 风格
  lib/
    data.ts                    # 确定性 demo 数据/函数
    prose.ts                   # extractProse() + {NAME}_SOURCE_PATH
  stages/
    {Stage1Name}.tsx
    {Stage2Name}.tsx
    {Stage3Name}.tsx
  widgets/
    {Widget}.tsx  ×N
```

注册进 `web/src/components/node/golden/index.ts` 的 `goldenSamples` map,key 为 `"17-graph-neural-networks/{NN-slug}"`。

## 4. 各节点 Stage / Widget 设计

### GCN(01-gcn)— 图卷积的归一化传播

- **Stage1 自环**(对应机制一):toy 图(5-6 节点),可切换"原始邻接 A vs 加自环 Ã=A+I",实时展示邻接矩阵/度数变化
- **Stage2 对称归一化**(对应机制二):同一张图,对比"未归一化求和聚合 vs D̃^(-1/2)ÃD̃^(-1/2) 归一化聚合"下,高度数节点是否碾压低度数节点的贡献权重
- **Stage3 逐层传播**(对应机制三):2 层 GCN 堆叠可视化,某中心节点的感受野从 1-hop 扩展到 2-hop(高亮受影响节点集合随层数变化)

### GraphSAGE(02-graphsage)— 采样 + 可学习聚合

- **Stage1 邻居采样**(机制一):点击一个节点,对比"使用全部邻居 vs 固定大小 k 的随机采样子集"
- **Stage2 聚合器对比**(机制二):同一组邻居特征向量,并排展示 mean / max-pool / LSTM(顺序敏感,不同排列给出不同结果)三种聚合器的输出差异
- **Stage3 归纳式泛化**(机制三):在图 A 上"训练"得到的聚合权重,直接应用到图 B 中一个训练时不存在的新节点上产出 embedding(对比 GCN 的直推式局限——GCN 需要固定邻接矩阵、无法处理新节点)

### GAT(03-gat)— 注意力加权聚合

- **Stage1 注意力系数**(机制一):选中心节点,展示共享前馈网络计算出的 attention logits,可调"温度"参数观察分布锐化/平滑
- **Stage2 mask+softmax**(机制二):展示原始 logits → 仅保留邻居 mask 后的 logits → softmax 归一化后的权重,三阶段数值对比
- **Stage3 多头注意力**(机制三):4 个头并排展示各自不同的注意力分布热力图,再对比 concat(中间层)vs average(输出层)两种多头输出组合方式

### GIN(04-gin)— 单射聚合与 WL-test

- **Stage1 非单射反例**(机制一,对应正文里已有的具体数值反例):构造两个不同的邻居特征多重集,展示在 mean/max 聚合下结果相同(信息塌缩),但 sum 聚合下结果不同
- **Stage2 ε 可调**(机制二):滑块调节 GIN 更新公式里的 (1+ε) 自身权重,实时观察 h_v^(k) 输出中"自身信息 vs 邻居聚合信息"占比的变化
- **Stage3 WL 染色对比**(机制三):小图上做 1-2 轮 Weisfeiler-Lehman 颜色迭代动画,与 GIN 的 sum+MLP 聚合过程并排对比,说明两者在区分能力上等价

### Graphormer(05-graphormer)— 结构编码注入注意力

- **Stage1 中心性编码**(机制一):节点度数直方图 → 查表映射为 embedding 向量的可视化
- **Stage2 空间编码**(机制二):选任意两个节点,计算最短路径距离,展示对应的 bias 项 b_φ(i,j) 如何叠加进 attention score 矩阵(热力图形式展示 Q·K^T/√d + bias 的叠加前后对比)
- **Stage3 边编码**(机制三):在 Stage2 基础上叠加路径上的边特征编码,展示完整 attention bias 矩阵相对纯空间编码的进一步调整

## 5. 复用与不复用的既有机制

**复用**:
- `MarkdownRenderer` 组件渲染 prose 片段
- `extractProse()` 的 H2/H3 正则提取模式(previousWork/intuition/mechanism1/mechanism2/mechanism3/synergy/keyCode/performance/aftermath)
- `Stage.module.css` 通用样式(可能需要为每个节点新建 `NodePage{Name}.module.css`,参照 mixtral 的做法)
- `AllGoldenSamples.smoke.test.tsx`(自动扫描 `goldenSamples` 注册表,无需为新节点单独加测试用例)
- `ProseCompleteness.test.tsx`(自动 glob 扫描所有 `lib/prose.ts`,无需单独加测试用例)

**不做**:
- 不真实训练/运行 GNN 模型,所有交互数据均为确定性构造(hash/固定小图/预设向量),不引入任何 ML 运行时依赖
- 不新增图数据可视化库依赖(用现有 SVG/CSS 手写小型 toy 图,与其余金标本一致的"手绘风"约定)
- 不改动 `foundations/` 或已有家族的金标本页面

## 6. 已知的坑(沿用前两轮沉淀的经验)

1. **`extractProse()` 的 H2/H3 匹配严格依赖 markdown 里 `## 核心思想` 下必须有 `### 直觉`、`### 机制一` 等三级标题**——写 Stage 组件前要先确认对应节点 md 文件里核心思想 section 的实际三级标题拼写,与 `H3_KEYS` 正则(`/^直觉/`、`/^机制一/` 等)完全匹配
2. **`ProseCompleteness.test.tsx` 要求每个 prose 字段提取结果非空**——如果某节点 md 缺某个三级标题或字段为空,vitest 会直接报红,写完 `lib/prose.ts` 后要跑一遍这个测试文件确认
3. **CommonMark 加粗定界符边界情况**(在 widget 内联文案或 prose 源文件里都可能触发)——`**` 紧贴标点时另一侧必须是空白,不能直接接普通字符
4. **`index.ts` 注册 key 必须与 `NodePage.tsx` 路由解析的 `${familyId}/${nodeSlug}` 完全一致**——即 `"17-graph-neural-networks/01-gcn"` 这种精确格式,拼错会导致该节点静默走回普通 markdown 渲染而非报错

## 7. 验收标准

- 5 个节点各自的 `NodePage{Name}.tsx` + `lib/data.ts` + `lib/prose.ts` + 3 个 stage + widgets 全部完成并注册进 `index.ts`
- `npx vitest run`(web/ 下)全部通过,包括自动扫描到的 `AllGoldenSamples.smoke.test.tsx` 里新增的 5 条冒烟测试用例、`ProseCompleteness.test.tsx` 里新增的 5 个 prose 模块
- `npx tsc --noEmit` 通过
- 浏览器验证:5 个节点详情页均渲染出交互式 Stage(而非纯 markdown 兜底渲染),各 widget 的核心交互(点击/滑块/切换)均可用且反馈符合设计描述,console 无 error

## 8. Out of scope(本轮明确不做)

- 不改动已完成的 markdown 正本内容(除非发现 prose 提取所需的三级标题缺失需要小幅补齐)
- 不引入图可视化第三方库(d3-force 等),继续用手写 SVG/CSS 小图
- 不做移动端专门适配(遵循现有金标本页面的响应式规则即可,不追加新规则)
