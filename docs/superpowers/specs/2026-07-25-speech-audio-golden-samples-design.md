# 语音/音频模型家族金标本交互页 · 设计

**日期**:2026-07-25
**作者**:通过 brainstorming 共同确定
**状态**:Design Approved,等待写实施计划

---

## 1. 背景与动机

第 18 个家族(`18-speech-audio`,Wav2Vec 2.0/HuBERT/Whisper/AudioLM/MusicGen)的 markdown 正本已在更早的轮次全部完成,按计划留待后续单独轮次补金标本交互页。本轮就是补这 5 个节点的交互页,补完后仓库里全部 84 个节点都会有金标本交互页。

## 2. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 覆盖范围 | 5 个节点全部做(Wav2Vec 2.0、HuBERT、Whisper、AudioLM、MusicGen) |
| 2 | 复用结构 | 完全复用现有金标本模式:hero + 3 个 Stage(对应机制一/二/三)+ footer(关键代码/性能数据/影响后续) |
| 3 | **prose 提取模板** | 与 World Models(16 家族)不同 —— 已用 `grep -n '^## \|^### '` 核对全部 5 个节点,`18-speech-audio/*.md` 用的是**扁平单层结构**(`## 核心思想 + 直觉` 本身就是一个 H2,不嵌套 `### 直觉`;`## 机制一/二/三` 也是顶层 H2),与 17-graph-neural-networks 家族完全一致。`lib/prose.ts` 直接复用 GNN 家族已验证过的扁平单层 `extractProse()` 模板,**不能**用 World Models 那套 mixtral 式双层模板 |
| 4 | 分支策略 | 直接在 `master` 分支做(与前几轮金标本一致) |

## 3. 各节点 Stage / Widget 设计

### Wav2Vec 2.0(01-wav2vec2)— CNN 编码器 + 量化对比学习 + 掩码预测

- **Stage1(CNN 特征编码器)**:原始波形(toy 序列)→ 多层一维卷积逐层下采样到约 50Hz 帧率,可调层数看压缩比如何变化
- **Stage2(量化模块 + 对比学习)**:连续特征 → 量化码本(离散化),展示对比学习任务:从若干候选(真实量化目标 + 干扰项)里让模型选出真实目标
- **Stage3(掩码预测)**:span masking 演示,选中的 mask 区间由 Transformer 上下文预测出量化目标,展示预测正确/错误的反馈

### HuBERT(02-hubert)— 离线聚类伪标签 + 掩码分类 + 迭代重聚类

- **Stage1(离线聚类生成伪标签)**:toy 特征点 → k-means 聚类分配可视化,展示每个特征点被分到哪个聚类
- **Stage2(BERT 式掩码预测)**:掩码分类任务(交叉熵,而非对比学习),展示分类头对被 mask 位置输出的类别概率分布
- **Stage3(迭代式重新聚类)**:点"跑下一轮重聚类"按钮,聚类边界逐轮细化(交互模式呼应 GIN 节点已验证过的 WL 染色迭代,但语义换成 HuBERT 的聚类精细化)

### Whisper(03-whisper)— log-mel + 标准 encoder-decoder + 多任务前缀

- **Stage1(log-mel 频谱输入)**:原始波形 → log-mel 频谱图可视化,展示频谱如何随时间/频率分布
- **Stage2(大规模弱监督数据过滤)**:切换"未过滤爬取数据" vs "过滤后数据",展示数据质量分布直方图的差异
- **Stage3(多任务统一格式)**:切换任务前缀按钮(转写/翻译/语言识别/时间戳),展示不同前缀 token 如何让同一个模型切换输出格式

### AudioLM(04-audiolm)— 语义 token + 声学 token(RVQ)+ 三阶段级联

- **Stage1(语义 token)**:音频 → 粗粒度语义 token 序列的提取演示,展示这套 token 采样率较低、携带长程结构信息
- **Stage2(声学 token,RVQ)**:多层残差量化码本可视化,第一层捕捉粗粒度信息,后续层逐层补充细节
- **Stage3(三阶段级联生成)**:逐步执行"语义 token 生成 → 粗声学 token 生成(条件于语义)→ 细声学 token 生成(条件于前两者)"三个阶段,每阶段一个按钮触发

### MusicGen(05-musicgen)— EnCodec RVQ + 码本交错 + 双重条件控制

- **Stage1(EnCodec 残差量化)**:多层并行 RVQ 码本可视化,与 AudioLM 的呈现方式呼应但强调"单阶段模型需要同时处理所有层"这一区别
- **Stage2(码本交错 delay pattern)**:K 层并行码本流按阶梯状延迟摊平成一条序列的可视化,展示为什么这样能让单个自回归 Transformer 处理多层码本
- **Stage3(文本 + 旋律双重条件)**:切换"仅文本条件" / "仅旋律条件" / "两者都开"三种模式,展示条件信号如何组合注入生成过程

## 4. 复用与不复用的既有机制

**复用**:
- `MarkdownRenderer` 组件渲染 prose 片段
- GNN 家族已验证的扁平单层 `extractProse()` 模板(H2_KEYS 直接匹配 前作进展/核心思想/机制一/机制二/机制三/三件套协同/关键代码/性能数据/影响,无需 H3 特殊处理)
- `Stage.module.css` 通用样式(每节点自建一份,内容与既有家族完全一致)
- `AllGoldenSamples.smoke.test.tsx`、`ProseCompleteness.test.tsx`(自动扫描,无需单独加测试用例)

**不做**:
- 不真实训练/运行任何模型,所有交互数据均为确定性构造,不引入音频处理/ML 运行时依赖
- 不引入音频播放器/波形库,用现有 SVG/CSS 手写小型 toy 演示
- 不改动已完成的 markdown 正本内容(除非发现 CommonMark 加粗定界符等已知坑,按前几轮惯例顺手修复)

## 5. 已知的坑(沿用前几轮沉淀的经验,这轮尤其要主动做)

1. **prose 提取模板必须用扁平单层**,不能照抄 World Models 轮的双层模板(这轮最容易踩错的坑,因为上一轮刚做完双层模板)
2. **SVG 条形图/曲线高度必须 `Math.min(..., cap)` 钳位**,禁止无上限缩放(本轮多次出现的 bug 类型)
3. **所有 toggle/选择按钮加 `aria-pressed={condition}`**
4. **不要硬编码重复 `lib/data.ts` 已导出的常量**(如网格大小等),必须 import 复用,避免脱离数据源静默漂移
5. **深色主题下的固定背景色搭配必须显式指定可读的文字颜色**(不能依赖继承 `var(--ink-primary)`,上一轮在 Sora 节点踩过这个坑)
6. **数学/示意函数落笔前先手动验证**:凡是"滑块调整参数 → 图表展示效果"的 widget,必须先手算/脚本验证参数变化时图表确实呈现出所声称的效果,不能想当然(上一轮反复出现"滑块调了但效果没变"或"公式本身有错"的 bug)
7. **`index.ts` 注册 key 必须与 `NodePage.tsx` 路由解析的 `${familyId}/${nodeSlug}` 完全一致**,即 `"18-speech-audio/01-wav2vec2"` 这种精确格式

## 6. 验收标准

- 5 个节点各自的 `NodePage{Name}.tsx` + `lib/data.ts` + `lib/prose.ts` + 3 个 stage + widgets 全部完成并注册进 `index.ts`
- `npx vitest run`(web/ 下)全部通过,包括自动扫描到的冒烟测试与 prose 完整性测试各新增 5 条
- `npx tsc --noEmit` 通过
- 浏览器验证:5 个节点详情页均渲染出交互式 Stage(而非纯 markdown 兜底渲染),各 widget 的核心交互均可用且反馈符合设计描述,console 无 error
- 补完这批后,首页/家族页应显示全部 84 个节点都有金标本交互页(无遗漏)

## 7. Out of scope(本轮明确不做)

- 不改动已完成的 markdown 正本内容(除非修复已知的 CommonMark 坑)
- 不引入音频/图表第三方库,继续用手写 SVG/CSS
- 不做移动端专门适配
