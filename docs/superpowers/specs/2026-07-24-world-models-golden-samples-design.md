# 世界模型 / 视频生成家族金标本交互页 · 设计

**日期**:2026-07-24
**作者**:通过 brainstorming 共同确定
**状态**:Design Approved,等待写实施计划

---

## 1. 背景与动机

第 16 个家族(`16-world-models`,World Models/Video Diffusion Models/DreamerV3/Sora/Genie/GameNGen)的 markdown 正本已在更早的轮次全部完成,按计划留待后续单独轮次补金标本交互页。本轮就是补这 6 个节点的交互页,是目前仓库里缺口存在最久的一批。

## 2. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 覆盖范围 | 6 个节点全部做(World Models、Video Diffusion Models、DreamerV3、Sora、Genie、GameNGen) |
| 2 | 复用结构 | 完全复用现有金标本模式:hero + 3 个 Stage(对应机制一/二/三)+ footer(关键代码/性能数据/影响后续) |
| 3 | **prose 提取模板** | **与 17/18 家族不同,与 mixtral 相同** —— 已用 `grep -n '^## \|^### '` 核对 `16-world-models/01-world-models.md` 的真实标题层级:`## 前作进展`、`## 核心思想:...`(H2,下嵌套 `### 直觉` 和 `### 三个必须同时跨过的坎` 两个 H3)、`## 机制一:...`/`## 机制二:...`/`## 机制三:...`(均为顶层 H2,不嵌套)、`## 三件套协同`、`## 关键代码`、`## 性能数据`、`## 影响 / 后续`。也就是说 `lib/prose.ts` 要直接复用 `mixtral/lib/prose.ts` 的双层提取模板(H2_KEYS 匹配 `前作进展/核心思想(→_coreInsight)/机制一/机制二/机制三/三件套协同/关键代码/性能数据/影响`,H3_KEYS 只在 `_coreInsight` 状态下匹配 `直觉`,`### 三个必须同时跨过的坎` 这个 H3 不提取、直接跳过留在 intuition 之外),**不能用 17/18 家族的扁平单层模板**。写第一个节点(Task 1)时仍需对该节点自己 `grep` 一遍确认,其余 5 个节点各自也要各自确认一遍标题措辞是否完全一致(不同节点的 H3 说法可能是"两个"或"三个"必须跨过的坎,词数不同但结构一致,不影响提取) |
| 4 | 分支策略 | 直接在 `master` 分支做(与前三轮 markdown/金标本正本一致) |

## 3. 各节点 Stage / Widget 设计

### World Models(01-world-models)— V+M+C 三段解耦

- **Stage1(V,VAE 视觉压缩)**:输入帧(toy 像素网格)→ VAE 压缩成潜向量,可调潜维度大小看重建质量权衡
- **Stage2(M,MDN-RNN 时序预测)**:用混合高斯(mixture density)预测下一潜状态,可调混合分量数,展示预测分布的多峰性/多样性
- **Stage3(C,梦境训练)**:完全在 M 自回归生成的"梦境"轨迹上跑 rollout(z→M→z'→M→z''→…,全程不碰真实 V/环境),可视化梦境轨迹与控制器输出

### Video Diffusion Models(02-video-diffusion-models)— 时空分解 + 联合训练

- **Stage1(时空分解架构)**:2D 空间卷积 / 1D 时间卷积 / 完整 3D 卷积三种方式的算力(FLOPs 量级)对比可视化
- **Stage2(图像/视频联合训练)**:切换"纯视频 batch" vs "图像+视频混合 batch",展示训练数据利用率/loss 稳定性的示意差异
- **Stage3(reconstruction guidance + 自回归扩展)**:滑动窗口自回归延长视频长度的演示,展示窗口如何逐步前移生成更长序列

### DreamerV3(03-dreamerv3)— RSSM + symlog + 纯 imagination 训练

- **Stage1(RSSM 离散隐变量)**:离散类别分布(categorical latent)采样可视化,可调类别数看隐状态表达力
- **Stage2(symlog 归一化)**:输入跨越不同量级的原始 reward 数值(如 Atari 的 ±1 分 vs Minecraft 的大额奖励),实时展示 symlog 变换后如何被压缩到可比范围
- **Stage3(纯 latent imagination 训练)**:actor-critic 完全在想象(imagined)轨迹上训练的 rollout 演示,不接触真实环境

### Sora(04-sora)— spacetime patches + DiT 规模化 + 原生分辨率

- **Stage1(spacetime patches)**:一段视频体(时空立方体)被切分成时空 patch 网格的可视化
- **Stage2(DiT 规模化)**:模型规模(参数量档位)vs 生成质量的示意曲线
- **Stage3(原生分辨率/长宽比/时长)**:不同长宽比/时长的输入都能被同一套变长 patch 序列统一处理的演示(切换几种预设分辨率/时长看 patch 数量如何自适应变化)

### Genie(05-genie)— tokenizer + 无监督 LAM + 动态模型

- **Stage1(视频 tokenizer)**:原始帧 → 离散 token 序列的压缩演示
- **Stage2(Latent Action Model)**:无监督推断相邻两帧之间的离散动作(latent action)分类,不依赖任何人工动作标注
- **Stage3(动态模型 + 交互式"玩")**:给定选择的离散动作,动态模型自回归生成下一帧,用户可以用几个离散动作按钮交互式"玩"生成出的世界

### GameNGen(06-gamengen)— RL 数据生成 + 条件 diffusion + 噪声增强抗漂移

- **Stage1(RL agent 自动生成训练数据)**:展示 RL agent 自我博弈产出训练数据轨迹,替代人类录屏
- **Stage2(条件 diffusion 预测下一帧)**:给定历史帧+动作条件,diffusion 模型预测下一帧,替代传统渲染步骤
- **Stage3(噪声增强抗漂移)**:开关"是否对条件帧加噪声增强",对比长程自回归多步生成后画面质量是否发生漂移退化

## 4. 复用与不复用的既有机制

**复用**:
- `MarkdownRenderer` 组件渲染 prose 片段
- `Stage.module.css` 通用样式(每节点自建一份,内容与既有家族完全一致)
- `AllGoldenSamples.smoke.test.tsx`(自动扫描 `goldenSamples` 注册表,无需为新节点单独加测试用例)
- `ProseCompleteness.test.tsx`(自动 glob 扫描所有 `lib/prose.ts`,无需单独加测试用例)

**不做**:
- 不真实训练/运行任何模型,所有交互数据均为确定性构造(hash/固定小样本/预设向量),不引入任何 ML 运行时依赖
- 不引入视频播放器/canvas 动画库,用现有 SVG/CSS 手写小型 toy 演示(与其余金标本一致的"手绘风"约定)
- 不改动已完成的 markdown 正本内容(除非发现 prose 提取所需的标题层级需要小幅核对确认,但不预期需要改动)

## 5. 已知的坑(沿用前几轮沉淀的经验)

1. **本家族标题层级与前两轮不同**——写 `lib/prose.ts` 前必须先 `grep` 精确核对具体标题层级,不能想当然套用扁平或双层模板
2. **`ProseCompleteness.test.tsx` 要求每个 prose 字段提取结果非空**——写完 `lib/prose.ts` 后要跑一遍这个测试文件确认
3. **CommonMark 加粗定界符边界情况**——`**` 紧贴标点时另一侧必须是空白,不能直接接普通字符
4. **`index.ts` 注册 key 必须与 `NodePage.tsx` 路由解析的 `${familyId}/${nodeSlug}` 完全一致**——即 `"16-world-models/01-world-models"` 这种精确格式
5. **SVG bar/条形图高度必须 clamp**——前几轮金标本反复出现无上限缩放导致视觉溢出的 bug,新建图表类 widget 时默认加 `Math.min(...)` 上限
6. **toggle/选择类按钮必须加 `aria-pressed`**——前几轮 code review 反复要求的可访问性基线

## 6. 验收标准

- 6 个节点各自的 `NodePage{Name}.tsx` + `lib/data.ts` + `lib/prose.ts` + 3 个 stage + widgets 全部完成并注册进 `index.ts`
- `npx vitest run`(web/ 下)全部通过,包括自动扫描到的 `AllGoldenSamples.smoke.test.tsx` 里新增的 6 条冒烟测试用例、`ProseCompleteness.test.tsx` 里新增的 6 个 prose 模块
- `npx tsc --noEmit` 通过
- 浏览器验证:6 个节点详情页均渲染出交互式 Stage(而非纯 markdown 兜底渲染),各 widget 的核心交互均可用且反馈符合设计描述,console 无 error

## 7. Out of scope(本轮明确不做)

- 不改动已完成的 markdown 正本内容
- 不引入图表/动画第三方库,继续用手写 SVG/CSS
- 不做移动端专门适配(遵循现有金标本页面的响应式规则即可)
