# 第 16 个家族:世界模型 / 视频生成 · 设计

**日期**:2026-07-20
**作者**:通过 brainstorming 共同确定
**状态**:Design Approved,等待写实施计划

---

## 1. 背景与动机

仓库目前有 15 个架构家族(`01-cnn` … `15-reasoning-o1-r1`),覆盖 CV / NLP / 生成模型 / PEFT / RLHF / MoE / RAG-Agent / 推理模型等主线,且全部 68 个节点已配齐"金标本"交互页面。世界模型(world models)与视频生成是 2018-2024 年间独立于上述 15 条主线之外、但近两年热度最高的新兴方向之一——它既不完全属于 `10-diffusion`(角度不同:diffusion 家族关注单帧图像生成机制,这个新家族关注时空一致性 / 可交互环境建模),也不属于任何现有家族,值得单独开一个第 16 个家族收录。

本轮范围**只写 markdown 正本**(6 篇节点 + 1 篇家族 README +配图 SVG),不做交互式金标本页面——金标本留到后续单独轮次按现有"开新一档金标本节点"的流程逐个补。

## 2. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 家族目录 id | `16-world-models` |
| 2 | 家族标题 | 世界模型 / 视频生成(World Models / Video Generation) |
| 3 | 家族色 token | `--family-16: #d946ef`(fuchsia,延续现有 15 个 token 从玫红→橙黄→绿→蓝→紫的色相环,落在 15 号紫`#a855f7`之后、与 01 号玫红`#db2777`区分开的洋红段) |
| 4 | 节点数量与排序 | 6 篇,**严格按年份升序**排列(与其他 15 个家族的 order 语义一致,不按"RL 决策支 / 视频生成支"分组) |
| 5 | 本轮交付物 | 只写 markdown 正本 + 配图 SVG + 家族注册,不做金标本交互页 |

## 3. 节点列表(最终版,严格按年份)

| order | 年份 | 节点 | 论文 | 一句话定位 |
|---|---|---|---|---|
| 01 | 2018 | World Models | *World Models*(Ha & Schmidhuber, NeurIPS 2018) | "世界模型"概念起源:VAE 学视觉表征(V)+ MDN-RNN 学时序动态(M)+ 极小的线性 Controller(C)在 M 生成的"梦境"里训练,首次证明智能体可以完全在自己学到的世界模型内部完成策略训练 |
| 02 | 2022 | Video Diffusion Models | *Video Diffusion Models*(Ho, Salimans et al., 2022) | 把 DDPM 的去噪框架从图像推广到视频:3D U-Net(分解时空卷积)+ 图像/视频联合训练 + 无分类器引导,是"用 diffusion 生成视频"这条路线的起点 |
| 03 | 2023 | DreamerV3 | *Mastering Diverse Domains through World Models*(Hafner et al., 2023) | 把 latent imagination 式的 model-based RL 规模化到跨领域通吃(Atari / DeepMind Control / Minecraft 拿到钻石等 150+ 任务),固定超参数不调参就能匹配甚至超过各领域的 model-free SOTA |
| 04 | 2024 | Sora | *Video generation models as world simulators*(OpenAI technical report, 2024) | 把 DiT 规模化到分钟级、多分辨率、多时长连贯视频:spacetime patches 统一表示不同长宽比/时长的时空数据,论文明确提出"video generation models are world simulators"的定位 |
| 05 | 2024 | Genie | *Genie: Generative Interactive Environments*(DeepMind, 2024) | 无监督地从海量无标注互联网视频里学出逐帧可控制的生成式环境:隐式学习出离散的"latent action"空间,不需要任何动作标注 |
| 06 | 2024 | GameNGen | *Diffusion Models Are Real-Time Game Engines*(Google, 2024) | 用条件 diffusion 模型完全替代传统游戏引擎的渲染循环,实时(20+ FPS)交互式生成可玩的 DOOM 画面,证明神经网络可以端到端承担游戏引擎的职责 |

**排序说明**:2024 年三篇(Sora / Genie / GameNGen)内部按公开时间先后排(Sora 2024.02 → Genie 2024.02 稍晚 → GameNGen 2024.08),与其他家族"同年多篇按实际发布顺序排"的既有惯例一致。

## 4. 每篇节点的写作规范(复用现有 15 个家族的锁死结构)

严格复用仓库已收敛的节点写作模板(参照 `13-moe-efficient/04-deepseek-v3.md` 等近期节点的实际章节结构,而非早期草创期的旧模板):

```
---
name: "..."
year: ...
family: "16-world-models"
order: ...
paper: "..."
authors: [...]
key_idea: "..."
---

# {{ name }} ({{ year }})

## 前作进展
## 核心思想:{{ 副标题 }}
### 直觉
## 机制一:{{ ... }}
## 机制二:{{ ... }}
## 机制三:{{ ... }}
## 三件套协同 — {{ ... }}
## 关键代码
## 性能数据
## 影响 / 后续
```

- 每篇至少配 1-2 张手绘风格 SVG 架构图,存到 `16-world-models/assets/`,复用现有 15 个家族统一的调色板与图注格式(`*图 N:...*`)
- 每篇正文里的机制拆解、数字、benchmark 结果必须来自真实论文内容,不得编造——写作阶段需要先读一遍原论文摘要/关键章节(如工具允许可用 WebFetch/WebSearch 查证),而不是凭训练记忆直接下笔
- 跨家族引用走相对链接,前置依赖指向 `foundations/`:例如 Sora 应链接 `../10-diffusion/05-dit.md`(DiT 架构复用)与 `../07-gpt-scaling/04-scaling-laws.md`(scaling 叙事呼应);DreamerV3 可链接 `../02-rnn-lstm/`(RNN 系历史脉络,虽然 DreamerV3 已经不用 RNN,但 World Models/01 用到,值得在"前作进展"里回顾)
- 本次遇到的、之前 15 个家族反复踩过的两个坑要主动规避:
  1. **CommonMark 加粗定界符边界情况**——`**` 紧贴标点(引号/问号/括号)时另一侧必须是空白或标点,不能直接接普通字符,写完后建议跑一遍 `AllGoldenSamples.smoke.test.tsx` 同款正则自查(或人工过一遍规则)
  2. **`$` 货币符号与 remark-math 冲突**——涉及 GPU 小时成本 / 训练费用等美元数字时,一律转义成 `\$`

## 5. 家族 README 写作规范

复用 `13-moe-efficient/README.md` 等近期家族 README 的章节结构:

```
# 世界模型 / 视频生成

> {{ 一句话定位引言,blockquote }}

## 一句话定位
## 概念本身
### {{ 子概念拆解,如"世界模型"三要素 V/M/C,或 RL 决策支 vs 视频生成支两条脉络 }}
## 子时间线
## 依赖与延伸
```

家族 README 需要明确解释这条主线与 `10-diffusion`(diffusion 机制本身)、`05-transformer`(DiT 复用 Transformer 主干)、`07-gpt-scaling`(scaling 叙事)、`02-rnn-lstm`(World Models 用到 RNN)之间的关系与边界,避免读者以为这是 diffusion 家族的重复内容。

## 6. 需要同步改动的注册文件

新家族要接入现有基础设施,需要改这几处(均为已有机制,非新增架构):

1. `scripts/generate_timeline.py` 的 `FAMILY_IDS` 列表 —— 追加 `"16-world-models"`
2. `web/src/types/family.ts` 的 `FamilyId` 联合类型 —— 追加 `"16-world-models"`
3. `web/src/styles/tokens.css` —— 新增 `--family-16: #d946ef; /* World Models/Video 洋红 */`
4. 写完全部 markdown 正本后运行 `python3 scripts/generate_timeline.py` 重新生成 `TIMELINE.md` 和 `web/src/data/families.json`(后者是脚本自动生成的产物,不手工编辑)

金标本相关的 `web/src/components/node/golden/index.ts`、`AllGoldenSamples.smoke.test.tsx`、`ProseCompleteness.test.tsx` 本轮不动——这些节点在没有金标本条目时会自动走 `NodePage.tsx` 的通用 markdown 渲染路径,不会导致任何测试失败(`ProseCompleteness.test.tsx` 只扫描 `golden/*/lib/prose.ts`,新节点没有这个文件,天然被跳过)。

## 7. 验收标准

- `python3 scripts/generate_timeline.py` 跑完后,`TIMELINE.md` 出现 16 个家族、且新增 6 行按年份正确插入到原有行之间
- `web/src/data/families.json` 出现 `"id": "16-world-models"` 家族块,6 个节点、`colorToken: "--family-16"`
- 6 篇 markdown frontmatter 齐全(`name/year/family/order/paper/authors/key_idea`),`family` 字段全部为 `"16-world-models"`,`order` 为 1-6 且与年份升序一致
- 每篇正文的 9 个必填章节(前作进展/核心思想+直觉/机制一二三/三件套协同/关键代码/性能数据/影响后续)全部非空
- `npx tsc --noEmit` 与 `npx vitest run`(web/ 下)全项目通过,不因新增 `FamilyId` 联合类型成员或 tokens.css 改动引入任何回归

## 8. Out of scope(本轮明确不做)

- 不写任何 `web/src/components/node/golden/16-world-models/*` 交互页面
- 不在首页/家族列表页做额外的视觉特殊处理(复用现有家族卡片组件即可)
- 不追加 `foundations/` 下的横切基础页面(如"视频 tokenization"独立成页),世界模型特有的技术点(spacetime patches、latent action 等)在各节点自己的"机制"章节里讲清楚即可,不拆共享基础页
