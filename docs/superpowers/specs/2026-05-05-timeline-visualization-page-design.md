# Daily-LLM 时间线可视化页面设计

## 背景

Daily-LLM 目前是以 Markdown 为主体的双语学习知识库，核心叙事是“每一个技术都是被前一代局限逼出来的”。仓库已有 `00-Timeline/README.md` 和 `00-Timeline/assets/timeline.svg`，但它们更适合静态阅读，不适合作为公网网站的第一屏交互入口。

本设计面向第一版独立网页：打开网站后，用户直接看到深度学习与大模型的横向演进时间线，并通过点选节点阅读对应年份的具体内容。

## 目标

- 首页就是时间线，不额外制作营销式 landing page。
- 横向主轴放在页面上方，作为第一视觉和核心导航。
- 下方展示当前节点内容，承担阅读和理解任务。
- 视觉风格素雅、暖色、清爽、鲜亮，延续仓库 Mermaid 暖色语义。
- 第一版可部署到公网，技术结构支持后续扩展，但不提前实现账号、进度、CMS 或复杂站内系统。

## 非目标

- 不做用户登录、收藏、学习进度或评论。
- 不把整个 Markdown 知识库一次性迁移成完整网站。
- 不引入后端服务；第一版先做可部署前端站点。
- 不在第一版实现自动解析 Markdown 生成页面数据。

## 技术形态

使用 `Vite + React + TypeScript` 建立独立前端站点，建议目录为 `web/`。

推荐结构：

```text
web/
  index.html
  package.json
  src/
    App.tsx
    main.tsx
    data/
      timeline.ts
    components/
      TimelineAxis.tsx
      TimelineContent.tsx
      TimelineWorkList.tsx
    styles/
      global.css
```

第一版时间线数据放在 `web/src/data/timeline.ts`。相较直接写死在组件里，结构化数据更利于维护；相较解析 Markdown，第一版更稳定、实现风险更低。后续可以增加脚本检查 `timeline.ts` 与 `00-Timeline/README.md` 的同步关系。

## 页面布局

桌面端采用“顶部横向主轴 + 下方内容阅读区”。

页面从上到下分为三段：

1. 顶部标题区：显示 Daily-LLM、页面标题、年份范围、当前节点。
2. 横向时间线区：显示 1948、2012–2025 的节点，支持横向滚动和点击切换。
3. 内容展示区：显示当前节点的完整摘要、结构化解释、同年关键工作和关联模块。

内容展示区建议使用两列：

- 主列：年份、标题、背景、发生了什么、解决了什么、留下什么新问题。
- 侧列：同年关键工作、所属模块链接、下一相关节点。

移动端保持同一信息架构，但改为单列：

- 标题区压缩。
- 时间线仍横向滚动，节点尺寸变小。
- 内容主列和侧列上下排列。

## 交互设计

用户可以通过两种方式浏览：

- 横向滚动时间线，查看完整演进链路。
- 点击年份节点，切换下方内容。

当前节点状态：

- 节点尺寸更大。
- 使用暖玫瑰或琥珀色高亮。
- 顶部状态显示当前年份和标题。
- 时间线进度条随当前节点位置更新。

内容切换：

- 点击节点后，下方内容即时更新。
- 第一版不强制页面跳转。
- 可以把当前节点同步到 URL hash，例如 `#2017`，便于分享和刷新恢复。

关联模块：

- 模块链接第一版指向仓库相对路径，例如 `../02-Language-Transformers/transformer-architecture/`。
- 后续如果扩展为完整站点，可迁移为站内路由。

## 数据模型

每个节点建议包含：

```ts
type TimelineNode = {
  year: string;
  title: string;
  shortTitle: string;
  phase: string;
  previousLimit: string;
  whatHappened: string;
  solved: string;
  newProblems: string;
  keyWorks: Array<{
    name: string;
    contribution: string;
    modulePath?: string;
  }>;
  relatedModules: Array<{
    label: string;
    path: string;
  }>;
};
```

第一版至少覆盖：

- 1948 Shannon 信息论
- 2012 AlexNet
- 2013 Word2Vec / VAE
- 2014 GAN / Seq2Seq / Attention / Adam
- 2015 ResNet / Batch Norm
- 2016 AlphaGo
- 2017 Transformer
- 2018 BERT / GPT-1
- 2019 GPT-2 / T5
- 2020 GPT-3 / Scaling Laws
- 2021 CLIP / Codex / LoRA
- 2022 ChatGPT / RLHF
- 2023 GPT-4 / LLaMA
- 2024 MoE / 长上下文 / o1
- 2025 DeepSeek R1 / Test-Time Compute

## 视觉规范

整体视觉遵循项目暖色语义：

- 页面背景：米白、浅暖色。
- 主时间轴：琥珀色。
- 数据/背景信息：浅琥珀。
- 计算/突破：浅玫瑰。
- 结果/解决：浅绿色。
- 局限/新问题：浅橙色。
- 演进/链接：浅蓝色。

约束：

- 不使用深色科技风。
- 不使用厚重阴影或大面积渐变。
- 不做营销页式大 Hero。
- 卡片用于内容块，不嵌套卡片。
- 字号和间距以阅读清晰为主，避免装饰性压过内容。

## 可访问性与响应式

- 节点使用按钮语义，支持键盘聚焦和回车切换。
- 当前节点需要有视觉状态和 `aria-current`。
- 颜色不作为唯一状态提示，节点尺寸、描边或文字状态同步变化。
- 横向时间线在小屏保留滚动，不压缩到不可读。
- 内容区在移动端单列展示，避免文字被时间线遮挡。

## 验证标准

第一版完成后应验证：

- `npm run build` 成功。
- 桌面宽度下，顶部主轴明确可见，下方内容不被遮挡。
- 移动宽度下，时间线可横向滚动，内容单列可读。
- 点击每个年份节点都能更新内容。
- 当前节点高亮和进度状态正确。
- 主要链接指向现有仓库章节或时间线内容。

## 后续扩展

第一版保留但不实现：

- 搜索年份、论文或模块。
- 按视觉、语言、规模、多模态、对齐、系统生产筛选。
- 完整章节站内路由。
- 从 Markdown 或统一 JSON 自动生成网页数据。
- 学习路径、进度、收藏或账号系统。
