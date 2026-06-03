# Daily-LLM 时间线可视化网页

这个目录是 Daily-LLM 的独立前端网页。第一版首页就是横向时间线：上方浏览深度学习与大模型的演进节点，下方阅读当前年份的背景、突破、解决的问题和后续瓶颈。

## 技术栈

- Vite
- React
- TypeScript
- Vitest + React Testing Library

## 本地运行

```bash
cd web
npm install
npm run dev
```

默认访问地址为 Vite 输出的本地 URL，通常是 `http://localhost:5173/` 或 `http://127.0.0.1:5173/`。

## 构建

```bash
cd web
npm run build
```

构建产物会生成在 `web/dist/`，该目录不进入 Git。

## 测试

```bash
cd web
npm test
```

当前测试覆盖：

- 时间线数据是否包含 1948、2012–2025 的核心节点
- 主要节点是否保留关联模块链接
- 默认节点、点击切换、URL hash 同步
- 当前时间线节点的 `aria-current` 状态

## 内容维护

时间线数据位于：

```text
web/src/data/timeline.ts
```

每个节点包含年份、标题、阶段、旧瓶颈、发生了什么、解决了什么、新问题、关键工作和关联模块。更新网页内容时，应同步检查 `../timeline/README.md` 是否需要补充或调整，避免网页和知识库主时间线脱节。

## 后续方向

- 继续补足每年关键工作的细节和模块链接。
- 增加按阶段筛选、搜索和站内章节路由。
- 等页面内容稳定后，再接入公网部署。
