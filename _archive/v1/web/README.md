# Daily-LLM 网页

Daily-LLM 知识库的公开 Web 前端,把 19 个家族 / 89 个节点整理为可探索、可阅读的可视化页面。

## 路由结构

- `/` — 主页,两种浏览模式可切换
  - **按时间**:全部节点放在密度感知的横向时间线上(密集年份拉宽、空白年份压缩),hover 显示节点摘要,点击进入节点详情
  - **按家族**:19 个家族卡片网格,每张卡片显示节点数和年份范围
- `/families/:familyId` — 家族页,展示该家族子时间线,每个节点配 mini-arch 缩略图
- `/families/:familyId/:nodeSlug` — 节点页;已注册节点优先使用交互式金标本,其余节点渲染通用 markdown 正文
- `/foundations` — 横切基础概念列表与正文

## 技术栈

- Vite + React + TypeScript
- React Router(SPA 路由)
- react-markdown + remark-gfm + remark-math + rehype-katex(数学公式)+ rehype-highlight(代码高亮)
- Mermaid(`.md` 内 ```mermaid 块按需懒加载)
- Framer Motion(过渡动效)
- Vitest + React Testing Library

## 本地运行

```bash
cd web
npm install
npm run dev
```

默认预览地址为 `http://127.0.0.1:5173/`;项目约定固定使用 5173 端口。

## 构建

```bash
cd web
npm run build
```

产物在 `web/dist/`(不进入 Git)。

## 测试

```bash
cd web
npm test
```

## 内容数据

节点元数据由仓库根的脚本生成,落到:

```text
web/src/data/families.json
```

每个节点带 frontmatter 字段(name / year / family / order / paper / authors / key_idea / path)。
markdown 正文从仓库根的家族目录(`01-cnn/05-resnet.md` 等)在构建期通过 `import.meta.glob` 懒加载;
家族目录下的 `assets/*.svg` 同样通过 glob 解析,markdown 里写相对路径(如 `assets/05-resnet-residual.svg`)即可。

更新内容时:

1. 直接编辑仓库根的节点 markdown
2. 跑生成脚本同步 `families.json`(若新增节点或改了 frontmatter)
3. dev server HMR 自动刷新

## 金标本路径

新加交互式金标本节点在 `components/node/golden/index.ts` 注册即可,路由层会优先用金标本组件代替通用 markdown 渲染。
