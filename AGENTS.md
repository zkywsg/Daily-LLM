# AGENTS.md · Daily-LLM 项目元信息

> 这里只放"我是什么项目、有什么约定"。
>
> 怎么写内容 → [docs/writing-style.md](docs/writing-style.md)
> Mermaid / 代码 / frontmatter → [docs/tech-conventions.md](docs/tech-conventions.md)

---

## 项目背景

- **仓库**：Daily-LLM · 深度学习与大模型双语学习知识库
- **主要语言**：中文优先，英文为辅
- **主轴**：以架构家族为单位，按时间排序

## 仓库结构

- `01-cnn/` … `15-reasoning-o1-r1/` — 15 个家族（仓库根，按时间）
- `foundations/` — 9 个横切基础子模块
- `TIMELINE.md` — 由 `scripts/generate_timeline.py` 自动生成
- `projects/` — 跨家族实战项目
- `web/` — 可视化网页
- `_archive/` — 只读历史素材

## 核心原则

- **单一正本**：每个工作只在它所在家族目录有正本，跨家族引用走相对链接
- **TIMELINE.md 不要手工编辑**：新增/修改节点后运行 `python3 scripts/generate_timeline.py`
- **`_archive/` 是只读素材区**：在写某个家族时可以取材，但不要在 `_archive/` 内新增或修改

## 开发服务器（web）

```bash
cd web && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

- 端口固定 `5173`，不要随意更换
- 如果 `5173` 被占用，先说明占用情况并询问是否停止旧服务或临时换端口
- 给用户预览地址默认 `http://127.0.0.1:5173/`

## 写作规范入口

- **三层模板与调性**：[docs/writing-style.md](docs/writing-style.md)
- **技术约定**：[docs/tech-conventions.md](docs/tech-conventions.md)
- **金标本示范节点**：[01-cnn/02-alexnet.md](01-cnn/02-alexnet.md)
