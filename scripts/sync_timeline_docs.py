#!/usr/bin/env python3
"""把 web/src/data/timeline.ts 作为单一数据源，同步生成 markdown：

· timeline/<year>/README.md      —— 每年一篇，含 4 段叙事 + 关键工作 + 前置基础 + 主题深挖
· timeline/README.md             —— 主索引（导航表 + 14 个年份链接 + 前史链接）
· tracks/<topic>/README.md       —— 追加/替换「## 涉及的时间线节点」段

设计原则：
- TS 文件是 SSOT（single source of truth），markdown 是派生物
- 脚本 idempotent：可任意次重跑，结果稳定
- 不动 track README 的其他正文，只维护一个用 marker 包围的 section

用法：
    python3 scripts/sync_timeline_docs.py            # dry-run，打印将生成的文件
    python3 scripts/sync_timeline_docs.py --apply    # 写盘
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TS_FILE = REPO_ROOT / "web" / "src" / "data" / "timeline.ts"

# ---------------------------------------------------------------------------
# TS → Python: 提取两个数组字面量并转 JSON
# ---------------------------------------------------------------------------

def strip_comments(s: str) -> str:
    # 去 /* ... */ 与 // ...
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"//[^\n]*", "", s)
    return s


def extract_array_literal(src: str, decl_marker: str) -> str:
    """定位 `export const <decl_marker>: ... = [` 后的数组字面量，返回 `[...]`（含方括号）。"""
    idx = src.find(decl_marker)
    if idx == -1:
        raise RuntimeError(f"找不到声明: {decl_marker}")
    # 跳过类型注解 `TimelineNode[]` 之后才找数组字面量本体
    eq = src.find("=", idx)
    if eq == -1:
        raise RuntimeError(f"{decl_marker} 后找不到 =")
    bracket = src.find("[", eq)
    if bracket == -1:
        raise RuntimeError(f"{decl_marker} 后找不到 [")
    # 按方括号深度往后扫，跨越字符串
    depth = 0
    i = bracket
    in_str = False
    str_ch = ""
    while i < len(src):
        ch = src[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == str_ch:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
                str_ch = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return src[bracket : i + 1]
        i += 1
    raise RuntimeError(f"{decl_marker} 的数组字面量未闭合")


def ts_array_to_json(literal: str) -> list:
    """TS 对象字面量 → JSON。仅处理本仓库实际用到的语法：

    - 无引号键 `key:` → `"key":`
    - 去尾逗号 `,}` `,]`
    - 字符串只用双引号（仓库内已是双引号）
    """
    s = literal
    # 给裸键加双引号（仅识别 ASCII 单词）
    s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s)
    # 去尾逗号
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return json.loads(s)


def load_data():
    raw = TS_FILE.read_text(encoding="utf-8")
    cleaned = strip_comments(raw)
    timeline_lit = extract_array_literal(cleaned, "timelineNodes")
    prehistory_lit = extract_array_literal(cleaned, "prehistoryNodes")
    return (
        ts_array_to_json(timeline_lit),
        ts_array_to_json(prehistory_lit),
    )


# ---------------------------------------------------------------------------
# 路径重写：TS 里用 `../foo/` 是相对 `web/` 的，生成 md 时要按目标文件位置重算
# ---------------------------------------------------------------------------

def fix_path_for(file_rel_dir: Path, ts_path: str) -> str:
    """TS 字段里的 path 是相对 `web/` 写的（`../foundations/...`）。

    转成"相对仓库根"后，再算相对当前生成文件位置的路径。
    """
    if ts_path.startswith("../"):
        root_rel = ts_path[3:]  # 去掉 "../"
    else:
        root_rel = ts_path
    target_abs = REPO_ROOT / root_rel
    from_abs = REPO_ROOT / file_rel_dir
    import os

    rel = os.path.relpath(target_abs, from_abs)
    if ts_path.endswith("/") and not rel.endswith("/"):
        rel += "/"
    return rel


# ---------------------------------------------------------------------------
# 生成 markdown
# ---------------------------------------------------------------------------

YEAR_TEMPLATE = """\
# {year} · {title}

> **阶段**：{phase}
>
> 时间线主线在这一年发生了什么、解决了什么、又留下了什么新问题。

[← 回到主时间线](../README.md)

---

## 之前卡在哪

{previous_limit}

## 发生了什么

{what_happened}

## 解决了什么

{solved}

## 留下了什么新问题

{new_problems}

---

## 同年关键工作

| 名字 | 贡献 |
|---|---|
{key_works_rows}

## 前置基础（Layer 0 · 工具箱）

> 看不懂当前内容时先来这里补课。

{prereq_list}

## 主题深挖（Layer 2 · 主题深挖）

> 想沿这条主线纵向往后追，进入下面的 track。

{tracks_list}

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
<!-- 不要手动编辑——改 timeline.ts 后重跑脚本 -->
"""


def render_year(node: dict) -> str:
    file_dir = Path(f"timeline/{node['year']}")

    def render_works(works: list[dict]) -> str:
        rows = []
        for w in works:
            name = w["name"]
            contrib = w["contribution"]
            if w.get("modulePath"):
                fixed = fix_path_for(file_dir, w["modulePath"])
                name = f"[{name}]({fixed})"
            rows.append(f"| **{name}** | {contrib} |")
        return "\n".join(rows)

    def render_links(items: list[dict]) -> str:
        if not items:
            return "_（暂无）_"
        lines = []
        for it in items:
            fixed = fix_path_for(file_dir, it["path"])
            lines.append(f"- [{it['label']}]({fixed})")
        return "\n".join(lines)

    return YEAR_TEMPLATE.format(
        year=node["year"],
        title=node["title"],
        phase=node["phase"],
        previous_limit=node["previousLimit"],
        what_happened=node["whatHappened"],
        solved=node["solved"],
        new_problems=node["newProblems"],
        key_works_rows=render_works(node["keyWorks"]),
        prereq_list=render_links(node["prerequisites"]),
        tracks_list=render_links(node["tracks"]),
    )


INDEX_TEMPLATE = """\
# 深度学习与大模型演进时间线

> 每一个技术的出现，背后都有一个"不得不解决"的问题。
> 这条时间线不是论文列表，而是一部"被逼出来的历史"。

<img src="assets/timeline.svg" alt="深度学习与大模型演进地图" width="100%">

主时间线从 2012 AlexNet 起。1948 / 1958 / 1986 / 1997 四个里程碑作为
「深度学习前史」单列，详见 [prehistory/](./prehistory/)。

## 导航

| 年份 | 核心事件 | 阶段 |
|---|---|---|
{rows}

## 深度学习前史（2012 之前的基石）

| 年份 | 里程碑 | 继续学 |
|---|---|---|
{prehistory_rows}

→ 完整阅读：[prehistory/](./prehistory/)

## 三层架构

| 层 | 入口 | 排序方式 |
|---|---|---|
| L0 · 基础工具箱 | [foundations/](../foundations/) | 按主题树，不按时间 |
| L1 · 编年主线（本目录） | [timeline/](.) | 按年份升序 |
| L2 · 主题深挖 | [tracks/](../tracks/) | 按主题深度递进 |

详见 [docs/restructure.md](../docs/restructure.md)。

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
"""


def render_index(timeline_nodes: list[dict], prehistory_nodes: list[dict]) -> str:
    rows = [
        f"| [{n['year']}](./{n['year']}/) | {n['title']} | {n['phase']} |"
        for n in timeline_nodes
    ]
    pre_rows = [
        f"| [{n['year']}](./prehistory/{n['year']}-{n['shortTitle'].lower()}.md) | {n['title']} | [{n['foundationLabel']}]({fix_path_for(Path('timeline'), n['foundationPath'])}) |"
        for n in prehistory_nodes
    ]
    return INDEX_TEMPLATE.format(
        rows="\n".join(rows),
        prehistory_rows="\n".join(pre_rows),
    )


# ---------------------------------------------------------------------------
# tracks/<topic>/README.md 追加 涉及的时间线节点 section
# ---------------------------------------------------------------------------

MARKER_BEGIN = "<!-- BEGIN: timeline-references (auto-generated) -->"
MARKER_END = "<!-- END: timeline-references -->"

# track 路径前缀 → tracks 目录名 映射；用 path 开头判断
# 例：`../tracks/vision/cnn-architectures/` → track="vision"
TRACK_PREFIX_RE = re.compile(r"^\.\./tracks/([^/]+)/")


def collect_track_refs(timeline_nodes: list[dict]) -> dict[str, list[dict]]:
    """对每个 tracks/<topic>，收集哪些时间线节点引用了它（按 tracks[] 字段）。"""
    by_track: dict[str, list[dict]] = {}
    for node in timeline_nodes:
        seen_tracks_in_this_year: set[str] = set()
        for t in node["tracks"]:
            m = TRACK_PREFIX_RE.match(t["path"])
            if not m:
                continue
            topic = m.group(1)
            if topic in seen_tracks_in_this_year:
                continue
            seen_tracks_in_this_year.add(topic)
            by_track.setdefault(topic, []).append(
                {
                    "year": node["year"],
                    "title": node["title"],
                    "via_label": t["label"],
                    "via_path": t["path"],
                }
            )
    return by_track


def render_track_section(topic: str, refs: list[dict]) -> str:
    file_dir = Path(f"tracks/{topic}")
    lines = [
        MARKER_BEGIN,
        "",
        "## 涉及的时间线节点",
        "",
        "> 本 track 在主时间线里被以下年份引用为「主题深挖」入口。",
        "",
        "| 年份 | 节点 | 引用入口 |",
        "|---|---|---|",
    ]
    for r in refs:
        year_link = f"[{r['year']}](../../timeline/{r['year']}/)"
        # 把 via_path 从「相对 web/」转成「相对当前 track README 位置」
        via_fixed = fix_path_for(file_dir, r["via_path"])
        via = f"[{r['via_label']}]({via_fixed})"
        lines.append(f"| {year_link} | {r['title']} | {via} |")
    lines += ["", MARKER_END]
    return "\n".join(lines)


def upsert_track_section(readme: Path, section: str) -> tuple[str, bool]:
    """在 readme 里 upsert section（按 marker 包围）。返回 (新内容, 是否变化)。"""
    if not readme.exists():
        return section + "\n", True
    text = readme.read_text(encoding="utf-8")
    if MARKER_BEGIN in text and MARKER_END in text:
        pattern = re.compile(
            re.escape(MARKER_BEGIN) + r".*?" + re.escape(MARKER_END),
            re.DOTALL,
        )
        new_text = pattern.sub(section, text)
    else:
        # 追加到文件末尾
        sep = "" if text.endswith("\n") else "\n"
        new_text = text + sep + "\n---\n\n" + section + "\n"
    return new_text, new_text != text


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    apply = "--apply" in sys.argv
    timeline_nodes, prehistory_nodes = load_data()

    writes: list[tuple[Path, str]] = []

    # 1) 每年一篇
    for node in timeline_nodes:
        out = REPO_ROOT / "timeline" / node["year"] / "README.md"
        writes.append((out, render_year(node)))

    # 2) 主索引
    writes.append(
        (REPO_ROOT / "timeline" / "README.md", render_index(timeline_nodes, prehistory_nodes))
    )

    # 3) 每个 track 的引用 section
    refs_by_track = collect_track_refs(timeline_nodes)
    for topic, refs in sorted(refs_by_track.items()):
        readme = REPO_ROOT / "tracks" / topic / "README.md"
        section = render_track_section(topic, refs)
        new_text, changed = upsert_track_section(readme, section)
        if changed:
            writes.append((readme, new_text))

    # report
    print(f"Plan: {len(writes)} files")
    for p, _content in writes:
        print(f"  · {p.relative_to(REPO_ROOT)}")

    if apply:
        for p, content in writes:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        print("\nApplied.")
    else:
        print("\nRun with --apply to write changes.")


if __name__ == "__main__":
    main()
