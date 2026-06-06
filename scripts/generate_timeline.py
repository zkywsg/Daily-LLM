#!/usr/bin/env python3
"""扫描仓库根下的家族目录（NN-xxx/），收集每个节点 markdown 的 frontmatter，
生成按年份升序的 TIMELINE.md 速查表。

用法：
    python3 scripts/generate_timeline.py [--root .] [--out TIMELINE.md]
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


FAMILY_DIR_RE = re.compile(r"^\d{2}-[a-z0-9\-]+$")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Node:
    name: str
    year: int
    family: str
    order: int
    key_idea: str
    paper: str
    path: Path


def parse_frontmatter(text: str) -> dict | None:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    block = m.group(1)
    data: dict = {}
    for line in block.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        key = key.strip()
        raw = raw.strip()
        if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
            raw = raw[1:-1]
        if raw.lstrip("-").isdigit():
            data[key] = int(raw)
        else:
            data[key] = raw
    return data


def collect_nodes(root: Path) -> list[Node]:
    nodes: list[Node] = []
    for fam_dir in sorted(root.iterdir()):
        if not fam_dir.is_dir() or not FAMILY_DIR_RE.match(fam_dir.name):
            continue
        for md in sorted(fam_dir.glob("*.md")):
            if md.name.lower() == "readme.md":
                continue
            fm = parse_frontmatter(md.read_text(encoding="utf-8"))
            if not fm or "year" not in fm or "name" not in fm:
                continue
            nodes.append(Node(
                name=str(fm.get("name", "")),
                year=int(fm.get("year", 0)),
                family=str(fm.get("family", fam_dir.name)),
                order=int(fm.get("order", 0)),
                key_idea=str(fm.get("key_idea", "")),
                paper=str(fm.get("paper", "")),
                path=md.relative_to(root),
            ))
        for sub in sorted(fam_dir.iterdir()):
            if not sub.is_dir():
                continue
            readme = sub / "README.md"
            if not readme.exists():
                continue
            fm = parse_frontmatter(readme.read_text(encoding="utf-8"))
            if not fm or "year" not in fm or "name" not in fm:
                continue
            nodes.append(Node(
                name=str(fm.get("name", "")),
                year=int(fm.get("year", 0)),
                family=str(fm.get("family", fam_dir.name)),
                order=int(fm.get("order", 0)),
                key_idea=str(fm.get("key_idea", "")),
                paper=str(fm.get("paper", "")),
                path=readme.relative_to(root),
            ))
    return nodes


def render(nodes: list[Node]) -> str:
    nodes_sorted = sorted(nodes, key=lambda n: (n.year, n.family, n.order))
    lines = [
        "# TIMELINE",
        "",
        "> 自动生成自各家族节点的 frontmatter。**请勿手工编辑。**",
        "> 重新生成：`python3 scripts/generate_timeline.py`",
        "",
        "| 年份 | 名字 | 家族 | 关键思想 | 路径 |",
        "|------|------|------|---------|------|",
    ]
    for n in nodes_sorted:
        lines.append(
            f"| {n.year} | **{n.name}** | `{n.family}` | {n.key_idea} | [{n.path}]({n.path}) |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="仓库根（默认当前目录）")
    parser.add_argument("--out", default="TIMELINE.md", help="输出文件路径")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    nodes = collect_nodes(root)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text(render(nodes), encoding="utf-8")
    print(f"wrote {out_path} ({len(nodes)} nodes)")


if __name__ == "__main__":
    main()
