#!/usr/bin/env python3
"""扫描仓库根下的家族目录（NN-xxx/），收集每个节点 markdown 的 frontmatter，
生成按年份升序的 TIMELINE.md 速查表。

用法：
    python3 scripts/generate_timeline.py [--root .] [--out TIMELINE.md]
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


FAMILY_DIR_RE = re.compile(r"^\d{2}-[a-z0-9\-]+$")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

FAMILY_IDS = [
    "01-cnn", "02-rnn-lstm", "03-word-embedding", "04-gan",
    "05-transformer", "06-bert-family", "07-gpt-scaling",
    "08-vit", "09-multimodal-clip", "10-diffusion",
    "11-peft-lora", "12-rlhf-alignment", "13-moe-efficient",
    "14-rag-agent", "15-reasoning-o1-r1",
]


@dataclass
class Node:
    name: str
    year: int
    family: str
    order: int
    key_idea: str
    paper: str
    path: Path
    authors: list[str] = field(default_factory=list)


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
            authors_raw = fm.get("authors", "")
            if isinstance(authors_raw, list):
                authors_list = [str(a).strip() for a in authors_raw if str(a).strip()]
            elif isinstance(authors_raw, str):
                s = authors_raw.strip()
                if s.startswith("[") and s.endswith("]"):
                    authors_list = [
                        a.strip().strip('"').strip("'")
                        for a in s[1:-1].split(",")
                        if a.strip()
                    ]
                else:
                    authors_list = []
            else:
                authors_list = []
            nodes.append(Node(
                name=str(fm.get("name", "")),
                year=int(fm.get("year", 0)),
                family=str(fm.get("family", fam_dir.name)),
                order=int(fm.get("order", 0)),
                key_idea=str(fm.get("key_idea", "")),
                paper=str(fm.get("paper", "")),
                path=md.relative_to(root),
                authors=authors_list,
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
            authors_raw = fm.get("authors", "")
            if isinstance(authors_raw, list):
                authors_list = [str(a).strip() for a in authors_raw if str(a).strip()]
            elif isinstance(authors_raw, str):
                s = authors_raw.strip()
                if s.startswith("[") and s.endswith("]"):
                    authors_list = [
                        a.strip().strip('"').strip("'")
                        for a in s[1:-1].split(",")
                        if a.strip()
                    ]
                else:
                    authors_list = []
            else:
                authors_list = []
            nodes.append(Node(
                name=str(fm.get("name", "")),
                year=int(fm.get("year", 0)),
                family=str(fm.get("family", fam_dir.name)),
                order=int(fm.get("order", 0)),
                key_idea=str(fm.get("key_idea", "")),
                paper=str(fm.get("paper", "")),
                path=readme.relative_to(root),
                authors=authors_list,
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


def parse_family_readme(readme_path: Path) -> tuple[str, str]:
    """从家族 README.md 提取 H1 label 和 blockquote blurb。"""
    if not readme_path.exists():
        return "", ""
    text = readme_path.read_text(encoding="utf-8")
    label = ""
    blurb = ""
    for line in text.splitlines():
        line_stripped = line.strip()
        if not label and line_stripped.startswith("# "):
            label = line_stripped[2:].strip()
        if not blurb and line_stripped.startswith("> "):
            raw = line_stripped[2:].strip()
            raw = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)
            raw = re.sub(r"\{\{[^}]+\}\}", "", raw).strip()
            if raw and not raw.startswith("_") and raw != "（待补充）":
                blurb = raw
        if label and blurb:
            break
    return label, blurb


def collect_families(root: Path) -> list[dict]:
    """扫描 15 家族目录，输出 families.json 用的 list。"""
    nodes_by_family: dict[str, list[Node]] = {}
    for node in collect_nodes(root):
        nodes_by_family.setdefault(node.family, []).append(node)

    families: list[dict] = []
    for fid in FAMILY_IDS:
        fam_dir = root / fid
        readme = fam_dir / "README.md"
        label, blurb = parse_family_readme(readme)
        if not label:
            label = fid
        if not blurb:
            blurb = "（待补充）"

        fam_nodes = sorted(nodes_by_family.get(fid, []), key=lambda n: n.order)
        assets_dir = fam_dir / "assets"

        node_dicts = []
        for n in fam_nodes:
            file_stem = n.path.stem  # e.g. "02-alexnet"
            assets: list[str] = []
            if assets_dir.exists():
                for svg in sorted(assets_dir.glob(f"{file_stem}-*.svg")):
                    assets.append(str(svg.relative_to(root)))
            node_dicts.append({
                "name": n.name,
                "year": n.year,
                "family": n.family,
                "order": n.order,
                "paper": n.paper,
                "authors": n.authors,
                "key_idea": n.key_idea,
                "path": str(n.path),
                "assets": assets,
            })

        years = [n.year for n in fam_nodes]
        year_range = [min(years), max(years)] if years else None

        family_num = fid.split("-")[0]
        families.append({
            "id": fid,
            "label": label,
            "blurb": blurb,
            "yearRange": year_range,
            "colorToken": f"--family-{family_num}",
            "nodes": node_dicts,
        })
    return families


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="仓库根（默认当前目录）")
    parser.add_argument("--out", default="TIMELINE.md", help="TIMELINE.md 输出路径")
    parser.add_argument(
        "--families-out",
        default="web/src/data/families.json",
        help="families.json 输出路径",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    # 1. TIMELINE.md
    nodes = collect_nodes(root)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text(render(nodes), encoding="utf-8")
    print(f"wrote {out_path} ({len(nodes)} nodes)")

    # 2. families.json
    families = collect_families(root)
    families_path = Path(args.families_out)
    if not families_path.is_absolute():
        families_path = root / families_path
    families_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "families": families,
    }
    families_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total_nodes = sum(len(f["nodes"]) for f in families)
    print(
        f"wrote {families_path} ({len(families)} families, {total_nodes} nodes)"
    )


if __name__ == "__main__":
    main()
