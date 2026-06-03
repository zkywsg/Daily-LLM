#!/usr/bin/env python3
"""把 markdown 里的旧路径替换为新路径，按当前文件深度动态计算相对路径。

把每个匹配视为：当前文件相对自身位置引用的目标。
1) 计算引用解析出的「绝对根路径」（root-relative）
2) 用旧→新映射表换成新的根路径
3) 重新计算相对当前文件的路径

用法：
    python3 scripts/rewrite_paths.py            # dry-run，打印替换
    python3 scripts/rewrite_paths.py --apply    # 写回文件
"""
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# foundations 子组归属
FOUND_SUBGROUP = {
    "linear-algebra": "math",
    "probability-information-theory": "math",
    "calculus": "math",
    "deep-learning-basics": "deep-learning",
    "backpropagation": "deep-learning",
    "activation-functions": "deep-learning",
    "loss-functions": "deep-learning",
    "normalization": "deep-learning",
    "optimization-scheduling": "deep-learning",
    "regularization": "deep-learning",
    "numerical-precision": "deep-learning",
    "embeddings": "representations",
    "tokenization": "representations",
    "softmax": "representations",
    "encoder-decoder": "structures",
    "attention-primer": "structures",
    "residual-connections": "structures",
    "inductive-bias": "structures",
}

# 顶层目录映射
TOP_RENAME = {
    "00-Timeline": "timeline",
    "01-Visual-Intelligence": "tracks/vision",
    "02-Language-Transformers": "tracks/language",
    "04-Alignment-OpenSource": "tracks/alignment",
    "05-Systems-Production": "tracks/systems",
    "06-Capstone-Projects": "projects",
}

OLD_PREFIXES = (
    "00-Prerequisites",
    "00-Timeline",
    "01-Visual-Intelligence",
    "02-Language-Transformers",
    "03-Scale-Multimodal",
    "04-Alignment-OpenSource",
    "05-Systems-Production",
    "06-Capstone-Projects",
)

# 把 root-relative 旧路径段映射为 root-relative 新路径段
def remap_old_to_new(old_root_path: str) -> str:
    parts = old_root_path.split("/")
    if not parts:
        return old_root_path
    head = parts[0]

    if head == "00-Prerequisites":
        if len(parts) < 2:
            return "foundations"
        topic = parts[1]
        subgroup = FOUND_SUBGROUP.get(topic)
        if subgroup is None:
            # unknown topic — leave under foundations/ root
            return "/".join(["foundations"] + parts[1:])
        return "/".join(["foundations", subgroup] + parts[1:])

    if head == "03-Scale-Multimodal":
        if len(parts) < 2:
            return "tracks/scale-multimodal"
        sub = parts[1]
        # subdir 'multimodal' 保持顶层；其余三类进 scale/
        if sub == "multimodal":
            return "/".join(["tracks", "scale-multimodal", "multimodal"] + parts[2:])
        if sub in {"pre-training", "prompt-engineering", "frameworks"}:
            return "/".join(["tracks", "scale-multimodal", "scale", sub] + parts[2:])
        return "/".join(["tracks", "scale-multimodal", sub] + parts[2:])

    new_head = TOP_RENAME.get(head)
    if new_head is None:
        return old_root_path
    return "/".join([new_head] + parts[1:])


# 匹配相对路径中的旧前缀，含可选的 `../` 前导段
OLD_RE = re.compile(
    r"((?:\.\.?/)*)("
    + "|".join(re.escape(p) for p in OLD_PREFIXES)
    + r")(/[^)\s\]]*?)?(?=[)\s\"'#?<>\]]|$)"
)


def rel_path_from(file_dir: Path, target_root_path: str) -> str:
    """从 file_dir（root-relative POSIX）回算到 target_root_path 的相对路径。"""
    abs_target = REPO_ROOT / target_root_path
    abs_from = REPO_ROOT / file_dir
    rel = os.path.relpath(abs_target, abs_from)
    return rel


def rewrite_text(text: str, file_path: Path) -> tuple[str, list[tuple[str, str]]]:
    """返回 (新文本, [(旧, 新), ...])

    策略：旧链接里嵌入的 `00-X/...` 等旧顶层名是「想到达哪个旧目录」的可靠标记，
    不再尝试物理解析（解析后是 broken 路径会失配）。
    直接：旧目标根路径 = <prefix>/<tail>；映射 → 新根路径；按当前文件位置回算相对路径。
    """
    file_dir_rel = file_path.parent.relative_to(REPO_ROOT)
    changes: list[tuple[str, str]] = []

    def replace(match: re.Match) -> str:
        full = match.group(0)
        prefix = match.group(2)
        tail = (match.group(3) or "").lstrip("/")

        # 旧目标根路径
        old_root = prefix + ("/" + tail if tail else "")
        new_root = remap_old_to_new(old_root)
        if new_root == old_root:
            return full

        had_trailing = full.endswith("/")

        # 新链接：root-relative 还是相对当前文件？
        # 顶层 docs（depth 0，root README/CLAUDE/...）保持 root-relative（无 ../ 前缀）
        if file_dir_rel == Path("."):
            new_rel = new_root
        else:
            new_rel = rel_path_from(file_dir_rel, new_root)

        if had_trailing and not new_rel.endswith("/"):
            new_rel += "/"

        changes.append((full, new_rel))
        return new_rel

    new_text = OLD_RE.sub(replace, text)
    return new_text, changes


def main():
    apply = "--apply" in sys.argv
    total = 0
    files_touched = 0

    SKIP_FILES = {"docs/restructure.md"}  # 是 old→new 说明书，本身不应被改

    md_files = [
        p
        for p in REPO_ROOT.rglob("*.md")
        if "node_modules" not in p.parts
        and ".git" not in p.parts
        and "dist" not in p.parts
        and "superpowers" not in p.parts  # 跳过历史归档
        and str(p.relative_to(REPO_ROOT)) not in SKIP_FILES
    ]

    for f in md_files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new_text, changes = rewrite_text(text, f)
        if changes:
            files_touched += 1
            total += len(changes)
            rel = f.relative_to(REPO_ROOT)
            print(f"\n=== {rel} ({len(changes)} changes) ===")
            for old, new in changes[:3]:
                print(f"  - {old}\n  + {new}")
            if len(changes) > 3:
                print(f"  ... +{len(changes) - 3} more")
            if apply:
                f.write_text(new_text, encoding="utf-8")

    print(f"\nTotal: {total} replacements in {files_touched} files")
    print("Run with --apply to write changes." if not apply else "Applied.")


if __name__ == "__main__":
    main()
