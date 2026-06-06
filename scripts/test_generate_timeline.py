"""测试 generate_timeline.py 能正确扫描家族节点 frontmatter 并生成 TIMELINE.md。"""
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path


def write_node(dir_path: Path, filename: str, frontmatter: dict) -> None:
    lines = ["---"]
    for k, v in frontmatter.items():
        if isinstance(v, list):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {v!r}" if isinstance(v, str) else f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append("# body")
    (dir_path / filename).write_text("\n".join(lines), encoding="utf-8")


def test_generate_timeline_basic(tmp_path: Path) -> None:
    fam = tmp_path / "01-cnn"
    fam.mkdir()
    (fam / "README.md").write_text("# CNN\n", encoding="utf-8")
    write_node(fam, "02-alexnet.md", {
        "name": "AlexNet",
        "year": 2012,
        "family": "01-cnn",
        "order": 2,
        "paper": "ImageNet Classification with Deep CNNs",
        "key_idea": "卷积破冰",
    })
    write_node(fam, "05-resnet.md", {
        "name": "ResNet",
        "year": 2015,
        "family": "01-cnn",
        "order": 5,
        "paper": "Deep Residual Learning",
        "key_idea": "残差连接",
    })

    script = Path(__file__).resolve().parent / "generate_timeline.py"
    result = subprocess.run(
        ["python3", str(script), "--root", str(tmp_path), "--out", str(tmp_path / "TIMELINE.md")],
        capture_output=True, text=True, check=True,
    )

    out = (tmp_path / "TIMELINE.md").read_text(encoding="utf-8")
    assert out.index("2012") < out.index("2015")
    assert "AlexNet" in out
    assert "ResNet" in out
    assert "01-cnn" in out
    assert "卷积破冰" in out
    assert "残差连接" in out


def test_generate_timeline_ignores_readmes(tmp_path: Path) -> None:
    fam = tmp_path / "02-rnn-lstm"
    fam.mkdir()
    (fam / "README.md").write_text("# RNN\n", encoding="utf-8")
    write_node(fam, "01-lstm.md", {
        "name": "LSTM", "year": 1997, "family": "02-rnn-lstm",
        "order": 1, "paper": "...", "key_idea": "门控",
    })

    script = Path(__file__).resolve().parent / "generate_timeline.py"
    subprocess.run(
        ["python3", str(script), "--root", str(tmp_path), "--out", str(tmp_path / "TIMELINE.md")],
        capture_output=True, text=True, check=True,
    )
    out = (tmp_path / "TIMELINE.md").read_text(encoding="utf-8")
    assert out.count("LSTM") == 1
