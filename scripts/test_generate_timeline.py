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


import json
from pathlib import Path


def test_generate_families_json(tmp_path: Path) -> None:
    """families.json 含 15 个家族 ID，已存在的家族被正确解析，其余为占位。"""
    fam_cnn = tmp_path / "01-cnn"
    fam_cnn.mkdir()
    (fam_cnn / "README.md").write_text(
        "# CNN 卷积神经网络\n\n> **测试一句话定位**\n", encoding="utf-8"
    )
    write_node(fam_cnn, "02-alexnet.md", {
        "name": "AlexNet",
        "year": 2012,
        "family": "01-cnn",
        "order": 2,
        "paper": "ImageNet Classification with Deep CNNs",
        "key_idea": "深 CNN 第一次跑赢手工特征",
    })
    assets = fam_cnn / "assets"
    assets.mkdir()
    (assets / "02-alexnet-arch.svg").write_text("<svg></svg>", encoding="utf-8")

    fam_empty = tmp_path / "02-rnn-lstm"
    fam_empty.mkdir()
    (fam_empty / "README.md").write_text(
        "# RNN / LSTM / GRU 循环网络\n", encoding="utf-8"
    )

    out_md = tmp_path / "TIMELINE.md"
    out_json = tmp_path / "web/src/data/families.json"

    script = Path(__file__).resolve().parent / "generate_timeline.py"
    subprocess.run(
        ["python3", str(script), "--root", str(tmp_path),
         "--out", str(out_md), "--families-out", str(out_json)],
        capture_output=True, text=True, check=True,
    )

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "generatedAt" in data
    assert isinstance(data["families"], list)
    assert len(data["families"]) == 15

    by_id = {f["id"]: f for f in data["families"]}

    # 01-cnn 真实数据
    cnn = by_id["01-cnn"]
    assert cnn["label"] == "CNN 卷积神经网络"
    assert cnn["blurb"] == "测试一句话定位"
    assert cnn["yearRange"] == [2012, 2012]
    assert cnn["colorToken"] == "--family-01"
    assert len(cnn["nodes"]) == 1
    node = cnn["nodes"][0]
    assert node["name"] == "AlexNet"
    assert node["year"] == 2012
    assert node["path"] == "01-cnn/02-alexnet.md"
    assert "01-cnn/assets/02-alexnet-arch.svg" in node["assets"]

    # 02-rnn-lstm 空但有 README
    rnn = by_id["02-rnn-lstm"]
    assert rnn["label"] == "RNN / LSTM / GRU 循环网络"
    assert rnn["yearRange"] is None
    assert rnn["nodes"] == []
    assert rnn["colorToken"] == "--family-02"

    # 13 个不存在的家族：仍有 entry，但 label 为 id 保底、blurb "（待补充）"
    others = [f for f in data["families"] if f["id"] not in ("01-cnn", "02-rnn-lstm")]
    assert len(others) == 13
    for f in others:
        assert f["nodes"] == []
        assert f["yearRange"] is None
