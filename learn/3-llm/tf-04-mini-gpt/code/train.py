"""训练迷你 GPT。运行：python train.py（约几十秒到几分钟，取决于机器）
输出 ../demo/model.js：权重、训练曲线、训练过程中的生成样例，供互动页直接使用。
python train.py --ablate 做对照实验：分别去掉残差 / 位置向量 / 因果掩码。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from corpus import entropy_floor, make_text, vocab
from minigpt import AdamW, MiniGPT, softmax

CHARS = vocab()
STOI = {c: i for i, c in enumerate(CHARS)}
encode = lambda s: np.array([STOI[c] for c in s])
decode = lambda ids: "".join(CHARS[i] for i in ids)
train_ids, val_ids = encode(make_text(6000, seed=1)), encode(make_text(600, seed=2))


def batch(data, rng, B, n):
    i = rng.integers(0, len(data) - n - 1, B)
    return np.stack([data[k:k + n] for k in i]), np.stack([data[k + 1:k + n + 1] for k in i])


def val_loss(m, n_batches=20):
    """固定的验证批次（来自另一份随机种子造的文本），只算损失不更新。"""
    rng = np.random.default_rng(123)
    out = []
    for _ in range(n_batches):
        x, y = batch(val_ids, rng, 64, m.cfg["ctx"])
        P = softmax(m.forward(x)[0])
        out.append(-np.log(np.take_along_axis(P, y[..., None], -1) + 1e-12).mean())
    return float(np.mean(out))


def sample(m, prompt, n=28, temperature=0.0, seed=0):
    return decode(m.generate(encode(prompt), n, temperature, np.random.default_rng(seed)))


def train(steps=3000, seed=0, log=False, **kw):
    m = MiniGPT(len(CHARS), seed=seed, **kw)
    opt = AdamW(m.p, lr=3e-3)
    rng = np.random.default_rng(seed + 100)
    curve, snaps = [], []
    marks = {0, 50, 100, 200, 400, 800, 1500, steps}
    for step in range(steps + 1):
        if step % 50 == 0 or step in marks:
            v = val_loss(m)
            curve.append([step, round(v, 4)])
        if step in marks:
            snaps.append({"step": step, "val": round(v, 3),
                          "texts": {p: sample(m, p, 24) for p in ("小猫", "熊猫住在", "兔子吃了")}})
            if log:
                print(f"step {step:5d}  验证损失 {v:.3f}  小猫 → {snaps[-1]['texts']['小猫']}")
        if step == steps:
            break
        x, y = batch(train_ids, rng, 32, m.cfg["ctx"])
        _, G = m.loss_and_grads(x, y)
        lr_scale = min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps)))
        opt.step(m.p, G, lr_scale)
    return m, curve, snaps


def accuracy_on_facts(m):
    """知识题：给出“X喜欢吃 / 住在 / 最爱 / 吃了”，贪心续写是否和规则一致。"""
    from corpus import ANIMALS, TEMPLATES, sentence
    ok = total = 0
    for s in ANIMALS:
        for t in range(len(TEMPLATES)):
            full = sentence(s, t)
            slot = {0: "喜欢吃", 1: "住在", 2: "最爱", 3: "吃了"}[t]
            head = s + slot
            got = sample(m, "。" + head, len(full) - len(head))[1:]
            ok += got == full
            total += 1
    return ok, total


def main():
    from optimal import optimal_loss
    floor, best = entropy_floor(), optimal_loss()
    if "--ablate" in sys.argv:
        rows = []
        cases = [("完整模型", {}), ("去掉残差", {"residual": False}), ("去掉位置向量", {"use_pos": False}),
                 ("去掉因果掩码", {"causal": False}),
                 ("6 层 · 有残差", {"n_layers": 6}), ("6 层 · 无残差", {"n_layers": 6, "residual": False})]
        for name, kw in cases:
            runs = []
            for seed in (0, 1, 2):
                m, curve, _ = train(steps=1500, seed=seed, **kw)
                ok, total = accuracy_on_facts(m)
                runs.append((curve[-1][1], ok, total, curve, sample(m, "小猫", 24)))
            vals = [r[0] for r in runs]
            rows.append({"name": name, "val_mean": round(float(np.mean(vals)), 3), "val_min": min(vals), "val_max": max(vals),
                         "correct": [r[1] for r in runs], "total": runs[0][2], "curve": runs[0][3], "sample": runs[0][4]})
            print(f"{name:12s} 验证损失 {np.mean(vals):.3f} [{min(vals):.3f}, {max(vals):.3f}]  知识题 {[r[1] for r in runs]}/{runs[0][2]}  {runs[0][4]}")
        out = Path(__file__).parent.parent / "demo" / "ablation.js"
        out.write_text("window.ABLATION=" + json.dumps({"floor": floor, "optimal": best, "steps": 1500, "seeds": 3, "rows": rows}, ensure_ascii=False) + ";\n", encoding="utf-8")
        return
    t0 = time.time()
    m, curve, snaps = train(steps=3000, log=True)
    ok, total = accuracy_on_facts(m)
    secs = time.time() - t0
    print(f"参数量 {m.n_params()}，用时 {secs:.0f} 秒，最终验证损失 {curve[-1][1]:.4f}（同样批次上的最优值 {best:.4f}），知识题 {ok}/{total}")
    test = encode("小猫喜欢吃鱼。熊猫住在")
    ref = m.forward(test[None, :])[0][0, -1]
    out = Path(__file__).parent.parent / "demo" / "model.js"
    out.write_text("window.MODEL=" + json.dumps({
        "cfg": m.cfg, "chars": CHARS, "n_params": m.n_params(), "floor": floor, "optimal": round(best, 4), "seconds": round(secs),
        "final_val": curve[-1][1], "facts": [ok, total], "curve": curve, "snapshots": snaps,
        "check": {"text": decode(test), "logits": [round(float(v), 4) for v in ref]},
        "weights": {k: [round(float(v), 5) for v in w.ravel()] for k, w in m.p.items()},
    }, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("已写入", out)


if __name__ == "__main__":
    main()
