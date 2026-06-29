import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ALEXNET_SOURCE_PATH } from "../lib/prose";
import { DropoutMaskGrid } from "../widgets/DropoutMaskGrid";
import { TrainVsValGap } from "../widgets/TrainVsValGap";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function DropoutStage({ mechanism2Prose }: Props) {
  const [p, setP] = useState(0.5);
  const [seed, setSeed] = useState(3);
  const [withDropout, setWithDropout] = useState(true);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Dropout — 防止 60M 参数过拟合
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        AlexNet 58M 参数集中在两层 4096 FC,纸面上参数量比训练样本(1.2M)还多 50 倍。
        Dropout 每个 forward 随机关闭一半 FC unit,推理时全开 + 权重 × p —
        等价于 2^4096 个子网的几何平均 bagging,把 train/val gap 从 5% 压到 1-2%。
      </p>

      <DropoutMaskGrid p={p} seed={seed} />
      <p className={styles.caption}>
        ↑ 256 个 unit 缩略代表 4096 维 FC 层。粉色 = 这一步被 mask 掉,
        反传也不更新。换 seed 看每个 batch 都是完全不同的子网。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={ALEXNET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <TrainVsValGap withDropout={withDropout} />
          <p className={styles.caption}>
            ↑ AlexNet 论文对照实验:无 dropout 时 train 几乎到 5%,
            但 val 卡在 43% — 模型把 1.2M 张图背下来了。
            加 dropout 后 train 收敛慢一点,但 val 提高 6% 到 37%。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
              <button type="button" onClick={() => setWithDropout(true)} style={btnStyle(withDropout)}>有 Dropout</button>
              <button type="button" onClick={() => setWithDropout(false)} style={btnStyle(!withDropout)}>无 Dropout</button>
            </div>
            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: 4 }}>
              <span>drop 概率 p</span><strong>{p.toFixed(2)}</strong>
            </label>
            <input type="range" min={0} max={0.9} step={0.05} value={p} onChange={(e) => setP(parseFloat(e.target.value))} style={{ width: "100%" }} />
            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "8px 0 4px" }}>
              <span>随机 seed</span><strong>{seed}</strong>
            </label>
            <input type="range" min={1} max={50} step={1} value={seed} onChange={(e) => setSeed(parseInt(e.target.value))} style={{ width: "100%" }} />
          </div>
        </div>
      </div>
    </div>
  );
}
