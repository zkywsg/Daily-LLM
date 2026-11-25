import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ALEXNET_SOURCE_PATH } from "../lib/prose";
import { ActivationCurves } from "../widgets/ActivationCurves";
import { GradientDecayBars } from "../widgets/GradientDecayBars";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
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

export function ReluStage({ intuitionProse, mechanism1Prose }: Props) {
  const [showGrad, setShowGrad] = useState(true);
  const [act, setAct] = useState<"sigmoid" | "tanh" | "relu">("sigmoid");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:ReLU 取代 sigmoid/tanh — 让深层网络可训
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        sigmoid 在 |x|&gt;5 几乎完全饱和(导数 &lt; 0.01),8 层反向乘下来梯度
        衰减到 10⁻⁵,前几层学不动。AlexNet 换成 ReLU = max(0, x),
        正区间梯度恒为 1,8 层串起来仍是 1,训练速度还快 6×。
      </p>

      <ActivationCurves showGrad={showGrad} />
      <p className={styles.caption}>
        ↑ 切到 "导数" 看 sigmoid/tanh 饱和区(粉色高亮)— |x|&gt;5 几乎归零,
        ReLU 正区间整齐站在 1 上。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setShowGrad(false)} style={btnStyle(!showGrad)}>forward</button>
        <button type="button" onClick={() => setShowGrad(true)} style={btnStyle(showGrad)}>导数</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={ALEXNET_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={ALEXNET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <GradientDecayBars activation={act} />
          <p className={styles.caption}>
            ↑ 8 层反向梯度幅度(log 尺度)。sigmoid 平均 ×0.25 → 第 8 层
            ≈ 1.5×10⁻⁵;ReLU 不衰减。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              activation
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {(["sigmoid", "tanh", "relu"] as const).map((a) => (
                <button key={a} type="button" onClick={() => setAct(a)} style={btnStyle(act === a)}>
                  {a}
                </button>
              ))}
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 8, lineHeight: 1.4 }}>
              在 ImageNet 上同架构 ReLU 比 tanh 训练速度快约 6× — 1 行
              代码改动改变了深度学习十年。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
