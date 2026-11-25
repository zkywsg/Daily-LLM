import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT2_SOURCE_PATH } from "../lib/prose";
import { PreVsPostLN } from "../widgets/PreVsPostLN";
import { SamplingDemo } from "../widgets/SamplingDemo";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

type Strategy = "greedy" | "temp" | "topk" | "topp";

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function PreLnSamplingStage({ mechanism3Prose, synergyProse }: Props) {
  const [lnSide, setLnSide] = useState<"post" | "pre" | "both">("both");
  const [strategy, setStrategy] = useState<Strategy>("temp");
  const [temp, setTemp] = useState(0.7);
  const [topK, setTopK] = useState(3);
  const [topP, setTopP] = useState(0.9);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Pre-LN + 改良初始化 — 让 48 层稳定训出来
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Post-LN 在深度 ≥ 24 层时梯度方差爆炸 / 训练发散。GPT-2 把 LN 移到 sublayer 输入
        前 (Pre-LN),残差成为 "纯通道" — 反传梯度沿残差直传不被 LN 挤压。
        加上残差权重 1/√N 缩放 + 上下文 512→1024 + 词表扩到 50257 + batch 524K token,
        共同让 1.5B/48 层第一次从随机初始化稳定收敛。
      </p>

      <PreVsPostLN side={lnSide} />
      <p className={styles.caption}>
        ↑ 左 Post-LN 把 LN 套在残差之后,深度大时梯度方差爆炸;
        右 Pre-LN 让残差直传,梯度方差近似不变。点按钮聚焦其中一边。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setLnSide("post")} style={btnStyle(lnSide === "post")}>Post-LN</button>
        <button type="button" onClick={() => setLnSide("pre")} style={btnStyle(lnSide === "pre")}>Pre-LN</button>
        <button type="button" onClick={() => setLnSide("both")} style={btnStyle(lnSide === "both")}>对比</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GPT2_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GPT2_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <SamplingDemo strategy={strategy} temperature={temp} topK={topK} topP={topP} />
          <p className={styles.caption}>
            ↑ GPT-2 续写时用什么采样?greedy = argmax 永远选 cat;
            temp 拉低更确定,拉高更发散;top-k / top-p 截断尾部避免选垃圾词。
            "独角兽报道" 那种自洽段落用的是 top-k=40 + temp=0.7。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              采样策略
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {(["greedy", "temp", "topk", "topp"] as const).map((s) => (
                <button key={s} type="button" onClick={() => setStrategy(s)} style={btnStyle(strategy === s)}>{s}</button>
              ))}
            </div>

            {strategy === "temp" && (
              <>
                <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "10px 0 4px" }}>
                  <span>temperature</span><strong>{temp.toFixed(2)}</strong>
                </label>
                <input type="range" min={0.1} max={2.0} step={0.05} value={temp} onChange={(e) => setTemp(parseFloat(e.target.value))} style={{ width: "100%" }} />
              </>
            )}
            {strategy === "topk" && (
              <>
                <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "10px 0 4px" }}>
                  <span>top-k</span><strong>{topK}</strong>
                </label>
                <input type="range" min={1} max={8} step={1} value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} style={{ width: "100%" }} />
              </>
            )}
            {strategy === "topp" && (
              <>
                <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "10px 0 4px" }}>
                  <span>top-p (nucleus)</span><strong>{topP.toFixed(2)}</strong>
                </label>
                <input type="range" min={0.1} max={1.0} step={0.05} value={topP} onChange={(e) => setTopP(parseFloat(e.target.value))} style={{ width: "100%" }} />
              </>
            )}
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 8, lineHeight: 1.4 }}>
              生产工程上 SD 默认 top-p=0.95 / GPT-2 用 top-k=40 / Llama 默认 temp=0.6 + top-p=0.9。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
