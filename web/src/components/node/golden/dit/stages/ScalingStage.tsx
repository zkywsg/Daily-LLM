import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DIT_SOURCE_PATH } from "../lib/prose";
import { ScalingCurve } from "../widgets/ScalingCurve";
import { DIT_SIZES } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function ScalingStage({ mechanism3Prose, synergyProse }: Props) {
  const [hIdx, setHIdx] = useState(3); // 默认高亮 XL

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Scaling — DiT-S/B/L/XL 干净的幂律
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        DiT 论文最有价值的实证不是 SOTA 数字,而是第一次给 diffusion 提供了像 LLM 那样
        干净的 scaling curve。S → B → L → XL 在 log(Gflops) - FID 平面上几乎线性,
        意味着 "按算力预算选模型" 工程方法论可以迁移过来 — 不需要再手工调 U-Net 通道。
      </p>

      <ScalingCurve highlightIdx={hIdx} />
      <p className={styles.caption}>
        ↑ FLOPs 横轴 log,FID 纵轴线性。点切换看每个 size 的具体数字。
        DiT-XL 是论文中最大,继续 scale 还有改善空间 — 直接说服行业投资百亿级 diffusion。
      </p>

      <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          高亮 size
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          <button type="button" onClick={() => setHIdx(-1)} style={btnStyle(hIdx === -1)}>全部</button>
          {DIT_SIZES.map((s, i) => (
            <button key={i} type="button" onClick={() => setHIdx(i)} style={btnStyle(hIdx === i)}>
              {s.name} · {s.paramsM}M
            </button>
          ))}
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DIT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DIT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么 Transformer 比 U-Net 更适合 diffusion?
            </div>
            <ol style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 18, lineHeight: 1.7, margin: 0 }}>
              <li><strong>无归纳偏置 = 优势</strong>:CNN local+multi-scale 偏置在大数据上反成限制,Transformer 从数据自己学 patch attention</li>
              <li><strong>adaLN-Zero 比 FiLM 灵活</strong>:逐层 γ/β/α 细粒度控制 · U-Net 的 timestep emb 只在通道</li>
              <li><strong>scaling 可预测</strong>:LLM 工程团队已知 "Transformer + 数据 + 算力 → loss 降",直接套用</li>
            </ol>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, fontStyle: "italic", lineHeight: 1.5 }}>
              ImageNet 256 SOTA:DiT-XL/2 FID 2.27 · 同等 FLOPs 下比 ADM (U-Net) 3.94 低 40%。
              这条幂律直接催生 Sora / SD3 / FLUX 等百亿级 diffusion。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
