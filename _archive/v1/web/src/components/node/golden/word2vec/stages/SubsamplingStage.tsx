import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WORD2VEC_SOURCE_PATH } from "../lib/prose";
import { SubsamplingCurve } from "../widgets/SubsamplingCurve";
import { SubsampledSentence } from "../widgets/SubsampledSentence";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const T_OPTIONS: Array<{ label: string; t: number }> = [
  { label: "t = 1e-3 (温和)", t: 1e-3 },
  { label: "t = 1e-4", t: 1e-4 },
  { label: "t = 1e-5 (Mikolov 默认)", t: 1e-5 },
  { label: "t = 1e-6 (激进)", t: 1e-6 },
];

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function SubsamplingStage({ mechanism3Prose, synergyProse }: Props) {
  const [tIdx, setTIdx] = useState(2);
  const [seed, setSeed] = useState(7);
  const t = T_OPTIONS[tIdx].t;

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Subsampling 高频词 — 把算力花在该花的地方
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        "the / of / a / is" 信息量几乎为零,但和任何词都共现,会把所有 embedding
        拖向无意义中心。Word2Vec 对每个词按
        P_discard = 1 − √(t / f) 随机丢弃 — 高频词大量被跳过,
        训练加速 2-10×,罕见词质量提升。
      </p>

      <SubsampledSentence t={t} seed={seed} />
      <p className={styles.caption}>
        ↑ 同一句话过 subsampling 前 vs 后。粉色 = 被丢的高频词。
        换 seed 看不同丢弃结果,丢的是不同位置但概率分布稳定。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={WORD2VEC_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={WORD2VEC_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <SubsamplingCurve t={t} />
          <p className={styles.caption}>
            ↑ 曲线是丢弃概率随 freq 的变化。低于阈值 t (蓝虚线) 的词全保留;
            "the" 在 5.6% 频率下,丢弃概率 &gt; 98%。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              threshold t
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {T_OPTIONS.map((opt, i) => (
                <button key={i} type="button" onClick={() => setTIdx(i)} style={btnStyle(i === tIdx)}>{opt.label}</button>
              ))}
            </div>
            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "10px 0 4px" }}>
              <span>随机 seed</span><strong>{seed}</strong>
            </label>
            <input type="range" min={1} max={50} step={1} value={seed} onChange={(e) => setSeed(parseInt(e.target.value))} style={{ width: "100%" }} />
          </div>
        </div>
      </div>
    </div>
  );
}
