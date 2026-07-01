import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FLAMINGO_SOURCE_PATH } from "../lib/prose";
import { PerceiverResamplerDiagram } from "../widgets/PerceiverResamplerDiagram";
import { GatedAttentionTrainCurve } from "../widgets/GatedAttentionTrainCurve";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
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

export function BridgeStage({ mechanism2Prose }: Props) {
  const [numPatches, setNumPatches] = useState(12);
  const [step, setStep] = useState(3000);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Perceiver Resampler + 间隔 Gated Cross-Attention
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        视觉编码器输出的 patch 数量随分辨率/帧数变化,Perceiver Resampler 用 64 个可学习
        query token 通过多层 cross-attention "采样"任意大小的视觉特征,输出永远固定 64 tokens。
        LLM 内部每隔 7 层插入一个新 cross-attention 层,用 tanh(α) 门控让训练初期完全是 identity,
        避免视觉模块破坏 LLM 原始能力。
      </p>

      <PerceiverResamplerDiagram numPatches={numPatches} />
      <p className={styles.caption}>
        ↑ 切换输入 patch 数量,输出始终是固定 64 tokens。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
        {[6, 12, 24].map((n) => (
          <button key={n} type="button" onClick={() => setNumPatches(n)} style={btnStyle(numPatches === n)}>{n} patches</button>
        ))}
      </div>

      <GatedAttentionTrainCurve currentStep={step} />
      <p className={styles.caption}>
        ↑ tanh(α) 初始 0,训练初期 Flamingo forward 严格等于原 LLM;
        1K-10K step 逐步学到非零,视觉影响才开启。拖动看训练进程。
      </p>
      <input type="range" min={0} max={15000} step={100} value={step}
             onChange={(e) => setStep(parseInt(e.target.value))} style={{ width: "100%", marginTop: 8 }} />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={FLAMINGO_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              公式
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)" }}>{`x ← x + tanh(α_attn)·CrossAttn(x, V)
      + tanh(α_ffn)·FFN(x)

α 初始化 = 0 → tanh(0) = 0
→ 训练初期完全 identity`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              这一 gated 初始化设计后被 LoRA / Adapter / Prompt Tuning 等 PEFT 工作沿用。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
