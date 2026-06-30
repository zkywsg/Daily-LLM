import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { BAHDANAU_SOURCE_PATH } from "../lib/prose";
import { BottleneckCompare } from "../widgets/BottleneckCompare";
import { LengthBleuCurves } from "../widgets/LengthBleuCurves";
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

export function BottleneckStage({ intuitionProse, mechanism1Prose }: Props) {
  const [side, setSide] = useState<"seq2seq" | "bahdanau" | "both">("both");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Encoder 保留每个位置的 hidden state — 不再压成一个 c
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Seq2Seq 把整句压成单一 d 维向量 c · 长句 (T=60) 时一个 1000 维装不下所有语义,
        encoder 早期信号被后续输入覆盖。Bahdanau 反问:让 decoder 每步直接回头看 encoder
        所有时刻 h_1..h_T,信息载体从 O(d) 升到 O(T·d) — 长句质量回到与短句平行。
      </p>

      <BottleneckCompare highlight={side} />
      <p className={styles.caption}>
        ↑ 上 Seq2Seq 漏斗压缩 → 单一 c → decoder 每步都看同一个 c;
        下 Bahdanau 保留所有 h_i,decoder 每步对应不同 c_t(粗绿线表示该步主要聚焦的 h_i)。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setSide("seq2seq")} style={btnStyle(side === "seq2seq")}>聚焦 Seq2Seq</button>
        <button type="button" onClick={() => setSide("bahdanau")} style={btnStyle(side === "bahdanau")}>聚焦 Bahdanau</button>
        <button type="button" onClick={() => setSide("both")} style={btnStyle(side === "both")}>对比</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={BAHDANAU_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={BAHDANAU_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <LengthBleuCurves />
          <p className={styles.caption}>
            ↑ 论文 WMT'14 EN→FR 长度分桶 BLEU。固定 c (粉) 在 &gt;60 词时跌到 12;
            attention (绿) 在 &gt;60 词时仍有 24,只比短句低 4 点。
            这是序列建模在长上下文上的第一次真正突破。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              工程细节
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>Encoder 用 BiGRU 把每位置 h_i 编码成 <code>[→h_i ; ←h_i]</code>,同时含左右上下文</li>
              <li>所有 T 个 h_i 全部保留 — encoder 输出是 T×2d 矩阵而不是单向量</li>
              <li>单向 RNN 也能加 attention,只是少了右上下文 — 双向是实现选择</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
