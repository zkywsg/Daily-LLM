import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GLOVE_SOURCE_PATH } from "../lib/prose";
import { TrainingPipeline } from "../widgets/TrainingPipeline";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

type Hl = "scan" | "train" | "reuse" | null;

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function GlobalMatrixStage({ mechanism3Prose, synergyProse }: Props) {
  const [hl, setHl] = useState<Hl>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:全局共现矩阵 — 一次扫语料替代每步采样
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Word2Vec 每步 SGD 都扫新 mini-batch 的窗口对,整个训练扫语料若干 epoch。
        GloVe 走完全不同的路线:预处理一次扫全语料构建 V×V 稀疏共现矩阵(带 distance
        weighting),训练只看这个矩阵。这种"全局统计 + 一次拟合"在 Common Crawl 840B
        token 规模上有显著工程优势。
      </p>

      <TrainingPipeline highlight={hl} />
      <p className={styles.caption}>
        ↑ 三步 pipeline:扫语料建矩阵 → AdaGrad 拟合 → 输出词向量。
        矩阵一次建好,训不同维度/超参不用重扫语料。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setHl(null)} style={btnStyle(hl === null)}>全部</button>
        <button type="button" onClick={() => setHl("scan")} style={btnStyle(hl === "scan")}>① 扫语料</button>
        <button type="button" onClick={() => setHl("train")} style={btnStyle(hl === "train")}>② 优化</button>
        <button type="button" onClick={() => setHl("reuse")} style={btnStyle(hl === "reuse")}>③ 输出</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GLOVE_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GLOVE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              Word2Vec vs GloVe 本质相同?
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              Levy & Goldberg 2014 证明:Word2Vec skip-gram NEG 数学上等价于隐式分解
              PMI 矩阵;GloVe 显式做 log X_ij 分解 — 两者本质都是矩阵分解,
              只是 framing 不同。GloVe 在大规模上略优,因为全局统计利用更充分。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
