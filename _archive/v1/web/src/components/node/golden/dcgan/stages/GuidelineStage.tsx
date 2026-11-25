import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DCGAN_SOURCE_PATH } from "../lib/prose";
import { LatentSpaceDiagram } from "../widgets/LatentSpaceDiagram";
import { ARCHITECTURE_GUIDELINES } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function GuidelineStage({ mechanism3Prose, synergyProse }: Props) {
  const [t, setT] = useState(0.5);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:架构指南 6 条 + Latent Space 算术 — 让任何人都能复现
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        DCGAN 论文最具影响力的部分是给出 6 条明确的架构 guideline,把成功率从
        &lt;30% 拉到 &gt;90%。论文里的副产品 latent space 算术 —— 把训完的 G 的
        z 向量做加减能得到语义合成的结果 —— 第一次让人直观看到 G 学到了语义结构化
        的 latent 空间。
      </p>

      <LatentSpaceDiagram t={t} />
      <p className={styles.caption}>
        ↑ 拖动滑块看 z 空间中从 "neutral woman" 插值到 "smiling woman" 时,
        G(z) 生成的表情如何连续过渡 —— 证明 latent 空间是语义结构化的,而不是随机噪声的堆砌。
      </p>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={t}
        onChange={(e) => setT(Number(e.target.value))}
        style={{ width: "100%", marginTop: 8 }}
      />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DCGAN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DCGAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              架构指南 6 条
            </div>
            <ol style={{ margin: 0, paddingLeft: 18, fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", lineHeight: 1.7 }}>
              {ARCHITECTURE_GUIDELINES.map((g) => (
                <li key={g}>{g}</li>
              ))}
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
