import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FLAMINGO_SOURCE_PATH } from "../lib/prose";
import { ShotCurveChart } from "../widgets/ShotCurveChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function M3wStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:M3W 交错图文序列 — 让 ICL 涌现到多模态
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        训练数据核心是 MultiModal MassiveWeb(M3W)— 43M 网页,每个是
        "text image text image text..." 的自然交错序列。LLM 预训练时已学到
        "看到上下文模式,猜下一个 token" 的元学习能力;M3W 提供"看到几个 (图,文本)
        模式,预测下一对"的多模态版本,ICL 自然涌现到多模态。没有 M3W,即使 70B LLM 也学不到多模态 ICL。
      </p>

      <ShotCurveChart />
      <p className={styles.caption}>
        ↑ VQAv2 / OK-VQA / TextVQA 三个任务从 0-shot 到 32-shot 的提升曲线,
        清晰的 in-context learning 信号。
      </p>

      <div style={{ marginTop: "var(--space-6)", padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
          四个训练数据集混合
        </div>
        <table style={{ width: "100%", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>数据集</th>
              <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>描述</th>
              <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>规模</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: "1px solid var(--border)", background: "rgba(236,72,153,0.06)" }}>
              <td style={{ padding: "8px", fontWeight: 700, color: "#ec4899" }}>M3W(关键)</td>
              <td style={{ padding: "8px" }}>交错图文网页</td>
              <td style={{ padding: "8px" }}>43M 网页,185M 图</td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "8px" }}>ALIGN</td>
              <td style={{ padding: "8px" }}>短 caption 图文对</td>
              <td style={{ padding: "8px" }}>1.8B 对</td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "8px" }}>LTIP</td>
              <td style={{ padding: "8px" }}>长 caption 图文对</td>
              <td style={{ padding: "8px" }}>312M 对</td>
            </tr>
            <tr>
              <td style={{ padding: "8px" }}>VTP</td>
              <td style={{ padding: "8px" }}>短视频 + caption</td>
              <td style={{ padding: "8px" }}>27M 对</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={FLAMINGO_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={FLAMINGO_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              示例
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-secondary)", whiteSpace: "pre-wrap" }}>{`Example 1: [image of dog] -> "A photo of a dog."
Example 2: [image of cat] -> "A photo of a cat."
Example 3: [image of bird] -> "A photo of a bird."
Query:     [image of fish] -> ?

Flamingo: "A photo of a fish."
完全没在这种任务上训过,只学到了 prompt 格式`}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
