import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FLOW_MATCHING_SOURCE_PATH } from "../lib/prose";
import { OBJECTIVE_COMPARE } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  synergyProse: string;
}

export function SynergyStage({ synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        三件套协同:直线路径 + 速度学习 + ODE 采样 缺一不可
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Flow Matching 能在 2023-2024 成为 diffusion 训练目标的现代化定型,不是单一改进,
        而是三件套同时调到协同点 —— 抽掉任何一件,剩下两件都撑不住"数学简洁 + 训练稳定
        + 采样高效"这个组合。
      </p>

      <div style={{ overflowX: "auto", marginBottom: "var(--space-6)" }}>
        <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "var(--fs-sm)" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", padding: "6px 10px", borderBottom: "2px solid var(--border)" }}></th>
              <th style={{ textAlign: "left", padding: "6px 10px", borderBottom: "2px solid var(--border)", color: "#ec4899" }}>DDPM</th>
              <th style={{ textAlign: "left", padding: "6px 10px", borderBottom: "2px solid var(--border)", color: "#3b82f6" }}>Flow Matching</th>
            </tr>
          </thead>
          <tbody>
            {OBJECTIVE_COMPARE.map((row) => (
              <tr key={row.dim}>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)", fontWeight: 600 }}>{row.dim}</td>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)", color: "var(--ink-secondary)" }}>{row.ddpm}</td>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)", color: "var(--ink-secondary)" }}>{row.flowMatching}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.caption}>↑ 数字来自源文档机制一的 DDPM vs Flow Matching 对比表。</p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={FLOW_MATCHING_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,协同就散了
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有直线路径,还学 ε</strong>:直线上的 ε 就是固定噪声 x₁ 本身,学到 trivial 解</li>
              <li><strong>只有速度学习,还用 schedule 曲线</strong>:任务复杂度回到 DDPM 水平,少步采样优势消失</li>
              <li><strong>只有直线 + 速度,还用 SDE 采样</strong>:随机性打折直线优势,大步 SDE 仍然质量崩</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
