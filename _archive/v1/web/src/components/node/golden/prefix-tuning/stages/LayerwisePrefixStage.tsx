import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { PREFIX_TUNING_SOURCE_PATH } from "../lib/prose";
import { KV_INJECTION } from "../lib/data";
import { PrefixKvInjectionDiagram } from "../widgets/PrefixKvInjectionDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function LayerwisePrefixStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Layer-wise Prefix on K/V
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Prefix Tuning 不改架构,只在<strong>每层</strong> attention 的 K/V 前面拼一段可学习的
        prefix —— m = {KV_INJECTION.prefixLen} 个"虚拟 token",像一段"持续在场
        的额外上下文"贯穿整个生成过程。关键限制:只加在 K/V,不加在 Q,避免
        改变 query 本身影响后续 token 的行为。
      </p>

      <PrefixKvInjectionDiagram />
      <p className={styles.caption}>
        ↑ K = [P_K, k₁...kₙ],V = [P_V, v₁...vₙ],Q 保持不变 —— prefix 只作为
        额外的 attend 目标出现,不改变 query 的计算方式。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={PREFIX_TUNING_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={PREFIX_TUNING_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么只加 K/V,不加 Q
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`Standard attention:
  K = [k_1, ..., k_n]
  V = [v_1, ..., v_n]
  Q · K^T → softmax → · V

Prefix Tuning attention:
  K = [P_K, k_1, ..., k_n]
  V = [P_V, v_1, ..., v_n]
  Q · K^T → softmax → · V
  (Q 不变)`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              如果 prefix 也加在 Q 上,相当于改变了 query 本身,后续每个 token
              生成 query 的方式都被扰动 —— 只加 K/V 让 prefix 的作用局限于
              "额外可 attend 的上下文",更稳定、更可控。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
