import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DEEPSEEK_V3_SOURCE_PATH } from "../lib/prose";
import { FineGrainedRoutingDiagram } from "../widgets/FineGrainedRoutingDiagram";
import { ParamActivationCompare } from "../widgets/ParamActivationCompare";
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

export function FineGrainedExpertsStage({ intuitionProse, mechanism1Prose }: Props) {
  const [granularity, setGranularity] = useState<"coarse" | "fine">("coarse");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Fine-Grained Experts + Shared Expert — 专家从粗变细
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Mixtral 8 个大 expert(top-2)已经证明开源 MoE 可行,但专精度有限。
        DeepSeek-V3 把 expert 切到 256 个更小的粒度(top-8),再加一个所有 token
        都走的 shared expert 承担通用能力 —— 路由组合空间从 C(8,2)=28 暴涨到
        C(256,8)≈10¹³,routed expert 才能真的专精。
      </p>

      <FineGrainedRoutingDiagram granularity={granularity} />
      <p className={styles.caption}>
        ↑ 切换看粗粒度(Mixtral 风格)vs 细粒度(V3 风格)专家布局;粉色 = 本次被路由选中的 expert。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setGranularity("coarse")} style={btnStyle(granularity === "coarse")}>粗粒度(Mixtral 8×top-2)</button>
        <button type="button" onClick={() => setGranularity("fine")} style={btnStyle(granularity === "fine")}>细粒度(V3 256×top-8 + shared)</button>
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <ParamActivationCompare />
        <p className={styles.caption}>
          ↑ 总参数(浅色)vs 每 token 激活参数(深色)。V3 总参数堪比超大 dense 模型,
          激活参数却只有 37B —— 细粒度专家让"容量"和"算力"脱钩。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={DEEPSEEK_V3_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={DEEPSEEK_V3_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              细粒度 vs 粗粒度
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.7, color: "var(--ink-primary)" }}>
              Mixtral 像"8 个大全能选手选 2 个",V3 像"256 个专才选 8 个"。
              更细的粒度 → 更容易 specialize 到具体模式(代码风格、数学步骤等),
              组合空间也大得多。
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              Shared expert 承担通用基础能力,routed expert 才能心无旁骛地专精。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
