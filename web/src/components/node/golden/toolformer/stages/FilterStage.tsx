import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { TOOLFORMER_SOURCE_PATH } from "../lib/prose";
import { PerplexityFilterDiagram } from "../widgets/PerplexityFilterDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function FilterStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Perplexity 过滤 — 用 loss 差当筛子
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        没有人工标注,怎么判断一个 tool call 是"有用的"?Toolformer 的答案
        极其优雅:看插入它之后,后续 token 的 perplexity 是否真的下降。
        只有当 L⁻(不带调用)- L⁺(带调用)超过阈值,才保留这条样本 —
        API 调用必须真的帮模型预测后续 token,不然就是"无用的装饰"。
      </p>

      <PerplexityFilterDiagram />
      <p className={styles.caption}>
        ↑ 四个候选调用的 loss 对比 — "今天天气?"的调用没有实质降低 loss,被丢弃。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={TOOLFORMER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              这一步是自监督的关键
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              LM 自己当裁判,完全不依赖人工判断"这个调用有没有用"。这个
              filter 让训练数据自然只保留高质量调用,把 Step 1 生成的
              海量嘈杂候选压缩成一份干净、高信号的微调数据集。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
