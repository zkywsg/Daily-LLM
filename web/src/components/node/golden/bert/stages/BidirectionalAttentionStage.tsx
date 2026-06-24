import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { BERT_SOURCE_PATH } from "../lib/prose";
import { AttentionMaskCompare } from "../widgets/AttentionMaskCompare";
import { BidirectionalDemo } from "../widgets/BidirectionalDemo";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

const DEMO_TOKENS = ["The", "cat", "sat", "on", "the", "mat"];

export function BidirectionalAttentionStage({
  intuitionProse,
  mechanism1Prose,
}: Props) {
  const [queryIdx, setQueryIdx] = useState(2); // "sat"

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Encoder + 双向 self-attention
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        GPT 用 decoder 因果 mask,每个 token 只看左侧;BERT 用 encoder
        放开 mask,每个 token 同时看左右上下文 —— 这是为什么 BERT 适合
        理解任务(分类、QA、NER),而 GPT 更适合生成。
      </p>

      <AttentionMaskCompare tokens={DEMO_TOKENS} />
      <p className={styles.caption}>
        ↑ 同一句子,两种 attention mask 矩阵对比。BERT 全连接(粉色)、
        GPT 下三角(深灰处被掩掉)。注意 BERT 不能从左到右生成 —— 训练时
        看不到的右侧上下文在推理时也不该用。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={BERT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={BERT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <BidirectionalDemo
            tokens={DEMO_TOKENS}
            queryIdx={queryIdx}
            onQueryIdxChange={setQueryIdx}
          />
          <p className={styles.caption}>
            点击任意 token 切换 query,看它的 attention 弧线 ——
            蓝色指向左侧(过去),粉色指向右侧(未来)。这正是 BERT 比 GPT
            多出的"右侧上下文"信息源。
          </p>
        </div>
      </div>
    </div>
  );
}
