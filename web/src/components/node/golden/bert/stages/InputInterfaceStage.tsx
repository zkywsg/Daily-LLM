import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { BERT_SOURCE_PATH } from "../lib/prose";
import { InputCompositionSVG } from "../widgets/InputCompositionSVG";
import { SpecialTokenRoles } from "../widgets/SpecialTokenRoles";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function InputInterfaceStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:[CLS]/[SEP]/Segment 统一输入接口
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        要让一个预训练模型适配分类、句对、序列标注全部任务,就需要一个统一的
        输入格式。BERT 用 [CLS] 占位"整段聚合"、[SEP] 分句、Segment emb 区分两段,
        加上 Position emb,三层 embedding 相加成最终输入。
      </p>

      <InputCompositionSVG />
      <p className={styles.caption}>
        ↑ 输入是三层 embedding 逐元素相加。[CLS] 在最前面,[SEP] 分两句,
        Segment A/B 让模型知道哪个 token 属于哪句 —— 句对任务必备。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={BERT_SOURCE_PATH} />
          </div>

          <h3
            style={{
              fontSize: "var(--fs-xl)",
              marginTop: "var(--space-8)",
              marginBottom: "var(--space-4)",
            }}
          >
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={BERT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <SpecialTokenRoles />
          <p className={styles.caption}>
            同一个 BERT 模型 → 三种下游任务,只需在 encoder 之后挂不同的 linear head。
            [CLS] 的输出向量是"句子级"任务的通用入口,token 级任务则用各自位置的输出。
          </p>
        </div>
      </div>
    </div>
  );
}
