import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { PREFIX_TUNING_SOURCE_PATH } from "../lib/prose";
import { FullFtVsPrefixCompareTable } from "../widgets/FullFtVsPrefixCompareTable";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function FrozenBaseCompareStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:冻结 base + 三件套协同
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Layer-wise 注入解决了"prefix 加在哪",MLP 重参数化解决了"怎么训得稳",
        但要真正省参数、防遗忘,还需要最后一步:Transformer 原参数完全冻结,
        梯度只流过 prefix encoder(P_small + MLP)。三个机制缺一不可,合起来
        才能让 0.1% 参数反超 Full FT。
      </p>

      <FullFtVsPrefixCompareTable />
      <p className={styles.caption}>
        ↑ Full FT vs Prefix Tuning:更新参数、推理开销、训练显存、多任务部署、
        灾难性遗忘 —— 五个维度的对比。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={PREFIX_TUNING_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={PREFIX_TUNING_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,都到不了"0.1% 反超"
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 Layer-wise</strong>:直接学全部层的 prefix 矩阵 → 优化 landscape 复杂,训练发散或收敛慢</li>
              <li><strong>只有 MLP 重参数化</strong>:只在 input 加 prefix(= Prompt Tuning)→ 小模型(&lt; 10B)效果跌很多</li>
              <li><strong>只有冻结 base</strong>:没有 prefix 也没有 adapter → 没引入任何任务相关参数,无法 fine-tune</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
