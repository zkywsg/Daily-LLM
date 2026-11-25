import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CONSTITUTIONAL_AI_SOURCE_PATH } from "../lib/prose";
import { RlaifVsRlhfDiagram } from "../widgets/RlaifVsRlhfDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function SlCaiStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:SL-CAI(Supervised Learning from AI Critiques)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        SFT 模型一开始"会回答任何问题",还不会拒绝有害请求。SL-CAI 让模型对着
        自己生成的有害回答反复做"按 constitution 批评 + 重写",通常迭代 4 轮,
        产物直接用于 SFT —— 把模型从"照单全收"调整到"会拒绝有害请求 + 给出
        无害替代",整个过程完全不需要人类介入。
      </p>

      <RlaifVsRlhfDiagram />
      <p className={styles.caption}>
        ↑ 人工标注流程(左)vs AI 标注流程(右)—— 同样产出约 30K 偏好对,
        人力成本从 $M 级压到 $K 级。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={CONSTITUTIONAL_AI_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              为什么要先做 SL-CAI 再进 RLAIF?
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              如果 SFT 模型直接进 RLAIF,它还不会拒绝有害请求 —— AI 自评时面对的候选
              回答"有害程度都差不多",偏好信号很弱。SL-CAI 先把模型调整到"会拒绝 +
              给无害替代",RLAIF 阶段的偏好打分才有区分度。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
