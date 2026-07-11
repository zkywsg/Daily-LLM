import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SELF_CONSISTENCY_SOURCE_PATH } from "../lib/prose";
import { MajorityVoteDiagram } from "../widgets/MajorityVoteDiagram";
import { AccuracyVsSamplesChart } from "../widgets/AccuracyVsSamplesChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function MajorityVoteStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Marginalize over r — 多数投票
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        推理过程 r 只是中间变量,我们真正关心的是答案 y。Self-Consistency 把
        argmax P(y, r | x) 改成 argmax Σ_r P(y, r | x) —— 用采样近似,
        统计每个唯一答案出现的次数,选出现最多的那个。三个机制协同后,
        GSM8K 从单次 56.5% 推到 N=40 投票的 74.4%,完全免费、无需重训。
      </p>

      <MajorityVoteDiagram />
      <p className={styles.caption}>
        ↑ 5 条采样路径中 4 条得到 196、1 条得到 168(算错),多数投票选出 196。
      </p>

      <div style={{ marginTop: "var(--space-8)" }}>
        <AccuracyVsSamplesChart />
        <p className={styles.caption}>
          ↑ 采样次数 N 越大,准确率越高,但呈对数线性增长、收益递减 ——
          第一次系统化展示了 test-time compute 是一条新的 scaling 轴。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={SELF_CONSISTENCY_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={SELF_CONSISTENCY_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              少一环都不行
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.8, color: "var(--ink-primary)" }}>
              <div>只有 Temperature → 答案不归一 → 投票分裂,不升反降</div>
              <div>只有 Normalization → T=0 全同一路径 → 等于 N 票同一答案</div>
              <div>只有多数投票 → 没有多样性也没归一化 → 投票空转</div>
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              三者首次组合,打开了 test-time compute scaling 这条新轴,直接影响后续 o1 / R1。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
