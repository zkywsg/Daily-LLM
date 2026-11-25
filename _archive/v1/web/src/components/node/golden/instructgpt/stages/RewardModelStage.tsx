import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { INSTRUCTGPT_SOURCE_PATH } from "../lib/prose";
import { RLHFPipelineFlow } from "../widgets/RLHFPipelineFlow";
import { PreferenceRanking } from "../widgets/PreferenceRanking";
import { BradleyTerryFormula } from "../widgets/BradleyTerryFormula";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function RewardModelStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Reward Model — 用偏好排序学打分器
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        让 SFT 模型给同一个 prompt 输出 N=4-9 个 candidate,labeler 对它们
        排序;然后训一个独立的 reward model r_φ,让它的输出跟人类排序一致。
        关键是用 \"pairwise loss\" 学相对顺序,而不是绝对分数 —— 评分员
        \"6.5 vs 7.2\" 不可比,\"A 比 B 好\" 才稳定。
      </p>

      <RLHFPipelineFlow activeStage={1} />
      <p className={styles.caption}>
        ↑ 现在进入 RM 训练阶段。SFT 模型在这里只用来生成候选,不再更新。
      </p>

      <PreferenceRanking />
      <p className={styles.caption}>
        ↑ 一个真实例子:同一 prompt 4 个候选,labeler 排出 #1-#4 顺序。
        RM 的目标就是给同样的相对顺序(右侧 logit 分数)。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <BradleyTerryFormula />
          <p className={styles.caption}>
            Bradley-Terry pairwise loss:对每对 (winner, loser),最大化
            σ(r_w − r_l)。N 个 candidate 一次能拆 C(N, 2) 个 pair,
            数据效率高 —— 这是 OpenAI 选 N=4-9 排序而不是\"打绝对分\"的关键。
          </p>
        </div>
      </div>
    </div>
  );
}
