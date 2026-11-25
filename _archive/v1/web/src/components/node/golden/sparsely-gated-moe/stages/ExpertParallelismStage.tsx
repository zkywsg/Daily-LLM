import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SPARSELY_GATED_MOE_SOURCE_PATH } from "../lib/prose";
import { ExpertParallelismDiagram } from "../widgets/ExpertParallelismDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ExpertParallelismStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Expert Parallelism — 让算力真正省下来
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Top-K gating 只是数学上稀疏,要真的省算力,2048 个 expert 得分布到多块
        GPU 上并行跑。Shazeer 团队用 128 块 K40 GPU,每 GPU 装 16 个 expert,
        靠 all-to-all 通信把 token 送到目标 expert 所在的 GPU —— 这是 MoE
        训练史上第一个跑通的分布式工程方案。
      </p>

      <ExpertParallelismDiagram />
      <p className={styles.caption}>
        ↑ 每 GPU 先算本地 gating,决定哪些 token 要去哪个 expert;all-to-all
        把 token 发送到目标 GPU;expert 本地计算;反向 all-to-all 把结果送回
        原 GPU 加权求和。All-to-all 是密集通信,带宽和延迟是最大工程瓶颈。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={SPARSELY_GATED_MOE_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={SPARSELY_GATED_MOE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div
            style={{
              padding: "var(--space-4)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
              fontSize: "var(--fs-sm)",
              lineHeight: 1.8,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 8 }}>缺一不可</div>
            <div style={{ color: "var(--ink-secondary)", marginBottom: 8 }}>
              只有 <strong>Top-K Gating</strong>:训着训着塌缩到少数 expert,
              稀疏度高但模型容量浪费。
            </div>
            <div style={{ color: "var(--ink-secondary)", marginBottom: 8 }}>
              只有 <strong>Auxiliary Loss</strong>:没有 top-K 截断,所有
              expert 都要算——退回 dense 网络,稀疏失效。
            </div>
            <div style={{ color: "var(--ink-secondary)" }}>
              只有 <strong>Expert Parallelism</strong>:工程能跑通,但缺
              sparse gating 还是 dense 算力;缺 aux loss 还是塌缩。
            </div>
          </div>
          <p className={styles.caption}>
            三者首次组合,让 137B 参数模型用 LSTM-Big 1/3 算力跑出 30% 更好的
            perplexity —— 这是稀疏 MoE 第一次被证明真的 work。
          </p>
        </div>
      </div>
    </div>
  );
}
