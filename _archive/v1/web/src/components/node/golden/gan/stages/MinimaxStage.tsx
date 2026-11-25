import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAN_SOURCE_PATH } from "../lib/prose";
import { NonSaturatingComparison } from "../widgets/NonSaturatingComparison";
import { ModeCollapseDemo } from "../widgets/ModeCollapseDemo";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function MinimaxStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Minimax 交替优化 + Non-Saturating Loss
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        D 和 G 交替走梯度:每次 D 走一步把自己变更敏锐,然后 G 走一步把
        自己变更狡猾。最终博弈均衡是 p_g = p_data,此时 D 怎么训都给 0.5。
        但原版 G loss 会饱和 —— Goodfellow 同篇论文里就用 non-saturating
        trick 救回梯度,实际所有 GAN 实现都用它。
      </p>

      <NonSaturatingComparison />
      <p className={styles.caption}>
        ↑ 训练初期 G 很烂,D(G(z)) 接近 0(D 一眼识破)。原版 log(1−D)
        在此处梯度饱和 ≈ 0 → G 学不动。Non-saturating −log(D) 在此处梯度爆炸,
        G 反而能快速逃出"被识破"的状态。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GAN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ModeCollapseDemo />
          <p className={styles.caption}>
            切换"正常收敛 / mode collapse"看 GAN 两种典型结局。理想情况 G
            分布 = 真分布;失败情况 G 只学到部分模式,生成多样性丢失。
          </p>
        </div>
      </div>
    </div>
  );
}
