import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VGG_SOURCE_PATH } from "../lib/prose";
import { SeedRelayDiagram } from "../widgets/SeedRelayDiagram";
import { ParamDistributionChart } from "../widgets/ParamDistributionChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function PretrainSeedingStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:预训练浅版 seeding — 三件套协同
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        2014 年没有 BatchNorm,从随机初始化直接训 16/19 层 VGG 经常不收敛。VGG 的解决方案
        是接力训练:先训 VGG-11,再用其卷积权重初始化更深版本对应位置的层——这是 BN 出现前
        训深网的标准工程做法。
      </p>

      <SeedRelayDiagram />
      <p className={styles.caption}>
        ↑ VGG-11(随机初始化能收敛)→ 迁移卷积权重 → VGG-13 → VGG-16 → VGG-19,每代新增的层
        用随机初始化,在已有权重基础上继续训。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={VGG_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同 — 缺一不可
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={VGG_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              VGG 的遗产:fc6 占 74% 参数
            </div>
            <ParamDistributionChart />
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "var(--space-2) 0 0", lineHeight: 1.6 }}>
              138M 参数中 74% 集中在 fc6 一层,所有 13 层 conv 合计仅 11%——这正是
              Inception 用 global average pooling 干掉 fc 的直接动机。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
