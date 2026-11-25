import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { EFFICIENTNET_SOURCE_PATH } from "../lib/prose";
import { RegularizationScheduleChart } from "../widgets/RegularizationScheduleChart";
import { ModelFamilyChart } from "../widgets/ModelFamilyChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function RegularizationStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:大模型同步加正则 — Dropout / Stochastic Depth 按 φ 线性涨
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        模型放大时,正则强度必须同步增加。B0 dropout=0.2、B7=0.5;stochastic depth
        丢弃率 B0=0.0、B4=0.2、B7=0.2,按 block 深度线性 schedule。论文消融显示 B7
        关闭 stochastic depth 时 Top-1 下降 0.7-1.0 个百分点。
      </p>

      <RegularizationScheduleChart />
      <p className={styles.caption}>
        ↑ Dropout / Stochastic Depth 都随模型规模(B0→B7)线性上涨 —— 正则强度不是
        "backbone 超参",而是"模型规模函数"。这一超参在 2019 年常被第三方复现忽略,
        导致早期 torchvision 的 B5-B7 比论文低约 1 个点。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={EFFICIENTNET_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={EFFICIENTNET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              B0-B7 帕累托线(拖动查看)
            </div>
            <ModelFamilyChart />
          </div>
        </div>
      </div>
    </div>
  );
}
