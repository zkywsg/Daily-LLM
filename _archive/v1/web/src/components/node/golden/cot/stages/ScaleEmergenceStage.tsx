import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { COT_SOURCE_PATH } from "../lib/prose";
import { ScaleEmergenceCurve } from "../widgets/ScaleEmergenceCurve";
import { SmallVsLargeComparison } from "../widgets/SmallVsLargeComparison";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ScaleEmergenceStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Scale 涌现 — CoT 在 ~100B 参数后才显著有效
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        CoT 不是免费午餐:小模型用 CoT 反而比 standard 更差,因为它推一步
        就跑偏。只有 ~62B 以上的模型才能稳定推完一条思维链 ——
        这是典型的"emergent ability",规模够大才解锁。
      </p>

      <ScaleEmergenceCurve />
      <p className={styles.caption}>
        ↑ 横轴 = 参数量(log)。粉色 = CoT,灰色 = Standard。两条曲线在
        ~62B 处分叉:此前 CoT 一直跟 Standard 持平甚至更差,此后陡升到 55%+。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={COT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={COT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <SmallVsLargeComparison />
          <p className={styles.caption}>
            同一道题,小模型 CoT 推到第 1 步就把"23"误算成"32",
            后面全错;大模型每步都对,稳定推出 9。CoT 需要的"内部正确性"
            是大模型才有的能力 —— 显式书写只是把它放大。
          </p>
        </div>
      </div>
    </div>
  );
}
