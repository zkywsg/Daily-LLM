import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DENSENET_SOURCE_PATH } from "../lib/prose";
import { BottleneckCompressionDiagram } from "../widgets/BottleneckCompressionDiagram";
import { ParamEfficiencyChart } from "../widgets/ParamEfficiencyChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function TransitionLayerStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-2xl)", marginBottom: "var(--space-4)" }}>
        Bottleneck + Compression(DenseNet-BC)
      </h2>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DENSENET_SOURCE_PATH} />
          </div>

          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-8)" }}>
            <h3 style={{ fontFamily: "var(--font-sans)", fontSize: "var(--fs-lg)", marginBottom: "var(--space-3)" }}>
              三件套协同
            </h3>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DENSENET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <BottleneckCompressionDiagram />
          <div style={{ marginTop: "var(--space-8)" }}>
            <ParamEfficiencyChart />
          </div>
        </div>
      </div>
    </div>
  );
}
