import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DENSENET_SOURCE_PATH } from "../lib/prose";
import { AddVsConcatToggle } from "../widgets/AddVsConcatToggle";
import { AddVsConcatDiagram } from "../widgets/AddVsConcatDiagram";
import { DenseBlockDiagram } from "../widgets/DenseBlockDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function DenseConnectivityStage({ intuitionProse, mechanism1Prose }: Props) {
  const [mode, setMode] = useState<"add" | "concat">("concat");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-2xl)", marginBottom: "var(--space-4)" }}>
        用 concat 替代 add
      </h2>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={DENSENET_SOURCE_PATH} />
          </div>

          <AddVsConcatToggle value={mode} onChange={setMode} />

          <p className={styles.caption}>
            切换看 ResNet 的加法(信息混叠,通道数不变)与 DenseNet 的拼接(信息保留,通道数翻倍)的区别。
          </p>

          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-8)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={DENSENET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <AddVsConcatDiagram mode={mode} />
          <div style={{ marginTop: "var(--space-8)" }}>
            <DenseBlockDiagram />
          </div>
        </div>
      </div>
    </div>
  );
}
