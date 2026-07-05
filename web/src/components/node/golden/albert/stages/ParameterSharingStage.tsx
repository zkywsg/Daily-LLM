import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ALBERT_SOURCE_PATH } from "../lib/prose";
import { ParameterSharingDiagram } from "../widgets/ParameterSharingDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function ParameterSharingStage({ mechanism2Prose }: Props) {
  const [activeLayer, setActiveLayer] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:跨层参数共享
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        BERT-large 24 层每层都有独立的 attention + FFN 参数,是模型参数的大头
        (~75%)。ALBERT 怀疑深层 Transformer 不同层学到的是"相似变换的不同尺度"，
        于是让所有 24 层共享同一组权重 —— 只存 1 份,forward 时重复用 24 次。
      </p>

      <ParameterSharingDiagram activeLayer={activeLayer} />
      <p className={styles.caption}>
        ↑ 拖动滑块模拟"当前在算第几层"—— BERT 每层切换到不同的权重组,ALBERT 始终用同一组 W。
      </p>
      <input
        type="range"
        min={1}
        max={8}
        step={1}
        value={activeLayer}
        onChange={(evt) => setActiveLayer(Number(evt.target.value))}
        style={{ width: "100%", marginTop: 12 }}
        aria-label="当前层数滑块"
      />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={ALBERT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              省显存,不省时间
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              参数共享把 BERT-large 的 attention+FFN 参数从 288M 压到 12M(少 24×),
              但 forward 依然要走满 24 层 —— 每层都是同一组参数,计算量(FLOPs)完全不变。
              这正是 ALBERT 部署上不如 DistilBERT 受欢迎的根本原因。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
