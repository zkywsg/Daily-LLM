import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { MIXTRAL_SOURCE_PATH } from "../lib/prose";
import { TopKCompare } from "../widgets/TopKCompare";
import { ExpertParallelism } from "../widgets/ExpertParallelism";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function TopKParallelismStage({ mechanism3Prose, synergyProse }: Props) {
  const [k, setK] = useState(2);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Top-2 Routing + Expert Parallelism — 推理与训练的工程基础
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Top-1 (Switch) 算力最便宜但容易把 token 学错给\"错 expert\";
        Top-4+ 算力翻倍但收益边际。Mixtral 选 k=2 是 sweet spot:质量大涨 +
        算力仅 2× dense。配合 expert parallelism(每 GPU 放 1 expert),
        all-to-all 把 token 送到对应 GPU,这套工程才让 47B MoE 真的能上线。
      </p>

      <TopKCompare selectedK={k} />
      <p className={styles.caption}>
        ↑ 切 k 看质量/算力权衡。k=2 是 \"质量陡升 + 算力还能接受\" 的拐点。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={MIXTRAL_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={MIXTRAL_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ExpertParallelism />
          <p className={styles.caption}>
            Mixtral 的部署方式:每 GPU 放 1 个 expert(共 8 个),token 经
            router 决策后 all-to-all 跨 GPU 通信去对应 GPU 跑 FFN,
            结果再 all-to-all 收回原 GPU 加权求和。
          </p>
          <div
            style={{
              marginTop: "var(--space-4)",
              padding: "var(--space-3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
            }}
          >
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginBottom: 6,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              top-k 选项
            </div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {[1, 2, 4, 8].map((opt) => (
                <button key={opt} type="button" onClick={() => setK(opt)} style={btnStyle(k === opt)}>
                  top-{opt}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
