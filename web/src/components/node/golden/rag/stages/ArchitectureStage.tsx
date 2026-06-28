import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { RAG_SOURCE_PATH } from "../lib/prose";
import { E2EvsPipeline } from "../widgets/E2EvsPipeline";
import { ArchitectureCompare } from "../widgets/ArchitectureCompare";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
  implementationProse: string;
}

export function ArchitectureStage({ mechanism3Prose, synergyProse, implementationProse }: Props) {
  const [mode, setMode] = useState<"e2e" | "pipeline">("pipeline");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:End-to-End vs Pipeline — 两条工程路线
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Lewis 2020 原版把 retriever (DPR) 和 generator (BART) 端到端联合训练
        —— retriever 学到的不是"通用语义相似",而是"对生成有用"。但工程复杂、
        必须自训。现代工业 RAG 把两者解耦,用现成 embed API + 现成 LLM 拼起来,
        上线快、可控性强,代价是 retriever 跟 generator 不能互相适配。
      </p>

      <E2EvsPipeline mode={mode} />
      <p className={styles.caption}>
        ↑ 切 toggle 看两种数据流。端到端有联合梯度回传(粉色虚线);
        Pipeline 各模块冻结,只调 prompt 和 k。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={RAG_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={RAG_SOURCE_PATH} />
          </div>

          {implementationProse && (
            <>
              <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
                实现细节
              </h3>
              <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
                <MarkdownRenderer markdown={implementationProse} sourcePath={RAG_SOURCE_PATH} />
              </div>
            </>
          )}
        </div>

        <div className={styles.stickyPanel}>
          <ArchitectureCompare />
          <p className={styles.caption}>
            两种路线在 retriever / generator / 训练 / 数据 / 代价 / 效果 6 维上的对比。
            学界从 Lewis 那条路出发,工业一路滑向 pipeline 因为 \"上线快\" 压倒 \"end-to-end optimal\"。
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
              架构路线
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {[
                { v: "e2e" as const, label: "Lewis 2020 端到端" },
                { v: "pipeline" as const, label: "现代工业 pipeline" },
              ].map((opt) => {
                const active = mode === opt.v;
                return (
                  <button
                    key={opt.v}
                    type="button"
                    onClick={() => setMode(opt.v)}
                    style={{
                      padding: "4px 12px",
                      fontSize: "var(--fs-sm)",
                      borderRadius: "var(--radius-sm)",
                      border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
                      background: active ? "var(--accent-link)" : "var(--bg-surface)",
                      color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
                      cursor: "pointer",
                    }}
                  >
                    {opt.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
