import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { MIXTRAL_SOURCE_PATH } from "../lib/prose";
import { ExpertLoadHistogram } from "../widgets/ExpertLoadHistogram";
import { LoadBalanceFormula } from "../widgets/LoadBalanceFormula";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function LoadBalanceStage({ mechanism2Prose }: Props) {
  const [balanced, setBalanced] = useState(false);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Load Balancing — 防止某些 expert 永远闲置
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        没约束的话 router 会很快学到\"少数几个 expert 一直最大\",其他 expert
        几乎收不到 token 永远在原地踏步 —— 47B 参数变成只用 \"热 expert\" 的
        几 B。一个 auxiliary loss 罚\"路由集中\",才能让所有 expert 真的参与训练。
      </p>

      <ExpertLoadHistogram balanced={balanced} />
      <p className={styles.caption}>
        ↑ 切换 toggle 看负载分布。不加 aux loss:E2/E3/E5/E7 几乎闲置;
        加 aux loss:8 个 expert 接近均匀。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={MIXTRAL_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <LoadBalanceFormula />
          <p className={styles.caption}>
            f_i · P_i 同时大才罚,逼 router 把概率散开;α=0.01 是平衡的 sweet spot ——
            这是 Switch Transformer 论文里的核心招式,Mixtral 直接继承。
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
              是否加 aux loss
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {[
                { v: false, label: "不加 → 不均衡" },
                { v: true, label: "加 → 均匀" },
              ].map((opt) => {
                const active = balanced === opt.v;
                return (
                  <button
                    key={String(opt.v)}
                    type="button"
                    onClick={() => setBalanced(opt.v)}
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
