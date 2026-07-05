import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SWITCH_TRANSFORMER_SOURCE_PATH } from "../lib/prose";
import { LoadBalancingDiagram } from "../widgets/LoadBalancingDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
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

export function LoadBalancingStage({ mechanism2Prose }: Props) {
  const [balanced, setBalanced] = useState(false);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:f·P Load Balancing Loss — 均衡的几何形式
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Top-1 是把双刃剑:路由简化了,但每个 expert 的"接客数"波动更大,
        稍有不均就会塌缩到几个热门 expert。Switch Transformer 用一个更简洁的
        辅助 loss —— L_aux = α·N·Σ f_i·P_i —— 让"被分配多少"(f_i)和
        "被想分配多少"(P_i)同时大才罚,逼 router 把概率真正散开。
      </p>

      <LoadBalancingDiagram balanced={balanced} />
      <p className={styles.caption}>
        ↑ 切换 toggle 看负载分布。不加 aux loss:top-1 路由极易塌缩到 2-3 个热门
        expert;加了之后:8 个 expert 接近均匀,参数真正被用起来。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={SWITCH_TRANSFORMER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div
            style={{
              padding: "var(--space-4)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
            }}
          >
            <div style={{ fontSize: "var(--fs-md)", fontFamily: "ui-monospace", fontWeight: 700, marginBottom: 10, color: "var(--ink-primary)" }}>
              L_aux = α · N · Σ f_i · P_i
            </div>
            <div style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", lineHeight: 1.6, marginBottom: 14 }}>
              f_i = 路由到 expert i 的 token 比例 · P_i = router 给 expert i 的平均概率 ·
              α = 0.01 —— 这是 Switch Transformer 论文里的核心招式,后续 GLaM / Mixtral 都直接继承。
            </div>
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
                { v: false, label: "不加 → 塌缩" },
                { v: true, label: "加 → 均匀" },
              ].map((opt) => (
                <button key={String(opt.v)} type="button" onClick={() => setBalanced(opt.v)} style={btnStyle(balanced === opt.v)}>
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
