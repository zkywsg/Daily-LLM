import { useState } from "react";
import { DistributionFittingScatter } from "./DistributionFittingScatter";

const TOTAL = 100;

// 切 toggle 演示:正常训练(4 模都学到) vs mode collapse(G 只学到 1 个模)。
// 都用 iter=80 时的状态对比 —— 正常的应该 4 模都聚拢,collapse 的只在左下角聚一团。
export function ModeCollapseDemo() {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <div>
      <DistributionFittingScatter iter={80} totalIter={TOTAL} modeCollapse={collapsed} />
      <div
        style={{
          marginTop: "var(--space-3)",
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
          训练结局
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {[
            { v: false, label: "正常收敛(4 模都学到)" },
            { v: true, label: "Mode Collapse(只学 1 模)" },
          ].map((opt) => {
            const active = collapsed === opt.v;
            return (
              <button
                key={String(opt.v)}
                type="button"
                onClick={() => setCollapsed(opt.v)}
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
        <div
          style={{
            fontSize: "var(--fs-xs)",
            color: "var(--ink-muted)",
            marginTop: 6,
            lineHeight: 1.4,
          }}
        >
          Mode Collapse:G 发现"只生成一种图就能骗到 D",懒得学其他模式。
          这是 GAN 训练的经典失败 —— 后来 WGAN-GP / Unrolled GAN / minibatch
          discrimination 都是为了缓解这个问题。
        </div>
      </div>
    </div>
  );
}
