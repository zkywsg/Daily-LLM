import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { COT_SOURCE_PATH } from "../lib/prose";
import { ZeroShotSpellBars } from "../widgets/ZeroShotSpellBars";
import { ZeroShotPromptDemo } from "../widgets/ZeroShotPromptDemo";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function ZeroShotCoTStage({ mechanism2Prose }: Props) {
  const [withSpell, setWithSpell] = useState(true);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Zero-shot CoT — 一句魔法咒语
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Kojima 2022 发现:不需要任何示例,只在 prompt 末尾加一句
        \"Let's think step by step\",GSM8K 准确率就从 17.6% 跳到 40.8%。
        模型一直都会推理,只是默认懒得写 —— 咒语只是把这个开关打开。
      </p>

      <ZeroShotPromptDemo withSpell={withSpell} />
      <p className={styles.caption}>
        切 toggle 看同一问题在两个 prompt 下的模型输出差异。
        不加咒语 → 直接错;加咒语 → 显式推理,答对。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={COT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ZeroShotSpellBars />
          <p className={styles.caption}>
            不同变体的精度。最佳是 \"Let's think step by step\"(40.8%),
            反向控制 \"Don't think. Just answer.\" 几乎跟无咒语持平 ——
            侧面证明咒语的关键是\"激活推理模式\",不是某个特定词的魔力。
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
              prompt 模式
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {[
                { v: false, label: "不加咒语" },
                { v: true, label: "加 \"step by step\"" },
              ].map((opt) => {
                const active = withSpell === opt.v;
                return (
                  <button
                    key={String(opt.v)}
                    type="button"
                    onClick={() => setWithSpell(opt.v)}
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
