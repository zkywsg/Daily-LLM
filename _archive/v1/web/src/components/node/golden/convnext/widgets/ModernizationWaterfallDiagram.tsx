import { useState } from "react";
import { MODERNIZATION_WATERFALL } from "../lib/data";

const W = 760;
const H = 340;

export function ModernizationWaterfallDiagram() {
  const [step, setStep] = useState(MODERNIZATION_WATERFALL.length - 1);

  const PAD_L = 50;
  const PAD_R = 20;
  const PAD_T = 50;
  const baseY = H - 70;
  const minAcc = 75;
  const maxAcc = 83;
  const scaleY = (baseY - PAD_T) / (maxAcc - minAcc);
  const colW = (W - PAD_L - PAD_R) / MODERNIZATION_WATERFALL.length;

  const yFor = (acc: number) => baseY - (acc - minAcc) * scaleY;

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
        role="img"
        aria-label="ResNet-50 到 ConvNeXt-T 的现代化路径累积精度瀑布图"
      >
        <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
          现代化路径:ResNet-50(76.1%)→ ConvNeXt-T(82.0%)
        </text>

        {/* baseline grid lines */}
        {[76, 78, 80, 82].map((v) => (
          <g key={v}>
            <line x1={PAD_L} x2={W - PAD_R} y1={yFor(v)} y2={yFor(v)} stroke="var(--border)" strokeWidth={1} strokeDasharray="2,3" />
            <text x={PAD_L - 8} y={yFor(v) + 3} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
              {v}%
            </text>
          </g>
        ))}

        {MODERNIZATION_WATERFALL.map((s, i) => {
          if (i > step) return null;
          const x = PAD_L + i * colW;
          const barW = colW - 10;
          const isFirst = i === 0;
          const isCurrent = i === step;
          const y = yFor(s.cumulative);
          const barTopFrom = isFirst ? baseY : yFor(MODERNIZATION_WATERFALL[i - 1].cumulative);
          const fill = isFirst ? "#9ca3af" : isCurrent ? "#ec4899" : "#3b82f6";
          const bg = isFirst ? "#f3f4f6" : isCurrent ? "#fce7f3" : "#dbeafe";

          return (
            <g key={s.label}>
              <rect x={x} y={PAD_T} width={barW} height={baseY - PAD_T} fill={bg} opacity={0.25} rx={2} />
              {isFirst ? (
                <rect x={x} y={y} width={barW} height={baseY - y} fill={fill} rx={2} />
              ) : (
                <rect x={x} y={Math.min(y, barTopFrom)} width={barW} height={Math.abs(barTopFrom - y)} fill={fill} rx={2} />
              )}
              <text x={x + barW / 2} y={y - 6} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">
                {s.cumulative}%
              </text>
              {!isFirst && (
                <text x={x + barW / 2} y={y - 18} textAnchor="middle" fontSize={9} fontWeight={700} fill={isCurrent ? "#ec4899" : "#3b82f6"}>
                  +{s.delta}
                </text>
              )}
              <text
                x={x + barW / 2}
                y={H - 46}
                textAnchor="middle"
                fontSize={8}
                fill="var(--ink-secondary)"
                style={{ writingMode: "horizontal-tb" }}
              >
                {s.label.length > 8 ? `${s.label.slice(0, 8)}…` : s.label}
              </text>
              <text x={x + barW / 2} y={H - 34} textAnchor="middle" fontSize={7.5} fill="var(--ink-muted)">
                {s.source}
              </text>
            </g>
          );
        })}
      </svg>

      <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
        <button
          type="button"
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          style={btnStyle(false, step === 0)}
        >
          ← 上一步
        </button>
        <button
          type="button"
          onClick={() => setStep((s) => Math.min(MODERNIZATION_WATERFALL.length - 1, s + 1))}
          disabled={step === MODERNIZATION_WATERFALL.length - 1}
          style={btnStyle(false, step === MODERNIZATION_WATERFALL.length - 1)}
        >
          下一步 →
        </button>
        <span style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)" }}>
          {step === 0 ? "起点:ResNet-50 原版" : `第 ${step} / ${MODERNIZATION_WATERFALL.length - 1} 步:${MODERNIZATION_WATERFALL[step].label}`}
        </span>
        <button
          type="button"
          onClick={() => setStep(MODERNIZATION_WATERFALL.length - 1)}
          style={btnStyle(true, false)}
        >
          显示全部
        </button>
      </div>
    </div>
  );
}

function btnStyle(active: boolean, disabled: boolean): React.CSSProperties {
  return {
    padding: "4px 12px",
    fontSize: "var(--fs-sm)",
    borderRadius: "var(--radius-sm)",
    border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
    background: active ? "var(--accent-link)" : "var(--bg-surface)",
    color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.5 : 1,
  };
}
