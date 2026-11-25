import { useState } from "react";
import { SINGLE_VS_COMPOUND_SCALING } from "../lib/data";

const W = 760;
const H = 360;

// x 轴用 log2(flops),flops ∈ {1,2,4,8,16}
function xFor(flops: number, padL: number, plotW: number) {
  const logMax = Math.log2(16);
  return padL + (Math.log2(flops) / logMax) * plotW;
}

export function SingleVsCompoundScalingChart() {
  const [visible, setVisible] = useState<Set<string>>(
    new Set(SINGLE_VS_COMPOUND_SCALING.map((c) => c.key))
  );

  const PAD_L = 46;
  const PAD_R = 24;
  const PAD_T = 44;
  const PAD_B = 40;
  const plotW = W - PAD_L - PAD_R;
  const baseY = H - PAD_B;
  const minAcc = 76;
  const maxAcc = 84;
  const scaleY = (baseY - PAD_T) / (maxAcc - minAcc);
  const yFor = (acc: number) => baseY - (acc - minAcc) * scaleY;

  const toggle = (key: string) => {
    setVisible((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
        role="img"
        aria-label="单轴缩放(只加深/只加宽/只加分辨率)与三轴复合缩放的 Top-1 准确率对比示意图"
      >
        <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
          单轴缩放 vs 复合缩放 — 相同 FLOPs 预算下的 Top-1 准确率(示意)
        </text>

        {/* y grid */}
        {[76, 78, 80, 82, 84].map((v) => (
          <g key={v}>
            <line x1={PAD_L} x2={W - PAD_R} y1={yFor(v)} y2={yFor(v)} stroke="var(--border)" strokeWidth={1} strokeDasharray="2,3" />
            <text x={PAD_L - 8} y={yFor(v) + 3} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
              {v}%
            </text>
          </g>
        ))}

        {/* x axis labels */}
        {[1, 2, 4, 8, 16].map((f) => (
          <text key={f} x={xFor(f, PAD_L, plotW)} y={baseY + 18} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
            {f}×
          </text>
        ))}
        <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={9.5} fill="var(--ink-muted)">
          相对 FLOPs(以 EfficientNet-B0 = 1× 为基准,对数刻度)
        </text>

        {SINGLE_VS_COMPOUND_SCALING.filter((c) => visible.has(c.key)).map((curve) => {
          const path = curve.points
            .map((p, i) => `${i === 0 ? "M" : "L"}${xFor(p.flops, PAD_L, plotW)},${yFor(p.top1)}`)
            .join(" ");
          const isCompound = curve.key === "compound";
          return (
            <g key={curve.key}>
              <path d={path} fill="none" stroke={curve.color} strokeWidth={isCompound ? 3 : 2} />
              {curve.points.map((p) => (
                <circle
                  key={`${curve.key}-${p.flops}`}
                  cx={xFor(p.flops, PAD_L, plotW)}
                  cy={yFor(p.top1)}
                  r={isCompound ? 3.5 : 2.5}
                  fill={curve.color}
                />
              ))}
              <text
                x={xFor(curve.points[curve.points.length - 1].flops, PAD_L, plotW) - 6}
                y={yFor(curve.points[curve.points.length - 1].top1) - 8}
                textAnchor="end"
                fontSize={9.5}
                fontWeight={isCompound ? 700 : 500}
                fill={curve.color}
              >
                {curve.label}
              </text>
            </g>
          );
        })}
      </svg>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
        {SINGLE_VS_COMPOUND_SCALING.map((curve) => {
          const active = visible.has(curve.key);
          return (
            <button
              key={curve.key}
              type="button"
              onClick={() => toggle(curve.key)}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                padding: "4px 10px",
                fontSize: "var(--fs-xs)",
                borderRadius: "var(--radius-sm)",
                border: `1px solid ${active ? curve.color : "var(--border)"}`,
                background: active ? curve.bg : "var(--bg-surface)",
                color: active ? curve.color : "var(--ink-muted)",
                cursor: "pointer",
                fontWeight: active ? 700 : 500,
              }}
            >
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: curve.color, opacity: active ? 1 : 0.3 }} />
              {curve.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
