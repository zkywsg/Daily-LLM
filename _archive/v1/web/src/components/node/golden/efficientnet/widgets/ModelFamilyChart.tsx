import { useState } from "react";
import { MODEL_FAMILY } from "../lib/data";

const W = 760;
const H = 400;

const EFFNET_VARIANTS = MODEL_FAMILY.filter((r) => r.isEfficientNet);

function xFor(params: number, padL: number, plotW: number, minLog: number, maxLog: number) {
  const log = Math.log10(Math.max(params, 1));
  return padL + ((log - minLog) / (maxLog - minLog)) * plotW;
}

export function ModelFamilyChart() {
  const [activeIdx, setActiveIdx] = useState(EFFNET_VARIANTS.length - 1);
  const active = EFFNET_VARIANTS[activeIdx];

  const PAD_L = 50;
  const PAD_R = 24;
  const PAD_T = 48;
  const PAD_B = 44;
  const plotW = W - PAD_L - PAD_R;
  const baseY = H - PAD_B;
  const minAcc = 75;
  const maxAcc = 85;
  const scaleY = (baseY - PAD_T) / (maxAcc - minAcc);
  const yFor = (acc: number) => baseY - (acc - minAcc) * scaleY;

  const minLog = Math.log10(4);
  const maxLog = Math.log10(600);

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
        role="img"
        aria-label="EfficientNet B0-B7 模型族与 ResNet/ResNeXt/GPipe 在参数量与 Top-1 准确率上的对比"
      >
        <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
          参数量 vs Top-1 准确率 — EfficientNet 帕累托线
        </text>

        {[76, 78, 80, 82, 84].map((v) => (
          <g key={v}>
            <line x1={PAD_L} x2={W - PAD_R} y1={yFor(v)} y2={yFor(v)} stroke="var(--border)" strokeWidth={1} strokeDasharray="2,3" />
            <text x={PAD_L - 8} y={yFor(v) + 3} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
              {v}%
            </text>
          </g>
        ))}

        {[10, 100].map((p) => (
          <text key={p} x={xFor(p, PAD_L, plotW, minLog, maxLog)} y={baseY + 18} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
            {p}M
          </text>
        ))}
        <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={9.5} fill="var(--ink-muted)">
          参数量(M,对数刻度)
        </text>

        {/* EfficientNet 帕累托曲线 */}
        <path
          d={EFFNET_VARIANTS.map((r, i) => `${i === 0 ? "M" : "L"}${xFor(r.params, PAD_L, plotW, minLog, maxLog)},${yFor(r.top1)}`).join(" ")}
          fill="none"
          stroke="#ec4899"
          strokeWidth={2}
        />

        {MODEL_FAMILY.map((row, i) => {
          const isActive = row.isEfficientNet && row.model === active.model;
          const cx = xFor(row.params, PAD_L, plotW, minLog, maxLog);
          const cy = yFor(row.top1);
          const fill = row.isEfficientNet ? (isActive ? "#ec4899" : "#f9a8d4") : "#9ca3af";
          return (
            <g key={`${row.model}-${i}`}>
              <circle cx={cx} cy={cy} r={isActive ? 7 : row.isEfficientNet ? 4 : 4} fill={fill} stroke={isActive ? "#ec4899" : "none"} strokeWidth={isActive ? 3 : 0} opacity={isActive ? 1 : 0.4} />
              {(isActive || !row.isEfficientNet) && (
                <text x={cx} y={cy - 10} textAnchor="middle" fontSize={8.5} fontWeight={isActive ? 700 : 500} fill={isActive ? "#ec4899" : "var(--ink-muted)"}>
                  {row.model}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      <div style={{ display: "flex", gap: 4, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
        <input
          type="range"
          min={0}
          max={EFFNET_VARIANTS.length - 1}
          step={1}
          value={activeIdx}
          onChange={(e) => setActiveIdx(Number(e.target.value))}
          style={{ flex: "1 1 200px", accentColor: "#ec4899" }}
          aria-label="拖动查看 EfficientNet-B0/B3/B5/B7"
        />
        <span style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: "#ec4899", minWidth: 220, textAlign: "right" }}>
          {active.model}:{active.params}M · {active.flops}B FLOPs · {active.top1}%
        </span>
      </div>
    </div>
  );
}
