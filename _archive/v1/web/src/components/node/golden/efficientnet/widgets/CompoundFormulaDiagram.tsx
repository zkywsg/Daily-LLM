import { useState } from "react";
import { PHI_STEPS, COMPOUND_CONSTANTS } from "../lib/data";

const W = 720;
const H = 300;

const AXES = [
  { key: "depthMul" as const, label: "depth = α^φ", color: "#ec4899", bg: "#fce7f3" },
  { key: "widthMul" as const, label: "width = β^φ", color: "#3b82f6", bg: "#dbeafe" },
  { key: "resMul" as const, label: "resolution = γ^φ", color: "#f59e0b", bg: "#fef3c7" },
];

export function CompoundFormulaDiagram() {
  const [phi, setPhi] = useState(0);
  const step = PHI_STEPS[phi];

  const PAD_L = 160;
  const PAD_T = 60;
  const barMaxW = 460;
  const rowH = 60;
  const maxMul = PHI_STEPS[PHI_STEPS.length - 1].depthMul; // depth 增长最快,做归一基准之一
  const maxScale = Math.max(maxMul, PHI_STEPS[PHI_STEPS.length - 1].widthMul, PHI_STEPS[PHI_STEPS.length - 1].resMul, 1);

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
        role="img"
        aria-label="Compound Coefficient φ 控制 depth/width/resolution 三轴同步放大"
      >
        <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
          φ = {step.phi}(EfficientNet-{step.variant})— 单参数联合缩放三轴
        </text>
        <text x={W / 2} y={38} textAnchor="middle" fontSize={9.5} fill="var(--ink-muted)">
          α={COMPOUND_CONSTANTS.alpha} · β={COMPOUND_CONSTANTS.beta} · γ={COMPOUND_CONSTANTS.gamma} ·{" "}
          {COMPOUND_CONSTANTS.constraint} ⇒ φ 每 +1,FLOPs ×2
        </text>

        {AXES.map((axis, i) => {
          const y = PAD_T + i * rowH;
          const mul = step[axis.key];
          const w = (mul / maxScale) * barMaxW;
          return (
            <g key={axis.key}>
              <text x={PAD_L - 12} y={y + 20} textAnchor="end" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
                {axis.label}
              </text>
              <rect x={PAD_L} y={y} width={barMaxW} height={28} fill={axis.bg} rx={4} />
              <rect x={PAD_L} y={y} width={Math.max(w, 3)} height={28} fill={axis.color} rx={4} />
              <text x={PAD_L + Math.max(w, 3) + 8} y={y + 19} fontSize={11} fontWeight={700} fill={axis.color}>
                ×{mul}
              </text>
            </g>
          );
        })}

        <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={9.5} fill="var(--ink-muted)">
          α/β/γ 在 B0 上做一次 grid search 得到,一旦定下,B1–B7 全部按 φ 推出来,不再手工调超参
        </text>
      </svg>

      <div style={{ display: "flex", gap: 4, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
        <input
          type="range"
          min={0}
          max={7}
          step={1}
          value={phi}
          onChange={(e) => setPhi(Number(e.target.value))}
          style={{ flex: "1 1 200px", accentColor: "#ec4899" }}
          aria-label="拖动查看 φ = 0..7 对应的 B0-B7"
        />
        <span style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: "var(--ink-primary)", minWidth: 90, textAlign: "right" }}>
          φ = {phi}(B{phi})
        </span>
      </div>
    </div>
  );
}
