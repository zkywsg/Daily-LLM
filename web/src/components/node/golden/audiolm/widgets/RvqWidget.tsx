import { useState } from "react";
import { TOY_SEMANTIC_VALUE, residualQuantize, partialReconstruction } from "../lib/data";

const W = 560;
const H = 220;

export function RvqWidget() {
  const [numLevels, setNumLevels] = useState(1);
  const { codes } = residualQuantize(TOY_SEMANTIC_VALUE);
  const recon = partialReconstruction(codes, numLevels);
  const error = Math.abs(TOY_SEMANTIC_VALUE - recon);

  const toX = (v: number) => 280 + v * 240;

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        使用的 RVQ 层数 = {numLevels}
        <input type="range" min={1} max={4} value={numLevels} onChange={(e) => setNumLevels(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`用 ${numLevels} 层 RVQ 重建,误差 ${error.toFixed(3)}`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          残差向量量化(RVQ):层数越多,重建越精确
        </text>
        <line x1={40} y1={H / 2} x2={W - 40} y2={H / 2} stroke="var(--border)" />
        <circle cx={toX(TOY_SEMANTIC_VALUE)} cy={H / 2} r={8} fill="#9ca3af" />
        <text x={toX(TOY_SEMANTIC_VALUE)} y={H / 2 - 16} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">真实值</text>
        <circle cx={toX(recon)} cy={H / 2} r={6} fill="#fb7185" />
        <text x={toX(recon)} y={H / 2 + 26} textAnchor="middle" fontSize={10} fill="#9d174d">重建值</text>
        <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
          codes = [{codes.slice(0, numLevels).join(", ")}] · 误差 = {error.toFixed(3)}
        </text>
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        第一层码本捕捉粗粒度信息,后续层逐层用残差方式补充更精细的细节——用的层数越多,重建值越接近真实值。
      </p>
    </div>
  );
}
