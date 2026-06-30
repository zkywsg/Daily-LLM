const W = 700;
const H = 280;

interface Props {
  guidanceScale: number;  // 0..20
  prompt: string;
}

// 可视化 CFG:从 epsilon_uncond 到 epsilon_uncond + s*(epsilon_cond - epsilon_uncond)
// 用一个 2D 向量空间表示

export function CfgScaleDemo({ guidanceScale, prompt }: Props) {
  const cx = W / 2;
  const cy = H / 2 + 10;

  // uncond direction (生成什么都可以)
  const uncondEnd = { x: cx + 60, y: cy + 30 };
  // cond direction (听 prompt)
  const condEnd = { x: cx + 110, y: cy - 30 };
  // CFG vector: uncond + s * (cond - uncond)
  const dx = condEnd.x - uncondEnd.x;
  const dy = condEnd.y - uncondEnd.y;
  const cfgEnd = { x: uncondEnd.x + dx * guidanceScale, y: uncondEnd.y + dy * guidanceScale };

  // describe quality based on scale
  let quality: string;
  let qualityColor: string;
  if (guidanceScale < 1) {
    quality = "noise · 不听 prompt";
    qualityColor = "#9ca3af";
  } else if (guidanceScale < 4) {
    quality = "弱听 prompt · 多样性高 / fidelity 低";
    qualityColor = "#3b82f6";
  } else if (guidanceScale <= 10) {
    quality = "✓ 听 prompt 适中 · 7.5 是默认甜蜜点";
    qualityColor = "#10b981";
  } else {
    quality = "过度强调 · 颜色饱和 / artifacts";
    qualityColor = "#ec4899";
  }

  function Arrow({ x1, y1, x2, y2, color, id, strokeW = 2 }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string; strokeW?: number }) {
    return (
      <g>
        <defs>
          <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
          </marker>
        </defs>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={strokeW} markerEnd={`url(#${id})`} />
      </g>
    );
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Classifier-free guidance scale demo">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Classifier-Free Guidance — ε = ε_uncond + s · (ε_cond − ε_uncond)
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        prompt: "{prompt}" · scale s = {guidanceScale.toFixed(1)}
      </text>

      {/* origin */}
      <circle cx={cx} cy={cy} r={4} fill="#1f2937" />
      <text x={cx - 8} y={cy + 4} textAnchor="end" fontSize={10} fill="#374151">latent</text>

      {/* uncond */}
      <Arrow x1={cx} y1={cy} x2={uncondEnd.x} y2={uncondEnd.y} color="#9ca3af" id="cfg-uncond" />
      <text x={uncondEnd.x + 4} y={uncondEnd.y + 14} fontSize={10} fontWeight={600} fill="#6b7280">ε_uncond</text>

      {/* cond */}
      <Arrow x1={cx} y1={cy} x2={condEnd.x} y2={condEnd.y} color="#3b82f6" id="cfg-cond" />
      <text x={condEnd.x + 4} y={condEnd.y} fontSize={10} fontWeight={600} fill="#1e40af">ε_cond</text>

      {/* CFG result */}
      <Arrow x1={cx} y1={cy} x2={cfgEnd.x} y2={cfgEnd.y} color="#ec4899" id="cfg-result" strokeW={2.8} />
      <text x={cfgEnd.x + 6} y={cfgEnd.y} fontSize={11} fontWeight={700} fill="#831843">ε_cfg</text>

      {/* quality label */}
      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={12} fontWeight={700} fill={qualityColor}>
        s = {guidanceScale.toFixed(1)} → {quality}
      </text>
    </svg>
  );
}
