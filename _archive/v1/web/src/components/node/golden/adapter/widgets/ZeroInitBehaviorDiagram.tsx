import { ZERO_INIT_FRAMES } from "../lib/data";

const W = 640;
const H = 240;

interface Props {
  phase: "before" | "after";
}

export function ZeroInitBehaviorDiagram({ phase }: Props) {
  const frame = ZERO_INIT_FRAMES[phase];
  const shift = frame.outputShift * 60; // px offset for visual "Δ"
  const wUpFill = phase === "before" ? "#f3f4f6" : "#fce7f3";
  const wUpStroke = phase === "before" ? "#9ca3af" : "#ec4899";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="零初始化行为对比">
      <text x={W / 2} y={18} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        {frame.label}
      </text>

      {/* input x */}
      <circle cx={80} cy={120} r={24} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.6} />
      <text x={80} y={124} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1d4ed8">x</text>

      {/* down -> relu -> up pipeline */}
      <rect x={150} y={70} width={70} height={26} fill="#fef3c7" stroke="#f59e0b" rx={5} />
      <text x={185} y={87} textAnchor="middle" fontSize={9} fontWeight={600} fill="#92400e">down</text>
      <line x1={104} y1={110} x2={150} y2={90} stroke="#9ca3af" strokeWidth={1.2} markerEnd="url(#az-arrow)" />

      <rect x={250} y={70} width={70} height={26} fill="#dbeafe" stroke="#3b82f6" rx={5} />
      <text x={285} y={87} textAnchor="middle" fontSize={9} fontWeight={600} fill="#1d4ed8">ReLU</text>
      <line x1={220} y1={83} x2={250} y2={83} stroke="#9ca3af" strokeWidth={1.2} markerEnd="url(#az-arrow)" />

      <rect x={350} y={70} width={70} height={26} fill={wUpFill} stroke={wUpStroke} strokeWidth={2} rx={5} />
      <text x={385} y={87} textAnchor="middle" fontSize={9} fontWeight={700} fill={phase === "before" ? "#6b7280" : "#be185d"}>
        up (W_up)
      </text>
      <line x1={320} y1={83} x2={350} y2={83} stroke="#9ca3af" strokeWidth={1.2} markerEnd="url(#az-arrow)" />
      <text x={385} y={62} textAnchor="middle" fontSize={9} fontWeight={700} fill={phase === "before" ? "#9ca3af" : "#ec4899"}>
        {phase === "before" ? "≈ 0" : "学到非零权重"}
      </text>

      {/* delta output */}
      <circle cx={480} cy={83} r={16} fill={phase === "before" ? "#f3f4f6" : "#fce7f3"} stroke={wUpStroke} strokeWidth={1.4} />
      <text x={480} y={87} textAnchor="middle" fontSize={9} fontWeight={700} fill={phase === "before" ? "#9ca3af" : "#be185d"}>Δ</text>
      <line x1={420} y1={83} x2={464} y2={83} stroke="#9ca3af" strokeWidth={1.2} markerEnd="url(#az-arrow)" />

      {/* residual sum -> output */}
      <circle cx={560} cy={120} r={22} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.8} />
      <text x={560} y={124} textAnchor="middle" fontSize={11} fontWeight={700} fill="#047857">+</text>
      <line x1={104} y1={128} x2={540} y2={128} stroke="#10b981" strokeWidth={1.4} strokeDasharray="4 2" markerEnd="url(#az-arrow-green)" />
      <line x1={480} y1={99} x2={565} y2={104} stroke={wUpStroke} strokeWidth={1.4} markerEnd="url(#az-arrow)" />

      <rect x={520} y={172} width={100} height={30} fill={phase === "before" ? "#f3f4f6" : "#ecfdf5"} stroke={phase === "before" ? "#9ca3af" : "#10b981"} rx={6} />
      <text x={570} y={191} textAnchor="middle" fontSize={10} fontWeight={700} fill={phase === "before" ? "#6b7280" : "#047857"}>
        h = x{phase === "after" ? " + Δ" : ""}
      </text>
      <line x1={560} y1={142} x2={570} y2={172} stroke="#10b981" strokeWidth={1.2} markerEnd="url(#az-arrow-green)" />

      {/* offset visualization bar */}
      <line x1={150} y1={220} x2={150 + shift + 340} y2={220} stroke="#e5e7eb" strokeWidth={6} strokeLinecap="round" />
      <line x1={150} y1={220} x2={150 + shift} y2={220} stroke={phase === "before" ? "#9ca3af" : "#ec4899"} strokeWidth={6} strokeLinecap="round" />
      <text x={150} y={236} fontSize={8} fill="var(--ink-muted)">base 表征不变</text>
      <text x={150 + shift + 4} y={236} fontSize={8} fill="var(--ink-muted)">
        {phase === "before" ? "偏移量 = 0(恒等映射)" : "偏移量 = Δ(任务特化)"}
      </text>

      <defs>
        <marker id="az-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
        <marker id="az-arrow-green" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#10b981" />
        </marker>
      </defs>
    </svg>
  );
}
