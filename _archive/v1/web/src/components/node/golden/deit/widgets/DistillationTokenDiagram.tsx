const W = 700;
const H = 300;

// Token 序列流入 encoder,CLS 和 DistillToken 各自接不同的监督信号。
export function DistillationTokenDiagram() {
  const tokens = ["CLS", "p1", "p2", "p3", "…", "p196", "DIST"];
  const tokenW = 68;
  const gap = 6;
  const startX = (W - (tokens.length * tokenW + (tokens.length - 1) * gap)) / 2;
  const tokenY = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="DeiT distillation token diagram">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        Distillation Token — 和 [CLS] 并列,接受独立监督信号
      </text>

      {/* token 序列 */}
      {tokens.map((t, i) => {
        const x = startX + i * (tokenW + gap);
        const isCls = t === "CLS";
        const isDist = t === "DIST";
        const fill = isCls ? "#dbeafe" : isDist ? "#fce7f3" : "#f3f4f6";
        const stroke = isCls ? "#3b82f6" : isDist ? "#ec4899" : "#9ca3af";
        return (
          <g key={t}>
            <rect x={x} y={tokenY} width={tokenW} height={40} rx={5} fill={fill} stroke={stroke} strokeWidth={isCls || isDist ? 2 : 1.2} />
            <text x={x + tokenW / 2} y={tokenY + 25} textAnchor="middle" fontSize={11} fontWeight={isCls || isDist ? 700 : 500} fill={isCls ? "#1e40af" : isDist ? "#9d174d" : "#4b5563"}>
              {t === "DIST" ? "[DIST]" : t === "CLS" ? "[CLS]" : t}
            </text>
          </g>
        );
      })}

      {/* arrow down to encoder */}
      <line x1={W / 2} y1={100} x2={W / 2} y2={130} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-dist)" />
      <defs>
        <marker id="arrow-dist" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>

      <rect x={W / 2 - 100} y={130} width={200} height={36} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} />
      <text x={W / 2} y={153} textAnchor="middle" fontSize={12} fontWeight={600} fill="#92400e">Transformer Encoder</text>

      {/* two output heads */}
      <line x1={W / 2 - 40} y1={166} x2={230} y2={200} stroke="#3b82f6" strokeWidth={1.5} />
      <line x1={W / 2 + 40} y1={166} x2={470} y2={200} stroke="#ec4899" strokeWidth={1.5} />

      <rect x={160} y={200} width={140} height={34} rx={5} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.5} />
      <text x={230} y={221} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1e40af">CLS output</text>

      <rect x={400} y={200} width={140} height={34} rx={5} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
      <text x={470} y={221} textAnchor="middle" fontSize={11} fontWeight={600} fill="#9d174d">Distill output</text>

      {/* supervision */}
      <line x1={230} y1={234} x2={230} y2={260} stroke="#3b82f6" strokeWidth={1.5} markerEnd="url(#arrow-dist)" />
      <text x={230} y={276} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e40af">y_true(ground truth)</text>

      <line x1={470} y1={234} x2={470} y2={260} stroke="#ec4899" strokeWidth={1.5} markerEnd="url(#arrow-dist)" />
      <text x={470} y={276} textAnchor="middle" fontSize={11} fontWeight={700} fill="#9d174d">y_teacher(CNN argmax)</text>

      <text x={W / 2} y={296} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        CLS 学真实标签,DIST 学 CNN teacher(RegNet)的预测 — 两个独立监督信号在同一 encoder 里共同训练
      </text>
    </svg>
  );
}
