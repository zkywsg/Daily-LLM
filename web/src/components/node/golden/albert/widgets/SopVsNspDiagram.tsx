interface Props {
  mode: "nsp" | "sop";
}

export function SopVsNspDiagram({ mode }: Props) {
  const W = 700;
  const H = 280;

  const isNsp = mode === "nsp";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="NSP 与 SOP 负例构造方式对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {isNsp ? "NSP:负例来自随机不相关句子" : "SOP:负例是同一对句子调换顺序"}
      </text>

      {/* Positive example row */}
      <text x={40} y={70} fontSize={11} fontWeight={700} fill="var(--ink-secondary)">正例</text>
      <rect x={110} y={52} width={220} height={32} rx={4} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} />
      <text x={220} y={72} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">句子 A</text>
      <rect x={350} y={52} width={220} height={32} rx={4} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} />
      <text x={460} y={72} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">句子 B(A 的下一句)</text>
      <text x={590} y={72} fontSize={16} fill="#10b981">✓</text>

      {/* Negative example row */}
      <text x={40} y={150} fontSize={11} fontWeight={700} fill="var(--ink-secondary)">负例</text>
      {isNsp ? (
        <>
          <rect x={110} y={132} width={220} height={32} rx={4} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} />
          <text x={220} y={152} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">句子 A</text>
          <rect x={350} y={132} width={220} height={32} rx={4} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.4} />
          <text x={460} y={152} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">随机文档的句子 X</text>
          <text x={590} y={152} fontSize={16} fill="#ec4899">✗</text>
        </>
      ) : (
        <>
          <rect x={110} y={132} width={220} height={32} rx={4} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} />
          <text x={220} y={152} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">句子 B(调换到前面)</text>
          <rect x={350} y={132} width={220} height={32} rx={4} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} />
          <text x={460} y={152} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">句子 A(调换到后面)</text>
          <text x={590} y={152} fontSize={16} fill="#ec4899">✗</text>
        </>
      )}

      <line x1={40} y1={185} x2={660} y2={185} stroke="var(--border)" strokeWidth={1} strokeDasharray="3 3" />

      <text x={W / 2} y={215} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
        {isNsp
          ? "主题完全不同 → 模型靠浅层主题信号就能区分,学不到句间逻辑"
          : "主题完全相同,只是顺序反了 → 模型必须学到真正的句间连贯性"}
      </text>
      <text x={W / 2} y={238} textAnchor="middle" fontSize={11} fontWeight={700} fill={isNsp ? "#ec4899" : "#10b981"}>
        {isNsp ? "RoBERTa 证明:NSP 几乎无用" : "ALBERT Table 5:MLM+SOP 82.1 F1 / 65.5 RACE，优于 MLM+NSP"}
      </text>
      <text x={W / 2} y={260} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        关键差异:SOP 的正负例是同一对句子,只是顺序不同 — 这是 hard negative 思想的早期实践
      </text>
    </svg>
  );
}
