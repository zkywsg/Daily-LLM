const W = 700;
const H = 280;

interface Props {
  revealed: boolean;
}

export function HiddenTraceDiagram({ revealed }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="thinking trace 隐藏 vs 展开 演示">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {revealed ? "如果 trace 公开(教学演示)" : "用户实际看到的界面"}
      </text>

      <g transform="translate(150, 50)">
        <rect x={0} y={0} width={400} height={36} fill="var(--bg-surface)" stroke="var(--border)" rx={8} />
        <circle cx={20} cy={18} r={6} fill={revealed ? "#10b981" : "#a855f7"}>
          {!revealed && <animate attributeName="opacity" values="1;0.4;1" dur="1.2s" repeatCount="indefinite" />}
        </circle>
        <text x={36} y={22} fontSize={11} fill="var(--ink-secondary)">
          {revealed ? "Thought for 32 seconds ▾(已展开)" : "Thought for 32 seconds ▸"}
        </text>
      </g>

      {revealed ? (
        <g transform="translate(150, 96)">
          <rect x={0} y={0} width={400} height={130} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1} rx={6} />
          <text x={14} y={20} fontSize={10} fontFamily="monospace" fill="#6b7280">让我先理解题目...</text>
          <text x={14} y={38} fontSize={10} fontFamily="monospace" fill="#6b7280">尝试方法 1:坐标几何...太复杂</text>
          <text x={14} y={56} fontSize={10} fontFamily="monospace" fill="#065f46" fontWeight={700}>Wait,换个角度,用对称性...</text>
          <text x={14} y={74} fontSize={10} fontFamily="monospace" fill="#065f46" fontWeight={700}>等等,那步公式用错了,重新算...</text>
          <text x={14} y={92} fontSize={10} fontFamily="monospace" fill="#6b7280">验证:代回原方程...对的</text>
          <text x={14} y={114} fontSize={11} fontWeight={700} fill="#1e40af">→ 8000+ reasoning tokens(单独计费,内容不返回)</text>
        </g>
      ) : (
        <g transform="translate(150, 96)">
          <rect x={0} y={0} width={400} height={60} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={6} />
          <text x={14} y={26} fontSize={11} fontWeight={700} fill="#065f46">答案是 23。</text>
          <text x={14} y={44} fontSize={9} fill="#065f46">(只显示最终 summary + answer)</text>
        </g>
      )}

      <text x={W / 2} y={H - 30} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {revealed
          ? "内部 8K+ token 的反思/回溯/自验证过程 — 真实产品中这一切被折叠隐藏"
          : "点击可展开查看内部 thinking 过程(教学演示,真实 o1 API 不返回 reasoning 内容)"}
      </text>
      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        动机:防止蒸馏 + 避免"未对齐内部独白"暴露 + 让用户更能接受"模型在思考"而非焦虑等待
      </text>
    </svg>
  );
}
