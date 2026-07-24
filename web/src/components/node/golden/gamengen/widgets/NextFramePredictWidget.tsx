const HISTORY_LEN = 4;

// 展示条件 diffusion "预测下一帧"这一步替代传统渲染引擎的渲染循环:
// 历史帧 + 动作条件 → diffusion 模型 → 下一帧。静态示意图,不需要交互状态。

export function NextFramePredictWidget() {
  const frames = Array.from({ length: HISTORY_LEN }, (_, i) => i);

  return (
    <div>
      <svg viewBox="0 0 500 200" style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="条件 diffusion 模型预测下一帧的数据流">
        {frames.map((i) => (
          <g key={i}>
            <rect x={20 + i * 70} y={60} width={55} height={55} rx={4} fill="var(--bg-subtle)" stroke="var(--border)" />
            <text x={20 + i * 70 + 27} y={130} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">t-{HISTORY_LEN - i}</text>
          </g>
        ))}
        <text x={310} y={30} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ink-primary)">+ 动作条件</text>
        <path d="M 300 90 L 340 90" stroke="var(--ink-muted)" markerEnd="url(#arrow)" strokeWidth={1.5} />
        <rect x={345} y={55} width={70} height={65} rx={6} fill="#fae8ff" stroke="#d946ef" strokeWidth={2} />
        <text x={380} y={92} textAnchor="middle" fontSize={10} fontWeight={700} fill="#86198f">Diffusion</text>
        <path d="M 415 90 L 450 90" stroke="var(--ink-muted)" markerEnd="url(#arrow)" strokeWidth={1.5} />
        <rect x={455} y={62} width={40} height={40} rx={4} fill="#d946ef" />
        <text x={475} y={120} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">帧 t</text>
        <defs>
          <marker id="arrow" markerWidth={8} markerHeight={8} refX={6} refY={4} orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="var(--ink-muted)" />
          </marker>
        </defs>
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        近期 {HISTORY_LEN} 帧历史 + 玩家动作输入 → 条件 diffusion 模型 → 直接预测出下一帧画面,整个过程完全替代传统游戏引擎的渲染循环。
      </p>
    </div>
  );
}
