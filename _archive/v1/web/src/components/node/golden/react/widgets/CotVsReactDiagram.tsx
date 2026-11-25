import { COT_HALLUCINATION, REACT_TRACE } from "../lib/data";

const W = 700;
const H = 260;

export function CotVsReactDiagram() {
  const finalAnswer = REACT_TRACE[REACT_TRACE.length - 1].action.replace("Finish[", "").replace("]", "");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="纯 CoT 幻觉 vs ReAct 正确答案对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        同一问题:纯 CoT 全程脑内推理 vs ReAct 每步基于查证信息
      </text>

      <g transform="translate(30, 40)">
        <text x={0} y={0} fontSize={11} fontWeight={700} fill="#be185d">纯 CoT(闭门造车)</text>
        <rect x={0} y={10} width={640} height={60} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.6} rx={6} />
        <text x={14} y={32} fontSize={10} fill="#9d174d">{COT_HALLUCINATION.thought}</text>
        <text x={14} y={52} fontSize={11} fontWeight={700} fill="#be185d">答案:{COT_HALLUCINATION.answer} ❌</text>
      </g>

      <g transform="translate(30, 120)">
        <text x={0} y={0} fontSize={9} fill="var(--ink-muted)" fontStyle="italic">{COT_HALLUCINATION.correction}</text>
      </g>

      <g transform="translate(30, 150)">
        <text x={0} y={0} fontSize={11} fontWeight={700} fill="#065f46">ReAct(每步查证)</text>
        <rect x={0} y={10} width={640} height={60} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.6} rx={6} />
        <text x={14} y={32} fontSize={10} fill="#065f46">先查颜色 → 再查形成原因 → 基于两次真实检索结果给答案</text>
        <text x={14} y={52} fontSize={11} fontWeight={700} fill="#065f46">答案:{finalAnswer} ✓</text>
      </g>

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        每一步都基于实际查到的信息,而不是参数化记忆里可能出错的知识
      </text>
    </svg>
  );
}
