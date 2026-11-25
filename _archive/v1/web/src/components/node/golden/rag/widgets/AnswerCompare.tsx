import { QUERIES } from "../lib/data";

interface Props {
  queryIdx: number;
}

const W = 700;
const H = 280;

// 无 RAG vs 有 RAG 的回答对比。
// 无 RAG:模型靠参数化知识硬编;时效性 / 私有知识无能为力。
// 有 RAG:能用 retrieved chunks 拼出准确答案。

const BARE_ANSWER: Record<number, string> = {
  0: "光合作用是植物利用阳光的过程。(没给出方程式)",
  1: "斯大林格勒战役是二战中很重要的战役。(模糊)",
  2: "git rebase 是一个 git 命令...(细节不准)",
};

export function AnswerCompare({ queryIdx }: Props) {
  const q = QUERIES[queryIdx];
  const bare = BARE_ANSWER[queryIdx] ?? "(模型没有可靠回答)";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Bare LLM vs RAG answer comparison">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        无 RAG vs 有 RAG 回答对比
      </text>

      {/* Bare LLM */}
      <rect x={20} y={40} width={W - 40} height={90} rx={5} fill="#fef2f2" stroke="#fca5a5" strokeWidth={1.5} />
      <text x={32} y={60} fontSize={11} fontWeight={700} fill="#7f1d1d">
        ❌ Bare LLM (只靠参数化知识)
      </text>
      <text x={32} y={86} fontSize={11} fontFamily="ui-monospace, monospace" fill="#1f2937">
        {bare.length > 80 ? bare.slice(0, 80) + "…" : bare}
      </text>
      <text x={32} y={114} fontSize={10} fontStyle="italic" fill="#7f1d1d">
        ↑ 模糊 / 错误 / 编造,无法核查
      </text>

      {/* RAG */}
      <rect x={20} y={148} width={W - 40} height={120} rx={5} fill="#ecfdf5" stroke="#86efac" strokeWidth={1.5} />
      <text x={32} y={168} fontSize={11} fontWeight={700} fill="#065f46">
        ✓ RAG (检索后回答)
      </text>
      {(() => {
        // 简单换行
        const lines: string[] = [];
        const words = q.answer.split("");
        let cur = "";
        for (const w of words) {
          cur += w;
          if (cur.length >= 38 || w === "。") {
            lines.push(cur);
            cur = "";
          }
        }
        if (cur) lines.push(cur);
        return lines.map((line, i) => (
          <text key={i} x={32} y={192 + i * 18} fontSize={11} fontFamily="ui-monospace, monospace" fill="#1f2937">
            {line}
          </text>
        ));
      })()}
    </svg>
  );
}
