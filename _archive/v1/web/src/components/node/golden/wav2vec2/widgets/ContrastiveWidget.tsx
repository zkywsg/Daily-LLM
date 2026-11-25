import { useState } from "react";
import { CODEBOOK_SIZE, codebookVector, frameToContinuousFeature, contrastiveScores } from "../lib/data";

const W = 680;
const H = 300;

export function ContrastiveWidget() {
  const [trueCode, setTrueCode] = useState(2);
  const z = frameToContinuousFeature(trueCode);
  const scores = contrastiveScores(z);
  const predictedCode = scores.indexOf(Math.max(...scores));

  const cx0 = 150, cy0 = 150, r = 100;
  const toXY = (v: [number, number]) => [cx0 + v[0] * r, cy0 - v[1] * r];

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)", flexWrap: "wrap" }}>
        {Array.from({ length: CODEBOOK_SIZE }, (_, i) => (
          <button
            key={i} type="button" onClick={() => setTrueCode(i)} aria-pressed={i === trueCode}
            style={{
              width: 30, height: 30, borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === trueCode ? "#fb7185" : "var(--border)"}`,
              background: i === trueCode ? "#fb7185" : "var(--bg-surface)",
              color: i === trueCode ? "#fff" : "var(--ink-secondary)", cursor: "pointer",
            }}
          >
            {i}
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`真实量化目标 ${trueCode},模型预测 ${predictedCode}`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          对比学习:z 与码本向量的相似度
        </text>
        {Array.from({ length: CODEBOOK_SIZE }, (_, i) => {
          const [x, y] = toXY(codebookVector(i));
          const isTrue = i === trueCode;
          return (
            <g key={i}>
              <circle cx={x} cy={y} r={14} fill={isTrue ? "#fb7185" : "var(--bg-subtle)"} stroke="var(--border)" />
              <text x={x} y={y + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill={isTrue ? "#fff" : "var(--ink-primary)"}>{i}</text>
            </g>
          );
        })}
        {(() => {
          const [zx, zy] = toXY(z);
          return <circle cx={zx} cy={zy} r={6} fill="#9d174d" />;
        })()}
        {scores.map((s, i) => {
          const x = 320 + (i % 3) * 110;
          const y = 110 + Math.floor(i / 3) * 100;
          const h = Math.min(s * 70, 70);
          return (
            <g key={i}>
              <rect x={x} y={y - h} width={30} height={h} fill={i === predictedCode ? "#fb7185" : "#9ca3af"} />
              <text x={x + 15} y={y + 18} textAnchor="middle" fontSize={9} fill="var(--ink-secondary)">码 {i}</text>
              <text x={x + 15} y={y - h - 4} textAnchor="middle" fontSize={9} fill="var(--ink-primary)">{s.toFixed(2)}</text>
            </g>
          );
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        深粉色圆点是连续特征 z(带小扰动);右侧柱状图是 z 与每个码本向量的相似度 softmax——{predictedCode === trueCode ? "模型正确选中了真实目标" : "模型选错了目标"}(码 {predictedCode})。
      </p>
    </div>
  );
}
