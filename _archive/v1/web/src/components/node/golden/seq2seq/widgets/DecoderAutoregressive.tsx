import { DEMO_TGT } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  teacherForcing: boolean;
  errorAtStep: number; // -1 = 无错误; N = 从第 N 步模型预测出错
}

function Arrow({ x1, y1, x2, y2, color, id }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string }) {
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function DecoderAutoregressive({ teacherForcing, errorAtStep }: Props) {
  const tokens = ["<SOS>", ...DEMO_TGT, "<EOS>"];
  const TILE_W = 80;
  const gap = 8;
  const startX = (W - (tokens.length * (TILE_W + gap) - gap)) / 2;
  const y = 100;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Decoder autoregressive generation">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Decoder 自回归生成 — {teacherForcing ? "Teacher Forcing(训练)" : "自由生成(推理)"}
      </text>

      {tokens.map((tok, i) => {
        if (i === 0) return null; // <SOS> 已作输入,不需要单独展示输出栏
        const isCorrupted = !teacherForcing && errorAtStep >= 0 && i > errorAtStep;
        const displayTok = isCorrupted ? "?" : tok;
        const x = startX + (i - 1) * (TILE_W + gap);
        return (
          <g key={i}>
            <rect x={x} y={y} width={TILE_W} height={30} rx={4}
                  fill={isCorrupted ? "#fce7f3" : "#ecfdf5"}
                  stroke={isCorrupted ? "#ec4899" : "#10b981"} strokeWidth={1.4} />
            <text x={x + TILE_W / 2} y={y + 20} textAnchor="middle" fontSize={12} fontWeight={600}
                  fill={isCorrupted ? "#831843" : "#065f46"}>
              {displayTok}
            </text>
          </g>
        );
      })}

      {/* input row */}
      {tokens.slice(0, -1).map((tok, i) => {
        const isCorrupted = !teacherForcing && errorAtStep >= 0 && i > errorAtStep;
        const inputTok = teacherForcing ? tok : (isCorrupted ? "?" : tok);
        const x = startX + i * (TILE_W + gap);
        return (
          <g key={i}>
            <rect x={x} y={40} width={TILE_W} height={30} rx={4}
                  fill={teacherForcing ? "#dbeafe" : (isCorrupted ? "#fef3c7" : "#dbeafe")}
                  stroke={teacherForcing ? "#3b82f6" : (isCorrupted ? "#f59e0b" : "#3b82f6")}
                  strokeWidth={1.4} />
            <text x={x + TILE_W / 2} y={60} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">
              {inputTok}
            </text>
            <Arrow x1={x + TILE_W / 2} y1={70} x2={x + TILE_W / 2} y2={98} color="#9ca3af" id={`da-${i}`} />
          </g>
        );
      })}

      <text x={startX - 8} y={60} textAnchor="end" fontSize={10} fill="#6b7280">输入 y_{"{t-1}"}:</text>
      <text x={startX - 8} y={120} textAnchor="end" fontSize={10} fill="#6b7280">输出 y_t:</text>

      <text x={W / 2} y={180} textAnchor="middle" fontSize={11} fontWeight={700}
            fill={teacherForcing ? "#065f46" : "#831843"}>
        {teacherForcing
          ? "输入永远用 ground truth,梯度不受预测错误影响"
          : errorAtStep >= 0
          ? `第 ${errorAtStep} 步预测错误后,后续输入全部基于错误延续 → 误差累积`
          : "推理时输入用模型自己上一步的预测"}
      </text>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        p(y_t | y_{"{<t}"}, x) = softmax(W_o s_t) · 逐字预测,后来成为所有 LM 的标准生成方式
      </text>
    </svg>
  );
}
