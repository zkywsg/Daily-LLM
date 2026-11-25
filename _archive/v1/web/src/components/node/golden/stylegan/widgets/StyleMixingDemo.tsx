const W = 700;
const H = 340;

interface Props {
  splitLayer: number;
}

// 用 SVG 画 3 张 "人脸" 的抽象化:A、B、mix。用 style 参数(colors, shapes) 演示
// A: 深色皮肤 + 圆脸 + 短发
// B: 浅色皮肤 + 长脸 + 长发
// mix: 前 splitLayer 层用 A,后续用 B

interface FaceStyle {
  skinColor: string;
  faceElongation: number;  // 1.0 圆脸, 1.3 长脸
  hairColor: string;
  hairLength: number;      // 0.3 短, 0.8 长
  eyeColor: string;
  lipColor: string;
}

const STYLE_A: FaceStyle = {
  skinColor: "#d4a574", faceElongation: 1.0, hairColor: "#3d2817",
  hairLength: 0.3, eyeColor: "#5a3820", lipColor: "#a04040"
};
const STYLE_B: FaceStyle = {
  skinColor: "#f3d5b5", faceElongation: 1.3, hairColor: "#c0a06a",
  hairLength: 0.8, eyeColor: "#3060a0", lipColor: "#cc6060"
};

// 混合:粗(0,1) → 姿态/脸型,中(2,3) → 发型,细(4-8) → 肤色/眼/唇
function mixStyle(split: number): FaceStyle {
  // faceElongation from layer 0-1 (coarse)
  const faceE = split > 0 ? STYLE_A.faceElongation : STYLE_B.faceElongation;
  // hairLength/color from layer 2-3 (medium)
  const hairL = split > 2 ? STYLE_A.hairLength : STYLE_B.hairLength;
  const hairC = split > 2 ? STYLE_A.hairColor : STYLE_B.hairColor;
  // skin/eye/lip from layer 4+ (fine)
  const skin = split > 4 ? STYLE_A.skinColor : STYLE_B.skinColor;
  const eye = split > 4 ? STYLE_A.eyeColor : STYLE_B.eyeColor;
  const lip = split > 4 ? STYLE_A.lipColor : STYLE_B.lipColor;
  return { skinColor: skin, faceElongation: faceE, hairColor: hairC, hairLength: hairL, eyeColor: eye, lipColor: lip };
}

function Face({ cx, cy, r, style, label, borderColor }: {
  cx: number; cy: number; r: number; style: FaceStyle; label: string; borderColor: string;
}) {
  const rx = r;
  const ry = r * style.faceElongation;
  return (
    <g>
      {/* border */}
      <rect x={cx - r - 8} y={cy - ry - 8} width={2 * r + 16} height={2 * ry + 24 + 16}
            rx={6} fill="none" stroke={borderColor} strokeWidth={2} strokeDasharray="4 3" />
      {/* 头发 */}
      <ellipse cx={cx} cy={cy - ry * 0.55} rx={rx * 1.15} ry={ry * 0.55 * (0.5 + style.hairLength * 0.9)}
               fill={style.hairColor} />
      {/* 脸 */}
      <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill={style.skinColor} />
      {/* 眼 */}
      <ellipse cx={cx - rx * 0.35} cy={cy - ry * 0.15} rx={rx * 0.08} ry={ry * 0.06} fill="#fff" />
      <circle cx={cx - rx * 0.35} cy={cy - ry * 0.15} r={rx * 0.05} fill={style.eyeColor} />
      <ellipse cx={cx + rx * 0.35} cy={cy - ry * 0.15} rx={rx * 0.08} ry={ry * 0.06} fill="#fff" />
      <circle cx={cx + rx * 0.35} cy={cy - ry * 0.15} r={rx * 0.05} fill={style.eyeColor} />
      {/* 唇 */}
      <ellipse cx={cx} cy={cy + ry * 0.5} rx={rx * 0.25} ry={ry * 0.06} fill={style.lipColor} />
      {/* label */}
      <text x={cx} y={cy + ry + 22} textAnchor="middle" fontSize={12} fontWeight={700} fill={borderColor}>{label}</text>
    </g>
  );
}

export function StyleMixingDemo({ splitLayer }: Props) {
  const mixed = mixStyle(splitLayer);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Style mixing demo faces">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Style Mixing — 前 {splitLayer} 层用 w_A · 后续用 w_B
      </text>

      <Face cx={110} cy={160} r={60} style={STYLE_A} label="人 A" borderColor="#ec4899" />

      {/* + */}
      <text x={240} y={165} textAnchor="middle" fontSize={28} fontWeight={700} fill="#9ca3af">+</text>

      <Face cx={370} cy={160} r={60} style={STYLE_B} label="人 B" borderColor="#10b981" />

      {/* → */}
      <text x={510} y={165} textAnchor="middle" fontSize={28} fontWeight={700} fill="#9ca3af">=</text>

      <Face cx={620} cy={160} r={60} style={mixed} label="mix 结果" borderColor="#f59e0b" />

      <text x={W / 2} y={H - 30} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        split=0 完全用 B · split=2 后 A 只贡献脸型 · split=4 后 A 贡献脸型+发型 · split≥9 完全用 A
      </text>
      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fill="#9ca3af">
        脸型来自粗层 (0-1) · 发型来自中层 (2-3) · 肤色/眼/唇来自细层 (4+)
      </text>
    </svg>
  );
}
