export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// GRU 单步:给定 r, z (0..1) 和 h_prev, h_cand,计算 h_t
export function gruStep(hPrev: number, r: number, z: number, xInfluence: number): { hCand: number; hNew: number } {
  // candidate 受 r 调制的历史 + x 影响
  const hCand = Math.tanh(r * hPrev * 0.8 + xInfluence);
  const hNew = (1 - z) * hPrev + z * hCand;
  return { hCand, hNew };
}

// 参数量对比 (d=hidden, x=input, 用 d=x=256 举例)
export interface ParamRow {
  unit: string;
  gates: number;
  formula: string;
  params: number;  // 在 d=256, x=256 下的参数量(单位:K)
}

const D = 256, X = 256;
export const PARAM_COMPARE: ParamRow[] = [
  { unit: "简单 RNN", gates: 1, formula: "d × (d+x)", params: D * (D + X) / 1000 },
  { unit: "GRU",       gates: 3, formula: "3 × d × (d+x)", params: 3 * D * (D + X) / 1000 },
  { unit: "LSTM",      gates: 4, formula: "4 × d × (d+x)", params: 4 * D * (D + X) / 1000 },
];

// 速度/参数对比(相对 LSTM=100%)
export const SPEED_COMPARE = {
  gru: { params: 75, trainSpeed: 82, matmulCount: 2 },
  lstm: { params: 100, trainSpeed: 100, matmulCount: 1 },
};

// h_t 轨迹模拟:20 step 序列,LSTM (h,C 两条线) vs GRU (单一 h)
export interface TrajectoryPoint {
  step: number;
  lstmH: number;
  lstmC: number;
  gruH: number;
}

export function simulateTrajectories(resetPattern: number[]): TrajectoryPoint[] {
  const steps = 20;
  const out: TrajectoryPoint[] = [];
  let lstmH = 0, lstmC = 0.5, gruH = 0.5;
  for (let t = 0; t < steps; t++) {
    const forget = 0.85 + 0.1 * Math.sin(t * 0.3);
    const inputG = 0.3 + 0.2 * Math.cos(t * 0.5);
    const cCand = Math.tanh(0.4 * Math.sin(t * 0.7));
    lstmC = forget * lstmC + inputG * cCand;
    const outG = 0.6 + 0.2 * Math.sin(t * 0.4);
    lstmH = outG * Math.tanh(lstmC);

    const z = sigmoid((forget - inputG) * 2); // 类比
    const r = resetPattern[t % resetPattern.length];
    const { hNew } = gruStep(gruH, r, z, 0.3 * Math.sin(t * 0.7));
    gruH = hNew;

    out.push({ step: t, lstmH, lstmC, gruH });
  }
  return out;
}
