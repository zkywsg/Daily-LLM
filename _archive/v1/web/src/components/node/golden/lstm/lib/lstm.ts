// LSTM 数学:单步 cell 状态更新 + 梯度衰减对比。

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * 单步 LSTM:
 *   f_t = σ(W_f · [h, x] + b_f)   forget gate
 *   i_t = σ(W_i · [h, x] + b_i)   input gate
 *   o_t = σ(W_o · [h, x] + b_o)   output gate
 *   g_t = tanh(W_g · [h, x] + b_g) candidate
 *   C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
 *   h_t = o_t ⊙ tanh(C_t)
 *
 * 这里把 W 简化为标量 demo:viewer 拖 f/i/o slider 就能直接看到 C 和 h 怎么变。
 */
export interface LstmStep {
  f: number;
  i: number;
  o: number;
  g: number;
  prevC: number;
  prevH: number;
  newC: number;
  newH: number;
}

export function lstmStep(
  f: number,
  i: number,
  o: number,
  g: number,
  prevC: number,
  prevH: number,
): LstmStep {
  const newC = f * prevC + i * g;
  const newH = o * Math.tanh(newC);
  void prevH; // 简化:这里只演示 h_t,prevH 在真模型里通过 W_h 参与 gate 计算
  return { f, i, o, g, prevC, prevH, newC, newH };
}

/**
 * 模拟梯度衰减(BPTT):
 *   Vanilla RNN: ∂L/∂h_0 ~ Π_{t} (W * f'(z))  → 反复乘 0.7 这种值,T 步后接近 0
 *   LSTM cell : ∂L/∂C_0 ~ Π_{t} f_t            → f_t 通常 ≈ 0.9-1.0,接近恒等,梯度保留
 *
 * 返回两条曲线随 timestep 衰减的值。
 */
export function gradientFlow(timesteps: number, forgetMean = 0.95): {
  rnn: number[];
  lstm: number[];
} {
  const rnn: number[] = [];
  const lstm: number[] = [];
  let rnnGrad = 1.0;
  let lstmGrad = 1.0;
  for (let t = 0; t < timesteps; t++) {
    // RNN: 每步乘 ~0.6(典型 tanh' · W)
    rnnGrad *= 0.6;
    // LSTM: 每步乘 forgetMean(forget gate 通常很接近 1)
    lstmGrad *= forgetMean;
    rnn.push(rnnGrad);
    lstm.push(lstmGrad);
  }
  return { rnn, lstm };
}

/**
 * 演示 h vs C 的两条 channel:
 * h 频繁更新(因每步都被 o_t · tanh(C) 重塑)
 * C 缓慢累积(因每步只是加一个 i_t · g_t)
 * 用一段 N 步\"输入信号\"驱动,固定 gate 值。
 */
export function simulateSequence(
  steps: number,
  fSeq: number[],
  iSeq: number[],
  oSeq: number[],
  gSeq: number[],
): { hHistory: number[]; cHistory: number[]; fHistory: number[] } {
  const hHistory: number[] = [];
  const cHistory: number[] = [];
  const fHistory: number[] = [];
  let C = 0;
  let h = 0;
  for (let t = 0; t < steps; t++) {
    const f = fSeq[t % fSeq.length];
    const i = iSeq[t % iSeq.length];
    const o = oSeq[t % oSeq.length];
    const g = gSeq[t % gSeq.length];
    const step = lstmStep(f, i, o, g, C, h);
    C = step.newC;
    h = step.newH;
    hHistory.push(h);
    cHistory.push(C);
    fHistory.push(f);
  }
  return { hHistory, cHistory, fHistory };
}

/**
 * 一段 demo \"句子\" 的逐 step 输入信号 (g 值)。
 * 让 viewer 看到 LSTM 能记住远处 token 的特征。
 */
export const DEMO_SCENARIOS = [
  {
    label: "稳定累积",
    desc: "f≈0.95, i≈0.3, g 在 0.5 附近 → C 缓慢上升,h 跟随",
    fSeq: [0.95],
    iSeq: [0.3],
    oSeq: [0.5],
    gSeq: [0.5],
  },
  {
    label: "周期性清零",
    desc: "f 在第 4/8/12 步 ≈ 0 → forget gate 主动清空 cell state",
    fSeq: [0.95, 0.95, 0.95, 0.02, 0.95, 0.95, 0.95, 0.02],
    iSeq: [0.4],
    oSeq: [0.6],
    gSeq: [0.7],
  },
  {
    label: "Vanilla RNN 衰减",
    desc: "f 固定 0.6 模拟 vanilla RNN → C 像 RNN h 一样指数衰减",
    fSeq: [0.6],
    iSeq: [0.1],
    oSeq: [0.5],
    gSeq: [0.8],
  },
];
