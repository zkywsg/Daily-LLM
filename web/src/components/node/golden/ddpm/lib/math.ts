// DDPM 数学 —— 所有 widget 只画,数学全在这里。

export type Schedule = "linear" | "cosine";

/** β_t 序列(0..T-1) */
export function betaSchedule(T: number, kind: Schedule): number[] {
  if (kind === "linear") {
    const start = 1e-4;
    const end = 0.02;
    return Array.from({ length: T }, (_, t) => start + (end - start) * (t / Math.max(1, T - 1)));
  }
  // cosine (Nichol & Dhariwal 2021):ᾱ_t = cos²((t/T + s) / (1 + s) · π/2)
  const s = 0.008;
  const alphaBar = (t: number) => {
    const x = (t / T + s) / (1 + s);
    return Math.cos((x * Math.PI) / 2) ** 2;
  };
  const out: number[] = [];
  for (let t = 1; t <= T; t++) {
    const b = 1 - alphaBar(t) / alphaBar(t - 1);
    out.push(Math.min(0.999, Math.max(1e-5, b)));
  }
  return out;
}

/** α_t = 1 - β_t */
export function alphaFromBeta(beta: number[]): number[] {
  return beta.map((b) => 1 - b);
}

/** ᾱ_t = Π_{s=1..t} α_s (累积) */
export function alphaBarCumulative(alpha: number[]): number[] {
  const out: number[] = [];
  let acc = 1;
  for (const a of alpha) {
    acc *= a;
    out.push(acc);
  }
  return out;
}

/**
 * Forward 一步:从 x_0 直接到 x_t(close-form,不用真走 Markov 链)
 *   x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε
 * 对每个像素独立采样高斯,这里 ε 由调用方给(可重现 demo)。
 */
export function forwardSample(
  x0: number[],
  alphaBar_t: number,
  epsilon: number[],
): number[] {
  const sigA = Math.sqrt(alphaBar_t);
  const sigB = Math.sqrt(1 - alphaBar_t);
  return x0.map((v, i) => sigA * v + sigB * epsilon[i]);
}

/** 信噪比 SNR(t) = ᾱ_t / (1 - ᾱ_t) */
export function snr(alphaBar_t: number): number {
  return alphaBar_t / Math.max(1e-8, 1 - alphaBar_t);
}

/**
 * Reverse 一步(对 x_0 已知时的真后验均值,DDPM 实际用 ε_θ 近似):
 *   μ_t = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε)
 * 返回 μ_t 的逐像素值。这里 ε 来自"假设我们能看到真噪声"——
 * 实际 DDPM 用网络 ε_θ(x_t, t) 预测它。
 */
export function reverseMean(
  xt: number[],
  beta_t: number,
  alphaBar_t: number,
  epsilon: number[],
): number[] {
  const alpha_t = 1 - beta_t;
  const c1 = 1 / Math.sqrt(alpha_t);
  const c2 = beta_t / Math.sqrt(1 - alphaBar_t);
  return xt.map((x, i) => c1 * (x - c2 * epsilon[i]));
}

/**
 * 构造一个用作 demo 的 "干净 1D 信号" —— 三段正弦叠加,
 * 振幅 1,长度 n。viewer 看它从清晰 → 噪声 → 清晰。
 */
export function makeDemoSignal(n: number): number[] {
  return Array.from({ length: n }, (_, i) => {
    const x = (i / n) * 2 * Math.PI;
    return 0.6 * Math.sin(2 * x) + 0.3 * Math.cos(5 * x);
  });
}

/**
 * 可重现高斯噪声:用 token 风格的字符种子。viewer 不希望每次 re-render
 * 都换噪声样子,固定种子用 mulberry32 出来的伪随机 + Box-Muller。
 */
export function seededGaussian(n: number, seed: number): number[] {
  let s = seed >>> 0 || 1;
  const rng = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out: number[] = [];
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(1e-9, rng());
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    out.push(r * Math.cos(2 * Math.PI * u2));
    if (i + 1 < n) out.push(r * Math.sin(2 * Math.PI * u2));
  }
  return out;
}
