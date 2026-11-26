// Transformer 01–03 用到的小实验。浏览器和 Node 用同一份代码，保证文章、互动页、图卡数字一致。
(function (root) {
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const gauss = (r) => () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const norm = (v) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
  const C = Math.sqrt(2 / Math.PI);
  const ACT = {
    relu: { f: (x) => Math.max(0, x), d: (x) => (x > 0 ? 1 : 0), name: "ReLU" },
    gelu: { f: (x) => 0.5 * x * (1 + Math.tanh(C * (x + 0.044715 * x ** 3))), d: (x) => { const t = Math.tanh(C * (x + 0.044715 * x ** 3)); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C * (1 + 3 * 0.044715 * x * x); }, name: "GELU" },
    silu: { f: (x) => x / (1 + Math.exp(-x)), d: (x) => { const s = 1 / (1 + Math.exp(-x)); return s + x * s * (1 - s); }, name: "SiLU" },
    linear: { f: (x) => x, d: () => 1, name: "无激活" },
  };

  // 归一化
  function layerNorm(x) { const n = x.length, mu = x.reduce((a, b) => a + b, 0) / n, s = Math.sqrt(x.reduce((a, b) => a + (b - mu) ** 2, 0) / n + 1e-5); return { y: x.map((v) => (v - mu) / s), mu, s }; }
  function rmsNorm(x) { const s = Math.sqrt(x.reduce((a, b) => a + b * b, 0) / x.length + 1e-5); return { y: x.map((v) => v / s), s }; }
  function lnBack(dy, c) { const n = dy.length, md = dy.reduce((a, b) => a + b, 0) / n, mdx = dy.reduce((a, b, i) => a + b * c.y[i], 0) / n; return dy.map((v, i) => (v - md - c.y[i] * mdx) / c.s); }

  // 堆很多层，量两件事：前向时向量有多长、反向时梯度还剩多少。
  // 每层的子网络 f(z) = GELU(W z)，W 的元素 ~ N(0, gain²/d)。不训练，只看初始化时的信号传播。
  const MODES = { plain: "无残差", res: "只有残差", pre: "残差 + Pre-LN", post: "残差 + Post-LN" };
  function depth(mode, L = 24, gain = 1, d = 64, seed = 1) {
    const g = gauss(rng(seed)), A = ACT.gelu;
    const Ws = Array.from({ length: L }, () => Array.from({ length: d }, () => Array.from({ length: d }, () => (g() * gain) / Math.sqrt(d))));
    let x = Array.from({ length: d }, g);
    const fwd = [norm(x) / Math.sqrt(d)], cache = [];
    for (let l = 0; l < L; l++) {
      const W = Ws[l], pre = mode === "pre" ? layerNorm(x) : null, z = pre ? pre.y : x;
      const u = W.map((r) => r.reduce((a, w, j) => a + w * z[j], 0)), f = u.map(A.f);
      let y = mode === "plain" ? f : x.map((v, i) => v + f[i]);
      const post = mode === "post" ? layerNorm(y) : null;
      if (post) y = post.y;
      cache.push({ W, u, pre, post }); x = y; fwd.push(norm(x) / Math.sqrt(d));
    }
    const fin = mode === "pre" ? layerNorm(x) : null; // Pre-LN 结构在最后再归一化一次
    let dx = Array.from({ length: d }, g);
    const top = norm(dx), grad = new Array(L + 1);
    if (fin) dx = lnBack(dx, fin);
    grad[L] = norm(dx) / top;
    for (let l = L - 1; l >= 0; l--) {
      const { W, u, pre, post } = cache[l];
      if (post) dx = lnBack(dx, post);
      const du = u.map((v, i) => A.d(v) * dx[i]);
      let dz = new Array(d).fill(0);
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) dz[j] += W[i][j] * du[i];
      if (pre) dz = lnBack(dz, pre);
      dx = mode === "plain" ? dz : dx.map((v, i) => v + dz[i]);
      grad[l] = norm(dx) / top;
    }
    return { fwd, grad };
  }
  // 多个随机种子取几何平均，避免单次抽样的偶然性
  function depthAvg(mode, L, gain, seeds = 5) {
    const R = Array.from({ length: seeds }, (_, s) => depth(mode, L, gain, 64, s + 1));
    const gm = (k) => R[0][k].map((_, i) => Math.exp(R.reduce((a, r) => a + Math.log(r[k][i] + 1e-300), 0) / seeds));
    return { fwd: gm("fwd"), grad: gm("grad") };
  }

  // FFN：用 k 个随机隐藏单元拟合一条曲线，只解输出层（最小二乘，带极小的正则）
  const TARGET = (x) => Math.sin(2.2 * x) + 0.4 * Math.sin(5 * x);
  function fitFFN(k, act = "relu", seed = 3) {
    const r = rng(seed), A = ACT[act];
    const hid = Array.from({ length: k }, (_, i) => { const w = (r() < 0.5 ? -1 : 1) * (1 + 2.5 * r()); const c = -2.4 + 4.8 * ((i + r()) / k); return { w, b: -w * c }; });
    const feat = (x) => [1, ...hid.map((h) => A.f(h.w * x + h.b))];
    const xs = Array.from({ length: 160 }, (_, i) => -2.5 + (5 * i) / 159);
    const F = xs.map(feat), n = k + 1;
    const M = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => F.reduce((a, f) => a + f[i] * f[j], 0) + (i === j ? 1e-6 : 0)));
    const v = Array.from({ length: n }, (_, i) => F.reduce((a, f, t) => a + f[i] * TARGET(xs[t]), 0));
    for (let c = 0; c < n; c++) { // 高斯消元（带列主元）
      let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
      [M[c], M[p]] = [M[p], M[c]]; [v[c], v[p]] = [v[p], v[c]];
      for (let i = c + 1; i < n; i++) { const q = M[i][c] / (M[c][c] || 1e-12); for (let j = c; j < n; j++) M[i][j] -= q * M[c][j]; v[i] -= q * v[c]; }
    }
    const w = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = v[i]; for (let j = i + 1; j < n; j++) s -= M[i][j] * w[j]; w[i] = s / (M[i][i] || 1e-12); }
    const pred = xs.map((x) => feat(x).reduce((a, f, i) => a + f * w[i], 0));
    const mse = pred.reduce((a, p, i) => a + (p - TARGET(xs[i])) ** 2, 0) / xs.length;
    return { xs, target: xs.map(TARGET), pred, mse };
  }

  // 参数量。cfg: d, layers, ff, vocab, pos(位置表长度), swiglu, bias, tied, norm('ln'|'rms')
  function params(c) {
    const b = c.bias ? 1 : 0;
    const attn = 4 * c.d * c.d + b * 4 * c.d;
    const ffn = (c.swiglu ? 3 : 2) * c.d * c.ff + b * (c.ff + c.d);
    const norm = 2 * c.d * (c.norm === "rms" ? 1 : 2);
    const layer = attn + ffn + norm;
    const embed = c.vocab * c.d, pos = (c.pos || 0) * c.d, finalNorm = c.d * (c.norm === "rms" ? 1 : 2), head = c.tied ? 0 : c.vocab * c.d;
    return { attn, ffn, norm, layer, layers: layer * c.layers, embed, pos, finalNorm, head, total: layer * c.layers + embed + pos + finalNorm + head };
  }
  const MODELS = {
    "GPT-2 small": { d: 768, layers: 12, ff: 3072, heads: 12, vocab: 50257, pos: 1024, swiglu: false, bias: true, tied: true, norm: "ln" },
    "LLaMA 7B": { d: 4096, layers: 32, ff: 11008, heads: 32, vocab: 32000, pos: 0, swiglu: true, bias: false, tied: false, norm: "rms" },
  };
  const api = { rng, ACT, MODES, layerNorm, rmsNorm, depth, depthAvg, fitFFN, TARGET, params, MODELS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.TF = api;
})(typeof window !== "undefined" ? window : globalThis);
