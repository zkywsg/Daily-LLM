// 线性代数的计算部分。文章配图、互动页、图卡共用；浏览器和 Node 通用。
(function (root) {
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { return () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }

  /* 向量 */
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const norm = (a, p = 2) => (p === Infinity ? Math.max(...a.map(Math.abs)) : Math.pow(a.reduce((s, v) => s + Math.pow(Math.abs(v), p), 0), 1 / p));
  const cos = (a, b) => dot(a, b) / (norm(a) * norm(b) || 1);
  const angle = (a, b) => (Math.acos(Math.max(-1, Math.min(1, cos(a, b)))) * 180) / Math.PI;
  const project = (a, b) => { const t = dot(a, b) / (dot(b, b) || 1); return b.map((v) => v * t); };
  // d 维随机向量两两夹角余弦的分布
  function cosSample(d, pairs = 4000, seed = 1) {
    const g = gauss(rng(seed)), out = [];
    for (let i = 0; i < pairs; i++) {
      const x = Array.from({ length: d }, g), y = Array.from({ length: d }, g);
      out.push(cos(x, y));
    }
    const mean = out.reduce((a, v) => a + v, 0) / pairs;
    const std = Math.sqrt(out.reduce((a, v) => a + (v - mean) ** 2, 0) / pairs);
    const near = out.filter((c) => Math.abs((Math.acos(c) * 180) / Math.PI - 90) <= 10).length / pairs;
    return { values: out, mean, std, near };
  }
  function histogram(values, bins = 41, lo = -1, hi = 1) {
    const h = new Array(bins).fill(0);
    values.forEach((v) => { const k = Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins))); h[k]++; });
    return h.map((c, i) => [lo + ((i + 0.5) * (hi - lo)) / bins, c / values.length]);
  }

  /* 矩阵（按行存的二维数组） */
  const mm = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
  const mv = (A, x) => A.map((r) => dot(r, x));
  const T = (A) => A[0].map((_, j) => A.map((r) => r[j]));
  const rot = (t) => [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]];
  const det2 = (M) => M[0][0] * M[1][1] - M[0][1] * M[1][0];

  // 2×2 特征分解。实特征值时返回单位特征向量；只有一个独立方向时 vecs 长度为 1
  function eig2(M) {
    const [[a, b], [c, d]] = M, tr = a + d, det = det2(M), disc = tr * tr - 4 * det;
    if (disc < -1e-10) return { complex: true, re: tr / 2, im: Math.sqrt(-disc) / 2, vals: [], vecs: [] };
    const q = Math.sqrt(Math.max(0, disc)), vals = [(tr + q) / 2, (tr - q) / 2];
    const unit = (v) => { const n = norm(v), s = v[0] < -1e-12 || (Math.abs(v[0]) <= 1e-12 && v[1] < 0) ? -1 : 1; return v.map((x) => (s * x) / n); };
    if (Math.abs(b) < 1e-9 && Math.abs(c) < 1e-9) {
      return Math.abs(a - d) < 1e-9 ? { complex: false, all: true, vals: [a, d], vecs: [[1, 0], [0, 1]] } : { complex: false, vals: [a, d], vecs: [[1, 0], [0, 1]] };
    }
    const vec = (l) => unit(Math.abs(b) > Math.abs(c) ? [b, l - a] : [l - d, c]);
    if (q < 1e-6) return { complex: false, defective: true, vals, vecs: [vec(vals[0])] };
    return { complex: false, vals, vecs: vals.map(vec) };
  }
  // 反复乘 A 并归一化
  function powerIter(A, v0, steps) {
    let v = v0.map((x) => x / norm(v0));
    const out = [{ v, rq: dot(v, mv(A, v)) }];
    for (let k = 0; k < steps; k++) { const w = mv(A, v), n = norm(w) || 1; v = w.map((x) => x / n); out.push({ v, rq: dot(v, mv(A, v)) }); }
    return out;
  }
  // 不归一化，记录每一步的长度
  function growth(A, x0, steps) {
    let x = x0.slice();
    const out = [norm(x)];
    for (let k = 0; k < steps; k++) { x = mv(A, x); out.push(norm(x)); }
    return out;
  }

  /* SVD：单边 Jacobi。返回 A ≈ U diag(s) Vᵀ，U、V 按列存为 U[列号]、V[列号] */
  function svd(A) {
    if (A.length < A[0].length) { const r = svd(T(A)); return { U: r.V, s: r.s, V: r.U }; }
    const n = A[0].length, C = T(A).map((c) => c.slice()), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    const spin = (M, p, q, c, s) => { const a = M[p], b = M[q]; for (let i = 0; i < a.length; i++) { const x = a[i], y = b[i]; a[i] = c * x - s * y; b[i] = s * x + c * y; } };
    for (let sweep = 0; sweep < 60; sweep++) {
      let moved = 0;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        const al = dot(C[p], C[p]), be = dot(C[q], C[q]), ga = dot(C[p], C[q]);
        if (Math.abs(ga) <= 1e-12 * Math.sqrt(al * be) || al * be < 1e-28) continue;
        const z = (be - al) / (2 * ga), t = Math.sign(z || 1) / (Math.abs(z) + Math.sqrt(1 + z * z)), c = 1 / Math.sqrt(1 + t * t);
        spin(C, p, q, c, c * t); spin(V, p, q, c, c * t); moved++;
      }
      if (!moved) break;
    }
    const s = C.map((c) => norm(c)), order = s.map((_, i) => i).sort((i, j) => s[j] - s[i]);
    return { U: order.map((i) => C[i].map((v) => (s[i] > 1e-12 ? v / s[i] : 0))), s: order.map((i) => s[i]), V: order.map((i) => V[i]) };
  }
  function rankK(f, k) {
    const m = f.U[0].length, n = f.V[0].length, R = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let t = 0; t < Math.min(k, f.s.length); t++) for (let i = 0; i < m; i++) { const u = f.U[t][i] * f.s[t]; for (let j = 0; j < n; j++) R[i][j] += u * f.V[t][j]; }
    return R;
  }
  const frob = (A) => Math.sqrt(A.reduce((a, r) => a + dot(r, r), 0));
  const relErr = (s, k) => Math.sqrt(s.slice(k).reduce((a, v) => a + v * v, 0) / s.reduce((a, v) => a + v * v, 0));

  /* 示例图片（取值 0~1）。demo 与 code/lowrank.py 用同一公式 */
  function picture(kind, n = 64) {
    const g = rng(7);
    return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
      const x = j / (n - 1), y = i / (n - 1);
      if (kind === "gradient") return 0.9 * x;
      if (kind === "checker") return (Math.floor((i * 8) / n) + Math.floor((j * 8) / n)) % 2 ? 0.85 : 0.1;
      if (kind === "noise") return g();
      let v = 0.25 * x;
      if ((x - 0.32) ** 2 + (y - 0.36) ** 2 < 0.045) v = 0.9;
      if (x > 0.55 && x < 0.9 && y > 0.5 && y < 0.85) v = 0.65;
      if (Math.abs(x + y - 1.25) < 0.025) v = 1.0;
      return v;
    }));
  }

  /* 迷你 GPT（需要先加载 3-llm 的 model.js） */
  function embeddingSim(model, chars) {
    const d = model.cfg.d, W = model.weights.E, row = (c) => { const i = model.chars.indexOf(c); return W.slice(i * d, (i + 1) * d); };
    return chars.map((a) => chars.map((b) => cos(row(a), row(b))));
  }
  const GPT_MATS = ["Wq", "Wk", "Wv", "Wo", "W1", "W2"].flatMap((n) => [`0.${n}`, `1.${n}`]);
  function gptSVD(model) {
    const { d, d_ff } = model.cfg, shape = (k) => (k.endsWith("W1") ? [d, d_ff] : k.endsWith("W2") ? [d_ff, d] : [d, d]);
    return GPT_MATS.map((k) => { const [r, c] = shape(k), W = model.weights[k]; return { name: k, r, c, f: svd(Array.from({ length: r }, (_, i) => W.slice(i * c, (i + 1) * c))) }; });
  }
  // 把 12 个权重矩阵换成秩 k 近似，返回新的模型对象（原对象不改）
  function truncateGPT(model, parts, k) {
    const weights = Object.assign({}, model.weights);
    parts.forEach((p) => (weights[p.name] = rankK(p.f, k).flat()));
    return Object.assign({}, model, { weights });
  }
  const api = { rng, gauss, dot, norm, cos, angle, project, cosSample, histogram, mm, mv, T, rot, det2, eig2, powerIter, growth, svd, rankK, frob, relErr, picture, embeddingSim, GPT_MATS, gptSVD, truncateGPT };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.LA = api;
})(typeof window !== "undefined" ? window : globalThis);
