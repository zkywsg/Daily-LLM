// 注意力教学引擎 · learn/3-llm/_shared
// 浏览器与 Node 通用。互动版和图卡都用它，保证数字一致。
// 重要：词表、特征和所有投影矩阵都是人工设定的“玩具”，用来演示机制，
//       不是训练得到的模型。未收录的词用随机向量，权重没有语义。
(function (root) {
  const FEATURES = ["有生命", "物体", "动作", "状态", "指代", "连接"];
  const D = FEATURES.length;   // d_model = 6
  const DK = 3;                // 每个头的 d_k = 3

  // 词表：词 → 6 维特征 [有生命, 物体, 动作, 状态, 指代, 连接]
  const LEX = {
    小猫: [1, 0, 0, 0.2, 0, 0], 小狗: [1, 0, 0, 0.2, 0, 0], 孩子: [1, 0, 0, 0.2, 0, 0], 老师: [1, 0, 0, 0.2, 0, 0],
    马路: [0.1, 1, 0, 0, 0, 0], 苹果: [0, 1, 0, 0, 0, 0], 球: [0, 1, 0, 0, 0, 0], 书: [0, 1, 0, 0, 0, 0], 桌子: [0, 1, 0, 0, 0, 0],
    过: [0, 0.2, 1, 0.1, 0, 0], 吃: [0, 0, 1, 0, 0, 0], 追: [0, 0, 1, 0, 0, 0], 看: [0, 0, 1, 0, 0, 0], 读: [0, 0, 1, 0, 0, 0], 推: [0, 0, 1, 0, 0, 0],
    太累了: [0.3, 0, 0, 1, 0, 0], 饿了: [0.3, 0, 0, 1, 0, 0], 很开心: [0.3, 0, 0, 1, 0, 0], 很大: [0, 0.3, 0, 1, 0, 0],
    它: [0.2, 0, 0, 0, 1, 0], 他: [0.3, 0, 0, 0, 1, 0], 她: [0.3, 0, 0, 0, 1, 0],
    因为: [0, 0, 0, 0.3, 0, 1], 但是: [0, 0, 0, 0.3, 0, 1], 所以: [0, 0, 0, 0.3, 0, 1], 没有: [0, 0, 0.3, 0.3, 0, 0.3],
  };
  const WORDS = Object.keys(LEX).sort((a, b) => b.length - a.length);

  function hash(s) { let h = 2166136261; for (const c of s) { h ^= c.codePointAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  // 最长匹配分词；标点丢弃；未收录的字单独成词并标记
  function tokenize(text) {
    const out = []; const s = text.replace(/[，。！？、,.!?\s]/g, "");
    for (let i = 0; i < s.length;) {
      const w = WORDS.find((w) => s.startsWith(w, i));
      if (w) { out.push({ text: w, known: true }); i += w.length; }
      else { out.push({ text: s[i], known: false }); i += 1; }
    }
    return out;
  }
  function embedToken(t) {
    if (t.known) return LEX[t.text].slice();
    const r = rng(hash(t.text)); return Array.from({ length: D }, () => +(r() * 0.8 - 0.4).toFixed(2));
  }
  // 位置编码（论文式的正弦/余弦）；演示里把底数取 100，便于在短句里看到差异
  function posEnc(pos, scale = 0.5, base = 100) {
    return Array.from({ length: D }, (_, i) => { const j = i >> 1; const a = pos / Math.pow(base, (2 * j) / D); return scale * (i % 2 ? Math.cos(a) : Math.sin(a)); });
  }
  function embed(tokens, opts = {}) {
    return tokens.map((t, i) => { const e = embedToken(t); if (opts.pos) { const p = posEnc(i); return e.map((v, d) => v + p[d]); } return e; });
  }

  // 每个头的投影矩阵（D×DK）。行：有生命 物体 动作 状态 指代 连接
  const HEADS = [
    { name: "头 1：指代", note: "代词去找有生命的词和状态词",
      Wq: [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0], [0, 0, 0.5], [2.6, 0, 1.2], [0, 0, 0.3]],
      Wk: [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
      Wv: [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]] },
    { name: "头 2：动作与对象", note: "动作去找宾语，连接词去找动作",
      Wq: [[0, 0, 1.2], [0, 0, 0], [2.4, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 2.4]],
      Wk: [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
      Wv: [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]] },
  ];

  const mv = (x, W) => W[0].map((_, j) => x.reduce((s, v, i) => s + v * W[i][j], 0));
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  function softmax(z) { const m = Math.max(...z.filter(isFinite)); const e = z.map((v) => (isFinite(v) ? Math.exp(v - m) : 0)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }

  // 单个头：X 为 n×D。opts: scale(是否除以√dk)、mask(因果掩码)
  function head(X, h, opts = {}) {
    const Q = X.map((x) => mv(x, h.Wq)), K = X.map((x) => mv(x, h.Wk)), V = X.map((x) => mv(x, h.Wv));
    const div = opts.scale === false ? 1 : Math.sqrt(DK);
    const S = Q.map((q, i) => K.map((k, j) => (opts.mask && j > i ? -Infinity : dot(q, k) / div)));
    const A = S.map(softmax);
    const O = A.map((a) => V[0].map((_, d) => a.reduce((s, w, j) => s + w * V[j][d], 0)));
    return { Q, K, V, S, A, O };
  }
  // 多头：各头输出拼接（Wo 为简化设为单位阵）
  function multihead(X, opts = {}) {
    const hs = HEADS.map((h) => head(X, h, opts));
    return { heads: hs, concat: X.map((_, i) => hs.flatMap((h) => h.O[i])) };
  }

  // 随机向量实验：缩放前后的平均最大权重
  function scaleExperiment(d, n, trials, seed) {
    const r = rng(seed); const g = () => Math.sqrt(-2 * Math.log(r() || 1e-12)) * Math.cos(2 * Math.PI * r());
    let a = 0, b = 0;
    for (let t = 0; t < trials; t++) {
      const q = Array.from({ length: d }, g); const raw = [];
      for (let j = 0; j < n; j++) raw.push(dot(q, Array.from({ length: d }, g)));
      a += Math.max(...softmax(raw)); b += Math.max(...softmax(raw.map((v) => v / Math.sqrt(d))));
    }
    return { unscaled: a / trials, scaled: b / trials };
  }
  // 打乱词序后，同一对词的注意力权重变了多少（无位置编码时应为 0）
  function shuffleDiff(tokens, perm, opts) {
    const base = head(embed(tokens, opts), HEADS[0], opts).A;
    const t2 = perm.map((i) => tokens[i]);
    const A2 = head(embed(t2, opts), HEADS[0], opts).A;
    let m = 0;
    perm.forEach((oi, ni) => perm.forEach((oj, nj) => { m = Math.max(m, Math.abs(base[oi][oj] - A2[ni][nj])); }));
    return m;
  }

  const api = { FEATURES, D, DK, LEX, HEADS, tokenize, embed, posEnc, head, multihead, softmax, scaleExperiment, shuffleDiff };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Attn = api;
})(typeof window !== "undefined" ? window : globalThis);
