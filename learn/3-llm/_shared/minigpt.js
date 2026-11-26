// 迷你 GPT 的浏览器 / Node 端推理（只做前向）。结构与 tf-04-mini-gpt/code/minigpt.py 一致。
(function (root) {
  const C = Math.sqrt(2 / Math.PI);
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(C * (x + 0.044715 * x * x * x)));
  function softmax(z, t = 1) {
    const m = Math.max(...z), e = z.map((v) => Math.exp((v - m) / t)), s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }
  function load(M) {
    const { d, d_ff, n_layers, n_heads, ctx, vocab } = M.cfg, W = M.weights, dk = d / n_heads;
    const mat = (k, r, c) => Array.from({ length: r }, (_, i) => W[k].slice(i * c, (i + 1) * c));
    const mul = (x, w) => { // 行向量 x 乘矩阵 w
      const out = new Array(w[0].length).fill(0);
      for (let i = 0; i < x.length; i++) { const xi = x[i], row = w[i]; for (let j = 0; j < out.length; j++) out[j] += xi * row[j]; }
      return out;
    };
    const ln = (x, g, b) => {
      const mu = x.reduce((a, v) => a + v, 0) / x.length;
      const s = Math.sqrt(x.reduce((a, v) => a + (v - mu) * (v - mu), 0) / x.length + 1e-5);
      return x.map((v, i) => g[i] * (v - mu) / s + b[i]);
    };
    const add = (a, b) => a.map((v, i) => v + b[i]);
    const E = mat("E", vocab, d), P = mat("P", ctx, d);
    const L = Array.from({ length: n_layers }, (_, l) => ({
      Wq: mat(l + ".Wq", d, d), Wk: mat(l + ".Wk", d, d), Wv: mat(l + ".Wv", d, d), Wo: mat(l + ".Wo", d, d),
      W1: mat(l + ".W1", d, d_ff), W2: mat(l + ".W2", d_ff, d), b1: W[l + ".b1"], b2: W[l + ".b2"],
      g1: W[l + ".ln1.g"], c1: W[l + ".ln1.b"], g2: W[l + ".ln2.g"], c2: W[l + ".ln2.b"],
    }));
    // ids: 整数数组（超过 ctx 时只取末尾）。返回最后一个位置的 logits、每层每头的注意力矩阵、逐层的“半成品”预测
    function forward(ids) {
      ids = ids.slice(-ctx);
      const n = ids.length;
      let x = ids.map((id, i) => add(E[id], P[i]));
      const attn = [], lens = [];
      const toLogits = (row) => { const y = ln(row, W["lnf.g"], W["lnf.b"]); return E.map((e) => e.reduce((a, v, k) => a + v * y[k], 0)); };
      lens.push(toLogits(x[n - 1]));
      for (const B of L) {
        const z = x.map((r) => ln(r, B.g1, B.c1));
        const Q = z.map((r) => mul(r, B.Wq)), K = z.map((r) => mul(r, B.Wk)), V = z.map((r) => mul(r, B.Wv));
        const O = Array.from({ length: n }, () => new Array(d).fill(0)), heads = [];
        for (let h = 0; h < n_heads; h++) {
          const A = [];
          for (let i = 0; i < n; i++) {
            const s = [];
            for (let j = 0; j <= i; j++) { let t = 0; for (let k = h * dk; k < (h + 1) * dk; k++) t += Q[i][k] * K[j][k]; s.push(t / Math.sqrt(dk)); }
            const a = softmax(s);
            for (let j = 0; j <= i; j++) for (let k = h * dk; k < (h + 1) * dk; k++) O[i][k] += a[j] * V[j][k];
            A.push(a.concat(new Array(n - i - 1).fill(0)));
          }
          heads.push(A);
        }
        attn.push(heads);
        x = x.map((r, i) => add(r, mul(O[i], B.Wo)));
        x = x.map((r) => { const u = add(mul(ln(r, B.g2, B.c2), B.W1), B.b1).map(gelu); return add(r, add(mul(u, B.W2), B.b2)); });
        lens.push(toLogits(x[n - 1]));
      }
      return { logits: lens[lens.length - 1], attn, lens, ids };
    }
    const stoi = new Map(M.chars.map((c, i) => [c, i]));
    return { forward, encode: (s) => [...s].filter((c) => stoi.has(c)).map((c) => stoi.get(c)), decode: (a) => a.map((i) => M.chars[i]).join(""), has: (c) => stoi.has(c) };
  }
  // 按温度和 top-k 得到下一个字的概率；temperature=0 表示总选最大
  function nextProbs(logits, temperature = 1, topk = 0) {
    if (temperature <= 0) { const m = logits.indexOf(Math.max(...logits)); return logits.map((_, i) => (i === m ? 1 : 0)); }
    let p = softmax(logits, temperature);
    if (topk > 0 && topk < p.length) {
      const cut = p.slice().sort((a, b) => b - a)[topk - 1];
      p = p.map((v) => (v >= cut ? v : 0));
      const s = p.reduce((a, b) => a + b, 0); p = p.map((v) => v / s);
    }
    return p;
  }
  // 20 道知识题：按语料规则续写。off = {layer, head} 时把那个头的输出置零（消融）
  const FACTS = { 小猫: ["鱼", "沙发上", "睡觉"], 小狗: ["骨头", "院子里", "跑步"], 小鸟: ["虫子", "大树上", "唱歌"], 兔子: ["胡萝卜", "草地上", "蹦跳"], 熊猫: ["竹子", "高山里", "爬树"] };
  const questions = () => Object.entries(FACTS).flatMap(([s, [f, p, a]]) => [["喜欢吃", f + "。"], ["住在", p + "。"], ["最爱", a + "。"], ["吃了", f + "，很开心。"]].map(([slot, rest]) => ({ head: s + slot, rest })));
  function ablate(M, off) {
    if (!off.length) return M;
    const { d, n_heads } = M.cfg, dk = d / n_heads, weights = Object.assign({}, M.weights);
    off.forEach(({ layer, head }) => { const w = weights[layer + ".Wo"].slice(); for (let i = head * dk; i < (head + 1) * dk; i++) for (let j = 0; j < d; j++) w[i * d + j] = 0; weights[layer + ".Wo"] = w; });
    return Object.assign({}, M, { weights });
  }
  function quiz(model) {
    return questions().map((q) => {
      const ids = model.encode("。" + q.head), n = [...q.rest].length;
      for (let i = 0; i < n; i++) { const l = model.forward(ids).logits; ids.push(l.indexOf(Math.max(...l))); }
      const got = model.decode(ids.slice(-n));
      return Object.assign({ got, ok: got === q.rest }, q);
    });
  }
  const api = { load, softmax, nextProbs, gelu, ablate, quiz, questions };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.MiniGPT = api;
})(typeof window !== "undefined" ? window : globalThis);
