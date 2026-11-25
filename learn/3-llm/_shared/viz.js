// 更多可视化：热力矩阵、全流程流水线、点积拖拽、softmax 面板、加权求和向量图
(function (root) {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const BLUE = "59,130,246", ORG = "229,165,10";
  const col = (v, m) => `rgba(${v >= 0 ? BLUE : ORG},${Math.min(0.9, Math.abs(v) / (m || 1) * 0.9 + 0.04).toFixed(2)})`;

  // 热力矩阵：返回 {svg,w,h}，可嵌入大图。x,y 为左上角
  function heat(M, o = {}) {
    const c = o.cell || 26, x = o.x || 0, y = o.y || 0, nums = o.nums !== false;
    const m = o.max || Math.max(1e-9, ...M.flat().map(Math.abs));
    let s = "";
    if (o.title) s += `<text x="${x}" y="${y - 36}" class="vt">${esc(o.title)}</text>`;
    if (o.sub) s += `<text x="${x}" y="${y - 14}" class="vs">${esc(o.sub)}</text>`;
    const G = o.gut || 40, lx0 = o.rows ? G : 0;
    (o.cols || []).forEach((t, j) => (s += `<text x="${x + lx0 + j * c + c / 2}" y="${y + 11}" text-anchor="middle" class="vs">${esc(t)}</text>`));
    const y0 = y + (o.cols ? 16 : 0), lx = o.rows ? G : 0;
    M.forEach((r, i) => {
      if (o.rows) s += `<text x="${x + G - 6}" y="${y0 + i * c + c / 2 + 4}" text-anchor="end" class="vs">${esc(o.rows[i])}</text>`;
      r.forEach((v, j) => {
        if (o.skip && o.skip(i, j)) return;
        s += `<rect x="${x + lx + j * c}" y="${y0 + i * c}" width="${c - 1}" height="${c - 1}" rx="2" fill="${col(v, m)}"/>`;
        if (nums) s += `<text x="${x + lx + j * c + c / 2}" y="${y0 + i * c + c / 2 + 3.5}" text-anchor="middle" class="vn">${o.fmt ? o.fmt(v) : v.toFixed(1)}</text>`;
      });
    });
    return { svg: s, w: lx + M[0].length * c, h: (o.cols ? 16 : 0) + M.length * c };
  }

  // 全流程：X → Q K V → S → A → O，每一步都是真实矩阵的热力图
  function pipeline(R, X, toks, sel) {
    const n = toks.length, labels = toks.map((t) => t.text), c = n > 8 ? 22 : 28;
    const dim = (k) => Array.from({ length: k }, (_, i) => "abcdef"[i]);
    const parts = [];
    let x = 10, top = 56, svg = "";
    const place = (M, o) => { const r = heat(M, Object.assign({ cell: c, x, y: top }, o)); svg += r.svg; const w = r.w; x += w + 66; return { x0: x - w - 66, w, h: r.h }; };
    const arrow = (px, py, t) => `<g><line x1="${px - 58}" y1="${py}" x2="${px - 8}" y2="${py}" stroke="#8a8a8a"/><path d="M${px - 8} ${py - 4}l6 4-6 4z" fill="#8a8a8a"/><text x="${px - 33}" y="${py - 7}" text-anchor="middle" class="vs">${t}</text></g>`;
    const Xp = place(X, { title: "X", sub: `${n}×6 词向量`, rows: labels, cols: ["有", "物", "动", "状", "指", "连"], nums: false });
    // 三块并排 Q K V（竖排节省宽度）
    const qx = x; svg += arrow(qx, top + 40 + c, "×W");
    const Qp = place(R.Q, { title: "Q", sub: `${n}×3 想找`, cols: dim(3), fmt: (v) => v.toFixed(1), nums: n <= 8 });
    svg += `<text x="${Qp.x0 + Qp.w / 2}" y="${top + Qp.h + 40}" text-anchor="middle" class="vs">K 与 V 同理</text>`;
    svg += arrow(x, top + 40 + c, "Q·Kᵀ");
    const Sp = place(R.S, { title: "S", sub: `${n}×${n} 分数`, cols: labels.map((t) => t.slice(0, 1)), nums: n <= 8, fmt: (v) => v.toFixed(1) });
    svg += arrow(x, top + 40 + c, "softmax");
    const Ap = place(R.A, { title: "A", sub: `${n}×${n} 权重（行和=1）`, cols: labels.map((t) => t.slice(0, 1)), nums: n <= 8, fmt: (v) => (v * 100).toFixed(0), max: 1 });
    svg += arrow(x, top + 40 + c, "·V");
    const Op = place(R.O, { title: "O", sub: `${n}×3 新表示`, cols: dim(3), fmt: (v) => v.toFixed(1), nums: n <= 8 });
    const W = x, H = top + 16 + n * c + 52;
    // 选中行描边（同一行贯穿所有矩阵）
    const ry = top + 16 + sel * c;
    svg += `<rect x="4" y="${ry - 2}" width="${W - 8}" height="${c + 3}" rx="4" fill="none" stroke="#3b82f6" stroke-dasharray="4 3"/>`;
    return `<div class="scroll"><svg class="viz" viewBox="0 0 ${W} ${H}" width="${W}" style="max-width:none">${svg}</svg></div>`;
  }

  // 点积拖拽：二维平面上拖 q、k，看夹角与点积
  function dotDrag(el) {
    const S = 260, C = S / 2, U = 40; // 1 单位 = 40px
    let q = [2.2, 1.2], k = [1.0, 2.0], drag = null;
    el.innerHTML = `<svg class="viz drag" viewBox="0 0 ${S} ${S}" width="${S}" height="${S}"></svg><div class="mono" id="dd-out"></div>`;
    const svg = el.querySelector("svg"), out = el.querySelector("#dd-out");
    const px = (v) => [C + v[0] * U, C - v[1] * U];
    function draw() {
      const [qx, qy] = px(q), [kx, ky] = px(k);
      const dot = q[0] * k[0] + q[1] * k[1], nq = Math.hypot(...q), nk = Math.hypot(...k), cs = dot / (nq * nk || 1);
      // k 在 q 方向上的投影点
      const t = dot / (nq * nq || 1), pj = px([q[0] * t, q[1] * t]);
      svg.innerHTML = `<line x1="0" y1="${C}" x2="${S}" y2="${C}" stroke="#1f1f1f"/><line x1="${C}" y1="0" x2="${C}" y2="${S}" stroke="#1f1f1f"/>` +
        `<line x1="${C}" y1="${C}" x2="${kx}" y2="${ky}" stroke="#e5a50a" stroke-width="2"/><line x1="${C}" y1="${C}" x2="${qx}" y2="${qy}" stroke="#3b82f6" stroke-width="2"/>` +
        `<line x1="${kx}" y1="${ky}" x2="${pj[0]}" y2="${pj[1]}" stroke="#8a8a8a" stroke-dasharray="3 3"/><circle cx="${pj[0]}" cy="${pj[1]}" r="3" fill="#8a8a8a"/>` +
        `<circle data-v="q" cx="${qx}" cy="${qy}" r="9" fill="#3b82f6" style="cursor:grab"/><circle data-v="k" cx="${kx}" cy="${ky}" r="9" fill="#e5a50a" style="cursor:grab"/>` +
        `<text x="${qx + 12}" y="${qy - 8}" class="vs" fill="#3b82f6">q</text><text x="${kx + 12}" y="${ky - 8}" class="vs" fill="#e5a50a">k</text>`;
      out.textContent = `q = [${q.map((v) => v.toFixed(1))}]   k = [${k.map((v) => v.toFixed(1))}]\nq·k = ${q[0].toFixed(1)}×${k[0].toFixed(1)} + ${q[1].toFixed(1)}×${k[1].toFixed(1)} = ${dot.toFixed(2)}\n夹角 cos = ${cs.toFixed(2)}（${cs > 0.7 ? "方向很像" : cs > 0 ? "方向有点像" : cs > -0.3 ? "几乎垂直" : "方向相反"}）\n点积 = |q|·|k|·cos = ${nq.toFixed(2)}×${nk.toFixed(2)}×${cs.toFixed(2)}`;
    }
    const pos = (e) => { const r = svg.getBoundingClientRect(), s = S / r.width; return [((e.clientX - r.left) * s - C) / U, (C - (e.clientY - r.top) * s) / U]; };
    svg.addEventListener("pointerdown", (e) => { const v = e.target.dataset && e.target.dataset.v; if (v) { drag = v; svg.setPointerCapture(e.pointerId); } });
    svg.addEventListener("pointermove", (e) => { if (!drag) return; const p = pos(e).map((x) => Math.max(-3, Math.min(3, x))); if (drag === "q") q = p; else k = p; draw(); });
    svg.addEventListener("pointerup", () => (drag = null));
    draw();
  }

  // softmax 面板：4 个可拖分数 + 温度，显示 exp 与概率
  function softmaxPanel(el) {
    const names = ["A", "B", "C", "D"]; let s = [2, 1, 0.2, -1], T = 1;
    el.innerHTML = `<div id="sp-sl"></div><div><label>温度 T（分数 ÷ T）<input type="range" id="sp-t" min="0.2" max="5" step="0.1" value="1"> <b id="sp-tv">1.0</b></label></div><div id="sp-out"></div>`;
    const sl = el.querySelector("#sp-sl");
    sl.innerHTML = names.map((n, i) => `<div class="row"><span>${n}</span><input type="range" min="-4" max="6" step="0.1" value="${s[i]}" data-i="${i}" style="flex:1"><span class="num" id="sp-v${i}"></span></div>`).join("");
    function draw() {
      const z = s.map((v) => v / T), m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), sum = e.reduce((a, b) => a + b), p = e.map((v) => v / sum);
      s.forEach((v, i) => (el.querySelector("#sp-v" + i).textContent = v.toFixed(1)));
      el.querySelector("#sp-tv").textContent = T.toFixed(1);
      const bar = (v, mx, c) => `<div class="bar"><i style="width:${Math.max(1, v / mx * 100).toFixed(1)}%;background:${c}"></i></div>`;
      const emx = Math.max(...e);
      el.querySelector("#sp-out").innerHTML = `<table class="mat" style="width:100%"><tr><th></th><th>分数 ÷ T</th><th>exp（减最大值后）</th><th>概率 = exp ÷ 总和</th></tr>` +
        names.map((n, i) => `<tr><th>${n}</th><td>${z[i].toFixed(2)}</td><td>${e[i].toFixed(3)}${bar(e[i], emx, "#8a8a8a")}</td><td><b>${(p[i] * 100).toFixed(1)}%</b>${bar(p[i], 1, "#3b82f6")}</td></tr>`).join("") +
        `<tr><th>和</th><td></td><td>${sum.toFixed(3)}</td><td><b>100%</b></td></tr></table><p class="note">T 越小，赢家越“一家独大”；T 越大，越接近平均。除以 √d_k 的作用类似把 T 调大一点，防止过早饱和。分数同时加同一个数，概率不变（这就是减最大值的原因）。</p>`;
    }
    sl.oninput = (ev) => { s[+ev.target.dataset.i] = +ev.target.value; draw(); };
    el.querySelector("#sp-t").oninput = (ev) => { T = +ev.target.value; draw(); };
    draw();
  }

  // 加权求和：每个词的 v（3 维）按权重缩放后堆叠成 o
  function stackSum(R, toks, sel) {
    const A = R.A[sel], V = R.V, d = 3, cols = ["#3b82f6", "#e5a50a", "#8a8a8a"], names = ["a", "b", "c"];
    const rows = toks.map((t, j) => {
      const parts = V[j].map((v, k) => `<span class="seg" style="width:${Math.max(0, Math.abs(A[j] * v) * 160).toFixed(1)}px;background:${cols[k]}" title="${names[k]}: ${(A[j] * v).toFixed(2)}"></span>`).join("");
      return `<div class="srow"><span style="width:56px">${esc(t.text)}</span><span class="num" style="width:44px">${(A[j] * 100).toFixed(0)}%</span><div class="stack">${parts}</div></div>`;
    }).join("");
    const tot = R.O[sel].map((v, k) => `<span class="seg" style="width:${Math.max(0, Math.abs(v) * 160).toFixed(1)}px;background:${cols[k]}"></span>`).join("");
    return `<div class="legend"><i style="background:${cols[0]}"></i>维度 a <i style="background:${cols[1]}"></i>维度 b <i style="background:${cols[2]}"></i>维度 c　每一行 = 权重 × 该词的 v，分段长度即贡献大小</div>${rows}<div class="srow tot"><span style="width:56px"><b>合计 o</b></span><span class="num" style="width:44px"></span><div class="stack">${tot}</div></div>`;
  }


  // 拆头示意：d_model 长条切成 h 段，每段 d_k
  function splitBar(d, h) {
    const W = 640, seg = W / h, cs = ["#3b82f6", "#e5a50a", "#8a8a8a", "#22c55e", "#a78bfa", "#f43f5e"];
    let s = `<svg class="viz" viewBox="0 0 ${W + 20} 96" width="100%" style="max-width:${W + 20}px"><text x="10" y="14" class="vs">d_model = ${d}：一个词的向量</text><rect x="10" y="22" width="${W}" height="18" rx="3" fill="#333"/>`;
    s += `<text x="10" y="62" class="vs">reshape 成 h = ${h} 段，每段 d_k = ${d / h}（每段交给一个头）</text>`;
    const show = Math.min(h, 96);
    for (let i = 0; i < show; i++) s += `<rect x="${10 + i * seg + 1}" y="70" width="${Math.max(1, seg - 2)}" height="18" rx="2" fill="${cs[i % cs.length]}"/>`;
    s += `<path d="M${10 + W / 2} 42v22" stroke="#8a8a8a" stroke-dasharray="3 3"/>`;
    return s + "</svg>";
  }

  // 多头：每个头一条流水线 + 拼接
  function concatView(M, toks, sel) {
    const labels = toks.map((t) => t.text), c = 28, n = toks.length;
    const parts = [];
    let x = 10, y = 48, svg = "";
    const put = (Mx, o) => { const r = heat(Mx, Object.assign({ cell: c, x, y }, o)); svg += r.svg; const x0 = x; x += r.w + 56; return { x0, w: r.w, h: r.h }; };
    const plus = (t) => `<text x="${x - 38}" y="${y + 16 + n * c / 2}" text-anchor="middle" style="font:16px var(--mono)" fill="#8a8a8a">${t}</text>`;
    put(M.heads[0].O, { title: "头 1 输出 O₁", sub: `${n}×3`, rows: labels, cols: ["a", "b", "c"], fmt: (v) => v.toFixed(1) });
    svg += plus("‖");
    put(M.heads[1].O, { title: "头 2 输出 O₂", sub: `${n}×3`, cols: ["a", "b", "c"], fmt: (v) => v.toFixed(1) });
    svg += plus("=");
    put(M.concat, { title: "拼接 concat", sub: `${n}×6（再乘 W_O）`, cols: ["a", "b", "c", "a", "b", "c"], fmt: (v) => v.toFixed(1) });
    const H = y + 16 + n * c + 12, ry = y + 16 + sel * c;
    svg += `<rect x="4" y="${ry - 2}" width="${x - 20}" height="${c + 3}" rx="4" fill="none" stroke="#3b82f6" stroke-dasharray="4 3"/>`;
    return `<div class="scroll"><svg class="viz" viewBox="0 0 ${x} ${H}" width="${x}" style="max-width:none">${svg}</svg></div>`;
  }

  // 掩码：分数 S → 加掩码 → S+M → softmax A
  function maskFlow(Rmask, Rplain, toks, on) {
    const n = toks.length, c = n > 8 ? 24 : 30, labels = toks.map((t) => t.text.slice(0, 1));
    const S = Rplain.S, Mk = S.map((r, i) => r.map((_, j) => (j > i ? -1 : 0)));
    const inf = (v) => (v <= -9 ? "−∞" : v.toFixed(1));
    const SM = S.map((r, i) => r.map((v, j) => (on && j > i ? -99 : v)));
    let x = 10, y = 48, svg = "";
    const put = (Mx, o) => { const r = heat(Mx, Object.assign({ cell: c, x, y, cols: labels }, o)); svg += r.svg; x += r.w + 60; };
    const ar = (t) => `<text x="${x - 40}" y="${y + 16 + n * c / 2}" text-anchor="middle" style="font:15px var(--mono)" fill="#8a8a8a">${t}</text>`;
    put(S, { title: "分数 S", sub: "q·k/√d", rows: toks.map((t) => t.text), fmt: (v) => v.toFixed(1), nums: n <= 7 });
    svg += ar("+");
    put(on ? Mk.map((r) => r.map((v) => v * 99)) : Mk.map((r) => r.map(() => 0)), { title: "掩码 M", sub: on ? "未来位置 = −∞" : "未开启（全 0）", fmt: (v) => (v < -9 ? "−∞" : "0"), nums: n <= 7, max: 99 });
    svg += ar("=");
    put(SM, { title: "S + M", sub: "", fmt: inf, nums: n <= 7, max: 3 });
    svg += ar("softmax");
    put(Rmask.A, { title: "权重 A", sub: "−∞ → 0，行和仍为 1", fmt: (v) => (v * 100).toFixed(0), nums: n <= 7, max: 1 });
    return `<div class="scroll"><svg class="viz" viewBox="0 0 ${x} ${y + 16 + n * c + 10}" width="${x}" style="max-width:none">${svg}</svg></div>`;
  }

  // 位置编码热力图 + 每一维随位置的波形
  function peView(posEnc, count, dims) {
    const P = Array.from({ length: count }, (_, i) => posEnc(i)), W = 520, H = 150, cs = ["#3b82f6", "#e5a50a", "#8a8a8a", "#22c55e", "#a78bfa", "#f43f5e"];
    const hm = heat(P, { cell: 18, x: 10, y: 34, title: "每行一个位置，每列一个维度", cols: Array.from({ length: dims }, (_, j) => "d" + (j + 1)), rows: P.map((_, i) => "位" + (i + 1)), nums: false });
    const hs = 34 + hm.h + 20, lx = 10, mn = Math.min(...P.flat()), mx = Math.max(...P.flat());
    let lines = `<text x="${lx}" y="${hs}" class="vt">每一维随位置的波形（慢的维度像“粗刻度”，快的像“细刻度”）</text>`;
    for (let j = 0; j < dims; j++) {
      const pts = P.map((r, i) => `${(lx + 30 + i * (W - 60) / Math.max(1, count - 1)).toFixed(1)},${(hs + 14 + H - (r[j] - mn) / (mx - mn || 1) * H).toFixed(1)}`).join(" ");
      lines += `<polyline points="${pts}" fill="none" stroke="${cs[j % 6]}" stroke-width="1.6"/><text x="${W - 22}" y="${hs + 14 + H - (P[count - 1][j] - mn) / (mx - mn || 1) * H}" class="vs" fill="${cs[j % 6]}">d${j + 1}</text>`;
    }
    return `<svg class="viz" viewBox="0 0 ${W + 10} ${hs + H + 30}" width="100%" style="max-width:${W + 10}px">${hm.svg}${lines}</svg>`;
  }

  // 张量形状：每一步的元素个数（对数长度条）+ 方块比例
  function shapeBars(steps, fmtN, mem) {
    const mx = Math.max(...steps.map((s) => s.el));
    return steps.map((s) => `<div class="srow"><span style="width:190px;font-size:12px;color:var(--mute)">${s.op}</span><div class="stack" style="flex:1"><span class="seg" style="width:${Math.max(0.6, s.el / mx * 100).toFixed(1)}%;background:${s.big ? "#e5a50a" : "#3b82f6"}"></span></div><span class="num" style="width:90px;text-align:right;font-size:12px">${mem(s.el)}</span></div>`).join("") + `<p class="note">条长与元素个数成正比（最长的是 100%）。橙色是 n×n 的分数表，序列一长它最先爆。</p>`;
  }
  // 方块图：[B,n,d] 与 [B,h,n,n] 面积对比（真实比例，超大时按比例缩小）
  function areaCompare(p) {
    const a = p.n * p.d, b = p.h * p.n * p.n, k = 240 / Math.sqrt(Math.max(a, b));
    const wa = Math.max(2, p.n * k * Math.sqrt(a / (p.n * p.d)) ), 
      A = { w: Math.max(2, Math.sqrt(a) * k), h: Math.max(2, Math.sqrt(a) * k) }, Bq = { w: Math.max(2, Math.sqrt(b) * k), h: Math.max(2, Math.sqrt(b) * k) };
    return `<svg class="viz" viewBox="0 0 620 270" width="100%" style="max-width:620px"><rect x="10" y="20" width="${A.w}" height="${A.h}" fill="#3b82f6"/><text x="10" y="14" class="vs">Q（n×d）= ${a.toLocaleString("zh-CN")} 个</text><rect x="${320}" y="20" width="${Bq.w}" height="${Bq.h}" fill="#e5a50a"/><text x="320" y="14" class="vs">分数表（h×n×n）= ${b.toLocaleString("zh-CN")} 个</text></svg><p class="note">两个方块的面积和元素个数成正比（每批 1 个句子）。拖动 n，看橙色方块多快超过蓝色。</p>`;
  }


  // 矩阵乘法可视化：L 的第 i 行 × R 的第 j 列 = 结果格 (i,j)
  function matmul(L, Rm, i, j, names, rl, cl) {
    const c = 34, fm = (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2));
    const Res = L.map((r, a) => Rm[0].map((_, b) => r.reduce((t, v, k) => t + v * Rm[k][b], 0)));
    let x = 10, y = 40, svg = "";
    const box = (M, nm, hr, hc, hcell, rows, cols) => {
      const r = heat(M, { cell: c, x, y, title: nm, fmt: fm, rows, cols, max: Math.max(1, ...M.flat().map(Math.abs)) }); svg += r.svg;
      const lx = rows ? 40 : 0, y0 = y + (cols ? 16 : 0);
      if (hr != null) svg += `<rect x="${x + lx - 2}" y="${y0 + hr * c - 2}" width="${M[0].length * c + 3}" height="${c + 2}" rx="3" fill="none" stroke="#e5a50a" stroke-width="2"/>`;
      if (hc != null) svg += `<rect x="${x + lx + hc * c - 2}" y="${y0 - 2}" width="${c + 2}" height="${M.length * c + 3}" rx="3" fill="none" stroke="#e5a50a" stroke-width="2"/>`;
      if (hcell) svg += `<rect x="${x + lx + hcell[1] * c - 2}" y="${y0 + hcell[0] * c - 2}" width="${c + 2}" height="${c + 2}" rx="3" fill="none" stroke="#fff" stroke-width="2"/>`;
      x += r.w + 46; return r;
    };
    box(L, names[0] + `（第 ${i + 1} 行）`, i, null, null, rl, ["a", "b", "c", "d"].slice(0, L[0].length));
    svg += `<text x="${x - 30}" y="${y + 40}" text-anchor="middle" style="font:18px var(--mono)" fill="#8a8a8a">×</text>`;
    box(Rm, names[1] + `（第 ${j + 1} 列）`, null, j, null, null, cl);
    svg += `<text x="${x - 30}" y="${y + 40}" text-anchor="middle" style="font:18px var(--mono)" fill="#8a8a8a">=</text>`;
    box(Res, names[2], null, null, [i, j], rl, cl);
    return `<div class="scroll"><svg class="viz" viewBox="0 0 ${x} ${y + 20 + Math.max(L.length, Rm.length) * c + 20}" width="${x}" style="max-width:none">${svg}</svg></div><p class="note">橙框：用到的一行和一列，对应位置相乘再相加，得到白框那一格。</p>`;
  }
  // 全貌：X、W_Q、W_K、W_V、Q、K、V、S、A、O 一排热力图，当前步输出高亮
  function chain(R, st, cur, TOK) {
    const c = 22, items = [["X", st.X], ["Q", R.Q], ["K", R.K], ["V", R.V], ["S", R.Sp], ["A", R.A], ["O", R.O]], keyOf = { 0: "Q", 1: "K", 2: "V", 3: "S", 4: "S", 5: "A", 6: "A", 7: "O" };
    let x = 10, svg = "";
    items.forEach(([nm, M], idx) => {
      const r = heat(M, { cell: c, x, y: 36, title: nm, fmt: (v) => (Math.abs(v) < 1 && v !== 0 ? v.toFixed(1) : String(Math.round(v * 10) / 10)), cols: M[0].length === 3 ? TOK : ["a", "b"], max: nm === "A" ? 1 : undefined });
      svg += r.svg;
      if (keyOf[cur] === nm) svg += `<rect x="${x - 4}" y="14" width="${r.w + 8}" height="${r.h + 52}" rx="5" fill="none" stroke="#3b82f6" stroke-width="2"/>`;
      x += r.w + 30;
    });
    return `<div class="scroll"><svg class="viz" viewBox="0 0 ${x} 142" width="${x}" style="max-width:none">${svg}</svg></div>`;
  }

  root.VIZ = { matmul, chain, heat, pipeline, dotDrag, softmaxPanel, stackSum, splitBar, concatView, maskFlow, peView, shapeBars, areaCompare };
})(window);
