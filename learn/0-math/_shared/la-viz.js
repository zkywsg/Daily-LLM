// 线性代数的图：平面（网格变形、箭头、单位圆）、灰度图、热力表、柱状图。返回 {svg,w,h}。
(function (root) {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const C = { blue: "#3b82f6", amber: "#e5a50a", green: "#22c55e", gray: "#8a8a8a", red: "#ef4444", ink: "#ededed", line: "#242424", dim: "#2a2a2a" };
  const apply = (M, p) => [M[0][0] * p[0] + M[0][1] * p[1], M[1][0] * p[0] + M[1][1] * p[1]];
  let uid = 0;

  // o: size, range（坐标范围 ±range）, title, M（画变形后的网格）, circle（画单位圆经 M 后的形状）, square（画单位方块）,
  //    vectors:[{v,color,label,dash,w}], segments:[{a,b,color,dash}], paths:[{pts,color}], dots:[{p,color,drag}]
  function plane(o) {
    const S = o.size || 300, R = o.range || 4, T = o.title ? 28 : 0, id = "clip" + (++uid) + "-" + (o.id || "p");
    const X = (x) => S / 2 + (x / R) * (S / 2), Y = (y) => T + S / 2 - (y / R) * (S / 2);
    const P = (p) => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1);
    const line = (a, b, c, w = 1, dash = "") => `<line x1="${X(a[0]).toFixed(1)}" y1="${Y(a[1]).toFixed(1)}" x2="${X(b[0]).toFixed(1)}" y2="${Y(b[1]).toFixed(1)}" stroke="${c}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
    let s = o.title ? `<text x="2" y="18" class="vt">${esc(o.title)}</text>` : "";
    s += `<clipPath id="${id}"><rect x="0" y="${T}" width="${S}" height="${S}"/></clipPath><rect x="0" y="${T}" width="${S}" height="${S}" rx="6" fill="#050505" stroke="${C.line}"/><g clip-path="url(#${id})">`;
    const N = Math.ceil(R);
    for (let i = -N; i <= N; i++) s += line([i, -R], [i, R], "#1b1b1b") + line([-R, i], [R, i], "#1b1b1b");
    if (o.M) for (let i = -8; i <= 8; i++) s += line(apply(o.M, [i, -8]), apply(o.M, [i, 8]), "#1e3a66") + line(apply(o.M, [-8, i]), apply(o.M, [8, i]), "#1e3a66");
    s += line([-R, 0], [R, 0], "#3a3a3a") + line([0, -R], [0, R], "#3a3a3a");
    if (o.square) s += `<polygon points="${[[0, 0], [1, 0], [1, 1], [0, 1]].map((p) => P(o.M ? apply(o.M, p) : p)).join(" ")}" fill="rgba(229,165,10,.22)" stroke="${C.amber}"/>`;
    if (o.circle) {
      const ring = Array.from({ length: 97 }, (_, i) => [Math.cos((i * Math.PI) / 48), Math.sin((i * Math.PI) / 48)]);
      if (o.M) s += `<polyline points="${ring.map(P).join(" ")}" fill="none" stroke="${C.gray}" stroke-dasharray="4 4"/>`;
      s += `<polyline points="${ring.map((p) => P(o.M ? apply(o.M, p) : p)).join(" ")}" fill="none" stroke="${C.blue}" stroke-width="2"/>`;
    }
    (o.paths || []).forEach((q) => (s += `<polyline points="${q.pts.map(P).join(" ")}" fill="${q.fill || "none"}" stroke="${q.color}" stroke-width="${q.w || 2}"${q.dash ? ` stroke-dasharray="${q.dash}"` : ""}/>`));
    (o.segments || []).forEach((g) => (s += line(g.a, g.b, g.color || C.gray, g.w || 1.5, g.dash || "")));
    (o.vectors || []).forEach((q) => {
      const x = X(q.v[0]), y = Y(q.v[1]), ox = X(0), oy = Y(0), a = Math.atan2(y - oy, x - ox), h = 9;
      s += `<line x1="${ox}" y1="${oy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${q.color}" stroke-width="${q.w || 2.5}"${q.dash ? ` stroke-dasharray="${q.dash}"` : ""}/>`;
      if (Math.hypot(x - ox, y - oy) > 6) s += `<path d="M${x.toFixed(1)} ${y.toFixed(1)}L${(x - h * Math.cos(a - 0.45)).toFixed(1)} ${(y - h * Math.sin(a - 0.45)).toFixed(1)}L${(x - h * Math.cos(a + 0.45)).toFixed(1)} ${(y - h * Math.sin(a + 0.45)).toFixed(1)}Z" fill="${q.color}"/>`;
      if (q.label) s += `<text x="${(x + (q.dx ?? 8)).toFixed(1)}" y="${(y + (q.dy ?? -6)).toFixed(1)}" style="fill:${q.color};font-weight:700">${esc(q.label)}</text>`;
    });
    (o.dots || []).forEach((d) => (s += `<circle cx="${X(d.p[0]).toFixed(1)}" cy="${Y(d.p[1]).toFixed(1)}" r="${d.r || 5}" fill="${d.color}"${d.drag ? ` data-drag="${d.drag}" style="cursor:grab"` : ""}/>`));
    return { svg: s + "</g>", w: S, h: S + T };
  }

  // 灰度图。取值按 0~1 截断
  function image(M, o = {}) {
    const c = o.cell || 3, T = o.title ? 26 : 0;
    let s = o.title ? `<text x="0" y="17" class="vt">${esc(o.title)}</text>` : "";
    M.forEach((r, i) => r.forEach((v, j) => { const g = Math.round(255 * Math.min(1, Math.max(0, v))); s += `<rect x="${(j * c).toFixed(2)}" y="${(T + i * c).toFixed(2)}" width="${(c + 0.35).toFixed(2)}" height="${(c + 0.35).toFixed(2)}" fill="rgb(${g},${g},${g})"/>`; }));
    return { svg: s, w: M[0].length * c, h: M.length * c + T };
  }

  // 带数值的相似度表：正为蓝，负为黄
  function heat(M, o = {}) {
    const c = o.cell || 40, G = o.rows ? 34 : 0, T = (o.title ? 24 : 0) + (o.cols ? 24 : 0), fmt = o.fmt || ((v) => v.toFixed(2));
    let s = o.title ? `<text x="${G}" y="16" class="vt">${esc(o.title)}</text>` : "";
    (o.cols || []).forEach((t, j) => (s += `<text x="${G + j * c + c / 2}" y="${T - 8}" text-anchor="middle" class="vs">${esc(t)}</text>`));
    M.forEach((r, i) => {
      if (o.rows) s += `<text x="${G - 8}" y="${T + i * c + c / 2 + 4}" text-anchor="end" class="vs">${esc(o.rows[i])}</text>`;
      r.forEach((v, j) => {
        const m = o.max || 1, a = Math.min(0.95, (Math.abs(v) / m) * 0.95 + 0.04);
        s += `<rect x="${G + j * c + 1}" y="${T + i * c + 1}" width="${c - 2}" height="${c - 2}" rx="3" fill="rgba(${v >= 0 ? "59,130,246" : "229,165,10"},${a.toFixed(2)})"/>`;
        s += `<text x="${G + j * c + c / 2}" y="${T + i * c + c / 2 + 4}" text-anchor="middle" style="font-size:${o.font || 11}px">${fmt(v)}</text>`;
      });
    });
    return { svg: s, w: G + M[0].length * c, h: T + M.length * c };
  }

  // 柱状图：前 on 根高亮
  function bars(vals, o = {}) {
    const w = o.w || 640, h = o.h || 160, T = o.title ? 26 : 4, m = Math.max(...vals, 1e-12), bw = (w - 8) / vals.length;
    let s = o.title ? `<text x="4" y="17" class="vt">${esc(o.title)}</text>` : "";
    vals.forEach((v, i) => { const bh = Math.max(1, (v / m) * (h - T - 18)); s += `<rect x="${(4 + i * bw).toFixed(1)}" y="${(h - 16 - bh).toFixed(1)}" width="${Math.max(1, bw - 2).toFixed(1)}" height="${bh.toFixed(1)}" fill="${o.on === undefined || i < o.on ? o.color || C.blue : "#2a2a2a"}"/>`; });
    s += `<text x="4" y="${h - 2}" class="vs">${esc(o.xlabel || "从大到小")}</text><text x="${w - 4}" y="${h - 2}" text-anchor="end" class="vs">最大 ${m.toFixed(m >= 10 ? 1 : 2)}</text>`;
    return { svg: s, w, h };
  }
  const wrap = (r, width) => `<svg class="viz" viewBox="0 0 ${r.w} ${r.h}" width="${width || "100%"}" style="${width ? "" : "max-width:" + r.w + "px"}" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC,sans-serif">${r.svg}</svg>`;
  const api = { plane, image, heat, bars, wrap, apply, C };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.LAV = api;
})(typeof window !== "undefined" ? window : globalThis);
