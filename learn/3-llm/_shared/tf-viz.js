// Transformer 系列的图：折线图（可对数）、结构框图、掩码矩阵。返回 SVG 内部片段 + 尺寸；浏览器和 Node 通用。
(function (root) {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const C = { blue: "#3b82f6", amber: "#e5a50a", green: "#22c55e", gray: "#8a8a8a", red: "#ef4444", ink: "#ededed", line: "#242424", dim: "#2a2a2a" };
  const wrap = (r, width) => `<svg class="viz" viewBox="0 0 ${r.w} ${r.h}" width="${width || "100%"}" style="${width ? "" : "max-width:" + r.w + "px"}" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC,sans-serif">${r.svg}</svg>`;

  // series: [{name,color,pts:[[x,y]],dash}]。o: w,h,log,title,xlabel,ylabel,yfmt,ref:{y,label}
  function lines(series, o = {}) {
    const w = o.w || 640, h = o.h || 320, L = 62, R = 16, T = o.title ? 34 : 14, B = 40;
    const all = series.flatMap((s) => s.pts), tr = (y) => (o.log ? Math.log10(Math.max(y, 1e-30)) : y);
    const xs = all.map((p) => p[0]), ys = all.map((p) => tr(p[1])).concat(o.ref ? [tr(o.ref.y)] : []);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = o.ymin !== undefined ? tr(o.ymin) : Math.min(...ys), y1 = o.ymax !== undefined ? tr(o.ymax) : Math.max(...ys);
    if (o.log) { y0 = Math.floor(y0); y1 = Math.max(Math.ceil(y1), y0 + 1); } else if (y1 === y0) y1 += 1;
    const X = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (w - L - R), Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (h - T - B);
    let s = o.title ? `<text x="${L}" y="18" class="vt">${esc(o.title)}</text>` : "";
    const ticks = [];
    if (o.log) { const step = Math.max(1, Math.ceil((y1 - y0) / 6)); for (let e = y0; e <= y1; e += step) ticks.push([e, e === 0 ? "1" : "1e" + e]); }
    else for (let i = 0; i <= 4; i++) { const v = y0 + ((y1 - y0) * i) / 4; ticks.push([v, o.yfmt ? o.yfmt(v) : v.toFixed(1)]); }
    ticks.forEach(([v, t]) => (s += `<line x1="${L}" x2="${w - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${C.line}"/><text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end" class="vs">${t}</text>`));
    for (let i = 0; i <= 4; i++) { const v = x0 + ((x1 - x0) * i) / 4; s += `<text x="${X(v)}" y="${h - B + 18}" text-anchor="middle" class="vs">${o.xfmt ? o.xfmt(v) : Math.round(v)}</text>`; }
    if (o.xlabel) s += `<text x="${(L + w - R) / 2}" y="${h - 4}" text-anchor="middle" class="vs">${esc(o.xlabel)}</text>`;
    if (o.ref) s += `<line x1="${L}" x2="${w - R}" y1="${Y(tr(o.ref.y))}" y2="${Y(tr(o.ref.y))}" stroke="${C.gray}" stroke-dasharray="5 4"/><text x="${w - R}" y="${Y(tr(o.ref.y)) - 6}" text-anchor="end" class="vs">${esc(o.ref.label)}</text>`;
    const cl = (v) => Math.min(y1, Math.max(y0, v));
    series.forEach((q) => (s += `<polyline fill="none" stroke="${q.color}" stroke-width="${o.sw || 2.2}" ${q.dash ? `stroke-dasharray="${q.dash}"` : ""} points="${q.pts.map(([a, b]) => X(a).toFixed(1) + "," + Y(cl(tr(b))).toFixed(1)).join(" ")}"/>`));
    let lx = L;
    const legendY = h + 14;
    if (!o.noLegend) series.forEach((q) => { s += `<rect x="${lx}" y="${legendY - 8}" width="14" height="4" fill="${q.color}"/><text x="${lx + 20}" y="${legendY - 2}" class="vs">${esc(q.name)}</text>`; lx += 38 + [...q.name].length * (o.charW || 12); });
    return { svg: s, w, h: h + (o.noLegend ? 0 : 22) };
  }

  // 一个 Transformer 块的结构图。kind: 'pre' | 'post'
  function block(kind, o = {}) {
    const w = 300, h = 470, cx = 170, bw = 150;
    let s = `<text x="${cx}" y="18" text-anchor="middle" class="vt">${kind === "pre" ? "Pre-LN（现代常用）" : "Post-LN（原论文）"}</text>`;
    const box = (y, t, col) => (s += `<rect x="${cx - bw / 2}" y="${y}" width="${bw}" height="38" rx="7" fill="${col === "n" ? "#10233f" : col === "a" ? C.blue : "#161616"}" stroke="${col === "a" ? C.blue : col === "n" ? C.blue : "#3a3a3a"}"/><text x="${cx}" y="${y + 25}" text-anchor="middle">${t}</text>`);
    const plus = (y) => (s += `<circle cx="${cx}" cy="${y}" r="13" fill="#000" stroke="${C.amber}" stroke-width="2"/><text x="${cx}" y="${y + 5}" text-anchor="middle" style="fill:${C.amber}">+</text>`);
    const v = (a, b) => (s += `<line x1="${cx}" y1="${a}" x2="${cx}" y2="${b}" stroke="#555" stroke-width="2"/>`);
    const skip = (from, to) => (s += `<path d="M${cx} ${from} H${cx - bw / 2 - 40} V${to} H${cx - 14}" fill="none" stroke="${C.amber}" stroke-width="2.5"/>`);
    s += `<text x="${cx}" y="${h - 6}" text-anchor="middle" class="vs">输入 x（自下而上）</text>`;
    // 自下而上两个子层
    [["注意力", 232], ["前馈 FFN", 36]].forEach(([name, top]) => {
      const base = top + 190;
      v(base + 4, base - 6);
      if (kind === "pre") { v(base - 6, base - 20); box(base - 58, "LayerNorm", "n"); v(base - 20, base - 20); v(base - 58, base - 72); box(base - 110, name, "a"); v(base - 110, base - 135); plus(base - 148); skip(base - 6, base - 148); v(base - 161, base - 186); }
      else { v(base - 6, base - 20); box(base - 58, name, "a"); v(base - 58, base - 72); plus(base - 84); skip(base - 6, base - 84); v(base - 97, base - 110); box(base - 148, "LayerNorm", "n"); v(base - 148, base - 186); }
    });
    s += `<text x="8" y="${h - 26}" class="vs" style="fill:${C.amber}">黄线 = 残差直通路</text>`;
    return { svg: s, w, h };
  }

  // n×m 的“谁能看谁”矩阵。allow(i,j) 返回是否可见
  function mask(rows, cols, allow, o = {}) {
    const c = o.cell || 30, G = o.gut || 62, T = o.title ? 62 : 34, w = G + cols.length * c + 6, h = T + rows.length * c + 6;
    let s = o.title ? `<text x="${G}" y="16" class="vt">${esc(o.title)}</text>` : "";
    cols.forEach((t, j) => (s += `<text x="${G + j * c + c / 2}" y="${T - 10}" text-anchor="middle" class="vs">${esc(t)}</text>`));
    rows.forEach((t, i) => {
      s += `<text x="${G - 8}" y="${T + i * c + c / 2 + 4}" text-anchor="end" class="vs">${esc(t)}</text>`;
      cols.forEach((_, j) => { const ok = allow(i, j); s += `<rect x="${G + j * c + 1}" y="${T + i * c + 1}" width="${c - 3}" height="${c - 3}" rx="3" fill="${ok ? o.color || C.blue : "#111"}" stroke="${ok ? "none" : C.dim}"/>`; });
    });
    return { svg: s, w, h };
  }
  const api = { lines, block, mask, wrap, C };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.TFV = api;
})(typeof window !== "undefined" ? window : globalThis);
