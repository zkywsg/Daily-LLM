// 页面通用小工具：点网格（点的大小 = 注意力权重）、向量热力表
(function (root) {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  // A: n×n 权重；sel: 选中的行；mask: 是否画因果掩码；cell: 单元格边长
  function dotGrid(tokens, A, opts = {}) {
    const n = tokens.length, cell = opts.cell || 44, L = 70, T = 52;
    const W = L + n * cell + 10, H = T + n * cell + 10;
    let s = `<svg class="grid" viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px" role="img" aria-label="注意力点网格">`;
    s += `<text class="mute" x="${L}" y="14" font-size="12">被看的词（键 k）→</text>`;
    tokens.forEach((t, j) => (s += `<text x="${L + j * cell + cell / 2}" y="${T - 10}" text-anchor="middle">${esc(t.text)}</text>`));
    tokens.forEach((t, i) => {
      const y = T + i * cell;
      if (i === opts.sel) s += `<rect class="selrow" x="0" y="${y}" width="${W}" height="${cell}" rx="6"/>`;
      s += `<text x="${L - 8}" y="${y + cell / 2 + 4}" text-anchor="end" ${i === opts.sel ? 'style="font-weight:700"' : ""}>${esc(t.text)}</text>`;
      tokens.forEach((_, j) => {
        const cx = L + j * cell + cell / 2, cy = y + cell / 2, w = A[i][j];
        if (opts.mask && j > i) { s += `<rect class="masked" x="${cx - cell / 2 + 4}" y="${cy - cell / 2 + 4}" width="${cell - 8}" height="${cell - 8}" rx="4"/>`; return; }
        s += `<circle cx="${cx}" cy="${cy}" r="${(Math.sqrt(w) * cell * 0.46).toFixed(1)}"/>`;
        if (opts.numbers) s += `<text class="mute" x="${cx}" y="${cy + 4}" text-anchor="middle" style="font-size:11px;fill:var(--ink)">${(w * 100).toFixed(0)}</text>`;
      });
      s += `<rect class="hit" data-row="${i}" x="0" y="${y}" width="${W}" height="${cell}"/>`;
    });
    return s + "</svg>";
  }
  function table(head, rows, fmt) {
    fmt = fmt || ((v) => (typeof v === "number" ? v.toFixed(2) : esc(v)));
    return `<div class="scroll"><table class="mat"><tr><th></th>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr>` +
      rows.map((r) => `<tr><th>${esc(r.label)}</th>${r.vals.map((v) => `<td class="h" style="background:rgba(37,99,235,${Math.min(0.55, Math.abs(v) * 0.5).toFixed(2)})">${fmt(v)}</td>`).join("")}</tr>`).join("") + "</table></div>";
  }
  root.UI = { dotGrid, table, esc };
})(window);
