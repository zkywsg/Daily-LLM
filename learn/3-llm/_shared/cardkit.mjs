// 图卡通用工具：深色、单一蓝色强调、1080×1920。Node 运行，用本机 Edge/Chrome 截图。
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createRequire } from "node:module";
globalThis.window = globalThis.window || {};
createRequire(import.meta.url)("./viz.js");
export const VIZ = globalThis.window.VIZ;
// 把 viz.js 产出的 SVG 片段包成卡片用的 SVG；scale 控制最终宽度
export const vizSvg = (inner, w, h, width = 900) => `<svg class="viz" viewBox="0 0 ${w} ${h}" width="${width}" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC,sans-serif">${inner}</svg>`;

export const browser = ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].find(existsSync);
if (!browser) throw new Error("找不到 Edge / Chrome");
export const pct = (w) => Math.round(w * 100);

// 安全区：上 170 / 左右 90 / 下 400（经验值，待在 TikTok 实机核对）
export const css = `
:root{--bg:#000000;--ink:#ededed;--mute:#8a8a8a;--line:#242424;--panel:#0a0a0a;--accent:#3b82f6;--dim:#2a2a2a}
*{box-sizing:border-box;margin:0}
html,body{width:1080px;height:1920px;background:var(--bg);color:var(--ink);font-family:"PingFang SC","Noto Sans SC",system-ui,sans-serif}
.card{width:1080px;height:1920px;padding:170px 90px 400px;display:flex;flex-direction:column}
.main{flex:1;display:flex;flex-direction:column;justify-content:center;margin-top:-40px}
.tag{font-size:34px;color:var(--accent);letter-spacing:3px;font-weight:700}
h1{font-size:96px;line-height:1.2;margin-top:34px;font-weight:700}
h2{font-size:76px;line-height:1.25;margin-top:28px;font-weight:700}
p{font-size:50px;line-height:1.55;margin-top:34px}
.mute{color:var(--mute)} .a{color:var(--accent)} .s{font-size:38px}
.small{font-size:30px;color:var(--mute);margin-top:24px}
.panel{background:var(--panel);border:2px solid var(--line);border-radius:12px;padding:36px 40px;margin-top:40px}
.row{display:grid;grid-template-columns:170px 1fr 120px;gap:20px;align-items:center;margin:16px 0;font-size:44px}
.bar{height:44px;background:var(--bg);border:2px solid var(--line);border-radius:10px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
.row.me .bar i{background:var(--dim)}.num{text-align:right;color:var(--mute);font-size:40px;font-variant-numeric:tabular-nums}
.big{font-size:60px;font-weight:700;line-height:1.5}
svg.viz text{fill:#ededed;font:20px ui-monospace,Menlo,monospace}svg.viz .vs{fill:#8a8a8a;font:20px "PingFang SC",sans-serif}svg.viz .vt{font:700 26px "PingFang SC",sans-serif}svg.viz .vn{fill:#fff;font:19px ui-monospace,Menlo,monospace}
.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}
`;

// 点网格 SVG（点的面积 = 权重）。opts: sel 高亮行, mask 画因果掩码, cell 格宽, numbers
export function dotGrid(tokens, A, o = {}) {
  const n = tokens.length, cell = o.cell || 110, L = o.left ?? 150, T = 120, W = L + n * cell + 10, H = T + n * cell + 10, fs = o.font || 34;
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC,sans-serif">`;
  tokens.forEach((t, j) => (s += `<text x="${L + j * cell + cell / 2}" y="${T - 22}" text-anchor="middle" font-size="${fs}" fill="#8a8a8a">${t.text}</text>`));
  tokens.forEach((t, i) => {
    const y = T + i * cell;
    if (i === o.sel) s += `<rect x="0" y="${y}" width="${W}" height="${cell}" rx="14" fill="#0f1c33"/>`;
    s += `<text x="${L - 14}" y="${y + cell / 2 + fs / 3}" text-anchor="end" font-size="${fs}" font-weight="${i === o.sel ? 800 : 500}" fill="${i === o.sel ? "#3b82f6" : "#ededed"}">${t.text}</text>`;
    tokens.forEach((_, j) => {
      const cx = L + j * cell + cell / 2, cy = y + cell / 2;
      if (o.mask && j > i) { s += `<rect x="${cx - cell / 2 + 10}" y="${cy - cell / 2 + 10}" width="${cell - 20}" height="${cell - 20}" rx="10" fill="#2a2a2a" opacity=".45"/>`; return; }
      s += `<circle cx="${cx}" cy="${cy}" r="${(Math.sqrt(A[i][j]) * cell * 0.47).toFixed(1)}" fill="#3b82f6"/>`;
      if (o.numbers) s += `<text x="${cx}" y="${cy + 9}" text-anchor="middle" font-size="26" font-weight="700" fill="#000000">${pct(A[i][j])}</text>`;
    });
  });
  return s + "</svg>";
}

export function bars(tokens, vals, fmt, meIdx) {
  const m = Math.max(...vals);
  return vals.map((v, j) => `<div class="row ${j === meIdx ? "me" : ""}"><span>${tokens[j].text}</span><div class="bar"><i style="width:${Math.max(2, (v / m) * 100).toFixed(1)}%"></i></div><span class="num">${fmt(v)}</span></div>`).join("");
}

// root: 知识点目录；label: 页脚名；cards: 每张的 body HTML 数组
export function build(root, label, cards) {
  const src = join(root, "cards/src"), out = join(root, "cards/out");
  mkdirSync(src, { recursive: true }); mkdirSync(out, { recursive: true });
  cards.forEach((body, i) => {
    const n = String(i + 1).padStart(2, "0"), f = join(src, `${n}.html`);
    writeFileSync(f, `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>${css}</style></head><body><div class="card"><div class="main">${body}</div><div class="small">${i + 1} / ${cards.length} · ${label}</div></div></body></html>`);
    execFileSync(browser, ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1080,1920", "--virtual-time-budget=2000", `--screenshot=${join(out, `${n}.png`)}`, `file://${f}`], { stdio: "ignore" });
  });
  console.log(label, "done", cards.length);
}
