// 01 注意力基础 · 图卡。用法：node cards/build.mjs（数字全部来自 _shared/engine.js）
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { writeFileSync } from "node:fs";
import { build, dotGrid, bars, pct, VIZ, vizSvg } from "../../_shared/cardkit.mjs";
const require = createRequire(import.meta.url);
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const E = require("../../_shared/engine.js");

const toks = E.tokenize("小猫没有过马路，因为它太累了"), IT = toks.findIndex((t) => t.text === "它");
const X = E.embed(toks), H = E.HEADS[0], R = E.head(X, H);
const top = R.A[IT].map((w, j) => [w, j]).sort((a, b) => b[0] - a[0]);
const exp = E.scaleExperiment(64, 8, 2000, 7);
const f2 = (v) => v.map((x) => x.toFixed(2)).join(", ");

function arcs() {
  const xs = toks.map((_, i) => 60 + i * 130), y0 = 300, from = xs[IT]; let s = `<svg width="900" height="380" viewBox="0 0 900 380" xmlns="http://www.w3.org/2000/svg">`;
  R.A[IT].forEach((w, j) => { if (j === IT) return; const x = xs[j], h = 40 + Math.abs(x - from) * 0.32;
    s += `<path d="M ${from} ${y0} Q ${(x + from) / 2} ${y0 - h * 2} ${x} ${y0}" fill="none" stroke="${w > 0.15 ? "#3b82f6" : "#2a2a2a"}" stroke-width="${(2 + w * 46).toFixed(1)}" stroke-linecap="round"/>`; });
  toks.forEach((t, i) => (s += `<text x="${xs[i]}" y="${y0 + 56}" text-anchor="middle" font-size="${i === IT ? 46 : 40}" font-weight="${i === IT ? 800 : 500}" fill="${i === IT ? "#3b82f6" : "#ededed"}" font-family="PingFang SC,sans-serif">${t.text}</text>`));
  return s + "</svg>";
}
const lab = toks.map((t) => t.text), c0 = 52;
const hS = VIZ.heat(R.S, { cell: c0, x: 0, y: 80, gut: 78, title: "分数 S", sub: "q·k/√d", rows: lab, cols: lab.map((t) => t.slice(0, 1)), fmt: (v) => v.toFixed(1), max: 2 });
const hA = VIZ.heat(R.A, { cell: c0, x: hS.w + 60, y: 80, gut: 78, title: "权重 A", sub: "每行和 = 100%", cols: lab.map((t) => t.slice(0, 1)), fmt: (v) => String(pct(v)), max: 1 });
const heatCard = `<div class="tag">看矩阵</div><h2>softmax 前后，<br>同一张表</h2><div style="margin-top:34px">${vizSvg(hS.svg + `<text x="${hS.w + 18}" y="${80 + 16 + 3.5 * c0}" class="vs" style="font-size:26px">→</text>` + hA.svg, hS.w + 60 + hA.w + 10, 80 + 16 + 7 * c0 + 10)}</div><p class="mute s" style="margin-top:20px">左：原始分数（可以是任意数）。右：每行变成和为 100% 的权重。</p>`;
const cards = [
 `<div class="tag">大模型怎么“读”</div><h1>它每写一个字，<br>都要和前面所有字<br><span class="a">打一遍分</span></h1><p class="mute">这套打分机制，叫注意力。<br>8 张图讲清楚 →</p>`,
 `<div class="tag">问题</div><h2>“小猫没有过马路，因为<span class="a">它</span>太累了”</h2><p>模型读到“它”，手里只有一个孤立的词，没有“小猫”的信息。</p><div style="margin-top:30px">${arcs()}</div><p class="mute s" style="margin-top:10px">线越粗，“它”越要从那个词取信息。（示意值，非真实模型）</p>`,
 `<div class="tag">第 1 步 · 变成向量</div><h2>每个词先变成一串数</h2><p>这里用 6 个数描述一个词。真实模型是几百到上万个，由训练得到。</p><div class="panel"><div class="big">“它” → x =<br><span class="a">[${f2(X[IT])}]</span></div><p class="mute s" style="margin-top:6px">依次是：有生命 / 物体 / 动作 / 状态 / 指代 / 连接</p></div>`,
 `<div class="tag">第 2 步 · 三个角色</div><h2>同一个词，变出 q、k、v</h2><p>x 分别乘三个矩阵：</p><div class="panel"><div class="big"><span class="a">q</span> 我想找什么<br><span class="a">k</span> 我能被怎样找到<br><span class="a">v</span> 被选中时我提供什么</div></div><p class="mute s">“它”的 q = [${f2(R.Q[IT])}]，带着“去找有生命的词”的倾向。</p>`,
 `<div class="tag">第 3 步 · 两两打分</div><h2>一张图看完全部：<br>点越大，权重越大</h2><div style="margin-top:20px">${dotGrid(toks, R.A, { sel: IT, cell: 112, left: 150, font: 32 })}</div><p class="mute s" style="margin-top:6px">每一行是一个提问的词。高亮行是“它”。</p>`,
 `<div class="tag">第 3 步 · 打分到权重</div><h2>q·k → softmax</h2><p>“它”和每个词的分数，变成加起来等于 100% 的权重：</p><div class="panel">${bars(toks, R.A[IT], (v) => pct(v) + "%", IT)}</div><p class="mute s" style="margin-top:20px">最关注“${toks[top[0][1]].text}”${pct(top[0][0])}%，其次“${toks[top[1][1]].text}”${pct(top[1][0])}%。</p>`,
 `<div class="tag">第 4 步 · 取信息</div><h2>加权求和</h2><p>用权重把各词的 v 混合，得到“它”的新表示：</p><div class="panel"><div class="big">o = <span class="a">${pct(top[0][0])}%</span>×${toks[top[0][1]].text}的v<br>+ <span class="a">${pct(top[1][0])}%</span>×${toks[top[1][1]].text}的v<br>+ ……其余词</div></div><p>现在的“它”，带着“小猫”的信息了。</p>`,
 `<div class="tag">一行公式 · 还有代价</div><h2>softmax(QKᵀ / <span class="a">√d</span>)·V</h2><div class="panel"><p class="mute s" style="margin:0">随机向量实验，d = 64：不除 √d 时，最大权重 ${pct(exp.unscaled)}%；除以 √d 后 ${pct(exp.scaled)}%。</p><div class="big" style="margin-top:14px">长度 ×10，分数表 <span class="a">×100</span></div><p class="mute s" style="margin:6px 0 0">1,000 词 → 100 万个分数；10,000 词 → 1 亿个（单层单头）</p></div>`,
];
cards.splice(5, 0, heatCard);
build(root, "注意力基础", cards);

// README 示意图
const rows = toks.map((t, j) => { const y = 56 + j * 36, w = Math.round(R.A[IT][j] * 520);
  return `<text x="70" y="${y + 15}" text-anchor="end" font-size="15" fill="#2a2a2a">${t.text}</text><rect x="82" y="${y}" width="${w}" height="22" rx="3" fill="${j === IT ? "#cbd5e1" : "#2563eb"}"/><text x="${90 + w}" y="${y + 16}" font-size="13" fill="#64748b">${pct(R.A[IT][j])}%</text>`; }).join("\n  ");
writeFileSync(join(root, "assets/attention-weights.svg"), `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 320" font-family="system-ui,-apple-system,sans-serif"><rect width="720" height="320" fill="#ffffff"/><text x="360" y="26" text-anchor="middle" font-size="16" font-weight="600" fill="#0f172a">“它”对句中各词的注意力权重（示意）</text>
  ${rows}
</svg>
`);
