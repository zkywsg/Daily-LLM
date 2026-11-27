// 线代 04 SVD · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { LA, LAV, MODEL, MiniGPT, TFV, build, fig, table, panel, grid } from "../../_shared/lacards.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), ".."), C = LAV.C;
const B = [[3, 0], [4, 5]], f = LA.svd(B);
let V = [[f.V[0][0], f.V[1][0]], [f.V[0][1], f.V[1][1]]], U = [[f.U[0][0], f.U[1][0]], [f.U[0][1], f.U[1][1]]];
if (LA.det2(V) < 0) { V = V.map((r) => [r[0], -r[1]]); U = U.map((r) => [r[0], -r[1]]); }
const Vt = LA.T(V), Sg = [[f.s[0], 0], [0, f.s[1]]];
const st = (title, M) => LAV.plane({ title, size: 220, range: 7, M, circle: true, vectors: [{ v: LA.mv(M, [V[0][0], V[1][0]]), color: C.amber }, { v: LA.mv(M, [V[0][1], V[1][1]]), color: C.green }] });
const steps = grid([st("① 单位圆", [[1, 0], [0, 1]]), st("② 乘 Vᵀ：转", Vt), st("③ 乘 Σ：拉", LA.mm(Sg, Vt)), st("④ 乘 U：再转", LA.mm(U, LA.mm(Sg, Vt)))], 2, 20);
const img = LA.picture("demo"), fi = LA.svd(img), pc = (k) => (LA.relErr(fi.s, k) * 100).toFixed(0) + "%";
const pics = grid([1, 4, 16].map((k) => LAV.image(LA.rankK(fi, k), { title: `秩 ${k} · 误差 ${pc(k)}`, cell: 3 })).concat(LAV.image(img, { title: "原图", cell: 3 })), 2, 24);
const parts = LA.gptSVD(MODEL), ks = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32];
const curve = ks.map((k) => ({ k, ok: MiniGPT.quiz(MiniGPT.load(LA.truncateGPT(MODEL, parts, k))).filter((q) => q.ok).length, kept: parts.reduce((a, p) => { const e = p.f.s.map((v) => v * v), t = e.reduce((x, y) => x + y, 0); return a + e.slice(0, k).reduce((x, y) => x + y, 0) / t; }, 0) / parts.length }));
const at = (k) => curve.find((c) => c.k === k);
const chart = TFV.lines([{ name: "保留能量", color: C.gray, dash: "5 4", pts: curve.map((c) => [c.k, c.kept]) }, { name: "答题正确率", color: C.blue, pts: curve.map((c) => [c.k, c.ok / 20]) }], { w: 440, h: 360, sw: 3.5, ymin: 0, ymax: 1, charW: 14, xlabel: "保留的秩（满秩 32）", yfmt: (v) => Math.round(v * 100) + "%", xfmt: (v) => Math.round(v) });
const store = (k) => (k * (64 * 2 + 1) / 4096 * 100).toFixed(0) + "%";
const cards = [
  `<div class="tag">线性代数 · 第 4 块</div><h1>一张图扔掉多少，<br>你<span class="a">还认得出</span>？</h1><p class="mute">奇异值分解 →</p>`,
  `<div class="tag">SVD</div><h2>任何矩阵 =<br><span class="a">转、拉、再转</span></h2>${fig(steps, 760)}<p class="s mute">A = U Σ Vᵀ。Σ 里的拉伸倍数叫奇异值。</p>`,
  `<div class="tag">手算</div><h2>只留最大的一项</h2>${panel(`A = [[3,0],[4,5]]<br>σ = √45 和 √5<br><br>σ₁u₁v₁ᵀ =<br><span class="a">[[1.5,1.5],[4.5,4.5]]</span><br><br>误差 = √5 = σ₂`, 48)}<p class="s mute">扔掉的奇异值，就是误差。</p>`,
  `<div class="tag">压缩图片</div><h2>从糊到清</h2>${fig(pics, 760)}<p class="s mute">64×64 合成图。保留的秩越高越清楚。</p>`,
  `<div class="tag">账要算清</div><h2>不是总能省</h2>${table(["秩", "要存的数", "误差"], [1, 8, 16, 32].map((k) => [String(k), store(k), pc(k)]), [3])}<p class="s">秩 32 存的比原图还多。</p><p class="s mute">只有秩远小于边长时才划算。</p>`,
  `<div class="tag">实验 · 压缩迷你 GPT</div><h2>留住 <span class="a">${Math.round(at(8).kept * 100)}%</span> 能量，<br>20 题只对 <span class="a">${at(8).ok}</span> 题</h2>${fig(chart, 880)}<p class="s mute">12 个权重矩阵全部截秩、不微调。秩 16 才全对。</p>`,
  `<div class="tag">LoRA</div><h2>低秩的是<br><span class="a">“改动”</span></h2>${panel("W′ = W + B A<br>W 冻结，只训练 B、A", 50)}${table(["d = 4096", "占完整矩阵"], [["r = 8", "0.39%"], ["r = 64", "3.12%"]])}<p class="s mute">上一张是压缩原模型，两回事。</p>`,
];
build(root, "SVD 与低秩", cards);
