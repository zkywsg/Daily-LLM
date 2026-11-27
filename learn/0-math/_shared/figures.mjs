// 生成线性代数四篇的配图（SVG）。用法：node learn/0-math/_shared/figures.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url)), learn = join(here, "../..");
globalThis.window = globalThis.window || {};
for (const f of ["0-math/_shared/la-engine.js", "0-math/_shared/la-viz.js", "3-llm/_shared/tf-viz.js", "3-llm/_shared/minigpt.js", "3-llm/tf-04-mini-gpt/demo/model.js"]) (0, eval)(readFileSync(join(learn, f), "utf8"));
const { LA, LAV, TFV, MiniGPT, MODEL } = window, C = LAV.C;

const css = `<style>text{fill:#ededed;font:13px "PingFang SC",sans-serif}.vs{fill:#8a8a8a;font-size:12px}.vt{font-weight:700;font-size:14px}</style>`;
// rows: 二维数组，每个元素是 {svg,w,h}
function file(path, rows, gap = 28) {
  let y = 14, W = 0, s = "";
  for (const parts of rows) {
    let x = 14, h = 0;
    for (const p of parts) { s += `<g transform="translate(${x},${y})">${p.svg}</g>`; x += p.w + gap; h = Math.max(h, p.h); }
    W = Math.max(W, x - gap + 14); y += h + 18;
  }
  const H = y - 4;
  writeFileSync(join(learn, "0-math", path), `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${css}<rect width="${W}" height="${H}" fill="#000"/>${s}</svg>\n`);
}
const note = (lines, w = 300) => ({ svg: lines.map((t, i) => `<text x="0" y="${16 + i * 22}" class="vs">${t}</text>`).join(""), w, h: lines.length * 22 + 6 });

/* 01 向量 */
const a = [3, 4], b = [4, 3], p = LA.project(a, b);
file("la-01-vectors/assets/vectors.svg", [[
  LAV.plane({ title: "两个向量与夹角", size: 300, range: 5.5, vectors: [{ v: a, color: C.blue, label: "a" }, { v: b, color: C.amber, label: "b", dy: 14 }], segments: [{ a, b, color: C.gray, dash: "4 4" }] }),
  LAV.plane({ title: "a 在 b 上的投影", size: 300, range: 5.5, vectors: [{ v: a, color: C.blue, label: "a" }, { v: b, color: C.amber, label: "b", dy: 14 }, { v: p, color: C.green, label: "投影", dx: 4, dy: 22, w: 4 }], segments: [{ a, b: p, color: C.gray, dash: "4 4" }] }),
]]);
const ball = (q, title) => LAV.plane({ title, size: 220, range: 1.5, paths: [{ color: C.blue, fill: "rgba(59,130,246,.18)", pts: Array.from({ length: 201 }, (_, i) => { const t = (i * Math.PI) / 100, v = [Math.cos(t), Math.sin(t)], n = LA.norm(v, q); return [v[0] / n, v[1] / n]; }) }] });
file("la-01-vectors/assets/balls.svg", [[ball(1, "L1：菱形"), ball(2, "L2：圆"), ball(Infinity, "L∞：正方形")]], 20);
file("la-01-vectors/assets/highdim.svg", [[2, 10, 100, 1000].map((d) => {
  const r = LA.cosSample(d, 4000);
  return TFV.lines([{ name: "比例", color: C.blue, pts: LA.histogram(r.values) }], { w: 250, h: 190, title: `${d} 维 · 标准差 ${r.std.toFixed(2)}`, ymin: 0, noLegend: true, xfmt: (v) => v.toFixed(1), yfmt: (v) => Math.round(v * 100) + "%" });
})], 12);
const ch = [..."猫狗鸟兔熊住喜最"];
file("la-01-vectors/assets/embed.svg", [[LAV.heat(LA.embeddingSim(MODEL, ch), { rows: ch, cols: ch, cell: 44, title: "迷你 GPT 词向量的余弦相似度" })]]);

/* 02 矩阵 */
const tf = (M, title) => LAV.plane({ title, size: 230, range: 3, M, square: true, vectors: [{ v: [M[0][0], M[1][0]], color: C.blue, label: "第1列", dx: -16, dy: 20 }, { v: [M[0][1], M[1][1]], color: C.green, label: "第2列", dx: -44 }] });
file("la-02-matrix/assets/transform.svg", [
  [tf([[1, 0], [0, 1]], "单位矩阵"), tf([[2, 1], [0, 1]], "[[2,1],[0,1]] 面积×2"), tf(LA.rot(Math.PI / 4), "旋转 45°")],
  [tf([[1.5, 0], [0, 0.5]], "缩放"), tf([[0, 1], [1, 0]], "翻转 行列式 −1"), tf([[1, 2], [0.5, 1]], "压扁到直线 行列式 0")],
], 18);
const R = [[0, -1], [1, 0]], S = [[2, 0], [0, 1]], e = [1, 0];
const step = (title, M, v, label) => LAV.plane({ title, size: 220, range: 2.6, M, square: true, vectors: [{ v, color: C.amber, label }] });
file("la-02-matrix/assets/order.svg", [
  [step("起点 e", [[1, 0], [0, 1]], e, "e"), step("先旋转 R", R, LA.mv(R, e), "Re"), step("再拉伸 S → (0,1)", LA.mm(S, R), LA.mv(S, LA.mv(R, e)), "SRe")],
  [step("起点 e", [[1, 0], [0, 1]], e, "e"), step("先拉伸 S", S, LA.mv(S, e), "Se"), step("再旋转 R → (0,2)", LA.mm(R, S), LA.mv(R, LA.mv(S, e)), "RSe")],
], 18);

/* 03 特征值 */
const A = [[2, 1], [1, 2]];
file("la-03-eigen/assets/eigen.svg", [[
  LAV.plane({ title: "变换前：单位圆", size: 300, range: 3.6, circle: true, vectors: [{ v: [0.707, 0.707], color: C.amber, label: "v₁" }, { v: [0.707, -0.707], color: C.green, label: "v₂", dy: 14 }, { v: [1, 0], color: C.gray, label: "u" }] }),
  LAV.plane({ title: "乘上 A 之后", size: 300, range: 3.6, M: A, circle: true, vectors: [{ v: LA.mv(A, [0.707, 0.707]), color: C.amber, label: "3·v₁" }, { v: LA.mv(A, [0.707, -0.707]), color: C.green, label: "1·v₂", dy: 14 }, { v: LA.mv(A, [1, 0]), color: C.gray, label: "Au", dy: 14 }] }),
]]);
const it = LA.powerIter(A, [1, 0], 4);
file("la-03-eigen/assets/power.svg", [[LAV.plane({ title: "反复乘 A：方向收向 v₁", size: 340, range: 1.25, circle: true, segments: [{ a: [-1.2, -1.2], b: [1.2, 1.2], color: C.amber, dash: "5 5" }], vectors: it.map((s, k) => ({ v: s.v, color: `rgba(59,130,246,${0.3 + 0.175 * k})`, label: k < 3 ? "k=" + k : "", dy: k === 0 ? 14 : -4 })) })]]);
const gr = LA.growth([[0.9, 5], [0, 0.9]], [0, 1], 60);
file("la-03-eigen/assets/growth.svg", [[TFV.lines([{ name: "向量长度 |Aᵏx|", color: C.blue, pts: gr.map((v, k) => [k, v]) }, { name: "只看特征值 0.9ᵏ", color: C.gray, dash: "5 4", pts: gr.map((_, k) => [k, 0.9 ** k]) }], { w: 620, h: 300, title: "特征值都是 0.9，长度却先涨到 19 倍", xlabel: "连乘次数 k", charW: 12 })]]);

/* 04 SVD */
const B = [[3, 0], [4, 5]], f = LA.svd(B);
// V 的两列为 f.V[0]、f.V[1]。为了画图，让 V 是纯旋转（行列式 +1）
let V = [[f.V[0][0], f.V[1][0]], [f.V[0][1], f.V[1][1]]], U = [[f.U[0][0], f.U[1][0]], [f.U[0][1], f.U[1][1]]];
if (LA.det2(V) < 0) { V = V.map((r) => [r[0], -r[1]]); U = U.map((r) => [r[0], -r[1]]); }
const Vt = LA.T(V), Sg = [[f.s[0], 0], [0, f.s[1]]];
const st = (title, M) => LAV.plane({ title, size: 220, range: 7, M, circle: true, vectors: [{ v: LA.mv(M, [V[0][0], V[1][0]]), color: C.amber, label: "" }, { v: LA.mv(M, [V[0][1], V[1][1]]), color: C.green, label: "" }] });
file("la-04-svd/assets/steps.svg", [[st("① 单位圆和 v₁、v₂", [[1, 0], [0, 1]]), st("② 乘 Vᵀ：转", Vt), st("③ 乘 Σ：拉", LA.mm(Sg, Vt)), st("④ 乘 U：再转 = A", LA.mm(U, LA.mm(Sg, Vt)))]], 14);
const img = LA.picture("demo"), fi = LA.svd(img);
file("la-04-svd/assets/ranks.svg", [[1, 4, 8].map((k) => LAV.image(LA.rankK(fi, k), { title: `秩 ${k} · 误差 ${(LA.relErr(fi.s, k) * 100).toFixed(0)}%`, cell: 3.2 })), [16, 32].map((k) => LAV.image(LA.rankK(fi, k), { title: `秩 ${k} · 误差 ${(LA.relErr(fi.s, k) * 100).toFixed(0)}%`, cell: 3.2 })).concat(LAV.image(img, { title: "原图", cell: 3.2 }))], 22);
const parts = LA.gptSVD(MODEL), ks = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32];
const curve = ks.map((k) => {
  const ok = MiniGPT.quiz(MiniGPT.load(LA.truncateGPT(MODEL, parts, k))).filter((q) => q.ok).length;
  const kept = parts.reduce((a, q) => { const e2 = q.f.s.map((v) => v * v), t = e2.reduce((x, y) => x + y, 0); return a + e2.slice(0, k).reduce((x, y) => x + y, 0) / t; }, 0) / parts.length;
  return { k, ok: ok / 20, kept };
});
file("la-04-svd/assets/gpt.svg", [[TFV.lines([{ name: "保留的能量（12 个矩阵平均）", color: C.gray, dash: "5 4", pts: curve.map((c) => [c.k, c.kept]) }, { name: "知识题正确率", color: C.blue, pts: curve.map((c) => [c.k, c.ok]) }], { w: 620, h: 320, title: "能量保留得多 ≠ 能力保留得住", xlabel: "每个权重矩阵保留的秩 k", ymin: 0, ymax: 1, yfmt: (v) => Math.round(v * 100) + "%", charW: 13 })]]);
console.log("gpt curve", curve.map((c) => `k=${c.k}: 题 ${c.ok * 20} 能量 ${(c.kept * 100).toFixed(1)}%`).join(" | "));
