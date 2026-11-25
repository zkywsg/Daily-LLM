// 02 手算 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { build, VIZ, vizSvg } from "../../_shared/cardkit.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const mm = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const T = (A) => A[0].map((_, j) => A.map((r) => r[j]));
const X = [[1, 0], [0, 1], [1, 1]], WQ = [[1, 0], [1, 1]], WK = [[1, 1], [0, 1]], WV = [[1, 0], [0, 2]];
const Q = mm(X, WQ), K = mm(X, WK), V = mm(X, WV), S = mm(Q, T(K)), Sp = S.map((r) => r.map((v) => v / Math.SQRT2));
const E = Sp.map((r) => r.map(Math.exp)), rs = E.map((r) => r.reduce((a, b) => a + b, 0)), A = E.map((r, i) => r.map((v) => v / rs[i])), O = mm(A, V);
const f = (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2));
const mat = (M, col = "#ededed") => `<table style="border-collapse:collapse;display:inline-table;margin:0 18px;vertical-align:middle;border-left:4px solid #8a8a8a;border-right:4px solid #8a8a8a;border-radius:10px">${M.map((r) => `<tr>${r.map((v) => `<td style="padding:6px 14px;font:600 44px ui-monospace,Menlo,monospace;color:${col};text-align:right">${f(v)}</td>`).join("")}</tr>`).join("")}</table>`;
const T3 = ["猫", "追", "鱼"], cc = 120;
const hA = VIZ.heat(A, { cell: cc, x: 0, y: 80, gut: 78, title: "A（权重）", sub: "每行和 = 1", rows: T3, cols: T3, fmt: (v) => v.toFixed(2), max: 1 });
const hO = VIZ.heat(O, { cell: cc, x: hA.w + 80, y: 80, gut: 78, title: "O = A·V", sub: "新表示", cols: ["a", "b"], fmt: (v) => v.toFixed(2) });
const heatCard = `<div class="tag">看全貌</div><h2>权重 A 越深，<br>拿的越多</h2><div style="margin-top:40px">${vizSvg(hA.svg + `<text x="${hA.w + 28}" y="${80 + 16 + 1.5 * cc}" class="vs" style="font-size:34px">→</text>` + hO.svg, hA.w + 80 + hO.w + 10, 80 + 16 + 3 * cc + 10)}</div><p class="mute s" style="margin-top:20px">颜色越深，数越大。“鱼”这一行：${Math.round(A[2][2] * 100)}% 的权重给了它自己，所以 O 里“鱼”的 v 占大头。</p>`;
const cards = [
 `<div class="tag">手算注意力</div><h1>能不能用纸笔，<br>把注意力<span class="a">算出来</span>？</h1><p class="mute">3 个词、2 维向量、8 步。<br>算过一遍，公式就不再是符号 →</p>`,
 `<div class="tag">题目</div><h2>猫 追 鱼</h2><div class="panel" style="text-align:center"><p class="s mute" style="margin:0 0 6px">每个词一个向量（每行一个词）</p>${mat(X)}</div><div class="panel" style="text-align:center"><p class="s mute" style="margin:0 0 6px">三个投影矩阵 W_Q · W_K · W_V</p>${mat(WQ)}${mat(WK)}${mat(WV)}</div>`,
 `<div class="tag">步骤 1–3</div><h2>X 乘三个矩阵</h2><div class="panel" style="text-align:center">Q ${mat(Q, "#3b82f6")}<br>K ${mat(K, "#3b82f6")}<br>V ${mat(V, "#3b82f6")}</div><p class="mute s">例：“鱼”的 q = [1,1]·[[1,0],[1,1]] = [2, 1]</p>`,
 `<div class="tag">步骤 4–6</div><h2>打分 → 缩放 → 取指数</h2><div class="panel" style="text-align:center"><span class="s mute">S = QKᵀ</span> ${mat(S)}</div><div class="panel" style="text-align:center"><span class="s mute">e^(S/√2)</span> ${mat(E)}</div>`,
 `<div class="tag">步骤 7–8</div><h2>行归一化，再取信息</h2><div class="panel" style="text-align:center"><span class="s mute">A 每行和为 1</span> ${mat(A, "#3b82f6")}</div><p class="s">“追”这行：${f(A[1][2])} 给了“鱼”。<br>所以 O_追 = ${A[1].map((w, k) => `${f(w)}·v${["猫", "追", "鱼"][k]}`).join(" + ")} = [${f(O[1][0])}, ${f(O[1][1])}]</p>`,
];
cards.push(heatCard);
build(root, "手算注意力", cards);
