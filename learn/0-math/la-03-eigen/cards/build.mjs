// 线代 03 特征值 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { LA, LAV, TFV, build, fig, table, panel } from "../../_shared/lacards.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), ".."), C = LAV.C;
const A = [[2, 1], [1, 2]], u = [0.707, 0.707], v = [0.707, -0.707];
const after = LAV.plane({ size: 320, range: 3.5, M: A, circle: true, vectors: [{ v: LA.mv(A, u), color: C.amber, label: "放大 3 倍", dx: -70, dy: -6 }, { v: LA.mv(A, v), color: C.green, label: "不变", dy: 18 }, { v: LA.mv(A, [1, 0]), color: C.ink, label: "方向变了", dx: -10, dy: 20 }] });
const it = LA.powerIter(A, [1, 0], 6);
const pw = LAV.plane({ size: 300, range: 1.5, circle: true, segments: [{ a: [-1.2, -1.2], b: [1.2, 1.2], color: C.amber, dash: "5 5" }], vectors: it.slice(0, 4).map((s, k) => ({ v: s.v, color: `rgba(59,130,246,${0.35 + 0.2 * k})`, label: k < 2 ? "第 " + k + " 步" : "", dy: k ? -4 : 16, dx: k ? 8 : -30 })) });
const P = [[0, 0, 1], [0.5, 0, 0], [0.5, 1, 0]];
let p = [1, 0, 0]; const walk = []; for (let k = 0; k <= 20; k++) { if ([0, 1, 5, 10, 20].includes(k)) walk.push([String(k), ...p.map((x) => x.toFixed(3))]); p = LA.mv(P, p); }
const gr = LA.growth([[0.9, 5], [0, 0.9]], [0, 1], 60), peak = Math.max(...gr);
const chart = TFV.lines([{ name: "实际长度", color: C.blue, pts: gr.map((y, k) => [k, y]) }, { name: "0.9 的 k 次方", color: C.gray, dash: "5 4", pts: gr.map((_, k) => [k, 0.9 ** k]) }], { w: 440, h: 360, sw: 3.5, ymin: 0, charW: 14, xlabel: "连乘次数" });
const cards = [
  `<div class="tag">线性代数 · 第 3 块</div><h1>有些方向，<br>怎么乘都<span class="a">不拐弯</span></h1><p class="mute">它们叫特征向量 →</p>`,
  `<div class="tag">几何直觉</div><h2>圆变椭圆</h2>${fig(after, 820)}<p class="s mute">Av = λv：只拉伸、不改方向。λ 是特征值。</p>`,
  `<div class="tag">手算</div><h2>A = [[2,1],[1,2]]</h2>${panel(`det(A − λI) = 0<br>(2−λ)² − 1 = 0<br>λ = <span class="a">3</span> 或 <span class="a">1</span><br><br>A[1,1] = [3,3] ✓<br>A[1,−1] = [1,−1] ✓`, 50)}`,
  `<div class="tag">幂迭代</div><h2>反复乘，<br>方向<span class="a">自己收拢</span></h2>${fig(pw, 700)}<p class="s">夹角：45° → ${LA.angle(it[1].v, [1, 1]).toFixed(1)}° → ${LA.angle(it[2].v, [1, 1]).toFixed(1)}° → … → ${LA.angle(it[6].v, [1, 1]).toFixed(2)}°</p><p class="s mute">每步大约缩到三分之一，就是 λ₂/λ₁。</p>`,
  `<div class="tag">网页排名</div><h2>随机点链接，<br>最后停在哪？</h2>${table(["步数", "A", "B", "C"], walk, [walk.length - 1])}<p class="s">A→B、C，B→C，C→A</p><p class="s mute">趋向 [0.4, 0.2, 0.4]：特征值 1 的特征向量。</p>`,
  `<div class="tag">反直觉</div><h2>特征值 0.9，<br>却先涨 <span class="a">${Math.round(peak)} 倍</span></h2>${fig(chart, 880)}<p class="s mute">A = [[0.9,5],[0,0.9]]。长期看特征值，单步看最大奇异值（${LA.svd([[0.9, 5], [0, 0.9]]).s[0].toFixed(2)}）。</p>`,
  `<div class="tag">和深度学习</div><h2>连乘 50 次</h2>${table(["缩放 g", "向量长度"], [["0.9", "0.041"], ["1.0", "8.0"], ["1.1", "942"]])}<p>差<span class="a">两万多倍</span>。</p><p class="s mute">64×64 随机矩阵、20 个种子的几何平均。这是初始化尺度要仔细选的原因之一。</p>`,
];
build(root, "特征值与特征向量", cards);
