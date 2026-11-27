// 线代 02 矩阵 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { LA, LAV, build, fig, table, panel, grid, bar } from "../../_shared/lacards.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), ".."), C = LAV.C;
const A = [[2, 1], [0, 1]], R = [[0, -1], [1, 0]], S = [[2, 0], [0, 1]], e = [1, 0];
const tf = (M, title, size = 300, range = 3, labels = true) => LAV.plane({ title, size, range, M, square: true, vectors: [{ v: [M[0][0], M[1][0]], color: C.blue, label: labels ? "第1列" : "", dx: -20, dy: 22 }, { v: [M[0][1], M[1][1]], color: C.green, label: labels ? "第2列" : "", dx: -40 }] });
const end = (M, v, title) => LAV.plane({ title, size: 220, range: 2.6, M, square: true, vectors: [{ v, color: C.amber, label: "[" + v.join(",") + "]", dx: 4 }] });
const q = [1.6, 0.4], k = [1.0, 1.2]; // 与讲解、互动页、transforms.py 相同
const rope = [[0, 3], [10, 13], [500, 503]].map(([m, n]) => [m, n, LA.dot(LA.mv(LA.rot(m * 0.3), q), LA.mv(LA.rot(n * 0.3), k)).toFixed(4)]);
const cards = [
  `<div class="tag">线性代数 · 第 2 块</div><h1>矩阵是一台<br><span class="a">搬网格</span>的机器</h1><p class="mute">看懂它只要盯住两列 →</p>`,
  `<div class="tag">几何直觉</div><h2>每一列 =<br>坐标轴<span class="a">搬去哪</span></h2>${fig(tf(A, "A = [[2,1],[0,1]]"), 800)}<p class="s mute">黄色是单位正方形变形后的样子。</p>`,
  `<div class="tag">手算</div><h2>按列加权</h2>${panel(`A x，x = [1, 2]<br><br>1 × <span class="a">[2, 0]</span><br>+ 2 × <span style="color:#22c55e">[1, 1]</span><br>= [${LA.mv(A, [1, 2]).join(", ")}]`, 54)}<p class="s mute">x 说的是沿两根轴各走几步，只是轴换了。</p>`,
  `<div class="tag">顺序</div><h2>先转后拉<br>≠ <span class="a">先拉后转</span></h2>${fig(grid([end(LA.mm(S, R), LA.mv(S, LA.mv(R, e)), "先转 R 后拉 S"), end(LA.mm(R, S), LA.mv(R, LA.mv(S, e)), "先拉 S 后转 R")], 2, 20), 900)}<p class="s mute">起点同样是 [1,0]。矩阵乘法一般不能交换。</p>`,
  `<div class="tag">行列式与秩</div><h2>面积变了<br>几倍？</h2>${table(["矩阵", "行列式", "含义"], [["[[2,1],[0,1]]", "2", "面积 ×2"], ["[[0,1],[1,0]]", "−1", "翻面"], ["[[1,2],[2,4]]", "0", "压成直线"]], [2])}${fig(tf([[1, 2], [0.5, 1]], "", 220, 2.6, false), 420)}<p class="s mute">压扁 = 秩变小 = 信息丢了，不可逆。</p>`,
  `<div class="tag">旋转 → RoPE</div><h2>一起转，<br>点积<span class="a">不变</span></h2>${table(["q 位置", "k 位置", "点积"], rope.map((r) => r.map(String)))}<p>只和<span class="a">位置差</span>有关。</p><p class="s mute">例子：q=[1.6,0.4]，k=[1.0,1.2]。</p><p class="s mute">这是旋转位置编码的核心。</p>`,
  `<div class="tag">代价</div><h2>换个括号，<br>快<span class="a">几十倍</span></h2><div class="panel">${bar("(AB)x", 100, "#8a8a8a", "114 ms")}${bar("A(Bx)", 2, "#3b82f6", "2 ms")}</div><p class="s">n = 3000：乘法次数 2.7×10¹⁰ 对 1.8×10⁷</p><p class="s mute">结果相同（只差浮点舍入）。耗时为作者机器实测，会随机器变化。</p>`,
];
build(root, "矩阵与线性变换", cards);
