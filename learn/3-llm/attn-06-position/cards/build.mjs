// 06 位置编码 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { build, VIZ, vizSvg } from "../../_shared/cardkit.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const rot = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1], q = [1, 0.3], k = [0.6, 0.9], th = 0.5;
const pairs = [[5, 2], [10, 7], [103, 100], [3, 0]];
const rows = pairs.map(([m, n]) => `<tr><td style="padding:8px 22px;font-size:40px">(${m}, ${n})</td><td style="padding:8px 22px;font-size:40px;color:#8a8a8a">${m - n}</td><td style="padding:8px 22px;font-size:40px;color:#3b82f6;font-weight:700">${dot(rot(q, m * th), rot(k, n * th)).toFixed(4)}</td></tr>`).join("");
// ALiBi 偏置
const n = 8, slopes = Array.from({ length: 4 }, (_, i) => Math.pow(2, -8 * (i + 1) / 8));
const bias = (m) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (j <= i ? -m * (i - j) : 0)));
const hb = VIZ.heat(bias(0.5), { cell: 58, x: 0, y: 70, title: "斜率 1/2（近视头）", skip: (i, j) => j > i, fmt: (v) => v.toFixed(1), max: 3.5 });
const hc = VIZ.heat(bias(1 / 16), { cell: 58, x: hb.w + 40, y: 70, title: "斜率 1/16（远视头）", skip: (i, j) => j > i, fmt: (v) => v.toFixed(1), max: 3.5 });
const cards = [
 `<div class="tag">注意力 · 位置</div><h1>模型怎么知道<br><span class="a">谁在谁前面</span>？</h1><p class="mute">三种思路：加编号、旋转、扣分。<br>5 张图讲清楚 →</p>`,
 `<div class="tag">加法派</div><h2>给每个位置<br>一个编号向量</h2><div class="panel"><div class="big" style="font-size:54px">x<sub>pos</sub> ← x<sub>pos</sub> + <span class="a">PE(pos)</span></div></div><p>正弦版（原论文）不需要训练；可学习版是一张查表。</p><p class="mute s">缺点：“差 3 个位置”这件事，要模型自己从两个编号里悟出来。</p>`,
 `<div class="tag">旋转派 · RoPE</div><h2>把 q、k 按位置<br><span class="a">转一个角度</span></h2><p>位置 m 的向量转 m·θ。再点积，结果只取决于角度差：</p><div class="panel" style="text-align:center"><div class="big" style="font-size:50px">⟨R(mθ)q, R(nθ)k⟩<br>= q<sup>T</sup> R((n−m)θ) k</div></div><p class="mute s">只依赖相对距离 n−m。旋转只用在 q、k 上，不用在 v 上。</p>`,
 `<div class="tag">验算</div><h2>距离同为 3，<br>点积一模一样</h2><div class="panel" style="text-align:center"><table style="border-collapse:collapse;margin:0 auto"><tr><td style="padding:8px 22px;font-size:30px;color:#8a8a8a">(m, n)</td><td style="padding:8px 22px;font-size:30px;color:#8a8a8a">m−n</td><td style="padding:8px 22px;font-size:30px;color:#8a8a8a">点积</td></tr>${rows}</table></div><p class="mute s">q=[1, 0.3]，k=[0.6, 0.9]，θ=0.5。</p>`,
 `<div class="tag">扣分派 · ALiBi</div><h2>离得越远，<br><span class="a">扣得越多</span></h2><div style="margin-top:20px">${vizSvg(hb.svg + hc.svg, hb.w + 40 + hc.w, 70 + 16 + n * 58 + 10)}</div><p class="s mute" style="margin-top:14px">在分数上减 斜率×距离。不同的头用不同斜率：有的只看近处，有的能看远。</p>`,
];
build(root, "位置编码", cards);
