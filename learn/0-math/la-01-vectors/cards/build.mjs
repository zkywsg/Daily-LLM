// 线代 01 向量 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { LA, LAV, MODEL, TFV, build, fig, table, panel, grid } from "../../_shared/lacards.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), ".."), C = LAV.C;
const a = [3, 4], b = [4, 3];
const pl = LAV.plane({ size: 300, range: 5.5, vectors: [{ v: a, color: C.blue, label: "a = [3,4]", dx: -62, dy: -10 }, { v: b, color: C.amber, label: "b = [4,3]", dx: -34, dy: 40 }] });
const ball = (q, title) => LAV.plane({ title, size: 200, range: 1.35, paths: [{ color: C.blue, fill: "rgba(59,130,246,.2)", pts: Array.from({ length: 201 }, (_, i) => { const t = (i * Math.PI) / 100, v = [Math.cos(t), Math.sin(t)], n = LA.norm(v, q); return [v[0] / n, v[1] / n]; }) }] });
const hists = [2, 10, 100, 1000].map((d) => { const r = LA.cosSample(d, 4000); return TFV.lines([{ name: "", color: C.blue, pts: LA.histogram(r.values) }], { w: 330, h: 230, title: `${d} 维`, ymin: 0, noLegend: true, sw: 3, xfmt: (v) => v.toFixed(1), yfmt: (v) => Math.round(v * 100) + "%" }); });
const ch = [..."猫狗鸟兔熊住喜最"], sim = LA.embeddingSim(MODEL, ch), at = (x, y) => sim[ch.indexOf(x)][ch.indexOf(y)].toFixed(2);
const cards = [
  `<div class="tag">线性代数 · 第 1 块</div><h1>AI 怎么知道<br>两个字<span class="a">像不像</span>？</h1><p class="mute">从一排数字开始 →</p>`,
  `<div class="tag">向量</div><h2>一排数，<br>也是<span class="a">一支箭头</span></h2>${fig(pl, 760)}<p class="s mute">大模型里每个词都是一个几百到几千维的向量。</p>`,
  `<div class="tag">手算</div><h2>点积</h2>${panel(`a·b = 3×4 + 4×3 = <span class="a">${LA.dot(a, b)}</span><br>|a| = |b| = ${LA.norm(a)}<br>cos = 24 ÷ 25 = <span class="a">${LA.cos(a, b).toFixed(2)}</span><br>夹角 ${LA.angle(a, b).toFixed(1)}°`, 48)}<p class="s mute">同向最大，垂直为 0，反向为负。除掉长度，就是余弦相似度。</p>`,
  `<div class="tag">范数</div><h2>“长度”<br>不止一种</h2>${fig(grid([ball(1, "L1"), ball(2, "L2"), ball(Infinity, "L∞")], 3, 16), 900)}${table(["[3,4] 的长度", "值"], [["L1 绝对值相加", "7"], ["L2 勾股定理", "5"], ["L∞ 取最大", "4"]])}`,
  `<div class="tag">高维的怪事</div><h2>随机两个方向，<br>几乎总是<span class="a">垂直</span></h2>${fig(grid(hists, 2, 20), 900)}<p class="s mute">夹角余弦的分布。维度越高越挤在 0。<br>标准差约等于 1/√d。</p>`,
  `<div class="tag">实验 · 迷你 GPT 的词向量</div><h2>它觉得<br>“兔”<span class="a">不像</span>“猫”</h2>${fig(LAV.heat(sim, { rows: ch, cols: ch, cell: 40 }), 860)}<p class="s mute">猫·狗 ${at("猫", "狗")}，兔·熊 ${at("兔", "熊")}，猫·兔 ${at("猫", "兔")}。</p>`,
  `<div class="tag">为什么</div><h2>相似的是<br><span class="a">用法</span></h2>${table(["字", "在语料里的位置"], [["猫 狗 鸟", "跟在“小”后面"], ["兔 熊", "主语开头"], ["住 喜 最", "紧接主语之后"]])}<p class="s">模型只见过文字之间的搭配。</p><p class="s mute">语料只有 20 种句子，换 3 个随机种子分组依旧。</p>`,
];
build(root, "向量与内积", cards);
