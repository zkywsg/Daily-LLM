// T04 迷你 GPT · 图卡。数字来自训练产物 demo/model.js 与 demo/ablation.js。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { TFV, MiniGPT, build, load, fig, table, panel } from "../../_shared/tfcards.mjs";
load("../tf-04-mini-gpt/demo/model.js"); load("../tf-04-mini-gpt/demo/ablation.js");
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const M = window.MODEL, G = MiniGPT.load(M), A = window.ABLATION.rows, n = M.n_params.toLocaleString("en-US");
const q = G.forward(G.encode("小鸟喜欢吃虫子。熊猫喜欢吃"));
const top = (l) => { const p = MiniGPT.softmax(l), i = p.indexOf(Math.max(...p)); return [M.chars[i], p[i]]; };
const curve = TFV.lines([{ name: "损失", color: TFV.C.blue, pts: M.curve }], { w: 460, h: 290, ymin: 0, ymax: 4, xlabel: "训练步数", ref: { y: M.optimal, label: "最优值 " + M.optimal.toFixed(3) }, noLegend: true, sw: 3 });
const cut = (s) => s.slice(0, 15) + "…";
const snaps = M.snapshots.filter((s) => [0, 50, 200, 3000].includes(s.step));
const subj = (who) => { const t = `小鸟最爱唱歌。${who}喜欢吃`, r = G.forward(G.encode(t)), a = r.attn[1][2], x = a[a.length - 1], i = t.lastIndexOf(who); return x[i] + x[i + 1]; };
const heads = (l) => [0, 1, 2, 3].map((h) => MiniGPT.quiz(MiniGPT.load(MiniGPT.ablate(M, [{ layer: l, head: h }]))).filter((v) => v.ok).length);
const h0 = heads(0), h1 = heads(1);
const cards = [
  `<div class="tag">Transformer · 拼起来</div><h1>${n} 个参数<br>从零训练<br>一个 <span class="a">GPT</span></h1><p class="mute">只用 NumPy，反向传播手写 →</p>`,
  `<div class="tag">给它一个小世界</div><h2>规则全知道，<br>才好批改</h2>${panel("小猫喜欢吃鱼。<br>熊猫住在高山里。<br>兔子最爱蹦跳。<br>小狗吃了骨头，很开心。", 52)}<p class="s mute">5 种动物 × 4 种句式，词表 ${M.chars.length} 个字。</p>`,
  `<div class="tag">看着它学会</div><h2>流畅 ≠ 正确</h2>${table(["步数", "续写“熊猫住在”之后"], snaps.map((s) => [s.step, cut(s.texts["熊猫住在"].slice(4))]), [2])}<p class="s mute">第 200 步：句子像样了，熊猫却“住在沙发上”。</p>`,
  `<div class="tag">训练曲线</div><h2>它学到头了吗？</h2>${fig(curve, 880)}${panel(`模型 <span class="a">${M.final_val.toFixed(4)}</span><br>最优值 ${M.optimal.toFixed(4)}`, 52)}<p class="s mute">下一句讲谁是随机的，没人能猜中，所以损失降不到 0。</p>`,
  `<div class="tag">打开看 · 逐层猜</div><h2>“熊猫喜欢吃 ___”</h2>${table(["读到哪一步", "第一猜", "把握"], q.lens.map((l, k) => { const [c, p] = top(l); return [k ? `第 ${k} 层后` : "进第 1 层前", c, (p * 100).toFixed(1) + "%"]; }), [2])}<p class="s mute">在这个例子里，答案是第 2 层写进去的。</p>`,
  `<div class="tag">别急着起名字</div><h2>一个头像是<br>在“找主语”…</h2>${table(["主语", "L1H2 放在主语上的权重"], ["熊猫", "小猫", "小鸟", "兔子"].map((w) => [w, subj(w).toFixed(2)]), [0])}<p class="s">换个动物就不灵了。<span class="a">一个漂亮的例子，说明不了一个头的功能。</span></p>`,
  `<div class="tag">关掉一个头试试</div><h2>20 道题<br>还能对几道？</h2>${table(["", "H0", "H1", "H2", "H3"], [["第 1 层", ...h0], ["第 2 层", ...h1]])}<p class="s mute">图中层号从 1 数起；互动页里记作 L0、L1。功能分散在多个头上。</p>`,
  `<div class="tag">对照实验</div><h2>损失越低，<br><span class="a">不一定越好</span></h2>${table(["设置", "损失", "知识题"], [0, 3, 2, 5].map((i) => [A[i].name, A[i].val_mean.toFixed(3), A[i].correct.join("/")]), [1])}<p class="s mute">去掉因果掩码 = 训练时偷看答案。3 个种子，满分 20。</p>`,
];
build(root, "从零训练迷你 GPT", cards);
