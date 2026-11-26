// T02 FFN 与 SwiGLU · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { TF, TFV, build, fig, row, table, panel } from "../../_shared/tfcards.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const xs = Array.from({ length: 161 }, (_, i) => -4 + i / 20);
const acts = TFV.lines([["relu", TFV.C.gray], ["gelu", TFV.C.blue], ["silu", TFV.C.amber]].map(([a, c]) => ({ name: TF.ACT[a].name, color: c, pts: xs.map((x) => [x, TF.ACT[a].f(x)]) })), { w: 460, h: 420, ymin: -0.6, ymax: 4, sw: 3, charW: 14 });
const fits = [2, 8, 32].map((k) => { const f = TF.fitFFN(k, "relu"); return TFV.lines([{ name: "目标", color: TFV.C.gray, dash: "5 4", pts: f.xs.map((x, i) => [x, f.target[i]]) }, { name: "拟合", color: TFV.C.blue, pts: f.xs.map((x, i) => [x, f.pred[i]]) }], { w: 560, h: 200, title: `${k} 个隐藏单元 · 误差 ${f.mse >= 0.001 ? f.mse.toFixed(3) : f.mse.toExponential(1)}`, ymin: -1.7, ymax: 1.7, noLegend: true, xfmt: (v) => v.toFixed(1), sw: 2.6 }); });
let y = 0; const stack = { svg: fits.map((p) => { const s = `<g transform="translate(0,${y})">${p.svg}</g>`; y += p.h + 6; return s; }).join(""), w: 560, h: 620 };
const g = TF.params(TF.MODELS["GPT-2 small"]), l = TF.params(TF.MODELS["LLaMA 7B"]), n = (v) => v.toLocaleString("en-US");
const bar = (t, w, c, v) => `<div class="row" style="grid-template-columns:200px 1fr 140px;font-size:40px"><span>${t}</span><div class="bar"><i style="width:${w}%;background:${c}"></i></div><span class="num">${v}</span></div>`;
const cards = [
  `<div class="tag">Transformer · 第 2 块积木</div><h1>参数大头<br><span class="a">不在注意力</span></h1><p class="mute">在后面那个“前馈网络” →</p>`,
  `<div class="tag">分工</div><h2>先交换信息，<br>再各自加工</h2>${table(["工序", "词与词之间"], [["注意力", "互相混合"], ["FFN", "互不相干"]], [1])}<p>FFN：把向量<span class="a">放大</span>到 4 倍宽，过激活函数，再<span class="a">缩回</span>来。</p>`,
  `<div class="tag">手算</div><h2>去掉激活函数<br>会怎样？</h2>${panel("x = [1, −1]<br>放大 u = [1, −1, −2]<br>ReLU → [1, 0, 0]<br>缩回 → <span class='a'>[2, 1]</span><br><br>不用 ReLU → [1, −6]<br>= x·(W₁W₂)", 44)}<p class="s mute">没有非线性，两层永远等于一层。</p>`,
  `<div class="tag">激活函数</div><h2>ReLU、GELU、SiLU</h2>${fig(acts, 880)}<p class="s mute">原论文用 ReLU；GPT-2 用 GELU；SwiGLU 的“门”用 SiLU。</p>`,
  `<div class="tag">为什么要宽</div><h2>单元越多，<br>拼得越细</h2>${fig(stack, 820)}<p class="s mute">玩具实验：隐藏层随机固定，只解输出层。</p>`,
  `<div class="tag">参数量</div><h2>FFN : 注意力 = <span class="a">2 : 1</span></h2><div class="panel">${bar("FFN", 100, "#3b82f6", "8d²")}${bar("注意力", 50, "#8a8a8a", "4d²")}</div><p class="s">GPT-2 small 单层：FFN 占 ${(g.ffn / g.layer * 100).toFixed(1)}%<br>LLaMA 7B 单层：FFN 占 ${(l.ffn / l.layer * 100).toFixed(1)}%</p><p class="s mute">按配置算出的总参数：${n(g.total)} 与 ${n(l.total)}。</p>`,
  `<div class="tag">SwiGLU</div><h2>加一道门，<br>参数<span class="a">不增加</span></h2>${panel("( SiLU(xW) ⊙ xV ) W₂<br><br>3 个矩阵<br>d_ff 缩到 2/3 × 4d<br>3 · d · 8d/3 = <span class='a'>8d²</span>", 44)}<p class="s mute">门接近 0 时，对应的内容就被关掉。</p>`,
];
build(root, "FFN 与 SwiGLU", cards);
