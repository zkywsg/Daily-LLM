// T01 残差与 LayerNorm · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { TF, TFV, build, load, fig, row, table, panel } from "../../_shared/tfcards.mjs";
load("../tf-04-mini-gpt/demo/ablation.js");
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const col = { plain: TFV.C.red, res: TFV.C.amber, pre: TFV.C.blue, post: TFV.C.green };
const R = Object.keys(TF.MODES).map((m) => [m, TF.depthAvg(m, 24, 1)]);
const e = (v) => v.toExponential(1).replace(/e([+-]?\d+)/, (_, k) => " × 10<sup>" + +k + "</sup>");
const chart = TFV.lines(R.map(([m, r]) => ({ name: TF.MODES[m], color: col[m], pts: r.grad.map((v, i) => [i, v]) })), { log: true, w: 460, h: 400, xlabel: "第几层（0 = 输入端）", noLegend: true, sw: 3 });
const legend = `<p class="s" style="margin-top:20px">${Object.entries(TF.MODES).map(([m, n], i) => `<span style="color:${col[m]}">● ${n}</span>${i === 1 ? "<br>" : "　"}`).join("")}</p>`;
const ln = TF.layerNorm([2, 4, 6, 8]).y, rms = TF.rmsNorm([2, 4, 6, 8]).y;
const A = window.ABLATION.rows;
const cards = [
  `<div class="tag">Transformer · 第 1 块积木</div><h1>大模型凭什么<br>能叠<span class="a">几十层</span>？</h1><p class="mute">两样东西：残差、归一化 →</p>`,
  `<div class="tag">一个层的结构</div><h2>子层外面，<br>包着同样两样</h2>${fig(row([TFV.block("post"), TFV.block("pre")], 20), 880)}<p class="s mute">左：原论文。右：LLaMA 等常用的写法。</p>`,
  `<div class="tag">残差</div><h2>不替换输入，<br>只<span class="a">加一笔修改</span></h2>${panel("x ← x + 子层(x)<br><br>反向传播时<br>每层连乘的是<br><span class='a'>( I + ∂f/∂x )</span>")}<p class="s mute">多出来的 I，让梯度有一条直接传回输入的路。</p>`,
  `<div class="tag">归一化 · 手算</div><h2>x = [2, 4, 6, 8]</h2>${panel(`LayerNorm：减均值，除标准差<br><span class="a mono">[${ln.map((v) => v.toFixed(2)).join(", ")}]</span><br><br>RMSNorm：只除均方根<br><span class="a mono">[${rms.map((v) => v.toFixed(2)).join(", ")}]</span>`, 40)}<p class="s mute">RMSNorm 少算一个均值，LLaMA 用的是它。</p>`,
  `<div class="tag">实验 · 24 层随机网络</div><h2>梯度还剩多少？</h2>${fig(chart, 880)}${legend}<p class="s mute">无残差：输入端只剩 ${e(R[0][1].grad[0])}；只有残差：反而涨到 ${e(R[1][1].grad[0])}。纵轴对数。</p>`,
  `<div class="tag">实验 · 真的训练</div><h2>6 层去掉残差，<br><span class="a">完全学不动</span></h2>${table(["迷你 GPT", "损失", "知识题"], [0, 1, 4, 5].map((i) => [A[i].name, A[i].val_mean.toFixed(2), A[i].correct.join("/")]), [3])}<p class="s mute">各 3 个随机种子，训练 1500 步，满分 20。只代表这个小模型。</p>`,
];
build(root, "残差与归一化", cards);
