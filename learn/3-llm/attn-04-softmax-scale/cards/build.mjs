// 04 softmax 与 √d · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import { build, VIZ, vizSvg } from "../../_shared/cardkit.mjs";
const require = createRequire(import.meta.url);
const E = require("../../_shared/engine.js");
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const ds = [4, 16, 64, 256, 1024], ex = ds.map((d) => [d, E.scaleExperiment(d, 8, 2000, 7)]);
const sm = (s) => { const m = Math.max(...s), e = s.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b); return e.map((v) => v / z); };
const jac = (a) => a.map((ai, i) => a.map((aj, j) => ai * ((i === j ? 1 : 0) - aj)));
const base = [1.5, 0.8, 0.2, -0.6], nm = ["A", "B", "C", "D"];
const J1 = jac(sm(base)), J2 = jac(sm(base.map((v) => v * 8)));
const jh = (J, x, t) => VIZ.heat(J, { cell: 78, x, y: 80, gut: 0, title: t, fmt: (v) => v.toFixed(2), max: 0.25 });
const j1 = jh(J1, 0, "k=1：正常"), j2 = jh(J2, j1.w + 70, "k=8：饱和");
const bars = ex.map(([d, r]) => `<div class="row" style="grid-template-columns:150px 1fr 1fr;gap:16px"><span>d=${d}</span><div class="bar"><i style="width:${r.unscaled * 100}%;background:#8a8a8a"></i></div><div class="bar"><i style="width:${r.scaled * 100}%"></i></div></div>`).join("");
const cards = [
 `<div class="tag">注意力 · softmax</div><h1>为什么注意力公式里<br>要<span class="a">除以 √d</span>？</h1><p class="mute">一个看起来很小的细节，<br>决定了模型学不学得动 →</p>`,
 `<div class="tag">问题</div><h2>维度越高，<br>点积<span class="a">波动越大</span></h2><p>q、k 各分量独立、均值 0、方差 1 时：</p><div class="panel"><div class="big">Var(q·k) = d<br><span class="mute" style="font-size:44px">标准差 = √d</span></div></div><p class="mute s">64 维时标准差约 8，1024 维约 32。</p>`,
 `<div class="tag">后果</div><h2>分数一大，<br>softmax 就“一家独大”</h2><div class="panel">${bars}<p class="s mute" style="margin:14px 0 0">灰：不缩放 · 蓝：除以 √d　（最大权重）</p></div><p class="s mute">不缩放时，d=1024 的最大权重平均约 ${Math.round(ex[4][1].unscaled * 100)}%；缩放后约 ${Math.round(ex[4][1].scaled * 100)}%。</p>`,
 `<div class="tag">梯度</div><h2>一家独大，<br>梯度就<span class="a">消失</span></h2><div style="margin-top:20px">${vizSvg(j1.svg + j2.svg, j1.w + 70 + j2.w, 80 + 16 + 4 * 78 + 10)}</div><p class="s mute" style="margin-top:14px">softmax 的梯度矩阵 J = diag(a) − aaᵀ。右边权重尖锐，J 几乎全 0：改分数几乎不影响输出，学不动。</p>`,
 `<div class="tag">解法</div><h2>除以 √d，<br>方差回到 1</h2><div class="panel"><div class="big">Var(q·k / √d) = d / d = <span class="a">1</span></div></div><p>分数尺度与维度无关，softmax 的输入分布固定，梯度不随维度变小。</p><p class="mute s">这是初始化附近的论证；训练后的 q、k 不再严格满足独立假设。</p>`,
];
build(root, "softmax 与 √d", cards);
