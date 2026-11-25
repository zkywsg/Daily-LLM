// 01 动机 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { build } from "../../_shared/cardkit.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const N = 8;
const chain = `<svg viewBox="0 0 900 260" width="900" xmlns="http://www.w3.org/2000/svg">${Array.from({ length: N }, (_, k) => { const x = 50 + k * 114; return `<circle cx="${x}" cy="60" r="30" fill="${k === 0 ? "#e5a50a" : k === N - 1 ? "#3b82f6" : "#0a0a0a"}" stroke="#2e2e2e" stroke-width="3"/>` + (k < N - 1 ? `<line x1="${x + 32}" y1="60" x2="${x + 82}" y2="60" stroke="#3b82f6" stroke-width="4"/>` : ""); }).join("")}<text x="0" y="130" font-size="34" fill="#8a8a8a" font-family="PingFang SC">RNN：7 步才传到</text>${Array.from({ length: N }, (_, k) => { const x = 50 + k * 114; return `<circle cx="${x}" cy="200" r="30" fill="${k === 0 ? "#e5a50a" : k === N - 1 ? "#3b82f6" : "#0a0a0a"}" stroke="#2e2e2e" stroke-width="3"/>`; }).join("")}<path d="M50 168 Q450 100 848 168" fill="none" stroke="#3b82f6" stroke-width="5"/></svg>`;
const cards = [
 `<div class="tag">注意力 · 起点</div><h1>注意力<br>是为了解决<span class="a">什么问题</span>？</h1><p class="mute">先看它的前辈 RNN 卡在哪 →</p>`,
 `<div class="tag">RNN</div><h2>像传话游戏</h2><p>一句话要一个词一个词地压进同一个向量。第 1 个词想影响第 50 个词，要经过 <b class="a">49 次转述</b>。</p><div style="margin-top:20px">${chain}</div>`,
 `<div class="tag">后果</div><h2>传得越远，<br>信号越弱</h2><div class="panel"><div class="big" style="font-size:50px">∂h<sub>T</sub>/∂x<sub>0</sub> ∝ ∏ w·(1−h<sub>t</sub>²)</div></div><p>每一步连乘一个小于 1 的数，几十步后接近 0。这叫<b class="a">梯度消失</b>：远处的信息学不到。</p>`,
 `<div class="tag">转机 · 2014</div><h2>别压缩了，<br><span class="a">回头看</span></h2><p>翻译模型原来把整句话压成一个固定长度的向量。Bahdanau 等人让它在每一步直接去看源句里的各个词，并学会该看哪里。</p><p class="mute s">这就是注意力的起点。</p>`,
 `<div class="tag">对比</div><h2>路径长度<br>从 n 变成 1</h2><div class="panel"><table style="width:100%;font-size:40px;border-collapse:collapse"><tr style="color:#8a8a8a"><td></td><td>注意力</td><td>RNN</td></tr><tr><td style="padding:12px 0">最大路径</td><td class="a">O(1)</td><td>O(n)</td></tr><tr><td style="padding:12px 0">顺序操作</td><td class="a">O(1)</td><td>O(n)</td></tr><tr><td style="padding:12px 0">每层计算</td><td>O(n²d)</td><td>O(nd²)</td></tr></table></div><p class="mute s">来源：Vaswani et al. 2017, Table 1。代价是 n² 的分数表。</p>`,
];
build(root, "为什么发明注意力", cards);
