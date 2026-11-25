// 03 掩码与位置 · 图卡。用法：node cards/build.mjs
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { build, dotGrid, VIZ, vizSvg } from "../../_shared/cardkit.mjs";
const require = createRequire(import.meta.url);
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const E = require("../../_shared/engine.js");
const toks = E.tokenize("小猫没有过马路，因为它太累了"), n = toks.length;
const Am = E.head(E.embed(toks), E.HEADS[0], { mask: true }).A;
const perm = [3, 0, 6, 2, 5, 1, 4];
const d0 = E.shuffleDiff(toks, perm, { pos: false }), d1 = E.shuffleDiff(toks, perm, { pos: true });
const shuffled = perm.map((i) => toks[i]);
const pe = toks.slice(0, 5).map((_, i) => E.posEnc(i));
const cell = (v) => `<td style="padding:6px 12px;background:rgba(59,130,246,${Math.min(0.6, Math.abs(v) * 0.9).toFixed(2)});font-size:32px;text-align:right">${v.toFixed(2)}</td>`;
const Rp = E.head(E.embed(toks), E.HEADS[0], {}), cm = 52, lab = toks.map((t) => t.text);
const hM = VIZ.heat(Rp.S.map((r, i) => r.map((_, j) => (j > i ? -99 : 0))), { cell: cm, x: 0, y: 80, gut: 78, title: "掩码 M", sub: "未来 = −∞", rows: lab, cols: lab.map((t) => t.slice(0, 1)), fmt: (v) => (v < -9 ? "−∞" : "0"), max: 99 });
const hA2 = VIZ.heat(Am, { cell: cm, x: hM.w + 60, y: 80, gut: 78, title: "softmax(S+M)", sub: "−∞ → 0", cols: lab.map((t) => t.slice(0, 1)), fmt: (v) => String(Math.round(v * 100)), max: 1 });
const maskCard = `<div class="tag">掩码 · 怎么实现</div><h2>加一张 −∞ 的表</h2><div style="margin-top:34px">${vizSvg(hM.svg + `<text x="${hM.w + 20}" y="${80 + 16 + 3.5 * cm}" class="vs" style="font-size:26px">→</text>` + hA2.svg, hM.w + 60 + hA2.w + 10, 80 + 16 + 7 * cm + 10)}</div><p class="mute s" style="margin-top:20px">先把分数加上这张表，再做 softmax。e^−∞ = 0，所以未来位置的权重恰好是 0。</p>`;
const cards = [
 `<div class="tag">注意力 · 补丁</div><h1>写字的模型，<br>为什么<span class="a">不许偷看</span><br>后面的字？</h1><p class="mute">掩码与位置。<br>6 张图讲清楚 →</p>`,
 `<div class="tag">因果掩码</div><h2>灰色的，看不见</h2><div style="margin-top:20px">${dotGrid(toks, Am, { mask: true, cell: 112, left: 150, font: 32, numbers: true })}</div><p class="mute s" style="margin-top:6px">第 i 个词只能看第 1 到 i 个词。数字是百分比权重。</p>`,
 `<div class="tag">为什么</div><h2>训练时一次预测所有位置</h2><div class="panel"><p class="s" style="margin:0;line-height:1.8">看到「小猫」→ 预测「没有」<br>看到「小猫 没有」→ 预测「过」<br>看到「小猫 没有 过」→ 预测「马路」<br>……</p></div><p>没有掩码，第 i 个位置能直接看到第 i+1 个词，也就是它要预测的答案。预测变成了抄答案。</p>`,
 `<div class="tag">位置 · 问题</div><h2>注意力本身，<br>看不见词序</h2><p>把词序打乱，每一对词之间的权重<b class="a">一模一样</b>：</p><div class="panel"><div style="font-size:42px;font-weight:700;line-height:1.7">${toks.map((t) => t.text).join(" ")}<br>↓ 打乱<br><span class="a">${shuffled.map((t) => t.text).join(" ")}</span></div><p class="s mute" style="margin:10px 0 0">同一对词的权重，最大变化 = <b class="a">${d0.toFixed(4)}</b></p></div>`,
 `<div class="tag">位置 · 解法</div><h2>把位置“加”进词向量</h2><p>给每个位置一个固定的正弦/余弦向量，加到词向量上。（论文 2017 的做法）</p><div class="panel"><table style="border-collapse:collapse;margin:0 auto">${pe.map((r, i) => `<tr><td style="padding:6px 12px;font-size:28px;white-space:nowrap;color:#8a8a8a">位置 ${i + 1}</td>${r.map(cell).join("")}</tr>`).join("")}</table></div><p class="mute s">加上后，同样打乱词序，权重最大变化变成 ${d1.toFixed(2)}：顺序被编码进来了。（教学参数，非论文原值）</p>`,
 `<div class="tag">回顾</div><h2>两个补丁</h2><div class="panel"><div class="big"><span class="a">掩码</span>：不许看未来<br><span class="a">位置</span>：让它知道顺序</div></div><p class="mute s">没讲到的：现代大模型更常用旋转位置编码（RoPE）等做法，这里只讲最初的正弦版本。</p>`,
];
cards.splice(2, 0, maskCard);
build(root, "掩码与位置", cards);
