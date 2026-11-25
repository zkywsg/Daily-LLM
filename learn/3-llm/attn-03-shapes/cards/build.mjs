// 03 形状 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { build, VIZ } from "../../_shared/cardkit.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const B = 2, n = 501, d = 768, h = 12, dk = d / h, fmt = (x) => x.toLocaleString("zh-CN");
const el1 = B * n * d, el2 = B * h * n * n, mib = (e) => (e * 2 / 2 ** 20).toFixed(1);
const n2 = 5010, el3 = B * h * n2 * n2, gib = (e) => (e * 2 / 2 ** 30).toFixed(1);
const flow = [["输入 X", `[B, n, d]`], ["拆头 reshape", `[B, n, h, dk]`], ["换轴 transpose", `[B, h, n, dk]`], ["分数 Q·Kᵀ", `[B, h, n, n]`, 1], ["加权 A·V", `[B, h, n, dk]`], ["拼头", `[B, n, d]`]];
const areaCard = `<div class="tag">谁最占内存</div><h2>橙色方块，<br>就是那张分数表</h2><div style="margin-top:30px;zoom:1.45">${VIZ.areaCompare({ n, d, h }).replace(/<p class="note">.*<\/p>/, "").replace('class="viz"', 'class="viz" font-family="PingFang SC"')}</div><p class="mute s">面积与元素个数成正比（n=${n}，d=${d}，h=${h}）。分数表是输入的 ${(el2 / el1).toFixed(0)} 倍。</p>`;
const cards = [
 `<div class="tag">注意力 · 形状</div><h1><span class="a">[B, n, d]</span> 进，<br><span class="a">[B, n, d]</span> 出</h1><p class="mute">中间发生了什么？<br>看懂任何一段注意力代码，只靠这一招 →</p>`,
 `<div class="tag">一次多头注意力</div><h2>形状怎么走</h2><div class="panel">${flow.map(([a, b, hi]) => `<div style="display:flex;justify-content:space-between;font-size:42px;margin:14px 0"><span class="mute">${a}</span><b class="${hi ? "a" : ""}" style="font-family:ui-monospace,Menlo,monospace">${b}</b></div>`).join("")}</div><p class="mute s">B 批大小 · n 序列长度 · d 向量维度 · h 头数 · dk = d/h</p>`,
 `<div class="tag">为什么贵</div><h2>只有一个张量<br>随 n² 增长</h2><div class="panel"><p class="s mute" style="margin:0">GPT-2 small：d=768，h=12，B=2，n=${n}</p><div class="big">输入 [${B}, ${n}, ${d}]<br><span class="mute" style="font-size:44px">${fmt(el1)} 个数 · ${mib(el1)} MiB</span></div><div class="big a">分数表 [${B}, ${h}, ${n}, ${n}]<br><span style="font-size:44px">${fmt(el2)} 个数 · ${mib(el2)} MiB</span></div></div><p class="mute s">每个数按 2 字节（fp16）算。</p>`,
 `<div class="tag">n 扩大 10 倍</div><h2>输入 ×10，<br>分数表 <span class="a">×100</span></h2><div class="panel"><div class="big">n = ${n2}<br><span class="mute" style="font-size:44px">输入 ≈ ${mib(B * n2 * d)} MiB</span><br><span class="a" style="font-size:56px">分数表 ≈ ${gib(el3)} GiB</span></div></div><p class="mute s">这就是长上下文难的来源。朴素实现的开销，FlashAttention 等会优化。</p>`,
 `<div class="tag">最难排查的 bug</div><h2>reshape 和 transpose<br>的顺序</h2><p>先<b class="a">拆头</b>（reshape），再<b class="a">换轴</b>（transpose）。合并时先换回，再 reshape。</p><div class="panel"><p class="s" style="margin:0">换轴忘了换回就 reshape：形状碰巧对得上，<b class="a">不报错、结果错</b>。</p></div>`,
];
cards.splice(3, 0, areaCard);
build(root, "注意力 · 形状", cards);
