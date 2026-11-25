// 02 多头注意力 · 图卡。用法：node cards/build.mjs
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { build, dotGrid, pct } from "../../_shared/cardkit.mjs";
const require = createRequire(import.meta.url);
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const E = require("../../_shared/engine.js");
const toks = E.tokenize("小猫没有过马路，因为它太累了"), IT = toks.findIndex((t) => t.text === "它"), GUO = toks.findIndex((t) => t.text === "过");
const M = E.multihead(E.embed(toks));
const f2 = (v) => v.map((x) => x.toFixed(2)).join(", ");
const per = 4 * 12288 * 12288, total = per * 96;
const fmtYi = (x) => (x / 1e8).toFixed(1).replace(/\.0$/, "");
const cards = [
 `<div class="tag">注意力 · 进阶</div><h1>GPT-3 的<br>每一层，有 <span class="a">96 个</span><br>注意力头</h1><p class="mute">一个头不够吗？<br>6 张图讲清楚 →</p>`,
 `<div class="tag">问题</div><h2>一个头，只有<br>一套找人的规则</h2><p>一个头的 q 和 k 只学一种“谁和谁相关”。</p><p>但一句话里有很多种关系：指代、动作和对象、修饰……</p><p class="mute s">所以并行放很多个头，每个头学自己的规则。</p>`,
 `<div class="tag">两个头</div><h2>同一句话，两张表</h2><div class="two" style="margin-top:20px"><div><p class="a s" style="margin:0;font-weight:700">头 1：指代</p>${dotGrid(toks, M.heads[0].A, { sel: IT, cell: 50, left: 78, font: 19 })}</div><div><p class="a s" style="margin:0;font-weight:700">头 2：动作与对象</p>${dotGrid(toks, M.heads[1].A, { sel: GUO, cell: 50, left: 78, font: 19 })}</div></div><p class="mute s" style="margin-top:10px">头 1 高亮“它”：找到“小猫”。头 2 高亮“过”：找到“马路”。（玩具模型，人工设定）</p>`,
 `<div class="tag">合并</div><h2>拼接，再混合</h2><p>每个头各出一个小向量，把它们<b>首尾拼起来</b>，再乘一个输出矩阵 W_O，得到这个词最终的新表示。</p><div class="panel"><div class="big">“它”<br>头 1 → <span class="a">[${f2(M.heads[0].O[IT])}]</span><br>头 2 → <span class="a">[${f2(M.heads[1].O[IT])}]</span><br>拼接 → 6 维</div></div>`,
 `<div class="tag">参数</div><h2>多头不是白加的</h2><p>每个头有自己的 W_Q、W_K、W_V，再加一个 W_O。当 头数 × d_k = d_model 时，一层注意力的参数约为：</p><div class="panel"><div class="big"><span class="a">4 × d_model²</span><br>GPT-3：4 × 12288²<br>≈ ${fmtYi(per)} 亿 / 层</div></div><p class="mute s">× 96 层 ≈ ${fmtYi(total)} 亿，约占 1750 亿总参数的三分之一（按公式推算）。</p>`,
 `<div class="tag">回顾</div><h2>多头 = 多套规则并行</h2><p>每个头各算一张注意力表，输出拼接，再用 W_O 合并。</p><div class="panel"><p class="mute s" style="margin:0">没讲到的：每个头真正学到了什么，往往很难解读。这里的两个头是为了教学人工设定的。</p></div><p class="s mute">下一张：注意力还有两个补丁，掩码和位置 →</p>`,
];
build(root, "多头注意力", cards);
