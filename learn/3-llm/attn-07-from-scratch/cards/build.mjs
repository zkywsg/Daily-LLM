// 07 从零实现 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { writeFileSync, mkdirSync } from "node:fs";
import { build, vizSvg } from "../../_shared/cardkit.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const code = (s) => `<pre style="font:34px/1.6 ui-monospace,Menlo,monospace;white-space:pre-wrap;color:#ededed">${s}</pre>`;

// 函数调用与形状流程图（同一份 SVG：README 用白底版，图卡用黑底版）
const steps = [
  ["X", "[n, d]", "输入：n 个词，每词 d 维"],
  ["Q = X·Wq   K = X·Wk   V = X·Wv", "[n, d]", "三次矩阵乘"],
  ["reshape + 换轴：拆成 h 个头", "[h, n, dk]", "d = h × dk"],
  ["S = Q·Kᵀ / √dk", "[h, n, n]", "每个头一张 n×n 分数表", 1],
  ["S = where(mask, −∞, S)", "[h, n, n]", "causal_mask：未来位置设为 −∞", 1],
  ["A = softmax(S)", "[h, n, n]", "softmax：先减最大值再 exp", 1],
  ["O = A·V", "[h, n, dk]", "加权取信息", 1],
  ["换轴 + reshape：拼接各头", "[n, d]", "h × dk = d"],
  ["out = concat · Wo", "[n, d]", "输出形状与输入相同"],
];
function flowSvg(dark) {
  const bg = dark ? "#000" : "#fff", ink = dark ? "#ededed" : "#0f172a", mu = dark ? "#8a8a8a" : "#64748b", bx = dark ? "#0a0a0a" : "#f8fafc", ln = dark ? "#2e2e2e" : "#cbd5e1", ac = dark ? "#3b82f6" : "#2563eb", ab = dark ? "#0f1c33" : "#dbeafe";
  const bh = 52, gap = 22, x0 = 20, bw = 420, H = 70 + steps.length * (bh + gap), W = 860;
  let t = `<rect width="${W}" height="${H}" fill="${bg}"/><text x="${x0}" y="32" font-size="20" font-weight="700" fill="${ink}">multi_head_attention 的一次调用：每一步的形状</text>`;
  steps.forEach((s, i) => {
    const y = 56 + i * (bh + gap), hi = s[3];
    t += `<rect x="${x0}" y="${y}" width="${bw}" height="${bh}" rx="6" fill="${hi ? ab : bx}" stroke="${hi ? ac : ln}"/><text x="${x0 + 14}" y="${y + 22}" font-size="15" fill="${ink}" font-family="ui-monospace,Menlo,monospace">${s[0]}</text><text x="${x0 + 14}" y="${y + 42}" font-size="13" fill="${mu}">${s[2]}</text><text x="${x0 + bw + 16}" y="${y + 32}" font-size="18" font-weight="700" fill="${ac}" font-family="ui-monospace,Menlo,monospace">${s[1]}</text>`;
    if (i < steps.length - 1) t += `<path d="M${x0 + 30} ${y + bh}v${gap - 4}" stroke="${mu}"/><path d="M${x0 + 26} ${y + bh + gap - 8}l4 6 4-6z" fill="${mu}"/>`;
  });
  const y1 = 56 + 3 * (bh + gap), y2 = 56 + 6 * (bh + gap) + bh;
  t += `<path d="M${x0 + bw + 150} ${y1}h10v${y2 - y1}h-10" fill="none" stroke="${ac}"/><text x="${x0 + bw + 176}" y="${(y1 + y2) / 2 - 6}" font-size="14" fill="${ac}" font-family="ui-monospace,Menlo,monospace">scaled_dot_product</text><text x="${x0 + bw + 176}" y="${(y1 + y2) / 2 + 12}" font-size="14" fill="${ac}" font-family="ui-monospace,Menlo,monospace">_attention()</text>`;
  return { t, W, H };
}
mkdirSync(join(root, "assets"), { recursive: true });
{ const f = flowSvg(false); writeFileSync(join(root, "assets/function-flow.svg"), `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${f.W} ${f.H}" font-family="system-ui,-apple-system,sans-serif">${f.t}</svg>`); }
const flowCard = `<div class="tag">代码结构</div><h2>一次调用，<br>形状这样走</h2><div style="margin-top:20px">${(() => { const f = flowSvg(true); return `<svg viewBox="0 0 ${f.W} ${f.H}" width="900" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC,sans-serif">${f.t}</svg>`; })()}</div>`;
const cards = [
 `<div class="tag">注意力 · 从零实现</div><h1>60 行 NumPy，<br>写出<span class="a">多头注意力</span></h1><p class="mute">再用 7 个测试证明它是对的。<br>不依赖任何框架 →</p>`,
 `<div class="tag">核心 4 行</div><h2>公式就是代码</h2><div class="panel">${code(`S = Q @ K.swapaxes(-1,-2) / sqrt(dk)\nS = where(mask, -inf, S)\nA = softmax(S, axis=-1)\nreturn A @ V`)}</div><p class="mute s">前面可以有任意多个批维度（B、h），多头不用重写。</p>`,
 `<div class="tag">最有用的一个测试</div><h2>改未来的词，<br>前面的输出<span class="a">不变</span></h2><p>这是检查自回归模型有没有“偷看后面”的标准做法：</p><div class="panel">${code(`X2[:, 4:] += 噪声   # 只改未来\nassert allclose(\n  out[:, :4], out2[:, :4])`)}</div>`,
 `<div class="tag">练习 1</div><h2>删掉“减最大值”，<br>会怎样？</h2><div class="panel">${code(`z = [1000, 1001, 999]\nexp(z) → inf\ninf / inf → <span style="color:#3b82f6">nan</span>`)}</div><p class="mute s">我实测：7 个测试里只有溢出测试失败，其余照常通过。这类 bug 只在极端输入下暴露，最容易漏测。</p>`,
 `<div class="tag">下一步</div><h2>你来试试</h2><div class="panel"><div class="big" style="font-size:48px">· 写滑动窗口掩码<br>· 实现 GQA（分组查询）<br>· 和 PyTorch 官方实现对照</div></div><p class="mute s">代码和练习答案都在仓库里：learn / 3-llm / attn-07-from-scratch</p>`,
];
cards.splice(1, 0, flowCard);
build(root, "从零实现", cards);
