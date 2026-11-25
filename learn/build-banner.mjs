// 生成 README 顶部横幅 .github/assets/banner.png：真实互动页截图 + 真实图卡
// 用法：node learn/build-banner.mjs（需要 Edge）
import { writeFileSync, readFileSync, mkdtempSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const assets = join(here, "../.github/assets");
const edge = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge";
const R = JSON.parse(readFileSync(join(here, "roadmap.json"), "utf8"));
const tmp = mkdtempSync(join(tmpdir(), "banner-"));
const shot = (url, w, h, out) => execFileSync(edge, ["--headless", "--disable-gpu", "--hide-scrollbars", `--window-size=${w},${h}`, `--screenshot=${out}`, url], { stdio: "ignore" });
const demo = join(tmp, "demo.png");
shot("file://" + join(here, "3-llm/01-attention/demo/index.html"), 1100, 3000, demo);
const card = (t, n) => "file://" + join(here, `3-llm/${t}/cards/out/${n}.png`);
const prog = R.layers.map((l) => {
  const ns = l.groups.flatMap((g) => g.nodes), d = ns.filter((x) => x.s === "done").length, p = ns.filter((x) => x.s === "partial").length;
  return `<div class="L"><span>${l.id} ${l.name}</span><b style="width:${Math.round(((d + p / 2) / ns.length) * 100)}%"></b></div>`;
}).join("");
const html = `<!doctype html><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0}
body{width:1600px;height:720px;background:#000;color:#e8eaed;font:16px/1.4 "PingFang SC","SF Pro Text",-apple-system,sans-serif;position:relative;overflow:hidden}
.glow{position:absolute;right:-120px;top:-160px;width:900px;height:900px;background:radial-gradient(closest-side,#0b2a66 0,transparent 70%);opacity:.55}
.l{position:absolute;left:64px;top:70px;width:600px}
.k{font:700 15px Menlo,monospace;letter-spacing:.14em;color:#3b82f6}
h1{font-size:68px;line-height:1.12;margin:16px 0 18px;letter-spacing:.01em}h1 em{font-style:normal;color:#3b82f6}
.t{font-size:23px;color:#a3aab5;line-height:1.55}
.st{display:flex;gap:34px;margin-top:30px}.st b{display:block;font-size:34px;color:#fff}.st span{color:#8b929c;font-size:14px}
.pg{margin-top:34px;width:520px}.pg .h{color:#8b929c;font-size:13px;margin-bottom:8px}
.L{display:flex;align-items:center;gap:12px;margin:5px 0;font-size:13px;color:#8b929c}.L span{width:150px}.L:after{content:"";flex:1;height:6px;border-radius:3px;background:#161b26;order:2;position:absolute}
.L{position:relative}.L b{display:block;height:6px;border-radius:3px;background:#3b82f6;position:relative;z-index:1;max-width:300px;margin-left:0}
.L{display:grid;grid-template-columns:150px 300px}.L:after{position:static;grid-column:2;grid-row:1}.L b{grid-column:2;grid-row:1}
.win{position:absolute;border:1px solid #2a3140;border-radius:12px;background:#08090c;overflow:hidden;box-shadow:0 24px 60px #000c}
.win .bar{height:30px;background:#12151c;display:flex;align-items:center;gap:7px;padding-left:12px;color:#6b7280;font-size:12px}
.win .bar i{width:9px;height:9px;border-radius:50%;background:#2a3140}
.win .v{position:relative;overflow:hidden}.win img{position:absolute;display:block}
.ph{position:absolute;width:210px;height:373px;border-radius:20px;border:2px solid #2a3140;overflow:hidden;background:#000;box-shadow:0 24px 60px #000c}
.ph img{width:100%;display:block}
.tag{position:absolute;font-size:13px;color:#7aa7f0;font-family:Menlo,monospace}
</style><body><div class="glow"></div>
<div class="l"><div class="k">DAILY-LLM</div>
<h1>从零学大模型<br><em>每一步都看得见</em></h1>
<div class="t">每个公式都能拖一拖、算一算。从一句话开始手算注意力，到 KV cache 与 FlashAttention，再配 TikTok 竖版图卡。</div>
<div class="st"><div><b>13</b><span>篇讲解</span></div><div><b>12</b><span>个互动页</span></div><div><b>73</b><span>张图卡</span></div><div><b>6 层</b><span>完整路线</span></div></div>
<div class="pg"><div class="h">路线进度（骨架 6 层，目前做完注意力）</div>${prog}</div></div>
<div class="win" style="left:700px;top:64px;width:540px;height:300px"><div class="bar"><i></i><i></i><i></i>&nbsp;互动页 · 点一行，看它在读谁</div><div class="v" style="height:270px"><img src="file://${demo}" style="width:760px;left:-135px;top:-782px"></div></div>
<div class="win" style="left:760px;top:392px;width:540px;height:262px"><div class="bar"><i></i><i></i><i></i>&nbsp;全流程：X → Q → S → A → O</div><div class="v" style="height:232px"><img src="file://${demo}" style="width:760px;left:-135px;top:-1500px"></div></div>
<div class="ph" style="left:1255px;top:48px;transform:rotate(-3deg)"><img src="${card("attn-04-softmax-scale", "04")}"></div>
<div class="ph" style="left:1345px;top:300px;transform:rotate(3deg)"><img src="${card("attn-06-position", "05")}"></div>
</body>`;
const hp = join(tmp, "banner.html");
writeFileSync(hp, html);
shot("file://" + hp, 1600, 720, join(assets, "banner.png"));
console.log("banner ok");
