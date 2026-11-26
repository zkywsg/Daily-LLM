// Transformer 系列图卡的公共部分：加载与文章、互动页同一份的实验代码，数字现场计算。
import { readFileSync } from "node:fs";
import { build } from "./cardkit.mjs";
const here = new URL(".", import.meta.url);
const load = (p) => (0, eval)(readFileSync(new URL(p, here), "utf8"));
load("./tf-engine.js"); load("./tf-viz.js"); load("./minigpt.js");
export const { TF, TFV, MiniGPT } = globalThis.window;
export { build, load };
const style = `<style>.tfc text{fill:#ededed;font:13px "PingFang SC",sans-serif}.tfc .vs{fill:#8a8a8a;font-size:12.5px}.tfc .vt{font-weight:700;font-size:15px}.tfc{display:block;margin:30px auto 0}
table.t{border-collapse:collapse;width:100%;margin-top:40px;font-size:36px}table.t td,table.t th{border:2px solid #242424;padding:18px 16px;text-align:right}table.t th{color:#8a8a8a;font-weight:500}table.t td:first-child,table.t th:first-child{text-align:left}table.t tr.on td{background:#0f1c33;color:#fff;font-weight:700}
.mono{font-family:ui-monospace,Menlo,monospace}</style>`;
export const fig = (r, width = 900) => `${style}<svg class="tfc" viewBox="0 0 ${r.w} ${r.h}" width="${width}" xmlns="http://www.w3.org/2000/svg">${r.svg}</svg>`;
export const row = (parts, gap = 30) => { let x = 0, h = 0, s = ""; for (const p of parts) { s += `<g transform="translate(${x},0)">${p.svg}</g>`; x += p.w + gap; h = Math.max(h, p.h); } return { svg: s, w: x - gap, h }; };
export const table = (head, rows, on = []) => `${style}<table class="t"><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>${rows.map((r, i) => `<tr class="${on.includes(i) ? "on" : ""}">${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</table>`;
export const panel = (html, size = 44) => `<div class="panel"><div class="big" style="font-size:${size}px">${html}</div></div>`;
