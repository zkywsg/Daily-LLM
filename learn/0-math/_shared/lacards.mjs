// 线性代数图卡的公共部分。数字和图都用与文章、互动页同一份代码现场计算。
import { build, fig, row, table, panel, load, MiniGPT, TFV } from "../../3-llm/_shared/tfcards.mjs";
load("../../0-math/_shared/la-engine.js");
load("../../0-math/_shared/la-viz.js");
load("../tf-04-mini-gpt/demo/model.js");
export const { LA, LAV, MODEL } = globalThis.window;
export { build, fig, row, table, panel, MiniGPT, TFV };
// 多个图排成网格
export const grid = (parts, cols, gap = 24) => {
  const w = Math.max(...parts.map((p) => p.w)), h = Math.max(...parts.map((p) => p.h));
  return { svg: parts.map((p, i) => `<g transform="translate(${(i % cols) * (w + gap)},${Math.floor(i / cols) * (h + gap)})">${p.svg}</g>`).join(""), w: cols * w + (cols - 1) * gap, h: Math.ceil(parts.length / cols) * (h + gap) - gap };
};
export const bar = (label, pct, color, value) => `<div class="row" style="grid-template-columns:200px 1fr 190px;font-size:40px"><span>${label}</span><div class="bar"><i style="width:${pct}%;background:${color}"></i></div><span class="num">${value}</span></div>`;
