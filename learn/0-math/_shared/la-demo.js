// 互动页共用的小工具：按钮组、可拖动的平面、2×2 矩阵输入。依赖 la-engine.js、la-viz.js。
(function (root) {
  const $ = (id) => document.getElementById(id);
  function chips(id, items, cur, cb) {
    const box = $(id); box.innerHTML = "";
    items.forEach(([k, name]) => {
      const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = name; b.dataset.k = k;
      b.setAttribute("aria-pressed", k === cur);
      b.onclick = () => { box.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", c === b)); cb(k); };
      box.appendChild(b);
    });
  }
  // 返回 draw(平面参数)；拖动带 drag 属性的点时调用 onDrag(名字, [x,y])
  function dragPlane(id, onDrag) {
    const el = $(id); let drag = null, opt = null;
    const draw = (o) => { opt = o; el.innerHTML = LAV.wrap(LAV.plane(Object.assign({ id }, o)), o.size || 300); };
    el.addEventListener("pointerdown", (e) => { const d = e.target.dataset && e.target.dataset.drag; if (d) { drag = d; el.setPointerCapture(e.pointerId); e.preventDefault(); } });
    el.addEventListener("pointermove", (e) => {
      if (!drag || !opt) return;
      const r = el.querySelector("svg").getBoundingClientRect(), S = opt.size || 300, T = opt.title ? 28 : 0, R = opt.range || 4, k = S / r.width;
      const clamp = (v) => Math.max(-R, Math.min(R, Math.round(v * 10) / 10));
      onDrag(drag, [clamp((((e.clientX - r.left) * k - S / 2) / (S / 2)) * R), clamp((-((e.clientY - r.top) * k - T - S / 2) / (S / 2)) * R)]);
    });
    el.addEventListener("pointerup", () => (drag = null));
    el.addEventListener("pointercancel", () => (drag = null));
    return draw;
  }
  // 2×2 矩阵输入框。返回 {get, set}
  function matInput(id, init, onChange) {
    const box = $(id);
    box.innerHTML = `<table class="mat">${[0, 1].map((i) => `<tr>${[0, 1].map((j) => `<td><input class="cell" type="number" step="0.1" id="${id}_${i}${j}" value="${init[i][j]}" aria-label="第 ${i + 1} 行第 ${j + 1} 列"></td>`).join("")}</tr>`).join("")}</table>`;
    const get = () => [0, 1].map((i) => [0, 1].map((j) => +$(`${id}_${i}${j}`).value || 0));
    box.querySelectorAll("input").forEach((x) => (x.oninput = () => onChange(get())));
    return { get, set: (M) => { [0, 1].forEach((i) => [0, 1].forEach((j) => ($(`${id}_${i}${j}`).value = +M[i][j].toFixed(3)))); onChange(get()); } };
  }
  const f = (v, n = 2) => { const s = v.toFixed(n); return /^-0\.?0*$/.test(s) ? s.slice(1) : s; };
  const vec = (v, n = 2) => "[" + v.map((x) => f(x, n)).join(", ") + "]";
  const mat = (M, n = 2) => "[" + M.map((r) => vec(r, n)).join(", ") + "]";
  root.D = { $, chips, dragPlane, matInput, f, vec, mat };
})(window);
