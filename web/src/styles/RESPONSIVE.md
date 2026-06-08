# Daily-LLM · Web 响应式惯例

> Mobile-first。默认样式服务最小屏，用 `@media (min-width: ...)` 逐档放大。

## 5 个断点（与 tokens.css 一致）

| Token | 值 | 典型设备 |
|---|---|---|
| `--bp-sm` | 640px | 大手机横屏 |
| `--bp-md` | 768px | 平板竖屏 |
| `--bp-lg` | 1024px | 平板横屏 / 小笔记本 |
| `--bp-xl` | 1280px | 笔记本 |
| `--bp-2xl` | 1536px | 大屏桌面 |

## 4 条核心规则

### 1. Grid 双栏在 `< 1024px` 塌单列

所有“左 prose + 右可视化”双栏布局：默认 `grid-template-columns: 1fr`，达到 `--bp-lg` 才切 `1fr 1fr`。

```css
.grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--space-6);
}

@media (min-width: 1024px) {
  .grid {
    grid-template-columns: 1fr 1fr;
    gap: var(--space-8);
  }
}
```

### 2. Sticky 在 `< 1024px` 失效

右图 `position: sticky` 在窄屏没意义（单列流式）。只在 `--bp-lg+` 启用：

```css
.stickyPanel {
  position: static;
}

@media (min-width: 1024px) {
  .stickyPanel {
    position: sticky;
    top: var(--space-8);
  }
}
```

### 3. SVG 流体宽度

所有内联 `<svg>` 必须：

```jsx
<svg viewBox="0 0 W H" style={{ maxWidth: "100%", height: "auto" }} ...>
```

`viewBox` 锁定纵横比，`max-width: 100%` 跟随容器，`height: auto` 等比。**不要** 写 `width={W}` `height={H}` 作为 HTML 属性（除非容器明确想要固定渲染尺寸）。

### 4. 大字号用 clamp 流体

Hero 等大字号：`font-size: clamp(min, preferred, max)`。
例：`clamp(2rem, 5vw + 1rem, 4rem)` —— 手机 ~32px，桌面 ~64px，中间流畅过渡。

普通正文字号 `var(--fs-md)` / `--fs-base` 不需要 clamp。

## 容器 padding 惯例

- 默认 `padding: var(--space-6) var(--space-4)`
- `@media (min-width: 768px)` 升到 `var(--space-12) var(--space-8)`

## 验证

调整 viewport 至：375 / 768 / 1024 / 1440 四个宽度逐个看。Chrome DevTools 设备工具栏切换。
