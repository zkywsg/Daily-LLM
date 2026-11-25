import { useEffect, useRef, useState } from "react";

interface MermaidBlockProps {
  code: string;
}

let mermaidPromise: Promise<typeof import("mermaid").default> | null = null;

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import("mermaid").then((m) => {
      m.default.initialize({
        startOnLoad: false,
        theme: "neutral",
        fontFamily: "var(--font-sans)",
      });
      return m.default;
    });
  }
  return mermaidPromise;
}

export function MermaidBlock({ code }: MermaidBlockProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadMermaid().then(async (mermaid) => {
      if (cancelled || !ref.current) return;
      try {
        const id = `mermaid-${Math.random().toString(36).slice(2)}`;
        const { svg } = await mermaid.render(id, code);
        if (!cancelled && ref.current) {
          ref.current.innerHTML = svg;
          // 桌面端窄于容器时让 svg 撑满容器(原 vbWidth 比例); 大图(超出容器)走横滚保原始可读尺寸
          const svgEl = ref.current.querySelector("svg");
          const vbWidth = svgEl?.viewBox?.baseVal?.width;
          if (svgEl && vbWidth) {
            svgEl.style.width = "auto";
            // 容器宽度不可用时退回 vbWidth
            const containerW = ref.current.clientWidth || vbWidth;
            // < 容器:不强制 px,让 svg 按 viewBox aspectRatio 撑满容器(min 100% 居中)
            // > 容器:固定 px,触发容器 overflow-x:auto 横滚
            if (vbWidth <= containerW) {
              svgEl.style.maxWidth = "100%";
              svgEl.style.height = "auto";
            } else {
              svgEl.style.width = `${Math.ceil(vbWidth)}px`;
              svgEl.style.maxWidth = "none";
            }
            svgEl.style.display = "block";
            svgEl.style.margin = "0 auto";
          }
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    });
    return () => {
      cancelled = true;
    };
  }, [code]);

  if (error) {
    return (
      <pre style={{ color: "var(--accent-warn)", padding: "var(--space-4)" }}>
        Mermaid render error: {error}
        {"\n\n"}
        {code}
      </pre>
    );
  }
  return (
    <div
      ref={ref}
      style={{
        margin: "var(--space-6) 0",
        overflowX: "auto",
        WebkitOverflowScrolling: "touch",
      }}
    />
  );
}
