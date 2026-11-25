import { useState } from "react";
import { extractSemanticToken } from "../lib/data";

export function SemanticTokenWidget() {
  const [framePos, setFramePos] = useState(0);
  const token = extractSemanticToken(framePos);

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        音频帧位置(模拟不同时刻的音频内容)
        <input type="range" min={0} max={5} step={0.5} value={framePos} onChange={(e) => setFramePos(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <div style={{ padding: "var(--space-4)", border: "1px solid #fb7185", borderRadius: "var(--radius-md)", background: "#fff1f2" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>语义 token id</div>
        <div style={{ fontSize: "var(--fs-2xl)", fontWeight: 700, color: "#9d174d" }}>{token}</div>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        语义 token 来自自监督音频模型(w2v-BERT)的中间层表征聚类离散化,采样率较低(每个 token 覆盖更长时间跨度),携带内容和说话人身份等长程结构信息。
      </p>
    </div>
  );
}
