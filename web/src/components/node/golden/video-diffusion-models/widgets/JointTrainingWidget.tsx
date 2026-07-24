import { useState } from "react";
import { simulateLossCurve } from "../lib/data";

const W = 680;
const H = 260;

export function JointTrainingWidget() {
  const [imageRatio, setImageRatio] = useState(0.3);
  const curve = simulateLossCurve(imageRatio);
  const maxL = Math.max(...curve);

  const toX = (i: number) => 40 + (i / (curve.length - 1)) * (W - 80);
  const toY = (v: number) => H - 40 - Math.min((v / maxL) * (H - 80), H - 80);
  const path = curve.map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i)} ${toY(v)}`).join(" ");

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        图像数据占比 = {Math.round(imageRatio * 100)}%(0% = 纯视频 batch,100% = 纯图像 batch)
        <input type="range" min={0} max={100} value={Math.round(imageRatio * 100)} onChange={(e) => setImageRatio(Number(e.target.value) / 100)} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`图像占比 ${Math.round(imageRatio * 100)}% 时的训练 loss 曲线示意`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          训练 loss 曲线示意(混合比例影响抖动幅度)
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} />
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        纯视频数据量小(标注/采集成本高),loss 曲线抖动大;混入大规模图像数据后曲线更平滑 —— 图像/视频联合训练复用了图像领域的数据规模优势。
      </p>
    </div>
  );
}
