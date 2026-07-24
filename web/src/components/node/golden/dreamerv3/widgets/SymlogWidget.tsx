import { DOMAIN_REWARDS, symlog } from "../lib/data";

const W = 680;
const H = 300;

export function SymlogWidget() {
  const maxRaw = Math.max(...DOMAIN_REWARDS.map((d) => d.raw));
  const maxSymlog = Math.max(...DOMAIN_REWARDS.map((d) => symlog(d.raw)));

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="不同领域原始 reward 与 symlog 变换后的对比">
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          跨领域 reward 量级 vs symlog 归一化后
        </text>
        {DOMAIN_REWARDS.map((d, i) => {
          const y = 50 + i * 60;
          const rawW = Math.min((d.raw / maxRaw) * 200, 200);
          const symW = Math.min((symlog(d.raw) / maxSymlog) * 200, 200);
          return (
            <g key={d.domain}>
              <text x={10} y={y + 4} fontSize={11} fill="var(--ink-secondary)">{d.domain}</text>
              <rect x={220} y={y - 8} width={rawW} height={7} fill="#9ca3af" />
              <text x={220 + rawW + 4} y={y - 2} fontSize={9} fill="var(--ink-muted)">raw={d.raw}</text>
              <rect x={220} y={y + 3} width={symW} height={7} fill="#d946ef" />
              <text x={220 + symW + 4} y={y + 9} fontSize={9} fill="var(--ink-muted)">symlog={symlog(d.raw).toFixed(2)}</text>
            </g>
          );
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        灰条 = 原始 reward(跨越 4 个数量级);粉条 = symlog 变换后,全部被压缩到相近的可比范围 —— 同一套超参数因此能跨领域通用,不需要为每个领域单独调 reward scale。
      </p>
    </div>
  );
}
