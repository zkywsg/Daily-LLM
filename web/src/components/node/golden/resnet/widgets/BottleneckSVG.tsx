import { BASIC_BLOCK_PARAMS, BOTTLENECK_PARAMS } from "../lib/curves";

interface Props {
  blockType: "basic" | "bottleneck";
  width?: number;
  height?: number;
}

export function BottleneckSVG({ blockType, width = 560, height = 360 }: Props) {
  const isBasic = blockType === "basic";
  const params = isBasic ? BASIC_BLOCK_PARAMS : BOTTLENECK_PARAMS;

  const layers = isBasic
    ? [
        { label: "Conv 3×3", sub: "64", x: 100 },
        { label: "ReLU", sub: "", x: 200 },
        { label: "Conv 3×3", sub: "64", x: 290 },
      ]
    : [
        { label: "Conv 1×1 ↓", sub: "64", x: 90 },
        { label: "ReLU", sub: "", x: 180 },
        { label: "Conv 3×3", sub: "64", x: 240 },
        { label: "ReLU", sub: "", x: 330 },
        { label: "Conv 1×1 ↑", sub: "256", x: 380 },
      ];

  const inputX = 20;
  const addX = isBasic ? 380 : 470;
  const outputX = isBasic ? 460 : 550;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ maxWidth: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label={`${blockType === "basic" ? "BasicBlock" : "Bottleneck"} 残差块结构`}
    >
      <text
        x={width / 2}
        y={30}
        textAnchor="middle"
        fontSize={16}
        fontWeight={600}
        fill="var(--ink-primary)"
      >
        {isBasic ? "BasicBlock" : "Bottleneck"}
      </text>

      <rect x={inputX} y={180} width={50} height={36} rx={6}
        fill="#fef3c7" stroke="#d97706" strokeWidth={1.5} />
      <text x={inputX + 25} y={203} textAnchor="middle" fontSize={14} fill="#92400e">
        x
      </text>

      {layers.map((layer, i) => (
        <g key={i}>
          <rect x={layer.x} y={180} width={70} height={36} rx={6}
            fill="#fce7f3" stroke="#db2777" strokeWidth={1.5} />
          <text
            x={layer.x + 35}
            y={layer.sub ? 197 : 203}
            textAnchor="middle"
            fontSize={12}
            fill="#9d174d"
          >
            {layer.label}
          </text>
          {layer.sub && (
            <text x={layer.x + 35} y={210} textAnchor="middle" fontSize={10} fill="#9d174d">
              {layer.sub}
            </text>
          )}
        </g>
      ))}

      <circle cx={addX} cy={198} r={16}
        fill="#fef3c7" stroke="#d97706" strokeWidth={2} />
      <text x={addX} y={204} textAnchor="middle" fontSize={18} fill="#92400e" fontWeight={700}>
        ⊕
      </text>

      <rect x={outputX} y={180} width={50} height={36} rx={6}
        fill="#fef3c7" stroke="#d97706" strokeWidth={1.5} />
      <text x={outputX + 25} y={203} textAnchor="middle" fontSize={14} fill="#92400e">
        y
      </text>

      <path
        d={`M ${inputX + 25} 180 C ${inputX + 25} 80, ${addX} 80, ${addX} 184`}
        stroke="#2563eb"
        strokeWidth={3}
        fill="none"
      />
      <text
        x={(inputX + addX) / 2}
        y={75}
        textAnchor="middle"
        fontSize={13}
        fontStyle="italic"
        fill="#1e40af"
      >
        identity
      </text>

      <text
        x={addX + 24}
        y={170}
        textAnchor="start"
        fontSize={14}
        fontStyle="italic"
        fill="var(--ink-primary)"
      >
        F(x) + x
      </text>

      <g transform={`translate(${width / 2}, 285)`}>
        <text textAnchor="middle" fontSize={13} fill="var(--ink-secondary)">
          参数量
        </text>
        <text y={26} textAnchor="middle" fontSize={22} fontWeight={600} fill="var(--ink-primary)">
          {params.toLocaleString()}
        </text>
      </g>
    </svg>
  );
}
