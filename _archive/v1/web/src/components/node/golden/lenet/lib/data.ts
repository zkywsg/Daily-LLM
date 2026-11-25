// LeNet-5 主干各层的形状变化(C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> output)
export interface LenetLayer {
  label: string;
  size: string; // 空间尺寸,如 "28×28"
  channels: number; // 通道数 / 特征图数
  kind: "input" | "conv" | "pool" | "fc" | "output";
  note?: string;
}

export const LENET_LAYERS: LenetLayer[] = [
  { label: "input", size: "32×32", channels: 1, kind: "input", note: "MNIST 28×28 zero-pad 到 32×32" },
  { label: "C1", size: "28×28", channels: 6, kind: "conv", note: "5×5 卷积,tanh" },
  { label: "S2", size: "14×14", channels: 6, kind: "pool", note: "2×2 平均池化" },
  { label: "C3", size: "10×10", channels: 16, kind: "conv", note: "5×5 卷积,部分连接表,tanh" },
  { label: "S4", size: "5×5", channels: 16, kind: "pool", note: "2×2 平均池化" },
  { label: "C5", size: "1×1", channels: 120, kind: "conv", note: "5×5 卷积,空间维度压成 1×1" },
  { label: "F6", size: "—", channels: 84, kind: "fc", note: "全连接,tanh" },
  { label: "output", size: "—", channels: 10, kind: "output", note: "原版 RBF,今天用 softmax" },
];

// 参数量对比:LeNet 整网 vs 同输入接 MLP 第一层
export interface ParamCompareRow {
  label: string;
  params: number; // 参数量(个)
  highlight?: boolean;
}

export const PARAM_COMPARE: ParamCompareRow[] = [
  { label: "LeNet-5 整网(卷积+池化+FC)", params: 60_000, highlight: true },
  { label: "MLP 仅第一层(32×32 → 1024 维隐层)", params: 1_049_600 },
];

// 局部连接 vs 全连接的参数量随隐层宽度变化(用于交互 diagram)
export interface ConnectivityPoint {
  hiddenUnits: number;
  mlpParams: number; // 1024 输入像素 × hiddenUnits
  convParams: number; // 固定:5×5 卷积核,与隐层宽度(通道数)近似线性但系数极小
}

export const CONNECTIVITY_GROWTH: ConnectivityPoint[] = [
  { hiddenUnits: 6, mlpParams: 1024 * 6, convParams: 5 * 5 * 6 },
  { hiddenUnits: 16, mlpParams: 1024 * 16, convParams: 5 * 5 * 16 },
  { hiddenUnits: 64, mlpParams: 1024 * 64, convParams: 5 * 5 * 64 },
  { hiddenUnits: 256, mlpParams: 1024 * 256, convParams: 5 * 5 * 256 },
  { hiddenUnits: 1024, mlpParams: 1024 * 1024, convParams: 5 * 5 * 1024 },
];

// 感受野随层数增长(相对 28×28 输入 C1 特征图而言,近似值,用于示意)
export interface ReceptiveFieldStep {
  label: string;
  fieldSize: number; // 感受野边长(像素,近似)
}

export const RECEPTIVE_FIELD_GROWTH: ReceptiveFieldStep[] = [
  { label: "C1", fieldSize: 5 },
  { label: "S2", fieldSize: 6 },
  { label: "C3", fieldSize: 14 },
  { label: "S4", fieldSize: 16 },
  { label: "C5", fieldSize: 32 },
];

// MNIST 错误率(论文报告)
export const MNIST_ERROR_RATE = 0.95; // %
export const LENET_TOTAL_PARAMS = 60_000;
