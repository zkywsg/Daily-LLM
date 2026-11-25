// MusicGen demo 数据:EnCodec 多层并行码本可视化 + 延迟交错模式(delay pattern)+
// 文本/旋律双重条件开关。全部确定性构造。

export const NUM_LEVELS = 4;
export const NUM_TIMESTEPS = 6;

/** 给定时间步 t 和码本层 level,生成一个确定性的码本 id(0-7),模拟 EnCodec
 * 每个时间步 NUM_LEVELS 个并行码本各自的取值。 */
export function encodecCode(t: number, level: number): number {
  let h = (t + 1) * 131 + (level + 1) * 977;
  h = h >>> 0;
  return h % 8;
}

export interface DelayCell {
  level: number;
  step: number; // 该帧位置对应的原始时间步,-1 表示 padding(尚未到达/已经结束)
  filled: boolean;
}

/** 延迟交错模式(delay pattern):第 level 层延迟 level 步开始,总长度
 * S = numTimesteps + numLevels - 1 帧,让单个自回归 Transformer 能按固定顺序
 * 逐帧预测所有层的 token,而不需要为每层单独训练模型或加阶段。 */
export function buildDelayPattern(numTimesteps: number, numLevels: number = NUM_LEVELS): DelayCell[][] {
  const S = numTimesteps + numLevels - 1;
  const rows: DelayCell[][] = [];
  for (let level = 0; level < numLevels; level++) {
    const row: DelayCell[] = [];
    for (let s = 0; s < S; s++) {
      const step = s - level;
      row.push({ level, step, filled: step >= 0 && step < numTimesteps });
    }
    rows.push(row);
  }
  return rows;
}

export type ConditionMode = "none" | "text" | "melody" | "both";

export const CONDITION_LABELS: Record<ConditionMode, string> = {
  none: "无条件(纯续写)",
  text: "仅文本条件",
  melody: "仅旋律条件",
  both: "文本 + 旋律双重条件",
};
