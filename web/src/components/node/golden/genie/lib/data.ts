// Genie demo 数据:帧 → 离散 token(tokenizer)+ 相邻帧推断离散动作(LAM)+
// 给定动作自回归生成下一帧(动态模型)。全部确定性 hash,不真跑训练。

export const NUM_ACTIONS = 8;
export const GRID_SIZE = 6;

/** 用简单参数化生成一个 toy "帧":以 (cx, cy) 为中心的圆形亮斑 */
export function makeFrame(cx: number, cy: number): number[] {
  return Array.from({ length: GRID_SIZE * GRID_SIZE }, (_, i) => {
    const x = i % GRID_SIZE, y = Math.floor(i / GRID_SIZE);
    const d = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
    return Math.max(0, 1 - d / 2.5);
  });
}

export const INITIAL_FRAME = { cx: 2.5, cy: 2.5 };

/** tokenizer:把帧(64 维)哈希成一个离散 token id(0-255) */
export function tokenizeFrame(frame: number[]): number {
  let h = 0;
  for (let i = 0; i < frame.length; i++) {
    h = (h * 31 + Math.round(frame[i] * 100)) >>> 0;
  }
  return h % 256;
}

/** LAM:给定相邻两帧的中心位移,无监督推断出的离散动作 id(0-7,8 个方向) */
export function inferLatentAction(prevPos: { cx: number; cy: number }, currPos: { cx: number; cy: number }): number {
  const dx = currPos.cx - prevPos.cx;
  const dy = currPos.cy - prevPos.cy;
  if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) return -1; // 无动作
  const angle = Math.atan2(dy, dx);
  const idx = Math.round(((angle + Math.PI) / (2 * Math.PI)) * NUM_ACTIONS) % NUM_ACTIONS;
  return idx;
}

export const ACTION_VECTORS: Array<{ dx: number; dy: number; label: string }> = Array.from({ length: NUM_ACTIONS }, (_, i) => {
  const angle = (i / NUM_ACTIONS) * 2 * Math.PI - Math.PI;
  return { dx: Math.cos(angle), dy: Math.sin(angle), label: `动作${i}` };
});

/** 动态模型:给定当前位置 + 选择的动作 id,自回归生成下一帧的位置(确定性物理规则模拟) */
export function dynamicsStep(pos: { cx: number; cy: number }, actionId: number): { cx: number; cy: number } {
  const v = ACTION_VECTORS[actionId];
  const nx = Math.max(0, Math.min(GRID_SIZE - 1, pos.cx + v.dx * 0.8));
  const ny = Math.max(0, Math.min(GRID_SIZE - 1, pos.cy + v.dy * 0.8));
  return { cx: nx, cy: ny };
}
