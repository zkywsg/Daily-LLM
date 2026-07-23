interface Props {
  onShuffle: () => void;
}

export function ShuffleButtonWidget({ onShuffle }: Props) {
  return (
    <button
      type="button"
      onClick={onShuffle}
      style={{ padding: "4px 12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)", marginBottom: "var(--space-3)" }}
    >
      打乱邻居顺序
    </button>
  );
}
