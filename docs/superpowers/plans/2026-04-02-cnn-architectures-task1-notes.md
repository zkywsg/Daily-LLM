# CNN Architectures Task 1 Execution Notes

Scope-lock evidence for Task 1 of `docs/superpowers/plans/2026-04-02-cnn-architectures.md`.

## Re-opened References

- Approved spec: `docs/superpowers/specs/2026-04-02-cnn-architectures-design.md`
- Style guide: `STYLE.md`

Constraints locked in from those reads:

- Keep the chapter as a problem-driven rewrite, not a new CNN survey.
- Preserve the required README structure from `STYLE.md`.
- Use the approved section flow: problem origin, learning goals, intuition, mechanism, architecture evolution, engineering traps, evolution note.
- Keep the ending as a bridge to attention / ViT rather than a full later-CNN expansion.
- Keep `你要记住` limited and aligned with the approved conclusions.

## Snapshot of Current Files

- `01-Visual-Intelligence/cnn-architectures/README.md`
  - Current state: existing Chinese chapter is the main rewrite target.
  - Will change: title framing, opening problem statement, learning goals, mechanism order, architecture evolution, engineering traps, evolution note.
  - May stay untouched: any appendix or unrelated navigation if already aligned.
- `01-Visual-Intelligence/cnn-architectures/README_EN.md`
  - Current state: existing English chapter mirrors the old structure.
  - Will change: section order and narrative to match the approved Chinese backbone.
  - May stay untouched: any text already consistent with the new structure.
- `01-Visual-Intelligence/README.md`
  - Current state: Phase 01 overview entry for CNN architecture.
  - Will change only if the one-line summary drifts from the rewritten chapter positioning.
  - May stay untouched: if the current summary still matches the new chapter scope.

## Temporary Rewrite Checklist

1. Open with image spatial structure, not benchmark-first framing.
2. Replace learning goals with the approved three questions.
3. Rebuild section 2 as convolution and feature maps, receptive field and downsampling, residual connections, and local-to-global limitation.
4. Rewrite architecture evolution as bottleneck -> response.
5. Keep only minimal convolution, conv block, and residual block code.
6. End with CNN strengths and boundary, then the attention / ViT bridge.
7. Mirror the same structure in `README_EN.md`.
8. Touch `01-Visual-Intelligence/README.md` only if summary drift exists.
