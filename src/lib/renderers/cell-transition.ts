export function animateTransition(
  targets: Map<string, { container: { x: number; y: number }; tx: number; ty: number }>,
  dur = 400
) {
  const t0 = performance.now();
  const snap = new Map<string, { sx: number; sy: number }>();
  targets.forEach((v, k) => snap.set(k, { sx: v.container.x, sy: v.container.y }));
  (function tick() {
    const p = Math.min((performance.now() - t0) / dur, 1);
    const e = p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2;
    targets.forEach((v, k) => {
      const s = snap.get(k)!;
      v.container.x = s.sx + (v.tx - s.sx) * e;
      v.container.y = s.sy + (v.ty - s.sy) * e;
    });
    if (p < 1) requestAnimationFrame(tick);
  })();
}
