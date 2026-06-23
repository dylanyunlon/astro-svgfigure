import { Graphics }  from '../../../upstream/pixijs-engine/src/scene/graphics/shared/Graphics';
import type { Container } from '../../../upstream/pixijs-engine/src/scene/container/Container';

export function createDebugOverlay(
  stage: Container,
  cells: { x: number; y: number; width: number; height: number }[],
) {
  const g = new Graphics();
  g.lineStyle(1, 0xFF0000, 0.3);
  cells.forEach((c) => g.drawRect(c.x, c.y, c.width, c.height));
  g.visible = false;
  stage.addChild(g);
  return {
    toggle() {
      g.visible = !g.visible;
    },
  };
}
