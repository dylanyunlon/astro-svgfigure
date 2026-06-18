import { Graphics } from "pixi.js";

export function createSelectionRing(w: number, h: number, color: number) {
  const g = new Graphics();
  g.lineStyle(2.5, color, 0.85);
  g.drawRoundedRect(-5, -5, w + 10, h + 10, 10);
  g.visible = false;
  return g;
}

export function showRing(g: Graphics) {
  g.visible = true;
}

export function hideRing(g: Graphics) {
  g.visible = false;
}
