/**
 * minimap-renderer.ts — M767
 * World overview thumbnail in the corner
 */









const SPECIES_COLORS = ['#3F51B5','#FF6F00','#2E7D32','#C62828','#455A64','#7B1FA2','#1565C0'];

export interface MinimapConfig {
  width: number;
  height: number;
  x: number;  // position on screen
  y: number;
  opacity: number;
  showDensityHeatmap: boolean;
  showRigidBodies: boolean;
  showViewport: boolean;
}

export function defaultMinimapConfig(canvasW: number, canvasH: number): MinimapConfig {
  return {
    width: 150, height: 100,
    x: canvasW - 160, y: canvasH - 110,
    opacity: 0.8,
    showDensityHeatmap: true, showRigidBodies: true, showViewport: true,
  };
}

export function renderMinimap(
  ctx: CanvasRenderingContext2D,
  config: MinimapConfig,
  worldW: number, worldH: number,
  particles: { x: number; y: number; species: number }[],
  rigidBodies: { x: number; y: number; w: number; h: number; species: number; label: string }[],
  viewportRect?: { x: number; y: number; w: number; h: number },
): void {
  const { x, y, width, height, opacity } = config;
  const sx = width / worldW, sy = height / worldH;

  ctx.save();
  ctx.globalAlpha = opacity;

  // Background
  ctx.fillStyle = 'rgba(10, 14, 23, 0.9)';
  ctx.beginPath();
  ctx.roundRect(x - 2, y - 2, width + 4, height + 4, 4);
  ctx.fill();
  ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Density heatmap (scatter particles as tiny dots)
  if (config.showDensityHeatmap && particles.length > 0) {
    const step = Math.max(1, Math.floor(particles.length / 2000));
    for (let i = 0; i < particles.length; i += step) {
      const p = particles[i];
      const px = x + p.x * sx, py = y + p.y * sy;
      const c = SPECIES_COLORS[p.species % 7] || '#888';
      ctx.fillStyle = c;
      ctx.globalAlpha = 0.4;
      ctx.fillRect(px, py, 1, 1);
    }
  }

  ctx.globalAlpha = opacity;

  // Rigid bodies
  if (config.showRigidBodies) {
    for (const rb of rigidBodies) {
      const rx = x + (rb.x - rb.w / 2) * sx;
      const ry = y + (rb.y - rb.h / 2) * sy;
      const rw = rb.w * sx, rh = rb.h * sy;
      ctx.fillStyle = SPECIES_COLORS[rb.species % 7] || '#666';
      ctx.globalAlpha = 0.7;
      ctx.fillRect(rx, ry, Math.max(rw, 2), Math.max(rh, 2));
    }
  }

  ctx.globalAlpha = opacity;

  // Viewport indicator
  if (config.showViewport && viewportRect) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(
      x + viewportRect.x * sx, y + viewportRect.y * sy,
      viewportRect.w * sx, viewportRect.h * sy,
    );
  }

  // Label
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '8px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('minimap', x + 3, y + height - 3);

  ctx.restore();
}
