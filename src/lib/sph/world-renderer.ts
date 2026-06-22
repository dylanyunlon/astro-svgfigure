import { World, Particle, RigidBody } from './world';

export const SPECIES_COLORS: Record<number, string> = {
  0: '#3F51B5',
  1: '#FF6F00',
  2: '#2E7D32',
  3: '#C62828',
  4: '#455A64',
  5: '#7B1FA2',
  6: '#1565C0',
};

export interface RenderOptions {
  showTrails: boolean;
  showDensity: boolean;
  showVelocity: boolean;
  showGrid: boolean;
  showBoundaryParticles: boolean;
  showForces: boolean;
}

export const DEFAULT_OPTIONS: RenderOptions = {
  showTrails: false,
  showDensity: false,
  showVelocity: false,
  showGrid: true,
  showBoundaryParticles: false,
  showForces: false,
};

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number, cellSize = 40): void {
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= width; x += cellSize) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
  }
  for (let y = 0; y <= height; y += cellSize) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  ctx.restore();
}

function drawBoundaryGlow(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  ctx.save();
  const glowSize = 32;
  const sides = [
    { x: 0, y: 0, w: width, h: glowSize, dir: 'top' },
    { x: 0, y: height - glowSize, w: width, h: glowSize, dir: 'bottom' },
    { x: 0, y: 0, w: glowSize, h: height, dir: 'left' },
    { x: width - glowSize, y: 0, w: glowSize, h: height, dir: 'right' },
  ];
  for (const s of sides) {
    let grad: CanvasGradient;
    if (s.dir === 'top')         grad = ctx.createLinearGradient(0, 0, 0, glowSize);
    else if (s.dir === 'bottom') grad = ctx.createLinearGradient(0, s.y, 0, s.y + glowSize);
    else if (s.dir === 'left')   grad = ctx.createLinearGradient(0, 0, glowSize, 0);
    else                         grad = ctx.createLinearGradient(s.x, 0, s.x + glowSize, 0);
    const inner = s.dir === 'bottom' || s.dir === 'right';
    grad.addColorStop(inner ? 1 : 0, 'rgba(100,160,255,0.18)');
    grad.addColorStop(inner ? 0 : 1, 'rgba(100,160,255,0.00)');
    ctx.fillStyle = grad;
    ctx.fillRect(s.x, s.y, s.w, s.h);
  }
  ctx.strokeStyle = 'rgba(100,160,255,0.35)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(0.75, 0.75, width - 1.5, height - 1.5);
  ctx.restore();
}

function drawTrails(ctx: CanvasRenderingContext2D, particles: Particle[]): void {
  ctx.save();
  for (const p of particles) {
    if (!p.trail || p.trail.length < 2) continue;
    const color = SPECIES_COLORS[p.species] ?? '#ffffff';
    ctx.beginPath();
    ctx.moveTo(p.trail[0].x, p.trail[0].y);
    for (let i = 1; i < p.trail.length; i++) {
      ctx.lineTo(p.trail[i].x, p.trail[i].y);
    }
    ctx.strokeStyle = hexToRgba(color, 0.25);
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }
  ctx.restore();
}

function drawParticles(ctx: CanvasRenderingContext2D, particles: Particle[], opts: RenderOptions): void {
  ctx.save();
  for (const p of particles) {
    if (!opts.showBoundaryParticles && p.isBoundary) continue;
    const color = SPECIES_COLORS[p.species] ?? '#ffffff';
    const speed = Math.hypot(p.vx ?? 0, p.vy ?? 0);
    const radius = 3 + Math.min(speed * 0.4, 3);

    if (opts.showDensity && p.density != null) {
      const d = Math.min(p.density / 20, 1);
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = hexToRgba(color, d * 0.15);
      ctx.fill();
    }

    // Glow halo
    const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius * 2);
    grd.addColorStop(0, hexToRgba(color, 0.6));
    grd.addColorStop(1, hexToRgba(color, 0.0));
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius * 2, 0, Math.PI * 2);
    ctx.fillStyle = grd;
    ctx.fill();

    // Solid core
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    if (opts.showVelocity && (p.vx || p.vy)) {
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + (p.vx ?? 0) * 6, p.y + (p.vy ?? 0) * 6);
      ctx.strokeStyle = hexToRgba(color, 0.7);
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (opts.showForces && (p.fx || p.fy)) {
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + (p.fx ?? 0) * 0.05, p.y + (p.fy ?? 0) * 0.05);
      ctx.strokeStyle = 'rgba(255,220,50,0.6)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
  ctx.restore();
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawRigidBodies(ctx: CanvasRenderingContext2D, bodies: RigidBody[]): void {
  ctx.save();
  for (const b of bodies) {
    const hw = b.width / 2, hh = b.height / 2;
    ctx.save();
    ctx.translate(b.x, b.y);
    ctx.rotate(b.angle ?? 0);
    roundRect(ctx, -hw, -hh, b.width, b.height, 6);
    ctx.fillStyle = 'rgba(255,255,255,0.07)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.50)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    if (b.pinned) {
      ctx.beginPath();
      ctx.arc(0, 0, 5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,220,50,0.9)';
      ctx.fill();
    }
    if (b.label) {
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(b.label, 0, 0);
    }
    ctx.restore();
  }
  ctx.restore();
}

function drawHUD(ctx: CanvasRenderingContext2D, world: World): void {
  ctx.save();
  const pad = 10, lineH = 16;
  const lines = [
    `particles: ${world.particles?.length ?? 0}`,
    `bodies:    ${world.rigidBodies?.length ?? 0}`,
    `tick:      ${world.tick ?? 0}`,
  ];
  const boxW = 130, boxH = lines.length * lineH + 12;
  roundRect(ctx, pad, pad, boxW, boxH, 5);
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  ctx.fill();
  ctx.fillStyle = 'rgba(180,200,255,0.85)';
  ctx.font = '11px monospace';
  ctx.textBaseline = 'top';
  lines.forEach((l, i) => ctx.fillText(l, pad + 8, pad + 6 + i * lineH));
  ctx.restore();
}

export function renderWorld(
  ctx: CanvasRenderingContext2D,
  world: World,
  options: Partial<RenderOptions> = {}
): void {
  const opts: RenderOptions = { ...DEFAULT_OPTIONS, ...options };
  const { width, height } = ctx.canvas;

  // 1. Dark background
  ctx.fillStyle = '#0d0f1a';
  ctx.fillRect(0, 0, width, height);

  // 2. Grid
  if (opts.showGrid) drawGrid(ctx, width, height);

  // 3. Boundary glow + border
  drawBoundaryGlow(ctx, width, height);

  // 4. Trails
  if (opts.showTrails) drawTrails(ctx, world.particles ?? []);

  // 5. Particles: density halo → glow → core → velocity/force arrows
  drawParticles(ctx, world.particles ?? [], opts);

  // 6. Rigid bodies: rounded rect + pin dot + label
  drawRigidBodies(ctx, world.rigidBodies ?? []);

  // 7. HUD overlay
  drawHUD(ctx, world);
}
