/**
 * trails.ts
 * Standalone trail-position recorder extracted for the v2 step loop.
 *
 * Records each particle's current (x, y) into its trail ring-buffer and
 * prunes entries for particles that have been removed from the world.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any








export function updateTrails(world: any): void {
  const maxLen: number = world.config?.trailLength ?? 20;
  const particles: Array<{ id: number; x: number; y: number }> = world.particles;
  const trails: Map<number, Array<{ x: number; y: number }>> = world.trails;

  if (!particles || !trails) return;

  for (const p of particles) {
    let trail = trails.get(p.id);
    if (!trail) {
      trail = [];
      trails.set(p.id, trail);
    }
    trail.push({ x: p.x, y: p.y });
    if (trail.length > maxLen) trail.shift();
  }

  // Remove trails for deleted particles
  const alive = new Set(particles.map((p) => p.id));
  for (const id of trails.keys()) {
    if (!alive.has(id)) {
      trails.delete(id);
    }
  }
}
