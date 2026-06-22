import { wrap } from '../../../upstream/comlink/src/comlink';
import type { SPHWorkerAPI } from './sph-worker';

export interface SPHFrameSnapshot {
  particles: Float64Array;
  particleCount: number;
  rigidBodies: Array<{
    id: string;
    x: number;
    y: number;
    angle: number;
    w: number;
    h: number;
    species: number;
    pinned: boolean;
  }>;
  stats: {
    particleCount: number;
    avgDensity: number;
    maxVelocity: number;
    kineticEnergy: number;
  };
  contacts: Array<{
    x: number;
    y: number;
    nx: number;
    ny: number;
    depth: number;
  }>;
}

let proxy: ReturnType<typeof wrap<SPHWorkerAPI>> | null = null;
let worker: Worker | null = null;

function getProxy(): ReturnType<typeof wrap<SPHWorkerAPI>> {
  if (!proxy) {
    worker = new Worker(new URL('./sph-worker.ts', import.meta.url), { type: 'module' });
    proxy = wrap<SPHWorkerAPI>(worker);
  }
  return proxy;
}

export async function initSPHWorld(
  width: number,
  height: number,
  qosProfile: string
): Promise<void> {
  const p = getProxy();
  await p.initSPHWorld(width, height, qosProfile);
}

export async function addFluid(
  x: number,
  y: number,
  w: number,
  h: number,
  spacing: number,
  species: number
): Promise<void> {
  const p = getProxy();
  await p.addFluid(x, y, w, h, spacing, species);
}

export async function addBody(
  id: string,
  x: number,
  y: number,
  w: number,
  h: number,
  species: number,
  pinned: boolean
): Promise<void> {
  const p = getProxy();
  await p.addBody(id, x, y, w, h, species, pinned);
}

export async function stepSPH(): Promise<SPHFrameSnapshot> {
  const p = getProxy();
  return await p.stepSPH() as SPHFrameSnapshot;
}

export async function setQoS(profileName: string): Promise<void> {
  const p = getProxy();
  await p.setQoS(profileName);
}

export async function raycast(
  ox: number,
  oy: number,
  dx: number,
  dy: number,
  maxDist: number
): Promise<any> {
  const p = getProxy();
  return await p.raycast(ox, oy, dx, dy, maxDist);
}

export function terminateSPHWorker(): void {
  if (proxy) {
    // Release the Comlink proxy before terminating the underlying worker
    (proxy as any)[Symbol.dispose]?.();
    proxy = null;
  }
  if (worker) {
    worker.terminate();
    worker = null;
  }
}
