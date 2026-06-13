/**
 * src/lib/renderer/index.ts
 *
 * Public API for the Nuke post-processing pipeline.
 *
 * Usage:
 * ```ts
 * import { Nuke, NukePass, BloomPass, DOFPass } from '$lib/renderer';
 *
 * const nuke = new Nuke(gl, width, height);
 * const bloom = new BloomPass(nuke, sceneRT, outputRT, { bloomStrength: 1.2 });
 * const dof   = new DOFPass(nuke, sceneRT, depthRT, outputRT, { focalZ: 0.5 });
 *
 * // Game loop:
 * nuke.render();
 * ```
 */

export { Nuke, NukeEvent } from './Nuke';
export type {
  RTOptions,
  PingPongPair,
  NukeListener,
  NukeEventType,
} from './Nuke';

export { NukePass, FULLSCREEN_VERT_SRC, UV_FROM_FRAG_COORD } from './NukePass';
export type { RenderTarget, UniformValue } from './NukePass';

export { BloomPass } from './passes/BloomPass';
export type { BloomPassConfig } from './passes/BloomPass';

export { DOFPass } from './passes/DOFPass';
export type { DOFPassConfig } from './passes/DOFPass';

// ── Nuke post-processing pipeline (xiaodi #32) ────────────────────────────────
export { Nuke, NukeEvent } from './Nuke';
export type { RTOptions, PingPongPair, NukeListener, NukeEventType } from './Nuke';

export { NukePass, FULLSCREEN_VERT_SRC, UV_FROM_FRAG_COORD } from './NukePass';
export type { RenderTarget, UniformValue } from './NukePass';

export { BloomPass } from './passes/BloomPass';
export type { BloomPassConfig } from './passes/BloomPass';

export { DOFPass } from './passes/DOFPass';
export type { DOFPassConfig } from './passes/DOFPass';
