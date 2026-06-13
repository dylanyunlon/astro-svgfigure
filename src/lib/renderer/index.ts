/**
 * src/lib/renderer/index.ts
 *
 * Public API for the composite render-to-texture system.
 *
 *   Scene            — base scene graph (children, add/remove, render traversal)
 *   FXScene          — offscreen RT-backed scene (manualRender, onCreate, texture)
 *   FXSceneCompositor— blits multiple FXScene textures onto the canvas
 *   RenderTarget     — raw FBO + texture attachment (single or MRT)
 */

export { Scene } from './Scene';
export type { Renderable, Camera, SceneOptions } from './Scene';
export { makeOrthoCameraForSize } from './Scene';

export { FXScene, FXSceneCompositor } from './FXScene';
export type {
  FXSceneOptions,
  CompositeLayer,
  OnCreateFn,
  OnRenderFn,
} from './FXScene';

export { RenderTarget } from './RenderTarget';
export type { RenderTargetOptions } from './RenderTarget';
