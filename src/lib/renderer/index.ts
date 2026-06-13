/**
 * src/lib/renderer/index.ts
 *
 * Public API for the composite render-to-texture system.
 *
 *   Scene            — base scene graph (children, add/remove, render traversal)
 *   FXScene          — offscreen RT-backed scene (manualRender, onCreate, texture)
 *   FXSceneCompositor— blits multiple FXScene textures onto the canvas
 *   RenderTarget     — raw FBO + texture attachment (single or MRT)
 *
 * AstroRenderer WebGL layer (xiaodi #31):
 *   AstroRenderer    — core WebGL2/1 renderer singleton
 *   AstroProgram     — shader program + uniform/attrib location cache
 *   AstroMesh        — VAO/VBO geometry wrapper
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

export { AstroRenderer, WEBGL1, WEBGL2, SHADOWS_LOW, SHADOWS_MED, SHADOWS_HIGH } from './AstroRenderer.js';
export type { WebGLVersion, ShadowQuality, AstroExtensions, Viewport, RendererOptions } from './AstroRenderer.js';

export { AstroProgram } from './AstroProgram.js';

export { AstroMesh } from './AstroMesh.js';
export type { DrawMode } from './AstroMesh.js';
