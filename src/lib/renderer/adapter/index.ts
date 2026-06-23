/**
 * adapter/index.ts — Renderer ↔ Program ↔ Material adapter 层入口
 *
 * 三个核心模块:
 *   RendererAdapter  统一 facade (beginFrame → useMaterial → draw → endFrame)
 *   ProgramFactory   ref-counted shader program 缓存池
 *   MaterialBinder   dirty-tracked GL state 绑定
 *
 * Usage:
 *   import {
 *     RendererAdapter,
 *     ProgramFactory,
 *     MaterialBinder,
 *   } from '$lib/renderer/adapter';
 */

export { RendererAdapter, RenderPassScope } from './RendererAdapter.js';
export type { DrawContext, ProgramHandle } from './RendererAdapter.js';

export { ProgramFactory } from './ProgramFactory.js';

export { MaterialBinder } from './MaterialBinder.js';
