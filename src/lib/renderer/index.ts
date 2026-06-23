/**
 * src/lib/renderer/index.ts
 *
 * Barrel export for the renderer adapter layer.
 * 100% coverage of all renderer modules.
 */

// ── AstroRenderer ─────────────────────────────────────────────────────────────
export { AstroRenderer, WEBGL1, WEBGL2, SHADOWS_LOW, SHADOWS_MED, SHADOWS_HIGH } from './AstroRenderer';
export type { WebGLVersion, ShadowQuality, AstroExtensions, Viewport, RendererOptions } from './AstroRenderer';

// ── AstroProgram ──────────────────────────────────────────────────────────────
export { AstroProgram } from './AstroProgram';

// ── AstroMesh ─────────────────────────────────────────────────────────────────
export { AstroMesh } from './AstroMesh';
export type { DrawMode } from './AstroMesh';

// ── Scene ─────────────────────────────────────────────────────────────────────
export { Scene, makeOrthoCameraForSize } from './Scene';
export type { Renderable, Camera, SceneOptions } from './Scene';

// ── FXScene ───────────────────────────────────────────────────────────────────
export { FXScene, FXSceneCompositor } from './FXScene';
export type { OnCreateFn, OnRenderFn, FXSceneOptions, CompositeLayer } from './FXScene';

// ── RenderTarget ──────────────────────────────────────────────────────────────
export { RenderTarget } from './RenderTarget';
export type { RenderTargetOptions } from './RenderTarget';

// ── InstancedMesh ─────────────────────────────────────────────────────────────
export { InstancedMesh, INSTANCED_VERT, INSTANCED_FRAG } from './InstancedMesh';
export type { InstanceData } from './InstancedMesh';

// ── CellInstanceManager ──────────────────────────────────────────────────────
export { CellInstanceManager } from './CellInstanceManager';
export type { CellBBox, CellParamsJson, SpeciesGroup } from './CellInstanceManager';

// ── UniformBuffer ─────────────────────────────────────────────────────────────
export { UniformBuffer, CELL_UBO_LAYOUT, NUKE_UBO_LAYOUT, VIEW_UBO_LAYOUT, createCellUBO, createNukeUBO, createViewUBO } from './UniformBuffer';
export type { UBOFieldType, UBOFieldDef, UBOLayoutDef } from './UniformBuffer';

// ── Nuke post-processing pipeline ─────────────────────────────────────────────
export { Nuke, NukeEvent } from './Nuke';
export type { RTOptions, PingPongPair, NukeListener, NukeEventType } from './Nuke';

export { NukePass, FULLSCREEN_VERT_SRC, UV_FROM_FRAG_COORD } from './NukePass';
export type { RenderTarget as NukeRenderTarget, UniformValue } from './NukePass';

// ── OcclusionQuery ────────────────────────────────────────────────────────────
export { OcclusionQueryManager, patchCellInstanceManagerWithOcclusion } from './OcclusionQuery';
export type { OcclusionQueryOptions, VisibilityMap } from './OcclusionQuery';

// ── Passes ────────────────────────────────────────────────────────────────────
export { BloomPass } from './passes/BloomPass';
export type { BloomPassConfig } from './passes/BloomPass';

export { DOFPass } from './passes/DOFPass';
export type { DOFPassConfig } from './passes/DOFPass';

export { KawaseBlurPass } from './passes/KawaseBlurPass';
export type { KawaseBlurPassOptions } from './passes/KawaseBlurPass';

export { KawaseBloomPass } from './passes/KawaseBloomPass';
export type { KawaseBloomPassOptions } from './passes/KawaseBloomPass';

// ── Geometry ──────────────────────────────────────────────────────────────────
export * from './geometry';

// ── Material ──────────────────────────────────────────────────────────────────
export * from './material';

// ── Adapter layer — Renderer ↔ Program ↔ Material bridge (M810) ─────────────
export { RendererAdapter, RenderPassScope } from './adapter/RendererAdapter';
export type { DrawContext, ProgramHandle } from './adapter/RendererAdapter';

export { ProgramFactory } from './adapter/ProgramFactory';

export { MaterialBinder } from './adapter/MaterialBinder';
