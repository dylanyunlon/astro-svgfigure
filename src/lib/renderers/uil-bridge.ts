/**
 * uil-bridge.ts — UIL JSON → PixiJS position / uniform / watch bridge
 *
 * Bridges upstream/uil (lo-th/uil) JSON parameter data into the PixiJS
 * cell rendering pipeline.  The AT (ActiveTheory) UIL parameter format
 * uses hierarchical keys like:
 *
 *   MESH_Element_{N}_{scene}position   → [x, y, z]
 *   MESH_Element_{N}_{scene}scale      → [sx, sy, sz]
 *   MESH_Element_{N}_{scene}rotation   → [rx, ry, rz]
 *   ShaderClass/ShaderClass/Element_{N}_{scene}/uParamName → value
 *
 * This module reads those keys, maps them to PixiJS Container.position /
 * Container.scale and shader Filter uniforms, and provides a polling watcher
 * that re-applies changes on each tick — the cell pub/sub core loop.
 *
 * Three entry points:
 *
 *   applyUILToStage(stage, uilJson, containerMap)
 *     Reads MESH_Element entries → sets Container.position / Container.scale
 *     for every matched cell container.
 *
 *   applyUILToShader(containerMap, uilJson)
 *     Reads ShaderClass/.../uXxx entries → sets uniform values on
 *     CilEyeSDFFilter / CilBoltSDFFilter / etc. attached to containers.
 *
 *   watchUILChanges(url, stage, containerMap, opts?)
 *     Polls a UIL JSON endpoint at a configurable interval, diffs against the
 *     previous snapshot, and re-applies only changed entries.  Returns a stop
 *     handle.  This is the cell pub/sub heartbeat — every poll cycle is a
 *     micro-epoch that pushes AT UIL parameter mutations into the live
 *     PixiJS scene graph.
 *
 * Upstream references:
 *   upstream/uil/src/core/Gui.js               — UIL parameter model
 *   upstream/uil/src/proto/Numeric.js           — numeric slider → value
 *   upstream/uil/src/proto/Slide.js             — slide range → value
 *   channels/physics/at_uil_categorized.json    — categorised AT UIL dump
 *   channels/physics/scene_mesh_at_params.json  — MESH_Element / shader params
 *   src/lib/renderer/material/CellMaterial.ts   — species ↔ shader class map
 *   src/lib/renderer/material/Material.ts       — importFromATParams()
 *   src/lib/renderers/pixi-cell-renderer.ts     — Container.__*Filter slots
 *   src/lib/renderers/sdf-species-filter.ts     — CilEye/Bolt/Vector/Plus/ArrowRight SDFFilter
 */

import type { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';

// ── Types ──────────────────────────────────────────────────────────────────

/**
 * A flat UIL JSON parameter object.
 * Keys follow AT conventions:
 *   - MESH_Element_{N}_{scene}{prop}   → mesh transform (position / scale / rotation)
 *   - {ShaderClass}/{ShaderClass}/Element_{N}_{scene}/{uParam} → shader uniform
 *   - INPUT_Config_{N}_{scene}_{prop}  → config flags (ignored by this bridge)
 */
export type UILJson = Record<string, unknown>;

/**
 * Parsed MESH_Element entry — represents one scene-graph node's transform.
 */
export interface UILMeshEntry {
  /** Element index from the key (e.g. 0 in MESH_Element_0_Home) */
  index: number;
  /** Scene/group name (e.g. "Home", "CleanRoom", "About") */
  scene: string;
  /** Transform property: position | scale | rotation */
  prop: 'position' | 'scale' | 'rotation';
  /** The [x, y, z] value array */
  value: [number, number, number];
}

/**
 * Parsed shader uniform entry.
 */
export interface UILShaderUniform {
  /** Shader class name (e.g. "GlassCubeShader", "FloorShader") */
  shaderClass: string;
  /** Element_N_scene identifier */
  elementKey: string;
  /** Uniform name (e.g. "uAlpha", "uTint", "uColor0") */
  uniformName: string;
  /** Raw value — number, number[], string (hex colour), boolean */
  value: unknown;
}

/**
 * Options for watchUILChanges.
 */
export interface UILWatchOptions {
  /** Polling interval in ms.  Default 500. */
  intervalMs?: number;
  /** Called after each successful apply cycle with the number of changed keys. */
  onApply?: (changedCount: number) => void;
  /** Called on fetch/parse errors. */
  onError?: (error: unknown) => void;
  /** If true, also applies shader uniforms (applyUILToShader).  Default true. */
  applyShaders?: boolean;
}

/**
 * Container map — cell_id → PixiJS Container.
 * Same structure used by pixi-cell-renderer's LiveCell map and
 * EpochCellBridge.registerContainer().
 */
export type ContainerMap = Map<string, Container>;

// ── Constants ──────────────────────────────────────────────────────────────

/** Regex for MESH_Element keys: MESH_Element_{index}_{scene}{prop} */
const RE_MESH_ELEMENT =
  /^MESH_Element_(\d+)_([A-Za-z][A-Za-z0-9_]*?)(position|scale|rotation)$/;

/** Regex for shader uniform keys: ShaderClass/ShaderClass/Element_{N}_{scene}/{uParam} */
const RE_SHADER_UNIFORM =
  /^([A-Za-z]\w+)\/\1\/Element_(\d+_[A-Za-z]\w+)\/(u[A-Za-z_]\w*|_tx_\w+)$/;

/** Default polling interval (ms) — matches pollCellChannels cadence. */
const DEFAULT_POLL_MS = 500;

/**
 * Scene name → cell_id mapping.
 * The AT UIL parameter system uses scene names (Home, About, CleanRoom…) while
 * the cell pub/sub loop uses cell_id strings.  This mapping is populated at
 * runtime by registerSceneMapping() or inferred from container keys.
 *
 * For the default Transformer topology the mapping is 1:1 (cell_id IS the
 * scene name lowercased).  Custom topologies can override via
 * registerSceneMapping().
 */
const _sceneToCell = new Map<string, string>();

// ── Scene ↔ cell_id registry ───────────────────────────────────────────────

/**
 * Register a mapping from AT UIL scene name to cell pub/sub cell_id.
 *
 * @example
 *   registerSceneMapping('Home', 'input_embed');
 *   registerSceneMapping('About', 'self_attn');
 */
export function registerSceneMapping(scene: string, cellId: string): void {
  _sceneToCell.set(scene, cellId);
}

/**
 * Bulk-register from an object.
 *
 * @example
 *   registerSceneMappings({
 *     Home: 'input_embed',
 *     About: 'self_attn',
 *     CleanRoom: 'ffn',
 *   });
 */
export function registerSceneMappings(map: Record<string, string>): void {
  for (const [scene, cellId] of Object.entries(map)) {
    _sceneToCell.set(scene, cellId);
  }
}

/**
 * Clear all scene → cell_id mappings.
 */
export function clearSceneMappings(): void {
  _sceneToCell.clear();
}

// ── Key parsing ────────────────────────────────────────────────────────────

/**
 * Parse a MESH_Element key into a structured entry, or null if it doesn't match.
 */
export function parseMeshElementKey(
  key: string,
  value: unknown,
): UILMeshEntry | null {
  const m = RE_MESH_ELEMENT.exec(key);
  if (!m) return null;

  const arr = value as number[] | undefined;
  if (!Array.isArray(arr) || arr.length < 2) return null;

  return {
    index: parseInt(m[1], 10),
    scene: m[2],
    prop: m[3] as 'position' | 'scale' | 'rotation',
    value: [arr[0] ?? 0, arr[1] ?? 0, arr[2] ?? 0],
  };
}

/**
 * Parse a shader uniform key into a structured entry, or null if it doesn't match.
 */
export function parseShaderUniformKey(
  key: string,
  value: unknown,
): UILShaderUniform | null {
  const m = RE_SHADER_UNIFORM.exec(key);
  if (!m) return null;

  return {
    shaderClass: m[1],
    elementKey: m[2],
    uniformName: m[3],
    value,
  };
}

// ── Resolve scene → cell container ─────────────────────────────────────────

/**
 * Resolve an AT scene name to a PixiJS Container in the container map.
 *
 * Resolution order:
 *   1. Explicit registerSceneMapping() → cell_id → containerMap.get(cell_id)
 *   2. Lowercase scene name as cell_id  → containerMap.get(scene.toLowerCase())
 *   3. Direct scene name                → containerMap.get(scene)
 *   4. null (no match)
 */
function resolveContainer(
  scene: string,
  containerMap: ContainerMap,
): Container | null {
  // 1. Explicit mapping
  const mapped = _sceneToCell.get(scene);
  if (mapped && containerMap.has(mapped)) {
    return containerMap.get(mapped)!;
  }

  // 2. Lowercase convention
  const lower = scene.toLowerCase();
  if (containerMap.has(lower)) {
    return containerMap.get(lower)!;
  }

  // 3. Exact name
  if (containerMap.has(scene)) {
    return containerMap.get(scene)!;
  }

  return null;
}

// ── Core: applyUILToStage ──────────────────────────────────────────────────

/**
 * applyUILToStage — reads MESH_Element entries from a UIL JSON object and
 * sets PixiJS Container position / scale on matched containers.
 *
 * The AT UIL format stores 3D transforms as [x, y, z] arrays.  For the 2D
 * PixiJS stage we project:
 *   position → container.position.set(x, y)      (z stored as metadata)
 *   scale    → container.scale.set(sx, sy)        (sz stored as metadata)
 *   rotation → container.rotation = rz (radians)  (rx, ry stored as metadata)
 *
 * @param stage         PixiJS root Container (for z-sorting after apply)
 * @param uilJson       Flat UIL JSON parameter object
 * @param containerMap  cell_id → Container
 * @returns             Number of containers that were updated
 */
export function applyUILToStage(
  stage: Container,
  uilJson: UILJson,
  containerMap: ContainerMap,
): number {
  let updated = 0;

  for (const [key, value] of Object.entries(uilJson)) {
    const entry = parseMeshElementKey(key, value);
    if (!entry) continue;

    const container = resolveContainer(entry.scene, containerMap);
    if (!container) continue;

    switch (entry.prop) {
      case 'position': {
        const [x, y, z] = entry.value;
        container.position.set(x, y);
        // Store z for 3D-aware sorting / depth compositing
        (container as any).__uilZ = z;
        updated++;
        break;
      }
      case 'scale': {
        const [sx, sy, sz] = entry.value;
        container.scale.set(sx, sy);
        (container as any).__uilScaleZ = sz;
        updated++;
        break;
      }
      case 'rotation': {
        const [_rx, _ry, rz] = entry.value;
        container.rotation = rz;
        // Store 3D rotation components for potential 3D projection
        (container as any).__uilRotationXY = [_rx, _ry];
        updated++;
        break;
      }
    }
  }

  // Re-sort stage children by z after transforms applied
  if (updated > 0) {
    _sortStageByZ(stage);
  }

  return updated;
}

/**
 * Sort stage children by __uilZ (ascending) so deeper containers render first.
 * Containers without __uilZ keep their current order via stable sort.
 */
function _sortStageByZ(stage: Container): void {
  const children = stage.children as Container[];
  if (children.length < 2) return;

  children.sort((a, b) => {
    const az = (a as any).__uilZ ?? 0;
    const bz = (b as any).__uilZ ?? 0;
    return az - bz;
  });

  // PixiJS requires explicit sortableChildren or manual re-ordering.
  // Setting zIndex triggers PixiJS's built-in sort if sortableChildren = true.
  (stage as any).sortableChildren = true;
  for (let i = 0; i < children.length; i++) {
    children[i].zIndex = i;
  }
}

// ── Core: applyUILToShader ─────────────────────────────────────────────────

/**
 * Well-known SDF filter uniform names → filter property names.
 * The SDF filters (CilEyeSDFFilter, CilBoltSDFFilter, etc.) expose uniforms
 * via typed properties.  This map bridges AT UIL uParam names to the
 * filter setter names.
 */
const UNIFORM_ALIAS: Record<string, string> = {
  uAlpha:           'opacity',
  uTime:            'time',
  uColor:           'fillColor',
  uColor0:          'fillColor',
  uTint:            'fillColor',
  uOpacity:         'opacity',
  uScale:           'arrowScale',
  uPulseSpeed:      'pulseSpeed',
  uPulseAmp:        'pulseAmp',
  uArmLength:       'armLength',
  uStrokeWidth:     'strokeWidth',
  uArrowLength:     'arrowLength',
  uFieldScale:      'fieldScale',
  uShaftWidth:      'shaftWidth',
  uBloomStrength:   'bloomStrength',
  uBloomRadius:     'bloomRadius',
  uFocalIntensity:  'focalIntensity',
  uPupilRadius:     'pupilRadius',
  uNumRays:         'numRays',
  uAmbientIntensity: 'ambientIntensity',
  uLightExposure:   'lightExposure',
  uShadowFar:       'shadowFar',
  uShadowBias:      'shadowBias',
};

/**
 * Known SDF filter slot names on Container (set by pixi-cell-renderer's
 * buildCellContainer):
 *   __eyeFilter, __boltFilter, __vectorFilter, __plusFilter, __arrowRightFilter
 *   __glowFilter, __bloomFilter
 */
const FILTER_SLOTS = [
  '__eyeFilter',
  '__boltFilter',
  '__vectorFilter',
  '__plusFilter',
  '__arrowRightFilter',
  '__glowFilter',
  '__bloomFilter',
] as const;

/**
 * applyUILToShader — reads shader uniform entries from a UIL JSON object and
 * applies them to SDF filter instances attached to cell containers.
 *
 * Parsing:
 *   Key format: ShaderClass/ShaderClass/Element_N_scene/uParamName
 *   → extract uParamName, resolve Element_N_scene → container,
 *   → set uniform on all attached SDF filters that have the property.
 *
 * Also handles flat material keys from at_uil_categorized.json "material"
 * section where the key is already a simple uXxx name (when the caller
 * pre-filters to a specific cell/species).
 *
 * @param containerMap  cell_id → Container
 * @param uilJson       Flat UIL JSON parameter object
 * @returns             Number of uniforms that were set
 */
export function applyUILToShader(
  containerMap: ContainerMap,
  uilJson: UILJson,
): number {
  let applied = 0;

  for (const [key, value] of Object.entries(uilJson)) {
    // ── 1. Try structured shader key ──────────────────────────────────────
    const parsed = parseShaderUniformKey(key, value);
    if (parsed) {
      // Element key is like "0_Home" → extract scene name after first underscore
      const underIdx = parsed.elementKey.indexOf('_');
      const scene = underIdx >= 0
        ? parsed.elementKey.slice(underIdx + 1)
        : parsed.elementKey;

      // Find the container for this scene
      for (const [cellId, container] of containerMap) {
        const match = resolveContainer(scene, containerMap) === container
          || cellId === scene
          || cellId === scene.toLowerCase();
        if (!match) continue;

        applied += _setFilterUniform(container, parsed.uniformName, value);
        break;
      }
      continue;
    }

    // ── 2. Try flat uXxx key (pre-filtered per-cell params) ──────────────
    if (key.startsWith('u') && key.length > 1 && key[1] === key[1].toUpperCase()) {
      // Broadcast to all containers (caller should pre-filter uilJson to one cell)
      for (const [, container] of containerMap) {
        applied += _setFilterUniform(container, key, value);
      }
    }
  }

  return applied;
}

/**
 * Set a uniform on any SDF filter attached to the container.
 * Returns 1 if at least one filter was updated, 0 otherwise.
 */
function _setFilterUniform(
  container: Container,
  uniformName: string,
  value: unknown,
): number {
  // Resolve alias
  const propName = UNIFORM_ALIAS[uniformName] ?? uniformName;
  let hit = 0;

  for (const slot of FILTER_SLOTS) {
    const filter = (container as any)[slot];
    if (!filter) continue;

    // Direct property set (typed filter API)
    if (propName in filter) {
      try {
        (filter as any)[propName] = value;
        hit = 1;
      } catch { /* read-only or type mismatch — skip */ }
      continue;
    }

    // Fallback: try resources.uniforms (raw PixiJS Filter uniform group)
    if (filter.resources?.uniforms) {
      const ug = filter.resources.uniforms;
      const uilKey = `u_${_camelToSnake(propName)}`;
      if (uilKey in ug) {
        try {
          ug[uilKey] = value;
          hit = 1;
        } catch { /* skip */ }
      } else if (uniformName in ug) {
        try {
          ug[uniformName] = value;
          hit = 1;
        } catch { /* skip */ }
      }
    }
  }

  return hit;
}

/** camelCase → snake_case helper for uniform name conversion. */
function _camelToSnake(s: string): string {
  return s.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);
}

// ── Core: watchUILChanges ──────────────────────────────────────────────────

/**
 * watchUILChanges — cell pub/sub core polling loop.
 *
 * Opens a polling loop against a UIL JSON endpoint (e.g. /api/uil-params or
 * a static at_uil_categorized.json file).  On each poll:
 *
 *   1. Fetch the JSON
 *   2. Diff against the previous snapshot (shallow key-level equality)
 *   3. Build a delta object containing only changed keys
 *   4. If delta is non-empty:
 *      a. applyUILToStage(stage, delta, containerMap)  — mesh transforms
 *      b. applyUILToShader(containerMap, delta)         — shader uniforms
 *      c. Call opts.onApply(changedCount)
 *   5. Store current snapshot for next diff
 *
 * The diff ensures that only actual mutations cause Container / Filter updates,
 * keeping the per-frame cost near zero when parameters are stable (converged
 * epoch).
 *
 * @param url           URL to fetch UIL JSON from
 * @param stage         PixiJS root Container
 * @param containerMap  cell_id → Container
 * @param opts          Watch options (interval, callbacks)
 * @returns             stop() function to cancel the watcher
 */
export function watchUILChanges(
  url: string,
  stage: Container,
  containerMap: ContainerMap,
  opts: UILWatchOptions = {},
): () => void {
  const intervalMs = opts.intervalMs ?? DEFAULT_POLL_MS;
  const applyShaders = opts.applyShaders ?? true;

  let prevSnapshot: UILJson = {};
  let stopped = false;
  let timer: ReturnType<typeof setInterval> | null = null;

  async function pollAndApply(): Promise<void> {
    if (stopped) return;

    try {
      const res = await fetch(url);
      if (!res.ok) {
        opts.onError?.(new Error(`UIL fetch failed: ${res.status} ${res.statusText}`));
        return;
      }

      const current: UILJson = await res.json();

      // ── Diff ──────────────────────────────────────────────────────────
      const delta: UILJson = {};
      let changedCount = 0;

      for (const [key, value] of Object.entries(current)) {
        if (!_shallowEqual(value, prevSnapshot[key])) {
          delta[key] = value;
          changedCount++;
        }
      }

      // Also detect removals (key was in prev but not in current)
      // — for mesh transforms this could mean a container should reset
      // We don't actively reset here, but the count is informational.
      for (const key of Object.keys(prevSnapshot)) {
        if (!(key in current)) {
          changedCount++;
        }
      }

      // ── Apply delta ───────────────────────────────────────────────────
      if (changedCount > 0) {
        applyUILToStage(stage, delta, containerMap);

        if (applyShaders) {
          applyUILToShader(containerMap, delta);
        }

        opts.onApply?.(changedCount);
      }

      // Store snapshot for next diff
      prevSnapshot = current;
    } catch (err) {
      opts.onError?.(err);
    }
  }

  // Initial fetch
  pollAndApply();

  // Schedule recurring polls
  timer = setInterval(pollAndApply, intervalMs);

  // ── Stop handle ────────────────────────────────────────────────────────
  return () => {
    stopped = true;
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  };
}

// ── Diff helper ────────────────────────────────────────────────────────────

/**
 * Shallow equality for UIL param values.
 * Handles: number, string, boolean, null, number[] arrays.
 */
function _shallowEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (typeof a !== typeof b) return false;

  // Array comparison (common for [x,y,z] position/scale/rotation)
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  }

  // Object (shouldn't appear often in flat UIL JSON, but handle gracefully)
  if (typeof a === 'object' && typeof b === 'object') {
    return JSON.stringify(a) === JSON.stringify(b);
  }

  return false;
}

// ── Convenience: one-shot apply from static JSON ───────────────────────────

/**
 * applyUILSnapshot — applies a complete UIL JSON snapshot in one call
 * (both mesh transforms and shader uniforms).  Useful for initial scene
 * setup or when loading a saved parameter state.
 *
 * @param stage         PixiJS root Container
 * @param uilJson       Full UIL JSON parameter object
 * @param containerMap  cell_id → Container
 * @returns             { meshUpdated, uniformsApplied }
 */
export function applyUILSnapshot(
  stage: Container,
  uilJson: UILJson,
  containerMap: ContainerMap,
): { meshUpdated: number; uniformsApplied: number } {
  const meshUpdated = applyUILToStage(stage, uilJson, containerMap);
  const uniformsApplied = applyUILToShader(containerMap, uilJson);
  return { meshUpdated, uniformsApplied };
}

// ── Cell pub/sub integration ───────────────────────────────────────────────

/**
 * UILCellBridge — connects a CellEventSource (SSE stream) to the UIL bridge
 * so that cell_update events carrying UIL-format params are automatically
 * applied to the PixiJS stage.
 *
 * This is the cell pub/sub core: the DataNotifier publishes cell_update
 * events → CellEventSource receives them → UILCellBridge.onCellUpdate()
 * extracts UIL params and applies transforms + uniforms.
 *
 * Usage:
 *   import { UILCellBridge } from './uil-bridge';
 *   import { getCellEventSource } from '$lib/CellEventSource';
 *
 *   const bridge = new UILCellBridge(app.stage, containerMap);
 *   const src = getCellEventSource();
 *   src.addListener((ev) => bridge.onCellUpdate(ev));
 *   src.connect();
 *   // later:
 *   bridge.destroy();
 */
export class UILCellBridge {
  private _stage: Container;
  private _containerMap: ContainerMap;
  private _destroyed = false;

  constructor(stage: Container, containerMap: ContainerMap) {
    this._stage = stage;
    this._containerMap = containerMap;
  }

  /**
   * Handle a cell_update event from CellEventSource.
   * Extracts UIL-format params from the event payload and applies them.
   *
   * Expected payload shape (from /api/cell/publish):
   *   {
   *     cell_id: string,
   *     params: {
   *       bbox?: { x, y, w, h },
   *       species_params?: Record<string, unknown>,  // UIL uniform params
   *       uil_mesh?: Record<string, unknown>,         // MESH_Element overrides
   *       ...other CellParamsJson fields...
   *     }
   *   }
   */
  onCellUpdate(event: { cell_id: string; params: Record<string, unknown> }): void {
    if (this._destroyed) return;

    const { cell_id, params } = event;
    const container = this._containerMap.get(cell_id);
    if (!container) return;

    // ── Apply bbox as position ──────────────────────────────────────────
    const bbox = params.bbox as { x?: number; y?: number; w?: number; h?: number } | undefined;
    if (bbox && bbox.x != null && bbox.y != null) {
      container.position.set(bbox.x, bbox.y);
    }

    // ── Apply UIL mesh transforms if present ────────────────────────────
    const uilMesh = params.uil_mesh as UILJson | undefined;
    if (uilMesh) {
      const singleMap = new Map<string, Container>([[cell_id, container]]);
      applyUILToStage(this._stage, uilMesh, singleMap);
    }

    // ── Apply species_params as shader uniforms ─────────────────────────
    const speciesParams = params.species_params as UILJson | undefined;
    if (speciesParams) {
      const singleMap = new Map<string, Container>([[cell_id, container]]);
      applyUILToShader(singleMap, speciesParams);
    }
  }

  /**
   * Update the container map (e.g. after scene rebuild).
   */
  setContainerMap(map: ContainerMap): void {
    this._containerMap = map;
  }

  /**
   * Clean up.
   */
  destroy(): void {
    this._destroyed = true;
  }
}
