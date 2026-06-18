/**
 * scene-layout.ts — AT Hydra INPUT_scenelayout + INPUT_Config parser
 *
 * Core pattern from Active Theory's Hydra engine: JSON-driven scene hierarchy.
 * UIL JSON stores two key prefixes that together define the full scene graph:
 *
 *   INPUT_scenelayout_{SceneName}_data  →  `{"layers":N,"groups":G}`
 *     Declares how many mesh layers and optional groups a scene has.
 *
 *   INPUT_Config_{index}_{SceneName}_{property}  →  per-layer config
 *     Each layer gets a numbered config slot (0, 1, 2, …) keyed to a scene
 *     name.  Properties include name, shader, geometry, blending, renderOrder,
 *     sortIndex, visible, transparent, depthTest, depthWrite, custom, parent.
 *
 * This module parses both prefixes into a typed SceneLayout, then applies it
 * to a PixiJS Container stage by creating child Containers sorted by
 * renderOrder.
 *
 * Upstream references:
 *   src/lib/SceneLayoutPresets.ts        — raw AT params (653 scene + 401 mesh)
 *   src/lib/fx-scene.ts                  — FXLayer / FXScene compositing
 *   upstream/active-theory/hydra/SceneLayout.js
 *   upstream/active-theory/hydra/SceneLayout.glsl
 */

import { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';

// ── Interfaces ────────────────────────────────────────────────────────────────

/**
 * Configuration for a single layer (mesh slot) within an AT scene.
 *
 * Maps 1:1 to the `INPUT_Config_{i}_{scene}_{prop}` key pattern found in
 * UIL JSON.  Every field except `name` and the ordering fields is optional
 * because the original UIL data is sparse — only explicitly-set properties
 * appear as keys.
 */
export interface SceneLayerConfig {
  /** Human-readable name for this layer (e.g. "glasscube", "particles"). */
  name: string;

  /** Geometry reference — file path, `World.BOX`, or JSON object as string. */
  geometry?: string;

  /** Shader class name (e.g. "GlassCubeShader", "TreeFBR"). */
  shader?: string;

  /** Blend mode string (e.g. "shader_normal_blending", "shader_additive_blending"). */
  blending?: string;

  /**
   * Render order — determines draw order within the scene.
   * Lower values draw first (behind); fractional values allowed.
   */
  renderOrder: number;

  /**
   * Sort index — secondary ordering hint used by AT's scene graph builder
   * when multiple layers share the same renderOrder.
   */
  sortIndex: number;

  /** Whether this layer is visible at init. */
  visible: boolean;

  /** Whether the material uses alpha transparency. */
  transparent: boolean;

  /** Whether the GPU depth test is enabled. */
  depthTest: boolean;

  /** Whether the GPU depth buffer write is enabled. */
  depthWrite: boolean;

  /** Custom class name (e.g. "Light", "GazeCamera", "Proton", "WorkSceneBackground"). */
  custom?: string;

  /** Parent group identifier (e.g. "group_0", "sl_TreeScene_group_0"). */
  parent?: string;
}

/**
 * Complete scene layout parsed from UIL JSON.
 *
 * `layers` and `groups` come from the `INPUT_scenelayout_{scene}_data` key.
 * `configs` is the assembled array of per-layer configs extracted from
 * `INPUT_Config_{i}_{scene}_{prop}` keys.
 */
export interface SceneLayout {
  /** Total number of layers declared by the scenelayout data entry. */
  layers: number;

  /** Number of groups declared (0 or absent means no named groups). */
  groups?: number;

  /** Parsed per-layer configurations, one per discovered config slot. */
  configs: SceneLayerConfig[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Coerce an AT config value to boolean.
 * AT stores booleans inconsistently: native `true`/`false`, string `"true"`/
 * `"false"`, or sometimes absent.  This normalises them all.
 */
function toBool(v: unknown, fallback: boolean): boolean {
  if (v === undefined || v === null) return fallback;
  if (typeof v === 'boolean') return v;
  if (typeof v === 'string') return v.toLowerCase() === 'true';
  return !!v;
}

/**
 * Coerce an AT config value to number.
 * Values may arrive as native numbers or numeric strings.
 */
function toNum(v: unknown, fallback: number): number {
  if (v === undefined || v === null) return fallback;
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Stringify a geometry value that might be a string or an object
 * (AT stores geometry as either a path string or a `{filename, prefix, relative, src}` object).
 */
function geoToString(v: unknown): string | undefined {
  if (v === undefined || v === null) return undefined;
  if (typeof v === 'string') return v || undefined;
  if (typeof v === 'object') {
    try { return JSON.stringify(v); } catch { return undefined; }
  }
  return String(v) || undefined;
}

// ── Config prefix regex ───────────────────────────────────────────────────────

/**
 * Matches `INPUT_Config_{index}_{sceneName}_{property}`.
 *
 * Groups:
 *   [1] index (numeric, e.g. "0", "14", "22")
 *   [2] sceneName + property tail — needs secondary split because scene names
 *       themselves can contain underscores (e.g. "home_scene", "work_scene").
 *
 * We build a more precise regex per-scene in parseSceneLayout().
 */
const CONFIG_PREFIX = /^INPUT_Config_(\d+)_(.+)$/;

// ── Known per-layer properties ────────────────────────────────────────────────

const LAYER_PROPS = new Set([
  'name', 'geometry', 'shader', 'blending', 'renderOrder', 'sortIndex',
  'visible', 'transparent', 'depthTest', 'depthWrite',
  'custom', 'customClass', 'parent',
  'side', 'billboard', 'scriptClass', 'wildcard', 'group',
]);

// ── parseSceneLayout ──────────────────────────────────────────────────────────

/**
 * Extract a SceneLayout from a UIL JSON blob for the given scene.
 *
 * Scans all keys for the two AT prefixes:
 *   • `INPUT_scenelayout_{sceneName}_data` — the layer/group count
 *   • `INPUT_Config_{i}_{sceneName}_{prop}` — per-layer properties
 *
 * The `sceneName` matching is case-sensitive and supports multi-word names
 * like `home_scene`, `TreeScene`, `work_page`, etc.
 *
 * @param uilJson - Flat key-value UIL JSON (same shape as SceneLayoutPresets.SCENE_PARAMS)
 * @param sceneName - Exact scene name to filter for (e.g. "home_scene", "TreeScene")
 * @returns Parsed SceneLayout with all discovered layer configs
 */
export function parseSceneLayout(
  uilJson: Record<string, any>,
  sceneName: string,
): SceneLayout {
  // ── 1. Extract scenelayout data ────────────────────────────────────────────
  const layoutKey = `INPUT_scenelayout_${sceneName}_data`;
  let layerCount = 0;
  let groupCount: number | undefined;

  const rawLayout = uilJson[layoutKey];
  if (rawLayout !== undefined) {
    const parsed = typeof rawLayout === 'string' ? JSON.parse(rawLayout) : rawLayout;
    layerCount = toNum(parsed.layers, 0);
    if (parsed.groups !== undefined) {
      groupCount = toNum(parsed.groups, 0);
    }
  }

  // ── 2. Collect INPUT_Config entries for this scene ─────────────────────────
  //
  // Key format: INPUT_Config_{index}_{sceneName}_{property}
  // We build an exact suffix match: `_{sceneName}_` must appear after the index.
  const configSuffix = `_${sceneName}_`;
  const slotMap = new Map<number, Record<string, any>>();

  for (const key of Object.keys(uilJson)) {
    const m = CONFIG_PREFIX.exec(key);
    if (!m) continue;

    const idx = parseInt(m[1], 10);
    const tail = m[2]; // e.g. "home_scene_renderOrder" or "TreeScene_shader"

    // Check that the tail starts with our sceneName followed by underscore
    if (!tail.startsWith(`${sceneName}_`)) continue;

    // Extract property name (everything after sceneName_)
    const prop = tail.slice(sceneName.length + 1);
    if (!prop) continue;

    let slot = slotMap.get(idx);
    if (!slot) {
      slot = {};
      slotMap.set(idx, slot);
    }
    slot[prop] = uilJson[key];
  }

  // ── 3. Also collect GROUP_ and INPUT_GROUP_ entries ────────────────────────
  //   GROUP_{sceneName}_group_{i}_name / GROUP_{sceneName}_group_{i}_sortIndex
  //   INPUT_GROUP_{sceneName}_group_{i}_name
  // These are stored for reference but don't directly become layer configs.

  // ── 4. Build SceneLayerConfig array ────────────────────────────────────────
  const configs: SceneLayerConfig[] = [];

  // Sort slot indices numerically for deterministic output
  const sortedSlots = Array.from(slotMap.keys()).sort((a, b) => a - b);

  for (const idx of sortedSlots) {
    const raw = slotMap.get(idx)!;

    // AT uses both "custom" and "customClass" in different scenes — normalise
    const customValue = raw.custom ?? raw.customClass;

    const config: SceneLayerConfig = {
      name:        typeof raw.name === 'string' ? raw.name : `layer_${idx}`,
      geometry:    geoToString(raw.geometry),
      shader:      raw.shader != null ? String(raw.shader) : undefined,
      blending:    raw.blending != null ? String(raw.blending) : undefined,
      renderOrder: toNum(raw.renderOrder, idx),
      sortIndex:   toNum(raw.sortIndex, idx),
      visible:     toBool(raw.visible, true),
      transparent: toBool(raw.transparent, false),
      depthTest:   toBool(raw.depthTest, true),
      depthWrite:  toBool(raw.depthWrite, true),
      custom:      customValue != null ? String(customValue) : undefined,
      parent:      raw.parent != null ? String(raw.parent) : (
                     raw.group != null ? String(raw.group) : undefined
                   ),
    };

    configs.push(config);
  }

  return {
    layers: layerCount,
    groups: groupCount,
    configs,
  };
}

// ── applySceneLayout ──────────────────────────────────────────────────────────

/**
 * Build a PixiJS Container hierarchy from a SceneLayout.
 *
 * Creates one child Container per SceneLayerConfig, sorted by renderOrder
 * (ascending — lowest draws first / sits behind).  Each container is labelled
 * and has its `visible` flag set from config.
 *
 * If a config specifies a `parent` group, the layer container is nested under
 * a group container that is created on first reference.  Group containers are
 * themselves sorted by the lowest renderOrder of their children.
 *
 * All containers are attached via `stage.addChild()` which gives them a
 * deterministic draw order matching AT Hydra's rendering pipeline.
 *
 * @param stage  - Root PixiJS Container to populate
 * @param layout - Parsed SceneLayout (from parseSceneLayout)
 * @returns Map from layer name → created Container, for downstream wiring
 */
export function applySceneLayout(
  stage: Container,
  layout: SceneLayout,
): Map<string, Container> {
  const result = new Map<string, Container>();

  // Sort configs by renderOrder, then sortIndex as tiebreaker
  const sorted = [...layout.configs].sort((a, b) => {
    const ro = a.renderOrder - b.renderOrder;
    return ro !== 0 ? ro : a.sortIndex - b.sortIndex;
  });

  // Track group containers for parent nesting
  const groupContainers = new Map<string, Container>();

  /**
   * Get or create a group container.  Group containers are interactiveChildren
   * pass-throughs — they just provide a structural grouping node.
   */
  function getGroup(groupName: string): Container {
    let group = groupContainers.get(groupName);
    if (!group) {
      group = new Container();
      group.label = groupName;
      group.visible = true;
      groupContainers.set(groupName, group);
    }
    return group;
  }

  // ── Phase 1: create all layer containers ──────────────────────────────────
  // Containers that belong to a group are attached to the group container.
  // Orphan layers and group containers themselves are collected for stage attachment.

  /** Tracks which groups have been seen and their minimum renderOrder. */
  const groupOrder = new Map<string, number>();

  for (const cfg of sorted) {
    const child = new Container();
    child.label = cfg.name;
    child.visible = cfg.visible;

    // Store config metadata on the container for downstream consumers
    // (shader wiring, material setup, etc.) via the label + custom data
    (child as any).__sceneLayerConfig = cfg;

    if (cfg.parent) {
      const group = getGroup(cfg.parent);
      group.addChild(child);

      // Track the minimum renderOrder for this group
      const existing = groupOrder.get(cfg.parent);
      if (existing === undefined || cfg.renderOrder < existing) {
        groupOrder.set(cfg.parent, cfg.renderOrder);
      }
    } else {
      // No parent — goes directly on stage; we add in Phase 2 with ordering
    }

    result.set(cfg.name, child);
  }

  // ── Phase 2: assemble onto stage in renderOrder ───────────────────────────
  // Build a unified list of items to add: top-level layers + group containers.

  interface StageEntry {
    container: Container;
    order: number;
    sortIdx: number;
  }

  const entries: StageEntry[] = [];

  for (const cfg of sorted) {
    if (!cfg.parent) {
      // Direct child of stage
      const child = result.get(cfg.name);
      if (child) {
        entries.push({
          container: child,
          order: cfg.renderOrder,
          sortIdx: cfg.sortIndex,
        });
      }
    }
  }

  // Add group containers at the minimum renderOrder of their children
  for (const [name, group] of groupContainers) {
    const order = groupOrder.get(name) ?? 0;
    entries.push({ container: group, order, sortIdx: 0 });
    result.set(name, group);
  }

  // Sort the unified list
  entries.sort((a, b) => {
    const ro = a.order - b.order;
    return ro !== 0 ? ro : a.sortIdx - b.sortIdx;
  });

  // Attach in order — addChild sequence determines draw order
  for (const entry of entries) {
    stage.addChild(entry.container);
  }

  return result;
}
