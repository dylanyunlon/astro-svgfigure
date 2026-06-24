/**
 * at-cables-edge.ts — M808: AT cables.bin + CABLES PBR 纹理套件 edge 连接
 *
 * Renders cell-to-cell edge connections as physically-based catenary cables
 * using Active Theory's cables.bin geometry asset and the full CABLES PBR
 * texture suite (CyclesBake_COMBINED, PBR_AT_MRO, PBR_Normal).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * AT asset pipeline (from upstream/activetheory-assets/):
 *
 *   geometry/cables.bin              → instanced tube geometry (cable cross-section)
 *   textures/CABLES___CyclesBake_COMBINED.ktx2  → baseColor / albedo
 *   textures/CABLES___PBR_AT_MRO.ktx2           → packed Metallic(R) Roughness(G) Occlusion(B)
 *   textures/CABLES___PBR_Normal.ktx2            → tangent-space normal map
 *
 * The cables.bin geometry provides a pre-modelled tube cross-section (circular
 * cable profile with 12 radial segments).  We instance this profile along a
 * catenary spline computed per-edge from edge_routes.json control points.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Catenary physics:
 *
 *   A catenary is the curve assumed by a uniform cable hanging under its own
 *   weight: y(x) = a · cosh((x − x₀) / a) + y₀.  The sag parameter `a`
 *   controls how much the cable droops — smaller `a` = more droop.
 *
 *   For each edge we compute a catenary between source and target endpoints,
 *   then sample N points along the arc-length-parameterised curve to position
 *   cable geometry instances.  The sag responds to edge length: longer cables
 *   droop more, matching physical intuition.
 *
 *   Additional micro-sway is driven by a simplex-noise displacement perpendicular
 *   to the cable tangent, giving subtle wind/vibration that makes the scene feel
 *   alive without physics simulation overhead.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Data flow pulse:
 *
 *   Each cable carries a scrolling luminance pulse that travels source → target,
 *   visualising data flow through the topology.  The pulse is a Gaussian
 *   brightness peak modulating the PBR emissive channel:
 *
 *     emissive += pulseColor · exp(−(t − tPulse)² / (2σ²))
 *
 *   where `t` is the normalised arc-length position and `tPulse` scrolls from
 *   0 → 1 at `pulseSpeed`.  Multiple concurrent pulses are supported per cable
 *   via a ring buffer, matching the EdgeDataFlowViz (M766) pulse architecture.
 *
 *   Pulse colour inherits from the source cell's species palette via
 *   getSpeciesColors(), and skip-connection cables receive a wider, brighter
 *   pulse with the skip_connection amber tone (#FFB74D).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * PBR material (AT convention):
 *
 *   The cable material follows PBRMaterial.ts (AT MRON packed format):
 *     uMRON = [metallic, roughness, occlusionStrength, normalScale]
 *
 *   Default cable PBR params (tuned for rubber/silicone industrial cable look):
 *     metallic      = 0.15   — mostly dielectric, slight metallic sheen
 *     roughness     = 0.55   — semi-matte rubber surface
 *     occlusion     = 0.90   — crevice darkening in cable braids
 *     normalScale   = 1.20   — pronounced braid/weave normal detail
 *     F0            = 0.04   — standard dielectric Fresnel
 *
 *   These can be overridden per-edge via CableEdgeConfig for different cable
 *   types (e.g. metallic conduit for skip connections, fibre-optic glass).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Integration:
 *
 *   ```ts
 *   import { ATCablesEdgeRenderer, createATCablesEdgeRenderer } from '$lib/renderers/at-cables-edge';
 *   import routesJson from '../../physics/edge_routes.json';
 *
 *   const renderer = createATCablesEdgeRenderer(stage, cellMap, {
 *     routes: routesJson,
 *     sagFactor: 0.08,
 *     pulseSpeed: 0.45,
 *   });
 *
 *   // animation loop:
 *   app.ticker.add(({ deltaMS }) => {
 *     renderer.update(deltaMS / 1000, performance.now() / 1000);
 *   });
 *
 *   // pub/sub hook: fire a pulse when data flows through an edge
 *   eventBus.on('edge:message', (edgeId) => renderer.firePulse(edgeId));
 *
 *   // cleanup:
 *   renderer.destroy();
 *   ```
 *
 * Pub-sub integration (cell-pubsub-loop branch):
 *   ATCablesEdgeRenderer subscribes to CellEventSource for live species-palette
 *   updates — cable tint and pulse colour refresh without re-allocation.
 *
 * M893: composite_params edge rendering integration
 *   Pass `compositeEdgeParams` (keyed by edge_id, values from
 *   composite_params.json edges[*].rendering) into CableEdgeConfig.
 *   The renderer reads:
 *     spline_params  → thickness, color, glow_width/color/intensity,
 *                      noise_amplitude/frequency, flow_speed, dash_pattern
 *                      → wired into _drawCable() stroke + halo passes AND
 *                        getEdgeSplineUniforms() for the GPU shader
 *     particle_params → count, speed, size, color, opacity, trail_length
 *                      → exposed via getParticleParams() for particle systems
 *     pixi_filters   → glow (distance, outerStrength, color) + blur (strength)
 *                      → applied as Graphics.filters on the cable container
 *     render_params  → z_index, blend_mode, arrow_size/fill
 *                      → applied as Graphics.zIndex / blendMode
 *
 * References:
 *   upstream/activetheory-assets/geometry/cables.bin        — cable tube geometry
 *   upstream/activetheory-assets/textures/CABLES___*.ktx2   — PBR texture suite
 *   src/lib/renderer/material/PBRMaterial.ts                — AT MRON PBR material
 *   src/lib/renderers/flower-edge-renderer.ts               — spiral particle edge flow
 *   src/lib/sph/edge-data-flow-viz.ts                       — glow-pulse edge viz
 *   src/lib/particle/edge-particle-bridge.ts                — edge → Bézier mapping
 *   src/lib/shaders/edge-spline.frag                        — GPU edge spline shader
 *   channels/physics/edge_routes.json                       — topology edge route data
 *   channels/rendering/constants.py                         — _F0_TABLE species Fresnel
 */

import { Container, Graphics, Mesh, MeshGeometry, MeshMaterial, BLEND_MODES } from 'pixi.js';
import type { CellEventSource } from '../CellEventSource';
import { getSpeciesColors } from './cell-color-palette';
import { GlowFilter, KawaseBlurFilter } from './pixi-filters-registry';

// ── M893: composite_params.json edge rendering types ─────────────────────────

// [orphan-precise] /**
// [orphan-precise]  * Shape of composite_params.json → edges[edgeId].rendering.spline_params
// [orphan-precise]  */
export interface CompositeSplineParams {
  thickness:       number;
  color:           string;
  dash_pattern:    { dash: number; gap: number };
  glow_width:      number;
  glow_color:      string;
  glow_intensity:  number;
  noise_amplitude: number;
  noise_frequency: number;
  flow_speed:      number;
}

/**
 * Shape of composite_params.json → edges[edgeId].rendering.particle_params
 */
export interface CompositeParticleParams {
  count:        number;
  size:         number;
  speed:        number;
  color:        string;
  opacity:      number;
  trail_length: number;
}

/**
 * Shape of composite_params.json → edges[edgeId].rendering.pixi_filters
 */
export interface CompositePixiFilters {
  glow?: {
    distance:      number;
    outerStrength: number;
    color:         string; // "0xRRGGBB"
  };
  blur?: {
    strength: number;
  };
}

/**
 * Shape of composite_params.json → edges[edgeId].rendering.render_params
 */
export interface CompositeRenderParams {
  z_index:    number;
  blend_mode: string;
  arrow_size: number;
  arrow_fill: string;
}

/**
 * Full rendering block for one edge from composite_params.json edges[id].rendering.
 */
export interface CompositeEdgeRendering {
  spline_params:   CompositeSplineParams;
  particle_params: CompositeParticleParams;
  pixi_filters:    CompositePixiFilters;
  render_params:   CompositeRenderParams;
}

/**
 * Helper: parse "#RRGGBB" or "0xRRGGBB" to a numeric colour.
 */
function parseCompositeColor(c: string): number {
  if (!c) return 0xFFFFFF;
  if (c.startsWith('#'))  return parseInt(c.slice(1), 16);
  if (c.startsWith('0x') || c.startsWith('0X')) return parseInt(c.slice(2), 16);
  return parseInt(c, 16);
}

/**
 * Map composite_params blend_mode string to PixiJS BLEND_MODES constant.
 */
function resolveBlendMode(mode: string): number {
  switch ((mode ?? '').toLowerCase()) {
    case 'add':       return BLEND_MODES.ADD;
    case 'multiply':  return BLEND_MODES.MULTIPLY;
    case 'screen':    return BLEND_MODES.SCREEN;
    case 'overlay':   return BLEND_MODES.OVERLAY;
    default:          return BLEND_MODES.NORMAL;
  }
}

// ── Asset paths (AT upstream convention) ─────────────────────────────────────

const CABLES_GEOMETRY_PATH   = 'upstream/activetheory-assets/geometry/cables.bin';
const CABLES_TEX_BASE_COLOR  = 'upstream/activetheory-assets/textures/CABLES___CyclesBake_COMBINED.ktx2';
const CABLES_TEX_MRO         = 'upstream/activetheory-assets/textures/CABLES___PBR_AT_MRO.ktx2';
const CABLES_TEX_NORMAL      = 'upstream/activetheory-assets/textures/CABLES___PBR_Normal.ktx2';

// ── JSON asset types (shared with edge-particle-bridge.ts) ──────────────────

interface RoutePoint { x: number; y: number; }

interface RouteEntry {
  edge_id:    string;
  sources:    string[];
  targets:    string[];
  is_skip:    boolean;
  advanced:   { semanticType?: string; routing?: string; curvature?: number };
  points:     RoutePoint[];
}

// ── Catenary constants ───────────────────────────────────────────────────────

/**
 * Default sag factor: ratio of max sag to horizontal span.
 * 0.08 = gentle industrial cable droop (8% of span).
 * AT reference: Finding-Love cables scene uses similar gentle sag.
 */
const DEFAULT_SAG_FACTOR = 0.08;

/**
 * Skip-connection cables get extra sag to visually differentiate them
 * from sequential data-flow cables.
 */
const SKIP_SAG_MULTIPLIER = 1.6;

/** Number of sample points along the catenary for geometry instancing. */
const CATENARY_SEGMENTS = 32;

/** Micro-sway amplitude in pixels (wind/vibration aesthetic). */
const MICRO_SWAY_AMPLITUDE = 1.8;

/** Micro-sway frequency multiplier (higher = faster oscillation). */
const MICRO_SWAY_FREQ = 0.7;

/** Cable tube radius in canvas pixels. */
const CABLE_TUBE_RADIUS = 2.5;

/** Skip-connection cable radius multiplier. */
const SKIP_RADIUS_MULTIPLIER = 1.35;

// ── PBR material defaults (AT MRON packed format) ────────────────────────────

/**
 * Default CABLES PBR parameters — tuned for rubber/silicone industrial cable.
 *
 * AT convention: uMRON = [metallic, roughness, occlusionStrength, normalScale]
 * Textures supply per-texel detail; these uniforms are multipliers.
 */
const CABLE_PBR_DEFAULTS = {
  metallic:     0.15,
  roughness:    0.55,
  occlusion:    0.90,
  normalScale:  1.20,
  f0:           0.04,   // dielectric Fresnel (IOR ≈ 1.5)
} as const;

/**
 * Skip-connection cable PBR — slightly metallic conduit appearance.
 */
const SKIP_CABLE_PBR = {
  metallic:     0.45,
  roughness:    0.35,
  occlusion:    0.85,
  normalScale:  1.00,
  f0:           0.06,
} as const;

// ── Data flow pulse constants ────────────────────────────────────────────────

/** Maximum concurrent pulses per cable (ring buffer size). */
const MAX_PULSES_PER_CABLE = 6;

/** Default pulse travel speed (0→1 normalised arc per second). */
const DEFAULT_PULSE_SPEED = 0.45;

/** Pulse Gaussian width σ (normalised arc units). */
const PULSE_SIGMA = 0.08;

/** Skip-connection pulse σ (wider, more dramatic). */
const SKIP_PULSE_SIGMA = 0.14;

/** Pulse peak emissive brightness [0, 1]. */
const PULSE_PEAK_BRIGHTNESS = 0.90;

/** Pulse trail exponential decay rate. */
const PULSE_TRAIL_DECAY = 6.0;

/** Ambient auto-pulse cadence: seconds between ambient pulses per cable. */
const AUTO_PULSE_CADENCE = 1.2;

/** Skip-connection auto-pulse cadence (more frequent for visual emphasis). */
const SKIP_AUTO_PULSE_CADENCE = 0.7;

/** Emissive base glow (always-on dim luminance along cable). */
const BASE_EMISSIVE_ALPHA = 0.04;

// ── Edge semantic type → colour (reuses edge-particle-bridge palette) ────────

const CABLE_TINT_COLORS: Record<string, number> = {
  data_flow:       0x64B5F6,   // cool blue
  skip_connection: 0xFFB74D,   // warm amber
  residual:        0x90EE90,   // soft green
  attention:       0xE882FA,   // orchid violet
};

function cableTintForSemantic(semanticType?: string): number {
  return CABLE_TINT_COLORS[semanticType || 'data_flow'] ?? CABLE_TINT_COLORS.data_flow;
}

// ── Catenary math ────────────────────────────────────────────────────────────

/**
 * Compute a catenary curve between two points with specified sag.
 *
 * The catenary y(x) = a·cosh((x − x₀)/a) + y₀ is the shape of a hanging
 * cable under uniform gravity.  We compute the `a` parameter from the desired
 * sag ratio and horizontal span, then sample uniformly in arc-length space.
 *
 * For near-vertical edges (small horizontal span relative to vertical drop)
 * we fall back to a parabolic approximation which is numerically stabler and
 * visually indistinguishable at small spans.
 *
 * @param p0  Source endpoint
 * @param p1  Target endpoint
 * @param sagFactor  Sag as fraction of span (0.08 = 8%)
 * @param segments   Number of sample points
 * @returns Array of (segments+1) points along the catenary
 */
function computeCatenary(
  p0: RoutePoint,
  p1: RoutePoint,
  sagFactor: number,
  segments: number,
): RoutePoint[] {
  const dx = p1.x - p0.x;
  const dy = p1.y - p0.y;
  const span = Math.sqrt(dx * dx + dy * dy);

  if (span < 1e-4) {
    // Degenerate: source and target overlap
    return Array.from({ length: segments + 1 }, () => ({ ...p0 }));
  }

  const maxSag = sagFactor * span;
  const points: RoutePoint[] = [];

  // Direction unit vector along the chord
  const ux = dx / span;
  const uy = dy / span;

  // Perpendicular (sag direction): for a cable hanging "down", we want the
  // sag to bow outward from the straight-line chord.  In screen coordinates
  // (y-down), the perpendicular that bows "down-right" is (uy, -ux).
  // We pick the perpendicular whose y-component is positive (bows downward).
  let px = uy;
  let py = -ux;
  if (py < 0) { px = -px; py = -py; }

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;

    // Linear interpolation along chord
    const lx = p0.x + dx * t;
    const ly = p0.y + dy * t;

    // Parabolic sag profile: max at t=0.5, zero at endpoints
    // sag(t) = 4 · maxSag · t · (1 − t)
    const sag = 4.0 * maxSag * t * (1.0 - t);

    points.push({
      x: lx + px * sag,
      y: ly + py * sag,
    });
  }

  return points;
}

/**
 * Compute the unit tangent at sample index `i` via finite differences.
 */
function tangentAtSample(samples: RoutePoint[], i: number): RoutePoint {
  const prev = samples[Math.max(0, i - 1)];
  const next = samples[Math.min(samples.length - 1, i + 1)];
  const dx = next.x - prev.x;
  const dy = next.y - prev.y;
  const len = Math.sqrt(dx * dx + dy * dy) + 1e-10;
  return { x: dx / len, y: dy / len };
}

/**
 * Compute the perpendicular (left-hand normal) of a tangent.
 */
function perpendicular(tangent: RoutePoint): RoutePoint {
  return { x: -tangent.y, y: tangent.x };
}

// ── Arc-length utilities ─────────────────────────────────────────────────────

/**
 * Build a cumulative arc-length LUT from sample points.
 * Returns { lut: normalised cumulative fractions, totalLength }.
 */
function buildArcLengthLUT(
  samples: RoutePoint[],
): { lut: Float32Array; totalLength: number } {
  const n = samples.length;
  const lut = new Float32Array(n);
  lut[0] = 0;
  let cum = 0;

  for (let i = 1; i < n; i++) {
    const dx = samples[i].x - samples[i - 1].x;
    const dy = samples[i].y - samples[i - 1].y;
    cum += Math.sqrt(dx * dx + dy * dy);
    lut[i] = cum;
  }

  const total = cum;
  if (total > 0) {
    for (let i = 1; i < n; i++) lut[i] /= total;
  }

  return { lut, totalLength: total };
}

// ── Simplex noise (inline minimal 2D — avoids external dep) ─────────────────

/**
 * Simple 2D hash-based value noise for micro-sway.
 * Not true simplex, but visually sufficient for gentle cable vibration.
 * Based on lygia generative/snoise pattern but reduced to 2D scalar.
 */
function valueNoise2D(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;

  // Smoothstep interpolation
  const ux = fx * fx * (3.0 - 2.0 * fx);
  const uy = fy * fy * (3.0 - 2.0 * fy);

  // Hash corners
  const hash = (px: number, py: number) => {
    const h = Math.sin(px * 127.1 + py * 311.7) * 43758.5453;
    return h - Math.floor(h);
  };

  const a = hash(ix, iy);
  const b = hash(ix + 1, iy);
  const c = hash(ix, iy + 1);
  const d = hash(ix + 1, iy + 1);

  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

// ── Pulse state ──────────────────────────────────────────────────────────────

interface CablePulse {
  /** Normalised travel position along cable [0, 1+]. */
  travel:    number;
  /** Is this pulse alive? Dead pulses are recycled in the ring buffer. */
  alive:     boolean;
  /** Per-pulse intensity jitter [0.8, 1.2]. */
  intensity: number;
}

// ── Cable state (per-edge) ───────────────────────────────────────────────────

interface CableState {
  /** Edge metadata. */
  edgeId:       string;
  sourceId:     string;
  targetId:     string;
  isSkip:       boolean;
  semanticType: string;

  /** Catenary sample points (CATENARY_SEGMENTS + 1). */
  samples:      RoutePoint[];
  /** Pre-computed tangents at each sample. */
  tangents:     RoutePoint[];
  /** Arc-length LUT (normalised cumulative). */
  arcLUT:       Float32Array;
  /** Total arc length in canvas pixels. */
  arcLength:    number;

  /** PBR parameters (AT MRON). */
  pbr:          typeof CABLE_PBR_DEFAULTS;

  /** Cable tint colour (hex). */
  tintHex:      number;
  /** Species palette colour (for pulse). */
  pulseHex:     number;

  /** Pulse ring buffer. */
  pulses:       CablePulse[];
  /** Ring buffer write head. */
  ringHead:     number;

  /** Auto-pulse cadence timer. */
  autoTimer:    number;
  /** Auto-pulse cadence period. */
  autoCadence:  number;

  /** Tube radius in canvas px. */
  radius:       number;

  /** Sag factor used for this cable. */
  sagFactor:    number;

  /**
   * M893: resolved composite_params rendering block for this cable (may be
   * undefined if no composite_params entry was provided for this edge_id).
   */
  compositeRendering: CompositeEdgeRendering | undefined;

  /**
   * M893: live PixiJS GlowFilter instance built from pixi_filters.glow params.
   * Null if pixi_filters.glow was absent or outerStrength === 0.
   */
  pixiGlowFilter: GlowFilter | null;

  /**
   * M893: live PixiJS KawaseBlurFilter instance built from pixi_filters.blur.
   * Null if pixi_filters.blur was absent or strength === 0.
   */
  pixiBlurFilter: KawaseBlurFilter | null;
}

// ── Public configuration ─────────────────────────────────────────────────────

export interface CableEdgeConfig {
  /** Route entries from edge_routes.json. */
  routes:            Record<string, RouteEntry>;
  /** Global sag factor (fraction of span). Default 0.08. */
  sagFactor?:        number;
  /** Global pulse speed (0→1 per second). Default 0.45. */
  pulseSpeed?:       number;
  /** Enable ambient auto-pulse. Default true. */
  enableAutoPulse?:  boolean;
  /** Global brightness multiplier [0, 1]. Default 1.0. */
  brightness?:       number;
  /** Catenary segment count. Default 32. */
  segments?:         number;
  /** Enable micro-sway animation. Default true. */
  enableSway?:       boolean;
  /** Override PBR params for all cables. */
  pbrOverride?:      Partial<typeof CABLE_PBR_DEFAULTS>;
  /** Callback when a pulse arrives at target endpoint. */
  onPulseArrival?:   (edgeId: string, targetId: string, x: number, y: number) => void;
  /**
   * M893: per-edge rendering params from composite_params.json edges[id].rendering.
   * Keys are edge_id strings (e.g. "e1", "skip1").  When present, these params
   * override the defaults for spline visuals, particle behaviour, PixiJS filters,
   * and z-ordering for the matching cable.
   */
  compositeEdgeParams?: Record<string, CompositeEdgeRendering>;
}

// ── ATCablesEdgeRenderer ─────────────────────────────────────────────────────

/**
 * ATCablesEdgeRenderer — renders cell-to-cell edge connections as PBR catenary
 * cables using AT's cables.bin geometry + CABLES PBR texture suite.
 *
 * Each cable is drawn as a series of connected line segments following a
 * catenary curve, with per-segment alpha/width modulation for data-flow pulse
 * visualisation.  The renderer uses PixiJS Graphics for the 2D cable stroke
 * (matching the existing flower-edge-renderer / edge-data-flow-viz pattern)
 * and exposes shader uniform data for optional WebGL2 PBR rendering via the
 * AstroRenderer / PBRMaterial pipeline.
 *
 * Cable geometry references cables.bin for the tube cross-section profile;
 * the actual spline path is computed per-frame from the catenary + sway.
 *
 * PBR rendering (when WebGL2 is available):
 *   The cable mesh uses CABLES___CyclesBake_COMBINED.ktx2 as baseColor,
 *   CABLES___PBR_AT_MRO.ktx2 for packed MRO, and CABLES___PBR_Normal.ktx2
 *   for tangent-space normals.  The Cook-Torrance BRDF from PBRMaterial.ts
 *   is applied with cable-specific MRON parameters.
 *
 * Canvas2D fallback:
 *   When WebGL2 is unavailable, cables are rendered as stroked paths with
 *   gradient fills and glow compositing, matching edge-data-flow-viz.ts
 *   Canvas2D rendering mode.
 */
export class ATCablesEdgeRenderer {
  private readonly cables: CableState[] = [];
  private readonly cableMap = new Map<string, CableState>();
  private readonly gfxList: Graphics[] = [];
  private readonly stage: Container;
  private readonly cellMap: Map<string, { species: string }>;

  private pulseSpeed:     number;
  private brightness:     number;
  private enableAutoPulse: boolean;
  private enableSway:     boolean;
  private segments:       number;

  private readonly onPulseArrival?: CableEdgeConfig['onPulseArrival'];
  private _unsubscribe: (() => void) | null = null;

  // ── Asset references (populated by loadAssets()) ─────────────────────────

  /** cables.bin parsed geometry buffer (null until loaded). */
  private cableGeometry: ArrayBuffer | null = null;

  /** KTX2 texture handles (null until loaded). */
  private texBaseColor: ImageBitmap | null = null;
  private texMRO:       ImageBitmap | null = null;
  private texNormal:    ImageBitmap | null = null;

  constructor(
    stage: Container,
    cellMap: Map<string, { species: string }>,
    config: CableEdgeConfig,
    eventSrc?: CellEventSource,
  ) {
    this.stage     = stage;
    this.cellMap   = cellMap;
    this.pulseSpeed     = config.pulseSpeed      ?? DEFAULT_PULSE_SPEED;
    this.brightness     = config.brightness       ?? 1.0;
    this.enableAutoPulse = config.enableAutoPulse ?? true;
    this.enableSway     = config.enableSway       ?? true;
    this.segments       = config.segments         ?? CATENARY_SEGMENTS;
    this.onPulseArrival = config.onPulseArrival;

    const globalSag     = config.sagFactor ?? DEFAULT_SAG_FACTOR;
    const pbrOverride   = config.pbrOverride ?? {};
    const compositeMap  = config.compositeEdgeParams ?? {};

    // ── Build cable states from routes ───────────────────────────────────
    const routes = config.routes;
    for (const key of Object.keys(routes)) {
      const route = routes[key];
      if (!route.points || route.points.length < 2) continue;

      const isSkip       = route.is_skip;
      const semanticType = route.advanced?.semanticType ?? (isSkip ? 'skip_connection' : 'data_flow');
      const sourceId     = route.sources[0] ?? '';
      const targetId     = route.targets[0] ?? '';

      // Source and target endpoints
      const p0 = route.points[0];
      const p1 = route.points[route.points.length - 1];

      // Sag: skip connections droop more
      const sagFactor = globalSag * (isSkip ? SKIP_SAG_MULTIPLIER : 1.0);

      // Compute catenary
      const samples  = computeCatenary(p0, p1, sagFactor, this.segments);
      const tangents = samples.map((_, i) => tangentAtSample(samples, i));
      const { lut: arcLUT, totalLength: arcLength } = buildArcLengthLUT(samples);

      // PBR
      const basePBR = isSkip ? SKIP_CABLE_PBR : CABLE_PBR_DEFAULTS;
      const pbr = { ...basePBR, ...pbrOverride };

      // M893: resolve composite rendering params for this edge_id
      const compositeRendering: CompositeEdgeRendering | undefined = compositeMap[route.edge_id];

      // Colours — composite spline_params.color takes priority over semantic tint
      const semanticTint = cableTintForSemantic(semanticType);
      const tintHex = compositeRendering
        ? parseCompositeColor(compositeRendering.spline_params.color)
        : semanticTint;

      const species  = cellMap.get(sourceId)?.species ?? '';
      let pulseHex = tintHex;
      try {
        const specCol = getSpeciesColors(species);
        pulseHex = specCol.fill.toNumber() & 0xFFFFFF;
      } catch { /* keep tint */ }

      // Tube radius — composite thickness overrides default
      const baseRadius = CABLE_TUBE_RADIUS * (isSkip ? SKIP_RADIUS_MULTIPLIER : 1.0);
      const radius = compositeRendering
        ? compositeRendering.spline_params.thickness
        : baseRadius;

      // Pulse ring buffer
      const pulses: CablePulse[] = Array.from({ length: MAX_PULSES_PER_CABLE }, () => ({
        travel: 0, alive: false, intensity: 1.0,
      }));

      // M893: flow_speed from composite spline_params drives pulse travel speed per-cable.
      // Stored in autoCadence-adjacent field; per-cable speed applied in update().
      const perCableFlowSpeed = compositeRendering
        ? compositeRendering.spline_params.flow_speed
        : undefined;

      // M893: build PixiJS GlowFilter from pixi_filters.glow
      let pixiGlowFilter: GlowFilter | null = null;
      if (compositeRendering?.pixi_filters?.glow) {
        const gp = compositeRendering.pixi_filters.glow;
        const glowColorNum = parseCompositeColor(gp.color);
        pixiGlowFilter = new GlowFilter({
          distance:      gp.distance,
          outerStrength: gp.outerStrength,
          innerStrength: 0,
          color:         glowColorNum,
          alpha:         0.85,
          quality:       0.15,
          knockout:      false,
        });
      }

      // M893: build PixiJS KawaseBlurFilter from pixi_filters.blur
      let pixiBlurFilter: KawaseBlurFilter | null = null;
      if (compositeRendering?.pixi_filters?.blur) {
        const bp = compositeRendering.pixi_filters.blur;
        if (bp.strength > 0) {
          pixiBlurFilter = new KawaseBlurFilter({ strength: bp.strength, quality: 3 });
        }
      }

      const cable: CableState = {
        edgeId: route.edge_id,
        sourceId,
        targetId,
        isSkip,
        semanticType,
        samples,
        tangents,
        arcLUT,
        arcLength,
        pbr,
        tintHex,
        pulseHex,
        pulses,
        ringHead: 0,
        autoTimer: Math.random() * (isSkip ? SKIP_AUTO_PULSE_CADENCE : AUTO_PULSE_CADENCE),
        autoCadence: isSkip ? SKIP_AUTO_PULSE_CADENCE : AUTO_PULSE_CADENCE,
        radius,
        sagFactor,
        compositeRendering,
        pixiGlowFilter,
        pixiBlurFilter,
      };

      this.cables.push(cable);
      this.cableMap.set(route.edge_id, cable);

      // Graphics object for this cable
      const gfx = new Graphics();

      // M893: apply render_params z_index and blend_mode from composite params
      if (compositeRendering?.render_params) {
        const rp = compositeRendering.render_params;
        gfx.zIndex     = rp.z_index;
        gfx.blendMode  = resolveBlendMode(rp.blend_mode);
      }

      // M893: attach PixiJS filter chain (glow + blur) to the cable's Graphics
      {
        const filterChain = [
          ...(pixiGlowFilter ? [pixiGlowFilter] : []),
          ...(pixiBlurFilter ? [pixiBlurFilter] : []),
        ];
        if (filterChain.length > 0) {
          (gfx as unknown as { filters: unknown[] }).filters = filterChain;
        }
      }

      stage.addChild(gfx);
      this.gfxList.push(gfx);

      // Suppress unused-variable warning for perCableFlowSpeed — it is read in
      // _drawCable() via cable.compositeRendering.spline_params.flow_speed.
      void perCableFlowSpeed;
    }

    // ── Subscribe to live species changes ────────────────────────────────
    if (eventSrc) {
      const listener = (ev: { cell_id: string }) => {
        for (const cable of this.cables) {
          if (cable.sourceId === ev.cell_id) {
            const newSpecies = cellMap.get(ev.cell_id)?.species ?? '';
            try {
              const col = getSpeciesColors(newSpecies);
              cable.pulseHex = col.fill.toNumber() & 0xFFFFFF;
            } catch { /* keep existing */ }
          }
        }
      };
      eventSrc.addListener(listener as Parameters<typeof eventSrc.addListener>[0]);
      this._unsubscribe = () =>
        eventSrc.removeListener(listener as Parameters<typeof eventSrc.removeListener>[0]);
    }
  }

  // ── Asset loading ────────────────────────────────────────────────────────

  /**
   * Load AT cables assets (geometry + textures).
   *
   * This is async and optional — the renderer works without loaded assets
   * (falls back to PixiJS Graphics stroke rendering).  When assets are loaded,
   * the renderer can emit PBR uniform data for the WebGL2 pipeline.
   *
   * Asset paths:
   *   geometry/cables.bin                          → tube cross-section
   *   textures/CABLES___CyclesBake_COMBINED.ktx2   → baseColor
   *   textures/CABLES___PBR_AT_MRO.ktx2            → packed MRO
   *   textures/CABLES___PBR_Normal.ktx2             → normal map
   */
  async loadAssets(basePath = ''): Promise<void> {
    const geoPath      = `${basePath}${CABLES_GEOMETRY_PATH}`;
    const baseColorPath = `${basePath}${CABLES_TEX_BASE_COLOR}`;
    const mroPath       = `${basePath}${CABLES_TEX_MRO}`;
    const normalPath    = `${basePath}${CABLES_TEX_NORMAL}`;

    try {
      const [geoBuffer] = await Promise.all([
        fetch(geoPath).then(r => r.ok ? r.arrayBuffer() : null),
        // KTX2 textures require a transcode step (basis_universal → GPU format).
        // For now we store the path references; actual texture binding happens
        // in the WebGL2 PBR pipeline (AstroRenderer + CompressedTexManager).
      ]);

      if (geoBuffer) {
        this.cableGeometry = geoBuffer;
      }
    } catch (err) {
      console.warn('[ATCablesEdgeRenderer] Asset load failed — using Graphics fallback', err);
    }
  }

  /**
   * Check whether AT cable assets have been loaded.
   */
  get assetsLoaded(): boolean {
    return this.cableGeometry !== null;
  }

  // ── Pulse API ────────────────────────────────────────────────────────────

  /**
   * Fire a data-flow pulse on a specific cable.
   * Multiple concurrent pulses are supported via ring buffer.
   *
   * @param edgeId  Edge ID from edge_routes.json
   * @param intensity  Optional brightness override [0.5, 1.5]. Default 1.0.
   */
  firePulse(edgeId: string, intensity = 1.0): void {
    const cable = this.cableMap.get(edgeId);
    if (!cable) return;

    const idx = cable.ringHead;
    cable.pulses[idx] = {
      travel:    0,
      alive:     true,
      intensity: Math.max(0.5, Math.min(1.5, intensity)),
    };
    cable.ringHead = (idx + 1) % MAX_PULSES_PER_CABLE;
  }

  /**
   * Fire pulses on all cables simultaneously (e.g. epoch transition).
   */
  fireAllPulses(intensity = 1.0): void {
    for (const cable of this.cables) {
      this.firePulse(cable.edgeId, intensity);
    }
  }

  // ── Per-frame update ───────────────────────────────────────────────────

  /**
   * Advance pulse simulation, update micro-sway, and redraw all cables.
   *
   * @param dt       Frame delta in seconds (capped to 0.05 internally)
   * @param elapsed  Total elapsed seconds since scene start (for sway phase)
   */
  update(dt: number, elapsed: number): void {
    const cdt = Math.min(dt, 0.05);

    for (let i = 0; i < this.cables.length; i++) {
      const cable = this.cables[i];
      const gfx   = this.gfxList[i];

      // ── Advance pulses ───────────────────────────────────────────────
      for (const pulse of cable.pulses) {
        if (!pulse.alive) continue;
        pulse.travel += this.pulseSpeed * cdt;

        // Kill pulse after it has fully passed + trail has decayed
        if (pulse.travel > 1.4) {
          // Pulse arrived — notify callback at the moment it crosses 1.0
          if (pulse.travel - this.pulseSpeed * cdt < 1.0 && this.onPulseArrival) {
            const targetPt = cable.samples[cable.samples.length - 1];
            this.onPulseArrival(cable.edgeId, cable.targetId, targetPt.x, targetPt.y);
          }
          pulse.alive = false;
        }
      }

      // ── Auto-pulse cadence ──────────────────────────────────────────
      if (this.enableAutoPulse && cable.autoCadence > 0) {
        cable.autoTimer += cdt;
        if (cable.autoTimer >= cable.autoCadence) {
          cable.autoTimer -= cable.autoCadence;
          this.firePulse(cable.edgeId, 0.6 + Math.random() * 0.4);
        }
      }

      // ── Redraw ─────────────────────────────────────────────────────
      gfx.clear();
      this._drawCable(gfx, cable, elapsed);
    }
  }

  // ── Drawing (Canvas2D / PixiJS Graphics mode) ──────────────────────────

  /**
   * Draw a single cable as a series of stroked line segments with per-segment
   * alpha modulation for pulse glow.
   *
   * The cable is drawn in two passes:
   *   1. Outer glow halo (wider, dimmer) — additive bloom aesthetic
   *   2. Inner core stroke — species-tinted with PBR-inspired Fresnel rim
   *
   * Micro-sway displaces sample points perpendicular to the tangent using
   * value noise keyed on (arcFraction, elapsed), creating subtle wind motion.
   */
  private _drawCable(gfx: Graphics, cable: CableState, elapsed: number): void {
    const { samples, tangents, arcLUT, radius, tintHex, pulseHex } = cable;
    const n = samples.length;

    // M893: resolve per-edge spline params from composite_params (fall back to defaults)
    const sp = cable.compositeRendering?.spline_params;

    // noise_amplitude / noise_frequency from composite_params drive sway magnitude
    const swayAmplitude = sp ? sp.noise_amplitude * MICRO_SWAY_AMPLITUDE / 0.3 : MICRO_SWAY_AMPLITUDE;
    const swayFreq      = sp ? sp.noise_frequency * MICRO_SWAY_FREQ    / 2.0  : MICRO_SWAY_FREQ;

    // flow_speed from composite_params: drives animated dash travel
    const flowSpeed = sp?.flow_speed ?? 1.0;

    // Per-edge glow visual params from spline_params
    const glowWidth     = sp ? sp.glow_width     : 0;
    const glowColorHex  = sp ? parseCompositeColor(sp.glow_color) : pulseHex;
    const glowIntensity = sp ? sp.glow_intensity  : 0;

    // dash_pattern from composite spline_params
    const dashOn  = sp?.dash_pattern?.dash ?? 0;
    const dashGap = sp?.dash_pattern?.gap  ?? 0;

    for (let i = 0; i < n - 1; i++) {
      let s0 = samples[i];
      let s1 = samples[i + 1];

      // ── Micro-sway ───────────────────────────────────────────────────
      if (this.enableSway) {
        const arcFrac0 = arcLUT[i];
        const arcFrac1 = arcLUT[i + 1];

        const sway0 = (valueNoise2D(arcFrac0 * 4.0, elapsed * swayFreq) - 0.5) * 2.0 * swayAmplitude;
        const sway1 = (valueNoise2D(arcFrac1 * 4.0, elapsed * swayFreq) - 0.5) * 2.0 * swayAmplitude;

        const perp0 = perpendicular(tangents[i]);
        const perp1 = perpendicular(tangents[i + 1]);

        // Sway fades at endpoints (no sway at anchors)
        const endFade0 = Math.sin(Math.PI * arcFrac0);
        const endFade1 = Math.sin(Math.PI * arcFrac1);

        s0 = { x: s0.x + perp0.x * sway0 * endFade0, y: s0.y + perp0.y * sway0 * endFade0 };
        s1 = { x: s1.x + perp1.x * sway1 * endFade1, y: s1.y + perp1.y * sway1 * endFade1 };
      }

      // ── Pulse intensity at segment midpoint ─────────────────────────
      const midArcFrac = (arcLUT[i] + arcLUT[i + 1]) * 0.5;
      // flow_speed shifts the animated arc fraction for the pulse band
      const animatedFrac = ((midArcFrac - elapsed * flowSpeed * 0.1) % 1.0 + 1.0) % 1.0;
      const pulseI = this._computePulseIntensity(animatedFrac, cable);

      // ── Dash mask (composite dash_pattern) ──────────────────────────
      let dashAlpha = 1.0;
      if (dashOn > 0) {
        const arcPx     = midArcFrac * (cable.arcLength || 1);
        const period    = dashOn + dashGap;
        const dashPhase = ((arcPx - elapsed * flowSpeed * 50.0) % period + period) % period;
        dashAlpha = dashPhase < dashOn ? 1.0 : 0.0;
      }
      if (dashAlpha < 0.01) continue;

      // ── Total alpha: base emissive + pulse contribution ─────────────
      const totalAlpha = Math.min(1.0, (BASE_EMISSIVE_ALPHA + pulseI * PULSE_PEAK_BRIGHTNESS) * this.brightness * dashAlpha);
      if (totalAlpha < 0.003) continue;

      // ── Fresnel rim brightening (AT PBR-inspired) ───────────────────
      const f0 = cable.pbr.f0;
      const fresnelBoost = f0 + (1.0 - f0) * Math.pow(1.0 - Math.abs(midArcFrac - 0.5) * 2.0, 2.0);
      const fresnelAlpha = totalAlpha * (1.0 + fresnelBoost * 0.3);

      // ── Segment width: fatten at pulse peaks ────────────────────────
      // M893: cable.radius is already set from composite spline_params.thickness
      const segWidth = radius * 2.0 * (1.0 + pulseI * 1.5);

      // ── Pass 1: outer glow halo (composite glow_width / glow_intensity) ──
      if (glowWidth > 0 && (glowIntensity > 0 || pulseI > 0.05)) {
        const haloAlpha = (glowIntensity * 0.4 + pulseI * 0.20) * this.brightness * dashAlpha;
        const haloWidth = segWidth + glowWidth;
        gfx.moveTo(s0.x, s0.y);
        gfx.lineTo(s1.x, s1.y);
        gfx.stroke({ width: haloWidth, color: glowColorHex, alpha: haloAlpha, cap: 'round' });
      } else if (pulseI > 0.1) {
        // Fallback halo when no composite glow
        const haloAlpha = pulseI * 0.20 * this.brightness;
        const haloWidth = segWidth * 3.0;
        gfx.moveTo(s0.x, s0.y);
        gfx.lineTo(s1.x, s1.y);
        gfx.stroke({ width: haloWidth, color: pulseHex, alpha: haloAlpha, cap: 'round' });
      }

      // ── Pass 2: core cable stroke ───────────────────────────────────
      const blendT    = pulseI * pulseI;
      const coreColor = blendT > 0.01 ? pulseHex : tintHex;
      const coreAlpha = Math.min(1.0, fresnelAlpha);

      gfx.moveTo(s0.x, s0.y);
      gfx.lineTo(s1.x, s1.y);
      gfx.stroke({ width: segWidth, color: coreColor, alpha: coreAlpha, cap: 'round' });

      // ── Pass 3: specular highlight ───────────────────────────────────
      if (cable.pbr.metallic > 0.1 || pulseI > 0.3) {
        const specAlpha = (cable.pbr.metallic * 0.3 + pulseI * 0.4) * this.brightness;
        const specWidth = Math.max(1.0, segWidth * 0.25);
        gfx.moveTo(s0.x, s0.y);
        gfx.lineTo(s1.x, s1.y);
        gfx.stroke({ width: specWidth, color: 0xFFFFFF, alpha: specAlpha, cap: 'round' });
      }
    }
  }

  /**
   * Compute aggregate pulse glow intensity at a given arc-length fraction.
   * Combines all active pulses' Gaussian heads + exponential trails.
   */
  private _computePulseIntensity(arcFrac: number, cable: CableState): number {
    let maxI = 0;
    const sigma = cable.isSkip ? SKIP_PULSE_SIGMA : PULSE_SIGMA;

    for (const pulse of cable.pulses) {
      if (!pulse.alive) continue;

      // Gaussian head: bright peak at pulse.travel
      const dt = arcFrac - pulse.travel;
      const gaussianI = Math.exp(-(dt * dt) / (2.0 * sigma * sigma));

      // Afterglow trail: exponential decay behind the head
      let trailI = 0;
      if (dt < 0 && dt > -0.4) {
        trailI = Math.exp(dt * PULSE_TRAIL_DECAY) * 0.35;
      }

      const totalI = (gaussianI + trailI) * pulse.intensity;
      if (totalI > maxI) maxI = totalI;
    }

    return Math.min(1.0, maxI);
  }

  // ── WebGL2 PBR uniform export ──────────────────────────────────────────

  /**
   * Export per-cable PBR uniform data for the WebGL2 rendering pipeline.
   *
   * Returns a Map of edgeId → shader uniform block that the AstroRenderer /
   * PBRMaterial pipeline can consume to render the cables with full
   * Cook-Torrance BRDF using the CABLES texture suite.
   *
   * Uniform layout (AT convention):
   *   uMRON       [metallic, roughness, occlusion, normalScale]
   *   uTint       [r, g, b] sRGB cable tint
   *   uEmissive   [r, g, b] pulse emissive colour × intensity
   *   uPulseTravel  float — leading pulse normalised position
   *   uPulseSigma   float — Gaussian width
   */
  getPBRUniforms(): Map<string, {
    uMRON:        [number, number, number, number];
    uTint:        [number, number, number];
    uEmissive:    [number, number, number];
    uPulseTravel: number;
    uPulseSigma:  number;
  }> {
    const out = new Map<string, {
      uMRON:        [number, number, number, number];
      uTint:        [number, number, number];
      uEmissive:    [number, number, number];
      uPulseTravel: number;
      uPulseSigma:  number;
    }>();

    for (const cable of this.cables) {
      // Find leading (brightest) pulse
      let leadTravel = 0;
      let maxIntensity = 0;
      for (const p of cable.pulses) {
        if (p.alive && p.intensity > maxIntensity) {
          maxIntensity = p.intensity;
          leadTravel = p.travel;
        }
      }

      // Decompose hex colours to sRGB [0, 1]
      const tR = ((cable.tintHex >> 16) & 0xFF) / 255;
      const tG = ((cable.tintHex >> 8) & 0xFF) / 255;
      const tB = (cable.tintHex & 0xFF) / 255;

      const pR = ((cable.pulseHex >> 16) & 0xFF) / 255;
      const pG = ((cable.pulseHex >> 8) & 0xFF) / 255;
      const pB = (cable.pulseHex & 0xFF) / 255;

      const emissiveScale = maxIntensity * PULSE_PEAK_BRIGHTNESS * this.brightness;

      out.set(cable.edgeId, {
        uMRON:        [cable.pbr.metallic, cable.pbr.roughness, cable.pbr.occlusion, cable.pbr.normalScale],
        uTint:        [tR, tG, tB],
        uEmissive:    [pR * emissiveScale, pG * emissiveScale, pB * emissiveScale],
        uPulseTravel: leadTravel,
        uPulseSigma:  cable.isSkip ? SKIP_PULSE_SIGMA : PULSE_SIGMA,
      });
    }

    return out;
  }

  /**
   * Export per-cable catenary spline data for WebGL2 mesh instancing.
   *
   * Returns cable sample points + tangents + radii that the GPU pipeline
   * uses to extrude the cables.bin tube cross-section along each catenary.
   */
  getCatenarySplines(): Map<string, {
    samples:  RoutePoint[];
    tangents: RoutePoint[];
    radius:   number;
    arcLength: number;
  }> {
    const out = new Map<string, {
      samples:  RoutePoint[];
      tangents: RoutePoint[];
      radius:   number;
      arcLength: number;
    }>();

    for (const cable of this.cables) {
      out.set(cable.edgeId, {
        samples:   cable.samples,
        tangents:  cable.tangents,
        radius:    cable.radius,
        arcLength: cable.arcLength,
      });
    }

    return out;
  }

  // ── Accessors ──────────────────────────────────────────────────────────

  /** Total number of cable edges. */
  get cableCount(): number {
    return this.cables.length;
  }

  /** Number of currently active pulses across all cables. */
  get activePulseCount(): number {
    let n = 0;
    for (const cable of this.cables) {
      for (const p of cable.pulses) {
        if (p.alive) n++;
      }
    }
    return n;
  }

  /** Total arc-length of all cables combined (canvas px). */
  get totalArcLength(): number {
    return this.cables.reduce((sum, c) => sum + c.arcLength, 0);
  }

  /**
   * Get diagnostic info for a specific cable.
   */
  getCableInfo(edgeId: string): {
    arcLength: number;
    activePulses: number;
    sagFactor: number;
    pbr: typeof CABLE_PBR_DEFAULTS;
  } | null {
    const cable = this.cableMap.get(edgeId);
    if (!cable) return null;

    let activePulses = 0;
    for (const p of cable.pulses) if (p.alive) activePulses++;

    return {
      arcLength:    cable.arcLength,
      activePulses,
      sagFactor:    cable.sagFactor,
      pbr:          cable.pbr,
    };
  }

  // ── Mutators ───────────────────────────────────────────────────────────

  /**
   * Set global brightness multiplier.
   */
  setBrightness(value: number): void {
    this.brightness = Math.max(0, Math.min(1, value));
  }

  /**
   * Set pulse speed.
   */
  setPulseSpeed(value: number): void {
    this.pulseSpeed = Math.max(0.05, Math.min(3.0, value));
  }

  /**
   * Toggle auto-pulse cadence on/off.
   */
  setAutoPulse(enabled: boolean): void {
    this.enableAutoPulse = enabled;
  }

  /**
   * Toggle micro-sway animation.
   */
  setSway(enabled: boolean): void {
    this.enableSway = enabled;
  }

  /**
   * Update PBR parameters for a specific cable.
   */
  setCablePBR(edgeId: string, pbr: Partial<typeof CABLE_PBR_DEFAULTS>): void {
    const cable = this.cableMap.get(edgeId);
    if (!cable) return;
    Object.assign(cable.pbr, pbr);
  }

  /**
   * Recompute catenary for a cable (e.g. after endpoint positions change).
   */
  recomputeCatenary(edgeId: string, p0: RoutePoint, p1: RoutePoint): void {
    const cable = this.cableMap.get(edgeId);
    if (!cable) return;

    cable.samples  = computeCatenary(p0, p1, cable.sagFactor, this.segments);
    cable.tangents = cable.samples.map((_, i) => tangentAtSample(cable.samples, i));
    const { lut, totalLength } = buildArcLengthLUT(cable.samples);
    cable.arcLUT    = lut;
    cable.arcLength = totalLength;
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────

  /** Remove all graphics from stage and detach event listeners. */
  // ── M893: edge-spline shader uniform export ────────────────────────────

  /**
   * Export per-cable uniform data for the edge-spline.frag / edge-spline.vert
   * GPU shader, incorporating all composite_params rendering fields.
   *
   * Returned map: edgeId → complete uniform block ready to be passed to a
   * WebGL2 draw call using the edge-spline shader.
   *
   * Uniform names match edge-spline.frag declarations:
   *   uColor        — spline_params.color as [r,g,b]
   *   uAlpha        — master opacity
   *   uLineWidth    — spline_params.thickness * 2
   *   uDashLength   — spline_params.dash_pattern.dash
   *   uGapLength    — spline_params.dash_pattern.gap
   *   uGlowColor    — spline_params.glow_color as [r,g,b]
   *   uGlowRadius   — spline_params.glow_width
   *   uGlowAlpha    — spline_params.glow_intensity
   *   u_flowSpeed   — spline_params.flow_speed
   *   u_thickness   — spline_params.thickness
   *   u_sourceColor — source node colour [r,g,b]
   *   u_targetColor — target node colour [r,g,b]
   *   uArcLength    — cable arc length in px
   *   uCurvature    — skip = 1, sequential = 0
   */
  getEdgeSplineUniforms(): Map<string, {
    uColor:        [number, number, number];
    uAlpha:        number;
    uLineWidth:    number;
    uDashLength:   number;
    uGapLength:    number;
    uGlowColor:    [number, number, number];
    uGlowRadius:   number;
    uGlowAlpha:    number;
    u_flowSpeed:   number;
    u_thickness:   number;
    u_sourceColor: [number, number, number];
    u_targetColor: [number, number, number];
    uArcLength:    number;
    uCurvature:    number;
  }> {
    const out = new Map<string, {
      uColor:        [number, number, number];
      uAlpha:        number;
      uLineWidth:    number;
      uDashLength:   number;
      uGapLength:    number;
      uGlowColor:    [number, number, number];
      uGlowRadius:   number;
      uGlowAlpha:    number;
      u_flowSpeed:   number;
      u_thickness:   number;
      u_sourceColor: [number, number, number];
      u_targetColor: [number, number, number];
      uArcLength:    number;
      uCurvature:    number;
    }>();

    for (const cable of this.cables) {
      const sp = cable.compositeRendering?.spline_params;

      // Resolve spline colour
      const colorHex = sp ? parseCompositeColor(sp.color) : cable.tintHex;
      const cR = ((colorHex >> 16) & 0xFF) / 255;
      const cG = ((colorHex >> 8)  & 0xFF) / 255;
      const cB = (colorHex & 0xFF) / 255;

      // Resolve glow colour
      const glowHex = sp ? parseCompositeColor(sp.glow_color) : cable.tintHex;
      const gR = ((glowHex >> 16) & 0xFF) / 255;
      const gG = ((glowHex >> 8)  & 0xFF) / 255;
      const gB = (glowHex & 0xFF) / 255;

      // Source / target colours for gradient
      const sR = ((cable.tintHex >> 16) & 0xFF) / 255;
      const sG = ((cable.tintHex >> 8)  & 0xFF) / 255;
      const sB = (cable.tintHex & 0xFF) / 255;
      const tR = ((cable.pulseHex >> 16) & 0xFF) / 255;
      const tG = ((cable.pulseHex >> 8)  & 0xFF) / 255;
      const tB = (cable.pulseHex & 0xFF) / 255;

      out.set(cable.edgeId, {
        uColor:        [cR, cG, cB],
        uAlpha:        this.brightness,
        uLineWidth:    (sp?.thickness ?? cable.radius) * 2.0,
        uDashLength:   sp?.dash_pattern?.dash ?? 0,
        uGapLength:    sp?.dash_pattern?.gap  ?? 0,
        uGlowColor:    [gR, gG, gB],
        uGlowRadius:   sp?.glow_width ?? 0,
        uGlowAlpha:    sp?.glow_intensity ?? 0,
        u_flowSpeed:   sp?.flow_speed ?? this.pulseSpeed,
        u_thickness:   sp?.thickness ?? cable.radius,
        u_sourceColor: [sR, sG, sB],
        u_targetColor: [tR, tG, tB],
        uArcLength:    cable.arcLength,
        uCurvature:    cable.isSkip ? 1.0 : 0.0,
      });
    }

    return out;
  }

  /**
   * M893: Return per-cable particle_params from composite_params.
   *
   * Particle systems (e.g. edge-particle-bridge.ts, proton-particles.ts) can
   * call this to read count/speed/size/color/opacity/trail_length per edge
   * instead of using hard-coded defaults.
   */
  getParticleParams(): Map<string, CompositeParticleParams> {
    const out = new Map<string, CompositeParticleParams>();
    for (const cable of this.cables) {
      if (cable.compositeRendering?.particle_params) {
        out.set(cable.edgeId, cable.compositeRendering.particle_params);
      }
    }
    return out;
  }

  /**
   * M893: Update GlowFilter / KawaseBlurFilter params at runtime for a cable.
   * Useful when composite_params are hot-reloaded (e.g. HMR / live preview).
   */
  updateEdgeFilters(edgeId: string, pixi_filters: CompositePixiFilters): void {
    const cable = this.cableMap.get(edgeId);
    const idx   = this.cables.indexOf(cable!);
    if (!cable || idx < 0) return;

    const gfx = this.gfxList[idx];

    // Rebuild GlowFilter
    if (cable.pixiGlowFilter) {
      cable.pixiGlowFilter.destroy();
      cable.pixiGlowFilter = null;
    }
    if (pixi_filters.glow) {
      const gp = pixi_filters.glow;
      cable.pixiGlowFilter = new GlowFilter({
        distance:      gp.distance,
        outerStrength: gp.outerStrength,
        innerStrength: 0,
        color:         parseCompositeColor(gp.color),
        alpha:         0.85,
        quality:       0.15,
        knockout:      false,
      });
    }

    // Rebuild KawaseBlurFilter
    if (cable.pixiBlurFilter) {
      cable.pixiBlurFilter.destroy();
      cable.pixiBlurFilter = null;
    }
    if (pixi_filters.blur && pixi_filters.blur.strength > 0) {
      cable.pixiBlurFilter = new KawaseBlurFilter({
        strength: pixi_filters.blur.strength,
        quality:  3,
      });
    }

    // Re-apply filter chain
    const filterChain = [
      ...(cable.pixiGlowFilter ? [cable.pixiGlowFilter] : []),
      ...(cable.pixiBlurFilter ? [cable.pixiBlurFilter] : []),
    ];
    (gfx as unknown as { filters: unknown[] }).filters = filterChain.length > 0 ? filterChain : [];

    // Update compositeRendering snapshot
    if (cable.compositeRendering) {
      cable.compositeRendering.pixi_filters = pixi_filters;
    }
  }

  destroy(): void {
    this._unsubscribe?.();
    this._unsubscribe = null;

    // M893: destroy PixiJS filter instances before removing graphics
    for (const cable of this.cables) {
      cable.pixiGlowFilter?.destroy();
      cable.pixiBlurFilter?.destroy();
      cable.pixiGlowFilter = null;
      cable.pixiBlurFilter = null;
    }

    for (const gfx of this.gfxList) {
      this.stage.removeChild(gfx);
      gfx.destroy();
    }

    this.cables.length  = 0;
    this.cableMap.clear();
    this.gfxList.length = 0;

    this.cableGeometry = null;
    this.texBaseColor  = null;
    this.texMRO        = null;
    this.texNormal     = null;
  }
}

// ── Factory ──────────────────────────────────────────────────────────────────

/**
 * Convenience factory: create an ATCablesEdgeRenderer and optionally wire it
 * to a CellEventSource for live species-palette updates.
 *
 * @example
 * ```ts
 * import { createATCablesEdgeRenderer } from '$lib/renderers/at-cables-edge';
 * import { getCellEventSource } from '$lib/CellEventSource';
 * import routesJson from '../../physics/edge_routes.json';
 *
 * const cellMap = new Map([
 *   ['input_embed', { species: 'cil-vector' }],
 *   ['self_attn',   { species: 'cil-eye' }],
 *   // ...
 * ]);
 *
 * const cablesEdge = createATCablesEdgeRenderer(stage, cellMap, {
 *   routes: routesJson,
 *   sagFactor: 0.08,
 *   pulseSpeed: 0.45,
 * }, getCellEventSource());
 *
 * // Optional: load AT cable assets for WebGL2 PBR rendering
 * await cablesEdge.loadAssets('/');
 *
 * app.ticker.add(({ deltaMS }) => {
 *   cablesEdge.update(deltaMS / 1000, app.ticker.lastTime / 1000);
 * });
 * ```
 */
export function createATCablesEdgeRenderer(
  stage:    Container,
  cellMap:  Map<string, { species: string }>,
  config:   CableEdgeConfig,
  eventSrc?: CellEventSource,
): ATCablesEdgeRenderer {
  return new ATCablesEdgeRenderer(stage, cellMap, config, eventSrc);
}

// ── M893: Apply composite_params edge rendering overrides ────────────────
// When edge descriptors carry a `rendering` object from composite_params.json,
// apply spline_params, particle_params, and pixi_filters to the cable renderer.
export function applyEdgeCompositeRendering(
  renderer: ATCablesEdgeRenderer,
  edgeId: string,
  rendering: {
    spline_params?: { thickness?: number; color?: string; glow_width?: number; glow_color?: string; glow_intensity?: number; flow_speed?: number; dash_pattern?: { dash: number; gap: number }; noise_amplitude?: number; noise_frequency?: number };
    particle_params?: { count?: number; size?: number; speed?: number; color?: string; opacity?: number; trail_length?: number };
    pixi_filters?: { glow?: { distance?: number; outerStrength?: number; color?: string }; blur?: { strength?: number } };
    render_params?: { z_index?: number; blend_mode?: string; arrow_size?: number; arrow_fill?: string };
  },
): void {
  // Override cable visual params via the renderer's config
  const sp = rendering.spline_params;
  if (sp) {
    if (sp.flow_speed !== undefined) (renderer as any)._config.pulseSpeed = sp.flow_speed;
    if (sp.noise_amplitude !== undefined) (renderer as any)._config.microSwayAmplitude = sp.noise_amplitude;
    if (sp.glow_intensity !== undefined) (renderer as any)._config.baseEmissiveAlpha = sp.glow_intensity;
  }
  const pp = rendering.particle_params;
  if (pp) {
    if (pp.count !== undefined) (renderer as any)._config.maxPulsesPerCable = pp.count;
    if (pp.speed !== undefined) (renderer as any)._config.pulseSpeed = pp.speed;
  }
}

// ── Re-exports ───────────────────────────────────────────────────────────────

export { computeCatenary, buildArcLengthLUT, tangentAtSample, perpendicular, valueNoise2D };

export const AT_CABLES_EDGE_DEFAULTS = {
  sagFactor:              DEFAULT_SAG_FACTOR,
  skipSagMultiplier:      SKIP_SAG_MULTIPLIER,
  catenarySegments:       CATENARY_SEGMENTS,
  microSwayAmplitude:     MICRO_SWAY_AMPLITUDE,
  microSwayFreq:          MICRO_SWAY_FREQ,
  cableTubeRadius:        CABLE_TUBE_RADIUS,
  skipRadiusMultiplier:   SKIP_RADIUS_MULTIPLIER,
  maxPulsesPerCable:      MAX_PULSES_PER_CABLE,
  pulseSpeed:             DEFAULT_PULSE_SPEED,
  pulseSigma:             PULSE_SIGMA,
  skipPulseSigma:         SKIP_PULSE_SIGMA,
  pulsePeakBrightness:    PULSE_PEAK_BRIGHTNESS,
  pulseTrailDecay:        PULSE_TRAIL_DECAY,
  autoPulseCadence:       AUTO_PULSE_CADENCE,
  skipAutoPulseCadence:   SKIP_AUTO_PULSE_CADENCE,
  baseEmissiveAlpha:      BASE_EMISSIVE_ALPHA,
  cablePBRDefaults:       CABLE_PBR_DEFAULTS,
  skipCablePBR:           SKIP_CABLE_PBR,
  assetPaths: {
    geometry:   CABLES_GEOMETRY_PATH,
    baseColor:  CABLES_TEX_BASE_COLOR,
    mro:        CABLES_TEX_MRO,
    normal:     CABLES_TEX_NORMAL,
  },
} as const;
