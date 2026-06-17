/**
 * pixi-cell-renderer.ts — Method 1: PixiJS Graphics cell renderer
 *
 * Cell 只输出参数（species、bbox、z），PixiJS 负责所有视觉：
 * - 圆角矩形 (Graphics.roundRect)
 * - Species 内部图案 (procedural Graphics draw calls)
 * - 每个 cell 背后的 bloom glow（BlurFilter + additive blend）
 * - 贝塞尔曲线连线 (Graphics.bezierCurveTo)
 * - Anti-aliasing（GPU 子像素平滑）
 *
 * Live Poll 模式 (pollCellChannels):
 * - 每 500ms fetch /api/cells 拉取最新 CellDescriptor[]
 * - 位置变化 → lerp 平滑过渡 (alpha += (target - current) * 0.1)
 * - 新 cell fade in (alpha 0→1)，消失 cell fade out (alpha 1→0) 后销毁
 * - edge layer 每帧跟着 cell 当前位置实时重绘
 *
 * M031: GlowFilter 外发光整合
 * M039: cil-eye SDF shader → PixiJS Filter
 *   buildCellContainer species==='cil-eye' 时挂载 CilEyeSDFFilter（sdf-species-filter.ts）
 *   到 Graphics pattern quad，替代 SPECIES_PATTERNS['cil-eye'] 画法。
 *   Ticker 驱动 container.__eyeFilter.time 做 SDF 径向光线旋转动画。
 * M045: cil-bolt SDF shader → PixiJS Filter
 *   buildCellContainer species==='cil-bolt' 时挂载 CilBoltSDFFilter（sdf-species-filter.ts）
 *   到 Graphics pattern quad，替代 SPECIES_PATTERNS['cil-bolt'] 画法。
 *   Ticker 驱动 container.__boltFilter.time 做 SDF 闪电动画。
 * M046: cil-vector / cil-plus / cil-arrow-right SDF Filters
 *   buildCellContainer 对应 species 挂载 SDF Filter 替代 SPECIES_PATTERNS Graphics 画法：
 *   cil-vector      → CilVectorSDFFilter   (arrow-grid, static)
 *   cil-plus        → CilPlusSDFFilter     (plus/cross SDF, Ticker-driven pulse, see M063)
 *   cil-arrow-right → CilArrowRightSDFFilter (tiled chevron scroll, Ticker-driven __arrowRightFilter)
 * M063: cil-plus SDF cross pulse animation
 *   CilPlusSDFFilter 升级添加 u_time + u_pulse_speed + u_pulse_amp 脉冲发光动画。
 *   u_cross_radius ↔ arm_length, u_cross_width ↔ stroke_width 从 species_params 读取。
 *   Ticker 驱动 container.__plusFilter.time 做十字呼吸脉冲效果。
 * M048: GodrayFilter per-species
 *   cil-vector → directional parallel GodrayFilter (angle 15°, high lacunarity)
 *   cil-bolt   → pulsing focal GodrayFilter (center=top, Ticker-driven gain oscillation)
 * - setGlow(container, 'hover' | 'select' | false) — hover/select 时叠加 GlowFilter
 * - GlowFilter 实例存于 container.__glowFilter，避免重复构造
 * - buildCellContainer() 预置 __glowFilter = null / __glowMode = null 槽位
 *
 * M050: cell hover tooltip
 *   __cellMeta 扩展携带 topology.in / topology.out (边数)，供 CellEventSystem
 *   tooltip 显示 label / species / 尺寸 / 连接数。hover 事件触发 setGlow('hover')
 *   在已有 M031 路径上无需额外改动；__cellMeta 结构变更向后兼容旧读取路径。
 *
 * Upstream reference:
 *   upstream/pixijs-engine/src/scene/graphics/shared/Graphics.ts
 *   upstream/pixijs-engine/src/filters/defaults/blur/
 *   upstream/pixijs-filters-v2/src/glow/GlowFilter.ts
 *   skills/pixijs/pixijs-filters/SKILL.md
 *   skills/pixijs/pixijs-graphics/SKILL.md
 */

import { Application } from '../../upstream/pixijs-engine/src/app/Application';
import { Container } from '../../upstream/pixijs-engine/src/scene/container/Container';
import { Graphics } from '../../upstream/pixijs-engine/src/scene/graphics/shared/Graphics';
import { Text } from '../../upstream/pixijs-engine/src/scene/text/Text';
import { TextStyle } from '../../upstream/pixijs-engine/src/scene/text/TextStyle';
import { Ticker } from '../../upstream/pixijs-engine/src/ticker/Ticker';
import { AdvancedBloomFilter } from '../../../upstream/pixijs-filters/src/advanced-bloom';
import { DropShadowFilter } from '../../../upstream/pixijs-filters/src/drop-shadow';
import { GodrayFilter } from '../../../upstream/pixijs-filters/src/godray';
import { MotionBlurFilter } from '../../../upstream/pixijs-filters/src/motion-blur';
import { AdjustmentFilter } from '../../../upstream/pixijs-filters/src/adjustment';
import { OutlineFilter } from '../../../upstream/pixijs-filters/src/outline';

// ── M031: GlowFilter — cell 选中/悬停外发光 ──────────────────────────────────
// GlowFilter 来自 upstream/pixijs-filters-v2/src/glow（pixi-filters-registry 已注册）。
// createCellGlow(mode) 按 hover/select 预设实例化，CellGlowMode 供 setGlow() 使用。
import { GlowFilter } from '../../../upstream/pixijs-filters-v2/src/glow';
import {
  createCellGlow,
  type CellGlowMode,
} from './pixi-filters-registry';

// ── M039: CilEyeSDFFilter — cil-eye SDF shader → PixiJS Filter ──────────────
//
// buildCellContainer 在 species === 'cil-eye' 时挂载此 Filter 到 pattern Graphics，
// 替代 SPECIES_PATTERNS['cil-eye'] Graphics 画法。
// Ticker 驱动 __eyeFilter.time 做径向光线旋转动画。
//
// ── M045: CilBoltSDFFilter — cil-bolt SDF shader → PixiJS Filter ─────────────
// 将 src/lib/shaders/cil-bolt.frag 的 SDF 闪电 shader 封装为 PixiJS Filter。
// buildCellContainer 在 species === 'cil-bolt' 时挂载此 Filter 到 pattern Graphics，
// 替代 SPECIES_PATTERNS['cil-bolt'] 的 Graphics 画法。
// Ticker 驱动 __boltFilter.time 做闪电动画。
//
// M046: CilVectorSDFFilter / CilPlusSDFFilter / CilArrowRightSDFFilter
// 同样来自 sdf-species-filter.ts，分别对应 cil-vector / cil-plus / cil-arrow-right。
// buildCellContainer 对应 species 时挂载：
//   cil-vector      → CilVectorSDFFilter (arrow-grid SDF, static)
//   cil-plus        → CilPlusSDFFilter   (plus/cross SDF, M063 Ticker-driven pulse animation)
//   cil-arrow-right → CilArrowRightSDFFilter (tiled scrolling chevron, Ticker-driven)
import { CilBoltSDFFilter, CilVectorSDFFilter, CilPlusSDFFilter, CilArrowRightSDFFilter, CilEyeSDFFilter } from './sdf-species-filter';

// ── Gaussian blur module (M007) ─────────────────────────────────────────────
// pixi-blur-cell adapts upstream/pixijs-engine BlurFilter for the cell render
// pipeline.  buildBloomFilterChain() places the pre-blur before the bloom
// compositor in the filter chain, matching the AdvancedBloomFilter internals
// (extract → gaussianBlur → composite) while giving us per-species tuning and
// Ticker-driven bloom-pulse animation.
import {
  buildBloomFilterChain,
  type CellBlurFilter,
} from './pixi-blur-cell';

// ── M017: cell-culling — viewport frustum skip ───────────────────────────────
// cullCells() sets container.visible = false for cells whose bbox is entirely
// outside the canvas screen rect, skipping the entire subtree render + filter
// evaluation for those cells.  Uses sharedCellCuller (64 px margin) so bloom
// halos don't pop in at the edge.  Mirrors upstream Culler._cullRecursive
// AABB test (upstream/pixijs-engine/src/culling/Culler.ts).
import { sharedCellCuller } from './cell-culling';

// ── params.json schema (M152 output) ───────────────────────────────────────

export interface ParamsJson {
  cell_id: string;
  species: string;
  bbox: { x: number; y: number; w: number; h: number; z?: number };
  z?: number;
  /** crowding_opacity — sets container.alpha */
  opacity?: number;
  /** hex string e.g. "#3F51B5" */
  fill_color?: string;
  /** hex string e.g. "#3F51B5" */
  stroke_color?: string;
  label?: string;
  font_size?: number;
  shadow?: {
    dx: number;
    dy: number;
    blur: number;
    opacity: number;
  };
  /**
   * bloom — AdvancedBloomFilter options emitted by postprocess_port.py emit_params().
   * Matches AdvancedBloomFilterOptions from upstream/pixijs-filters (AT HydraBloom port).
   *
   * bloom_variants.json naming (M044):
   *   bloomStrength      → maps to AdvancedBloomFilter.bloomScale
   *   radius             → maps to AdvancedBloomFilter.blur (Kawase kernel radius)
   *   luminosityThreshold → maps to AdvancedBloomFilter.threshold (alias for threshold)
   *
   * Keys: bloomStrength, radius, luminosityThreshold, threshold, bloomScale,
   *        brightness, blur, quality, pixelSize, tint, blurOrigin.
   */
  bloom?: {
    /**
     * AT bloom_variants.json "bloomStrength" — drives AdvancedBloomFilter.bloomScale.
     * Takes precedence over bloomScale when both are present.
     */
    bloomStrength?: number;
    /**
     * AT bloom_variants.json "bloomRadius" / "radius" — drives AdvancedBloomFilter.blur.
     * Interpreted as a 0–1 normalised radius; mapped to blur pixels via BLOOM_RADIUS_SCALE.
     * Takes precedence over blur when both are present.
     */
    radius?: number;
    /**
     * AT bloom_variants.json "luminosityThreshold" — alias for threshold.
     * Takes precedence over threshold when both are present.
     */
    luminosityThreshold?: number;
    threshold?: number;
    /** Direct AdvancedBloomFilter.bloomScale override (lower priority than bloomStrength). */
    bloomScale?: number;
    brightness?: number;
    /** Direct AdvancedBloomFilter.blur override (lower priority than radius). */
    blur?: number;
    quality?: number;
    pixelSize?: { x: number; y: number };
    /** [r, g, b] in 0–1 range — AT HydraBloom tint */
    tint?: [number, number, number];
    /** normalised screen-space light shaft origin */
    blurOrigin?: { x: number; y: number };
  };
  species_params?: Record<string, unknown>;
  epoch?: number;
}

// ── Cell descriptor — this is ALL the LLM needs to produce ──────────────────

export interface CellDescriptor {
  cell_id: string;
  label: string;
  species: string;
  bbox: { x: number; y: number; w: number; h: number };
  z: number;
  topology: {
    incoming_edges: string[];
    outgoing_edges: string[];
  };
  /** Optional params.json overlay (M152 output) — merged at build time or runtime */
  params?: ParamsJson;
}

export interface EdgeDescriptor {
  id: string;
  source: string;
  target: string;
  type: 'normal' | 'skip_connection';
}

// ── Species colour palette ──────────────────────────────────────────────────

const SPECIES_COLOURS: Record<string, { fill: number; stroke: number; glow: number }> = {
  'cil-eye':         { fill: 0x5C6BC0, stroke: 0x3949AB, glow: 0x7986CB },
  'cil-vector':      { fill: 0x66BB6A, stroke: 0x388E3C, glow: 0x81C784 },
  'cil-bolt':        { fill: 0xFFA726, stroke: 0xF57C00, glow: 0xFFCC80 },
  'cil-plus':        { fill: 0xEC407A, stroke: 0xC62828, glow: 0xF48FB1 },
  'cil-arrow-right': { fill: 0x78909C, stroke: 0x455A64, glow: 0xB0BEC5 },
  'cil-filter':      { fill: 0xAB47BC, stroke: 0x7B1FA2, glow: 0xCE93D8 },
  'cil-code':        { fill: 0x26A69A, stroke: 0x00796B, glow: 0x80CBC4 },
  'cil-layers':      { fill: 0x42A5F5, stroke: 0x1565C0, glow: 0x90CAF9 },
  'cil-loop':        { fill: 0xFFCA28, stroke: 0xF9A825, glow: 0xFFE082 },
  'cil-graph':       { fill: 0x78909C, stroke: 0x37474F, glow: 0xB0BEC5 },
};

function hexToNum(hex: string): number {
  return parseInt(hex.replace('#', ''), 16);
}

function getColours(
  species: string,
  fillOverride?: string,
  strokeOverride?: string,
): { fill: number; stroke: number; glow: number } {
  const base = SPECIES_COLOURS[species] ?? { fill: 0x90A4AE, stroke: 0x607D8B, glow: 0xB0BEC5 };
  return {
    fill:   fillOverride   ? hexToNum(fillOverride)   : base.fill,
    stroke: strokeOverride ? hexToNum(strokeOverride) : base.stroke,
    glow:   base.glow,
  };
}

// ── Species pattern drawers ─────────────────────────────────────────────────

type PatternDrawer = (g: Graphics, w: number, h: number, col: number, sp?: Record<string, unknown>) => void;

const SPECIES_PATTERNS: Record<string, PatternDrawer> = {
  'cil-eye': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2;
    const rOuter = (sp?.r_outer as number | undefined) ?? Math.min(w, h) * 0.35;
    const ringCount = (sp?.ring_count as number | undefined) ?? 3;
    const pupilRadius = (sp?.pupil_radius as number | undefined) ?? rOuter * 0.15;
    const rInnerRatio = (sp?.r_inner_ratio as number | undefined) ?? 0.3;
    for (let i = ringCount; i >= 1; i--) {
      g.circle(cx, cy, rOuter * (i / ringCount));
      g.fill({ color: col, alpha: 0.08 * i * (3 / ringCount) });
    }
    g.circle(cx, cy, pupilRadius);
    g.fill({ color: col, alpha: 0.5 });
    // draw iris line using r_inner_ratio
    g.circle(cx, cy, rOuter * rInnerRatio);
    g.stroke({ color: col, width: 0.8, alpha: 0.25 });
  },

  'cil-vector': (g, w, h, col, sp) => {
    const pad = 8;
    const arrowCount = (sp?.arrow_count as number | undefined) ?? 1;
    const arrowLength = (sp?.arrow_length as number | undefined) ?? (w - pad * 4);
    const angleSpread = (sp?.angle_spread as number | undefined) ?? 0;
    const cx = w / 2, cy = h / 2;
    const startX = cx - arrowLength / 2;
    for (let i = 0; i < arrowCount; i++) {
      const angle = arrowCount > 1 ? angleSpread * (i / (arrowCount - 1) - 0.5) : 0;
      const endX = startX + arrowLength * Math.cos(angle);
      const endY = cy + arrowLength * Math.sin(angle);
      g.moveTo(startX, cy);
      g.lineTo(endX, endY);
      g.moveTo(endX, endY);
      g.lineTo(endX - 6 * Math.cos(angle - 0.4), endY - 6 * Math.sin(angle - 0.4));
      g.moveTo(endX, endY);
      g.lineTo(endX - 6 * Math.cos(angle + 0.4), endY - 6 * Math.sin(angle + 0.4));
      g.stroke({ color: col, width: 1.5, alpha: 0.4 });
    }
  },

  'cil-bolt': (g, w, h, col, sp) => {
    const n = (sp?.zigzag_segments as number | undefined) ?? (sp?.num_rays as number | undefined) ?? 5;
    const dy = h / (n + 1), amp = w * 0.15;
    g.moveTo(w / 2, 6);
    for (let i = 1; i <= n; i++) {
      const x = w / 2 + (i % 2 === 1 ? amp : -amp);
      g.lineTo(x, dy * i + 6);
    }
    g.lineTo(w / 2, h - 6);
    g.stroke({ color: col, width: 1.5, alpha: 0.35 });
  },

  'cil-plus': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2;
    const arm = (sp?.arm_length as number | undefined) ?? Math.min(w, h) * 0.3;
    const strokeWidth = (sp?.stroke_width as number | undefined) ?? 2;
    const dashCorners = (sp?.dash_corners as boolean | undefined) ?? false;
    g.moveTo(cx - arm, cy); g.lineTo(cx + arm, cy);
    g.moveTo(cx, cy - arm); g.lineTo(cx, cy + arm);
    g.stroke({ color: col, width: strokeWidth, alpha: 0.3 });
    if (dashCorners) {
      const d = arm * 0.6;
      for (const [ox, oy] of [[-d,-d],[d,-d],[d,d],[-d,d]]) {
        g.circle(cx + ox, cy + oy, 1.5);
        g.fill({ color: col, alpha: 0.2 });
      }
    }
  },

  'cil-arrow-right': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2, sz = Math.min(w, h) * 0.25;
    g.moveTo(cx - sz, cy - sz);
    g.lineTo(cx + sz * 0.5, cy);
    g.lineTo(cx - sz, cy + sz);
    g.stroke({ color: col, width: 2, alpha: 0.4 });
  },

  'cil-filter': (g, w, h, col, sp) => {
    const pad = 10, gw = (w - pad * 2) / 3, gh = (h - pad * 2) / 3;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        g.rect(pad + c * gw + 1, pad + r * gh + 1, gw - 2, gh - 2);
        g.stroke({ color: col, width: 0.8, alpha: 0.2 });
      }
    }
  },

  'cil-code': (g, w, h, col, sp) => {
    const bx = 12, by = 8;
    g.moveTo(bx, by); g.lineTo(bx - 4, h / 2); g.lineTo(bx, h - by);
    g.moveTo(w - bx, by); g.lineTo(w - bx + 4, h / 2); g.lineTo(w - bx, h - by);
    g.stroke({ color: col, width: 1.5, alpha: 0.3 });
  },

  'cil-layers': (g, w, h, col, sp) => {
    for (let i = 0; i < 3; i++) {
      const off = i * 4;
      g.roundRect(6 + off, 6 + off, w - 12 - off * 2, h - 12 - off * 2, 3);
      g.stroke({ color: col, width: 1, alpha: 0.15 + i * 0.1 });
    }
  },

  'cil-loop': (g, w, h, col, sp) => {
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) * 0.3;
    g.arc(cx, cy, r, -Math.PI * 0.8, Math.PI * 0.5);
    g.stroke({ color: col, width: 1.5, alpha: 0.4 });
    const ax = cx + r * Math.cos(Math.PI * 0.5);
    const ay = cy + r * Math.sin(Math.PI * 0.5);
    g.moveTo(ax - 4, ay - 4); g.lineTo(ax, ay); g.lineTo(ax + 4, ay - 4);
    g.stroke({ color: col, width: 1.5, alpha: 0.4 });
  },

  'cil-graph': (g, w, h, col, sp) => {
    const pts = [[w*0.25, h*0.3], [w*0.6, h*0.25], [w*0.75, h*0.6], [w*0.35, h*0.7]];
    for (const [x, y] of pts) {
      g.circle(x, y, 3);
      g.fill({ color: col, alpha: 0.35 });
    }
    g.moveTo(pts[0][0], pts[0][1]); g.lineTo(pts[1][0], pts[1][1]);
    g.lineTo(pts[2][0], pts[2][1]); g.lineTo(pts[3][0], pts[3][1]);
    g.lineTo(pts[0][0], pts[0][1]);
    g.stroke({ color: col, width: 1, alpha: 0.2 });
  },
};

// ── Bloom glow factory ──────────────────────────────────────────────────────

/**
 * BLOOM_RADIUS_SCALE — normalised bloom_variants.json radius (0–1) →
 * AdvancedBloomFilter blur pixels.
 * radius 0 = no blur (2 px floor), radius 1 = 16 px.
 */
const BLOOM_RADIUS_SCALE = 16;

/**
 * BLOOM_DEFAULTS — per-species AdvancedBloomFilter defaults (M044).
 *
 * Values are calibrated to match the AT HydraBloom aesthetic for each species:
 *   bloomStrength  → AdvancedBloomFilter.bloomScale (intensity of bloom overlay)
 *   threshold      → luminance threshold for bloom extraction (0 = all glow, 1 = bright-only)
 *   radius         → normalised Kawase blur radius (0–1); multiplied by BLOOM_RADIUS_SCALE
 *
 * These defaults mirror bloom_variants.json scene presets mapped onto the cil-* species:
 *   cil-eye   (attention)   → intense, wide, low-threshold — like AT "home" scene
 *   cil-bolt  (FFN/MLP)     → strong, tight, medium-threshold — like AT "workbloom"
 *   cil-loop  (loop ctrl)   → warm moderate glow — like AT "sv_tree"
 *   cil-vector (embedding)  → wide diffuse glow — like AT "homebloom"
 *   cil-plus  (add&norm)    → subtle, almost none — like AT "globalbloom"
 *   cil-layers (layers)     → mid-depth stack glow — like AT "cleanroom"
 *   cil-filter (filter)     → moderate — like AT "sv_contact"
 *   cil-code  (code/output) → restrained — like AT "sv_footer"
 *   cil-graph (graph)       → minimal utility — like AT "sv_work"
 *   cil-arrow-right (flow)  → soft directional — like AT "sv_home"
 */
interface BloomDefaults {
  bloomStrength: number;
  threshold: number;
  radius: number;
  /** Base bloom pulse amplitude (fraction of bloomStrength); Ticker multiplies this. */
  pulseAmplitude: number;
  /** Pulse frequency in radians/second. */
  pulseFrequency: number;
}

const BLOOM_DEFAULTS: Record<string, BloomDefaults> = {
  // Attention (self-attn) — bright, radiant, wide halo
  'cil-eye': {
    bloomStrength:  2.0,
    threshold:      0.0,
    radius:         1.0,
    pulseAmplitude: 0.25,
    pulseFrequency: 1.2,
  },
  // FFN / MLP — electric, intense, fast flicker
  'cil-bolt': {
    bloomStrength:  1.8,
    threshold:      0.1,
    radius:         0.75,
    pulseAmplitude: 0.30,
    pulseFrequency: 2.0,
  },
  // Embedding / positional encoding — wide diffuse
  'cil-vector': {
    bloomStrength:  1.2,
    threshold:      0.0,
    radius:         0.80,
    pulseAmplitude: 0.15,
    pulseFrequency: 0.8,
  },
  // Add & Norm — neutral, very subtle
  'cil-plus': {
    bloomStrength:  0.3,
    threshold:      0.0,
    radius:         0.20,
    pulseAmplitude: 0.08,
    pulseFrequency: 0.6,
  },
  // Skip connection / routing — directional soft glow
  'cil-arrow-right': {
    bloomStrength:  0.6,
    threshold:      0.0,
    radius:         0.50,
    pulseAmplitude: 0.12,
    pulseFrequency: 0.9,
  },
  // Filter / selection — moderate, organised
  'cil-filter': {
    bloomStrength:  0.8,
    threshold:      0.0,
    radius:         0.50,
    pulseAmplitude: 0.15,
    pulseFrequency: 1.0,
  },
  // Code / output — restrained readability glow
  'cil-code': {
    bloomStrength:  0.7,
    threshold:      0.2,
    radius:         0.44,
    pulseAmplitude: 0.10,
    pulseFrequency: 0.7,
  },
  // Layers — mid-depth stack depth cue
  'cil-layers': {
    bloomStrength:  1.0,
    threshold:      0.2,
    radius:         0.625,
    pulseAmplitude: 0.12,
    pulseFrequency: 0.85,
  },
  // Loop / control flow — warm cycling glow
  'cil-loop': {
    bloomStrength:  0.8,
    threshold:      0.0,
    radius:         0.70,
    pulseAmplitude: 0.18,
    pulseFrequency: 1.4,
  },
  // Graph / topology — minimal utility
  'cil-graph': {
    bloomStrength:  0.5,
    threshold:      0.0,
    radius:         0.50,
    pulseAmplitude: 0.08,
    pulseFrequency: 0.6,
  },
};

/** Fallback bloom defaults for unknown species. */
const DEFAULT_BLOOM: BloomDefaults = {
  bloomStrength:  1.0,
  threshold:      0.1,
  radius:         0.5,
  pulseAmplitude: 0.15,
  pulseFrequency: 1.0,
};

/**
 * createGlowSprite — build the bloom halo behind each cell (M044).
 *
 * Resolves AdvancedBloomFilter parameters from three sources (highest → lowest priority):
 *   1. params.json bloom field (emitted by postprocess_port.py / bloom_variants.json):
 *        bloomStrength  → bloomScale   (M044: bloom_variants.json naming)
 *        luminosityThreshold → threshold  (M044 alias)
 *        radius         → blur pixels via BLOOM_RADIUS_SCALE
 *        bloomScale / threshold / blur → direct fallbacks
 *   2. BLOOM_DEFAULTS[species] — per-species calibrated defaults (M044 table)
 *   3. AdvancedBloomFilter library defaults
 *
 * The base bloomScale is stored on __bloomFilterBaseScale so the Ticker pulse
 * loop can modulate it without fighting the constructor value.
 */
function createGlowSprite(
  w: number,
  h: number,
  glowColor: number,
  species: string,
  bloomParams?: ParamsJson['bloom'],
): Graphics {
  const glow = new Graphics();
  const pad = 20;
  glow.roundRect(-pad, -pad, w + pad * 2, h + pad * 2, 12);
  glow.fill({ color: glowColor, alpha: 0.25 });

  // ── M044: Resolve AdvancedBloomFilter options ───────────────────────────
  // Per-species defaults from BLOOM_DEFAULTS table (mirrors bloom_variants.json)
  const speciesDefaults = BLOOM_DEFAULTS[species] ?? DEFAULT_BLOOM;

  // bloomStrength (bloom_variants.json naming) takes priority over bloomScale
  const resolvedBloomScale =
    bloomParams?.bloomStrength ??
    bloomParams?.bloomScale    ??
    speciesDefaults.bloomStrength;

  // luminosityThreshold (bloom_variants.json alias) takes priority over threshold
  const resolvedThreshold =
    bloomParams?.luminosityThreshold ??
    bloomParams?.threshold           ??
    speciesDefaults.threshold;

  // radius (bloom_variants.json, 0–1 normalised) → blur pixels
  // Direct blur override takes lowest priority
  const resolvedBlur = bloomParams?.radius !== undefined
    ? Math.max(2, bloomParams.radius * BLOOM_RADIUS_SCALE)
    : (bloomParams?.blur ?? Math.max(2, speciesDefaults.radius * BLOOM_RADIUS_SCALE));

  const bloom = new AdvancedBloomFilter({
    threshold:  resolvedThreshold,
    bloomScale: resolvedBloomScale,
    brightness: bloomParams?.brightness ?? 1.0,
    blur:       resolvedBlur,
    quality:    bloomParams?.quality    ?? 4,
    pixelSize:  bloomParams?.pixelSize  ?? { x: 1, y: 1 },
  });

  // ── M007: pre-blur Gaussian stage before bloom compositor ───────────────
  // buildBloomFilterChain() inserts a species-tuned CellBlurFilter *before*
  // AdvancedBloomFilter in the chain:
  //   glow texture → [CellBlurFilter] → [AdvancedBloomFilter compositor]
  const { filters, blurFilter } = buildBloomFilterChain(species, [bloom]);
  glow.filters = filters as any[];

  // ── Expose handles for Ticker-driven bloom pulse (M044) ─────────────────
  // __bloomFilter.bloomScale is modulated per-frame in the Ticker loop.
  // __bloomFilterBaseScale anchors the species/params base so the pulse is
  // a proportional oscillation rather than an absolute offset.
  (glow as any).__bloomBlur            = blurFilter;
  (glow as any).__bloomFilter          = bloom;
  (glow as any).__bloomFilterBaseScale = resolvedBloomScale;
  (glow as any).__bloomPulseAmplitude  = speciesDefaults.pulseAmplitude;
  (glow as any).__bloomPulseFrequency  = speciesDefaults.pulseFrequency;

  return glow;
}

// ── Cell container builder ──────────────────────────────────────────────────

function buildCellContainer(desc: CellDescriptor): Container {
  const { bbox, species, label, z } = desc;
  const { w, h } = bbox;

  // ── Resolve params.json overrides ──────────────────────────────────────
  const p = desc.params;
  const cols = getColours(species, p?.fill_color, p?.stroke_color);
  const speciesParams = p?.species_params;
  const crowdingOpacity = (p?.opacity !== undefined) ? Math.max(0.15, Math.min(1, p.opacity)) : 1;
  const fontSize = p?.font_size ?? 11;

  const container = new Container();
  container.position.set(bbox.x, bbox.y);
  container.zIndex = z;

  const glow = createGlowSprite(w, h, cols.glow, species, p?.bloom);
  container.addChild(glow);

  const body = new Graphics();
  body.roundRect(0, 0, w, h, 8);
  body.fill({ color: cols.fill, alpha: 0.9 });
  body.roundRect(0, 0, w, h, 8);
  body.stroke({ color: cols.stroke, width: 1.5, alpha: 0.8 });

  // ── DropShadowFilter from params.json shadow field ──────────────────────
  if (p?.shadow) {
    const { dx, dy, blur, opacity } = p.shadow;
    const dropShadow = new DropShadowFilter({
      offset: { x: dx, y: dy },
      blur,
      alpha: opacity,
      color: 0x000000,
      quality: 4,
    });
    // Compose with bloom on the body: [dropShadow, ...existing]
    body.filters = [dropShadow];
  }

  container.addChild(body);

  const pattern = new Graphics();
  if (species === 'cil-bolt') {
    // M045: cil-bolt → CilBoltSDFFilter replaces the Graphics draw path.
    // Draw a transparent background quad so the filter has a surface to paint on.
    // The SDF shader covers the entire [0,w]×[0,h] area and handles its own
    // coordinate mapping (vTextureCoord → [-1,1] NDC) internally.
    pattern.rect(0, 0, w, h);
    pattern.fill({ color: 0x000000, alpha: 0 }); // transparent — shader draws over it

    // Resolve fill colour from params.json or species palette
    const fc = cols.fill;
    const r = ((fc >> 16) & 0xff) / 255;
    const g = ((fc >>  8) & 0xff) / 255;
    const b = ( fc        & 0xff) / 255;

    const boltFilter = new CilBoltSDFFilter({
      fillColor:   [r, g, b],
      opacity:     1.0,
      zigzagCount: (speciesParams?.zigzag_segments as number | undefined)
                   ?? (speciesParams?.num_rays    as number | undefined)
                   ?? 6,
      amplitude:   (speciesParams?.amplitude as number | undefined) ?? 0.35,
      time:        0,
    });
    pattern.filters = [boltFilter];

    // Expose on container for Ticker-driven animation (see renderCellGraph / renderCellGraphLive)
    (container as any).__boltFilter = boltFilter;
  } else if (species === 'cil-vector') {
    // M046: cil-vector → CilVectorSDFFilter (arrow-grid SDF, replaces SPECIES_PATTERNS draw).
    // Transparent quad; shader renders arrow grid over the entire cell area.
    pattern.rect(0, 0, w, h);
    pattern.fill({ color: 0x000000, alpha: 0 });

    const fc = cols.fill;
    const r = ((fc >> 16) & 0xff) / 255;
    const g = ((fc >>  8) & 0xff) / 255;
    const b = ( fc        & 0xff) / 255;

    const vectorFilter = new CilVectorSDFFilter({
      fillColor:   [r, g, b],
      opacity:     1.0,
      arrowCount:  (speciesParams?.arrow_count  as number | undefined) ?? 4,
      angleSpread: (speciesParams?.angle_spread as number | undefined) ?? 0.4,
    });
    pattern.filters = [vectorFilter];

    // No Ticker animation needed (static grid); expose for external override if desired
    (container as any).__vectorFilter = vectorFilter;
  } else if (species === 'cil-plus') {
    // M063: cil-plus → CilPlusSDFFilter (plus/cross SDF, Ticker-driven pulse animation).
    // M046 baseline: transparent quad, SDF renders cross over the cell area.
    // M063 upgrade: adds u_time-driven glow pulse (u_cross_radius / u_cross_width uniforms).
    pattern.rect(0, 0, w, h);
    pattern.fill({ color: 0x000000, alpha: 0 });

    const fc = cols.fill;
    const r = ((fc >> 16) & 0xff) / 255;
    const g = ((fc >>  8) & 0xff) / 255;
    const b = ( fc        & 0xff) / 255;

    // arm_length / stroke_width from species_params if present; normalise pixel → [-1,1] NDC
    // SPECIES_PATTERNS uses Math.min(w,h)*0.3 for arm — translate to ~0.55 in NDC
    const armLengthPx   = speciesParams?.arm_length   as number | undefined;
    const strokeWidthPx = speciesParams?.stroke_width as number | undefined;
    const armLength   = armLengthPx   != null ? armLengthPx   / (Math.min(w, h) / 2) : 0.55;
    const strokeWidth = strokeWidthPx != null ? strokeWidthPx / (Math.min(w, h) / 2) : 0.12;

    // M063: pulse params from species_params; defaults calibrated for Add & Norm breathing rhythm
    const pulseSpeed = (speciesParams?.pulse_speed as number | undefined) ?? 2.0;
    const pulseAmp   = (speciesParams?.pulse_amp   as number | undefined) ?? 0.3;

    const plusFilter = new CilPlusSDFFilter({
      fillColor:   [r, g, b],
      opacity:     1.0,
      armLength,
      strokeWidth,
      time:       0,
      pulseSpeed,
      pulseAmp,
    });
    pattern.filters = [plusFilter];

    // Expose on container for Ticker-driven pulse animation (M063)
    (container as any).__plusFilter = plusFilter;
  } else if (species === 'cil-arrow-right') {
    // M046: cil-arrow-right → CilArrowRightSDFFilter (tiled scrolling chevron SDF).
    // Ticker drives __arrowRightFilter.time for horizontal scroll animation.
    pattern.rect(0, 0, w, h);
    pattern.fill({ color: 0x000000, alpha: 0 });

    const fc = cols.fill;
    const r = ((fc >> 16) & 0xff) / 255;
    const g = ((fc >>  8) & 0xff) / 255;
    const b = ( fc        & 0xff) / 255;

    const arrowRightFilter = new CilArrowRightSDFFilter({
      fillColor:  [r, g, b],
      opacity:    1.0,
      arrowWidth: (speciesParams?.arrow_width as number | undefined) ?? 0.08,
      time:       0,
    });
    pattern.filters = [arrowRightFilter];

    // Expose for Ticker-driven scroll animation
    (container as any).__arrowRightFilter = arrowRightFilter;
  } else if (species === 'cil-eye') {
    // M053: cil-eye → CilEyeSDFFilter (pupil + iris ring + radial rays + sclera halo SDF).
    // Ticker drives __eyeFilter.time for rotating radial ray animation.
    // Uniform完善: params.json species_params 完整映射到 CilEyeSDFFilter uniforms:
    //   ring_count    → numRays       (辐射光线数 / 同心环数)
    //   pupil_radius  → pupilRadius   (像素→NDC归一化: px / (min(w,h)/2))
    //   r_outer       → focalIntensity (外环半径→焦点强度: r_outer / min(w,h) * 2, clamp 0-2)
    //   r_inner_ratio → bloomRadius   (内环比率 → bloom 半径缩放, 直接映射)
    //   bloom_strength → bloomStrength (直接)
    //   ambient_intensity → ambientIntensity (直接)
    //   ambient_color  → ambientColor  ([r,g,b])
    //   light_exposure → lightExposure (直接)
    //   shadow_far     → shadowFar     (直接)
    //   shadow_bias    → shadowBias    (直接)
    pattern.rect(0, 0, w, h);
    pattern.fill({ color: 0x000000, alpha: 0 });

    const fc = cols.fill;
    const r = ((fc >> 16) & 0xff) / 255;
    const g = ((fc >>  8) & 0xff) / 255;
    const b = ( fc        & 0xff) / 255;

    // ── M053: params.json species_params → uniform mapping ─────────────────
    // Normalisation reference: min(w,h)/2 maps pixel-space radii to NDC half-space [0,1].
    const halfMin = Math.min(w, h) / 2;

    // ring_count: 同心环/辐射条数 (int) → numRays (shader u_numRays)
    // Fallback chain: ring_count → num_rays → default 8
    const numRaysVal =
      (speciesParams?.ring_count  as number | undefined) ??
      (speciesParams?.num_rays    as number | undefined) ??
      8;

    // pupil_radius: pixel space → NDC [-1,1] half-space
    // params.json pupil_radius=4.2 with h=50 → 4.2/25=0.168, clamped to [0.05, 0.6]
    const pupilRadiusPx  = (speciesParams?.pupil_radius as number | undefined);
    const pupilRadiusVal = pupilRadiusPx != null
      ? Math.max(0.05, Math.min(0.6, pupilRadiusPx / halfMin))
      : 0.22;

    // r_outer: outer ring pixel radius → focalIntensity scale
    // r_outer=21 with halfMin=25 → 21/25=0.84; map [0,1] → focalIntensity [0.3, 2.0]
    const rOuterPx       = (speciesParams?.r_outer as number | undefined);
    const focalIntensityVal = rOuterPx != null
      ? Math.max(0.3, Math.min(2.0, (rOuterPx / halfMin) * 1.5))
      : (speciesParams?.focal_intensity as number | undefined) ?? 1.0;

    // r_inner_ratio: inner ring ratio [0,1] → bloomRadius (ring width factor)
    // r_inner_ratio=0.3 maps to bloomRadius≈0.6 (multiply by 2 to span [0,1])
    const rInnerRatioVal = (speciesParams?.r_inner_ratio as number | undefined);
    const bloomRadiusVal = rInnerRatioVal != null
      ? Math.max(0.2, Math.min(2.0, rInnerRatioVal * 2.0))
      : (speciesParams?.bloom_radius as number | undefined) ?? 1.0;

    // ambient_color: hex string or [r,g,b] array from params
    const ambientColorRaw = speciesParams?.ambient_color;
    let ambientColorVal: [number, number, number] = [0.047, 0.929, 0.565]; // default #0bed90
    if (typeof ambientColorRaw === 'string' && ambientColorRaw.startsWith('#')) {
      const c = parseInt(ambientColorRaw.replace('#', ''), 16);
      ambientColorVal = [((c >> 16) & 0xff) / 255, ((c >> 8) & 0xff) / 255, (c & 0xff) / 255];
    } else if (Array.isArray(ambientColorRaw) && ambientColorRaw.length === 3) {
      ambientColorVal = ambientColorRaw as [number, number, number];
    }

    const eyeFilter = new CilEyeSDFFilter({
      fillColor:        [r, g, b],
      opacity:          1.0,
      // M053: 完整 uniform 映射
      numRays:          numRaysVal,
      pupilRadius:      pupilRadiusVal,
      focalIntensity:   focalIntensityVal,
      bloomStrength:    (speciesParams?.bloom_strength    as number | undefined) ?? 1.2,
      bloomRadius:      bloomRadiusVal,
      ambientIntensity: (speciesParams?.ambient_intensity as number | undefined) ?? 3.44,
      ambientColor:     ambientColorVal,
      lightExposure:    (speciesParams?.light_exposure    as number | undefined) ?? 0.86,
      shadowFar:        (speciesParams?.shadow_far        as number | undefined) ?? 40.0,
      shadowBias:       (speciesParams?.shadow_bias       as number | undefined) ?? 0.001,
      time:             0,
    });
    pattern.filters = [eyeFilter];

    // Expose for Ticker-driven ray rotation animation
    (container as any).__eyeFilter = eyeFilter;
  } else {
    const drawer = SPECIES_PATTERNS[species] ?? SPECIES_PATTERNS['cil-code'];
    drawer(pattern, w, h, cols.stroke, speciesParams);
  }
  container.addChild(pattern);

  const style = new TextStyle({
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize,
    fill: 0xFFFFFF,
    fontWeight: '500',
  });
  const txt = new Text({ text: label, style });
  txt.anchor.set(0.5);
  txt.position.set(w / 2, h / 2);
  container.addChild(txt);

  // ── Species post-process filters ────────────────────────────────────────
  const speciesFilters: any[] = [];

  if (species === 'cil-eye') {
    // GodrayFilter: focal light beams — gives "iris illumination" feel
    const godray = new GodrayFilter({
      angle: 30,
      gain: 0.4,
      lacunarity: 2.5,
      parallel: false,
      center: { x: w / 2, y: h / 2 },
      alpha: 0.6,
      time: 0,
    });
    speciesFilters.push(godray);
    // Animate godray time via ticker tag stored on container
    (container as any).__godray = godray;
  } else if (species === 'cil-vector') {
    // M048: GodrayFilter per-species — cil-vector (embedding / positional encoding).
    // Directional parallel rays emanating from upper-left at a shallow angle,
    // evoking a diffuse "information flow" shaft of light across the cell.
    // parallel=true uses angle (not a focal point), giving crisp directional rays
    // that suit the embedding arrow visual already drawn inside the cell.
    // Low gain + high lacunarity = many fine rays rather than one bright beam.
    const godray = new GodrayFilter({
      angle:      15,       // shallow diagonal — matches arrow direction in cil-vector pattern
      gain:       0.25,     // low intensity: subtle ambient rays, not overpowering
      lacunarity: 3.5,      // high lacunarity = dense, fine-grained ray texture
      parallel:   true,     // directional (angle-based), not focal
      alpha:      0.45,
      time:       0,
    });
    speciesFilters.push(godray);
    (container as any).__godray = godray;
  } else if (species === 'cil-bolt') {
    // M048: GodrayFilter per-species — cil-bolt (FFN / MLP).
    // Pulsing non-parallel rays from a focal point at cell top-centre,
    // simulating electric discharge bursting outward from the apex.
    // The focal point (center) tracks cell centre-top so rays splay radially.
    // gain and alpha are animated by the Ticker (see __godrayPulse* below).
    const godray = new GodrayFilter({
      angle:      0,        // unused when parallel=false; kept at default
      gain:       0.55,     // moderate base gain; Ticker will pulse this
      lacunarity: 2.0,      // fewer but wider rays = bold electric look
      parallel:   false,    // focal point mode: rays burst from center
      center:     { x: w / 2, y: 0 },   // apex of cell — radial burst downward
      alpha:      0.7,
      time:       0,
    });
    speciesFilters.push(godray);
    // Expose for Ticker-driven pulse animation
    (container as any).__godray              = godray;
    (container as any).__godrayPulseBaseGain = 0.55;   // base gain to oscillate around
    (container as any).__godrayPulseAmp      = 0.35;   // pulse amplitude (± fraction of base)
    (container as any).__godrayPulseFreq     = 3.5;    // fast pulse frequency (rad/s) — electric feel
  } else if (species === 'cil-layers') {
    // MotionBlurFilter: depth-of-field sense for layered stacking
    const motionBlur = new MotionBlurFilter({
      velocity: { x: 3, y: 1.5 },
      kernelSize: 5,
      offset: 0,
    });
    speciesFilters.push(motionBlur);
  }

  if (speciesFilters.length > 0) {
    // Compose with any existing body filters
    const existing = (container.filters as any[] | null) ?? [];
    container.filters = [...existing, ...speciesFilters];
  }

  // ── OutlineFilter slot (applied on hover/select via setOutline helper) ──
  (container as any).__outlineFilter = null;
  (container as any).__baseFilters = (container.filters as any[] | null) ?? [];

  // ── M031: GlowFilter slot — hover/select 外发光 ────────────────────────
  // GlowFilter 实例在首次 setGlow() 时按 mode 创建，后续复用同一实例以避免
  // GLSL 重编译（distance/quality 已烘焙进 GLSL loop）。
  // __glowFilter: 当前挂载的 GlowFilter 实例（null = 未激活）
  // __glowMode:   'hover' | 'select' | null — 标记当前 glow 类型，用于去重
  (container as any).__glowFilter = null as GlowFilter | null;
  (container as any).__glowMode   = null as CellGlowMode | null;

  // ── crowding_opacity: applies after fade-in completes ──────────────────
  // Store as userData so the poll loop can respect it when fading in
  (container as any).__targetAlpha = crowdingOpacity;
  container.alpha = 0; // caller sets to 0 for fade-in; will stop at __targetAlpha

  // ── CellMeta stamp — consumed by CellEventSystem.attachToCellContainers ─
  // Stamped here so CellEventSystem can auto-register without a separate lookup.
  // Mirrors AT's HitManager tagging pattern: object.__meta = descriptor.
  // M050: include topology edge counts so tooltip can display connectivity info.
  (container as any).__cellMeta = {
    cell_id:  desc.cell_id,
    label:    desc.label,
    species:  desc.species,
    bbox:     { ...desc.bbox },
    topology: {
      in:  desc.topology?.incoming_edges?.length ?? 0,
      out: desc.topology?.outgoing_edges?.length ?? 0,
    },
  };

  return container;
}

// ── Edge renderer ───────────────────────────────────────────────────────────

function drawEdges(
  g: Graphics,
  edges: EdgeDescriptor[],
  cellMap: Map<string, { x: number; y: number; w: number; h: number }>,
): void {
  g.clear();
  for (const edge of edges) {
    const src = cellMap.get(edge.source);
    const tgt = cellMap.get(edge.target);
    if (!src || !tgt) continue;

    const sx = src.x + src.w / 2;
    const sy = src.y + src.h;
    const tx = tgt.x + tgt.w / 2;
    const ty = tgt.y;

    if (edge.type === 'skip_connection') {
      const cx = Math.max(sx, tx) + 80;
      g.moveTo(sx, sy);
      g.bezierCurveTo(cx, sy, cx, ty, tx, ty);
      g.stroke({ color: 0x4CAF50, width: 2, alpha: 0.6 });
    } else {
      const mid_y = (sy + ty) / 2;
      g.moveTo(sx, sy);
      g.bezierCurveTo(sx, mid_y, tx, mid_y, tx, ty);
      g.stroke({ color: 0x999999, width: 1.5, alpha: 0.5 });
    }

    const angle = Math.atan2(ty - sy, tx - sx);
    const arrLen = 8;
    g.moveTo(tx - arrLen * Math.cos(angle - 0.4), ty - arrLen * Math.sin(angle - 0.4));
    g.lineTo(tx, ty);
    g.lineTo(tx - arrLen * Math.cos(angle + 0.4), ty - arrLen * Math.sin(angle + 0.4));
    g.stroke({ color: 0x999999, width: 1.5, alpha: 0.5 });
  }
}

// ── EdgeParticleSystem — WebGL2 Transform Feedback GPU 粒子流 ──────────────
//
// 架构：双缓冲 Ping-Pong Transform Feedback（参考 upstream/webgl2-particles-2）
//   - Simulation Pass: RASTERIZER_DISCARD + transformFeedback，更新粒子沿 bezier 路径的 t 值
//   - Render Pass: gl.POINTS，gl_PointSize 动态，fragment 输出发光点精灵
//   - 每条 edge 最多 PARTICLES_PER_EDGE 颗粒子，全部驻留 GPU 显存
//
// 粒子属性（per vertex，packed 进单 vec4）：
//   xyz = bezier 参数化坐标 (x, y, t)   w = lifetime [0..1]
//
// Simulation Shader:
//   - 沿 bezier 曲线步进 t += speed（速度略随机）
//   - t > 1.0 时粒子在起点复位，随机偏移以避免同步
//
// Render Shader:
//   - 坐标从 NDC 映射：position = (x/W*2-1, -(y/H*2-1))
//   - gl_PointSize 按 lifetime 曲线缩放（中段最大，两端收缩）
//   - fragment: 圆形点精灵 + 径向渐变发光

export class EdgeParticleSystem {
  private gl: WebGL2RenderingContext;
  private simProgram: WebGLProgram;
  private renderProgram: WebGLProgram;
  private tf: WebGLTransformFeedback;

  // Ping-Pong buffers — each holds all particle data as vec4 arrays
  private bufA: WebGLBuffer;
  private bufB: WebGLBuffer;
  private vaoA: WebGLVertexArrayObject;  // reads A, writes B
  private vaoB: WebGLVertexArrayObject;  // reads B, writes A
  private currentPing = 0; // 0 = read A write B, 1 = read B write A

  private particleCount = 0;
  private canvasW: number;
  private canvasH: number;

  // Per-edge bezier control points (uploaded as uniform arrays)
  private edgeUniforms: Float32Array; // [sx, sy, cx1, cy1, cx2, cy2, tx, ty] * MAX_EDGES
  private edgeCount = 0;

  static readonly PARTICLES_PER_EDGE = 32;
  static readonly MAX_EDGES          = 128;
  private static readonly VERT_FLOATS = 4; // vec4: x, y, t, lifetime

  // ── Shader sources ────────────────────────────────────────────────────────

  private static SIM_VERT = /* glsl */`#version 300 es
precision highp float;

// Input state (ping)
in vec4 a_particle; // x, y, t, lifetime

// Bezier edge data: packed as pairs of vec4
// [p0.xy, cp1.xy] and [cp2.xy, p1.xy] for each edge
uniform vec4 u_edges_a[${EdgeParticleSystem.MAX_EDGES}]; // p0.xy, cp1.xy
uniform vec4 u_edges_b[${EdgeParticleSystem.MAX_EDGES}]; // cp2.xy, p1.xy
uniform float u_speed_base;
uniform float u_time;

// Output state (pong) — captured by Transform Feedback
out vec4 v_particle;

// Simple LCG hash for per-particle pseudo-random speed jitter
float lcg(float seed) {
  return fract(sin(seed * 127.1 + 311.7) * 43758.5453);
}

// Cubic bezier position
vec2 bezier(vec2 p0, vec2 cp1, vec2 cp2, vec2 p1, float tt) {
  float u  = 1.0 - tt;
  float u2 = u * u;
  float u3 = u2 * u;
  float t2 = tt * tt;
  float t3 = t2 * tt;
  return u3*p0 + 3.0*u2*tt*cp1 + 3.0*u*t2*cp2 + t3*p1;
}

void main() {
  float px        = a_particle.x;
  float py        = a_particle.y;
  float t         = a_particle.z;
  float lifetime  = a_particle.w;

  // Each particle belongs to edge index = gl_VertexID / PARTICLES_PER_EDGE
  int edgeIdx = gl_VertexID / ${EdgeParticleSystem.PARTICLES_PER_EDGE};
  float particleSlot = float(gl_VertexID % ${EdgeParticleSystem.PARTICLES_PER_EDGE});

  vec4 ea = u_edges_a[edgeIdx]; // p0.xy, cp1.xy
  vec4 eb = u_edges_b[edgeIdx]; // cp2.xy, p1.xy
  vec2 p0  = ea.xy;
  vec2 cp1 = ea.zw;
  vec2 cp2 = eb.xy;
  vec2 p1  = eb.zw;

  // Per-particle speed jitter using LCG seeded by slot
  float jitter = 0.5 + lcg(particleSlot + float(edgeIdx) * 31.0) * 0.5;
  float speed  = u_speed_base * jitter;

  t += speed;

  // Respawn at staggered offsets so particles don't all bunch at start
  if (t > 1.0) {
    float phase = lcg(particleSlot * 17.0 + u_time * 0.001 + float(edgeIdx));
    t = phase * 0.35; // respawn in first third to maintain visual density
    lifetime = 0.0;
  } else {
    lifetime = min(1.0, lifetime + 0.04);
  }

  vec2 pos = bezier(p0, cp1, cp2, p1, clamp(t, 0.0, 1.0));

  v_particle = vec4(pos.x, pos.y, t, lifetime);
}
`;

  private static SIM_FRAG = /* glsl */`#version 300 es
precision highp float;
void main() {}  // RASTERIZER_DISCARD — fragment never executes
`;

  private static RENDER_VERT = /* glsl */`#version 300 es
precision highp float;

in vec4 a_particle; // x, y, t, lifetime

uniform vec2 u_resolution; // canvas W, H
uniform float u_point_size_max;
uniform vec4 u_color; // r, g, b, a base tint

out float v_lifetime;
out vec4  v_color;

void main() {
  float px = a_particle.x;
  float py = a_particle.y;
  float lifetime = a_particle.w;

  // Screen → NDC
  vec2 ndc = vec2(
    px / u_resolution.x * 2.0 - 1.0,
    -(py / u_resolution.y * 2.0 - 1.0)
  );
  gl_Position = vec4(ndc, 0.0, 1.0);

  // Size bell curve: peaks at lifetime ~ 0.5
  float bell = lifetime * (1.0 - lifetime) * 4.0;
  gl_PointSize = max(1.5, u_point_size_max * bell);

  v_lifetime = lifetime;
  v_color    = u_color;
}
`;

  private static RENDER_FRAG = /* glsl */`#version 300 es
precision highp float;

in float v_lifetime;
in vec4  v_color;
out vec4 fragColor;

void main() {
  // gl_PointCoord: [0,1]^2, center at (0.5, 0.5)
  vec2 uv   = gl_PointCoord - 0.5;
  float d   = dot(uv, uv); // squared distance from center
  if (d > 0.25) discard;   // clip to circle (r=0.5)

  // Radial glow: bright core, soft halo
  float core = 1.0 - smoothstep(0.0, 0.08, d);
  float halo = 1.0 - smoothstep(0.08, 0.25, d);
  float glow = core * 0.9 + halo * 0.3;

  fragColor = vec4(v_color.rgb * glow, v_color.a * glow * v_lifetime);
}
`;

  // ── Constructor ────────────────────────────────────────────────────────────

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: false,
      antialias: false,
    });
    if (!gl) throw new Error('[EdgeParticleSystem] WebGL2 not available');
    this.gl = gl;
    this.canvasW = canvas.width;
    this.canvasH = canvas.height;

    // ── Compile simulation program (with transform feedback) ──────────────
    this.simProgram = this._buildSimProgram();

    // ── Compile render program ─────────────────────────────────────────────
    this.renderProgram = this._compileProgram(
      EdgeParticleSystem.RENDER_VERT,
      EdgeParticleSystem.RENDER_FRAG,
    );

    // ── Transform Feedback object ──────────────────────────────────────────
    this.tf = gl.createTransformFeedback()!;

    // ── Allocate max ping-pong buffers (MAX_EDGES * PARTICLES_PER_EDGE * 4 floats) ──
    const maxParticles = EdgeParticleSystem.MAX_EDGES * EdgeParticleSystem.PARTICLES_PER_EDGE;
    const bytes = maxParticles * EdgeParticleSystem.VERT_FLOATS * 4;
    this.bufA = this._allocBuf(bytes);
    this.bufB = this._allocBuf(bytes);
    this.vaoA = this._makeVAO(this.bufA);
    this.vaoB = this._makeVAO(this.bufB);

    // ── Edge uniform scratch ───────────────────────────────────────────────
    this.edgeUniforms = new Float32Array(EdgeParticleSystem.MAX_EDGES * 8);
  }

  // ── Upload edge bezier data and seed particles ─────────────────────────────

  setEdges(
    edges: EdgeDescriptor[],
    cellMap: Map<string, { x: number; y: number; w: number; h: number }>,
  ): void {
    const { PARTICLES_PER_EDGE, MAX_EDGES, VERT_FLOATS } = EdgeParticleSystem;
    const validEdges: Array<{
      sx: number; sy: number;
      cx1: number; cy1: number;
      cx2: number; cy2: number;
      tx: number; ty: number;
      color: [number, number, number];
    }> = [];

    for (const edge of edges) {
      if (validEdges.length >= MAX_EDGES) break;
      const src = cellMap.get(edge.source);
      const tgt = cellMap.get(edge.target);
      if (!src || !tgt) continue;

      const sx  = src.x + src.w / 2;
      const sy  = src.y + src.h;
      const tx  = tgt.x + tgt.w / 2;
      const ty  = tgt.y;

      let cx1: number, cy1: number, cx2: number, cy2: number;
      if (edge.type === 'skip_connection') {
        const cx = Math.max(sx, tx) + 80;
        cx1 = cx; cy1 = sy;
        cx2 = cx; cy2 = ty;
      } else {
        const mid_y = (sy + ty) / 2;
        cx1 = sx; cy1 = mid_y;
        cx2 = tx; cy2 = mid_y;
      }

      const color: [number, number, number] = edge.type === 'skip_connection'
        ? [0.3, 0.85, 0.4]  // green tint for skip connections
        : [0.5, 0.75, 1.0]; // blue-white for normal edges

      validEdges.push({ sx, sy, cx1, cy1, cx2, cy2, tx, ty, color });
    }

    this.edgeCount    = validEdges.length;
    this.particleCount = this.edgeCount * PARTICLES_PER_EDGE;

    if (this.particleCount === 0) return;

    // Build packed u_edges_a / u_edges_b uniform data
    for (let i = 0; i < validEdges.length; i++) {
      const e = validEdges[i];
      const base = i * 8;
      this.edgeUniforms[base + 0] = e.sx;
      this.edgeUniforms[base + 1] = e.sy;
      this.edgeUniforms[base + 2] = e.cx1;
      this.edgeUniforms[base + 3] = e.cy1;
      this.edgeUniforms[base + 4] = e.cx2;
      this.edgeUniforms[base + 5] = e.cy2;
      this.edgeUniforms[base + 6] = e.tx;
      this.edgeUniforms[base + 7] = e.ty;
    }

    // Seed initial particle state with staggered t values
    const seed = new Float32Array(this.particleCount * VERT_FLOATS);
    for (let e = 0; e < validEdges.length; e++) {
      const ev = validEdges[e];
      for (let p = 0; p < PARTICLES_PER_EDGE; p++) {
        const idx  = (e * PARTICLES_PER_EDGE + p) * VERT_FLOATS;
        const t    = p / PARTICLES_PER_EDGE;  // evenly staggered
        // Evaluate bezier at initial t to get starting position
        const pos  = this._evalBezier(ev.sx, ev.sy, ev.cx1, ev.cy1, ev.cx2, ev.cy2, ev.tx, ev.ty, t);
        seed[idx + 0] = pos[0];
        seed[idx + 1] = pos[1];
        seed[idx + 2] = t;
        seed[idx + 3] = t; // lifetime starts at same phase
      }
    }

    // Upload to both buffers so first ping-pong has valid data
    const gl = this.gl;
    const byteLen = this.particleCount * VERT_FLOATS * 4;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufA);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, seed.subarray(0, this.particleCount * VERT_FLOATS));
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufB);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, seed.subarray(0, this.particleCount * VERT_FLOATS));
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Store per-edge colors for render pass (simple: use first edge color for all, or cycle)
    this._edgeColors = validEdges.map(e => e.color);
  }

  private _edgeColors: Array<[number, number, number]> = [];

  // ── Tick: one simulation step + one render pass ────────────────────────────

  tick(time: number): void {
    if (this.particleCount === 0) return;
    const gl = this.gl;

    // WebGL2 blending for additive glow
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive — brightens on overlap

    this._simPass(time);
    this._renderPass();

    gl.disable(gl.BLEND);
  }

  // ── Simulation pass (Transform Feedback) ──────────────────────────────────

  private _simPass(time: number): void {
    const gl = this.gl;
    const { PARTICLES_PER_EDGE, MAX_EDGES, VERT_FLOATS } = EdgeParticleSystem;

    gl.useProgram(this.simProgram);

    // Upload edge bezier data
    const locA = gl.getUniformLocation(this.simProgram, 'u_edges_a[0]');
    const locB = gl.getUniformLocation(this.simProgram, 'u_edges_b[0]');
    // Pack into two separate Float32Arrays (vec4 arrays)
    const ua = new Float32Array(MAX_EDGES * 4);
    const ub = new Float32Array(MAX_EDGES * 4);
    for (let i = 0; i < this.edgeCount; i++) {
      const base = i * 8;
      ua[i*4+0] = this.edgeUniforms[base+0]; // p0.x
      ua[i*4+1] = this.edgeUniforms[base+1]; // p0.y
      ua[i*4+2] = this.edgeUniforms[base+2]; // cp1.x
      ua[i*4+3] = this.edgeUniforms[base+3]; // cp1.y
      ub[i*4+0] = this.edgeUniforms[base+4]; // cp2.x
      ub[i*4+1] = this.edgeUniforms[base+5]; // cp2.y
      ub[i*4+2] = this.edgeUniforms[base+6]; // p1.x
      ub[i*4+3] = this.edgeUniforms[base+7]; // p1.y
    }
    gl.uniform4fv(locA, ua);
    gl.uniform4fv(locB, ub);
    gl.uniform1f(gl.getUniformLocation(this.simProgram, 'u_speed_base'), 0.004);
    gl.uniform1f(gl.getUniformLocation(this.simProgram, 'u_time'), time);

    // Bind read VAO and write buffer for TF
    const [readVAO, writeBuf] = this.currentPing === 0
      ? [this.vaoA, this.bufB]
      : [this.vaoB, this.bufA];

    gl.bindVertexArray(readVAO);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.tf);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, writeBuf);

    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, this.particleCount);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);

    // Flip ping-pong
    this.currentPing ^= 1;
  }

  // ── Render pass (gl.POINTS sprite) ────────────────────────────────────────

  private _renderPass(): void {
    const gl = this.gl;
    const { PARTICLES_PER_EDGE } = EdgeParticleSystem;

    gl.useProgram(this.renderProgram);
    gl.uniform2f(gl.getUniformLocation(this.renderProgram, 'u_resolution'), this.canvasW, this.canvasH);
    gl.uniform1f(gl.getUniformLocation(this.renderProgram, 'u_point_size_max'), 7.0);

    // Read VAO is now the pong side (we just wrote to it in sim pass)
    const readVAO = this.currentPing === 0 ? this.vaoA : this.vaoB;
    gl.bindVertexArray(readVAO);

    const colorLoc = gl.getUniformLocation(this.renderProgram, 'u_color');

    // Draw each edge's particle band with its own color
    for (let e = 0; e < this.edgeCount; e++) {
      const col = this._edgeColors[e] ?? [0.6, 0.8, 1.0];
      gl.uniform4f(colorLoc, col[0], col[1], col[2], 0.85);
      const offset = e * PARTICLES_PER_EDGE;
      gl.drawArrays(gl.POINTS, offset, PARTICLES_PER_EDGE);
    }

    gl.bindVertexArray(null);
  }

  // ── destroy ────────────────────────────────────────────────────────────────

  destroy(): void {
    const gl = this.gl;
    gl.deleteBuffer(this.bufA);
    gl.deleteBuffer(this.bufB);
    gl.deleteVertexArray(this.vaoA);
    gl.deleteVertexArray(this.vaoB);
    gl.deleteTransformFeedback(this.tf);
    gl.deleteProgram(this.simProgram);
    gl.deleteProgram(this.renderProgram);
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private _evalBezier(
    sx: number, sy: number,
    cx1: number, cy1: number,
    cx2: number, cy2: number,
    tx: number, ty: number,
    t: number,
  ): [number, number] {
    const u  = 1 - t;
    const u2 = u * u;
    const u3 = u2 * u;
    const t2 = t * t;
    const t3 = t2 * t;
    return [
      u3*sx + 3*u2*t*cx1 + 3*u*t2*cx2 + t3*tx,
      u3*sy + 3*u2*t*cy1 + 3*u*t2*cy2 + t3*ty,
    ];
  }

  private _allocBuf(bytes: number): WebGLBuffer {
    const gl = this.gl;
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, bytes, gl.DYNAMIC_COPY);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return buf;
  }

  private _makeVAO(buf: WebGLBuffer): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    // attrib 0: a_particle (vec4)
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return vao;
  }

  private _buildSimProgram(): WebGLProgram {
    const gl = this.gl;

    const vs = this._compileShader(gl.VERTEX_SHADER, EdgeParticleSystem.SIM_VERT);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, EdgeParticleSystem.SIM_FRAG);

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);

    // Declare Transform Feedback varyings BEFORE linking
    gl.transformFeedbackVaryings(prog, ['v_particle'], gl.SEPARATE_ATTRIBS);

    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('[EdgeParticleSystem] Sim program link error: ' + gl.getProgramInfoLog(prog));
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private _compileProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const gl = this.gl;
    const vs = this._compileShader(gl.VERTEX_SHADER, vertSrc);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('[EdgeParticleSystem] Render program link error: ' + gl.getProgramInfoLog(prog));
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private _compileShader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error('[EdgeParticleSystem] Shader compile error: ' + gl.getShaderInfoLog(sh));
    }
    return sh;
  }
}

// ── Live cell state (used by poll loop) ────────────────────────────────────

interface LiveCell {
  desc: CellDescriptor;
  /** current rendered position (lerp target) */
  curX: number;
  curY: number;
  /** target position from latest poll */
  tgtX: number;
  tgtY: number;
  container: Container;
  /** fade direction: +1 = fading in, -1 = fading out, 0 = stable */
  fadeDir: 0 | 1 | -1;
}

const LERP_FACTOR  = 0.1;   // position lerp per frame
const FADE_SPEED   = 0.05;  // alpha change per frame

// ── pollCellChannels ────────────────────────────────────────────────────────

/**
 * pollCellChannels — starts a 500ms polling loop against /api/cells.
 *
 * Behaviour:
 *   1. Every 500ms fetch /api/cells → CellDescriptor[]
 *   2. New cells: spawn container at alpha=0, fade in to 1
 *   3. Removed cells: fade out to 0, then destroy
 *   4. Existing cells: lerp position toward new bbox (alpha += (target-current)*0.1)
 *   5. Edge layer redraws every frame based on current live positions
 *
 * @param app        Running PixiJS Application
 * @param edges      EdgeDescriptor[] (static topology — edges don't change)
 * @param edgeLayer  Graphics node dedicated to edge drawing
 * @returns          stop() to cancel polling + animation
 */
export function pollCellChannels(
  app: Application,
  edges: EdgeDescriptor[],
  edgeLayer: Graphics,
  particleSystem?: EdgeParticleSystem | null,
): () => void {
  // Map of live cells keyed by cell_id
  const live = new Map<string, LiveCell>();

  let pollHandle: ReturnType<typeof setInterval> | null = null;
  let stopped = false;

  // ── Per-frame tick: lerp positions + fade + redraw edges ───────────────
  function tick(_ticker: Ticker): void {
    if (stopped) return;

    // Build bbox snapshot for edge drawing
    const bboxSnap = new Map<string, { x: number; y: number; w: number; h: number }>();

    for (const [id, lc] of live) {
      // Lerp position
      lc.curX += (lc.tgtX - lc.curX) * LERP_FACTOR;
      lc.curY += (lc.tgtY - lc.curY) * LERP_FACTOR;
      lc.container.position.set(lc.curX, lc.curY);

      // Fade
      const targetAlpha: number = (lc.container as any).__targetAlpha ?? 1;
      if (lc.fadeDir === 1) {
        lc.container.alpha = Math.min(targetAlpha, lc.container.alpha + FADE_SPEED);
        if (lc.container.alpha >= targetAlpha) lc.fadeDir = 0;
      } else if (lc.fadeDir === -1) {
        lc.container.alpha = Math.max(0, lc.container.alpha - FADE_SPEED);
        if (lc.container.alpha <= 0) {
          // Fully faded — destroy and remove
          app.stage.removeChild(lc.container);
          lc.container.destroy({ children: true });
          live.delete(id);
          continue;
        }
      }

      // Record current rendered bbox for edge drawing
      const { w, h } = lc.desc.bbox;
      bboxSnap.set(id, { x: lc.curX, y: lc.curY, w, h });
    }

    // ── M017: frustum culling — skip offscreen cells ──────────────────────
    // Run AFTER lerp so curX/curY reflect the current frame position.
    // sharedCellCuller sets container.visible = false for any cell whose bbox
    // is fully outside app.screen (+ 64 px bloom margin).  Invisible containers
    // skip the entire subtree render and all filter evaluations (godray / bloom
    // / bloom) at zero GPU cost.  Fade-out cells are exempt so the alpha
    // animation can complete before destroy().
    // __fadeDir is stamped onto container so CellCuller can read it:
    for (const [, lc] of live) {
      (lc.container as any).__fadeDir = lc.fadeDir;
    }
    sharedCellCuller.cull(live, app.screen);

    // Redraw edges at current (lerped) positions
    drawEdges(edgeLayer, edges, bboxSnap);

    // Sync particle paths to current cell positions (every frame — lerp smoothness)
    if (particleSystem && bboxSnap.size > 0) {
      particleSystem.setEdges(edges, bboxSnap);
    }
  }

  app.ticker.add(tick);

  // ── Poll loop: fetch /api/cells every 500ms ────────────────────────────
  async function fetchAndReconcile(): Promise<void> {
    if (stopped) return;
    try {
      const res = await fetch('/api/cells');
      if (!res.ok) return;
      const incoming: CellDescriptor[] = await res.json();

      const seen = new Set<string>();

      for (const desc of incoming) {
        seen.add(desc.cell_id);
        const tgtX = desc.bbox.x;
        const tgtY = desc.bbox.y;

        if (live.has(desc.cell_id)) {
          // Existing cell — update target position
          const lc = live.get(desc.cell_id)!;
          lc.tgtX = tgtX;
          lc.tgtY = tgtY;
          // Also update desc so edge dimensions stay correct
          lc.desc = desc;
          // Cancel any ongoing fade-out if cell reappears
          if (lc.fadeDir === -1) lc.fadeDir = 1;
        } else {
          // New cell — spawn at target, alpha=0, fade in
          const container = buildCellContainer(desc);
          container.alpha = 0;
          app.stage.addChild(container);

          const lc: LiveCell = {
            desc,
            curX: tgtX,
            curY: tgtY,
            tgtX,
            tgtY,
            container,
            fadeDir: 1,
          };
          live.set(desc.cell_id, lc);
        }
      }

      // Cells in live but NOT in incoming → fade out
      for (const [id, lc] of live) {
        if (!seen.has(id) && lc.fadeDir !== -1) {
          lc.fadeDir = -1;
        }
      }
    } catch (err) {
      console.warn('[pollCellChannels] fetch error:', err);
    }
  }

  // Initial fetch, then schedule
  fetchAndReconcile();
  pollHandle = setInterval(fetchAndReconcile, 500);

  // ── Return stop handle ─────────────────────────────────────────────────
  return () => {
    stopped = true;
    if (pollHandle !== null) clearInterval(pollHandle);
    app.ticker.remove(tick);
    if (particleSystem) particleSystem.destroy();
  };
}

// ── Main renderer (static, one-shot) ───────────────────────────────────────

export async function renderCellGraph(
  canvas: HTMLCanvasElement,
  cells: CellDescriptor[],
  edges: EdgeDescriptor[],
): Promise<Application> {
  const app = new Application();
  await app.init({
    canvas,
    width: canvas.width,
    height: canvas.height,
    backgroundColor: 0x1A1A2E,
    antialias: true,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  app.stage.sortableChildren = true;

  const cellMap = new Map<string, CellDescriptor>();
  for (const c of cells) cellMap.set(c.cell_id, c);

  // Build bbox map for edge drawing
  const bboxMap = new Map<string, { x: number; y: number; w: number; h: number }>();
  for (const c of cells) bboxMap.set(c.cell_id, c.bbox);

  // M047: Draw edges first — zIndex -1 guarantees edge layer is below all
  // cell containers (cell.z ≥ 1) regardless of incoming data.
  const edgeLayer = new Graphics();
  edgeLayer.zIndex = -1;
  drawEdges(edgeLayer, edges, bboxMap);
  app.stage.addChild(edgeLayer);

  // Draw cells
  for (const cell of cells) {
    const container = buildCellContainer(cell);
    // In static mode: skip fade-in, go straight to target alpha
    container.alpha = (container as any).__targetAlpha ?? 1;
    app.stage.addChild(container);
  }

  // ── GPU edge particle system (WebGL2 Transform Feedback) ───────────────
  try {
    const particleSystem = new EdgeParticleSystem(canvas);
    particleSystem.setEdges(edges, bboxMap);
    const t0 = performance.now();
    app.ticker.add(() => particleSystem.tick(performance.now() - t0));
    (app as any).__edgeParticleSystem = particleSystem;
  } catch (err) {
    // WebGL2 not available — silently degrade to static edges only
    console.warn('[renderCellGraph] EdgeParticleSystem unavailable:', err);
  }

  // ── Global AdjustmentFilter: subtle brightness/contrast/saturation lift ─
  const adjustment = new AdjustmentFilter({
    gamma: 1.0,
    contrast: 1.05,
    saturation: 1.1,
    brightness: 1.02,
    red: 1,
    green: 1,
    blue: 1,
  });
  app.stage.filters = [adjustment];

  // ── Animate per-species filters (godray time + M048 pulse, bloom pulse) ──
  //
  // M044 bloom pulse: per-frame AdvancedBloomFilter.bloomScale modulation.
  // M045: cil-bolt CilBoltSDFFilter time — __boltFilter.time driven per frame.
  // M046: cil-arrow-right CilArrowRightSDFFilter time — __arrowRightFilter.time per frame.
  // M063: cil-plus CilPlusSDFFilter time — __plusFilter.time driven per frame (cross pulse).
  // M048 godray pulse: cil-bolt __godrayPulse* fields drive gain oscillation.
  //   __bloomFilter          — AdvancedBloomFilter instance
  //   __bloomFilterBaseScale — species/params resolved base bloomScale
  //   __bloomPulseAmplitude  — species-calibrated pulse amplitude (BLOOM_DEFAULTS)
  //   __bloomPulseFrequency  — species-calibrated pulse frequency (rad/s)
  // The Ticker drives: bloomScale = base * (1 + amplitude * sin(freq * elapsed))
  // This gives each species a distinct breathing rhythm while keeping the
  // overall feel consistent with the AT HydraBloom animation pattern.
  const t0anim = performance.now();
  app.ticker.add(() => {
    const elapsed = (performance.now() - t0anim) / 1000;
    for (const child of app.stage.children) {
      // M045: cil-bolt SDF lightning animation — update CilBoltSDFFilter time
      const boltFilter = (child as any).__boltFilter as CilBoltSDFFilter | undefined;
      if (boltFilter) {
        boltFilter.time = elapsed;
      }

      // M046: cil-arrow-right SDF scroll animation — update CilArrowRightSDFFilter time
      const arrowRightFilter = (child as any).__arrowRightFilter as CilArrowRightSDFFilter | undefined;
      if (arrowRightFilter) {
        arrowRightFilter.time = elapsed;
      }

      // M063: cil-plus SDF cross pulse animation — update CilPlusSDFFilter time
      const plusFilter = (child as any).__plusFilter as CilPlusSDFFilter | undefined;
      if (plusFilter) {
        plusFilter.time = elapsed;
      }

      // M039: cil-eye SDF ray rotation animation — update CilEyeSDFFilter time
      const eyeFilter = (child as any).__eyeFilter as CilEyeSDFFilter | undefined;
      if (eyeFilter) {
        eyeFilter.time = elapsed;
      }

      const godray = (child as any).__godray as GodrayFilter | undefined;
      if (godray) {
        godray.time = elapsed;
        // M048: cil-bolt pulsing gain
        const pulseBaseGain = (child as any).__godrayPulseBaseGain as number | undefined;
        const pulseAmp      = (child as any).__godrayPulseAmp      as number | undefined;
        const pulseFreq     = (child as any).__godrayPulseFreq     as number | undefined;
        if (pulseBaseGain !== undefined) {
          godray.gain = pulseBaseGain * (1 + (pulseAmp ?? 0.35) * Math.sin((pulseFreq ?? 3.5) * elapsed));
        }
      }

      // M044: bloom pulse — animate AdvancedBloomFilter.bloomScale + pre-blur strength
      // First child of each cell container is always the glow sprite.
      const glowChild = (child as any).children?.[0];
      if (glowChild) {
        // M007: pre-blur Gaussian pulse (±20% strength oscillation on CellBlurFilter)
        const bloomBlur = (glowChild as any).__bloomBlur as CellBlurFilter | undefined;
        if (bloomBlur) {
          bloomBlur.setStrengthForBloom(1 + 0.2 * Math.sin(elapsed * 1.4));
        }

        // M044: AdvancedBloomFilter.bloomScale pulse — species-specific amplitude & frequency
        const bloomFilter = (glowChild as any).__bloomFilter as AdvancedBloomFilter | undefined;
        const baseScale   = (glowChild as any).__bloomFilterBaseScale as number | undefined;
        const pulseAmp    = (glowChild as any).__bloomPulseAmplitude  as number | undefined;
        const pulseFreq   = (glowChild as any).__bloomPulseFrequency  as number | undefined;
        if (bloomFilter && baseScale !== undefined) {
          const amp  = pulseAmp  ?? 0.15;
          const freq = pulseFreq ?? 1.0;
          bloomFilter.bloomScale = baseScale * (1 + amp * Math.sin(freq * elapsed));
        }
      }
    }
  });

  return app;
}

// ── Live poll renderer (uses pollCellChannels) ──────────────────────────────

/**
 * renderCellGraphLive — initialise a PixiJS canvas in live-poll mode.
 *
 * No initial cells are rendered; the poll loop populates the stage.
 * Returns both the Application and a stop() handle.
 */
export async function renderCellGraphLive(
  canvas: HTMLCanvasElement,
  edges: EdgeDescriptor[],
): Promise<{ app: Application; stop: () => void }> {
  const app = new Application();
  await app.init({
    canvas,
    width: canvas.width,
    height: canvas.height,
    backgroundColor: 0x1A1A2E,
    antialias: true,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  app.stage.sortableChildren = true;

  // M047: edgeLayer zIndex -1 — guaranteed below all cell containers (cell.z ≥ 1)
  const edgeLayer = new Graphics();
  edgeLayer.zIndex = -1;
  app.stage.addChild(edgeLayer);

  // ── GPU edge particle system (WebGL2 Transform Feedback) ───────────────
  let particleSystem: EdgeParticleSystem | null = null;
  try {
    particleSystem = new EdgeParticleSystem(canvas);
    const t0 = performance.now();
    app.ticker.add(() => particleSystem!.tick(performance.now() - t0));
  } catch (err) {
    console.warn('[renderCellGraphLive] EdgeParticleSystem unavailable:', err);
  }

  const stop = pollCellChannels(app, edges, edgeLayer, particleSystem);

  // ── Global AdjustmentFilter ─────────────────────────────────────────────
  const adjustment = new AdjustmentFilter({
    gamma: 1.0,
    contrast: 1.05,
    saturation: 1.1,
    brightness: 1.02,
    red: 1,
    green: 1,
    blue: 1,
  });
  app.stage.filters = [adjustment];

  // ── Animate per-species filters (godray time + pulse, bloom pulse) ──────────
  //
  // M044: live-mode bloom pulse — identical per-species pulse logic as
  // renderCellGraph (static mode).  AdvancedBloomFilter.bloomScale oscillates
  // around each cell's resolved base value using species-specific amplitude and
  // frequency stored at glow sprite build time (createGlowSprite).
  //
  // M045: cil-bolt CilBoltSDFFilter time — live mode drives __boltFilter.time.
  // M046: cil-arrow-right CilArrowRightSDFFilter time — live mode drives __arrowRightFilter.time.
  // M063: cil-plus CilPlusSDFFilter time — live mode drives __plusFilter.time (cross pulse).
  //
  // M048: cil-bolt godray gain pulse — same __godrayPulse* fields as static mode.
  const t0anim = performance.now();
  app.ticker.add(() => {
    const elapsed = (performance.now() - t0anim) / 1000;
    for (const child of app.stage.children) {
      // M045: cil-bolt SDF lightning animation — update CilBoltSDFFilter time
      const boltFilter = (child as any).__boltFilter as CilBoltSDFFilter | undefined;
      if (boltFilter) {
        boltFilter.time = elapsed;
      }

      // M046: cil-arrow-right SDF scroll animation — update CilArrowRightSDFFilter time
      const arrowRightFilter = (child as any).__arrowRightFilter as CilArrowRightSDFFilter | undefined;
      if (arrowRightFilter) {
        arrowRightFilter.time = elapsed;
      }

      // M063: cil-plus SDF cross pulse animation — update CilPlusSDFFilter time
      const plusFilter = (child as any).__plusFilter as CilPlusSDFFilter | undefined;
      if (plusFilter) {
        plusFilter.time = elapsed;
      }

      // M039: cil-eye SDF ray rotation animation — update CilEyeSDFFilter time
      const eyeFilter = (child as any).__eyeFilter as CilEyeSDFFilter | undefined;
      if (eyeFilter) {
        eyeFilter.time = elapsed;
      }

      const godray = (child as any).__godray as GodrayFilter | undefined;
      if (godray) {
        godray.time = elapsed;
        // M048: cil-bolt pulsing gain
        const pulseBaseGain = (child as any).__godrayPulseBaseGain as number | undefined;
        const pulseAmp      = (child as any).__godrayPulseAmp      as number | undefined;
        const pulseFreq     = (child as any).__godrayPulseFreq     as number | undefined;
        if (pulseBaseGain !== undefined) {
          godray.gain = pulseBaseGain * (1 + (pulseAmp ?? 0.35) * Math.sin((pulseFreq ?? 3.5) * elapsed));
        }
      }

      // M044: bloom pulse — glow sprite is always first child of each cell container
      const glowChild = (child as any).children?.[0];
      if (glowChild) {
        // M007: pre-blur Gaussian pulse
        const bloomBlur = (glowChild as any).__bloomBlur as CellBlurFilter | undefined;
        if (bloomBlur) {
          bloomBlur.setStrengthForBloom(1 + 0.2 * Math.sin(elapsed * 1.4));
        }

        // M044: AdvancedBloomFilter.bloomScale per-species pulse
        const bloomFilter = (glowChild as any).__bloomFilter as AdvancedBloomFilter | undefined;
        const baseScale   = (glowChild as any).__bloomFilterBaseScale as number | undefined;
        const pulseAmp    = (glowChild as any).__bloomPulseAmplitude  as number | undefined;
        const pulseFreq   = (glowChild as any).__bloomPulseFrequency  as number | undefined;
        if (bloomFilter && baseScale !== undefined) {
          const amp  = pulseAmp  ?? 0.15;
          const freq = pulseFreq ?? 1.0;
          bloomFilter.bloomScale = baseScale * (1 + amp * Math.sin(freq * elapsed));
        }
      }
    }
  });

  return { app, stop };
}

// ── setOutline — apply/remove OutlineFilter on hover or selection ───────────
//
// Usage:
//   setOutline(container, true, 0xFFFFFF, 2);  // highlight on hover
//   setOutline(container, false);               // remove outline
//
// Note: filter chain is rebuilt via _rebuildFilters() so that OutlineFilter
// and GlowFilter (M031) coexist without clobbering each other.
//
export function setOutline(
  container: Container,
  active: boolean,
  color: number = 0xFFFFFF,
  thickness: number = 2,
  alpha: number = 0.85,
): void {
  if (active) {
    // Avoid stacking duplicate outline filters
    if ((container as any).__outlineFilter) return;
    const outline = new OutlineFilter({ thickness, color, alpha, quality: 0.1 });
    (container as any).__outlineFilter = outline;
  } else {
    (container as any).__outlineFilter = null;
  }
  _rebuildFilters(container);
}

// ── setGlow — apply/remove GlowFilter on hover or selection (M031) ──────────
//
// GlowFilter 外发光：hover 时软青色外光，select 时金色强外光。
// 两种模式对应 pixi-filters-registry.ts 中的 CELL_GLOW_PRESETS。
//
// Filter 合成顺序（filter chain）:
//   [__baseFilters…, outlineFilter?, glowFilter]
// GlowFilter 放在末位保证外发光叠加在 outline 之外，视觉上最醒目。
//
// 实例复用策略：
//   同一 container 的 hover→hover 不重建；hover→select 销毁旧实例重建新的，
//   因为 distance/quality 已在 GLSL 编译时硬编码无法运行时改变。
//
// Usage:
//   setGlow(container, 'hover');    // hover 时外发光
//   setGlow(container, 'select');   // select 时外发光
//   setGlow(container, false);      // 移除外发光
//
export function setGlow(
  container: Container,
  mode: CellGlowMode | false,
): void {
  const prevGlow = (container as any).__glowFilter as GlowFilter | null;
  const prevMode = (container as any).__glowMode   as CellGlowMode | null;

  if (mode === false) {
    // ── Remove glow ──────────────────────────────────────────────────────
    if (!prevGlow) return;               // already off — no-op
    (container as any).__glowFilter = null;
    (container as any).__glowMode   = null;
    _rebuildFilters(container);
    return;
  }

  // ── Apply / update glow ─────────────────────────────────────────────────
  if (prevMode === mode && prevGlow) return;  // same mode already active — no-op

  // Destroy old instance if switching modes (distance baked in GLSL)
  if (prevGlow && prevMode !== mode) {
    prevGlow.destroy?.();
    (container as any).__glowFilter = null;
    (container as any).__glowMode   = null;
  }

  const glow = createCellGlow(mode);
  (container as any).__glowFilter = glow;
  (container as any).__glowMode   = mode;
  _rebuildFilters(container);
}

/**
 * _rebuildFilters — reassemble container.filters from slots after any change.
 *
 * Order: [...__baseFilters, outlineFilter?, glowFilter?]
 * Kept internal; setOutline / setGlow are the public API.
 */
function _rebuildFilters(container: Container): void {
  const base    = (container as any).__baseFilters    as any[]       ?? [];
  const outline = (container as any).__outlineFilter  as any | null;
  const glow    = (container as any).__glowFilter     as any | null;

  const chain: any[] = [...base];
  if (outline) chain.push(outline);
  if (glow)    chain.push(glow);

  container.filters = chain.length > 0 ? chain : null;
}

// ── HUD layer — pixijs-ui canvas-native controls (M170) ────────────────────
//
// HUD 组件使用 upstream/pixijs-ui 的原生 PixiJS 控件，直接在 canvas 内渲染：
//
//   Slider      — 控制 epoch 播放速度 (0.25×–4×)
//   ProgressBar — 显示收敛进度 (0–100%)
//   Button      — 手动收敛触发按钮 (带 Graphics 背景)
//   FancyButton — 工具栏按钮 (导出 / 暂停 / 复位)
//
// 所有组件通过纯 PixiJS Graphics 构建纹理，无需外部图片资源。
// HUD 层 zIndex 固定为 9999 保证在所有 cell/edge 层之上。

import { Slider } from '../../../upstream/pixijs-ui/src/Slider';
import { ProgressBar as UIProgressBar } from '../../../upstream/pixijs-ui/src/ProgressBar';
import { Button as UIButton, ButtonContainer } from '../../../upstream/pixijs-ui/src/Button';
import { FancyButton } from '../../../upstream/pixijs-ui/src/FancyButton';

export interface HUDState {
  /** Current epoch playback speed multiplier (0.25–4). */
  speedMultiplier: number;
  /** Convergence progress 0–100. */
  convergenceProgress: number;
  /** Whether epoch playback is currently paused. */
  paused: boolean;
}

export interface HUDCallbacks {
  onSpeedChange?: (multiplier: number) => void;
  onManualConverge?: () => void;
  onExport?: () => void;
  onPauseToggle?: (paused: boolean) => void;
  onReset?: () => void;
}

/**
 * HUDLayer — builds and manages a pixijs-ui HUD on top of the cell canvas.
 *
 * Layout (bottom toolbar, 40px tall):
 *   [⏸ Pause] [↺ Reset]  |  ───speed slider───  |  ══progress bar══  |  [⚡ Converge] [⬇ Export]
 *
 * @example
 * const hud = new HUDLayer(app, { speedMultiplier: 1, convergenceProgress: 0, paused: false });
 * hud.setCallbacks({ onSpeedChange: (v) => console.log('speed', v) });
 * // Update progress at any time:
 * hud.setProgress(42);
 */
export class HUDLayer {
  private app: Application;
  private container: Container;
  private slider!: Slider;
  private progressBar!: UIProgressBar;
  private pauseBtn!: FancyButton;
  private resetBtn!: FancyButton;
  private convergeBtn!: FancyButton;
  private exportBtn!: FancyButton;
  private speedLabel!: Text;
  private state: HUDState;
  private callbacks: HUDCallbacks = {};

  // HUD layout constants
  private static readonly HUD_H       = 48;
  private static readonly PAD         = 12;
  private static readonly BTN_W       = 80;
  private static readonly BTN_H       = 32;
  private static readonly SLIDER_W    = 160;
  private static readonly PROGRESS_W  = 180;
  private static readonly BG_COLOR    = 0x0D0D1A;
  private static readonly BG_ALPHA    = 0.82;
  private static readonly ACCENT      = 0x5C6BC0;
  private static readonly GREEN       = 0x43A047;
  private static readonly ORANGE      = 0xFFA726;
  private static readonly TEXT_STYLE  = {
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize: 11,
    fill: 0xCCCCDD,
    fontWeight: '500' as const,
  };

  constructor(app: Application, initialState: Partial<HUDState> = {}) {
    this.app = app;
    this.state = {
      speedMultiplier:    initialState.speedMultiplier    ?? 1,
      convergenceProgress: initialState.convergenceProgress ?? 0,
      paused:             initialState.paused             ?? false,
    };

    this.container = new Container();
    this.container.zIndex = 9999;
    this.container.label  = 'hud-layer';
    app.stage.addChild(this.container);

    this._buildBackground();
    this._buildSpeedSlider();
    this._buildProgressBar();
    this._buildButtons();
    this._layoutComponents();
  }

  // ── Public API ────────────────────────────────────────────────────────────

  setCallbacks(cb: HUDCallbacks): void {
    this.callbacks = { ...this.callbacks, ...cb };
  }

  /** Update convergence progress bar (0–100). */
  setProgress(pct: number): void {
    this.state.convergenceProgress = Math.max(0, Math.min(100, pct));
    this.progressBar.progress = this.state.convergenceProgress;
  }

  /** Programmatically set speed slider value. */
  setSpeed(multiplier: number): void {
    // Slider uses log scale: value 0–100 maps to 0.25×–4×
    this.state.speedMultiplier = multiplier;
    this.slider.value = this._speedToSlider(multiplier);
    this._updateSpeedLabel(multiplier);
  }

  /** Reposition HUD if canvas is resized. */
  resize(w: number, h: number): void {
    this._layoutComponents(w, h);
  }

  destroy(): void {
    this.app.stage.removeChild(this.container);
    this.container.destroy({ children: true });
  }

  // ── Private builders ──────────────────────────────────────────────────────

  private _buildBackground(): void {
    const bg = new Graphics();
    // Will be sized in _layoutComponents
    (this.container as any).__hudBg = bg;
    this.container.addChildAt(bg, 0);
  }

  private _buildSpeedSlider(): void {
    const { SLIDER_W, ACCENT } = HUDLayer;

    // Build bg/fill/thumb textures via Graphics → RenderTexture
    const bgG = new Graphics();
    bgG.roundRect(0, 0, SLIDER_W, 6, 3);
    bgG.fill({ color: 0x2A2A4A });

    const fillG = new Graphics();
    fillG.roundRect(0, 0, SLIDER_W, 6, 3);
    fillG.fill({ color: ACCENT });

    const thumbG = new Graphics();
    thumbG.circle(0, 0, 8);
    thumbG.fill({ color: 0xFFFFFF });
    thumbG.circle(0, 0, 5);
    thumbG.fill({ color: ACCENT });

    this.slider = new Slider({
      bg:     bgG,
      fill:   fillG,
      slider: thumbG,
      min:    0,
      max:    100,
      value:  this._speedToSlider(this.state.speedMultiplier),
      step:   1,
    });

    this.slider.onUpdate.connect((val) => {
      const mult = this._sliderToSpeed(val);
      this.state.speedMultiplier = mult;
      this._updateSpeedLabel(mult);
      this.callbacks.onSpeedChange?.(mult);
    });

    // Speed label
    this.speedLabel = new Text({
      text: this._formatSpeed(this.state.speedMultiplier),
      style: new TextStyle({ ...HUDLayer.TEXT_STYLE, fontSize: 10 }),
    });

    const sliderLabel = new Text({
      text: 'SPEED',
      style: new TextStyle({ ...HUDLayer.TEXT_STYLE, fontSize: 9, fill: 0x7777AA }),
    });

    (this.slider as any).__speedLabel    = this.speedLabel;
    (this.slider as any).__sliderLabel   = sliderLabel;

    this.container.addChild(sliderLabel);
    this.container.addChild(this.slider);
    this.container.addChild(this.speedLabel);
  }

  private _buildProgressBar(): void {
    const { PROGRESS_W, GREEN } = HUDLayer;

    const pbBgG = new Graphics();
    pbBgG.roundRect(0, 0, PROGRESS_W, 10, 5);
    pbBgG.fill({ color: 0x1E1E3A });

    const pbFillG = new Graphics();
    pbFillG.roundRect(0, 0, PROGRESS_W, 10, 5);
    pbFillG.fill({ color: GREEN });

    this.progressBar = new UIProgressBar({
      bg:       pbBgG,
      fill:     pbFillG,
      progress: this.state.convergenceProgress,
    });

    const pbLabel = new Text({
      text: 'CONVERGENCE',
      style: new TextStyle({ ...HUDLayer.TEXT_STYLE, fontSize: 9, fill: 0x7777AA }),
    });

    (this.progressBar as any).__pbLabel = pbLabel;
    this.container.addChild(pbLabel);
    this.container.addChild(this.progressBar);
  }

  private _buildButtons(): void {
    const { BTN_W, BTN_H, ACCENT, ORANGE, GREEN } = HUDLayer;

    // ── Helper: build a FancyButton with Graphics textures ─────────────
    const makeBtn = (
      label: string,
      baseColor: number,
      hoverColor: number,
    ): FancyButton => {
      const mkView = (color: number): Graphics => {
        const g = new Graphics();
        g.roundRect(0, 0, BTN_W, BTN_H, 6);
        g.fill({ color, alpha: 0.9 });
        g.roundRect(0, 0, BTN_W, BTN_H, 6);
        g.stroke({ color: 0xFFFFFF, width: 0.5, alpha: 0.15 });
        return g;
      };

      return new FancyButton({
        defaultView:  mkView(baseColor),
        hoverView:    mkView(hoverColor),
        pressedView:  mkView(hoverColor),
        text:         label,
        padding:      4,
        textOffset:   { default: { y: 0 } },
      });
    };

    // ⏸ Pause / ▶ Resume
    this.pauseBtn = makeBtn(
      this.state.paused ? '▶ Resume' : '⏸ Pause',
      0x2A2A50, 0x3A3A70,
    );
    this.pauseBtn.onPress.connect(() => {
      this.state.paused = !this.state.paused;
      // Update label via internal text accessor
      (this.pauseBtn as any).text = this.state.paused ? '▶ Resume' : '⏸ Pause';
      this.callbacks.onPauseToggle?.(this.state.paused);
    });

    // ↺ Reset
    this.resetBtn = makeBtn('↺ Reset', 0x2A3040, 0x3A4060);
    this.resetBtn.onPress.connect(() => {
      this.callbacks.onReset?.();
    });

    // ⚡ Manual Converge
    this.convergeBtn = makeBtn('⚡ Converge', 0x1A3320, 0x2A5030);
    this.convergeBtn.onPress.connect(() => {
      // Flash progress bar green momentarily
      this.progressBar.progress = Math.min(100, this.state.convergenceProgress + 5);
      this.callbacks.onManualConverge?.();
    });

    // ⬇ Export
    this.exportBtn = makeBtn('⬇ Export', 0x3A2010, 0x5A3820);
    this.exportBtn.onPress.connect(() => {
      this.callbacks.onExport?.();
    });

    this.container.addChild(this.pauseBtn);
    this.container.addChild(this.resetBtn);
    this.container.addChild(this.convergeBtn);
    this.container.addChild(this.exportBtn);
  }

  private _layoutComponents(
    w?: number,
    h?: number,
  ): void {
    const cw = w  ?? this.app.screen.width;
    const ch = h  ?? this.app.screen.height;
    const { HUD_H, PAD, BTN_W, BTN_H, SLIDER_W, PROGRESS_W } = HUDLayer;

    const barY = ch - HUD_H;

    // ── Background ────────────────────────────────────────────────────────
    const bg = (this.container as any).__hudBg as Graphics;
    if (bg) {
      bg.clear();
      bg.rect(0, barY, cw, HUD_H);
      bg.fill({ color: HUDLayer.BG_COLOR, alpha: HUDLayer.BG_ALPHA });
      // top border line
      bg.moveTo(0, barY);
      bg.lineTo(cw, barY);
      bg.stroke({ color: HUDLayer.ACCENT, width: 1, alpha: 0.3 });
    }

    const midY  = barY + HUD_H / 2;
    const labelH = 10;

    // ── Left: Pause + Reset buttons ───────────────────────────────────────
    let x = PAD;

    this.pauseBtn.position.set(x, midY - BTN_H / 2);
    x += BTN_W + PAD;

    this.resetBtn.position.set(x, midY - BTN_H / 2);
    x += BTN_W + PAD * 2;

    // ── Divider ───────────────────────────────────────────────────────────
    // (drawn into bg Graphics is sufficient; skip explicit line for simplicity)

    // ── Centre-left: Speed Slider ─────────────────────────────────────────
    const sliderLabel = (this.slider as any).__sliderLabel as Text | undefined;
    if (sliderLabel) {
      sliderLabel.position.set(x, barY + 7);
    }
    this.slider.position.set(x, midY - 3);

    const speedVal = (this.slider as any).__speedLabel as Text | undefined;
    if (speedVal) {
      speedVal.position.set(x + SLIDER_W + 6, midY - 5);
    }
    x += SLIDER_W + 46 + PAD * 2;

    // ── Centre-right: Progress Bar ────────────────────────────────────────
    const pbLabel = (this.progressBar as any).__pbLabel as Text | undefined;
    if (pbLabel) {
      pbLabel.position.set(x, barY + 7);
    }
    this.progressBar.position.set(x, midY - 5);
    x += PROGRESS_W + PAD * 2;

    // ── Right: Converge + Export ──────────────────────────────────────────
    // Pin to right edge
    const rightX = cw - PAD - BTN_W;
    this.exportBtn.position.set(rightX, midY - BTN_H / 2);
    this.convergeBtn.position.set(rightX - BTN_W - PAD, midY - BTN_H / 2);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  /** Map slider value 0–100 to speed multiplier 0.25–4 (log scale). */
  private _sliderToSpeed(val: number): number {
    // val 0→100 maps to log2(0.25)=-2 … log2(4)=2
    const logVal = -2 + (val / 100) * 4;
    return Math.round(Math.pow(2, logVal) * 100) / 100;
  }

  /** Inverse: speed multiplier → slider value 0–100. */
  private _speedToSlider(mult: number): number {
    const logVal = Math.log2(Math.max(0.25, Math.min(4, mult)));
    return Math.round(((logVal + 2) / 4) * 100);
  }

  private _formatSpeed(mult: number): string {
    return `${mult.toFixed(2)}×`;
  }

  private _updateSpeedLabel(mult: number): void {
    this.speedLabel.text = this._formatSpeed(mult);
  }
}

/**
 * attachHUD — convenience wrapper: creates a HUDLayer and wires it to the
 * given Application.  Returns the HUDLayer instance so callers can call
 * hud.setProgress(), hud.setSpeed(), hud.setCallbacks(), hud.resize() etc.
 *
 * @example
 * const { app, stop } = await renderCellGraphLive(canvas, edges);
 * const hud = attachHUD(app, { convergenceProgress: 0, speedMultiplier: 1 });
 * hud.setCallbacks({ onManualConverge: () => triggerConverge() });
 */
export function attachHUD(
  app: Application,
  initialState: Partial<HUDState> = {},
  callbacks: HUDCallbacks = {},
): HUDLayer {
  const hud = new HUDLayer(app, initialState);
  hud.setCallbacks(callbacks);

  // Auto-resize on window resize (optional but ergonomic)
  const onResize = () => hud.resize(app.screen.width, app.screen.height);
  window.addEventListener('resize', onResize);

  // Expose cleanup on the hud instance
  const origDestroy = hud.destroy.bind(hud);
  (hud as any).destroy = () => {
    window.removeEventListener('resize', onResize);
    origDestroy();
  };

  return hud;
}

// ── params.json loader helper ───────────────────────────────────────────────

/**
 * mergeParamsJson — fetches a params.json file for a given cell_id and
 * merges it into a CellDescriptor so that fill_color, stroke_color, shadow,
 * opacity, and species_params are all available to buildCellContainer.
 *
 * Usage:
 *   const cell = await mergeParamsJson(desc, '/channels/cell/self_attn/params.json');
 */
export async function mergeParamsJson(
  desc: CellDescriptor,
  paramsUrl: string,
): Promise<CellDescriptor> {
  try {
    const res = await fetch(paramsUrl);
    if (!res.ok) return desc;
    const p: ParamsJson = await res.json();
    return { ...desc, params: p };
  } catch {
    return desc;
  }
}
