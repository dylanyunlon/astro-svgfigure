/**
 * sdf-cell-renderer.ts — Method 3: SDF shader cell renderer
 *
 * 每个 cell 是一个 quad（两个三角形），fragment shader 里用距离场函数算形状：
 *   - sdRoundBox 画圆角矩形
 *   - sdCircle 画同心环
 *   - 正弦函数画 zigzag
 * 所有图案都是数学公式，分辨率无关——放大到任意倍数都不模糊。
 * 自带 glow（exp(-d*d*0.008)）。
 *
 * 这就是 MSDF 的原理，只是没走 msdfgen 二进制转换那步，直接在 shader 里手写 SDF。
 *
 * M071: cil-eye SDF Filter → PixiJS Mesh 挂载 + uniform injection from species_params
 *   species === 'cil-eye' 时，不再使用 inline speciesEye() SDF 函数，
 *   而是编译 cil-eye.frag 为 CilEyeSDFFilter (sdf-species-filter.ts)，
 *   挂载到 Graphics quad overlay 上。Uniforms 从 agent_params.json 的
 *   species_params 注入：ring_count → numRays, pupil_radius → pupilRadius (px→NDC),
 *   r_outer → focalIntensity, r_inner_ratio → bloomRadius, 以及 AT bloom/ambient/shadow。
 *   Ticker 驱动 __eyeFilter.time 做 SDF 径向光线旋转动画。
 *   参考 M039 (commit b7811e0) 的实现模式。
 *
 * Upstream reference:
 *   skills/pixijs/pixijs-custom-rendering/SKILL.md
 *   upstream/pixijs-engine/src/scene/mesh/shared/Mesh.ts
 */

import {
  Application,
  Container,
  Filter,
  Graphics,
  Mesh,
  MeshGeometry,
  Shader,
  UniformGroup,
  Text,
  TextStyle,
} from 'pixi.js';
// Resolved to upstream/pixijs-engine via tsconfig paths — no npm install needed

import type { CellDescriptor, EdgeDescriptor } from './pixi-cell-renderer';

// ── M071: CilEyeSDFFilter — cil-eye.frag SDF shader → PixiJS Filter ─────────
// For species === 'cil-eye', mount CilEyeSDFFilter on a Graphics quad overlay
// instead of using the inline speciesEye() SDF function.  Uniforms are injected
// from agent_params.json species_params (via CellDescriptor.params.species_params).
import {
  CilEyeSDFFilter,
  CilBoltSDFFilter,
  CilVectorSDFFilter,
  CilPlusSDFFilter,
  CilArrowRightSDFFilter,
} from './sdf-species-filter';
import { getSpeciesPalette } from './cell-color-palette';

// ── SDF Fragment Shader (GLSL ES 3.0) ───────────────────────────────────────
// All species patterns are pure math — resolution-independent.

const SDF_FRAGMENT = `#version 300 es
precision highp float;

in vec2 vUV;
out vec4 finalColor;

uniform float uTime;
uniform vec2  uSize;        // cell width, height in pixels
uniform vec3  uFillColor;   // species fill colour (RGB 0-1)
uniform vec3  uStrokeColor; // species stroke colour
uniform vec3  uGlowColor;   // bloom glow colour
uniform float uCornerRadius;
uniform int   uSpeciesId;   // 0=eye,1=vector,2=bolt,3=plus,4=arrow,5=filter,6=code,7=layers,8=loop,9=graph

// ── SDF primitives ──────────────────────────────────────────────────────────

float sdRoundBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float sdCircle(vec2 p, float r) {
    return length(p) - r;
}

float sdLine(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// ── Species SDF patterns ────────────────────────────────────────────────────

float speciesEye(vec2 uv, vec2 sz) {
    // Concentric circles — radial heatmap
    float r = min(sz.x, sz.y) * 0.35;
    float d = sdCircle(uv, r);
    float rings = sin(d * 25.0) * 0.5 + 0.5;
    float pupil = smoothstep(r * 0.2, r * 0.15, length(uv));
    return mix(rings * 0.3, 1.0, pupil);
}

float speciesVector(vec2 uv, vec2 sz) {
    // Arrow pointing right
    float arrow = sdLine(uv, vec2(-sz.x * 0.35, 0.0), vec2(sz.x * 0.25, 0.0));
    float head1 = sdLine(uv, vec2(sz.x * 0.25, 0.0), vec2(sz.x * 0.15, sz.y * 0.15));
    float head2 = sdLine(uv, vec2(sz.x * 0.25, 0.0), vec2(sz.x * 0.15, -sz.y * 0.15));
    float d = min(arrow, min(head1, head2));
    return smoothstep(2.0, 0.5, d) * 0.4;
}

float speciesBolt(vec2 uv, vec2 sz) {
    // Zigzag activation function
    float x_norm = (uv.x / sz.x + 0.5);
    float zigzag = sin(x_norm * 3.14159 * 4.0) * sz.y * 0.2;
    float d = abs(uv.y - zigzag);
    return smoothstep(3.0, 0.5, d) * 0.35;
}

float speciesPlus(vec2 uv, vec2 sz) {
    // Cross
    float arm = min(sz.x, sz.y) * 0.3;
    float dh = sdLine(uv, vec2(-arm, 0.0), vec2(arm, 0.0));
    float dv = sdLine(uv, vec2(0.0, -arm), vec2(0.0, arm));
    float d = min(dh, dv);
    return smoothstep(2.5, 0.5, d) * 0.3;
}

float speciesFilter(vec2 uv, vec2 sz) {
    // 3x3 grid
    float gx = fract((uv.x / sz.x + 0.5) * 3.0);
    float gy = fract((uv.y / sz.y + 0.5) * 3.0);
    float gridLine = min(
        smoothstep(0.05, 0.0, gx) + smoothstep(0.95, 1.0, gx),
        1.0
    );
    float gridLine2 = min(
        smoothstep(0.05, 0.0, gy) + smoothstep(0.95, 1.0, gy),
        1.0
    );
    return max(gridLine, gridLine2) * 0.2;
}

float speciesLoop(vec2 uv, vec2 sz) {
    // Arc
    float r = min(sz.x, sz.y) * 0.3;
    float d = abs(sdCircle(uv, r));
    float angle = atan(uv.y, uv.x);
    float mask = smoothstep(-2.5, 1.5, angle); // partial arc
    return smoothstep(2.5, 0.5, d) * mask * 0.4;
}

// ── Main fragment ───────────────────────────────────────────────────────────

void main() {
    // Map UV to cell-local coordinates (centered)
    vec2 uv = (vUV - 0.5) * uSize;
    vec2 halfSize = uSize * 0.5;

    // 1. Cell body — rounded rectangle SDF
    float d = sdRoundBox(uv, halfSize - 1.0, uCornerRadius);

    // 2. Fill + stroke
    float fill = smoothstep(1.0, -1.0, d);                  // anti-aliased fill
    float stroke = smoothstep(2.0, 0.5, abs(d)) * 0.8;      // anti-aliased stroke

    // 3. Glow — exp(-d²) falloff, extends BEYOND the cell bbox
    float glow = exp(-d * d * 0.008) * 0.3;

    // 4. Species pattern (inside cell only)
    float pattern = 0.0;
    if (d < 0.0) {
        if      (uSpeciesId == 0) pattern = speciesEye(uv, halfSize);
        else if (uSpeciesId == 1) pattern = speciesVector(uv, halfSize);
        else if (uSpeciesId == 2) pattern = speciesBolt(uv, halfSize);
        else if (uSpeciesId == 3) pattern = speciesPlus(uv, halfSize);
        else if (uSpeciesId == 5) pattern = speciesFilter(uv, halfSize);
        else if (uSpeciesId == 8) pattern = speciesLoop(uv, halfSize);
    }

    // 5. Compose final colour
    vec3 col = uGlowColor * glow;                           // glow layer
    col = mix(col, uFillColor, fill);                       // fill over glow
    col = mix(col, uStrokeColor, stroke);                   // stroke over fill
    col += uStrokeColor * pattern;                          // pattern additive

    float alpha = max(glow, max(fill, stroke));
    finalColor = vec4(col, alpha);
}
`;

// ── SDF Vertex Shader ───────────────────────────────────────────────────────

const SDF_VERTEX = `#version 300 es
precision highp float;

in vec2 aPosition;
in vec2 aUV;

out vec2 vUV;

void main() {
    vUV = aUV;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ── Species ID mapping ──────────────────────────────────────────────────────

const SPECIES_ID: Record<string, number> = {
  'cil-eye': 0, 'cil-vector': 1, 'cil-bolt': 2, 'cil-plus': 3,
  'cil-arrow-right': 4, 'cil-filter': 5, 'cil-code': 6,
  'cil-layers': 7, 'cil-loop': 8, 'cil-graph': 9,
};

const SPECIES_COLOURS: Record<string, { fill: [number,number,number]; stroke: [number,number,number]; glow: [number,number,number] }> = {
  'cil-eye':         { fill: [0.36,0.42,0.75], stroke: [0.22,0.29,0.67], glow: [0.47,0.53,0.80] },
  'cil-vector':      { fill: [0.40,0.73,0.42], stroke: [0.22,0.56,0.24], glow: [0.51,0.78,0.52] },
  'cil-bolt':        { fill: [1.00,0.65,0.15], stroke: [0.96,0.49,0.00], glow: [1.00,0.80,0.50] },
  'cil-plus':        { fill: [0.93,0.25,0.48], stroke: [0.78,0.16,0.16], glow: [0.96,0.56,0.69] },
  'cil-arrow-right': { fill: [0.47,0.56,0.61], stroke: [0.27,0.35,0.39], glow: [0.69,0.75,0.77] },
  'cil-filter':      { fill: [0.67,0.28,0.74], stroke: [0.48,0.12,0.64], glow: [0.81,0.58,0.85] },
  'cil-code':        { fill: [0.15,0.65,0.60], stroke: [0.00,0.47,0.42], glow: [0.50,0.80,0.77] },
  'cil-layers':      { fill: [0.26,0.65,0.96], stroke: [0.08,0.40,0.75], glow: [0.56,0.79,0.98] },
  'cil-loop':        { fill: [1.00,0.79,0.16], stroke: [0.98,0.66,0.15], glow: [1.00,0.88,0.51] },
  'cil-graph':       { fill: [0.47,0.56,0.61], stroke: [0.22,0.28,0.31], glow: [0.69,0.75,0.77] },
};

function getSDFColours(species: string) {
  return SPECIES_COLOURS[species] ?? SPECIES_COLOURS['cil-code'];
}

// ── Build SDF mesh for a single cell ────────────────────────────────────────

function buildSDFCellMesh(desc: CellDescriptor): Mesh {
  const { bbox, species, z } = desc;
  const pad = 30; // extra padding for glow

  // Quad geometry (2 triangles) in clip space
  // We position the quad using the bbox coordinates
  const x0 = bbox.x - pad;
  const y0 = bbox.y - pad;
  const x1 = bbox.x + bbox.w + pad;
  const y1 = bbox.y + bbox.h + pad;

  const geometry = new MeshGeometry({
    positions: new Float32Array([x0, y0, x1, y0, x1, y1, x0, y1]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  });

  const cols = getSDFColours(species);

  const uniforms = new UniformGroup({
    uTime: { value: 0, type: 'f32' },
    uSize: { value: new Float32Array([bbox.w, bbox.h]), type: 'vec2<f32>' },
    uFillColor: { value: new Float32Array(cols.fill), type: 'vec3<f32>' },
    uStrokeColor: { value: new Float32Array(cols.stroke), type: 'vec3<f32>' },
    uGlowColor: { value: new Float32Array(cols.glow), type: 'vec3<f32>' },
    uCornerRadius: { value: 8.0, type: 'f32' },
    uSpeciesId: { value: SPECIES_ID[species] ?? 6, type: 'i32' },
  });

  const shader = Shader.from({
    gl: { vertex: SDF_VERTEX, fragment: SDF_FRAGMENT },
    resources: { uniforms },
  });

  const mesh = new Mesh({ geometry, shader });
  mesh.zIndex = z;

  return mesh;
}

// ── M071: Build CilEyeSDFFilter with uniform injection from species_params ──
//
// Mirrors the M039/M053 approach in pixi-cell-renderer.ts:
//   1. Create a transparent Graphics quad covering the cell bbox
//   2. Instantiate CilEyeSDFFilter with uniforms derived from species_params
//   3. Mount the filter on the Graphics quad
//   4. Wrap in a Container positioned at the cell bbox
//
// species_params mapping (from agent_params.json):
//   ring_count        → numRays          (radial ray / concentric ring count)
//   pupil_radius      → pupilRadius      (px → NDC: px / (min(w,h)/2), clamped [0.05, 0.6])
//   r_outer           → focalIntensity   (px → scale: r_outer/halfMin * 1.5, clamped [0.3, 2.0])
//   r_inner_ratio     → bloomRadius      (ratio → bloom width: ratio * 2.0, clamped [0.2, 2.0])
//   bloom_strength    → bloomStrength    (direct, default 1.2)
//   ambient_intensity → ambientIntensity (direct, default 3.44)
//   ambient_color     → ambientColor     (hex string or [r,g,b], default #0bed90)
//   light_exposure    → lightExposure    (direct, default 0.86)
//   shadow_far        → shadowFar        (direct, default 40.0)
//   shadow_bias       → shadowBias       (direct, default 0.001)
//   num_rays          → numRays          (fallback for ring_count)
//   focal_intensity   → focalIntensity   (fallback for r_outer)

function buildCilEyeFilterContainer(desc: CellDescriptor): Container {
  const { bbox, species, z } = desc;
  const { w, h } = bbox;
  const speciesParams = desc.params?.species_params;

  const container = new Container();
  container.position.set(bbox.x, bbox.y);
  container.zIndex = z;

  // Transparent Graphics quad — the Filter renders into this
  const pattern = new Graphics();
  pattern.rect(0, 0, w, h);
  pattern.fill({ color: 0x000000, alpha: 0 });

  // Fill colour from species palette → [r, g, b] 0-1
  const cols = getSDFColours(species);
  const fc = cols.fill;

  // M071: Use CilEyeSDFFilter.fromSpeciesParams for uniform injection.
  // The factory handles all species_params → uniform mapping (ring_count → numRays,
  // pupil_radius → pupilRadius in NDC, r_outer → focalIntensity, etc.)
  const eyeFilter = CilEyeSDFFilter.fromSpeciesParams(
    [fc[0], fc[1], fc[2]],
    w,
    h,
    speciesParams,
  );
  pattern.filters = [eyeFilter];

  // Expose filter for Ticker-driven radial ray rotation animation
  (container as any).__eyeFilter = eyeFilter;

  container.addChild(pattern);

  return container;
}

// ── M211: createSpeciesSDFFilter — species → SDF Filter routing ─────────────
//
// Factory that maps a species string to the corresponding SDF Filter instance,
// using each filter class's fromSpeciesParams factory (or direct constructor for
// CilBoltSDFFilter which lacks one).  Fill colour is resolved from the
// JSON-driven species palette (getSpeciesPalette) so callers don't need to
// hard-code colour tables.
//
// @param species — species identifier (e.g. 'cil-eye', 'cil-bolt', ...)
// @param cellW   — cell width in pixels (for px → NDC normalisation)
// @param cellH   — cell height in pixels
// @param sp      — species_params from agent_params.json (optional)
// @returns         the matching SDF Filter, or null for unknown species

export function createSpeciesSDFFilter(
  species: string,
  cellW: number,
  cellH: number,
  sp?: Record<string, unknown>,
): Filter | null {
  // Resolve fill colour from JSON-driven palette → [r, g, b] 0-1
  const palette = getSpeciesPalette(species);
  const primary = palette.primary;
  const fillColor: [number, number, number] = [
    ((primary >> 16) & 0xff) / 255,
    ((primary >>  8) & 0xff) / 255,
    ( primary        & 0xff) / 255,
  ];

  switch (species) {
    case 'cil-eye':
      return CilEyeSDFFilter.fromSpeciesParams(fillColor, cellW, cellH, sp);

    case 'cil-bolt':
      // CilBoltSDFFilter has no fromSpeciesParams — construct directly with
      // species_params mapping: zigzag_segments → zigzagCount, amplitude → amplitude
      return new CilBoltSDFFilter({
        fillColor,
        opacity:     1.0,
        zigzagCount: (sp?.zigzag_segments as number | undefined)
                     ?? (sp?.num_rays      as number | undefined)
                     ?? 6,
        amplitude:   (sp?.amplitude as number | undefined) ?? 0.35,
        time:        0,
      });

    case 'cil-vector':
      return CilVectorSDFFilter.fromSpeciesParams(fillColor, cellW, cellH, sp);

    case 'cil-plus':
      return CilPlusSDFFilter.fromSpeciesParams(fillColor, cellW, cellH, sp);

    case 'cil-arrow-right':
      return CilArrowRightSDFFilter.fromSpeciesParams(fillColor, cellW, cellH, sp);

    default:
      return null;
  }
}

// ── Export: SDF renderer entry point ─────────────────────────────────────────

export async function renderCellGraphSDF(
  canvas: HTMLCanvasElement,
  cells: CellDescriptor[],
  edges: EdgeDescriptor[],
): Promise<Application> {
  const app = new Application();
  await app.init({
    canvas,
    width: canvas.width,
    height: canvas.height,
    backgroundColor: 0x0D1117,
    antialias: true,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  app.stage.sortableChildren = true;

  // Render cells
  for (const cell of cells) {
    if (cell.species === 'cil-eye') {
      // M071: cil-eye → CilEyeSDFFilter mounted on Graphics quad overlay.
      // Compiles cil-eye.frag as a PixiJS Filter with full uniform injection
      // from agent_params.json species_params, replacing the inline speciesEye()
      // SDF function in the monolithic shader.
      const eyeContainer = buildCilEyeFilterContainer(cell);
      app.stage.addChild(eyeContainer);

      // Label overlay — positioned relative to the container
      const style = new TextStyle({
        fontFamily: 'Inter, system-ui, sans-serif',
        fontSize: 11,
        fill: 0xFFFFFF,
        fontWeight: '500',
      });
      const txt = new Text({ text: cell.label, style });
      txt.anchor.set(0.5);
      txt.position.set(cell.bbox.w / 2, cell.bbox.h / 2);
      eyeContainer.addChild(txt);
    } else {
      // All other species — use the inline monolithic SDF shader as before
      const mesh = buildSDFCellMesh(cell);
      app.stage.addChild(mesh);

      // Label overlay (text still uses PixiJS Text — SDF text needs msdfgen)
      const style = new TextStyle({
        fontFamily: 'Inter, system-ui, sans-serif',
        fontSize: 11,
        fill: 0xFFFFFF,
        fontWeight: '500',
      });
      const txt = new Text({ text: cell.label, style });
      txt.anchor.set(0.5);
      txt.position.set(cell.bbox.x + cell.bbox.w / 2, cell.bbox.y + cell.bbox.h / 2);
      txt.zIndex = cell.z + 1;
      app.stage.addChild(txt);
    }
  }

  // Animate time uniform
  app.ticker.add(() => {
    const t = performance.now() / 1000;
    for (const child of app.stage.children) {
      // Inline SDF Mesh — update uTime
      if (child instanceof Mesh && child.shader?.resources?.uniforms) {
        (child.shader.resources.uniforms as any).uniforms.uTime = t;
      }

      // M071: cil-eye CilEyeSDFFilter — update time for radial ray rotation
      const eyeFilter = (child as any).__eyeFilter as CilEyeSDFFilter | undefined;
      if (eyeFilter) {
        eyeFilter.time = t;
      }
    }
  });

  return app;
}
