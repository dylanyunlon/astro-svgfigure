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
 * Upstream reference:
 *   skills/pixijs/pixijs-custom-rendering/SKILL.md
 *   upstream/pixijs-engine/src/scene/mesh/shared/Mesh.ts
 */

import {
  Application,
  Container,
  Mesh,
  MeshGeometry,
  Shader,
  UniformGroup,
  Text,
  TextStyle,
} from 'pixi.js';
// Resolved to upstream/pixijs-engine via tsconfig paths — no npm install needed

import type { CellDescriptor, EdgeDescriptor } from './pixi-cell-renderer';

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

  // Render cells as SDF quads
  for (const cell of cells) {
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

  // Animate time uniform
  app.ticker.add(() => {
    const t = performance.now() / 1000;
    for (const child of app.stage.children) {
      if (child instanceof Mesh && child.shader?.resources?.uniforms) {
        (child.shader.resources.uniforms as any).uniforms.uTime = t;
      }
    }
  });

  return app;
}
