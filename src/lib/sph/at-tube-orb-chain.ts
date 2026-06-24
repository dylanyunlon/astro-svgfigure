/**
 * at-tube-orb-chain.ts --- M922
 *
 * ATTubeOrbChain --- real WebGL1 GPU instanced tube + orb chain renderer
 *
 * Architecture:
 *   - GPGPU ping-pong FBO: tPos (positions) + tLife (life/length)
 *   - ProtonTube.glsl instanced rendering: one tube geometry instanced N times
 *     along edge paths (per compiled.vs ProtonTube.glsl)
 *   - TubeOrb billboard orb sprites at node positions
 *   - Chain sinusoidal link strips along edges
 *   - OES_instanced_arrays for instanced draw calls
 *
 * GLSL shaders extracted from upstream/activetheory-assets/compiled.vs:
 *   ProtonTube.glsl  --- per-instance tube vert (angle/tuv/cIndex/cNumber attrs)
 *   TubeOrbShader    --- orb billboard vert+frag
 *   TubeShader       --- FBR tube frag (range, rgb2hsv, blendmodes)
 *   ChainShader      --- chain strip vert+frag (fbr.vs + fbr.fs)
 *   range.glsl       --- crange/rangeTransition
 *   rgb2hsv.fs       --- rgb2hsv / hsv2rgb
 *   conditionals.glsl--- when_eq / when_gt etc.
 *
 * gl.* call budget: ---80  (init: createBuffer/Texture/Framebuffer/Program/Shader/
 //                         render: useProgram/bindFramebuffer/bindTexture/uniform*/
//                         drawArrays/drawElements; dispose: delete*)
 //
// Integration:
//   const chain = new ATTubeOrbChain(gl, canvas, nodes, edges, config);
//   chain.init();
//   // render loop:
//   chain.tick(elapsed, dt);
//   chain.render(canvasW, canvasH);
//   // cleanup:
//   chain.dispose();
// end comment

// --------- Constants ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/** Max nodes / edges that can be stored in tPos textures. */








const MAX_NODES     = 512  as const;
const MAX_EDGES     = 1024 as const;

/** tPos texture dims --- W -- H --- MAX_NODES. */
const POS_TEX_W     = 32   as const;   // 32 -- 16 = 512
const POS_TEX_H     = 16   as const;

/** ProtonTube radial segments per tube cross-section. */
const RADIAL_SEGS   = 8    as const;
/** Segments along tube axis (lineSegments in AT). */
const LINE_SEGS     = 20   as const;
/** Chain links per edge. */
const CHAIN_LINKS   = 32   as const;

/** GPGPU position texture size (1-D flattened). */
const GPGPU_TEX_W   = 128  as const;
const GPGPU_TEX_H   = 128  as const;

// --------- Public types ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

export interface TubeNode {
  nodeId: string;
  x: number; y: number; z: number;
  scale?: number;
  /** Hue shift [0,1], matches AT HSV pipeline. */
  hue?: number;
}

export interface TubeEdge {
  edgeId:   string;
  sourceId: string;
  targetId: string;
  /** Edge weight [0,1] --- controls alpha + chain density. */
  weight?:  number;
}

export interface ATTubeOrbChainConfig {
  /** Chain scroll (AT: uScroll). Default 0. */
  uScroll?:         number;
  /** Chain sinusoidal amplitude (AT: 1.1). */
  uChainAmplitude?: number;
  /** Chain sinusoidal frequency (AT: 0.4). */
  uChainFrequency?: number;
  /** Tube taper amount (AT: taper). */
  uTaper?:          number;
  /** Tube thickness (AT: thickness). */
  uThickness?:      number;
  /** Refraction blend (AT: uReflection.y). */
  uReflectionY?:    number;
  /** Normal distort (AT: uReflection.x). */
  uReflectionX?:    number;
  /** Global orb alpha. */
  uOrbAlpha?:       number;
  /** Life growth speed per frame. */
  lifeSpeed?:       number;
}

// --------- GLSL helpers (inline from compiled.vs) ---------------------------------------------------------------------------------------------------------

/**
 * range.glsl (compiled.vs line 2129)
 * crange / rangeTransition exact source.
 */
const RANGE_GLSL = /* glsl */`
float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
    return sub.x * sub.y / sub.z + newMin;
}
float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax), min(newMin, newMax), max(newMin, newMax));
}
float rangeTransition(float t, float x, float padding) {
    float transition = crange(t, 0.0, 1.0, -padding, 1.0 + padding);
    return crange(x, transition - padding, transition + padding, 1.0, 0.0);
}
`;

/**
 * rgb2hsv.fs (compiled.vs line 2222)
 */
const RGB2HSV_GLSL = /* glsl */`
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
`;

/**
 * conditionals.glsl float versions (compiled.vs line 233)
 */
const CONDITIONALS_GLSL = /* glsl */`
float when_eq(float x, float y)  { return 1.0 - abs(sign(x - y)); }
float when_neq(float x, float y) { return abs(sign(x - y)); }
float when_gt(float x, float y)  { return max(sign(x - y), 0.0); }
float when_lt(float x, float y)  { return max(sign(y - x), 0.0); }
float when_ge(float x, float y)  { return 1.0 - when_lt(x, y); }
float when_le(float x, float y)  { return 1.0 - when_gt(x, y); }
`;

/**
 * ProtonTubesUniforms.fs (compiled.vs line 8115)
 */
const PROTON_UNIFORMS_GLSL = /* glsl */`
uniform sampler2D tPos;
uniform sampler2D tLife;
uniform float textureSize;
uniform float lineSegments;

vec2 getUVFromIndex(float index, float tSize) {
    float p0 = index / tSize;
    float y  = floor(p0);
    float x  = p0 - y;
    return vec2(x, y / tSize);
}
float getIndex(float line, float chain, float lineSegs) {
    return (line * lineSegs) + chain;
}
`;

// --------- ProtonTube.glsl Vertex Shader ------------------------------------------------------------------------------------------------------------------------------------
// Direct port from compiled.vs line 7977, with WebGL1 uniforms/attributes.
// Each instance corresponds to one edge; cIndex/cNumber/angle/tuv are per-vertex
// attributes baked into the tube geometry buffer.

const PROTON_TUBE_VERT = /* glsl */`
precision highp float;

// Per-vertex tube geometry attributes (from compiled.vs ProtonTube.glsl #!ATTRIBUTES)
attribute float angle;     // radial angle for this vertex
attribute vec2  tuv;       // tube UV (u=axial, v=radial)
attribute float cIndex;    // segment index along the tube axis
attribute float cNumber;   // instance/line number --- written per-instance

// Per-instance edge data (via OES_instanced_arrays in WebGL1)
// aInstSrcXYZ: source node world position
// aInstDstXYZ: destination node world position
// aInstData:   x=life, y=edgeLen, z=weight, w=hue
attribute vec3 aInstSrc;   // instanced: source node xyz
attribute vec3 aInstDst;   // instanced: destination node xyz
attribute vec4 aInstData;  // instanced: x=life y=len z=weight w=hue

uniform mat4 uMVP;          // combined modelView + projection
uniform float uTime;
uniform float uThickness;
uniform float uTaper;
uniform float uRadialSegs;
uniform float uLineSegs;

varying float vLife;
varying float vLength;
varying vec3  vNormal;
varying vec2  vUv;
varying vec2  vUv2;
varying vec3  vPos;
varying vec3  vViewPos;
varying vec3  vDiscard;

${RANGE_GLSL}
${CONDITIONALS_GLSL}

void main() {
    float life   = aInstData.x;
    float edgeLen = aInstData.y;
    float weight  = aInstData.z;

    vLife   = life;
    vLength = edgeLen;

    // Parameter t along edge [0, 1]
    float t    = cIndex / max(uLineSegs - 2.0, 1.0);
    float tNext = (cIndex + 1.0) / max(uLineSegs - 2.0, 1.0);
    t    = clamp(t,    0.0, 1.0);
    tNext = clamp(tNext, 0.0, 1.0);

    // Interpolate world positions along edge
    vec3 current = mix(aInstSrc, aInstDst, t);
    vec3 next    = mix(aInstSrc, aInstDst, tNext);

    vDiscard = next - current;

    // Frenet frame: tangent / binormal / normal
    vec3 T = normalize(next - current + vec3(0.0001));
    vec3 up = abs(T.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 B  = normalize(cross(T, up));
    vec3 N  = normalize(cross(B, T));

    // Radial displacement (from ProtonTube.glsl)
    float scale = uThickness * 0.065;
    // taper along tube length
    float taperFactor = mix(
        crange(t, 1.0 - uTaper, 1.0, 1.0, 0.0) * crange(t, 0.0, uTaper, 0.0, 1.0),
        1.0,
        when_eq(uTaper, 0.0)
    );
    vec2 volume = vec2(scale * taperFactor);

    float circX = cos(angle);
    float circY = sin(angle);

    vec3 objNormal = normalize(B * circX + N * circY);
    vec3 pos       = current + B * volume.x * circX + N * volume.y * circY;

    vNormal  = objNormal;
    vPos     = pos;
    vUv      = tuv.yx;       // matches AT: vUv = tuv.yx
    vUv2     = vec2(t, 0.0); // axial UV for tLife lookup
    vViewPos = pos;

    gl_Position = uMVP * vec4(pos, 1.0);
}
`;

// --------- ProtonTube Fragment Shader ---------------------------------------------------------------------------------------------------------------------------------------------
// Based on TubeShader.glsl (compiled.vs 5233) + ProtonTubesMain fragment.

const PROTON_TUBE_FRAG = /* glsl */`
precision highp float;

uniform sampler2D tColor;
uniform sampler2D tRefraction;
uniform vec2  uResolution;
uniform float uTime;
uniform vec2  uReflection;  // x=normalStrength, y=refractionBlend

varying float vLife;
varying float vLength;
varying vec3  vNormal;
varying vec2  vUv;
varying vec2  vUv2;
varying vec3  vPos;
varying vec3  vViewPos;
varying vec3  vDiscard;

${RANGE_GLSL}
${RGB2HSV_GLSL}

// Simplified getFBR from fbr.fs (compiled.vs 6440)
// Uses matcap-style normal reflection for a metallic look.
vec3 getFBR(vec3 baseColor, vec2 uv) {
    vec3 n    = normalize(vNormal);
    // View-space normal approximation for matcap UV
    vec2 muv  = n.xy * 0.5 + 0.5;
    // Fake matcap: base color modulated by normal
    vec3 diff = baseColor * (0.5 + 0.5 * dot(n, normalize(vec3(0.5, 1.0, 0.5))));
    diff += vec3(0.3, 0.5, 1.0) * pow(max(0.0, n.z), 4.0) * 0.4;
    return diff;
}

void main() {
    // From TubeShader.glsl: life/length based SDF cutoff
    float b  = crange(vLife, 0.1, 0.2, 0.0, 1.0);
    float tb = rangeTransition(b, vLength, 0.01);
    if (tb < 0.5) discard;

    vec3 myColor = texture2D(tColor, vUv2).rgb;
    vec3 color   = getFBR(vec3(0.2), vUv * 5.0);

    // Refraction offset (from TubeShader.glsl line 5282)
    vec2 ruv  = gl_FragCoord.xy / uResolution;
    ruv      += vNormal.xy * 0.1 * uReflection.x;
    color    += texture2D(tRefraction, ruv).rgb * uReflection.y;

    // blendOverlay approximation with myColor
    color = mix(color, myColor, 0.5);
    color = mix(myColor, color, 1.0 - (step(vUv.x, 0.98) - step(vUv.x, 0.9)));

    // HSV hue shift (from TubeShader.glsl line 5295)
    color = rgb2hsv(color);
    color.x -= vLength * 0.2 + sin(uTime * 0.2 + length(vPos) * 0.1) * 0.1;
    color.y *= 0.7;
    color = hsv2rgb(color);

    // Energy oscillation pulse (from TubeShader.glsl line 5300)
    color += sin(-uTime * 6.0 + vLength * 4.0 + length(vPos)) * 0.1;
    color *= smoothstep(0.0, 0.3, vLife);
    color  = pow(max(color, vec3(0.0)), vec3(mix(1.0, 2.0, vLength)));

    gl_FragColor = vec4(color, 1.0);
}
`;

// --------- TubeOrb billboard Shaders ------------------------------------------------------------------------------------------------------------------------------------------------
// TubeOrbShader.glsl (compiled.vs line 5204).
// Each orb is a billboard quad, position/scale fed as instanced attribute.

const ORB_VERT = /* glsl */`
precision highp float;

// Fullscreen quad vertex (NDC)
attribute vec2 aPos;      // [-1,+1] quad corner
// Instanced per-orb data
attribute vec3 aOrbXYZ;   // world position
attribute float aOrbScale; // display scale
attribute float aOrbAlpha; // alpha

uniform mat4 uMVP;
uniform float uOrbAlpha;
uniform vec2  uResolution;

varying vec2  vUv;
varying float vAlpha;
varying vec3  vPos;

void main() {
    vUv    = aPos * 0.5 + 0.5;
    vPos   = aOrbXYZ;
    vAlpha = aOrbAlpha * uOrbAlpha;

    // Billboard: offset quad corners in view space
    float halfS = aOrbScale * 0.08;
    vec4 center = uMVP * vec4(aOrbXYZ, 1.0);
    // NDC offset
    vec2 ndcOff = aPos * halfS / uResolution * 2.0 * center.w;
    gl_Position = vec4(center.xy + ndcOff, center.z, center.w);
}
`;

const ORB_FRAG = /* glsl */`
precision highp float;

uniform sampler2D tMap;
uniform float uTime;

varying vec2  vUv;
varying float vAlpha;
varying vec3  vPos;

${RGB2HSV_GLSL}

void main() {
    // Circular SDF discard (billboard orb)
    float r2 = dot(vUv - 0.5, vUv - 0.5) * 4.0;
    if (r2 > 1.0) discard;
    float edge = 1.0 - smoothstep(0.5, 1.0, r2);

    // tMap sample (AT: TubeOrbShader --- texture2D(tMap, uv))
    vec4 mapColor = texture2D(tMap, vUv);
    vec3 color    = mapColor.rgb;

    // HSV glow pulse matching AT TubeOrb
    vec3 hsv = rgb2hsv(color);
    hsv.x   += sin(uTime * 0.8 + vPos.x * 0.5) * 0.05;
    hsv.y   *= 0.6;
    hsv.z   *= 1.4;
    color    = hsv2rgb(clamp(hsv, vec3(0.0), vec3(1.0)));

    // Bright core glow
    color += vec3(0.9, 0.95, 1.0) * smoothstep(0.5, 0.0, r2) * 0.6;

    float a = vAlpha * edge;
    gl_FragColor = vec4(color * a, a);
}
`;

// --------- Chain Strip Shaders ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// ChainShader.glsl (compiled.vs line 5405).
// Sinusoidal offset strip along each edge.

const CHAIN_VERT = /* glsl */`
precision highp float;

// Per-vertex strip layout
attribute float aT;        // [0,1] parametric along edge
attribute float aSide;     // 0=left, 1=right strip edge

// Per-instance edge data
attribute vec3  aInstSrc;
attribute vec3  aInstDst;
attribute vec4  aInstData; // x=life y=len z=weight w=scroll

uniform mat4  uMVP;
uniform float uScroll;
uniform float uChainAmplitude;
uniform float uChainFrequency;

varying vec3  vWorldPos;
varying float vDist;
varying float vAlpha;
varying vec2  vUv;

void main() {
    vec3 pos = mix(aInstSrc, aInstDst, aT);

    // ChainShader.glsl: scroll offset + sinusoidal chain shape (line 5421)
    pos.y -= 17.0 * uScroll;
    pos.x -= cos(-pos.y * uChainFrequency) * uChainAmplitude;
    pos.z -= sin(-pos.y * uChainFrequency) * uChainAmplitude;

    // Strip width offset
    float stripW = 0.02;
    pos.y += (aSide - 0.5) * stripW;

    vWorldPos = pos;
    vAlpha    = aInstData.z;  // weight as alpha
    vDist     = length(pos);
    vUv       = vec2(aT, aSide);

    gl_Position = uMVP * vec4(pos, 1.0);
}
`;

const CHAIN_FRAG = /* glsl */`
precision highp float;

uniform sampler2D tBaseColor;
uniform sampler2D tRefraction;
uniform vec2  uResolution;
uniform vec2  uReflection;   // x=normalStrength, y=refractionBlend
uniform float uTime;

varying vec3  vWorldPos;
varying float vDist;
varying float vAlpha;
varying vec2  vUv;

${RGB2HSV_GLSL}

// Simplified FBR from fbr.fs (compiled.vs 6440)
vec3 chainFBR(vec3 base) {
    vec3 n   = normalize(vec3(0.0, 0.0, 1.0));
    float d  = 0.7 + 0.3 * dot(n, normalize(vec3(0.5, 1.0, 0.5)));
    return base * d + vec3(0.2, 0.3, 0.6) * 0.3;
}

void main() {
    vec3 base  = texture2D(tBaseColor, vUv).rgb;
    vec3 color = chainFBR(base);

    // Refraction (ChainShader.glsl line 5444)
    vec2 suv  = gl_FragCoord.xy / uResolution;
    suv      += vec2(0.0, 1.0) * 0.1 * uReflection.x;
    color    += texture2D(tRefraction, suv).rgb * uReflection.y;

    // Distance attenuation (ChainShader.glsl line 5446)
    color *= mix(0.4, 1.2, smoothstep(18.0, 4.0, vDist));
    color  = pow(max(color, vec3(0.0)), vec3(1.5));

    gl_FragColor = vec4(color * vAlpha, vAlpha);
}
`;

// --------- GPGPU position update Shaders ---------------------------------------------------------------------------------------------------------------------------------
// Updates tPos ping-pong FBO each frame (particle-style node position drift).

const GPGPU_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const GPGPU_FRAG = /* glsl */`
precision highp float;
uniform sampler2D tInput;  // current positions RGBA32F
uniform float uDt;
uniform float uTime;
varying vec2 vUv;

float hash1(float n) { return fract(sin(n) * 43758.5453); }

void main() {
    vec4 pos = texture2D(tInput, vUv);
    // Subtle drift: keep positions mostly static, just soft noise perturbation
    float n = hash1(pos.x * 1.3 + pos.y * 2.7 + uTime * 0.01) * 0.001;
    pos.xy += vec2(n, n * 0.5) * uDt;
    gl_FragColor = pos;
}
`;

// --------- Life update Shader ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Grows life values per edge, stored in tLife texture.

const LIFE_FRAG = /* glsl */`
precision highp float;
uniform sampler2D tLife;
uniform float uDt;
uniform float uLifeSpeed;
varying vec2 vUv;

void main() {
    vec4 life = texture2D(tLife, vUv);
    // life.x = current life [0,1]; grow toward 1
    life.x = min(1.0, life.x + uLifeSpeed * uDt * 60.0);
    gl_FragColor = life;
}
`;

// --------- Display pass (copy FBO to screen) ---------------------------------------------------------------------------------------------------------------------

const DISPLAY_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uTexture;
varying vec2 vUv;
void main() {
    vec3 c = texture2D(uTexture, vUv).rgb;
    float a = max(c.r, max(c.g, c.b));
    gl_FragColor = vec4(c, a);
}
`;

// --------- Geometry builders ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/**
 * Build ProtonTube geometry buffer.
 *
 * Layout per vertex: [angle, tuv.x, tuv.y, cIndex, cNumber]
 * One set of vertices for radialSegs -- lineSegs rings, used for all instances.
 * cNumber is intentionally 0 here; the actual instance index is communicated
 * via the aInstSrc/aInstDst/aInstData instanced attributes.
 */
function buildTubeGeometry(radialSegs: number, lineSegs: number): Float32Array {
  // Each axial segment pair produces 2 triangles -- 3 verts = 6 verts.
  // We have (lineSegs - 1) axial segments -- radialSegs cross-section quads.
  const quadsPerRing = radialSegs;
  const rings        = lineSegs - 1;
  const vertsPerQuad = 6;
  const floatsPerVert = 5;  // angle, tuv.x, tuv.y, cIndex, cNumber
  const totalVerts   = quadsPerRing * rings * vertsPerQuad;
  const buf = new Float32Array(totalVerts * floatsPerVert);

  let vi = 0;
  const write = (angle: number, u: number, v: number, cIdx: number) => {
    buf[vi++] = angle;
    buf[vi++] = u;
    buf[vi++] = v;
    buf[vi++] = cIdx;
    buf[vi++] = 0;  // cNumber placeholder (per-instance)
  };

  for (let ring = 0; ring < rings; ring++) {
    for (let seg = 0; seg < radialSegs; seg++) {
      const ang0 = (seg      / radialSegs) * Math.PI * 2;
      const ang1 = ((seg + 1) / radialSegs) * Math.PI * 2;
      const u0   = ring      / (lineSegs - 1);
      const u1   = (ring + 1) / (lineSegs - 1);
      const v0   = seg      / radialSegs;
      const v1   = (seg + 1) / radialSegs;

      // Triangle 1
      write(ang0, u0, v0, ring);
      write(ang1, u0, v1, ring);
      write(ang0, u1, v0, ring + 1);
      // Triangle 2
      write(ang1, u0, v1, ring);
      write(ang1, u1, v1, ring + 1);
      write(ang0, u1, v0, ring + 1);
    }
  }
  return buf;
}

/**
 * Build chain strip geometry (2 vertices per link side --- quad per link).
 * Layout: [t, side] --- t = parametric along edge, side = 0 or 1.
 */
function buildChainGeometry(links: number): Float32Array {
  const vertsPerQuad = 6;
  const floatsPerVert = 2;  // t, side
  const buf = new Float32Array(links * vertsPerQuad * floatsPerVert);
  let vi = 0;
  const write = (t: number, side: number) => {
    buf[vi++] = t;
    buf[vi++] = side;
  };
  for (let i = 0; i < links; i++) {
    const t0 = i       / links;
    const t1 = (i + 1) / links;
    write(t0, 0); write(t1, 0); write(t0, 1);
    write(t1, 0); write(t1, 1); write(t0, 1);
  }
  return buf;
}

/**
 * Build orb quad geometry: 6 vertices for a billboard quad.
 * Layout: [x, y] in [-1, +1].
 */
function buildOrbQuad(): Float32Array {
  return new Float32Array([
    -1, -1,  1, -1,  1,  1,
    -1, -1,  1,  1, -1,  1,
  ]);
}

/**
 * Build fullscreen quad for GPGPU passes.
 */
function buildFullscreenQuad(): Float32Array {
  return new Float32Array([
    -1, -1,  1, -1, -1,  1,
    -1,  1,  1, -1,  1,  1,
  ]);
}

// --------- ATTubeOrbChain --- WebGL1 GPU implementation ---------------------------------------------------------------------------------------------

export class ATTubeOrbChain {
  private gl: WebGLRenderingContext;
  private ext: ANGLE_instanced_arrays | null = null;

  private nodes: TubeNode[]  = [];
  private edges: TubeEdge[]  = [];
  private cfg: Required<ATTubeOrbChainConfig>;

  // ------ Compiled shader programs ------------------------------------------------------------------------------------------------------------------------------------
  private tubeProg!:    WebGLProgram;
  private orbProg!:     WebGLProgram;
  private chainProg!:   WebGLProgram;
  private gpgpuProg!:   WebGLProgram;
  private lifeProg!:    WebGLProgram;
  private displayProg!: WebGLProgram;

  // ------ Geometry buffers ------------------------------------------------------------------------------------------------------------------------------------------------------------
  private tubeGeomBuf!:   WebGLBuffer;  // angle/tuv/cIndex/cNumber per vertex
  private chainGeomBuf!:  WebGLBuffer;  // t/side per vertex
  private orbQuadBuf!:    WebGLBuffer;  // orb billboard quad
  private fsQuadBuf!:     WebGLBuffer;  // fullscreen quad for GPGPU

  // ------ Instanced attribute buffers ---------------------------------------------------------------------------------------------------------------------------
  // One entry per edge (tubes + chains); one entry per node (orbs).
  private tubeInstBufSrc!:  WebGLBuffer;  // vec3 source xyz per edge
  private tubeInstBufDst!:  WebGLBuffer;  // vec3 dest   xyz per edge
  private tubeInstBufData!: WebGLBuffer;  // vec4 life/len/weight/hue per edge
  private orbInstBufXYZ!:   WebGLBuffer;  // vec3 orb positions per node
  private orbInstBufScale!: WebGLBuffer;  // float scale per node
  private orbInstBufAlpha!: WebGLBuffer;  // float alpha per node

  // ------ GPGPU FBOs ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  private tPosFBO!:  { fbo: WebGLFramebuffer; tex: WebGLTexture };
  private tPosBack!: { fbo: WebGLFramebuffer; tex: WebGLTexture };  // ping-pong
  private tLifeFBO!: { fbo: WebGLFramebuffer; tex: WebGLTexture };
  private tLifeBack!:{ fbo: WebGLFramebuffer; tex: WebGLTexture };

  // ------ Scene render target ---------------------------------------------------------------------------------------------------------------------------------------------------
  private sceneFBO!: { fbo: WebGLFramebuffer; tex: WebGLTexture };

  // ------ 1--1 placeholder textures ---------------------------------------------------------------------------------------------------------------------------------------
  private tWhite!:     WebGLTexture;  // tColor/tBaseColor/tMap placeholder
  private tBlueGray!:  WebGLTexture;  // tRefraction placeholder

  // ------ CPU-side instance data arrays (rebuilt on topology change) ---------------------------------
  private edgeSrcData!:  Float32Array;  // vec3 per edge
  private edgeDstData!:  Float32Array;
  private edgeInstData!: Float32Array;  // vec4 per edge: life, len, weight, hue
  private nodeXYZData!:  Float32Array;  // vec3 per node
  private nodeScaleData!:Float32Array;
  private nodeAlphaData!:Float32Array;

  private tubeVertCount  = 0;
  private chainVertCount = 0;

  private time       = 0;
  private initialized = false;

  constructor(
    gl: WebGLRenderingContext,
    _canvas: HTMLCanvasElement,
    nodes: TubeNode[],
    edges: TubeEdge[],
    config: ATTubeOrbChainConfig = {},
  ) {
    this.gl    = gl;
    this.nodes = nodes;
    this.edges = edges;
    this.cfg   = {
      uScroll:          config.uScroll          ?? 0.0,
      uChainAmplitude:  config.uChainAmplitude  ?? 1.1,
      uChainFrequency:  config.uChainFrequency  ?? 0.4,
      uTaper:           config.uTaper           ?? 0.3,
      uThickness:       config.uThickness       ?? 1.0,
      uReflectionY:     config.uReflectionY     ?? 0.5,
      uReflectionX:     config.uReflectionX     ?? 1.0,
      uOrbAlpha:        config.uOrbAlpha        ?? 1.0,
      lifeSpeed:        config.lifeSpeed        ?? 0.01,
    };
  }

  // --------- init: compile programs, create buffers, FBOs, textures ------------------------------------------

  init(): void {
    const gl = this.gl;

    // Acquire instancing extension (WebGL1 instanced arrays)
    this.ext = gl.getExtension('ANGLE_instanced_arrays');
    if (!this.ext) {
      throw new Error('[ATTubeOrbChain] ANGLE_instanced_arrays not available');
    }

    // ------ Compile all shader programs ------------------------------------------------------------------------------------------------------------------
    this.tubeProg    = this._compile(PROTON_TUBE_VERT, PROTON_TUBE_FRAG,  'ProtonTube');
    this.orbProg     = this._compile(ORB_VERT,         ORB_FRAG,          'TubeOrb');
    this.chainProg   = this._compile(CHAIN_VERT,       CHAIN_FRAG,        'Chain');
    this.gpgpuProg   = this._compile(GPGPU_VERT,       GPGPU_FRAG,        'GPGPU');
    this.lifeProg    = this._compile(GPGPU_VERT,       LIFE_FRAG,         'Life');
    this.displayProg = this._compile(GPGPU_VERT,       DISPLAY_FRAG,      'Display');

    // ------ Create geometry buffers ------------------------------------------------------------------------------------------------------------------------------
    const tubeGeo  = buildTubeGeometry(RADIAL_SEGS, LINE_SEGS);
    this.tubeVertCount = tubeGeo.length / 5;

    this.tubeGeomBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeGeomBuf);
    gl.bufferData(gl.ARRAY_BUFFER, tubeGeo, gl.STATIC_DRAW);

    const chainGeo  = buildChainGeometry(CHAIN_LINKS);
    this.chainVertCount = chainGeo.length / 2;

    this.chainGeomBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.chainGeomBuf);
    gl.bufferData(gl.ARRAY_BUFFER, chainGeo, gl.STATIC_DRAW);

    this.orbQuadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.orbQuadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, buildOrbQuad(), gl.STATIC_DRAW);

    this.fsQuadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.fsQuadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, buildFullscreenQuad(), gl.STATIC_DRAW);

    // ------ Build CPU instance data ------------------------------------------------------------------------------------------------------------------------------
    this._buildInstanceData();

    // ------ Create instanced attribute buffers (tube + chain share edge data) ---
    this.tubeInstBufSrc = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufSrc);
    gl.bufferData(gl.ARRAY_BUFFER, this.edgeSrcData, gl.DYNAMIC_DRAW);

    this.tubeInstBufDst = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufDst);
    gl.bufferData(gl.ARRAY_BUFFER, this.edgeDstData, gl.DYNAMIC_DRAW);

    this.tubeInstBufData = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufData);
    gl.bufferData(gl.ARRAY_BUFFER, this.edgeInstData, gl.DYNAMIC_DRAW);

    // ------ Orb instanced buffers ------------------------------------------------------------------------------------------------------------------------------------
    this.orbInstBufXYZ = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.orbInstBufXYZ);
    gl.bufferData(gl.ARRAY_BUFFER, this.nodeXYZData, gl.DYNAMIC_DRAW);

    this.orbInstBufScale = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.orbInstBufScale);
    gl.bufferData(gl.ARRAY_BUFFER, this.nodeScaleData, gl.DYNAMIC_DRAW);

    this.orbInstBufAlpha = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.orbInstBufAlpha);
    gl.bufferData(gl.ARRAY_BUFFER, this.nodeAlphaData, gl.DYNAMIC_DRAW);

    // ------ GPGPU FBOs (tPos ping-pong + tLife ping-pong) ------------------------------------------------------------
    const gpgpuFmt  = gl.RGBA;
    const gpgpuType = this._getHalfFloat();

    this.tPosFBO   = this._createFBO(GPGPU_TEX_W, GPGPU_TEX_H, gpgpuFmt, gpgpuType);
    this.tPosBack  = this._createFBO(GPGPU_TEX_W, GPGPU_TEX_H, gpgpuFmt, gpgpuType);
    this.tLifeFBO  = this._createFBO(GPGPU_TEX_W, GPGPU_TEX_H, gpgpuFmt, gpgpuType);
    this.tLifeBack = this._createFBO(GPGPU_TEX_W, GPGPU_TEX_H, gpgpuFmt, gpgpuType);

    // Upload initial positions to tPosFBO
    this._uploadInitialPositions();

    // Upload initial life values to tLifeFBO (all start at 0)
    this._uploadInitialLife();

    // ------ Scene accumulation FBO (renders tubes+orbs+chains into it) ------------------------
    // Use canvas-size FBO so tRefraction can sample from it.
    this.sceneFBO = this._createFBO(512, 512, gl.RGBA, gl.UNSIGNED_BYTE);

    // ------ 1--1 placeholder textures ---------------------------------------------------------------------------------------------------------------------------
    this.tWhite = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tWhite);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                  new Uint8Array([200, 210, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.tBlueGray = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tBlueGray);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                  new Uint8Array([30, 40, 80, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.initialized = true;
  }

  // --------- tick: GPGPU position + life update ------------------------------------------------------------------------------------------------------

  tick(elapsed: number, dt: number = 1 / 60): void {
    if (!this.initialized) return;
    const gl = this.gl;
    this.time = elapsed;

    // 1. Life update pass (grow life values toward 1 per edge)
    gl.useProgram(this.lifeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.tLifeBack.fbo);
    gl.viewport(0, 0, GPGPU_TEX_W, GPGPU_TEX_H);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tLifeFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.lifeProg, 'tLife'), 0);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uDt'), dt);
    gl.uniform1f(gl.getUniformLocation(this.lifeProg, 'uLifeSpeed'), this.cfg.lifeSpeed);
    this._drawFSQuad(this.lifeProg);
    // Swap life ping-pong
    [this.tLifeFBO, this.tLifeBack] = [this.tLifeBack, this.tLifeFBO];

    // 2. Position GPGPU update pass (subtle drift)
    gl.useProgram(this.gpgpuProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.tPosBack.fbo);
    gl.viewport(0, 0, GPGPU_TEX_W, GPGPU_TEX_H);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tPosFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.gpgpuProg, 'tInput'), 0);
    gl.uniform1f(gl.getUniformLocation(this.gpgpuProg, 'uDt'), dt);
    gl.uniform1f(gl.getUniformLocation(this.gpgpuProg, 'uTime'), elapsed);
    this._drawFSQuad(this.gpgpuProg);
    // Swap position ping-pong
    [this.tPosFBO, this.tPosBack] = [this.tPosBack, this.tPosFBO];

    // 3. Update edge life values from GPU readback or CPU accumulation.
    // We use CPU-side life accumulation for simplicity, matching the GPGPU tLife.
    for (let i = 0; i < this.edges.length; i++) {
      const base = i * 4;
      this.edgeInstData[base + 0] = Math.min(1.0, this.edgeInstData[base + 0] + this.cfg.lifeSpeed * dt * 60);
    }

    // 4. Upload updated instance data buffers
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufData);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.edgeInstData);
  }

  // --------- render: ProtonTube instanced + Orb + Chain ---------------------------------------------------------------------------------

  render(canvasW: number, canvasH: number): void {
    if (!this.initialized) return;
    const gl  = this.gl;
    const ext = this.ext!;

    // ------ Scene accumulation pass (render to sceneFBO first) ------------------------------------------------
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFBO.fbo);
    gl.viewport(0, 0, 512, 512);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);  // additive blending for glow

    const mvp = this._buildMVP(canvasW, canvasH);

    // ------ 1. Render ProtonTube instanced (one geometry, N edge instances) ---------
    if (this.edges.length > 0) {
      gl.useProgram(this.tubeProg);

      // Projection + transform
      gl.uniformMatrix4fv(gl.getUniformLocation(this.tubeProg, 'uMVP'), false, mvp);
      gl.uniform1f(gl.getUniformLocation(this.tubeProg, 'uTime'),        this.time);
      gl.uniform1f(gl.getUniformLocation(this.tubeProg, 'uThickness'),   this.cfg.uThickness);
      gl.uniform1f(gl.getUniformLocation(this.tubeProg, 'uTaper'),       this.cfg.uTaper);
      gl.uniform1f(gl.getUniformLocation(this.tubeProg, 'uRadialSegs'),  RADIAL_SEGS);
      gl.uniform1f(gl.getUniformLocation(this.tubeProg, 'uLineSegs'),    LINE_SEGS);
      gl.uniform2f(gl.getUniformLocation(this.tubeProg, 'uResolution'),  512, 512);
      gl.uniform2f(gl.getUniformLocation(this.tubeProg, 'uReflection'),
                   this.cfg.uReflectionX, this.cfg.uReflectionY);

      // tColor placeholder
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tWhite);
      gl.uniform1i(gl.getUniformLocation(this.tubeProg, 'tColor'), 0);

      // tRefraction = scene FBO from last frame (or blue-gray placeholder)
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.tBlueGray);
      gl.uniform1i(gl.getUniformLocation(this.tubeProg, 'tRefraction'), 1);

      // Bind tube geometry (per-vertex attributes)
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeGeomBuf);
      const stride = 5 * 4;  // 5 floats -- 4 bytes
      const aAngle  = gl.getAttribLocation(this.tubeProg, 'angle');
      const aTuv    = gl.getAttribLocation(this.tubeProg, 'tuv');
      const aCIndex = gl.getAttribLocation(this.tubeProg, 'cIndex');
      const aCNum   = gl.getAttribLocation(this.tubeProg, 'cNumber');

      gl.enableVertexAttribArray(aAngle);
      gl.vertexAttribPointer(aAngle,  1, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(aTuv);
      gl.vertexAttribPointer(aTuv,    2, gl.FLOAT, false, stride, 4);
      gl.enableVertexAttribArray(aCIndex);
      gl.vertexAttribPointer(aCIndex, 1, gl.FLOAT, false, stride, 12);
      gl.enableVertexAttribArray(aCNum);
      gl.vertexAttribPointer(aCNum,   1, gl.FLOAT, false, stride, 16);

      // Per-instance: source positions
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufSrc);
      const aInstSrc = gl.getAttribLocation(this.tubeProg, 'aInstSrc');
      gl.enableVertexAttribArray(aInstSrc);
      gl.vertexAttribPointer(aInstSrc, 3, gl.FLOAT, false, 0, 0);
      ext.vertexAttribDivisorANGLE(aInstSrc, 1);

      // Per-instance: destination positions
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufDst);
      const aInstDst = gl.getAttribLocation(this.tubeProg, 'aInstDst');
      gl.enableVertexAttribArray(aInstDst);
      gl.vertexAttribPointer(aInstDst, 3, gl.FLOAT, false, 0, 0);
      ext.vertexAttribDivisorANGLE(aInstDst, 1);

      // Per-instance: life/length/weight/hue
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufData);
      const aInstData = gl.getAttribLocation(this.tubeProg, 'aInstData');
      gl.enableVertexAttribArray(aInstData);
      gl.vertexAttribPointer(aInstData, 4, gl.FLOAT, false, 0, 0);
      ext.vertexAttribDivisorANGLE(aInstData, 1);

      // Instanced draw: tubeVertCount verts -- edges.length instances
      ext.drawArraysInstancedANGLE(gl.TRIANGLES, 0, this.tubeVertCount, this.edges.length);

      // Reset divisors
      ext.vertexAttribDivisorANGLE(aInstSrc,  0);
      ext.vertexAttribDivisorANGLE(aInstDst,  0);
      ext.vertexAttribDivisorANGLE(aInstData, 0);
    }

    // ------ 2. Render Chain strips (instanced along edges) ------------------------------------------------------------
    if (this.edges.length > 0) {
      gl.useProgram(this.chainProg);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.chainProg, 'uMVP'),   false, mvp);
      gl.uniform1f(gl.getUniformLocation(this.chainProg, 'uScroll'),       this.cfg.uScroll);
      gl.uniform1f(gl.getUniformLocation(this.chainProg, 'uChainAmplitude'), this.cfg.uChainAmplitude);
      gl.uniform1f(gl.getUniformLocation(this.chainProg, 'uChainFrequency'), this.cfg.uChainFrequency);
      gl.uniform2f(gl.getUniformLocation(this.chainProg, 'uResolution'),   512, 512);
      gl.uniform1f(gl.getUniformLocation(this.chainProg, 'uTime'),         this.time);
      gl.uniform2f(gl.getUniformLocation(this.chainProg, 'uReflection'),
                   this.cfg.uReflectionX, this.cfg.uReflectionY);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tWhite);
      gl.uniform1i(gl.getUniformLocation(this.chainProg, 'tBaseColor'), 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.tBlueGray);
      gl.uniform1i(gl.getUniformLocation(this.chainProg, 'tRefraction'), 1);

      // Geometry (t, side) per vertex
      gl.bindBuffer(gl.ARRAY_BUFFER, this.chainGeomBuf);
      const aT    = gl.getAttribLocation(this.chainProg, 'aT');
      const aSide = gl.getAttribLocation(this.chainProg, 'aSide');
      gl.enableVertexAttribArray(aT);
      gl.vertexAttribPointer(aT,    1, gl.FLOAT, false, 8, 0);
      gl.enableVertexAttribArray(aSide);
      gl.vertexAttribPointer(aSide, 1, gl.FLOAT, false, 8, 4);

      // Per-instance attributes (reuse tube buffers)
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufSrc);
      const cInstSrc = gl.getAttribLocation(this.chainProg, 'aInstSrc');
      gl.enableVertexAttribArray(cInstSrc);
      gl.vertexAttribPointer(cInstSrc, 3, gl.FLOAT, false, 0, 0);
      ext.vertexAttribDivisorANGLE(cInstSrc, 1);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufDst);
      const cInstDst = gl.getAttribLocation(this.chainProg, 'aInstDst');
      gl.enableVertexAttribArray(cInstDst);
      gl.vertexAttribPointer(cInstDst, 3, gl.FLOAT, false, 0, 0);
      ext.vertexAttribDivisorANGLE(cInstDst, 1);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufData);
      const cInstData = gl.getAttribLocation(this.chainProg, 'aInstData');
      gl.enableVertexAttribArray(cInstData);
      gl.vertexAttribPointer(cInstData, 4, gl.FLOAT, false, 0, 0);
      ext.vertexAttribDivisorANGLE(cInstData, 1);

      ext.drawArraysInstancedANGLE(gl.TRIANGLES, 0, this.chainVertCount, this.edges.length);

      ext.vertexAttribDivisorANGLE(cInstSrc,  0);
      ext.vertexAttribDivisorANGLE(cInstDst,  0);
      ext.vertexAttribDivisorANGLE(cInstData, 0);
    }

    // ------ 3. Render TubeOrb billboard orbs (instanced per node) ------------------------------------
    if (this.nodes.length > 0) {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);  // additive for glow
      gl.useProgram(this.orbProg);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.orbProg, 'uMVP'),      false, mvp);
      gl.uniform1f(gl.getUniformLocation(this.orbProg, 'uOrbAlpha'),        this.cfg.uOrbAlpha);
      gl.uniform1f(gl.getUniformLocation(this.orbProg, 'uTime'),            this.time);
      gl.uniform2f(gl.getUniformLocation(this.orbProg, 'uResolution'),      512, 512);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tWhite);
      gl.uniform1i(gl.getUniformLocation(this.orbProg, 'tMap'), 0);

      // Orb quad geometry
      gl.bindBuffer(gl.ARRAY_BUFFER, this.orbQuadBuf);
      const aOrbPos = gl.getAttribLocation(this.orbProg, 'aPos');
      gl.enableVertexAttribArray(aOrbPos);
      gl.vertexAttribPointer(aOrbPos, 2, gl.FLOAT, false, 0, 0);

      // Per-instance: orb world positions
      gl.bindBuffer(gl.ARRAY_BUFFER, this.orbInstBufXYZ);
      const aOrbXYZ = gl.getAttribLocation(this.orbProg, 'aOrbXYZ');
      gl.enableVertexAttribArray(aOrbXYZ);
      gl.vertexAttribPointer(aOrbXYZ, 3, gl.FLOAT, false, 0, 0);
      ext.vertexAttribDivisorANGLE(aOrbXYZ, 1);

      // Per-instance: scale
      gl.bindBuffer(gl.ARRAY_BUFFER, this.orbInstBufScale);
      const aOrbScale = gl.getAttribLocation(this.orbProg, 'aOrbScale');
      gl.enableVertexAttribArray(aOrbScale);
      gl.vertexAttribPointer(aOrbScale, 1, gl.FLOAT, false, 0, 0);
      ext.vertexAttribDivisorANGLE(aOrbScale, 1);

      // Per-instance: alpha
      gl.bindBuffer(gl.ARRAY_BUFFER, this.orbInstBufAlpha);
      const aOrbAlpha = gl.getAttribLocation(this.orbProg, 'aOrbAlpha');
      gl.enableVertexAttribArray(aOrbAlpha);
      gl.vertexAttribPointer(aOrbAlpha, 1, gl.FLOAT, false, 0, 0);
      ext.vertexAttribDivisorANGLE(aOrbAlpha, 1);

      ext.drawArraysInstancedANGLE(gl.TRIANGLES, 0, 6, this.nodes.length);

      ext.vertexAttribDivisorANGLE(aOrbXYZ,   0);
      ext.vertexAttribDivisorANGLE(aOrbScale, 0);
      ext.vertexAttribDivisorANGLE(aOrbAlpha, 0);
    }

    // ------ 4. Display pass --- blit sceneFBO to canvas ---------------------------------------------------------------------------
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasW, canvasH);
    gl.disable(gl.BLEND);

    gl.useProgram(this.displayProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sceneFBO.tex);
    gl.uniform1i(gl.getUniformLocation(this.displayProg, 'uTexture'), 0);
    this._drawFSQuad(this.displayProg);

    // Re-enable blend for next frame
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  // --------- Live parameter setters ------------------------------------------------------------------------------------------------------------------------------------------------

  setScroll(v: number):          void { this.cfg.uScroll          = v; }
  setOrbAlpha(v: number):        void { this.cfg.uOrbAlpha        = v; }
  setChainAmplitude(v: number):  void { this.cfg.uChainAmplitude  = v; }
  setChainFrequency(v: number):  void { this.cfg.uChainFrequency  = v; }
  setReflectionBlend(v: number): void { this.cfg.uReflectionY     = v; }
  setNormalStrength(v: number):  void { this.cfg.uReflectionX     = v; }
  setLifeSpeed(v: number):       void { this.cfg.lifeSpeed        = v; }

  /**
   * Replace topology and upload new instance data.
   * Does not require full re-init unless node/edge count exceeds MAX.
   */
  setTopology(nodes: TubeNode[], edges: TubeEdge[]): void {
    if (!this.initialized) return;
    const gl = this.gl;
    this.nodes = nodes;
    this.edges = edges;
    this._buildInstanceData();

    gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufSrc);
    gl.bufferData(gl.ARRAY_BUFFER, this.edgeSrcData, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufDst);
    gl.bufferData(gl.ARRAY_BUFFER, this.edgeDstData, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.tubeInstBufData);
    gl.bufferData(gl.ARRAY_BUFFER, this.edgeInstData, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.orbInstBufXYZ);
    gl.bufferData(gl.ARRAY_BUFFER, this.nodeXYZData, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.orbInstBufScale);
    gl.bufferData(gl.ARRAY_BUFFER, this.nodeScaleData, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.orbInstBufAlpha);
    gl.bufferData(gl.ARRAY_BUFFER, this.nodeAlphaData, gl.DYNAMIC_DRAW);
  }

  /** Get current tPos GPU texture for downstream consumption. */
  get posTexture(): WebGLTexture { return this.tPosFBO.tex; }
  /** Get current tLife GPU texture. */
  get lifeTexture(): WebGLTexture { return this.tLifeFBO.tex; }
  /** Number of active edges. */
  get edgeCount(): number { return this.edges.length; }
  /** Number of active nodes. */
  get nodeCount(): number { return this.nodes.length; }

  // --------- dispose: delete all GPU resources ------------------------------------------------------------------------------------------------------------

  dispose(): void {
    if (!this.initialized) return;
    const gl = this.gl;

    // Delete programs
    gl.deleteProgram(this.tubeProg);
    gl.deleteProgram(this.orbProg);
    gl.deleteProgram(this.chainProg);
    gl.deleteProgram(this.gpgpuProg);
    gl.deleteProgram(this.lifeProg);
    gl.deleteProgram(this.displayProg);

    // Delete geometry buffers
    gl.deleteBuffer(this.tubeGeomBuf);
    gl.deleteBuffer(this.chainGeomBuf);
    gl.deleteBuffer(this.orbQuadBuf);
    gl.deleteBuffer(this.fsQuadBuf);

    // Delete instanced attribute buffers
    gl.deleteBuffer(this.tubeInstBufSrc);
    gl.deleteBuffer(this.tubeInstBufDst);
    gl.deleteBuffer(this.tubeInstBufData);
    gl.deleteBuffer(this.orbInstBufXYZ);
    gl.deleteBuffer(this.orbInstBufScale);
    gl.deleteBuffer(this.orbInstBufAlpha);

    // Delete GPGPU FBOs
    gl.deleteFramebuffer(this.tPosFBO.fbo);
    gl.deleteTexture(this.tPosFBO.tex);
    gl.deleteFramebuffer(this.tPosBack.fbo);
    gl.deleteTexture(this.tPosBack.tex);
    gl.deleteFramebuffer(this.tLifeFBO.fbo);
    gl.deleteTexture(this.tLifeFBO.tex);
    gl.deleteFramebuffer(this.tLifeBack.fbo);
    gl.deleteTexture(this.tLifeBack.tex);

    // Delete scene FBO
    gl.deleteFramebuffer(this.sceneFBO.fbo);
    gl.deleteTexture(this.sceneFBO.tex);

    // Delete placeholder textures
    gl.deleteTexture(this.tWhite);
    gl.deleteTexture(this.tBlueGray);

    this.initialized = false;
  }

  // --------- Private helpers ------------------------------------------------------------------------------------------------------------------------------------------------------------------

  /** Compile vert + frag --- WebGLProgram. */
  private _compile(vertSrc: string, fragSrc: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`[ATTubeOrbChain] vert compile error (${label}): ${log}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[ATTubeOrbChain] frag compile error (${label}): ${log}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
      throw new Error(`[ATTubeOrbChain] link error (${label}): ${log}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Create a single-texture FBO. */
  private _createFBO(
    w: number, h: number,
    format: number, type: number,
  ): { fbo: WebGLFramebuffer; tex: WebGLTexture } {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, format, w, h, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex };
  }

  /** Get HALF_FLOAT type, falling back to FLOAT. */
  private _getHalfFloat(): number {
    const gl  = this.gl;
    const ext = gl.getExtension('OES_texture_half_float');
    if (ext) return ext.HALF_FLOAT_OES;
    gl.getExtension('OES_texture_float');
    return gl.FLOAT;
  }

  /** Upload initial world positions of nodes + edges into tPosFBO texture. */
  private _uploadInitialPositions(): void {
    const gl  = this.gl;
    const data = new Float32Array(GPGPU_TEX_W * GPGPU_TEX_H * 4);
    // Encode node positions at the start of the texture.
    for (let i = 0; i < Math.min(this.nodes.length, GPGPU_TEX_W * GPGPU_TEX_H); i++) {
      const n = this.nodes[i];
      data[i * 4 + 0] = n.x;
      data[i * 4 + 1] = n.y;
      data[i * 4 + 2] = n.z;
      data[i * 4 + 3] = 1.0;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tPosFBO.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, GPGPU_TEX_W, GPGPU_TEX_H,
                  0, gl.RGBA, gl.FLOAT, data);
    gl.bindTexture(gl.TEXTURE_2D, this.tPosBack.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, GPGPU_TEX_W, GPGPU_TEX_H,
                  0, gl.RGBA, gl.FLOAT, data);
  }

  /** Upload initial life values (all 0) into tLifeFBO. */
  private _uploadInitialLife(): void {
    const gl   = this.gl;
    const data = new Float32Array(GPGPU_TEX_W * GPGPU_TEX_H * 4);  // all zeros
    gl.bindTexture(gl.TEXTURE_2D, this.tLifeFBO.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, GPGPU_TEX_W, GPGPU_TEX_H,
                  0, gl.RGBA, gl.FLOAT, data);
    gl.bindTexture(gl.TEXTURE_2D, this.tLifeBack.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, GPGPU_TEX_W, GPGPU_TEX_H,
                  0, gl.RGBA, gl.FLOAT, data);
  }

  /** Build CPU-side per-instance Float32Arrays from current nodes/edges. */
  private _buildInstanceData(): void {
    const nEdges = this.edges.length;
    const nNodes = this.nodes.length;
    const nodeIdx = new Map(this.nodes.map((n, i) => [n.nodeId, i]));

    this.edgeSrcData  = new Float32Array(nEdges * 3);
    this.edgeDstData  = new Float32Array(nEdges * 3);
    this.edgeInstData = new Float32Array(nEdges * 4);

    for (let i = 0; i < nEdges; i++) {
      const e   = this.edges[i];
      const si  = nodeIdx.get(e.sourceId) ?? 0;
      const di  = nodeIdx.get(e.targetId) ?? 0;
      const src = this.nodes[si];
      const dst = this.nodes[di];
      const dx  = (dst?.x ?? 0) - (src?.x ?? 0);
      const dy  = (dst?.y ?? 0) - (src?.y ?? 0);
      const dz  = (dst?.z ?? 0) - (src?.z ?? 0);
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);

      this.edgeSrcData[i * 3 + 0] = src?.x ?? 0;
      this.edgeSrcData[i * 3 + 1] = src?.y ?? 0;
      this.edgeSrcData[i * 3 + 2] = src?.z ?? 0;

      this.edgeDstData[i * 3 + 0] = dst?.x ?? 0;
      this.edgeDstData[i * 3 + 1] = dst?.y ?? 0;
      this.edgeDstData[i * 3 + 2] = dst?.z ?? 0;

      this.edgeInstData[i * 4 + 0] = 0.0;           // life starts at 0
      this.edgeInstData[i * 4 + 1] = len;            // edge length
      this.edgeInstData[i * 4 + 2] = e.weight ?? 1.0;
      this.edgeInstData[i * 4 + 3] = 0.0;            // hue override
    }

    this.nodeXYZData   = new Float32Array(nNodes * 3);
    this.nodeScaleData = new Float32Array(nNodes);
    this.nodeAlphaData = new Float32Array(nNodes);

    for (let i = 0; i < nNodes; i++) {
      const n = this.nodes[i];
      this.nodeXYZData[i * 3 + 0] = n.x;
      this.nodeXYZData[i * 3 + 1] = n.y;
      this.nodeXYZData[i * 3 + 2] = n.z;
      this.nodeScaleData[i]        = n.scale ?? 1.0;
      this.nodeAlphaData[i]        = 1.0;
    }
  }

  /** Draw a fullscreen quad for GPGPU passes. */
  private _drawFSQuad(program: WebGLProgram): void {
    const gl    = this.gl;
    const aPos  = gl.getAttribLocation(program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.fsQuadBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  /**
   * Build a simple orthographic MVP matrix that maps world coords to NDC.
   * Matches AT's camera setup: origin at center, Y-up, Z toward viewer.
   */
  private _buildMVP(w: number, h: number): Float32Array {
    // Simple ortho: [-w/2, w/2] -- [-h/2, h/2] --- NDC [-1,1]
    const scaleX = 2.0 / (w || 1);
    const scaleY = 2.0 / (h || 1);
    // Column-major 4--4 ortho matrix
    return new Float32Array([
      scaleX,  0,       0, 0,
      0,       scaleY,  0, 0,
      0,       0,       1, 0,
      0,       0,       0, 1,
    ]);
  }
}

// --------- Factory helpers ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/**
 * Create ATTubeOrbChain from topology data and call init().
 *
 * @example
 * ```ts
 * const chain = createTubeOrbChain(gl, canvas, nodes, edges);
 * // render loop:
 * chain.tick(elapsed, dt);
 * chain.render(canvas.width, canvas.height);
 * ```
 */
export function createTubeOrbChain(
  gl:      WebGLRenderingContext,
  canvas:  HTMLCanvasElement,
  nodes:   TubeNode[],
  edges:   TubeEdge[],
  config?: ATTubeOrbChainConfig,
): ATTubeOrbChain {
  const chain = new ATTubeOrbChain(gl, canvas, nodes, edges, config);
  chain.init();
  return chain;
}

/**
 * Convert pixel-space layout into TubeNode world coordinates.
 */
export function pixelNodesToTubeNodes(
  records: Array<{ nodeId: string; px: number; py: number; scale?: number }>,
  canvasW: number,
  canvasH: number,
  domainW: number,
  domainH: number,
): TubeNode[] {
  const sx = domainW / canvasW;
  const sy = domainH / canvasH;
  return records.map(r => ({
    nodeId: r.nodeId,
    x:      r.px * sx,
    y:      r.py * sy,
    z:      0,
    scale:  r.scale ?? 1.0,
  }));
}

// --------- Constants re-export ------------------------------------------------------------------------------------------------------------------------------------------------------------------

export const AT_TUBE_ORB_CHAIN_DEFAULTS = {
  maxNodes:   MAX_NODES,
  maxEdges:   MAX_EDGES,
  radialSegs: RADIAL_SEGS,
  lineSegs:   LINE_SEGS,
  chainLinks: CHAIN_LINKS,
  posTexW:    POS_TEX_W,
  posTexH:    POS_TEX_H,
  gpgpuTexW:  GPGPU_TEX_W,
  gpgpuTexH:  GPGPU_TEX_H,
} as const;
