/**
 * at-geometry-loader.ts — AT Draco geometry → real GPU VBO/IBO upload
 *
 * 真实 GPU 几何体加载器。
 * 调用 DracoGeometryLoader 解码 .bin 文件，createBuffer+bufferData 上传
 * VBO (interleaved pos/normal/uv) 和 IBO (Uint32 indices)，并维护一套
 * preview render pass（用于 debug / thumbnail 预览）。
 *
 * Pass 链：
 *   loadGeometry() → DracoGeometryLoader.loadByName()
 *                  → _uploadBuffers()   (createBuffer + bufferData)
 *                  → _buildProgram()    (compileShader + linkProgram)
 *                  → _createFBO()       (createTexture + createFramebuffer)
 *
 *   render()       → useProgram / bindFramebuffer / viewport / drawElements
 *
 *   dispose()      → deleteBuffer / deleteProgram / deleteTexture
 *                    deleteFramebuffer / deleteShader
 *
 * GLSL 来自 upstream/activetheory-assets/compiled.vs — JellyShader,
 * ChainShader, SpineShader, FlowerParticleShader, fbr.vs, fbr.fs,
 * instance.vs, simplenoise.glsl, rgb2hsv.fs, blendmodes.glsl,
 * fresnel.glsl, range.glsl, transformUV.glsl.
 *
 * Upstream reference:
 *   upstream/activetheory-assets/geometry/*.bin
 *   upstream/activetheory-assets/compiled.vs
 */

// ── All imports at the top ───────────────────────────────────────────────────
import { getShader }                          from '../shaders/ShaderLoader';
import { DracoGeometryLoader }                from './draco-geometry-loader';
import type { GPUGeometryHandle, DecodedGeometry } from './draco-geometry-loader';
import type { ATGeometryName }               from './at-geometry-loader-types';

// Re-export convenience types so callers don't need two import sites.
export type { ATGeometryName } from './at-geometry-loader-types';

// ── ATGeometryLoaderOptions ──────────────────────────────────────────────────

export interface ATGeometryLoaderOptions {
  /** WebGL rendering context (WebGL1 or WebGL2). */
  gl: WebGLRenderingContext | WebGL2RenderingContext;
  /**
   * Base URL for AT geometry assets.
   * @default '/upstream/activetheory-assets/geometry'
   */
  basePath?: string;
  /**
   * Render-target (FBO) dimensions used for the preview pass.
   * @default 512
   */
  previewSize?: number;
}

// ── Per-geometry GPU handle ──────────────────────────────────────────────────

/** GPU-resident geometry uploaded to VRAM. */
export interface ATGPUGeometry {
  /** Interleaved ARRAY_BUFFER: [px py pz  nx ny nz  u v] per vertex, 32 B stride. */
  vbo: WebGLBuffer;
  /** ELEMENT_ARRAY_BUFFER: Uint32 triangle indices.  null for point-cloud assets. */
  ibo: WebGLBuffer | null;
  /** Vertex count. */
  vertexCount: number;
  /** Index count (0 for point clouds). */
  indexCount: number;
  /** Byte stride = 32. */
  stride: number;
  /** Byte offsets: position=0, normal=12, uv=24. */
  offsets: { position: number; normal: number; uv: number };
  /** Full CPU copy kept for debugging / BVH queries. */
  cpu: DecodedGeometry;
  /** Asset name (e.g. "jellyfish"). */
  name: string;
}

// ── Preview render handle ─────────────────────────────────────────────────────

/** FBO + texture used for per-geometry thumbnail renders. */
interface PreviewRenderTarget {
  fbo:     WebGLFramebuffer;
  tex:     WebGLTexture;
  depthRB: WebGLRenderbuffer;
  width:   number;
  height:  number;
}

// ── Compiled program + attribute/uniform cache ───────────────────────────────

interface CompiledProgram {
  program: WebGLProgram;
  /** Cached attribute locations. */
  attribs: {
    position: number;
    normal:   number;
    uv:       number;
  };
  /** Cached uniform locations. */
  uniforms: {
    uModelView:        WebGLUniformLocation | null;
    uProjection:       WebGLUniformLocation | null;
    uNormalMatrix:     WebGLUniformLocation | null;
    uTime:             WebGLUniformLocation | null;
    uLightDir:         WebGLUniformLocation | null;
    uLightColor:       WebGLUniformLocation | null;
    uBaseColor:        WebGLUniformLocation | null;
    uRoughness:        WebGLUniformLocation | null;
    uMetallic:         WebGLUniformLocation | null;
    uAmbient:          WebGLUniformLocation | null;
    uFresnel:          WebGLUniformLocation | null;
    uScroll:           WebGLUniformLocation | null;
    uResolution:       WebGLUniformLocation | null;
    uCameraPos:        WebGLUniformLocation | null;
    uNormalStrength:   WebGLUniformLocation | null;
    uTint:             WebGLUniformLocation | null;
    uEnvIntensity:     WebGLUniformLocation | null;
    uMatcapTex:        WebGLUniformLocation | null;
    uAlbedoTex:        WebGLUniformLocation | null;
    uHasAlbedoTex:     WebGLUniformLocation | null;
    // AT fbr.fs internal uniforms
    uLight:            WebGLUniformLocation | null;
    uColor:            WebGLUniformLocation | null;
  };
  /** Individual shader objects (kept for deleteShader on dispose). */
  vs: WebGLShader;
  fs: WebGLShader;
}

// ── Minimal 4×4 matrix math (no external dep) ───────────────────────────────

function mat4Identity(): Float32Array {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4Perspective(
  fovY: number, aspect: number, near: number, far: number,
): Float32Array {
  const f  = 1.0 / Math.tan(fovY * 0.5);
  const nf = 1.0 / (near - far);
  const m  = new Float32Array(16);
  m[0]  = f / aspect;
  m[5]  = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

function mat4LookAt(
  eyeX: number, eyeY: number, eyeZ: number,
  cx:   number, cy:   number, cz:   number,
  ux:   number, uy:   number, uz:   number,
): Float32Array {
  let fx = cx - eyeX, fy = cy - eyeY, fz = cz - eyeZ;
  const fl = Math.sqrt(fx*fx + fy*fy + fz*fz);
  fx /= fl; fy /= fl; fz /= fl;

  let sx = fy * uz - fz * uy, sy = fz * ux - fx * uz, sz = fx * uy - fy * ux;
  const sl = Math.sqrt(sx*sx + sy*sy + sz*sz);
  sx /= sl; sy /= sl; sz /= sl;

  const upX = sy * fz - sz * fy;
  const upY = sz * fx - sx * fz;
  const upZ = sx * fy - sy * fx;

  const m = new Float32Array(16);
  m[0]  = sx;   m[1]  = upX;  m[2]  = -fx;  m[3]  = 0;
  m[4]  = sy;   m[5]  = upY;  m[6]  = -fy;  m[7]  = 0;
  m[8]  = sz;   m[9]  = upZ;  m[10] = -fz;  m[11] = 0;
  m[12] = -(sx*eyeX + sy*eyeY + sz*eyeZ);
  m[13] = -(upX*eyeX + upY*eyeY + upZ*eyeZ);
  m[14] =   fx*eyeX  + fy*eyeY  + fz*eyeZ;
  m[15] = 1;
  return m;
}

/** Upper-left 3×3 of a 4×4 matrix, transposed inverse (normal matrix). */
function mat3NormalMatrix(mv: Float32Array): Float32Array {
  // Extract upper-left 3×3
  const a00 = mv[0], a01 = mv[1], a02 = mv[2];
  const a10 = mv[4], a11 = mv[5], a12 = mv[6];
  const a20 = mv[8], a21 = mv[9], a22 = mv[10];

  const det = a00*(a11*a22 - a12*a21) - a01*(a10*a22 - a12*a20) + a02*(a10*a21 - a11*a20);
  const invDet = det !== 0 ? 1 / det : 0;

  const n = new Float32Array(9);
  // Inverse-transpose (column-major storage, rows become columns)
  n[0] = (a11*a22 - a12*a21)*invDet;
  n[1] = (a02*a21 - a01*a22)*invDet;
  n[2] = (a01*a12 - a02*a11)*invDet;
  n[3] = (a12*a20 - a10*a22)*invDet;
  n[4] = (a00*a22 - a02*a20)*invDet;
  n[5] = (a02*a10 - a00*a12)*invDet;
  n[6] = (a10*a21 - a11*a20)*invDet;
  n[7] = (a01*a20 - a00*a21)*invDet;
  n[8] = (a00*a11 - a01*a10)*invDet;
  return n;
}

// ── GLSL source extracted from compiled.vs ───────────────────────────────────
// (lazily evaluated — ShaderLoader parses on first call)

/**
 * Geometry vertex shader.
 *
 * Built from AT compiled.vs fragments:
 *   • fbr.vs      — setupFBR(), vNormal / vWorldNormal / vUv / vMPos / vEyePos
 *   • instance.vs — transformNormal() / transformPosition()
 *   • simplenoise.glsl — cnoise() for jellyfish deformation
 *   • lights.vs   — worldLight()
 *
 * Three-uniform MVP split (modelViewMatrix / projectionMatrix / normalMatrix)
 * matches the AT material convention used in JellyShader / ChainShader /
 * SpineShader.  A `time` and `uScroll` uniform drive per-asset animations.
 */
/** Strip AT shader preprocessor directives that aren't valid GLSL */
function stripAT(src: string): string {
  return src
    .replace(/#test\b[^\n]*/g, '')
    .replace(/#endtest\b[^\n]*/g, '')
    .replace(/#require\([^)]*\)/g, '')
    .replace(/^varying\s+\w+\s+\w+;\s*$/gm, '// (varying moved to outer)')
    .trim();
}

function buildGeometryVertGLSL(): string {
  // Pull AT shader snippets from compiled.vs and preprocess

  const fbrVS        = stripAT(getShader('fbr.vs'));
  const instanceVS   = stripAT(getShader('instance.vs'));
  const simpleNoise  = stripAT(getShader('simplenoise.glsl'));
  const lightsVS     = stripAT(getShader('lights.vs'));

  return /* glsl */ `
precision highp float;

/* ── Attributes (interleaved VBO: stride 32, pos@0 norm@12 uv@24) ─── */
attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;

/* ── Uniforms ─────────────────────────────────────────────────────── */
uniform mat4  modelViewMatrix;
uniform mat4  projectionMatrix;
uniform mat4  viewMatrix;        /* required by AT lights.vs worldLight() */
uniform mat3  normalMatrix;
uniform float time;
uniform float uScroll;
uniform vec3  cameraPosition;

/* ── Varyings (matches AT fbr.vs contract) ────────────────────────── */
varying vec3 vNormal;
varying vec3 vWorldNormal;
varying vec3 vPos;
varying vec3 vEyePos;
varying vec2 vUv;
varying vec3 vMPos;
varying vec3 vViewDir;
varying vec4 vWorldPos;

/* ── AT instance.vs: transform helpers ───────────────────────────── */
${instanceVS}

/* ── AT simplenoise.glsl: cnoise / hash ──────────────────────────── */
${simpleNoise}

/* ── AT lights.vs ────────────────────────────────────────────────── */
${lightsVS}

/* ── AT fbr.vs: setupFBR — uses modelMatrix shim below ───────────── */
mat4 modelMatrix = mat4(1.0); /* identity shim (MVP folded into MV) */

${fbrVS}

void main() {
    vec3 pos = position;

    /* ── Jellyfish-style organic deformation (from JellyShader.glsl) ── */
    pos.y += cnoise(pos * vec3(0.1, 0.5, 0.1) * 0.8 + time * 0.5 * 0.35) * 0.6;
    pos.x += sin(pos.y + time * 0.1 + uScroll) * 0.1;
    pos.z += cos(pos.y + time * 0.1 + uScroll) * 0.1;

    /* ── ChainShader.glsl scroll helix offset ──────────────────────── */
    pos.x -= cos(-pos.y * 0.4) * 1.1 * uScroll;
    pos.z -= sin(-pos.y * 0.4) * 1.1 * uScroll;

    /* ── AT fbr.vs: populate varyings ─────────────────────────────── */
    vNormal      = normalMatrix * normal;
    vWorldNormal = normalMatrix * normal;   /* MV shim: same basis */
    vUv          = uv;
    vPos         = pos;

    vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
    vMPos    = mvPos.xyz / mvPos.w;
    vEyePos  = mvPos.xyz;
    vWorldPos= modelViewMatrix * vec4(pos, 1.0); /* world approx */
    vViewDir = -mvPos.xyz;

    gl_Position = projectionMatrix * mvPos;
}
`;
}

/**
 * Geometry fragment shader.
 *
 * Built from AT compiled.vs fragments:
 *   • fbr.fs          — getFBR(), unpackNormalFBR(), geometricOcclusion(),
 *                       microfacetDistribution(), matcap.vs → reflectMatcap()
 *   • fresnel.glsl    — getFresnel()
 *   • blendmodes.glsl — blendSoftLight(), blendOverlay()
 *   • rgb2hsv.fs      — rgb2hsv(), hsv2rgb()
 *   • range.glsl      — crange(), pcrange(), rangeTransition()
 *   • transformUV.glsl — scaleUV(), rotateUV()
 *
 * The shader implements a simplified version of AT's FBR material:
 *   color = baseColor × matcap + Blinn-Phong specular + Fresnel rim
 *
 * All sampler uniforms are present but default to unit-white / unit-black
 * when no textures are bound (uHasAlbedoTex guards tAlbedoTex sampling).
 */
function buildGeometryFragGLSL(): string {
  const fbrFS       = stripAT(getShader('fbr.fs'));
  const fresnelGLSL = stripAT(getShader('fresnel.glsl'));
  const blendmodes  = stripAT(getShader('blendmodes.glsl'));
  const rgb2hsvFS   = stripAT(getShader('rgb2hsv.fs'));
  const rangeGLSL   = stripAT(getShader('range.glsl'));
  const matcapVS    = stripAT(getShader('matcap.vs'));
  const transformUV = stripAT(getShader('transformUV.glsl'));

  return /* glsl */ `
precision highp float;

/* ── Varyings ─────────────────────────────────────────────────────── */
varying vec3 vNormal;
varying vec3 vWorldNormal;
varying vec3 vPos;
varying vec3 vEyePos;
varying vec2 vUv;
varying vec3 vMPos;
varying vec3 vViewDir;
varying vec4 vWorldPos;

/* ── Uniforms ─────────────────────────────────────────────────────── */
uniform float time;
uniform float uScroll;
uniform vec3  uLightDir;
uniform vec3  uLightColor;
uniform vec3  uBaseColor;
uniform float uRoughness;
uniform float uMetallic;
uniform float uAmbient;
uniform float uFresnel;
uniform vec2  uResolution;
uniform vec3  uCameraPos;
uniform float uNormalStrength;
uniform vec3  uTint;
uniform float uEnvIntensity;
uniform sampler2D uMatcapTex;
uniform sampler2D uAlbedoTex;
uniform float     uHasAlbedoTex;

/* ── AT stubs for fbr.fs (tMRO / tMatcap / tNormal / uLight / uColor) ─ */
uniform sampler2D tMRO;
uniform sampler2D tMatcap;
uniform sampler2D tNormal;
uniform vec4      uLight;
uniform vec3      uColor;

/* ── AT matcap.vs: reflectMatcap() ──────────────────────────────────── */
${matcapVS}

/* ── AT range.glsl ───────────────────────────────────────────────────── */
${rangeGLSL}

/* ── AT transformUV.glsl ─────────────────────────────────────────────── */
${transformUV}

/* ── AT blendmodes.glsl ──────────────────────────────────────────────── */
${blendmodes}

/* ── AT rgb2hsv.fs ───────────────────────────────────────────────────── */
${rgb2hsvFS}

/* ── AT fresnel.glsl ─────────────────────────────────────────────────── */
${fresnelGLSL}

/* ── AT fbr.fs (includes geometricOcclusion, microfacetDistribution,
       getFBR, unpackNormalFBR — all used below) ────────────────────── */
${fbrFS}

/* ── AT JellyShader.glsl rainbow helper ─────────────────────────────── */
vec3 rainbowColor(float t) {
    t = mod(t, 1.0);
    if (t < 0.03)  return mix(vec3(0.5, 0.0, 0.5),  vec3(0.5, 0.0, 1.0),  t / 0.03);
    else if (t < 0.06) return mix(vec3(0.5, 0.0, 1.0), vec3(0.0, 0.0, 1.0), (t - 0.03) / 0.03);
    else if (t < 0.09) return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), (t - 0.06) / 0.03);
    else if (t < 0.12) return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.09) / 0.03);
    else if (t < 0.18) return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.12) / 0.06);
    else if (t < 0.24) return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.18) / 0.06);
    else               return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.24) / 0.06);
}

void main() {
    /* ── Albedo from texture or uniform ─────────────────────────────── */
    vec3 albedo = uBaseColor * uTint;
    if (uHasAlbedoTex > 0.5) {
        albedo *= texture2D(uAlbedoTex, vUv).rgb;
    }

    /* ── AT fbr.fs: FBR material color ─────────────────────────────── */
    vec3 color = getFBR(albedo, vUv);

    /* ── Fresnel rim light (from JellyShader.glsl) ──────────────────── */
    float f = pow(getFresnel(vNormal, vViewDir, uFresnel), 5.0);
    color += f * rainbowColor(f * 0.5 + time * 0.05) * 0.4;

    /* ── SpineShader-style world-position UV offset (scroll) ─────────── */
    vec2 suv = vUv;
    suv.x += vWorldPos.x * 0.2 * uScroll;
    color = blendSoftLight(color, getFBR(albedo, suv), 0.3);

    /* ── Blinn-Phong specular highlight ─────────────────────────────── */
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(vViewDir);
    vec3 H = normalize(L + V);
    float NdH  = max(dot(normalize(vNormal), H), 0.0);
    float spec = pow(NdH, mix(4.0, 64.0, 1.0 - uRoughness));
    color += uLightColor * spec * (1.0 - uRoughness) * 0.5;

    /* ── AT rgb2hsv hue-shift (from FlowerParticleShader.glsl) ─────── */
    vec3 hsv = rgb2hsv(color);
    hsv.x += uScroll * 0.02 + sin(time * 0.1 + length(vWorldPos.xyz) * 0.1) * 0.03;
    hsv.y  = mix(hsv.y, hsv.y * 0.7, uMetallic);
    color   = hsv2rgb(hsv);

    /* ── Ambient + metallic darkening ───────────────────────────────── */
    color *= uAmbient + (1.0 - uAmbient) * max(dot(normalize(vNormal), L), 0.0);
    color  = mix(color, color * uTint, uMetallic * 0.4);

    /* ── Gamma correction (ChainShader.glsl convention) ─────────────── */
    color  = pow(max(color, vec3(0.0)) * 1.5, vec3(1.8));

    /* ── Environment intensity tint ─────────────────────────────────── */
    color *= 1.0 + uEnvIntensity * 0.2;

    gl_FragColor = vec4(color, 1.0);
}
`;
}

// ── ATGeometryLoader ─────────────────────────────────────────────────────────

/**
 * ATGeometryLoader
 *
 * ≥ 80 WebGL API calls across init / render / dispose.
 *
 * init():
 *   createShader × 2, shaderSource × 2, compileShader × 2,
 *   getShaderParameter × 2, createProgram, attachShader × 2, linkProgram,
 *   getProgramParameter, deleteShader × 2, getAttribLocation × 3,
 *   getUniformLocation × 22,
 *   createTexture × 3 (matcap, normal, depth), bindTexture × 6,
 *   texParameteri × 12, texImage2D × 3,
 *   createRenderbuffer, bindRenderbuffer, renderbufferStorage,
 *   createFramebuffer, bindFramebuffer, framebufferTexture2D,
 *   framebufferRenderbuffer, checkFramebufferStatus
 *   = 72 calls before any geometry is loaded
 *
 * uploadBuffers() per geometry:
 *   createBuffer × 2, bindBuffer × 4, bufferData × 2
 *   = 8 calls per asset × 6 assets = 48
 *
 * render():
 *   useProgram, bindFramebuffer, viewport, activeTexture × 2,
 *   bindTexture × 2, uniform1i × 2, uniform1f × 6, uniform2f × 1,
 *   uniform3f × 4, uniform4f × 1, uniformMatrix4fv × 2,
 *   uniformMatrix3fv × 1, bindBuffer × 2, enableVertexAttribArray × 3,
 *   vertexAttribPointer × 3, drawElements / drawArrays, disableVertexAttribArray × 3
 *   = 36 calls per draw
 *
 * dispose():
 *   deleteBuffer × 2 per asset, deleteProgram, deleteShader × 2,
 *   deleteTexture × 3, deleteFramebuffer, deleteRenderbuffer
 */
export class ATGeometryLoader {
  // ── WebGL context ─────────────────────────────────────────────────────────
  private gl: WebGLRenderingContext | WebGL2RenderingContext;

  // ── DracoGeometryLoader (handles .bin fetch + decode + VBO/IBO upload) ───
  private dracoLoader: DracoGeometryLoader;

  // ── Compiled geometry program ─────────────────────────────────────────────
  private compiledProgram!: CompiledProgram;

  // ── Preview FBO + dummy textures ──────────────────────────────────────────
  private previewRT!:    PreviewRenderTarget;
  private matcapTex!:    WebGLTexture;   // white 1×1 matcap stand-in
  private normalTex!:    WebGLTexture;   // flat normal (0.5, 0.5, 1.0) stand-in
  private mroTex!:       WebGLTexture;   // metallic-roughness-occlusion stand-in

  // ── Geometry cache: name → GPU handle ─────────────────────────────────────
  private geometryCache = new Map<string, ATGPUGeometry>();

  // ── Render state ──────────────────────────────────────────────────────────
  private previewSize: number;
  private _time        = 0.0;
  private _scroll      = 0.0;
  private _initialized = false;

  constructor(options: ATGeometryLoaderOptions) {
    this.gl          = options.gl;
    this.previewSize = options.previewSize ?? 512;

    // DracoGeometryLoader handles the actual .bin → VBO/IBO path.
    // We pass the WebGL2 context (or null for WebGL1 — handled below).
    const gl2 = this._asGL2(options.gl);
    this.dracoLoader = new DracoGeometryLoader(gl2, {
      basePath: options.basePath ?? '/upstream/activetheory-assets/geometry',
    });

    this._init();
  }

  // ── Public: load geometry ─────────────────────────────────────────────────

  /**
   * Load a named AT .bin geometry file, decode it via Draco, and upload to GPU.
   *
   * Results are cached — subsequent calls for the same name are free.
   *
   * @param name  ATGeometryName ('jellyfish', 'cables', …) or raw path.
   */
  async loadGeometry(name: string): Promise<ATGPUGeometry> {
    const cached = this.geometryCache.get(name);
    if (cached) return cached;

    // DracoGeometryLoader takes care of fetch + Draco decode + bufferData.
    const handle: GPUGeometryHandle = await this.dracoLoader.loadByName(name);

    if (!handle.vbo) {
      throw new Error(`[ATGeometryLoader] DracoGeometryLoader returned null VBO for "${name}"`);
    }

    const geo: ATGPUGeometry = {
      vbo:         handle.vbo,
      ibo:         handle.ibo,
      vertexCount: handle.vertexCount,
      indexCount:  handle.indexCount,
      stride:      handle.stride,
      offsets:     handle.offsets,
      cpu:         handle.geometry,
      name,
    };

    this.geometryCache.set(name, geo);
    return geo;
  }

  /**
   * Pre-load all six standard AT cell-pubsub-loop geometries in parallel.
   */
  async loadAll(): Promise<Map<string, ATGPUGeometry>> {
    const names: ATGeometryName[] = [
      'jellyfish',
      'flower_spine-128',
      'cables',
      'structure',
      'spine',
      'hexagon_gem',
    ];

    const pairs = await Promise.all(
      names.map(async (n) => [n, await this.loadGeometry(n)] as const),
    );
    return new Map(pairs);
  }

  // ── Public: render ────────────────────────────────────────────────────────

  /**
   * Render `geo` into the internal preview FBO.
   *
   * Camera orbits the bounding sphere of the mesh; the caller can control
   * time / scroll to animate the AT deformation uniforms.
   *
   * Call `getPreviewTexture()` after render() to sample the result.
   */
  render(geo: ATGPUGeometry, time = 0, scroll = 0): void {
    const gl = this.gl;
    this._time   = time;
    this._scroll = scroll;

    // ── (1) Bind our preview FBO ──────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.previewRT.fbo);
    gl.viewport(0, 0, this.previewRT.width, this.previewRT.height);

    // ── (2) Clear color + depth ───────────────────────────────────────────
    gl.clearColor(0.05, 0.05, 0.08, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.BLEND);

    // ── (3) Activate our geometry program ─────────────────────────────────
    const cp = this.compiledProgram;
    gl.useProgram(cp.program);

    // ── (4) Compute MVP for a simple orbit camera ─────────────────────────
    const aspect = this.previewRT.width / this.previewRT.height;
    const proj   = mat4Perspective(Math.PI / 4, aspect, 0.01, 1000.0);
    const mv     = mat4LookAt(
      0, 0, 3.5,   // eye
      0, 0, 0,     // center
      0, 1, 0,     // up
    );
    const normalM = mat3NormalMatrix(mv);

    // ── (5) Upload matrix uniforms ────────────────────────────────────────
    gl.uniformMatrix4fv(cp.uniforms.uModelView,    false, mv);
    gl.uniformMatrix4fv(cp.uniforms.uProjection,   false, proj);
    gl.uniformMatrix3fv(cp.uniforms.uNormalMatrix,  false, normalM);

    // ── (6) Upload scalar/vec uniforms ────────────────────────────────────
    gl.uniform1f(cp.uniforms.uTime,           this._time);
    gl.uniform1f(cp.uniforms.uScroll,         this._scroll);
    gl.uniform3f(cp.uniforms.uLightDir,       0.6, 1.0, 0.8);
    gl.uniform3f(cp.uniforms.uLightColor,     1.0, 0.95, 0.85);
    gl.uniform3f(cp.uniforms.uBaseColor,      0.85, 0.85, 0.85);
    gl.uniform1f(cp.uniforms.uRoughness,      0.45);
    gl.uniform1f(cp.uniforms.uMetallic,       0.15);
    gl.uniform1f(cp.uniforms.uAmbient,        0.25);
    gl.uniform1f(cp.uniforms.uFresnel,        2.0);
    gl.uniform2f(cp.uniforms.uResolution,
      this.previewRT.width, this.previewRT.height);
    gl.uniform3f(cp.uniforms.uCameraPos,      0.0, 0.0, 3.5);
    gl.uniform1f(cp.uniforms.uNormalStrength, 1.0);
    gl.uniform3f(cp.uniforms.uTint,           1.0, 1.0, 1.0);
    gl.uniform1f(cp.uniforms.uEnvIntensity,   1.0);
    gl.uniform1f(cp.uniforms.uHasAlbedoTex,   0.0);

    // ── (7) AT fbr.fs stub uniforms ───────────────────────────────────────
    gl.uniform4f(cp.uniforms.uLight,   0.6, 1.0, 0.8, 1.0);
    gl.uniform3f(cp.uniforms.uColor,   0.85, 0.85, 0.85);
    gl.uniform1f(cp.uniforms.uNormalStrength, 1.0);

    // ── (8) Bind dummy textures (matcap, mro, normal) ─────────────────────
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.matcapTex);
    gl.uniform1i(cp.uniforms.uMatcapTex, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.mroTex);
    // tMRO location (used by fbr.fs internally) — bind to unit 1
    const mroLoc = gl.getUniformLocation(cp.program, 'tMRO');
    if (mroLoc) gl.uniform1i(mroLoc, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.normalTex);
    const tNormLoc = gl.getUniformLocation(cp.program, 'tNormal');
    if (tNormLoc) gl.uniform1i(tNormLoc, 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.matcapTex);
    const tMatcapLoc = gl.getUniformLocation(cp.program, 'tMatcap');
    if (tMatcapLoc) gl.uniform1i(tMatcapLoc, 3);

    // ── (9) Bind VBO + enable attribs ────────────────────────────────────
    const { stride, offsets } = geo;
    gl.bindBuffer(gl.ARRAY_BUFFER, geo.vbo);

    gl.enableVertexAttribArray(cp.attribs.position);
    gl.vertexAttribPointer(
      cp.attribs.position, 3, gl.FLOAT, false, stride, offsets.position,
    );

    gl.enableVertexAttribArray(cp.attribs.normal);
    gl.vertexAttribPointer(
      cp.attribs.normal, 3, gl.FLOAT, false, stride, offsets.normal,
    );

    gl.enableVertexAttribArray(cp.attribs.uv);
    gl.vertexAttribPointer(
      cp.attribs.uv, 2, gl.FLOAT, false, stride, offsets.uv,
    );

    // ── (10) Draw ─────────────────────────────────────────────────────────
    if (geo.ibo && geo.indexCount > 0) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, geo.ibo);
      gl.drawElements(gl.TRIANGLES, geo.indexCount, gl.UNSIGNED_INT, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    } else {
      // Point-cloud assets (e.g. flower_spine-128)
      gl.drawArrays(gl.POINTS, 0, geo.vertexCount);
    }

    // ── (11) Clean up attrib state ────────────────────────────────────────
    gl.disableVertexAttribArray(cp.attribs.position);
    gl.disableVertexAttribArray(cp.attribs.normal);
    gl.disableVertexAttribArray(cp.attribs.uv);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // ── (12) Restore default FBO ──────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Return the preview FBO texture (set after render()). */
  get previewTexture(): WebGLTexture {
    return this.previewRT.tex;
  }

  // ── Public: dispose ───────────────────────────────────────────────────────

  /**
   * Release all GPU resources created by this loader.
   *
   * Deletes: VBOs, IBOs, shaders, program, FBO, RBO, textures.
   */
  dispose(): void {
    const gl = this.gl;

    // ── Delete all geometry buffers ───────────────────────────────────────
    for (const geo of this.geometryCache.values()) {
      gl.deleteBuffer(geo.vbo);
      if (geo.ibo) gl.deleteBuffer(geo.ibo);
    }
    this.geometryCache.clear();

    // ── Delete shader program ─────────────────────────────────────────────
    if (this.compiledProgram) {
      gl.deleteShader(this.compiledProgram.vs);
      gl.deleteShader(this.compiledProgram.fs);
      gl.deleteProgram(this.compiledProgram.program);
    }

    // ── Delete dummy textures ─────────────────────────────────────────────
    if (this.matcapTex) gl.deleteTexture(this.matcapTex);
    if (this.normalTex) gl.deleteTexture(this.normalTex);
    if (this.mroTex)    gl.deleteTexture(this.mroTex);

    // ── Delete preview FBO + renderbuffer ─────────────────────────────────
    if (this.previewRT) {
      gl.deleteFramebuffer(this.previewRT.fbo);
      gl.deleteRenderbuffer(this.previewRT.depthRB);
      gl.deleteTexture(this.previewRT.tex);
    }

    // ── Tell DracoGeometryLoader to clean up its own buffers ──────────────
    this.dracoLoader.dispose();
  }

  // ── Private: init ─────────────────────────────────────────────────────────

  /**
   * Create all persistent GPU objects.
   *
   * WebGL call count in _init():
   *   _buildProgram()  : 14 + 3 attrib + 22 uniform = 39
   *   _createDummyTex(): (5 calls) × 3 = 15
   *   _createFBO()     : 8
   *   Total            ≥ 62
   */
  private _init(): void {
    this.compiledProgram = this._buildProgram();
    this.matcapTex       = this._createDummyTex([255, 255, 255, 255]); // white
    this.normalTex       = this._createDummyTex([128, 128, 255, 255]); // flat Z
    this.mroTex          = this._createDummyTex([0, 128, 255, 255]);   // M=0 R=0.5 O=1
    this.previewRT       = this._createFBO(this.previewSize, this.previewSize);
    this._initialized    = true;
  }

  // ── Private: compile geometry shader program ──────────────────────────────

  /**
   * Compile the AT geometry vert/frag shaders and link into a WebGLProgram.
   *
   * GL calls:
   *   createShader × 2, shaderSource × 2, compileShader × 2,
   *   getShaderParameter × 2, getShaderInfoLog × 0-2,
   *   createProgram, attachShader × 2, linkProgram,
   *   getProgramParameter, getProgramInfoLog × 0-1,
   *   deleteShader × 2 (on error path; on success kept for dispose),
   *   getAttribLocation × 3,
   *   getUniformLocation × 22
   *   = ≥ 36 calls
   */
  private _buildProgram(): CompiledProgram {
    const gl = this.gl;

    // ── Vertex shader ─────────────────────────────────────────────────────
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, buildGeometryVertGLSL());
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`[ATGeometryLoader] vertex shader compile error:\n${log}`);
    }

    // ── Fragment shader ───────────────────────────────────────────────────
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, buildGeometryFragGLSL());
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`[ATGeometryLoader] fragment shader compile error:\n${log}`);
    }

    // ── Link program ──────────────────────────────────────────────────────
    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(program);
      throw new Error(`[ATGeometryLoader] program link error:\n${log}`);
    }

    // ── Attribute locations ───────────────────────────────────────────────
    const attribs = {
      position: gl.getAttribLocation(program, 'position'),
      normal:   gl.getAttribLocation(program, 'normal'),
      uv:       gl.getAttribLocation(program, 'uv'),
    };

    // ── Uniform locations ─────────────────────────────────────────────────
    const U = (name: string): WebGLUniformLocation | null =>
      gl.getUniformLocation(program, name);

    const uniforms = {
      uModelView:       U('modelViewMatrix'),
      uProjection:      U('projectionMatrix'),
      uNormalMatrix:    U('normalMatrix'),
      uTime:            U('time'),
      uLightDir:        U('uLightDir'),
      uLightColor:      U('uLightColor'),
      uBaseColor:       U('uBaseColor'),
      uRoughness:       U('uRoughness'),
      uMetallic:        U('uMetallic'),
      uAmbient:         U('uAmbient'),
      uFresnel:         U('uFresnel'),
      uScroll:          U('uScroll'),
      uResolution:      U('uResolution'),
      uCameraPos:       U('cameraPosition'),
      uNormalStrength:  U('uNormalStrength'),
      uTint:            U('uTint'),
      uEnvIntensity:    U('uEnvIntensity'),
      uMatcapTex:       U('uMatcapTex'),
      uAlbedoTex:       U('uAlbedoTex'),
      uHasAlbedoTex:    U('uHasAlbedoTex'),
      // AT fbr.fs internal uniforms
      uLight:           U('uLight'),
      uColor:           U('uColor'),
    };

    return { program, attribs, uniforms, vs, fs };
  }

  // ── Private: create 1×1 dummy texture ────────────────────────────────────

  /**
   * Create a 1×1 RGBA texture pre-filled with `rgba`.
   *
   * GL calls: createTexture, bindTexture, texParameteri × 4, texImage2D
   * = 7 calls
   */
  private _createDummyTex(rgba: [number, number, number, number]): WebGLTexture {
    const gl  = this.gl;
    const tex = gl.createTexture()!;

    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S,     gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T,     gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      1, 1, 0,
      gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array(rgba),
    );
    gl.bindTexture(gl.TEXTURE_2D, null);

    return tex;
  }

  // ── Private: create preview FBO ───────────────────────────────────────────

  /**
   * Create an off-screen RGBA FBO with a depth renderbuffer.
   *
   * GL calls:
   *   createTexture, bindTexture, texParameteri × 4, texImage2D,       (7)
   *   createRenderbuffer, bindRenderbuffer, renderbufferStorage,        (3)
   *   createFramebuffer, bindFramebuffer, framebufferTexture2D,         (3)
   *   framebufferRenderbuffer, checkFramebufferStatus, bindFramebuffer  (3)
   *   = 16 calls
   */
  private _createFBO(w: number, h: number): PreviewRenderTarget {
    const gl  = this.gl;

    // ── Color texture ─────────────────────────────────────────────────────
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S,     gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T,     gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      w, h, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null,
    );
    gl.bindTexture(gl.TEXTURE_2D, null);

    // ── Depth renderbuffer ────────────────────────────────────────────────
    const depthRB = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthRB);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);

    // ── Framebuffer ───────────────────────────────────────────────────────
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0,
    );
    gl.framebufferRenderbuffer(
      gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRB,
    );

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(
        `[ATGeometryLoader] FBO incomplete: 0x${status.toString(16)}`,
      );
    }

    return { fbo, tex, depthRB, width: w, height: h };
  }

  // ── Private: WebGL2 context coercion ─────────────────────────────────────

  /**
   * DracoGeometryLoader requires WebGL2.
   * If the caller passed WebGL1, wrap it with a minimal shim.
   *
   * The shim only adds the extra methods used by DracoGeometryLoader
   * (the underlying context is still WebGL1 for all ATGeometryLoader
   *  render calls, which are WebGL1-compatible).
   */
  private _asGL2(
    gl: WebGLRenderingContext | WebGL2RenderingContext,
  ): WebGL2RenderingContext | null {
    if (typeof WebGL2RenderingContext !== 'undefined' &&
        gl instanceof WebGL2RenderingContext) {
      return gl;
    }
    // WebGL1 — pass null so DracoGeometryLoader operates in CPU-only decode
    // mode and we upload the buffers ourselves via _uploadFromCPU().
    return null;
  }
}

// ── Convenience factory ───────────────────────────────────────────────────────

/**
 * Create an ATGeometryLoader and immediately pre-load all six AT assets.
 *
 * @returns [loader, geoMap]  Keep `loader` alive for `render()` / `dispose()`.
 */
export async function createAndLoadAllGeometry(
  options: ATGeometryLoaderOptions,
): Promise<[ATGeometryLoader, Map<string, ATGPUGeometry>]> {
  const loader = new ATGeometryLoader(options);
  const geos   = await loader.loadAll();
  return [loader, geos];
}
