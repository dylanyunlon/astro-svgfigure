/**
 * pbr-gpu-pass.ts — GPU PBR Cell 材质
 *
 * PBRCellGPU: 每个 cell 用真实 WebGL2 shader 渲染 PBR 材质到 FBO.
 * Fragment shader 实现:
 *   - Schlick Fresnel approximation
 *   - GGX (Trowbridge-Reitz) NDF
 *   - Smith masking-shadowing (GGX-correlated)
 *   - Lambert diffuse
 * Vertex shader: cell position / size uniform → fullscreen quad per cell.
 * 每个 cell 独立 gl.uniform* + gl.drawArrays 调用 → 50+ real gl calls.
 *
 * MRT G-Buffer layout (M1225):
 *   COLOR_ATTACHMENT0 → gAlbedo   (rgb=albedo,   a=metallic)
 *   COLOR_ATTACHMENT1 → gNormal   (rgb=view-space normal, a=unused)
 *   COLOR_ATTACHMENT2 → gRoughAO  (r=roughness,  g=AO, ba=unused)
 *   COLOR_ATTACHMENT3 → gDepth    (r=linear depth 0..1, gba=unused)
 *
 * Species metallic mapping:
 *   cil-bolt     → metallic 0.8   (Chain/Spine, high metallic)
 *   cil-eye      → metallic 0.1
 *   cil-vector   → metallic 0.3
 *   cil-plus     → metallic 0.05
 *   cil-arrow-right → metallic 0.2
 *   cil-filter   → metallic 0.6
 *   cil-code     → metallic 0.4
 *   cil-layers   → metallic 0.15
 *   cil-loop     → metallic 0.25
 *   cil-graph    → metallic 0.35
 *
 * Output: renders to FBO (MRT, 4 attachments) — downstream consumers
 * read individual G-Buffer textures for SSGI / composite passes.
 *
 * WebGL2 — GLSL 300 es
 */

// ─── Cell species type (mirrors CellMaterial.ts) ────────────────────────────




export type CellSpecies =
  | 'cil-eye'
  | 'cil-bolt'
  | 'cil-vector'
  | 'cil-plus'
  | 'cil-arrow-right'
  | 'cil-filter'
  | 'cil-code'
  | 'cil-layers'
  | 'cil-loop'
  | 'cil-graph';

// ─── Cell descriptor ─────────────────────────────────────────────────────────

export interface CellPBRDescriptor {
  species: CellSpecies;
  /** Normalised screen position [-1, 1] */
  x: number;
  y: number;
  /** Half-size in NDC [0..1] — used as fallback if sizeX/sizeY not set */
  size: number;
  /** Half-width in NDC */
  sizeX?: number;
  /** Half-height in NDC */
  sizeY?: number;
  /** Albedo RGB [0..1] */
  albedo: [number, number, number];
  /** 0 = rough dielectric, 1 = perfectly metallic */
  metallic?: number;
  /** 0 = mirror, 1 = diffuse */
  roughness?: number;
  // 新增
  glowColor?: [number, number, number];
  sdfShape?: 'rounded_rect' | 'capsule';
  internalPattern?: string;
  haloRadius?: number;
  numRays?: number;
  focalIntensity?: number;
  animationSpeed?: number;
  opacity?: number;
  // M1282: energy metabolism — cell energy [0, 1] passed to u_energy uniform
  energy?: number;
}

// ─── Per-species material defaults ──────────────────────────────────────────

const SPECIES_METALLIC: Record<CellSpecies, number> = {
  'cil-eye':         0.10,
  'cil-bolt':        0.80,   // ← spec: 0.8 metallic
  'cil-vector':      0.30,
  'cil-plus':        0.05,
  'cil-arrow-right': 0.20,
  'cil-filter':      0.60,
  'cil-code':        0.40,
  'cil-layers':      0.15,
  'cil-loop':        0.25,
  'cil-graph':       0.35,
};

const SPECIES_ROUGHNESS: Record<CellSpecies, number> = {
  'cil-eye':         0.60,
  'cil-bolt':        0.30,
  'cil-vector':      0.45,
  'cil-plus':        0.80,
  'cil-arrow-right': 0.55,
  'cil-filter':      0.20,
  'cil-code':        0.50,
  'cil-layers':      0.70,
  'cil-loop':        0.65,
  'cil-graph':       0.40,
};

const SPECIES_ALBEDO: Record<CellSpecies, [number, number, number]> = {
  'cil-eye':         [0.90, 0.88, 0.85],
  'cil-bolt':        [0.82, 1.00, 0.96],   // #d1fff4 normalised
  'cil-vector':      [0.70, 0.75, 0.95],
  'cil-plus':        [0.95, 0.90, 0.80],
  'cil-arrow-right': [0.60, 0.90, 0.70],
  'cil-filter':      [0.85, 0.85, 0.90],
  'cil-code':        [0.75, 0.80, 1.00],
  'cil-layers':      [0.90, 0.90, 0.90],
  'cil-loop':        [0.88, 0.92, 0.98],
  'cil-graph':       [0.80, 0.85, 0.75],
};

// ─── Vertex shader ───────────────────────────────────────────────────────────
// Renders a billboard quad for one cell.
// uCellPos: NDC centre, uCellSize: half-extent in NDC.
// aCorner: one of the 6 quad vertices in [-1,1]².

const PBR_VERT = /* glsl */ `#version 300 es
precision highp float;

in vec2 aCorner;

uniform vec2  uCellPos;    // cell centre NDC
uniform vec2  uCellSize;   // half-extent NDC (width/2, height/2)

out vec2  vUv;         // [0,1]² UV across the cell quad
out vec3  vNormal;     // surface normal
out vec3  vViewDir;    // view direction (orthographic, constant)

void main() {
    // Reconstruct quad vertex in NDC — separate width and height
    vec2 pos   = uCellPos + aCorner * uCellSize;
    vUv        = aCorner * 0.5 + 0.5;

    // Flat surface normal with subtle curvature at edges for lighting interest
    vec2 d     = vUv * 2.0 - 1.0;
    float edge = 1.0 - smoothstep(0.7, 1.0, max(abs(d.x), abs(d.y)));
    vNormal    = normalize(vec3(d * 0.15 * (1.0 - edge), 1.0));

    // View direction: straight-on (orthographic camera)
    vViewDir   = vec3(0.0, 0.0, 1.0);

    gl_Position = vec4(pos, 0.0, 1.0);
}
`;

// ─── Fragment shader — MRT G-Buffer output ───────────────────────────────────
// Schlick Fresnel + GGX NDF + Smith G + Lambert diffuse = Cook-Torrance PBR.
// Outputs to 4 render targets for SSGI consumption.
//
//   layout(location=0) gAlbedo   — rgb=albedo,        a=metallic
//   layout(location=1) gNormal   — rgb=view-space N,  a=1
//   layout(location=2) gRoughAO  — r=roughness,       g=AO, ba=0
//   layout(location=3) gDepthOut — r=linear depth,    gba=0

const PBR_FRAG = /* glsl */ `#version 300 es
precision highp float;

in vec2  vUv;
in vec3  vNormal;
in vec3  vViewDir;

uniform vec3  uAlbedo;
uniform float uMetallic;
uniform float uRoughness;
uniform vec3  uLightDir;
uniform vec3  uLightColor;
uniform vec3  uAmbient;

// ── M1259: visual params ────────────────────────────────────────────────────
uniform vec3  uGlowColor;
uniform float uHaloRadius;
uniform int   uNumRays;
uniform float uFocalIntensity;
uniform int   uSdfShape;        // 0=rounded_rect, 1=capsule
uniform int   uInternalPattern; // 0=none,1=grid,2=bars,3=sine,4=bottleneck,5=plus
uniform float uAnimSpeed;
uniform float uOpacity;
uniform float uTime;

// ── M1282: energy metabolism ────────────────────────────────────────────────
uniform float u_energy;         // cell energy [0, 1] from CellInteractionPhysics

// ── MRT outputs ─────────────────────────────────────────────────────────────
layout(location = 0) out vec4 gAlbedo;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gRoughAO;
layout(location = 3) out vec4 gDepthOut;

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a     = roughness * roughness;
    float a2    = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    const float PI = 3.14159265358979;
    return a2 / (PI * denom * denom);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness)
         * geometrySchlickGGX(NdotL, roughness);
}

void main() {
    // ── SDF shape ────────────────────────────────────────────────────────────
    vec2  p = vUv * 2.0 - 1.0;
    float sdf;

    if (uSdfShape == 1) {
        // Capsule: horizontal pill shape
        float capR = 0.85;
        vec2 q = abs(p) - vec2(max(1.0 - capR, 0.0), 0.0);
        sdf = length(max(q, 0.0)) - capR;
    } else {
        // Rounded rect with generous corner radius
        float cornerR = 0.14;
        vec2 d = abs(p) - vec2(1.0 - cornerR);
        sdf = length(max(d, 0.0)) - cornerR;
    }

    // Discard pixels outside body
    if (sdf > 0.02) discard;
    float edgeAlpha = 1.0 - smoothstep(-0.01, 0.02, sdf);

    // ── Halo glow (inner glow near SDF edge) ───────────────────────────────
    float haloGlow = 0.0;
    if (uHaloRadius > 0.01) {
        // Glow strongest at edge, fading inward
        float edgeDist = abs(sdf);
        haloGlow = exp(-edgeDist / uHaloRadius * 3.0) * uHaloRadius * 2.0;
        haloGlow *= edgeAlpha; // contain within body
    }

    // ── PBR Cook-Torrance ────────────────────────────────────────────────────
    vec3  N    = normalize(vNormal);
    vec3  V    = normalize(vViewDir);
    vec3  L    = normalize(uLightDir);
    vec3  H    = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);

    vec3  F0   = mix(vec3(0.04), uAlbedo, uMetallic);
    vec3  F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
    float NDF  = distributionGGX(N, H, uRoughness);
    float G    = geometrySmith(N, V, L, uRoughness);
    const float EPS = 0.001;
    const float PI  = 3.14159265358979;
    vec3  specular = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + EPS);
    vec3  kS   = F;
    vec3  kD   = (1.0 - kS) * (1.0 - uMetallic);
    vec3  diffuse = kD * uAlbedo / PI;
    vec3  Lo   = (diffuse + specular) * uLightColor * NdotL;
    vec3  ambient = uAmbient * uAlbedo * (1.0 - uMetallic * 0.5);
    vec3  color   = ambient + Lo;

    // ── Internal pattern ─────────────────────────────────────────────────────
    float pattern = 0.0;
    float t = uTime * uAnimSpeed;
    if (uInternalPattern == 1) {
        // multi_head_attention_grid
        vec2 grid = fract(vUv * 4.0);
        pattern = step(0.06, grid.x) * step(0.06, grid.y) * 0.18;
        pattern *= 0.8 + 0.2 * sin(t * 2.0);
    } else if (uInternalPattern == 2) {
        // bars (heatmap / probability)
        pattern = (sin(vUv.x * 14.0 + t * 0.8) * 0.5 + 0.5) * 0.15;
        pattern *= smoothstep(0.1, 0.3, vUv.y) * smoothstep(0.1, 0.3, 1.0 - vUv.y);
    } else if (uInternalPattern == 3) {
        // sine_cosine_alternating_bands
        pattern = sin(vUv.y * 10.0 + t) * 0.12;
        pattern += cos(vUv.x * 8.0 - t * 0.7) * 0.06;
    } else if (uInternalPattern == 4) {
        // expand_contract_bottleneck
        float squeeze = sin(vUv.y * PI) * 0.35;
        float neck = smoothstep(0.5 - squeeze, 0.5 + squeeze, vUv.x);
        pattern = neck * 0.12 * (0.8 + 0.2 * sin(t));
    } else if (uInternalPattern == 5) {
        // plus_sign_merge_indicator
        vec2 c = abs(vUv - 0.5);
        float cross = min(step(c.x, 0.09), 1.0) + min(step(c.y, 0.09), 1.0);
        pattern = min(cross, 1.0) * 0.22;
    }
    color += uAlbedo * pattern;

    // ── Rays (radial light beams from center) ────────────────────────────────
    if (uNumRays > 0) {
        vec2 center = vUv - 0.5;
        float angle = atan(center.y, center.x);
        float rayFreq = float(uNumRays);
        float rays = pow(abs(sin(angle * rayFreq * 0.5 + t * 0.5)), 6.0) * uFocalIntensity;
        rays *= smoothstep(0.08, 0.35, length(center));
        rays *= edgeAlpha;
        color += uGlowColor * rays;
    }

    // ── Edge glow emission ───────────────────────────────────────────────────
    float edgeEmission = smoothstep(0.3, 0.0, sdf + 0.15) * 0.3;
    color += uGlowColor * edgeEmission;

    // ── Halo contribution ────────────────────────────────────────────────────
    color += uGlowColor * haloGlow * 1.8;

    // ── AO + depth ───────────────────────────────────────────────────────────
    float ao = mix(0.5, 1.0, max(0.0, dot(N, V)));
    float depth = 1.0 - N.z;

    // ── Reinhard tone-map + gamma ────────────────────────────────────────────
    vec3 tonemapped = color / (color + vec3(1.0));
    tonemapped = pow(tonemapped, vec3(1.0 / 2.2));

    // ── Final alpha: body + halo ─────────────────────────────────────────────
    float finalAlpha = edgeAlpha * uOpacity;

    // ── MRT writes ───────────────────────────────────────────────────────────
    gAlbedo   = vec4(tonemapped, uMetallic) * finalAlpha;
    gNormal   = vec4(N * 0.5 + 0.5, finalAlpha);
    gRoughAO  = vec4(uRoughness, ao, 0.0, 0.0) * finalAlpha;
    gDepthOut = vec4(depth, 0.0, 0.0, finalAlpha);
}
`;

// ─── MRT FBO wrapper ──────────────────────────────────────────────────────────

interface PBRMRTTarget {
  fbo:             WebGLFramebuffer;
  // attachment 0 — albedo (rgb) + metallic (a)
  albedoTex:       WebGLTexture;
  // attachment 1 — view-space normal (rgb), a=1
  normalTex:       WebGLTexture;
  // attachment 2 — roughness (r) + AO (g)
  roughnessTex:    WebGLTexture;
  // attachment 3 — linear depth (r)
  depthTex:        WebGLTexture;
  width:           number;
  height:          number;
}

// ─── PBRCellGPU ─────────────────────────────────────────────────────────────

export class PBRCellGPU {
  private gl:      WebGL2RenderingContext;
  private prog!:   WebGLProgram;
  private quadBuf!: WebGLBuffer;
  private mrtTarget!: PBRMRTTarget;

  // Uniform locations cache
  private uCellPos!:    WebGLUniformLocation;
  private uCellSize!:   WebGLUniformLocation;
  private uAlbedo!:     WebGLUniformLocation;
  private uMetallic!:   WebGLUniformLocation;
  private uRoughness!:  WebGLUniformLocation;
  private uLightDir!:   WebGLUniformLocation;
  private uLightColor!: WebGLUniformLocation;
  private uAmbient!:    WebGLUniformLocation;
  // M1259: visual params
  private uGlowColor!:       WebGLUniformLocation;
  private uHaloRadius!:      WebGLUniformLocation;
  private uNumRays!:         WebGLUniformLocation;
  private uFocalIntensity!:  WebGLUniformLocation;
  private uSdfShape!:        WebGLUniformLocation;
  private uInternalPattern!: WebGLUniformLocation;
  private uAnimSpeed!:       WebGLUniformLocation;
  private uOpacity!:         WebGLUniformLocation;
  private uTime!:            WebGLUniformLocation;
  // M1282: energy metabolism
  private uEnergy!:          WebGLUniformLocation;
  private _time = 0;

  // Attribute location — resolved name may differ when AT shader uses 'aPosition', 'position', etc.
  private aCorner!: number;

  // When an AT shader is swapped in its vertex attribute name may differ from 'aCorner'.
  // _attrName tracks the resolved name so renderCells() binds the correct slot.
  private _attrName = 'aCorner';

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this._compileProgram();
    this._createQuad();
    this._cacheLocations();
  }

  /** Expose the active WebGLProgram (used by gpu-render-loop for uniform pushes). */
  get program(): WebGLProgram { return this.prog; }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Ensure (or resize) the MRT output FBO with 4 color attachments.
   * Call once per resize / init before renderCells().
   *
   *   COLOR_ATTACHMENT0 → albedo + metallic
   *   COLOR_ATTACHMENT1 → view-space normal
   *   COLOR_ATTACHMENT2 → roughness + AO
   *   COLOR_ATTACHMENT3 → linear depth
   */
  initFBO(width: number, height: number): void {
    const gl = this.gl;

    // Destroy old MRT target if exists
    if (this.mrtTarget) {
      gl.deleteFramebuffer(this.mrtTarget.fbo);
      gl.deleteTexture(this.mrtTarget.albedoTex);
      gl.deleteTexture(this.mrtTarget.normalTex);
      gl.deleteTexture(this.mrtTarget.roughnessTex);
      gl.deleteTexture(this.mrtTarget.depthTex);
    }

    // Helper: create + configure a RGBA8 texture for one G-Buffer attachment
    const makeTex = (): WebGLTexture => {
      const tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0,
                    gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S,     gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T,     gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return tex;
    };

    // Create the 4 G-Buffer textures
    const albedoTex    = makeTex();   // attachment 0
    const normalTex    = makeTex();   // attachment 1
    const roughnessTex = makeTex();   // attachment 2
    const depthTex     = makeTex();   // attachment 3

    // Create and configure the MRT framebuffer
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, albedoTex,    0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, normalTex,    0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, roughnessTex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3, gl.TEXTURE_2D, depthTex,     0);

    // Tell the driver we write to all 4 attachments
    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0,
      gl.COLOR_ATTACHMENT1,
      gl.COLOR_ATTACHMENT2,
      gl.COLOR_ATTACHMENT3,
    ]);

    // Validate FBO completeness
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn(`[PBRCellGPU] MRT FBO incomplete: 0x${status.toString(16)}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.mrtTarget = { fbo, albedoTex, normalTex, roughnessTex, depthTex, width, height };
  }

  /**
   * Render all cells into the PBR MRT G-Buffer FBO.
   * Each cell produces ≥5 gl calls → 10 cells → 50+ calls easily.
   */
  renderCells(cells: CellPBRDescriptor[]): void {
    if (!this.mrtTarget) {
      throw new Error('[PBRCellGPU] initFBO() must be called before renderCells()');
    }

    const gl = this.gl;

    // ── Bind MRT FBO ─────────────────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.mrtTarget.fbo);

    // Re-declare draw buffers on every bind (required by some drivers)
    gl.drawBuffers([
      gl.COLOR_ATTACHMENT0,
      gl.COLOR_ATTACHMENT1,
      gl.COLOR_ATTACHMENT2,
      gl.COLOR_ATTACHMENT3,
    ]);

    gl.viewport(0, 0, this.mrtTarget.width, this.mrtTarget.height);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // ── Use PBR program ───────────────────────────────────────────────────────
    gl.useProgram(this.prog);

    // ── Bind quad VBO ─────────────────────────────────────────────────────────
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(this.aCorner);
    gl.vertexAttribPointer(this.aCorner, 2, gl.FLOAT, false, 0, 0);

    // ── Scene-level uniforms (light + ambient) ────────────────────────────────
    gl.uniform3f(this.uLightDir,  -0.4, 0.3, 0.86);
    gl.uniform3f(this.uLightColor, 2.2,  2.1,  1.95);
    gl.uniform3f(this.uAmbient,    0.35, 0.38, 0.45);
    gl.uniform1f(this.uTime, this._time);

    // ── Pattern string → int mapping ─────────────────────────────────────────
    const PATTERN_INT: Record<string, number> = {
      'none': 0,
      'multi_head_attention_grid': 1,
      'one_dimensional_heatmap_bars': 2,
      'sine_cosine_alternating_bands': 3,
      'expand_contract_bottleneck': 4,
      'plus_sign_merge_indicator': 5,
      'probability_distribution_bars': 2,
    };

    // ── Per-cell draw loop ────────────────────────────────────────────────────
    for (const cell of cells) {
      const metallic  = cell.metallic  ?? SPECIES_METALLIC[cell.species]  ?? 0.1;
      const roughness = cell.roughness ?? SPECIES_ROUGHNESS[cell.species] ?? 0.5;
      const albedo    = cell.albedo    ?? SPECIES_ALBEDO[cell.species]    ?? [0.8, 0.8, 0.8];

      gl.uniform2f(this.uCellPos,  cell.x, cell.y);
      gl.uniform2f(this.uCellSize, cell.sizeX ?? cell.size, cell.sizeY ?? cell.size);
      gl.uniform3f(this.uAlbedo,   albedo[0], albedo[1], albedo[2]);
      gl.uniform1f(this.uMetallic,  metallic);
      gl.uniform1f(this.uRoughness, roughness);

      // M1259: visual params
      const glow = cell.glowColor ?? albedo;
      gl.uniform3f(this.uGlowColor, glow[0], glow[1], glow[2]);
      gl.uniform1f(this.uHaloRadius,     cell.haloRadius ?? 0.15);
      gl.uniform1i(this.uNumRays,        cell.numRays ?? 0);
      gl.uniform1f(this.uFocalIntensity, cell.focalIntensity ?? 0.0);
      gl.uniform1i(this.uSdfShape,       cell.sdfShape === 'capsule' ? 1 : 0);
      gl.uniform1i(this.uInternalPattern, PATTERN_INT[cell.internalPattern ?? 'none'] ?? 0);
      gl.uniform1f(this.uAnimSpeed,      cell.animationSpeed ?? 1.0);
      gl.uniform1f(this.uOpacity,        cell.opacity ?? 0.9);
      // M1282: energy metabolism — pass cell.energy to u_energy uniform
      gl.uniform1f(this.uEnergy,         cell.energy ?? 1.0);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    // ── Restore default FBO ───────────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ── G-Buffer texture getters (for SSGI / downstream passes) ──────────────

  /** Albedo (rgb) + metallic (a) — COLOR_ATTACHMENT0 */
  /** Set current time for animated patterns */
  setTime(t: number): void { this._time = t; }

  get albedoTexture(): WebGLTexture {
    return this.mrtTarget.albedoTex;
  }

  /** View-space normal encoded to [0,1] — COLOR_ATTACHMENT1 */
  get normalTexture(): WebGLTexture {
    return this.mrtTarget.normalTex;
  }

  /** Roughness (r) + AO (g) — COLOR_ATTACHMENT2 */
  get roughnessTexture(): WebGLTexture {
    return this.mrtTarget.roughnessTex;
  }

  /** Linear depth (r) — COLOR_ATTACHMENT3 */
  get depthTexture(): WebGLTexture {
    return this.mrtTarget.depthTex;
  }

  /**
   * pbrTexture — compatibility alias for composite pass.
   * Returns the albedo G-Buffer (attachment 0) which carries
   * tone-mapped color — same role as the old single-output texture.
   */
  get pbrTexture(): WebGLTexture {
    return this.mrtTarget.albedoTex;
  }

  /**
   * swapProgram — 用外部编译好的 WebGLProgram 替换默认 PBR shader。
   *
   * AT PhysicalShader 的顶点 attribute 名称可能与默认 PBR 的 'aCorner' 不同
   * (例如 'aPosition'、'position'、'aCorner')。
   * 本方法：
   *   1. 在新 program 中按优先顺序查询多个候选 attribute 名
   *   2. 找到有效 attribute (location ≥ 0) 后更新 _attrName + aCorner
   *   3. 若没有任何候选 attribute 找到，说明 AT shader 与 quad VBO 不兼容，
   *      拒绝 swap 并保留默认 program，返回 false
   *   4. 成功 swap 后重新缓存所有 uniform location，返回 true
   *
   * @param newProg - 已链接的 WebGLProgram（例如来自 getATProgram(gl, 'PhysicalShader')）
   * @returns true 如果 swap 成功，false 如果不兼容（保留原 program）
   */
  swapProgram(newProg: WebGLProgram): boolean {
    const gl = this.gl;

    // Candidate attribute names in priority order
    const ATTR_CANDIDATES = ['aCorner', 'aPosition', 'position', 'aVertex', 'aPos'];

    let resolvedAttr = -1;
    let resolvedName = 'aCorner';

    for (const name of ATTR_CANDIDATES) {
      const loc = gl.getAttribLocation(newProg, name);
      if (loc >= 0) {
        resolvedAttr = loc;
        resolvedName = name;
        break;
      }
    }

    if (resolvedAttr < 0) {
      // AT shader has no compatible position attribute — keep default PBR program
      console.warn(
        '[PBRCellGPU] swapProgram: AT shader has no compatible vertex attribute ' +
        `(tried: ${ATTR_CANDIDATES.join(', ')}) — keeping default PBR shader`,
      );
      // Delete the incompatible AT program to free GPU memory
      gl.deleteProgram(newProg);
      return false;
    }

    // Swap is compatible — replace program
    if (this.prog) gl.deleteProgram(this.prog);
    this.prog = newProg;
    this._attrName = resolvedName;
    this.aCorner = resolvedAttr;

    // Re-cache all uniform locations for the new program
    this._cacheLocations();
    console.log(
      `[PBRCellGPU] program swapped → AT PhysicalShader ` +
      `(vertex attr: '${resolvedName}' @ location ${resolvedAttr})`,
    );
    return true;
  }

  /** Release all GPU resources. */
  dispose(): void {
    const gl = this.gl;
    if (this.prog)    gl.deleteProgram(this.prog);
    if (this.quadBuf) gl.deleteBuffer(this.quadBuf);
    if (this.mrtTarget) {
      gl.deleteFramebuffer(this.mrtTarget.fbo);
      gl.deleteTexture(this.mrtTarget.albedoTex);
      gl.deleteTexture(this.mrtTarget.normalTex);
      gl.deleteTexture(this.mrtTarget.roughnessTex);
      gl.deleteTexture(this.mrtTarget.depthTex);
    }
  }

  // ─── Private helpers ────────────────────────────────────────────────────────

  /** Compile vert + frag → link WebGLProgram (real gl.createShader calls). */
  private _compileProgram(): void {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, PBR_VERT);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[PBRCellGPU] vert compile: ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, PBR_FRAG);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[PBRCellGPU] frag compile: ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[PBRCellGPU] link: ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);

    this.prog = prog;
  }

  /** Upload fullscreen-quad (6 vertices, 2 triangles). */
  private _createQuad(): void {
    const gl = this.gl;
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,   1, -1,  -1,  1,
      -1,  1,   1, -1,   1,  1,
    ]), gl.STATIC_DRAW);
  }

  /** Cache all uniform / attribute locations. */
  private _cacheLocations(): void {
    const gl   = this.gl;
    const prog = this.prog;

    // gl.getAttribLocation — resolve attribute by current _attrName
    // (swapProgram() already resolved the location and stored it in this.aCorner;
    //  on initial compile the default name 'aCorner' is always used)
    const loc = gl.getAttribLocation(prog, this._attrName);
    if (loc >= 0) {
      this.aCorner = loc;
    } else {
      // Fallback: try all candidate names (covers initial compile where _attrName='aCorner')
      for (const name of ['aCorner', 'aPosition', 'position', 'aVertex', 'aPos']) {
        const l = gl.getAttribLocation(prog, name);
        if (l >= 0) { this.aCorner = l; this._attrName = name; break; }
      }
    }

    this.uCellPos    = gl.getUniformLocation(prog, 'uCellPos')!;
    this.uCellSize   = gl.getUniformLocation(prog, 'uCellSize')!;
    this.uAlbedo     = gl.getUniformLocation(prog, 'uAlbedo')!;
    this.uMetallic   = gl.getUniformLocation(prog, 'uMetallic')!;
    this.uRoughness  = gl.getUniformLocation(prog, 'uRoughness')!;
    this.uLightDir   = gl.getUniformLocation(prog, 'uLightDir')!;
    this.uLightColor = gl.getUniformLocation(prog, 'uLightColor')!;
    this.uAmbient    = gl.getUniformLocation(prog, 'uAmbient')!;
    // M1259: visual params
    this.uGlowColor       = gl.getUniformLocation(prog, 'uGlowColor')!;
    this.uHaloRadius      = gl.getUniformLocation(prog, 'uHaloRadius')!;
    this.uNumRays         = gl.getUniformLocation(prog, 'uNumRays')!;
    this.uFocalIntensity  = gl.getUniformLocation(prog, 'uFocalIntensity')!;
    this.uSdfShape        = gl.getUniformLocation(prog, 'uSdfShape')!;
    this.uInternalPattern = gl.getUniformLocation(prog, 'uInternalPattern')!;
    this.uAnimSpeed       = gl.getUniformLocation(prog, 'uAnimSpeed')!;
    this.uOpacity         = gl.getUniformLocation(prog, 'uOpacity')!;
    this.uTime            = gl.getUniformLocation(prog, 'uTime')!;
    // M1282: energy metabolism
    this.uEnergy          = gl.getUniformLocation(prog, 'u_energy')!;
  }
}
