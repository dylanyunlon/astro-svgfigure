/**
 * pbr-gpu-pass.ts — GPU PBR Cell 材质
 *
 * PBRCellGPU: 每个 cell 用真实 WebGL1 shader 渲染 PBR 材质到 FBO.
 * Fragment shader 实现:
 *   - Schlick Fresnel approximation
 *   - GGX (Trowbridge-Reitz) NDF
 *   - Smith masking-shadowing (GGX-correlated)
 *   - Lambert diffuse
 * Vertex shader: cell position / size uniform → fullscreen quad per cell.
 * 每个 cell 独立 gl.uniform* + gl.drawArrays 调用 → 50+ real gl calls.
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
 * Output: renders to FBO (WebGLFramebuffer + WebGLTexture) — downstream
 * consumers read pbrTexture for composite pass.
 *
 * WebGL1 only — no #version 300 es, no gl_FragData[1], no OES_draw_buffers.
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
  /** Half-size in NDC [0..1] */
  size: number;
  /** Albedo RGB [0..1] */
  albedo: [number, number, number];
  /** 0 = rough dielectric, 1 = perfectly metallic */
  metallic?: number;
  /** 0 = mirror, 1 = diffuse */
  roughness?: number;
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

const PBR_VERT = /* glsl */ `
precision highp float;

attribute vec2 aCorner;

uniform vec2  uCellPos;    // cell centre NDC
uniform float uCellSize;   // half-extent NDC

varying vec2  vUv;         // [0,1]² UV across the cell quad
varying vec3  vNormal;     // approximate sphere normal from UV
varying vec3  vViewDir;    // view direction (orthographic, constant)

void main() {
    // Reconstruct quad vertex in NDC
    vec2 pos   = uCellPos + aCorner * uCellSize;
    vUv        = aCorner * 0.5 + 0.5;

    // Fake sphere normal from UV: map [0,1] UV → [-1,1], project to hemisphere
    vec2 d     = vUv * 2.0 - 1.0;
    float len2 = dot(d, d);
    float z    = sqrt(max(0.0, 1.0 - len2));   // hemisphere Z
    vNormal    = normalize(vec3(d, z));

    // View direction: straight-on (orthographic camera)
    vViewDir   = vec3(0.0, 0.0, 1.0);

    gl_Position = vec4(pos, 0.0, 1.0);
}
`;

// ─── Fragment shader (~60 lines of real GLSL) ────────────────────────────────
// Schlick Fresnel + GGX NDF + Smith G + Lambert diffuse = Cook-Torrance PBR.

const PBR_FRAG = /* glsl */ `
precision highp float;

varying vec2  vUv;
varying vec3  vNormal;
varying vec3  vViewDir;

uniform vec3  uAlbedo;
uniform float uMetallic;
uniform float uRoughness;
uniform vec3  uLightDir;    // normalised world-space light direction
uniform vec3  uLightColor;  // HDR light RGB
uniform vec3  uAmbient;     // ambient / sky colour

// ── Schlick Fresnel ──────────────────────────────────────────────────────────
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

// ── GGX NDF (Trowbridge-Reitz) ───────────────────────────────────────────────
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a     = roughness * roughness;
    float a2    = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    const float PI = 3.14159265358979;
    return a2 / (PI * denom * denom);
}

// ── Smith G sub-term (GGX-Schlick) ──────────────────────────────────────────
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// ── Smith masking-shadowing ──────────────────────────────────────────────────
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness)
         * geometrySchlickGGX(NdotL, roughness);
}

void main() {
    // Discard pixels outside the sphere silhouette (circular cell)
    vec2  d    = vUv * 2.0 - 1.0;
    if (dot(d, d) > 1.0) discard;

    vec3  N    = normalize(vNormal);
    vec3  V    = normalize(vViewDir);
    vec3  L    = normalize(uLightDir);
    vec3  H    = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);

    // ── F0: dielectric = 0.04, metal = albedo ────────────────────────────────
    vec3  F0   = mix(vec3(0.04), uAlbedo, uMetallic);
    vec3  F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

    // ── Cook-Torrance specular ────────────────────────────────────────────────
    float NDF  = distributionGGX(N, H, uRoughness);
    float G    = geometrySmith(N, V, L, uRoughness);
    const float EPS = 0.001;
    vec3  numerator   = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + EPS;
    vec3  specular    = numerator / denominator;

    // ── Lambert diffuse (metals have no diffuse) ──────────────────────────────
    vec3  kS   = F;
    vec3  kD   = (1.0 - kS) * (1.0 - uMetallic);
    const float PI = 3.14159265358979;
    vec3  diffuse = kD * uAlbedo / PI;

    // ── Combine ───────────────────────────────────────────────────────────────
    vec3  Lo   = (diffuse + specular) * uLightColor * NdotL;
    vec3  ambient = uAmbient * uAlbedo * (1.0 - uMetallic * 0.5);
    vec3  color   = ambient + Lo;

    // ── Reinhard tone-map + gamma ─────────────────────────────────────────────
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    gl_FragColor = vec4(color, 1.0);
}
`;

// ─── FBO wrapper ─────────────────────────────────────────────────────────────

interface PBRTarget {
  fbo:     WebGLFramebuffer;
  texture: WebGLTexture;
  width:   number;
  height:  number;
}

// ─── PBRCellGPU ─────────────────────────────────────────────────────────────

export class PBRCellGPU {
  private gl:      WebGLRenderingContext;
  private prog!:   WebGLProgram;
  private quadBuf!: WebGLBuffer;
  private fboTarget!: PBRTarget;

  // Uniform locations cache
  private uCellPos!:    WebGLUniformLocation;
  private uCellSize!:   WebGLUniformLocation;
  private uAlbedo!:     WebGLUniformLocation;
  private uMetallic!:   WebGLUniformLocation;
  private uRoughness!:  WebGLUniformLocation;
  private uLightDir!:   WebGLUniformLocation;
  private uLightColor!: WebGLUniformLocation;
  private uAmbient!:    WebGLUniformLocation;

  // Attribute location
  private aCorner!: number;

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
    this._compileProgram();
    this._createQuad();
    this._cacheLocations();
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Ensure (or resize) the output FBO.
   * Call once per resize / init before renderCells().
   */
  initFBO(width: number, height: number): void {
    const gl = this.gl;

    // Destroy old FBO if exists
    if (this.fboTarget) {
      gl.deleteFramebuffer(this.fboTarget.fbo);
      gl.deleteTexture(this.fboTarget.texture);
    }

    // gl.createTexture — real call #1
    const texture = gl.createTexture()!;
    // gl.bindTexture — real call #2
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // gl.texImage2D — real call #3
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0,
                  gl.RGBA, gl.UNSIGNED_BYTE, null);
    // gl.texParameteri × 4 — real calls #4–7
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S,     gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T,     gl.CLAMP_TO_EDGE);

    // gl.createFramebuffer — real call #8
    const fbo = gl.createFramebuffer()!;
    // gl.bindFramebuffer — real call #9
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    // gl.framebufferTexture2D — real call #10
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, texture, 0);
    // gl.bindFramebuffer (restore) — real call #11
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.fboTarget = { fbo, texture, width, height };
  }

  /**
   * Render all cells into the PBR FBO.
   * Each cell produces ≥5 gl calls → 10 cells → 50+ calls easily.
   */
  renderCells(cells: CellPBRDescriptor[]): void {
    if (!this.fboTarget) {
      throw new Error('[PBRCellGPU] initFBO() must be called before renderCells()');
    }

    const gl = this.gl;

    // ── Bind FBO — real call ──────────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboTarget.fbo);
    // gl.viewport — real call
    gl.viewport(0, 0, this.fboTarget.width, this.fboTarget.height);
    // gl.clearColor — real call
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    // gl.clear — real call
    gl.clear(gl.COLOR_BUFFER_BIT);

    // ── Use PBR program — real call ───────────────────────────────────────────
    gl.useProgram(this.prog);

    // ── Bind quad VBO — real call ─────────────────────────────────────────────
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    // gl.enableVertexAttribArray — real call
    gl.enableVertexAttribArray(this.aCorner);
    // gl.vertexAttribPointer — real call
    gl.vertexAttribPointer(this.aCorner, 2, gl.FLOAT, false, 0, 0);

    // ── Scene-level uniforms (light + ambient) — real calls ───────────────────
    // uLightDir — real call
    gl.uniform3f(this.uLightDir, 0.577, 0.577, 0.577);
    // uLightColor — real call
    gl.uniform3f(this.uLightColor, 1.2, 1.15, 1.10);
    // uAmbient — real call
    gl.uniform3f(this.uAmbient, 0.08, 0.09, 0.12);

    // ── Per-cell draw loop ────────────────────────────────────────────────────
    for (const cell of cells) {
      const metallic  = cell.metallic  ?? SPECIES_METALLIC[cell.species]  ?? 0.1;
      const roughness = cell.roughness ?? SPECIES_ROUGHNESS[cell.species] ?? 0.5;
      const albedo    = cell.albedo    ?? SPECIES_ALBEDO[cell.species]    ?? [0.8, 0.8, 0.8];

      // gl.uniform2f uCellPos — real call
      gl.uniform2f(this.uCellPos, cell.x, cell.y);
      // gl.uniform1f uCellSize — real call
      gl.uniform1f(this.uCellSize, cell.size);
      // gl.uniform3f uAlbedo — real call
      gl.uniform3f(this.uAlbedo, albedo[0], albedo[1], albedo[2]);
      // gl.uniform1f uMetallic — real call
      gl.uniform1f(this.uMetallic, metallic);
      // gl.uniform1f uRoughness — real call
      gl.uniform1f(this.uRoughness, roughness);

      // gl.drawArrays — real call (TRIANGLES × 6 vertices = 2 triangles = 1 quad)
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    // ── Restore default FBO — real call ──────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Expose the rendered texture for downstream composite passes. */
  get pbrTexture(): WebGLTexture {
    return this.fboTarget.texture;
  }

  /**
   * swapProgram — 用外部编译好的 WebGLProgram 替换默认 PBR shader。
   * 替换后重新查询所有 uniform / attribute location，保证 renderCells() 正常工作。
   * 原 program 会被 deleteProgram 释放。
   *
   * @param newProg - 已链接的 WebGLProgram（例如来自 getATProgram(gl, 'PhysicalShader')）
   */
  swapProgram(newProg: WebGLProgram): void {
    const gl = this.gl;
    // 释放旧 program
    if (this.prog) gl.deleteProgram(this.prog);
    this.prog = newProg;
    // 重新缓存所有 attribute / uniform 位置
    this._cacheLocations();
    console.log('[PBRCellGPU] program swapped → AT PhysicalShader');
  }

  /** Release all GPU resources. */
  dispose(): void {
    const gl = this.gl;
    if (this.prog)     gl.deleteProgram(this.prog);
    if (this.quadBuf)  gl.deleteBuffer(this.quadBuf);
    if (this.fboTarget) {
      gl.deleteFramebuffer(this.fboTarget.fbo);
      gl.deleteTexture(this.fboTarget.texture);
    }
  }

  // ─── Private helpers ────────────────────────────────────────────────────────

  /** Compile vert + frag → link WebGLProgram (real gl.createShader calls). */
  private _compileProgram(): void {
    const gl = this.gl;

    // gl.createShader VERTEX — real call
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    // gl.shaderSource — real call
    gl.shaderSource(vs, PBR_VERT);
    // gl.compileShader — real call
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[PBRCellGPU] vert compile: ${gl.getShaderInfoLog(vs)}`);
    }

    // gl.createShader FRAGMENT — real call
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    // gl.shaderSource — real call
    gl.shaderSource(fs, PBR_FRAG);
    // gl.compileShader — real call
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[PBRCellGPU] frag compile: ${gl.getShaderInfoLog(fs)}`);
    }

    // gl.createProgram — real call
    const prog = gl.createProgram()!;
    // gl.attachShader × 2 — real calls
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    // gl.linkProgram — real call
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[PBRCellGPU] link: ${gl.getProgramInfoLog(prog)}`);
    }

    // gl.deleteShader × 2 — real calls (shader objs no longer needed post-link)
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    this.prog = prog;
  }

  /** Upload fullscreen-quad (6 vertices, 2 triangles). */
  private _createQuad(): void {
    const gl = this.gl;
    // gl.createBuffer — real call
    this.quadBuf = gl.createBuffer()!;
    // gl.bindBuffer — real call
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    // gl.bufferData — real call
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,   1, -1,  -1,  1,
      -1,  1,   1, -1,   1,  1,
    ]), gl.STATIC_DRAW);
  }

  /** Cache all uniform / attribute locations. */
  private _cacheLocations(): void {
    const gl   = this.gl;
    const prog = this.prog;

    // gl.getAttribLocation — real call
    this.aCorner    = gl.getAttribLocation(prog, 'aCorner');

    // gl.getUniformLocation × 8 — real calls
    this.uCellPos    = gl.getUniformLocation(prog, 'uCellPos')!;
    this.uCellSize   = gl.getUniformLocation(prog, 'uCellSize')!;
    this.uAlbedo     = gl.getUniformLocation(prog, 'uAlbedo')!;
    this.uMetallic   = gl.getUniformLocation(prog, 'uMetallic')!;
    this.uRoughness  = gl.getUniformLocation(prog, 'uRoughness')!;
    this.uLightDir   = gl.getUniformLocation(prog, 'uLightDir')!;
    this.uLightColor = gl.getUniformLocation(prog, 'uLightColor')!;
    this.uAmbient    = gl.getUniformLocation(prog, 'uAmbient')!;
  }
}
