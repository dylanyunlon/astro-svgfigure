/**
 * cell-mesh-renderer.ts — M1261: 3D mesh renderer for cells
 *
 * Replaces 2D quad PBR with actual 3D GLB meshes (from Rodin or any source).
 * Each species maps to a GLB file in public/models/.
 * Uses orthographic projection matching the auto-fit camera in gpu-render-loop.
 *
 * Architecture:
 *   1. Load GLB per species via threed-pipeline.ts GLTFLoader
 *   2. Build instanced draw data from CellData[]
 *   3. Render with correct ortho projection: cell pixel space → NDC
 *
 * Coordinate spaces:
 *   Cell pixel space:  x ∈ [0, 2052], y ∈ [0, 3965]
 *   Canvas pixel:      x ∈ [0, W],    y ∈ [0, H]
 *   NDC:               x ∈ [-1, 1],   y ∈ [-1, 1]
 *
 *   The auto-fit camera in gpu-render-loop computes camScale/camOffX/camOffY
 *   to map cell space → canvas space. This renderer builds an ortho matrix
 *   that does cell space → NDC directly, matching that same mapping.
 */

// ─── Shader sources ──────────────────────────────────────────────────────────

const MESH_VERT = /* glsl */ `#version 300 es
precision highp float;

// Per-vertex
in vec3 aPosition;
in vec3 aNormal;
in vec2 aUV;

// Per-instance (set via uniform for now; instanced attrs later)
uniform mat4 uModelMatrix;
uniform mat4 uViewProjMatrix;

uniform vec3 uAlbedo;

out vec3 vNormal;
out vec3 vWorldPos;
out vec2 vUV;
out vec3 vAlbedo;

void main() {
    vec4 worldPos = uModelMatrix * vec4(aPosition, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal   = mat3(uModelMatrix) * aNormal;
    vUV       = aUV;
    vAlbedo   = uAlbedo;
    gl_Position = uViewProjMatrix * worldPos;
}
`;

const MESH_FRAG = /* glsl */ `#version 300 es
precision highp float;

in vec3 vNormal;
in vec3 vWorldPos;
in vec2 vUV;
in vec3 vAlbedo;

uniform vec3  uLightDir;
uniform vec3  uLightColor;
uniform vec3  uAmbient;
uniform float uOpacity;
uniform vec3  uGlowColor;
uniform float uTime;

out vec4 fragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    float NdotL = max(dot(N, L), 0.0);

    // Simple PBR-ish lighting
    vec3 diffuse = vAlbedo * uLightColor * NdotL;
    vec3 ambient = uAmbient * vAlbedo;

    // Fresnel rim glow
    vec3 V = vec3(0.0, 0.0, 1.0); // ortho camera
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 rim = uGlowColor * fresnel * 0.4;

    vec3 color = ambient + diffuse + rim;

    // Tone map
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    fragColor = vec4(color, uOpacity);
}
`;

// ─── Placeholder cube geometry ───────────────────────────────────────────────

function createPlaceholderCube(): {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  indices: Uint16Array;
} {
  // Unit cube centered at origin, [-0.5, 0.5]
  // 24 vertices (4 per face, unique normals)
  const p = 0.5;
  // prettier-ignore
  const positions = new Float32Array([
    // Front face
    -p, -p,  p,   p, -p,  p,   p,  p,  p,  -p,  p,  p,
    // Back face
    -p, -p, -p,  -p,  p, -p,   p,  p, -p,   p, -p, -p,
    // Top face
    -p,  p, -p,  -p,  p,  p,   p,  p,  p,   p,  p, -p,
    // Bottom face
    -p, -p, -p,   p, -p, -p,   p, -p,  p,  -p, -p,  p,
    // Right face
     p, -p, -p,   p,  p, -p,   p,  p,  p,   p, -p,  p,
    // Left face
    -p, -p, -p,  -p, -p,  p,  -p,  p,  p,  -p,  p, -p,
  ]);

  // prettier-ignore
  const normals = new Float32Array([
    // Front
    0,0,1, 0,0,1, 0,0,1, 0,0,1,
    // Back
    0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
    // Top
    0,1,0, 0,1,0, 0,1,0, 0,1,0,
    // Bottom
    0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0,
    // Right
    1,0,0, 1,0,0, 1,0,0, 1,0,0,
    // Left
    -1,0,0, -1,0,0, -1,0,0, -1,0,0,
  ]);

  // prettier-ignore
  const uvs = new Float32Array([
    0,0, 1,0, 1,1, 0,1,
    0,0, 1,0, 1,1, 0,1,
    0,0, 1,0, 1,1, 0,1,
    0,0, 1,0, 1,1, 0,1,
    0,0, 1,0, 1,1, 0,1,
    0,0, 1,0, 1,1, 0,1,
  ]);

  // prettier-ignore
  const indices = new Uint16Array([
    0,1,2,  0,2,3,     // front
    4,5,6,  4,6,7,     // back
    8,9,10, 8,10,11,   // top
    12,13,14, 12,14,15, // bottom
    16,17,18, 16,18,19, // right
    20,21,22, 20,22,23, // left
  ]);

  return { positions, normals, uvs, indices };
}

// ─── Orthographic projection matrix ──────────────────────────────────────────

/**
 * Build a column-major 4x4 orthographic projection matrix.
 * Maps [left, right] × [bottom, top] × [near, far] → NDC [-1,1]³
 */
function ortho(
  left: number, right: number,
  bottom: number, top: number,
  near: number, far: number,
): Float32Array {
  const lr = 1 / (right - left);
  const bt = 1 / (top - bottom);
  const nf = 1 / (far - near);
  // Column-major
  return new Float32Array([
    2 * lr,        0,             0,             0,
    0,             2 * bt,        0,             0,
    0,             0,            -2 * nf,        0,
    -(right+left)*lr, -(top+bottom)*bt, -(far+near)*nf, 1,
  ]);
}

/**
 * Build a model matrix: translate to (cx, cy, 0) and scale to (sx, sy, sz).
 * Column-major 4x4.
 */
function modelMatrix(
  cx: number, cy: number, cz: number,
  sx: number, sy: number, sz: number,
): Float32Array {
  return new Float32Array([
    sx, 0,  0,  0,
    0,  sy, 0,  0,
    0,  0,  sz, 0,
    cx, cy, cz, 1,
  ]);
}

/**
 * Apply a rotation about the Z axis to a column-major 4x4 matrix in place.
 * Multiplies model = model * Rz(angle), so it rotates in the model's local frame
 * before the existing translation/scale takes effect.
 */
function rotateZ(mat: Float32Array, angle: number): void {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  // Columns 0 and 1 are affected (X and Y basis vectors).
  for (let i = 0; i < 3; i++) {
    const a = mat[i];       // column 0, row i
    const b = mat[4 + i];   // column 1, row i
    mat[i]     = a * c + b * s;
    mat[4 + i] = -a * s + b * c;
  }
}

/**
 * Apply a rotation about the Y axis to a column-major 4x4 matrix in place.
 * Multiplies model = model * Ry(angle).
 */
function rotateY(mat: Float32Array, angle: number): void {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  // Columns 0 and 2 are affected (X and Z basis vectors).
  for (let i = 0; i < 3; i++) {
    const a = mat[i];       // column 0, row i
    const b = mat[8 + i];   // column 2, row i
    mat[i]     = a * c - b * s;
    mat[8 + i] = a * s + b * c;
  }
}

// ─── CellMeshRenderer ────────────────────────────────────────────────────────

import type { CellData } from './gpu-render-loop';
import { SPECIES_GEOMETRY } from './procedural-cell-geometries';

/** Per-species GPU mesh (VBO + IBO + VAO) */
interface SpeciesMesh {
  vao: WebGLVertexArrayObject;
  indexCount: number;
  indexType: number; // gl.UNSIGNED_SHORT or gl.UNSIGNED_INT
}

export class CellMeshRenderer {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;

  // Uniform locations
  private uModelMatrix!:    WebGLUniformLocation;
  private uViewProjMatrix!: WebGLUniformLocation;
  private uAlbedo!:         WebGLUniformLocation;
  private uLightDir!:       WebGLUniformLocation;
  private uLightColor!:     WebGLUniformLocation;
  private uAmbient!:        WebGLUniformLocation;
  private uOpacity!:        WebGLUniformLocation;
  private uGlowColor!:      WebGLUniformLocation;
  private uTime!:           WebGLUniformLocation;

  // Attribute locations
  private aPosition!: number;
  private aNormal!:   number;
  private aUV!:       number;

  // Per-species mesh cache
  private meshes = new Map<string, SpeciesMesh>();

  // FBO for rendering to texture (so composite can consume it)
  private fbo: WebGLFramebuffer | null = null;
  private colorTex: WebGLTexture | null = null;
  private depthRB: WebGLRenderbuffer | null = null;
  private fboW = 0;
  private fboH = 0;

  private _time = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.prog = this._compileProgram();
    this._resolveLocations();
    this._uploadPlaceholder();
    // Auto-load GLB files (async, falls back to placeholder until loaded)
    this.loadAllSpeciesGLB().catch(e =>
      console.warn('[CellMeshRenderer] GLB auto-load failed, using procedural fallback:', e)
    );
  }

  /**
   * Load GLB files for all 5 species from /models/{species}.glb.
   * Non-blocking — cells render with procedural geometry until GLB loads.
   */
  async loadAllSpeciesGLB(): Promise<void> {
    console.info('[CellMeshRenderer] loadAllSpeciesGLB starting...');
    const species = ['cil-eye', 'cil-bolt', 'cil-vector', 'cil-plus', 'cil-arrow-right'];
    const results = await Promise.allSettled(
      species.map(s => this.loadSpeciesMesh(s, `/models/${s}.glb`))
    );
    const loaded = results.filter(r => r.status === 'fulfilled').length;
    console.info(`[CellMeshRenderer] GLB load: ${loaded}/${species.length} species`);
  }

  // ── Public API ──────────────────────────────────────────────────────────

  /** Set current time (seconds) for animation */
  setTime(t: number): void { this._time = t; }

  /** Output texture for composite pass */
  get outputTexture(): WebGLTexture | null { return this.colorTex; }

  /**
   * Render all cells as 3D meshes.
   *
   * Camera params come from gpu-render-loop's auto-fit camera:
   *   camScale, camOffX, camOffY map cell pixel space → canvas pixel space.
   *   W, H are canvas dimensions.
   *
   * We build an ortho projection that maps canvas pixel space → NDC.
   * Model matrix per cell: translate to fitted position, scale to fitted size.
   */
  render(
    cells: CellData[],
    camScale: number,
    camOffX: number,
    camOffY: number,
    W: number,
    H: number,
  ): void {
    const gl = this.gl;

    // ── Ensure FBO ────────────────────────────────────────────────────────
    if (!this.fbo || this.fboW !== W || this.fboH !== H) {
      this._initFBO(W, H);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(this.prog);

    // ── Orthographic view-projection: canvas pixel → NDC ────────────────
    // Canvas pixel (0,0) is top-left, but NDC (-1,-1) is bottom-left.
    // ortho(left=0, right=W, bottom=H, top=0, near=-100, far=100)
    // This flips Y so that pixel Y-down matches cell Y-down.
    const vpMatrix = ortho(0, W, H, 0, -100, 100);
    gl.uniformMatrix4fv(this.uViewProjMatrix, false, vpMatrix);

    // ── Scene-level uniforms ──────────────────────────────────────────────
    gl.uniform3f(this.uLightDir, -0.4, 0.3, 0.86);
    gl.uniform3f(this.uLightColor, 2.0, 1.95, 1.85);
    gl.uniform3f(this.uAmbient, 0.35, 0.38, 0.45);
    gl.uniform1f(this.uTime, this._time);

    // ── Per-cell rendering ────────────────────────────────────────────────
    for (const cell of cells) {
      // Cell centre in canvas pixel space
      const cx = cell.x * camScale + camOffX + cell.w * camScale * 0.5;
      const cy = cell.y * camScale + camOffY + cell.h * camScale * 0.5;
      const cz = cell.z ?? 0;

      // Cell size in canvas pixels
      const sw = cell.w * camScale;
      const sh = cell.h * camScale;
      // Z scale: average of w/h to keep proportional
      const sz = Math.min(sw, sh) * 0.5;

      const model = modelMatrix(cx, cy, cz, sw, sh, sz);

      // ── Species-specific 3D motion ──────────────────────────────────────
      const t = this._time;
      switch (cell.species) {
        case 'cil-eye':
          // Slow continuous spin about Z.
          rotateZ(model, t * 0.3);
          break;
        case 'cil-bolt':
          // Pulsing rotation about Y.
          rotateY(model, Math.sin(t * 3.0) * 0.5);
          break;
        case 'cil-vector':
          // Small X-axis translation pulse, no rotation.
          model[12] += Math.sin(t * 2) * 2;
          break;
        case 'cil-plus':
          // Back-and-forth wobble about Z.
          rotateZ(model, Math.sin(t * 1.5) * 0.8);
          break;
        case 'cil-arrow-right':
        default:
          // Static.
          break;
      }

      gl.uniformMatrix4fv(this.uModelMatrix, false, model);

      gl.uniform3f(this.uAlbedo, cell.albedo[0], cell.albedo[1], cell.albedo[2]);
      gl.uniform1f(this.uOpacity, cell.opacity ?? 0.9);
      gl.uniform3f(this.uGlowColor,
        cell.glowColor?.[0] ?? cell.albedo[0],
        cell.glowColor?.[1] ?? cell.albedo[1],
        cell.glowColor?.[2] ?? cell.albedo[2],
      );

      // Get species mesh (placeholder cube for now)
      const mesh = this.meshes.get(cell.species) ?? this.meshes.get('_placeholder')!;
      gl.bindVertexArray(mesh.vao);
      gl.drawElements(gl.TRIANGLES, mesh.indexCount, mesh.indexType, 0);
    }

    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Load a GLB model for a specific species.
   * Call this when Rodin GLBs are ready: renderer.loadSpeciesMesh('cil-eye', '/models/cil-eye.glb')
   * Until called, the placeholder cube is used.
   */
  async loadSpeciesMesh(species: string, glbUrl: string): Promise<void> {
    try {
      const { GLTFLoader } = await import('../threed-pipeline');
      // Standard GLBs (no KHR_draco_mesh_compression) — do NOT pass a
      // dracoThread. With no options the loader sets this.draco = null and
      // parses POSITION/NORMAL/TEXCOORD/indices directly from the bin chunk,
      // so no Web Worker / blob-URL importScripts path is ever touched.
      const loader = new GLTFLoader();
      const scene = await loader.load(glbUrl);

      // Use the first mesh found
      const firstMesh = scene.meshes.values().next().value;
      if (!firstMesh) {
        console.error(`[CellMeshRenderer] No mesh found in ${glbUrl}`);
        return;
      }

      const mesh = this._uploadGeometry(
        firstMesh.positions,
        firstMesh.normals ?? new Float32Array(firstMesh.positions.length),
        firstMesh.uvs ?? new Float32Array(firstMesh.vertexCount * 2),
        firstMesh.indices ?? null,
        firstMesh.vertexCount,
      );
      this.meshes.set(species, mesh);
      console.info(`[CellMeshRenderer] Loaded ${species} from ${glbUrl}: ${firstMesh.vertexCount} verts`);
    } catch (e) {
      console.error(`[CellMeshRenderer] Failed to load ${species} from ${glbUrl}:`, e);
      throw e;
    }
  }

  // ── Internal ────────────────────────────────────────────────────────────

  private _uploadPlaceholder(): void {
    // Upload per-species procedural geometries
    for (const [species, createFn] of Object.entries(SPECIES_GEOMETRY)) {
      const geo = createFn();
      const mesh = this._uploadGeometry(
        geo.positions,
        geo.normals,
        geo.uvs,
        geo.indices,
        geo.positions.length / 3,
      );
      this.meshes.set(species, mesh);
    }

    // Fallback: use cube for unknown species
    const cube = createPlaceholderCube();
    const fallback = this._uploadGeometry(
      cube.positions,
      cube.normals,
      cube.uvs,
      cube.indices,
      24,
    );
    this.meshes.set('_placeholder', fallback);
  }

  private _uploadGeometry(
    positions: Float32Array,
    normals: Float32Array,
    uvs: Float32Array,
    indices: Uint16Array | Uint32Array | null,
    vertexCount: number,
  ): SpeciesMesh {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    // Position
    const posBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.aPosition);
    gl.vertexAttribPointer(this.aPosition, 3, gl.FLOAT, false, 0, 0);

    // Normal
    const normBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.aNormal);
    gl.vertexAttribPointer(this.aNormal, 3, gl.FLOAT, false, 0, 0);

    // UV
    const uvBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.aUV);
    gl.vertexAttribPointer(this.aUV, 2, gl.FLOAT, false, 0, 0);

    // Index buffer
    let indexCount = vertexCount;
    let indexType: number = gl.UNSIGNED_SHORT;
    if (indices) {
      const idxBuf = gl.createBuffer()!;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
      indexCount = indices.length;
      indexType = indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    }

    gl.bindVertexArray(null);

    return { vao, indexCount, indexType };
  }

  private _initFBO(W: number, H: number): void {
    const gl = this.gl;

    // Clean up old
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    if (this.colorTex) gl.deleteTexture(this.colorTex);
    if (this.depthRB) gl.deleteRenderbuffer(this.depthRB);

    this.colorTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.depthRB = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.depthRB);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, W, H);

    this.fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.colorTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depthRB);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.error('[CellMeshRenderer] FBO incomplete:', status);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.fboW = W;
    this.fboH = H;
  }

  private _compileProgram(): WebGLProgram {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, MESH_VERT);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(vs);
      throw new Error(`[CellMeshRenderer] vertex shader error:\n${log}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, MESH_FRAG);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(fs);
      throw new Error(`[CellMeshRenderer] fragment shader error:\n${log}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      throw new Error(`[CellMeshRenderer] link error:\n${log}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private _resolveLocations(): void {
    const gl = this.gl;
    const prog = this.prog;

    this.aPosition = gl.getAttribLocation(prog, 'aPosition');
    this.aNormal   = gl.getAttribLocation(prog, 'aNormal');
    this.aUV       = gl.getAttribLocation(prog, 'aUV');

    this.uModelMatrix    = gl.getUniformLocation(prog, 'uModelMatrix')!;
    this.uViewProjMatrix = gl.getUniformLocation(prog, 'uViewProjMatrix')!;
    this.uAlbedo         = gl.getUniformLocation(prog, 'uAlbedo')!;
    this.uLightDir       = gl.getUniformLocation(prog, 'uLightDir')!;
    this.uLightColor     = gl.getUniformLocation(prog, 'uLightColor')!;
    this.uAmbient        = gl.getUniformLocation(prog, 'uAmbient')!;
    this.uOpacity        = gl.getUniformLocation(prog, 'uOpacity')!;
    this.uGlowColor      = gl.getUniformLocation(prog, 'uGlowColor')!;
    this.uTime           = gl.getUniformLocation(prog, 'uTime')!;
  }
}
