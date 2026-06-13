/**
 * CloudFog.ts — Volumetric fog effect (AT CloudFog module analogue)
 *
 * Technique: N semi-transparent quads (AT default: 20) are distributed across
 * a 3-D bounding volume.  Each quad is rendered with cloud-fog.frag which
 * samples an animated 2-D noise function and weights the result by depth and
 * edge fade.  The accumulated alpha of all planes fakes volume scattering.
 *
 * AT CloudFog parameter reference:
 *   alpha=1.8, planes=20, noise=1, speed=0.7
 *   width=[-4,4], height=[-1,4], depth=[-2,-2]
 *   fadeDist=[2,4], cullDistance=999, scale=6
 *
 * Rendering:
 *   - Pure WebGL2 (no Three.js / PixiJS dependency) so it composes with any
 *     host renderer that exposes a raw WebGL2RenderingContext.
 *   - Planes are sorted back-to-front at construction time and rendered with
 *     ONE_MINUS_SRC_ALPHA blending (standard alpha).
 *   - The component owns its GL resources and must be disposed via .destroy().
 *
 * Usage:
 *   const fog = new CloudFog(gl, { planes: 20, alpha: 1.8, noise: 1 });
 *   // in render loop:
 *   fog.render(projectionMatrix, viewMatrix, elapsedSeconds);
 *   // cleanup:
 *   fog.destroy();
 *
 * Author: xiaodi <xiaodi@astro.dev>
 */

// ── Inline shader source (imported at build time via ?raw or inlined here) ────
// We inline the GLSL so CloudFog.ts is a single self-contained module that
// works without bundler plugin configuration.

const VERT_SRC = /* glsl */ `#version 300 es
precision highp float;

in vec2  aPosition;
in vec2  aUV;

uniform mat4  uProjection;
uniform mat4  uView;
uniform mat4  uModel;
uniform float uDepth01;

out vec2  vUV;
out float vDepth01;

void main() {
    vUV      = aUV;
    vDepth01 = uDepth01;
    vec4 worldPos = uModel * vec4(aPosition, 0.0, 1.0);
    gl_Position   = uProjection * uView * worldPos;
}
`;

const FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;

in vec2  vUV;
in float vDepth01;

uniform float uAlpha;
uniform float uNoise;
uniform float uSpeed;
uniform float uScale;
uniform float uTime;
uniform int   uLayerIndex;
uniform int   uPlaneCount;
uniform float uFadeNear;
uniform float uFadeFar;
uniform vec3  uFogColor;

out vec4 finalColor;

// ── Simplex 2-D noise (Gustavson, public domain) ─────────────────────────────
vec3 _mod289v3(vec3 x){ return x - floor(x*(1./289.))*289.; }
vec2 _mod289v2(vec2 x){ return x - floor(x*(1./289.))*289.; }
vec3 _permute(vec3 x){ return _mod289v3(((x*34.)+10.)*x); }

float snoise2(vec2 v){
    const vec4 C = vec4(0.211324865405187,0.366025403784439,-0.577350269189626,0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1  = (x0.x > x0.y) ? vec2(1.,0.) : vec2(0.,1.);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy  -= i1;
    i = _mod289v2(i);
    vec3 p = _permute(_permute(i.y+vec3(0.,i1.y,1.))+i.x+vec3(0.,i1.x,1.));
    vec3 m = max(0.5-vec3(dot(x0,x0),dot(x12.xy,x12.xy),dot(x12.zw,x12.zw)),0.);
    m=m*m; m=m*m;
    vec3 x  = 2.*fract(p*C.www)-1.;
    vec3 h  = abs(x)-0.5;
    vec3 ox = floor(x+0.5);
    vec3 a0 = x-ox;
    m *= 1.79284291400159-0.85373472095314*(a0*a0+h*h);
    vec3 g;
    g.x  = a0.x *x0.x   +h.x *x0.y;
    g.yz = a0.yz*x12.xz +h.yz*x12.yw;
    return 130.*dot(m,g);
}

float fbm(vec2 p){
    float n = snoise2(p)*0.6 + snoise2(p*2.1)*0.4;
    return clamp(n*0.5+0.5, 0., 1.);
}

void main(){
    float layerT  = float(uLayerIndex)/max(float(uPlaneCount)-1.,1.);
    float phaseX  = uTime*uSpeed*(0.7+layerT*0.3);
    float phaseY  = uTime*uSpeed*(0.5-layerT*0.2);
    vec2 noiseUV  = vUV*uScale+vec2(phaseX,phaseY);
    float density = fbm(noiseUV);
    float fogDensity = mix(0.65, density, clamp(uNoise,0.,1.));

    float depthFade = smoothstep(0., uFadeNear/(uFadeNear+uFadeFar), vDepth01)
                    * smoothstep(1., 1.-uFadeNear/(uFadeNear+uFadeFar), vDepth01);

    vec2  centered = vUV*2.-1.;
    float edgeFade = 1.-smoothstep(0.6, 1., length(centered));

    float layerAlpha = uAlpha*fogDensity*depthFade*edgeFade*mix(0.4,1.,layerT);
    float alpha = clamp(layerAlpha*0.12, 0., 1.);
    if(alpha < 0.002) discard;
    finalColor = vec4(uFogColor*alpha, alpha);
}
`;

// ── Types ──────────────────────────────────────────────────────────────────────

/** AT CloudFog parameter block — matches upstream UIL API surface. */
export interface CloudFogOptions {
  /**
   * Master opacity scale across all planes. AT default: 1.8.
   * Values > 1 are allowed; final per-fragment alpha is clamped.
   */
  alpha?: number;

  /** Number of stacked fog planes. AT default: 20. */
  planes?: number;

  /**
   * Noise intensity in [0, 1]. 0 = flat uniform fog, 1 = full noise.
   * AT default: 1.
   */
  noise?: number;

  /** Animation drift speed in world-units/second. AT default: 0.7. */
  speed?: number;

  /** Noise domain scale — larger values produce coarser fog clumps. AT default: 6. */
  scale?: number;

  /** World-space volume extents [minX, maxX]. AT default: [-4, 4]. */
  width?: [number, number];

  /** World-space volume extents [minY, maxY]. AT default: [-1, 4]. */
  height?: [number, number];

  /**
   * World-space depth of the fog slab [frontZ, backZ].
   * Both values are the same in AT (-2) meaning a flat slab; provide a range
   * to spread planes across Z. Default: [-2, -6].
   */
  depth?: [number, number];

  /**
   * Fade distances [near, far] in world units — planes near either edge of the
   * volume fade out.  AT default: [2, 4].
   */
  fadeDist?: [number, number];

  /**
   * Maximum camera distance at which the fog is rendered.
   * Planes beyond this are culled.  AT default: 999.
   */
  cullDistance?: number;

  /** Fog tint colour as [r, g, b] in [0, 1]. Default: white [1, 1, 1]. */
  fogColor?: [number, number, number];
}

/** Resolved, fully-specified options (all fields defined). */
type ResolvedOptions = Required<CloudFogOptions>;

// ── Quad geometry helpers ──────────────────────────────────────────────────────

/** Unit quad: 4 vertices, 2 triangles, interleaved [x, y, u, v]. */
const QUAD_INTERLEAVED = new Float32Array([
  // x      y      u    v
  -0.5,  -0.5,   0.0, 0.0,
   0.5,  -0.5,   1.0, 0.0,
   0.5,   0.5,   1.0, 1.0,
  -0.5,   0.5,   0.0, 1.0,
]);

const QUAD_INDICES = new Uint16Array([0, 1, 2, 0, 2, 3]);

// ── 4×4 matrix utilities (column-major, matches WebGL convention) ──────────────

type Mat4 = Float32Array;

function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** Scale then translate a copy of the identity matrix — sufficient for fog planes. */
function mat4PlaneModel(
  tx: number, ty: number, tz: number,
  sx: number, sy: number,
): Mat4 {
  const m = mat4Identity();
  m[0]  = sx;   // scale X
  m[5]  = sy;   // scale Y
  m[12] = tx;   // translate X
  m[13] = ty;   // translate Y
  m[14] = tz;   // translate Z
  return m;
}

// ── Shader compilation helpers ─────────────────────────────────────────────────

function compileShader(
  gl: WebGL2RenderingContext,
  type: GLenum,
  src: string,
): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? 'unknown error';
    gl.deleteShader(shader);
    throw new Error(`[CloudFog] Shader compile error:\n${log}`);
  }
  return shader;
}

function linkProgram(
  gl: WebGL2RenderingContext,
  vert: WebGLShader,
  frag: WebGLShader,
): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vert);
  gl.attachShader(prog, frag);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog) ?? 'unknown error';
    gl.deleteProgram(prog);
    throw new Error(`[CloudFog] Program link error:\n${log}`);
  }
  return prog;
}

// ── Fog plane descriptor ───────────────────────────────────────────────────────

interface FogPlane {
  /** Index within the plane array (0 = back). */
  index: number;
  /** Normalised position [0, 1] within volume depth. */
  depth01: number;
  /** Pre-built model matrix for this plane. */
  model: Mat4;
}

// ── CloudFog ──────────────────────────────────────────────────────────────────

/**
 * CloudFog — volumetric fog renderer matching AT's CloudFog UIL module.
 *
 * Owns all WebGL resources for the effect.  Call `.destroy()` to release them.
 *
 * @example
 * ```ts
 * const fog = new CloudFog(gl);
 * // animation loop:
 * fog.render(projectionMat4, viewMat4, performance.now() / 1000);
 * ```
 */
export class CloudFog {
  // ── GL resources ────────────────────────────────────────────────────────────
  private readonly gl:   WebGL2RenderingContext;
  private readonly prog: WebGLProgram;
  private readonly vao:  WebGLVertexArrayObject;
  private readonly vbo:  WebGLBuffer;
  private readonly ebo:  WebGLBuffer;

  // ── Uniform locations ────────────────────────────────────────────────────────
  private readonly uProjection: WebGLUniformLocation;
  private readonly uView:       WebGLUniformLocation;
  private readonly uModel:      WebGLUniformLocation;
  private readonly uDepth01:    WebGLUniformLocation;
  private readonly uAlpha:      WebGLUniformLocation;
  private readonly uNoise:      WebGLUniformLocation;
  private readonly uSpeed:      WebGLUniformLocation;
  private readonly uScale:      WebGLUniformLocation;
  private readonly uTime:       WebGLUniformLocation;
  private readonly uLayerIndex: WebGLUniformLocation;
  private readonly uPlaneCount: WebGLUniformLocation;
  private readonly uFadeNear:   WebGLUniformLocation;
  private readonly uFadeFar:    WebGLUniformLocation;
  private readonly uFogColor:   WebGLUniformLocation;

  // ── Scene data ───────────────────────────────────────────────────────────────
  private readonly planes: FogPlane[];
  private readonly opts:   ResolvedOptions;

  // ── Diagnostics ──────────────────────────────────────────────────────────────
  /** Total render calls since construction (increments each .render()). */
  public frameCount = 0;

  // ── Constructor ──────────────────────────────────────────────────────────────

  constructor(gl: WebGL2RenderingContext, options: CloudFogOptions = {}) {
    this.gl = gl;

    // Resolve options with AT defaults
    this.opts = {
      alpha:        options.alpha        ?? 1.8,
      planes:       options.planes       ?? 20,
      noise:        options.noise        ?? 1.0,
      speed:        options.speed        ?? 0.7,
      scale:        options.scale        ?? 6.0,
      width:        options.width        ?? [-4, 4],
      height:       options.height       ?? [-1, 4],
      depth:        options.depth        ?? [-2, -6],
      fadeDist:     options.fadeDist     ?? [2, 4],
      cullDistance: options.cullDistance ?? 999,
      fogColor:     options.fogColor     ?? [1, 1, 1],
    };

    // ── Compile shaders ──────────────────────────────────────────────────────
    const vert = compileShader(gl, gl.VERTEX_SHADER,   VERT_SRC);
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
    this.prog  = linkProgram(gl, vert, frag);
    gl.deleteShader(vert);
    gl.deleteShader(frag);

    // ── Uniform locations ────────────────────────────────────────────────────
    const u = (name: string) => {
      const loc = gl.getUniformLocation(this.prog, name);
      if (loc === null) {
        // Non-fatal: GLSL optimizer may have pruned unused uniforms.
        console.warn(`[CloudFog] Uniform "${name}" not found in program.`);
        return null as unknown as WebGLUniformLocation;
      }
      return loc;
    };
    this.uProjection = u('uProjection');
    this.uView       = u('uView');
    this.uModel      = u('uModel');
    this.uDepth01    = u('uDepth01');
    this.uAlpha      = u('uAlpha');
    this.uNoise      = u('uNoise');
    this.uSpeed      = u('uSpeed');
    this.uScale      = u('uScale');
    this.uTime       = u('uTime');
    this.uLayerIndex = u('uLayerIndex');
    this.uPlaneCount = u('uPlaneCount');
    this.uFadeNear   = u('uFadeNear');
    this.uFadeFar    = u('uFadeFar');
    this.uFogColor   = u('uFogColor');

    // ── Shared quad geometry ─────────────────────────────────────────────────
    this.vao = gl.createVertexArray()!;
    this.vbo = gl.createBuffer()!;
    this.ebo = gl.createBuffer()!;

    gl.bindVertexArray(this.vao);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, QUAD_INTERLEAVED, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, QUAD_INDICES, gl.STATIC_DRAW);

    const aPosition = gl.getAttribLocation(this.prog, 'aPosition');
    const aUV       = gl.getAttribLocation(this.prog, 'aUV');

    const STRIDE = 4 * Float32Array.BYTES_PER_ELEMENT; // 4 floats per vertex
    if (aPosition >= 0) {
      gl.enableVertexAttribArray(aPosition);
      gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, STRIDE, 0);
    }
    if (aUV >= 0) {
      gl.enableVertexAttribArray(aUV);
      gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, STRIDE, 2 * Float32Array.BYTES_PER_ELEMENT);
    }

    gl.bindVertexArray(null);

    // ── Build plane descriptors (back → front order for correct blending) ────
    this.planes = this._buildPlanes();
  }

  // ── Private: plane layout ──────────────────────────────────────────────────

  private _buildPlanes(): FogPlane[] {
    const { planes, width, height, depth } = this.opts;
    const [minX, maxX] = width;
    const [minY, maxY] = height;
    const [frontZ, backZ] = depth;  // frontZ is typically less negative

    const planeW = maxX - minX;
    const planeH = maxY - minY;
    const centerX = (minX + maxX) * 0.5;
    const centerY = (minY + maxY) * 0.5;

    const result: FogPlane[] = [];

    for (let i = 0; i < planes; i++) {
      // Distribute planes uniformly from back (i=0) to front (i=planes-1).
      const t      = planes > 1 ? i / (planes - 1) : 0.5;
      // Lerp in Z from backZ (more negative) to frontZ (less negative)
      const z      = backZ + (frontZ - backZ) * t;
      // Slight Y jitter per layer to break perfect regularity
      const yOff   = Math.sin(i * 1.37) * (planeH * 0.05);

      const model = mat4PlaneModel(centerX, centerY + yOff, z, planeW, planeH);

      result.push({ index: i, depth01: t, model });
    }

    // Back-to-front already (i=0 is back); painter's algorithm for additive blend
    return result;
  }

  // ── Public: update options at runtime ────────────────────────────────────────

  /**
   * Hot-update fog parameters without recreating GL resources.
   * Only updates the resolved option values; plane geometry is NOT rebuilt.
   * To change width / height / depth / planes, destroy and recreate.
   */
  public setOptions(patch: Partial<CloudFogOptions>): void {
    Object.assign(this.opts, patch);
  }

  // ── Public: render ────────────────────────────────────────────────────────────

  /**
   * Render all fog planes.
   *
   * Must be called every frame inside the host renderer's render loop.
   *
   * @param projection  Column-major 4×4 projection matrix (Float32Array, 16 elements)
   * @param view        Column-major 4×4 view/camera matrix (Float32Array, 16 elements)
   * @param timeSeconds Elapsed time in seconds (drives noise animation)
   * @param cameraZ     Optional camera Z for cullDistance check (default 0)
   */
  public render(
    projection: Float32Array,
    view:       Float32Array,
    timeSeconds: number,
    cameraZ = 0,
  ): void {
    const { gl, opts } = this;

    // ── Cull entire effect if camera is too far ──────────────────────────────
    const [frontZ] = opts.depth;
    if (Math.abs(cameraZ - frontZ) > opts.cullDistance) return;

    // ── GL state setup ───────────────────────────────────────────────────────
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);

    // Standard alpha blending (SRC_ALPHA + ONE_MINUS_SRC_ALPHA)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    // Fog planes write colour only — no depth writes to avoid z-fighting
    gl.depthMask(false);

    // ── Shared uniforms (same for all planes) ────────────────────────────────
    gl.uniformMatrix4fv(this.uProjection, false, projection);
    gl.uniformMatrix4fv(this.uView,       false, view);
    gl.uniform1f(this.uAlpha,      opts.alpha);
    gl.uniform1f(this.uNoise,      opts.noise);
    gl.uniform1f(this.uSpeed,      opts.speed);
    gl.uniform1f(this.uScale,      opts.scale);
    gl.uniform1f(this.uTime,       timeSeconds);
    gl.uniform1i(this.uPlaneCount, opts.planes);
    gl.uniform1f(this.uFadeNear,   opts.fadeDist[0]);
    gl.uniform1f(this.uFadeFar,    opts.fadeDist[1]);
    gl.uniform3fv(this.uFogColor,  opts.fogColor);

    // ── Per-plane draw calls ─────────────────────────────────────────────────
    for (const plane of this.planes) {
      gl.uniformMatrix4fv(this.uModel,      false, plane.model);
      gl.uniform1f(this.uDepth01,   plane.depth01);
      gl.uniform1i(this.uLayerIndex, plane.index);

      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    }

    // ── Restore GL state ─────────────────────────────────────────────────────
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
    gl.useProgram(null);

    this.frameCount++;
  }

  // ── Public: dispose ───────────────────────────────────────────────────────────

  /**
   * Release all WebGL resources owned by this CloudFog instance.
   * After calling destroy() the instance must not be used.
   */
  public destroy(): void {
    const { gl } = this;
    gl.deleteProgram(this.prog);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.vbo);
    gl.deleteBuffer(this.ebo);
  }

  // ── Getters ───────────────────────────────────────────────────────────────────

  /** Read-only view of the resolved options. */
  get options(): Readonly<ResolvedOptions> {
    return this.opts;
  }

  /** Number of draw calls issued per frame (equals plane count). */
  get drawCallsPerFrame(): number {
    return this.opts.planes;
  }
}
