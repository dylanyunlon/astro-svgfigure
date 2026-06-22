/**
 * caustics-background.ts — AT caustic light-pattern world background
 *
 * Source lineage:
 *   src/lib/shaders/caustics.frag           (M605) — analytic water caustics
 *   upstream/webgl-water/renderer.js        (Evan Wallace, MIT) — refraction model
 *   src/lib/renderers/water-background.ts   (M704) — mount / lifecycle pattern
 *   src/lib/SceneLayoutPresets.ts line 769  — AT caustic_plane.bin (CleanRoom)
 *
 * Architecture:
 *   A single fullscreen-triangle pass renders real-time analytic water caustics
 *   onto a virtual pool floor. The technique avoids ray-marching entirely:
 *
 *   1. WAVE SYNTHESIS — Four sine-wave octaves (different directions, frequencies,
 *      amplitudes, phase velocities) are summed to produce a time-evolving,
 *      turbulent water height field h(uv, t).
 *
 *   2. ANALYTIC NORMAL — The surface normal is computed from the closed-form
 *      gradient ∇h (no finite differences on the height, only on the projection).
 *
 *   3. SNELL REFRACTION — Each surface point refracts the incoming light direction
 *      through the water surface via GLSL `refract()`, projecting the ray onto
 *      a virtual pool floor at configurable depth.
 *
 *   4. JACOBIAN CAUSTICS — A 5-tap finite-difference Jacobian of the displaced
 *      floor UV measures photon convergence. Bright spots form where rays converge.
 *
 *   5. MULTI-DEPTH BLEND — Three Jacobian layers at 60%, 100%, and 155% of
 *      uWaterDepth plus a low-frequency envelope provide volumetric depth.
 *
 *   6. FLOOR LIGHTING — Lambertian diffuse from the refracted light direction,
 *      a Blinn-Phong specular glint on the water surface, Fresnel edge vignette,
 *      and premultiplied alpha output.
 *
 * Public API (mirrors water-background.ts):
 *   const bg = await mountCausticsBackground(containerEl, options?)
 *   bg.update(dt)            — advance time + render (call each frame)
 *   bg.setFloorColor(hex)    — update pool floor tint
 *   bg.setCausticColor(hex)  — update caustic highlight tint
 *   bg.setDepth(d)           — update virtual pool depth
 *   bg.setWaveAmp(a)         — update wave amplitude
 *   bg.setWaveSpeed(s)       — update wave phase velocity
 *   bg.stop()                — tear down canvas + GL resources
 *
 * Research: xiaodi #M746 — cell-pubsub-loop
 */

import { createProgram } from './hydra-gl-layer';

// ─────────────────────────────────────────────────────────────────────────────
// Public config / handle
// ─────────────────────────────────────────────────────────────────────────────

export interface CausticsBackgroundOptions {
  /**
   * Pool floor colour (dark blue by default, matching AT CleanRoom scene).
   * Default '#0f2e52'
   */
  floorColor?: string;

  /**
   * Caustic highlight colour (pale cyan — refracted sunlight on the pool floor).
   * Default '#bfeaff'
   */
  causticColor?: string;

  /**
   * Normalised world-space light direction [x, y, z].
   * Default [0.3, 1.0, 0.5] — top-front-right sun.
   */
  lightDir?: [number, number, number];

  /**
   * Virtual pool depth — controls how far the refracted rays travel before
   * hitting the floor, which scales the caustic pattern size.
   * Default 0.5
   */
  waterDepth?: number;

  /**
   * Index of refraction for water (air→water).
   * Default 1.333
   */
  ior?: number;

  /**
   * Master wave amplitude — controls the turbulence intensity.
   * Default 0.08
   */
  waveAmp?: number;

  /**
   * Master wave phase velocity — controls how fast the pattern moves.
   * Default 0.4
   */
  waveSpeed?: number;

  /**
   * Jacobian power-curve exponent — higher = sharper caustic lines.
   * Default 2.5
   */
  causticSharpness?: number;

  /**
   * Output luminance scale for caustic highlights.
   * Default 3.0
   */
  causticBrightness?: number;

  /**
   * Fresnel edge darkening half-width [0, 1] — frames the pool.
   * Default 0.12
   */
  fresnelEdge?: number;

  /**
   * Background CSS colour behind the GL canvas.
   * Default '#060e1a' (very dark navy)
   */
  backgroundColor?: string;
}

export interface CausticsBackgroundHandle {
  /** Advance time + render. Call from rAF / Ticker. */
  update(dt: number): void;

  /** Change the pool floor colour. Accepts '#rrggbb'. */
  setFloorColor(hex: string): void;

  /** Change the caustic highlight colour. Accepts '#rrggbb'. */
  setCausticColor(hex: string): void;

  /** Update virtual pool depth (refraction displacement scale). */
  setDepth(d: number): void;

  /** Update wave amplitude. */
  setWaveAmp(a: number): void;

  /** Update wave phase velocity. */
  setWaveSpeed(s: number): void;

  /** Tear down GL resources and remove the canvas element. */
  stop(): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — vertex (fullscreen triangle from gl_VertexID, no attributes)
// ─────────────────────────────────────────────────────────────────────────────

const VERT = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL — caustics fragment
//
// Self-contained port of src/lib/shaders/caustics.frag with all lygia helpers
// inlined. The shader runs entirely in a single fullscreen pass — no textures,
// no simulation buffers, pure analytic math.
// ─────────────────────────────────────────────────────────────────────────────

const CAUSTICS_FRAG = `#version 300 es
precision highp float;

in  vec2  vUV;
out vec4  fragColor;

// ── Uniforms ──────────────────────────────────────────────────────────────────

uniform float     uTime;
uniform vec2      uTexelSize;       // 1 / render-target resolution

uniform float     uWaterDepth;      // e.g. 0.5
uniform float     uIOR;             // e.g. 1.333

uniform float     uWaveAmp;         // e.g. 0.08
uniform float     uWaveSpeed;       // e.g. 0.40

uniform float     uCausticSharpness;  // e.g. 2.5
uniform float     uCausticBrightness; // e.g. 3.0

uniform vec3      uFloorColor;      // e.g. vec3(0.06, 0.18, 0.32)
uniform vec3      uCausticColor;    // e.g. vec3(0.75, 0.92, 1.00)
uniform vec3      uLightDir;        // e.g. normalize(vec3(0.3, 1.0, 0.5))
uniform float     uFresnelEdge;     // e.g. 0.12

// ── Lygia math helpers (inlined) ──────────────────────────────────────────────

#define saturate(V) clamp(V, 0.0, 1.0)

// ── Wave octaves ──────────────────────────────────────────────────────────────

#define NUM_OCTAVES 4

const vec4 WAVE_DA[NUM_OCTAVES] = vec4[NUM_OCTAVES](
    vec4( 0.97, 0.24, 1.00, 1.00),
    vec4(-0.50, 0.87, 2.13, 0.55),
    vec4( 0.21,-0.98, 3.97, 0.28),
    vec4(-0.83,-0.56, 6.71, 0.14)
);

const vec2 WAVE_SP[NUM_OCTAVES] = vec2[NUM_OCTAVES](
    vec2(1.00, 0.000),
    vec2(1.32, 1.047),
    vec2(0.81, 2.094),
    vec2(1.61, 3.665)
);

float waveHeight(vec2 p) {
    float h = 0.0;
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        vec2  dir   = WAVE_DA[i].xy;
        float freq  = WAVE_DA[i].z;
        float amp   = WAVE_DA[i].w * uWaveAmp;
        float speed = WAVE_SP[i].x * uWaveSpeed;
        float phase = WAVE_SP[i].y;
        float arg = dot(dir, p) * freq - uTime * speed + phase;
        h += amp * sin(arg);
    }
    return h;
}

vec3 waveNormal(vec2 p) {
    float dhdx = 0.0;
    float dhdy = 0.0;
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        vec2  dir   = WAVE_DA[i].xy;
        float freq  = WAVE_DA[i].z;
        float amp   = WAVE_DA[i].w * uWaveAmp;
        float speed = WAVE_SP[i].x * uWaveSpeed;
        float phase = WAVE_SP[i].y;
        float arg  = dot(dir, p) * freq - uTime * speed + phase;
        float darg = amp * freq * cos(arg);
        dhdx += darg * dir.x;
        dhdy += darg * dir.y;
    }
    return vec3(-dhdx, 1.0, -dhdy);
}

vec2 refractedFloorUV(vec2 uv, vec3 N, vec3 lightDir, float depth, float ior) {
    vec3 I      = normalize(-lightDir);
    vec3 refRay = refract(I, N, ior);
    float t     = depth / max(abs(refRay.y), 1e-4);
    vec2  disp  = refRay.xz * t;
    return uv + disp;
}

float causticJacobian(vec2 uv, vec3 lightDir, float depth, float ior, float eps) {
    vec2 ex = vec2(eps, 0.0);
    vec2 ey = vec2(0.0, eps);

    vec3 Nx  = normalize(waveNormal(uv + ex));
    vec2 Px  = refractedFloorUV(uv + ex, Nx,  lightDir, depth, ior);
    vec3 Nx2 = normalize(waveNormal(uv - ex));
    vec2 Px2 = refractedFloorUV(uv - ex, Nx2, lightDir, depth, ior);

    vec3 Ny  = normalize(waveNormal(uv + ey));
    vec2 Py  = refractedFloorUV(uv + ey, Ny,  lightDir, depth, ior);
    vec3 Ny2 = normalize(waveNormal(uv - ey));
    vec2 Py2 = refractedFloorUV(uv - ey, Ny2, lightDir, depth, ior);

    vec2 dFdx = (Px - Px2) / (2.0 * eps);
    vec2 dFdy = (Py - Py2) / (2.0 * eps);

    float det = dFdx.x * dFdy.y - dFdx.y * dFdy.x;
    return max(0.0, det);
}

float singleCausticLayer(vec2 uv, vec3 lightDir, float depth, float ior) {
    float eps       = uTexelSize.x * 2.0;
    float raw       = causticJacobian(uv, lightDir, depth, ior, eps);
    float intensity = saturate(pow(max(raw - 0.5, 0.0) * 0.8, uCausticSharpness));
    return intensity;
}

float fresnelEdge(vec2 uv, float halfWidth) {
    vec2 edge = smoothstep(0.0, halfWidth, uv) *
                smoothstep(0.0, halfWidth, 1.0 - uv);
    return edge.x * edge.y;
}

void main() {
    vec3 L = normalize(uLightDir);
    vec3 N = normalize(waveNormal(vUV));
    float eta = 1.0 / max(uIOR, 1.0);

    // Multi-layer caustic accumulation
    float c0 = singleCausticLayer(vUV, L, uWaterDepth * 0.60, eta);
    float c1 = singleCausticLayer(vUV, L, uWaterDepth * 1.00, eta);
    float c2 = singleCausticLayer(vUV, L, uWaterDepth * 1.55, eta);

    float envelope = 0.5 + 0.5 * sin(
        dot(vUV - 0.5, vec2(1.3, 0.7)) * 3.1 - uTime * uWaveSpeed * 0.25
    );

    float caustic = c0 * 0.50
                  + c1 * 0.30
                  + c2 * 0.12
                  + envelope * 0.08;

    caustic = saturate(caustic * uCausticBrightness);

    // Floor lighting
    vec3  refRay       = refract(normalize(-L), N, eta);
    float floorDiffuse = max(0.0, -refRay.y);

    vec3 floorCol = uFloorColor * (0.35 + 0.65 * floorDiffuse);
    vec3 litColor = floorCol + uCausticColor * caustic;

    // Surface specular glint
    vec3  viewDir  = vec3(0.0, 0.0, 1.0);
    vec3  halfVec  = normalize(L + viewDir);
    float specular = pow(max(dot(N, halfVec), 0.0), 96.0) * 0.45;
    litColor += uCausticColor * specular;

    // Edge vignette
    float edge = fresnelEdge(vUV, uFresnelEdge);
    litColor  *= edge;

    float alpha = edge;
    fragColor = vec4(litColor * alpha, alpha);
}`;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function hexToRgb01(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16);
  return [
    ((n >> 16) & 0xff) / 255,
    ((n >>  8) & 0xff) / 255,
    (       n  & 0xff) / 255,
  ];
}

/** Set float uniform, silently skip missing locations. */
function uf(gl: WebGL2RenderingContext, prog: WebGLProgram, name: string, v: number): void {
  const loc = gl.getUniformLocation(prog, name);
  if (loc !== null) gl.uniform1f(loc, v);
}

function u2f(gl: WebGL2RenderingContext, prog: WebGLProgram, name: string, x: number, y: number): void {
  const loc = gl.getUniformLocation(prog, name);
  if (loc !== null) gl.uniform2f(loc, x, y);
}

function u3f(gl: WebGL2RenderingContext, prog: WebGLProgram, name: string, r: number, g: number, b: number): void {
  const loc = gl.getUniformLocation(prog, name);
  if (loc !== null) gl.uniform3f(loc, r, g, b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Core CausticsBackground class
// ─────────────────────────────────────────────────────────────────────────────

class CausticsBackground {
  private readonly gl: WebGL2RenderingContext;
  private readonly canvas: HTMLCanvasElement;

  private prog: WebGLProgram;
  private vao: WebGLVertexArrayObject;

  // Mutable params
  private floorColor:        [number, number, number];
  private causticColor:      [number, number, number];
  private lightDir:          [number, number, number];
  private waterDepth:        number;
  private ior:               number;
  private waveAmp:           number;
  private waveSpeed:         number;
  private causticSharpness:  number;
  private causticBrightness: number;
  private fresnelEdge:       number;

  private uTime  = 0;
  private W      = 0;
  private H      = 0;
  private ro:    ResizeObserver;
  private stopped = false;

  constructor(
    gl: WebGL2RenderingContext,
    canvas: HTMLCanvasElement,
    container: HTMLElement,
    opts: Required<CausticsBackgroundOptions>,
  ) {
    this.gl     = gl;
    this.canvas = canvas;

    this.floorColor        = hexToRgb01(opts.floorColor);
    this.causticColor      = hexToRgb01(opts.causticColor);
    this.lightDir          = opts.lightDir;
    this.waterDepth        = opts.waterDepth;
    this.ior               = opts.ior;
    this.waveAmp           = opts.waveAmp;
    this.waveSpeed         = opts.waveSpeed;
    this.causticSharpness  = opts.causticSharpness;
    this.causticBrightness = opts.causticBrightness;
    this.fresnelEdge       = opts.fresnelEdge;

    // ── Program ──────────────────────────────────────────────────────────────
    this.prog = createProgram(gl, VERT, CAUSTICS_FRAG);

    // ── Empty VAO (fullscreen triangle from gl_VertexID) ─────────────────────
    this.vao = gl.createVertexArray()!;

    // ── Resize handling ──────────────────────────────────────────────────────
    this.ro = new ResizeObserver(() => this._resize(container));
    this.ro.observe(container);
    this._resize(container);
  }

  private _resize(container: HTMLElement): void {
    if (this.stopped) return;
    const rect = container.getBoundingClientRect();
    const dpr  = Math.min(window.devicePixelRatio || 1, 2);
    const W    = Math.max(1, Math.floor(rect.width  * dpr));
    const H    = Math.max(1, Math.floor(rect.height * dpr));
    if (W === this.W && H === this.H) return;
    this.W = W; this.H = H;
    this.canvas.width  = W;
    this.canvas.height = H;
  }

  // ── update — advance time + render ─────────────────────────────────────────

  update(dt: number): void {
    if (this.stopped || !this.W || !this.H) return;
    const gl = this.gl;

    const safeDt = Math.min(dt, 0.1);
    this.uTime  += safeDt;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.W, this.H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // premultiplied alpha

    gl.useProgram(this.prog);

    // Upload uniforms
    uf( gl, this.prog, 'uTime',              this.uTime);
    u2f(gl, this.prog, 'uTexelSize',         1.0 / this.W, 1.0 / this.H);

    uf( gl, this.prog, 'uWaterDepth',        this.waterDepth);
    uf( gl, this.prog, 'uIOR',               this.ior);
    uf( gl, this.prog, 'uWaveAmp',           this.waveAmp);
    uf( gl, this.prog, 'uWaveSpeed',         this.waveSpeed);
    uf( gl, this.prog, 'uCausticSharpness',  this.causticSharpness);
    uf( gl, this.prog, 'uCausticBrightness', this.causticBrightness);
    uf( gl, this.prog, 'uFresnelEdge',       this.fresnelEdge);

    u3f(gl, this.prog, 'uFloorColor',
        this.floorColor[0], this.floorColor[1], this.floorColor[2]);
    u3f(gl, this.prog, 'uCausticColor',
        this.causticColor[0], this.causticColor[1], this.causticColor[2]);
    u3f(gl, this.prog, 'uLightDir',
        this.lightDir[0], this.lightDir[1], this.lightDir[2]);

    // Draw fullscreen triangle
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);

    gl.disable(gl.BLEND);
  }

  // ── Setters ────────────────────────────────────────────────────────────────

  setFloorColor(hex: string):   void { this.floorColor   = hexToRgb01(hex); }
  setCausticColor(hex: string): void { this.causticColor  = hexToRgb01(hex); }
  setDepth(d: number):          void { this.waterDepth    = d; }
  setWaveAmp(a: number):        void { this.waveAmp       = a; }
  setWaveSpeed(s: number):      void { this.waveSpeed     = s; }

  // ── Teardown ───────────────────────────────────────────────────────────────

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.ro.disconnect();

    const gl = this.gl;
    gl.deleteProgram(this.prog);
    gl.deleteVertexArray(this.vao);
    this.canvas.remove();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Noop handle (returned when WebGL2 is unavailable)
// ─────────────────────────────────────────────────────────────────────────────

function createNoopHandle(): CausticsBackgroundHandle {
  return {
    update:         (_dt: number)  => undefined,
    setFloorColor:  (_hex: string) => undefined,
    setCausticColor:(_hex: string) => undefined,
    setDepth:       (_d: number)   => undefined,
    setWaveAmp:     (_a: number)   => undefined,
    setWaveSpeed:   (_s: number)   => undefined,
    stop:           ()             => undefined,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Public mount function
// ─────────────────────────────────────────────────────────────────────────────

/**
 * mountCausticsBackground
 *
 * Creates a full-screen WebGL2 caustics canvas behind `container`
 * (zIndex = -1) using the analytic multi-layer Jacobian caustics technique
 * from caustics.frag (M605).
 *
 * Unlike water-background.ts which simulates a ping-pong PDE height field,
 * this renderer is purely analytic — no simulation textures needed. Four
 * sine-wave octaves synthesize the water surface, and a 5-tap Jacobian
 * stencil computes photon convergence in real time.
 *
 * @param container  Host element — needs `position: relative` or `absolute`.
 * @param options    Parameter overrides; see CausticsBackgroundOptions.
 * @returns          CausticsBackgroundHandle for per-frame update + control.
 *
 * @example
 * ```ts
 * import { mountCausticsBackground } from '@/lib/renderers/caustics-background';
 *
 * const bg = await mountCausticsBackground(document.getElementById('scene')!);
 *
 * // Per-frame (rAF / Pixi Ticker):
 * ticker.add((dt) => bg.update(dt));
 *
 * // React to theme change:
 * bg.setFloorColor('#1a0a2e');   // deep violet pool floor
 * bg.setCausticColor('#e0c0ff'); // lavender caustic highlights
 *
 * // Teardown:
 * bg.stop();
 * ```
 */
export async function mountCausticsBackground(
  container: HTMLElement,
  options: CausticsBackgroundOptions = {},
): Promise<CausticsBackgroundHandle> {

  // ── Resolve defaults ────────────────────────────────────────────────────────
  const opts: Required<CausticsBackgroundOptions> = {
    floorColor:        options.floorColor        ?? '#0f2e52',
    causticColor:      options.causticColor       ?? '#bfeaff',
    lightDir:          options.lightDir           ?? [0.3, 1.0, 0.5],
    waterDepth:        options.waterDepth         ?? 0.5,
    ior:               options.ior                ?? 1.333,
    waveAmp:           options.waveAmp            ?? 0.08,
    waveSpeed:         options.waveSpeed          ?? 0.4,
    causticSharpness:  options.causticSharpness   ?? 2.5,
    causticBrightness: options.causticBrightness  ?? 3.0,
    fresnelEdge:       options.fresnelEdge        ?? 0.12,
    backgroundColor:   options.backgroundColor    ?? '#060e1a',
  };

  // Normalise light direction
  const [lx, ly, lz] = opts.lightDir;
  const len = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1;
  opts.lightDir = [lx / len, ly / len, lz / len];

  // ── Create full-screen canvas behind container ─────────────────────────────
  const canvas = document.createElement('canvas');
  canvas.id = 'caustics-background';
  canvas.style.cssText = [
    'position: absolute',
    'inset: 0',
    'width: 100%',
    'height: 100%',
    `background-color: ${opts.backgroundColor}`,
    'z-index: -1',
    'pointer-events: none',
    'display: block',
  ].join('; ');

  if (getComputedStyle(container).position === 'static') {
    container.style.position = 'relative';
  }
  container.insertBefore(canvas, container.firstChild);

  // ── Acquire WebGL2 context ─────────────────────────────────────────────────
  const gl = canvas.getContext('webgl2', {
    alpha:                true,
    antialias:            false,
    premultipliedAlpha:   true,
    preserveDrawingBuffer: false,
  }) as WebGL2RenderingContext | null;

  if (!gl) {
    console.warn('[CausticsBackground] WebGL2 not available — background disabled');
    canvas.remove();
    return createNoopHandle();
  }

  // ── Instantiate renderer ───────────────────────────────────────────────────
  let bg: CausticsBackground;
  try {
    bg = new CausticsBackground(gl, canvas, container, opts);
  } catch (err) {
    console.error('[CausticsBackground] init error:', err);
    canvas.remove();
    return createNoopHandle();
  }

  // ── Return public handle ───────────────────────────────────────────────────
  return {
    update:          (dt)  => bg.update(dt),
    setFloorColor:   (hex) => bg.setFloorColor(hex),
    setCausticColor: (hex) => bg.setCausticColor(hex),
    setDepth:        (d)   => bg.setDepth(d),
    setWaveAmp:      (a)   => bg.setWaveAmp(a),
    setWaveSpeed:    (s)   => bg.setWaveSpeed(s),
    stop:            ()    => bg.stop(),
  };
}
