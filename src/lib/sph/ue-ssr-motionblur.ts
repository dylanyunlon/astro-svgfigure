/**
 * ue-ssr-motionblur.ts — M937: UE5 Screen Space Reflection + Motion Blur Port
 * ─────────────────────────────────────────────────────────────────────────────
 * Real WebGL1 GPU implementation of UE5 SSR + Motion Blur passes.
 * Architecture mirrors fluid-gpu-pass.ts (82gl) and at-terrain-environment.ts (254gl):
 *   init():    createProgram / createFramebuffer / createTexture / createBuffer
 *   render():  useProgram / bindFramebuffer / drawArrays
 *   dispose(): deleteProgram / deleteFramebuffer / deleteTexture / deleteBuffer
 *
 * Pass chain (per frame):
 *   Pass 1: Hi-Z Downsample  — depth → hierarchical-Z pyramid (6 mips)
 *   Pass 2: SSR Ray March    — screen-space ray march against Hi-Z depth
 *   Pass 3: Temporal Denoise — blend SSR result with history buffer
 *   Pass 4: Velocity Flatten — per-pixel velocity → 16×16 tile max velocities
 *   Pass 5: Motion Blur Apply — gather blur along tile velocity direction
 *   Pass 6: Composite        — SSR reflection + motion blur → final output
 *
 * GLSL sources extracted from compiled.vs pattern (ShaderLoader).
 * Research: xiaodi #M937 — cell-pubsub-loop
 */

// ─── constants ────────────────────────────────────────────────────────────────









const SSR_TILE_SIZE           = 8   as const;
const VELOCITY_TILE_SIZE      = 16  as const;
const CONFIG_MAX_RANGE_SIZE   = 2   as const;
const MAX_MOTION_BLUR_SAMPLES = 32  as const;
const MIN_MOTION_BLUR_VELOCITY = 0.5 as const;
const MAX_RAY_MARCH_ITERATIONS = 64  as const;
const DEFAULT_HIZ_LEVELS      = 6   as const;
const SSR_RAY_MAX_DIST        = 1000.0 as const;

// ─── fullscreen quad vertex shader (shared by all passes) ─────────────────────

const QUAD_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── Pass 1: Hi-Z downsample — max-depth mipmap reduction ─────────────────────
// Each invocation reads a 2×2 neighbourhood and writes max depth.
// First mip copies SceneDepth verbatim; subsequent mips read the previous result.

const HIZ_DOWNSAMPLE_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uDepth;        // source (prev mip or scene depth)
uniform vec2      uTexelSize;    // 1/width, 1/height of source mip
varying vec2      vUv;
void main() {
    // Sample 2×2 block and take max (conservative Hi-Z)
    float d0 = texture2D(uDepth, vUv + vec2( 0.0,          0.0         ) * uTexelSize).r;
    float d1 = texture2D(uDepth, vUv + vec2( uTexelSize.x, 0.0         )             ).r;
    float d2 = texture2D(uDepth, vUv + vec2( 0.0,          uTexelSize.y)             ).r;
    float d3 = texture2D(uDepth, vUv + uTexelSize                                     ).r;
    gl_FragColor = vec4(max(max(d0, d1), max(d2, d3)));
}
`;

// ─── Pass 2: SSR Ray March — Hi-Z accelerated screen-space reflections ─────────
// Reflects view ray about GBuffer normal, marches in clip space against Hi-Z,
// writes hit colour + confidence into SSR FBO.

const SSR_RAY_MARCH_FRAG = /* glsl */`
precision highp float;

uniform sampler2D uSceneColor;   // scene colour (for hit lookup)
uniform sampler2D uSceneDepth;   // full-res linear depth
uniform sampler2D uNormal;       // RGB world-space normal (encoded: *0.5+0.5)
uniform sampler2D uRoughness;    // R = roughness
uniform sampler2D uHiZ0;         // Hi-Z mip 0  (half res)
uniform sampler2D uHiZ1;         // Hi-Z mip 1  (quarter res)
uniform sampler2D uHiZ2;         // Hi-Z mip 2
uniform sampler2D uHiZ3;         // Hi-Z mip 3
uniform sampler2D uHiZ4;         // Hi-Z mip 4
uniform mat4      uInvProj;      // inverse projection
uniform mat4      uProj;         // projection
uniform vec2      uViewSize;     // viewport (w, h) in pixels
uniform float     uNear;
uniform float     uFar;
uniform float     uTime;
varying vec2      vUv;

// Reconstruct view-space position from depth
vec3 viewPosFromDepth(vec2 uv, float depth) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 vp  = uInvProj * ndc;
    return vp.xyz / vp.w;
}

// Project view-space position to UV
vec2 projectToUV(vec3 vPos) {
    vec4 clip = uProj * vec4(vPos, 1.0);
    vec2 ndc  = clip.xy / clip.w;
    return ndc * 0.5 + 0.5;
}

// Sample appropriate Hi-Z mip level
float sampleHiZ(vec2 uv, int level) {
    if (level <= 0) return texture2D(uSceneDepth, uv).r;
    if (level == 1) return texture2D(uHiZ0, uv).r;
    if (level == 2) return texture2D(uHiZ1, uv).r;
    if (level == 3) return texture2D(uHiZ2, uv).r;
    if (level == 4) return texture2D(uHiZ3, uv).r;
    return texture2D(uHiZ4, uv).r;
}

// Hash-based jitter for per-pixel ray offset (reduces banding)
float hash(vec2 p) {
    p = fract(p * vec2(443.8975, 397.2973));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

void main() {
    vec2 uv       = vUv;
    float depth   = texture2D(uSceneDepth, uv).r;

    // Sky / background — no SSR
    if (depth >= 0.9999) {
        gl_FragColor = vec4(0.0);
        return;
    }

    float roughness = texture2D(uRoughness, uv).r;
    // Skip diffuse-dominant surfaces
    if (roughness > 0.8) {
        gl_FragColor = vec4(0.0);
        return;
    }

    // Decode GBuffer normal (world-space stored as *0.5+0.5)
    vec3 worldNormal = texture2D(uNormal, uv).rgb * 2.0 - 1.0;
    worldNormal      = normalize(worldNormal);

    vec3 vPos   = viewPosFromDepth(uv, depth);
    // Transform world normal to view space (approximate: assume uniform scale)
    // For WebGL1 we pass normals in view-space via the GBuffer
    vec3 vNorm  = normalize(worldNormal); // treated as view-space here

    vec3 vDir   = normalize(vPos);        // view-space ray from camera
    vec3 vRefl  = reflect(vDir, vNorm);   // reflected direction

    // Jitter to reduce temporal aliasing
    float jitter = hash(uv + uTime * 0.01) * 0.5;

    vec3  rayPos  = vPos;
    float stepLen = 0.5 + jitter * 0.5;
    bool  hit     = false;
    vec2  hitUV   = vec2(0.0);
    float hitConf = 0.0;
    int   hiZMip  = 2;          // start coarse

    for (int i = 0; i < 64; i++) {
        rayPos += vRefl * stepLen;

        vec2 rUV = projectToUV(rayPos);
        if (rUV.x < 0.0 || rUV.x > 1.0 || rUV.y < 0.0 || rUV.y > 1.0) break;

        float sceneZ = sampleHiZ(rUV, hiZMip);
        // Convert Hi-Z stored depth [0,1] to view-space Z for comparison
        vec4 sNDC = vec4(rUV * 2.0 - 1.0, sceneZ * 2.0 - 1.0, 1.0);
        vec4 sVP  = uInvProj * sNDC;
        float sceneViewZ = -(sVP.z / sVP.w);  // positive in front
        float rayViewZ   = -rayPos.z;

        if (rayViewZ > sceneViewZ + 0.1) {
            // Ray is behind surface at coarse level — refine
            if (hiZMip > 0) {
                hiZMip--;
                stepLen *= 0.5;
                rayPos  -= vRefl * stepLen;  // step back half
            } else {
                // Full precision hit
                hit     = true;
                hitUV   = rUV;
                // Confidence fades with distance and glancing angle
                float dist  = length(vPos - rayPos);
                float angle = max(0.0, dot(-vDir, vNorm));
                // Fade at screen edges to hide artefacts
                float edgeX = 1.0 - abs(rUV.x - 0.5) * 2.0;
                float edgeY = 1.0 - abs(rUV.y - 0.5) * 2.0;
                float edge  = clamp(min(edgeX, edgeY) * 4.0, 0.0, 1.0);
                hitConf = angle * edge * (1.0 - roughness);
                break;
            }
        } else {
            // Miss at this level — step forward and coarsen
            stepLen *= 1.3;
            if (float(i) > 20.0 && hiZMip < 4) hiZMip++;
        }
    }

    if (hit) {
        vec3 refColor = texture2D(uSceneColor, hitUV).rgb;
        gl_FragColor  = vec4(refColor, hitConf);
    } else {
        gl_FragColor  = vec4(0.0);
    }
}
`;

// ─── Pass 3: Temporal Denoise — confidence-weighted history blend ───────────────

const TEMPORAL_DENOISE_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uCurrent;   // this frame's SSR
uniform sampler2D uHistory;   // previous frame's denoised SSR
uniform float     uBlend;     // base history blend factor (0.85..0.95)
varying vec2      vUv;
void main() {
    vec4 curr = texture2D(uCurrent, vUv);
    vec4 hist = texture2D(uHistory, vUv);
    // confidence in curr.a drives alpha: high conf → trust current
    float conf  = clamp(curr.a, 0.0, 1.0);
    float alpha = mix(uBlend, 0.2, conf);  // low conf → lean on history
    vec3  col   = mix(curr.rgb, hist.rgb, alpha);
    float a     = mix(curr.a,  hist.a,  alpha);
    gl_FragColor = vec4(col, a);
}
`;

// ─── Pass 4: Velocity Flatten — per-pixel velocity to 16×16 tile max ───────────
// We render to a VELOCITY_TILE_SIZE-downsampled target and write the max
// magnitude velocity of each tile (using a max-over-neighbourhood approach).
// Since WebGL1 has no compute, we implement this as a fragment shader that
// manually gathers a VELOCITY_TILE_SIZE×VELOCITY_TILE_SIZE window.

const VELOCITY_FLATTEN_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uVelocity;    // full-res RG velocity (screen-space pixels/frame)
uniform vec2      uPixelSize;   // 1/fullResW, 1/fullResH
uniform vec2      uTileSize;    // VELOCITY_TILE_SIZE (16.0, 16.0)
varying vec2      vUv;
void main() {
    // Tile origin in UV space
    vec2 tileOriginUV = floor(vUv * (vec2(1.0) / (uTileSize * uPixelSize)))
                        * uTileSize * uPixelSize;
    vec2 maxVel  = vec2(0.0);
    float maxLen = 0.0;
    vec2 secVel  = vec2(0.0);
    float secLen = 0.0;
    float secTheta = 1e9;

    for (int ty = 0; ty < 16; ty++) {
        for (int tx = 0; tx < 16; tx++) {
            vec2 sUV = tileOriginUV + (vec2(float(tx), float(ty)) + 0.5) * uPixelSize;
            if (sUV.x > 1.0 || sUV.y > 1.0) continue;
            vec2  v   = texture2D(uVelocity, sUV).rg;
            float len = length(v);
            if (len > maxLen) {
                // demote old max to secondary if direction differs
                if (maxLen > 0.0) {
                    float cosA = dot(normalize(v), normalize(maxVel));
                    if (cosA < 0.7) {   // > ~45° apart → keep as secondary
                        secVel = maxVel;
                        secLen = maxLen;
                    }
                }
                maxVel = v;
                maxLen = len;
            } else if (len > secLen) {
                if (maxLen > 0.0) {
                    float cosA = dot(normalize(v), normalize(maxVel));
                    if (cosA < 0.7) {
                        secVel = v;
                        secLen = len;
                    }
                }
            }
        }
    }
    // Pack: RG = primary velocity, BA = secondary velocity
    gl_FragColor = vec4(maxVel, secVel);
}
`;

// ─── Pass 5: Motion Blur Apply — gather blur along velocity direction ───────────
// Samples uSceneColor along the primary velocity direction from the tile buffer.
// Supports a secondary direction blend (CONFIG_MAX_RANGE_SIZE == 2).

const MOTION_BLUR_APPLY_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uSceneColor;  // base scene colour (pre-SSR-composite)
uniform sampler2D uVelTile;     // velocity tile buffer (RGBA: primary.xy, secondary.xy)
uniform vec2      uViewSize;    // full viewport size
uniform vec2      uTileCount;   // (tileCountX, tileCountY)
uniform float     uStrength;    // global MB strength
uniform float     uMaxSamples;  // max blur samples (float cast of MAX_MOTION_BLUR_SAMPLES)
varying vec2      vUv;

vec4 gatherBlur(vec2 baseUV, vec2 vel, float strength) {
    float velLen = length(vel) * strength;
    if (velLen < 0.5) return texture2D(uSceneColor, baseUV);

    vec2  dir       = normalize(vel) / uViewSize;
    int   sCount    = int(clamp(velLen, 1.0, uMaxSamples));
    vec4  accum     = vec4(0.0);
    float wSum      = 0.0;

    for (int i = 0; i < 32; i++) {
        if (i >= sCount) break;
        float t      = (float(i) + 0.5) / float(sCount);
        float weight = 1.0 - t;   // nearer → heavier
        vec2  sUV    = baseUV + dir * velLen * (t - 0.5);
        sUV          = clamp(sUV, vec2(0.001), vec2(0.999));
        accum       += texture2D(uSceneColor, sUV) * weight;
        wSum        += weight;
    }
    return accum / max(wSum, 0.001);
}

void main() {
    // Look up tile
    vec2 tileUV = (floor(vUv * uTileCount) + 0.5) / uTileCount;
    vec4 tile   = texture2D(uVelTile, tileUV);

    vec2 vel0   = tile.rg;   // primary velocity (pixels)
    vec2 vel1   = tile.ba;   // secondary velocity

    vec4 blur0  = gatherBlur(vUv, vel0, uStrength);
    float len1  = length(vel1);
    if (len1 > 0.5) {
        vec4  blur1  = gatherBlur(vUv, vel1, uStrength);
        float blend  = clamp(len1 / (length(vel0) + 0.001), 0.0, 0.5);
        gl_FragColor = mix(blur0, blur1, blend);
    } else {
        gl_FragColor = blur0;
    }
}
`;

// ─── Pass 6: Composite — SSR reflection + motion-blurred scene ─────────────────

const COMPOSITE_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uBlurred;     // motion-blurred scene
uniform sampler2D uSSR;         // denoised SSR reflection (RGB = colour, A = conf)
uniform sampler2D uRoughness;   // roughness to modulate SSR
uniform float     uSSRStrength; // global SSR intensity
varying vec2      vUv;
void main() {
    vec4  blur     = texture2D(uBlurred, vUv);
    vec4  ssr      = texture2D(uSSR,     vUv);
    float roughness = texture2D(uRoughness, vUv).r;
    // SSR contribution: attenuated by roughness and confidence
    float ssrAlpha = ssr.a * uSSRStrength * (1.0 - roughness);
    vec3  col      = mix(blur.rgb, ssr.rgb, ssrAlpha);
    gl_FragColor   = vec4(col, blur.a);
}
`;

// ─── Interfaces ───────────────────────────────────────────────────────────────

export interface SSRMotionBlurConfig {
  width:                number;
  height:               number;
  hiZLevels?:           number;
  raySamples?:          number;
  velocityTileSize?:    number;
  motionBlurStrength?:  number;
  enableSSR?:           boolean;
  enableMotionBlur?:    boolean;
  temporalFilterWeight?: number;
}

export interface CellVelocityData {
  cellId:        string;
  worldPos:      Float32Array;   // [x, y, z]
  prevWorldPos:  Float32Array;   // [x, y, z]
  screenVelocity: Float32Array;  // [vx, vy] screen-space pixels/frame
}

export interface SSRReflectionData {
  reflectionColor: Float32Array; // [r, g, b, a]
  confidence:      number;
  rayDistance:     number;
}

export interface MotionBlurStats {
  frameCount:           number;
  avgMotionMagnitude:   number;
  maxMotionMagnitude:   number;
  staticPixelCount:     number;
  dynamicPixelCount:    number;
  ssrHitCount:          number;
  ssrMissCount:         number;
  averageRayIterations: number;
  cpuTimeMs:            number;
  gpuTimeMs:            number;
}

// ─── Single / Double FBO helpers (internal) ───────────────────────────────────

interface SingleRT {
  fbo:   WebGLFramebuffer;
  tex:   WebGLTexture;
  w:     number;
  h:     number;
}

interface DoubleRT {
  read:     WebGLFramebuffer;
  write:    WebGLFramebuffer;
  readTex:  WebGLTexture;
  writeTex: WebGLTexture;
  w:        number;
  h:        number;
}

// ─── Main class ───────────────────────────────────────────────────────────────

export class UESSRMotionBlur {
  private readonly gl: WebGLRenderingContext;
  private readonly cfg: Required<SSRMotionBlurConfig>;

  // ── Programs ─────────────────────────────────────────────────────────────────
  private hiZProg!:      WebGLProgram;   // Pass 1: Hi-Z downsample
  private ssrProg!:      WebGLProgram;   // Pass 2: SSR ray march
  private denoiseProg!:  WebGLProgram;   // Pass 3: temporal denoise
  private flattenProg!:  WebGLProgram;   // Pass 4: velocity flatten
  private mbProg!:       WebGLProgram;   // Pass 5: motion blur apply
  private compositeProg!: WebGLProgram;  // Pass 6: composite

  // ── Hi-Z pyramid (6 half-res FBOs, each mip is half the previous) ────────────
  private hiZMips!:      SingleRT[];     // [mip0=half, mip1=quarter, …, mip5]

  // ── SSR buffers ──────────────────────────────────────────────────────────────
  private ssrRT!:        SingleRT;       // raw SSR hit colour + confidence
  private ssrHistory!:   DoubleRT;       // temporal history (ping-pong)

  // ── Motion blur buffers ───────────────────────────────────────────────────────
  private velTileRT!:    SingleRT;       // velocity tile buffer (tileCountX × tileCountY)
  private mbRT!:         SingleRT;       // motion-blur applied scene

  // ── Composite output ─────────────────────────────────────────────────────────
  private compositeRT!:  SingleRT;       // final SSR + MB composite

  // ── Quad geometry ────────────────────────────────────────────────────────────
  private quadBuf!:      WebGLBuffer;    // 2 triangles ([-1,-1 … 1,1])

  // ── External inputs (bound each frame) ───────────────────────────────────────
  private sceneColorTex:   WebGLTexture | null = null;
  private sceneDepthTex:   WebGLTexture | null = null;
  private normalTex:       WebGLTexture | null = null;
  private roughnessTex:    WebGLTexture | null = null;
  private velocityTex:     WebGLTexture | null = null;

  // ── CPU-side matrices ─────────────────────────────────────────────────────────
  private projMatrix:    Float32Array = new Float32Array(16);
  private invProjMatrix: Float32Array = new Float32Array(16);

  // ── Per-cell velocity tracking ────────────────────────────────────────────────
  private readonly cellVelocityMap: Map<string, CellVelocityData> = new Map();

  private time = 0.0;

  // ─── Constructor / factory ────────────────────────────────────────────────────

  constructor(gl: WebGLRenderingContext, cfg: SSRMotionBlurConfig) {
    this.gl  = gl;
    this.cfg = {
      hiZLevels:           cfg.hiZLevels          ?? DEFAULT_HIZ_LEVELS,
      raySamples:          cfg.raySamples          ?? 12,
      velocityTileSize:    cfg.velocityTileSize    ?? VELOCITY_TILE_SIZE,
      motionBlurStrength:  cfg.motionBlurStrength  ?? 1.0,
      enableSSR:           cfg.enableSSR           ?? true,
      enableMotionBlur:    cfg.enableMotionBlur     ?? true,
      temporalFilterWeight: cfg.temporalFilterWeight ?? 0.88,
      width:  cfg.width,
      height: cfg.height,
    };
    this._init();
  }

  // ─── Public API ───────────────────────────────────────────────────────────────

  /**
   * Supply external scene textures produced upstream each frame.
   * Must be called before render().
   */
  updateSceneData(
    sceneColor: WebGLTexture,
    sceneDepth: WebGLTexture,
    normal:     WebGLTexture,
    roughness:  WebGLTexture,
    velocity:   WebGLTexture,
  ): void {
    this.sceneColorTex = sceneColor;
    this.sceneDepthTex = sceneDepth;
    this.normalTex     = normal;
    this.roughnessTex  = roughness;
    this.velocityTex   = velocity;
  }

  /** Supply view/projection matrices (column-major Float32Array[16]). */
  setProjectionMatrices(proj: Float32Array, invProj: Float32Array): void {
    this.projMatrix.set(proj);
    this.invProjMatrix.set(invProj);
  }

  /** Register a cell's screen-space velocity for the current frame. */
  registerCellVelocity(data: CellVelocityData): void {
    this.cellVelocityMap.set(data.cellId, data);
  }

  /** Clear cell velocities — call after render, before next frame's registrations. */
  clearCellVelocities(): void {
    this.cellVelocityMap.clear();
  }

  /**
   * Execute full SSR + Motion Blur pipeline for one frame.
   * @param dt delta time in seconds
   */
  render(dt: number): void {
    this.time += dt;
    const { enableSSR, enableMotionBlur } = this.cfg;

    if (enableSSR) {
      this._passHiZBuild();
      this._passSSRRayMarch();
      this._passTemporalDenoise();
    }

    if (enableMotionBlur) {
      this._passVelocityFlatten();
      this._passMotionBlurApply();
    }

    this._passComposite();
  }

  /** Alias to match cell-pubsub-loop tick interface. */
  tick(dt: number): void { this.render(dt); }

  /** Returns the composite output texture (SSR + MB merged with scene). */
  get resultTexture(): WebGLTexture { return this.compositeRT.tex; }

  /** Returns the raw denoised SSR reflection texture. */
  get ssrTexture(): WebGLTexture { return this.ssrHistory.readTex; }

  /** Returns the motion-blur-applied scene texture. */
  get mbTexture(): WebGLTexture { return this.mbRT.tex; }

  /** Returns normalised [0,1] motion blur intensity for a registered cell. */
  getCellMotionBlurIntensity(cellId: string): number {
    const d = this.cellVelocityMap.get(cellId);
    if (!d) return 0;
    const mag = Math.hypot(d.screenVelocity[0], d.screenVelocity[1]);
    return Math.min(1.0, mag / (MAX_MOTION_BLUR_SAMPLES * this.cfg.motionBlurStrength));
  }

  /** Release all GPU resources. */
  dispose(): void {
    const gl = this.gl;

    // Programs
    gl.deleteProgram(this.hiZProg);
    gl.deleteProgram(this.ssrProg);
    gl.deleteProgram(this.denoiseProg);
    gl.deleteProgram(this.flattenProg);
    gl.deleteProgram(this.mbProg);
    gl.deleteProgram(this.compositeProg);

    // Hi-Z mips
    for (const mip of this.hiZMips) {
      gl.deleteFramebuffer(mip.fbo);
      gl.deleteTexture(mip.tex);
    }

    // SSR
    gl.deleteFramebuffer(this.ssrRT.fbo);
    gl.deleteTexture(this.ssrRT.tex);
    gl.deleteFramebuffer(this.ssrHistory.read);
    gl.deleteFramebuffer(this.ssrHistory.write);
    gl.deleteTexture(this.ssrHistory.readTex);
    gl.deleteTexture(this.ssrHistory.writeTex);

    // Motion blur
    gl.deleteFramebuffer(this.velTileRT.fbo);
    gl.deleteTexture(this.velTileRT.tex);
    gl.deleteFramebuffer(this.mbRT.fbo);
    gl.deleteTexture(this.mbRT.tex);

    // Composite
    gl.deleteFramebuffer(this.compositeRT.fbo);
    gl.deleteTexture(this.compositeRT.tex);

    // Quad
    gl.deleteBuffer(this.quadBuf);
  }

  // ─── Private: init ────────────────────────────────────────────────────────────

  private _init(): void {
    // 1. Compile all programs
    this.hiZProg       = this._compile(QUAD_VERT, HIZ_DOWNSAMPLE_FRAG,   'hiZ');
    this.ssrProg       = this._compile(QUAD_VERT, SSR_RAY_MARCH_FRAG,    'ssr');
    this.denoiseProg   = this._compile(QUAD_VERT, TEMPORAL_DENOISE_FRAG, 'denoise');
    this.flattenProg   = this._compile(QUAD_VERT, VELOCITY_FLATTEN_FRAG, 'flatten');
    this.mbProg        = this._compile(QUAD_VERT, MOTION_BLUR_APPLY_FRAG,'mb');
    this.compositeProg = this._compile(QUAD_VERT, COMPOSITE_FRAG,        'composite');

    // 2. Create fullscreen quad
    this._buildQuad();

    // 3. Create FBOs
    this._createHiZMips();
    this._createSSRBuffers();
    this._createMotionBlurBuffers();
    this._createCompositeBuffer();
  }

  // ─── Private: program compile ─────────────────────────────────────────────────

  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[UESSRMotionBlur] vert compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[UESSRMotionBlur] frag compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[UESSRMotionBlur] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  // ─── Private: geometry ────────────────────────────────────────────────────────

  private _buildQuad(): void {
    const gl = this.gl;
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1,  -1, 1,
       1, -1,  1,  1,  -1, 1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  // ─── Private: FBO creation ────────────────────────────────────────────────────

  /** Six half-resolution Hi-Z mip levels. */
  private _createHiZMips(): void {
    const gl     = this.gl;
    const levels = this.cfg.hiZLevels;
    this.hiZMips = [];

    let w = Math.max(1, Math.floor(this.cfg.width  / 2));
    let h = Math.max(1, Math.floor(this.cfg.height / 2));

    for (let i = 0; i < levels; i++) {
      const tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      const fbo = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

      this.hiZMips.push({ fbo, tex, w, h });

      w = Math.max(1, Math.floor(w / 2));
      h = Math.max(1, Math.floor(h / 2));
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** SSR raw hit buffer + temporal history ping-pong. */
  private _createSSRBuffers(): void {
    const { width: w, height: h } = this.cfg;
    this.ssrRT      = this._createSingleRT(w, h);
    this.ssrHistory = this._createDoubleRT(w, h);
  }

  /** Velocity tile buffer + motion-blurred scene buffer. */
  private _createMotionBlurBuffers(): void {
    const { width, height, velocityTileSize } = this.cfg;
    const tw = Math.ceil(width  / velocityTileSize);
    const th = Math.ceil(height / velocityTileSize);
    this.velTileRT = this._createSingleRT(tw, th);
    this.mbRT      = this._createSingleRT(width, height);
  }

  /** Final composite output. */
  private _createCompositeBuffer(): void {
    const { width, height } = this.cfg;
    this.compositeRT = this._createSingleRT(width, height);
  }

  private _createSingleRT(w: number, h: number): SingleRT {
    const gl  = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return { fbo, tex, w, h };
  }

  private _createDoubleRT(w: number, h: number): DoubleRT {
    const a = this._createSingleRT(w, h);
    const b = this._createSingleRT(w, h);
    return { read: a.fbo, write: b.fbo, readTex: a.tex, writeTex: b.tex, w, h };
  }

  // ─── Private: render passes ───────────────────────────────────────────────────

  /**
   * Pass 1 — Hi-Z Pyramid Build
   * Iteratively downsample SceneDepth into 6 half-res mips (max depth per 2×2).
   * Each mip serves as the Hi-Z level for ray march coarse intersection tests.
   */
  private _passHiZBuild(): void {
    if (!this.sceneDepthTex) return;
    const gl = this.gl;

    gl.useProgram(this.hiZProg);

    let srcTex: WebGLTexture = this.sceneDepthTex;
    let srcW = this.cfg.width;
    let srcH = this.cfg.height;

    for (let i = 0; i < this.hiZMips.length; i++) {
      const mip = this.hiZMips[i];

      gl.bindFramebuffer(gl.FRAMEBUFFER, mip.fbo);
      gl.viewport(0, 0, mip.w, mip.h);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.uniform1i(gl.getUniformLocation(this.hiZProg, 'uDepth'), 0);
      gl.uniform2f(gl.getUniformLocation(this.hiZProg, 'uTexelSize'),
                   1.0 / srcW, 1.0 / srcH);

      this._drawQuad(this.hiZProg);

      srcTex = mip.tex;
      srcW   = mip.w;
      srcH   = mip.h;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Pass 2 — SSR Ray March
   * For every pixel: reflect view ray about GBuffer normal, march in clip space
   * using Hi-Z pyramid for early exit. Writes RGB hit colour + alpha confidence.
   */
  private _passSSRRayMarch(): void {
    if (!this.sceneColorTex || !this.sceneDepthTex || !this.normalTex || !this.roughnessTex) return;
    const gl = this.gl;
    const { width, height } = this.cfg;

    gl.useProgram(this.ssrProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.ssrRT.fbo);
    gl.viewport(0, 0, width, height);

    // Bind all input textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sceneColorTex);
    gl.uniform1i(gl.getUniformLocation(this.ssrProg, 'uSceneColor'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.sceneDepthTex);
    gl.uniform1i(gl.getUniformLocation(this.ssrProg, 'uSceneDepth'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.normalTex);
    gl.uniform1i(gl.getUniformLocation(this.ssrProg, 'uNormal'), 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.roughnessTex);
    gl.uniform1i(gl.getUniformLocation(this.ssrProg, 'uRoughness'), 3);

    // Hi-Z mips (up to 5)
    const hiZNames = ['uHiZ0','uHiZ1','uHiZ2','uHiZ3','uHiZ4'];
    for (let i = 0; i < Math.min(5, this.hiZMips.length); i++) {
      gl.activeTexture(gl.TEXTURE4 + i);
      gl.bindTexture(gl.TEXTURE_2D, this.hiZMips[i].tex);
      gl.uniform1i(gl.getUniformLocation(this.ssrProg, hiZNames[i]), 4 + i);
    }

    // Matrices and uniforms
    gl.uniformMatrix4fv(gl.getUniformLocation(this.ssrProg, 'uInvProj'), false, this.invProjMatrix);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.ssrProg, 'uProj'),    false, this.projMatrix);
    gl.uniform2f(gl.getUniformLocation(this.ssrProg, 'uViewSize'), width, height);
    gl.uniform1f(gl.getUniformLocation(this.ssrProg, 'uNear'),  0.1);
    gl.uniform1f(gl.getUniformLocation(this.ssrProg, 'uFar'),   SSR_RAY_MAX_DIST);
    gl.uniform1f(gl.getUniformLocation(this.ssrProg, 'uTime'),  this.time);

    this._drawQuad(this.ssrProg);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Pass 3 — Temporal Denoise
   * Blends this frame's raw SSR with the temporal history using confidence weights.
   * High-confidence pixels trust the current frame; low-confidence pixels lean on
   * the history to suppress noise. Output goes to ssrHistory.write, then swaps.
   */
  private _passTemporalDenoise(): void {
    const gl = this.gl;
    const { width, height } = this.cfg;

    gl.useProgram(this.denoiseProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.ssrHistory.write);
    gl.viewport(0, 0, width, height);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.ssrRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.denoiseProg, 'uCurrent'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.ssrHistory.readTex);
    gl.uniform1i(gl.getUniformLocation(this.denoiseProg, 'uHistory'), 1);

    gl.uniform1f(gl.getUniformLocation(this.denoiseProg, 'uBlend'), this.cfg.temporalFilterWeight);

    this._drawQuad(this.denoiseProg);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Swap history ping-pong
    [this.ssrHistory.read,    this.ssrHistory.write   ] =
    [this.ssrHistory.write,   this.ssrHistory.read    ];
    [this.ssrHistory.readTex, this.ssrHistory.writeTex] =
    [this.ssrHistory.writeTex, this.ssrHistory.readTex];
  }

  /**
   * Pass 4 — Velocity Flatten
   * Gathers per-pixel velocity from the full-res velocity buffer and reduces each
   * VELOCITY_TILE_SIZE × VELOCITY_TILE_SIZE block down to its two dominant velocities.
   * Result is written to a small tile-resolution RT (velTileRT).
   */
  private _passVelocityFlatten(): void {
    if (!this.velocityTex) return;
    const gl = this.gl;
    const { width, height, velocityTileSize } = this.cfg;

    gl.useProgram(this.flattenProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velTileRT.fbo);
    gl.viewport(0, 0, this.velTileRT.w, this.velTileRT.h);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocityTex);
    gl.uniform1i(gl.getUniformLocation(this.flattenProg, 'uVelocity'), 0);

    gl.uniform2f(gl.getUniformLocation(this.flattenProg, 'uPixelSize'),
                 1.0 / width, 1.0 / height);
    gl.uniform2f(gl.getUniformLocation(this.flattenProg, 'uTileSize'),
                 velocityTileSize, velocityTileSize);

    this._drawQuad(this.flattenProg);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Pass 5 — Motion Blur Apply
   * For every pixel, looks up its tile's dominant velocity, then gathers samples
   * along that direction (and optionally a secondary direction) to produce
   * temporal-smear motion blur. Result goes to mbRT.
   */
  private _passMotionBlurApply(): void {
    if (!this.sceneColorTex) return;
    const gl = this.gl;
    const { width, height, velocityTileSize, motionBlurStrength } = this.cfg;

    gl.useProgram(this.mbProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.mbRT.fbo);
    gl.viewport(0, 0, width, height);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sceneColorTex);
    gl.uniform1i(gl.getUniformLocation(this.mbProg, 'uSceneColor'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.velTileRT.tex);
    gl.uniform1i(gl.getUniformLocation(this.mbProg, 'uVelTile'), 1);

    gl.uniform2f(gl.getUniformLocation(this.mbProg, 'uViewSize'), width, height);
    gl.uniform2f(gl.getUniformLocation(this.mbProg, 'uTileCount'),
                 this.velTileRT.w, this.velTileRT.h);
    gl.uniform1f(gl.getUniformLocation(this.mbProg, 'uStrength'),    motionBlurStrength);
    gl.uniform1f(gl.getUniformLocation(this.mbProg, 'uMaxSamples'),  MAX_MOTION_BLUR_SAMPLES);

    this._drawQuad(this.mbProg);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Pass 6 — Composite
   * Merges the motion-blurred scene with the denoised SSR reflections, modulated
   * by surface roughness. Final result written to compositeRT.
   */
  private _passComposite(): void {
    const gl = this.gl;
    const { width, height, enableSSR, enableMotionBlur } = this.cfg;

    gl.useProgram(this.compositeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.compositeRT.fbo);
    gl.viewport(0, 0, width, height);

    // Blurred or straight scene colour
    const blurSource = (enableMotionBlur && this.mbRT)
      ? this.mbRT.tex
      : (this.sceneColorTex ?? this.mbRT.tex);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, blurSource);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uBlurred'), 0);

    // SSR reflection (or a clear 1×1 if SSR disabled)
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.ssrHistory.readTex);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uSSR'), 1);

    // Roughness for reflection attenuation
    const roughSrc = this.roughnessTex ?? this.ssrRT.tex;
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, roughSrc);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'uRoughness'), 2);

    gl.uniform1f(gl.getUniformLocation(this.compositeProg, 'uSSRStrength'),
                 enableSSR ? 1.0 : 0.0);

    this._drawQuad(this.compositeProg);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ─── Private: draw quad helper ────────────────────────────────────────────────

  private _drawQuad(prog: WebGLProgram): void {
    const gl    = this.gl;
    const posLoc = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}

// ─── Standalone factory ───────────────────────────────────────────────────────

export function createUESSRMotionBlur(
  gl: WebGLRenderingContext,
  width: number,
  height: number,
  options?: Partial<SSRMotionBlurConfig>,
): UESSRMotionBlur {
  return new UESSRMotionBlur(gl, { width, height, ...options });
}

// ─── Velocity utility ─────────────────────────────────────────────────────────

/**
 * Compute a cell's screen-space velocity (pixels/frame) given world positions
 * and the current/previous view-projection matrices.
 */
export function computeCellScreenVelocity(
  cellId:        string,
  currWorldPos:  [number, number, number],
  prevWorldPos:  [number, number, number],
  currViewProj:  Float32Array,  // 4×4 column-major
  prevViewProj:  Float32Array,
  vpW:           number,
  vpH:           number,
): CellVelocityData {
  const mvp4 = (m: Float32Array, v: [number,number,number]): [number,number,number,number] => {
    const [x,y,z] = v;
    return [
      m[0]*x + m[4]*y + m[ 8]*z + m[12],
      m[1]*x + m[5]*y + m[ 9]*z + m[13],
      m[2]*x + m[6]*y + m[10]*z + m[14],
      m[3]*x + m[7]*y + m[11]*z + m[15],
    ];
  };
  const toScreen = (clip: [number,number,number,number]): [number,number] => {
    const ndcX = clip[0] / clip[3];
    const ndcY = clip[1] / clip[3];
    return [(ndcX + 1) * 0.5 * vpW, (1 - ndcY) * 0.5 * vpH];
  };

  const [cx, cy] = toScreen(mvp4(currViewProj, currWorldPos));
  const [px, py] = toScreen(mvp4(prevViewProj, prevWorldPos));

  return {
    cellId,
    worldPos:       new Float32Array(currWorldPos),
    prevWorldPos:   new Float32Array(prevWorldPos),
    screenVelocity: new Float32Array([cx - px, cy - py]),
  };
}

// ─── Compositor helper ────────────────────────────────────────────────────────

export class SSRMotionBlurCompositor {
  private readonly ssrMb: UESSRMotionBlur;
  private blendMode:   'additive' | 'blend' | 'overlay' = 'blend';
  private ssrIntensity = 1.0;
  private mbIntensity  = 1.0;

  constructor(ssrMb: UESSRMotionBlur) { this.ssrMb = ssrMb; }

  setBlendMode(m: 'additive' | 'blend' | 'overlay'): void { this.blendMode = m; }
  setSSRIntensity(v: number):          void { this.ssrIntensity = Math.min(1, Math.max(0, v)); }
  setMotionBlurIntensity(v: number):   void { this.mbIntensity  = Math.min(1, Math.max(0, v)); }

  /** Returns the live composite output texture from the wrapped UESSRMotionBlur. */
  getOutputTexture(): WebGLTexture { return this.ssrMb.resultTexture; }
}

// ─── Adaptive quality ─────────────────────────────────────────────────────────

export class AdaptiveMotionBlurQuality {
  private targetFrameTime = 16.66;
  private currentQuality: 'low' | 'medium' | 'high' | 'ultra' = 'high';
  private raySamplesBudget = 12;
  private readonly history: number[] = [];
  private readonly maxHistory = 30;

  recordFrameTime(ms: number): void {
    this.history.push(ms);
    if (this.history.length > this.maxHistory) this.history.shift();
  }
  getAverageFrameTime(): number {
    if (!this.history.length) return 0;
    return this.history.reduce((a, b) => a + b, 0) / this.history.length;
  }
  shouldAdjustQuality(): boolean {
    return Math.abs(this.getAverageFrameTime() - this.targetFrameTime) > 3.0;
  }
  adjustQualityBasedOnPerformance(): void {
    const avg = this.getAverageFrameTime();
    if (avg > this.targetFrameTime * 1.2) {
      switch (this.currentQuality) {
        case 'ultra':  this.currentQuality = 'high';   this.raySamplesBudget = 12; break;
        case 'high':   this.currentQuality = 'medium'; this.raySamplesBudget =  8; break;
        case 'medium': this.currentQuality = 'low';    this.raySamplesBudget =  4; break;
      }
    } else if (avg < this.targetFrameTime * 0.8) {
      switch (this.currentQuality) {
        case 'low':    this.currentQuality = 'medium'; this.raySamplesBudget =  8; break;
        case 'medium': this.currentQuality = 'high';   this.raySamplesBudget = 12; break;
        case 'high':   this.currentQuality = 'ultra';  this.raySamplesBudget = 16; break;
      }
    }
  }
  getCurrentQuality(): string   { return this.currentQuality; }
  getRaySampleBudget(): number   { return this.raySamplesBudget; }
}

// ─── Stats collector ──────────────────────────────────────────────────────────

export class MotionBlurStatsCollector {
  private stats: MotionBlurStats = {
    frameCount: 0, avgMotionMagnitude: 0, maxMotionMagnitude: 0,
    staticPixelCount: 0, dynamicPixelCount: 0,
    ssrHitCount: 0, ssrMissCount: 0, averageRayIterations: 0,
    cpuTimeMs: 0, gpuTimeMs: 0,
  };
  private magnitudes: number[] = [];
  private iterations: number[] = [];

  recordMotionMagnitude(mag: number): void {
    this.magnitudes.push(mag);
    this.stats.maxMotionMagnitude = Math.max(this.stats.maxMotionMagnitude, mag);
  }
  recordRayIteration(count: number): void { this.iterations.push(count); }
  recordSSRHit():  void { this.stats.ssrHitCount++;  }
  recordSSRMiss(): void { this.stats.ssrMissCount++; }
  recordPixelType(isStatic: boolean): void {
    isStatic ? this.stats.staticPixelCount++ : this.stats.dynamicPixelCount++;
  }
  recordTiming(cpu: number, gpu: number): void {
    this.stats.cpuTimeMs = cpu;
    this.stats.gpuTimeMs = gpu;
  }
  finalize(): MotionBlurStats {
    if (this.magnitudes.length)
      this.stats.avgMotionMagnitude = this.magnitudes.reduce((a,b)=>a+b,0) / this.magnitudes.length;
    if (this.iterations.length)
      this.stats.averageRayIterations = this.iterations.reduce((a,b)=>a+b,0) / this.iterations.length;
    this.stats.frameCount++;
    return { ...this.stats };
  }
  reset(): void { this.magnitudes = []; this.iterations = []; }
}

// ─── Re-exports for downstream consumers ─────────────────────────────────────

export {
  SSR_TILE_SIZE,
  VELOCITY_TILE_SIZE,
  CONFIG_MAX_RANGE_SIZE,
  MAX_MOTION_BLUR_SAMPLES,
  MIN_MOTION_BLUR_VELOCITY,
  MAX_RAY_MARCH_ITERATIONS,
  DEFAULT_HIZ_LEVELS,
  SSR_RAY_MAX_DIST,
};
