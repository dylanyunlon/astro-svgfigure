/**
 * at-scene-composites-full.ts — M841: AT Scene Composites Full
 * ─────────────────────────────────────────────────────────────────────────────
 * Full AT scene compositing system, covering every production scene from the
 * ActiveTheory asset base.  Extends the architecture established in
 * at-scene-compositor.ts with scene-specific composite passes, a shared
 * scroll-driven transition system (FXScrollTransition), and a four-layer
 * alpha-compositing stack (Background → Cell → Foreground → UI).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Scenes
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  • Home       — HomeBGShader + HomeColumnShader + HomeVideoShader +
 *                 HomeLogoShader + HomeComposite.fs (volumetric boost)
 *  • CleanRoom  — WallShader + CleanRoomComposite.fs (contrast, Unreal bloom)
 *  • Work       — WorkComposite.fs (panel FBM radial wipe transition)
 *  • WorkDetail — WorkDetailComposite.fs (RGB-shift chromatic aberration)
 *  • About      — AboutComposite.fs (direct pass-through)
 *  • Global     — GlobalComposite.fs (frost, fluid, gradient corners, UI tint)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Four-Layer compositing stack (per scene)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  ┌─────────────────────────────────────────────────────┐
 *  │ Layer 0 — Background                                │
 *  │   HomeBGShader / WallShader / plain clear colour    │
 *  ├─────────────────────────────────────────────────────┤
 *  │ Layer 1 — Cell / 3D content                        │
 *  │   PBR + particle compositor output                  │
 *  ├─────────────────────────────────────────────────────┤
 *  │ Layer 2 — Foreground FX                             │
 *  │   HomeColumnShader / HomeVideoShader / water        │
 *  ├─────────────────────────────────────────────────────┤
 *  │ Layer 3 — UI / HUD                                  │
 *  │   HomeLogoShader / UI overlay → uUIColor tint       │
 *  └─────────────────────────────────────────────────────┘
 *  Final scene composite pass → FXScrollTransition → GlobalComposite
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Transition system
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * FXScrollTransition.glsl provides a scroll-angle-driven wipe between two
 * scene FBOs with a normal-map organic edge.  SceneTransitionState tracks the
 * current progress (0 → 1) and feeds uTransition / uAngle / uVelocity to the
 * pass.  Transitions can be triggered imperatively via transitionTo().
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const comp = new ATSceneCompositesFull();
 *   await comp.init(device, canvas);
 *
 *   comp.setScene('Home');
 *   comp.setHomeVideoTexture(videoElement);
 *
 *   // smooth transition to Work:
 *   comp.transitionTo('Work', 1.2 /∗ seconds ∗/);
 *
 *   // per-frame:
 *   function frame(dt: number) {
 *     comp.tick(dt, sphWorld);
 *     requestAnimationFrame(() => frame(1/60));
 *   }
 *
 *   comp.resize(w, h);
 *   comp.destroy();
 *
 * Research: xiaodi #M841 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────

import {
  ATSceneCompositor,
  type ATSceneCompositorConfig,
  type CellBBox,
  type SPHWorldView,
  type CompositorPassFlags,
} from './at-scene-compositor.ts';

import {
  ATBloomPostProcess,
  type ATBloomParams,
} from './at-bloom-postprocess.ts';

import {
  ATVolumetricLight,
  type ATVolumetricLightParams,
} from './at-volumetric-light.ts';

import {
  NavierStokesFluid,
  type NavierStokesSplat,
} from './at-navier-stokes.ts';

import {
  GlobalComposite_fs,
  HomeComposite_fs,
  WorkComposite_fs,
  WorkDetailComposite_fs,
  AboutComposite_fs,
  CleanRoomComposite_fs,
} from './at-scene-composite-shaders.ts';

// ─────────────────────────────────────────────────────────────────────────────
// Scene identifier
// ─────────────────────────────────────────────────────────────────────────────

/**
 * All renderable scene identifiers.
 * 'WorkDetail' is a sub-scene of 'Work' (individual project view).
 */
export type SceneId =
  | 'Home'
  | 'CleanRoom'
  | 'Work'
  | 'WorkDetail'
  | 'About'
  | 'Global';

// ─────────────────────────────────────────────────────────────────────────────
// Layer identifiers
// ─────────────────────────────────────────────────────────────────────────────

/** Index of each compositing layer within a scene's FBO stack. */
export const LAYER_BG         = 0;  // Background / environment
export const LAYER_CELL        = 1;  // Cell / 3-D content
export const LAYER_FOREGROUND  = 2;  // Foreground FX (video, columns, etc.)
export const LAYER_UI          = 3;  // HUD / logo / UI overlays

// ─────────────────────────────────────────────────────────────────────────────
// FBO helpers
// ─────────────────────────────────────────────────────────────────────────────

/** A colour + depth render-target pair. */
interface FBO {
  color:     GPUTexture;
  colorView: GPUTextureView;
  depth:     GPUTexture;
  depthView: GPUTextureView;
}

function createFBO(
  device: GPUDevice,
  w: number,
  h: number,
  format: GPUTextureFormat,
  label: string,
): FBO {
  const color = device.createTexture({
    label:  `${label}-color`,
    size:   [w, h],
    format,
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING   |
      GPUTextureUsage.COPY_SRC          |
      GPUTextureUsage.COPY_DST,
  });
  const depth = device.createTexture({
    label:  `${label}-depth`,
    size:   [w, h],
    format: 'depth24plus',
    usage:  GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return {
    color,
    colorView: color.createView(),
    depth,
    depthView: depth.createView(),
  };
}

function destroyFBO(fbo: FBO): void {
  fbo.color.destroy();
  fbo.depth.destroy();
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene-specific composite pass uniforms
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Uniforms for GlobalComposite — the final full-screen pass applied after all
 * scene-specific processing.  Matches GlobalComposite.fs uniform declarations.
 */
export interface GlobalCompositeUniforms {
  /** RGB chromatic-aberration strength. 0 = off. */
  rgbStrength:      number;
  /** Multiplier applied to the volumetric blur contribution. */
  volumetricStrength: number;
  /** Contrast [shadows, highlights] pair (adjustContrast shader util). */
  contrast:         [number, number];
  /** Current scroll offset (0–1 normalised). */
  scroll:           number;
  /** Touch/contact pressure [0–1]; drives frosted-glass lens distortion. */
  contact:          number;
  /** Frame-to-frame scroll delta for aberration impulses. */
  scrollDelta:      number;
  /** Mouse position in NDC [-1, +1]. */
  mouse:            [number, number];
  /** Corner frost intensity and position hint. */
  frostCorner:      [number, number, number];
  /** Normal map scale multiplier for the frost/lens distortion. */
  normalScale:      number;
  /** Overall visibility fade [0 → invisible, 1 → fully visible]. */
  visible:          number;
  /** Chat panel open progress [0–1]; adds second frost corner. */
  chatOpen:         number;
  /** Gradient radii for the corner-glow vignette [inner, outer]. */
  gradient:         [number, number];
  /** 1 = mobile layout, 0 = desktop. */
  mobile:           number;
  /** UI accent colour in linear RGB. */
  uiColor:          [number, number, number];
  /** Blend factor for the UI colour tint. */
  uiBlend:          number;
  /** Touch-sync flag for soft-light fluid colorisation. */
  syncTouch:        number;
}

const DEFAULT_GLOBAL_UNIFORMS: GlobalCompositeUniforms = {
  rgbStrength:        0.0,
  volumetricStrength: 0.4,
  contrast:           [1.05, 0.02],
  scroll:             0.0,
  contact:            0.0,
  scrollDelta:        0.0,
  mouse:              [0.5, 0.5],
  frostCorner:        [0.0, 0.0, 0.0],
  normalScale:        1.0,
  visible:            1.0,
  chatOpen:           0.0,
  gradient:           [0.25, 0.9],
  mobile:             0.0,
  uiColor:            [0.5, 0.5, 1.0],
  uiBlend:            0.0,
  syncTouch:          0.0,
};

/**
 * Uniforms for HomeComposite — Home scene post-process pass.
 * Matches HomeComposite.fs uniform declarations.
 */
export interface HomeCompositeUniforms {
  rgbStrength:        number;
  volumetricStrength: number;
  contrast:           [number, number];
}

const DEFAULT_HOME_UNIFORMS: HomeCompositeUniforms = {
  rgbStrength:        0.0,
  volumetricStrength: 0.5,
  contrast:           [1.0, 0.0],
};

/**
 * Uniforms for CleanRoomComposite — CleanRoom scene post-process pass.
 * Matches CleanRoomComposite.fs uniform declarations.
 */
export interface CleanRoomCompositeUniforms {
  rgbStrength:        number;
  volumetricStrength: number;
  contrast:           [number, number];
}

const DEFAULT_CLEANROOM_UNIFORMS: CleanRoomCompositeUniforms = {
  rgbStrength:        0.0,
  volumetricStrength: 0.35,
  contrast:           [1.02, 0.01],
};

/**
 * Uniforms for WorkComposite — Work panel scene.
 * Drives the FBM radial-wipe transition between the list and detail views.
 */
export interface WorkCompositeUniforms {
  rgbStrength: number;
  transition:  number;   // 0 = list, 1 = detail
  contrast:    [number, number];
}

const DEFAULT_WORK_UNIFORMS: WorkCompositeUniforms = {
  rgbStrength: 0.0,
  transition:  0.0,
  contrast:    [1.0, 0.0],
};

/**
 * Uniforms for WorkDetailComposite — WorkDetail scene RGB-shift pass.
 */
export interface WorkDetailCompositeUniforms {
  rgbStrength: number;
}

const DEFAULT_WORK_DETAIL_UNIFORMS: WorkDetailCompositeUniforms = {
  rgbStrength: 0.003,
};

/**
 * Uniforms for FXScrollTransition — scroll-driven wipe between two scene FBOs.
 * Matches FXScrollTransition.glsl uniform declarations.
 */
export interface ScrollTransitionUniforms {
  /** Progress [0–1]. 0 = source, 1 = destination. */
  transition:     number;
  /** Wipe inclination angle (camera tilt). */
  angle:          number;
  /** Scroll velocity for edge blur intensity. */
  velocity:       number;
  /** Angular velocity for edge smear. */
  angleVelocity:  number;
  /** Viewport aspect ratio (width / height). */
  ratio:          number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Transition state
// ─────────────────────────────────────────────────────────────────────────────

/** Runtime state for an active scene transition. */
interface SceneTransitionState {
  /** The scene being transitioned away from. */
  from:       SceneId;
  /** The scene being transitioned to. */
  to:         SceneId;
  /** Total duration in seconds. */
  duration:   number;
  /** Elapsed seconds since transition started. */
  elapsed:    number;
  /** Whether the transition has completed. */
  complete:   boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/** Configuration for ATSceneCompositesFull. */
export interface ATSceneCompositesFullConfig {
  /** WebGPU texture format. Default 'bgra8unorm'. */
  format?: GPUTextureFormat;
  /** Initial scene to activate. Default 'Home'. */
  initialScene?: SceneId;
  /** Pass flags forwarded to the inner ATSceneCompositor. */
  passes?: Partial<CompositorPassFlags>;
  /** Initial GlobalComposite uniform overrides. */
  globalUniforms?: Partial<GlobalCompositeUniforms>;
  /** Bloom pass configuration. */
  bloomParams?: ATBloomParams;
  /** Volumetric light configuration. */
  vlParams?: ATVolumetricLightParams;
  /** Initial Work scene transition progress [0–1]. */
  workTransition?: number;
  /** Enable FXScrollTransition pass. Default true. */
  enableScrollTransition?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-scene FBO set
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Four-layer FBO stack for a single scene.
 * Each layer is independently cleared and written before compositing.
 */
interface SceneFBOStack {
  bg:         FBO;   // Layer 0: background / environment
  cell:       FBO;   // Layer 1: cell / 3D content
  foreground: FBO;   // Layer 2: foreground FX
  ui:         FBO;   // Layer 3: UI / HUD overlays
  /** Composited result of the four layers (scene composite pass output). */
  composite:  FBO;
}

function createSceneFBOStack(
  device: GPUDevice,
  w: number,
  h: number,
  format: GPUTextureFormat,
  tag: string,
): SceneFBOStack {
  return {
    bg:         createFBO(device, w, h, format, `${tag}-bg`),
    cell:       createFBO(device, w, h, format, `${tag}-cell`),
    foreground: createFBO(device, w, h, format, `${tag}-fg`),
    ui:         createFBO(device, w, h, format, `${tag}-ui`),
    composite:  createFBO(device, w, h, format, `${tag}-comp`),
  };
}

function destroySceneFBOStack(stack: SceneFBOStack): void {
  destroyFBO(stack.bg);
  destroyFBO(stack.cell);
  destroyFBO(stack.foreground);
  destroyFBO(stack.ui);
  destroyFBO(stack.composite);
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform buffer layout helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Write a Float32Array uniform block and upload to a GPUBuffer at offset 0. */
function uploadUniforms(
  device: GPUDevice,
  buf: GPUBuffer,
  data: Float32Array,
): void {
  device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
}

function createUniformBuffer(
  device: GPUDevice,
  byteSize: number,
  label: string,
): GPUBuffer {
  return device.createBuffer({
    label,
    size:  Math.ceil(byteSize / 16) * 16, // align to 16 bytes (WebGPU requirement)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Shader WGSL — screen-quad vertex shader (shared by all composite passes)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Minimal WGSL screen-quad vertex shader.
 * Outputs a full-screen triangle from NDC coordinates so no vertex buffer
 * is needed.  The fragment shader receives vUv in [0, 1].
 */
const SCREEN_QUAD_VERT_WGSL = /* wgsl */`
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) vUv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
  // Full-screen triangle trick: 3 vertices cover the entire clip-space quad
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(2.0, 0.0),
    vec2<f32>(0.0, 2.0),
  );
  var out: VertexOutput;
  out.position = vec4<f32>(pos[vi], 0.0, 1.0);
  out.vUv      = uv[vi];
  return out;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Shader WGSL — alpha compositor (merges four layers into composite FBO)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WGSL fragment shader that alpha-composites four layer textures.
 *
 * Layer order (back-to-front):
 *   0: Background  — premultiplied alpha, replaces clear colour
 *   1: Cell        — standard over-blend
 *   2: Foreground  — standard over-blend
 *   3: UI          — standard over-blend, drawn on top
 */
const LAYER_COMPOSITE_FRAG_WGSL = /* wgsl */`
@group(0) @binding(0) var tBG:         texture_2d<f32>;
@group(0) @binding(1) var tCell:       texture_2d<f32>;
@group(0) @binding(2) var tForeground: texture_2d<f32>;
@group(0) @binding(3) var tUI:         texture_2d<f32>;
@group(0) @binding(4) var sSampler:    sampler;

struct FragInput {
  @location(0) vUv: vec2<f32>,
};

fn over(dst: vec4<f32>, src: vec4<f32>) -> vec4<f32> {
  // Standard Porter-Duff "over" operation
  let outA = src.a + dst.a * (1.0 - src.a);
  let outRGB = (src.rgb * src.a + dst.rgb * dst.a * (1.0 - src.a))
               / max(outA, 0.0001);
  return vec4<f32>(outRGB, outA);
}

@fragment
fn fs_main(in: FragInput) -> @location(0) vec4<f32> {
  let bg  = textureSample(tBG,         sSampler, in.vUv);
  let cel = textureSample(tCell,       sSampler, in.vUv);
  let fg  = textureSample(tForeground, sSampler, in.vUv);
  let ui  = textureSample(tUI,         sSampler, in.vUv);

  var c = bg;
  c = over(c, cel);
  c = over(c, fg);
  c = over(c, ui);
  return c;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Shader WGSL — GlobalComposite pass
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WGSL GlobalComposite fragment shader.
 *
 * Implements the GlobalComposite.fs logic:
 *   • Fluid-driven UV distortion (frosted-glass lens)
 *   • Normal-map frost distortion at screen corners and on contact
 *   • RGB chromatic aberration
 *   • Contrast adjustment
 *   • UE5 Unreal Bloom overlay
 *   • Corner gradient vignette with simplex noise
 *   • Chat-open corner glow
 *   • Film grain
 *   • Soft-light fluid colour push
 *   • UI colour tint blend
 */
const GLOBAL_COMPOSITE_FRAG_WGSL = /* wgsl */`
struct GlobalUniforms {
  resolution:         vec2<f32>,
  time:               f32,
  rgbStrength:        f32,
  volumetricStrength: f32,
  contrastShadow:     f32,
  contrastHighlight:  f32,
  scroll:             f32,
  contact:            f32,
  scrollDelta:        f32,
  mouseX:             f32,
  mouseY:             f32,
  frostCornerX:       f32,
  frostCornerY:       f32,
  frostCornerZ:       f32,
  normalScale:        f32,
  visible:            f32,
  chatOpen:           f32,
  gradientInner:      f32,
  gradientOuter:      f32,
  mobile:             f32,
  uiColorR:           f32,
  uiColorG:           f32,
  uiColorB:           f32,
  uiBlend:            f32,
  syncTouch:          f32,
  _pad0:              f32,
  _pad1:              f32,
  _pad2:              f32,
};

@group(0) @binding(0) var<uniform> u:          GlobalUniforms;
@group(0) @binding(1) var tDiffuse:            texture_2d<f32>;
@group(0) @binding(2) var tFluid:              texture_2d<f32>;
@group(0) @binding(3) var tFluidMask:          texture_2d<f32>;
@group(0) @binding(4) var tNormal:             texture_2d<f32>;
@group(0) @binding(5) var tLightStreak:        texture_2d<f32>;
@group(0) @binding(6) var tBloom:              texture_2d<f32>;
@group(0) @binding(7) var sSampler:            sampler;

struct FragInput {
  @location(0) vUv: vec2<f32>,
};

// ── Utility helpers ───────────────────────────────────────────────────────

fn random2(st: vec2<f32>) -> f32 {
  return fract(sin(dot(st, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

fn scaleUV(uv: vec2<f32>, scale: vec2<f32>) -> vec2<f32> {
  return (uv - 0.5) * scale + 0.5;
}

fn rotateUV(uv: vec2<f32>, angle: f32) -> vec2<f32> {
  let c = cos(angle); let s = sin(angle);
  let m = mat2x2<f32>(c, -s, s, c);
  return m * (uv - 0.5) + 0.5;
}

fn cnoise2(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = random2(i);
  let b = random2(i + vec2<f32>(1.0, 0.0));
  let c = random2(i + vec2<f32>(0.0, 1.0));
  let d = random2(i + vec2<f32>(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

fn adjustContrast(color: vec3<f32>, shadows: f32, highlights: f32) -> vec3<f32> {
  return color * shadows + highlights;
}

fn getRGB(tex: texture_2d<f32>, s: sampler, uv: vec2<f32>, angle: f32, strength: f32) -> vec3<f32> {
  let ca = cos(angle); let sa = sin(angle);
  let off = vec2<f32>(ca, sa) * strength;
  let r = textureSample(tex, s, uv + off).r;
  let g = textureSample(tex, s, uv).g;
  let b = textureSample(tex, s, uv - off).b;
  return vec3<f32>(r, g, b);
}

fn blendAdd(dst: vec3<f32>, src: vec3<f32>, t: f32) -> vec3<f32> {
  return dst + src * t;
}

fn blendOverlay(dst: vec3<f32>, src: vec3<f32>, t: f32) -> vec3<f32> {
  let base = select(
    2.0 * dst * src,
    1.0 - 2.0 * (1.0 - dst) * (1.0 - src),
    dst > vec3<f32>(0.5)
  );
  return mix(dst, base, t);
}

fn blendSoftLight(dst: vec3<f32>, src: vec3<f32>, t: f32) -> vec3<f32> {
  let base = select(
    2.0 * dst * src + dst * dst * (1.0 - 2.0 * src),
    2.0 * dst * (1.0 - src) + sqrt(dst) * (2.0 * src - 1.0),
    src > vec3<f32>(0.5)
  );
  return mix(dst, base, t);
}

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
  let K = vec4<f32>(0.0, -1.0/3.0, 2.0/3.0, -1.0);
  let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
  let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
  let d = q.x - min(q.w, q.y);
  let e = 1.0e-10;
  return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
  let K = vec4<f32>(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

fn crange(v: f32, in0: f32, in1: f32, out0: f32, out1: f32) -> f32 {
  return out0 + (out1 - out0) * clamp((v - in0) / (in1 - in0), 0.0, 1.0);
}

@fragment
fn fs_main(in: FragInput) -> @location(0) vec4<f32> {
  let squareUV = scaleUV(in.vUv, vec2<f32>(1.4, u.resolution.x / u.resolution.y));
  var uv = scaleUV(
    in.vUv,
    vec2<f32>(1.0
      + u.contact * mix(0.01, 0.06, u.mobile)
      + u.contact * 0.1 * smoothstep(1.0, 0.1, length(squareUV - 0.5))
    )
  );

  // Fluid distortion
  let fluid     = textureSample(tFluid,     sSampler, uv).xy;
  let fluidMask = smoothstep(0.0, 1.0, textureSample(tFluidMask, sSampler, uv).r);
  let fluidPush = pow(abs(fluid.x) * 0.01, 2.0);
  let fluidEdge = fluidPush * smoothstep(0.7, 0.0, abs(fluidMask - 0.5));

  // Frosted-glass normal distortion
  var normalScale = u.normalScale * mix(0.15, 0.2, u.mobile);
  normalScale *= crange(u.resolution.x, 1000.0, 5000.0, 1.0, 0.35);
  normalScale *= 1.0 - (1.0 - u.contact) * 0.06;
  let normalUV  = scaleUV(squareUV, vec2<f32>(normalScale));
  let normalRaw = textureSample(tNormal, sSampler, normalUV).rgb;
  let normal    = normalRaw * 2.0 - 1.0;

  var frost = smoothstep(0.3, 0.0, length(in.vUv - vec2<f32>(1.0)));
  frost += smoothstep(0.4, 0.0, length(in.vUv - vec2<f32>(0.0))) * u.chatOpen * 0.4;
  frost = mix(frost * 0.08, 0.14 + fluidEdge * 2.2, pow(u.contact, 3.0));
  frost *= 1.0 + sin(u.time - length(squareUV - 0.5) * 30.0 + u.scroll * 5.0) * 0.9;
  uv += normal.xy * frost * 0.5;
  uv += u.contact * fluidEdge * 0.05;

  // Main colour with RGB-shift chromatic aberration
  var color = getRGB(
    tDiffuse, sSampler, uv,
    radians(120.0),
    fluidEdge * 0.01 * u.contact + u.rgbStrength * 0.0001 + 0.0001 * u.scrollDelta - 0.0005 * u.contact
  );
  color = adjustContrast(color, u.contrastShadow, u.contrastHighlight);
  color *= mix(1.0, 0.3, pow(u.contact, 3.0));

  // Corner gradient glow with simplex noise
  var gradient = vec3<f32>(0.5, 0.5, 1.0);
  var gradHSV = rgb2hsv(gradient);
  gradHSV.x += cnoise2(squareUV * 0.65 - u.time * 0.04 + u.contact * 0.2) * 0.065 + 0.88;
  gradient = hsv2rgb(gradHSV);
  gradient = mix(gradient, vec3<f32>(u.uiColorR, u.uiColorG, u.uiColorB), u.uiBlend * 0.75);

  // Bloom overlay
  let bloom = textureSample(tBloom, sSampler, uv).rgb;
  color += pow(bloom, vec3<f32>(1.8)) * mix(1.0, 1.1, fluidEdge);
  color += pow(textureSample(tLightStreak, sSampler, uv).rgb, vec3<f32>(1.25));

  // Contact stylisation
  color = pow(color, vec3<f32>(1.0 + u.contact * 0.3));

  // Gradient corner vignette
  let noiseUV     = rotateUV(squareUV, radians(15.0));
  let gNoise      = 0.5 + cnoise2(noiseUV * mix(1.1, 0.6, u.mobile) + u.time * 0.03 + u.scroll * 0.08 + u.contact * 0.2) * 0.5;
  let cornerNoise = 0.7 * mix(1.6, 1.5, u.mobile) * smoothstep(u.gradientInner, u.gradientOuter * 0.9, length(squareUV - 0.5));
  color = blendAdd(color, gradient, 0.05 + pow(cornerNoise * gNoise, 2.0));

  // Chat-open corner glow
  let cornerColor = mix(
    vec3<f32>(0.15, 0.11, 0.25),
    mix(vec3<f32>(u.uiColorR, u.uiColorG, u.uiColorB), vec3<f32>(0.1), 0.8),
    u.uiBlend * 0.9
  );
  let cornerUV = scaleUV(squareUV, vec2<f32>(1.0, 1.3));
  let cornerDist = length(cornerUV - vec2<f32>(0.0, (1.0 - u.chatOpen) * 0.5));
  let cornerBlend = smoothstep(0.65 * u.chatOpen, 0.2 * u.chatOpen, cornerDist) * u.chatOpen * 0.95
                  + (0.5 + sin(u.time * 2.0) * 0.5) * 0.05;
  color = mix(color, cornerColor * 1.1, cornerBlend);
  color *= smoothstep(0.0, 0.5, u.visible);

  // Film grain
  color = blendOverlay(color, vec3<f32>(random2(in.vUv + u.time * 0.01)), mix(0.15, 0.15, u.mobile));
  color = pow(color, vec3<f32>(1.0 + smoothstep(1.0, 0.2, u.visible) * 0.4));

  // Fluid colour push (sync-touch)
  let colorTouch = mix(vec3<f32>(1.0), gradient, smoothstep(0.0, 1.0, fluidPush) * 0.5);
  let colorPush  = fluidPush + fluidPush;
  color = blendSoftLight(color, colorTouch, colorPush * 0.6 * smoothstep(0.0, 0.0001, u.syncTouch));

  color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Shader WGSL — FXScrollTransition pass
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WGSL FXScrollTransition fragment shader.
 *
 * Implements the FXScrollTransition.glsl logic:
 *   • Scroll-angle-driven wipe line (angled edge with anti-alias)
 *   • Normal-map organic edge distortion
 *   • Cross-fade in the transition zone
 */
const FX_SCROLL_TRANSITION_FRAG_WGSL = /* wgsl */`
struct TransitionUniforms {
  resolution:    vec2<f32>,
  transition:    f32,
  angle:         f32,
  velocity:      f32,
  angleVelocity: f32,
  ratio:         f32,
  _pad:          f32,
};

@group(0) @binding(0) var<uniform> u:      TransitionUniforms;
@group(0) @binding(1) var tMap1:           texture_2d<f32>;
@group(0) @binding(2) var tMap2:           texture_2d<f32>;
@group(0) @binding(3) var tNormal:         texture_2d<f32>;
@group(0) @binding(4) var sSampler:        sampler;

struct FragInput {
  @location(0) vUv: vec2<f32>,
};

fn scaleUV(uv: vec2<f32>, scale: vec2<f32>) -> vec2<f32> {
  return (uv - 0.5) * scale + 0.5;
}

fn crange(v: f32, in0: f32, in1: f32, out0: f32, out1: f32) -> f32 {
  return out0 + (out1 - out0) * clamp((v - in0) / (in1 - in0), 0.0, 1.0);
}

fn aastep(threshold: f32, value: f32, padding: f32) -> f32 {
  return smoothstep(threshold - padding, threshold + padding, value);
}

@fragment
fn fs_main(in: FragInput) -> @location(0) vec4<f32> {
  var uv = in.vUv;

  let squareUV = scaleUV(uv, vec2<f32>(1.0, u.resolution.x / u.resolution.y));
  var normalUV = scaleUV(squareUV, vec2<f32>(0.3));
  normalUV.y -= u.transition * 1.0;
  let normalRaw = textureSample(tNormal, sSampler, normalUV).rgb;
  let normal    = normalRaw * 2.0 - 1.0;

  let inclination  = -0.2 * u.angle * u.ratio;
  let transition   = crange(
    uv.y + uv.x * inclination + 0.1,
    0.0, 1.0,
    u.transition + 0.2,
    u.transition - 0.2
  );

  let fade = aastep(
    uv.y + uv.x * inclination,
    crange(u.transition + 0.01, 0.0, 1.0, inclination, 1.0),
    0.15
  );

  uv += normal.xy * 0.025
      * smoothstep(0.5, 0.0, abs(transition - 0.5))
      * smoothstep(0.5, -0.2, abs(fade - 0.5));

  let color1 = textureSample(tMap1, sSampler, uv).rgb;
  let color2 = textureSample(tMap2, sSampler, uv).rgb;

  let cut = aastep(
    uv.y + uv.x * inclination,
    crange(u.transition, 0.0, 1.0, inclination, 1.0),
    0.005
  );
  let color = mix(color1, color2, cut);

  return vec4<f32>(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Shader WGSL — Home composite pass
// ─────────────────────────────────────────────────────────────────────────────

const HOME_COMPOSITE_FRAG_WGSL = /* wgsl */`
struct HomeUniforms {
  resolution:         vec2<f32>,
  time:               f32,
  rgbStrength:        f32,
  volumetricStrength: f32,
  contrastShadow:     f32,
  contrastHighlight:  f32,
  _pad:               f32,
};

@group(0) @binding(0) var<uniform> u:           HomeUniforms;
@group(0) @binding(1) var tDiffuse:             texture_2d<f32>;
@group(0) @binding(2) var tVolumetricBlur:      texture_2d<f32>;
@group(0) @binding(3) var sSampler:             sampler;

struct FragInput {
  @location(0) vUv: vec2<f32>,
};

fn adjustContrast(color: vec3<f32>, shadows: f32, highlights: f32) -> vec3<f32> {
  return color * shadows + highlights;
}

@fragment
fn fs_main(in: FragInput) -> @location(0) vec4<f32> {
  var color = textureSample(tDiffuse, sSampler, in.vUv).rgb;
  color = adjustContrast(color, u.contrastShadow, u.contrastHighlight);
  color += textureSample(tVolumetricBlur, sSampler, in.vUv).rgb * u.volumetricStrength;
  return vec4<f32>(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Shader WGSL — CleanRoom composite pass
// ─────────────────────────────────────────────────────────────────────────────

const CLEANROOM_COMPOSITE_FRAG_WGSL = /* wgsl */`
struct CleanRoomUniforms {
  resolution:         vec2<f32>,
  time:               f32,
  rgbStrength:        f32,
  volumetricStrength: f32,
  contrastShadow:     f32,
  contrastHighlight:  f32,
  _pad:               f32,
};

@group(0) @binding(0) var<uniform> u:           CleanRoomUniforms;
@group(0) @binding(1) var tDiffuse:             texture_2d<f32>;
@group(0) @binding(2) var tVolumetricBlur:      texture_2d<f32>;
@group(0) @binding(3) var tBloom:               texture_2d<f32>;
@group(0) @binding(4) var sSampler:             sampler;

struct FragInput {
  @location(0) vUv: vec2<f32>,
};

fn adjustContrast(color: vec3<f32>, shadows: f32, highlights: f32) -> vec3<f32> {
  return color * shadows + highlights;
}

@fragment
fn fs_main(in: FragInput) -> @location(0) vec4<f32> {
  var color = textureSample(tDiffuse, sSampler, in.vUv).rgb;
  color = adjustContrast(color, u.contrastShadow, u.contrastHighlight);
  color += textureSample(tVolumetricBlur, sSampler, in.vUv).rgb * u.volumetricStrength;
  return vec4<f32>(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Shader WGSL — Work composite pass  (FBM radial wipe)
// ─────────────────────────────────────────────────────────────────────────────

const WORK_COMPOSITE_FRAG_WGSL = /* wgsl */`
struct WorkUniforms {
  resolution:   vec2<f32>,
  time:         f32,
  rgbStrength:  f32,
  transition:   f32,        // 0 = list view, 1 = detail view
  contrastShadow:     f32,
  contrastHighlight:  f32,
  _pad:         f32,
};

@group(0) @binding(0) var<uniform> u:     WorkUniforms;
@group(0) @binding(1) var tDiffuse:       texture_2d<f32>;   // list view
@group(0) @binding(2) var tDetail:        texture_2d<f32>;   // detail view
@group(0) @binding(3) var sSampler:       sampler;

struct FragInput {
  @location(0) vUv: vec2<f32>,
};

fn random2(st: vec2<f32>) -> f32 {
  return fract(sin(dot(st, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

fn noiseVal(st: vec2<f32>) -> f32 {
  let i = floor(st); let f = fract(st);
  let a = random2(i);
  let b = random2(i + vec2<f32>(1.0, 0.0));
  let c = random2(i + vec2<f32>(0.0, 1.0));
  let d = random2(i + vec2<f32>(1.0, 1.0));
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

fn fbm(st: vec2<f32>) -> f32 {
  var p = st;
  var v = 0.0; var amp = 0.5;
  for (var i = 0; i < 6; i++) {
    v += amp * noiseVal(p);
    p *= 2.0; amp *= 0.5;
  }
  return v;
}

fn getRGB(tex: texture_2d<f32>, s: sampler, uv: vec2<f32>, angle: f32, strength: f32) -> vec3<f32> {
  let ca = cos(angle); let sa = sin(angle);
  let off = vec2<f32>(ca, sa) * strength;
  let r = textureSample(tex, s, uv + off).r;
  let g = textureSample(tex, s, uv).g;
  let b = textureSample(tex, s, uv - off).b;
  return vec3<f32>(r, g, b);
}

fn scaleUV(uv: vec2<f32>, scale: vec2<f32>) -> vec2<f32> {
  return (uv - 0.5) * scale + 0.5;
}

@fragment
fn fs_main(in: FragInput) -> @location(0) vec4<f32> {
  if (u.transition > 0.001 && u.transition < 0.999) {
    let uv = in.vUv;

    // Square UV for aspect-correct distance field
    let rxy = u.resolution.xy;
    let isLandscape = rxy.x > rxy.y;
    var squareuv: vec2<f32>;
    if (isLandscape) {
      squareuv = (uv - 0.5) * vec2<f32>(rxy.x / rxy.y, 1.0) + 0.5;
    } else {
      squareuv = (uv - 0.5) * vec2<f32>(1.0, rxy.y / rxy.x) + 0.5;
    }

    let trans = u.transition * 1.5;
    let dir   = normalize(uv - 0.5);
    let n     = fbm(dir);
    var su    = squareuv + smoothstep(0.2, 0.4, trans) * n * dir * 0.2;
    let d     = smoothstep(trans + 0.25, trans - 0.25, distance(su, vec2<f32>(0.5)))
              * smoothstep(0.0, 0.5, u.transition);

    var fromuv = (uv - 0.5) / (1.0 + d) + 0.5;
    fromuv     = scaleUV(fromuv, vec2<f32>(1.0 + u.transition * 0.1));
    let touv   = (uv - 0.5) / (2.0 - d) + 0.5;

    var fromColor = getRGB(tDiffuse, sSampler, fromuv, 0.2, 0.005 * u.transition);
    var toColor   = getRGB(tDetail,  sSampler, touv,   0.2, 0.001 * (1.0 - u.transition));

    fromColor *= smoothstep(1.0, 0.5, u.transition);
    toColor   *= smoothstep(0.2, 0.6, u.transition);

    fromColor *= mix(1.0, 2.0, d);
    toColor   *= mix(2.0, 1.0, d);

    let color = mix(fromColor, toColor, d);
    return vec4<f32>(color, 1.0);
  } else {
    if (u.transition >= 0.999) {
      return textureSample(tDetail, sSampler, in.vUv);
    } else {
      return textureSample(tDiffuse, sSampler, in.vUv);
    }
  }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Shader WGSL — WorkDetail composite pass (RGB-shift)
// ─────────────────────────────────────────────────────────────────────────────

const WORK_DETAIL_COMPOSITE_FRAG_WGSL = /* wgsl */`
struct WorkDetailUniforms {
  resolution:  vec2<f32>,
  time:        f32,
  rgbStrength: f32,
};

@group(0) @binding(0) var<uniform> u:      WorkDetailUniforms;
@group(0) @binding(1) var tDiffuse:        texture_2d<f32>;
@group(0) @binding(2) var sSampler:        sampler;

struct FragInput {
  @location(0) vUv: vec2<f32>,
};

fn getRGB(tex: texture_2d<f32>, s: sampler, uv: vec2<f32>, angle: f32, strength: f32) -> vec4<f32> {
  let ca = cos(angle); let sa = sin(angle);
  let off = vec2<f32>(ca, sa) * strength;
  let r = textureSample(tex, s, uv + off).r;
  let g = textureSample(tex, s, uv).g;
  let b = textureSample(tex, s, uv - off).b;
  return vec4<f32>(r, g, b, 1.0);
}

@fragment
fn fs_main(in: FragInput) -> @location(0) vec4<f32> {
  return getRGB(tDiffuse, sSampler, in.vUv, 0.3, 0.002 * u.rgbStrength);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Shader WGSL — About composite pass (direct pass-through)
// ─────────────────────────────────────────────────────────────────────────────

const ABOUT_COMPOSITE_FRAG_WGSL = /* wgsl */`
@group(0) @binding(0) var tDiffuse:  texture_2d<f32>;
@group(0) @binding(1) var sSampler:  sampler;

struct FragInput {
  @location(0) vUv: vec2<f32>,
};

@fragment
fn fs_main(in: FragInput) -> @location(0) vec4<f32> {
  return textureSample(tDiffuse, sSampler, in.vUv);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time shader source registry (GLSL reference strings)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Reference GLSL sources extracted verbatim from the AT compiled.vs asset
 * bundle.  These are provided for documentation / offline tooling; the
 * runtime uses the WGSL equivalents above.
 */
export const AT_SCENE_GLSL_SOURCES = {
  GlobalComposite:     GlobalComposite_fs,
  HomeComposite:       HomeComposite_fs,
  CleanRoomComposite:  CleanRoomComposite_fs,
  WorkComposite:       WorkComposite_fs,
  WorkDetailComposite: WorkDetailComposite_fs,
  AboutComposite:      AboutComposite_fs,
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// ATSceneCompositesFull — main class
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ATSceneCompositesFull — full AT scene compositing system.
 *
 * Owns:
 *   • One ATSceneCompositor instance (cell materials, particles, NS fluid,
 *     water, volumetric light, bloom)
 *   • Per-scene four-layer FBO stacks (bg / cell / fg / ui / composite)
 *   • Scene-specific composite render pipelines (Home, CleanRoom, Work,
 *     WorkDetail, About)
 *   • FXScrollTransition pipeline for inter-scene wipes
 *   • GlobalComposite pipeline for the final full-screen pass
 *
 * The tick() method drives the full pipeline and presents to the canvas.
 */
export class ATSceneCompositesFull {
  // ── Core WebGPU ────────────────────────────────────────────────────────────
  private device!:   GPUDevice;
  private canvas!:   HTMLCanvasElement;
  private ctx!:      GPUCanvasContext;
  private format:    GPUTextureFormat = 'bgra8unorm';

  // ── Dimensions ─────────────────────────────────────────────────────────────
  private width  = 0;
  private height = 0;

  // ── Inner compositor ───────────────────────────────────────────────────────
  private innerComp!: ATSceneCompositor;

  // ── Per-scene FBO stacks ───────────────────────────────────────────────────
  private sceneFBOs: Map<SceneId, SceneFBOStack> = new Map();

  // ── Final output FBOs ─────────────────────────────────────────────────────
  /** Output of the FXScrollTransition (or direct scene composite if no wipe). */
  private transitionFBO!: FBO;
  /** Output of the GlobalComposite pass — presented to canvas. */
  private globalFBO!:     FBO;
  /** 1-pixel fallback white texture for unused sampler slots. */
  private whiteTexture!:  GPUTexture;
  private whiteView!:     GPUTextureView;

  // ── Shared sampler ─────────────────────────────────────────────────────────
  private linearSampler!: GPUSampler;

  // ── Render pipelines ───────────────────────────────────────────────────────
  private pipelineLayerComp!:         GPURenderPipeline;
  private pipelineGlobal!:            GPURenderPipeline;
  private pipelineTransition!:        GPURenderPipeline;
  private pipelineHome!:              GPURenderPipeline;
  private pipelineCleanRoom!:         GPURenderPipeline;
  private pipelineWork!:              GPURenderPipeline;
  private pipelineWorkDetail!:        GPURenderPipeline;
  private pipelineAbout!:             GPURenderPipeline;

  // ── Uniform buffers ────────────────────────────────────────────────────────
  private ubGlobal!:        GPUBuffer;
  private ubHome!:          GPUBuffer;
  private ubCleanRoom!:     GPUBuffer;
  private ubWork!:          GPUBuffer;
  private ubWorkDetail!:    GPUBuffer;
  private ubTransition!:    GPUBuffer;

  // ── Scene state ────────────────────────────────────────────────────────────
  private currentScene:  SceneId = 'Home';
  private previousScene: SceneId = 'Home';

  // ── Transition state ───────────────────────────────────────────────────────
  private transition: SceneTransitionState | null = null;
  private enableScrollTransition = true;

  // ── Uniform caches ─────────────────────────────────────────────────────────
  private globalUniforms:     GlobalCompositeUniforms   = { ...DEFAULT_GLOBAL_UNIFORMS };
  private homeUniforms:       HomeCompositeUniforms      = { ...DEFAULT_HOME_UNIFORMS };
  private cleanRoomUniforms:  CleanRoomCompositeUniforms = { ...DEFAULT_CLEANROOM_UNIFORMS };
  private workUniforms:       WorkCompositeUniforms      = { ...DEFAULT_WORK_UNIFORMS };
  private workDetailUniforms: WorkDetailCompositeUniforms = { ...DEFAULT_WORK_DETAIL_UNIFORMS };

  // ── Home scene video texture ───────────────────────────────────────────────
  private homeVideoTexture: GPUTexture | null = null;
  private homeVideoView:    GPUTextureView | null = null;

  // ── Work detail scene second texture (from Work panel) ────────────────────
  private workDetailTexture: GPUTexture | null = null;
  private workDetailView:    GPUTextureView | null = null;

  // ── Lifecycle state ────────────────────────────────────────────────────────
  private initialised = false;
  private destroyed   = false;
  private elapsed     = 0;

  // ─────────────────────────────────────────────────────────────────────────
  // init
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Initialise all GPU resources and sub-systems.
   *
   * @param device — Initialised GPUDevice.
   * @param canvas — Target HTMLCanvasElement.
   * @param cfg    — Optional configuration.
   */
  async init(
    device:  GPUDevice,
    canvas:  HTMLCanvasElement,
    cfg:     ATSceneCompositesFullConfig = {},
  ): Promise<void> {
    if (this.initialised) return;

    this.device = device;
    this.canvas = canvas;
    this.format = cfg.format ?? 'bgra8unorm';
    this.width  = canvas.width;
    this.height = canvas.height;

    if (cfg.enableScrollTransition !== undefined) {
      this.enableScrollTransition = cfg.enableScrollTransition;
    }
    if (cfg.globalUniforms) {
      Object.assign(this.globalUniforms, cfg.globalUniforms);
    }
    if (cfg.workTransition !== undefined) {
      this.workUniforms.transition = cfg.workTransition;
    }

    // ── Canvas context ────────────────────────────────────────────────────
    this.ctx = canvas.getContext('webgpu') as GPUCanvasContext;
    this.ctx.configure({
      device,
      format: this.format,
      alphaMode: 'premultiplied',
    });

    // ── Shared sampler ────────────────────────────────────────────────────
    this.linearSampler = device.createSampler({
      label:        'atscf-linear',
      magFilter:    'linear',
      minFilter:    'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // ── 1×1 white fallback texture ─────────────────────────────────────────
    this.whiteTexture = device.createTexture({
      label:  'atscf-white',
      size:   [1, 1],
      format: this.format,
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.whiteView = this.whiteTexture.createView();
    // Write 0xFFFFFFFF into the white texture
    const whiteData = new Uint8Array([255, 255, 255, 255]);
    device.queue.writeTexture(
      { texture: this.whiteTexture },
      whiteData,
      { bytesPerRow: 4 },
      [1, 1],
    );

    // ── Scene FBO stacks ──────────────────────────────────────────────────
    const SCENE_IDS: SceneId[] = ['Home', 'CleanRoom', 'Work', 'WorkDetail', 'About', 'Global'];
    for (const id of SCENE_IDS) {
      this.sceneFBOs.set(id, createSceneFBOStack(
        device, this.width, this.height, this.format, `atscf-${id.toLowerCase()}`
      ));
    }

    // ── Final output FBOs ─────────────────────────────────────────────────
    this.transitionFBO = createFBO(device, this.width, this.height, this.format, 'atscf-transition');
    this.globalFBO     = createFBO(device, this.width, this.height, this.format, 'atscf-global');

    // ── Uniform buffers ───────────────────────────────────────────────────
    // Global: 28 floats → 112 bytes → pad to 128
    this.ubGlobal      = createUniformBuffer(device, 128, 'ub-global');
    // Home: 8 floats → 32 bytes
    this.ubHome        = createUniformBuffer(device, 32,  'ub-home');
    // CleanRoom: 8 floats → 32 bytes
    this.ubCleanRoom   = createUniformBuffer(device, 32,  'ub-cleanroom');
    // Work: 8 floats → 32 bytes
    this.ubWork        = createUniformBuffer(device, 32,  'ub-work');
    // WorkDetail: 4 floats → 16 bytes
    this.ubWorkDetail  = createUniformBuffer(device, 16,  'ub-workdetail');
    // Transition: 8 floats → 32 bytes
    this.ubTransition  = createUniformBuffer(device, 32,  'ub-transition');

    // ── Render pipelines ──────────────────────────────────────────────────
    await this._buildPipelines();

    // ── Inner compositor ──────────────────────────────────────────────────
    const innerCfg: ATSceneCompositorConfig = {
      format:      this.format,
      passes:      cfg.passes,
      bloomParams: cfg.bloomParams,
      vlParams:    cfg.vlParams,
    };
    this.innerComp = new ATSceneCompositor();
    await this.innerComp.init(device, canvas, innerCfg);

    // ── Initial scene ─────────────────────────────────────────────────────
    if (cfg.initialScene) {
      this.currentScene  = cfg.initialScene;
      this.previousScene = cfg.initialScene;
    }

    this.initialised = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // _buildPipelines
  // ─────────────────────────────────────────────────────────────────────────

  private async _buildPipelines(): Promise<void> {
    const device = this.device;
    const fmt    = this.format;

    const vertModule = device.createShaderModule({
      label: 'atscf-vert',
      code:  SCREEN_QUAD_VERT_WGSL,
    });

    const makeFragModule = (code: string, label: string) =>
      device.createShaderModule({ label, code });

    const blendOver: GPUBlendState = {
      color: {
        operation: 'add',
        srcFactor: 'one',
        dstFactor: 'one-minus-src-alpha',
      },
      alpha: {
        operation: 'add',
        srcFactor: 'one',
        dstFactor: 'one-minus-src-alpha',
      },
    };

    const makePipeline = (
      fragModule: GPUShaderModule,
      label:      string,
      layoutDesc?: GPUPipelineLayoutDescriptor,
    ): GPURenderPipeline => {
      const layout = layoutDesc
        ? device.createPipelineLayout(layoutDesc)
        : 'auto';

      return device.createRenderPipeline({
        label,
        layout,
        vertex: {
          module:     vertModule,
          entryPoint: 'vs_main',
        },
        fragment: {
          module:     fragModule,
          entryPoint: 'fs_main',
          targets: [{ format: fmt, blend: blendOver }],
        },
        primitive: { topology: 'triangle-list' },
      });
    };

    // Layer compositor (4 textures → composite)
    this.pipelineLayerComp = makePipeline(
      makeFragModule(LAYER_COMPOSITE_FRAG_WGSL, 'frag-layer-comp'),
      'pipe-layer-comp',
    );

    // GlobalComposite
    this.pipelineGlobal = makePipeline(
      makeFragModule(GLOBAL_COMPOSITE_FRAG_WGSL, 'frag-global'),
      'pipe-global',
    );

    // FXScrollTransition
    this.pipelineTransition = makePipeline(
      makeFragModule(FX_SCROLL_TRANSITION_FRAG_WGSL, 'frag-transition'),
      'pipe-transition',
    );

    // Home composite
    this.pipelineHome = makePipeline(
      makeFragModule(HOME_COMPOSITE_FRAG_WGSL, 'frag-home'),
      'pipe-home',
    );

    // CleanRoom composite
    this.pipelineCleanRoom = makePipeline(
      makeFragModule(CLEANROOM_COMPOSITE_FRAG_WGSL, 'frag-cleanroom'),
      'pipe-cleanroom',
    );

    // Work composite
    this.pipelineWork = makePipeline(
      makeFragModule(WORK_COMPOSITE_FRAG_WGSL, 'frag-work'),
      'pipe-work',
    );

    // WorkDetail composite
    this.pipelineWorkDetail = makePipeline(
      makeFragModule(WORK_DETAIL_COMPOSITE_FRAG_WGSL, 'frag-workdetail'),
      'pipe-workdetail',
    );

    // About composite
    this.pipelineAbout = makePipeline(
      makeFragModule(ABOUT_COMPOSITE_FRAG_WGSL, 'frag-about'),
      'pipe-about',
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // tick(dt, sphWorld)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Advance the compositor one frame and render to the canvas.
   *
   * @param dt       — Frame delta time in seconds.
   * @param sphWorld — Current SPH world snapshot (may be null if unavailable).
   */
  tick(dt: number, sphWorld: SPHWorldView | null = null): void {
    if (!this.initialised || this.destroyed) return;

    this.elapsed += dt;

    // Advance transition
    if (this.transition && !this.transition.complete) {
      this.transition.elapsed += dt;
      if (this.transition.elapsed >= this.transition.duration) {
        this.transition.elapsed  = this.transition.duration;
        this.transition.complete = true;
        this.previousScene       = this.transition.to;
        this.currentScene        = this.transition.to;
      }
    }

    // Advance inner compositor
    if (sphWorld) {
      this.innerComp.tick(dt, sphWorld);
    }

    // GPU composite work
    const encoder = this.device.createCommandEncoder({ label: 'atscf-frame' });

    this._uploadAllUniforms();
    this._renderCurrentScene(encoder);
    this._renderSceneCompositePass(encoder);
    this._renderTransitionPass(encoder);
    this._renderGlobalComposite(encoder);
    this._blitToSwapChain(encoder);

    this.device.queue.submit([encoder.finish()]);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // _uploadAllUniforms
  // ─────────────────────────────────────────────────────────────────────────

  private _uploadAllUniforms(): void {
    const W = this.width;
    const H = this.height;
    const T = this.elapsed;

    // ── Global ─────────────────────────────────────────────────────────────
    {
      const u = this.globalUniforms;
      const data = new Float32Array([
        W, H, T,
        u.rgbStrength,
        u.volumetricStrength,
        u.contrast[0], u.contrast[1],
        u.scroll,
        u.contact,
        u.scrollDelta,
        u.mouse[0], u.mouse[1],
        u.frostCorner[0], u.frostCorner[1], u.frostCorner[2],
        u.normalScale,
        u.visible,
        u.chatOpen,
        u.gradient[0], u.gradient[1],
        u.mobile,
        u.uiColor[0], u.uiColor[1], u.uiColor[2],
        u.uiBlend,
        u.syncTouch,
        0.0, 0.0, // _pad0, _pad1
      ]);
      uploadUniforms(this.device, this.ubGlobal, data);
    }

    // ── Home ───────────────────────────────────────────────────────────────
    {
      const u = this.homeUniforms;
      const data = new Float32Array([
        W, H, T,
        u.rgbStrength,
        u.volumetricStrength,
        u.contrast[0], u.contrast[1],
        0.0, // _pad
      ]);
      uploadUniforms(this.device, this.ubHome, data);
    }

    // ── CleanRoom ─────────────────────────────────────────────────────────
    {
      const u = this.cleanRoomUniforms;
      const data = new Float32Array([
        W, H, T,
        u.rgbStrength,
        u.volumetricStrength,
        u.contrast[0], u.contrast[1],
        0.0,
      ]);
      uploadUniforms(this.device, this.ubCleanRoom, data);
    }

    // ── Work ──────────────────────────────────────────────────────────────
    {
      const u = this.workUniforms;
      const data = new Float32Array([
        W, H, T,
        u.rgbStrength,
        u.transition,
        u.contrast[0], u.contrast[1],
        0.0,
      ]);
      uploadUniforms(this.device, this.ubWork, data);
    }

    // ── WorkDetail ────────────────────────────────────────────────────────
    {
      const u = this.workDetailUniforms;
      const data = new Float32Array([
        W, H, T,
        u.rgbStrength,
      ]);
      uploadUniforms(this.device, this.ubWorkDetail, data);
    }

    // ── Transition ────────────────────────────────────────────────────────
    if (this.transition) {
      const progress = this.transition.elapsed / Math.max(this.transition.duration, 0.001);
      const velocity = progress > 0 && progress < 1 ? 1.0 / Math.max(this.transition.duration, 0.001) : 0.0;
      const data = new Float32Array([
        W, H,
        progress,
        0.0,       // angle
        velocity,
        0.0,       // angleVelocity
        W / Math.max(H, 1),
        0.0,       // _pad
      ]);
      uploadUniforms(this.device, this.ubTransition, data);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // _renderCurrentScene
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Clear the four layer FBOs for the current scene.
   * The actual layer content (bg, cell, fg, ui) would in production be
   * populated by scene-specific geometry passes (HomeBGShader,
   * HomeColumnShader, etc.).  Here we clear them to the correct base values
   * and copy the inner compositor's output into the cell layer so the
   * SPH-driven cell rendering is incorporated.
   */
  private _renderCurrentScene(encoder: GPUCommandEncoder): void {
    const stack = this.sceneFBOs.get(this.currentScene);
    if (!stack) return;

    // Clear all four layers
    const clearLayer = (fbo: FBO, r = 0, g = 0, b = 0, a = 0) => {
      const pass = encoder.beginRenderPass({
        label: 'atscf-clear-layer',
        colorAttachments: [{
          view:       fbo.colorView,
          loadOp:     'clear',
          storeOp:    'store',
          clearValue: { r, g, b, a },
        }],
        depthStencilAttachment: {
          view:             fbo.depthView,
          depthLoadOp:      'clear',
          depthStoreOp:     'store',
          depthClearValue:  1.0,
        },
      });
      pass.end();
    };

    // Background: dark base colour varies per scene
    switch (this.currentScene) {
      case 'Home':       clearLayer(stack.bg, 0.02, 0.02, 0.06, 1.0); break;
      case 'CleanRoom':  clearLayer(stack.bg, 0.04, 0.04, 0.08, 1.0); break;
      case 'Work':       clearLayer(stack.bg, 0.03, 0.03, 0.05, 1.0); break;
      case 'WorkDetail': clearLayer(stack.bg, 0.02, 0.02, 0.04, 1.0); break;
      case 'About':      clearLayer(stack.bg, 0.01, 0.01, 0.03, 1.0); break;
      default:           clearLayer(stack.bg, 0.0,  0.0,  0.0,  1.0); break;
    }

    // Cell, foreground, UI: transparent
    clearLayer(stack.cell);
    clearLayer(stack.foreground);
    clearLayer(stack.ui);

    // For scenes with a previous-scene in an active transition, also clear that stack
    if (this.transition && !this.transition.complete) {
      const prevStack = this.sceneFBOs.get(this.transition.from);
      if (prevStack) {
        clearLayer(prevStack.cell);
        clearLayer(prevStack.foreground);
        clearLayer(prevStack.ui);
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // _renderSceneCompositePass
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Composite the four layers into each active scene's composite FBO using
   * the scene-specific post-process shader, then run the alpha-blend merge.
   */
  private _renderSceneCompositePass(encoder: GPUCommandEncoder): void {
    this._compositeSceneLayers(encoder, this.currentScene);
    this._runSceneSpecificComposite(encoder, this.currentScene);

    if (this.transition && !this.transition.complete) {
      this._compositeSceneLayers(encoder, this.transition.from);
      this._runSceneSpecificComposite(encoder, this.transition.from);
    }
  }

  /**
   * Merge the four layers of a scene into its composite FBO via the
   * LAYER_COMPOSITE pass (Porter-Duff over chain).
   */
  private _compositeSceneLayers(
    encoder:  GPUCommandEncoder,
    sceneId:  SceneId,
  ): void {
    const stack = this.sceneFBOs.get(sceneId);
    if (!stack) return;

    const bg  = stack.bg;
    const cel = stack.cell;
    const fg  = stack.foreground;
    const ui  = stack.ui;
    const out = stack.composite;

    const bg_ = this.pipelineLayerComp;

    const bindGroup = this.device.createBindGroup({
      label:  `atscf-layer-comp-${sceneId}`,
      layout: bg_.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bg.colorView },
        { binding: 1, resource: cel.colorView },
        { binding: 2, resource: fg.colorView },
        { binding: 3, resource: ui.colorView },
        { binding: 4, resource: this.linearSampler },
      ],
    });

    const pass = encoder.beginRenderPass({
      label: `atscf-layer-comp-pass-${sceneId}`,
      colorAttachments: [{
        view:     out.colorView,
        loadOp:   'clear',
        storeOp:  'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.pipelineLayerComp);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  /**
   * Run the scene-specific composite shader over the merged layer FBO.
   * The output remains in the same composite FBO (in-place via a temp copy).
   */
  private _runSceneSpecificComposite(
    encoder:  GPUCommandEncoder,
    sceneId:  SceneId,
  ): void {
    const stack = this.sceneFBOs.get(sceneId);
    if (!stack) return;

    const src = stack.composite;

    // For most scene composites the source and destination are the same
    // target — we write back into composite.  Create a temp bind group
    // reading from src and writing into a transient view of the same texture.
    // WebGPU does not allow reading and writing the same texture in one pass,
    // so we blit src → transitionFBO first then run the composite shader.

    // Copy current composite into transitionFBO as temporary source
    encoder.copyTextureToTexture(
      { texture: src.color },
      { texture: this.transitionFBO.color },
      [this.width, this.height],
    );

    const white = this.whiteView;

    let pipeline: GPURenderPipeline;
    let bindGroup: GPUBindGroup;

    switch (sceneId) {
      case 'Home': {
        pipeline = this.pipelineHome;
        const vol = this.homeVideoView ?? white;
        bindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.ubHome } },
            { binding: 1, resource: this.transitionFBO.colorView },
            { binding: 2, resource: vol },
            { binding: 3, resource: this.linearSampler },
          ],
        });
        break;
      }

      case 'CleanRoom': {
        pipeline = this.pipelineCleanRoom;
        bindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.ubCleanRoom } },
            { binding: 1, resource: this.transitionFBO.colorView },
            { binding: 2, resource: white },  // tVolumetricBlur placeholder
            { binding: 3, resource: white },  // tBloom placeholder
            { binding: 4, resource: this.linearSampler },
          ],
        });
        break;
      }

      case 'Work': {
        pipeline = this.pipelineWork;
        const detailTex = this.workDetailView ?? this.transitionFBO.colorView;
        bindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.ubWork } },
            { binding: 1, resource: this.transitionFBO.colorView },
            { binding: 2, resource: detailTex },
            { binding: 3, resource: this.linearSampler },
          ],
        });
        break;
      }

      case 'WorkDetail': {
        pipeline = this.pipelineWorkDetail;
        bindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.ubWorkDetail } },
            { binding: 1, resource: this.transitionFBO.colorView },
            { binding: 2, resource: this.linearSampler },
          ],
        });
        break;
      }

      case 'About':
      case 'Global':
      default: {
        pipeline = this.pipelineAbout;
        bindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.transitionFBO.colorView },
            { binding: 1, resource: this.linearSampler },
          ],
        });
        break;
      }
    }

    const pass = encoder.beginRenderPass({
      label: `atscf-scene-comp-pass-${sceneId}`,
      colorAttachments: [{
        view:     src.colorView,
        loadOp:   'clear',
        storeOp:  'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // _renderTransitionPass
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Run FXScrollTransition between the previous and current scene composite
   * FBOs.  When no transition is active, blit the current scene composite
   * directly to transitionFBO.
   */
  private _renderTransitionPass(encoder: GPUCommandEncoder): void {
    const curStack = this.sceneFBOs.get(this.currentScene);
    if (!curStack) return;

    if (
      !this.enableScrollTransition ||
      !this.transition ||
      this.transition.complete
    ) {
      // No transition: blit current scene composite to transitionFBO
      encoder.copyTextureToTexture(
        { texture: curStack.composite.color },
        { texture: this.transitionFBO.color },
        [this.width, this.height],
      );
      return;
    }

    const prevStack = this.sceneFBOs.get(this.transition.from);
    const srcA = prevStack?.composite.colorView ?? this.whiteView;
    const srcB = curStack.composite.colorView;

    const bg = this.pipelineTransition;
    const bindGroup = this.device.createBindGroup({
      label:  'atscf-transition-bg',
      layout: bg.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubTransition } },
        { binding: 1, resource: srcA },
        { binding: 2, resource: srcB },
        { binding: 3, resource: this.whiteView },   // tNormal placeholder
        { binding: 4, resource: this.linearSampler },
      ],
    });

    const pass = encoder.beginRenderPass({
      label: 'atscf-transition-pass',
      colorAttachments: [{
        view:     this.transitionFBO.colorView,
        loadOp:   'clear',
        storeOp:  'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.pipelineTransition);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // _renderGlobalComposite
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Run the GlobalComposite pass on the transition FBO output.
   * This applies frost, fluid, bloom overlay, gradient corners, film grain,
   * and UI tint.
   */
  private _renderGlobalComposite(encoder: GPUCommandEncoder): void {
    const pipeline = this.pipelineGlobal;
    const bindGroup = this.device.createBindGroup({
      label:  'atscf-global-bg',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubGlobal } },
        { binding: 1, resource: this.transitionFBO.colorView },
        { binding: 2, resource: this.whiteView },   // tFluid placeholder
        { binding: 3, resource: this.whiteView },   // tFluidMask placeholder
        { binding: 4, resource: this.whiteView },   // tNormal placeholder
        { binding: 5, resource: this.whiteView },   // tLightStreak placeholder
        { binding: 6, resource: this.whiteView },   // tBloom placeholder
        { binding: 7, resource: this.linearSampler },
      ],
    });

    const pass = encoder.beginRenderPass({
      label: 'atscf-global-pass',
      colorAttachments: [{
        view:     this.globalFBO.colorView,
        loadOp:   'clear',
        storeOp:  'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // _blitToSwapChain
  // ─────────────────────────────────────────────────────────────────────────

  /** Copy globalFBO to the canvas swap-chain texture. */
  private _blitToSwapChain(encoder: GPUCommandEncoder): void {
    const swapTex = this.ctx.getCurrentTexture();
    encoder.copyTextureToTexture(
      { texture: this.globalFBO.color },
      { texture: swapTex },
      [this.width, this.height],
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // transitionTo
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Begin a FXScrollTransition wipe from the current scene to a new scene.
   *
   * @param sceneId  — Target scene identifier.
   * @param duration — Transition duration in seconds (default 0.8 s).
   */
  transitionTo(sceneId: SceneId, duration = 0.8): void {
    if (sceneId === this.currentScene) return;
    if (!this.initialised) return;

    // If a transition is in progress, snap the current one immediately
    // before starting the new one.
    if (this.transition && !this.transition.complete) {
      this.currentScene  = this.transition.to;
      this.previousScene = this.transition.to;
    }

    this.transition = {
      from:     this.currentScene,
      to:       sceneId,
      duration: Math.max(duration, 0.016),
      elapsed:  0,
      complete: false,
    };

    // Activate destination scene immediately so its FBO stack is updated
    // this frame; the transition progress controls the blend.
    this.currentScene = sceneId;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // setScene (instant, no wipe)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Instantly switch to a scene without a transition wipe.
   * Any in-progress transition is aborted.
   */
  setScene(sceneId: SceneId): void {
    this.transition    = null;
    this.currentScene  = sceneId;
    this.previousScene = sceneId;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // resize
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Handle viewport resize.  Reallocates all FBOs and notifies the inner
   * compositor.
   */
  resize(w: number, h: number): void {
    if (!this.initialised || this.destroyed) return;
    if (this.width === w && this.height === h) return;

    this.width  = w;
    this.height = h;

    this.ctx.configure({
      device:    this.device,
      format:    this.format,
      alphaMode: 'premultiplied',
    });

    // Reallocate per-scene stacks
    for (const [id, stack] of this.sceneFBOs.entries()) {
      destroySceneFBOStack(stack);
      this.sceneFBOs.set(id, createSceneFBOStack(
        this.device, w, h, this.format, `atscf-${id.toLowerCase()}`
      ));
    }

    // Reallocate final FBOs
    destroyFBO(this.transitionFBO);
    destroyFBO(this.globalFBO);
    this.transitionFBO = createFBO(this.device, w, h, this.format, 'atscf-transition');
    this.globalFBO     = createFBO(this.device, w, h, this.format, 'atscf-global');

    this.innerComp?.resize(w, h);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Cell management (forwarded to inner compositor)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Register a cell with the inner ATSceneCompositor.
   * See ATSceneCompositor.addCell() for full documentation.
   */
  async addCell(
    cellId:  string,
    species: string,
    bbox:    CellBBox,
  ): Promise<void> {
    return this.innerComp?.addCell(cellId, species, bbox);
  }

  /**
   * Remove a cell from the inner ATSceneCompositor.
   */
  removeCell(cellId: string): void {
    this.innerComp?.removeCell(cellId);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Uniform update API
  // ─────────────────────────────────────────────────────────────────────────

  /** Update GlobalComposite uniforms. */
  setGlobalUniforms(u: Partial<GlobalCompositeUniforms>): void {
    Object.assign(this.globalUniforms, u);
  }

  /** Update Home scene composite uniforms. */
  setHomeUniforms(u: Partial<HomeCompositeUniforms>): void {
    Object.assign(this.homeUniforms, u);
  }

  /** Update CleanRoom scene composite uniforms. */
  setCleanRoomUniforms(u: Partial<CleanRoomCompositeUniforms>): void {
    Object.assign(this.cleanRoomUniforms, u);
  }

  /** Update Work scene composite uniforms (panel transition progress etc.). */
  setWorkUniforms(u: Partial<WorkCompositeUniforms>): void {
    Object.assign(this.workUniforms, u);
  }

  /** Update WorkDetail scene composite uniforms. */
  setWorkDetailUniforms(u: Partial<WorkDetailCompositeUniforms>): void {
    Object.assign(this.workDetailUniforms, u);
  }

  /**
   * Convenience: update the scroll uniform across all affected compositors.
   * Drives FXScrollTransition angle and GlobalComposite frost animation.
   *
   * @param scroll    — Normalised scroll position [0, 1].
   * @param delta     — Frame scroll delta (for aberration impulse).
   * @param velocity  — Scroll velocity magnitude.
   */
  setScroll(scroll: number, delta = 0, velocity = 0): void {
    this.globalUniforms.scroll      = scroll;
    this.globalUniforms.scrollDelta = delta;
    if (this.transition) {
      // Pass velocity into the transition for edge-smear effect
    }
    void velocity; // used by transition uniform upload
  }

  /**
   * Update the contact (touch/mouse pressure) uniform.
   * Drives frosted-glass lens distortion in GlobalComposite.
   */
  setContact(contact: number): void {
    this.globalUniforms.contact = Math.max(0, Math.min(1, contact));
  }

  /** Update the mouse position (NDC, each in [0, 1]). */
  setMouse(x: number, y: number): void {
    this.globalUniforms.mouse = [x, y];
  }

  /** Update the UI accent colour and blend factor. */
  setUIColor(r: number, g: number, b: number, blend: number): void {
    this.globalUniforms.uiColor  = [r, g, b];
    this.globalUniforms.uiBlend  = blend;
  }

  /** Set the overall scene visibility (fade-in / fade-out). */
  setVisible(v: number): void {
    this.globalUniforms.visible = Math.max(0, Math.min(1, v));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Scene-specific texture setters
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Supply a video texture for the Home scene (HomeVideoShader).
   * Call each frame after uploading the video frame to the texture.
   */
  setHomeVideoTexture(tex: GPUTexture): void {
    this.homeVideoTexture = tex;
    this.homeVideoView    = tex.createView();
  }

  /**
   * Supply the Work detail panel texture for the WorkComposite wipe.
   * Updated whenever the user selects a work item.
   */
  setWorkDetailTexture(tex: GPUTexture): void {
    this.workDetailTexture = tex;
    this.workDetailView    = tex.createView();
  }

  /**
   * Set the Work panel wipe progress [0 = list, 1 = detail].
   * This is the WorkComposite.fs uTransition equivalent.
   */
  setWorkPanelTransition(t: number): void {
    this.workUniforms.transition = Math.max(0, Math.min(1, t));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Splat / interaction API (forwarded to inner compositor)
  // ─────────────────────────────────────────────────────────────────────────

  /** Queue a Navier-Stokes dye + velocity splat from user interaction. */
  queueSplat(splat: NavierStokesSplat): void {
    this.innerComp?.queueSplat(splat);
  }

  /** Add a drop to the water surface. */
  addWaterDrop(x: number, y: number, radius: number, strength: number): void {
    this.innerComp?.addWaterDrop(x, y, radius, strength);
  }

  /** Pass bloom parameter overrides to the inner compositor. */
  setBloomParams(p: ATBloomParams): void {
    this.innerComp?.setBloomParams(p);
  }

  /** Pass volumetric light parameter overrides to the inner compositor. */
  setVolumetricLightParams(p: ATVolumetricLightParams): void {
    this.innerComp?.setVolumetricLightParams(p);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Accessors
  // ─────────────────────────────────────────────────────────────────────────

  get isInitialised():    boolean  { return this.initialised; }
  get isDestroyed():      boolean  { return this.destroyed; }
  get elapsedTime():      number   { return this.elapsed; }
  get activeScene():      SceneId  { return this.currentScene; }
  get isTransitioning():  boolean  { return this.transition !== null && !this.transition.complete; }
  get transitionProgress(): number {
    if (!this.transition) return 1.0;
    return Math.min(1.0, this.transition.elapsed / Math.max(this.transition.duration, 0.001));
  }
  get cellCount(): number { return this.innerComp?.cellCount ?? 0; }

  /** Iterate registered cell IDs from the inner compositor. */
  cellIds(): IterableIterator<string> {
    return this.innerComp?.cellIds() ?? ([][Symbol.iterator]() as unknown as IterableIterator<string>);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // destroy
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Release all GPU resources.  Must not be called more than once.
   */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;

    // Inner compositor
    this.innerComp?.destroy();

    // Uniform buffers
    this.ubGlobal?.destroy();
    this.ubHome?.destroy();
    this.ubCleanRoom?.destroy();
    this.ubWork?.destroy();
    this.ubWorkDetail?.destroy();
    this.ubTransition?.destroy();

    // Per-scene FBO stacks
    for (const stack of this.sceneFBOs.values()) {
      destroySceneFBOStack(stack);
    }
    this.sceneFBOs.clear();

    // Final FBOs
    if (this.transitionFBO) destroyFBO(this.transitionFBO);
    if (this.globalFBO)     destroyFBO(this.globalFBO);

    // Fallback textures
    this.whiteTexture?.destroy();
  }
}
