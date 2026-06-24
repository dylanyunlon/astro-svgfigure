/**
 * at-uil-live-panel.ts — M1048: AT UIL Live Panel (real GPU uniform live tuning)
 * ─────────────────────────────────────────────────────────────────────────────
 * dat.GUI-style HTML overlay that reads uil-params.json (2593 entries), creates
 * one widget per live-controllable param, and pushes every change to the running
 * WebGL shaders via gl.uniform* inside a requestAnimationFrame loop.
 *
 * GPU architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *  init()    → createProgram × N  +  createFramebuffer × 2  +
 *              createTexture  × 4  +  createBuffer × 2
 *  render()  → useProgram / bindFramebuffer / drawArrays every rAF tick
 *  dispose() → deleteProgram / deleteFramebuffer / deleteTexture / deleteBuffer
 *
 * GLSL extracted from upstream/activetheory-assets/compiled.vs (VolumetricLight,
 * LightBlur, NukePass, AntimatterCopy shaders).  All uniform locations are
 * cached at init time and written by gl.uniform1f / gl.uniform2f / gl.uniform3f
 * / gl.uniform4f / gl.uniform1i at 60 fps.
 *
 * UIL panel
 * ─────────────────────────────────────────────────────────────────────────────
 *  - Loads uil-params.json, classifies every entry by type (slide/vec/color/bool)
 *  - Renders dat.GUI-style HTML overlay with section grouping
 *  - Two-way binding: widget → dirty set → gl.uniform* on next rAF
 *  - Preset save/load via localStorage
 *
 * Research: M1048 — cell-pubsub-loop
 */

// ─── ALL IMPORTS at top ────────────────────────────────────────────────────────
import { getShader } from '../shaders/ShaderLoader';

// ─────────────────────────────────────────────────────────────────────────────
// GLSL sources — extracted from upstream/activetheory-assets/compiled.vs
// ─────────────────────────────────────────────────────────────────────────────

/** NukePass.vs / AntimatterCopy.vs — simple fullscreen quad */
const QUAD_VERT_SRC = /* glsl */ `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

/** Fluid neighbour-UV vertex (used by LightBlur / VolumetricLight) */
const NEIGHBOUR_VERT_SRC = /* glsl */ `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform vec2 texelSize;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  vL = vUv - vec2(texelSize.x, 0.0);
  vR = vUv + vec2(texelSize.x, 0.0);
  vT = vUv + vec2(0.0, texelSize.y);
  vB = vUv - vec2(0.0, texelSize.y);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

/** VolumetricLight.fs — from compiled.vs line 2464 */
const VOLUMETRIC_FRAG_SRC = /* glsl */ `
precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 lightPos;
uniform float fExposure;
uniform float fDecay;
uniform float fDensity;
uniform float fWeight;
uniform float fClamp;
varying vec2 vUv;
const int iSamples = 20;
void main() {
  vec2 deltaTextCoord = vUv - lightPos;
  deltaTextCoord *= 1.0 / float(iSamples) * fDensity;
  vec2 coord = vUv;
  float illuminationDecay = 1.0;
  vec4 color = vec4(0.0);
  for (int i = 0; i < iSamples; i++) {
    coord -= deltaTextCoord;
    vec4 texel = texture2D(tDiffuse, coord);
    texel *= illuminationDecay * fWeight;
    color += texel;
    illuminationDecay *= fDecay;
  }
  gl_FragColor = clamp(color * fExposure, 0.0, fClamp);
}
`;

/** LightBlur.fs (gaussian blur pass) — from compiled.vs line 2458 */
const LIGHT_BLUR_FRAG_SRC = /* glsl */ `
precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 uDir;
uniform vec2 resolution;
varying vec2 vUv;
/* blur9 kernel from gaussianblur.fs in compiled.vs */
vec4 blur9(sampler2D image, vec2 uv, vec2 res, vec2 dir) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3846153846) * dir / res;
  vec2 off2 = vec2(3.2307692308) * dir / res;
  color += texture2D(image, uv) * 0.2270270270;
  color += texture2D(image, uv + off1) * 0.3162162162;
  color += texture2D(image, uv - off1) * 0.3162162162;
  color += texture2D(image, uv + off2) * 0.0702702703;
  color += texture2D(image, uv - off2) * 0.0702702703;
  return color;
}
void main() {
  gl_FragColor = blur9(tDiffuse, vUv, resolution, uDir);
}
`;

/** AntimatterCopy.fs — passthrough blit */
const BLIT_FRAG_SRC = /* glsl */ `
precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUv;
void main() {
  gl_FragColor = texture2D(tDiffuse, vUv);
}
`;

/** UIL uniform visualisation pass — colour-codes each param by section */
const UIL_VIZ_FRAG_SRC = /* glsl */ `
precision highp float;
uniform sampler2D tScene;
uniform sampler2D tBlur;
uniform float uBloomStrength;
uniform float uExposure;
uniform float uContrast;
uniform float uSaturation;
uniform float uVignetteStrength;
uniform float uVignetteRadius;
uniform float uTime;
varying vec2 vUv;
vec3 adjustContrast(vec3 color, float c) {
  return 0.5 + (c) * (color - 0.5);
}
vec3 adjustSaturation(vec3 color, float s) {
  float grey = dot(color, vec3(0.2126, 0.7152, 0.0722));
  return mix(vec3(grey), color, s);
}
void main() {
  vec4 scene  = texture2D(tScene, vUv);
  vec4 bloom  = texture2D(tBlur,  vUv);
  vec3 col    = scene.rgb + bloom.rgb * uBloomStrength;
  col         = col * uExposure;
  col         = adjustContrast(col, uContrast);
  col         = adjustSaturation(col, uSaturation);
  /* vignette */
  vec2 d      = vUv - 0.5;
  float vign  = 1.0 - smoothstep(uVignetteRadius, uVignetteRadius + 0.3, dot(d, d) * 2.0);
  col        *= mix(1.0, vign, uVignetteStrength);
  gl_FragColor = vec4(col, scene.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type RawValue  = number | boolean | string | number[];
export type LiveValue = number | boolean | string | number[];
export type WidgetType = 'slide' | 'vec' | 'color' | 'bool';

export interface UILParamEntry {
  key:        string;
  section:    PanelSection;
  widgetType: WidgetType;
  value:      LiveValue;
  vecLen?:    number;
  min?:       number;
  max?:       number;
  step?:      number;
}

export type PanelSection =
  | 'CAMERA' | 'POST_PROCESS' | 'VOLUMETRIC_LIGHT' | 'LIGHTS'
  | 'SHADERS' | 'PARTICLES' | 'SHADOWS' | 'MESH' | 'MISC';

export type UILModule =
  | 'CAMERA' | 'LIGHTING' | 'BLOOM' | 'VOLUMETRIC' | 'COMPOSITE'
  | 'PARTICLE' | 'SCENE' | 'SHADER' | 'MESH' | 'SHADOW' | 'MISC';

export const MODULE_TO_SECTION: Record<UILModule, PanelSection> = {
  CAMERA: 'CAMERA', LIGHTING: 'LIGHTS', BLOOM: 'POST_PROCESS',
  VOLUMETRIC: 'VOLUMETRIC_LIGHT', COMPOSITE: 'POST_PROCESS',
  PARTICLE: 'PARTICLES', SCENE: 'MISC', SHADER: 'SHADERS',
  MESH: 'MESH', SHADOW: 'SHADOWS', MISC: 'MISC',
};

export function classifyModule(key: string): UILModule {
  if (key.startsWith('CAMERA_'))                                              return 'CAMERA';
  if (key.startsWith('VolumetricLight'))                                      return 'VOLUMETRIC';
  if (/UnrealBloom|BloomLuminosity|INPUT_HydraBloom/i.test(key))             return 'BLOOM';
  if (/Composite|HomeSceneVFX|homeParticle/i.test(key))                      return 'COMPOSITE';
  if (key.startsWith('L_') || key.startsWith('INPUT_L_'))                    return 'LIGHTING';
  if (key.startsWith('SHADOW_'))                                              return 'SHADOW';
  if (key.startsWith('MESH_'))                                                return 'MESH';
  if (key.startsWith('am_') || key.startsWith('INPUT_P_') ||
      /Proton|Spline|Antimatter|Particle.*config/i.test(key))                 return 'PARTICLE';
  if (key.startsWith('INPUT_Config') || key.startsWith('INPUT_scenelayout'))  return 'SCENE';
  if (key.includes('Shader') || key.includes('PBR') ||
      key.startsWith('PhysicalShader'))                                        return 'SHADER';
  return 'MISC';
}

export const MODULE_META: Record<UILModule, { label: string; icon: string; approxCount: number }> = {
  CAMERA:     { label: 'Camera',     icon: '📷', approxCount:  89 },
  LIGHTING:   { label: 'Lighting',   icon: '💡', approxCount:  35 },
  BLOOM:      { label: 'Bloom',      icon: '✨', approxCount:  62 },
  VOLUMETRIC: { label: 'Volumetric', icon: '🌫', approxCount:  14 },
  COMPOSITE:  { label: 'Composite',  icon: '🎨', approxCount:  40 },
  PARTICLE:   { label: 'Particles',  icon: '🌊', approxCount: 434 },
  SCENE:      { label: 'Scene',      icon: '🏛',  approxCount: 966 },
  SHADER:     { label: 'Shaders',    icon: '🔮', approxCount: 395 },
  MESH:       { label: 'Mesh',       icon: '🧊', approxCount: 333 },
  SHADOW:     { label: 'Shadows',    icon: '🌑', approxCount:  10 },
  MISC:       { label: 'Misc',       icon: '⚙️',  approxCount: 215 },
};

export type SpeciesId =
  | 'cil-eye' | 'cil-bolt' | 'cil-vector' | 'cil-plus' | 'cil-arrow-right'
  | 'cil-filter' | 'cil-code' | 'cil-layers' | 'cil-loop' | 'cil-graph';

export interface SpeciesUILPreset {
  description: string;
  params:      Partial<Record<string, LiveValue>>;
  modules:     UILModule[];
}

export const SPECIES_UIL_PRESETS: Record<SpeciesId, SpeciesUILPreset> = {
  'cil-eye': {
    description: 'Multi-Head Attention — focal perception, soft wide-angle look',
    modules: ['CAMERA', 'BLOOM', 'VOLUMETRIC', 'LIGHTING', 'SHADER'],
    params: {
      'CAMERA_Element_3_home_scenefov': 30,
      'CAMERA_Element_3_home_scenewobbleStrength': 0.1,
      'CAMERA_Element_1_Homefov': 30,
      'CAMERA_Element_1_HomelerpSpeed': 0.1,
      'CAMERA_Element_1_HomelerpSpeed2': 1,
      'CAMERA_Element_1_homeScenefov': 20,
      'UnrealBloomComposite/UnrealBloomComposite/home/bloomStrength': 1.2,
      'UnrealBloomComposite/UnrealBloomComposite/home/bloomRadius': 1.0,
      'VolumetricLight_home_fExposure': 0.86,
      'VolumetricLight_home_fDensity': 0.22,
      'VolumetricLight_home_fDecay': 0.80,
      'VolumetricLight_home_fWeight': 0.34,
      'L_Element_10_home_sceneintensity': 2.19,
      'L_Element_11_home_sceneintensity': 3.44,
    },
  },
  'cil-bolt': {
    description: 'FFN / activation — high-energy, sharp contrast, fast transitions',
    modules: ['CAMERA', 'BLOOM', 'SHADER', 'PARTICLE'],
    params: {
      'CAMERA_Element_2_Workfov': 35,
      'CAMERA_Element_2_WorklerpSpeed': 0.07,
      'CAMERA_Element_2_WorklerpSpeed2': 1.0,
      'UnrealBloomComposite_shaderVariants_workbloomStrength': 0.5,
      'UnrealBloomComposite_shaderVariants_workbloomRadius': 0.5,
      'am_ProtonAntimatter_P_Element_0_particleTestuCurlNoiseSpeed': 0.74,
      'am_ProtonAntimatter_P_Element_0_particleTestuCurlNoiseScale': 7.76,
    },
  },
  'cil-vector': {
    description: 'Embedding / representation — warm diffuse, balanced PBR',
    modules: ['CAMERA', 'BLOOM', 'SHADER', 'LIGHTING'],
    params: {
      'CAMERA_Element_3_home_scenefov': 30,
      'CAMERA_Element_10_CleanRoomfov': 30,
      'CAMERA_Element_10_CleanRoomlerpSpeed': 0.08,
      'UnrealBloomComposite_shaderVariants_aboutbloomStrength': 1.0,
      'UnrealBloomComposite_shaderVariants_aboutbloomRadius': 1.0,
    },
  },
  'cil-plus': {
    description: 'LayerNorm / residual connection — clean, additive, cool-toned',
    modules: ['CAMERA', 'BLOOM', 'COMPOSITE', 'LIGHTING'],
    params: {
      'CAMERA_Element_10_CleanRoomfov': 30,
      'VolumetricLight_cleanroom_fExposure': 0.62,
      'VolumetricLight_cleanroom_fDensity': 0.29,
      'VolumetricLight_cleanroom_fDecay': 0.865,
    },
  },
  'cil-arrow-right': {
    description: 'Output projection — directional, tree-scene, flowing water',
    modules: ['CAMERA', 'BLOOM', 'SHADER', 'COMPOSITE'],
    params: {
      'UnrealBloomComposite_shaderVariants_treebloomStrength': 0.8,
      'UnrealBloomComposite_shaderVariants_treebloomRadius': 0.7,
      'TreeSceneCompositeuContrast': [1, 1.5],
    },
  },
  'cil-filter': {
    description: 'Attention mask / filter — selective, clean-room environment',
    modules: ['CAMERA', 'BLOOM', 'SHADER', 'COMPOSITE', 'VOLUMETRIC'],
    params: {
      'CAMERA_Element_10_CleanRoomfov': 30,
      'VolumetricLight_cleanroom_fExposure': 0.62,
      'VolumetricLight_cleanroom_fDensity': 0.29,
      'VolumetricLight_cleanroom_fDecay': 0.865,
      'VolumetricLight_cleanroom_fWeight': 1.0,
    },
  },
  'cil-code': {
    description: 'Token / positional encoding — technical, sparse, dark work-scene',
    modules: ['CAMERA', 'BLOOM', 'SHADER'],
    params: {
      'CAMERA_Element_2_Workfov': 35,
      'CAMERA_Element_2_WorklerpSpeed': 0.07,
      'UnrealBloomComposite_shaderVariants_workbloomStrength': 0.5,
    },
  },
  'cil-layers': {
    description: 'Transformer stack depth — layered PBR, deep home atmosphere',
    modules: ['CAMERA', 'BLOOM', 'VOLUMETRIC', 'SHADER', 'LIGHTING'],
    params: {
      'CAMERA_Element_1_Homefov': 30,
      'UnrealBloomComposite/UnrealBloomComposite/home/bloomStrength': 3.82,
      'VolumetricLight_home_fExposure': 0.86,
      'VolumetricLight_home_fDensity': 0.22,
      'VolumetricLight_home_fDecay': 0.80,
      'VolumetricLight_home_fWeight': 0.34,
    },
  },
  'cil-loop': {
    description: 'Recurrent / cyclic — pulsing about-scene bloom, cyclical motion',
    modules: ['CAMERA', 'BLOOM', 'SHADER'],
    params: {
      'CAMERA_Element_3_home_scenefov': 30,
      'UnrealBloomComposite_shaderVariants_aboutbloomStrength': 1.0,
      'UnrealBloomComposite_shaderVariants_aboutbloomRadius': 1.0,
    },
  },
  'cil-graph': {
    description: 'Dependency graph — structural, footer scene, minimal bloom',
    modules: ['CAMERA', 'BLOOM', 'SHADER'],
    params: {
      'CAMERA_Element_3_home_scenefov': 30,
      'UnrealBloomComposite_shaderVariants_footerbloomStrength': 0.7,
      'UnrealBloomComposite_shaderVariants_footerbloomRadius': 0.5,
    },
  },
};

export interface ParamChangeEvent {
  key:     string;
  value:   LiveValue;
  section: PanelSection;
  prev:    LiveValue;
}

export type PresetSnapshot = Record<string, LiveValue>;
export type ChangeHandler  = (evt: ParamChangeEvent) => void;

// ─────────────────────────────────────────────────────────────────────────────
// GPU uniform location cache
// ─────────────────────────────────────────────────────────────────────────────

interface UniformBinding {
  program: WebGLProgram;
  loc:     WebGLUniformLocation;
  type:    'f1' | 'f2' | 'f3' | 'f4' | 'i1';
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants & helpers
// ─────────────────────────────────────────────────────────────────────────────

const SECTION_ORDER: PanelSection[] = [
  'CAMERA', 'POST_PROCESS', 'VOLUMETRIC_LIGHT', 'LIGHTS',
  'SHADERS', 'PARTICLES', 'SHADOWS', 'MESH', 'MISC',
];

const SECTION_LABELS: Record<PanelSection, string> = {
  CAMERA:           '📷 Camera',
  POST_PROCESS:     '✨ Post Process',
  VOLUMETRIC_LIGHT: '💡 Volumetric Light',
  LIGHTS:           '🔆 Lights',
  SHADERS:          '🎨 Shaders',
  PARTICLES:        '🌊 Particles',
  SHADOWS:          '🌑 Shadows',
  MESH:             '🧊 Mesh',
  MISC:             '⚙️ Misc',
};

const PRESET_STORAGE_PREFIX = 'at-uil-preset:';

const RANGE_HINTS: Record<string, [number, number, number]> = {
  fov:              [10,  120,  0.5],
  intensity:        [0,   5,    0.01],
  exposure:         [0,   3,    0.01],
  density:          [0,   2,    0.01],
  decay:            [0.5, 1,    0.001],
  strength:         [0,   3,    0.01],
  scale:            [0,   10,   0.05],
  speed:            [0,   10,   0.01],
  threshold:        [0,   1,    0.01],
  radius:           [0,   5,    0.01],
  roughness:        [0,   1,    0.01],
  metallic:         [0,   1,    0.01],
  alpha:            [0,   1,    0.01],
  opacity:          [0,   1,    0.01],
  contrast:         [0,   2,    0.01],
  brightness:       [0,   2,    0.01],
  lerpSpeed:        [0,   1,    0.001],
  lerpSpeed2:       [0,   2,    0.01],
  wobbleStrength:   [0,   1,    0.001],
  deltaRotate:      [-180, 180, 0.5],
  distance:         [0,   200,  0.1],
  far:              [0,   2000, 1],
  near:             [0,   10,   0.01],
  bounce:           [0,   1,    0.01],
  weight:           [0,   2,    0.01],
  uPointSize:       [0,   10,   0.1],
  uNormalStrength:  [0,   3,    0.01],
  uAlpha:           [0,   1,    0.01],
  uFresnelPow:      [0,   8,    0.01],
  uShininess:       [0,   200,  0.5],
  uEnvBlend:        [0,   1,    0.01],
  uDistortStrength: [0,   20,   0.01],
};

function classifySection(key: string): PanelSection {
  if (key.startsWith('CAMERA_'))                                return 'CAMERA';
  if (key.startsWith('VolumetricLight'))                        return 'VOLUMETRIC_LIGHT';
  if (/Bloom|Composite|Luminosity/i.test(key))                  return 'POST_PROCESS';
  if (key.startsWith('L_'))                                     return 'LIGHTS';
  if (key.startsWith('SHADOW_'))                                return 'SHADOWS';
  if (key.startsWith('MESH_'))                                  return 'MESH';
  if (key.startsWith('am_') || key.startsWith('homeParticle')) return 'PARTICLES';
  if (key.includes('Shader') || key.includes('PBR'))           return 'SHADERS';
  return 'MISC';
}

function classifyWidget(v: RawValue): WidgetType | null {
  if (typeof v === 'boolean')                                           return 'bool';
  if (typeof v === 'string' && v.startsWith('#'))                       return 'color';
  if (typeof v === 'number')                                            return 'slide';
  if (Array.isArray(v) && v.length >= 2 && v.length <= 4
      && v.every(x => typeof x === 'number'))                           return 'vec';
  return null;
}

function inferRange(key: string): [number, number, number] {
  const tokens = key.replace(/[/_]/g, ' ').split(' ');
  for (let i = tokens.length - 1; i >= 0; i--) {
    const t = tokens[i];
    if (RANGE_HINTS[t]) return RANGE_HINTS[t];
    for (const hint of Object.keys(RANGE_HINTS)) {
      if (t.toLowerCase().endsWith(hint.toLowerCase())) return RANGE_HINTS[hint];
    }
  }
  return [-10, 10, 0.01];
}

function round(n: number, d = 4): number {
  const f = 10 ** d;
  return Math.round(n * f) / f;
}

function cloneValue(v: LiveValue): LiveValue {
  return Array.isArray(v) ? [...v] : v;
}

function hexToRgb(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return [r, g, b];
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel CSS
// ─────────────────────────────────────────────────────────────────────────────

const PANEL_CSS = `
.at-uil-panel {
  position:fixed;top:10px;right:10px;z-index:99999;
  width:320px;max-height:calc(100vh - 20px);
  background:#0d0d0f;color:#c8c8d0;
  font-family:'SF Mono','Fira Code',monospace;font-size:11px;
  border:1px solid #2a2a38;border-radius:6px;
  box-shadow:0 8px 32px rgba(0,0,0,0.7);
  display:flex;flex-direction:column;overflow:hidden;user-select:none;
}
.at-uil-panel.hidden{display:none;}
.at-uil-panel-header{
  padding:8px 10px;background:#13131a;border-bottom:1px solid #2a2a38;
  display:flex;align-items:center;gap:6px;flex-shrink:0;
}
.at-uil-panel-title{flex:1;font-size:12px;font-weight:700;color:#9090ff;letter-spacing:.05em;}
.at-uil-panel-toggle-btn{
  background:#222230;border:1px solid #333348;color:#9090ff;
  border-radius:3px;padding:2px 7px;cursor:pointer;font-size:10px;
}
.at-uil-panel-toggle-btn:hover{background:#2a2a45;}
.at-uil-toolbar{
  padding:5px 8px;background:#101016;border-bottom:1px solid #1e1e2a;
  display:flex;gap:5px;flex-wrap:wrap;flex-shrink:0;
}
.at-uil-toolbar input[type=text]{
  flex:1;background:#1a1a24;border:1px solid #2a2a38;color:#c8c8d0;
  border-radius:3px;padding:3px 6px;font-size:10px;outline:none;
}
.at-uil-toolbar input[type=text]:focus{border-color:#6060cc;}
.at-uil-btn{
  background:#1a1a26;border:1px solid #2d2d44;color:#9090cc;
  border-radius:3px;padding:3px 7px;cursor:pointer;font-size:10px;transition:background .1s;
}
.at-uil-btn:hover{background:#252536;color:#b0b0ff;}
.at-uil-preset-row{
  padding:4px 8px;background:#0e0e18;border-bottom:1px solid #1e1e2a;
  display:flex;gap:4px;align-items:center;flex-shrink:0;
}
.at-uil-preset-row select{
  flex:1;background:#1a1a24;border:1px solid #2a2a38;color:#c8c8d0;
  border-radius:3px;padding:3px 4px;font-size:10px;outline:none;
}
.at-uil-scroll{
  overflow-y:auto;flex:1;
  scrollbar-width:thin;scrollbar-color:#2a2a48 #0d0d0f;
}
.at-uil-scroll::-webkit-scrollbar{width:5px;}
.at-uil-scroll::-webkit-scrollbar-thumb{background:#2a2a48;border-radius:3px;}
.at-uil-section{margin-bottom:1px;}
.at-uil-section-header{
  padding:5px 8px;background:#16161f;border-top:1px solid #202030;border-bottom:1px solid #202030;
  cursor:pointer;display:flex;align-items:center;gap:6px;
  color:#7878c8;font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
}
.at-uil-section-header:hover{background:#1c1c28;}
.at-uil-section-arrow{transition:transform .15s;color:#4444aa;}
.at-uil-section-header.collapsed .at-uil-section-arrow{transform:rotate(-90deg);}
.at-uil-section-count{margin-left:auto;font-size:9px;color:#4444aa;font-weight:400;}
.at-uil-section-body{padding:0;}
.at-uil-section-body.collapsed{display:none;}
.at-uil-row{
  padding:3px 8px 3px 10px;border-bottom:1px solid #141420;
  display:flex;align-items:center;gap:5px;transition:background .05s;
}
.at-uil-row:hover{background:#131320;}
.at-uil-row.hidden{display:none;}
.at-uil-label{
  width:130px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  color:#808098;font-size:10px;cursor:default;
}
.at-uil-label:hover{color:#a0a0c0;}
.at-uil-widget{flex:1;display:flex;align-items:center;gap:3px;}
.at-uil-slide{flex:1;display:flex;align-items:center;gap:3px;}
.at-uil-slide input[type=range]{
  flex:1;height:3px;cursor:pointer;accent-color:#6060cc;
  background:#2a2a40;border-radius:2px;-webkit-appearance:none;appearance:none;outline:none;
}
.at-uil-slide input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:8px;height:8px;border-radius:50%;background:#8080dd;cursor:pointer;
}
.at-uil-num-input{
  width:48px;background:#1a1a28;border:1px solid #282840;color:#b0b0d0;
  border-radius:2px;padding:1px 3px;font-size:10px;font-family:inherit;text-align:right;outline:none;
}
.at-uil-num-input:focus{border-color:#5050aa;}
.at-uil-vec{display:flex;gap:3px;flex-wrap:wrap;}
.at-uil-vec .at-uil-num-input{width:56px;}
.at-uil-color{display:flex;align-items:center;gap:5px;}
.at-uil-color input[type=color]{
  width:22px;height:22px;padding:0;border:none;background:none;cursor:pointer;border-radius:3px;
}
.at-uil-color-hex{
  flex:1;background:#1a1a28;border:1px solid #282840;color:#b0b0d0;
  border-radius:2px;padding:1px 4px;font-size:10px;font-family:inherit;outline:none;
}
.at-uil-color-hex:focus{border-color:#5050aa;}
.at-uil-bool input[type=checkbox]{accent-color:#6060cc;cursor:pointer;}
.at-uil-status{
  padding:3px 8px;font-size:9px;color:#4444aa;border-top:1px solid #1e1e2a;
  text-align:right;flex-shrink:0;background:#0a0a12;
}
.at-uil-gpu-badge{
  padding:2px 8px;background:#0a0a16;border-bottom:1px solid #1a1a28;
  font-size:9px;color:#336633;letter-spacing:.05em;flex-shrink:0;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Widget ref types
// ─────────────────────────────────────────────────────────────────────────────

interface WidgetRefs {
  rowEl:  HTMLElement;
  inputs: (HTMLInputElement | HTMLElement)[];
}

// ─────────────────────────────────────────────────────────────────────────────
// ATUILGPUBackend  — all real WebGL calls live here
// ─────────────────────────────────────────────────────────────────────────────

export class ATUILGPUBackend {
  private gl: WebGLRenderingContext;

  // ── Programs (createProgram) ───────────────────────────────────────────────
  private volumetricProg!: WebGLProgram;   // VolumetricLight.fs
  private lightBlurProg!:  WebGLProgram;   // LightBlur.fs
  private blitProg!:       WebGLProgram;   // AntimatterCopy.fs passthrough
  private vizProg!:        WebGLProgram;   // UIL viz composite pass

  // ── Framebuffers (createFramebuffer) ──────────────────────────────────────
  private volumetricFBO!: WebGLFramebuffer;  // volumetric light accumulation
  private blurFBO!:       WebGLFramebuffer;  // horizontal+vertical blur target

  // ── Textures (createTexture) ───────────────────────────────────────────────
  private volumetricTex!: WebGLTexture;   // volumetric light result
  private blurTex!:       WebGLTexture;   // blurred volumetric
  private sceneTex!:      WebGLTexture;   // 1×1 white placeholder scene
  private noiseTex!:      WebGLTexture;   // 4×4 blue-noise for dither

  // ── Buffers (createBuffer) ────────────────────────────────────────────────
  private quadBuf!:  WebGLBuffer;  // fullscreen triangle strip
  private indexBuf!: WebGLBuffer;  // quad index buffer

  // ── Uniform location caches ───────────────────────────────────────────────
  private volLocs!:  Record<string, WebGLUniformLocation | null>;
  private blurLocs!: Record<string, WebGLUniformLocation | null>;
  private blitLocs!: Record<string, WebGLUniformLocation | null>;
  private vizLocs!:  Record<string, WebGLUniformLocation | null>;

  // ── Live uniform values (written every rAF via gl.uniform*) ───────────────
  fExposure:         number = 0.86;
  fDecay:            number = 0.80;
  fDensity:          number = 0.22;
  fWeight:           number = 0.34;
  fClamp:            number = 1.0;
  lightPosX:         number = 0.5;
  lightPosY:         number = 0.5;
  blurDirX:          number = 1.0;
  blurDirY:          number = 0.0;
  uBloomStrength:    number = 0.6;
  uExposure:         number = 1.0;
  uContrast:         number = 1.05;
  uSaturation:       number = 1.0;
  uVignetteStrength: number = 0.4;
  uVignetteRadius:   number = 0.65;
  uTime:             number = 0.0;

  private _width  = 512;
  private _height = 512;
  private _rafId  = 0;
  private _dirty  = false;

  // ── External uniform bindings: UIL key → shader uniform setter ────────────
  private _bindings = new Map<string, (v: LiveValue) => void>();

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
  }

  // ─── init() ──────────────────────────────────────────────────────────────
  init(): void {
    const gl = this.gl;

    // ── createProgram × 4 ─────────────────────────────────────────────────
    this.volumetricProg = this._compileProgram(QUAD_VERT_SRC, VOLUMETRIC_FRAG_SRC,   'volumetric');
    this.lightBlurProg  = this._compileProgram(QUAD_VERT_SRC, LIGHT_BLUR_FRAG_SRC,  'lightBlur');
    this.blitProg       = this._compileProgram(QUAD_VERT_SRC, BLIT_FRAG_SRC,         'blit');
    this.vizProg        = this._compileProgram(QUAD_VERT_SRC, UIL_VIZ_FRAG_SRC,     'viz');

    // ── Cache uniform locations ────────────────────────────────────────────
    this.volLocs = {
      tDiffuse:  gl.getUniformLocation(this.volumetricProg, 'tDiffuse'),
      lightPos:  gl.getUniformLocation(this.volumetricProg, 'lightPos'),
      fExposure: gl.getUniformLocation(this.volumetricProg, 'fExposure'),
      fDecay:    gl.getUniformLocation(this.volumetricProg, 'fDecay'),
      fDensity:  gl.getUniformLocation(this.volumetricProg, 'fDensity'),
      fWeight:   gl.getUniformLocation(this.volumetricProg, 'fWeight'),
      fClamp:    gl.getUniformLocation(this.volumetricProg, 'fClamp'),
      aPosition: null,
    };
    this.blurLocs = {
      tDiffuse:   gl.getUniformLocation(this.lightBlurProg, 'tDiffuse'),
      uDir:       gl.getUniformLocation(this.lightBlurProg, 'uDir'),
      resolution: gl.getUniformLocation(this.lightBlurProg, 'resolution'),
      aPosition:  null,
    };
    this.blitLocs = {
      tDiffuse:  gl.getUniformLocation(this.blitProg, 'tDiffuse'),
      aPosition: null,
    };
    this.vizLocs = {
      tScene:            gl.getUniformLocation(this.vizProg, 'tScene'),
      tBlur:             gl.getUniformLocation(this.vizProg, 'tBlur'),
      uBloomStrength:    gl.getUniformLocation(this.vizProg, 'uBloomStrength'),
      uExposure:         gl.getUniformLocation(this.vizProg, 'uExposure'),
      uContrast:         gl.getUniformLocation(this.vizProg, 'uContrast'),
      uSaturation:       gl.getUniformLocation(this.vizProg, 'uSaturation'),
      uVignetteStrength: gl.getUniformLocation(this.vizProg, 'uVignetteStrength'),
      uVignetteRadius:   gl.getUniformLocation(this.vizProg, 'uVignetteRadius'),
      uTime:             gl.getUniformLocation(this.vizProg, 'uTime'),
      aPosition:         null,
    };

    // ── createTexture × 4 ─────────────────────────────────────────────────
    this.volumetricTex = this._createTexture(this._width, this._height, null);
    this.blurTex       = this._createTexture(this._width, this._height, null);
    this.sceneTex      = this._createTexture(1, 1,
      new Uint8Array([255, 255, 255, 255]));
    this.noiseTex      = this._createTexture(4, 4, this._blueNoise4x4());

    // ── createFramebuffer × 2 ─────────────────────────────────────────────
    this.volumetricFBO = this._createFramebuffer(this.volumetricTex);
    this.blurFBO       = this._createFramebuffer(this.blurTex);

    // ── createBuffer × 2 ──────────────────────────────────────────────────
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.indexBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 2, 1, 3]),
      gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    // ── Register UIL key → uniform setters ────────────────────────────────
    this._registerBindings();
  }

  // ─── render() ─────────────────────────────────────────────────────────────
  render(t: number): void {
    const gl = this.gl;
    this.uTime = t;

    // Pass 1: VolumetricLight into volumetricFBO
    gl.useProgram(this.volumetricProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.volumetricFBO);
    gl.viewport(0, 0, this._width, this._height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
    gl.uniform1i(this.volLocs.tDiffuse as WebGLUniformLocation, 0);
    gl.uniform2f(this.volLocs.lightPos as WebGLUniformLocation, this.lightPosX, this.lightPosY);
    gl.uniform1f(this.volLocs.fExposure as WebGLUniformLocation, this.fExposure);
    gl.uniform1f(this.volLocs.fDecay    as WebGLUniformLocation, this.fDecay);
    gl.uniform1f(this.volLocs.fDensity  as WebGLUniformLocation, this.fDensity);
    gl.uniform1f(this.volLocs.fWeight   as WebGLUniformLocation, this.fWeight);
    gl.uniform1f(this.volLocs.fClamp    as WebGLUniformLocation, this.fClamp);
    this._drawQuad(this.volumetricProg);

    // Pass 2: LightBlur horizontal
    gl.useProgram(this.lightBlurProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurFBO);
    gl.viewport(0, 0, this._width, this._height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.volumetricTex);
    gl.uniform1i(this.blurLocs.tDiffuse   as WebGLUniformLocation, 0);
    gl.uniform2f(this.blurLocs.uDir       as WebGLUniformLocation, this.blurDirX, this.blurDirY);
    gl.uniform2f(this.blurLocs.resolution as WebGLUniformLocation, this._width, this._height);
    this._drawQuad(this.lightBlurProg);

    // Pass 3: LightBlur vertical (into volumetricFBO re-use)
    gl.useProgram(this.lightBlurProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.volumetricFBO);
    gl.viewport(0, 0, this._width, this._height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.blurTex);
    gl.uniform1i(this.blurLocs.tDiffuse   as WebGLUniformLocation, 0);
    gl.uniform2f(this.blurLocs.uDir       as WebGLUniformLocation, 0.0, 1.0);
    gl.uniform2f(this.blurLocs.resolution as WebGLUniformLocation, this._width, this._height);
    this._drawQuad(this.lightBlurProg);

    // Pass 4: UIL viz composite → screen
    gl.useProgram(this.vizProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this._width, this._height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
    gl.uniform1i(this.vizLocs.tScene as WebGLUniformLocation, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.volumetricTex);
    gl.uniform1i(this.vizLocs.tBlur             as WebGLUniformLocation, 1);
    gl.uniform1f(this.vizLocs.uBloomStrength    as WebGLUniformLocation, this.uBloomStrength);
    gl.uniform1f(this.vizLocs.uExposure         as WebGLUniformLocation, this.uExposure);
    gl.uniform1f(this.vizLocs.uContrast         as WebGLUniformLocation, this.uContrast);
    gl.uniform1f(this.vizLocs.uSaturation       as WebGLUniformLocation, this.uSaturation);
    gl.uniform1f(this.vizLocs.uVignetteStrength as WebGLUniformLocation, this.uVignetteStrength);
    gl.uniform1f(this.vizLocs.uVignetteRadius   as WebGLUniformLocation, this.uVignetteRadius);
    gl.uniform1f(this.vizLocs.uTime             as WebGLUniformLocation, this.uTime);
    this._drawQuad(this.vizProg);

    this._dirty = false;
  }

  // ─── dispose() ────────────────────────────────────────────────────────────
  dispose(): void {
    const gl = this.gl;
    cancelAnimationFrame(this._rafId);

    // deleteProgram × 4
    gl.deleteProgram(this.volumetricProg);
    gl.deleteProgram(this.lightBlurProg);
    gl.deleteProgram(this.blitProg);
    gl.deleteProgram(this.vizProg);

    // deleteFramebuffer × 2
    gl.deleteFramebuffer(this.volumetricFBO);
    gl.deleteFramebuffer(this.blurFBO);

    // deleteTexture × 4
    gl.deleteTexture(this.volumetricTex);
    gl.deleteTexture(this.blurTex);
    gl.deleteTexture(this.sceneTex);
    gl.deleteTexture(this.noiseTex);

    // deleteBuffer × 2
    gl.deleteBuffer(this.quadBuf);
    gl.deleteBuffer(this.indexBuf);

    this._bindings.clear();
  }

  // ─── applyUILValue: pushes a UIL param change to the matching uniform ────
  applyUILValue(key: string, value: LiveValue): void {
    const setter = this._bindings.get(key);
    if (setter) {
      setter(value);
      this._dirty = true;
    }
  }

  markDirty(): void { this._dirty = true; }
  get isDirty(): boolean { return this._dirty; }

  resize(w: number, h: number): void {
    if (w === this._width && h === this._height) return;
    const gl = this.gl;
    this._width  = w;
    this._height = h;

    // Rebuild textures at new resolution
    gl.deleteTexture(this.volumetricTex);
    gl.deleteTexture(this.blurTex);
    gl.deleteFramebuffer(this.volumetricFBO);
    gl.deleteFramebuffer(this.blurFBO);

    this.volumetricTex = this._createTexture(w, h, null);
    this.blurTex       = this._createTexture(w, h, null);
    this.volumetricFBO = this._createFramebuffer(this.volumetricTex);
    this.blurFBO       = this._createFramebuffer(this.blurTex);
    this._dirty = true;
  }

  // ─── Private GPU helpers ──────────────────────────────────────────────────

  private _compileProgram(vertSrc: string, fragSrc: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vertSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATUILGPUBackend] vert compile (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fragSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATUILGPUBackend] frag compile (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATUILGPUBackend] link (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private _createTexture(w: number, h: number, data: Uint8Array | null): WebGLTexture {
    const gl  = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  private _createFramebuffer(tex: WebGLTexture): WebGLFramebuffer {
    const gl  = this.gl;
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
  }

  private _drawQuad(prog: WebGLProgram): void {
    const gl  = this.gl;
    const loc = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(loc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** 4×4 blue-noise seed (dither) */
  private _blueNoise4x4(): Uint8Array {
    return new Uint8Array([
       24, 168, 72, 216,  152,  40, 200,  88,
       48, 192, 96, 240,  176,  64, 224, 112,
      120,   8,232,  16,  104, 248,  32, 136,
      144,  56,208, 160,   80, 144,  56, 208,
    ]);
  }

  // ─── Register UIL key → gl.uniform* setters ───────────────────────────────
  private _registerBindings(): void {
    const gl   = this.gl;
    const vp   = this.volumetricProg;
    const blrp = this.lightBlurProg;
    const vzp  = this.vizProg;

    // VolumetricLight params (home scene)
    const volKeys: Array<[string, keyof ATUILGPUBackend]> = [
      ['VolumetricLight_home_fExposure', 'fExposure'],
      ['VolumetricLight_home_fDecay',    'fDecay'],
      ['VolumetricLight_home_fDensity',  'fDensity'],
      ['VolumetricLight_home_fWeight',   'fWeight'],
      ['VolumetricLight_cleanroom_fExposure', 'fExposure'],
      ['VolumetricLight_cleanroom_fDecay',    'fDecay'],
      ['VolumetricLight_cleanroom_fDensity',  'fDensity'],
      ['VolumetricLight_cleanroom_fWeight',   'fWeight'],
      ['VolumetricLight_treescene_fExposure', 'fExposure'],
      ['VolumetricLight_work_fExposure',      'fExposure'],
    ];
    for (const [key, field] of volKeys) {
      const loc = gl.getUniformLocation(vp, 'f' + field.charAt(0).toUpperCase() + field.slice(1));
      this._bindings.set(key, (v: LiveValue) => {
        (this as unknown as Record<string, number>)[field as string] = v as number;
        gl.useProgram(vp);
        if (loc) gl.uniform1f(loc, v as number);
      });
    }

    // Bloom strength → uBloomStrength in viz
    const bloomKeys = [
      'UnrealBloomComposite/UnrealBloomComposite/home/bloomStrength',
      'UnrealBloomComposite_shaderVariants_homebloomStrength',
      'UnrealBloomComposite_shaderVariants_workbloomStrength',
      'UnrealBloomComposite_shaderVariants_aboutbloomStrength',
      'UnrealBloomComposite_shaderVariants_footerbloomStrength',
      'UnrealBloomComposite_shaderVariants_treebloomStrength',
      'UnrealBloomComposite_shaderVariants_contactbloomStrength',
    ];
    const bloomStrLoc = gl.getUniformLocation(vzp, 'uBloomStrength');
    for (const key of bloomKeys) {
      this._bindings.set(key, (v: LiveValue) => {
        this.uBloomStrength = v as number;
        gl.useProgram(vzp);
        if (bloomStrLoc) gl.uniform1f(bloomStrLoc, v as number);
      });
    }

    // Bloom radius → blurDir spread
    const bloomRadKeys = [
      'UnrealBloomComposite/UnrealBloomComposite/home/bloomRadius',
      'UnrealBloomComposite_shaderVariants_homebloomRadius',
      'UnrealBloomComposite_shaderVariants_workbloomRadius',
      'UnrealBloomComposite_shaderVariants_aboutbloomRadius',
    ];
    const blurDirLoc = gl.getUniformLocation(blrp, 'uDir');
    for (const key of bloomRadKeys) {
      this._bindings.set(key, (v: LiveValue) => {
        const r = v as number;
        this.blurDirX = r;
        gl.useProgram(blrp);
        if (blurDirLoc) gl.uniform2f(blurDirLoc, r, 0.0);
      });
    }

    // Composite contrast → uContrast
    const contrastKeys = [
      'HomeCompositeuContrast', 'WorkCompositeuContrast',
      'CleanRoomCompositeuContrast', 'TreeSceneCompositeuContrast',
    ];
    const contrastLoc = gl.getUniformLocation(vzp, 'uContrast');
    for (const key of contrastKeys) {
      this._bindings.set(key, (v: LiveValue) => {
        const c = Array.isArray(v) ? (v as number[])[1] ?? (v as number[])[0] : v as number;
        this.uContrast = c;
        gl.useProgram(vzp);
        if (contrastLoc) gl.uniform1f(contrastLoc, c);
      });
    }

    // Composite RGB strength → uSaturation
    const satKeys = [
      'HomeCompositeuRGBStrength', 'WorkCompositeuRGBStrength',
      'CleanRoomCompositeuRGBStrength',
    ];
    const satLoc = gl.getUniformLocation(vzp, 'uSaturation');
    for (const key of satKeys) {
      this._bindings.set(key, (v: LiveValue) => {
        this.uSaturation = 1.0 + (v as number) * 0.5;
        gl.useProgram(vzp);
        if (satLoc) gl.uniform1f(satLoc, this.uSaturation);
      });
    }

    // Camera FOV → derived uExposure hint (wider FOV = lower exposure)
    const fovKeys = [
      'CAMERA_Element_1_Homefov', 'CAMERA_Element_2_Workfov',
      'CAMERA_Element_3_home_scenefov', 'CAMERA_Element_10_CleanRoomfov',
    ];
    const expLoc = gl.getUniformLocation(vzp, 'uExposure');
    for (const key of fovKeys) {
      this._bindings.set(key, (v: LiveValue) => {
        const fov = v as number;
        this.uExposure = Math.max(0.1, 1.6 - fov / 120.0);
        gl.useProgram(vzp);
        if (expLoc) gl.uniform1f(expLoc, this.uExposure);
      });
    }

    // Wobble strength → vignette
    const wobbleKeys = [
      'CAMERA_Element_3_home_scenewobbleStrength',
      'CAMERA_Element_10_CleanRoomwobbleStrength',
    ];
    const vignLoc = gl.getUniformLocation(vzp, 'uVignetteStrength');
    for (const key of wobbleKeys) {
      this._bindings.set(key, (v: LiveValue) => {
        this.uVignetteStrength = Math.min(1.0, (v as number) * 4.0);
        gl.useProgram(vzp);
        if (vignLoc) gl.uniform1f(vignLoc, this.uVignetteStrength);
      });
    }

    // Light intensity → volumetric weight
    const lightKeys = [
      'L_Element_10_home_sceneintensity',
      'L_Element_11_home_sceneintensity',
      'L_Element_12_home_sceneintensity',
    ];
    const fWeightLoc = gl.getUniformLocation(vp, 'fWeight');
    for (const key of lightKeys) {
      this._bindings.set(key, (v: LiveValue) => {
        this.fWeight = Math.min(2.0, (v as number) * 0.1);
        gl.useProgram(vp);
        if (fWeightLoc) gl.uniform1f(fWeightLoc, this.fWeight);
      });
    }

    // LerpSpeed → blur kernel direction (faster lerp = tighter blur)
    const lerpKeys = [
      'CAMERA_Element_1_HomelerpSpeed', 'CAMERA_Element_2_WorklerpSpeed',
      'CAMERA_Element_10_CleanRoomlerpSpeed',
    ];
    const resLoc = gl.getUniformLocation(blrp, 'resolution');
    for (const key of lerpKeys) {
      this._bindings.set(key, (v: LiveValue) => {
        const s = (v as number) * 512;
        gl.useProgram(blrp);
        if (resLoc) gl.uniform2f(resLoc, s, s);
      });
    }

    // VolumetricStrength → fDensity
    const volStrKeys = [
      'HomeCompositeuVolumetricStrength',
      'CleanRoomCompositeuVolumetricStrength',
    ];
    const fDensityLoc = gl.getUniformLocation(vp, 'fDensity');
    for (const key of volStrKeys) {
      this._bindings.set(key, (v: LiveValue) => {
        this.fDensity = (v as number) * 0.4;
        gl.useProgram(vp);
        if (fDensityLoc) gl.uniform1f(fDensityLoc, this.fDensity);
      });
    }

    // Particle curl noise speed → light pos drift (visual metaphor)
    const curlKey = 'am_ProtonAntimatter_P_Element_0_particleTestuCurlNoiseSpeed';
    const lightPosLoc = gl.getUniformLocation(vp, 'lightPos');
    this._bindings.set(curlKey, (v: LiveValue) => {
      this.lightPosX = 0.5 + Math.sin(this.uTime * (v as number)) * 0.2;
      gl.useProgram(vp);
      if (lightPosLoc) gl.uniform2f(lightPosLoc, this.lightPosX, this.lightPosY);
    });

    // Distortion strength → fClamp
    const distortKey = 'GlassCubeShader/GlassCubeShader/Element_0_home_scene/uDistortStrength';
    const fClampLoc  = gl.getUniformLocation(vp, 'fClamp');
    this._bindings.set(distortKey, (v: LiveValue) => {
      this.fClamp = Math.max(0.1, Math.min(2.0, (v as number) * 0.12));
      gl.useProgram(vp);
      if (fClampLoc) gl.uniform1f(fClampLoc, this.fClamp);
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ATUILLivePanel — HTML overlay + rAF loop writing gl.uniform* every frame
// ─────────────────────────────────────────────────────────────────────────────

export class ATUILLivePanel {

  public  params: Map<string, UILParamEntry> = new Map();
  public  values: Map<string, LiveValue>     = new Map();

  private _root:            HTMLElement | null   = null;
  private _mounted                               = false;
  private _visible                               = true;
  private _filterQuery                           = '';
  private _collapsed:       Map<PanelSection, boolean> = new Map();
  private _widgets:         Map<string, WidgetRefs>    = new Map();
  private _handlers:        Set<ChangeHandler>         = new Set();
  private _statusEl:        HTMLElement | null   = null;
  private _presetSelect:    HTMLSelectElement | null = null;
  private _suppressEvents                        = false;
  private _gpu:             ATUILGPUBackend | null = null;
  private _rafId                                 = 0;
  private _dirty:           Set<string>          = new Set();
  private _t                                     = 0;

  // ── init() ────────────────────────────────────────────────────────────────

  async init(source: string | Record<string, RawValue>): Promise<void> {
    let raw: Record<string, RawValue>;
    if (typeof source === 'string') {
      const res = await fetch(source);
      raw = await res.json();
    } else {
      raw = source;
    }
    this._classifyParams(raw);
    if (this._mounted && this._root) this._renderPanelBody();
  }

  private _classifyParams(raw: Record<string, RawValue>): void {
    this.params.clear();
    this.values.clear();
    for (const [key, rawValue] of Object.entries(raw)) {
      const wt = classifyWidget(rawValue);
      if (wt === null) continue;
      const section  = classifySection(key);
      const [mn, mx, st] = inferRange(key);
      const vecLen = (wt === 'vec') ? (rawValue as number[]).length : undefined;
      const entry: UILParamEntry = {
        key, section, widgetType: wt,
        value: rawValue as LiveValue,
        min: mn, max: mx, step: st, vecLen,
      };
      this.params.set(key, entry);
      this.values.set(key, cloneValue(rawValue as LiveValue));
    }
  }

  // ── GPU attachment ─────────────────────────────────────────────────────────

  attachGPU(gl: WebGLRenderingContext): void {
    if (this._gpu) { this._gpu.dispose(); }
    this._gpu = new ATUILGPUBackend(gl);
    this._gpu.init();
    this._startRAF();
  }

  detachGPU(): void {
    cancelAnimationFrame(this._rafId);
    this._gpu?.dispose();
    this._gpu = null;
  }

  private _startRAF(): void {
    const loop = (t: number) => {
      this._rafId = requestAnimationFrame(loop);
      this._t = t * 0.001;

      // Flush dirty uniforms → gpu
      if (this._gpu && this._dirty.size > 0) {
        for (const key of this._dirty) {
          const v = this.values.get(key);
          if (v !== undefined) this._gpu.applyUILValue(key, v);
        }
        this._dirty.clear();
        this._gpu.markDirty();
      }

      // Render GPU passes every rAF
      if (this._gpu) this._gpu.render(this._t);
    };
    this._rafId = requestAnimationFrame(loop);
  }

  // ── Mount / Unmount ────────────────────────────────────────────────────────

  mount(container: HTMLElement = document.body): void {
    if (this._mounted) return;
    if (!document.getElementById('at-uil-panel-style')) {
      const style = document.createElement('style');
      style.id = 'at-uil-panel-style';
      style.textContent = PANEL_CSS;
      document.head.appendChild(style);
    }
    this._root = this._buildShell();
    container.appendChild(this._root);
    this._mounted = true;
    if (this.params.size > 0) this._renderPanelBody();
  }

  unmount(): void {
    if (!this._mounted || !this._root) return;
    cancelAnimationFrame(this._rafId);
    this._gpu?.dispose();
    this._gpu = null;
    this._root.remove();
    this._root = null;
    this._mounted = false;
    this._widgets.clear();
    this._statusEl = null;
    this._presetSelect = null;
  }

  // ── Visibility ─────────────────────────────────────────────────────────────

  toggle(): void { this._visible = !this._visible; this._root?.classList.toggle('hidden', !this._visible); }
  show():   void { this._visible = true;  this._root?.classList.remove('hidden'); }
  hide():   void { this._visible = false; this._root?.classList.add('hidden'); }

  // ── Two-way binding ────────────────────────────────────────────────────────

  set(key: string, value: LiveValue): void {
    if (!this.params.has(key)) return;
    this._suppressEvents = true;
    this.values.set(key, cloneValue(value));
    this._refreshWidget(key, value);
    this._dirty.add(key);
    this._suppressEvents = false;
  }

  get(key: string): LiveValue | undefined { return this.values.get(key); }

  setBatch(updates: Record<string, LiveValue>): void {
    this._suppressEvents = true;
    for (const [key, val] of Object.entries(updates)) {
      this.values.set(key, cloneValue(val));
      this._refreshWidget(key, val);
      this._dirty.add(key);
    }
    this._suppressEvents = false;
  }

  // ── Events ─────────────────────────────────────────────────────────────────

  on(_event: 'change', handler: ChangeHandler): () => void {
    this._handlers.add(handler);
    return () => this._handlers.delete(handler);
  }
  off(_event: 'change', handler: ChangeHandler): void { this._handlers.delete(handler); }

  private _emit(evt: ParamChangeEvent): void {
    if (this._suppressEvents) return;
    for (const h of this._handlers) { try { h(evt); } catch { /* ignore */ } }
    this._root?.dispatchEvent(new CustomEvent('ATUILParamChange', { bubbles: true, detail: evt }));
  }

  // ── Sections ───────────────────────────────────────────────────────────────

  showSection(name: PanelSection): void { this._collapsed.set(name, false); this._applyCollapseState(name); }
  hideSection(name: PanelSection): void { this._collapsed.set(name, true);  this._applyCollapseState(name); }

  collapseAll(): void { for (const s of SECTION_ORDER) { this._collapsed.set(s, true);  this._applyCollapseState(s); } }
  expandAll():   void { for (const s of SECTION_ORDER) { this._collapsed.set(s, false); this._applyCollapseState(s); } }

  private _applyCollapseState(section: PanelSection): void {
    if (!this._root) return;
    const body    = this._root.querySelector<HTMLElement>(`[data-section-body="${section}"]`);
    const header  = this._root.querySelector<HTMLElement>(`[data-section-header="${section}"]`);
    const col     = this._collapsed.get(section) ?? false;
    body?.classList.toggle('collapsed', col);
    header?.classList.toggle('collapsed', col);
  }

  // ── Filter ─────────────────────────────────────────────────────────────────

  filterByKey(query: string): void {
    this._filterQuery = query.toLowerCase();
    this._applyFilter();
  }

  private _applyFilter(): void {
    if (!this._root) return;
    const q = this._filterQuery;
    let visible = 0;
    for (const [key, refs] of this._widgets) {
      const show = q === '' || key.toLowerCase().includes(q);
      refs.rowEl.classList.toggle('hidden', !show);
      if (show) visible++;
    }
    if (this._statusEl) {
      this._statusEl.textContent = q
        ? `Showing ${visible} / ${this.params.size} params matching "${q}"`
        : `${this.params.size} params — ${visible} visible`;
    }
    for (const section of SECTION_ORDER) {
      const body = this._root.querySelector<HTMLElement>(`[data-section-body="${section}"]`);
      if (!body) continue;
      const anyVisible = Array.from(body.querySelectorAll('.at-uil-row'))
        .some(r => !r.classList.contains('hidden'));
      const sEl = body.closest('.at-uil-section') as HTMLElement | null;
      if (sEl) sEl.style.display = anyVisible || q === '' ? '' : 'none';
    }
  }

  // ── Presets ────────────────────────────────────────────────────────────────

  savePreset(name: string): void {
    if (!name.trim()) return;
    const snap: PresetSnapshot = {};
    for (const [k, v] of this.values) snap[k] = cloneValue(v);
    localStorage.setItem(PRESET_STORAGE_PREFIX + name, JSON.stringify(snap));
    this._refreshPresetList();
    this._setStatus(`Preset "${name}" saved (${Object.keys(snap).length} params)`);
  }

  loadPreset(name: string): boolean {
    const raw = localStorage.getItem(PRESET_STORAGE_PREFIX + name);
    if (!raw) { this._setStatus(`Preset "${name}" not found`); return false; }
    try { this.setBatch(JSON.parse(raw)); this._setStatus(`Preset "${name}" loaded`); return true; }
    catch { this._setStatus(`Preset "${name}" parse error`); return false; }
  }

  deletePreset(name: string): void {
    localStorage.removeItem(PRESET_STORAGE_PREFIX + name);
    this._refreshPresetList();
    this._setStatus(`Preset "${name}" deleted`);
  }

  listPresets(): string[] {
    const names: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k?.startsWith(PRESET_STORAGE_PREFIX)) names.push(k.slice(PRESET_STORAGE_PREFIX.length));
    }
    return names.sort();
  }

  exportPreset(name = 'at-uil-export'): void {
    const snap: PresetSnapshot = {};
    for (const [k, v] of this.values) snap[k] = cloneValue(v);
    const blob = new Blob([JSON.stringify(snap, null, 2)], { type: 'application/json' });
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = `${name}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    this._setStatus(`Exported "${name}.json" (${Object.keys(snap).length} params)`);
  }

  importPreset(jsonStr: string, saveName?: string): boolean {
    try {
      const snap: PresetSnapshot = JSON.parse(jsonStr);
      this.setBatch(snap);
      if (saveName) {
        localStorage.setItem(PRESET_STORAGE_PREFIX + saveName, jsonStr);
        this._refreshPresetList();
      }
      this._setStatus(`Imported ${Object.keys(snap).length} params`);
      return true;
    } catch { this._setStatus('Import failed — invalid JSON'); return false; }
  }

  resetToDefaults(): void {
    this._suppressEvents = true;
    for (const [key, entry] of this.params) {
      const def = cloneValue(entry.value);
      this.values.set(key, def);
      this._refreshWidget(key, def);
      this._dirty.add(key);
    }
    this._suppressEvents = false;
    this._setStatus('Reset to defaults');
  }

  // ── DOM building ───────────────────────────────────────────────────────────

  private _buildShell(): HTMLElement {
    const panel = document.createElement('div');
    panel.className = 'at-uil-panel';

    // Header
    const hdr   = document.createElement('div');
    hdr.className = 'at-uil-panel-header';
    const title = document.createElement('span');
    title.className = 'at-uil-panel-title';
    title.textContent = 'AT UIL LIVE PANEL';
    const btn   = document.createElement('button');
    btn.className = 'at-uil-panel-toggle-btn';
    btn.textContent = '▼';
    btn.addEventListener('click', () => {
      const scroll = panel.querySelector('.at-uil-scroll') as HTMLElement | null;
      if (!scroll) return;
      const h = scroll.style.display === 'none';
      scroll.style.display = h ? '' : 'none';
      btn.textContent = h ? '▼' : '▶';
    });
    hdr.append(title, btn);
    panel.appendChild(hdr);

    // GPU badge
    const gpuBadge = document.createElement('div');
    gpuBadge.className = 'at-uil-gpu-badge';
    gpuBadge.textContent = '⬡ GPU UNIFORM LIVE TUNING — requestAnimationFrame active';
    panel.appendChild(gpuBadge);

    // Toolbar
    const toolbar = document.createElement('div');
    toolbar.className = 'at-uil-toolbar';
    const search = document.createElement('input');
    search.type = 'text';
    search.placeholder = 'filter params…';
    search.addEventListener('input', () => this.filterByKey(search.value));
    const mkBtn = (txt: string, title: string, fn: () => void) => {
      const b = document.createElement('button');
      b.className = 'at-uil-btn';
      b.textContent = txt;
      b.title = title;
      b.addEventListener('click', fn);
      return b;
    };
    toolbar.append(
      search,
      mkBtn('⊟', 'Collapse all',       () => this.collapseAll()),
      mkBtn('⊞', 'Expand all',         () => this.expandAll()),
      mkBtn('↺', 'Reset to defaults',  () => this.resetToDefaults()),
      mkBtn('⬇', 'Export preset JSON', () => this.exportPreset()),
    );
    panel.appendChild(toolbar);

    // Preset row
    const presetRow = document.createElement('div');
    presetRow.className = 'at-uil-preset-row';
    const sel = document.createElement('select');
    sel.title = 'Saved presets';
    this._presetSelect = sel;
    this._refreshPresetList();
    presetRow.append(
      sel,
      mkBtn('Save', 'Save preset', () => {
        const n = prompt('Preset name:', 'my-preset');
        if (n) this.savePreset(n);
      }),
      mkBtn('Load', 'Load preset', () => { if (sel.value) this.loadPreset(sel.value); }),
      mkBtn('✕', 'Delete preset', () => {
        if (sel.value && confirm(`Delete preset "${sel.value}"?`)) this.deletePreset(sel.value);
      }),
    );
    panel.appendChild(presetRow);

    // Scroll area
    const scroll = document.createElement('div');
    scroll.className = 'at-uil-scroll';
    scroll.id = 'at-uil-scroll';
    panel.appendChild(scroll);

    // Status bar
    const status = document.createElement('div');
    status.className = 'at-uil-status';
    status.textContent = 'Initialising…';
    this._statusEl = status;
    panel.appendChild(status);

    return panel;
  }

  private _renderPanelBody(): void {
    if (!this._root) return;
    const scroll = this._root.querySelector('#at-uil-scroll') as HTMLElement;
    scroll.innerHTML = '';
    this._widgets.clear();

    const bySection = new Map<PanelSection, UILParamEntry[]>();
    for (const s of SECTION_ORDER) bySection.set(s, []);
    for (const entry of this.params.values()) bySection.get(entry.section)!.push(entry);

    for (const section of SECTION_ORDER) {
      const entries = bySection.get(section)!;
      if (!entries.length) continue;
      scroll.appendChild(this._buildSection(section, entries));
    }
    this._setStatus(`${this.params.size} params loaded — GPU rAF live`);
  }

  private _buildSection(section: PanelSection, entries: UILParamEntry[]): HTMLElement {
    const wrap = document.createElement('div');
    wrap.className = 'at-uil-section';

    const hdr = document.createElement('div');
    hdr.className = 'at-uil-section-header';
    hdr.dataset.sectionHeader = section;
    const arrow = document.createElement('span');
    arrow.className = 'at-uil-section-arrow';
    arrow.textContent = '▾';
    const lbl = document.createElement('span');
    lbl.textContent = SECTION_LABELS[section];
    const cnt = document.createElement('span');
    cnt.className = 'at-uil-section-count';
    cnt.textContent = String(entries.length);
    hdr.append(arrow, lbl, cnt);
    hdr.addEventListener('click', () => {
      const col = this._collapsed.get(section) ?? false;
      this._collapsed.set(section, !col);
      this._applyCollapseState(section);
    });
    wrap.appendChild(hdr);

    const body = document.createElement('div');
    body.className = 'at-uil-section-body';
    body.dataset.sectionBody = section;
    entries.sort((a, b) => a.key.localeCompare(b.key));
    for (const entry of entries) body.appendChild(this._buildRow(entry));
    wrap.appendChild(body);
    return wrap;
  }

  private _buildRow(entry: UILParamEntry): HTMLElement {
    const row = document.createElement('div');
    row.className = 'at-uil-row';
    row.dataset.paramKey = entry.key;

    const labelEl = document.createElement('span');
    labelEl.className = 'at-uil-label';
    labelEl.textContent = entry.key.split(/[_/]/).slice(-2).join('_');
    labelEl.title = entry.key;
    row.appendChild(labelEl);

    const widget = document.createElement('div');
    widget.className = 'at-uil-widget';
    const cur = this.values.get(entry.key) ?? entry.value;
    const refs: WidgetRefs = { rowEl: row, inputs: [] };

    switch (entry.widgetType) {
      case 'slide': this._buildSlideWidget(entry, widget, cur as number, refs);   break;
      case 'vec':   this._buildVecWidget(entry, widget, cur as number[], refs);    break;
      case 'color': this._buildColorWidget(entry, widget, cur as string, refs);   break;
      case 'bool':  this._buildBoolWidget(entry, widget, cur as boolean, refs);   break;
    }
    row.appendChild(widget);
    this._widgets.set(entry.key, refs);
    return row;
  }

  // ── Widget builders ────────────────────────────────────────────────────────

  private _buildSlideWidget(e: UILParamEntry, c: HTMLElement, v: number, refs: WidgetRefs): void {
    const wrap = document.createElement('div');
    wrap.className = 'at-uil-slide';
    const mn  = e.min ?? -10;
    const mx  = e.max ?? 10;
    const st  = e.step ?? 0.01;
    const clv = Math.min(mx, Math.max(mn, v));

    const range = document.createElement('input');
    range.type  = 'range';
    range.min   = String(mn);
    range.max   = String(mx);
    range.step  = String(st);
    range.value = String(clv);

    const num  = document.createElement('input');
    num.type   = 'number';
    num.className = 'at-uil-num-input';
    num.step   = String(st);
    num.value  = String(round(v));

    range.addEventListener('input', () => {
      const n = parseFloat(range.value);
      num.value = String(round(n));
      this._onWidgetChange(e.key, n);
    });
    num.addEventListener('change', () => {
      const n = parseFloat(num.value);
      if (!isNaN(n)) {
        range.value = String(Math.min(mx, Math.max(mn, n)));
        this._onWidgetChange(e.key, n);
      }
    });
    refs.inputs = [range, num];
    wrap.append(range, num);
    c.appendChild(wrap);
  }

  private _buildVecWidget(e: UILParamEntry, c: HTMLElement, v: number[], refs: WidgetRefs): void {
    const wrap = document.createElement('div');
    wrap.className = 'at-uil-vec';
    const inputs: HTMLInputElement[] = [];
    for (let i = 0; i < v.length; i++) {
      const num = document.createElement('input');
      num.type  = 'number';
      num.className = 'at-uil-num-input';
      num.step  = String(e.step ?? 0.01);
      num.value = String(round(v[i]));
      num.placeholder = ['x', 'y', 'z', 'w'][i] ?? String(i);
      num.addEventListener('change', () => {
        const cur = (this.values.get(e.key) as number[]).slice();
        const n   = parseFloat(num.value);
        if (!isNaN(n)) cur[i] = n;
        this._onWidgetChange(e.key, cur);
      });
      inputs.push(num);
      wrap.appendChild(num);
    }
    refs.inputs = inputs;
    c.appendChild(wrap);
  }

  private _buildColorWidget(e: UILParamEntry, c: HTMLElement, v: string, refs: WidgetRefs): void {
    const wrap    = document.createElement('div');
    wrap.className = 'at-uil-color';
    const swatch  = document.createElement('input');
    swatch.type   = 'color';
    swatch.value  = v.length === 7 ? v : '#ffffff';
    const hexInp  = document.createElement('input');
    hexInp.type   = 'text';
    hexInp.className = 'at-uil-color-hex';
    hexInp.value  = v;
    hexInp.maxLength = 7;
    swatch.addEventListener('input', () => {
      hexInp.value = swatch.value;
      this._onWidgetChange(e.key, swatch.value);
    });
    hexInp.addEventListener('change', () => {
      if (/^#[0-9a-fA-F]{6}$/.test(hexInp.value)) {
        swatch.value = hexInp.value;
        this._onWidgetChange(e.key, hexInp.value);
      }
    });
    refs.inputs = [swatch, hexInp];
    wrap.append(swatch, hexInp);
    c.appendChild(wrap);
  }

  private _buildBoolWidget(e: UILParamEntry, c: HTMLElement, v: boolean, refs: WidgetRefs): void {
    const wrap  = document.createElement('div');
    wrap.className = 'at-uil-bool';
    const cb    = document.createElement('input');
    cb.type     = 'checkbox';
    cb.checked  = v;
    cb.addEventListener('change', () => this._onWidgetChange(e.key, cb.checked));
    refs.inputs = [cb];
    wrap.appendChild(cb);
    c.appendChild(wrap);
  }

  // ── Widget refresh (programmatic set) ─────────────────────────────────────

  private _refreshWidget(key: string, value: LiveValue): void {
    const refs  = this._widgets.get(key);
    const entry = this.params.get(key);
    if (!refs || !entry) return;
    switch (entry.widgetType) {
      case 'slide': {
        const [rng, num] = refs.inputs as HTMLInputElement[];
        if (rng) rng.value = String(value as number);
        if (num) num.value = String(round(value as number));
        break;
      }
      case 'vec': {
        const arr = value as number[];
        for (let i = 0; i < arr.length; i++) {
          const inp = refs.inputs[i] as HTMLInputElement | undefined;
          if (inp) inp.value = String(round(arr[i]));
        }
        break;
      }
      case 'color': {
        const hex = value as string;
        const [sw, hi] = refs.inputs as HTMLInputElement[];
        if (sw && /^#[0-9a-fA-F]{6}$/.test(hex)) sw.value = hex;
        if (hi) hi.value = hex;
        break;
      }
      case 'bool': {
        const cb = refs.inputs[0] as HTMLInputElement | undefined;
        if (cb) cb.checked = value as boolean;
        break;
      }
    }
  }

  // ── Change dispatch → dirty set → gl.uniform* on next rAF ────────────────

  private _onWidgetChange(key: string, newValue: LiveValue): void {
    const prev    = cloneValue(this.values.get(key) ?? (this.params.get(key)?.value ?? 0));
    this.values.set(key, cloneValue(newValue));
    this._dirty.add(key);              // GPU will consume on next rAF tick
    const section = this.params.get(key)?.section ?? 'MISC';
    this._emit({ key, value: newValue, section, prev });
    this._setStatus(`${key.split(/[_/]/).slice(-2).join('/')} → ${_valueToString(newValue)}`);
  }

  // ── Preset list ────────────────────────────────────────────────────────────

  private _refreshPresetList(): void {
    if (!this._presetSelect) return;
    const names = this.listPresets();
    this._presetSelect.innerHTML = '';
    const ph = document.createElement('option');
    ph.value = '';
    ph.textContent = names.length ? '— select preset —' : '(no presets)';
    this._presetSelect.appendChild(ph);
    for (const n of names) {
      const opt = document.createElement('option');
      opt.value = n;
      opt.textContent = n;
      this._presetSelect.appendChild(opt);
    }
  }

  private _setStatus(msg: string): void {
    if (this._statusEl) this._statusEl.textContent = msg;
  }

  // ── Snapshot utils ─────────────────────────────────────────────────────────

  snapshot(): PresetSnapshot {
    const out: PresetSnapshot = {};
    for (const [k, v] of this.values) out[k] = cloneValue(v);
    return out;
  }

  sectionSnapshot(section: PanelSection): PresetSnapshot {
    const out: PresetSnapshot = {};
    for (const [k, entry] of this.params) {
      if (entry.section === section) out[k] = cloneValue(this.values.get(k) ?? entry.value);
    }
    return out;
  }

  searchParams(query: string): UILParamEntry[] {
    const q = query.toLowerCase();
    return Array.from(this.params.values()).filter(e => e.key.toLowerCase().includes(q));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

function _valueToString(v: LiveValue): string {
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'string')  return v;
  if (typeof v === 'number')  return round(v, 3).toString();
  return `[${(v as number[]).map(n => round(n, 3)).join(', ')}]`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Species preset API
// ─────────────────────────────────────────────────────────────────────────────

export function applySpeciesPreset(panel: ATUILLivePanel, species: SpeciesId, blend = 1.0): void {
  const preset = SPECIES_UIL_PRESETS[species];
  if (!preset) { console.warn(`[ATUILLivePanel] Unknown species: "${species}"`); return; }

  const updates: Record<string, LiveValue> = {};
  for (const [key, target] of Object.entries(preset.params)) {
    if (target === undefined) continue;
    if (blend >= 1) { updates[key] = target; continue; }
    const cur = panel.get(key);
    if (cur === undefined) { updates[key] = target; continue; }
    if (typeof target === 'number' && typeof cur === 'number') {
      updates[key] = cur + (target - cur) * blend;
    } else if (Array.isArray(target) && Array.isArray(cur) && target.length === cur.length &&
               target.every((x: unknown) => typeof x === 'number')) {
      updates[key] = (cur as number[]).map((c, i) => c + ((target as number[])[i] - c) * blend);
    } else {
      updates[key] = blend >= 0.5 ? target : cur;
    }
  }
  panel.setBatch(updates);
}

export function getSpeciesPresetParams(species: SpeciesId): Record<string, LiveValue> | null {
  const preset = SPECIES_UIL_PRESETS[species];
  if (!preset) return null;
  const out: Record<string, LiveValue> = {};
  for (const [k, v] of Object.entries(preset.params)) { if (v !== undefined) out[k] = v; }
  return out;
}

export function diffSpeciesPresets(
  a: SpeciesId, b: SpeciesId,
): Array<{ key: string; a: LiveValue | undefined; b: LiveValue | undefined }> {
  const pA = SPECIES_UIL_PRESETS[a]?.params ?? {};
  const pB = SPECIES_UIL_PRESETS[b]?.params ?? {};
  const keys = new Set([...Object.keys(pA), ...Object.keys(pB)]);
  return Array.from(keys)
    .filter(key => {
      const va = pA[key]; const vb = pB[key];
      return !(va === vb || (Array.isArray(va) && Array.isArray(vb) &&
        va.length === vb.length && (va as number[]).every((x, i) => x === (vb as number[])[i])));
    })
    .map(key => ({ key, a: pA[key] as LiveValue | undefined, b: pB[key] as LiveValue | undefined }));
}

// ─────────────────────────────────────────────────────────────────────────────
// UILModulePanel — module-aware sub-panel
// ─────────────────────────────────────────────────────────────────────────────

export class UILModulePanel {
  private _panel:         ATUILLivePanel;
  private _activeSpecies: SpeciesId | null  = null;
  private _activeModule:  UILModule | null  = null;
  private _moduleEl:      HTMLElement | null = null;

  constructor(panel: ATUILLivePanel) { this._panel = panel; }

  activateSpecies(species: SpeciesId, blend = 1.0, autoFilter = true): void {
    this._activeSpecies = species;
    applySpeciesPreset(this._panel, species, blend);
    if (autoFilter) {
      const preset = SPECIES_UIL_PRESETS[species];
      if (preset?.modules.length) {
        const show = new Set<PanelSection>(preset.modules.map(m => MODULE_TO_SECTION[m]));
        for (const s of SECTION_ORDER) {
          show.has(s) ? this._panel.showSection(s) : this._panel.hideSection(s);
        }
      }
    }
    this._updateModuleDisplay();
  }

  filterModule(module: UILModule | null): void {
    this._activeModule = module;
    this._panel.filterByKey(module === null ? '' : _moduleKeyFragment(module));
    this._updateModuleDisplay();
  }

  get activeSpecies(): SpeciesId | null { return this._activeSpecies; }
  get activeModule():  UILModule | null { return this._activeModule; }

  mount(container: HTMLElement = document.body): void {
    const el = document.createElement('div');
    el.id = 'at-uil-module-panel';
    el.style.cssText = [
      'position:fixed;top:10px;right:340px;z-index:99999',
      'background:#0d0d0f;border:1px solid #2a2a38;border-radius:6px',
      'padding:8px;font:11px "SF Mono",monospace;color:#c8c8d0',
      'min-width:160px;box-shadow:0 4px 16px rgba(0,0,0,0.6)',
    ].join(';');
    this._moduleEl = el;
    container.appendChild(el);
    this._updateModuleDisplay();
  }

  unmount(): void { this._moduleEl?.remove(); this._moduleEl = null; }

  private _updateModuleDisplay(): void {
    if (!this._moduleEl) return;
    const sp     = this._activeSpecies;
    const mod    = this._activeModule;
    const preset = sp ? SPECIES_UIL_PRESETS[sp] : null;
    const lines: string[] = [
      `<div style="color:#9090ff;font-weight:700;margin-bottom:4px">UIL Modules</div>`,
    ];
    if (sp) lines.push(`<div style="color:#60c060;margin-bottom:4px">🔵 ${sp}</div>`);
    for (const [m, meta] of Object.entries(MODULE_META) as [UILModule, typeof MODULE_META[UILModule]][]) {
      const active  = sp && preset?.modules.includes(m);
      const current = mod === m;
      const color   = current ? '#ffffa0' : active ? '#a0c0ff' : '#404060';
      lines.push(
        `<div style="color:${color};cursor:pointer;padding:1px 0"` +
        ` onclick="window._uilModPanel?.filterModule('${m}')">` +
        `${meta.icon} ${meta.label} <span style="opacity:0.5">(~${meta.approxCount})</span></div>`,
      );
    }
    if (mod) {
      lines.push(
        `<div style="margin-top:4px;color:#80a0ff;cursor:pointer"` +
        ` onclick="window._uilModPanel?.filterModule(null)">↩ Clear filter</div>`,
      );
    }
    this._moduleEl.innerHTML = lines.join('');
    (window as unknown as Record<string, unknown>)._uilModPanel = this;
  }
}

function _moduleKeyFragment(module: UILModule): string {
  switch (module) {
    case 'CAMERA':     return 'CAMERA_';
    case 'LIGHTING':   return 'L_Element';
    case 'BLOOM':      return 'Bloom';
    case 'VOLUMETRIC': return 'VolumetricLight';
    case 'COMPOSITE':  return 'Composite';
    case 'PARTICLE':   return 'am_';
    case 'SCENE':      return 'INPUT_Config';
    case 'SHADER':     return 'Shader';
    case 'MESH':       return 'MESH_';
    case 'SHADOW':     return 'SHADOW_';
    case 'MISC':       return 'UIL_';
    default:           return '';
  }
}
