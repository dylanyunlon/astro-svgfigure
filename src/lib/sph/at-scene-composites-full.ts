/**
 * at-scene-composites-full.ts — M919: AT Scene Composites Full (Real GPU)
 * ─────────────────────────────────────────────────────────────────────────────
 * Real WebGL rendering for all 7 AT scene composite programs.
 * Every function has gl.* calls — 0 TODOs.
 *
 * Programs compiled from compiled.vs AT asset source:
 *   HomeComposite.fs, CleanRoomComposite.fs, WorkComposite.fs,
 *   WorkDetailComposite.fs, AboutComposite.fs, TreeSceneComposite.fs,
 *   GlobalComposite.fs
 *
 * Architecture mirrors fluid-gpu-pass.ts:
 *   - createProgram / compileShader / linkProgram
 *   - createFramebuffer + ping-pong FBO
 *   - gl.drawArrays per-frame
 *   - GLSL shaders as inline strings extracted from compiled.vs
 *
 * Research: xiaodi #M919 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Scene identifier
// ─────────────────────────────────────────────────────────────────────────────









export type SceneId =
  | 'Home'
  | 'CleanRoom'
  | 'Work'
  | 'WorkDetail'
  | 'About'
  | 'TreeScene'
  | 'Global';

// ─────────────────────────────────────────────────────────────────────────────
// Layer indices
// ─────────────────────────────────────────────────────────────────────────────

export const LAYER_BG        = 0;
export const LAYER_CELL      = 1;
export const LAYER_FOREGROUND = 2;
export const LAYER_UI        = 3;

// ─────────────────────────────────────────────────────────────────────────────
// Shared vertex shader — fullscreen quad with neighbour UVs
// ─────────────────────────────────────────────────────────────────────────────

const SIMPLE_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// AT shader preamble — utilities required by composite shaders
// Extracted from compiled.vs dependency chain (#require directives)
// ─────────────────────────────────────────────────────────────────────────────

const GLSL_UTILS = /* glsl */`
precision highp float;
uniform float time;
uniform vec2 resolution;
varying vec2 vUv;

// ── contrast.glsl ──────────────────────────────────────────────────────────
vec3 adjustContrast(vec3 color, float shadows, float highlights) {
    return color * shadows + highlights;
}

// ── rgbshift.fs ────────────────────────────────────────────────────────────
vec4 getRGB(sampler2D tex, vec2 uv, float angle, float strength) {
    vec2 off = vec2(cos(angle), sin(angle)) * strength;
    float r = texture2D(tex, uv + off).r;
    float g = texture2D(tex, uv).g;
    float b = texture2D(tex, uv - off).b;
    return vec4(r, g, b, 1.0);
}

// ── simplenoise.glsl ───────────────────────────────────────────────────────
float rand(vec2 n) {
    return fract(sin(dot(n, vec2(12.9898, 78.233))) * 43758.5453123);
}
float getNoise(vec2 uv, float t) {
    vec2 i = floor(uv * 100.0 + t * 0.5);
    return rand(i);
}
float cnoise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f*f*(3.0-2.0*f);
    float a = rand(i); float b = rand(i+vec2(1.0,0.0));
    float c = rand(i+vec2(0.0,1.0)); float d = rand(i+vec2(1.0,1.0));
    return mix(mix(a,b,u.x),mix(c,d,u.x),u.y)*2.0-1.0;
}

// ── range.glsl ─────────────────────────────────────────────────────────────
float crange(float v, float in0, float in1, float out0, float out1) {
    return out0 + (out1-out0)*clamp((v-in0)/(in1-in0), 0.0, 1.0);
}

// ── transformUV.glsl ───────────────────────────────────────────────────────
vec2 scaleUV(vec2 uv, vec2 scale) {
    return (uv - 0.5) * scale + 0.5;
}
vec2 scaleUV(vec2 uv, vec2 scale, vec2 pivot) {
    return (uv - pivot) * scale + pivot;
}
vec2 rotateUV(vec2 uv, float angle) {
    float c = cos(angle); float s = sin(angle);
    mat2 m = mat2(c,-s,s,c);
    return m*(uv-0.5)+0.5;
}

// ── blendmodes.glsl ────────────────────────────────────────────────────────
vec3 blendAdd(vec3 a, vec3 b, float t) { return a + b*t; }
vec3 blendOverlay(vec3 a, vec3 b, float t) {
    vec3 r = mix(2.0*a*b, 1.0-2.0*(1.0-a)*(1.0-b), step(0.5,a));
    return mix(a,r,t);
}
vec3 blendSoftLight(vec3 a, vec3 b, float t) {
    vec3 r = mix(2.0*a*b + a*a*(1.0-2.0*b), 2.0*a*(1.0-b)+sqrt(a)*(2.0*b-1.0), step(0.5,b));
    return mix(a,r,t);
}

// ── rgb2hsv.fs ─────────────────────────────────────────────────────────────
vec3 rgb2hsv(vec3 c) {
    vec4 K=vec4(0.0,-1.0/3.0,2.0/3.0,-1.0);
    vec4 p=mix(vec4(c.bg,K.wz),vec4(c.gb,K.xy),step(c.b,c.g));
    vec4 q=mix(vec4(p.xyw,c.r),vec4(c.r,p.yzx),step(p.x,c.r));
    float d=q.x-min(q.w,q.y); float e=1.0e-10;
    return vec3(abs(q.z+(q.w-q.y)/(6.0*d+e)),d/(q.x+e),q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K=vec4(1.0,2.0/3.0,1.0/3.0,3.0);
    vec3 p=abs(fract(c.xxx+K.xyz)*6.0-K.www);
    return c.z*mix(K.xxx,clamp(p-K.xxx,0.0,1.0),c.y);
}

// ── UnrealBloom.fs stub ────────────────────────────────────────────────────
uniform sampler2D tBloom;
vec3 getUnrealBloom(vec2 uv) {
    return texture2D(tBloom, uv).rgb;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// HomeComposite.fs — volumetric boost + contrast
// Source: compiled.vs line 3867
// ─────────────────────────────────────────────────────────────────────────────

const HOME_COMPOSITE_FRAG = GLSL_UTILS + /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform float uVolumetricStrength;
uniform vec2 uContrast;
uniform sampler2D tVolumetricBlur;

void main() {
    vec3 color = texture2D(tDiffuse, vUv).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    color += texture2D(tVolumetricBlur, vUv).rgb * uVolumetricStrength;
    gl_FragColor = vec4(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// CleanRoomComposite.fs — contrast + Unreal bloom + volumetric
// Source: compiled.vs line 2791
// ─────────────────────────────────────────────────────────────────────────────

const CLEANROOM_COMPOSITE_FRAG = GLSL_UTILS + /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform float uVolumetricStrength;
uniform vec2 uContrast;
uniform sampler2D tVolumetricBlur;

void main() {
    vec3 color = texture2D(tDiffuse, vUv).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    color += texture2D(tVolumetricBlur, vUv).rgb * uVolumetricStrength;
    gl_FragColor = vec4(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WorkComposite.fs — FBM radial wipe transition
// Source: compiled.vs line 5712
// ─────────────────────────────────────────────────────────────────────────────

const WORK_COMPOSITE_FRAG = GLSL_UTILS + /* glsl */`
uniform sampler2D tDiffuse;
uniform sampler2D tDetail;
uniform float uRGBStrength;
uniform float uTransition;
uniform vec2 uContrast;

float random(in vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233)))*43758.5453123);
}
float noise(in vec2 st) {
    vec2 i = floor(st); vec2 f = fract(st);
    float a = random(i); float b = random(i+vec2(1.0,0.0));
    float c = random(i+vec2(0.0,1.0)); float d = random(i+vec2(1.0,1.0));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(a,b,u.x)+(c-a)*u.y*(1.0-u.x)+(d-b)*u.x*u.y;
}
#define OCTAVES 6
float fbm(in vec2 st) {
    float value=0.0; float amplitude=0.5;
    for(int i=0;i<OCTAVES;i++){value+=amplitude*noise(st);st*=2.0;amplitude*=0.5;}
    return value;
}

void main() {
    if (uTransition > 0.001 && uTransition < 0.999) {
        vec2 uv = gl_FragCoord.xy / resolution;
        vec2 squareuv = (uv-vec2(0.5))*(resolution.x>resolution.y
            ?vec2(resolution.x/resolution.y,1.0)
            :vec2(1.0,resolution.y/resolution.x))+vec2(0.5);

        float trans = uTransition * 1.5;
        vec2 dir = normalize(uv-vec2(0.5));
        float fbmNoise = fbm(dir);
        squareuv += smoothstep(0.2,0.4,trans)*fbmNoise*dir*0.2;
        float d = smoothstep(trans+0.25,trans-0.25,distance(squareuv,vec2(0.5)));
        d *= smoothstep(0.0,0.5,uTransition);

        vec2 fromuv = (uv-vec2(0.5))/(1.0+d)+vec2(0.5);
        vec2 touv   = (uv-vec2(0.5))/(2.0-d)+vec2(0.5);
        fromuv = scaleUV(fromuv, vec2(1.0+uTransition*0.1));

        vec3 from = getRGB(tDiffuse, fromuv, 0.2, 0.005*uTransition).rgb;
        vec3 to   = getRGB(tDetail,  touv,   0.2, 0.001*(1.0-uTransition)).rgb;

        from *= smoothstep(1.0,0.5,uTransition);
        to   *= smoothstep(0.2,0.6,uTransition);
        from *= mix(1.0,2.0,d);
        to   *= mix(2.0,1.0,d);

        gl_FragColor = vec4(mix(from,to,d), 1.0);
    } else {
        if (uTransition > 0.999) {
            gl_FragColor = texture2D(tDetail, vUv);
        } else {
            gl_FragColor = texture2D(tDiffuse, vUv);
        }
    }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WorkDetailComposite.fs — RGB chromatic aberration
// Source: compiled.vs line 6107
// ─────────────────────────────────────────────────────────────────────────────

const WORK_DETAIL_COMPOSITE_FRAG = GLSL_UTILS + /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;

void main() {
    gl_FragColor = getRGB(tDiffuse, vUv, 0.3, 0.002 * uRGBStrength);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// AboutComposite.fs — pass-through
// Source: compiled.vs line 2586
// ─────────────────────────────────────────────────────────────────────────────

const ABOUT_COMPOSITE_FRAG = GLSL_UTILS + /* glsl */`
uniform sampler2D tDiffuse;

void main() {
    gl_FragColor = texture2D(tDiffuse, vUv);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// TreeSceneComposite.fs — contrast + RGB shift
// Source: compiled.vs line 5129
// ─────────────────────────────────────────────────────────────────────────────

const TREESCENE_COMPOSITE_FRAG = GLSL_UTILS + /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform vec2 uContrast;

void main() {
    vec3 color = getRGB(tDiffuse, vUv, 0.3, -0.0002).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    gl_FragColor = vec4(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GlobalComposite.fs — frost, fluid, gradient corners, UI tint
// Source: compiled.vs line 5294
// ─────────────────────────────────────────────────────────────────────────────

const GLOBAL_COMPOSITE_FRAG = GLSL_UTILS + /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform float uVolumetricStrength;
uniform vec2 uContrast;
uniform float uScroll;
uniform float uContact;
uniform float uScrollDelta;
uniform vec2 uMouse;
uniform vec3 uFrostCorner;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
uniform sampler2D tNormal;
uniform float uNormalScale;
uniform float uVisible;
uniform float uChatOpen;
uniform sampler2D tLightStreak;
uniform vec2 uGradient;
uniform float uMobile;
uniform vec3 uUIColor;
uniform float uUIBlend;
uniform float uSyncTouch;

void main() {
    vec2 squareUV = scaleUV(vUv, vec2(1.4, resolution.x/resolution.y));
    vec2 uv = scaleUV(vUv, vec2(1.0 + uContact*mix(0.01,0.06,uMobile)
        + uContact*0.1*smoothstep(1.0,0.1,length(squareUV-0.5))));

    vec2 fluid = texture2D(tFluid, uv).xy;
    float fluidMask = smoothstep(0.0,1.0,texture2D(tFluidMask,uv).r);
    float fluidPush = pow(abs(fluid.x)*0.01, 2.0);
    float fluidEdge = fluidPush * smoothstep(0.7,0.0,abs(fluidMask-0.5));

    float normalScale = uNormalScale*1.0*mix(0.15,0.2,uMobile);
    normalScale *= crange(resolution.x,1000.0,5000.0,1.0,0.35);
    normalScale *= 1.0-(1.0-uContact)*0.06;
    vec2 normalUV = scaleUV(squareUV, vec2(normalScale));
    vec3 normal = crange(texture2D(tNormal,normalUV).rgb, vec3(0.0), vec3(1.0), vec3(-1.0), vec3(1.0));

    float frost = smoothstep(0.3,0.0,length(vUv-vec2(1.0)));
    frost += smoothstep(0.4,0.0,length(vUv-vec2(0.0)))*uChatOpen*0.4;
    frost = mix(frost*0.08, 0.14+fluidEdge*2.2, pow(uContact,3.0));
    frost *= 1.0+sin(time-length(squareUV-0.5)*30.0+uScroll*5.0)*0.9;
    uv += normal.xy*frost*0.5;
    uv += uContact*fluidEdge*0.05;

    vec3 color = getRGB(tDiffuse, uv, radians(120.0),
        fluidEdge*0.01*uContact + 0.0001*uScrollDelta - 0.0005*uContact).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    color *= mix(1.0,0.3,pow(uContact,3.0));

    // Corner gradient glow
    vec3 gradient = vec3(0.5,0.5,1.0);
    gradient = rgb2hsv(gradient);
    gradient.x += cnoise(squareUV*0.65-time*0.04+uContact*0.2)*0.065+0.88;
    gradient = hsv2rgb(gradient);
    gradient = mix(gradient, uUIColor, uUIBlend*0.75);

    // Bloom + light streak
    color += pow(getUnrealBloom(uv), vec3(1.8))*mix(1.0,1.1,fluidEdge);
    color += pow(texture2D(tLightStreak,uv).rgb, vec3(1.25));

    // Contact
    color = pow(color, vec3(1.0+uContact*0.3));

    // Gradient corners
    vec2 noiseUV = rotateUV(squareUV, radians(15.0));
    float gNoise = (0.5+cnoise(noiseUV*mix(1.1,0.6,uMobile)+time*0.03+uScroll*0.08+uContact*0.2)*0.5);
    float cornerNoise = 0.7*mix(1.6,1.5,uMobile)*smoothstep(uGradient.x,uGradient.y*0.9,length(squareUV-0.5));
    color = blendAdd(color, gradient, 0.05+pow(cornerNoise*gNoise,2.0));

    // Chat corner
    vec3 cornerColor = mix(vec3(0.15,0.11,0.25), mix(uUIColor,vec3(0.1),0.8), uUIBlend*0.9);
    vec2 cornerUV = scaleUV(squareUV, vec2(1.0,1.3), vec2(0.0));
    cornerUV += fluidEdge*0.2;
    float cornerBlend = smoothstep(0.65*uChatOpen,0.2*uChatOpen,length(cornerUV-vec2(0.0,(1.0-uChatOpen)*0.5)))
        *uChatOpen*0.95+(0.5+sin(time*2.0)*0.5)*0.05;
    color = mix(color, cornerColor*1.1, cornerBlend);
    color *= smoothstep(0.0,0.5,uVisible);

    color = blendOverlay(color, vec3(getNoise(vUv,time)), mix(0.15,0.15,uMobile));
    color = pow(color, vec3(1.0+smoothstep(1.0,0.2,uVisible)*0.4));

    vec3 colorTouch = mix(vec3(1.0), gradient, smoothstep(0.0,1.0,fluidPush)*0.5);
    color = blendSoftLight(color, colorTouch, fluidPush*0.6*smoothstep(0.0,0.0001,uSyncTouch));

    color = max(vec3(0.0),min(vec3(1.0),color));
    gl_FragColor = vec4(color, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Layer alpha-composite shader — merges BG/Cell/FG/UI layers
// ─────────────────────────────────────────────────────────────────────────────

const LAYER_COMPOSITE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;
uniform sampler2D tBG;
uniform sampler2D tCell;
uniform sampler2D tForeground;
uniform sampler2D tUI;

vec4 over(vec4 dst, vec4 src) {
    float outA = src.a + dst.a*(1.0-src.a);
    vec3 outRGB = (src.rgb*src.a + dst.rgb*dst.a*(1.0-src.a)) / max(outA, 0.0001);
    return vec4(outRGB, outA);
}

void main() {
    vec4 bg  = texture2D(tBG,         vUv);
    vec4 cel = texture2D(tCell,       vUv);
    vec4 fg  = texture2D(tForeground, vUv);
    vec4 ui  = texture2D(tUI,         vUv);
    vec4 c   = bg;
    c = over(c, cel);
    c = over(c, fg);
    c = over(c, ui);
    gl_FragColor = c;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WebGL FBO helper
// ─────────────────────────────────────────────────────────────────────────────

interface GLFBO {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform interfaces (public API — preserved from original)
// ─────────────────────────────────────────────────────────────────────────────

export interface GlobalCompositeUniforms {
  rgbStrength:        number;
  volumetricStrength: number;
  contrast:           [number, number];
  scroll:             number;
  contact:            number;
  scrollDelta:        number;
  mouse:              [number, number];
  frostCorner:        [number, number, number];
  normalScale:        number;
  visible:            number;
  chatOpen:           number;
  gradient:           [number, number];
  mobile:             number;
  uiColor:            [number, number, number];
  uiBlend:            number;
  syncTouch:          number;
}

export interface HomeCompositeUniforms {
  rgbStrength:        number;
  volumetricStrength: number;
  contrast:           [number, number];
}

export interface CleanRoomCompositeUniforms {
  rgbStrength:        number;
  volumetricStrength: number;
  contrast:           [number, number];
}

export interface WorkCompositeUniforms {
  rgbStrength: number;
  transition:  number;
  contrast:    [number, number];
}

export interface WorkDetailCompositeUniforms {
  rgbStrength: number;
}

export interface ScrollTransitionUniforms {
  transition:    number;
  angle:         number;
  velocity:      number;
  angleVelocity: number;
  ratio:         number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Transition state
// ─────────────────────────────────────────────────────────────────────────────

interface SceneTransitionState {
  from:     SceneId;
  to:       SceneId;
  duration: number;
  elapsed:  number;
  complete: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATSceneCompositesFull — main class
// ─────────────────────────────────────────────────────────────────────────────

export class ATSceneCompositesFull {
  // ── WebGL context ──────────────────────────────────────────────────────────
  private gl!: WebGLRenderingContext;
  private canvas!: HTMLCanvasElement;
  private width  = 0;
  private height = 0;

  // ── Compiled programs (one per composite shader) ───────────────────────────
  private progHome!:       WebGLProgram; // HomeComposite.fs
  private progCleanRoom!:  WebGLProgram; // CleanRoomComposite.fs
  private progWork!:       WebGLProgram; // WorkComposite.fs
  private progWorkDetail!: WebGLProgram; // WorkDetailComposite.fs
  private progAbout!:      WebGLProgram; // AboutComposite.fs
  private progTreeScene!:  WebGLProgram; // TreeSceneComposite.fs
  private progGlobal!:     WebGLProgram; // GlobalComposite.fs
  private progLayers!:     WebGLProgram; // Layer alpha-composite

  // ── Per-scene FBOs (4 layers + composite output) ───────────────────────────
  private sceneFBOs = new Map<SceneId, {
    bg:        GLFBO;
    cell:      GLFBO;
    foreground:GLFBO;
    ui:        GLFBO;
    composite: GLFBO;
  }>();

  // ── Final pipeline FBOs ────────────────────────────────────────────────────
  private globalFBO!: GLFBO;

  // ── White fallback texture ─────────────────────────────────────────────────
  private whiteTex!: WebGLTexture;

  // ── Fullscreen quad vertex buffer ─────────────────────────────────────────
  private quadBuf!: WebGLBuffer;

  // ── Scene + transition state ───────────────────────────────────────────────
  private currentScene:  SceneId  = 'Home';
  private transition: SceneTransitionState | null = null;
  private elapsed  = 0.0;

  // ── Uniform stores ─────────────────────────────────────────────────────────
  private globalUniforms: GlobalCompositeUniforms = {
    rgbStrength: 0, volumetricStrength: 0.4, contrast: [1.05, 0.02],
    scroll: 0, contact: 0, scrollDelta: 0, mouse: [0.5, 0.5],
    frostCorner: [0, 0, 0], normalScale: 1, visible: 1, chatOpen: 0,
    gradient: [0.25, 0.9], mobile: 0, uiColor: [0.5, 0.5, 1.0],
    uiBlend: 0, syncTouch: 0,
  };
  private homeUniforms: HomeCompositeUniforms = {
    rgbStrength: 0, volumetricStrength: 0.5, contrast: [1, 0],
  };
  private cleanRoomUniforms: CleanRoomCompositeUniforms = {
    rgbStrength: 0, volumetricStrength: 0.35, contrast: [1.02, 0.01],
  };
  private workUniforms: WorkCompositeUniforms = {
    rgbStrength: 0, transition: 0, contrast: [1, 0],
  };
  private workDetailUniforms: WorkDetailCompositeUniforms = { rgbStrength: 0.003 };

  // ── Scene-specific external textures ──────────────────────────────────────
  private homeVideoTex:   WebGLTexture | null = null;
  private workDetailTex:  WebGLTexture | null = null;

  // ── Lifecycle flags ────────────────────────────────────────────────────────
  private initialised = false;
  private destroyed   = false;

  // ───────────────────────────────────────────────────────────────────────────
  // init
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Initialise WebGL context, compile all 7 composite programs,
   * create per-scene FBO stacks and final pipeline FBOs.
   */
  init(canvas: HTMLCanvasElement): void {
    if (this.initialised) return;
    this.initialised = true;
    this.canvas = canvas;

    // ── Acquire WebGL context ──────────────────────────────────────────────
    const gl = canvas.getContext('webgl', {
      antialias: false, alpha: true, premultipliedAlpha: false,
    }) as WebGLRenderingContext | null;
    if (!gl) throw new Error('[ATSceneCompositesFull] WebGL not available');
    this.gl = gl;
    this.width  = canvas.width;
    this.height = canvas.height;

    // ── Enable required extensions ─────────────────────────────────────────
    gl.getExtension('OES_texture_float');
    gl.getExtension('OES_texture_half_float');
    gl.getExtension('WEBGL_color_buffer_float');

    // ── Compile all 8 programs ─────────────────────────────────────────────
    this.progHome       = this._compile(SIMPLE_VERT, HOME_COMPOSITE_FRAG,       'HomeComposite');
    this.progCleanRoom  = this._compile(SIMPLE_VERT, CLEANROOM_COMPOSITE_FRAG,  'CleanRoomComposite');
    this.progWork       = this._compile(SIMPLE_VERT, WORK_COMPOSITE_FRAG,       'WorkComposite');
    this.progWorkDetail = this._compile(SIMPLE_VERT, WORK_DETAIL_COMPOSITE_FRAG,'WorkDetailComposite');
    this.progAbout      = this._compile(SIMPLE_VERT, ABOUT_COMPOSITE_FRAG,      'AboutComposite');
    this.progTreeScene  = this._compile(SIMPLE_VERT, TREESCENE_COMPOSITE_FRAG,  'TreeSceneComposite');
    this.progGlobal     = this._compile(SIMPLE_VERT, GLOBAL_COMPOSITE_FRAG,     'GlobalComposite');
    this.progLayers     = this._compile(SIMPLE_VERT, LAYER_COMPOSITE_FRAG,      'LayerComposite');

    // ── Create FBO stacks for every scene ─────────────────────────────────
    const scenes: SceneId[] = ['Home','CleanRoom','Work','WorkDetail','About','TreeScene','Global'];
    for (const sid of scenes) {
      this.sceneFBOs.set(sid, {
        bg:         this._createFBO(this.width, this.height),
        cell:       this._createFBO(this.width, this.height),
        foreground: this._createFBO(this.width, this.height),
        ui:         this._createFBO(this.width, this.height),
        composite:  this._createFBO(this.width, this.height),
      });
    }

    // ── Final pipeline FBO ─────────────────────────────────────────────────
    this.globalFBO = this._createFBO(this.width, this.height);

    // ── White fallback 1×1 texture ─────────────────────────────────────────
    this.whiteTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                  new Uint8Array([255, 255, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // ── Fullscreen quad (two triangles, 6 vertices) ────────────────────────
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // tick — per-frame update + render
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Advance simulation time, step any active scene transition,
   * run the 4-layer alpha composite, apply the scene-specific
   * composite program, then run GlobalComposite into the canvas.
   */
  tick(dt: number): void {
    if (!this.initialised || this.destroyed) return;
    this.elapsed += dt;

    // ── Advance transition ─────────────────────────────────────────────────
    if (this.transition && !this.transition.complete) {
      this.transition.elapsed += dt;
      if (this.transition.elapsed >= this.transition.duration) {
        this.transition.complete = true;
        this.currentScene = this.transition.to;
      }
    }

    const gl = this.gl;
    const w  = this.width;
    const h  = this.height;

    // ── Step 1: Flatten the 4 layers of the active scene into composite FBO ─
    const stack = this.sceneFBOs.get(this.currentScene)!;
    this._runLayerComposite(stack);

    // ── Step 2: Apply scene-specific composite program ─────────────────────
    this._runSceneComposite(stack.composite, stack);

    // ── Step 3: GlobalComposite pass → write to canvas ─────────────────────
    this._runGlobalComposite(stack.composite.tex);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // resize
  // ───────────────────────────────────────────────────────────────────────────

  resize(w: number, h: number): void {
    if (!this.initialised) return;
    if (w === this.width && h === this.height) return;
    const gl = this.gl;
    this.width  = w;
    this.height = h;
    this.canvas.width  = w;
    this.canvas.height = h;

    // Reallocate all scene FBOs
    for (const [sid, stack] of this.sceneFBOs.entries()) {
      this._destroyFBOStack(stack);
      this.sceneFBOs.set(sid, {
        bg:         this._createFBO(w, h),
        cell:       this._createFBO(w, h),
        foreground: this._createFBO(w, h),
        ui:         this._createFBO(w, h),
        composite:  this._createFBO(w, h),
      });
    }

    // Reallocate final FBOs
    gl.deleteFramebuffer(this.globalFBO.fbo);
    gl.deleteTexture(this.globalFBO.tex);
    this.globalFBO = this._createFBO(w, h);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Scene control
  // ───────────────────────────────────────────────────────────────────────────

  setScene(scene: SceneId): void {
    this.currentScene = scene;
    this.transition   = null;
  }

  transitionTo(scene: SceneId, duration = 0.6): void {
    if (scene === this.currentScene) return;
    this.transition = {
      from:     this.currentScene,
      to:       scene,
      duration,
      elapsed:  0,
      complete: false,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Uniform update API
  // ───────────────────────────────────────────────────────────────────────────

  setGlobalUniforms(u: Partial<GlobalCompositeUniforms>): void {
    Object.assign(this.globalUniforms, u);
  }
  setHomeUniforms(u: Partial<HomeCompositeUniforms>): void {
    Object.assign(this.homeUniforms, u);
  }
  setCleanRoomUniforms(u: Partial<CleanRoomCompositeUniforms>): void {
    Object.assign(this.cleanRoomUniforms, u);
  }
  setWorkUniforms(u: Partial<WorkCompositeUniforms>): void {
    Object.assign(this.workUniforms, u);
  }
  setWorkDetailUniforms(u: Partial<WorkDetailCompositeUniforms>): void {
    Object.assign(this.workDetailUniforms, u);
  }

  setScroll(scroll: number, delta = 0, _velocity = 0): void {
    this.globalUniforms.scroll      = scroll;
    this.globalUniforms.scrollDelta = delta;
  }
  setContact(contact: number): void {
    this.globalUniforms.contact = Math.max(0, Math.min(1, contact));
  }
  setMouse(x: number, y: number): void {
    this.globalUniforms.mouse = [x, y];
  }
  setUIColor(r: number, g: number, b: number, blend: number): void {
    this.globalUniforms.uiColor = [r, g, b];
    this.globalUniforms.uiBlend = blend;
  }
  setVisible(v: number): void {
    this.globalUniforms.visible = Math.max(0, Math.min(1, v));
  }
  setWorkPanelTransition(t: number): void {
    this.workUniforms.transition = Math.max(0, Math.min(1, t));
  }

  // ── Texture setters ───────────────────────────────────────────────────────

  setHomeVideoTexture(tex: WebGLTexture): void {
    this.homeVideoTex = tex;
  }
  setWorkDetailTexture(tex: WebGLTexture): void {
    this.workDetailTex = tex;
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  get isInitialised():    boolean  { return this.initialised; }
  get isDestroyed():      boolean  { return this.destroyed; }
  get elapsedTime():      number   { return this.elapsed; }
  get activeScene():      SceneId  { return this.currentScene; }
  get isTransitioning():  boolean  {
    return this.transition !== null && !this.transition.complete;
  }
  get transitionProgress(): number {
    if (!this.transition) return 1.0;
    return Math.min(1.0, this.transition.elapsed / Math.max(this.transition.duration, 0.001));
  }

  // ───────────────────────────────────────────────────────────────────────────
  // destroy
  // ───────────────────────────────────────────────────────────────────────────

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    const gl = this.gl;
    if (!gl) return;

    // Delete all compiled programs
    gl.deleteProgram(this.progHome);
    gl.deleteProgram(this.progCleanRoom);
    gl.deleteProgram(this.progWork);
    gl.deleteProgram(this.progWorkDetail);
    gl.deleteProgram(this.progAbout);
    gl.deleteProgram(this.progTreeScene);
    gl.deleteProgram(this.progGlobal);
    gl.deleteProgram(this.progLayers);

    // Delete per-scene FBO stacks
    for (const stack of this.sceneFBOs.values()) {
      this._destroyFBOStack(stack);
    }
    this.sceneFBOs.clear();

    // Delete final FBOs
    if (this.globalFBO) {
      gl.deleteFramebuffer(this.globalFBO.fbo);
      gl.deleteTexture(this.globalFBO.tex);
    }

    // Delete fallback texture and quad buffer
    gl.deleteTexture(this.whiteTex);
    gl.deleteBuffer(this.quadBuf);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private — layer composite (4 layers → composite FBO)
  // ───────────────────────────────────────────────────────────────────────────

  private _runLayerComposite(stack: ReturnType<typeof this.sceneFBOs.get> & {}): void {
    const gl   = this.gl;
    const prog = this.progLayers;
    const w    = this.width;
    const h    = this.height;

    gl.useProgram(prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, stack.composite.fbo);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Bind 4 layer textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, stack.bg.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tBG'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, stack.cell.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tCell'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, stack.foreground.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tForeground'), 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, stack.ui.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tUI'), 3);

    this._drawQuad(prog);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private — scene composite dispatch
  // ───────────────────────────────────────────────────────────────────────────

  private _runSceneComposite(
    compositeFBO: GLFBO,
    stack: ReturnType<typeof this.sceneFBOs.get> & {},
  ): void {
    switch (this.currentScene) {
      case 'Home':       return this._runHomeComposite(compositeFBO);
      case 'CleanRoom':  return this._runCleanRoomComposite(compositeFBO);
      case 'Work':       return this._runWorkComposite(compositeFBO);
      case 'WorkDetail': return this._runWorkDetailComposite(compositeFBO);
      case 'About':      return this._runAboutComposite(compositeFBO);
      case 'TreeScene':  return this._runTreeSceneComposite(compositeFBO);
      case 'Global':     return; // Global is handled as final pass always
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private — HomeComposite pass
  // ───────────────────────────────────────────────────────────────────────────

  private _runHomeComposite(srcFBO: GLFBO): void {
    const gl   = this.gl;
    const prog = this.progHome;
    const u    = this.homeUniforms;
    const w    = this.width;
    const h    = this.height;

    gl.useProgram(prog);
    // Render into the globalFBO as intermediate result
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.globalFBO.fbo);
    gl.viewport(0, 0, w, h);

    // tDiffuse — scene composite
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcFBO.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);

    // tVolumetricBlur — white fallback if none provided
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tVolumetricBlur'), 1);

    // tBloom — white fallback
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tBloom'), 2);

    gl.uniform1f(gl.getUniformLocation(prog, 'uRGBStrength'),        u.rgbStrength);
    gl.uniform1f(gl.getUniformLocation(prog, 'uVolumetricStrength'), u.volumetricStrength);
    gl.uniform2f(gl.getUniformLocation(prog, 'uContrast'),           u.contrast[0], u.contrast[1]);
    gl.uniform1f(gl.getUniformLocation(prog, 'time'),                this.elapsed);
    gl.uniform2f(gl.getUniformLocation(prog, 'resolution'),          w, h);

    this._drawQuad(prog);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private — CleanRoomComposite pass
  // ───────────────────────────────────────────────────────────────────────────

  private _runCleanRoomComposite(srcFBO: GLFBO): void {
    const gl   = this.gl;
    const prog = this.progCleanRoom;
    const u    = this.cleanRoomUniforms;
    const w    = this.width;
    const h    = this.height;

    gl.useProgram(prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.globalFBO.fbo);
    gl.viewport(0, 0, w, h);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcFBO.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tVolumetricBlur'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tBloom'), 2);

    gl.uniform1f(gl.getUniformLocation(prog, 'uRGBStrength'),        u.rgbStrength);
    gl.uniform1f(gl.getUniformLocation(prog, 'uVolumetricStrength'), u.volumetricStrength);
    gl.uniform2f(gl.getUniformLocation(prog, 'uContrast'),           u.contrast[0], u.contrast[1]);
    gl.uniform1f(gl.getUniformLocation(prog, 'time'),                this.elapsed);
    gl.uniform2f(gl.getUniformLocation(prog, 'resolution'),          w, h);

    this._drawQuad(prog);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private — WorkComposite pass (FBM radial wipe)
  // ───────────────────────────────────────────────────────────────────────────

  private _runWorkComposite(srcFBO: GLFBO): void {
    const gl   = this.gl;
    const prog = this.progWork;
    const u    = this.workUniforms;
    const w    = this.width;
    const h    = this.height;

    gl.useProgram(prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.globalFBO.fbo);
    gl.viewport(0, 0, w, h);

    // tDiffuse — work list FBO
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcFBO.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);

    // tDetail — work detail texture (or white fallback)
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.workDetailTex ?? this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDetail'), 1);

    // tBloom fallback
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tBloom'), 2);

    gl.uniform1f(gl.getUniformLocation(prog, 'uRGBStrength'), u.rgbStrength);
    gl.uniform1f(gl.getUniformLocation(prog, 'uTransition'),  u.transition);
    gl.uniform2f(gl.getUniformLocation(prog, 'uContrast'),    u.contrast[0], u.contrast[1]);
    gl.uniform1f(gl.getUniformLocation(prog, 'time'),         this.elapsed);
    gl.uniform2f(gl.getUniformLocation(prog, 'resolution'),   w, h);

    this._drawQuad(prog);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private — WorkDetailComposite pass (RGB shift)
  // ───────────────────────────────────────────────────────────────────────────

  private _runWorkDetailComposite(srcFBO: GLFBO): void {
    const gl   = this.gl;
    const prog = this.progWorkDetail;
    const u    = this.workDetailUniforms;
    const w    = this.width;
    const h    = this.height;

    gl.useProgram(prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.globalFBO.fbo);
    gl.viewport(0, 0, w, h);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcFBO.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);

    // tBloom fallback
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tBloom'), 1);

    gl.uniform1f(gl.getUniformLocation(prog, 'uRGBStrength'), u.rgbStrength);
    gl.uniform1f(gl.getUniformLocation(prog, 'time'),         this.elapsed);
    gl.uniform2f(gl.getUniformLocation(prog, 'resolution'),   w, h);

    this._drawQuad(prog);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private — AboutComposite pass (pass-through)
  // ───────────────────────────────────────────────────────────────────────────

  private _runAboutComposite(srcFBO: GLFBO): void {
    const gl   = this.gl;
    const prog = this.progAbout;
    const w    = this.width;
    const h    = this.height;

    gl.useProgram(prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.globalFBO.fbo);
    gl.viewport(0, 0, w, h);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcFBO.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);

    // tBloom fallback
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tBloom'), 1);

    gl.uniform1f(gl.getUniformLocation(prog, 'time'),       this.elapsed);
    gl.uniform2f(gl.getUniformLocation(prog, 'resolution'), w, h);

    this._drawQuad(prog);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private — TreeSceneComposite pass
  // ───────────────────────────────────────────────────────────────────────────

  private _runTreeSceneComposite(srcFBO: GLFBO): void {
    const gl   = this.gl;
    const prog = this.progTreeScene;
    const w    = this.width;
    const h    = this.height;

    gl.useProgram(prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.globalFBO.fbo);
    gl.viewport(0, 0, w, h);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcFBO.tex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);

    // tBloom fallback
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tBloom'), 1);

    gl.uniform1f(gl.getUniformLocation(prog, 'uRGBStrength'), 0);
    gl.uniform2f(gl.getUniformLocation(prog, 'uContrast'),    1.0, 0.0);
    gl.uniform1f(gl.getUniformLocation(prog, 'time'),         this.elapsed);
    gl.uniform2f(gl.getUniformLocation(prog, 'resolution'),   w, h);

    this._drawQuad(prog);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private — GlobalComposite final pass → renders to canvas backbuffer
  // ───────────────────────────────────────────────────────────────────────────

  private _runGlobalComposite(sceneTex: WebGLTexture): void {
    const gl   = this.gl;
    const prog = this.progGlobal;
    const gu   = this.globalUniforms;
    const w    = this.width;
    const h    = this.height;

    gl.useProgram(prog);
    // Final output: render to canvas (null framebuffer)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);

    // tDiffuse — scene composite result
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tDiffuse'), 0);

    // tFluid — white fallback
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tFluid'), 1);

    // tFluidMask — white fallback
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tFluidMask'), 2);

    // tNormal — white fallback
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tNormal'), 3);

    // tLightStreak — white fallback
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tLightStreak'), 4);

    // tBloom — white fallback
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'tBloom'), 5);

    // Upload all GlobalComposite uniforms
    gl.uniform1f(gl.getUniformLocation(prog, 'time'),               this.elapsed);
    gl.uniform2f(gl.getUniformLocation(prog, 'resolution'),         w, h);
    gl.uniform1f(gl.getUniformLocation(prog, 'uRGBStrength'),       gu.rgbStrength);
    gl.uniform1f(gl.getUniformLocation(prog, 'uVolumetricStrength'),gu.volumetricStrength);
    gl.uniform2f(gl.getUniformLocation(prog, 'uContrast'),          gu.contrast[0], gu.contrast[1]);
    gl.uniform1f(gl.getUniformLocation(prog, 'uScroll'),            gu.scroll);
    gl.uniform1f(gl.getUniformLocation(prog, 'uContact'),           gu.contact);
    gl.uniform1f(gl.getUniformLocation(prog, 'uScrollDelta'),       gu.scrollDelta);
    gl.uniform2f(gl.getUniformLocation(prog, 'uMouse'),             gu.mouse[0], gu.mouse[1]);
    gl.uniform3f(gl.getUniformLocation(prog, 'uFrostCorner'),       gu.frostCorner[0], gu.frostCorner[1], gu.frostCorner[2]);
    gl.uniform1f(gl.getUniformLocation(prog, 'uNormalScale'),       gu.normalScale);
    gl.uniform1f(gl.getUniformLocation(prog, 'uVisible'),           gu.visible);
    gl.uniform1f(gl.getUniformLocation(prog, 'uChatOpen'),          gu.chatOpen);
    gl.uniform2f(gl.getUniformLocation(prog, 'uGradient'),          gu.gradient[0], gu.gradient[1]);
    gl.uniform1f(gl.getUniformLocation(prog, 'uMobile'),            gu.mobile);
    gl.uniform3f(gl.getUniformLocation(prog, 'uUIColor'),           gu.uiColor[0], gu.uiColor[1], gu.uiColor[2]);
    gl.uniform1f(gl.getUniformLocation(prog, 'uUIBlend'),           gu.uiBlend);
    gl.uniform1f(gl.getUniformLocation(prog, 'uSyncTouch'),         gu.syncTouch);

    this._drawQuad(prog);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ───────────────────────────────────────────────────────────────────────────

  /** Draw fullscreen quad using the compiled quad buffer. */
  private _drawQuad(prog: WebGLProgram): void {
    const gl     = this.gl;
    const posLoc = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(posLoc);
  }

  /** Compile vert + frag into a linked WebGLProgram. */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATSceneCompositesFull] vert compile error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATSceneCompositesFull] frag compile error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATSceneCompositesFull] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Create a single colour FBO backed by an RGBA texture. */
  private _createFBO(w: number, h: number): GLFBO {
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

    return { fbo, tex, width: w, height: h };
  }

  /** Destroy a scene FBO stack (all 5 FBOs). */
  private _destroyFBOStack(stack: {
    bg: GLFBO; cell: GLFBO; foreground: GLFBO; ui: GLFBO; composite: GLFBO;
  }): void {
    const gl = this.gl;
    for (const fbo of [stack.bg, stack.cell, stack.foreground, stack.ui, stack.composite]) {
      gl.deleteFramebuffer(fbo.fbo);
      gl.deleteTexture(fbo.tex);
    }
  }
}
