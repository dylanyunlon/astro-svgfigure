import { FluidGPU } from './fluid-gpu-pass';
import { BloomGPU } from './bloom-gpu-pass';
import { ShadowGPU } from './shadow-gpu-pass';
import { EdgeGPU } from './edge-gpu-pass';
import { MSDFTextGPU } from './msdf-gpu-pass';
import { CompositeGPU } from './composite-gpu-pass';
import { ParticleGPU } from './particle-gpu-pass';
import { PBRCellGPU } from './pbr-gpu-pass';
import { GlassGPU } from './glass-gpu-pass';
import { SDFIconGPU, createSDFIconGPU } from './sdf-gpu-pass';
import { initATShaderPipeline, listATShaders, getATProgram } from './at-shader-pipeline-bridge';
import { ATGeometryLoader } from './at-geometry-loader';
import { KTX2TextureLoader } from './ktx2-texture-loader';
import { safeCompile, checkFBO, drainErrors, setupContextLost } from './gpu-error-guard';
import { NukePass } from '../renderer/NukePass';
import { UELumenGI } from './ue-lumen-gi';
import { GPUPerfMonitor } from './gpu-perf-monitor';
import { parseUILParams, type UILParamsJson } from '../renderers/at-uil-bridge';
import uilParamsJson from '../../../upstream/activetheory-assets/uil-params.json';
import { ATVolumetricLight } from './at-volumetric-light';
import { ATWaterSurface } from './at-water-surface';
import { UEAtmosphereSky } from './ue-atmosphere-sky';
import { UEBloomTonemap } from './ue-bloom-tonemap';
import { ATJellyfishCell, type JellyfishInstance } from './at-jellyfish-cell';
import { ATFlowerParticleRenderer, type FlowerEdgeSpline } from './at-flower-particle';

/**
 * gpu-render-loop.ts — M966: 真正的 GPU 渲染主循环
 *
 * 串联所有 *-gpu-pass.ts 模块, 每帧执行完整渲染管线。
 * 集成 gpu-error-guard (context lost + safeCompile + FBO check + drainErrors)
 * 和 gpu-perf-monitor (per-pass CPU 计时 + FPS + drawcall 统计)。
 */




// ─── Cell 数据接口 ────────────────────────────────────────────





export interface CellData {
  cell_id: string;
  species: string;
  x: number; y: number; w: number; h: number;
  z: number;
  metallic: number;
  roughness: number;
  albedo: [number, number, number];
  label: string;
}

export interface EdgeData {
  edge_id: string;
  source: string;
  target: string;
  controlPoints: [number, number][];
  color: [number, number, number];
}

// ─── Species → 材质映射 ────────────────────────────────────────

const SPECIES_MATERIAL: Record<string, { metallic: number; roughness: number; albedo: [number,number,number] }> = {
  'cil-eye':         { metallic: 0.04, roughness: 0.6,  albedo: [0.247, 0.318, 0.71]  },
  'cil-bolt':        { metallic: 0.8,  roughness: 0.3,  albedo: [1.0,   0.435, 0.0]   },
  'cil-vector':      { metallic: 0.15, roughness: 0.5,  albedo: [0.18,  0.49,  0.196] },
  'cil-plus':        { metallic: 0.1,  roughness: 0.55, albedo: [0.776, 0.157, 0.157] },
  'cil-arrow-right': { metallic: 0.3,  roughness: 0.45, albedo: [0.271, 0.353, 0.392] },
};

// ─── GPU Render Loop ────────────────────────────────────────────

export class GPURenderLoop {
  private gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;

  // GPU passes — 每个都有真实 gl 调用
  private fluid: FluidGPU;
  private bloom: BloomGPU;
  private shadow: ShadowGPU;
  private edge: EdgeGPU;
  private msdf: MSDFTextGPU;
  private composite: CompositeGPU;
  private particle: ParticleGPU | null = null; // WebGL2 only
  private pbr: PBRCellGPU | null = null;
  private glass: GlassGPU | null = null;
  private sdfIcon: SDFIconGPU | null = null;
  // AT asset loaders — Draco geometry + KTX2 textures
  private geometryLoader: ATGeometryLoader | null = null;
  private textureLoader: KTX2TextureLoader | null = null;
  private nukePass: NukePass | null = null;
  private lumenGI: UELumenGI | null = null;
  private volumetricLight: ATVolumetricLight | null = null;
  private waterSurface: ATWaterSurface | null = null;
  private atmosphereSky: UEAtmosphereSky | null = null;
  private ueBloomTonemap: UEBloomTonemap | null = null;
  private atJellyfish: ATJellyfishCell | null = null;
  private atJellyfishInstances: JellyfishInstance[] = [];
  private atFlower: ATFlowerParticleRenderer | null = null;

  // Perf + error monitoring
  private perf: GPUPerfMonitor;
  private frameCount = 0;

  // UIL params — loaded once from uil-params.json, live-patchable at runtime
  private uil: UILParamsJson = uilParamsJson as UILParamsJson;

  // UIL-driven camera state
  private cameraWobbleStrength = 0.1;

  // UIL-driven shadow state
  private shadowLightDir: [number, number, number] = [-0.5, -1.0, -0.3];

  // M1219: auto-fit camera state (updated each frame)
  private _camScale = 1;
  private _camOffX = 0;
  private _camOffY = 0;

  // 状态
  private cells: CellData[] = [];
  private edges: EdgeData[] = [];
  private _edgesDirty = true;
  private _placeholderTex: WebGLTexture | null = null;
  private _pbrFBOReady = false;
  private mouseX = 0;
  private mouseY = 0;
  private prevMouseX = 0;
  private prevMouseY = 0;
  private running = false;
  private lastTime = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;

    // ── Perf monitor ──
    this.perf = new GPUPerfMonitor();

    // ── Context lost / restored handlers ──
    const onLost = () => {
      console.warn('[GPURenderLoop] context lost — pausing loop');
      this.running = false;
    };
    const onRestored = () => {
      console.log('[GPURenderLoop] context restored — resuming loop');
      this.running = true;
      this._initPasses();
      this.start();
    };
    setupContextLost(canvas, onLost, onRestored);

    const gl = canvas.getContext('webgl2', {
      alpha: true, antialias: true, premultipliedAlpha: false,
    });
    if (!gl) throw new Error('[GPURenderLoop] WebGL2 not available');
    this.gl = gl;

    // WebGL1 扩展在 WebGL2 中已内置，无需手动启用
    gl.getExtension('EXT_color_buffer_float');

    // 初始化全部 GPU pass
    this._initPasses();

    // 鼠标追踪
    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      this.prevMouseX = this.mouseX;
      this.prevMouseY = this.mouseY;
      this.mouseX = (e.clientX - rect.left) / rect.width;
      this.mouseY = 1.0 - (e.clientY - rect.top) / rect.height;
    });

    // 异步加载 AT compiled.vs shader bundle
    // 加载完成后 172 个 shader 的 #require 依赖全部递归解析
    initATShaderPipeline('/activetheory/compiled.vs')
      .then(() => {
        const names = listATShaders();
        console.log(`[GPURenderLoop] AT shaders ready: ${names.length} shaders`);

        // M1226: wire AT PhysicalShader into PBR pass via swapProgram
        // Use AT production shaders when available; fall back to default PBR on any error.
        if (this.pbr) {
          try {
            const atPhysical = getATProgram(this.gl, 'PhysicalShader');
            if (atPhysical) {
              // swapProgram returns false when AT shader has no compatible vertex attribute
              // (AT PhysicalShader may use 'position'/'aPosition' instead of 'aCorner').
              // In that case swapProgram already freed the AT program and kept the default.
              const swapped = this.pbr.swapProgram(atPhysical.program);
              if (swapped) {
                console.log('[GPURenderLoop] PBR pass: AT PhysicalShader active (production shaders)');
              } else {
                console.warn(
                  '[GPURenderLoop] AT PhysicalShader vertex attribute incompatible with quad VBO ' +
                  '— PBR pass keeps default shader (non-fatal)',
                );
              }
            } else {
              console.warn('[GPURenderLoop] AT PhysicalShader not found in compiled.vs — PBR keeps default shader');
            }
          } catch (e) {
            // Any unexpected error (e.g. GL context state) must not crash the render loop
            console.warn('[GPURenderLoop] AT PhysicalShader swap failed (non-fatal):', e);
          }
        }
      })
      .catch((e) => console.warn('[GPURenderLoop] AT shader load failed (non-fatal):', e));
  }

  /** 初始化 / 重新初始化所有 GPU pass (context restore 时复用) */
  private _initPasses(): void {
    const gl = this.gl;
    const canvas = this.canvas;

    this.fluid = new FluidGPU(gl, {
      simWidth: 256, simHeight: 256,
      dyeWidth: canvas.width, dyeHeight: canvas.height,
    });
    this.bloom = new BloomGPU(gl, canvas.width, canvas.height);
    this.shadow = new ShadowGPU(gl, 1024);
    this.edge = new EdgeGPU(gl);
    this.msdf = new MSDFTextGPU(gl);
    this.composite = new CompositeGPU(gl, canvas.width, canvas.height);

    // PBR cell surface pass
    try { this.pbr = new PBRCellGPU(gl); } catch (e) { console.warn('[GPURenderLoop] PBR init failed, using fallback:', e); }

    // Glass Fresnel pass
    try { this.glass = new GlassGPU(gl); } catch (e) { console.warn('[GPURenderLoop] Glass init failed:', e); }

    // ── Lumen GI (SSGI) ──
    try { this.lumenGI = new UELumenGI(gl); } catch (e) { console.warn('[GPURenderLoop] LumenGI init failed (non-fatal):', e); }

    // ── AT Volumetric Light ──
    try {
      this.volumetricLight = new ATVolumetricLight(gl, { width: canvas.width, height: canvas.height });
    } catch (e) { console.warn('[GPURenderLoop] ATVolumetricLight init failed (non-fatal):', e); }

    // ── AT Water Surface ──
    try {
      this.waterSurface = new ATWaterSurface(gl);
    } catch (e) { console.warn('[GPURenderLoop] ATWaterSurface init failed (non-fatal):', e); }

    // ── UE Atmosphere Sky (background sky scattering) ──
    try {
      this.atmosphereSky = new UEAtmosphereSky(gl as unknown as WebGLRenderingContext);
      this.atmosphereSky.init();
    } catch (e) { console.warn('[GPURenderLoop] UEAtmosphereSky init failed (non-fatal):', e); }

    // ── UE Bloom + ACES Tonemap (post composite) ──
    try {
      this.ueBloomTonemap = new UEBloomTonemap(gl as unknown as WebGLRenderingContext, canvas.width, canvas.height);
    } catch (e) { console.warn('[GPURenderLoop] UEBloomTonemap init failed (non-fatal):', e); }

    // ── AT Jellyfish Cell renderer (M1225) ──
    // Replaces rectangular cell visuals with AT jellyfish.bin Draco mesh.
    // load() is async (Draco decode); rendering silently skips until ready.
    try {
      this.atJellyfish = new ATJellyfishCell();
      this.atJellyfish.load(gl).catch((e) => {
        console.warn('[GPURenderLoop] ATJellyfishCell load failed (non-fatal):', e);
      });
    } catch (e) { console.warn('[GPURenderLoop] ATJellyfishCell init failed (non-fatal):', e); }

    // ── AT Flower Particle renderer (M1225) ──
    // Initialised with an empty edge list; edges are wired once setScene() runs.
    try {
      this.atFlower = new ATFlowerParticleRenderer(canvas, [], {
        uTimeMultiplier: 0.17,
        uSize: 8.0,
      });
    } catch (e) { console.warn('[GPURenderLoop] ATFlowerParticleRenderer init failed (non-fatal):', e); }

    // SDF species icon pass
    try { this.sdfIcon = createSDFIconGPU(gl); } catch (e) { console.warn('[GPURenderLoop] SDF init failed:', e); }

    // ── M1120: Load AT assets — Draco geometry + KTX2 textures ──
    // These are the actual 3D models and PBR textures from AT's production site.
    // Once loaded, cells use jellyfish/flower/hexagon meshes instead of flat rectangles,
    // and PBR shaders get real albedo/normal/MRO textures.
    const GEOMETRY_BASE = '/assets/geometry/';
    const TEXTURE_BASE = '/assets/textures/';

    // Geometry: Draco .bin → GPU VBO/IBO
    try {
      this.geometryLoader = new ATGeometryLoader({ gl });
      // Load cell geometries (mapping from AT asset → cell species)
      const geometryManifest = [
        { name: 'jellyfish',  url: `${GEOMETRY_BASE}jellyfish.bin`,       cell: 'self_attn' },
        { name: 'cables',     url: `${GEOMETRY_BASE}cables.bin`,          cell: 'edges' },
        { name: 'flower',     url: `${GEOMETRY_BASE}flower_spine-128.bin`, cell: 'ffn' },
        { name: 'hexagon',    url: `${GEOMETRY_BASE}hexagon_gem.bin`,     cell: 'add_norm' },
        { name: 'pillars',    url: `${GEOMETRY_BASE}pillars.bin`,         cell: 'input_embed' },
        { name: 'structure',  url: `${GEOMETRY_BASE}structure.bin`,       cell: 'pos_encode' },
        { name: 'spine',      url: `${GEOMETRY_BASE}spine.bin`,           cell: 'output' },
      ];
      for (const { name, url } of geometryManifest) {
        this.geometryLoader.load(url).then(() => {
          console.log(`[GPURenderLoop] geometry loaded: ${name}`);
        }).catch(() => { /* non-fatal */ });
      }
    } catch (e) { console.warn('[GPURenderLoop] geometry loader init failed:', e); }

    // Textures: KTX2 → GPU Texture2D (PBR albedo/normal/MRO)
    try {
      this.textureLoader = new KTX2TextureLoader(gl);
      const textureManifest = [
        'CABLES___CyclesBake_COMBINED.ktx2',
        'CABLES___PBR_AT_MRO.ktx2',
        'CABLES___PBR_Normal.ktx2',
        'PILLARS___CyclesBake_COMBINED.ktx2',
        'PILLARS___PBR_AT_MRO.ktx2',
        'PILLARS___PBR_Normal.ktx2',
      ];
      for (const name of textureManifest) {
        this.textureLoader.load(`${TEXTURE_BASE}${name}`).then(() => {
          console.log(`[GPURenderLoop] texture loaded: ${name}`);
        }).catch(() => { /* non-fatal */ });
      }
    } catch (e) { console.warn('[GPURenderLoop] texture loader init failed:', e); }

    // WebGL2 particle — 直接复用同一个 gl2 context
    this.particle = new ParticleGPU(gl, []);
  }

  /** 设置 cell 和 edge 数据 */
  setScene(cells: CellData[], edges: EdgeData[]): void {
    this.cells = cells;
    this.edges = edges;
    this._edgesDirty = true;

    // ── Rebuild AT jellyfish instances from current cell list ──
    if (this.atJellyfish) {
      this.atJellyfishInstances = cells.map(c =>
        this.atJellyfish!.createVariant(
          c.cell_id,
          c.species,
          c.x + c.w / 2,
          c.y + c.h / 2,
          c.z,
        )
      );
    }

    // ── Rebuild AT flower particle edges ──
    if (this.atFlower) {
      const flowerEdges: FlowerEdgeSpline[] = edges.map(e => ({
        edgeId:   e.edge_id,
        sourceId: e.source,
        targetId: e.target,
        weight:   1,
        points:   e.controlPoints.map(([x, y]) => ({ x, y, z: 0 })),
      }));
      this.atFlower.setEdges(flowerEdges);
    }
  }

  /**
   * 运行时替换 UIL 参数 (hot-reload / PubSub 推送时调用)。
   * 不需要重新创建 GPU pass — 下一帧 _pushUILUniforms 会自动拾取。
   */
  setUILParams(params: UILParamsJson): void {
    this.uil = params;
  }

  /**
   * _pushUILUniforms — 在每帧 render 开始时把 UIL 参数推到各 pass 的 shader uniform。
   *
   * UIL key 格式 (来自 uil-params.json):
   *   bloom:   "UnrealBloomComposite/UnrealBloomComposite/globalbloom/bloomStrength"
   *            "UnrealBloomComposite/UnrealBloomComposite/globalbloom/bloomRadius"
   *   shadow:  "SHADOW_Element_9_home_scene*" → lightDir via L_Element / CAMERA entries
   *   fluid:   "VolumetricLight_home_fDensity"  (drives fluid curl strength)
   *   pbr:     "ATPBR/ATPBR/Element_6_homeScene/uEnv", "uMRON", "uTint"
   *   camera:  "CAMERA_Element_3_home_scenewobbleStrength"
   */
  private _pushUILUniforms(): void {
    const gl = this.gl;
    const uil = this.uil;

    // ── helpers ──────────────────────────────────────────────────────────────
    const getNum = (key: string, fallback: number): number => {
      const v = uil[key];
      if (typeof v === 'number') return v;
      if (typeof v === 'string') { const n = parseFloat(v); return isNaN(n) ? fallback : n; }
      return fallback;
    };
    const getVec3 = (key: string, fallback: [number,number,number]): [number,number,number] => {
      const v = uil[key];
      if (Array.isArray(v) && v.length >= 3) return [Number(v[0]), Number(v[1]), Number(v[2])];
      return fallback;
    };
    const setUniform1f = (program: WebGLProgram, name: string, value: number): void => {
      gl.useProgram(program);
      const loc = gl.getUniformLocation(program, name);
      if (loc !== null) gl.uniform1f(loc, value);
    };
    const setUniform3f = (program: WebGLProgram, name: string, v: [number,number,number]): void => {
      gl.useProgram(program);
      const loc = gl.getUniformLocation(program, name);
      if (loc !== null) gl.uniform3f(loc, v[0], v[1], v[2]);
    };
    const setUniform4f = (program: WebGLProgram, name: string, arr: number[]): void => {
      gl.useProgram(program);
      const loc = gl.getUniformLocation(program, name);
      if (loc !== null) gl.uniform4f(loc, arr[0] ?? 0, arr[1] ?? 0, arr[2] ?? 0, arr[3] ?? 0);
    };

    // ── 1. Bloom pass ─────────────────────────────────────────────────────────
    // UIL keys for globalbloom variant (primary post-process bloom):
    //   "UnrealBloomComposite/UnrealBloomComposite/globalbloom/bloomStrength" = 0.3
    //   "UnrealBloomComposite/UnrealBloomComposite/globalbloom/bloomRadius"   = 0.2
    // Home scene bloom:
    //   "UnrealBloomComposite/UnrealBloomComposite/home/bloomStrength"        = 3.82
    //   "UnrealBloomComposite/UnrealBloomComposite/home/bloomRadius"          = 1.0
    if (this.bloom?.program) {
      const globalStrength = getNum(
        'UnrealBloomComposite/UnrealBloomComposite/globalbloom/bloomStrength', 0.3,
      );
      const globalRadius = getNum(
        'UnrealBloomComposite/UnrealBloomComposite/globalbloom/bloomRadius', 0.2,
      );
      const homeStrength = getNum(
        'UnrealBloomComposite/UnrealBloomComposite/home/bloomStrength', 3.82,
      );
      const homeRadius = getNum(
        'UnrealBloomComposite/UnrealBloomComposite/home/bloomRadius', 1.0,
      );
      const lumThreshold = getNum(
        'UnrealBloomLuminosity/UnrealBloomLuminosity/globalbloom/luminosityThreshold', 0.0,
      );
      setUniform1f(this.bloom.program, 'globalBloom',        globalStrength);
      setUniform1f(this.bloom.program, 'globalBloomRadius',  globalRadius);
      setUniform1f(this.bloom.program, 'homeBloomStrength',  homeStrength);
      setUniform1f(this.bloom.program, 'homeBloomRadius',    homeRadius);
      setUniform1f(this.bloom.program, 'luminosityThreshold', lumThreshold);
    }

    // ── 2. Shadow pass ────────────────────────────────────────────────────────
    // UIL shadow light position comes from SHADOW_Element_9_home_scene
    //   "SHADOW_Element_9_home_sceneposition": [0, 6.51, 0]
    // We derive a light direction vector from that position.
    if (this.shadow?.program) {
      const shadowPos = getVec3('SHADOW_Element_9_home_sceneposition', [0, 6.51, 0]);
      // normalise position → direction (pointing from pos toward origin)
      const len = Math.sqrt(shadowPos[0]**2 + shadowPos[1]**2 + shadowPos[2]**2) || 1;
      this.shadowLightDir = [
        -shadowPos[0] / len,
        -shadowPos[1] / len,
        -shadowPos[2] / len,
      ];
      setUniform3f(this.shadow.program, 'uLightDir', this.shadowLightDir);

      // Shadow far plane from UIL
      const shadowFar = getNum('SHADOW_Element_9_home_scenefar', 40);
      setUniform1f(this.shadow.program, 'uShadowFar', shadowFar);
    }

    // ── 3. Fluid pass ─────────────────────────────────────────────────────────
    // UIL: "VolumetricLight_home_fDensity" = 0.22  → drives fluid curl strength
    //      "VolumetricLight_home_fDecay"   = 0.80  → fluid dissipation
    if (this.fluid?.program) {
      const fluidDensity = getNum('VolumetricLight_home_fDensity', 0.22);
      const fluidDecay   = getNum('VolumetricLight_home_fDecay',   0.80);
      setUniform1f(this.fluid.program, 'uCurlStrength',  fluidDensity);
      setUniform1f(this.fluid.program, 'uDissipation',   fluidDecay);
    }

    // ── 4. PBR pass ───────────────────────────────────────────────────────────
    // Primary AT PBR scene: ATPBR/ATPBR/Element_6_homeScene
    //   "ATPBR/ATPBR/Element_6_homeScene/uEnv":  [1.5, 1]
    //   "ATPBR/ATPBR/Element_6_homeScene/uMRON": [1, 1.3, 1, 1]
    //   "ATPBR/ATPBR/Element_6_homeScene/uTint": "#e5f1ff"
    if (this.pbr?.program) {
      // uEnv — environment intensity [specular, diffuse]
      const uEnvRaw = uil['ATPBR/ATPBR/Element_6_homeScene/uEnv'];
      if (Array.isArray(uEnvRaw) && uEnvRaw.length >= 2) {
        gl.useProgram(this.pbr.program);
        const loc = gl.getUniformLocation(this.pbr.program, 'uEnv');
        if (loc !== null) gl.uniform2f(loc, Number(uEnvRaw[0]), Number(uEnvRaw[1]));
      }

      // uMRON — metallic, roughness, occlusion, normal strength [4f]
      const uMRONRaw = uil['ATPBR/ATPBR/Element_6_homeScene/uMRON'];
      if (Array.isArray(uMRONRaw)) {
        setUniform4f(this.pbr.program, 'uMRON', uMRONRaw.map(Number));
      }

      // uTint — hex color string → vec3
      const uTintRaw = uil['ATPBR/ATPBR/Element_6_homeScene/uTint'];
      if (typeof uTintRaw === 'string' && uTintRaw.startsWith('#')) {
        const hex = uTintRaw.slice(1);
        const r = parseInt(hex.slice(0,2), 16) / 255;
        const g = parseInt(hex.slice(2,4), 16) / 255;
        const b = parseInt(hex.slice(4,6), 16) / 255;
        setUniform3f(this.pbr.program, 'uTint', [r, g, b]);
      }

      // PhysicalShader uParams (Fresnel, roughness overrides)
      const physParams = uil['PhysicalShader/PhysicalShader/uParams'];
      if (Array.isArray(physParams)) {
        setUniform4f(this.pbr.program, 'uParams', physParams.map(Number));
      }

      // Shadow far / light dir (keep in sync with shadow pass)
      setUniform3f(this.pbr.program, 'uLightDir', this.shadowLightDir);
    }

    // ── 5. Camera wobble → particle emitter strength ──────────────────────────
    // UIL: "CAMERA_Element_3_home_scenewobbleStrength" = 0.1
    // We expose this as cameraWobbleStrength and forward it to the particle pass
    // as a velocity/turbulence scale.
    const wobble = getNum('CAMERA_Element_3_home_scenewobbleStrength', 0.1);
    this.cameraWobbleStrength = wobble;
    if (this.particle?.program) {
      setUniform1f(this.particle.program, 'uTurbulence', this.cameraWobbleStrength);
    }

    // ── 6. Glass pass ─────────────────────────────────────────────────────────
    // UIL: "GlassCubeShader/GlassCubeShader/Element_0_home_scene/uDistortStrength" = 8.06
    //      "GlassCubeShader/GlassCubeShader/Element_0_home_scene/uFresnelPow"      = 1.5
    //      "GlassCubeShader/GlassCubeShader/Element_0_home_scene/uRefractionRatio" = 1.0
    if (this.glass?.program) {
      const distort   = getNum('GlassCubeShader/GlassCubeShader/Element_0_home_scene/uDistortStrength', 8.06);
      const fresnelPow = getNum('GlassCubeShader/GlassCubeShader/Element_0_home_scene/uFresnelPow', 1.5);
      const refracRatio = getNum('GlassCubeShader/GlassCubeShader/Element_0_home_scene/uRefractionRatio', 1.0);
      setUniform1f(this.glass.program, 'uDistortStrength', distort);
      setUniform1f(this.glass.program, 'uFresnelPow',      fresnelPow);
      setUniform1f(this.glass.program, 'uRefractionRatio', refracRatio);

      // Fresnel tint color
      const fresnelColor = uil['GlassCubeShader/GlassCubeShader/Element_0_home_scene/uFresnelColor'];
      if (typeof fresnelColor === 'string' && fresnelColor.startsWith('#')) {
        const hex = fresnelColor.slice(1);
        const r = parseInt(hex.slice(0,2), 16) / 255;
        const g = parseInt(hex.slice(2,4), 16) / 255;
        const b = parseInt(hex.slice(4,6), 16) / 255;
        setUniform3f(this.glass.program, 'uFresnelColor', [r, g, b]);
      }
    }
  }

  /** 单帧渲染 — 执行全部 GPU pass，每 pass 带 error guard + perf 计时 */
  frame(dt: number): void {
    const gl = this.gl;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const time = performance.now() / 1000;
    this.frameCount++;

    this.perf.frameStart();

    // ── M1219: Auto-fit camera — scale all cells into the viewport ──────────
    // Cells live in a 0-2052 × 0-3965 pixel space but the canvas is ~1920×1080.
    // Compute a uniform scale + offset so every cell is visible.
    const PADDING = 40; // pixels of breathing room around bounding box
    let camMinX = Infinity, camMinY = Infinity, camMaxX = -Infinity, camMaxY = -Infinity;
    for (const c of this.cells) {
      if (c.x < camMinX) camMinX = c.x;
      if (c.y < camMinY) camMinY = c.y;
      if (c.x + c.w > camMaxX) camMaxX = c.x + c.w;
      if (c.y + c.h > camMaxY) camMaxY = c.y + c.h;
    }
    // Fallback when no cells are loaded yet
    if (!isFinite(camMinX)) { camMinX = 0; camMinY = 0; camMaxX = W; camMaxY = H; }
    const bbW = camMaxX - camMinX;
    const bbH = camMaxY - camMinY;
    const camScale = Math.min(
      W / (bbW + PADDING * 2),
      H / (bbH + PADDING * 2),
    );
    // After scaling, centre the bounding box inside the canvas
    const camOffX = (W - bbW * camScale) / 2 - camMinX * camScale;
    const camOffY = (H - bbH * camScale) / 2 - camMinY * camScale;

    /** Transform a cell's pixel-space x → NDC (−1 … 1) using auto-fit camera */
    const toNdcX = (px: number): number => ((px * camScale + camOffX) / W) * 2 - 1;
    /** Transform a cell's pixel-space y → NDC (−1 … 1) using auto-fit camera */
    const toNdcY = (py: number): number => ((py * camScale + camOffY) / H) * 2 - 1;
    /** Transform a cell's pixel-space x → fitted canvas pixel */
    const fitX = (px: number): number => px * camScale + camOffX;
    /** Transform a cell's pixel-space y → fitted canvas pixel */
    const fitY = (py: number): number => py * camScale + camOffY;
    /** Scale a dimension (w or h) by camScale */
    const fitD = (d: number): number => d * camScale;
    // ── End auto-fit setup ──────────────────────────────────────────────────
    // Re-upload edge control points whenever the camera changes
    if (camScale !== this._camScale || camOffX !== this._camOffX || camOffY !== this._camOffY) {
      this._edgesDirty = true;
    }
    this._camScale = camScale;
    this._camOffX = camOffX;
    this._camOffY = camOffY;

    // ── UIL params → GPU uniforms (每帧开始时推送) ──
    this._pushUILUniforms();

    // ── Pass 0: UE Atmosphere Sky (background — before all scene passes) ──
    if (this.atmosphereSky) {
      const t = this.perf.passStart('atmosphereSky');
      try {
        this.atmosphereSky.render(W, H);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] atmosphereSky pass error:', e); }
      this.perf.passEnd('atmosphereSky', t);
    }

    // ── Pass 1: Fluid (鼠标流体 → FBO) ──
    {
      const t = this.perf.passStart('fluid');
      try {
        this.fluid.step(this.mouseX, this.mouseY, this.prevMouseX, this.prevMouseY, dt);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('fluid', t);
    }

    // ── Pass 2: Shadow map → FBO ──
    {
      const t = this.perf.passStart('shadow');
      try {
        // shadow.step(cellPositions: Float32Array, cellCount, positionTex?)
        const posArr = new Float32Array(this.cells.length * 4);
        for (let i = 0; i < this.cells.length; i++) {
          const c = this.cells[i];
          posArr[i * 4 + 0] = fitX(c.x);
          posArr[i * 4 + 1] = fitY(c.y);
          posArr[i * 4 + 2] = fitD(c.w);
          posArr[i * 4 + 3] = fitD(c.h);
        }
        this.shadow.step(posArr, this.cells.length);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('shadow', t);
    }

    // ── Pass 3: PBR cell surface → FBO ──
    let cellTex: WebGLTexture;
    {
      const t = this.perf.passStart('pbr');
      try {
        if (this.pbr) {
          // Convert CellData → CellPBRDescriptor
          const descs = this.cells.map(c => ({
            species: c.species as any,
            x: toNdcX(c.x + c.w / 2),
            y: toNdcY(c.y + c.h / 2),
            size: Math.max(fitD(c.w), fitD(c.h)) / Math.max(W, H),
            albedo: c.albedo as [number, number, number],
            metallic: c.metallic,
            roughness: c.roughness,
          }));
          if (!this._pbrFBOReady) {
            this.pbr.initFBO(W, H);
            this._pbrFBOReady = true;
          }
          this.pbr.renderCells(descs);
          cellTex = this.pbr.pbrTexture;
        } else {
          cellTex = this._renderCellsFallback();
        }
      } catch (e) {
        if (this.frameCount <= 3) console.warn('[GPURenderLoop] PBR pass error:', e);
        cellTex = this._renderCellsFallback();
      }
      this.perf.passEnd('pbr', t);
    }

    // ── Pass 3b: AT Jellyfish cell renderer (M1225) ──
    // Renders AT jellyfish.bin mesh for each cell, instanced.
    // Runs after PBR so the translucent jellyfish layer composites on top.
    if (this.atJellyfish && this.atJellyfish.loaded && this.atJellyfishInstances.length > 0) {
      const t = this.perf.passStart('atJellyfish');
      try {
        // Animate all instances this frame
        for (const inst of this.atJellyfishInstances) {
          this.atJellyfish.animate(time, inst, dt);
        }
        // Identity view/proj — cells are already in world (pixel) space;
        // the jellyfish shader maps via modelMatrix written by animate().
        const identityMat = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
        this.atJellyfish.render(
          this.gl,
          this.atJellyfishInstances,
          identityMat,
          identityMat,
          time,
        );
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] ATJellyfish pass error:', e); }
      this.perf.passEnd('atJellyfish', t);
    }

    // ── Pass 3c: AT Flower particle renderer (M1225) ──
    // Runs Life TF → Pos FBO → Render passes for GPU spline particles.
    // Placed after PBR so particles layer over cell surfaces.
    if (this.atFlower) {
      const t = this.perf.passStart('atFlower');
      try {
        this.atFlower.tick(time, dt);
        this.atFlower.render(W, H);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] ATFlower pass error:', e); }
      this.perf.passEnd('atFlower', t);
    }
    {
      const t = this.perf.passStart('bloom');
      try {
        this.bloom.step(cellTex);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('bloom', t);
    }

    // ── Pass 5: Glass → per-cell Fresnel sheen (M1212: per-cell quads, not fullscreen) ──
    if (this.glass) {
      const t = this.perf.passStart('glass');
      try {
        // 将 CellData 转换为 GlassCellRect (像素坐标)
        const glassRects = this.cells.map(c => ({ x: fitX(c.x), y: fitY(c.y), w: fitD(c.w), h: fitD(c.h) }));
        this.glass.render(cellTex, this.bloom.outputTexture, time, glassRects, W, H);
      } catch (e) { /* non-fatal */ }
      this.perf.passEnd('glass', t);
    }

    // ── Pass 5b: Lumen GI (Screen-Space Global Illumination) ──
    // M1225: feed the real PBR G-Buffer textures instead of placeholders.
    if (this.lumenGI) {
      const t = this.perf.passStart('ssgi');
      try {
        const placeholder = this._placeholderTex ?? (this._placeholderTex = this._create1x1Tex());
        // Use real G-Buffer textures from the MRT PBR pass when available.
        const depthTex     = this.pbr ? this.pbr.depthTexture     : placeholder;
        const normalTex    = this.pbr ? this.pbr.normalTexture    : placeholder;
        const albedoTex    = this.pbr ? this.pbr.albedoTexture    : cellTex;
        const roughnessTex = this.pbr ? this.pbr.roughnessTexture : placeholder;
        this.lumenGI.render(dt, {
          depthTex,
          normalTex,
          albedoTex,
          roughnessTex,
          sceneColorTex: cellTex,
          viewMatrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
          projMatrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
          cameraPos: [0, 0, 5],
          frameIndex: this.frameCount,
        }, W, H);
      } catch (e) { /* non-fatal — SSGI degrades gracefully without G-Buffer */ }
      this.perf.passEnd('ssgi', t);
    }

    // ── NukePass (HDR tonemap + LUT) ──
    if (this.nukePass) {
      try { this.nukePass.render(gl); } catch (_) { /* non-fatal */ }
    }

    // ── AT Water Surface pass (背景层，在 fluid 之后) ──
    if (this.waterSurface) {
      const t = this.perf.passStart('waterSurface');
      try {
        // Build identity MVP for orthographic fullscreen
        const mvp = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
        const eye: [number, number, number] = [0, 0, 5];
        this.waterSurface.render(dt, mvp, eye, W, H);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] waterSurface pass error:', e); }
      this.perf.passEnd('waterSurface', t);
    }

    // ── AT Volumetric Light pass (光照叠加层) ──
    if (this.volumetricLight) {
      const t = this.perf.passStart('volumetricLight');
      try {
        this.volumetricLight.render(cellTex);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] volumetricLight pass error:', e); }
      this.perf.passEnd('volumetricLight', t);
    }

    // ── Pass 6: Composite → canvas (merge FBO layers) ──
    {
      const t = this.perf.passStart('composite');
      try {
        // Use placeholder for passes that render directly (edge/particle)
        const placeholder = this._placeholderTex ?? (this._placeholderTex = this._create1x1Tex());
        this.composite.render({
          cell:     cellTex,
          edge:     placeholder,
          particle: placeholder,
          bloom:    this.bloom.outputTexture,
          shadow:   this.shadow.shadowFactorTexture,
          fluid:    this.fluid.dyeTexture,
        }, W, H, time);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('composite', t);
    }

    // ── UE Bloom + ACES Tonemap (post-composite, UE-grade) ──
    if (this.ueBloomTonemap) {
      const t = this.perf.passStart('ueBloomTonemap');
      try {
        // Use the composite output (or cellTex fallback) as scene input
        const sceneTex = this.composite
          ? (this.composite as any).outputTexture ?? cellTex
          : cellTex;
        this.ueBloomTonemap.render(sceneTex, W, H);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] ueBloomTonemap pass error:', e); }
      this.perf.passEnd('ueBloomTonemap', t);
    }

    // ── Direct overlay passes (render on top of composite) ──
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Edge splines
    {
      const t = this.perf.passStart('edge');
      try {
        // Convert EdgeData → EdgeControlPoints and upload once
        if (this._edgesDirty) {
          const ecp = this.edges.map(e => ({
            id: e.edge_id,
            isSkip: e.edge_id.startsWith('skip'),
            p0: [fitX((e.controlPoints[0] ?? [0,0])[0]), fitY((e.controlPoints[0] ?? [0,0])[1])] as [number,number],
            p1: [fitX((e.controlPoints[1] ?? e.controlPoints[0] ?? [0,0])[0]), fitY((e.controlPoints[1] ?? e.controlPoints[0] ?? [0,0])[1])] as [number,number],
            p2: [fitX((e.controlPoints[2] ?? e.controlPoints[1] ?? [0,0])[0]), fitY((e.controlPoints[2] ?? e.controlPoints[1] ?? [0,0])[1])] as [number,number],
            p3: [fitX((e.controlPoints[3] ?? e.controlPoints[2] ?? [0,0])[0]), fitY((e.controlPoints[3] ?? e.controlPoints[2] ?? [0,0])[1])] as [number,number],
            sourceColor: e.color as [number,number,number],
          }));
          this.edge.setEdges(ecp);
          this._edgesDirty = false;
        }
        this.edge.render(time);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('edge', t);
    }

    // SDF species icons
    if (this.sdfIcon) {
      const t = this.perf.passStart('sdf');
      try {
        // Group cells by species → SpeciesBatch[]
        const bySpecies = new Map<string, Array<{x:number,y:number,size:number,opacity:number}>>();
        for (const c of this.cells) {
          if (!bySpecies.has(c.species)) bySpecies.set(c.species, []);
          bySpecies.get(c.species)!.push({
            x: fitX(c.x + c.w / 2), y: fitY(c.y + c.h / 2),
            size: Math.max(fitD(c.w), fitD(c.h)) * 0.6, opacity: 1.0,
          });
        }
        const batches = [...bySpecies.entries()].map(([species, instances]) => ({
          species: species as any, instances,
        }));
        this.sdfIcon.render(batches, dt);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('sdf', t);
    }

    // Particles
    if (this.particle) {
      const t = this.perf.passStart('particle');
      try {
        this.particle.render(W, H);
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('particle', t);
    }

    // MSDF text labels
    {
      const t = this.perf.passStart('msdf');
      try {
        this.msdf.drawAllCellLabels();
      } catch (e) { if (this.frameCount <= 3) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('msdf', t);
    }

    // M1211: removed M1210 direct-cell-rendering — it was clearing the entire
    // framebuffer AFTER composite output, destroying all PBR/bloom/shadow work.
    // The GPU passes are not silently failing; M1210 was overwriting them.

    // ── Drain accumulated GL errors ──
    drainErrors(gl);

    this.perf.frameEnd();

    // ── Log perf stats every 120 frames ──
    if (this.frameCount % 120 === 0) {
      console.log('[GPURenderLoop] perf:', this.perf.stats);
    }
  }

  /** 1×1 transparent placeholder texture for composite inputs that have no FBO */
  private _create1x1Tex(): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 0]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  /** 简单 fallback: 用纯色 quad 画 cell (pbr-gpu-pass 未到时) */
  private _fallbackTex: WebGLTexture | null = null;
  private _fallbackFBO: WebGLFramebuffer | null = null;
  private _renderCellsFallback(): WebGLTexture {
    const gl = this.gl;
    const W = this.canvas.width;
    const H = this.canvas.height;

    // 首次创建或尺寸变化时重建
    if (!this._fallbackTex) {
      this._fallbackTex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, this._fallbackTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      this._fallbackFBO = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._fallbackFBO);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._fallbackTex, 0);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fallbackFBO);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // 画每个 cell 为一个纯色矩形（像素坐标，Y 翻转因为 WebGL origin 在左下）
    for (const cell of this.cells) {
      const mat = SPECIES_MATERIAL[cell.species] ?? SPECIES_MATERIAL['cil-eye'];
      const fx = cell.x * this._camScale + this._camOffX;
      const fy = cell.y * this._camScale + this._camOffY;
      const fw = cell.w * this._camScale;
      const fh = cell.h * this._camScale;
      const px = Math.floor(fx);
      const py = Math.floor(H - fy - fh); // flip Y
      const pw = Math.floor(fw);
      const ph = Math.floor(fh);
      if (pw <= 0 || ph <= 0) continue;
      gl.enable(gl.SCISSOR_TEST);
      gl.scissor(px, py, pw, ph);
      gl.clearColor(mat.albedo[0], mat.albedo[1], mat.albedo[2], 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return this._fallbackTex;
  }

  /** 启动 requestAnimationFrame 循环 */
  start(): void {
    this.running = true;
    this.lastTime = performance.now() / 1000;
    const loop = () => {
      if (!this.running) return;
      const now = performance.now() / 1000;
      const dt = Math.min(now - this.lastTime, 1/30);
      this.lastTime = now;
      this.frame(dt);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
  }
}
