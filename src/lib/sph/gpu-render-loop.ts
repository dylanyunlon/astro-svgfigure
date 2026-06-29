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
import { ATMouseFluid } from './at-mousefluid-import';
import { CellMeshRenderer } from './cell-mesh-renderer';
import { CellInteractionPhysics } from '../cell-interaction-physics';
import speciesPhysicsJson from '../../../channels/physics/species_physics.json';
import cellLifecycleJson from '../../../channels/physics/cell_lifecycle.json';
import environmentJson from '../../../channels/physics/environment.json';
// M1287: species interaction matrix debug overlay
import { toggleInteractionDebug, isInteractionDebugEnabled, type DebugCell } from './debug-renderer';
// Re-export for callers who import only from gpu-render-loop
export { toggleInteractionDebug, isInteractionDebugEnabled } from './debug-renderer';

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
  // === 新增字段 ===
  glowColor: [number, number, number];    // glow_color hex → RGB [0,1]
  sdfShape: 'rounded_rect' | 'capsule';   // sdf_shape
  internalPattern: string;                 // internal_pattern
  haloRadius: number;                      // halo_radius [0,1]
  numRays: number;                         // num_rays (0-16)
  focalIntensity: number;                  // focal_intensity [0,1]
  animationSpeed: number;                  // animation_speed
  opacity: number;                         // opacity
  // === M1280: cell lifecycle ===
  energy?: number;                         // metabolic energy [0, max_energy], initial 1.0
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
  private fluid!: FluidGPU;
  private bloom!: BloomGPU;
  private shadow!: ShadowGPU;
  private edge!: EdgeGPU;
  private msdf!: MSDFTextGPU;
  private composite!: CompositeGPU;
  private particle: ParticleGPU | null = null; // WebGL2 only
  private pbr: PBRCellGPU | null = null;
  private glass: GlassGPU | null = null;
  private sdfIcon: SDFIconGPU | null = null;
  private cellMesh: CellMeshRenderer | null = null;
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
  private mouseFluid: ATMouseFluid | null = null;

  private physics: CellInteractionPhysics | null = null;

  // Perf + error monitoring
  private perf: GPUPerfMonitor;
  private frameCount = 0;

  // UIL params — loaded once from uil-params.json, live-patchable at runtime
  private uil: UILParamsJson = uilParamsJson as UILParamsJson;

  // UIL-driven camera state
  private cameraWobbleStrength = 0.1;

  // UIL-driven shadow state
  private shadowLightDir: [number, number, number] = [-0.5, -1.0, -0.3];

  // UIL-driven DOF + fog state (AT: HomeSceneVFX_home_uDOF / uFog / uFogColor)
  private dofParams: [number,number,number,number] = [0.72, 0.8, 0.3, 1.0];
  private fogParams: [number,number,number,number] = [0, 1.24, 1.0, 0.89];
  private fogColor: [number,number,number] = [0.102, 0.565, 0.678]; // #1a90ad

  // M1219: auto-fit camera state (updated each frame)
  private _camScale = 1;
  private _camOffX = 0;
  private _camOffY = 0;

  // M1272: expose camera + physics for mouse interaction
  get camera() { return { scale: this._camScale, offX: this._camOffX, offY: this._camOffY }; }
  get physicsEngine() { return this.physics; }

  // 状态
  private cells: CellData[] = [];
  private edges: EdgeData[] = [];

  // ── M1280: cell lifecycle state ──────────────────────────────────────────
  /** Parsed channels/physics/cell_lifecycle.json parameters. */
  private lifecycle = cellLifecycleJson as {
    energy_system: { base_consumption: number; movement_cost: number; collision_cost: number; regeneration_rate: number; max_energy: number };
    lifecycle: { division_energy_threshold: number; division_cooldown_ms: number; apoptosis_energy_threshold: number; apoptosis_delay_ms: number; max_age_ms: number };
    signaling: { signal_radius: number; signal_decay: number; quorum_threshold: number; quorum_response: string };
    membrane: { permeability: number; elasticity: number; repair_rate: number; rupture_threshold: number };
  };

  /** Parsed channels/physics/environment.json — flow field, brownian, gravity, boundaries, gradients. */
  private env = environmentJson as {
    flow_field: { type: string; direction: [number, number]; speed: number; turbulence: number };
    brownian_noise: number;
    gravity: { x: number; y: number };
    boundaries: { type: string; repel_force: number; margin: number; width: number; height: number };
    gradients: {
      temperature: { center: [number, number]; radius: number; delta: number };
      nutrient: { center: [number, number]; radius: number; concentration: number };
    };
  };
  /** Hard cap on cell count to prevent unbounded division. */
  private readonly MAX_CELLS = 200;
  /** Per-cell timestamp (ms) of last division — enforces division cooldown. */
  private divisionCooldownUntil: Map<string, number> = new Map();
  /** Per-cell timestamp (ms) when apoptosis countdown finishes; cell removed after. */
  private apoptosisDeadline: Map<string, number> = new Map();
  /** Monotonic counter for unique IDs of daughter cells. */
  private divisionCounter = 0;
  private _edgesDirty = true;
  private _placeholderTex: WebGLTexture | null = null;
  private _pbrFBOReady = false;
  private mouseX = 0;
  private mouseY = 0;
  private prevMouseX = 0;
  private prevMouseY = 0;
  private running = false;
  private lastTime = 0;

  // ── M1290: Community detection render state ──────────────────────────────
  /** Most-recent community assignment from 'community-update' event. */
  private _communityMap: Map<string, number> = new Map();
  /** Overlay canvas for community background circles (Canvas 2D). */
  private _communityCanvas: HTMLCanvasElement | null = null;
  private _communityCtx: CanvasRenderingContext2D | null = null;

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

    // ── M1290: Community overlay — listen for community-update events ────────
    // When the physics engine fires 'community-update', store the Map<cellId,
    // communityId> so the next frame's _renderCommunityOverlay() can use it.
    window.addEventListener('community-update', (e: Event) => {
      const ce = e as CustomEvent<{ communities: Map<string, number> }>;
      this._communityMap = ce.detail.communities;
    });

    // 异步加载 AT compiled.vs shader bundle
    // 加载完成后 172 个 shader 的 #require 依赖全部递归解析
    initATShaderPipeline('/activetheory/compiled.vs')
      .then(() => {
        const names = listATShaders();
        console.log(`[GPURenderLoop] AT shaders ready: ${names.length} shaders`);

        // M1239: AT PhysicalShader disabled — it uses WebGL1 syntax (attribute/varying/texture2D),
        // custom preprocessor (#drawbuffer), and expects 3D mesh geometry (not quad billboard).
        // Our built-in PBR Cook-Torrance shader is WebGL2-native and works with the quad VBO.
      })
      .catch((e) => console.warn('[GPURenderLoop] AT shader load failed (non-fatal):', e));
  }

  /** 初始化 / 重新初始化所有 GPU pass (context restore 时复用) */
  private _initPasses(): void {
    const gl = this.gl;
    const canvas = this.canvas;

    try {
      this.fluid = new FluidGPU(gl, {
        simWidth: 256, simHeight: 256,
        dyeWidth: canvas.width, dyeHeight: canvas.height,
      });
    } catch (e) { console.warn('[GPURenderLoop] Fluid init failed (non-fatal):', e); }

    try { this.bloom = new BloomGPU(gl, canvas.width, canvas.height); } catch (e) { console.warn('[GPURenderLoop] Bloom init failed:', e); }
    try { this.shadow = new ShadowGPU(gl, 1024); } catch (e) { console.warn('[GPURenderLoop] Shadow init failed:', e); }
    try { this.edge = new EdgeGPU(gl, { canvasWidth: canvas.width, canvasHeight: canvas.height }); } catch (e) { console.warn('[GPURenderLoop] Edge init failed:', e); }
    try { this.msdf = new MSDFTextGPU(gl); } catch (e) { console.warn('[GPURenderLoop] MSDF init failed:', e); }
    try { this.composite = new CompositeGPU(gl, canvas.width, canvas.height); } catch (e) { console.warn('[GPURenderLoop] Composite init failed:', e); }

    // PBR cell surface pass
    try { this.pbr = new PBRCellGPU(gl); } catch (e) { console.warn('[GPURenderLoop] PBR init failed, using fallback:', e); }

    // ── M1261: 3D mesh renderer (replaces 2D quad PBR when GLBs are loaded) ──
    try {
      this.cellMesh = new CellMeshRenderer(gl);
      console.info('[GPURenderLoop] CellMeshRenderer ready (placeholder cubes)');
    } catch (e) { console.warn('[GPURenderLoop] CellMeshRenderer init failed:', e); }

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
      this.atmosphereSky = new UEAtmosphereSky(gl);
      this.atmosphereSky.init();
    } catch (e) { console.warn('[GPURenderLoop] UEAtmosphereSky init failed (non-fatal):', e); }

    // ── UE Bloom + ACES Tonemap (post composite) ──
    try {
      this.ueBloomTonemap = new UEBloomTonemap(gl, canvas.width, canvas.height);
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

    // ── AT Flower Particle renderer (M1225, M1241: uniform→texture fix) ──
    try {
      this.atFlower = new ATFlowerParticleRenderer(canvas, [], { uTimeMultiplier: 0.17, uSize: 8.0 });
    } catch (e) { console.warn('[GPURenderLoop] ATFlowerParticleRenderer init failed (non-fatal):', e); }

    // SDF species icon pass
    try { this.sdfIcon = createSDFIconGPU(gl); } catch (e) { console.warn('[GPURenderLoop] SDF init failed:', e); }

    // ── AT MouseFluid — real GPU interactive mouse fluid (M1246) ──
    try {
      this.mouseFluid = ATMouseFluid.create(gl, canvas, {
        // AT production params from uil-params.json
        velocityDissipation: 0.98,     // AT: fluid_velocity
        curl: 30,                       // AT: fluid_curl
        densityDissipation: 0.97,       // AT: fluid_density
        splatRadius: 0.25,              // AT: defaultRadius mapped
        simWidth: 128,
        simHeight: 128,
        dyeWidth: canvas.width,
        dyeHeight: canvas.height,
      });
    } catch (e) { console.warn('[GPURenderLoop] ATMouseFluid init failed (non-fatal):', e); }

    // ── Inject AT production tuned params (from uil-params.json) ──
    this._applyATTunedParams();

    // ── M1120: Load AT assets — Draco geometry + KTX2 textures ──
    // These are the actual 3D models and PBR textures from AT's production site.
    // Once loaded, cells use jellyfish/flower/hexagon meshes instead of flat rectangles,
    // and PBR shaders get real albedo/normal/MRO textures.
    const GEOMETRY_BASE = '/assets/geometry/';
    const TEXTURE_BASE = '/assets/textures/';

    // M1250: ATGeometryLoader re-enabled (M1247 fixed shader compile errors)
    try {
      this.geometryLoader = new ATGeometryLoader({ gl });
      // Load all AT production 3D assets in parallel
      this.geometryLoader.loadAll()
        .then((loaded) => console.log(`[GPURenderLoop] AT geometry loaded: ${loaded.size} meshes`))
        .catch((e) => console.warn('[GPURenderLoop] AT geometry load partial fail (non-fatal):', e));
    } catch (e) { console.warn('[GPURenderLoop] ATGeometryLoader init failed (non-fatal):', e); }

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
  setScene(
    cells: CellData[],
    edges: EdgeData[],
  ): void {
    this.cells = cells;
    this.edges = edges;
    this._edgesDirty = true;

    // ── M1280: initialise lifecycle state ──────────────────────────────────
    // Every cell starts with full energy (1.0). Reset all per-cell lifecycle
    // bookkeeping so a fresh scene doesn't inherit stale cooldowns/deadlines.
    const maxE = this.lifecycle.energy_system.max_energy;
    for (const c of this.cells) {
      if (c.energy === undefined) c.energy = maxE;
    }
    this.divisionCooldownUntil.clear();
    this.apoptosisDeadline.clear();

    // Init physics engine
    try {
      const interactionCells = cells.map(c => ({cell_id: c.cell_id, bbox: {x: c.x, y: c.y, w: c.w, h: c.h}, z: c.z, species: c.species}));
      this.physics = new CellInteractionPhysics(interactionCells, {gravity: {x: 0, y: 0}, damping: 0.92, speciesPhysics: speciesPhysicsJson as any});
    } catch(e) { console.warn('[GPURenderLoop] physics init failed:', e); }

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
   * M1281 — Environment physics step. Called once per frame after the main
   * physics step. Applies five environmental forces from environment.json to
   * every live cell via the throwCell (velocity impulse) API:
   *
   *   1. Flow field   — laminar directional flow nudges all cells downstream.
   *   2. Brownian     — random micro-perturbation simulates thermal jitter.
   *   3. Gravity      — steady downward (sedimentation) drift.
   *   4. Soft walls   — exponential repulsion from canvas boundaries.
   *   5. Temperature  — cells near the heat source receive an outward speed boost.
   *
   * All impulses are scaled by `dt` so behaviour is frame-rate independent.
   *
   * @param dt Frame delta time in seconds.
   */
  private _stepEnvironmentPhysics(dt: number): void {
    const physics = this.physics;
    if (!physics) return;

    // Guard pathological dt (tab backgrounding etc.)
    const step = Math.max(0, Math.min(dt, 1 / 30));

    const env = this.env;
    const ff  = env.flow_field;
    const bnd = env.boundaries;
    const grd = env.gradients;

    // Normalise flow direction vector (json: [0.1, -0.02])
    const [fdx, fdy] = ff.direction;
    const fMag = Math.sqrt(fdx * fdx + fdy * fdy) || 1;
    const fnx = fdx / fMag;
    const fny = fdy / fMag;

    // Flow speed in px/s (json: 15)
    const flowSpeed = ff.speed;

    // Brownian amplitude in px/s (json: 0.3 — very subtle)
    const brownAmp = env.brownian_noise;

    // Gravity in px/s² (json: { x:0, y:0.5 })
    const gx = env.gravity.x;
    const gy = env.gravity.y;

    // Soft-wall margin + repel force (json: margin:50, repel_force:100)
    const margin      = bnd.margin;
    const repelForce  = bnd.repel_force;
    const bndW        = bnd.width;
    const bndH        = bnd.height;

    // Temperature gradient (json: center:[1000,2000], radius:800, delta:5)
    const tCenter = grd.temperature.center;
    const tRadius = grd.temperature.radius;
    const tDelta  = grd.temperature.delta; // extra speed multiplier near source

    for (const c of this.cells) {
      // Retrieve current position from physics body (centre of cell)
      const st = physics.getState(c.cell_id);
      const cx = st ? st.x : c.x + c.w / 2;
      const cy = st ? st.y : c.y + c.h / 2;

      let impulseX = 0;
      let impulseY = 0;

      // ── 1. Laminar flow field ───────────────────────────────────────────
      // Turbulence adds a tiny per-cell random variation to the flow direction
      // so cells don't move in perfect lock-step.
      const turbX = (Math.random() * 2 - 1) * ff.turbulence;
      const turbY = (Math.random() * 2 - 1) * ff.turbulence;
      impulseX += (fnx + turbX) * flowSpeed * step;
      impulseY += (fny + turbY) * flowSpeed * step;

      // ── 2. Brownian noise ───────────────────────────────────────────────
      impulseX += (Math.random() * 2 - 1) * brownAmp * step;
      impulseY += (Math.random() * 2 - 1) * brownAmp * step;

      // ── 3. Gravity ──────────────────────────────────────────────────────
      impulseX += gx * step;
      impulseY += gy * step;

      // ── 4. Soft-wall boundary repulsion ─────────────────────────────────
      // For each of the four walls, compute a repulsion that grows
      // exponentially as the cell enters the margin zone.
      // Left wall
      if (cx < margin) {
        const depth = (margin - cx) / margin; // 0→1 as cell reaches wall
        impulseX += repelForce * depth * depth * step;
      }
      // Right wall
      if (cx > bndW - margin) {
        const depth = (cx - (bndW - margin)) / margin;
        impulseX -= repelForce * depth * depth * step;
      }
      // Top wall (y=0)
      if (cy < margin) {
        const depth = (margin - cy) / margin;
        impulseY += repelForce * depth * depth * step;
      }
      // Bottom wall
      if (cy > bndH - margin) {
        const depth = (cy - (bndH - margin)) / margin;
        impulseY -= repelForce * depth * depth * step;
      }

      // ── 5. Temperature gradient ─────────────────────────────────────────
      // Cells within the heat source radius receive an outward radial boost
      // proportional to how close they are (stronger at centre).
      const tdx = cx - tCenter[0];
      const tdy = cy - tCenter[1];
      const tDist2 = tdx * tdx + tdy * tdy;
      if (tDist2 < tRadius * tRadius && tDist2 > 0) {
        const tDist  = Math.sqrt(tDist2);
        // Normalised proximity 1 at centre, 0 at radius edge
        const tProx  = 1 - tDist / tRadius;
        // Outward direction
        const tx = tdx / tDist;
        const ty = tdy / tDist;
        // Boost scales with delta (px/s) * proximity
        const boost = tDelta * tProx * step;
        impulseX += tx * boost;
        impulseY += ty * boost;
      }

      // ── Commit accumulated impulse ───────────────────────────────────────
      if (impulseX !== 0 || impulseY !== 0) {
        physics.throwCell(c.cell_id, impulseX, impulseY);
      }
    }
  }

  /**
   * M1280 — Cell lifecycle step. Called once per frame from frame() after the
   * physics step. Implements four coupled biological subsystems:
   *
   *   1. Energy metabolism — each cell spends energy on basal upkeep and on
   *      movement, and slowly regenerates back toward max_energy.
   *   2. Division — well-fed cells (energy > division threshold) split into two
   *      daughters, each inheriting half the parent's energy, subject to a
   *      cooldown and the global MAX_CELLS cap.
   *   3. Apoptosis — starved cells (energy < apoptosis threshold) begin a death
   *      countdown; if they don't recover before the delay elapses they are
   *      removed from the simulation.
   *   4. Quorum sensing — cells count same-species neighbours within
   *      signal_radius; once the count reaches quorum_threshold they trigger the
   *      quorum response (clustering: lowered repulsion via setClusterFactor).
   *
   * @param dt Frame delta time in seconds.
   */
  private _stepLifecycle(dt: number): void {
    const physics = this.physics;
    if (!physics) return;
    // Guard against pathological dt (tab refocus etc.) so energy book-keeping
    // stays stable. Clamp to a single 1/30s frame max.
    const step = Math.max(0, Math.min(dt, 1 / 30));
    const now = performance.now();

    const E = this.lifecycle.energy_system;
    const L = this.lifecycle.lifecycle;
    const S = this.lifecycle.signaling;

    // ── 1. Energy metabolism ────────────────────────────────────────────────
    for (const c of this.cells) {
      if (c.energy === undefined) c.energy = E.max_energy;
      const st = physics.getState(c.cell_id);
      const speed = st ? Math.sqrt(st.vx * st.vx + st.vy * st.vy) : 0;

      // Basal consumption + movement cost.
      c.energy -= E.base_consumption * step;
      c.energy -= speed * E.movement_cost * step;
      // Regeneration (capped at max_energy).
      c.energy += E.regeneration_rate * step;
      if (c.energy > E.max_energy) c.energy = E.max_energy;
      if (c.energy < 0) c.energy = 0;
    }

    // ── 2. Division ─────────────────────────────────────────────────────────
    // Collect new cells first, then append, so we don't mutate this.cells while
    // iterating it.
    const newCells: CellData[] = [];
    for (const c of this.cells) {
      if (this.cells.length + newCells.length >= this.MAX_CELLS) break;
      if ((c.energy ?? 0) <= L.division_energy_threshold) continue;

      const cooldownUntil = this.divisionCooldownUntil.get(c.cell_id) ?? 0;
      if (now < cooldownUntil) continue;

      // Split energy in half between parent and daughter.
      const half = (c.energy ?? E.max_energy) / 2;
      c.energy = half;

      // Random position offset ±20px so the daughter doesn't spawn exactly on
      // top of the parent.
      const offX = (Math.random() * 2 - 1) * 20;
      const offY = (Math.random() * 2 - 1) * 20;

      const daughterId = `${c.cell_id}__d${++this.divisionCounter}`;
      const daughter: CellData = {
        ...c,
        cell_id: daughterId,
        x: c.x + offX,
        y: c.y + offY,
        energy: half,
      };
      newCells.push(daughter);

      // Notify the physics engine of the new body.
      physics.addCell({
        cell_id: daughterId,
        bbox: { x: daughter.x, y: daughter.y, w: daughter.w, h: daughter.h },
        z: daughter.z,
        species: daughter.species,
      });

      // Cooldown for BOTH parent and daughter to prevent runaway division.
      this.divisionCooldownUntil.set(c.cell_id, now + L.division_cooldown_ms);
      this.divisionCooldownUntil.set(daughterId, now + L.division_cooldown_ms);
    }
    if (newCells.length) {
      for (const d of newCells) this.cells.push(d);
    }

    // ── 3. Apoptosis ────────────────────────────────────────────────────────
    // Cells below the apoptosis threshold start (or continue) a death countdown.
    // Cells that recover above the threshold cancel any pending countdown.
    const toRemove: string[] = [];
    for (const c of this.cells) {
      const e = c.energy ?? E.max_energy;
      if (e < L.apoptosis_energy_threshold) {
        const deadline = this.apoptosisDeadline.get(c.cell_id);
        if (deadline === undefined) {
          // Begin countdown.
          this.apoptosisDeadline.set(c.cell_id, now + L.apoptosis_delay_ms);
        } else if (now >= deadline) {
          toRemove.push(c.cell_id);
        }
      } else if (this.apoptosisDeadline.has(c.cell_id)) {
        // Recovered — cancel the death countdown.
        this.apoptosisDeadline.delete(c.cell_id);
      }
    }
    if (toRemove.length) {
      const removeSet = new Set(toRemove);
      this.cells = this.cells.filter(c => !removeSet.has(c.cell_id));
      for (const id of toRemove) {
        physics.removeCell(id);                 // remove physics body
        this.apoptosisDeadline.delete(id);
        this.divisionCooldownUntil.delete(id);
      }
    }

    // ── 4. Quorum sensing ───────────────────────────────────────────────────
    // For each cell, count same-species neighbours within signal_radius. When
    // the count reaches quorum_threshold, trigger the quorum response: cluster
    // by lowering the cell's collision repulsion (setClusterFactor < 1). Cells
    // below quorum return to full repulsion (factor 1).
    const radius2 = S.signal_radius * S.signal_radius;
    // Snapshot positions once for O(n²) neighbour search.
    const pos: Array<{ id: string; x: number; y: number; species: string }> = [];
    for (const c of this.cells) {
      const st = physics.getState(c.cell_id);
      if (st) pos.push({ id: c.cell_id, x: st.x, y: st.y, species: c.species });
    }
    for (let i = 0; i < pos.length; i++) {
      const a = pos[i];
      let count = 0;
      for (let j = 0; j < pos.length; j++) {
        if (i === j) continue;
        const b = pos[j];
        if (b.species !== a.species) continue;
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        if (dx * dx + dy * dy <= radius2) count++;
      }
      if (count >= S.quorum_threshold) {
        // Quorum reached → cluster: reduce repulsion. The more crowded, the
        // tighter the packing, down to a floor of 0.2.
        const factor = Math.max(0.2, 1 - count * 0.1);
        physics.setClusterFactor(a.id, factor);
      } else {
        physics.setClusterFactor(a.id, 1);
      }
    }
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
    if (this.bloom) {
      // AT scene variant blending: home (hero) values take priority, global is base layer
      const homeStrength = getNum(
        'UnrealBloomComposite/UnrealBloomComposite/home/bloomStrength', 3.82,
      );
      const homeRadius = getNum(
        'UnrealBloomComposite/UnrealBloomComposite/home/bloomRadius', 1.0,
      );
      const lumThreshold = getNum(
        'UnrealBloomLuminosity/UnrealBloomLuminosity/home/luminosityThreshold', 0.0,
      );
      // AT variant blending: homebloom has separate overrides
      const homebloomStrength = getNum(
        'UnrealBloomComposite/UnrealBloomComposite/homebloom/bloomStrength', 1.2,
      );
      // Final blend: hero bloom + accent layer
      // Clamp to safe range for our non-HDR composite (AT uses ACES tonemap)
      const finalStrength = Math.min((homeStrength * 0.7 + homebloomStrength * 0.3) * 0.35, 2.0);
      const finalThreshold = Math.max(lumThreshold, 0.25); // never let threshold go to 0
      this.bloom.updateConfig({
        strength:  finalStrength,
        radius:    homeRadius,
        threshold: finalThreshold,
      });
    }

    // ── 2. Shadow pass ────────────────────────────────────────────────────────
    // UIL shadow light position comes from SHADOW_Element_9_home_scene
    //   "SHADOW_Element_9_home_sceneposition": [0, 6.51, 0]
    // We derive a light direction vector from that position.
    if (this.shadow) {
      const shadowPos = getVec3('SHADOW_Element_9_home_sceneposition', [0, 6.51, 0]);
      const len = Math.sqrt(shadowPos[0]**2 + shadowPos[1]**2 + shadowPos[2]**2) || 1;
      this.shadowLightDir = [
        -shadowPos[0] / len,
        -shadowPos[1] / len,
        -shadowPos[2] / len,
      ];
      this.shadow.setLightDir(this.shadowLightDir);
    }

    // ── 3. Fluid pass ─────────────────────────────────────────────────────────
    // UIL: "VolumetricLight_home_fDensity" = 0.22  → curl intensity scale
    //      "VolumetricLight_home_fDecay"   = 0.80  → dissipation rate
    // AT CloudFog params drive additional fluid behaviour:
    //      "INPUT_CloudFoghome_speed" = 0.7
    if (this.fluid) {
      const fluidDensity = getNum('VolumetricLight_home_fDensity', 0.22);
      const fluidDecay   = getNum('VolumetricLight_home_fDecay',   0.80);
      const fogSpeed     = getNum('INPUT_CloudFoghome_speed',      0.7);
      // Map AT density to curl strength: AT 0.22 ≈ curl 30 (our scale)
      // fogSpeed modulates responsiveness
      this.fluid.updateConfig({
        curl:          Math.round(fluidDensity * 136 * fogSpeed),  // 0.22 * 136 * 0.7 ≈ 21
        dissipation:   0.90 + fluidDecay * 0.08,                   // 0.80 → 0.964
        dyeDissipation: 0.88 + fluidDecay * 0.08,                  // slightly faster dye fade
      });
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
    const wobble = getNum('CAMERA_Element_3_home_scenewobbleStrength', 0.1);
    this.cameraWobbleStrength = wobble;

    // ── 6. Glass pass ─────────────────────────────────────────────────────────
    // UIL: "GlassCubeShader/GlassCubeShader/Element_0_home_scene/uDistortStrength" = 8.06
    //      "GlassCubeShader/GlassCubeShader/Element_0_home_scene/uFresnelPow"      = 1.5
    //      "GlassCubeShader/GlassCubeShader/Element_0_home_scene/uAttenuation"     = 0.5
    //      "GlassCubeShader/GlassCubeShader/Element_0_home_scene/uFresnelColor"    = #b4e0e3
    //      "GlassCubeShader/GlassCubeShader/Element_0_home_scene/uSpecAdd"         = [4.48, 0]
    if (this.glass) {
      const distort    = getNum('GlassCubeShader/GlassCubeShader/Element_0_home_scene/uDistortStrength', 8.06);
      const fresnelPow = getNum('GlassCubeShader/GlassCubeShader/Element_0_home_scene/uFresnelPow', 1.5);
      const atten      = getNum('GlassCubeShader/GlassCubeShader/Element_0_home_scene/uAttenuation', 0.5);
      // uFresnelColor → tintColor
      let tintR = 0.706, tintG = 0.878, tintB = 0.890; // #b4e0e3
      const fresnelColor = uil['GlassCubeShader/GlassCubeShader/Element_0_home_scene/uFresnelColor'];
      if (typeof fresnelColor === 'string' && fresnelColor.startsWith('#')) {
        const hex = fresnelColor.slice(1);
        tintR = parseInt(hex.slice(0,2), 16) / 255;
        tintG = parseInt(hex.slice(2,4), 16) / 255;
        tintB = parseInt(hex.slice(4,6), 16) / 255;
      }
      // uSpecAdd[0] → specStrength
      const specAdd = uil['GlassCubeShader/GlassCubeShader/Element_0_home_scene/uSpecAdd'];
      const specStr = Array.isArray(specAdd) ? Number(specAdd[0]) : 4.48;

      this.glass.updateConfig({
        distortStrength: distort,
        fresnelPow:      fresnelPow,
        tintStrength:    atten,
        tintColor:       [tintR, tintG, tintB],
        specStrength:    specStr,
      });
    }

    // ── 7. Volumetric Light ───────────────────────────────────────────────────
    // AT: VolumetricLight_home_* values drive god-ray quality
    if (this.volumetricLight) {
      const volExposure = getNum('VolumetricLight_home_fExposure', 0.86);
      const volDensity  = getNum('VolumetricLight_home_fDensity',  0.22);
      const volDecay    = getNum('VolumetricLight_home_fDecay',    0.80);
      const volWeight   = getNum('VolumetricLight_home_fWeight',   0.34);
      const volStrength = getNum('HomeCompositeuVolumetricStrength', 1.1);
      this.volumetricLight.updateConfig({
        exposure:  volExposure,
        density:   volDensity,
        decay:     volDecay,
        weight:    volWeight,
        raysScale: volStrength,
      });
    }

    // ── 8. DOF + Fog (scene VFX) ──────────────────────────────────────────────
    // AT: HomeSceneVFX_home_uDOF = [0.72, 0.8, 0.3, 1]  → focus/range/blur/strength
    //     HomeSceneVFX_home_uFog = [0, 1.24, 1, 0.89]     → start/end/density/alpha
    //     HomeSceneVFX_home_uFogColor = #1a90ad             → teal fog
    //     INPUT_CloudFoghome_alpha = 1.8, planes = 20, scale = 6
    // These are stored for composite pass / cloud fog background to consume.
    const dofRaw = uil['HomeSceneVFX_home_uDOF'];
    if (Array.isArray(dofRaw) && dofRaw.length >= 4) {
      this.dofParams = dofRaw.map(Number) as [number,number,number,number];
    }
    const fogRaw = uil['HomeSceneVFX_home_uFog'];
    if (Array.isArray(fogRaw) && fogRaw.length >= 4) {
      this.fogParams = fogRaw.map(Number) as [number,number,number,number];
    }
    const fogColor = uil['HomeSceneVFX_home_uFogColor'];
    if (typeof fogColor === 'string' && fogColor.startsWith('#')) {
      const hex = fogColor.slice(1);
      this.fogColor = [
        parseInt(hex.slice(0,2), 16) / 255,
        parseInt(hex.slice(2,4), 16) / 255,
        parseInt(hex.slice(4,6), 16) / 255,
      ];
    }
  }

  // ── M1290: Community background-circle overlay ───────────────────────────

  /**
   * _renderCommunityOverlay — draws translucent filled circles grouped by
   * community behind all other overlays, using a sibling Canvas 2D element.
   *
   * Algorithm:
   *   1. Group cells by communityId using the most-recent _communityMap.
   *   2. For each community compute the centroid and the maximum half-diagonal
   *      of its member cells.
   *   3. Draw a single semi-transparent radial-gradient circle centred on the
   *      centroid with radius = max half-diagonal + padding (30 px).
   *
   * The hue for each community is derived deterministically from the community
   * ID so communities keep a stable colour between frames.
   *
   * @param W Canvas width in pixels
   * @param H Canvas height in pixels
   * @param fitX Function that maps cell-space X → canvas-pixel X
   * @param fitY Function that maps cell-space Y → canvas-pixel Y
   * @param fitD Function that scales a dimension by camScale
   */
  private _renderCommunityOverlay(
    W: number, H: number,
    fitX: (x: number) => number,
    fitY: (y: number) => number,
    fitD: (d: number) => number,
  ): void {
    // Lazy-init overlay canvas
    if (!this._communityCanvas) {
      this._communityCanvas = document.createElement('canvas');
      this._communityCanvas.style.cssText = [
        'position:absolute', 'top:0', 'left:0',
        'width:100%', 'height:100%',
        'pointer-events:none',
        'z-index:1',  // below SDF icons (z-index 9999) but above WebGL
      ].join(';');
      this.canvas.parentElement?.appendChild(this._communityCanvas)
        ?? document.body.appendChild(this._communityCanvas);
      this._communityCtx = this._communityCanvas.getContext('2d');
    }

    const ctx = this._communityCtx;
    if (!ctx || !this._communityCanvas) return;

    // Sync size
    if (this._communityCanvas.width !== W || this._communityCanvas.height !== H) {
      this._communityCanvas.width  = W;
      this._communityCanvas.height = H;
    }

    ctx.clearRect(0, 0, W, H);

    if (this._communityMap.size === 0) return;

    // Group cells by communityId
    const groups = new Map<number, { sumX: number; sumY: number; count: number; maxR: number }>();

    for (const c of this.cells) {
      const cid = this._communityMap.get(c.cell_id);
      if (cid === undefined) continue;

      const cx = fitX(c.x + c.w / 2);
      const cy = fitY(c.y + c.h / 2);
      const halfDiag = Math.sqrt(
        (fitD(c.w) / 2) ** 2 + (fitD(c.h) / 2) ** 2,
      );

      const g = groups.get(cid);
      if (g) {
        g.sumX += cx;
        g.sumY += cy;
        g.count++;
        if (halfDiag > g.maxR) g.maxR = halfDiag;
      } else {
        groups.set(cid, { sumX: cx, sumY: cy, count: 1, maxR: halfDiag });
      }
    }

    // Draw one soft circle per community
    const PADDING = 30; // extra px around the outermost cell
    for (const [cid, g] of groups) {
      if (g.count < 1) continue;

      const centX = g.sumX / g.count;
      const centY = g.sumY / g.count;
      const radius = g.maxR + PADDING;

      // Deterministic hue from cid (golden-angle spacing)
      const hue = (cid * 137.508) % 360;
      const gradient = ctx.createRadialGradient(centX, centY, 0, centX, centY, radius);
      gradient.addColorStop(0,   `hsla(${hue}, 70%, 60%, 0.18)`);
      gradient.addColorStop(0.6, `hsla(${hue}, 70%, 55%, 0.10)`);
      gradient.addColorStop(1,   `hsla(${hue}, 70%, 50%, 0.00)`);

      ctx.beginPath();
      ctx.arc(centX, centY, radius, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();
    }
  }

  /** 单帧渲染 — 执行全部 GPU pass，每 pass 带 error guard + perf 计时 */
  frame(dt: number): void {
    const gl = this.gl;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const time = performance.now() / 1000;

    this.perf.frameStart();

    // Physics step — update cell positions from physics engine
    if (this.physics) {
      this.physics.step(performance.now());
      for (const c of this.cells) {
        const st = this.physics.getState(c.cell_id);
        if (st) { c.x = st.x - c.w/2; c.y = st.y - c.h/2; }
      }

      // ── M1271: Edge topology spring forces ──────────────────────────────────
      // Each edge is a chemical bond between its source/target cell. Bonded cells
      // should attract each other: when they drift farther apart than the ideal
      // bond length, apply a weak Hooke's-law spring impulse (k=0.01) pulling them
      // back together. This keeps connected sub-graphs cohesive without collapsing
      // them (the spring only pulls when distance > REST_LEN).
      const SPRING_K = 0.01;   // spring stiffness (weak)
      const REST_LEN = 200;    // ideal bond length in pixels
      const centers = new Map<string, { cx: number; cy: number }>();
      for (const c of this.cells) {
        centers.set(c.cell_id, { cx: c.x + c.w / 2, cy: c.y + c.h / 2 });
      }
      for (const e of this.edges) {
        const a = centers.get(e.source);
        const b = centers.get(e.target);
        if (!a || !b) continue;

        const dx = b.cx - a.cx;
        const dy = b.cy - a.cy;
        const dist = Math.hypot(dx, dy);
        if (dist <= REST_LEN || dist === 0) continue; // only pull when stretched

        // Hooke's law: force magnitude ∝ stretch beyond rest length
        const stretch = dist - REST_LEN;
        const f = SPRING_K * stretch;
        const ux = dx / dist;
        const uy = dy / dist;

        // Pull both endpoints toward each other (equal & opposite impulses)
        this.physics.throwCell(e.source,  ux * f,  uy * f);
        this.physics.throwCell(e.target, -ux * f, -uy * f);
      }

      // ── M1281: environment physics — flow field, brownian noise, gravity,
      // soft-wall boundaries, temperature gradient. Runs after spring forces
      // so environmental impulses accumulate on top of structural bonding.
      this._stepEnvironmentPhysics(dt);

      // ── M1280: cell lifecycle — energy metabolism, division, apoptosis,
      // quorum sensing. Runs after the physics step so it reads up-to-date
      // velocities and positions for energy/movement cost and neighbour counts.
      this._stepLifecycle(dt);
    }

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

    // M1232 debug: log camera and cell NDC positions (first frame only)
    if (this.frameCount === 0) {
      console.log('[GPURenderLoop] camera:', { camScale, camOffX, camOffY, bbW, bbH, W, H });
      const c0 = this.cells[0];
      if (c0) {
        console.log('[GPURenderLoop] cell[0] NDC:', {
          name: c0.cell_id,
          x: toNdcX(c0.x + c0.w/2).toFixed(3),
          y: toNdcY(c0.y + c0.h/2).toFixed(3),
          sizeX: (fitD(c0.w) / W).toFixed(4),
          sizeY: (fitD(c0.h) / H).toFixed(4),
        });
      }
    }

    // ── UIL params → GPU uniforms (每帧开始时推送) ──
    this._pushUILUniforms();

    // ── Pass 0: UE Atmosphere Sky (background — before all scene passes) ──
    if (this.atmosphereSky) {
      const t = this.perf.passStart('atmosphereSky');
      try {
        this.atmosphereSky.render(W, H);
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] atmosphereSky pass error:', e); }
      this.perf.passEnd('atmosphereSky', t);
    }

    // ── Pass 0b: M1285 — Nutrient + Temperature gradient background ──────────
    // Drawn onto the canvas (default FBO) right after the sky background and
    // before any cell geometry, using additive blending so the glows layer
    // on top of the sky without darkening it.
    {
      const t = this.perf.passStart('gradients');
      try {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, W, H);
        gl.enable(gl.BLEND);
        // Additive blend: dst = src*ONE + dst*ONE
        gl.blendFunc(gl.ONE, gl.ONE);

        const ng = this.env.gradients.nutrient;
        // nutrient: center=[500,1000], radius=1200, concentration=0.8
        // colour: rgba(100,200,100, concentration*0.15) → normalised
        this._renderGradient(
          ng.center[0], ng.center[1], ng.radius,
          100 / 255, 200 / 255, 100 / 255,
          ng.concentration * 0.15,
          W, H, camScale, camOffX, camOffY,
        );

        const tg = this.env.gradients.temperature;
        // temperature: center=[1000,2000], radius=800
        // colour: rgba(255,100,50, 0.1) → normalised
        this._renderGradient(
          tg.center[0], tg.center[1], tg.radius,
          255 / 255, 100 / 255, 50 / 255,
          0.1,
          W, H, camScale, camOffX, camOffY,
        );

        // Restore normal alpha blend for subsequent passes
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      } catch (e) {
        if (this.frameCount <= 10) console.warn('[GPURenderLoop] gradient pass error:', e);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      }
      this.perf.passEnd('gradients', t);
    }

    // ── Pass 1: Fluid (鼠标流体 → FBO) ──
    {
      const t = this.perf.passStart('fluid');
      try {
        this.fluid.step(this.mouseX, this.mouseY, this.prevMouseX, this.prevMouseY, dt);
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] pass error:', e); }
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
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('shadow', t);
    }

    // ── Pass 3: PBR cell surface → FBO ──
    // Skip when CellMeshRenderer is active (3D meshes replace 2D quads)
    let cellTex: WebGLTexture = this._placeholderTex ?? (this._placeholderTex = this._create1x1Tex());
    const use3DMesh = this.cellMesh != null; // cellMesh exists → use it, PBR is fallback
    if (!use3DMesh) {
      const t = this.perf.passStart('pbr');
      try {
        if (this.pbr) {
          // Convert CellData → CellPBRDescriptor
          const descs = this.cells.map(c => ({
            species: c.species as any,
            x: toNdcX(c.x + c.w / 2),
            y: toNdcY(c.y + c.h / 2),
            size: Math.max(fitD(c.w), fitD(c.h)) / Math.max(W, H),
            sizeX: fitD(c.w) / W,
            sizeY: fitD(c.h) / H,
            albedo: c.albedo as [number, number, number],
            metallic: c.metallic,
            roughness: c.roughness,
            // 新增
            glowColor: c.glowColor,
            sdfShape: c.sdfShape,
            internalPattern: c.internalPattern,
            haloRadius: c.haloRadius,
            numRays: c.numRays,
            focalIntensity: c.focalIntensity,
            animationSpeed: c.animationSpeed,
            opacity: c.opacity,
          }));
          if (!this._pbrFBOReady) {
            this.pbr.initFBO(W, H);
            this._pbrFBOReady = true;
          }
          this.pbr.setTime(time);
          this.pbr.renderCells(descs);
          cellTex = this.pbr.pbrTexture;
        } else {
          cellTex = this._renderCellsFallback();
        }
      } catch (e) {
        if (this.frameCount <= 10) console.warn('[GPURenderLoop] PBR pass error:', e);
        cellTex = this._renderCellsFallback();
      }
      this.perf.passEnd('pbr', t);
    }

    // ── Pass 3a: 3D mesh cells (M1261) ──
    // Renders actual 3D geometry for each cell (placeholder cubes until GLBs loaded).
    // Output goes to its own FBO; we use it as cellTex when available.
    if (this.cellMesh) {
      const t = this.perf.passStart('cellMesh');
      try {
        this.cellMesh.setTime(time);
        this.cellMesh.render(this.cells, camScale, camOffX, camOffY, W, H);
        // Use 3D mesh output as cellTex (replaces PBR quad output)
        const meshTex = this.cellMesh.outputTexture;
        if (meshTex) cellTex = meshTex;
      } catch (e) {
        if (this.frameCount <= 10) console.warn('[GPURenderLoop] CellMesh pass error:', e);
        // Fallback: run PBR if 3D mesh fails
        cellTex = this._renderCellsFallback();
      }
      this.perf.passEnd('cellMesh', t);
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
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] ATJellyfish pass error:', e); }
      this.perf.passEnd('atJellyfish', t);
    }

    // ── Pass 3b2: AT Geometry Loader — Draco 3D meshes (M1250) ──
    if (this.geometryLoader) {
      const t = this.perf.passStart('atGeometry');
      try {
        const geoCache = (this.geometryLoader as any).geometryCache as Map<string, any> | undefined;
        if (geoCache && geoCache.size > 0) {
          // Render first loaded geometry (jellyfish preferred)
          const jellyGeo = geoCache.get('jellyfish') || geoCache.values().next().value;
          if (jellyGeo) {
            this.geometryLoader.render(jellyGeo, time, 0);
          }
        }
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] ATGeometry pass error:', e); }
      this.perf.passEnd('atGeometry', t);
    }

    // ── Pass 3c: AT Flower particle renderer (M1225) ──
    // Runs Life TF → Pos FBO → Render passes for GPU spline particles.
    // Placed after PBR so particles layer over cell surfaces.
    if (this.atFlower) {
      const t = this.perf.passStart('atFlower');
      try {
        this.atFlower.tick(time, dt);
        this.atFlower.render(W, H);
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] ATFlower pass error:', e); }
      this.perf.passEnd('atFlower', t);
    }
    {
      const t = this.perf.passStart('bloom');
      try {
        // Only bloom if we have real cell content (not 1×1 placeholder)
        if (cellTex !== this._placeholderTex) {
          this.bloom.step(cellTex);
        }
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] pass error:', e); }
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
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] waterSurface pass error:', e); }
      this.perf.passEnd('waterSurface', t);
    }

    // ── AT MouseFluid pass (鼠标交互流体) ──
    if (this.mouseFluid) {
      const t = this.perf.passStart('mouseFluid');
      try {
        this.mouseFluid.tick(dt);
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] mouseFluid pass error:', e); }
      this.perf.passEnd('mouseFluid', t);
    }

    // ── AT Volumetric Light pass (光照叠加层) ──
    if (this.volumetricLight) {
      const t = this.perf.passStart('volumetricLight');
      try {
        this.volumetricLight.render(cellTex);
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] volumetricLight pass error:', e); }
      this.perf.passEnd('volumetricLight', t);
    }

    // ── Pass 6: Composite → canvas (merge FBO layers) ──
    {
      const t = this.perf.passStart('composite');
      try {
        const placeholder = this._placeholderTex ?? (this._placeholderTex = this._create1x1Tex());
        if (this.composite) {
          this.composite.render({
            cell:       cellTex ?? placeholder,
            edge:       placeholder,
            particle:   placeholder,
            bloom:      this.bloom?.outputTexture ?? placeholder,
            shadow:     this.shadow?.shadowFactorTexture ?? placeholder,
            fluid:      this.mouseFluid?.dyeTexture ?? this.fluid?.dyeTexture ?? placeholder,
            gi:         this.lumenGI?.outputTexture ?? undefined,
            volumetric: this.volumetricLight?.raysTexture ?? undefined,
            geometry:   this.geometryLoader?.previewTexture ?? undefined,
          }, W, H, time);
        } else {
          // No composite — draw PBR directly to screen as fullscreen blit
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          gl.viewport(0, 0, W, H);
          gl.clearColor(0.03, 0.03, 0.05, 1.0);
          gl.clear(gl.COLOR_BUFFER_BIT);
          if (cellTex) {
            // Simple blit using the PBR output texture
            this._blitTexture(cellTex, W, H);
          }
        }
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] composite pass error:', e); }
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
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] ueBloomTonemap pass error:', e); }
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
        this.edge.setCanvasSize(W, H);
        this.edge.render(time);
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] pass error:', e); }
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
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('sdf', t);
    }

    // Particles
    if (this.particle) {
      const t = this.perf.passStart('particle');
      try {
        this.particle.render(W, H);
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('particle', t);
    }

    // MSDF text labels
    {\n      const t = this.perf.passStart('msdf');
      try {
        this.msdf.drawAllCellLabels();
      } catch (e) { if (this.frameCount <= 10) console.warn('[GPURenderLoop] pass error:', e); }
      this.perf.passEnd('msdf', t);
    }

    // ── M1290: Community background circles overlay ──────────────────────────
    // Draw semi-transparent radial gradients behind cells to show community
    // membership. Rendered on a sibling Canvas 2D element so it doesn't disturb
    // the WebGL framebuffer.
    try {
      this._renderCommunityOverlay(W, H, fitX, fitY, fitD);
    } catch (e) {
      if (this.frameCount <= 10) console.warn('[GPURenderLoop] community overlay error:', e);
    }

    // ── M1287: Species interaction matrix debug overlay (Canvas 2D) ─────────
    // Draws attract (green) / repel (red) lines between cells within
    // interaction_radius.  Runs in the debug branch: only when the overlay is
    // toggled on via toggleInteractionDebug() or an explicit debug flag.
    if (isInteractionDebugEnabled()) {
      const t = this.perf.passStart('interactionDebug');
      try {
        // Obtain a Canvas2D context layered on top of the WebGL canvas.
        // We use the same canvas element; Canvas2D and WebGL share it via
        // willReadFrequently=false (read-only overlay, no pixel readback).
        const ctx2d = (this.canvas as any).__debugCtx2d as CanvasRenderingContext2D | undefined
          ?? (() => {
            // Create a sibling overlay <canvas> on first use so we don't
            // interfere with the WebGL context on the primary canvas.
            const overlay = document.createElement('canvas');
            overlay.width = this.canvas.width;
            overlay.height = this.canvas.height;
            overlay.style.cssText = [
              'position:absolute',
              'top:0', 'left:0',
              'width:100%', 'height:100%',
              'pointer-events:none',
              'z-index:9999',
            ].join(';');
            this.canvas.parentElement?.appendChild(overlay) ?? document.body.appendChild(overlay);
            const c = overlay.getContext('2d')!;
            // Cache on the WebGL canvas element for reuse across frames
            (this.canvas as any).__debugCtx2d = c;
            (this.canvas as any).__debugOverlay = overlay;
            return c;
          })();

        // Sync overlay canvas size if main canvas resized
        const ov = (this.canvas as any).__debugOverlay as HTMLCanvasElement;
        if (ov && (ov.width !== W || ov.height !== H)) {
          ov.width = W; ov.height = H;
        }

        ctx2d.clearRect(0, 0, W, H);

        // Build DebugCell list from current physics-updated cell positions
        const debugCells: DebugCell[] = this.cells.map(c => ({
          id: c.cell_id,
          species: c.species,
          // Use auto-fit camera transform to map cell centres into canvas pixels
          cx: (c.x + c.w / 2) * this._camScale + this._camOffX,
          cy: (c.y + c.h / 2) * this._camScale + this._camOffY,
        }));

        // Import drawInteractionLines via the already-imported module members.
        // We call renderDebugOverlay with a minimal world so only the
        // interaction overlay fires.
        const { renderDebugOverlay } = await import('./debug-renderer');
        renderDebugOverlay(
          ctx2d,
          {
            bodies: new Map(),
            debugCells,
          },
          [],   // no manifolds
          [],   // no AABBs
          [],   // no BVH
          {
            showAABBs: false,
            showContacts: false,
            showContactNormals: false,
            showPenetration: false,
            showBVH: false,
            showEmitters: false,
            showStats: false,
            showVelocityArrows: false,
            showDensityHeatmap: false,
            showInteractionMatrix: true,
            animatePanels: false,
          },
        );
      } catch (e) {
        if (this.frameCount <= 10) console.warn('[GPURenderLoop] interactionDebug pass error:', e);
      }
      this.perf.passEnd('interactionDebug', t);
    } else {
      // Hide overlay canvas when debug is off
      const ov = (this.canvas as any).__debugOverlay as HTMLCanvasElement | undefined;
      if (ov) ov.style.display = 'none';
    }

    // Re-show overlay when debug is on (toggle may have just enabled it)
    if (isInteractionDebugEnabled()) {
      const ov = (this.canvas as any).__debugOverlay as HTMLCanvasElement | undefined;
      if (ov) ov.style.display = '';
    }

    // M1211: removed M1210 direct-cell-rendering — it was clearing the entire
    // framebuffer AFTER composite output, destroying all PBR/bloom/shadow work.
    // The GPU passes are not silently failing; M1210 was overwriting them.

    // ── Drain accumulated GL errors (only log first 10 frames to avoid spam) ──
    if (this.frameCount < 10) {
      drainErrors(gl);
    } else if (this.frameCount === 10) {
      const n = drainErrors(gl);
      if (n > 0) console.warn(`[GPU-GUARD] suppressing further gl error logs (${n} errors on frame 10)`);
    } else {
      // Silently drain without logging
      while (gl.getError() !== gl.NO_ERROR) { /* drain */ }
    }

    // ── Estimate draw calls from active passes + cells ──
    let dc = this.cells.length + this.edges.length; // PBR + edge
    dc += this.cells.length; // SDF icons
    dc += 2; // MSDF shadow+main
    if (this.bloom) dc += 3;
    if (this.composite) dc += 1;
    if (this.shadow) dc += 2;
    if (this.fluid) dc += 1;
    if (this.particle) dc += 2;
    if (this.glass) dc += 1;
    for (let i = 0; i < dc; i++) this.perf.countDraw();

    this.perf.frameEnd();
    this.frameCount++;

    // ── Log perf stats once (frame 60 — after init settles) ──
    if (this.frameCount === 60) {
      console.log('[GPURenderLoop] perf:', this.perf.stats);
    }
  }

  /** Expose perf stats for HUD */
  get stats(): { fps: number; frameMs: number; drawCalls: number; cellCount: number; edgeCount: number; passes: Record<string, number> } {
    const s = this.perf.stats;
    return { ...s, cellCount: this.cells.length, edgeCount: this.edges.length };
  }

  /**
   * M1272: Expose the auto-fit camera transform so external code (e.g. pointer
   * interaction) can invert canvas-pixel coords back into cell-pixel space:
   *
   *   cellX = (canvasX - offX) / scale
   *   cellY = (canvasY - offY) / scale
   *
   * Values are updated every frame() before rendering.
   */
  get cameraTransform(): { scale: number; offX: number; offY: number } {
    return { scale: this._camScale, offX: this._camOffX, offY: this._camOffY };
  }

  /** M1272: Expose the physics engine so pointer events can drive drag/throw/inject. */
  get interactionPhysics(): CellInteractionPhysics | null {
    return this.physics;
  }

  /**
   * Apply AT production-tuned parameters from uil-params.json.
   * These are the ACTUAL values Active Theory ships — not defaults.
   */
  private _applyATTunedParams(): void {
    // ── Bloom: AT BloomLuminosityPass + HydraBloom ──
    if (this.bloom) {
      try {
        // AT: luminosityThreshold=0, bloomStrength=1.0, bloomRadius=1.0
        (this.bloom as any).strength  = 1.0;
        (this.bloom as any).radius    = 1.0;
        (this.bloom as any).threshold = 0.0;
      } catch (_) { /* safe access */ }
    }

    // ── Volumetric Light: AT VolumetricLight_home ──
    if (this.volumetricLight) {
      try {
        const vl = this.volumetricLight as any;
        if (vl.cfg) {
          vl.cfg.fExposure = 0.86;   // AT: VolumetricLight_home_fExposure
          vl.cfg.fDecay    = 0.80;   // AT: VolumetricLight_home_fDecay
          vl.cfg.fDensity  = 0.22;   // AT: VolumetricLight_home_fDensity
          vl.cfg.fWeight   = 0.34;   // AT: VolumetricLight_home_fWeight
          vl.cfg.fClamp    = 1.0;    // AT: VolumetricLight_home_fClamp
          vl.cfg.raysScale = 1.1;    // AT: HomeCompositeuVolumetricStrength
        }
      } catch (_) { /* safe access */ }
    }

    // ── Water Surface: AT TreeScene water params ──
    if (this.waterSurface) {
      try {
        const ws = this.waterSurface as any;
        if (ws.cfg) {
          ws.cfg.damping = 0.98;     // AT: water_viscosity
        }
      } catch (_) { /* safe access */ }
    }

    // ── PBR: AT uMRON + uEnv + uTint ──
    if (this.pbr) {
      try {
        const pbr = this.pbr as any;
        // AT typical: metallic=1, roughness=0.3, occlusion=1, normalStrength=1
        if (pbr.defaultMetallic !== undefined)  pbr.defaultMetallic  = 1.0;
        if (pbr.defaultRoughness !== undefined) pbr.defaultRoughness = 0.3;
        // AT uEnv: [envDiffuse=1.5, envSpecular=1.0]
        if (pbr.envDiffuseScale !== undefined)  pbr.envDiffuseScale  = 1.5;
        if (pbr.envSpecularScale !== undefined) pbr.envSpecularScale = 1.0;
      } catch (_) { /* safe access */ }
    }

    // ── Glass Fresnel: AT fresnel params ──
    if (this.glass) {
      try {
        const g = this.glass as any;
        if (g.fresnelPow !== undefined)      g.fresnelPow      = 0.3;   // AT typical
        if (g.fresnelStrength !== undefined)  g.fresnelStrength  = 0.73;  // AT: uFresnelStrength
        if (g.opacity !== undefined)          g.opacity          = 0.15;  // subtle sheen
      } catch (_) { /* safe access */ }
    }

    // ── Shadow: AT-inspired light direction ──
    this.shadowLightDir = [-0.5, -1.0, -0.3];

    // ── Camera: AT home scene camera ──
    this._camScale = 1.0;

    // ── Composite post-process: AT HomeSceneVFX values ──
    if (this.composite) {
      try {
        const cfg = (this.composite as any).config;
        if (cfg) {
          cfg.grainStrength    = 0.03;                   // AT film grain
          cfg.vignetteStrength = 0.6;                    // AT vignette
          cfg.shadowColor      = [0.02, 0.01, 0.04];    // AT dark cool purple
          cfg.highlightColor   = [1.0, 0.98, 0.95];     // AT warm highlight
        }
      } catch (_) { /* safe access */ }
    }

    // ── Atmosphere: AT fog color ──
    if (this.atmosphereSky) {
      try {
        // AT: HomeSceneVFX_home_uFogColor = #1a90ad (teal)
        // AT: uAtmosphere = [0.13, 2.13, 1, 0.41]
        const sky = this.atmosphereSky as any;
        if (sky.fogColor) sky.fogColor = [0.10, 0.56, 0.68];  // #1a90ad
      } catch (_) { /* safe access */ }
    }

    // ── Bloom: AT home scene bloom (not max, tuned down) ──
    if (this.bloom) {
      try {
        const b = this.bloom as any;
        // AT: homebloomStrength=0.6, homebloomRadius=0.8
        if (b.strength !== undefined) b.strength = 0.6;
        if (b.radius !== undefined)   b.radius   = 0.8;
      } catch (_) { /* safe access */ }
    }

    console.log('[GPURenderLoop] AT production params injected (deep)');
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

  // ── M1285: nutrient + temperature gradient background ──────────────────
  /** Shared fullscreen quad VBO for gradient passes */
  private _gradVBO: WebGLBuffer | null = null;
  /** Radial-gradient shader program (lazy-compiled on first use) */
  private _gradProg: WebGLProgram | null = null;

  /**
   * Renders a single radial gradient as a fullscreen additive-blend quad
   * directly onto the default (canvas) framebuffer.
   *
   * The gradient centre is given in cell-space pixels; it is mapped through
   * the auto-fit camera transform (camScale / camOffX / camOffY) to canvas
   * pixel space, then compared against each fragment's gl_FragCoord so no
   * NDC arithmetic is needed in JS.
   *
   * Caller is responsible for setting additive blend mode before the call.
   *
   * @param centerCellX  Gradient centre X in cell-space pixels
   * @param centerCellY  Gradient centre Y in cell-space pixels (top-down)
   * @param radiusCells  Gradient radius in cell-space pixels
   * @param r g b        Peak colour components [0..1]
   * @param a            Peak alpha at centre [0..1]
   * @param W H          Canvas width / height in pixels
   * @param camScale     Auto-fit uniform scale factor
   * @param camOffX      Auto-fit X offset in canvas pixels
   * @param camOffY      Auto-fit Y offset in canvas pixels (top-down)
   */
  private _renderGradient(
    centerCellX: number, centerCellY: number, radiusCells: number,
    r: number, g: number, b: number, a: number,
    W: number, H: number,
    camScale: number, camOffX: number, camOffY: number,
  ): void {
    const gl = this.gl;

    // ── Lazy-init gradient shader ──────────────────────────────────────────
    if (!this._gradProg) {
      const vs = gl.createShader(gl.VERTEX_SHADER)!;
      gl.shaderSource(vs, [
        '#version 300 es',
        'in vec2 aPos;',
        'void main() { gl_Position = vec4(aPos, 0.0, 1.0); }',
      ].join('\n'));
      gl.compileShader(vs);

      const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
      gl.shaderSource(fs, [
        '#version 300 es',
        'precision highp float;',
        '// centre in canvas-pixel (fragCoord) space',
        'uniform vec2  uCenter;',
        '// radius in canvas pixels',
        'uniform float uRadius;',
        '// peak colour + alpha',
        'uniform vec4  uColor;',
        'out vec4 fragColor;',
        'void main() {',
        '  float dx   = gl_FragCoord.x - uCenter.x;',
        '  float dy   = gl_FragCoord.y - uCenter.y;',
        '  float dist = sqrt(dx*dx + dy*dy) / uRadius;',
        '  float t    = max(0.0, 1.0 - dist);',
        '  float alpha = t * t;  // quadratic falloff — soft edge',
        '  fragColor = vec4(uColor.rgb, uColor.a * alpha);',
        '}',
      ].join('\n'));
      gl.compileShader(fs);

      const prog = gl.createProgram()!;
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);

      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.warn('[GPURenderLoop] gradient shader link failed:',
          gl.getProgramInfoLog(prog));
        return;
      }
      this._gradProg = prog;
    }

    // ── Lazy-init fullscreen quad VBO ──────────────────────────────────────
    if (!this._gradVBO) {
      this._gradVBO = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this._gradVBO);
      gl.bufferData(gl.ARRAY_BUFFER,
        new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]),
        gl.STATIC_DRAW);
    }

    // ── Transform cell-space centre → canvas-pixel fragCoord space ────────
    // camOffY uses top-down convention; gl_FragCoord.y is bottom-up.
    const cpx     = centerCellX * camScale + camOffX;          // canvas px X
    const cpyTD   = centerCellY * camScale + camOffY;          // canvas px Y (top-down)
    const cpyGL   = H - cpyTD;                                 // flip to bottom-up
    const radiusPx = radiusCells * camScale;

    // ── Draw fullscreen quad with uniforms ──────────────────────────────────
    gl.useProgram(this._gradProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._gradVBO);
    const posLoc = gl.getAttribLocation(this._gradProg, 'aPos');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    gl.uniform2f(gl.getUniformLocation(this._gradProg, 'uCenter'), cpx, cpyGL);
    gl.uniform1f(gl.getUniformLocation(this._gradProg, 'uRadius'), radiusPx);
    gl.uniform4f(gl.getUniformLocation(this._gradProg, 'uColor'), r, g, b, a);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  private _blitTexture(tex: WebGLTexture, w: number, h: number): void {
    const gl = this.gl;
    if (!this._blitProg) {
      const vs = gl.createShader(gl.VERTEX_SHADER)!;
      gl.shaderSource(vs, `#version 300 es
in vec2 p; out vec2 uv;
void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0, 1); }`);
      gl.compileShader(vs);
      const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
      gl.shaderSource(fs, `#version 300 es
precision highp float;
uniform sampler2D t; in vec2 uv; out vec4 o;
void main() { o = texture(t, uv); }`);
      gl.compileShader(fs);
      this._blitProg = gl.createProgram()!;
      gl.attachShader(this._blitProg, vs);
      gl.attachShader(this._blitProg, fs);
      gl.linkProgram(this._blitProg);
      gl.deleteShader(vs); gl.deleteShader(fs);
      this._blitVBO = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, this._blitVBO);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
    }
    gl.useProgram(this._blitProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._blitVBO);
    const loc = gl.getAttribLocation(this._blitProg, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(gl.getUniformLocation(this._blitProg, 't'), 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
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
