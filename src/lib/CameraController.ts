/**
 * CameraController.ts
 * M172 — OGL Camera + Orbit: zoom / pan / focus-on-cell
 *
 * Wraps upstream/ogl Camera + Orbit to provide:
 *   • zoom()           — programmatic dolly in/out
 *   • pan()            — programmatic pan in world-space XY
 *   • focusOnCell()    — animate camera to frame a CellParams bbox
 *   • mount()          — attach OGL Orbit event listeners to a canvas
 *   • unmount()        — remove event listeners
 *   • update()         — call every rAF frame
 *
 * AT Camera Presets (66 params) are preserved as static data below.
 */

// ─── Minimal OGL type stubs (no module bundler path alias needed) ─────────────
// We import at runtime via dynamic import so this file stays pure TS without
// requiring the project to add upstream/ogl to tsconfig paths.
// For type-checking purposes we declare the shapes we actually use.

export interface OGLVec3Like {
  x: number; y: number; z: number;
  set(x: number, y: number, z: number): OGLVec3Like;
  copy(v: OGLVec3Like): OGLVec3Like;
  add(v: OGLVec3Like): OGLVec3Like;
  sub(v: OGLVec3Like): OGLVec3Like;
  distance(): number;
}

export interface OGLCameraLike {
  position: OGLVec3Like;
  fov: number;
  near: number;
  far: number;
  type: 'perspective' | 'orthographic';
  lookAt(target: OGLVec3Like): OGLCameraLike;
  perspective(opts?: { fov?: number; near?: number; far?: number; aspect?: number }): OGLCameraLike;
  updateMatrixWorld(): void;
}

export interface OGLOrbitLike {
  enabled: boolean;
  target: OGLVec3Like;
  minDistance: number;
  maxDistance: number;
  zoomStyle: string;
  update(): void;
  forcePosition(): void;
  remove(): void;
}

// ─── AT Camera Preset types (unchanged from xiaodi #67) ──────────────────────

export interface CameraPreset {
  position?: [number, number, number];
  lookAt?: [number, number, number];
  groupPos?: [number, number, number];
  moveXY?: [number, number];
  fov?: number;
  far?: number;
  near?: number;
  lerpSpeed?: number;
  lerpSpeed2?: number;
  rotation?: [number, number, number];
  cameraRotation?: [number, number, number];
  viewportFocus?: [number, number];
  deltaRotate?: number;
  wobbleStrength?: number;
}

// ─── All 66 AT Camera Parameters (exact values) ──────────────────────────────

export const AT_CAMERA_PARAMS = {
  // Element_0 / Footer
  CAMERA_Element_0_FootergroupPos:        [0, 0, 0]            as [number,number,number],
  CAMERA_Element_0_FooterlookAt:          [0, 0, 0]            as [number,number,number],
  CAMERA_Element_0_FootermoveXY:          [0, 0]               as [number,number],
  CAMERA_Element_0_Footerposition:        [0, -1.03, 40]       as [number,number,number],

  // Element_10 / CleanRoom
  CAMERA_Element_10_CleanRoomfov:         30,
  CAMERA_Element_10_CleanRoomgroupPos:    [0, 0, 0]            as [number,number,number],
  CAMERA_Element_10_CleanRoomlerpSpeed:   0.08,
  CAMERA_Element_10_CleanRoomlookAt:      [0, 0.8, -0.3]       as [number,number,number],
  CAMERA_Element_10_CleanRoommoveXY:      [0, 0]               as [number,number],
  CAMERA_Element_10_CleanRoomposition:    [0, 1, 5]            as [number,number,number],
  CAMERA_Element_10_CleanRoomrotation:    [0.99, 0, 0]         as [number,number,number],

  // Element_10 / TreeScene
  CAMERA_Element_10_TreeScenegroupPos:    [0, 0, 0]            as [number,number,number],
  CAMERA_Element_10_TreeScenelookAt:      [-2, 4.74, 6]        as [number,number,number],
  CAMERA_Element_10_TreeScenemoveXY:      [0, 0]               as [number,number],
  CAMERA_Element_10_TreeSceneposition:    [0, 5, 32]           as [number,number,number],

  // Element_1 / About
  CAMERA_Element_1_AboutmoveXY:           [0, 0]               as [number,number],
  CAMERA_Element_1_Aboutposition:         [0, 0, 6]            as [number,number,number],

  // Element_1 / ContactUs
  CAMERA_Element_1_ContactUsposition:     [0, 0, 6]            as [number,number,number],

  // Element_1 / Contact
  CAMERA_Element_1_ContactgroupPos:       [0, 0, 0.004]        as [number,number,number],
  CAMERA_Element_1_Contactposition:       [0, 0, 6]            as [number,number,number],

  // Element_1 / Home
  CAMERA_Element_1_Homefov:               30,
  CAMERA_Element_1_HomegroupPos:          [0, 0, 0]            as [number,number,number],
  CAMERA_Element_1_HomelerpSpeed:         0.1,
  CAMERA_Element_1_HomelerpSpeed2:        1,
  CAMERA_Element_1_HomelookAt:            [0, 4.59, 0]         as [number,number,number],
  CAMERA_Element_1_HomemoveXY:            [0, 0]               as [number,number],
  CAMERA_Element_1_Homeposition:          [0, 2, 40]           as [number,number,number],

  // Element_1 / JellyfishDemo
  CAMERA_Element_1_JellyfishDemomoveXY:   [1, 1]               as [number,number],
  CAMERA_Element_1_JellyfishDemoposition: [0, 0, 6]            as [number,number,number],

  // Element_1 / WorkDetailParticles
  CAMERA_Element_1_WorkDetailParticleslookAt:   [0, 0, 5]      as [number,number,number],
  CAMERA_Element_1_WorkDetailParticlesmoveXY:   [-4, 4]        as [number,number],
  CAMERA_Element_1_WorkDetailParticlesposition: [0, 0, 22]     as [number,number,number],

  // Element_1 / WorkDetail
  CAMERA_Element_1_WorkDetailgroupPos:    [0, 0, 9.33]         as [number,number,number],
  CAMERA_Element_1_WorkDetaillerpSpeed:   0.07,
  CAMERA_Element_1_WorkDetaillookAt:      [0, 0, -10]          as [number,number,number],
  CAMERA_Element_1_WorkDetailmoveXY:      [-1, 0.5]            as [number,number],
  CAMERA_Element_1_WorkDetailposition:    [0, 0, 1]            as [number,number,number],
  CAMERA_Element_1_WorkDetailviewportFocus: [0, 0]             as [number,number],

  // Element_1 / homeScene
  CAMERA_Element_1_homeScenefar:          100,
  CAMERA_Element_1_homeScenefov:          20,
  CAMERA_Element_1_homeScenelookAt:       [0, 3, 0]            as [number,number,number],
  CAMERA_Element_1_homeSceneposition:     [0, 3, 15]           as [number,number,number],

  // Element_1 / particleTest
  CAMERA_Element_1_particleTestposition:  [0, 0, 10]           as [number,number,number],

  // Element_2 / Work
  CAMERA_Element_2_WorkcameraRotation:    [0, 0, 0]            as [number,number,number],
  CAMERA_Element_2_Workfov:               35,
  CAMERA_Element_2_WorkgroupPos:          [0, 0, 0]            as [number,number,number],
  CAMERA_Element_2_WorklerpSpeed:         0.07,
  CAMERA_Element_2_WorklerpSpeed2:        1,
  CAMERA_Element_2_WorklookAt:            [0, 0, -4]           as [number,number,number],
  CAMERA_Element_2_WorkmoveXY:            [0, 0]               as [number,number],
  CAMERA_Element_2_Workposition:          [0, 0, 2]            as [number,number,number],
  CAMERA_Element_2_Workrotation:          [0, 196.07, 0]       as [number,number,number],

  // Element_2 / work_page
  CAMERA_Element_2_work_pagemoveXY:       [0.5, 0.5]           as [number,number],
  CAMERA_Element_2_work_pageposition:     [0, 0, 2]            as [number,number,number],

  // Element_3 / home_scene
  CAMERA_Element_3_home_scenedeltaRotate:     3,
  CAMERA_Element_3_home_scenefov:             30,
  CAMERA_Element_3_home_scenegroupPos:        [0, 1.95, 8.020000000000001] as [number,number,number],
  CAMERA_Element_3_home_scenemoveXY:          [0.4, 0.2]       as [number,number],
  CAMERA_Element_3_home_sceneposition:        [0, 0, 8]        as [number,number,number],
  CAMERA_Element_3_home_scenerotation:        [0, 0, 0]        as [number,number,number],
  CAMERA_Element_3_home_scenewobbleStrength:  0.1,

  // Element_4 / ParticleTest
  CAMERA_Element_4_ParticleTestgroupPos:  [0, 0, 0]            as [number,number,number],
  CAMERA_Element_4_ParticleTestlookAt:    [0, 6.2, 2]          as [number,number,number],
  CAMERA_Element_4_ParticleTestmoveXY:    [0, 0]               as [number,number],
  CAMERA_Element_4_ParticleTestposition:  [0, 8, 35]           as [number,number,number],
  CAMERA_Element_4_ParticleTestrotation:  [0, 0, 0]            as [number,number,number],
} as const;

// ─── Scene / Species Presets (derived from AT params) ────────────────────────

export const CAMERA_PRESETS: Record<string, CameraPreset> = {
  Footer: {
    groupPos:   AT_CAMERA_PARAMS.CAMERA_Element_0_FootergroupPos,
    lookAt:     AT_CAMERA_PARAMS.CAMERA_Element_0_FooterlookAt,
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_0_FootermoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_0_Footerposition,
  },
  CleanRoom: {
    fov:        AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomfov,
    groupPos:   AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomgroupPos,
    lerpSpeed:  AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomlerpSpeed,
    lookAt:     AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomlookAt,
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoommoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomposition,
    rotation:   AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomrotation,
  },
  TreeScene: {
    groupPos:   AT_CAMERA_PARAMS.CAMERA_Element_10_TreeScenegroupPos,
    lookAt:     AT_CAMERA_PARAMS.CAMERA_Element_10_TreeScenelookAt,
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_10_TreeScenemoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_10_TreeSceneposition,
  },
  About: {
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_1_AboutmoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_Aboutposition,
  },
  ContactUs: {
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_ContactUsposition,
  },
  Contact: {
    groupPos:   AT_CAMERA_PARAMS.CAMERA_Element_1_ContactgroupPos,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_Contactposition,
  },
  Home: {
    fov:        AT_CAMERA_PARAMS.CAMERA_Element_1_Homefov,
    groupPos:   AT_CAMERA_PARAMS.CAMERA_Element_1_HomegroupPos,
    lerpSpeed:  AT_CAMERA_PARAMS.CAMERA_Element_1_HomelerpSpeed,
    lerpSpeed2: AT_CAMERA_PARAMS.CAMERA_Element_1_HomelerpSpeed2,
    lookAt:     AT_CAMERA_PARAMS.CAMERA_Element_1_HomelookAt,
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_1_HomemoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_Homeposition,
  },
  JellyfishDemo: {
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_1_JellyfishDemomoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_JellyfishDemoposition,
  },
  WorkDetailParticles: {
    lookAt:     AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailParticleslookAt,
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailParticlesmoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailParticlesposition,
  },
  WorkDetail: {
    groupPos:       AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailgroupPos,
    lerpSpeed:      AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetaillerpSpeed,
    lookAt:         AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetaillookAt,
    moveXY:         AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailmoveXY,
    position:       AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailposition,
    viewportFocus:  AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailviewportFocus,
  },
  homeScene: {
    far:      AT_CAMERA_PARAMS.CAMERA_Element_1_homeScenefar,
    fov:      AT_CAMERA_PARAMS.CAMERA_Element_1_homeScenefov,
    lookAt:   AT_CAMERA_PARAMS.CAMERA_Element_1_homeScenelookAt,
    position: AT_CAMERA_PARAMS.CAMERA_Element_1_homeSceneposition,
  },
  particleTest: {
    position: AT_CAMERA_PARAMS.CAMERA_Element_1_particleTestposition,
  },
  Work: {
    cameraRotation: AT_CAMERA_PARAMS.CAMERA_Element_2_WorkcameraRotation,
    fov:            AT_CAMERA_PARAMS.CAMERA_Element_2_Workfov,
    groupPos:       AT_CAMERA_PARAMS.CAMERA_Element_2_WorkgroupPos,
    lerpSpeed:      AT_CAMERA_PARAMS.CAMERA_Element_2_WorklerpSpeed,
    lerpSpeed2:     AT_CAMERA_PARAMS.CAMERA_Element_2_WorklerpSpeed2,
    lookAt:         AT_CAMERA_PARAMS.CAMERA_Element_2_WorklookAt,
    moveXY:         AT_CAMERA_PARAMS.CAMERA_Element_2_WorkmoveXY,
    position:       AT_CAMERA_PARAMS.CAMERA_Element_2_Workposition,
    rotation:       AT_CAMERA_PARAMS.CAMERA_Element_2_Workrotation,
  },
  work_page: {
    moveXY:   AT_CAMERA_PARAMS.CAMERA_Element_2_work_pagemoveXY,
    position: AT_CAMERA_PARAMS.CAMERA_Element_2_work_pageposition,
  },
  home_scene: {
    deltaRotate:    AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenedeltaRotate,
    fov:            AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenefov,
    groupPos:       AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenegroupPos,
    moveXY:         AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenemoveXY,
    position:       AT_CAMERA_PARAMS.CAMERA_Element_3_home_sceneposition,
    rotation:       AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenerotation,
    wobbleStrength: AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenewobbleStrength,
  },
  ParticleTest: {
    groupPos: AT_CAMERA_PARAMS.CAMERA_Element_4_ParticleTestgroupPos,
    lookAt:   AT_CAMERA_PARAMS.CAMERA_Element_4_ParticleTestlookAt,
    moveXY:   AT_CAMERA_PARAMS.CAMERA_Element_4_ParticleTestmoveXY,
    position: AT_CAMERA_PARAMS.CAMERA_Element_4_ParticleTestposition,
    rotation: AT_CAMERA_PARAMS.CAMERA_Element_4_ParticleTestrotation,
  },
};

// ─── Cell bbox type (subset of CellParams) ────────────────────────────────────

export interface CellBbox {
  x: number;
  y: number;
  w: number;
  h: number;
  z?: number;
}

export interface FocusCell {
  cell_id: string;
  bbox: CellBbox;
}

// ─── OGLCameraController options ─────────────────────────────────────────────

export interface OGLCameraControllerOptions {
  /**
   * Scale factor mapping cell-space units → world-space units.
   * CellRenderer lays out cells in a 2-D coordinate system (pixels).
   * The OGL scene uses a smaller world space.  Default: 0.01 (1px = 0.01 world)
   */
  worldScale?: number;

  /** Perspective FOV in degrees. Default: 45 */
  fov?: number;

  /** Camera near plane. Default: 0.1 */
  near?: number;

  /** Camera far plane. Default: 1000 */
  far?: number;

  /** Minimum dolly distance. Default: 1 */
  minDistance?: number;

  /** Maximum dolly distance. Default: 500 */
  maxDistance?: number;

  /** Orbit ease (0–1). Default: 0.25 (OGL default) */
  ease?: number;

  /** Orbit inertia (0–1). Default: 0.85 (OGL default) */
  inertia?: number;

  /** Orbit pan speed. Default: 0.1 */
  panSpeed?: number;

  /** Orbit zoom speed. Default: 1 */
  zoomSpeed?: number;

  /** Margin (world units) added around a cell bbox when focusing. Default: 2 */
  focusMargin?: number;

  /** Duration (ms) of the focus lerp animation. Default: 600 */
  focusDurationMs?: number;
}

// ─── Internal lerp state ─────────────────────────────────────────────────────

interface LerpState {
  active: boolean;
  startTime: number;
  durationMs: number;
  fromPos: [number, number, number];
  toPos: [number, number, number];
  fromTarget: [number, number, number];
  toTarget: [number, number, number];
}

// ─── OGLCameraController ─────────────────────────────────────────────────────

/**
 * OGLCameraController
 *
 * Owns one OGL Camera + one OGL Orbit.  Exposes three interaction verbs:
 *
 *   zoom(delta)            — dolly camera forward/back by `delta` world units
 *   pan(dx, dy)            — translate camera + orbit target by (dx, dy) world units
 *   focusOnCell(cell)      — smoothly frame a CellParams (or any FocusCell) bbox
 *
 * Mouse/touch interaction (wheel, drag, pinch) is handled by OGL Orbit natively
 * once `mount(canvas)` is called.
 *
 * Call `update()` every rAF frame to tick the Orbit and advance lerp animations.
 *
 * @example
 *   const ctrl = new OGLCameraController({ worldScale: 0.01, fov: 45 });
 *
 *   // Set up OGL renderer + scene …
 *   await ctrl.init(gl);
 *   ctrl.mount(canvas);
 *
 *   function frame() {
 *     ctrl.update();
 *     renderer.render({ scene, camera: ctrl.camera });
 *     requestAnimationFrame(frame);
 *   }
 *   requestAnimationFrame(frame);
 *
 *   // On cell click:
 *   ctrl.focusOnCell(clickedCell);
 */
export class OGLCameraController {
  // ── Public OGL objects (available after init()) ──────────────────────────
  camera!: OGLCameraLike;
  orbit!: OGLOrbitLike;

  // ── Options ──────────────────────────────────────────────────────────────
  private readonly _opts: Required<OGLCameraControllerOptions>;

  // ── Internal state ───────────────────────────────────────────────────────
  private _lerp: LerpState = {
    active: false,
    startTime: 0,
    durationMs: 600,
    fromPos: [0, 0, 0],
    toPos: [0, 0, 0],
    fromTarget: [0, 0, 0],
    toTarget: [0, 0, 0],
  };

  private _currentPreset: CameraPreset | null = null;
  private _mounted = false;

  constructor(opts: OGLCameraControllerOptions = {}) {
    this._opts = {
      worldScale:     opts.worldScale     ?? 0.01,
      fov:            opts.fov            ?? 45,
      near:           opts.near           ?? 0.1,
      far:            opts.far            ?? 1000,
      minDistance:    opts.minDistance    ?? 1,
      maxDistance:    opts.maxDistance    ?? 500,
      ease:           opts.ease           ?? 0.25,
      inertia:        opts.inertia        ?? 0.85,
      panSpeed:       opts.panSpeed       ?? 0.1,
      zoomSpeed:      opts.zoomSpeed      ?? 1,
      focusMargin:    opts.focusMargin    ?? 2,
      focusDurationMs: opts.focusDurationMs ?? 600,
    };
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /**
   * init — dynamically imports OGL Camera + Orbit from upstream/ogl and
   * creates the camera positioned at [0, 0, 50] looking at origin.
   *
   * @param gl   WebGL rendering context (used only to satisfy OGL Camera ctor)
   * @param aspect  viewport aspect ratio (width / height). Default: 1
   */
  async init(gl: WebGLRenderingContext | WebGL2RenderingContext, aspect = 1): Promise<void> {
    // Dynamic import: upstream/ogl ships plain ES modules with no CJS wrapper.
    // We resolve relative to the upstream directory so no tsconfig path alias
    // is needed — this file is always at src/lib/CameraController.ts.
    const oglBase = new URL('../../upstream/ogl/src', import.meta.url).href;

    const { Camera } = await import(/* @vite-ignore */ `${oglBase}/core/Camera.js`);
    const { Orbit  } = await import(/* @vite-ignore */ `${oglBase}/extras/Orbit.js`);

    const { fov, near, far } = this._opts;

    this.camera = new Camera(gl, { fov, near, far, aspect }) as OGLCameraLike;
    this.camera.position.set(0, 0, 50);
    this.camera.lookAt({ x: 0, y: 0, z: 0 } as OGLVec3Like);

    this.orbit = new Orbit(this.camera, {
      ease:        this._opts.ease,
      inertia:     this._opts.inertia,
      enablePan:   true,
      enableZoom:  true,
      enableRotate: true,
      panSpeed:    this._opts.panSpeed,
      zoomSpeed:   this._opts.zoomSpeed,
      minDistance: this._opts.minDistance,
      maxDistance: this._opts.maxDistance,
      // Start with no DOM element so we don't accidentally listen to `document`.
      // Call mount(canvas) to attach to the right element.
      element:     document.createElement('div'), // dummy; replaced in mount()
    }) as OGLOrbitLike;

    // Detach the dummy element immediately; mount() will attach the real canvas.
    this.orbit.remove();
  }

  /**
   * mount — attach OGL Orbit input listeners to `canvas`.
   * Must be called after init().
   */
  async mount(canvas: HTMLCanvasElement): Promise<void> {
    if (this._mounted) this.unmount();

    const oglBase = new URL('../../upstream/ogl/src', import.meta.url).href;
    const { Orbit } = await import(/* @vite-ignore */ `${oglBase}/extras/Orbit.js`);

    // Re-create Orbit with the real canvas element so event listeners target
    // only the canvas, not the whole document.
    const { fov } = this._opts;
    void fov; // used by camera already

    // Preserve current camera position / target before re-creating orbit.
    const prevPos    = [this.camera.position.x, this.camera.position.y, this.camera.position.z] as [number,number,number];
    const prevTarget = [this.orbit.target.x, this.orbit.target.y, this.orbit.target.z] as [number,number,number];

    this.orbit = new Orbit(this.camera, {
      element:      canvas,
      ease:         this._opts.ease,
      inertia:      this._opts.inertia,
      enablePan:    true,
      enableZoom:   true,
      enableRotate: true,
      panSpeed:     this._opts.panSpeed,
      zoomSpeed:    this._opts.zoomSpeed,
      minDistance:  this._opts.minDistance,
      maxDistance:  this._opts.maxDistance,
    }) as OGLOrbitLike;

    // Restore position and resync orbit's internal spherical coords.
    this.camera.position.set(...prevPos);
    this.orbit.target.set(...prevTarget);
    this.orbit.forcePosition();

    this._mounted = true;
  }

  /** unmount — remove Orbit event listeners. */
  unmount(): void {
    if (this.orbit) this.orbit.remove();
    this._mounted = false;
  }

  // ── Per-frame update ──────────────────────────────────────────────────────

  /**
   * update — call every requestAnimationFrame tick.
   * Ticks the Orbit (applies mouse/touch input + inertia) and advances any
   * in-progress focus lerp animation.
   */
  update(): void {
    if (!this.camera || !this.orbit) return;

    // Advance focus lerp
    if (this._lerp.active) {
      const now = performance.now();
      const t = Math.min(1, (now - this._lerp.startTime) / this._lerp.durationMs);
      const ease = OGLCameraController._easeInOut(t);

      const pos = OGLCameraController._lerpV3(this._lerp.fromPos, this._lerp.toPos, ease);
      const tgt = OGLCameraController._lerpV3(this._lerp.fromTarget, this._lerp.toTarget, ease);

      this.camera.position.set(...pos);
      this.orbit.target.set(...tgt);
      this.orbit.forcePosition();

      if (t >= 1) this._lerp.active = false;
    }

    this.orbit.update();
  }

  // ── Interaction verbs ─────────────────────────────────────────────────────

  /**
   * zoom — dolly the camera along the view axis by `delta` world units.
   * Positive delta = zoom in (camera moves toward target).
   * Negative delta = zoom out.
   *
   * Clamps result to [minDistance, maxDistance] from the orbit target.
   */
  zoom(delta: number): void {
    if (!this.camera || !this.orbit) return;

    const tx = this.orbit.target.x;
    const ty = this.orbit.target.y;
    const tz = this.orbit.target.z;

    const dx = this.camera.position.x - tx;
    const dy = this.camera.position.y - ty;
    const dz = this.camera.position.z - tz;

    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist === 0) return;

    // Direction from target to camera (unit vector)
    const invDist = 1 / dist;
    const ux = dx * invDist;
    const uy = dy * invDist;
    const uz = dz * invDist;

    // New distance after dolly
    const newDist = Math.max(
      this._opts.minDistance,
      Math.min(this._opts.maxDistance, dist - delta),
    );

    this.camera.position.set(
      tx + ux * newDist,
      ty + uy * newDist,
      tz + uz * newDist,
    );
    this.orbit.forcePosition();
  }

  /**
   * pan — translate the camera and orbit target together by (dx, dy) in
   * world-space XY (no depth change).
   *
   * Positive dx = right, positive dy = up (matches OpenGL convention).
   */
  pan(dx: number, dy: number): void {
    if (!this.camera || !this.orbit) return;

    this.camera.position.x += dx;
    this.camera.position.y += dy;
    this.orbit.target.x   += dx;
    this.orbit.target.y   += dy;
    this.orbit.forcePosition();
  }

  /**
   * focusOnCell — smoothly animate the camera to frame the given cell's bbox.
   *
   * The bbox is given in cell-space (pixels). It is converted to world-space
   * using `worldScale`, then the camera is positioned at a Z distance that
   * exactly fits the bbox vertically in the frustum with `focusMargin` padding.
   *
   * @param cell     Any object with a `bbox: { x, y, w, h }` field.
   * @param duration Override animation duration in ms (default: focusDurationMs)
   */
  focusOnCell(cell: FocusCell, duration?: number): void {
    if (!this.camera || !this.orbit) return;

    const s = this._opts.worldScale;
    const margin = this._opts.focusMargin;

    // Cell centre in world space
    const worldCx = (cell.bbox.x + cell.bbox.w * 0.5) * s;
    const worldCy = (cell.bbox.y + cell.bbox.h * 0.5) * s;

    // Half-height of bbox in world space (with margin)
    const halfH = (cell.bbox.h * 0.5) * s + margin;

    // Distance from target at which the half-height fills the frustum vertically.
    // tan(fov/2) = halfH / dist  →  dist = halfH / tan(fov/2)
    const fovRad = (this.camera.fov * Math.PI) / 180;
    const dist = halfH / Math.tan(fovRad * 0.5);

    // Clamp to orbit limits
    const clampedDist = Math.max(this._opts.minDistance, Math.min(this._opts.maxDistance, dist));

    // Snapshot current pose
    const fromPos: [number, number, number] = [
      this.camera.position.x,
      this.camera.position.y,
      this.camera.position.z,
    ];
    const fromTarget: [number, number, number] = [
      this.orbit.target.x,
      this.orbit.target.y,
      this.orbit.target.z,
    ];

    // Target pose: camera directly in front of cell centre
    const toTarget: [number, number, number] = [worldCx, worldCy, 0];
    const toPos: [number, number, number]    = [worldCx, worldCy, clampedDist];

    const ms = duration ?? this._opts.focusDurationMs;

    this._lerp = {
      active:      true,
      startTime:   performance.now(),
      durationMs:  ms,
      fromPos,
      toPos,
      fromTarget,
      toTarget,
    };

    console.debug(
      `[CameraController] focusOnCell "${cell.cell_id}" → world(${worldCx.toFixed(2)}, ${worldCy.toFixed(2)}) dist=${clampedDist.toFixed(2)}`,
    );
  }

  // ── AT Preset API (backwards compat with xiaodi #67) ─────────────────────

  getPreset(scene: string): CameraPreset | undefined {
    return CAMERA_PRESETS[scene];
  }

  /**
   * applyPreset — apply an AT camera preset to the OGL camera.
   * If the OGL camera hasn't been init()-ed yet, just stores the preset and
   * returns it (same behaviour as the original static controller).
   */
  applyPreset(scene: string): CameraPreset | null {
    const preset = CAMERA_PRESETS[scene];
    if (!preset) {
      console.warn(`[CameraController] No preset for scene: "${scene}"`);
      return null;
    }
    this._currentPreset = preset;

    if (this.camera) {
      if (preset.position) {
        this.camera.position.set(...preset.position);
      }
      if (preset.fov !== undefined) {
        this.camera.fov = preset.fov;
        this.camera.perspective({ fov: preset.fov });
      }
      if (preset.lookAt && this.orbit) {
        this.orbit.target.set(...preset.lookAt);
        this.orbit.forcePosition();
      }
    }
    return preset;
  }

  getCurrent(): CameraPreset | null {
    return this._currentPreset;
  }

  listScenes(): string[] {
    return Object.keys(CAMERA_PRESETS);
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  /** Smooth-step ease in-out (cubic). */
  private static _easeInOut(t: number): number {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
  }

  private static _lerpV3(
    a: [number, number, number],
    b: [number, number, number],
    t: number,
  ): [number, number, number] {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  }
}

// ─── Default singleton (backwards compat) ────────────────────────────────────
// Consumers that just need the preset API can continue to import this default.
// Consumers that need zoom/pan/focus must use `new OGLCameraController()` and
// call `await ctrl.init(gl)` + `ctrl.mount(canvas)` themselves.

export default new OGLCameraController();
