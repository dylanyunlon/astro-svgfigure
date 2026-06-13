/**
 * xr-system.ts — WebXR / WebVR abstraction layer
 *
 * Covers:
 *   XRDeviceManager    — session lifecycle, device enumeration, frame loop
 *   VRInput            — unified gamepad + XR input-source polling
 *   VRAbstractHand     — joint-pose hand tracking abstraction
 *   VRHandFingerTip    — per-finger-tip pose & pinch detection
 *   VRControllerBeam   — ray-cast pointer beam renderer (WebGL lines)
 *   UserInputVR        — high-level action map (select / squeeze / thumbstick)
 *   WEBVRPolyfill      — legacy WebVR → WebXR shim
 *
 * All classes are tree-shakeable; import only what you need.
 */

// ─── Types ────────────────────────────────────────────────────────────────────

export interface XRSessionConfig {
  mode: XRSessionMode;
  requiredFeatures?: string[];
  optionalFeatures?: string[];
  domOverlay?: { root: Element };
}

export interface XRFrameCallback {
  (time: number, frame: XRFrame): void;
}

export interface VRInputState {
  selectPressed: boolean;
  selectValue: number;
  squeezePressed: boolean;
  squeezeValue: number;
  thumbstick: { x: number; y: number };
  touchpad: { x: number; y: number };
  buttonA: boolean;
  buttonB: boolean;
}

export interface HandJointPose {
  position: Float32Array; // [x, y, z]
  orientation: Float32Array; // [qx, qy, qz, qw]
  radius: number;
}

export interface FingerTipState {
  joint: XRHandJoint;
  pose: HandJointPose | null;
  pinching: boolean;
  pinchStrength: number; // 0–1
}

export interface BeamHit {
  point: Float32Array; // [x, y, z]
  normal: Float32Array;
  distance: number;
  object: EventTarget | null;
}

// ─── XRDeviceManager ─────────────────────────────────────────────────────────

export class XRDeviceManager extends EventTarget {
  private static _instance: XRDeviceManager | null = null;

  private _xr: XRSystem | null = null;
  private _session: XRSession | null = null;
  private _rafId: number = 0;
  private _callbacks: Set<XRFrameCallback> = new Set();
  private _referenceSpace: XRReferenceSpace | XRBoundedReferenceSpace | null = null;
  private _baseLayer: XRWebGLLayer | null = null;
  private _supported: boolean = false;
  private _sessionMode: XRSessionMode = 'inline';

  static getInstance(): XRDeviceManager {
    if (!XRDeviceManager._instance) {
      XRDeviceManager._instance = new XRDeviceManager();
    }
    return XRDeviceManager._instance;
  }

  constructor() {
    super();
    this._xr = typeof navigator !== 'undefined' && 'xr' in navigator
      ? (navigator as any).xr as XRSystem
      : null;
  }

  get session(): XRSession | null { return this._session; }
  get referenceSpace(): XRReferenceSpace | XRBoundedReferenceSpace | null { return this._referenceSpace; }
  get baseLayer(): XRWebGLLayer | null { return this._baseLayer; }
  get isSupported(): boolean { return this._supported; }
  get isPresenting(): boolean { return this._session !== null; }

  async checkSupport(mode: XRSessionMode = 'immersive-vr'): Promise<boolean> {
    if (!this._xr) return false;
    try {
      this._supported = await this._xr.isSessionSupported(mode);
    } catch {
      this._supported = false;
    }
    return this._supported;
  }

  async requestSession(
    gl: WebGL2RenderingContext,
    config: XRSessionConfig = { mode: 'immersive-vr' }
  ): Promise<XRSession> {
    if (!this._xr) throw new Error('WebXR not available');
    if (this._session) return this._session;

    const opts: XRSessionInit = {
      requiredFeatures: config.requiredFeatures ?? ['local-floor'],
      optionalFeatures: config.optionalFeatures ?? ['bounded-floor', 'hand-tracking', 'dom-overlay'],
    };
    if (config.domOverlay) (opts as any).domOverlay = config.domOverlay;

    this._session = await this._xr.requestSession(config.mode, opts);
    this._sessionMode = config.mode;

    await gl.makeXRCompatible();
    this._baseLayer = new XRWebGLLayer(this._session, gl);
    this._session.updateRenderState({ baseLayer: this._baseLayer });

    try {
      this._referenceSpace = await this._session.requestReferenceSpace('local-floor');
    } catch {
      this._referenceSpace = await this._session.requestReferenceSpace('local');
    }

    this._session.addEventListener('end', () => this._onSessionEnd());
    this._startFrameLoop();

    this.dispatchEvent(new CustomEvent('sessionstart', { detail: { session: this._session } }));
    return this._session;
  }

  private _startFrameLoop(): void {
    const loop = (time: number, frame: XRFrame) => {
      this._rafId = this._session!.requestAnimationFrame(loop);
      this._callbacks.forEach(cb => cb(time, frame));
    };
    this._rafId = this._session!.requestAnimationFrame(loop);
  }

  addFrameCallback(cb: XRFrameCallback): void { this._callbacks.add(cb); }
  removeFrameCallback(cb: XRFrameCallback): void { this._callbacks.delete(cb); }

  async endSession(): Promise<void> {
    if (!this._session) return;
    await this._session.end();
  }

  private _onSessionEnd(): void {
    this._session = null;
    this._referenceSpace = null;
    this._baseLayer = null;
    this._callbacks.clear();
    this.dispatchEvent(new Event('sessionend'));
  }

  /** Enumerate available XR input sources from the active session. */
  getInputSources(): XRInputSource[] {
    if (!this._session) return [];
    return Array.from(this._session.inputSources);
  }

  dispose(): void {
    this.endSession();
    XRDeviceManager._instance = null;
  }
}

// ─── VRInput ─────────────────────────────────────────────────────────────────

export class VRInput {
  private _source: XRInputSource;
  private _state: VRInputState;
  private _prevState: VRInputState;

  constructor(source: XRInputSource) {
    this._source = source;
    this._state = VRInput._emptyState();
    this._prevState = VRInput._emptyState();
  }

  private static _emptyState(): VRInputState {
    return {
      selectPressed: false,
      selectValue: 0,
      squeezePressed: false,
      squeezeValue: 0,
      thumbstick: { x: 0, y: 0 },
      touchpad: { x: 0, y: 0 },
      buttonA: false,
      buttonB: false,
    };
  }

  get source(): XRInputSource { return this._source; }
  get handedness(): XRHandedness { return this._source.handedness; }
  get state(): Readonly<VRInputState> { return this._state; }
  get prevState(): Readonly<VRInputState> { return this._prevState; }

  /** Poll the underlying Gamepad each frame. */
  update(): void {
    this._prevState = { ...this._state, thumbstick: { ...this._state.thumbstick }, touchpad: { ...this._state.touchpad } };
    const gp = this._source.gamepad;
    if (!gp) return;

    // Standard XR Gamepad button layout
    this._state.selectPressed = gp.buttons[0]?.pressed ?? false;
    this._state.selectValue  = gp.buttons[0]?.value  ?? 0;
    this._state.squeezePressed = gp.buttons[1]?.pressed ?? false;
    this._state.squeezeValue  = gp.buttons[1]?.value  ?? 0;
    this._state.buttonA = gp.buttons[4]?.pressed ?? false;
    this._state.buttonB = gp.buttons[5]?.pressed ?? false;

    if (gp.axes.length >= 4) {
      this._state.thumbstick = { x: gp.axes[2], y: gp.axes[3] };
      this._state.touchpad   = { x: gp.axes[0], y: gp.axes[1] };
    } else if (gp.axes.length >= 2) {
      this._state.thumbstick = { x: gp.axes[0], y: gp.axes[1] };
    }
  }

  isSelectJustPressed(): boolean  { return this._state.selectPressed  && !this._prevState.selectPressed; }
  isSelectJustReleased(): boolean { return !this._state.selectPressed && this._prevState.selectPressed; }
  isSqueezeJustPressed(): boolean { return this._state.squeezePressed && !this._prevState.squeezePressed; }

  /** Deadzone-filtered thumbstick. */
  getThumbstick(deadzone = 0.12): { x: number; y: number } {
    const { x, y } = this._state.thumbstick;
    const len = Math.sqrt(x * x + y * y);
    if (len < deadzone) return { x: 0, y: 0 };
    const scale = (len - deadzone) / (1 - deadzone) / len;
    return { x: x * scale, y: y * scale };
  }
}

// ─── VRAbstractHand ──────────────────────────────────────────────────────────

const FINGER_TIPS: XRHandJoint[] = [
  'thumb-tip',
  'index-finger-tip',
  'middle-finger-tip',
  'ring-finger-tip',
  'pinky-finger-tip',
];

export class VRAbstractHand {
  private _source: XRInputSource;
  private _handedness: XRHandedness;
  private _joints: Map<XRHandJoint, HandJointPose | null> = new Map();
  private _fingerTips: VRHandFingerTip[] = [];

  constructor(source: XRInputSource) {
    if (!source.hand) throw new Error('XRInputSource does not have hand tracking');
    this._source = source;
    this._handedness = source.handedness;
    this._fingerTips = FINGER_TIPS.map(j => new VRHandFingerTip(j));
  }

  get handedness(): XRHandedness { return this._handedness; }
  get fingerTips(): readonly VRHandFingerTip[] { return this._fingerTips; }
  get indexTip(): VRHandFingerTip { return this._fingerTips[1]; }
  get thumbTip(): VRHandFingerTip  { return this._fingerTips[0]; }

  update(frame: XRFrame, refSpace: XRReferenceSpace | XRBoundedReferenceSpace): void {
    const hand = this._source.hand;
    if (!hand) return;

    for (const [joint] of hand) {
      const pose = frame.getJointPose(hand.get(joint)!, refSpace);
      if (pose) {
        const t = pose.transform;
        this._joints.set(joint, {
          position:    new Float32Array([t.position.x, t.position.y, t.position.z]),
          orientation: new Float32Array([t.orientation.x, t.orientation.y, t.orientation.z, t.orientation.w]),
          radius: pose.radius ?? 0.005,
        });
      } else {
        this._joints.set(joint, null);
      }
    }

    for (const tip of this._fingerTips) {
      const pose = this._joints.get(tip.joint) ?? null;
      tip.update(pose, this._joints.get('thumb-tip') ?? null);
    }
  }

  getJointPose(joint: XRHandJoint): HandJointPose | null {
    return this._joints.get(joint) ?? null;
  }

  /** Rough palm-normal estimation from wrist + index-metacarpal. */
  getPalmNormal(): Float32Array {
    const wrist = this._joints.get('wrist');
    const meta  = this._joints.get('index-finger-metacarpal');
    if (!wrist || !meta) return new Float32Array([0, 1, 0]);
    const dx = meta.position[0] - wrist.position[0];
    const dy = meta.position[1] - wrist.position[1];
    const dz = meta.position[2] - wrist.position[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    return new Float32Array([dx / len, dy / len, dz / len]);
  }
}

// ─── VRHandFingerTip ─────────────────────────────────────────────────────────

export class VRHandFingerTip {
  readonly joint: XRHandJoint;
  private _pose: HandJointPose | null = null;
  private _pinching: boolean = false;
  private _pinchStrength: number = 0;

  private static PINCH_ENTER_DIST = 0.025; // metres
  private static PINCH_EXIT_DIST  = 0.035;

  constructor(joint: XRHandJoint) {
    this.joint = joint;
  }

  get pose(): HandJointPose | null { return this._pose; }
  get pinching(): boolean { return this._pinching; }
  get pinchStrength(): number { return this._pinchStrength; }

  update(pose: HandJointPose | null, thumbPose: HandJointPose | null): void {
    this._pose = pose;
    if (!pose || !thumbPose || this.joint === 'thumb-tip') {
      this._pinchStrength = 0;
      this._pinching = false;
      return;
    }

    const dx = pose.position[0] - thumbPose.position[0];
    const dy = pose.position[1] - thumbPose.position[1];
    const dz = pose.position[2] - thumbPose.position[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    const enter = VRHandFingerTip.PINCH_ENTER_DIST;
    const exit  = VRHandFingerTip.PINCH_EXIT_DIST;
    this._pinchStrength = Math.max(0, Math.min(1, 1 - (dist - enter) / (exit - enter)));

    if (this._pinching) {
      if (dist > exit) this._pinching = false;
    } else {
      if (dist < enter) this._pinching = true;
    }
  }

  /** World-space position as [x, y, z] or null. */
  get worldPosition(): Float32Array | null {
    return this._pose?.position ?? null;
  }
}

// ─── VRControllerBeam ────────────────────────────────────────────────────────

export interface BeamOptions {
  color?: [number, number, number, number]; // RGBA 0–1
  maxLength?: number;
  width?: number;
}

export class VRControllerBeam {
  private _source: XRInputSource;
  private _color: [number, number, number, number];
  private _maxLength: number;
  private _width: number;
  private _active: boolean = true;
  private _lastHit: BeamHit | null = null;

  // WebGL resources (lazy-created on first draw)
  private _vao: WebGLVertexArrayObject | null = null;
  private _vbo: WebGLBuffer | null = null;
  private _program: WebGLProgram | null = null;
  private _gl: WebGL2RenderingContext | null = null;

  constructor(source: XRInputSource, opts: BeamOptions = {}) {
    this._source = source;
    this._color     = opts.color     ?? [0.4, 0.8, 1.0, 0.85];
    this._maxLength = opts.maxLength ?? 10;
    this._width     = opts.width     ?? 0.004;
  }

  get active(): boolean { return this._active; }
  set active(v: boolean) { this._active = v; }
  get lastHit(): BeamHit | null { return this._lastHit; }

  private _ensureGL(gl: WebGL2RenderingContext): void {
    if (this._gl === gl) return;
    this._gl = gl;

    const vert = `#version 300 es
      precision highp float;
      in vec3 aPos;
      uniform mat4 uMVP;
      void main() { gl_Position = uMVP * vec4(aPos, 1.0); }`;
    const frag = `#version 300 es
      precision mediump float;
      uniform vec4 uColor;
      out vec4 fragColor;
      void main() { fragColor = uColor; }`;

    const compile = (src: string, type: number) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src); gl.compileShader(sh); return sh;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(vert, gl.VERTEX_SHADER));
    gl.attachShader(prog, compile(frag, gl.FRAGMENT_SHADER));
    gl.linkProgram(prog);
    this._program = prog;

    this._vbo = gl.createBuffer();
    this._vao = gl.createVertexArray();
    gl.bindVertexArray(this._vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._vbo);
    gl.bufferData(gl.ARRAY_BUFFER, 6 * 4, gl.DYNAMIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'aPos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  /**
   * Render the beam for the current frame.
   * @param gl   WebGL2 context
   * @param frame XRFrame
   * @param refSpace reference space
   * @param viewProjection column-major 4×4 MVP matrix (Float32Array[16])
   */
  render(
    gl: WebGL2RenderingContext,
    frame: XRFrame,
    refSpace: XRReferenceSpace | XRBoundedReferenceSpace,
    viewProjection: Float32Array,
  ): void {
    if (!this._active) return;
    this._ensureGL(gl);

    const pose = frame.getPose(this._source.targetRaySpace, refSpace);
    if (!pose) return;

    const t = pose.transform;
    const ox = t.position.x, oy = t.position.y, oz = t.position.z;
    // forward direction from quaternion
    const qx = t.orientation.x, qy = t.orientation.y,
          qz = t.orientation.z, qw = t.orientation.w;
    const fx = 2 * (qx * qz + qw * qy);
    const fy = 2 * (qy * qz - qw * qx);
    const fz = 1 - 2 * (qx * qx + qy * qy);

    const ex = ox + fx * this._maxLength;
    const ey = oy + fy * this._maxLength;
    const ez = oz + fz * this._maxLength;

    const data = new Float32Array([ox, oy, oz, ex, ey, ez]);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._vbo!);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);

    gl.useProgram(this._program!);
    gl.uniformMatrix4fv(gl.getUniformLocation(this._program!, 'uMVP'), false, viewProjection);
    gl.uniform4fv(gl.getUniformLocation(this._program!, 'uColor'), this._color);
    gl.bindVertexArray(this._vao!);
    gl.lineWidth(Math.max(1, this._width * 1000));
    gl.drawArrays(gl.LINES, 0, 2);
    gl.bindVertexArray(null);
  }

  /** Simple sphere-intersection ray cast against a list of AABBs/spheres. */
  castRay(
    frame: XRFrame,
    refSpace: XRReferenceSpace | XRBoundedReferenceSpace,
    targets: Array<{ center: Float32Array; radius: number; ref: EventTarget }>,
  ): BeamHit | null {
    const pose = frame.getPose(this._source.targetRaySpace, refSpace);
    if (!pose) { this._lastHit = null; return null; }

    const t = pose.transform;
    const ox = t.position.x, oy = t.position.y, oz = t.position.z;
    const qx = t.orientation.x, qy = t.orientation.y,
          qz = t.orientation.z, qw = t.orientation.w;
    const fx = 2 * (qx * qz + qw * qy);
    const fy = 2 * (qy * qz - qw * qx);
    const fz = 1 - 2 * (qx * qx + qy * qy);

    let nearest: BeamHit | null = null;

    for (const tgt of targets) {
      const cx = tgt.center[0] - ox;
      const cy = tgt.center[1] - oy;
      const cz = tgt.center[2] - oz;
      const b = fx * cx + fy * cy + fz * cz;
      const c = cx * cx + cy * cy + cz * cz - tgt.radius * tgt.radius;
      const disc = b * b - c;
      if (disc < 0) continue;
      const dist = b - Math.sqrt(disc);
      if (dist < 0 || dist > this._maxLength) continue;
      if (!nearest || dist < nearest.distance) {
        nearest = {
          point: new Float32Array([ox + fx * dist, oy + fy * dist, oz + fz * dist]),
          normal: new Float32Array([
            (ox + fx * dist - tgt.center[0]) / tgt.radius,
            (oy + fy * dist - tgt.center[1]) / tgt.radius,
            (oz + fz * dist - tgt.center[2]) / tgt.radius,
          ]),
          distance: dist,
          object: tgt.ref,
        };
      }
    }

    this._lastHit = nearest;
    return nearest;
  }

  dispose(gl: WebGL2RenderingContext): void {
    if (this._vao) gl.deleteVertexArray(this._vao);
    if (this._vbo) gl.deleteBuffer(this._vbo);
    if (this._program) gl.deleteProgram(this._program);
    this._vao = this._vbo = this._program = null;
  }
}

// ─── UserInputVR ─────────────────────────────────────────────────────────────

export interface VRActionMap {
  'select':      boolean;
  'selectDown':  boolean;
  'selectUp':    boolean;
  'squeeze':     boolean;
  'squeezeDown': boolean;
  'thumbstick':  { x: number; y: number };
  'buttonA':     boolean;
  'buttonB':     boolean;
  'pinch':       boolean; // hand-tracking index pinch
}

export class UserInputVR extends EventTarget {
  private _manager: XRDeviceManager;
  private _inputs: Map<XRInputSource, VRInput> = new Map();
  private _hands:  Map<XRInputSource, VRAbstractHand> = new Map();
  private _actions: Map<'left' | 'right', VRActionMap> = new Map();

  constructor(manager: XRDeviceManager) {
    super();
    this._manager = manager;
    manager.addFrameCallback(this._onFrame.bind(this));
  }

  getActions(hand: 'left' | 'right'): Readonly<VRActionMap> {
    return this._actions.get(hand) ?? {
      select: false, selectDown: false, selectUp: false,
      squeeze: false, squeezeDown: false,
      thumbstick: { x: 0, y: 0 },
      buttonA: false, buttonB: false, pinch: false,
    };
  }

  private _onFrame(_time: number, frame: XRFrame): void {
    const session = this._manager.session;
    if (!session) return;

    const refSpace = this._manager.referenceSpace;
    if (!refSpace) return;

    // Sync input source list
    const activeSources = new Set(session.inputSources);
    for (const [src] of this._inputs) {
      if (!activeSources.has(src)) { this._inputs.delete(src); this._hands.delete(src); }
    }
    for (const src of activeSources) {
      if (!this._inputs.has(src)) {
        this._inputs.set(src, new VRInput(src));
        if (src.hand) {
          try { this._hands.set(src, new VRAbstractHand(src)); } catch { /* ignore */ }
        }
      }
    }

    // Update & build action maps
    for (const [src, input] of this._inputs) {
      input.update();
      const hand = src.handedness === 'left' ? 'left' : 'right';
      const htracking = this._hands.get(src);
      if (htracking) htracking.update(frame, refSpace);

      const pinch = htracking?.indexTip.pinching ?? false;
      const ts = input.getThumbstick();

      this._actions.set(hand, {
        select:      input.state.selectPressed,
        selectDown:  input.isSelectJustPressed(),
        selectUp:    input.isSelectJustReleased(),
        squeeze:     input.state.squeezePressed,
        squeezeDown: input.isSqueezeJustPressed(),
        thumbstick:  ts,
        buttonA:     input.state.buttonA,
        buttonB:     input.state.buttonB,
        pinch,
      });

      if (input.isSelectJustPressed()) this.dispatchEvent(new CustomEvent('select', { detail: { hand, source: src } }));
      if (pinch) this.dispatchEvent(new CustomEvent('pinch', { detail: { hand, tip: htracking?.indexTip } }));
    }
  }

  dispose(): void {
    this._manager.removeFrameCallback(this._onFrame.bind(this));
    this._inputs.clear();
    this._hands.clear();
  }
}

// ─── WEBVRPolyfill ───────────────────────────────────────────────────────────

/**
 * Minimal WebVR → WebXR polyfill shim.
 * Bridges legacy `navigator.getVRDisplays()` environments into the
 * XRDeviceManager session flow.
 */
export class WEBVRPolyfill {
  private static _installed = false;

  /** Install the shim onto navigator if WebXR is absent but WebVR is present. */
  static install(): boolean {
    if (WEBVRPolyfill._installed) return true;
    if (typeof navigator === 'undefined') return false;

    const nav = navigator as any;
    if ('xr' in nav) return true; // native WebXR present, no shim needed

    if (!('getVRDisplays' in nav)) {
      console.warn('[WEBVRPolyfill] Neither WebXR nor WebVR found.');
      return false;
    }

    // Stub navigator.xr with isSessionSupported + requestSession
    nav.xr = {
      async isSessionSupported(mode: XRSessionMode): Promise<boolean> {
        if (mode !== 'immersive-vr') return false;
        try {
          const displays: any[] = await nav.getVRDisplays();
          return displays.length > 0;
        } catch { return false; }
      },
      async requestSession(_mode: XRSessionMode, _opts?: XRSessionInit): Promise<XRSession> {
        throw new Error('[WEBVRPolyfill] Full WebVR→WebXR session bridging requires a complete polyfill library (e.g. webxr-polyfill npm package).');
      },
      addEventListener: () => {},
      removeEventListener: () => {},
    };

    WEBVRPolyfill._installed = true;
    console.info('[WEBVRPolyfill] Legacy WebVR shim installed on navigator.xr');
    return true;
  }

  static get isInstalled(): boolean { return WEBVRPolyfill._installed; }
}
