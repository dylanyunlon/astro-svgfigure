/**
 * renderers/hydra-css.ts — CSS-driven GPU animation layer: HydraCSS, HydraObject, FXController, FXScroll
 */

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Vec2 { x: number; y: number; }
export interface CSSTransform {
  translateX?: number | string;
  translateY?: number | string;
  translateZ?: number | string;
  rotateX?: number;  // degrees
  rotateY?: number;
  rotateZ?: number;
  scaleX?: number;
  scaleY?: number;
  scale?: number;
  skewX?: number;
  skewY?: number;
  perspective?: number;
}

export type HydraProp = CSSTransform & {
  opacity?: number;
  filter?: string;
  clipPath?: string;
  willChange?: string;
};

// ─── HydraObject ─────────────────────────────────────────────────────────────

export interface HydraObjectOptions {
  id?: string;
  el: HTMLElement;
  initialProps?: HydraProp;
  gpu?: boolean;  // force GPU compositing layer
}

/**
 * Lightweight CSS-transform wrapper for single DOM elements.
 * Batches all prop changes and flushes them in a single rAF.
 */
export class HydraObject {
  readonly el: HTMLElement;
  readonly id: string;
  private props: HydraProp;
  private dirty = false;
  private animId = 0;

  constructor(opts: HydraObjectOptions) {
    this.el = opts.el;
    this.id = opts.id ?? `hydra-${Math.random().toString(36).slice(2)}`;
    this.props = { ...(opts.initialProps ?? {}) };

    if (opts.gpu !== false) {
      this.el.style.willChange = opts.initialProps?.willChange ?? 'transform, opacity';
      this.el.style.backfaceVisibility = 'hidden';
    }

    if (opts.initialProps) this.apply(opts.initialProps);
  }

  set(props: Partial<HydraProp>): void {
    Object.assign(this.props, props);
    this.scheduleFlushed();
  }

  setImmediate(props: Partial<HydraProp>): void {
    Object.assign(this.props, props);
    this.flush();
  }

  get(key: keyof HydraProp): HydraProp[keyof HydraProp] | undefined {
    return this.props[key];
  }

  private scheduleFlushed(): void {
    if (this.dirty) return;
    this.dirty = true;
    this.animId = requestAnimationFrame(() => { this.flush(); this.dirty = false; });
  }

  private flush(): void {
    this.apply(this.props);
  }

  private apply(props: HydraProp): void {
    const parts: string[] = [];

    if (props.perspective !== undefined) parts.push(`perspective(${props.perspective}px)`);
    if (props.translateX !== undefined) parts.push(`translateX(${numPx(props.translateX)})`);
    if (props.translateY !== undefined) parts.push(`translateY(${numPx(props.translateY)})`);
    if (props.translateZ !== undefined) parts.push(`translateZ(${numPx(props.translateZ)})`);
    if (props.rotateX !== undefined) parts.push(`rotateX(${props.rotateX}deg)`);
    if (props.rotateY !== undefined) parts.push(`rotateY(${props.rotateY}deg)`);
    if (props.rotateZ !== undefined) parts.push(`rotateZ(${props.rotateZ}deg)`);
    if (props.scale !== undefined) parts.push(`scale(${props.scale})`);
    if (props.scaleX !== undefined) parts.push(`scaleX(${props.scaleX})`);
    if (props.scaleY !== undefined) parts.push(`scaleY(${props.scaleY})`);
    if (props.skewX !== undefined) parts.push(`skewX(${props.skewX}deg)`);
    if (props.skewY !== undefined) parts.push(`skewY(${props.skewY}deg)`);

    if (parts.length) this.el.style.transform = parts.join(' ');
    if (props.opacity !== undefined) this.el.style.opacity = String(props.opacity);
    if (props.filter !== undefined) this.el.style.filter = props.filter;
    if (props.clipPath !== undefined) this.el.style.clipPath = props.clipPath;
  }

  dispose(): void {
    cancelAnimationFrame(this.animId);
    this.el.style.willChange = '';
  }
}

// ─── HydraCSS ─────────────────────────────────────────────────────────────────

export interface HydraCSSOptions {
  root?: HTMLElement;
  useCompositor?: boolean;
}

/**
 * Scene graph of HydraObjects with per-frame update loop.
 */
export class HydraCSS {
  private objects = new Map<string, HydraObject>();
  private updaters = new Map<string, (dt: number, obj: HydraObject) => void>();
  private animId = 0;
  private running = false;
  private lastTime = 0;
  readonly root: HTMLElement;

  constructor(opts: HydraCSSOptions = {}) {
    this.root = opts.root ?? (typeof document !== 'undefined' ? document.body : {} as HTMLElement);
  }

  add(opts: HydraObjectOptions): HydraObject {
    const obj = new HydraObject(opts);
    this.objects.set(obj.id, obj);
    return obj;
  }

  remove(id: string): void {
    const obj = this.objects.get(id);
    obj?.dispose();
    this.objects.delete(id);
    this.updaters.delete(id);
  }

  addUpdater(id: string, fn: (dt: number, obj: HydraObject) => void): void {
    this.updaters.set(id, fn);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    const loop = (now: number) => {
      if (!this.running) return;
      const dt = this.lastTime ? (now - this.lastTime) / 1000 : 0;
      this.lastTime = now;
      for (const [id, fn] of this.updaters) {
        const obj = this.objects.get(id);
        if (obj) fn(dt, obj);
      }
      this.animId = requestAnimationFrame(loop);
    };
    this.animId = requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.animId);
  }

  get(id: string): HydraObject | undefined { return this.objects.get(id); }
  get objectCount(): number { return this.objects.size; }

  dispose(): void {
    this.stop();
    for (const obj of this.objects.values()) obj.dispose();
    this.objects.clear();
    this.updaters.clear();
  }
}

// ─── FXController ─────────────────────────────────────────────────────────────

export interface FXControllerOptions {
  container: HTMLElement;
  autoStart?: boolean;
}

export type FXPhase = 'idle' | 'intro' | 'active' | 'outro' | 'done';

export interface FXControllerEvent {
  type: 'phaseChange';
  from: FXPhase;
  to: FXPhase;
}

/**
 * Orchestrates multiple HydraObjects through timed phases (intro → active → outro).
 */
export class FXController {
  private readonly hydra: HydraCSS;
  private phase: FXPhase = 'idle';
  private phaseTimeout = 0;
  private listeners: Array<(e: FXControllerEvent) => void> = [];
  readonly container: HTMLElement;

  constructor(opts: FXControllerOptions) {
    this.container = opts.container;
    this.hydra = new HydraCSS({ root: opts.container });
    if (opts.autoStart) this.hydra.start();
  }

  add(opts: HydraObjectOptions): HydraObject { return this.hydra.add(opts); }
  remove(id: string): void { this.hydra.remove(id); }
  get(id: string): HydraObject | undefined { return this.hydra.get(id); }

  addUpdater(id: string, fn: (dt: number, obj: HydraObject) => void): void {
    this.hydra.addUpdater(id, fn);
  }

  playIntro(durationMs = 600): Promise<void> {
    return this.transitionPhase('intro', 'active', durationMs);
  }

  playOutro(durationMs = 400): Promise<void> {
    return this.transitionPhase('outro', 'done', durationMs);
  }

  on(fn: (e: FXControllerEvent) => void): () => void {
    this.listeners.push(fn);
    return () => { this.listeners = this.listeners.filter(l => l !== fn); };
  }

  private transitionPhase(from: FXPhase, to: FXPhase, ms: number): Promise<void> {
    return new Promise(res => {
      this.setPhase(from);
      clearTimeout(this.phaseTimeout);
      this.phaseTimeout = window.setTimeout(() => { this.setPhase(to); res(); }, ms);
    });
  }

  private setPhase(p: FXPhase): void {
    const prev = this.phase;
    this.phase = p;
    this.container.dataset.fxPhase = p;
    this.listeners.forEach(fn => fn({ type: 'phaseChange', from: prev, to: p }));
  }

  start(): void { this.hydra.start(); }
  stop(): void { this.hydra.stop(); }
  get currentPhase(): FXPhase { return this.phase; }

  dispose(): void {
    clearTimeout(this.phaseTimeout);
    this.hydra.dispose();
  }
}

// ─── FXScroll ─────────────────────────────────────────────────────────────────

export interface FXScrollBinding {
  object: HydraObject;
  /** Map of scroll progress (0..1) → prop value */
  tracks: Partial<Record<keyof HydraProp, [number, number]>>;  // [startVal, endVal]
  /** Scroll range [start, end] in 0..1 domain */
  range?: [number, number];
  easing?: (t: number) => number;
}

export interface FXScrollOptions {
  scrollTarget?: HTMLElement | Window;
  onScrollProgress?: (p: number) => void;
}

/**
 * Ties CSS transform props to scroll position using scrub animation.
 */
export class FXScroll {
  private bindings: FXScrollBinding[] = [];
  private progress = 0;
  private readonly opts: FXScrollOptions;
  private boundHandler: () => void;

  constructor(opts: FXScrollOptions = {}) {
    this.opts = opts;
    this.boundHandler = this.handleScroll.bind(this);
  }

  mount(): void {
    const t = this.opts.scrollTarget ?? window;
    (t as EventTarget).addEventListener('scroll', this.boundHandler, { passive: true });
    this.handleScroll(); // sync on mount
  }

  unmount(): void {
    const t = this.opts.scrollTarget ?? window;
    (t as EventTarget).removeEventListener('scroll', this.boundHandler);
  }

  bind(binding: FXScrollBinding): void {
    this.bindings.push(binding);
  }

  unbind(obj: HydraObject): void {
    this.bindings = this.bindings.filter(b => b.object !== obj);
  }

  private handleScroll(): void {
    const t = this.opts.scrollTarget ?? window;
    const scrollY = t === window ? window.scrollY : (t as HTMLElement).scrollTop;
    const maxScroll = t === window
      ? document.documentElement.scrollHeight - window.innerHeight
      : (t as HTMLElement).scrollHeight - (t as HTMLElement).clientHeight;
    this.progress = maxScroll > 0 ? scrollY / maxScroll : 0;
    this.opts.onScrollProgress?.(this.progress);
    this.applyBindings();
  }

  private applyBindings(): void {
    for (const binding of this.bindings) {
      const [rangeStart, rangeEnd] = binding.range ?? [0, 1];
      let t = (this.progress - rangeStart) / (rangeEnd - rangeStart);
      t = Math.max(0, Math.min(1, t));
      if (binding.easing) t = binding.easing(t);

      const patch: Partial<HydraProp> = {};
      for (const [key, [from, to]] of Object.entries(binding.tracks)) {
        (patch as any)[key] = from + (to - from) * t;
      }
      binding.object.set(patch);
    }
  }

  get scrollProgress(): number { return this.progress; }

  dispose(): void { this.unmount(); }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function numPx(v: number | string): string {
  return typeof v === 'number' ? `${v}px` : v;
}
