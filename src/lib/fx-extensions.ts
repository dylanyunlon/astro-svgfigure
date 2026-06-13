/**
 * fx-extensions.ts — FX layer, stencil, asset control, scroll transitions, scene visibility/compositor, UI helper, DhCwa
 */

// ─── FXLayer ──────────────────────────────────────────────────────────────────

export interface FXLayerOptions {
  name: string;
  zIndex?: number;
  opacity?: number;
  blendMode?: GlobalCompositeOperation;
  visible?: boolean;
}

export class FXLayer {
  readonly name: string;
  zIndex: number;
  opacity: number;
  blendMode: GlobalCompositeOperation;
  visible: boolean;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  constructor(opts: FXLayerOptions) {
    this.name = opts.name;
    this.zIndex = opts.zIndex ?? 0;
    this.opacity = opts.opacity ?? 1;
    this.blendMode = opts.blendMode ?? 'source-over';
    this.visible = opts.visible ?? true;
  }

  mount(container: HTMLElement, width: number, height: number): HTMLCanvasElement {
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    this.canvas.style.position = 'absolute';
    this.canvas.style.inset = '0';
    this.canvas.style.zIndex = String(this.zIndex);
    this.canvas.style.opacity = String(this.opacity);
    this.canvas.style.mixBlendMode = this.blendMode;
    this.canvas.style.display = this.visible ? '' : 'none';
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    return this.canvas;
  }

  resize(w: number, h: number): void {
    if (!this.canvas) return;
    this.canvas.width = w;
    this.canvas.height = h;
  }

  clear(): void {
    this.ctx?.clearRect(0, 0, this.canvas?.width ?? 0, this.canvas?.height ?? 0);
  }

  setOpacity(o: number): void {
    this.opacity = o;
    if (this.canvas) this.canvas.style.opacity = String(o);
  }

  show(): void { this.visible = true; if (this.canvas) this.canvas.style.display = ''; }
  hide(): void { this.visible = false; if (this.canvas) this.canvas.style.display = 'none'; }

  get context(): CanvasRenderingContext2D | null { return this.ctx; }

  dispose(): void {
    this.canvas?.remove();
    this.canvas = null;
    this.ctx = null;
  }
}

// ─── FXStencil ────────────────────────────────────────────────────────────────

export interface StencilRegion {
  id: string;
  x: number; y: number; w: number; h: number;
  cornerRadius?: number;
}

export class FXStencil {
  private regions: StencilRegion[] = [];
  private dirtyFlag = false;

  addRegion(r: StencilRegion): void {
    this.regions.push(r);
    this.dirtyFlag = true;
  }

  removeRegion(id: string): void {
    this.regions = this.regions.filter(r => r.id !== id);
    this.dirtyFlag = true;
  }

  clearRegions(): void { this.regions = []; this.dirtyFlag = true; }

  apply(ctx: CanvasRenderingContext2D): void {
    ctx.save();
    ctx.beginPath();
    for (const r of this.regions) {
      const radius = r.cornerRadius ?? 0;
      if (radius > 0) {
        ctx.roundRect(r.x, r.y, r.w, r.h, radius);
      } else {
        ctx.rect(r.x, r.y, r.w, r.h);
      }
    }
    ctx.clip();
    this.dirtyFlag = false;
  }

  restore(ctx: CanvasRenderingContext2D): void {
    ctx.restore();
  }

  applyToGL(gl: WebGL2RenderingContext): void {
    gl.enable(gl.STENCIL_TEST);
    gl.clear(gl.STENCIL_BUFFER_BIT);
    gl.stencilFunc(gl.ALWAYS, 1, 0xff);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
  }

  get isDirty(): boolean { return this.dirtyFlag; }
  get regionCount(): number { return this.regions.length; }
}

// ─── FXAssetsController ───────────────────────────────────────────────────────

export type AssetKind = 'image' | 'video' | 'audio' | 'json' | 'glb' | 'wasm';

export interface FXAssetEntry {
  id: string;
  url: string;
  kind: AssetKind;
  priority?: number;
}

export interface FXAssetLoaded {
  id: string;
  kind: AssetKind;
  data: unknown;
}

export type FXAssetProgress = { loaded: number; total: number; percent: number };

export class FXAssetsController {
  private queue: FXAssetEntry[] = [];
  private cache = new Map<string, unknown>();
  private onProgressCb: ((p: FXAssetProgress) => void) | null = null;

  enqueue(entry: FXAssetEntry): void {
    this.queue.push(entry);
  }

  onProgress(fn: (p: FXAssetProgress) => void): void {
    this.onProgressCb = fn;
  }

  async loadAll(): Promise<FXAssetLoaded[]> {
    const total = this.queue.length;
    const results: FXAssetLoaded[] = [];
    let loaded = 0;

    const sorted = [...this.queue].sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));

    for (const entry of sorted) {
      if (this.cache.has(entry.id)) {
        results.push({ id: entry.id, kind: entry.kind, data: this.cache.get(entry.id) });
        loaded++;
        this.onProgressCb?.({ loaded, total, percent: loaded / total });
        continue;
      }
      const data = await this.fetchAsset(entry);
      this.cache.set(entry.id, data);
      results.push({ id: entry.id, kind: entry.kind, data });
      loaded++;
      this.onProgressCb?.({ loaded, total, percent: loaded / total });
    }

    this.queue = [];
    return results;
  }

  get<T = unknown>(id: string): T | undefined {
    return this.cache.get(id) as T | undefined;
  }

  evict(id: string): void {
    this.cache.delete(id);
  }

  private async fetchAsset(entry: FXAssetEntry): Promise<unknown> {
    switch (entry.kind) {
      case 'image': return this.loadImage(entry.url);
      case 'json': return fetch(entry.url).then(r => r.json());
      case 'audio': return fetch(entry.url).then(r => r.arrayBuffer());
      case 'glb': return fetch(entry.url).then(r => r.arrayBuffer());
      case 'wasm': return fetch(entry.url).then(r => r.arrayBuffer());
      case 'video': return entry.url; // video src is the asset
      default: return fetch(entry.url).then(r => r.blob());
    }
  }

  private loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = rej;
      img.src = url;
    });
  }
}

// ─── FXScrollTransition ───────────────────────────────────────────────────────

export type ScrollTransitionKind = 'fade' | 'slide-up' | 'slide-down' | 'scale' | 'parallax';

export interface FXScrollTransitionOptions {
  kind?: ScrollTransitionKind;
  threshold?: number;     // 0..1 intersection ratio
  duration?: string;      // CSS duration, e.g. '0.5s'
  easing?: string;
  offset?: string;
  once?: boolean;
}

export class FXScrollTransition {
  private readonly kind: ScrollTransitionKind;
  private readonly duration: string;
  private readonly easing: string;
  private readonly threshold: number;
  private readonly once: boolean;
  private observer: IntersectionObserver | null = null;
  private tracked = new WeakMap<Element, boolean>();

  constructor(opts: FXScrollTransitionOptions = {}) {
    this.kind = opts.kind ?? 'fade';
    this.duration = opts.duration ?? '0.6s';
    this.easing = opts.easing ?? 'cubic-bezier(0.4, 0, 0.2, 1)';
    this.threshold = opts.threshold ?? 0.15;
    this.once = opts.once ?? true;
  }

  observe(el: Element): void {
    if (typeof IntersectionObserver === 'undefined') return;
    this.applyInitialStyle(el as HTMLElement);
    if (!this.observer) {
      this.observer = new IntersectionObserver(entries => {
        for (const entry of entries) {
          const entered = this.tracked.get(entry.target);
          if (entry.isIntersecting && !entered) {
            this.applyEnterStyle(entry.target as HTMLElement);
            this.tracked.set(entry.target, true);
            if (this.once) this.observer?.unobserve(entry.target);
          } else if (!entry.isIntersecting && !this.once) {
            this.applyInitialStyle(entry.target as HTMLElement);
            this.tracked.set(entry.target, false);
          }
        }
      }, { threshold: this.threshold });
    }
    this.observer.observe(el);
  }

  unobserve(el: Element): void { this.observer?.unobserve(el); }

  private applyInitialStyle(el: HTMLElement): void {
    el.style.transition = 'none';
    switch (this.kind) {
      case 'fade': el.style.opacity = '0'; break;
      case 'slide-up': el.style.opacity = '0'; el.style.transform = 'translateY(40px)'; break;
      case 'slide-down': el.style.opacity = '0'; el.style.transform = 'translateY(-40px)'; break;
      case 'scale': el.style.opacity = '0'; el.style.transform = 'scale(0.85)'; break;
      case 'parallax': /* handled externally */ break;
    }
  }

  private applyEnterStyle(el: HTMLElement): void {
    el.style.transition = `opacity ${this.duration} ${this.easing}, transform ${this.duration} ${this.easing}`;
    el.style.opacity = '1';
    el.style.transform = 'none';
  }

  dispose(): void {
    this.observer?.disconnect();
    this.observer = null;
  }
}

// ─── FXSceneVisibility ────────────────────────────────────────────────────────

export interface FXScene {
  id: string;
  el?: HTMLElement | null;
  render?: (dt: number) => void;
  onEnter?: () => void;
  onLeave?: () => void;
}

export class FXSceneVisibility {
  private scenes = new Map<string, FXScene>();
  private activeId: string | null = null;

  register(scene: FXScene): void {
    this.scenes.set(scene.id, scene);
  }

  unregister(id: string): void {
    this.scenes.delete(id);
  }

  activate(id: string, transition = true): void {
    const prev = this.activeId ? this.scenes.get(this.activeId) : null;
    const next = this.scenes.get(id);
    if (!next) return;

    if (prev) {
      prev.onLeave?.();
      if (prev.el && transition) prev.el.style.visibility = 'hidden';
    }

    this.activeId = id;
    next.onEnter?.();
    if (next.el) {
      next.el.style.visibility = 'visible';
      if (transition) {
        next.el.style.opacity = '0';
        requestAnimationFrame(() => {
          next.el!.style.transition = 'opacity 0.4s ease';
          next.el!.style.opacity = '1';
        });
      }
    }
  }

  deactivateAll(): void {
    for (const scene of this.scenes.values()) {
      scene.onLeave?.();
      if (scene.el) scene.el.style.visibility = 'hidden';
    }
    this.activeId = null;
  }

  get active(): FXScene | null {
    return this.activeId ? this.scenes.get(this.activeId) ?? null : null;
  }

  isActive(id: string): boolean { return this.activeId === id; }
}

// ─── FXSceneCompositor ────────────────────────────────────────────────────────

export interface FXSceneCompositorOptions {
  width: number;
  height: number;
}

export class FXSceneCompositor {
  private readonly layers: FXLayer[] = [];
  private container: HTMLElement | null = null;
  private width: number;
  private height: number;
  private animationId = 0;
  private running = false;
  private lastTime = 0;

  constructor(opts: FXSceneCompositorOptions) {
    this.width = opts.width;
    this.height = opts.height;
  }

  mount(container: HTMLElement): void {
    this.container = container;
    container.style.position = 'relative';
    container.style.overflow = 'hidden';
    container.style.width = `${this.width}px`;
    container.style.height = `${this.height}px`;
  }

  addLayer(opts: FXLayerOptions): FXLayer {
    const layer = new FXLayer(opts);
    if (this.container) layer.mount(this.container, this.width, this.height);
    // Keep sorted by zIndex
    this.layers.push(layer);
    this.layers.sort((a, b) => a.zIndex - b.zIndex);
    return layer;
  }

  removeLayer(name: string): void {
    const idx = this.layers.findIndex(l => l.name === name);
    if (idx < 0) return;
    this.layers[idx].dispose();
    this.layers.splice(idx, 1);
  }

  resize(w: number, h: number): void {
    this.width = w; this.height = h;
    for (const layer of this.layers) layer.resize(w, h);
    if (this.container) {
      this.container.style.width = `${w}px`;
      this.container.style.height = `${h}px`;
    }
  }

  start(renderFn: (dt: number, layers: FXLayer[]) => void): void {
    this.running = true;
    const loop = (now: number) => {
      if (!this.running) return;
      const dt = this.lastTime ? (now - this.lastTime) / 1000 : 0;
      this.lastTime = now;
      renderFn(dt, this.layers);
      this.animationId = requestAnimationFrame(loop);
    };
    this.animationId = requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.animationId);
  }

  dispose(): void {
    this.stop();
    for (const layer of this.layers) layer.dispose();
    this.layers.length = 0;
  }

  getLayer(name: string): FXLayer | undefined {
    return this.layers.find(l => l.name === name);
  }
}

// ─── FragUIHelper ─────────────────────────────────────────────────────────────

export interface FragUIHelperOptions {
  prefix?: string;
  root?: HTMLElement | null;
}

/**
 * Utility for wiring DOM fragment UI elements (query shortcuts, event helpers).
 */
export class FragUIHelper {
  private readonly prefix: string;
  private readonly root: HTMLElement;

  constructor(opts: FragUIHelperOptions = {}) {
    this.prefix = opts.prefix ?? '';
    this.root = opts.root ?? (typeof document !== 'undefined' ? document.body : {} as HTMLElement);
  }

  $<T extends HTMLElement = HTMLElement>(selector: string): T | null {
    return this.root.querySelector<T>(this.prefixed(selector));
  }

  $$<T extends HTMLElement = HTMLElement>(selector: string): T[] {
    return Array.from(this.root.querySelectorAll<T>(this.prefixed(selector)));
  }

  on<K extends keyof HTMLElementEventMap>(
    selector: string,
    event: K,
    handler: (e: HTMLElementEventMap[K]) => void,
    opts?: AddEventListenerOptions,
  ): () => void {
    const el = this.$(selector);
    if (!el) return () => {};
    el.addEventListener(event, handler as EventListener, opts);
    return () => el.removeEventListener(event, handler as EventListener, opts);
  }

  show(selector: string): void { const el = this.$(selector); if (el) el.style.display = ''; }
  hide(selector: string): void { const el = this.$(selector); if (el) el.style.display = 'none'; }
  toggle(selector: string, force?: boolean): void {
    const el = this.$(selector);
    if (!el) return;
    const visible = el.style.display !== 'none';
    el.style.display = (force ?? !visible) ? '' : 'none';
  }

  setText(selector: string, text: string): void {
    const el = this.$(selector);
    if (el) el.textContent = text;
  }

  addClass(selector: string, ...classes: string[]): void { this.$(selector)?.classList.add(...classes); }
  removeClass(selector: string, ...classes: string[]): void { this.$(selector)?.classList.remove(...classes); }

  private prefixed(selector: string): string {
    return this.prefix ? `${this.prefix} ${selector}` : selector;
  }
}

// ─── FXDhCwa ──────────────────────────────────────────────────────────────────

export interface FXDhCwaOptions {
  targetElement?: HTMLElement | null;
  glowColor?: string;
  glowRadius?: number;
  animationSpeed?: number;
  enabled?: boolean;
}

/**
 * Dynamic highlight / chromatic-wave-aberration effect layer.
 * Applies a moving chromatic glow to a container element.
 */
export class FXDhCwa {
  private readonly opts: Required<FXDhCwaOptions>;
  private animId = 0;
  private time = 0;
  private running = false;
  private layer: FXLayer | null = null;

  constructor(opts: FXDhCwaOptions = {}) {
    this.opts = {
      targetElement: opts.targetElement ?? null,
      glowColor: opts.glowColor ?? '120, 80, 255',
      glowRadius: opts.glowRadius ?? 60,
      animationSpeed: opts.animationSpeed ?? 0.5,
      enabled: opts.enabled ?? true,
    };
  }

  mount(container: HTMLElement, w: number, h: number): void {
    this.layer = new FXLayer({ name: 'fx-dhcwa', zIndex: 999, blendMode: 'screen', opacity: 0.4 });
    this.layer.mount(container, w, h);
  }

  start(): void {
    if (!this.opts.enabled || this.running) return;
    this.running = true;
    const tick = () => {
      if (!this.running) return;
      this.time += 0.016;
      this.render();
      this.animId = requestAnimationFrame(tick);
    };
    this.animId = requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.animId);
  }

  private render(): void {
    const ctx = this.layer?.context;
    const canvas = (this.layer as any)?.canvas as HTMLCanvasElement | null;
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const cx = canvas.width / 2 + Math.sin(this.time * this.opts.animationSpeed) * canvas.width * 0.3;
    const cy = canvas.height / 2 + Math.cos(this.time * this.opts.animationSpeed * 0.7) * canvas.height * 0.3;
    const r = this.opts.glowRadius;

    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 2.5);
    grad.addColorStop(0, `rgba(${this.opts.glowColor}, 0.9)`);
    grad.addColorStop(0.4, `rgba(${this.opts.glowColor}, 0.3)`);
    grad.addColorStop(1, `rgba(${this.opts.glowColor}, 0)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  setEnabled(e: boolean): void {
    this.opts.enabled = e;
    if (!e) { this.stop(); this.layer?.clear(); } else { this.start(); }
  }

  dispose(): void {
    this.stop();
    this.layer?.dispose();
  }
}
