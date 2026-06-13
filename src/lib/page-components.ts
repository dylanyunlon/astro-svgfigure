/**
 * page-components.ts — Site page component logic: work listings, detail views, interaction patterns, contact, footer, playground, theory, player
 */

// ─── Types ────────────────────────────────────────────────────────────────────

export interface WorkItem {
  id: string;
  title: string;
  slug: string;
  year: number;
  tags: string[];
  thumbnailUrl: string;
  featured?: boolean;
  order?: number;
}

export interface WorkItemFilter {
  tag?: string;
  year?: number;
  featured?: boolean;
  query?: string;
}

// ─── WorkItems ────────────────────────────────────────────────────────────────

export interface WorkItemsOptions {
  items?: WorkItem[];
  onSelect?: (item: WorkItem) => void;
  sortBy?: 'order' | 'year' | 'title';
}

export class WorkItems {
  private items: WorkItem[];
  private filtered: WorkItem[];
  private readonly onSelect?: (item: WorkItem) => void;
  private readonly sortBy: 'order' | 'year' | 'title';
  private activeFilter: WorkItemFilter = {};

  constructor(opts: WorkItemsOptions = {}) {
    this.items = opts.items ?? [];
    this.filtered = [...this.items];
    this.onSelect = opts.onSelect;
    this.sortBy = opts.sortBy ?? 'order';
  }

  setItems(items: WorkItem[]): void {
    this.items = items;
    this.applyFilter(this.activeFilter);
  }

  applyFilter(filter: WorkItemFilter): void {
    this.activeFilter = filter;
    let result = [...this.items];
    if (filter.tag) result = result.filter(i => i.tags.includes(filter.tag!));
    if (filter.year) result = result.filter(i => i.year === filter.year);
    if (filter.featured !== undefined) result = result.filter(i => !!i.featured === filter.featured);
    if (filter.query) {
      const q = filter.query.toLowerCase();
      result = result.filter(i => i.title.toLowerCase().includes(q) || i.tags.some(t => t.toLowerCase().includes(q)));
    }
    this.filtered = this.sort(result);
  }

  clearFilter(): void {
    this.activeFilter = {};
    this.filtered = this.sort([...this.items]);
  }

  select(id: string): void {
    const item = this.items.find(i => i.id === id);
    if (item) this.onSelect?.(item);
  }

  get visibleItems(): WorkItem[] { return this.filtered; }
  get allTags(): string[] { return [...new Set(this.items.flatMap(i => i.tags))].sort(); }
  get allYears(): number[] { return [...new Set(this.items.map(i => i.year))].sort((a, b) => b - a); }

  private sort(items: WorkItem[]): WorkItem[] {
    return items.sort((a, b) => {
      if (this.sortBy === 'year') return b.year - a.year;
      if (this.sortBy === 'title') return a.title.localeCompare(b.title);
      return (a.order ?? 0) - (b.order ?? 0);
    });
  }
}

// ─── WorkDetail ───────────────────────────────────────────────────────────────

export interface WorkDetailData {
  id: string;
  title: string;
  year: number;
  tags: string[];
  description: string;
  media: Array<{ kind: 'image' | 'video'; url: string; alt?: string }>;
  links?: Array<{ label: string; url: string }>;
  credits?: string;
  nextSlug?: string;
  prevSlug?: string;
}

export interface WorkDetailOptions {
  onNavigate?: (slug: string) => void;
  onClose?: () => void;
}

export class WorkDetail {
  private data: WorkDetailData | null = null;
  private readonly onNavigate?: (slug: string) => void;
  private readonly onClose?: () => void;
  private open = false;

  constructor(opts: WorkDetailOptions = {}) {
    this.onNavigate = opts.onNavigate;
    this.onClose = opts.onClose;
  }

  load(data: WorkDetailData): void {
    this.data = data;
    this.open = true;
  }

  close(): void {
    this.open = false;
    this.data = null;
    this.onClose?.();
  }

  navigateNext(): void {
    if (this.data?.nextSlug) this.onNavigate?.(this.data.nextSlug);
  }

  navigatePrev(): void {
    if (this.data?.prevSlug) this.onNavigate?.(this.data.prevSlug);
  }

  get isOpen(): boolean { return this.open; }
  get current(): WorkDetailData | null { return this.data; }
  get hasNext(): boolean { return !!this.data?.nextSlug; }
  get hasPrev(): boolean { return !!this.data?.prevSlug; }
}

// ─── WorkDetailContent ────────────────────────────────────────────────────────

export interface WorkDetailContentOptions {
  animateIn?: boolean;
  mediaAutoplay?: boolean;
}

export class WorkDetailContent {
  private container: HTMLElement | null = null;
  private scrollProgress = 0;
  private readonly animateIn: boolean;
  private readonly mediaAutoplay: boolean;
  private onScrollListeners: Array<(p: number) => void> = [];

  constructor(opts: WorkDetailContentOptions = {}) {
    this.animateIn = opts.animateIn ?? true;
    this.mediaAutoplay = opts.mediaAutoplay ?? false;
  }

  mount(el: HTMLElement): void {
    this.container = el;
    el.addEventListener('scroll', this.handleScroll, { passive: true });
    if (this.animateIn) this.triggerEntrance();
  }

  unmount(): void {
    this.container?.removeEventListener('scroll', this.handleScroll);
    this.container = null;
  }

  scrollTo(y: number, smooth = true): void {
    this.container?.scrollTo({ top: y, behavior: smooth ? 'smooth' : 'instant' });
  }

  onScroll(fn: (progress: number) => void): () => void {
    this.onScrollListeners.push(fn);
    return () => { this.onScrollListeners = this.onScrollListeners.filter(l => l !== fn); };
  }

  get progress(): number { return this.scrollProgress; }

  private handleScroll = (): void => {
    if (!this.container) return;
    const { scrollTop, scrollHeight, clientHeight } = this.container;
    this.scrollProgress = scrollHeight > clientHeight ? scrollTop / (scrollHeight - clientHeight) : 0;
    this.onScrollListeners.forEach(fn => fn(this.scrollProgress));
  };

  private triggerEntrance(): void {
    if (!this.container) return;
    this.container.style.opacity = '0';
    this.container.style.transform = 'translateY(20px)';
    requestAnimationFrame(() => {
      if (!this.container) return;
      this.container.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      this.container.style.opacity = '1';
      this.container.style.transform = '';
    });
  }
}

// ─── TubesInteraction ─────────────────────────────────────────────────────────

export interface TubeNode { id: string; x: number; y: number; active?: boolean; }
export interface TubeEdge { from: string; to: string; energy?: number; }

export interface TubesInteractionOptions {
  onNodeClick?: (node: TubeNode) => void;
  onEdgeHover?: (edge: TubeEdge, on: boolean) => void;
  animationSpeed?: number;
}

export class TubesInteraction {
  private nodes: TubeNode[] = [];
  private edges: TubeEdge[] = [];
  private hovered: string | null = null;
  private energy = new Map<string, number>(); // edge key -> energy level
  private readonly opts: Required<TubesInteractionOptions>;

  constructor(opts: TubesInteractionOptions = {}) {
    this.opts = {
      onNodeClick: opts.onNodeClick ?? (() => {}),
      onEdgeHover: opts.onEdgeHover ?? (() => {}),
      animationSpeed: opts.animationSpeed ?? 1,
    };
  }

  setGraph(nodes: TubeNode[], edges: TubeEdge[]): void {
    this.nodes = nodes;
    this.edges = edges;
    this.energy.clear();
    for (const e of edges) this.energy.set(this.edgeKey(e), e.energy ?? 0);
  }

  clickNode(id: string): void {
    const node = this.nodes.find(n => n.id === id);
    if (!node) return;
    node.active = !node.active;
    this.opts.onNodeClick(node);
    this.propagateEnergy(id);
  }

  hoverEdge(from: string, to: string, on: boolean): void {
    const edge = this.edges.find(e => e.from === from && e.to === to);
    if (!edge) return;
    this.hovered = on ? this.edgeKey(edge) : null;
    this.opts.onEdgeHover(edge, on);
  }

  update(dt: number): void {
    for (const [key, level] of this.energy.entries()) {
      if (level > 0) this.energy.set(key, Math.max(0, level - dt * this.opts.animationSpeed));
    }
  }

  getEnergy(from: string, to: string): number {
    return this.energy.get(this.edgeKey({ from, to })) ?? 0;
  }

  private propagateEnergy(nodeId: string): void {
    for (const edge of this.edges) {
      if (edge.from === nodeId || edge.to === nodeId) {
        this.energy.set(this.edgeKey(edge), 1.0);
      }
    }
  }

  private edgeKey(e: { from: string; to: string }): string { return `${e.from}:${e.to}`; }
}

// ─── MoveNode ─────────────────────────────────────────────────────────────────

export interface MoveNodeOptions {
  containerId?: string;
  snapGrid?: number;
  onMove?: (id: string, x: number, y: number) => void;
}

export class MoveNode {
  private dragging: string | null = null;
  private offset = { x: 0, y: 0 };
  private positions = new Map<string, { x: number; y: number }>();
  private readonly snapGrid: number;
  private readonly onMove?: (id: string, x: number, y: number) => void;

  constructor(opts: MoveNodeOptions = {}) {
    this.snapGrid = opts.snapGrid ?? 0;
    this.onMove = opts.onMove;
  }

  startDrag(id: string, pointerX: number, pointerY: number): void {
    this.dragging = id;
    const pos = this.positions.get(id) ?? { x: 0, y: 0 };
    this.offset = { x: pointerX - pos.x, y: pointerY - pos.y };
  }

  moveDrag(pointerX: number, pointerY: number): void {
    if (!this.dragging) return;
    let x = pointerX - this.offset.x;
    let y = pointerY - this.offset.y;
    if (this.snapGrid > 0) {
      x = Math.round(x / this.snapGrid) * this.snapGrid;
      y = Math.round(y / this.snapGrid) * this.snapGrid;
    }
    this.positions.set(this.dragging, { x, y });
    this.onMove?.(this.dragging, x, y);
  }

  endDrag(): void { this.dragging = null; }

  setPosition(id: string, x: number, y: number): void { this.positions.set(id, { x, y }); }
  getPosition(id: string): { x: number; y: number } { return this.positions.get(id) ?? { x: 0, y: 0 }; }
  get isDragging(): boolean { return !!this.dragging; }
  get draggedId(): string | null { return this.dragging; }
}

// ─── Contact ──────────────────────────────────────────────────────────────────

export interface ContactFormData {
  name: string;
  email: string;
  subject?: string;
  message: string;
  honeypot?: string; // spam trap
}

export interface ContactOptions {
  endpointUrl?: string;
  onSuccess?: (data: ContactFormData) => void;
  onError?: (err: Error) => void;
}

export class Contact {
  private sending = false;
  private sent = false;
  private errorMsg: string | null = null;
  private readonly opts: ContactOptions;

  constructor(opts: ContactOptions = {}) {
    this.opts = opts;
  }

  async submit(data: ContactFormData): Promise<boolean> {
    if (this.sending) return false;
    if (data.honeypot) return false; // bot trap
    if (!this.validate(data)) return false;

    this.sending = true;
    this.errorMsg = null;

    try {
      if (this.opts.endpointUrl) {
        const res = await fetch(this.opts.endpointUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: data.name, email: data.email, subject: data.subject, message: data.message }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
      }
      this.sent = true;
      this.opts.onSuccess?.(data);
      return true;
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      this.errorMsg = err.message;
      this.opts.onError?.(err);
      return false;
    } finally {
      this.sending = false;
    }
  }

  reset(): void { this.sent = false; this.sending = false; this.errorMsg = null; }

  validate(data: ContactFormData): boolean {
    if (!data.name.trim() || !data.message.trim()) return false;
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(data.email);
  }

  get isSending(): boolean { return this.sending; }
  get isSent(): boolean { return this.sent; }
  get error(): string | null { return this.errorMsg; }
}

// ─── Footer ───────────────────────────────────────────────────────────────────

export interface FooterLink { label: string; url: string; external?: boolean; }

export interface FooterOptions {
  links?: FooterLink[];
  year?: number;
  author?: string;
}

export class Footer {
  readonly links: FooterLink[];
  readonly year: number;
  readonly author: string;

  constructor(opts: FooterOptions = {}) {
    this.links = opts.links ?? [];
    this.year = opts.year ?? new Date().getFullYear();
    this.author = opts.author ?? '';
  }

  get copyright(): string {
    return `© ${this.year}${this.author ? ` ${this.author}` : ''}`;
  }

  addLink(link: FooterLink): void { this.links.push(link); }

  render(container: HTMLElement): void {
    container.innerHTML = '';
    const copy = document.createElement('span');
    copy.textContent = this.copyright;
    container.appendChild(copy);

    for (const link of this.links) {
      const a = document.createElement('a');
      a.href = link.url;
      a.textContent = link.label;
      if (link.external) { a.target = '_blank'; a.rel = 'noopener noreferrer'; }
      container.appendChild(a);
    }
  }
}

// ─── Playground ───────────────────────────────────────────────────────────────

export interface PlaygroundModule {
  id: string;
  name: string;
  init: (container: HTMLElement) => Promise<void> | void;
  destroy?: () => void;
}

export class Playground {
  private modules = new Map<string, PlaygroundModule>();
  private active: PlaygroundModule | null = null;
  private container: HTMLElement | null = null;

  register(mod: PlaygroundModule): void {
    this.modules.set(mod.id, mod);
  }

  mount(container: HTMLElement): void {
    this.container = container;
  }

  async activate(id: string): Promise<void> {
    if (this.active) {
      this.active.destroy?.();
      this.active = null;
    }
    const mod = this.modules.get(id);
    if (!mod || !this.container) return;
    this.active = mod;
    this.container.innerHTML = '';
    await mod.init(this.container);
  }

  deactivate(): void {
    this.active?.destroy?.();
    this.active = null;
    if (this.container) this.container.innerHTML = '';
  }

  get moduleIds(): string[] { return [...this.modules.keys()]; }
  get activeModule(): PlaygroundModule | null { return this.active; }
}

// ─── Theory ───────────────────────────────────────────────────────────────────

export interface TheorySection {
  id: string;
  title: string;
  content: string; // HTML or markdown
  subsections?: TheorySection[];
}

export interface TheoryOptions {
  sections?: TheorySection[];
  onSectionChange?: (id: string) => void;
}

export class Theory {
  private sections: TheorySection[];
  private activeSectionId: string | null = null;
  private readonly onSectionChange?: (id: string) => void;

  constructor(opts: TheoryOptions = {}) {
    this.sections = opts.sections ?? [];
    this.onSectionChange = opts.onSectionChange;
  }

  setSections(sections: TheorySection[]): void { this.sections = sections; }

  navigate(id: string): void {
    const section = this.findSection(id);
    if (!section) return;
    this.activeSectionId = id;
    this.onSectionChange?.(id);
  }

  get active(): TheorySection | null {
    return this.activeSectionId ? this.findSection(this.activeSectionId) ?? null : null;
  }

  get allSections(): TheorySection[] { return this.sections; }

  get tableOfContents(): Array<{ id: string; title: string; depth: number }> {
    const toc: Array<{ id: string; title: string; depth: number }> = [];
    const walk = (sections: TheorySection[], depth: number) => {
      for (const s of sections) {
        toc.push({ id: s.id, title: s.title, depth });
        if (s.subsections) walk(s.subsections, depth + 1);
      }
    };
    walk(this.sections, 0);
    return toc;
  }

  private findSection(id: string, sections = this.sections): TheorySection | undefined {
    for (const s of sections) {
      if (s.id === id) return s;
      if (s.subsections) {
        const found = this.findSection(id, s.subsections);
        if (found) return found;
      }
    }
    return undefined;
  }
}

// ─── Player ───────────────────────────────────────────────────────────────────

export interface PlayerTrack {
  id: string;
  title: string;
  artist?: string;
  src: string;
  duration?: number; // seconds
  coverUrl?: string;
}

export interface PlayerOptions {
  tracks?: PlayerTrack[];
  autoplay?: boolean;
  loop?: boolean;
  onTrackChange?: (track: PlayerTrack) => void;
  onStateChange?: (playing: boolean) => void;
}

export class Player {
  private tracks: PlayerTrack[];
  private currentIndex = 0;
  private audio: HTMLAudioElement | null = null;
  private _playing = false;
  private readonly opts: Required<PlayerOptions>;

  constructor(opts: PlayerOptions = {}) {
    this.tracks = opts.tracks ?? [];
    this.opts = {
      tracks: this.tracks,
      autoplay: opts.autoplay ?? false,
      loop: opts.loop ?? false,
      onTrackChange: opts.onTrackChange ?? (() => {}),
      onStateChange: opts.onStateChange ?? (() => {}),
    };
    this.initAudio();
  }

  private initAudio(): void {
    if (typeof Audio === 'undefined') return;
    this.audio = new Audio();
    this.audio.loop = false;
    this.audio.addEventListener('ended', () => this.next());
    this.audio.addEventListener('play', () => { this._playing = true; this.opts.onStateChange(true); });
    this.audio.addEventListener('pause', () => { this._playing = false; this.opts.onStateChange(false); });
    if (this.tracks.length > 0) this.loadTrack(0);
  }

  private loadTrack(index: number): void {
    if (!this.audio || index < 0 || index >= this.tracks.length) return;
    this.currentIndex = index;
    this.audio.src = this.tracks[index].src;
    this.opts.onTrackChange(this.tracks[index]);
  }

  play(): void { this.audio?.play().catch(() => {}); }
  pause(): void { this.audio?.pause(); }
  toggle(): void { this._playing ? this.pause() : this.play(); }

  next(): void {
    const nextIdx = (this.currentIndex + 1) % this.tracks.length;
    if (nextIdx === 0 && !this.opts.loop) { this.pause(); return; }
    this.loadTrack(nextIdx);
    if (this._playing) this.play();
  }

  prev(): void {
    const prevIdx = (this.currentIndex - 1 + this.tracks.length) % this.tracks.length;
    this.loadTrack(prevIdx);
    if (this._playing) this.play();
  }

  seekTo(seconds: number): void { if (this.audio) this.audio.currentTime = seconds; }

  setVolume(v: number): void { if (this.audio) this.audio.volume = Math.max(0, Math.min(1, v)); }

  addTrack(track: PlayerTrack): void { this.tracks.push(track); }
  setTracks(tracks: PlayerTrack[]): void { this.tracks = tracks; this.loadTrack(0); }

  get currentTrack(): PlayerTrack | null { return this.tracks[this.currentIndex] ?? null; }
  get isPlaying(): boolean { return this._playing; }
  get currentTime(): number { return this.audio?.currentTime ?? 0; }
  get duration(): number { return this.audio?.duration ?? 0; }
  get volume(): number { return this.audio?.volume ?? 1; }

  dispose(): void {
    this.audio?.pause();
    if (this.audio) this.audio.src = '';
    this.audio = null;
  }
}
