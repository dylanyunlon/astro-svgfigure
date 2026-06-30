/**
 * cell-geometry-channel.ts — 从 channel 读取 cell 的 geometry.json
 *
 * 每个 cell agent (Claude 对话) 写 channels/cell/{id}/geometry.json。
 * 此模块轮询这些文件，将 geometry 数据注入 GPU 渲染循环。
 *
 * 数据流:
 *   Cell Agent → geometry.json → (SSE or poll) → CellGeometryChannel → GPURenderLoop
 *
 * 这是 cell 和 GPU 之间的唯一接口。cell 不直接操作 WebGL。
 */

// ── Types ──────────────────────────────────────────────────────────────────────

export interface Lobe {
  angle: number;
  distance: number;
  radius: number;
}

export interface Pseudopod {
  target_cell: string;
  length: number;
  width: number;
  curl: number;
}

export interface CellGeometry {
  cell_id: string;
  tick: number;
  timestamp_ms: number;

  transform: {
    x: number;
    y: number;
    scale: number;
    rotation: number;
  };

  sdf: {
    type: 'metaball';
    base_radius: number;
    lobes: Lobe[];
    noise_amplitude: number;
    noise_frequency: number;
  };

  surface: {
    albedo: [number, number, number];
    roughness: number;
    metallic: number;
    opacity: number;
    glow_color: [number, number, number];
    glow_intensity: number;
  };

  membrane: {
    thickness: number;
    wobble_amplitude: number;
    wobble_frequency: number;
    permeability_visual: number;
  };

  pseudopods: Pseudopod[];

  internal_motion: {
    cytoplasm_flow_angle: number;
    cytoplasm_flow_speed: number;
    organelle_drift: number;
  };
}

// ── Channel Reader ─────────────────────────────────────────────────────────────

const MAX_LOBES = 8;

export class CellGeometryChannel {
  private geometries: Map<string, CellGeometry> = new Map();
  private pollInterval: number | null = null;
  private eventSource: EventSource | null = null;
  private cellIds: string[] = [];

  /**
   * Initialize with cell IDs from composite_params.
   * Starts polling or SSE listening for geometry updates.
   */
  init(cellIds: string[]): void {
    this.cellIds = cellIds;
    // Try SSE first, fall back to polling
    this._connectSSE();
  }

  /** Get geometry for a cell. Returns null if cell hasn't written geometry.json yet. */
  get(cellId: string): CellGeometry | null {
    return this.geometries.get(cellId) ?? null;
  }

  /** Set geometry for a cell directly (from external SSE handler). */
  set(cellId: string, geom: CellGeometry): void {
    if (geom && geom.sdf) {
      this._validate(geom);
      this.geometries.set(cellId, geom);
    }
  }

  /** All geometries that have been written by cell agents. */
  getAll(): Map<string, CellGeometry> {
    return this.geometries;
  }

  /** Whether a cell has organic geometry (vs fallback rectangle). */
  hasGeometry(cellId: string): boolean {
    return this.geometries.has(cellId);
  }

  /**
   * Convert CellGeometry to GPU-friendly uniform data.
   * Returns a flat Float32Array for uploading to WebGL.
   *
   * Layout per cell (32 floats):
   *   [0-1]   position (x, y)
   *   [2-3]   scale, rotation
   *   [4]     base_radius
   *   [5]     noise_amplitude
   *   [6]     noise_frequency
   *   [7]     lobe_count
   *   [8-31]  lobes (up to MAX_LOBES × 3: angle, distance, radius)
   */
  toGPUData(cellId: string): Float32Array | null {
    const g = this.geometries.get(cellId);
    if (!g) return null;

    const data = new Float32Array(32);
    data[0] = g.transform.x;
    data[1] = g.transform.y;
    data[2] = g.transform.scale;
    data[3] = g.transform.rotation;
    data[4] = g.sdf.base_radius;
    data[5] = g.sdf.noise_amplitude;
    data[6] = g.sdf.noise_frequency;

    const lobes = g.sdf.lobes.slice(0, MAX_LOBES);
    data[7] = lobes.length;
    for (let i = 0; i < lobes.length; i++) {
      data[8 + i * 3 + 0] = lobes[i].angle;
      data[8 + i * 3 + 1] = lobes[i].distance;
      data[8 + i * 3 + 2] = lobes[i].radius;
    }

    return data;
  }

  /**
   * Convert surface properties to GPU-friendly data.
   * Layout (12 floats):
   *   [0-2]   albedo RGB
   *   [3]     roughness
   *   [4]     metallic
   *   [5]     opacity
   *   [6-8]   glow_color RGB
   *   [9]     glow_intensity
   *   [10]    membrane_thickness
   *   [11]    membrane_wobble_amplitude
   */
  toSurfaceData(cellId: string): Float32Array | null {
    const g = this.geometries.get(cellId);
    if (!g) return null;

    const data = new Float32Array(12);
    data[0] = g.surface.albedo[0];
    data[1] = g.surface.albedo[1];
    data[2] = g.surface.albedo[2];
    data[3] = g.surface.roughness;
    data[4] = g.surface.metallic;
    data[5] = g.surface.opacity;
    data[6] = g.surface.glow_color[0];
    data[7] = g.surface.glow_color[1];
    data[8] = g.surface.glow_color[2];
    data[9] = g.surface.glow_intensity;
    data[10] = g.membrane.thickness;
    data[11] = g.membrane.wobble_amplitude;

    return data;
  }

  // ── SSE ──────────────────────────────────────────────────────────────────────

  private _connectSSE(): void {
    try {
      this.eventSource = new EventSource('/api/cell-events');
      this.eventSource.addEventListener('geometry_update', (e: MessageEvent) => {
        try {
          const payload = JSON.parse(e.data);
          const geom: CellGeometry = payload.geometry ?? payload;
          if (geom.cell_id && geom.sdf) {
            this._validate(geom);
            this.geometries.set(geom.cell_id, geom);
          }
        } catch { /* ignore malformed */ }
      });
      this.eventSource.onerror = () => {
        // SSE failed, fall back to polling
        this.eventSource?.close();
        this.eventSource = null;
        this._startPolling();
      };
    } catch {
      this._startPolling();
    }
  }

  private _startPolling(): void {
    if (this.pollInterval) return;
    // Poll every 500ms
    this.pollInterval = window.setInterval(() => this._poll(), 500);
  }

  private async _poll(): Promise<void> {
    for (const id of this.cellIds) {
      try {
        const resp = await fetch(`/channels/cell/${id}/geometry.json`, { cache: 'no-cache' });
        if (!resp.ok) continue;
        const geom: CellGeometry = await resp.json();
        if (geom.cell_id && geom.sdf) {
          this._validate(geom);
          this.geometries.set(id, geom);
        }
      } catch { /* cell hasn't written geometry yet */ }
    }
  }

  /** Clamp values to safe ranges */
  private _validate(g: CellGeometry): void {
    g.sdf.base_radius = Math.max(5, Math.min(200, g.sdf.base_radius));
    g.sdf.noise_amplitude = Math.max(0, Math.min(0.1, g.sdf.noise_amplitude));
    g.sdf.noise_frequency = Math.max(1, Math.min(10, g.sdf.noise_frequency));
    g.sdf.lobes = g.sdf.lobes.slice(0, MAX_LOBES);
    for (const l of g.sdf.lobes) {
      l.radius = Math.max(3, Math.min(60, l.radius));
      l.distance = Math.max(0, Math.min(100, l.distance));
    }
    g.surface.opacity = Math.max(0, Math.min(1, g.surface.opacity));
    g.surface.glow_intensity = Math.max(0, Math.min(2, g.surface.glow_intensity));
    g.membrane.thickness = Math.max(0.5, Math.min(10, g.membrane.thickness));
  }

  dispose(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    this.eventSource?.close();
    this.eventSource = null;
  }
}
