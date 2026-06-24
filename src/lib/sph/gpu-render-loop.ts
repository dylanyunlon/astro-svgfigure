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
import { safeCompile, checkFBO, drainErrors, setupContextLost } from './gpu-error-guard';
import { GPUPerfMonitor } from './gpu-perf-monitor';

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
  private gl: WebGLRenderingContext;
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

  // Perf + error monitoring
  private perf: GPUPerfMonitor;
  private frameCount = 0;

  // 状态
  private cells: CellData[] = [];
  private edges: EdgeData[] = [];
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

    const gl = canvas.getContext('webgl', {
      alpha: true, antialias: true, premultipliedAlpha: false,
    });
    if (!gl) throw new Error('[GPURenderLoop] WebGL not available');
    this.gl = gl;

    // 启用必要扩展
    gl.getExtension('OES_texture_float');
    gl.getExtension('OES_texture_half_float');
    gl.getExtension('OES_texture_half_float_linear');
    gl.getExtension('OES_standard_derivatives');
    gl.getExtension('WEBGL_color_buffer_float');

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
    initATShaderPipeline('/upstream/activetheory-assets/compiled.vs')
      .then(() => {
        const names = listATShaders();
        console.log(`[GPURenderLoop] AT shaders ready: ${names.length} shaders`);

        // M982: 用 AT PhysicalShader 替换 PBR pass 的默认 shader
        if (this.pbr) {
          const atPhysical = getATProgram(this.gl, 'PhysicalShader');
          if (atPhysical) {
            this.pbr.swapProgram(atPhysical.program);
            console.log('[GPURenderLoop] PBR pass: AT PhysicalShader active');
          } else {
            console.warn('[GPURenderLoop] AT PhysicalShader not found — PBR keeps default shader');
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

    // SDF species icon pass
    try { this.sdfIcon = createSDFIconGPU(gl); } catch (e) { console.warn('[GPURenderLoop] SDF init failed:', e); }

    // WebGL2 particle (optional)
    const gl2 = canvas.getContext('webgl2');
    if (gl2) {
      this.particle = new ParticleGPU(gl2, 5000);
    }
  }

  /** 设置 cell 和 edge 数据 */
  setScene(cells: CellData[], edges: EdgeData[]): void {
    this.cells = cells;
    this.edges = edges;
  }

  /** 单帧渲染 — 执行全部 GPU pass，每 pass 带 error guard + perf 计时 */
  frame(dt: number): void {
    const gl = this.gl;
    this.frameCount++;

    this.perf.frameStart();

    // ── Pass 1: Fluid (鼠标流体) ──
    {
      const t = this.perf.passStart('fluid');
      try {
        this.fluid.step(this.mouseX, this.mouseY, this.prevMouseX, this.prevMouseY, dt);
      } catch (e) {
        console.error('[GPURenderLoop] fluid pass error:', e);
      }
      this.perf.passEnd('fluid', t);
    }

    // ── Pass 2: Shadow map ──
    const lightDir: [number, number, number] = [-0.5, -1.0, -0.3];
    {
      const t = this.perf.passStart('shadow');
      try {
        this.shadow.render(this.cells, lightDir);
      } catch (e) {
        console.error('[GPURenderLoop] shadow pass error:', e);
      }
      this.perf.passEnd('shadow', t);
    }

    // ── Pass 3: PBR cell surface ──
    let cellFBO: WebGLTexture;
    {
      const t = this.perf.passStart('pbr');
      try {
        if (this.pbr) {
          this.pbr.render(this.cells, this.shadow.outputTexture, lightDir);
          cellFBO = this.pbr.outputTexture;
        } else {
          cellFBO = this._renderCellsFallback();
        }
      } catch (e) {
        console.error('[GPURenderLoop] pbr pass error:', e);
        cellFBO = this._renderCellsFallback();
      }
      this.perf.passEnd('pbr', t);
    }

    // ── Pass 3b: Glass overlay (Fresnel + refraction on cell bodies) ──
    if (this.glass) {
      const t = this.perf.passStart('glass');
      try {
        this.glass.render(this.cells, cellFBO);
      } catch (e) {
        console.error('[GPURenderLoop] glass pass error:', e);
      }
      this.perf.passEnd('glass', t);
    }

    // ── Pass 4: Edge 样条线 ──
    {
      const t = this.perf.passStart('edge');
      try {
        this.edge.render(this.edges);
      } catch (e) {
        console.error('[GPURenderLoop] edge pass error:', e);
      }
      this.perf.passEnd('edge', t);
    }

    // ── Pass 5: SDF Species icon (instanced) ──
    if (this.sdfIcon) {
      const t = this.perf.passStart('sdf');
      try {
        this.sdfIcon.render(this.cells);
      } catch (e) {
        console.error('[GPURenderLoop] sdf pass error:', e);
      }
      this.perf.passEnd('sdf', t);
    }

    // ── Pass 6: Particle ──
    if (this.particle) {
      const t = this.perf.passStart('particle');
      try {
        this.particle.step(dt);
        this.particle.render();
      } catch (e) {
        console.error('[GPURenderLoop] particle pass error:', e);
      }
      this.perf.passEnd('particle', t);
    }

    // ── Pass 7: Bloom ──
    {
      const t = this.perf.passStart('bloom');
      try {
        this.bloom.step(cellFBO);
      } catch (e) {
        console.error('[GPURenderLoop] bloom pass error:', e);
      }
      this.perf.passEnd('bloom', t);
    }

    // ── Pass 8: MSDF text ──
    {
      const t = this.perf.passStart('msdf');
      try {
        this.msdf.render(this.cells.map(c => ({
          text: c.label, x: c.x, y: c.y, scale: 1.0,
        })));
      } catch (e) {
        console.error('[GPURenderLoop] msdf pass error:', e);
      }
      this.perf.passEnd('msdf', t);
    }

    // ── Pass 9: Final composite → screen ──
    {
      const t = this.perf.passStart('composite');
      try {
        this.composite.render({
          cellTexture: cellFBO,
          edgeTexture: this.edge.outputTexture,
          particleTexture: this.particle?.outputTexture ?? null,
          bloomTexture: this.bloom.outputTexture,
          shadowTexture: this.shadow.outputTexture,
          fluidTexture: this.fluid.dyeTexture,
          sdfTexture: this.sdfIcon?.outputTexture ?? null,
          glassTexture: this.glass?.outputTexture ?? null,
        });
      } catch (e) {
        console.error('[GPURenderLoop] composite pass error:', e);
      }
      this.perf.passEnd('composite', t);
    }

    // ── Drain any accumulated GL errors ──
    drainErrors(gl);

    this.perf.frameEnd();

    // ── Log perf stats every 60 frames ──
    if (this.frameCount % 60 === 0) {
      console.log('[GPURenderLoop] perf stats:', this.perf.stats);
    }
  }

  /** 简单 fallback: 用纯色 quad 画 cell (pbr-gpu-pass 未到时) */
  private _renderCellsFallback(): WebGLTexture {
    const gl = this.gl;
    // 创建临时 FBO
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.canvas.width, this.canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // FBO 完整性校验
    checkFBO(gl, 'fallback-cell');

    // 画每个 cell 为一个纯色矩形 (简化 PBR)
    for (const cell of this.cells) {
      const mat = SPECIES_MATERIAL[cell.species] ?? SPECIES_MATERIAL['cil-eye'];
      // 用 scissor 画矩形
      const px = Math.floor(cell.x / 1000 * this.canvas.width);
      const py = Math.floor(cell.y / 800 * this.canvas.height);
      const pw = Math.floor(cell.w / 1000 * this.canvas.width);
      const ph = Math.floor(cell.h / 800 * this.canvas.height);
      gl.enable(gl.SCISSOR_TEST);
      gl.scissor(px, py, pw, ph);
      gl.clearColor(mat.albedo[0], mat.albedo[1], mat.albedo[2], 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.disable(gl.SCISSOR_TEST);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    return tex;
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
