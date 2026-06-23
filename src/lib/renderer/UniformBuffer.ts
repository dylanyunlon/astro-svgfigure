/**
 * UniformBuffer.ts — Uniform Buffer Objects (UBO) 批量传 uniform
 *
 * 对标 AT Renderer.UBO:
 *   - WebGL2: gl.createBuffer() + gl.bindBufferBase(gl.UNIFORM_BUFFER, ...)
 *   - gl.bufferSubData 高效局部更新
 *   - WebGL1 降级：退回到普通 uniform calls (AstroProgram setters)
 *
 * 支持的布局:
 *   CellUBO  — per-cell 图元数据
 *   NukeUBO  — 后期特效参数
 *   ViewUBO  — 相机 / 投影 / 分辨率
 *
 * std140 对齐规则 (WebGL2 UBO 强制要求):
 *   float      → 4 bytes  (offset 必须是 4 的倍数)
 *   vec2       → 8 bytes  (offset 必须是 8 的倍数)
 *   vec4       → 16 bytes (offset 必须是 16 的倍数)
 *   mat4       → 64 bytes (4×vec4)
 */

import { AstroRenderer, WEBGL2 } from './AstroRenderer';
import type { AstroProgram } from './AstroProgram';

// ── UBO 布局描述 ──────────────────────────────────────────────────────────────

export type UBOFieldType = 'float' | 'vec2' | 'vec4' | 'mat4';

export interface UBOFieldDef {
  name:   string;
  type:   UBOFieldType;
  /** std140 byte offset — must be pre-computed by caller or use UBOLayout helpers */
  offset: number;
}

export interface UBOLayoutDef {
  blockName:    string;
  byteLength:   number;   // total aligned size (multiple of 16)
  fields:       UBOFieldDef[];
  bindingPoint?: number;  // auto-assigned if omitted
}

// ── std140 field sizes (bytes) ────────────────────────────────────────────────

const FIELD_BYTES: Record<UBOFieldType, number> = {
  float: 4,
  vec2:  8,
  vec4:  16,
  mat4:  64,
};

// ── Pre-defined Layouts ───────────────────────────────────────────────────────

/**
 * CellUBO — per-cell 图元数据
 *
 * layout std140:
 *   bbox      vec4  offset  0  (16 bytes)
 *   fillColor vec4  offset 16  (16 bytes)
 *   opacity   float offset 32  (4 bytes)
 *   time      float offset 36  (4 bytes)
 *   _pad      vec2  offset 40  (8 bytes) — pad to 48 → round up to 48
 * total: 48 bytes
 */
export const CELL_UBO_LAYOUT: UBOLayoutDef = {
  blockName:  'CellUBO',
  byteLength: 48,
  fields: [
    { name: 'bbox',      type: 'vec4',  offset:  0 },
    { name: 'fillColor', type: 'vec4',  offset: 16 },
    { name: 'opacity',   type: 'float', offset: 32 },
    { name: 'time',      type: 'float', offset: 36 },
  ],
};

/**
 * NukeUBO — 后期特效
 *
 * layout std140:
 *   bloomStrength float offset  0  (4 bytes)
 *   bloomRadius   float offset  4  (4 bytes)
 *   _pad          vec2  offset  8  (8 bytes) — align dofParams to 16
 *   dofParams     vec4  offset 16  (16 bytes)
 * total: 32 bytes
 */
export const NUKE_UBO_LAYOUT: UBOLayoutDef = {
  blockName:  'NukeUBO',
  byteLength: 32,
  fields: [
    { name: 'bloomStrength', type: 'float', offset:  0 },
    { name: 'bloomRadius',   type: 'float', offset:  4 },
    { name: 'dofParams',     type: 'vec4',  offset: 16 },
  ],
};

/**
 * ViewUBO — 相机 / 投影 / 分辨率
 *
 * layout std140:
 *   viewMatrix  mat4  offset   0  (64 bytes)
 *   projMatrix  mat4  offset  64  (64 bytes)
 *   resolution  vec2  offset 128  (8 bytes)
 *   _pad        vec2  offset 136  (8 bytes) — total 144
 * total: 144 bytes
 */
export const VIEW_UBO_LAYOUT: UBOLayoutDef = {
  blockName:  'ViewUBO',
  byteLength: 144,
  fields: [
    { name: 'viewMatrix', type: 'mat4', offset:   0 },
    { name: 'projMatrix', type: 'mat4', offset:  64 },
    { name: 'resolution', type: 'vec2', offset: 128 },
  ],
};

// ── UniformBuffer ─────────────────────────────────────────────────────────────

export class UniformBuffer {
  readonly layout:   UBOLayoutDef;
  readonly renderer: AstroRenderer;

  /** Binding point assigned at construction time */
  readonly bindingPoint: number;

  /** Whether UBOs are actually in use (false = WebGL1 fallback mode) */
  readonly isWebGL2: boolean;

  private _buffer:    WebGLBuffer | null = null;
  private _cpuBuffer: ArrayBuffer;
  private _f32View:   Float32Array;

  // ── Static binding-point counter (global, shared across all UniformBuffer instances) ──
  private static _nextBindingPoint = 0;

  // ── Constructor ─────────────────────────────────────────────────────────────

  constructor(renderer: AstroRenderer, layout: UBOLayoutDef) {
    this.renderer = renderer;
    this.layout   = layout;
    this.isWebGL2 = renderer.version === WEBGL2;

    // Assign a binding point (auto-increment or from layout definition)
    this.bindingPoint = layout.bindingPoint ?? UniformBuffer._nextBindingPoint++;

    // Allocate CPU-side shadow buffer (always, for both WebGL1 fallback reads and WebGL2 upload)
    this._cpuBuffer = new ArrayBuffer(layout.byteLength);
    this._f32View   = new Float32Array(this._cpuBuffer);

    if (this.isWebGL2) {
      this._allocateGPUBuffer();
    }
  }

  // ── GPU buffer allocation ────────────────────────────────────────────────────

  private _allocateGPUBuffer(): void {
    const gl = this.renderer.context as WebGL2RenderingContext;
    this._buffer = gl.createBuffer();
    if (!this._buffer) throw new Error('[UniformBuffer] gl.createBuffer() failed');

    gl.bindBuffer(gl.UNIFORM_BUFFER, this._buffer);
    gl.bufferData(gl.UNIFORM_BUFFER, this._cpuBuffer.byteLength, gl.DYNAMIC_DRAW);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, this.bindingPoint, this._buffer);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
  }

  // ── Field write helpers ──────────────────────────────────────────────────────

  /**
   * Write a float field by name into the CPU shadow buffer.
   * Call upload() / update() to push to GPU.
   */
  setFloat(fieldName: string, value: number): this {
    const field = this._field(fieldName, 'float');
    if (!field) return this;
    this._f32View[field.offset / 4] = value;
    return this;
  }

  setVec2(fieldName: string, x: number, y: number): this {
    const field = this._field(fieldName, 'vec2');
    if (!field) return this;
    const i = field.offset / 4;
    this._f32View[i]     = x;
    this._f32View[i + 1] = y;
    return this;
  }

  setVec4(fieldName: string, x: number, y: number, z: number, w: number): this {
    const field = this._field(fieldName, 'vec4');
    if (!field) return this;
    const i = field.offset / 4;
    this._f32View[i]     = x;
    this._f32View[i + 1] = y;
    this._f32View[i + 2] = z;
    this._f32View[i + 3] = w;
    return this;
  }

  /** mat4: expects Float32Array[16], column-major */
  setMat4(fieldName: string, mat: Float32Array): this {
    const field = this._field(fieldName, 'mat4');
    if (!field) return this;
    this._f32View.set(mat, field.offset / 4);
    return this;
  }

  // ── Bulk update (high-level) ─────────────────────────────────────────────────

  /**
   * Batch-set multiple fields and upload in one call.
   *
   * data keys must match UBOFieldDef.name values in the layout.
   * Values:
   *   float  → number
   *   vec2   → [x, y]
   *   vec4   → [x, y, z, w]
   *   mat4   → Float32Array[16]
   */
  update(data: Record<string, number | number[] | Float32Array>): void {
    for (const [key, val] of Object.entries(data)) {
      const field = this.layout.fields.find(f => f.name === key);
      if (!field) continue;

      if (field.type === 'float') {
        this.setFloat(key, val as number);
      } else if (field.type === 'vec2') {
        const v = val as number[];
        this.setVec2(key, v[0], v[1]);
      } else if (field.type === 'vec4') {
        const v = val as number[];
        this.setVec4(key, v[0], v[1], v[2], v[3]);
      } else if (field.type === 'mat4') {
        this.setMat4(key, val instanceof Float32Array ? val : new Float32Array(val as number[]));
      }
    }
    this.upload();
  }

  // ── GPU upload ───────────────────────────────────────────────────────────────

  /**
   * Push CPU shadow buffer → GPU via gl.bufferSubData.
   * WebGL2 only; no-op in WebGL1 (use fallbackBind instead).
   */
  upload(): void {
    if (!this.isWebGL2 || !this._buffer) return;
    const gl = this.renderer.context as WebGL2RenderingContext;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._buffer);
    // bufferSubData with srcOffset=0 for full upload — fastest path
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, this._f32View);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
  }

  /**
   * Upload only a sub-range of the buffer (byte offsets).
   * Useful when only one field changed (e.g. time each frame).
   */
  uploadRange(byteOffset: number, byteLength: number): void {
    if (!this.isWebGL2 || !this._buffer) return;
    const gl  = this.renderer.context as WebGL2RenderingContext;
    const sub = new Float32Array(this._cpuBuffer, byteOffset, byteLength / 4);
    gl.bindBuffer(gl.UNIFORM_BUFFER, this._buffer);
    gl.bufferSubData(gl.UNIFORM_BUFFER, byteOffset, sub);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
  }

  // ── WebGL1 fallback ──────────────────────────────────────────────────────────

  /**
   * WebGL1 降级: 用普通 uniform calls 将 CPU buffer 中的数据推给 program。
   * 调用方在 WebGL1 环境下用这个替代 upload()。
   *
   * 命名约定: GLSL uniform 名 = UBOFieldDef.name (无 block 前缀)。
   */
  fallbackBind(program: AstroProgram): void {
    if (this.isWebGL2) return; // no-op in WebGL2 — use upload() instead

    for (const field of this.layout.fields) {
      const i = field.offset / 4;
      switch (field.type) {
        case 'float':
          program.setFloat(field.name, this._f32View[i]);
          break;
        case 'vec2':
          program.setVec2(field.name, this._f32View[i], this._f32View[i + 1]);
          break;
        case 'vec4':
          program.setVec4(field.name,
            this._f32View[i], this._f32View[i + 1],
            this._f32View[i + 2], this._f32View[i + 3]);
          break;
        case 'mat4':
          program.setMatrix4(field.name, this._f32View.slice(i, i + 16));
          break;
      }
    }
  }

  // ── Bind point ───────────────────────────────────────────────────────────────

  /**
   * Re-bind this UBO's buffer to its binding point.
   * Usually done once per frame before draw calls.
   */
  bind(): void {
    if (!this.isWebGL2 || !this._buffer) return;
    const gl = this.renderer.context as WebGL2RenderingContext;
    gl.bindBufferBase(gl.UNIFORM_BUFFER, this.bindingPoint, this._buffer);
  }

  // ── CPU read-back ────────────────────────────────────────────────────────────

  /** Read a float field from CPU shadow buffer (no GPU round-trip). */
  getFloat(fieldName: string): number {
    const field = this._field(fieldName, 'float');
    return field ? this._f32View[field.offset / 4] : 0;
  }

  // ── Internal helpers ─────────────────────────────────────────────────────────

  private _field(name: string, expectedType: UBOFieldType): UBOFieldDef | null {
    const f = this.layout.fields.find(x => x.name === name);
    if (!f) {
      if (import.meta.env?.DEV) {
        console.warn(`[UniformBuffer:${this.layout.blockName}] unknown field "${name}"`);
      }
      return null;
    }
    if (f.type !== expectedType && import.meta.env?.DEV) {
      console.warn(
        `[UniformBuffer:${this.layout.blockName}] field "${name}" is ${f.type}, called as ${expectedType}`
      );
    }
    return f;
  }

  // ── Dispose ──────────────────────────────────────────────────────────────────

  dispose(): void {
    if (this._buffer) {
      this.renderer.context.deleteBuffer(this._buffer);
      this._buffer = null;
    }
  }
}

// ── Factory helpers ───────────────────────────────────────────────────────────

/** Create a CellUBO with the pre-defined layout. */
export function createCellUBO(renderer: AstroRenderer): UniformBuffer {
  return new UniformBuffer(renderer, CELL_UBO_LAYOUT);
}

/** Create a NukeUBO with the pre-defined layout. */
export function createNukeUBO(renderer: AstroRenderer): UniformBuffer {
  return new UniformBuffer(renderer, NUKE_UBO_LAYOUT);
}

/** Create a ViewUBO with the pre-defined layout. */
export function createViewUBO(renderer: AstroRenderer): UniformBuffer {
  return new UniformBuffer(renderer, VIEW_UBO_LAYOUT);
}

// ── GLSL snippet helpers ──────────────────────────────────────────────────────

/**
 * Generate the GLSL uniform block declaration for a layout.
 * Useful for shader source generation.
 *
 * Example output:
 *   layout(std140) uniform CellUBO {
 *     vec4  bbox;
 *     vec4  fillColor;
 *     float opacity;
 *     float time;
 *   };
 */
export function glslUniformBlock(layout: UBOLayoutDef): string {
  const glslType: Record<UBOFieldType, string> = {
    float: 'float',
    vec2:  'vec2',
    vec4:  'vec4',
    mat4:  'mat4',
  };
  const fields = layout.fields
    .map(f => `  ${glslType[f.type]} ${f.name};`)
    .join('\n');
  return `layout(std140) uniform ${layout.blockName} {\n${fields}\n};`;
}
