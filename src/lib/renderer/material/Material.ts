/**
 * Material.ts — 基础材质类
 *
 * AT Renderer 的材质系统基础层。对标 AT 中所有 shader class 共享的属性：
 *   - program       WebGL program 引用
 *   - uniforms      key-value uniform 存储（与 at_uil_categorized.json material 参数对齐）
 *   - blending      渲染混合模式（AT: shader_normal_blending / shader_additive_blending）
 *   - depthTest     深度测试
 *   - depthWrite    深度写入
 *   - side          渲染面（AT: shader_front_side / shader_back_side / shader_double_side）
 *
 * AT 分析：INPUT_Config 中的 transparent / blending / depthTest / depthWrite / side
 * 参数均由此基类统一管理。
 */

import type { AstroProgram } from '../AstroProgram';

// ── Types ────────────────────────────────────────────────────────────────────

export type BlendingMode =
  | 'shader_normal_blending'
  | 'shader_additive_blending'
  | 'shader_multiply_blending'
  | 'shader_no_blending';

export type SideMode =
  | 'shader_front_side'
  | 'shader_back_side'
  | 'shader_double_side';

export type UniformValue =
  | number
  | number[]
  | Float32Array
  | WebGLTexture
  | boolean
  | string;

export interface TextureDescriptor {
  src: string;
  compressed?: boolean | 'ktx2';
  filename?: string;
  prefix?: string;
  relative?: string;
  useCompressed?: boolean;
  hotreload?: boolean;
}

// ── Material ─────────────────────────────────────────────────────────────────

export class Material {
  /** AT: shader program (AstroProgram 封装) */
  program: AstroProgram | null = null;

  /** Flat uniform map — 与 at_uil_categorized.json material 键值对齐 */
  uniforms: Map<string, UniformValue> = new Map();

  /** AT INPUT_Config blending 字段 */
  blending: BlendingMode = 'shader_normal_blending';

  /** AT INPUT_Config depthTest 字段 */
  depthTest: boolean = true;

  /** AT INPUT_Config depthWrite 字段 */
  depthWrite: boolean = true;

  /** AT INPUT_Config side 字段 */
  side: SideMode = 'shader_front_side';

  /** AT INPUT_Config transparent 字段 */
  transparent: boolean = false;

  /** 材质名称 (对应 AT shader class name，如 'PBR', 'ATPBR', 'JellyShader'…) */
  readonly name: string;

  constructor(name: string = 'Material') {
    this.name = name;
  }

  // ── Uniform setters ──────────────────────────────────────────────────────────

  setUniform(name: string, value: UniformValue): this {
    this.uniforms.set(name, value);
    return this;
  }

  getUniform(name: string): UniformValue | undefined {
    return this.uniforms.get(name);
  }

  /**
   * 批量从 AT UIL JSON 参数对象中导入 uniform。
   * 参数 key 格式：'ShaderClass/ShaderClass/Element_N_scene/uParamName'
   * 此方法接受已经过 prefix 剥离的 flat 对象（只包含 uXxx / _tx_xxx 等参数名）。
   */
  importFromATParams(params: Record<string, UniformValue>): this {
    for (const [key, value] of Object.entries(params)) {
      this.uniforms.set(key, value);
    }
    return this;
  }

  // ── WebGL bind ───────────────────────────────────────────────────────────────

  /**
   * 绑定材质到 WebGL 上下文。
   * - 激活 program
   * - 配置 blending / depthTest / depthWrite / side
   * - 上传所有 uniform（需要 program 已设置）
   */
  bind(gl: WebGL2RenderingContext): void {
    if (!this.program) return;

    // Use program
    this.program.use();

    // Depth
    if (this.depthTest) {
      gl.enable(gl.DEPTH_TEST);
    } else {
      gl.disable(gl.DEPTH_TEST);
    }
    gl.depthMask(this.depthWrite);

    // Blending
    this._applyBlending(gl);

    // Face culling
    this._applySide(gl);

    // Upload uniforms
    this._uploadUniforms(gl);
  }

  // ── Private helpers ───────────────────────────────────────────────────────────

  private _applyBlending(gl: WebGL2RenderingContext): void {
    switch (this.blending) {
      case 'shader_normal_blending':
        gl.enable(gl.BLEND);
        gl.blendEquation(gl.FUNC_ADD);
        gl.blendFuncSeparate(
          gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA,
          gl.ONE,       gl.ONE_MINUS_SRC_ALPHA,
        );
        break;

      case 'shader_additive_blending':
        gl.enable(gl.BLEND);
        gl.blendEquation(gl.FUNC_ADD);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
        break;

      case 'shader_multiply_blending':
        gl.enable(gl.BLEND);
        gl.blendEquation(gl.FUNC_ADD);
        gl.blendFunc(gl.DST_COLOR, gl.ZERO);
        break;

      case 'shader_no_blending':
      default:
        gl.disable(gl.BLEND);
        break;
    }
  }

  private _applySide(gl: WebGL2RenderingContext): void {
    switch (this.side) {
      case 'shader_front_side':
        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.BACK);
        break;
      case 'shader_back_side':
        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.FRONT);
        break;
      case 'shader_double_side':
        gl.disable(gl.CULL_FACE);
        break;
    }
  }

  private _uploadUniforms(gl: WebGL2RenderingContext): void {
    if (!this.program) return;
    let textureUnit = 0;

    for (const [name, value] of this.uniforms) {
      // Skip texture descriptor objects — handled by subclass texture binding
      if (typeof value === 'object' && value !== null && !ArrayBuffer.isView(value) && !Array.isArray(value)) {
        continue;
      }

      const loc = this.program.uniformLocation(name);
      if (loc === null) continue;

      if (typeof value === 'number') {
        gl.uniform1f(loc, value);
      } else if (typeof value === 'boolean') {
        gl.uniform1i(loc, value ? 1 : 0);
      } else if (Array.isArray(value)) {
        switch (value.length) {
          case 1: gl.uniform1f(loc, value[0]); break;
          case 2: gl.uniform2fv(loc, value); break;
          case 3: gl.uniform3fv(loc, value); break;
          case 4: gl.uniform4fv(loc, value); break;
          case 9: gl.uniformMatrix3fv(loc, false, value); break;
          case 16: gl.uniformMatrix4fv(loc, false, value); break;
        }
      } else if (value instanceof Float32Array) {
        switch (value.length) {
          case 2: gl.uniform2fv(loc, value); break;
          case 3: gl.uniform3fv(loc, value); break;
          case 4: gl.uniform4fv(loc, value); break;
          case 9: gl.uniformMatrix3fv(loc, false, value); break;
          case 16: gl.uniformMatrix4fv(loc, false, value); break;
        }
      } else if (value instanceof WebGLTexture) {
        gl.activeTexture(gl.TEXTURE0 + textureUnit);
        gl.bindTexture(gl.TEXTURE_2D, value);
        gl.uniform1i(loc, textureUnit);
        textureUnit++;
      }
    }
  }

  // ── Disposal ─────────────────────────────────────────────────────────────────

  dispose(): void {
    this.uniforms.clear();
    this.program = null;
  }
}
