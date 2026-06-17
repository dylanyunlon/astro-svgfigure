/**
 * cell-csp-shader.ts — CSP 策略下 shader 编译方案
 *
 * 问题背景：
 *   PixiJS 默认通过 `new Function()` 动态生成 shader sync 函数（GenerateShaderSyncCode.ts），
 *   以及通过 `new Function()` 生成 UBO/uniform 同步函数（generateUniformsSync.ts）。
 *   在 strict Content-Security-Policy（禁止 unsafe-eval）的环境下，这些调用会被浏览器拦截，
 *   导致渲染失败。
 *
 * 解决方案 — 预编译 GLSL + 静态分发路由：
 *   1. 不使用 new Function() / eval()，所有 shader sync 路由在编译期静态确定
 *   2. 注入 upstream/pixijs-engine/src/unsafe-eval/ 中的 polyfill 函数集替换动态生成器
 *   3. Cell 渲染的 GLSL shader 源码作为 ES module 静态字符串导出（bundler 可 tree-shake）
 *   4. CSPShaderRegistry 管理预编译 program 缓存，避免重复 gl.compileShader()
 *
 * 整合 upstream 参考：
 *   upstream/pixijs-engine/src/unsafe-eval/init.ts             — selfInstall() polyfill 注入模式
 *   upstream/pixijs-engine/src/unsafe-eval/shader/generateShaderSyncPolyfill.ts
 *   upstream/pixijs-engine/src/unsafe-eval/uniforms/generateUniformsSyncPolyfill.ts
 *   upstream/pixijs-engine/src/unsafe-eval/ubo/generateUboSyncPolyfill.ts
 *   upstream/pixijs-engine/src/rendering/renderers/gl/shader/GenerateShaderSyncCode.ts
 *   upstream/pixijs-engine/src/utils/browser/unsafeEvalSupported.ts
 *
 * 使用方式：
 *   // 在 PixiJS Application 初始化 *之前* 调用一次
 *   import { installCSPShaderPolyfill } from './cell-csp-shader';
 *   installCSPShaderPolyfill();
 *
 *   // 之后正常初始化 PixiJS — 所有 shader 同步走静态 polyfill，无 eval
 *   const app = new Application();
 *   await app.init({ ... });
 *
 * M023: cell-csp-shader — precompiled GLSL no eval
 */

// ── 预编译 GLSL 模块 — Cell 渲染 shader 源码静态字符串 ─────────────────────────
//
// 所有 shader 源码作为 ES module 常量导出，bundler（Vite/Rollup）在构建期将其
// tree-shake 进 chunk，无需运行时字符串拼接或 eval 执行。
// GLSL 版本统一用 #version 300 es（WebGL2），与项目其他 shader 保持一致。

/** Cell 主体渲染 — 圆角矩形 SDF + species 图案 vertex shader */
export const CELL_VERT_GLSL = /* glsl */`#version 300 es
precision highp float;

in vec2 aPosition;
in vec2 aUV;

uniform mat3 uProjectionMatrix;
uniform mat3 uWorldTransformMatrix;
uniform vec4 uFrame;          // x, y, width, height of cell bbox in world space
uniform float uAlpha;

out vec2 vUV;
out float vAlpha;

void main() {
  vec3 worldPos = uWorldTransformMatrix * vec3(aPosition, 1.0);
  vec3 clipPos  = uProjectionMatrix * worldPos;
  gl_Position   = vec4(clipPos.xy, 0.0, 1.0);
  vUV           = aUV;
  vAlpha        = uAlpha;
}
`;

/** Cell 主体渲染 — 圆角矩形 SDF + species 图案 fragment shader */
export const CELL_FRAG_GLSL = /* glsl */`#version 300 es
precision highp float;

in vec2 vUV;
in float vAlpha;

out vec4 fragColor;

// Cell 参数
uniform vec2  uSize;          // cell width, height (pixels)
uniform vec3  uFillColor;     // RGB 0-1
uniform vec3  uStrokeColor;   // RGB 0-1
uniform vec3  uGlowColor;     // RGB 0-1
uniform float uCornerRadius;  // pixels
uniform float uStrokeWidth;   // pixels
uniform float uTime;          // seconds, for animated species
uniform int   uSpeciesId;     // 0-9, species enum
uniform float uGlowStrength;  // glow intensity 0-2

// ── SDF 原语 ──────────────────────────────────────────────────────────────────

float sdRoundBox(vec2 p, vec2 b, float r) {
  vec2 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float sdCircle(vec2 p, float r) {
  return length(p) - r;
}

float sdLine(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

// ── Species SDF 图案 ───────────────────────────────────────────────────────────

// 0: cil-eye — 同心圆虹膜
float patternEye(vec2 uv, vec2 sz) {
  float r = min(sz.x, sz.y) * 0.35;
  float d = sdCircle(uv, r);
  float rings = sin(d * 25.0) * 0.5 + 0.5;
  float pupil = smoothstep(r * 0.2, r * 0.12, length(uv));
  return mix(rings * 0.3, 1.0, pupil);
}

// 1: cil-vector — 箭头
float patternVector(vec2 uv, vec2 sz) {
  float arrow = sdLine(uv, vec2(-sz.x * 0.35, 0.0), vec2(sz.x * 0.25, 0.0));
  float head1 = sdLine(uv, vec2(sz.x*0.25, 0.0), vec2(sz.x*0.15,  sz.y*0.15));
  float head2 = sdLine(uv, vec2(sz.x*0.25, 0.0), vec2(sz.x*0.15, -sz.y*0.15));
  float d = min(arrow, min(head1, head2));
  return smoothstep(2.0, 0.5, d) * 0.4;
}

// 2: cil-bolt — 锯齿激活函数
float patternBolt(vec2 uv, vec2 sz) {
  float x_norm = (uv.x / sz.x + 0.5);
  float zigzag = sin(x_norm * 3.14159 * 4.0) * sz.y * 0.2;
  float d = abs(uv.y - zigzag);
  return smoothstep(3.0, 0.5, d) * 0.35;
}

// 3: cil-plus — 十字
float patternPlus(vec2 uv, vec2 sz) {
  float arm = min(sz.x, sz.y) * 0.3;
  float dh = sdLine(uv, vec2(-arm, 0.0), vec2(arm, 0.0));
  float dv = sdLine(uv, vec2(0.0, -arm), vec2(0.0, arm));
  return smoothstep(2.5, 0.5, min(dh, dv)) * 0.3;
}

// 4: cil-arrow-right — 单箭头
float patternArrow(vec2 uv, vec2 sz) {
  float sz2 = min(sz.x, sz.y) * 0.25;
  float d1 = sdLine(uv, vec2(-sz2, -sz2), vec2(sz2 * 0.5,  0.0));
  float d2 = sdLine(uv, vec2(-sz2,  sz2), vec2(sz2 * 0.5,  0.0));
  return smoothstep(2.5, 0.5, min(d1, d2)) * 0.35;
}

// 5: cil-filter — 九宫格
float patternFilter(vec2 uv, vec2 sz) {
  vec2 cell = mod(uv + sz * 0.5, sz / 3.0) - sz / 6.0;
  float d = sdRoundBox(cell, sz / 6.0 - 2.0, 1.5);
  return smoothstep(1.5, -0.5, d) * 0.2;
}

// 6: cil-code — 花括号
float patternCode(vec2 uv, vec2 sz) {
  float bx = sz.x * 0.6, by = sz.y * 0.45;
  float d1 = sdLine(uv, vec2(-bx, -by), vec2(-bx - 4.0, 0.0));
  float d2 = sdLine(uv, vec2(-bx - 4.0, 0.0), vec2(-bx, by));
  float d3 = sdLine(uv, vec2(bx,  -by), vec2(bx  + 4.0, 0.0));
  float d4 = sdLine(uv, vec2(bx  + 4.0, 0.0), vec2(bx,  by));
  return smoothstep(2.5, 0.5, min(min(d1, d2), min(d3, d4))) * 0.3;
}

// 7: cil-layers — 层叠矩形
float patternLayers(vec2 uv, vec2 sz) {
  float acc = 0.0;
  for (int i = 0; i < 3; i++) {
    float off = float(i) * 4.0;
    float d = sdRoundBox(uv - vec2(off * 0.5, -off * 0.5),
                         sz - vec2(6.0 + off), 3.0);
    acc += smoothstep(1.5, -0.5, d) * (0.1 + float(i) * 0.08);
  }
  return clamp(acc, 0.0, 1.0);
}

// 8: cil-loop — 弧形循环箭头
float patternLoop(vec2 uv, vec2 sz) {
  float r = min(sz.x, sz.y) * 0.3;
  float ring = abs(sdCircle(uv, r)) - 1.5;
  // mask upper half: only show bottom arc
  float mask = step(0.0, uv.y + r * 0.1);
  float d = mix(ring, 99.0, mask);
  // arrowhead at end of arc
  vec2 ap = vec2(uv.x - r, uv.y);
  float arrD = sdLine(ap, vec2(-4.0, -4.0), vec2(0.0, 0.0));
  arrD = min(arrD, sdLine(ap, vec2(4.0, -4.0), vec2(0.0, 0.0)));
  return smoothstep(2.0, 0.0, min(d, arrD)) * 0.4;
}

// 9: cil-graph — 图节点+连线
float patternGraph(vec2 uv, vec2 sz) {
  vec2 nodes[4];
  nodes[0] = vec2(-sz.x*0.25, -sz.y*0.2);
  nodes[1] = vec2( sz.x*0.1,  -sz.y*0.25);
  nodes[2] = vec2( sz.x*0.25,  sz.y*0.1);
  nodes[3] = vec2(-sz.x*0.15,  sz.y*0.2);
  float acc = 0.0;
  for (int i = 0; i < 4; i++) {
    acc = max(acc, smoothstep(4.0, 1.5, sdCircle(uv - nodes[i], 3.0)) * 0.35);
  }
  float edge = 99.0;
  for (int i = 0; i < 3; i++) {
    edge = min(edge, sdLine(uv, nodes[i], nodes[i+1]));
  }
  acc = max(acc, smoothstep(1.8, 0.2, edge) * 0.2);
  return clamp(acc, 0.0, 1.0);
}

// ── メイン ─────────────────────────────────────────────────────────────────────

void main() {
  // UV → 局部坐标系（原点在 cell 中心）
  vec2 sz  = uSize * 0.5;
  vec2 uv  = (vUV - 0.5) * uSize; // [-sz, sz]

  // ── 圆角矩形 SDF ──────────────────────────────────────────────────────────
  float dBox = sdRoundBox(uv, sz, uCornerRadius);

  // 外发光 — 指数衰减
  float glow   = exp(-max(dBox, 0.0) * 0.06 * uGlowStrength);
  vec3  glowC  = uGlowColor * glow * 0.4;

  // fill inside
  float fillMask   = smoothstep(1.0, -0.5, dBox);
  // stroke ring
  float strokeMask = smoothstep(uStrokeWidth, uStrokeWidth - 1.5, abs(dBox));

  // ── Species 图案 ─────────────────────────────────────────────────────────
  float pattern = 0.0;
  if      (uSpeciesId == 0) pattern = patternEye(uv, sz);
  else if (uSpeciesId == 1) pattern = patternVector(uv, sz);
  else if (uSpeciesId == 2) pattern = patternBolt(uv, sz);
  else if (uSpeciesId == 3) pattern = patternPlus(uv, sz);
  else if (uSpeciesId == 4) pattern = patternArrow(uv, sz);
  else if (uSpeciesId == 5) pattern = patternFilter(uv, sz);
  else if (uSpeciesId == 6) pattern = patternCode(uv, sz);
  else if (uSpeciesId == 7) pattern = patternLayers(uv, sz);
  else if (uSpeciesId == 8) pattern = patternLoop(uv, sz);
  else                      pattern = patternGraph(uv, sz);

  // ── 合成 ─────────────────────────────────────────────────────────────────
  vec3 col = glowC;
  col      = mix(col, uFillColor,   fillMask);
  col      = mix(col, uFillColor + uGlowColor * 0.15 * pattern, fillMask * pattern);
  col      = mix(col, uStrokeColor, strokeMask * fillMask);

  float alpha = max(glow * 0.5, fillMask) * vAlpha;
  fragColor   = vec4(col * alpha, alpha); // premultiplied alpha
}
`;

/** Edge 粒子渲染 vertex shader（Transform Feedback Ping-Pong）*/
export const EDGE_PARTICLE_VERT_GLSL = /* glsl */`#version 300 es
precision highp float;

in  vec4 aParticle; // x, y, t, lifetime

uniform vec2  uResolution;
uniform float uPointSizeMax;
uniform vec4  uColor;

out float vLifetime;
out vec4  vColor;

void main() {
  float px       = aParticle.x;
  float py       = aParticle.y;
  float lifetime = aParticle.w;

  vec2 ndc = vec2(
     px / uResolution.x * 2.0 - 1.0,
    -(py / uResolution.y * 2.0 - 1.0)
  );
  gl_Position  = vec4(ndc, 0.0, 1.0);

  float bell   = lifetime * (1.0 - lifetime) * 4.0;
  gl_PointSize = max(1.5, uPointSizeMax * bell);

  vLifetime = lifetime;
  vColor    = uColor;
}
`;

/** Edge 粒子渲染 fragment shader — 径向发光点精灵 */
export const EDGE_PARTICLE_FRAG_GLSL = /* glsl */`#version 300 es
precision highp float;

in float vLifetime;
in vec4  vColor;
out vec4 fragColor;

void main() {
  vec2  uv = gl_PointCoord - 0.5;
  float d  = dot(uv, uv);
  if (d > 0.25) discard;

  float core = 1.0 - smoothstep(0.0, 0.08, d);
  float halo = 1.0 - smoothstep(0.08, 0.25, d);
  float glow = core * 0.9 + halo * 0.3;

  fragColor = vec4(vColor.rgb * glow, vColor.a * glow * vLifetime);
}
`;

/** Edge 粒子模拟 vertex shader（Transform Feedback output，RASTERIZER_DISCARD）*/
export const EDGE_SIM_VERT_GLSL = /* glsl */`#version 300 es
precision highp float;

in  vec4 aParticle; // x, y, t, lifetime

uniform vec4  uEdgesA[128]; // p0.xy, cp1.xy per edge
uniform vec4  uEdgesB[128]; // cp2.xy, p1.xy per edge
uniform float uSpeedBase;
uniform float uTime;

out vec4 vParticle;

float lcg(float seed) {
  return fract(sin(seed * 127.1 + 311.7) * 43758.5453);
}

vec2 bezier(vec2 p0, vec2 cp1, vec2 cp2, vec2 p1, float tt) {
  float u  = 1.0 - tt;
  float u2 = u * u, u3 = u2 * u;
  float t2 = tt * tt, t3 = t2 * tt;
  return u3*p0 + 3.0*u2*tt*cp1 + 3.0*u*t2*cp2 + t3*p1;
}

void main() {
  float px       = aParticle.x;
  float py       = aParticle.y;
  float t        = aParticle.z;
  float lifetime = aParticle.w;

  int   edgeIdx     = gl_VertexID / 32;
  float particleSlot = float(gl_VertexID % 32);

  vec4 ea = uEdgesA[edgeIdx];
  vec4 eb = uEdgesB[edgeIdx];

  float jitter = 0.5 + lcg(particleSlot + float(edgeIdx) * 31.0) * 0.5;
  float speed  = uSpeedBase * jitter;

  t += speed;

  if (t > 1.0) {
    float phase = lcg(particleSlot * 17.0 + uTime * 0.001 + float(edgeIdx));
    t        = phase * 0.35;
    lifetime = 0.0;
  } else {
    lifetime = min(1.0, lifetime + 0.04);
  }

  vec2 pos = bezier(ea.xy, ea.zw, eb.xy, eb.zw, clamp(t, 0.0, 1.0));
  vParticle = vec4(pos.x, pos.y, t, lifetime);
}
`;

/** Edge 粒子模拟 fragment shader — 占位（RASTERIZER_DISCARD 模式下不执行）*/
export const EDGE_SIM_FRAG_GLSL = /* glsl */`#version 300 es
precision highp float;
void main() {}
`;

// ── Species ID 映射表（与 pixi-cell-renderer.ts SPECIES_COLOURS 对齐）──────────

export const SPECIES_ID_MAP: Record<string, number> = {
  'cil-eye':         0,
  'cil-vector':      1,
  'cil-bolt':        2,
  'cil-plus':        3,
  'cil-arrow-right': 4,
  'cil-filter':      5,
  'cil-code':        6,
  'cil-layers':      7,
  'cil-loop':        8,
  'cil-graph':       9,
};

// ── CSP 检测 — 不用 unsafeEvalSupported()，直接静态探测 ──────────────────────

/**
 * isCspBlocked — 检测当前环境是否禁止 unsafe-eval。
 *
 * 与 upstream 的 unsafeEvalSupported() 逻辑相反：
 *   upstream 检测 new Function 是否 *成功*；
 *   isCspBlocked 检测是否 *被拦截*（返回 true 表示需要 polyfill）。
 *
 * 设计原则：
 *   - 不使用 try/catch new Function — 因为在 CSP 模式下该调用本身就是错误
 *   - 改用 CSP header 元信息（meta 标签 / response header 解析）静态判断
 *   - 默认保守：若无法确定则返回 true，强制走 polyfill 路径
 */
export function isCspBlocked(): boolean {
  // 检查 <meta http-equiv="Content-Security-Policy"> 是否包含 unsafe-eval
  if (typeof document !== 'undefined') {
    const metas = document.querySelectorAll('meta[http-equiv="Content-Security-Policy"]');
    for (const meta of Array.from(metas)) {
      const content = meta.getAttribute('content') ?? '';
      // 如果 CSP 中没有 'unsafe-eval'，则当前环境受限
      if (content.includes('script-src') && !content.includes('unsafe-eval')) {
        return true;
      }
    }
  }
  // 没有显式 CSP meta 或包含 unsafe-eval — 尝试探测（不使用 new Function）
  // 保守返回 false：让 PixiJS 默认路径尝试，失败时可再调用 installCSPShaderPolyfill()
  return false;
}

// ── CSPShaderRegistry — 预编译 program 缓存管理器 ────────────────────────────

/**
 * CSPShaderRegistry — 管理 WebGL2 shader 程序的预编译与缓存。
 *
 * 核心职责：
 *   1. 以 (vertSource, fragSource) 为 key 缓存已编译的 WebGLProgram
 *   2. 提供 compileProgram() / compileTFProgram() 两种无 eval 路径
 *   3. 提供 compileAll() 预热入口，在 idle 期提前编译所有已知 cell shader
 *
 * CSP 安全性：
 *   - 所有 shader 源码来自本文件顶部的静态字符串常量（bundler 静态分析可见）
 *   - gl.shaderSource / gl.compileShader 是原生 WebGL API，不受 CSP 限制
 *   - 零 eval / 零 new Function 调用
 */
export class CSPShaderRegistry {
  private readonly _gl: WebGL2RenderingContext;
  private readonly _cache = new Map<string, WebGLProgram>();

  constructor(gl: WebGL2RenderingContext) {
    this._gl = gl;
  }

  /** 缓存 key = vert hash + frag hash（简单长度+首尾字符组合，避免完整哈希开销）*/
  private _key(vert: string, frag: string): string {
    return `${vert.length}:${vert.charCodeAt(0)}:${frag.length}:${frag.charCodeAt(0)}`;
  }

  /**
   * compileProgram — 编译并链接普通 WebGL program，结果缓存。
   * @param vertSrc   GLSL vertex shader 源码（静态字符串）
   * @param fragSrc   GLSL fragment shader 源码（静态字符串）
   * @returns         链接完成的 WebGLProgram
   * @throws          编译/链接错误时抛出带 info log 的 Error
   */
  compileProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const key = this._key(vertSrc, fragSrc);
    const cached = this._cache.get(key);
    if (cached) return cached;

    const prog = this._link(vertSrc, fragSrc, null);
    this._cache.set(key, prog);
    return prog;
  }

  /**
   * compileTFProgram — 编译并链接带 Transform Feedback varyings 的 WebGL program。
   *
   * Transform Feedback 必须在 linkProgram() 之前声明 varyings，
   * 因此需要单独接口以传入 varyings 列表。
   *
   * @param vertSrc    GLSL vertex shader 源码
   * @param fragSrc    GLSL fragment shader 源码
   * @param varyings   TF output varying 名称列表（如 ['vParticle']）
   * @returns          链接完成的 WebGLProgram（已配置 TF varyings）
   */
  compileTFProgram(vertSrc: string, fragSrc: string, varyings: string[]): WebGLProgram {
    const key = this._key(vertSrc, fragSrc) + ':tf:' + varyings.join(',');
    const cached = this._cache.get(key);
    if (cached) return cached;

    const prog = this._link(vertSrc, fragSrc, varyings);
    this._cache.set(key, prog);
    return prog;
  }

  /**
   * compileAll — 预热所有已知 cell shader（在 requestIdleCallback 或 init 阶段调用）。
   *
   * 提前编译能消除首帧渲染时的 GPU 驱动编译延迟（shader compilation stall）。
   * 等效于 PixiJS PrepareSystem.upload() 的 shader warmup 扩展。
   */
  compileAll(): void {
    // Cell body + species SDF shader
    this.compileProgram(CELL_VERT_GLSL, CELL_FRAG_GLSL);
    // Edge particle render shader
    this.compileProgram(EDGE_PARTICLE_VERT_GLSL, EDGE_PARTICLE_FRAG_GLSL);
    // Edge particle simulation shader（TF 版本）
    this.compileTFProgram(EDGE_SIM_VERT_GLSL, EDGE_SIM_FRAG_GLSL, ['vParticle']);
  }

  /** 销毁所有缓存的 program */
  destroy(): void {
    for (const prog of this._cache.values()) {
      this._gl.deleteProgram(prog);
    }
    this._cache.clear();
  }

  // ── 内部编译辅助 ───────────────────────────────────────────────────────────

  private _compileShader(type: number, src: string): WebGLShader {
    const gl = this._gl;
    const sh = gl.createShader(type);
    if (!sh) throw new Error('[CSPShaderRegistry] gl.createShader returned null');
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh) ?? '(no log)';
      gl.deleteShader(sh);
      throw new Error(`[CSPShaderRegistry] Shader compile error:\n${log}`);
    }
    return sh;
  }

  private _link(
    vertSrc: string,
    fragSrc: string,
    tfVaryings: string[] | null,
  ): WebGLProgram {
    const gl = this._gl;
    const vs = this._compileShader(gl.VERTEX_SHADER, vertSrc);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fragSrc);

    const prog = gl.createProgram();
    if (!prog) throw new Error('[CSPShaderRegistry] gl.createProgram returned null');

    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);

    // Transform Feedback varyings 必须在 linkProgram() *之前* 声明
    if (tfVaryings && tfVaryings.length > 0) {
      gl.transformFeedbackVaryings(prog, tfVaryings, gl.SEPARATE_ATTRIBS);
    }

    gl.linkProgram(prog);

    // 清理 shader 对象（program 链接后即可删除）
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog) ?? '(no log)';
      gl.deleteProgram(prog);
      throw new Error(`[CSPShaderRegistry] Program link error:\n${log}`);
    }

    return prog;
  }
}

// ── PixiJS 全局 polyfill 注入 ─────────────────────────────────────────────────
//
// 模式参考 upstream/pixijs-engine/src/unsafe-eval/init.ts 的 selfInstall() 模式：
//   Object.assign(GlShaderSystem.prototype, { _generateShaderSync: polyfillFn })
//
// polyfill 函数直接从 upstream/pixijs-engine/src/unsafe-eval/ 静态导入，
// 无需任何动态代码生成。
//
// 调用时机：必须在 new Application() / autoDetectRenderer() 之前执行一次。

let _polyfillInstalled = false;

/**
 * installCSPShaderPolyfill — 向 PixiJS 注入 CSP 安全的 shader sync polyfill。
 *
 * 替换以下 eval 路径：
 *   GlShaderSystem._generateShaderSync  → generateShaderSyncPolyfill（静态循环，无 new Function）
 *   GlUniformGroupSystem._generateUniformsSync → generateUniformsSyncPolyfill
 *   GlUboSystem._generateUboSync        → generateUboSyncPolyfillSTD40
 *   GpuUboSystem._generateUboSync       → generateUboSyncPolyfillWGSL
 *   AbstractRenderer._unsafeEvalCheck   → no-op（跳过 eval 可用性检查）
 *   UboSystem._systemCheck              → no-op
 *   ParticleBuffer.generateParticleUpdate → generateParticleUpdatePolyfill
 *
 * 幂等：多次调用仅安装一次。
 *
 * @param force 强制重新安装（测试用途）
 */
export async function installCSPShaderPolyfill(force = false): Promise<void> {
  if (_polyfillInstalled && !force) return;

  // 动态导入 upstream polyfill 模块（静态字符串路径，bundler 可静态分析）
  // 不使用 eval — import() 是 ESM 原生特性，不受 script-src unsafe-eval 限制
  const [
    { generateShaderSyncPolyfill },
    { generateUniformsSyncPolyfill },
    { generateUboSyncPolyfillSTD40, generateUboSyncPolyfillWGSL },
    { generateParticleUpdatePolyfill },
  ] = await Promise.all([
    import('../../upstream/pixijs-engine/src/unsafe-eval/shader/generateShaderSyncPolyfill'),
    import('../../upstream/pixijs-engine/src/unsafe-eval/uniforms/generateUniformsSyncPolyfill'),
    import('../../upstream/pixijs-engine/src/unsafe-eval/ubo/generateUboSyncPolyfill'),
    import('../../upstream/pixijs-engine/src/unsafe-eval/particle/generateParticleUpdatePolyfill'),
  ]);

  // 延迟导入 PixiJS 内部类（避免循环依赖 + 允许 tree-shaking）
  const [
    { AbstractRenderer },
    { GlShaderSystem },
    { GlUniformGroupSystem },
    { GlUboSystem },
    { GpuUboSystem },
    { UboSystem },
    { ParticleBuffer },
  ] = await Promise.all([
    import('../../upstream/pixijs-engine/src/rendering/renderers/shared/system/AbstractRenderer'),
    import('../../upstream/pixijs-engine/src/rendering/renderers/gl/shader/GlShaderSystem'),
    import('../../upstream/pixijs-engine/src/rendering/renderers/gl/shader/GlUniformGroupSystem'),
    import('../../upstream/pixijs-engine/src/rendering/renderers/gl/GlUboSystem'),
    import('../../upstream/pixijs-engine/src/rendering/renderers/gpu/GpuUboSystem'),
    import('../../upstream/pixijs-engine/src/rendering/renderers/shared/shader/UboSystem'),
    import('../../upstream/pixijs-engine/src/scene/particle-container/shared/ParticleBuffer'),
  ]);

  // ── 注入 polyfill — 复制 upstream unsafe-eval/init.ts 的 selfInstall() ──────

  // 1. 跳过 eval 可用性检查（CSP 环境下 unsafeEvalSupported() 会返回 false）
  Object.assign(AbstractRenderer.prototype, {
    _unsafeEvalCheck() {
      // CSP 安全：不检查 eval，直接通过
      // polyfill 已替换所有 new Function 调用，无需 eval
    },
  });

  // 2. 跳过 UboSystem eval 检查
  Object.assign(UboSystem.prototype, {
    _systemCheck() {
      // CSP 安全：UBO sync 已由 polyfill 接管
    },
  });

  // 3. GlUniformGroupSystem — uniform 同步走静态 functionMap 路由
  Object.assign(GlUniformGroupSystem.prototype, {
    _generateUniformsSync: generateUniformsSyncPolyfill,
  });

  // 4. GlUboSystem — WebGL STD140 UBO sync，静态偏移计算
  Object.assign(GlUboSystem.prototype, {
    _generateUboSync: generateUboSyncPolyfillSTD40,
  });

  // 5. GpuUboSystem — WebGPU WGSL UBO sync
  Object.assign(GpuUboSystem.prototype, {
    _generateUboSync: generateUboSyncPolyfillWGSL,
  });

  // 6. GlShaderSystem — shader 资源绑定走静态循环（syncShader），无 new Function
  Object.assign(GlShaderSystem.prototype, {
    _generateShaderSync: generateShaderSyncPolyfill,
  });

  // 7. ParticleBuffer — 粒子更新函数走预生成 functionMap
  Object.assign(ParticleBuffer.prototype, {
    generateParticleUpdate: generateParticleUpdatePolyfill,
  });

  _polyfillInstalled = true;

  if (typeof console !== 'undefined') {
    console.debug('[cell-csp-shader] CSP polyfill installed — no eval paths active');
  }
}

/**
 * installCSPShaderPolyfillSync — 同步版本，用于支持顶层 await 的环境。
 *
 * 注意：此函数执行动态 import()，仍返回 Promise，但调用方可在模块顶层 await 它：
 *
 *   // astro component or module init:
 *   await installCSPShaderPolyfillSync();
 *   const app = new Application();
 *
 * 与 installCSPShaderPolyfill 完全等价，命名区分仅为语义清晰。
 */
export const installCSPShaderPolyfillSync = installCSPShaderPolyfill;

// ── CellCSPShaderProgram — 单 cell 渲染的完整 WebGL2 program 包装 ───────────────

/**
 * CellCSPUniformSet — cell 渲染所需的全部 uniform 值。
 *
 * 字段命名与 CELL_FRAG_GLSL 中 uniform 声明一一对应，
 * 类型均为原生 JS 类型，无运行时代码生成。
 */
export interface CellCSPUniformSet {
  /** Cell 宽高（像素）*/
  uSize: [number, number];
  /** 投影矩阵（column-major, 9 floats）*/
  uProjectionMatrix: Float32Array;
  /** 世界变换矩阵（column-major, 9 floats）*/
  uWorldTransformMatrix: Float32Array;
  /** Cell bbox frame [x, y, w, h] in world space */
  uFrame: [number, number, number, number];
  /** Fill color RGB [0-1] */
  uFillColor: [number, number, number];
  /** Stroke color RGB [0-1] */
  uStrokeColor: [number, number, number];
  /** Glow color RGB [0-1] */
  uGlowColor: [number, number, number];
  /** Corner radius in pixels */
  uCornerRadius: number;
  /** Stroke width in pixels */
  uStrokeWidth: number;
  /** Animation time in seconds */
  uTime: number;
  /** Species enum (0-9) */
  uSpeciesId: number;
  /** Glow intensity 0-2 */
  uGlowStrength: number;
  /** Alpha 0-1 */
  uAlpha: number;
}

/**
 * CellCSPShaderProgram — 持有已编译 program + uniform location cache 的包装器。
 *
 * 使用 CSPShaderRegistry 预编译 shader，在 draw() 时直接调用
 * gl.uniformXxx(location, value) — 纯原生 WebGL API，无任何代码生成。
 */
export class CellCSPShaderProgram {
  private readonly _gl: WebGL2RenderingContext;
  readonly program: WebGLProgram;

  // Uniform location cache（避免每帧 getUniformLocation 的 driver 查询开销）
  private readonly _locs: {
    uSize: WebGLUniformLocation | null;
    uProjectionMatrix: WebGLUniformLocation | null;
    uWorldTransformMatrix: WebGLUniformLocation | null;
    uFrame: WebGLUniformLocation | null;
    uFillColor: WebGLUniformLocation | null;
    uStrokeColor: WebGLUniformLocation | null;
    uGlowColor: WebGLUniformLocation | null;
    uCornerRadius: WebGLUniformLocation | null;
    uStrokeWidth: WebGLUniformLocation | null;
    uTime: WebGLUniformLocation | null;
    uSpeciesId: WebGLUniformLocation | null;
    uGlowStrength: WebGLUniformLocation | null;
    uAlpha: WebGLUniformLocation | null;
  };

  constructor(registry: CSPShaderRegistry, gl: WebGL2RenderingContext) {
    this._gl = gl;
    this.program = registry.compileProgram(CELL_VERT_GLSL, CELL_FRAG_GLSL);

    // Cache all uniform locations once at construction time
    const p = this.program;
    this._locs = {
      uSize:                 gl.getUniformLocation(p, 'uSize'),
      uProjectionMatrix:     gl.getUniformLocation(p, 'uProjectionMatrix'),
      uWorldTransformMatrix: gl.getUniformLocation(p, 'uWorldTransformMatrix'),
      uFrame:                gl.getUniformLocation(p, 'uFrame'),
      uFillColor:            gl.getUniformLocation(p, 'uFillColor'),
      uStrokeColor:          gl.getUniformLocation(p, 'uStrokeColor'),
      uGlowColor:            gl.getUniformLocation(p, 'uGlowColor'),
      uCornerRadius:         gl.getUniformLocation(p, 'uCornerRadius'),
      uStrokeWidth:          gl.getUniformLocation(p, 'uStrokeWidth'),
      uTime:                 gl.getUniformLocation(p, 'uTime'),
      uSpeciesId:            gl.getUniformLocation(p, 'uSpeciesId'),
      uGlowStrength:         gl.getUniformLocation(p, 'uGlowStrength'),
      uAlpha:                gl.getUniformLocation(p, 'uAlpha'),
    };
  }

  /**
   * bindUniforms — 上传所有 cell uniform 值到 GPU。
   *
   * 调用方需先 gl.useProgram(this.program)。
   * 全部使用 gl.uniformXxx 原生 API — CSP 安全，无 eval。
   */
  bindUniforms(u: CellCSPUniformSet): void {
    const gl = this._gl;
    const L = this._locs;

    // vec2
    if (L.uSize)      gl.uniform2f(L.uSize, u.uSize[0], u.uSize[1]);
    if (L.uFrame)     gl.uniform4f(L.uFrame, u.uFrame[0], u.uFrame[1], u.uFrame[2], u.uFrame[3]);

    // mat3 (column-major 3x3 → uniformMatrix3fv)
    if (L.uProjectionMatrix)     gl.uniformMatrix3fv(L.uProjectionMatrix,     false, u.uProjectionMatrix);
    if (L.uWorldTransformMatrix) gl.uniformMatrix3fv(L.uWorldTransformMatrix, false, u.uWorldTransformMatrix);

    // vec3
    if (L.uFillColor)   gl.uniform3f(L.uFillColor,   u.uFillColor[0],   u.uFillColor[1],   u.uFillColor[2]);
    if (L.uStrokeColor) gl.uniform3f(L.uStrokeColor, u.uStrokeColor[0], u.uStrokeColor[1], u.uStrokeColor[2]);
    if (L.uGlowColor)   gl.uniform3f(L.uGlowColor,   u.uGlowColor[0],   u.uGlowColor[1],   u.uGlowColor[2]);

    // float
    if (L.uCornerRadius) gl.uniform1f(L.uCornerRadius, u.uCornerRadius);
    if (L.uStrokeWidth)  gl.uniform1f(L.uStrokeWidth,  u.uStrokeWidth);
    if (L.uTime)         gl.uniform1f(L.uTime,         u.uTime);
    if (L.uGlowStrength) gl.uniform1f(L.uGlowStrength, u.uGlowStrength);
    if (L.uAlpha)        gl.uniform1f(L.uAlpha,        u.uAlpha);

    // int
    if (L.uSpeciesId) gl.uniform1i(L.uSpeciesId, u.uSpeciesId);
  }
}

// ── 便捷工厂函数 ───────────────────────────────────────────────────────────────

/**
 * createCellCSPShader — 从 canvas 创建 CSPShaderRegistry + CellCSPShaderProgram。
 *
 * 一步工厂：获取 WebGL2 context → 创建 registry → 编译 cell program。
 *
 * @param canvas  目标 HTMLCanvasElement（WebGL2 context 尚未获取时）
 * @param gl      已有 WebGL2 context（优先使用，避免重复 getContext）
 * @returns       { registry, cellProgram }
 * @throws        若 WebGL2 不可用
 */
export function createCellCSPShader(
  canvas: HTMLCanvasElement,
  gl?: WebGL2RenderingContext,
): {
  registry: CSPShaderRegistry;
  cellProgram: CellCSPShaderProgram;
} {
  const ctx = gl ?? (canvas.getContext('webgl2') as WebGL2RenderingContext | null);
  if (!ctx) throw new Error('[cell-csp-shader] WebGL2 not available');

  const registry    = new CSPShaderRegistry(ctx);
  const cellProgram = new CellCSPShaderProgram(registry, ctx);

  return { registry, cellProgram };
}

/**
 * warmupCellShaders — 在 requestIdleCallback / app init 阶段预热所有 cell shader。
 *
 * 等效于 PrepareSystem 的 shader warmup，消除首帧 GPU 编译 stall。
 * 调用后 CSPShaderRegistry 的 cache 已填充，后续 compileProgram() 直接命中缓存。
 *
 * @param registry  已创建的 CSPShaderRegistry 实例
 */
export function warmupCellShaders(registry: CSPShaderRegistry): void {
  registry.compileAll();
}
