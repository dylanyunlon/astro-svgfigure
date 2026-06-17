/**
 * CloudFogBackground.ts — M049: CloudFog 全屏背景层
 *
 * 任务规格:
 *   - 使用 src/lib/shaders/cloud-fog.frag / .vert + CloudFog.ts
 *   - 创建全屏 Mesh 背景 Canvas，zIndex = -1 最底层
 *   - Ticker (requestAnimationFrame) 驱动 u_time 云雾流动
 *   - 颜色从 composite_params.json palette 读取 (zenith / horizon / nadir)
 *
 * 架构:
 *   - 在宿主元素下插入一个独立 <canvas> (position: absolute, z-index: -1)
 *   - 创建 WebGL2RenderingContext，实例化 CloudFog（来自 CloudFog.ts）
 *   - 使用内置 requestAnimationFrame Ticker 更新 u_time（elapsed seconds）
 *   - 正交投影矩阵 + 固定 identity view，将全屏 quad 覆盖整个 canvas
 *   - palette.zenith 作为 fogColor，horizon/nadir 参与背景底色
 *
 * 用法:
 *   import { mountCloudFogBackground } from '@/lib/CloudFogBackground'
 *   const stop = mountCloudFogBackground(containerEl)
 *   // 清理:
 *   stop()
 *
 * Author: claude <claude@astro.dev>
 */

import { CloudFog, type CloudFogOptions } from './CloudFog'

// ── 从 composite_params.json palette 解析 fogColor ──────────────────────────

interface Palette {
  zenith:  string  // e.g. "#D9CAC1"
  horizon: string  // e.g. "#A09D9D"
  nadir:   string  // e.g. "#615F5E"
}

/** 将 CSS hex 色 → [r,g,b] 0-1 浮点 */
function hexToRgb01(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16)
  return [
    ((n >> 16) & 0xff) / 255,
    ((n >>  8) & 0xff) / 255,
    (       n  & 0xff) / 255,
  ]
}

/** 将 CSS hex → 24-bit number (for CSS background-color) */
function hexToNum(hex: string): number {
  return parseInt(hex.replace('#', ''), 16)
}

// ── 4×4 正交投影 (column-major, WebGL convention) ─────────────────────────────

/**
 * 正交投影矩阵：将 [left,right] × [bottom,top] × [near,far] 映射到 NDC [-1,1]³
 */
function ortho(
  left: number, right: number,
  bottom: number, top: number,
  near: number, far: number,
): Float32Array {
  const m = new Float32Array(16)
  m[0]  =  2 / (right - left)
  m[5]  =  2 / (top   - bottom)
  m[10] = -2 / (far   - near)
  m[12] = -(right + left)   / (right - left)
  m[13] = -(top   + bottom) / (top   - bottom)
  m[14] = -(far   + near)   / (far   - near)
  m[15] =  1
  return m
}

/** 4×4 identity（column-major） */
function mat4Id(): Float32Array {
  const m = new Float32Array(16)
  m[0] = m[5] = m[10] = m[15] = 1
  return m
}

// ── CloudFogBackground 公共 API ───────────────────────────────────────────────

export interface CloudFogBackgroundOptions {
  /**
   * 覆盖 composite_params.json palette.zenith 的 fog 颜色。
   * 传入时忽略 palette 自动读取。格式: "#RRGGBB"
   */
  fogColorHex?: string

  /** 透传给 CloudFog 的额外参数（alpha, planes, speed, …） */
  fogOptions?: Omit<CloudFogOptions, 'fogColor'>

  /**
   * palette JSON 路径（相对于网站根）。默认 '/channels/composite_params.json'。
   * 设为 null 可跳过远端 fetch，完全使用 fogColorHex 或内置默认值。
   */
  paletteUrl?: string | null
}

/**
 * mountCloudFogBackground
 *
 * 在 `container` 内插入全屏 WebGL2 fog canvas（zIndex=-1），
 * 用 requestAnimationFrame Ticker 驱动 CloudFog.render() 的 u_time。
 *
 * @param container  宿主元素（需有 position: relative 或 absolute）
 * @param options    可选 palette/fog 参数覆盖
 * @returns          stop() — 销毁所有 GL 资源并移除 canvas
 */
export async function mountCloudFogBackground(
  container: HTMLElement,
  options: CloudFogBackgroundOptions = {},
): Promise<() => void> {

  // ── 1. 尝试加载 composite_params.json 获取 palette ──────────────────────
  let palette: Palette = { zenith: '#D9CAC1', horizon: '#A09D9D', nadir: '#615F5E' }

  const paletteUrl = options.paletteUrl === undefined
    ? '/channels/composite_params.json'
    : options.paletteUrl

  if (paletteUrl) {
    try {
      const res  = await fetch(paletteUrl)
      if (res.ok) {
        const json = await res.json()
        if (json?.palette) {
          palette = { ...palette, ...json.palette }
        }
      }
    } catch {
      // offline / not found — use defaults
    }
  }

  // fogColor: zenith from palette (lightest tone → most visible as fog tint)
  const fogColorHex = options.fogColorHex ?? palette.zenith
  const fogColor    = hexToRgb01(fogColorHex)

  // Background gradient uses horizon → nadir (darker tones)
  const bgHex = palette.nadir   // darkest as canvas background

  // ── 2. 创建全屏 <canvas>（zIndex = -1） ──────────────────────────────────
  const canvas = document.createElement('canvas')
  canvas.id    = 'cloud-fog-bg'
  canvas.style.cssText = [
    'position: absolute',
    'inset: 0',
    'width: 100%',
    'height: 100%',
    `background-color: #${hexToNum(bgHex).toString(16).padStart(6, '0')}`,
    'z-index: -1',
    'pointer-events: none',
    'display: block',
  ].join('; ')

  // container 需要 position:relative/absolute 才能正确定位子元素
  const existingPosition = getComputedStyle(container).position
  if (existingPosition === 'static') {
    container.style.position = 'relative'
  }

  // 插入为第一个子元素，其他内容在其上方
  container.insertBefore(canvas, container.firstChild)

  // ── 3. 初始化 WebGL2 ──────────────────────────────────────────────────────
  const gl = canvas.getContext('webgl2', {
    alpha:              true,
    antialias:          false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
  }) as WebGL2RenderingContext | null

  if (!gl) {
    console.warn('[CloudFogBackground] WebGL2 not available — fog disabled')
    return () => { canvas.remove() }
  }

  // ── 4. 实例化 CloudFog ────────────────────────────────────────────────────
  //  AT CloudFog 参数 + palette 颜色
  const fogOpts: CloudFogOptions = {
    alpha:    1.8,
    planes:   20,
    noise:    1.0,
    speed:    0.7,
    scale:    6.0,
    fadeDist: [2, 4],
    ...(options.fogOptions ?? {}),
    fogColor,
  }

  let fog: CloudFog | null = null
  try {
    fog = new CloudFog(gl, fogOpts)
  } catch (err) {
    console.error('[CloudFogBackground] CloudFog init failed:', err)
    canvas.remove()
    return () => {}
  }

  // ── 5. 处理 Canvas 尺寸 ───────────────────────────────────────────────────
  let W = 0, H = 0

  function resize() {
    const rect = container.getBoundingClientRect()
    const dpr  = window.devicePixelRatio || 1
    const newW = Math.max(1, Math.floor(rect.width  * dpr))
    const newH = Math.max(1, Math.floor(rect.height * dpr))
    if (newW === W && newH === H) return
    W = newW; H = newH
    canvas.width  = W
    canvas.height = H
    gl.viewport(0, 0, W, H)
  }

  resize()

  const ro = new ResizeObserver(resize)
  ro.observe(container)

  // ── 6. 构建投影矩阵 (正交，覆盖 CloudFog 的 world volume) ────────────────
  //  CloudFog 默认 width=[-4,4], height=[-1,4], depth=[-2,-6]
  //  用简单正交投影覆盖这个范围；view = identity（camera at origin, 朝 -Z）
  const VIEW = mat4Id()

  function buildProjection(): Float32Array {
    // aspect-correct ortho — 使 fog 水平 fill
    const aspect = W / Math.max(H, 1)
    const halfW  = 4.0            // matches CloudFog default width half-extent
    const halfH  = halfW / aspect
    return ortho(-halfW, halfW, -halfH, halfH, -10, 10)
  }

  // ── 7. Ticker — requestAnimationFrame 驱动 u_time ────────────────────────
  let rafId    = -1
  let startMs  = -1
  let stopped  = false

  function frame(nowMs: number) {
    if (stopped) return
    if (startMs < 0) startMs = nowMs
    const elapsedSec = (nowMs - startMs) * 0.001   // seconds → u_time

    // 清除背景（透明，让 CSS background-color 显现）
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)

    // 更新 CloudFog 的 fogColor（动态色，可选）
    // 此处直接用 palette zenith — 如需呼吸效果可在此 lerp

    const proj = buildProjection()
    fog!.render(proj, VIEW, elapsedSec)

    rafId = requestAnimationFrame(frame)
  }

  rafId = requestAnimationFrame(frame)

  // ── 8. 返回 stop() ────────────────────────────────────────────────────────
  return function stop() {
    stopped = true
    if (rafId >= 0) { cancelAnimationFrame(rafId); rafId = -1 }
    ro.disconnect()
    try { fog?.destroy() } catch { /* ok */ }
    canvas.remove()
  }
}
