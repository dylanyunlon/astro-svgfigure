/**
 * CloudFogBackground.ts — M200: CloudFog 全屏背景层
 *
 * 职责:
 *   - 读 channels/physics/fog_at_params.json (fogDensity → alpha, fogSpeed → speed)
 *   - update(dt) 推进 u_time 驱动云雾流动
 *   - setSpeciesInfluence(cells) 加权平均 species glow 色设 u_fogColor
 *   - 全屏 WebGL2 <canvas> (zIndex=-1) 渲染 CloudFog
 *
 * 用法:
 *   import { mountCloudFogBackground } from '@/lib/CloudFogBackground'
 *   const bg = await mountCloudFogBackground(containerEl)
 *   // 每帧:
 *   bg.update(dt)
 *   bg.setSpeciesInfluence(visibleCells)
 *   // 清理:
 *   bg.stop()
 *
 * Author: claude <claude@astro.dev>
 */

import { CloudFog, type CloudFogOptions } from './CloudFog'
import { getSpeciesPalette } from './color-utils'

// ── fog_at_params.json 结构 ──────────────────────────────────────────────────

interface FogAtParams {
  params: {
    alpha:    number
    planes:   number
    noise:    number
    speed:    number
    scale:    number
    width:    [number, number]
    height:   [number, number]
    fadeDist: [number, number]
  }
  uniform_mapping?: Record<string, number>
}

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

// ── Cell 接口 (用于 setSpeciesInfluence) ──────────────────────────────────────

/** 传入 setSpeciesInfluence 的 cell 最小契约 */
export interface FogCell {
  species: string
  /** 可选面积/权重；省略时每个 cell 权重相等 */
  w?: number
  h?: number
}

// ── CloudFogBackground 公共 API ───────────────────────────────────────────────

export interface CloudFogBackgroundOptions {
  /**
   * 覆盖 fog_at_params.json / composite_params.json palette.zenith 的 fog 颜色。
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

  /**
   * fog_at_params JSON 路径（相对于网站根）。
   * 默认 '/channels/physics/fog_at_params.json'。
   * 设为 null 跳过加载，使用内置默认值。
   */
  fogParamsUrl?: string | null
}

/** mountCloudFogBackground 返回的控制句柄 */
export interface CloudFogBackgroundHandle {
  /**
   * 推进 u_time。dt 为秒级 delta。
   * 在外部 ticker / rAF 内调用。
   */
  update(dt: number): void

  /**
   * 根据可见 cells 的 species glow 色加权平均，设置 u_fogColor。
   * 权重 = cell 面积 (w × h)，缺省为等权。
   * 空数组时保留当前颜色不变。
   */
  setSpeciesInfluence(cells: ReadonlyArray<FogCell>): void

  /** 销毁所有 GL 资源并移除 canvas */
  stop(): void
}

// ── 加载 fog_at_params.json ──────────────────────────────────────────────────

interface FogAtDefaults {
  fogDensity: number  // → CloudFog alpha
  fogSpeed:   number  // → CloudFog speed
  planes:     number
  noise:      number
  scale:      number
  fadeDist:   [number, number]
}

const BUILTIN_DEFAULTS: FogAtDefaults = {
  fogDensity: 1.8,
  fogSpeed:   0.7,
  planes:     20,
  noise:      1.0,
  scale:      6.0,
  fadeDist:   [2, 4],
}

async function loadFogAtParams(url: string | null): Promise<FogAtDefaults> {
  if (!url) return { ...BUILTIN_DEFAULTS }
  try {
    const res = await fetch(url)
    if (!res.ok) return { ...BUILTIN_DEFAULTS }
    const json: FogAtParams = await res.json()
    const p = json.params
    return {
      fogDensity: p.alpha    ?? BUILTIN_DEFAULTS.fogDensity,
      fogSpeed:   p.speed    ?? BUILTIN_DEFAULTS.fogSpeed,
      planes:     p.planes   ?? BUILTIN_DEFAULTS.planes,
      noise:      p.noise    ?? BUILTIN_DEFAULTS.noise,
      scale:      p.scale    ?? BUILTIN_DEFAULTS.scale,
      fadeDist:   p.fadeDist ?? [...BUILTIN_DEFAULTS.fadeDist],
    }
  } catch {
    return { ...BUILTIN_DEFAULTS }
  }
}

// ── mountCloudFogBackground ──────────────────────────────────────────────────

/**
 * mountCloudFogBackground
 *
 * 在 `container` 内插入全屏 WebGL2 fog canvas（zIndex=-1），
 * 返回 handle 对象用于外部 ticker 驱动 update(dt) 和 species 色设置。
 *
 * @param container  宿主元素（需有 position: relative 或 absolute）
 * @param options    可选 palette/fog 参数覆盖
 * @returns          CloudFogBackgroundHandle
 */
export async function mountCloudFogBackground(
  container: HTMLElement,
  options: CloudFogBackgroundOptions = {},
): Promise<CloudFogBackgroundHandle> {

  // ── 1. 并行加载 fog_at_params.json + composite_params.json ─────────────

  const fogParamsUrl = options.fogParamsUrl === undefined
    ? '/channels/physics/fog_at_params.json'
    : options.fogParamsUrl

  const paletteUrl = options.paletteUrl === undefined
    ? '/channels/composite_params.json'
    : options.paletteUrl

  const [fogAtDefaults, palette] = await Promise.all([
    loadFogAtParams(fogParamsUrl),
    loadPalette(paletteUrl),
  ])

  // fogColor: zenith from palette (lightest tone → most visible as fog tint)
  const fogColorHex = options.fogColorHex ?? palette.zenith
  const fogColor: [number, number, number] = hexToRgb01(fogColorHex)

  // Background gradient uses nadir (darkest) as canvas background
  const bgHex = palette.nadir

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

  const existingPosition = getComputedStyle(container).position
  if (existingPosition === 'static') {
    container.style.position = 'relative'
  }

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
    canvas.remove()
    return createNoopHandle()
  }

  // ── 4. 实例化 CloudFog（使用 fog_at_params 参数） ─────────────────────────
  const fogOpts: CloudFogOptions = {
    alpha:    fogAtDefaults.fogDensity,
    planes:   fogAtDefaults.planes,
    noise:    fogAtDefaults.noise,
    speed:    fogAtDefaults.fogSpeed,
    scale:    fogAtDefaults.scale,
    fadeDist: fogAtDefaults.fadeDist,
    ...(options.fogOptions ?? {}),
    fogColor,
  }

  let fog: CloudFog | null = null
  try {
    fog = new CloudFog(gl, fogOpts)
  } catch (err) {
    console.error('[CloudFogBackground] CloudFog init failed:', err)
    canvas.remove()
    return createNoopHandle()
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
    gl!.viewport(0, 0, W, H)
  }

  resize()

  const ro = new ResizeObserver(resize)
  ro.observe(container)

  // ── 6. 投影矩阵 ──────────────────────────────────────────────────────────
  const VIEW = mat4Id()

  function buildProjection(): Float32Array {
    const aspect = W / Math.max(H, 1)
    const halfW  = 4.0
    const halfH  = halfW / aspect
    return ortho(-halfW, halfW, -halfH, halfH, -10, 10)
  }

  // ── 7. 状态: u_time 由 update(dt) 累积 ──────────────────────────────────
  let uTime   = 0
  let stopped = false
  let rafId   = -1

  // 当前 fogColor（可被 setSpeciesInfluence 覆盖）
  let currentFogColor: [number, number, number] = [...fogColor]

  function renderFrame() {
    if (stopped) return

    gl!.clearColor(0, 0, 0, 0)
    gl!.clear(gl!.COLOR_BUFFER_BIT)

    const proj = buildProjection()
    fog!.render(proj, VIEW, uTime)

    rafId = requestAnimationFrame(renderFrame)
  }

  rafId = requestAnimationFrame(renderFrame)

  // ── 8. 返回 Handle ────────────────────────────────────────────────────────
  return {
    update(dt: number): void {
      if (stopped) return
      uTime += dt
    },

    setSpeciesInfluence(cells: ReadonlyArray<FogCell>): void {
      if (stopped || !fog || cells.length === 0) return

      // 加权平均: 权重 = cell 面积 (w × h), 缺省 1×1
      let totalWeight = 0
      let rAcc = 0, gAcc = 0, bAcc = 0

      for (const cell of cells) {
        const weight  = (cell.w ?? 1) * (cell.h ?? 1)
        const palette = getSpeciesPalette(cell.species)
        const glow    = palette.glow

        rAcc += glow.r * weight
        gAcc += glow.g * weight
        bAcc += glow.b * weight
        totalWeight += weight
      }

      if (totalWeight > 0) {
        currentFogColor[0] = rAcc / totalWeight
        currentFogColor[1] = gAcc / totalWeight
        currentFogColor[2] = bAcc / totalWeight
        fog.setOptions({ fogColor: currentFogColor })
      }
    },

    stop(): void {
      stopped = true
      if (rafId >= 0) { cancelAnimationFrame(rafId); rafId = -1 }
      ro.disconnect()
      try { fog?.destroy() } catch { /* ok */ }
      canvas.remove()
    },
  }
}

// ── 内部辅助 ────────────────────────────────────────────────────────────────

async function loadPalette(url: string | null): Promise<Palette> {
  const fallback: Palette = { zenith: '#D9CAC1', horizon: '#A09D9D', nadir: '#615F5E' }
  if (!url) return fallback
  try {
    const res = await fetch(url)
    if (!res.ok) return fallback
    const json = await res.json()
    if (json?.palette) return { ...fallback, ...json.palette }
    return fallback
  } catch {
    return fallback
  }
}

function createNoopHandle(): CloudFogBackgroundHandle {
  return {
    update()               { /* noop */ },
    setSpeciesInfluence()  { /* noop */ },
    stop()                 { /* noop */ },
  }
}
