/**
 * sprite-route-handler.ts — Sprite pipeline result router
 *
 * Transplanted from src/pages/generate/index.astro (commit 08394c8)
 * with algorithmic modifications (~20%):
 *   - Extracted from inline <script> into importable module
 *   - State machine pattern replaces sequential if/else
 *   - Diagnostic hooks at every stage transition
 *   - Timing instrumentation for performance profiling
 *
 * Upstream references:
 *   withastro/astro (packages/astro/src/core/app/pipeline.ts — Pipeline.create)
 *   kieler/elkjs (src/js/elk-api.js — PromisedWorker.postMessage)
 *
 * Original flow (08394c8):
 *   sprite-generate API → elk_graph → elkToSvg() → showSkeleton()
 *   → switch to Skeleton tab → InteractiveSvgEditor renders sprites
 *   (previously went to tldraw Editor tab, which was removed)
 */

// ── Types ──────────────────────────────────────────────────

interface SpriteApiResponse {
  success: boolean
  elk_graph?: Record<string, unknown>
  error?: string
  diagnostics?: {
    injection?: {
      refs_stamped?: number
      [k: string]: unknown
    }
    [k: string]: unknown
  }
}

interface PreviewController {
  showSkeleton: (svgMarkup: string, layoutedData?: Record<string, unknown>) => void
}

type ElkToSvgFn = (elkGraph: Record<string, unknown>) => string | null

/** State machine stages for the sprite routing pipeline */
type RouteStage =
  | 'idle'
  | 'fetching'
  | 'converting_svg'
  | 'loading_editor'
  | 'switching_tab'
  | 'done'
  | 'error'

interface StageTransition {
  from: RouteStage
  to: RouteStage
  ts: number
  detail?: string
}

// ── Diagnostic trace ───────────────────────────────────────

let _routeTrace: StageTransition[] = []

function recordTransition(from: RouteStage, to: RouteStage, detail?: string) {
  const entry: StageTransition = {
    from, to,
    ts: typeof performance !== 'undefined' ? performance.now() : Date.now(),
    detail,
  }
  _routeTrace.push(entry)

  // Also console.log for live debugging
  console.log(
    `[sprite-router] ${from} → ${to}` +
    (detail ? ` | ${detail}` : '') +
    ` @ ${entry.ts.toFixed(1)}ms`
  )
}

/** Get the full routing trace for post-mortem analysis */
export function getSpriteRouteTrace(): StageTransition[] {
  return [..._routeTrace]
}

/** Clear the trace buffer */
export function clearSpriteRouteTrace(): void {
  _routeTrace = []
}

/** Print the trace as a table to the console */
export function printSpriteRouteTrace(): void {
  console.group('[sprite-router] route trace')
  console.table(_routeTrace.map((t, i) => ({
    '#': i,
    from: t.from,
    to: t.to,
    elapsed: i > 0 ? `+${(t.ts - _routeTrace[i - 1].ts).toFixed(1)}ms` : '0ms',
    detail: t.detail ?? '',
  })))
  console.groupEnd()
}

// ── Main router ────────────────────────────────────────────
// Replaces the inline sprite result handling in generate/index.astro
// with a state-machine pattern for clarity and debuggability.

export interface SpriteRouteOptions {
  /** The preview controller that owns showSkeleton() */
  preview: PreviewController | null

  /** Async import for elkToSvg — keeps this module tree-shakeable */
  importElkToSvg: () => Promise<{ elkToSvg: ElkToSvgFn }>

  /** Status text element to update */
  statusElement: HTMLElement | null

  /** CSS selector for the Skeleton tab button */
  skeletonTabSelector?: string

  /** Called when routing completes successfully */
  onSuccess?: (info: { refsStamped: number; svgGenerated: boolean }) => void

  /** Called on error */
  onError?: (error: string) => void
}

/**
 * Route a sprite-generate API response into the interactive editor.
 *
 * This is the transplanted logic from 08394c8's generate/index.astro:
 *   1. Generate static SVG (for export/fallback)
 *   2. Load enriched graph into InteractiveSvgEditor via showSkeleton
 *   3. Switch to Skeleton tab
 *   4. Update status text
 *
 * The key architectural change from the pre-08394c8 code:
 *   - Previously routed to tldraw Editor tab with retry polling
 *   - Now routes to the Skeleton tab's InteractiveSvgEditor directly
 *   - No retry loop needed — InteractiveSvgEditor is always ready
 */
export async function routeSpriteResult(
  apiData: SpriteApiResponse,
  options: SpriteRouteOptions,
): Promise<boolean> {
  const {
    preview,
    importElkToSvg,
    statusElement,
    skeletonTabSelector = '#tab-skeleton',
    onSuccess,
    onError,
  } = options

  let currentStage: RouteStage = 'idle'

  const transition = (to: RouteStage, detail?: string) => {
    recordTransition(currentStage, to, detail)
    currentStage = to
  }

  // ── Stage 1: Validate API response ──
  if (!apiData.success || !apiData.elk_graph) {
    const errMsg = apiData.error || 'Unknown sprite error'
    transition('error', errMsg)
    if (statusElement) statusElement.textContent = `Sprite error: ${errMsg}`
    onError?.(errMsg)
    return false
  }

  transition('converting_svg', `elk_graph keys: ${Object.keys(apiData.elk_graph).join(',')}`)

  // ── Stage 2: Generate static SVG (for export/fallback) ──
  let staticSvg = ''
  try {
    const { elkToSvg } = await importElkToSvg()
    const rendered = elkToSvg(apiData.elk_graph)
    if (rendered) {
      staticSvg = rendered
      transition('loading_editor', `SVG generated: ${staticSvg.length} chars`)
    } else {
      transition('loading_editor', 'elkToSvg returned null — proceeding with empty SVG')
    }
  } catch (err) {
    console.warn('[sprite-router] elkToSvg failed, proceeding with empty SVG:', err)
    transition('loading_editor', `elkToSvg threw: ${err}`)
  }

  // ── Stage 3: Load enriched graph into interactive editor ──
  // showSkeleton with layoutedData triggers initInteractiveEditor,
  // which renders sprite images via InteractiveNode.spriteUrl
  if (preview) {
    try {
      preview.showSkeleton(staticSvg, apiData.elk_graph)
      transition('switching_tab', 'showSkeleton called')
    } catch (err) {
      console.error('[sprite-router] showSkeleton failed:', err)
      transition('error', `showSkeleton threw: ${err}`)
      onError?.(`Editor load failed: ${err}`)
      return false
    }
  } else {
    transition('switching_tab', 'no preview controller — skipping showSkeleton')
  }

  // ── Stage 4: Switch to Skeleton tab ──
  // This is the primary editing surface (08394c8 replaced tldraw with this)
  try {
    const tabBtn = document.querySelector(skeletonTabSelector) as HTMLElement | null
    if (tabBtn) {
      tabBtn.click()
      transition('done', 'tab switched')
    } else {
      transition('done', `tab button not found: ${skeletonTabSelector}`)
    }
  } catch (err) {
    // Tab switching is non-critical — editor still works
    transition('done', `tab switch failed: ${err}`)
  }

  // ── Stage 5: Update status ──
  const diag = apiData.diagnostics || {}
  const injDiag = diag.injection || diag
  const refsStamped = (injDiag as Record<string, unknown>).refs_stamped as number || 0

  if (statusElement) {
    statusElement.textContent = `Sprites done: ${refsStamped} injected. Drag nodes to rearrange.`
  }

  onSuccess?.({ refsStamped, svgGenerated: staticSvg.length > 0 })

  // ── Debug: dump final state with full graph inspection ──
  const elkChildren = (apiData.elk_graph as Record<string, unknown[]>).children ?? []
  const graphStats = _inspectElkGraph(apiData.elk_graph)

  console.group('[sprite-router] ✓ ROUTING COMPLETE')
  console.log('refs stamped:', refsStamped)
  console.log('SVG generated:', staticSvg.length > 0, `(${staticSvg.length} chars)`)
  console.log('ELK graph stats:', graphStats)
  console.log('trace entries:', _routeTrace.length)
  console.log('total time:', _routeTrace.length > 1
    ? `${(_routeTrace[_routeTrace.length-1].ts - _routeTrace[0].ts).toFixed(0)}ms`
    : 'N/A')
  console.groupEnd()

  return true
}

// ── Graph inspection helper ───────────────────────────────
// Walks the ELK graph and prints sprite coverage statistics

interface ElkGraphStats {
  totalNodes: number
  spriteNodes: number
  operatorNodes: number
  groupNodes: number
  leafNodes: number
  familyCount: number
  spriteCoverage: string
  families: Record<string, number>
  missingSprites: string[]
}

function _inspectElkGraph(elkGraph: Record<string, unknown>): ElkGraphStats {
  const stats: ElkGraphStats = {
    totalNodes: 0, spriteNodes: 0, operatorNodes: 0,
    groupNodes: 0, leafNodes: 0, familyCount: 0,
    spriteCoverage: '0%', families: {}, missingSprites: [],
  }

  const stack: Record<string, unknown>[] = []
  const children = elkGraph.children as Record<string, unknown>[] | undefined
  if (Array.isArray(children)) stack.push(...children)

  while (stack.length > 0) {
    const node = stack.pop()!
    stats.totalNodes++

    const kids = node.children as Record<string, unknown>[] | undefined
    if (Array.isArray(kids) && kids.length > 0) {
      stats.groupNodes++
      stack.push(...kids)
    } else {
      stats.leafNodes++
    }

    if (node.isOperator) stats.operatorNodes++

    const spriteRef = node.spriteRef as Record<string, unknown> | undefined
    const renderMode = node.renderMode as string | undefined

    if (renderMode === 'sprite') {
      if (spriteRef?.url) {
        stats.spriteNodes++
      } else {
        stats.missingSprites.push(node.id as string ?? '?')
      }
    }

    const famId = node.familyId as string | undefined
    if (famId) {
      stats.families[famId] = (stats.families[famId] ?? 0) + 1
    }
  }

  stats.familyCount = Object.keys(stats.families).length
  const spriteEligible = stats.leafNodes - stats.operatorNodes - stats.groupNodes
  stats.spriteCoverage = spriteEligible > 0
    ? `${Math.round(100 * stats.spriteNodes / Math.max(spriteEligible, 1))}%`
    : 'N/A'

  return stats
}

/** Print comprehensive routing report — batch progress + ELK graph analysis */
export function printFullRouteReport(elkGraph?: Record<string, unknown>): void {
  console.log(`\n${'━'.repeat(50)}`)
  console.log('  SPRITE ROUTE FULL REPORT')
  console.log(`${'━'.repeat(50)}`)

  printSpriteRouteTrace()

  if (elkGraph) {
    const stats = _inspectElkGraph(elkGraph)
    console.group('[sprite-router] ELK graph analysis')
    console.log(`  Total nodes:       ${stats.totalNodes}`)
    console.log(`  Leaf nodes:        ${stats.leafNodes}`)
    console.log(`  Group nodes:       ${stats.groupNodes}`)
    console.log(`  Operator nodes:    ${stats.operatorNodes}`)
    console.log(`  Sprite nodes:      ${stats.spriteNodes}`)
    console.log(`  Sprite coverage:   ${stats.spriteCoverage}`)
    console.log(`  Families:          ${stats.familyCount}`)
    if (Object.keys(stats.families).length > 0) {
      console.log(`  Family breakdown:`)
      for (const [fid, count] of Object.entries(stats.families)) {
        console.log(`    ${fid}: ${count} members`)
      }
    }
    if (stats.missingSprites.length > 0) {
      console.warn(`  ⚠ Missing sprites:`, stats.missingSprites)
    }
    console.groupEnd()
  }

  console.log(`${'━'.repeat(50)}\n`)
}

// ── Legacy compatibility shim ──────────────────────────────
// Drop-in for code that still expects the old inline pattern:
//   const data = await res.json()
//   if (data.success && data.elk_graph) { ... }
//
// Usage:
//   import { handleSpriteResponse } from './sprite-route-handler'
//   const success = await handleSpriteResponse(res, preview, statusEl)

export async function handleSpriteResponse(
  response: Response,
  preview: PreviewController | null,
  statusElement: HTMLElement | null,
): Promise<boolean> {
  let data: SpriteApiResponse
  try {
    data = await response.json()
  } catch (err) {
    console.error('[sprite-router] JSON parse failed:', err)
    if (statusElement) statusElement.textContent = 'Sprite error: invalid response'
    return false
  }

  return routeSpriteResult(data, {
    preview,
    importElkToSvg: () => import('../to-svg'),
    statusElement,
  })
}

export default {
  routeSpriteResult,
  handleSpriteResponse,
  getSpriteRouteTrace,
  clearSpriteRouteTrace,
  printSpriteRouteTrace,
}
