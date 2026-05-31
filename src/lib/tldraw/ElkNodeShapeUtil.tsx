/**
 * ElkNodeShapeUtil — custom tldraw shape for ELK graph nodes (M300).
 *
 * Reference: tldraw/apps/examples/.../customMermaidShapeUtil.tsx
 *   FlowchartShapeUtil extends BaseBoxShapeUtil, carries mermaidNodeId,
 *   renders with HTMLContainer + RichTextLabel, getIndicatorPath returns
 *   a rect Path2D.  We follow that pattern exactly.
 *
 * Diff from reference:
 *   - Props: mermaidNodeId → elkId, renderMode, familyId, spriteUrl, iconHint, label
 *   - Component: RichTextLabel → three-mode renderer (blob / sprite / text)
 *   - No retry button (sprite retry lives in SpritePanelOverlay, M307)
 */
import {
  BaseBoxShapeUtil,
  HTMLContainer,
  T,
} from 'tldraw'
import type {
  RecordProps,
  TLShape,
} from 'tldraw'

// ═══════════════════════════════════════════════════════════════════════════
//  §1  Shape type declaration
// ═══════════════════════════════════════════════════════════════════════════

export const ELK_NODE_TYPE = 'elk-node' as const

declare module 'tldraw' {
  export interface TLGlobalShapePropsMap {
    [ELK_NODE_TYPE]: {
      w: number
      h: number
      elkId: string
      label: string
      renderMode: string     // 'text' | 'icon' | 'sprite'
      familyId: string
      spriteUrl: string      // data:image/png;base64,... or empty
      spriteFormat: string   // 'png' | 'svg' | 'stack' | ''
      iconHint: string
      isOperator: boolean
      fillColor: string      // skeleton group background color
      depth: number          // nesting depth (0 = top-level group)
      isGroup: boolean
    }
  }
}

export type ElkNodeShape = TLShape<typeof ELK_NODE_TYPE>

// ═══════════════════════════════════════════════════════════════════════════
//  §2  Color palettes — match to-svg.ts SKELETON_FILLS and FAMILY_PALETTES
// ═══════════════════════════════════════════════════════════════════════════

const SKELETON_FILLS = [
  { bg: '#EBF5EB', stroke: '#A5D6A7' },
  { bg: '#FFF3E0', stroke: '#FFCC80' },
  { bg: '#FFF8E1', stroke: '#FFE082' },
  { bg: '#E3F2FD', stroke: '#90CAF9' },
  { bg: '#F3E5F5', stroke: '#CE93D8' },
]

const FAMILY_PALETTES: string[][] = [
  ['#B3D4FC', '#9EC5F0', '#89B6E4', '#74A7D8'],
  ['#F8BBD0', '#F48FB1', '#F06292', '#EC407A'],
  ['#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A'],
  ['#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726'],
  ['#D1C4E9', '#B39DDB', '#9575CD', '#7E57C2'],
]

function familyColor(familyId: string, index: number): string {
  if (!familyId) return FAMILY_PALETTES[0][0]
  let h = 0
  for (let i = 0; i < familyId.length; i++) h = Math.imul(h ^ familyId.charCodeAt(i), 16777619)
  const palette = FAMILY_PALETTES[((h >>> 0) % FAMILY_PALETTES.length)]
  return palette[index % palette.length]
}

// ═══════════════════════════════════════════════════════════════════════════
//  §3  ShapeUtil — follows FlowchartShapeUtil pattern exactly
// ═══════════════════════════════════════════════════════════════════════════

export class ElkNodeShapeUtil extends BaseBoxShapeUtil<ElkNodeShape> {
  static override type = ELK_NODE_TYPE
  static override props: RecordProps<ElkNodeShape> = {
    w: T.number,
    h: T.number,
    elkId: T.string,
    label: T.string,
    renderMode: T.string,
    familyId: T.string,
    spriteUrl: T.string,
    spriteFormat: T.string,
    iconHint: T.string,
    isOperator: T.boolean,
    fillColor: T.string,
    depth: T.number,
    isGroup: T.boolean,
  }

  override getDefaultProps() {
    return {
      w: 160,
      h: 50,
      elkId: '',
      label: '',
      renderMode: 'text',
      familyId: '',
      spriteUrl: '',
      spriteFormat: '',
      iconHint: '',
      isOperator: false,
      fillColor: '',
      depth: 0,
      isGroup: false,
    }
  }

  override canEdit() { return false }
  override canResize() { return true }

  override component(shape: ElkNodeShape) {
    return <ElkNodeComponent shape={shape} />
  }

  override getIndicatorPath(shape: ElkNodeShape) {
    const path = new Path2D()
    path.roundRect(0, 0, shape.props.w, shape.props.h, 8)
    return path
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  §4  Component — three-mode renderer (matches to-svg.ts Path A-E)
// ═══════════════════════════════════════════════════════════════════════════

function ElkNodeComponent({ shape }: { shape: ElkNodeShape }) {
  const { props } = shape
  const { w, h, label, renderMode, isGroup, isOperator, depth,
          spriteUrl, spriteFormat, familyId, fillColor } = props

  // Path A: Skeleton group — colored background panel
  if (isGroup) {
    const palette = SKELETON_FILLS[depth % SKELETON_FILLS.length]
    return (
      <HTMLContainer style={{ width: w, height: h }}>
        <div style={{
          width: '100%', height: '100%',
          borderRadius: 12,
          backgroundColor: depth <= 1 ? palette.bg : 'transparent',
          border: depth <= 1 ? `1.5px solid ${palette.stroke}` : 'none',
          position: 'relative',
        }}>
          <span style={{
            position: 'absolute', top: 6, left: 10,
            fontSize: 11, fontWeight: 700, fontStyle: 'italic',
            color: depth <= 1 ? palette.stroke : '#888',
            fontFamily: 'system-ui, -apple-system, sans-serif',
          }}>
            {label}
          </span>
        </div>
      </HTMLContainer>
    )
  }

  // Path B: Operator — circle + glyph
  if (isOperator) {
    const r = Math.min(w, h) * 0.4
    return (
      <HTMLContainer style={{ width: w, height: h }}>
        <div style={{
          width: '100%', height: '100%',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            width: r * 2, height: r * 2, borderRadius: '50%',
            border: '1.5px solid #4A4A4A', backgroundColor: '#FFFFFF',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: r * 1.2, fontWeight: 300, color: '#1A1A1A',
          }}>
            {label}
          </div>
        </div>
      </HTMLContainer>
    )
  }

  // Path C: Sprite — blob IS the body, small label below
  if (renderMode === 'sprite') {
    const color = fillColor || familyColor(familyId, 0)
    const hasRealSprite = spriteUrl && spriteFormat !== 'stack'

    return (
      <HTMLContainer style={{ width: w, height: h }}>
        <div style={{
          width: '100%', height: '100%',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
        }}>
          {hasRealSprite ? (
            <img
              src={spriteUrl}
              alt={label}
              style={{
                width: w * 0.85, height: Math.max((h - 20) * 0.9, 40),
                objectFit: 'contain',
                borderRadius: 4,
              }}
              draggable={false}
            />
          ) : (
            // Organic blob placeholder — colored rounded rect
            <div style={{
              width: w * 0.7, height: Math.max((h - 16) * 0.7, 30),
              borderRadius: '40% 60% 55% 45% / 50% 40% 60% 50%',
              backgroundColor: color,
              opacity: 0.7,
            }} />
          )}
          <span style={{
            fontSize: Math.max(7, Math.min(10, w / Math.max(label.length, 1) * 1.1)),
            color: '#1A1A1A', fontStyle: 'italic',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            marginTop: 2, textAlign: 'center',
            maxWidth: w, overflow: 'hidden', textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            {label}
          </span>
        </div>
      </HTMLContainer>
    )
  }

  // Path D: Label-only (h <= 30)
  if (h <= 30) {
    return (
      <HTMLContainer style={{ width: w, height: h }}>
        <div style={{
          width: '100%', height: '100%',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 11, fontWeight: 600, color: '#1A1A1A',
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}>
          {label}
        </div>
      </HTMLContainer>
    )
  }

  // Path E: Regular leaf node — white box + centered label
  return (
    <HTMLContainer style={{ width: w, height: h }}>
      <div style={{
        width: '100%', height: '100%',
        borderRadius: 8,
        backgroundColor: '#FFFFFF',
        border: '0.8px solid #4A4A4A',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 12, fontWeight: 500, color: '#1A1A1A',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        textAlign: 'center', padding: '4px 8px',
        overflow: 'hidden',
      }}>
        {label}
      </div>
    </HTMLContainer>
  )
}