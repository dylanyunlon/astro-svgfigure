/**
 * ElkCanvas — React component wrapping tldraw for ELK graph editing (M305).
 *
 * Reference: tldraw/apps/examples/.../CustomShapeMermaids.tsx
 *   <Tldraw components={components} shapeUtils={customShapes} />
 *   then on "Apply", calls createMermaidDiagram(editor, text, opts)
 *
 * Diff from reference:
 *   - Input: ELK layouted JSON prop instead of mermaid text
 *   - On mount/update: elkToTldraw(editor, graph) instead of createMermaidDiagram
 *   - onExport callback: tldrawToElk(editor) returns updated ELK JSON
 *   - No mermaid parsing, no CSS animations, no pipeline state atom
 */
import { useCallback, useEffect, useRef } from 'react'
import { Tldraw } from 'tldraw'
import type { Editor, TLComponents } from 'tldraw'
import 'tldraw/tldraw.css'
import { ElkNodeShapeUtil } from './ElkNodeShapeUtil'
import { elkToTldraw, tldrawToElk } from './elkBridge'

// Register custom shape utils — same pattern as FlowchartShapeUtil registration
const customShapeUtils = [ElkNodeShapeUtil]

// Minimal UI components — hide tldraw's default shape tools since users
// only interact with ELK nodes (not freehand drawing)
const components: TLComponents = {}

interface ElkCanvasProps {
  /** ELK layouted JSON graph. When this changes, canvas re-imports. */
  layoutedJson: any
  /** Called when user clicks "Confirm Layout" — returns updated ELK JSON. */
  onExport?: (elkGraph: any) => void
  /** Called when the tldraw editor instance is ready. */
  onEditorReady?: (editor: Editor) => void
  /** CSS class for the container div. */
  className?: string
  /** Read-only mode (no dragging/editing). */
  readOnly?: boolean
}

/**
 * Wrap tldraw as an ELK graph editor.
 *
 * Usage:
 *   <ElkCanvas
 *     layoutedJson={layoutData}
 *     onExport={(graph) => setElkGraph(graph)}
 *     onEditorReady={(editor) => editorRef.current = editor}
 *   />
 */
export function ElkCanvas({
  layoutedJson,
  onExport,
  onEditorReady,
  className,
  readOnly,
}: ElkCanvasProps) {
  const editorRef = useRef<Editor | null>(null)
  const graphRef = useRef<any>(null)

  // Handle editor mount
  const handleMount = useCallback((editor: Editor) => {
    editorRef.current = editor
    onEditorReady?.(editor)

    // Import initial graph if available
    if (layoutedJson) {
      graphRef.current = layoutedJson
      elkToTldraw(editor, layoutedJson, { clearFirst: true })
      // Zoom to fit the imported graph
      editor.zoomToFit()
    }
  }, [layoutedJson, onEditorReady])

  // Re-import when layoutedJson changes
  useEffect(() => {
    const editor = editorRef.current
    if (!editor || !layoutedJson) return

    // Skip if same reference (avoid re-import on parent re-render)
    if (layoutedJson === graphRef.current) return
    graphRef.current = layoutedJson

    elkToTldraw(editor, layoutedJson, { clearFirst: true })
    editor.zoomToFit()
  }, [layoutedJson])

  // Export function for parent to call
  const handleExport = useCallback(() => {
    const editor = editorRef.current
    if (!editor) return
    const elkGraph = tldrawToElk(editor)
    onExport?.(elkGraph)
  }, [onExport])

  return (
    <div
      className={className}
      style={{ width: '100%', height: '100%', position: 'relative' }}
    >
      <Tldraw
        shapeUtils={customShapeUtils}
        components={components}
        onMount={handleMount}
        options={readOnly ? { maxPages: 1 } : undefined}
      />

      {/* Export button overlay */}
      {onExport && (
        <button
          type="button"
          onClick={handleExport}
          style={{
            position: 'absolute', bottom: 16, right: 16,
            padding: '8px 16px', borderRadius: 8,
            backgroundColor: '#4A4A4A', color: '#FFFFFF',
            border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: 600,
            fontFamily: 'system-ui, -apple-system, sans-serif',
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
            zIndex: 100,
          }}
        >
          Confirm Layout
        </button>
      )}
    </div>
  )
}

export { elkToTldraw, tldrawToElk } from './elkBridge'
export { ElkNodeShapeUtil, ELK_NODE_TYPE } from './ElkNodeShapeUtil'