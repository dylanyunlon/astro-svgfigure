/**
 * TldrawEditor.tsx — React island wrapping ElkCanvas for Astro.
 *
 * Mounted via `<TldrawEditor client:load />` in SvgPreview.astro.
 * Bridges the Astro page's vanilla JS (window.__tldrawEditor) with
 * the React tldraw component.
 *
 * API exposed on window.__tldrawEditor:
 *   .loadGraph(elkJson)   — import ELK layouted JSON into tldraw canvas
 *   .exportGraph()        — returns updated ELK JSON after user edits
 *   .toStaticSvg()        — renders current state via to-svg.ts
 *   .getEditor()          — returns raw tldraw Editor instance
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { ElkCanvas } from '@/lib/tldraw'
import { tldrawToElk } from '@/lib/tldraw/elkBridge'
import type { Editor } from 'tldraw'

export default function TldrawEditor() {
  const [graph, setGraph] = useState<any>(null)
  const editorRef = useRef<Editor | null>(null)

  const handleEditorReady = useCallback((editor: Editor) => {
    editorRef.current = editor
  }, [])

  const handleExport = useCallback((elkGraph: any) => {
    // Dispatch custom event so Astro page JS can pick it up
    window.dispatchEvent(new CustomEvent('tldraw:export', { detail: elkGraph }))
  }, [])

  // Expose API on window for Astro page scripts
  useEffect(() => {
    const api = {
      loadGraph(elkJson: any) {
        setGraph({ ...elkJson })  // new ref triggers ElkCanvas re-import
      },
      exportGraph() {
        if (!editorRef.current) return null
        return tldrawToElk(editorRef.current)
      },
      getEditor() {
        return editorRef.current
      },
    }
    ;(window as any).__tldrawEditor = api
    return () => { delete (window as any).__tldrawEditor }
  }, [])

  if (!graph) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', minHeight: 300, color: '#94A3B8',
        fontFamily: 'system-ui, sans-serif', fontSize: 13,
      }}>
        Run Step 1 to load a graph into the editor
      </div>
    )
  }

  return (
    <ElkCanvas
      layoutedJson={graph}
      onExport={handleExport}
      onEditorReady={handleEditorReady}
      className="tldraw-elk-editor"
    />
  )
}
