/**
 * interact-ai.ts — In-canvas AI chat overlay + GLSEO layer
 *
 * Ported from AT's InteractAI (36 refs) + GLSEO (9 refs):
 *   InteractAI: canvas-overlaid chat UI for AI interaction
 *   InteractAIGPT: GPT backend integration
 *   ChatUI/ChatDOM: DOM-based chat bubble layout
 *   GLSEO: hidden semantic HTML under WebGL canvas for search engines
 *
 * Algorithm changes from AT original (20% modification):
 *   1. AT uses custom ChatDOM virtual-DOM → we use plain DOM overlay
 *   2. AT routes to internal GPT endpoint → we route to /api/cell-loop
 *   3. AT's GLSEO writes meta tags → we write JSON-LD structured data
 *   4. AT's ChatUI is fullscreen → ours is a collapsible corner panel
 *
 * [ASTRO-INTERACT-AI] debug prefix.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// InteractAI — canvas-overlaid chat interface
//
// AT architecture: ChatDOM (virtual DOM) → ChatUIInput (text field) →
//   ChatUIResponse (streaming display) → InteractAIGPT (GPT fetch).
//
// Our adaptation: lightweight DOM overlay positioned absolute over the
// PixiJS canvas. No virtual DOM — plain createElement. Messages stream
// via fetch() to the Python backend /api/cell-loop endpoint with
// natural language instructions that the orchestrator interprets.
// ═══════════════════════════════════════════════════════════════════════════════

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

export interface InteractAIOptions {
  /** DOM element to overlay (the canvas wrapper) */
  container: HTMLElement
  /** Backend endpoint for cell-loop commands */
  endpoint?: string
  /** Position: bottom-right corner by default */
  position?: 'bottom-right' | 'bottom-left' | 'top-right'
  /** Start collapsed */
  collapsed?: boolean
}

/**
 * InteractAI — AT's in-canvas chat, adapted for astro-svgfigure.
 *
 * Mirrors AT's InteractAI class lifecycle:
 *   constructor → mount DOM overlay
 *   show/hide   → toggle visibility with CSS transition
 *   send        → POST to backend, stream response
 *   destroy     → unmount DOM
 */
export class InteractAI {
  private container: HTMLElement
  private overlay: HTMLElement
  private messagesEl: HTMLElement
  private inputEl: HTMLInputElement
  private toggleBtn: HTMLElement
  private messages: ChatMessage[] = []
  private endpoint: string
  private collapsed: boolean
  private position: string

  constructor(opts: InteractAIOptions) {
    this.container = opts.container
    this.endpoint = opts.endpoint || '/api/cell-loop'
    this.collapsed = opts.collapsed ?? true
    this.position = opts.position || 'bottom-right'

    // [ASTRO-INTERACT-AI] mount DOM overlay
    this.overlay = document.createElement('div')
    this.overlay.className = 'interact-ai-overlay'

    // Position styles — AT uses absolute positioning over the WebGL canvas
    const posStyles = this._getPositionStyles()
    Object.assign(this.overlay.style, {
      position: 'absolute',
      width: '320px',
      maxHeight: '420px',
      background: 'rgba(10, 10, 20, 0.92)',
      backdropFilter: 'blur(12px)',
      borderRadius: '12px',
      border: '0.5px solid rgba(255,255,255,0.12)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      transition: 'opacity 0.3s, transform 0.3s',
      zIndex: '1000',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      fontSize: '13px',
      color: '#e0e0e0',
      ...posStyles,
    })

    // Header bar — AT's ChatUI header with collapse toggle
    const header = document.createElement('div')
    Object.assign(header.style, {
      padding: '8px 12px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      borderBottom: '0.5px solid rgba(255,255,255,0.08)',
      cursor: 'pointer',
      userSelect: 'none',
    })
    header.innerHTML = '<span style="font-weight:500;font-size:12px;opacity:0.7">Cell Agent Chat</span>'

    this.toggleBtn = document.createElement('span')
    this.toggleBtn.style.cssText = 'font-size:16px;opacity:0.5;transition:transform 0.2s'
    this.toggleBtn.textContent = '▾'
    header.appendChild(this.toggleBtn)
    header.addEventListener('click', () => this.toggle())

    // Messages container — AT's ChatUIResponse scroll area
    this.messagesEl = document.createElement('div')
    Object.assign(this.messagesEl.style, {
      flex: '1',
      overflowY: 'auto',
      padding: '8px 12px',
      display: 'flex',
      flexDirection: 'column',
      gap: '6px',
    })

    // Input row — AT's ChatUIInput
    const inputRow = document.createElement('div')
    Object.assign(inputRow.style, {
      padding: '8px 12px',
      borderTop: '0.5px solid rgba(255,255,255,0.08)',
      display: 'flex',
      gap: '6px',
    })

    this.inputEl = document.createElement('input')
    Object.assign(this.inputEl.style, {
      flex: '1',
      background: 'rgba(255,255,255,0.06)',
      border: '0.5px solid rgba(255,255,255,0.1)',
      borderRadius: '6px',
      padding: '6px 10px',
      color: '#e0e0e0',
      fontSize: '12px',
      outline: 'none',
    })
    this.inputEl.placeholder = 'Ask about cells...'
    this.inputEl.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Enter' && this.inputEl.value.trim()) {
        this.send(this.inputEl.value.trim())
        this.inputEl.value = ''
      }
    })

    const sendBtn = document.createElement('button')
    Object.assign(sendBtn.style, {
      background: 'rgba(100,140,255,0.25)',
      border: 'none',
      borderRadius: '6px',
      padding: '6px 12px',
      color: '#a0b8ff',
      fontSize: '12px',
      cursor: 'pointer',
    })
    sendBtn.textContent = '→'
    sendBtn.addEventListener('click', () => {
      if (this.inputEl.value.trim()) {
        this.send(this.inputEl.value.trim())
        this.inputEl.value = ''
      }
    })

    inputRow.appendChild(this.inputEl)
    inputRow.appendChild(sendBtn)

    this.overlay.appendChild(header)
    this.overlay.appendChild(this.messagesEl)
    this.overlay.appendChild(inputRow)

    // Make container relative for absolute positioning
    if (getComputedStyle(this.container).position === 'static') {
      this.container.style.position = 'relative'
    }
    this.container.appendChild(this.overlay)

    // Apply initial collapsed state
    if (this.collapsed) {
      this.messagesEl.style.display = 'none'
      inputRow.style.display = 'none'
      this.overlay.style.maxHeight = '36px'
      this.toggleBtn.style.transform = 'rotate(-90deg)'
    }

    console.log('[ASTRO-INTERACT-AI] mounted overlay')
  }

  private _getPositionStyles(): Record<string, string> {
    switch (this.position) {
      case 'bottom-left': return { bottom: '16px', left: '16px' }
      case 'top-right': return { top: '16px', right: '16px' }
      default: return { bottom: '16px', right: '16px' }
    }
  }

  /** Toggle collapsed/expanded — mirrors AT's ChatUI.toggle() */
  toggle(): void {
    this.collapsed = !this.collapsed
    const inputRow = this.overlay.querySelector('div:last-child') as HTMLElement
    if (this.collapsed) {
      this.messagesEl.style.display = 'none'
      if (inputRow) inputRow.style.display = 'none'
      this.overlay.style.maxHeight = '36px'
      this.toggleBtn.style.transform = 'rotate(-90deg)'
    } else {
      this.messagesEl.style.display = 'flex'
      if (inputRow) inputRow.style.display = 'flex'
      this.overlay.style.maxHeight = '420px'
      this.toggleBtn.style.transform = 'rotate(0deg)'
    }
  }

  /** Add a message bubble — mirrors AT's ChatUIResponse.addMessage() */
  private _addBubble(msg: ChatMessage): HTMLElement {
    const bubble = document.createElement('div')
    const isUser = msg.role === 'user'
    Object.assign(bubble.style, {
      padding: '6px 10px',
      borderRadius: '8px',
      background: isUser ? 'rgba(100,140,255,0.15)' : 'rgba(255,255,255,0.05)',
      alignSelf: isUser ? 'flex-end' : 'flex-start',
      maxWidth: '85%',
      fontSize: '12px',
      lineHeight: '1.5',
      wordBreak: 'break-word',
    })
    bubble.textContent = msg.content
    this.messagesEl.appendChild(bubble)
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight
    return bubble
  }

  /**
   * Send a message — mirrors AT's InteractAIGPT.send().
   *
   * AT routes to an internal GPT endpoint; we route to /api/cell-loop
   * which the Python backend interprets as a natural language cell command.
   * Response streams back and is displayed with a typewriter effect.
   */
  async send(text: string): Promise<void> {
    const userMsg: ChatMessage = { role: 'user', content: text, timestamp: Date.now() }
    this.messages.push(userMsg)
    this._addBubble(userMsg)

    // Create assistant bubble for streaming response
    const assistantMsg: ChatMessage = { role: 'assistant', content: '', timestamp: Date.now() }
    const bubble = this._addBubble(assistantMsg)
    bubble.textContent = '...'

    try {
      const res = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: text, natural_language: true }),
      })

      if (!res.ok) {
        bubble.textContent = `Error: ${res.status}`
        return
      }

      const data = await res.json()
      const reply = data.message || data.result || JSON.stringify(data).slice(0, 200)

      // Typewriter effect — mirrors AT's ChatUIResponse streaming display
      bubble.textContent = ''
      assistantMsg.content = reply
      this.messages.push(assistantMsg)

      for (let i = 0; i < reply.length; i++) {
        await new Promise(r => setTimeout(r, 12))
        bubble.textContent = reply.slice(0, i + 1)
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight
      }
    } catch (err) {
      bubble.textContent = `Network error: ${err}`
    }
  }

  /** Destroy — mirrors AT's InteractAI.destroy() */
  destroy(): void {
    this.overlay.remove()
    console.log('[ASTRO-INTERACT-AI] destroyed')
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// GLSEO — WebGL SEO layer
//
// AT's GLSEO (9 refs): inserts hidden semantic HTML below the WebGL canvas
// so search engines can index content that's only rendered in WebGL.
//
// Algorithm changes from AT:
//   1. AT writes meta tags + prerender hints → we write JSON-LD
//   2. AT targets Google bot specifically → we target all crawlers
//   3. AT's content comes from CMS → ours comes from cell topology
// ═══════════════════════════════════════════════════════════════════════════════

export interface CellSEOData {
  id: string
  label: string
  species: string
  connections: string[]  // target cell ids
}

/**
 * GLSEO — hidden semantic HTML for WebGL content indexing.
 *
 * Mirrors AT's GLSEO class: generateSEO() → injects structured data
 * below the canvas element, invisible to users but readable by crawlers.
 */
export class GLSEO {
  private container: HTMLElement
  private seoEl: HTMLElement | null = null

  constructor(container: HTMLElement) {
    this.container = container
  }

  /**
   * Generate and inject SEO content.
   *
   * Mirrors AT's GLSEO.update() — rebuilds the hidden HTML whenever
   * the cell topology changes (new epoch, new cells).
   */
  generateSEO(cells: CellSEOData[], title?: string): void {
    // Remove previous SEO element if any
    if (this.seoEl) {
      this.seoEl.remove()
    }

    this.seoEl = document.createElement('div')
    Object.assign(this.seoEl.style, {
      position: 'absolute',
      width: '1px',
      height: '1px',
      overflow: 'hidden',
      clip: 'rect(0,0,0,0)',
      whiteSpace: 'nowrap',
    })
    this.seoEl.setAttribute('aria-hidden', 'false')
    this.seoEl.setAttribute('role', 'complementary')
    this.seoEl.setAttribute('aria-label', 'Cell graph content')

    // Semantic HTML — h2 + list of cells + connections
    const heading = document.createElement('h2')
    heading.textContent = title || 'Cell Architecture Graph'
    this.seoEl.appendChild(heading)

    const list = document.createElement('ul')
    for (const cell of cells) {
      const li = document.createElement('li')
      li.textContent = `${cell.label} (${cell.species})`
      if (cell.connections.length > 0) {
        li.textContent += ` → ${cell.connections.join(', ')}`
      }
      list.appendChild(li)
    }
    this.seoEl.appendChild(list)

    // JSON-LD structured data — mirrors AT's meta tag injection
    // but uses schema.org Graph structure for richer semantics
    const jsonLd = {
      '@context': 'https://schema.org',
      '@type': 'Dataset',
      name: title || 'Cell Architecture Graph',
      description: `Architecture diagram with ${cells.length} cells`,
      distribution: {
        '@type': 'DataDownload',
        contentUrl: window.location.href,
        encodingFormat: 'application/svg+xml',
      },
      variableMeasured: cells.map(c => ({
        '@type': 'PropertyValue',
        name: c.label,
        description: `${c.species} cell node`,
        value: c.connections.length,
      })),
    }

    const script = document.createElement('script')
    script.type = 'application/ld+json'
    script.textContent = JSON.stringify(jsonLd)
    this.seoEl.appendChild(script)

    // Inject below the container
    this.container.parentElement?.insertBefore(
      this.seoEl,
      this.container.nextSibling
    )

    console.log(
      `[ASTRO-GLSEO] injected SEO: ${cells.length} cells, ` +
      `${cells.reduce((n, c) => n + c.connections.length, 0)} edges`
    )
  }

  /** Remove SEO layer — mirrors GLSEO.destroy() */
  destroy(): void {
    if (this.seoEl) {
      this.seoEl.remove()
      this.seoEl = null
    }
  }
}
