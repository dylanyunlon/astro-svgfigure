# Research #94 — Cytoscape.js + VivaGraph: Bio Network Visualization

**小弟 #94 — xiaodi**

---

## 1. Cytoscape.js (`upstream/cytoscape-js`)

**Repo:** https://github.com/cytoscape/cytoscape.js  
**Version cloned:** 3.35.0-unstable  
**License:** MIT  
**Stars:** ~10k  

### What it is
Cytoscape.js is the de-facto standard graph library for bioinformatics network visualization.
Used in: protein-protein interaction (PPI) networks, gene regulatory networks, metabolic pathways, drug-target networks.

### Source architecture (`src/`)

```
src/
├── collection/          # Node/edge collection ops (traversal, filter, degree, style, animation)
│   └── algorithms/      # Graph algorithms: BFS, DFS, Dijkstra, A*, Kruskal, PageRank, betweenness…
├── core/                # Cytoscape instance: add/remove elements, events, layout, export, viewport
│   └── animation/       # Core animation loop
├── define/              # Mixin/definition helpers for data, events, animation
├── extensions/
│   ├── layout/          # Bundled layouts: breadthfirst, circle, concentric, cose, fcose…
│   └── renderer/
│       ├── base/        # Renderer base class
│       ├── canvas/      # Default Canvas 2D renderer (full-featured)
│       └── null/        # Null renderer (headless / server-side)
└── selector/            # CSS-style selector engine for querying graph elements
```

### Key design points
- **Renderer plug-in architecture**: The default renderer is Canvas 2D (`extensions/renderer/canvas/`). A separate community package `cytoscape-three` provides a Three.js/WebGL backend. The core is renderer-agnostic.
- **Algorithm suite**: Built-in graph algorithms (shortest path, MST, clustering, centrality) — critical for bio network analysis.
- **Selector language**: CSS-like selectors (`cy.$('[degree > 5]')`) for querying nodes/edges.
- **Style system**: Declarative stylesheet (similar to CSS) for mapping data attributes → visual properties.
- **Layouts**: Cose (compound spring embedder) and Fcose are the main force-directed layouts used for bio networks.
- **Export**: PNG, JPEG, JSON, base64.
- **Headless mode**: Can run in Node.js for server-side graph processing.

### WebGL situation
Core canvas renderer is Canvas 2D. WebGL is available via third-party extensions:
- `cytoscape-three` (Three.js backend)
- `cytoscape-gl` (experimental WebGL)

For massive graphs (>50k nodes), switching to a WebGL renderer is the main scaling strategy.

---

## 2. VivaGraphJS (`upstream/vivagraph`)

**Repo:** https://github.com/anvaka/VivaGraphJS  
**License:** BSD  

### What it is
VivaGraphJS is a pure WebGL graph rendering library focused on performance. Where Cytoscape.js is feature-rich and bio-focused, VivaGraph is performance-first.

### Source architecture (`src/`)

```
src/
├── Algorithms/          # Graph traversal algos
├── Input/               # Input handling
├── Layout/              # Force-directed layout engine
├── Utils/               # Utilities
├── View/                # Renderer abstraction, SVG renderer, Canvas renderer
├── WebGL/               # WebGL renderer — the main performance backend
│   ├── webgl.js                 # WebGL context management
│   ├── webglAtlas.js            # Texture atlas for node sprites
│   ├── webglImage.js            # Image-based node rendering
│   ├── webglImageNodeProgram.js # GLSL shader program for image nodes
│   ├── webglInputEvents.js      # Mouse/touch events on WebGL canvas
│   ├── webglLine.js             # Line (edge) rendering
│   ├── webglLinkProgram.js      # GLSL shader program for edges
│   ├── webglNodeProgram.js      # GLSL shader program for nodes
│   ├── webglSquare.js           # Square primitive for nodes
│   ├── webglAtlas.js            # Sprite/texture atlas
│   └── texture.js               # Texture management
└── viva.js              # Main entry point / namespace
```

### Key design points
- **First-class WebGL**: Unlike Cytoscape.js, WebGL is the primary (not optional) backend.
- **GLSL shaders**: Custom vertex/fragment shaders for nodes (squares, images) and edges (lines).
- **Texture atlas**: Sprite-based node rendering for high node counts.
- **Layout**: Force-directed layout (Fruchterman-Reingold) with optional WebWorker offloading.
- **Scale**: Targets graphs with 100k+ nodes where Canvas 2D breaks down.

---

## 3. Comparison for Bio Network Use

| Feature | Cytoscape.js | VivaGraphJS |
|---|---|---|
| Primary renderer | Canvas 2D (WebGL via plugins) | WebGL native |
| Bio algorithms built-in | ✅ Rich (centrality, clustering, etc.) | ❌ Basic traversal only |
| Style/CSS system | ✅ Full declarative | ❌ Minimal |
| Compound nodes | ✅ Yes (groups/clusters) | ❌ No |
| Max scale | ~10k–50k nodes comfortably | 100k+ nodes |
| Bio community adoption | ✅ Very high | Low |
| Plugin ecosystem | ✅ Rich (fcose, cola, dagre, etc.) | Minimal |
| Headless/server | ✅ Yes | ❌ No |

### Recommendation for astro-svgfigure
- **Use Cytoscape.js** as the primary graph engine for bio networks (PPI, GRN, pathways). The algorithm suite and bio community plugins are irreplaceable.
- **Reference VivaGraphJS WebGL shaders** (`webglNodeProgram.js`, `webglLinkProgram.js`) as implementation reference for building a high-performance WebGL renderer layer on top of Cytoscape's renderer interface.
- The hybrid approach: Cytoscape data model + VivaGraph-style WebGL rendering is the path to both feature richness and scalability.

---

## 4. Integration Notes

### Cytoscape WebGL path
```js
// Mount with canvas renderer (default)
const cy = cytoscape({ container: el, elements: bioData, style: [...] });

// Or swap to community WebGL renderer
const cy = cytoscape({ renderer: { name: 'three' }, ... });
```

### VivaGraph WebGL path
```js
const graph = Viva.Graph.graph();
graph.addNode('TP53', { label: 'TP53', type: 'protein' });
graph.addLink('TP53', 'MDM2');

const renderer = Viva.Graph.View.renderer(graph, {
  graphics: Viva.Graph.View.webglGraphics()
});
renderer.run();
```

---

*Researched by 小弟 #94 (xiaodi) — 2026-06-15*
