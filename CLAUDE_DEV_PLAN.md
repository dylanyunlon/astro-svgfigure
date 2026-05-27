# Claude Development Plan — astro-svgfigure Topology Pipeline
## Codename: CCCL f984c90 Multi-Pass Decomposition

### Completed Progress

| Claude # | Status | Tasks | Key Deliverables |
|----------|--------|-------|-----------------|
| 1st | ✅ Done | M1-M4 | user_intent_parser, finalize_pass, region_planner, per_region_generator |
| 2nd | ✅ Done | M5-M6 | canvas_compositor (DoubleBuffer), layered_pipeline (dispatch) |
| 3rd | ✅ Done | M7 | server_layered_routes, /api/topology-layered endpoint |
| 4th | ✅ Done | M8-M9 + removebg | prompt_templates (PolicySelector+Iconify), layout_within_bbox, remove.bg Canva key pool Tier 0 |
| 5th | ✅ Done | M10-M11 | RegionCanvas.astro interactive layer editor, cross-region edge routing |
| 6th | ✅ Done | M12-M13 | DoubleBuffer refinement pass, e2e test suite M1-M16 |
| 7th | ✅ Done | M14 | MasterGo per-region export: MastergoLayer, layered_to_mastergo_layout, /api/mastergo-export, MasterGo Import API format |
| 8th | ✅ Done | M15 | icon_extractor 3-pass pipeline (saturation detection + two-pass removebg) |
| 9th | ✅ Done | M16 | Async parallel region generation (asyncio.gather + dependency graph) |

### Upcoming Plan

| Claude # | Tasks | Description |
|----------|-------|-------------|
| 10th | M17 | Replace topology_gen.py monolith — wire /api/topology-layered as default |
| 11th | M18 | Prompt engineering — few-shot examples per diagram type, test with GenDB/UICopilot |
| 12th | M19 | Icon post-processing — auto-crop whitespace, normalize to 32×32, embed as data URI |
| 13th | M20 | Cross-region edge bundling — parallel edges merge, orthogonal routing with smoothing |
| 14th | M21 | Canvas zoom/pan — frontend pinch-zoom, minimap, viewport sync |
| 15th | M22 | Region drag-and-drop — reposition regions, auto-reflow edges |
| 16th | M23 | Export pipeline — SVG/PNG/PDF export with embedded icons and fonts |
| 17th | M24 | Batch topology — multiple diagrams from one document (paper→figures) |
| 18th | M25 | Style system — color themes per diagram type, dark mode support |
| 19th | M26 | Animation pipeline integration — topology→animation frame sequence |
| 20th | M27 | Real-time collaboration — WebSocket region locking, concurrent edits |
| 21st | M28 | Performance — lazy region rendering, virtual scroll for large diagrams |
| 22nd | M29 | Accessibility — ARIA labels, keyboard navigation, screen reader |
| 23rd | M30 | i18n — Chinese/English/Japanese entity names, layout direction |
| 24th | M31 | Version history — undo/redo stack, snapshot comparison |
| 25th | M32 | Template library — prebuilt layouts for common architectures |
| 26th | M33 | AI feedback loop — user edits train better region planning |
| 27th | M34 | Plugin system — custom node renderers, icon packs, layout algorithms |
| 28th | M35 | CI/CD — GitHub Actions: lint, test, build, deploy preview |
| 29th | M36 | Documentation — API docs, user guide, architecture decision records |
| 30th | M37 | Load testing — 100 concurrent topology requests, P99 latency |
| 31st | M38 | Security audit — input sanitization, API key rotation, CORS policy |
| 32nd | M39 | Mobile optimization — responsive canvas, touch gestures |
| 33rd | M40 | Offline mode — service worker, local rembg fallback |
| 34th | M41 | Analytics — usage tracking, error reporting, quality metrics |
| 35th | M42 | MasterGo plugin — direct import/export from MasterGo desktop |
| 36th | M43 | Figma plugin — FigJam integration, auto-sync |
| 37th | M44 | VS Code extension — diagram preview in editor sidebar |
| 38th | M45 | Final integration — all modules wired, production deployment |

### Architecture Mapping (CCCL f984c90)

```
CCCL dispatch()                    Our Pipeline
───────────────                    ────────────
extract_bin_op       (pure func)  → user_intent_parser.py     (M2)
finalize_pass()      (template)   → finalize_pass.py          (M4)
DeviceTopKHistogram  (pass 0)     → region_planner.py         (M1)
DeviceTopKKernel     (pass 1..N)  → per_region_generator.py   (M3)
DoubleBuffer+compose (dispatch)   → canvas_compositor.py      (M5)
dispatch()           (orchestr)   → layered_pipeline.py       (M6)
DeviceTopK::TopK()   (public API) → server_layered_routes.py  (M7)
PolicySelector       (tuning)     → prompt_templates/          (M8)
Per-pass refinement  (radix bits) → layout_within_bbox         (M9)
Key pool rotation    (fallback)   → removebg_canva_client.py  (Tier 0)
```
