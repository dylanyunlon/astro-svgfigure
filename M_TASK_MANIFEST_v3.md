# M-Task Manifest v3 — 10 Missions for 10 Claudes
## Codename: invoke_last_filter (NVIDIA CCCL f984c90, phase 3)

> **Recap**: M1-M6 complete — the kernel extraction + dispatch function:
>   M2 `user_intent_parser.py` — `extract_bin_op` (pure classifier, 559 lines)
>   M4 `finalize_pass.py` — `finalize_pass()` template (445 lines)
>   M1 `region_planner.py` — `DeviceTopKHistogramKernel` (510 lines)
>   M3 `per_region_generator.py` — fused Filter+Histogram kernel (450 lines)
>   M5 `canvas_compositor.py` — `DoubleBuffer` + compose (777 lines)
>   M6 `layered_pipeline.py` — complete `dispatch()` function (390 lines)
>
> **What's left**: the public API that calls dispatch(), the frontend
> that renders the result, and the refinement passes that use DoubleBuffer.
>
> CCCL f984c90 total: +334 -285 lines (net +49).
> Our total so far: 3131 new lines across 6 modules.
> Target: replace the 1217-line `topology_gen.py` monolith.
> Net delta when done: smaller codebase, more capability.

---

## M7 — Wire `server.py`: `/api/topology-layered` ← THIS CLAUDE
**CCCL**: The public API `DeviceTopK::TopK()` delegates to `dispatch()`
**We do**: HTTP endpoints delegate to `generate_layered_topology()`

- POST `/api/topology-layered` → full pipeline
- POST `/api/region-regenerate` → single-region re-dispatch
- SSE streaming: emit events per completed region
- Response: `{elk, layers, regions, diagnostics, canvas}`
- Pydantic request/response models with validation

**STATUS: I will implement this NOW.**

---

## M8 — `prompt_templates/` — The PolicySelector
**CCCL**: `PolicySelector` maps GPU arch → optimal tuning params
**We do**: Map diagram_type → optimal prompt per pipeline stage

Create `backend/pipeline/topology/prompt_templates/`:
- `__init__.py`: `select_template(diagram_type, stage) -> PromptTemplate`
- Per-type templates: engineering, architecture, flowchart, recursive, comparison
- Each: system prompt + user template + JSON schema + example output
- Replace hardcoded prompts in region_planner.py and per_region_generator.py

---

## M9 — Sugiyama Per-Region: `layout_within_bbox()`
**CCCL**: Each pass refines with higher-precision radix bits
**We do**: Apply Sugiyama within each region's bbox constraint

Add to `sugiyama.py`:
- `layout_within_bbox(nodes, edges, widths, heights, bbox) -> positions`
- Coordinate compression: pack nodes within region bounds
- Virtual nodes at region boundaries for cross-region edge stubs
- Integration point: called after per_region_generator, before compositor

---

## M10 — Frontend: `RegionCanvas.astro` — Interactive Layer Editor
**CCCL**: Output flows through existing API unchanged
**We do**: Draggable canvas with per-region layer management

Create `src/components/pipeline/RegionCanvas.astro`:
- Render regions as draggable/resizable boxes on canvas
- Per-region: SVG thumbnail, "regenerate" button, style controls
- Layer panel: reorder, hide/show, lock (MasterGo/Figma-style)
- Calls `/api/region-regenerate` on single-region updates

---

## M11 — Cross-Region Edge Routing (A*)
**CCCL**: `invoke_last_filter` resolves final candidates from buffer
**We do**: Route edges that cross region boundaries

Enhance `canvas_compositor.py`:
- A* pathfinding on region boundary graph
- Orthogonal routing with corner smoothing
- Edge bundling for parallel cross-region connections
- Avoid routing through region interiors

---

## M12 — Refinement Pass: DoubleBuffer Second Radix Pass
**CCCL**: Passes 2+ refine with higher radix bits
**We do**: Second LLM pass refines regions based on neighbor context

Add to `per_region_generator.py`:
- `refine_region(region, current_subgraph, neighbor_subgraphs) -> refined`
- LLM adjusts: node sizes for visual balance, edge alignment
- Uses DoubleBuffer: reads Current(), writes Alternate(), swap
- Convergence check: skip refinement if ΔNodes < threshold

---

## M13 — End-to-End Test: The CCCL Architecture Test
Create `tests/test_layered_pipeline.py`:
- Input: CCCL f984c90 description (histogram kernel, DoubleBuffer, dispatch)
- Validate: ≥5 regions, all edges valid, no bbox overlaps
- Benchmark: LLM calls, tokens, time per region
- Regression: deterministic grid layout for simple inputs

---

## M14 — MasterGo Export: Full Layout Object Generation
Enhance `mastergo_schema.py`:
- Per-region layer metadata (name, visibility, lock, z-index)
- Region → MastergoElement mapping with precise bbox
- Canvas-level composition (layer ordering)
- JSON export compatible with MasterGo import API

---

## M15 — Performance: Async Parallel Region Generation
**CCCL**: Thread blocks execute independently on different SMs
**We do**: Independent regions generate in parallel with asyncio.gather

Modify `per_region_generator.py`:
- `generate_all_regions_parallel()` using asyncio.gather
- Dependency graph: parallel for independent, sequential for dependent
- Rate limiting: max 3 concurrent LLM calls
- Progressive rendering: emit subgraphs as they complete

---

## M16 — Integration: Replace `topology_gen.py` Monolith
The culmination — make `/api/topology-layered` the default:
- Update frontend to call new endpoint
- Deprecate `/api/topology` → redirect to layered
- Remove duplicated code from topology_gen.py
- Final line count comparison: monolith vs decomposed pipeline

---

## Execution Order

```
M7  (server endpoint)        ← THIS CLAUDE: wire the public API
M8  (prompt templates)       ← next Claude
M9  (Sugiyama per-region)    ← next Claude
M10 (frontend canvas)        ← next Claude
M11 (cross-region routing)   ← next Claude
M12 (refinement pass)        ← next Claude
M13 (e2e test)               ← next Claude
M14 (mastergo export)        ← next Claude
M15 (parallel generation)    ← next Claude
M16 (monolith replacement)   ← next Claude
```
