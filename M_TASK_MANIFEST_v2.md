# M-Task Manifest v2 — 10 Missions for 10 Claudes
## Codename: DoubleBuffer (NVIDIA CCCL f984c90, phase 2)

> **Recap**: Claudes 1-2 completed M1-M4 — the kernel extraction:
>   - M2 `user_intent_parser.py` — the `extract_bin_op` (pure classifier)
>   - M4 `finalize_pass.py` — the `finalize_pass()` template
>   - M1 `region_planner.py` — the `DeviceTopKHistogramKernel` (pass 0)
>   - M3 `per_region_generator.py` — the fused Filter+Histogram (passes 1..N)
>
> **What's missing**: the dispatch function.
>
> In CCCL f984c90, the real elegance is in `dispatch()`:
>
>     // Pass 0: dedicated histogram-only kernel
>     { launcher.doit(histogram_kernel, ...); }
>
>     // Passes 1..N: fused filter + histogram
>     DoubleBuffer<key_in_t> key_bufs(alloc[3], alloc[2]);
>     int pass = 1;
>     for (; pass < num_passes; pass++) {
>         launcher.doit(topk_kernel, key_bufs.Current(), key_bufs.Alternate(), ...);
>         key_bufs.selector ^= 1;  // zero-copy swap
>     }
>
>     // Final: invoke_last_filter on key_bufs.Current()
>
> Three clean phases: histogram → refine loop → final filter.
> Each reads from Current(), writes to Alternate(), swaps.
> The buffer management is invisible to the kernels.
>
> We need the same: a dispatch function that composes our kernels.

---

## M5 — `canvas_compositor.py` — The DoubleBuffer + dispatch()
**CCCL**: `DoubleBuffer<key_in_t>` with `selector ^= 1`, plus the dispatch
loop that wires pass 0 → passes 1..N → final filter
**We do**: Compose N region subgraphs into one unified canvas

This is the **critical missing piece**. We have kernels (M1-M4) but no
dispatch function to wire them together. Like CCCL's dispatch():
  1. Take histogram output (region plan) + kernel outputs (subgraphs)
  2. Place each subgraph at its region's bbox (absolute coordinate transform)
  3. Resolve cross-region edges (edges whose targets are in other regions)
  4. DoubleBuffer: draft → validate → refine → final
  5. Export: unified ELK graph + MastergoLayout + per-region layer data

**STATUS: I will implement this NOW.**

---

## M6 — `layered_pipeline.py` — The Complete dispatch() Function
**CCCL**: The `dispatch()` function in `dispatch_topk.cuh` that orchestrates
pass 0 (histogram), passes 1..N (fused), and the final filter
**We do**: Single async function that runs the full pipeline

Create `backend/pipeline/topology/layered_pipeline.py`:
- `async def generate_layered_topology(text, ai_engine, **opts) -> LayeredResult`
- Orchestration: intent_parse → region_plan → N × per_region_gen → compose
- DoubleBuffer pattern: first draft → validation pass → refinement pass
- Early exit: if num_passes == 1 (simple input), skip refinement
- PassContext flows through every stage (like CCCL's `counter` struct)
- Returns: `{elk, mastergo, regions, layers, diagnostics}`

---

## M7 — Wire `server.py`: `/api/topology-layered`
**CCCL**: The public API (`DeviceTopK::TopK()`) delegates to dispatch()
**We do**: The HTTP endpoint delegates to layered_pipeline

Add to `server.py`:
- POST `/api/topology-layered` → calls `generate_layered_topology()`
- POST `/api/region-regenerate` → re-runs one region's per_region_gen
- Response includes per-region data for frontend layer editing
- Streaming: SSE events for each completed region (progressive rendering)

---

## M8 — `prompt_templates/` — The PolicySelector
**CCCL**: `PolicySelector` maps GPU arch → optimal tuning params
**We do**: Map diagram_type → optimal prompt per pipeline stage

Create `backend/pipeline/topology/prompt_templates/`:
- `__init__.py`: `get_template(diagram_type, stage) -> str`
- Templates per diagram type × pipeline stage (plan, generate, refine)
- Each template includes: system prompt, user template, JSON schema
- Tested against intent_parser output categories

---

## M9 — Sugiyama Per-Region: Constrained Internal Layout
**CCCL**: Each pass refines the candidate set with better precision
**We do**: Apply Sugiyama within each region's bbox constraint

Modify `sugiyama.py` to add:
- `layout_within_bbox(nodes, edges, widths, heights, bbox) -> positions`
- Coordinate compression: pack nodes within region bounds
- Cross-region edge stubs: virtual nodes at region boundaries
- Integration with per_region_generator: post-LLM layout refinement

---

## M10 — Frontend: `RegionCanvas.astro` — Interactive Layer Editor
**CCCL**: Output flows through the existing API unchanged
**We do**: Draggable canvas with per-region layer management

Create `src/components/pipeline/RegionCanvas.astro`:
- Canvas renders all regions as draggable/resizable boxes
- Per-region: SVG thumbnail, "regenerate" button, style controls
- Layer panel: reorder, hide/show, lock (like MasterGo/Figma)
- Export: save positions for server recomposition

---

## M11 — Cross-Region Edge Routing
**CCCL**: `invoke_last_filter` resolves the final set of candidates
**We do**: Route edges that cross region boundaries

Add to `canvas_compositor.py`:
- Edge routing algorithm: A* on region boundary graph
- Avoid crossing region interiors (route along edges)
- Orthogonal routing with corner smoothing
- Edge bundling for parallel cross-region connections

---

## M12 — Refinement Pass: The Second Radix Pass
**CCCL**: Passes 2+ refine the candidate set with higher radix bits
**We do**: Second LLM pass to refine generated regions based on neighbors

Add to `per_region_generator.py`:
- `async def refine_region(region, current_subgraph, neighbors) -> refined`
- Input: current subgraph + neighboring regions' subgraphs
- LLM adjusts: node sizes to match visual balance, edge alignment
- Like CCCL: reads from Current(), writes to Alternate(), then swap

---

## M13 — End-to-End Test: The CCCL Architecture Test
**CCCL**: The PR includes benchmark-only changes and restores bench.yaml
**We do**: Generate the CCCL f984c90 architecture as a test figure

Create `tests/test_layered_pipeline.py`:
- Test input: "DeviceTopK multi-pass radix selection with histogram kernel
  extraction, DoubleBuffer candidate management, and finalize_pass template"
- Validate: ≥5 regions, all edges valid, no bbox overlaps
- Benchmark: LLM calls, tokens, time per region

---

## M14 — MasterGo Export: Full Layout Object Generation
**CCCL**: The output feeds into the existing CUB device API
**We do**: Export to MasterGo-compatible layer format

Enhance `mastergo_schema.py`:
- Per-region layer metadata (layer name, visibility, lock state)
- Region-to-MastergoElement mapping with precise bbox
- Canvas-level composition data (layer ordering, z-index)
- JSON export compatible with MasterGo import

---

## Execution Order

```
M5  (canvas compositor)        ← THIS CLAUDE: the dispatch() function
M6  (layered pipeline)         ← THIS CLAUDE: wire the kernels together
M7  (server endpoint)          ← next Claude
M8  (prompt templates)         ← next Claude
M9  (Sugiyama per-region)      ← next Claude
M10 (frontend canvas)          ← next Claude
M11 (cross-region routing)     ← next Claude
M12 (refinement pass)          ← next Claude
M13 (e2e test)                 ← next Claude
M14 (mastergo export)          ← next Claude
```

> The design principle: CCCL f984c90 is +334/-285 lines, net +49.
> The code got SHORTER while gaining a new kernel, removing a template
> parameter, and extracting a shared template.  Our code should do the
> same: the total line count of the pipeline should DECREASE as we
> replace monolithic topology_gen.py with composable kernels.
