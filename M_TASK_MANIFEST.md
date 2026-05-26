# M-Task Manifest — 10 Missions for 10 Claudes
## Codename: Extract Histogram (inspired by NVIDIA CCCL f984c90)

> **Design philosophy**: CCCL f984c90 extracts the first histogram-only pass from
> a fused `filter_and_histogram` kernel into its own `DeviceTopKHistogramKernel`.
> The result: the first pass runs with optimal occupancy (no filter logic dead weight),
> the loop starts at `pass = 1` instead of branching on `IsFirstPass`, and a shared
> `finalize_pass()` template replaces 40 lines of duplicated last-block coordination.
>
> We apply the same decomposition to `topology_gen.py` and the pipeline.
> Our monolith: user_text → single LLM call → pray for good JSON.
> Our target: a multi-pass pipeline where each pass has a dedicated kernel.

---

## M1 — `region_planner.py` — The Histogram Kernel
**What CCCL did**: Extract histogram-only pass into `DeviceTopKHistogramKernel`
**What we do**: Extract region planning into its own LLM pass

Create `backend/pipeline/topology/region_planner.py`:
- Input: full user text (NO truncation — the `text[:12000]` crime stops here)
- Output: `List[PlannedRegion]` — each with `{id, name, bbox, description, style_hints}`
- The LLM's only job: read user text, output a spatial plan with concrete bboxes
- Like CCCL's histogram kernel: does ONE thing (count bins), does it over the FULL input
- Parse user intent patterns: detect "git clone", "tree", "from C start D" chains
- Adaptive canvas: estimate total area from region count, don't hardcode 900×500

---

## M2 — `user_intent_parser.py` — The `extract_bin_op` Equivalent
**What CCCL did**: `extract_bin_op` maps keys to radix bins — pure function, no side effects
**What we do**: Map user text to structured intent — pure parse, no LLM call

Create `backend/pipeline/topology/user_intent_parser.py`:
- Regex + heuristic pass (zero LLM calls) that classifies user input:
  - `engineering_flow`: "git clone → tree → branch → experiment"
  - `academic_paper`: "encoder → decoder → attention → output"
  - `architecture_diagram`: "module A contains B, C, D"
  - `comparison_matrix`: "X vs Y vs Z"
  - `recursive_pattern`: "from C start, implement D, let E do F, then G..."
- Extract action verbs, technical nouns, containment relationships
- Output: `UserIntent` dataclass with `{diagram_type, entities, relationships, style_cues}`
- This runs BEFORE any LLM call — feeds into region_planner and entity_extractor

---

## M3 — `per_region_generator.py` — The Filter+Histogram Kernel (passes 1..N)
**What CCCL did**: After pass 0 (histogram-only), passes 1..N run fused filter+histogram
**What we do**: After region planning (pass 0), per-region generation runs N times

Create `backend/pipeline/topology/per_region_generator.py`:
- Input: one `PlannedRegion` + full user context + global entity registry
- Output: region-local ELK subgraph with nodes, edges, precise sizes
- Each region is an independent LLM call — like CCCL's thread blocks
- The LLM sees: (a) full user description, (b) this region's bbox constraint,
  (c) what other regions exist (for cross-region edge planning)
- Buffer management: like CCCL's `DoubleBuffer<key_in_t>`, swap between
  "draft" and "refined" versions of each region

---

## M4 — `finalize_pass.py` — The `finalize_pass()` Template
**What CCCL did**: Extract 40 lines of duplicated last-block coordination into
a single `finalize_pass()` template with a caller-supplied `counter_update_fn` lambda
**What we do**: Extract shared post-LLM validation into reusable finalize logic

Create `backend/pipeline/topology/finalize_pass.py`:
- `finalize_topology_pass(raw_llm_output, pass_context) -> ValidatedOutput`
- JSON parse + repair (move from topology_gen.py's 200-line `_repair_json`)
- Edge integrity validation (sources/targets exist)
- Bbox overlap detection and resolution
- Counter updates: track total entities, edges, passes completed
- Like CCCL's `finalize_pass`: one template, many callers

---

## M5 — `canvas_compositor.py` — The `DoubleBuffer` + Buffer Swap
**What CCCL did**: `DoubleBuffer<key_in_t>` with `selector ^= 1` for zero-copy swaps
**What we do**: Compose N region outputs into one canvas with layer management

Create `backend/pipeline/topology/canvas_compositor.py`:
- Input: List[RegionOutput] (each with local ELK subgraph + bbox)
- Output: unified ELK graph with absolute coordinates
- Layer system: each region is a layer (like MasterGo's layer model)
- Collision resolution: if regions overlap, negotiate (shrink, offset, z-order)
- Double-buffer for iterative refinement: draft → validate → refine → final
- Export: to ELK, to MastergoLayout, to SVG regions for per-region image gen

---

## M6 — `prompt_templates/` — The Policy Selector
**What CCCL did**: `PolicySelector` maps GPU arch → tuning params per kernel
**What we do**: Map diagram_type → optimal prompt template per pipeline stage

Create `backend/pipeline/topology/prompt_templates/`:
- `__init__.py`: `select_template(diagram_type, pass_stage) -> PromptTemplate`
- `engineering_flow.py`: prompts for git-clone/tree/experiment workflows
- `architecture.py`: prompts for nested module/submodule hierarchies
- `academic_paper.py`: prompts for encoder-decoder-attention pipelines
- `recursive_chain.py`: prompts for "from C start D, let E do F" patterns
- Each template includes: system prompt, user prompt template, example output,
  validation schema (JSON Schema for structured output)

---

## M7 — Wire `server.py` endpoint: `/api/topology-layered`
**What CCCL did**: The dispatch function wires pass 0 (histogram kernel) +
passes 1..N (fused kernel) + final filter into one launch sequence
**What we do**: Wire region_planner → per_region_generator[] → compositor

New endpoint in `server.py`:
- POST `/api/topology-layered`
- Request: `{text, canvas_width?, canvas_height?, output_format?}`
- Pipeline: intent_parse → region_plan → N × per_region_gen → finalize → compose
- Response: `{regions: [...], elk: {...}, mastergo: [...], diagnostics: {...}}`
- Each region can be individually re-generated (POST `/api/region-regenerate`)

---

## M8 — Frontend: `RegionCanvas.astro` — Interactive Layer Editor
**What CCCL did**: The output flows through the existing DeviceTopK API unchanged
**What we do**: The output renders in a draggable canvas that users can rearrange

Create `src/components/pipeline/RegionCanvas.astro`:
- Canvas view showing all regions as draggable, resizable boxes
- Each region shows its generated content (SVG thumbnail or placeholder)
- Drag to reposition, resize handles on corners
- "Regenerate this region" button per region
- Layer panel (like MasterGo): reorder, hide/show, lock layers
- Export: save positions back to server for final composition

---

## M9 — Sugiyama Integration: Per-Region Internal Layout
**What CCCL did**: Each radix pass refines the candidate set with better precision
**What we do**: Apply Sugiyama within each region for crossing-minimized internal layout

Modify `sugiyama.py` to accept a region's subgraph:
- `layout_region(nodes, edges, bbox_constraint) -> positioned_nodes`
- Respects the region's bbox: all nodes positioned within [x, y, w, h]
- Internal crossing minimization per region (local Sugiyama)
- Cross-region edges handled by compositor (M5) at the global level
- Coordinate compression: pack nodes tightly within region bounds

---

## M10 — End-to-End Test: The "CCCL Figure" Test
**What CCCL did**: The PR includes benchmark-only changes and restores bench.yaml
**What we do**: Create a test that generates the CCCL f984c90 architecture as a figure

Create `tests/test_layered_pipeline.py`:
- Input text: "DeviceTopK uses a multi-pass radix selection algorithm.
  Pass 0 runs a histogram-only kernel over the full input. Passes 1..N
  run a fused filter+histogram kernel with double-buffered candidate sets.
  The last block in each pass coordinates via atomicInc to run prefix sum..."
- Expected output: ≥5 regions (histogram kernel, fused kernel, double buffer,
  last-block coordination, dispatch loop)
- Validate: all edges reference existing nodes, no bbox overlaps,
  region count matches complexity estimate
- Benchmark: measure total LLM calls, total tokens, time per region

---

## Execution Order

```
M2 (intent parser, no LLM)     ← I will do this NOW
M4 (finalize_pass, shared util) ← I will do this NOW
M1 (region planner)             ← I will do this NOW
M3 (per-region generator)       ← I will do this NOW
M5 (canvas compositor)          ← I will do this NOW
M6 (prompt templates)           ← next Claude
M7 (server endpoint)            ← next Claude
M8 (frontend canvas)            ← next Claude
M9 (Sugiyama integration)       ← next Claude
M10 (end-to-end test)           ← next Claude
```

> The first 5 are the algorithmic core. Like CCCL f984c90, the kernel
> extraction and buffer management are the hard part. The dispatch
> wiring (M7) and UI (M8) follow naturally once the kernels exist.
