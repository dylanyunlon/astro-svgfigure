# src/lib/sph — SPH Physics & Rendering Engine

Real-time 2-D Smoothed Particle Hydrodynamics engine with a full WebGPU
rendering pipeline.  The system maps Transformer-analogy cell species onto
fluid-physical bodies, driven by Apollo CyberRT QoS profiles that translate
messaging semantics (reliability, durability, history depth, message rate) into
physical parameters (viscosity, boundary friction, trail length, emitter rate).

**106 TypeScript modules · ~63 300 lines · 32 shader host files · 113 inline WGSL/GLSL shaders**

---

## Table of Contents

1. [Full Module Dependency Graph (ASCII)](#full-module-dependency-graph)
2. [Per-Frame Data Flow](#per-frame-data-flow)
3. [Render Compositor — 13-Pass Pipeline](#render-compositor--13-pass-pipeline)
4. [All 84 sph/ Modules](#all-84-sph-modules)
5. [All 22 collision/ Modules](#all-22-collision-modules)
6. [All 32 Shader Host Files](#all-32-shader-host-files)
7. [QoS → Physics Mapping](#qos--physics-mapping)
8. [Species Visual DNA](#species-visual-dna)
9. [Quick Start](#quick-start)

---

## Full Module Dependency Graph

```
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │                            EXTERNAL INPUTS                                           │
 │                                                                                      │
 │   SSE /api/cell-events          /api/cells           Apollo QoS Profiles             │
 │         │                          │                        │                        │
 └─────────┼──────────────────────────┼────────────────────────┼────────────────────────┘
           │                          │                        │
           ▼                          ▼                        ▼
 ┌─────────────────────┐  ┌────────────────────┐  ┌──────────────────────────┐
 │  sph-epoch-bridge   │  │  cell-body-bridge  │  │  qos-spatial-bridge     │
 │  (SSE ↔ SPHWorld)   │  │  (cells → rigid)   │  │  (8 Apollo profiles)    │
 └────────┬────────────┘  └────────┬───────────┘  │  qosSpatial             │
          │                        │               │  (5 lightweight presets) │
          │                        │               └────────────┬─────────────┘
          ▼                        ▼                            │
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  SIMULATION CORE                                                                   │
 │                                                                                    │
 │  ┌──────────────────────────────────────────────────────────────────┐               │
 │  │  world-stepper  ◄──── Main loop: stepWorld() per substep        │               │
 │  │       │                                                          │               │
 │  │       ├── dfsph-solver ◄── sph-kernels                          │               │
 │  │       ├── fluid-rigid-coupling ◄── rigid-body                   │               │
 │  │       ├── spatial-hash (CPU neighbour grid)                      │               │
 │  │       ├── world-boundary (domain walls)                          │               │
 │  │       └── emitter-strategy ◄── qos-spatial-bridge               │               │
 │  └───────────────┬──────────────────────────────────────────────────┘               │
 │                  │                                                                  │
 │                  ▼                                                                  │
 │  ┌──────────────────────────────────────────────────────────────────┐               │
 │  │  collision/ subsystem                                            │               │
 │  │       │                                                          │               │
 │  │       ├── collision-world (broad → narrow → solve → correct)     │               │
 │  │       │       ├── sort-and-sweep (broad phase)                   │               │
 │  │       │       ├── bvh-tree ◄── aabb-manager                     │               │
 │  │       │       ├── gjk-epa (narrow: convex detect + penetration)  │               │
 │  │       │       ├── sat-solver (narrow: OBB overlap)               │               │
 │  │       │       ├── contact-manifold (contact generation)          │               │
 │  │       │       ├── impulse-solver ◄── constraints                 │               │
 │  │       │       └── scene-query (ray-cast, overlap, closest)       │               │
 │  │       └── CollisionEvents (BEGIN / STAY / END callbacks)         │               │
 │  └───────────────┬──────────────────────────────────────────────────┘               │
 │                  │                                                                  │
 └──────────────────┼──────────────────────────────────────────────────────────────────┘
                    │
                    ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  GPU INFRASTRUCTURE                                                                │
 │                                                                                    │
 │  ┌─────────────────────────────────────────────────────────────────────────────┐   │
 │  │  SPHWorld  (WebGPU buffers + compute pipelines + render passes)             │   │
 │  │       │                                                                     │   │
 │  │       ├── SPHGPUOrchestrator (density / force / integrate compute shaders)  │   │
 │  │       │       └── SpatialHashGrid (GPU neighbour hash)                      │   │
 │  │       │               └── NeighborListBuilder (CSR neighbour lists)         │   │
 │  │       ├── BoundaryModel (GPU wall particle buffers)                         │   │
 │  │       └── ParticleRenderer (instanced circle draw)                          │   │
 │  └────────────────────────────────────┬────────────────────────────────────────┘   │
 │                                       │                                            │
 │  sph-bridge ◄── sph-worker            │   (Comlink Web Worker offload)             │
 │                                       │                                            │
 └───────────────────────────────────────┼────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  AT RENDER PIPELINE  (render-compositor.ts — 13-pass forward pipeline)             │
 │                                                                                    │
 │  ┌─ Pass 0 ── environment-fx (brick grid + voronoise + chromatic aberr) ────────┐ │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 1 ── at-navier-stokes (splat → advect → vorticity → pressure) ────────┐ │
 │  │              ◄── interactive-fluid (mouse/touch → splat queue)               │  │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 2 ── at-flower-particle + at-spline-particle (GPGPU lifecycle) ────────┐ │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 3 ── cell-material-system (per-cell PBR / Matcap → geoFBO) ───────────┐ │
 │  │              ◄── species-shader-registry  ◄── at-pbr-material               │  │
 │  │              ◄── physics-uniform-bridge                                      │  │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 4 ── at-flower-particle .render() (instanced quads) ──────────────────┐ │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 5 ── at-spline-particle .render() (instanced quads) ──────────────────┐ │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 6 ── particle-compositor (bitonic sort + alpha + glow → sceneFBO) ─────┐ │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 7 ── at-water-surface (wave sim + mesh + splash → waterFBO) ───────────┐ │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 8 ── at-volumetric-light (occlusion → radial blur → Mie → vlFBO) ─────┐ │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 9 ── atmosphere (Rayleigh + Mie + depth fog → atmoFBO) ────────────────┐ │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 10 ── at-bloom-postprocess (bright extract → blur → composite) ────────┐ │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 11 ── post-process (Kuwahara / Film Grain / Chromatic Aberr) ──────────┐ │
 │  └──────────────────────────────────────────────────────────────┬───────────────┘  │
 │                                                                 ▼                  │
 │  ┌─ Pass 12 ── LUT colour grade (3-D trilinear LUT → swap-chain → canvas) ─────┐ │
 │  └──────────────────────────────────────────────────────────────────────────────┘  │
 │                                                                                    │
 └────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  VISUAL IDENTITY STACK                                                             │
 │                                                                                    │
 │  species-visual-dna ──┬── species-shader-registry (SDF + material + pattern)       │
 │                       ├── cell-material-system (5 Transformer species PBR/matcap)  │
 │                       ├── physics-uniform-bridge (live SPH → shader uniforms)      │
 │                       ├── uil-species-live (2593+ AT UIL params)                   │
 │                       └── cell-visual-identity (physics → morphology derivation)   │
 │                                                                                    │
 │  organic-sdf ◄── species-shader-registry (flower / koch / julia SDF shapes)       │
 │  chromatic-adaptation ◄── color-palette ◄── qosSpatial                            │
 │  lut-generator ◄── qosSpatial (3-D LUT per QoS zone)                             │
 │  tone-mapping (ACES filmic HDR)                                                    │
 │                                                                                    │
 └────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  PROCEDURAL PATTERNS & MORPHOGENESIS                                               │
 │                                                                                    │
 │  reaction-diffusion (Gray-Scott WebGPU compute)                                    │
 │  turing-pattern (Gray-Scott species-specific modes)                                │
 │  morphogenesis (L-system plant growth → Bézier paths)                              │
 │  differential-growth (organic fractal folds)                                       │
 │  natural-patterns (Voronoi + Worley noise WebGPU)                                  │
 │  phyllotaxis (Fibonacci golden-angle spirals)                                      │
 │  physarum-sim (GPU slime-mould agents) ◄── physarum-edge-bridge                   │
 │                                                                                    │
 └────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  FLOW FIELDS & FLUID BACKGROUNDS                                                   │
 │                                                                                    │
 │  curl-flow-field (WebGPU 3-D curl noise)                                           │
 │  noise-flow-field (curl + FBM overlay)                                             │
 │  boids-compute (WebGPU separation + alignment + cohesion)                          │
 │  flowmap-bridge (SPH velocity → OGL Flowmap texture)                               │
 │  ogl-flowmap-bridge (pure-TS Flowmap.js port)                                      │
 │  ocean-background (Gerstner wave mesh) ◄── ocean-bridge                           │
 │  lattice-boltzmann-bg (LBM macro flow + SPH overlay)                               │
 │                                                                                    │
 └────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  VISUAL EFFECTS                                                                    │
 │                                                                                    │
 │  collision-fx-system ◄── CollisionEvents + at-flower-particle                     │
 │  contact-sparks ◄── CollisionEvents                                               │
 │  ripple-effect ◄── CollisionEvents + ParticleRenderer                             │
 │  water-caustics (CPU caustics port)                                                │
 │  curl-aura (curl-noise halo WebGL2) ◄── particle-instancing                      │
 │  screen-space-reflections ◄── cell-material-system + physics-uniform-bridge       │
 │  environment-fog (depth fog + god rays)                                            │
 │  environment-fx (bio-lab atmosphere)                                               │
 │  transition-system (appear / disappear / morph) ◄── cell-visual-identity          │
 │                                                                                    │
 └────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  UTILITIES & INFRASTRUCTURE                                                        │
 │                                                                                    │
 │  types (shared interfaces + constants: MAX_PARTICLES=50000, WORKGROUP_SIZE=256)    │
 │  performance-budget (4-tier adaptive: ULTRA / HIGH / MEDIUM / LOW)                 │
 │  audio-physics-bridge (SPH state ↔ audio synthesis)                                │
 │  world-serializer (binary snapshot: serialize / deserialize)                       │
 │  world-renderer (Canvas2D debug fallback)                                          │
 │  debug-renderer (AABB wireframes, contact normals, BVH, force arrows)             │
 │  integration-test (end-to-end test harness)                                        │
 │  index (barrel re-export for entire sph/)                                          │
 │                                                                                    │
 └────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Per-Frame Data Flow

The complete data flow for a single frame, from epoch SSE event to canvas
pixel output:

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 1 — Epoch SSE                                               │
   │                                                                     │
   │  Server pushes SSE via /api/cell-events:                            │
   │    epoch_completed  │  cell_params_updated  │  topology_updated     │
   │                                                                     │
   │  sph-epoch-bridge.ts receives, parses, exponential backoff reconnect│
   └───────────────────────────┬─────────────────────────────────────────┘
                               │ { cellId, species, bbox, qosProfile }
                               ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 2 — Cell Registry                                           │
   │                                                                     │
   │  cell-body-bridge.ts maps 7 Transformer cells from /api/cells      │
   │  into SPH rigid body parameters via species_physics.json:           │
   │                                                                     │
   │    cellId ──► species (attention/ffn/layernorm/embedding/softmax)   │
   │          ──► position, size, mass, inertia, restitution, friction   │
   │          ──► boundary particle count                                │
   │                                                                     │
   │  qos-spatial-bridge.ts resolves Apollo QoS profile:                 │
   │    reliability → viscosity                                          │
   │    durability  → boundary friction                                  │
   │    depth       → trail length                                       │
   │    mps         → emitter rate                                       │
   │    priority    → force multiplier                                   │
   └───────────────────────────┬─────────────────────────────────────────┘
                               │ RigidBody[] + SpatialConfig
                               ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 3 — Physics World (world-stepper.ts per substep)            │
   │                                                                     │
   │  for each substep (1–4, adaptive via performance-budget):           │
   │                                                                     │
   │    3a. spatial-hash.buildSpatialHash(particles)                     │
   │    3b. dfsph-solver: density prediction → pressure correction       │
   │           └── sph-kernels: cubic spline, spiky grad, poly6         │
   │    3c. fluid-rigid-coupling: two-way SPH ↔ rigid body forces       │
   │    3d. emitter-strategy: spawn particles per QoS pattern            │
   │    3e. rigid-body: Euler integrate position/angle                   │
   │    3f. world-boundary: clamp/reflect at domain walls               │
   │                                                                     │
   │  GPU path (SPHGPUOrchestrator):                                     │
   │    dispatch density_shader → force_shader → integrate_shader        │
   │    hash_count → prefix_sum → prefix_sum_add (neighbour rebuild)     │
   └───────────────────────────┬─────────────────────────────────────────┘
                               │ Particle[] + RigidBody[] positions/velocities
                               ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 4 — Collision (collision/ subsystem)                        │
   │                                                                     │
   │  4a. Broad phase:  sort-and-sweep on AABB x-projections            │
   │        └── aabb-manager computes per-body AABBs                     │
   │        └── bvh-tree for spatial acceleration                        │
   │                                                                     │
   │  4b. Narrow phase: gjk-epa (convex) or sat-solver (OBB)            │
   │        └── GJK simplex walk → EPA penetration depth                 │
   │        └── contact-manifold: Sutherland-Hodgman clipping            │
   │                                                                     │
   │  4c. Solve:  impulse-solver (sequential impulse, N iterations)      │
   │        └── constraints: NonPenetration + Friction + Restitution     │
   │        └── PositionSolver: Baumgarte position correction            │
   │                                                                     │
   │  4d. Events: CollisionEvents dispatches BEGIN / STAY / END          │
   │        └── collision-fx-system: flower burst at contacts            │
   │        └── contact-sparks: directional spark particles              │
   │        └── ripple-effect: WebGPU ping-pong wave at impact           │
   └───────────────────────────┬─────────────────────────────────────────┘
                               │ ContactConstraint[] + CollisionEvent[]
                               ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 5 — AT Render (render-compositor.ts 13-pass pipeline)       │
   │                                                                     │
   │  Pass  0: environment-fx → envFBO (bio-lab background)              │
   │  Pass  1: at-navier-stokes → NS textures (vel + dye)               │
   │  Pass  2: at-flower-particle + at-spline-particle compute           │
   │  Pass  3: cell-material-system → geoFBO (per-cell PBR/Matcap)      │
   │  Pass  4: at-flower-particle render (instanced quads)               │
   │  Pass  5: at-spline-particle render (instanced quads)               │
   │  Pass  6: particle-compositor → sceneFBO (sort + alpha + glow)      │
   │  Pass  7: at-water-surface → waterFBO (wave sim + mesh)             │
   │  Pass  8: at-volumetric-light → vlFBO (god rays)                    │
   │  Pass  9: atmosphere → atmoFBO (Rayleigh + Mie + depth fog)         │
   │  Pass 10: at-bloom-postprocess → bloomFBO (extract → blur → add)    │
   │  Pass 11: post-process → postFBO (Kuwahara / grain / chroma)        │
   │  Pass 12: LUT colour grade → swap-chain                             │
   └───────────────────────────┬─────────────────────────────────────────┘
                               │ FBO chain
                               ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 6 — Composite                                               │
   │                                                                     │
   │  render-compositor chains all FBOs through pass-through blits:      │
   │                                                                     │
   │    envFBO ──► geoFBO (blend) ──► sceneFBO (particles composited)    │
   │    ──► waterFBO (overlay) ──► vlFBO (additive god rays)             │
   │    ──► atmoFBO (atmospheric fog) ──► bloomFBO (additive bloom)      │
   │    ──► postFBO (artistic style) ──► LUT pass (colour grade)         │
   │                                                                     │
   │  Optional overlays composited in parallel:                          │
   │    screen-space-reflections (SSR: hi-z ray march → resolve)         │
   │    curl-aura (WebGL2 curl-noise halos)                              │
   │    particle-instancing (WebGL2 instanced soft particles)            │
   │    edge-flow-renderer (QoS-tinted spline flow)                      │
   │    debug-renderer (AABB wireframes, contact normals)                │
   └───────────────────────────┬─────────────────────────────────────────┘
                               │ final pixel data
                               ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  STAGE 7 — Canvas                                                  │
   │                                                                     │
   │  ATRenderPipeline.ts: WebGPU swap-chain presentSurface()            │
   │    fallback: Canvas2D via world-renderer.ts                         │
   │                                                                     │
   │  Frame budget managed by performance-budget.ts:                     │
   │    ULTRA (16ms) → HIGH (20ms) → MEDIUM (28ms) → LOW (40ms)         │
   │    Auto-downgrades substeps, particle caps, render resolution       │
   └─────────────────────────────────────────────────────────────────────┘
```

### Concise one-line summary

```
epoch SSE → sph-epoch-bridge → cell-body-bridge (cell registry) → world-stepper
(physics world) → collision/ (broad→narrow→solve→events) → render-compositor
(13 AT passes) → FBO composite chain → LUT grade → canvas swap-chain
```

---

## Render Compositor — 13-Pass Pipeline

The definitive top-level `render-compositor.ts` (M745) unifies all AT rendering
modules.  It supersedes both `at-scene-compositor.ts` (M730, 9-pass) and
`at-render-pipeline.ts` (M720) by combining their responsibilities and adding
environment FX, atmosphere, artistic post-process, per-pass GPU timing, and
performance budget.

| Pass | Module | Input | Output | Description |
|------|--------|-------|--------|-------------|
| 0 | `environment-fx` | — | envFBO | Bio-lab background: brick grid + voronoise + chromatic aberration |
| 1 | `at-navier-stokes` | mouse/touch + SPH vel | NS vel/dye tex | Grid fluid: splat → advect → vorticity → divergence → pressure → gradient |
| 2 | `at-flower-particle` + `at-spline-particle` | tPos0 | tPos1 | GPGPU lifecycle compute: petal spirals + Catmull-Rom splines |
| 3 | `cell-material-system` | species registry | geoFBO | Per-cell Cook-Torrance PBR or Matcap Fresnel material render |
| 4 | `at-flower-particle` | tPos1 | geoFBO+ | Instanced quad draw: flower petal particles |
| 5 | `at-spline-particle` | tPos1 | geoFBO+ | Instanced quad draw: Bézier spline particles |
| 6 | `particle-compositor` | all particles | sceneFBO | GPU bitonic sort → back-to-front alpha → additive glow halo |
| 7 | `at-water-surface` | wave state | waterFBO | Wave simulation + Gerstner mesh + water-particle overlay |
| 8 | `at-volumetric-light` | scene depth | vlFBO | Screen-space god rays: occlusion → radial blur → Mie scatter |
| 9 | `atmosphere` | scene + depth | atmoFBO | Rayleigh scattering + Mie phase + depth fog + sky gradient |
| 10 | `at-bloom-postprocess` | scene HDR | bloomFBO | UE5-style: bright extract → separable Gaussian blur → additive |
| 11 | `post-process` | bloomFBO | postFBO | Artistic styles: Kuwahara oil paint / Film Grain / Chromatic Aberration |
| 12 | LUT (inline) | postFBO | canvas | 3-D trilinear LUT colour grade → swap-chain surface |

---

## All 84 sph/ Modules

One-line purpose for each of the 84 TypeScript modules in `src/lib/sph/`
(excluding the 22 `collision/` modules listed separately below).

```
ATRenderPipeline.ts        Unified render pipeline facade — probes WebGPU, auto-falls back to Canvas2D; single API: addFluid / addObstacle / step / render / tick
BoundaryModel.ts           GPU boundary model — wall particle buffers, boundary density contribution in compute shaders
NeighborListBuilder.ts     Builds Compressed Sparse Row (CSR) neighbour lists from SpatialHashGrid results
ParticleRenderer.ts        GPU particle renderer — instanced circle drawing with per-species colour, glow, and size; hosts SPLAT/COMPOSITE/PARTICLE shaders
SPHGPUOrchestrator.ts      GPU compute orchestrator — dispatches density, force, integrate, hash, prefix-sum compute shaders per substep with buffer ping-pong
SPHWorld.ts                WebGPU SPH world — manages GPU buffers, compute pipelines, and render passes for the full particle simulation
SpatialHashGrid.ts         GPU-accelerated spatial hash grid for neighbour search in compute shaders
at-bloom-postprocess.ts    AT UnrealBloom WGSL port (M714) — bright extract → separable Gaussian blur → additive composite; per-species bloom via createATBloomForSpecies()
at-flower-particle.ts      AT FlowerParticleShader WebGPU/WGSL port (M710) — petal-shaped particle lifecycle, edge-route-to-flower-spline conversion; 6 inline shaders
at-navier-stokes.ts        AT Navier-Stokes fluid compute WGSL port (M715) — grid-based incompressible fluid: splat → advect → vorticity → divergence → pressure → gradient
at-pbr-material.ts         AT PBR lighting WGSL port — Cook-Torrance BRDF (GGX/Smith), matcap Fresnel, thin-film iridescence, uniform packing for per-cell material instances
at-render-pipeline.ts      Pipeline orchestrator (M720) — chains AT modules in fixed-function forward pipeline with FBO pass-through; precursor to render-compositor
at-scene-compositor.ts     9-pass scene compositor (M730) — SPH readback → NS fluid → particle compute → sort/composite → PBR → water → volumetric → bloom → final
at-shader-utils.ts         AT shader utility library WGSL port — easing functions (30 curves), range remapping, Photoshop blend modes; WGSL_EASES + WGSL_RANGE + WGSL_BLEND_MODES
at-spline-particle.ts      AT SplineParticleLife WebGPU/WGSL port (M713) — Bézier spline particle instances with lifecycle presets; 6 inline shaders
at-volumetric-light.ts     AT VolumetricLight WGSL port (M716) — screen-space god rays: occlusion → radial blur → Mie scatter → composite; 7 inline shaders
at-water-surface.ts        AT WaterCeilingShader WebGPU/WGSL port (M711) — wave simulation, mesh render, water-particle overlay; 9 inline shaders (most of any module)
atmosphere.ts              Atmospheric scattering + fog post-process — Rayleigh scattering phase function, depth fog, sky gradient; ATMOSPHERE_PRESETS
audio-physics-bridge.ts    Audio ↔ physics bridge (M748) — bidirectional mapping between SPH state (density, velocity, collisions) and audio synthesis parameters
boids-compute.ts           WebGPU Boids — separation + alignment + cohesion in 3 compute passes; host uploads once, GPU advances per tick()
cell-body-bridge.ts        Cell registry → RigidBody bridge — maps 7 Transformer cells from /api/cells into SPH rigid body parameters via species_physics.json
cell-material-system.ts    Per-species material system (M719) — 5 Transformer species: attention=iridescent, ffn=glass, layernorm=marble, embedding=membrane, softmax=luminous
cell-visual-identity.ts    Physics → visual derivation (M731) — species → morphology (jellyfish/petal/coral/mycelium/crystal), QoS → border/flow/decoration/sparks
chromatic-adaptation.ts    Chromatic adaptation colour system (M584) — particle colours shift along natural gradients based on density/velocity/temperature; batch-resolve
collision-fx-system.ts     Collision FX flower burst (M741) — spawns flower-shaped particle bursts at contact points with impulse-scaled intensity
color-palette.ts           Dynamic colour palette per QoS profile (M566) — lygia blend modes (Screen, Overlay) ported to TS; resolveParticleColor() + rgbaToCss()
contact-sparks.ts          Contact spark system (M587) — short-lived directional spark particles at collision contacts
curl-aura.ts               Curl-noise aura halos (M749) — WebGL2 concentric SDF rings with curl-noise distortion around cell bodies; 2 inline GLSL shaders
curl-flow-field.ts         WebGPU 3-D curl-noise flow field (M606) — 2 inline WGSL compute shaders for curl advection + SPH injection
debug-renderer.ts          Collision debug overlay — AABB wireframes, contact normals, BVH hierarchy, force field arrows, easing animations (lygia port)
dfsph-solver.ts            Divergence-Free SPH pressure solver (Bender & Koschier 2015) — iterative density/divergence correction on CPU Particle[]
differential-growth.ts     Differential growth simulation — organic fractal folds (coral reefs, cerebral cortex, romanesco broccoli)
edge-flow-renderer.ts      Edge-flow particle renderer (M742) — dual-mode WebGPU compute + Canvas2D fallback; QoS-driven colour/speed/trail per topology edge; 4 inline shaders
emitter-strategy.ts        QoS-driven particle emitter patterns — Continuous / HighFreqStream / LowFreqPulse / ConstantField / BurstWave; patternForProfile()
environment-fog.ts         Environment fog system — depth fog + god rays + Mie scatter composite; 6 inline WGSL shaders; exported as ENVIRONMENT_FOG_WGSL bundle
environment-fx.ts          "Bio-laboratory" atmospheric effects — brick-tile grid + voronoise scatter + chromatic aberration; 2 inline WGSL shaders
flowmap-bridge.ts          SPH velocity → OGL Flowmap texture bridge (M573) — rasterises particle velocities to low-res Float32 RG grid with dissipation
fluid-rigid-coupling.ts    Two-way SPH ↔ rigid body coupling — boundary volume, density contribution, pressure/viscosity forces, momentum transfer
fluid-surface-mesh.ts      SPH → surface mesh reconstruction (M746) — Marching Squares 2D: scalar field → contour → ear-clipping triangulation
index.ts                   Barrel re-export for the entire sph/ module — re-exports all 83 sibling modules + collision/ subsystem
integration-test.ts        End-to-end integration test harness for the SPH pipeline — validates world-stepper + qos-spatial-bridge
interactive-fluid.ts       Mouse splat → advect → pressure fluid interaction (M743) — wires DOM pointer events to NavierStokesFluid compute pipeline
lattice-boltzmann-bg.ts    Dual-layer fluid — Lattice Boltzmann Method macro background flow + SPH particle overlay; 2 inline WGSL shaders
lut-generator.ts           3-D LUT colour grading generator (M624) — generates 17³ cube LUT textures with QoS-zone-weighted tone mapping
morphogenesis.ts           L-system plant-growth morphogenesis — graph edges become living branches via Lindenmayer systems; Bézier path particles; fromPreset()
natural-patterns.ts        Natural cell-surface textures via WebGPU compute — Voronoi + Worley noise (lygia port); per-species FBM; 1 inline WGSL shader
noise-flow-field.ts        Curl-noise + FBM flow field overlay for SPH particles — simplex basis + FBM + curl (lygia-inlined WGSL); 4 inline shaders
ocean-background.ts        Gerstner wave ocean background — 4-wave sum displacement on subdivided grid mesh + splash compute; 5 inline WGSL shaders
ocean-bridge.ts            WebGPU-Ocean ↔ cell-pubsub-loop bridge (M603) — wraps upstream SPH + MLSMPM simulators and FluidRenderer
ogl-flowmap-bridge.ts      OGL Flowmap bridge (M614) — ports Flowmap.js ping-pong to pure TypeScript/CPU, replaces mouse input with SPH velocity field
organic-sdf.ts             Inlined SDF primitives for organic cell outlines — flowerSDF, kochSDF, juliaSDF + species-parameterised organicOutline()
particle-compositor.ts     Unified particle render compositor (M718) — depth-sorts flower + spline particles, draws alpha-blended + additive-glow layers; 6 inline shaders
particle-instancing.ts     GPU instanced particle rendering (M745) — WebGL2 instanced pipeline replacing per-particle Canvas2D draw; 2 inline GLSL shaders
performance-budget.ts      4-tier adaptive performance budget (ULTRA/HIGH/MEDIUM/LOW) — auto-downgrades substeps, particle caps, render resolution on measured frame time
phyllotaxis.ts             Fibonacci golden-angle phyllotaxis spiral — sunflower/pinecone packing pattern generation
physarum-edge-bridge.ts    Physarum ↔ EdgeFlowRenderer bridge (M742) — GPU trail readback → CPU pheromone sampling → particle speed modulation (chemotaxis)
physarum-sim.ts            GPU Physarum polycephalum slime-mould simulation (Jones 2010) — agent-based trail deposition on diffusion grid; 1 inline WGSL shader
physics-uniform-bridge.ts  Bridges live SPH world state to per-cell shader uniforms — samples density, velocity, pressure, vorticity, kinetic energy from neighbourhood
post-process.ts            Full-screen post-process pipeline — Kuwahara (oil painting), Film Grain, Chromatic Aberration; independently stackable; 1 inline WGSL shader
qos-spatial-bridge.ts      Apollo CyberRT QoS → SPH physics mapping — 8 named profiles; reliability→viscosity, durability→friction, depth→trail, mps→emitter, priority→force
qosSpatial.ts              Lightweight QoS preset → SpatialConfig — 5 presets; reliability→stiffness, mps→viscosity, depth→persistence, durability→density; interpolateConfigs()
reaction-diffusion.ts      Gray-Scott reaction-diffusion WebGPU compute pipeline — per-species feed/kill parameters via GrayScottSpecies; 2 inline WGSL shaders
render-compositor.ts       Definitive 13-pass render compositor (M745) — supersedes at-scene-compositor + at-render-pipeline; adds env FX, atmosphere, post-process, GPU timing
rigid-body.ts              Rigid body state — position, velocity, angle, angular velocity, inverse mass/inertia; Euler integration, impulse application, boundary particle sampling
ripple-effect.ts           Ripple collision effect — WebGPU ping-pong wave simulation at impact points; 3 inline WGSL shaders
screen-space-reflections.ts SSR cell surface reflections (M747) — UE5-style hi-z ray march: depth pyramid → half-res march → temporal resolve → composite; 5 inline WGSL shaders
spatial-hash.ts            CPU spatial hash grid for neighbour queries — buildSpatialHash() → queryNeighbors(); O(1) cell lookup, tuneable cell size
species-shader-registry.ts Declarative shader stack registry — per-species SDF (flower/koch/julia/supershape/capsule/hexagon/star/roundbox/polygon), material, pattern, bloom, physics bindings
species-visual-dna.ts      Single-call facade: shader registry + physics sampling + UIL modulation → one VisualDNA struct per cell per frame
sph-bridge.ts              Comlink Web Worker bridge — initSPHWorld / addFluid / addBody / stepSPH / setQoS / raycast / terminateSPHWorker; offloads simulation to background thread
sph-epoch-bridge.ts        Backend epoch loop ↔ frontend SPHWorld bridge (M514) — listens SSE /api/cell-events, syncs rigid bodies, exponential backoff reconnect
sph-kernels.ts             SPH kernel math — cubic spline, spiky gradient, Poly6, viscosity Laplacian; includes selfTest() for kernel normalisation validation
sph-worker.ts              Web Worker entry point — receives Comlink-proxied calls, runs simulation loop off-main-thread
spline-particle-life.ts    Spline particle life system — Bézier spline-bound particles with lifecycle phases (spawn/flow/fade), edge-route conversion
tone-mapping.ts            ACES filmic HDR tone mapping — RRT + ODT fit matrices (Stephen Hill / Krzysztof Narkowicz); Color3 / Mat3 types
transition-system.ts       Cell appear/disappear/transform transitions (M748) — scale, dissolve (particle scatter), morph (curl-noise path interpolation); physics-aware timing
turing-pattern.ts          Gray-Scott Turing pattern generator via WebGPU compute — species-specific pattern modes; 2 inline WGSL shaders
types.ts                   Shared interfaces — GPUBufferSet, SimParams, ParticleData, QoSProfile, SpatialConfig, RigidBody, ContactConstraint; MAX_PARTICLES=50000, WORKGROUP_SIZE=256
uil-species-live.ts        AT UIL params × SPH physics live interpolation — 2593+ AT scene parameters per species, modulated by physics state in real time
water-caustics.ts          CPU port of WebGL water caustics (M613) — drop, stepSimulation, updateNormals
world-boundary.ts          Domain boundary definitions (rect, circle, polygon) and enforcement — wall particles, clamping, reflection; BoundaryShape union type
world-renderer.ts          Canvas2D world renderer — species-coloured particles, rigid body outlines, contacts, BVH debug; used by debug views and Canvas2D fallback
world-serializer.ts        Binary serialization for World snapshots — "SPHW" magic header, particle + rigid body + config; serialize() / deserialize()
world-stepper.ts           Main simulation loop — orchestrates DFSPH pressure solve, rigid body integration, fluid-rigid coupling, boundary enforcement, emitter spawning per substep
```

---

## All 22 collision/ Modules

One-line purpose for each module in `src/lib/sph/collision/`:

```
AABB.ts              Standalone AABB utilities — overlap, union, perimeter, expand, fromCircle, fromPoints, contains, center
BVHTree.ts           Bounding Volume Hierarchy (PascalCase) — surface-area heuristic insertion, refit, ray-cast, overlap queries
CollisionEvents.ts   Event dispatcher — BEGIN / STAY / END contact phases, pair caching, callback registration
CollisionWorld.ts    Monolithic collision world (PascalCase) — createCircleBody, createBoxBody, computeContactInfo
ContactSolver.ts     Class-based impulse contact resolution (PascalCase variant)
EPA.ts               Standalone Expanding Polytope Algorithm on raw Vec2[] arrays for penetration depth
GJK.ts               Class-based Circle / Polygon shapes with collide() — GJK simplex walk
PositionSolver.ts    Baumgarte position correction pass
SAT.ts               Separating Axis Theorem (PascalCase variant)
SceneQuery.ts        Full-featured scene query — CircleShape / AABBShape / CapsuleShape / ConvexPolygonShape; ray-cast, overlap, closest-point
SortAndSweep.ts      Sort-and-Sweep broad phase (PascalCase variant)
aabb-manager.ts      AABB computation, expansion, merge, area, ray-cast, containment tests
bvh-tree.ts          BVH for spatial acceleration — surface-area heuristic insertion, refit, ray-cast, overlap
collision-world.ts   Top-level collision world — broad (Sort-and-Sweep) → narrow (GJK/EPA) → contact → impulse solve → position correct
constraints.ts       Constraint types — NonPenetrationConstraint, FrictionConstraint, RestitutionConstraint
contact-manifold.ts  Contact point generation — Sutherland-Hodgman polygon clipping, warm starting, combined friction/restitution
gjk-epa.ts           GJK collision detection + EPA penetration for convex shapes — createBoxShape, createCircleShape
impulse-solver.ts    Sequential impulse constraint solver — configurable iterations, Baumgarte stabilisation
index.ts             Barrel re-export for the entire collision subsystem
sat-solver.ts        Separating Axis Theorem for OBB-OBB overlap tests
scene-query.ts       Ray-cast, overlap, and closest-point queries against the collision world
sort-and-sweep.ts    Sweep-and-Prune broad phase on AABB x-axis projections
```

---

## All 32 Shader Host Files

Each host `.ts` file contains one or more inline WGSL (or GLSL) shader strings.
Total: **113 distinct shader constants** across 32 files.

```
SPHGPUOrchestrator.ts       6 WGSL  DENSITY_SHADER            — SPH density accumulation compute
                                     FORCE_SHADER              — pressure + viscosity force compute
                                     INTEGRATE_SHADER          — symplectic Euler position/velocity integration
                                     HASH_COUNT_SHADER         — spatial hash cell counting
                                     PREFIX_SUM_SHADER         — parallel prefix sum (Blelloch scan)
                                     PREFIX_SUM_ADD_SHADER     — prefix sum block-add propagation

ParticleRenderer.ts         3 WGSL  SPLAT_SHADER              — particle-to-density-field splatting
                                     COMPOSITE_SHADER          — density field → colour composite
                                     PARTICLE_SHADER           — instanced per-species circle draw

at-navier-stokes.ts         7 WGSL  GRID_UNIFORM_WGSL         — shared grid resolution/dx uniform struct
                                     SPLAT_WGSL                — Gaussian dye/velocity injection
                                     ADVECTION_WGSL            — semi-Lagrangian velocity/dye advection
                                     VORTICITY_WGSL            — vorticity confinement force compute
                                     DIVERGENCE_WGSL           — velocity divergence field compute
                                     PRESSURE_WGSL             — Jacobi pressure iteration
                                     GRADIENT_WGSL             — pressure gradient subtraction (projection)

at-flower-particle.ts       6 WGSL  NOISE_WGSL                — simplex noise for petal turbulence
                                     UNIFORMS_WGSL             — shared uniform struct (time, bounds, params)
                                     SPLINE_WGSL               — Catmull-Rom spline evaluation helpers
                                     COMPUTE_SHADER            — petal lifecycle update compute
                                     VERTEX_SHADER             — instanced quad vertex transform
                                     FRAGMENT_SHADER           — petal SDF + colour + alpha fragment

at-spline-particle.ts       6 WGSL  NOISE_WGSL                — simplex noise for spline jitter
                                     UNIFORMS_WGSL             — shared uniform struct (time, bounds, params)
                                     SPLINE_WGSL               — Catmull-Rom + Bézier evaluation helpers
                                     COMPUTE_SHADER            — spline lifecycle update compute
                                     VERTEX_SHADER             — instanced quad vertex transform
                                     FRAGMENT_SHADER           — spline particle SDF + glow fragment

at-water-surface.ts         9 WGSL  WGSL_MATH                 — math helpers (PI, clamp01, smoothstep)
                                     WGSL_UNIFORMS             — water simulation uniform struct
                                     WGSL_WAVE_STEP            — wave equation propagation compute
                                     WGSL_WAVE_NORMAL          — normal map computation from height field
                                     WGSL_WAVE_DROP            — circular wave drop injection compute
                                     WGSL_WATER_RENDER         — water surface mesh render (reflection + refraction)
                                     WGSL_WATER_RENDER_FLAT    — flat water fallback render
                                     WGSL_PARTICLE_UPDATE      — water-particle physics update compute
                                     WGSL_PARTICLE_RENDER      — water-particle instanced render

at-pbr-material.ts          7 WGSL  WGSL_MATH_HELPERS         — saturate, pow5, PI constants
                                     WGSL_PBR_BRDF             — Cook-Torrance BRDF: F_Schlick, D_GGX, G_SmithGGX, pbrDirect, pbrAmbientSimple
                                     WGSL_FRESNEL              — Schlick Fresnel edge light (fresnel_f, fresnelRim)
                                     WGSL_IRIDESCENCE          — thin-film interference rainbow (iridescence)
                                     WGSL_MATCAP_FRESNEL_FRAG  — Matcap + Fresnel fragment (simplex noise distort)
                                     WGSL_PBR_FRAG             — full PBR fragment shader
                                     WGSL_FULLSCREEN_VS        — full-screen triangle vertex shader

at-bloom-postprocess.ts     5 WGSL  WGSL_BLOOM_UNIFORMS       — bloom parameter uniform struct
                                     WGSL_LUMINANCE            — Rec.709 luminance helper function
                                     WGSL_LUMINOSITY           — bright threshold extract vertex+fragment
                                     WGSL_GAUSSIAN_BLUR        — separable Gaussian blur vertex+fragment
                                     WGSL_COMPOSITE            — scene + bloom additive composite

at-volumetric-light.ts      7 WGSL  WGSL_VL_UNIFORMS          — volumetric light uniform struct
                                     WGSL_FULLSCREEN_VERT      — full-screen triangle vertex shader
                                     WGSL_PERLIN_NOISE         — Perlin turbulence noise helpers
                                     WGSL_OCCLUSION            — occlusion mask fragment
                                     WGSL_GOD_RAYS             — radial blur fragment (64-sample march)
                                     WGSL_MIE_SCATTER          — Mie scattering phase fragment
                                     WGSL_COMPOSITE            — additive composite fragment

particle-compositor.ts      6 WGSL  KEY_EXTRACT_WGSL          — depth-key extraction compute
                                     BITONIC_SORT_WGSL         — GPU bitonic merge sort compute
                                     ALPHA_VERTEX_WGSL         — alpha-blended particle vertex
                                     ALPHA_FRAGMENT_WGSL       — alpha-blended particle fragment
                                     GLOW_VERTEX_WGSL          — additive glow particle vertex
                                     GLOW_FRAGMENT_WGSL        — additive glow particle fragment

environment-fog.ts          6 WGSL  WGSL_UNIFORMS             — EnvFogUniforms struct
                                     WGSL_FULLSCREEN_VERT      — full-screen triangle vertex
                                     WGSL_DEPTH_FOG            — depth fog fragment
                                     WGSL_OCCLUSION            — occlusion extract fragment
                                     WGSL_GOD_RAYS             — god ray radial blur fragment
                                     WGSL_COMPOSITE            — fog-rays composite (Mie + ACES)

screen-space-reflections.ts 5 WGSL  WGSL_SSR_MATH             — SSR math helpers (reflection, hash)
                                     WGSL_HIZ_DOWNSAMPLE       — hi-z depth pyramid downsample compute
                                     WGSL_SSR_MARCH            — half-res hi-z ray march compute
                                     WGSL_SSR_RESOLVE          — temporal resolve + Fresnel fade
                                     WGSL_SSR_COMPOSITE        — energy-conserving reflection composite

ocean-background.ts         5 WGSL  WGSL_SNOISE               — 3D simplex noise helper
                                     WGSL_GERSTNER             — Gerstner wave displacement function
                                     OCEAN_MESH_SHADER         — ocean grid vertex + fragment (4-wave sum)
                                     SPLASH_COMPUTE_SHADER     — splash particle physics compute
                                     SPLASH_RENDER_SHADER      — splash particle instanced render

cell-material-system.ts     5 WGSL  WGSL_PATCH_ATTENTION      — attention species iridescent metallic fragment patch
                                     WGSL_PATCH_FFN            — ffn species glass refraction fragment patch
                                     WGSL_PATCH_LAYERNORM      — layernorm species matcap marble fragment patch
                                     WGSL_PATCH_EMBEDDING      — embedding species organic membrane fragment patch
                                     WGSL_PATCH_SOFTMAX        — softmax species luminous energy fragment patch

noise-flow-field.ts         4 WGSL  WGSL_SIMPLEX_BASIS        — simplex noise basis (lygia-inlined)
                                     WGSL_FBM                  — fractal Brownian motion accumulator
                                     WGSL_CURL                 — 2D curl operator on noise field
                                     NOISE_FORCE_SHADER        — particle force injection from curl field

edge-flow-renderer.ts       4 WGSL  GPU_UNIFORMS_WGSL         — edge-flow uniform struct
                                     GPU_QOS_WGSL              — QoS colour/speed/trail lookup table
                                     GPU_SPLINE_WGSL           — Catmull-Rom spline evaluation
                                     GPU_NOISE_WGSL            — simplex noise for flow jitter

at-shader-utils.ts          3 WGSL  WGSL_EASES                — 30 easing functions (quad/cubic/quart/quint/sine/expo/circ/back/elastic/bounce × In/Out/InOut)
                                     WGSL_RANGE                — range / crange / rangeMirror remapping (float + vec2/3/4)
                                     WGSL_BLEND_MODES          — 13 Photoshop blend modes (Add/Multiply/Screen/Overlay/SoftLight/…)

boids-compute.ts            3 WGSL  UNIFORMS_WGSL             — boids simulation uniform struct
                                     INFLUENCE_SHADER          — separation + alignment + cohesion accumulation
                                     INTEGRATE_SHADER          — velocity/position integration with speed clamping

ripple-effect.ts            3 WGSL  RIPPLE_PROPAGATE_SHADER   — wave equation propagation compute
                                     RIPPLE_STAMP_SHADER       — circular impulse stamp compute
                                     RIPPLE_COMPOSITE_SHADER   — ripple overlay composite render

reaction-diffusion.ts       2 WGSL  GS_COMPUTE_SHADER_PER_SPECIES — per-species Gray-Scott reaction-diffusion compute
                                     GS_COMPUTE_SHADER             — global Gray-Scott reaction-diffusion compute

turing-pattern.ts           2 WGSL  INIT_SHADER_SRC           — random initialisation compute
                                     STEP_SHADER_SRC           — Gray-Scott timestep compute

lattice-boltzmann-bg.ts     2 WGSL  LBM_SHADER                — D2Q9 Lattice Boltzmann collision + streaming compute
                                     SPH_LBM_COUPLE_SHADER     — SPH ↔ LBM two-way coupling compute

environment-fx.ts           2 WGSL  WGSL_COMPUTE              — brick-tile grid + voronoise + chroma compute
                                     WGSL_RENDER               — full-screen render (sample compute output)

curl-flow-field.ts          2 WGSL  WGSL_CURL_COMPUTE         — 3D curl-noise flow field advection compute
                                     WGSL_SPH_INJECT           — SPH velocity field injection into curl grid

particle-instancing.ts      2 GLSL  VERT_SRC                  — WebGL2 instanced soft-particle vertex shader
                                     FRAG_SRC                  — WebGL2 instanced soft-particle fragment shader (SDF circle + glow)

curl-aura.ts                2 GLSL  AURA_VERT                 — WebGL2 aura quad vertex (screen-aligned billboard)
                                     AURA_FRAG                 — WebGL2 aura fragment (concentric SDF rings + curl-noise distortion)

render-compositor.ts        1 WGSL  LUT_PASS_WGSL             — 3-D trilinear LUT colour grade final pass
at-render-pipeline.ts       1 WGSL  LUT_PASS_WGSL             — 3-D trilinear LUT colour grade (precursor variant)
post-process.ts             1 WGSL  POST_PROCESS_WGSL         — Kuwahara + Film Grain + Chromatic Aberration full-screen
atmosphere.ts               1 WGSL  ATMOSPHERE_WGSL           — Rayleigh scattering + Mie phase + depth fog + sky gradient
natural-patterns.ts         1 WGSL  COMPUTE_SHADER_SRC        — Voronoi + Worley + FBM natural texture compute
physarum-sim.ts             1 WGSL  WGSL_SIMPLEX2             — 2D simplex noise for Physarum agent heading perturbation
```

---

## QoS → Physics Mapping

Two mapping modules translate Apollo CyberRT QoS messaging semantics into
physical simulation parameters:

### qos-spatial-bridge.ts (Full Apollo Profiles)

8 named profiles sourced from `apollo/cyber/transport/qos/qos_profile_conf.cc`:

| Profile | Reliability | Durability | Depth | Priority | Use Case |
|---------|-------------|------------|-------|----------|----------|
| DEFAULT | RELIABLE | VOLATILE | 1 | 1 | General control messages |
| SENSOR_DATA | BEST_EFFORT | VOLATILE | 5 | 0 | Lidar / camera / radar streams |
| PARAMETERS | RELIABLE | TRANSIENT_LOCAL | 1000 | 2 | Parameter server |
| SERVICES_DEFAULT | RELIABLE | VOLATILE | 10 | 2 | RPC service calls |
| PARAM_EVENT | RELIABLE | TRANSIENT_LOCAL | 1000 | 2 | Parameter change events |
| SYSTEM_DEFAULT | RELIABLE | VOLATILE | 1 | 1 | System infrastructure |
| TF_STATIC | RELIABLE | TRANSIENT_LOCAL | 1 | 1 | Static coordinate transforms |
| TOPO_CHANGE | RELIABLE | TRANSIENT_LOCAL | 10 | 3 | Graph topology updates |

**Mapping formulas:**

| QoS Field | → Physical Parameter | Rule |
|-----------|---------------------|------|
| reliability | viscosity | RELIABLE → 0.02 (thick ordered flow), BEST_EFFORT → 0.001 (thin turbulent) |
| durability | boundary friction | TRANSIENT_LOCAL → 0.95 (sticky walls), VOLATILE → 0.30 (slippery) |
| history depth | trail length | `min(depth × 3, 30)` |
| mps | emitter rate | 0 (unlimited) → 120 particles/s, else `min(mps × 1.5, 120)` |
| priority (0–3) | force multiplier | `1.0 + priority × 0.5` (range 1.0–2.5) |

### qosSpatial.ts (Lightweight Presets)

5 presets used by `SPHWorld.ts` for GPU-path physics:

| QoS Field | → Physical Parameter | Rule |
|-----------|---------------------|------|
| reliability | boundary stiffness | RELIABLE → 50 000, BEST_EFFORT → 8 000 |
| mps | viscosity | `max(0.001, 0.1 / √mps)`, or 0.01 when mps = 0 |
| history depth | persistence | `depth × 0.5` |
| durability | rest density | TRANSIENT_LOCAL → 1 200, VOLATILE → 1 000 |

`interpolateConfigs(a, b, t)` provides smooth lerp transitions between any two
spatial configurations for animated QoS profile switching.

---

## Species Visual DNA

Every cell's visual identity is assembled from three layers, unified by
`species-visual-dna.ts` into a single `VisualDNA` struct:

### Layer 1 — Shader Stack (species-shader-registry.ts)

Static, declarative definition of the rendering pipeline per species:

| Dimension | Options |
|-----------|---------|
| SDF shape | flower, koch, julia, supershape, capsule, hexagon, star, roundbox, polygon |
| Material type | matcap, pbr, iridescence |
| Pattern shader | Filename reference in `src/lib/shaders/` |
| Bloom params | strength, radius, threshold, pulse amplitude, pulse frequency |
| Physics bindings | Maps density/velocity/pressure/vorticity/kinetic energy → visual targets |

### Layer 2 — Cell Material System (cell-material-system.ts)

5 Transformer-analogy species with full PBR/matcap params + WGSL fragment patches:

| Species | Material Identity | Visual Character |
|---------|------------------|------------------|
| `attention` | Iridescent metallic | Multi-head shimmer, thin-film interference, cyan-violet-gold |
| `ffn` | Glass refraction | Snell-law caustic flicker, depth-of-field haze |
| `layernorm` | Matcap marble | Smooth normalised surface, veined stone |
| `embedding` | Organic membrane | Translucent lipid bilayer, subsurface scatter |
| `softmax` | Luminous energy | HDR emissive glow, bloom-saturated hot-core |

### Layer 3 — Physics-Driven Modulation

At runtime, `physics-uniform-bridge.ts` samples the SPH neighbourhood around each
rigid body, producing density ratio, velocity magnitude, pressure, vorticity, and
kinetic energy.  These feed into:

- **species-shader-registry** physics bindings → bloom strength, SDF distortion,
  material fresnel, pattern speed/contrast
- **uil-species-live** → 2593+ AT UIL scene parameters modulated per-species
  per-frame
- **cell-visual-identity** → morphology archetype selection
  (jellyfish / petal / coral / mycelium / crystal)

### Cell Visual Identity Derivation Chain

```
species          → base morphology
QoS reliability  → border sharpness  (RELIABLE = crisp, BEST_EFFORT = fuzzy)
QoS mps          → internal flow speed
force_field      → decoration direction & intensity
collision contacts → contact spark/ripple intensity
```

---

## Quick Start

### Minimal simulation (CPU path)

```typescript
import { createWorld, addFluidBlock, stepWorld } from '$lib/sph';
import { qosToSpatial, APOLLO_PROFILES } from '$lib/sph';

const qos = qosToSpatial(APOLLO_PROFILES.SENSOR_DATA);
const world = createWorld(800, 600, qos);

addFluidBlock(world, 100, 100, 300, 200, 8, 'attention');

function loop() {
  stepWorld(world);
  // read world.particles, world.rigidBodies, etc.
  requestAnimationFrame(loop);
}
loop();
```

### ATRenderPipeline (auto WebGPU / Canvas2D)

```typescript
import { ATRenderPipeline } from '$lib/sph';

const pipe = await ATRenderPipeline.create(canvas, {
  preferredBackend: 'webgpu',
  particleCount: 65_536,
  substeps: 4,
});

pipe.addFluid(0.1, 0.05, 0.4, 0.4, 0.008, 0);
pipe.addObstacle(0.5, 0.5, 0.06);

requestAnimationFrame(function loop(t) {
  pipe.tick(t);
  requestAnimationFrame(loop);
});
```

### Species Visual DNA (per-frame per-cell)

```typescript
import { getVisualDNA, initVisualDNA } from '$lib/sph';

await initVisualDNA();

// inside render loop, per cell:
const dna = getVisualDNA('attention', world, bodyId);
// dna.shader   → SpeciesShaderConfig
// dna.bloom    → RuntimeBloom { strength, radius, threshold, ... }
// dna.material → RuntimeMaterial { fresnelStrength, iridThickness, ... }
// dna.pattern  → RuntimePattern { speed, contrast }
// dna.sdf      → RuntimeSdf { distort }
// dna.uilBag   → SpeciesUniformBag (2593+ AT params)
```

### Backend sync via epoch bridge

```typescript
import { SPHEpochBridge } from '$lib/sph';

const bridge = new SPHEpochBridge(sphWorld, {
  domainW: 3.0, domainH: 3.0,
  canvasW: canvas.width, canvasH: canvas.height,
});
bridge.start();
// Listens to SSE /api/cell-events, syncs rigid bodies automatically.
// On teardown:
bridge.stop();
```

### QoS profile switching

```typescript
import { setQoS } from '$lib/sph';

// Switch the worker-thread simulation to a new QoS profile:
await setQoS('TOPO_CHANGE');
// Physics parameters update immediately on the next step.
```
