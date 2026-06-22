# src/lib/sph — SPH Physics & Rendering Engine

Real-time 2-D Smoothed Particle Hydrodynamics engine with a full WebGPU rendering
pipeline.  The system maps Transformer-analogy cell species onto fluid-physical
bodies, driven by Apollo CyberRT QoS profiles that translate messaging semantics
(reliability, durability, history depth, message rate) into physical parameters
(viscosity, boundary friction, trail length, emitter rate).

```
cell-pubsub-loop  ──  SSE epoch stream
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  sph-epoch-bridge          Backend ↔ Frontend sync      │
│  cell-body-bridge          Cell registry → RigidBody    │
│  qos-spatial-bridge        QoS profile → SpatialPhysics │
└────────────┬────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────┐
│  SIMULATION CORE                                        │
│  world-stepper   ←  dfsph-solver, sph-kernels,         │
│                     fluid-rigid-coupling, rigid-body,   │
│                     world-boundary, spatial-hash         │
│  collision/*     ←  BVH, GJK/EPA, SAT, impulse solver  │
└────────────┬────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────┐
│  AT RENDER PIPELINE                                     │
│  at-scene-compositor  (9-pass forward pipeline)         │
│    ① SPH readback                                       │
│    ② Navier-Stokes fluid step                           │
│    ③ Flower + Spline particle compute                   │
│    ④ Particle sort + composite                          │
│    ⑤ PBR / Matcap material pass                         │
│    ⑥ Water surface                                      │
│    ⑦ Volumetric light (god rays)                        │
│    ⑧ Bloom post-process                                 │
│    ⑨ Final composite → canvas                           │
└────────────┬────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────┐
│  VISUAL IDENTITY                                        │
│  species-visual-dna     Complete per-cell render config  │
│  species-shader-registry  SDF + material + pattern stack │
│  cell-material-system   Per-species PBR/matcap + WGSL   │
│  cell-visual-identity   Physics → morphology derivation  │
└─────────────────────────────────────────────────────────┘
```

---

## Module Catalog (95 files, ~52 700 lines)

### Simulation Core

| Module | Description |
|--------|-------------|
| `world-stepper.ts` | Main simulation loop — orchestrates DFSPH pressure solve, rigid body integration, fluid-rigid coupling, boundary enforcement, and emitter spawning per substep. Entry point: `createWorld()` → `stepWorld()`. |
| `dfsph-solver.ts` | Divergence-Free SPH pressure solver. Iterative density/divergence correction (Bender & Koschier 2015) on CPU-side `Particle[]`. |
| `sph-kernels.ts` | SPH kernel math library — Cubic spline, Spiky gradient, Poly6, viscosity Laplacian. Includes `selfTest()` for kernel normalisation validation. |
| `spatial-hash.ts` | CPU spatial hash grid for neighbour queries. `buildSpatialHash()` → `queryNeighbors()`. O(1) cell lookup, tuneable cell size. |
| `rigid-body.ts` | Rigid body state (position, velocity, angle, angular velocity, inverse mass/inertia) with Euler integration, impulse application, and boundary particle sampling. |
| `fluid-rigid-coupling.ts` | Two-way SPH ↔ rigid body coupling — boundary volume computation, density contribution from rigid walls, pressure/viscosity coupling forces, momentum transfer. |
| `world-boundary.ts` | Domain boundary definitions (rect, circle, polygon) and enforcement — wall particle generation, clamping, reflection. Configurable `BoundaryShape` union type. |
| `emitter-strategy.ts` | QoS-driven particle emitter patterns — `ContinuousPattern`, `HighFreqStreamPattern`, `LowFreqPulsePattern`, `ConstantFieldPattern`, `BurstWavePattern`. Maps Apollo QoS profiles to emission behaviour via `patternForProfile()`. |
| `performance-budget.ts` | 4-tier adaptive performance budget (ULTRA / HIGH / MEDIUM / LOW). Auto-downgrades substeps, particle caps, and render resolution based on measured frame time. |
| `types.ts` | Shared interfaces — `GPUBufferSet`, `SimParams`, `ParticleData`, `QoSProfile`, `SpatialConfig`, `RigidBody`, `ContactConstraint`. Constants: `MAX_PARTICLES` (50 000), `WORKGROUP_SIZE` (256). |

### Collision Subsystem (`collision/`)

| Module | Description |
|--------|-------------|
| `collision-world.ts` | Top-level collision world — broad phase (Sort-and-Sweep) → narrow phase (GJK/EPA) → contact generation → impulse solve → position correction. |
| `sort-and-sweep.ts` | Sweep-and-Prune broad phase on AABB projections. |
| `aabb-manager.ts` | AABB computation, expansion, merge, area, ray-cast, containment tests. |
| `bvh-tree.ts` | Bounding Volume Hierarchy for spatial acceleration — surface-area heuristic insertion, refit, ray-cast, overlap queries. |
| `gjk-epa.ts` | GJK collision detection + EPA penetration depth for convex shapes (`createBoxShape`, `createCircleShape`). |
| `sat-solver.ts` | Separating Axis Theorem for OBB-OBB overlap tests. |
| `contact-manifold.ts` | Contact point generation, warm starting, combined friction/restitution coefficients. |
| `impulse-solver.ts` | Sequential impulse constraint solver — configurable iteration count, Baumgarte stabilisation. |
| `constraints.ts` | Constraint types — `NonPenetrationConstraint`, `FrictionConstraint`, `RestitutionConstraint`. |
| `scene-query.ts` | Ray-cast, overlap, and closest-point queries against the collision world. |
| `CollisionEvents.ts` | Event dispatcher — `BEGIN`, `STAY`, `END` contact phases, pair caching, callback registration. |
| `ContactSolver.ts` | Class-based impulse contact resolution (PascalCase variant). |
| `PositionSolver.ts` | Baumgarte position correction pass. |
| `AABB.ts` | Standalone AABB utilities — overlap, union, perimeter, expand, fromCircle, fromPoints, contains, center. |
| `CollisionWorld.ts` | Monolithic collision world (PascalCase variant) with `createCircleBody`, `createBoxBody`, `computeContactInfo`. |
| `BVHTree.ts` | BVH tree (PascalCase variant). |
| `EPA.ts` | Standalone EPA on raw `Vec2[]` arrays. |
| `GJK.ts` | Class-based `Circle` / `Polygon` shapes with `collide()`. |
| `SAT.ts` | SAT (PascalCase variant). |
| `SceneQuery.ts` | Full-featured scene query with shape union (`CircleShape`, `AABBShape`, `CapsuleShape`, `ConvexPolygonShape`). |
| `SortAndSweep.ts` | Sort-and-Sweep (PascalCase variant). |
| `index.ts` | Barrel re-export for the entire collision subsystem. |

### AT Render Pipeline

| Module | Description |
|--------|-------------|
| `ATRenderPipeline.ts` | Unified render pipeline facade — probes WebGPU at construction, auto-falls back to Canvas2D. Single API surface (`addFluid`, `addObstacle`, `step`, `render`, `tick`). |
| `at-render-pipeline.ts` | Pipeline orchestrator — chains all AT rendering modules in a fixed-function forward pipeline, FBO pass-through between stages. |
| `at-scene-compositor.ts` | 9-pass scene compositor — SPH readback → Navier-Stokes → particle compute → sort/composite → PBR material → water surface → volumetric light → bloom → final composite. Manages per-cell material instances via species-shader-registry + cell-material-system. |
| `at-flower-particle.ts` | AT FlowerParticleShader WebGPU/WGSL port (M710) — petal-shaped particle lifecycle, edge-route-to-flower-spline conversion. |
| `at-spline-particle.ts` | AT SplineParticleLife WebGPU/WGSL port (M713) — Bézier spline particle instances, canvas-route-to-edge-spline conversion, lifecycle presets. |
| `at-navier-stokes.ts` | AT Navier-Stokes fluid compute WGSL port (M715) — grid-based incompressible fluid with mouse/touch splat injection, dye advection. |
| `at-pbr-material.ts` | AT PBR lighting system WGSL port — Cook-Torrance BRDF (GGX/Smith), matcap Fresnel, uniform packing for per-cell material instances. |
| `at-bloom-postprocess.ts` | AT UnrealBloom post-process WGSL port (M714) — bright extract → separable Gaussian blur → additive composite. Per-species bloom strength via `createATBloomForSpecies()`. |
| `at-volumetric-light.ts` | AT VolumetricLight WGSL port (M716) — screen-space god rays via occlusion → radial blur → Mie scatter. |
| `at-water-surface.ts` | AT WaterCeilingShader WebGPU/WGSL port (M711) — wave simulation, mesh render, water-particle overlay. |
| `at-shader-utils.ts` | AT shader utility library WGSL port — easing functions, range mapping, blend modes. Provides `WGSL_EASES`, `WGSL_RANGE`, `WGSL_BLEND_MODES` string constants for shader composition. |
| `particle-compositor.ts` | Unified particle render compositor (M718) — depth-sorts flower + spline particles across all active renderers, draws alpha and additive-glow layers. |
| `particle-instancing.ts` | GPU instanced particle rendering (M745) — replaces per-particle Canvas2D draw with single-draw-call WebGL2 instanced pipeline. |
| `post-process.ts` | Full-screen post-process pipeline — three art styles: Kuwahara (oil painting), Film Grain, Chromatic Aberration. Independently stackable. |
| `atmosphere.ts` | Atmospheric scattering + fog post-process — Rayleigh scattering phase function, depth fog, sky gradient. Includes `ATMOSPHERE_PRESETS`. |

### GPU Infrastructure

| Module | Description |
|--------|-------------|
| `SPHWorld.ts` | WebGPU SPH world — manages GPU buffers, compute pipelines, and render passes for the full particle simulation. |
| `SPHGPUOrchestrator.ts` | GPU compute orchestrator — dispatches density, pressure, force, and integration compute shaders per substep. Manages buffer ping-ponging and readback. |
| `ParticleRenderer.ts` | GPU particle renderer — instanced circle drawing with per-species colour, glow, and size. |
| `SpatialHashGrid.ts` | GPU-accelerated spatial hash grid for neighbour search in compute shaders. |
| `NeighborListBuilder.ts` | Builds CSR (Compressed Sparse Row) neighbour lists from spatial hash results. |
| `BoundaryModel.ts` | GPU boundary model — wall particle buffers, boundary density contribution in compute shaders. |
| `sph-bridge.ts` | Comlink Web Worker bridge — `initSPHWorld`, `addFluid`, `addBody`, `stepSPH`, `setQoS`, `raycast`, `terminateSPHWorker`. Offloads simulation to a background thread. |
| `sph-worker.ts` | Web Worker entry point — receives Comlink-proxied calls, runs the simulation loop off-main-thread. |

### QoS Mapping

| Module | Description |
|--------|-------------|
| `qos-spatial-bridge.ts` | Apollo CyberRT QoS → SPH physics mapping. 8 named profiles (DEFAULT, SENSOR_DATA, PARAMETERS, SERVICES_DEFAULT, PARAM_EVENT, SYSTEM_DEFAULT, TF_STATIC, TOPO_CHANGE). Maps reliability → viscosity, durability → boundary friction, history depth → trail length, mps → emitter rate, priority → force multiplier. |
| `qosSpatial.ts` | Lightweight QoS preset → SpatialConfig mapping. 5 presets (DEFAULT, SENSOR_DATA, PARAMETERS, TF_STATIC, TOPO_CHANGE). Maps reliability → boundary stiffness, mps → viscosity, history depth → persistence, durability → rest density. Also provides `interpolateConfigs()` for smooth QoS transitions. |
| `qos-spatial-bridge.ts` | Also exports `APOLLO_PROFILES` (all 8 entries) and `PROFILE_DESCRIPTIONS` for UI tooltip/legend rendering. |

### Species Visual DNA

| Module | Description |
|--------|-------------|
| `species-visual-dna.ts` | Single-call facade that concatenates shader registry + physics sampling + UIL modulation into one `VisualDNA` struct per cell per frame. The complete render-time visual config — SDF, material, pattern, bloom, physics uniforms, UIL bag — in a single lookup. |
| `species-shader-registry.ts` | Declarative shader stack registry — per-species `SpeciesShaderConfig` defining SDF shape (flower/koch/julia/supershape/capsule/hexagon/star/roundbox/polygon), material type (matcap/pbr/iridescence), pattern shader, bloom params, and physics bindings. |
| `cell-material-system.ts` | Per-species material system (M719) — 5 Transformer-analogy species with distinct visual identities: attention → iridescent metallic, ffn → glass refraction, layernorm → matcap marble, embedding → organic membrane, softmax → luminous energy. Each carries PBR/matcap params + WGSL fragment patches + physics modulators. |
| `cell-visual-identity.ts` | Physics → visual derivation (M731). Maps species → morphology (jellyfish/petal/coral/mycelium/crystal), QoS reliability → border sharpness, QoS mps → internal flow speed, force field → decoration direction, contacts → spark intensity. Output: `VisualProfile` data bag. |
| `physics-uniform-bridge.ts` | Bridges live SPH world state to per-cell shader uniforms — samples density, velocity, pressure, vorticity, kinetic energy from the SPH neighbourhood around a rigid body. |
| `uil-species-live.ts` | AT UIL params × SPH physics live interpolation — 2593+ AT scene parameters per species, modulated by physics state in real time. |
| `organic-sdf.ts` | Inlined SDF primitives for organic cell outlines — `flowerSDF`, `kochSDF`, `juliaSDF`, plus species-parameterised `organicOutline()`. |

### Color & Lighting

| Module | Description |
|--------|-------------|
| `chromatic-adaptation.ts` | Chromatic adaptation colour system (M584) — particle colours shift along natural colour gradients based on physics state (density, velocity, temperature). Batch-resolve, CSS/glow string conversion. |
| `color-palette.ts` | Dynamic colour palette per QoS profile (M566) — lygia blend modes (Screen, Overlay) ported to TypeScript. `resolveParticleColor()`, `rgbaToCss()`, `rgbaToU8()`. |
| `lut-generator.ts` | 3-D LUT colour grading generator (M624) — generates 17³ cube LUT textures, QoS-zone-weighted tone mapping. `classifyQoSZone()`, `classifyQoSProfileName()`. |

### Procedural Patterns & Morphogenesis

| Module | Description |
|--------|-------------|
| `reaction-diffusion.ts` | Gray-Scott reaction-diffusion WebGPU compute pipeline. Per-species feed/kill parameters via `GrayScottSpecies`. |
| `turing-pattern.ts` | Gray-Scott Turing pattern generator using WebGPU compute. Species-specific pattern modes. |
| `morphogenesis.ts` | L-system plant-growth morphogenesis — graph edges become living branches via Lindenmayer systems; particles stream along resulting Bézier paths. Named presets via `fromPreset()`. |
| `differential-growth.ts` | Differential growth simulation — organic fractal folds (coral reefs, cerebral cortex, romanesco). |
| `natural-patterns.ts` | Natural cell-surface textures via WebGPU compute — Voronoi + Worley noise (lygia port). Per-species FBM parameters. |
| `phyllotaxis.ts` | Fibonacci golden-angle phyllotaxis spiral — sunflower / pinecone packing pattern generation. |
| `physarum-sim.ts` | GPU Physarum polycephalum slime-mould simulation (Jones 2010). Agent-based trail deposition on a diffusion grid. |

### Flow Fields & Fluid Backgrounds

| Module | Description |
|--------|-------------|
| `curl-flow-field.ts` | WebGPU 3-D curl-noise flow field (M606). |
| `noise-flow-field.ts` | Curl-noise + FBM flow field overlay for SPH particles (lygia inlined WGSL). |
| `boids-compute.ts` | WebGPU Boids — separation + alignment + cohesion in 3 compute passes. Host uploads once; GPU advances per `tick()`. |
| `flowmap-bridge.ts` | SPH velocity field → OGL Flowmap texture bridge (M573) — rasterises particle velocities to a low-res Float32 RG grid with dissipation. |
| `ogl-flowmap-bridge.ts` | OGL Flowmap bridge (M614) — ports upstream Flowmap.js ping-pong to pure TypeScript/CPU, replaces mouse input with SPH velocity field. |
| `ocean-background.ts` | Gerstner wave ocean background — 4-wave sum displacement on a subdivided grid mesh. |
| `ocean-bridge.ts` | WebGPU-Ocean ↔ cell-pubsub-loop bridge (M603) — wraps upstream SPH + MLSMPM simulators and FluidRenderer. |
| `lattice-boltzmann-bg.ts` | Dual-layer fluid — Lattice Boltzmann Method macro background flow + SPH particle overlay. |

### Visual Effects

| Module | Description |
|--------|-------------|
| `collision-fx-system.ts` | Collision FX flower burst (M741) — spawns flower-shaped particle bursts at contact points with impulse-scaled intensity. |
| `contact-sparks.ts` | Contact spark system (M587) — short-lived directional spark particles at collision contacts. |
| `ripple-effect.ts` | Ripple collision effect — WebGPU ping-pong wave simulation at impact points. |
| `water-caustics.ts` | CPU port of WebGL water caustics — drop, stepSimulation, updateNormals (M613). |
| `environment-fx.ts` | "Bio-laboratory" atmospheric effects — three layered FX for a cyberpunk lab atmosphere. Includes presets. |
| `debug-renderer.ts` | Collision debug overlay — AABB wireframes, contact normals, BVH hierarchy, force field arrows, easing animations (lygia port). |
| `world-renderer.ts` | Canvas2D world renderer — species-coloured particles, rigid body outlines, contact points, BVH debug. Used by debug views and the Canvas2D fallback path. |

### Bridges & Integration

| Module | Description |
|--------|-------------|
| `sph-epoch-bridge.ts` | Backend epoch loop ↔ frontend SPHWorld bridge (M514). Listens to SSE `/api/cell-events`, syncs rigid bodies on `epoch_completed`, `cell_params_updated`, `topology_updated`. Exponential backoff reconnect. |
| `cell-body-bridge.ts` | Cell registry → RigidBody bridge. Maps 7 Transformer cells from `/api/cells` into SPH rigid body parameters using `species_physics.json`. |
| `audio-physics-bridge.ts` | Audio ↔ physics bridge (M748) — bidirectional mapping between SPH state and audio synthesis parameters. |

### Utilities

| Module | Description |
|--------|-------------|
| `spline-particle-life.ts` | Spline particle life system — Bézier spline-bound particles with lifecycle phases, edge-route conversion. |
| `integration-test.ts` | End-to-end integration test harness for the SPH pipeline. |
| `index.ts` | Barrel re-export for the entire `sph/` module. |

---

## AT Render Pipeline (9-Pass Detail)

The `at-scene-compositor.ts` orchestrates a fixed-function forward pipeline each frame:

**Pass 1 — SPH World Readback.** Read particle positions and velocities from
`SPHWorld` CPU buffers.  Derive per-cell bounding boxes, densities, and flow
vectors that drive physics-modulated material parameters downstream.

**Pass 2 — Navier-Stokes Fluid.** Advance the `at-navier-stokes` grid by one
timestep.  Inject mouse/touch splats and cell-centre dye impulses computed from
SPH velocity field.

**Pass 3 — Particle Compute.** Dispatch `at-flower-particle` and `at-spline-particle`
lifecycle compute passes.  Both renderers advance their ping-pong position textures
(`tPos0` ↔ `tPos1`).

**Pass 4 — Particle Sort & Composite.** `particle-compositor` depth-sorts all
active particles from both renderers, then draws alpha-blended and additive-glow
layers to the intermediate FBO.

**Pass 5 — PBR Material.** Per-cell `at-pbr-material` (or `ATMatcapFresnel`) render
pass.  Species-specific Cook-Torrance BRDF parameters are resolved through
`cell-material-system` + `species-shader-registry` for each registered cell.

**Pass 6 — Water Surface.** `at-water-surface` wave simulation, mesh render, and
water-particle overlay composited into the scene.

**Pass 7 — Volumetric Light.** `at-volumetric-light` screen-space god rays:
occlusion pass → radial blur → Mie scatter composite.

**Pass 8 — Bloom Post-Process.** `at-bloom-postprocess` UE5-style pipeline:
bright extract → separable Gaussian blur → additive composite.  Per-species
bloom strength set by `species-shader-registry` and modulated by
`physics-uniform-bridge` density ratio.

**Pass 9 — Final Composite.** Bloom output presented to the swap-chain surface
(canvas).  The `ATRenderPipeline` facade manages WebGPU ↔ Canvas2D fallback
selection transparently.

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
| Physics bindings | Maps density/velocity/pressure/vorticity/kinetic energy → visual targets (bloom, SDF distort, material fresnel, pattern speed, etc.) |

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

- **species-shader-registry** physics bindings → bloom strength, SDF distortion, material fresnel, pattern speed/contrast
- **uil-species-live** → 2593+ AT UIL scene parameters modulated per-species per-frame
- **cell-visual-identity** → morphology archetype selection (jellyfish / petal / coral / mycelium / crystal)

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
