# RESEARCH_86_molstar — Molstar WebGL Renderer Deep-Dive
> Researcher: xiaodi #86 | Branch: cell-pubsub-loop

Molstar (https://github.com/molstar/molstar) is the industry-standard publication-quality
molecular 3D viewer, powering PDBe and RCSB PDB.  This document maps its renderer
architecture to our own `src/lib/renderer/` gaps and flags what can be ported directly.

---

## 1. Renderer Architecture

```
mol-canvas3d/               ← Canvas / camera / controls layer
  passes/draw.ts            ← DrawPass: multi-pass pipeline (opaque, transparent,
                               post-process, WBOIT/DPOIT, bloom, DoF, SSAO, SMAA, FXAA)
  camera/                   ← Perspective + orthographic + stereo cameras
  controls/trackball.ts     ← Arcball orbit controls

mol-gl/                     ← Pure WebGL abstraction layer
  renderer.ts               ← Renderer: iterates Scene.Groups, issues render variants
                               (color | pick | depth | marking | emissive | tracing)
  scene.ts                  ← Scene + Scene.Group: holds GraphicsRenderable[],
                               manages commit-queue, computes aggregate bounding sphere,
                               sorts by program ID for minimal GL state changes
  render-object.ts          ← GraphicsRenderObject<T>: typed union
                               T ∈ {mesh, points, spheres, cylinders, text, lines,
                                    direct-volume, image, texture-mesh}
  renderable.ts             ← GraphicsRenderable: wraps RenderItem + state + materialId
  renderable/               ← One file per primitive type (spheres.ts, cylinders.ts …)
                               Each exports a *Values type (GPU buffer descriptors) and
                               a factory function that calls webgl/render-item.ts
  webgl/
    context.ts              ← WebGLContext wrapper (extensions, resource pools)
    program.ts              ← Shader program compilation + uniform cache
    render-item.ts          ← Low-level draw call (VAO bind, uniform upload, draw*)
    buffer.ts / texture.ts / framebuffer.ts …

mol-geo/                    ← CPU geometry & data layer (no WebGL)
  geometry/
    spheres/spheres.ts      ← Spheres data structure (center + group typed arrays)
    cylinders/cylinders.ts  ← Cylinders (start/end/cap/scale typed arrays)
    mesh/mesh.ts            ← Indexed triangle mesh
    geometry.ts             ← GeometryUtils interface (createValues, updateValues…)
    base.ts                 ← BaseGeometry: common params (alpha, quality, material)
  primitive/                ← Pure CPU mesh generators (no GPU state)
    icosahedron.ts          ← Base icosahedron (12 verts, 20 faces)
    sphere.ts               ← Sphere(detail) → Polyhedron subdivision of icosahedron
    cylinder.ts             ← Cylinder(props) → frustum/cone mesh
    polyhedron.ts           ← Subdivided polyhedron (midpoint subdivision)
```

### Key design insight: two-tier geometry

| Layer | Our equivalent | Molstar |
|-------|---------------|---------|
| CPU data | `Geometry` base class | `Spheres`, `Cylinders`, `Mesh` in `mol-geo/geometry/` |
| GPU renderable | `InstancedMesh` + `AstroRenderer` | `GraphicsRenderable` in `mol-gl/` |

---

## 2. How Molstar Draws Spheres — Impostor Shader (NOT mesh subdivision)

**TL;DR: spheres are drawn as quads with a ray-cast impostor shader, not as meshes.**

### Vertex stage (`mol-gl/shader/spheres.vert.ts`)

Each atom/sphere is a **2-triangle billboard quad** expanded in clip space to cover the
sphere's projected footprint.  Two projection algorithms are used:

- `sphereProjection()` — Mara/McGuire 2013 tight bounding rectangle for symmetric
  projections. Computes exact min/max screen extents analytically.
- `quadraticProjection()` — Sigg et al. GPU quadratic surface method, for
  asymmetric (XR) projections.

The center position is stored in a **texture** (`tPositionGroup`, dimensions `uTexDim`)
rather than a VBO, enabling very large atom counts without per-vertex data explosion.

### Fragment stage (`mol-gl/shader/spheres.frag.ts`)

`SphereImpostor(out vec3 modelPos, out vec3 cameraPos, out vec3 cameraNormal, …)`:

1. Reconstruct a ray from the camera through the current fragment.
2. Solve the quadratic `det = B² + r² - |d|²` for ray-sphere intersection.
3. If `det < 0` → `discard` (outside the sphere silhouette).
4. Write the **real depth** of the intersection point via `gl_FragDepth`, so spheres
   occlude geometry correctly despite being rendered as flat quads.
5. Compute the analytic normal at the hit point → Phong/PBR lighting.

**Result**: pixel-perfect spheres at any resolution, no tessellation artifacts,
essentially free LOD, and correct depth/picking.

### CPU data structure (`mol-geo/geometry/spheres/spheres.ts`)

```ts
interface Spheres {
  sphereCount: number;
  centerBuffer: ValueCell<Float32Array>;  // xyz per sphere
  groupBuffer:  ValueCell<Float32Array>;  // group id per sphere
  shaderData: {
    positionGroup: ValueCell<TextureImage<Float32Array>>; // packed into texture
    texDim: ValueCell<Vec2>;
    lodLevels: ValueCell<...>;   // distance-based detail switching
    sizeFactor: ValueCell<number>;
  }
}
```

---

## 3. How Molstar Draws Connections — Cylinder Impostor (NOT capsule mesh)

**TL;DR: bonds/sticks are also ray-cast impostors, not tessellated capsule meshes.**

### Vertex stage (`mol-gl/shader/cylinders.vert.ts`)

Each bond is an **oriented bounding-box billboard** (8 corners) aligned to the cylinder
axis, built entirely in the shader from `aStart`, `aEnd`, `aScale`, and per-vertex
`aMapping` (vec3 corner selector):

```glsl
vec3 left = cross(camDir, dir);   // screen-space left
vec3 up   = cross(left, dir);     // screen-space up
vModelPosition += aMapping.x * dir + aMapping.y * left + aMapping.z * up;
```

`aCap` flag controls flat disk end-caps.  `aColorMode` supports dual-color
interpolation for half-bond coloring.

### Fragment stage (`mol-gl/shader/cylinders.frag.ts`)

`CylinderImpostor(rayOrigin, rayDir, start, end, radius, …)`:

Adapted from Inigo Quilez's Shadertoy cylinder SDF (MIT):

```glsl
float k2 = baba - bard*bard;
float k1 = baba*dot(oc,rd) - baoc*bard;
float k0 = baba*dot(oc,oc) - baoc*baoc - r*r*baba;
float h  = k1*k1 - k2*k0;
if (h < 0.0) return false;   // miss
```

Handles body, top-cap (disk), and bottom-cap (disk) intersections separately.
Writes corrected `gl_FragDepth` for proper Z-ordering.

**Capsule appearance** comes from pairing a sphere impostor at each atom center with
cylinder impostors for bonds — they visually merge because atom sphere radius ≥ bond radius.

### CPU data structure (`mol-geo/geometry/cylinders/cylinders.ts`)

```ts
interface Cylinders {
  cylinderCount: number;
  mappingBuffer: ValueCell<Float32Array>;  // corner of billboard box
  indexBuffer:   ValueCell<Uint32Array>;
  groupBuffer:   ValueCell<Float32Array>;
  startBuffer:   ValueCell<Float32Array>;  // bond start xyz
  endBuffer:     ValueCell<Float32Array>;  // bond end xyz
  scaleBuffer:   ValueCell<Float32Array>;  // per-bond radius scale
  capBuffer:     ValueCell<Float32Array>;  // 0=none,1=top,2=bottom,3=both
  colorModeBuffer: ValueCell<Float32Array>;
}
```

---

## 4. Primitive Mesh Generators (`mol-geo/primitive/`) — CPU only, MIT licensed

Pure TypeScript returning `{ vertices, normals, indices }` typed arrays. No WebGL, no
state — safe to import anywhere.

| File | Our gap | Notes |
|------|---------|-------|
| `icosahedron.ts` | ❌ Missing | 12-vertex base icosahedron, golden-ratio |
| `sphere.ts` | SphereGeometry uses UV sphere | `Sphere(detail)`: detail=0→20 faces, 1→80, 2→320 |
| `polyhedron.ts` | ❌ Missing | Midpoint subdivision; uniform triangles |
| `cylinder.ts` | CylinderGeometry exists | Adds `topCap`/`bottomCap` flags; conical frustum |
| `torus.ts` | ❌ Missing | Useful for ring/ER membrane depictions |
| `box.ts` | BoxGeometry exists | Skip |

---

## 5. What to Port into `src/lib/renderer/geometry/`

### HIGH PRIORITY — Direct port (MIT licensed ✓)

#### A. `IcosahedronSphereGeometry.ts`

Port `icosahedron.ts` + `sphere.ts` + `polyhedron.ts` as a drop-in alternative to
our current `SphereGeometry` (UV sphere).

```ts
export class IcosahedronSphereGeometry extends Geometry {
  constructor(detail = 2) {
    super();
    const { vertices, normals, indices } = Sphere(detail); // ported from molstar
    this.setAttribute('position', new GeometryAttribute(vertices, 3));
    this.setAttribute('normal',   new GeometryAttribute(normals,  3));
    this.setIndex(indices);
  }
}
```

Advantages over our current UV sphere:
- No pole distortion / degenerate triangles at poles
- More uniform triangle area → better shading at low segment counts
- `detail=1` (80 tris) looks better than UV 16×8 (256 tris)

#### B. `TorusGeometry.ts`

Port `torus.ts` for annular membrane / endoplasmic reticulum cross-sections.

### MEDIUM PRIORITY — Sphere Impostor Shader (high impact for cell viewer)

For rendering hundreds of organelles at interactive framerates:

1. Add `src/lib/renderer/geometry/SpheresImpostorGeometry.ts` that packs
   `centerBuffer` → texture image.
2. Port `spheres.vert.ts` + `spheres.frag.ts` → GLSL files under
   `src/lib/renderer/assets/shaders/`.
3. Add `SpheresRenderable` using `AstroProgram`.

Performance win: 10–100× vs mesh spheres for >1000 instances with correct depth/picking.

### MEDIUM PRIORITY — Cylinder Impostor Shader

Same pattern for filament/membrane tube rendering.

---

## 6. Architecture Comparison Table

| Concept | Molstar | Our Renderer |
|---------|---------|-------------|
| Sphere primitive | Ray-cast impostor quad | UV sphere mesh (SphereGeometry) |
| Cylinder primitive | Ray-cast impostor billboard | Tessellated mesh (CylinderGeometry) |
| Icosphere | `Sphere(detail)` via polyhedron subdivision | ❌ Missing |
| Scene graph | `Scene → Scene.Group[] → GraphicsRenderable[]` | `Scene → InstancedMesh[]` |
| Render variants | color / pick / depth / marking / emissive / tracing | color only |
| Transparency | WBOIT + DPOIT + blended | needs work |
| Post-processing | SSAO, bloom, DoF, FXAA, SMAA, shadow, outline | NukePass (custom) |
| LOD | `lodLevels` ValueCell per geometry | ❌ Missing |
| GPU instancing | per-instance transform texture | InstancedMesh (our impl) |
| Picking | `renderPick` variant → object/instance/group id | OcclusionQuery only |

---

## 7. Immediate Next Actions for #86

1. **Port `IcosahedronSphereGeometry`** (`icosahedron.ts` + `sphere.ts` + `polyhedron.ts`)
   → `src/lib/renderer/geometry/IcosahedronSphereGeometry.ts`
   → update `src/lib/renderer/geometry/index.ts` exports

2. **Spike sphere impostor shader** — `SphereImpostorGeometry` + matching GLSL in
   `src/lib/renderer/`, wire into `CellInstanceManager` for organelle rendering.

3. **Add `TorusGeometry`** for ring-shaped membrane / ER cross-sections.

4. **File ticket**: implement picking render variant (port `renderPick` from `renderer.ts`)
   so clicking on cells works in 3D.

---

## References

- `upstream/molstar/src/mol-gl/shader/spheres.{vert,frag}.ts` — impostor sphere GLSL
- `upstream/molstar/src/mol-gl/shader/cylinders.{vert,frag}.ts` — impostor cylinder GLSL
- `upstream/molstar/src/mol-geo/primitive/` — MIT-licensed CPU mesh generators
- Mara & McGuire 2013 — tight projected sphere bounding rectangle
- Sigg et al. — GPU quadratic surface projection
- Inigo Quilez — cylinder SDF (https://www.shadertoy.com/view/4lcSRn)
