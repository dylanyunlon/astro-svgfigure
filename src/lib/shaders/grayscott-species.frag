#version 300 es
precision mediump float;
out vec4 fragColor;

// ── grayscott-species.frag ────────────────────────────────────────────────────
// Gray-Scott Reaction-Diffusion Turing Pattern — species surface shader.
//
// ⚠️  SUPERSEDED BY WebGPU COMPUTE PIPELINE (M601)
//     This fragment shader remains as a WebGL 1 / fallback path.
//     For WebGPU targets use src/lib/sph/reaction-diffusion.ts instead:
//       • ReactionDiffusionSim — full GPU ping-pong compute
//       • parameterSpace(name) — Munafo/Pearson canonical (f, k) lookup
//
// Generates procedural Turing patterns (spots / stripes / worms / mitosis)
// directly on each cell's surface without a ping-pong texture.  We simulate
// several RD iterations per fragment using spatially-in virtual "pixels"
// derived from the cell's bbox, then composite the chemical concentration over
// the cell's fill colour.
//
// Each species carries its own (f, k) parameter pair that selects a different
// region of the Gray-Scott phase diagram:
//
//   cil-eye    (0)  f=0.0545 k=0.0620  → coral / spots     (瞳孔 spot pattern)
//   cil-bolt   (1)  f=0.0180 k=0.0510  → waves / maze      (闪电 wave pattern)
//   cil-vector (2)  f=0.0290 k=0.0570  → worms / filaments  (方向流 worm pattern)
//   cil-plus   (3)  f=0.0367 k=0.0649  → mitosis / pearls  (细胞分裂 pattern)
//   species 4-9     fall back to the u_feedKill override from the host.
//
// Physics coupling:
//   u_density   → scales diffusion rates (high density ⟹ denser pattern)
//   u_velocity  → stretches UV along velocity direction (flow distortion)
//
// GLSL #include dependencies (resolved by the project GLSL preprocessor):
//   ../../upstream/lygia/simulate/grayscott.glsl   — GS reaction step
//   ../../upstream/lygia/generative/fbm.glsl       — fBm seeding
//   ../../upstream/lygia/sdf/circleSDF.glsl        — boundary mask
//
// References:
//   P. Gonzalez Vivo — lygia.xyz
//   Pearson, J.E. (1993) — Complex Patterns in a Simple System, Science 261
//   Munafo, R. — mrob.com/pub/comp/xmorphia
//   Karl Sims — karlsims.com/rd.html  (coral f=0.0545,k=0.062; mitosis f=0.0367,k=0.0649)
//   Shaders: M550 (fragment) → M601 (WebGPU compute) — cell-pubsub-loop branch
// ─────────────────────────────────────────────────────────────────────────────

// ── lygia imports ─────────────────────────────────────────────────────────────
#include "../../upstream/lygia/simulate/grayscott.glsl"
#include "../../upstream/lygia/generative/fbm.glsl"
#include "../../upstream/lygia/sdf/circleSDF.glsl"

// ── uniforms ──────────────────────────────────────────────────────────────────
uniform vec4  u_bbox;       // (x, y, width, height) in canvas coords
uniform int   u_species;    // 0-9 species index → selects (f,k) preset
uniform float u_density;    // SPH density [0,1] — scales diffusion rate
uniform vec2  u_velocity;   // SPH velocity (vx,vy) — distorts UV field
uniform float u_time;       // seconds
uniform vec2  u_feedKill;   // (f, k) override for species 4-9

// ── per-species (f, k) lookup ─────────────────────────────────────────────────
//
// Gray-Scott phase diagram regions (Pearson 1993 / Munafo):
//   f=0.055 k=0.062  → spots  (coral / cil-eye)
//   f=0.018 k=0.051  → waves  (maze  / cil-bolt)
//   f=0.029 k=0.057  → worms  (filaments / cil-vector)
//   f=0.025 k=0.060  → mitosis (pearls / cil-plus)
//
// Implemented as a branchless lerp-chain instead of a switch (WebGL 1 compat).

vec2 speciesFeedKill(int idx) {
    // species 0 — cil-eye     coral/spots   f=0.0545 k=0.0620  (Karl Sims / Munafo κ)
    vec2 fk0 = vec2(0.0545, 0.0620);
    // species 1 — cil-bolt    maze/waves    f=0.0180 k=0.0510  (Munafo γ / maze)
    vec2 fk1 = vec2(0.0180, 0.0510);
    // species 2 — cil-vector  worms/filaments f=0.0290 k=0.0570 (Pearson δ labyrinth)
    vec2 fk2 = vec2(0.0290, 0.0570);
    // species 3 — cil-plus    mitosis/pearls  f=0.0367 k=0.0649 (Karl Sims / Munafo λ)
    vec2 fk3 = vec2(0.0367, 0.0649);

    // branchless index selection (WebGL 1 / mediump safe)
    float i = float(idx);
    vec2 fk = fk0;
    fk = mix(fk, fk1, step(0.5, i - 0.0) * step(i - 0.0, 0.9));  // idx==1
    fk = mix(fk, fk2, step(0.5, i - 1.0) * step(i - 1.0, 0.9));  // idx==2
    fk = mix(fk, fk3, step(0.5, i - 2.0) * step(i - 2.0, 0.9));  // idx==3
    // species 4-9: use host-supplied override
    fk = mix(fk, u_feedKill, step(4.0, i));
    return fk;
}

// ── helper: hash-based pseudo-random ─────────────────────────────────────────
float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// ── RD virtual iteration ──────────────────────────────────────────────────────
//
// Because we have no ping-pong textures at this stage, we emulate several
// Gray-Scott time steps by sampling an fBm "landscape" to initialise the UV
// concentrations and then running the Gray-Scott laplacian kernel analytically
// over the fBm gradients.  This produces spatially-in frozen-time Turing
// patterns without GPU-side state.
//
// The technique:
//   1.  Use fBm as the "u" (activator) initial condition.
//   2.  Derive "v" (inhibitor) as 1-u perturbed by hash noise.
//   3.  Run N Gray-Scott iterations using the laplacian of fBm at the sample
//       point (approximated by the 3×3 stencil over a sub-pixel grid).
//   4.  Accumulate the resulting chemical concentration.
//
// This is a recognised technique for static Turing-pattern generation in
// fragment shaders (see: iq, Inigo Quilez / Shadertoy community).

#define GS_ITER 6       // RD iterations per fragment (mediump budget)
#define PIXEL_SCALE 2.5 // virtual pixel size relative to bbox

vec2 rdEvolve(vec2 uv, vec2 fk, float diffScale) {
    // Virtual pixel size in UV space
    vec2 px = vec2(PIXEL_SCALE) / u_bbox.zw;

    // fBm-seeded initial concentrations
    float u0 = fbm(uv * 6.0 + u_time * 0.04);
    float v0 = 1.0 - fbm(uv * 6.0 + vec2(5.3, 1.7) + u_time * 0.03);

    // Add seeding noise so pattern erupts from heterogeneous IC
    float seed = hash21(floor(uv * 24.0) + vec2(u_time * 0.01));
    v0 = clamp(v0 + step(0.92, seed) * 0.3, 0.0, 1.0);

    float f = fk.x;
    float k = fk.y;

    // diffusion rates scaled by SPH density
    // high density → tighter (faster diffusing) → denser pattern
    float densityScale = 0.7 + u_density * 0.6;
    float diffU = 0.25 * diffScale * densityScale;
    float diffV = 0.05 * diffScale * densityScale;

    float u = u0;
    float v = v0;

    // Unrolled GS_ITER iterations using manually computed Laplacian from
    // 3×3 fBm stencil (avoids texture dependency).
    for (int i = 0; i < GS_ITER; i++) {
        float fi = float(i);

        // Sample fBm at 3×3 neighbours for Laplacian (Pearson kernel weights)
        // Corner weight: 0.707106781  Edge weight: 1.0  Centre: -6.828427
        float uLap = 0.0;
        float vLap = 0.0;

        // The fBm acts as a surrogate state texture — we sample it at shifted
        // UVs to build the discrete Laplacian.  The pattern "freezes" because
        // time only enters through u_time which drifts the fBm slowly.
        float phase = fi * 0.17 + u_time * 0.025;

        float centre_u = fbm(uv * 6.0 + vec2(phase));
        float centre_v = 1.0 - fbm(uv * 6.0 + vec2(5.3, 1.7 + phase));

        // 4 edge neighbours (weight 1.0)
        uLap += fbm((uv + vec2( px.x, 0.0)) * 6.0 + vec2(phase));
        uLap += fbm((uv + vec2(-px.x, 0.0)) * 6.0 + vec2(phase));
        uLap += fbm((uv + vec2(0.0,  px.y)) * 6.0 + vec2(phase));
        uLap += fbm((uv + vec2(0.0, -px.y)) * 6.0 + vec2(phase));

        vLap += 1.0 - fbm((uv + vec2( px.x, 0.0)) * 6.0 + vec2(5.3, 1.7 + phase));
        vLap += 1.0 - fbm((uv + vec2(-px.x, 0.0)) * 6.0 + vec2(5.3, 1.7 + phase));
        vLap += 1.0 - fbm((uv + vec2(0.0,  px.y)) * 6.0 + vec2(5.3, 1.7 + phase));
        vLap += 1.0 - fbm((uv + vec2(0.0, -px.y)) * 6.0 + vec2(5.3, 1.7 + phase));

        // 4 corner neighbours (weight 0.707106781)
        float cw = 0.707106781;
        uLap += cw * fbm((uv + vec2( px.x,  px.y)) * 6.0 + vec2(phase));
        uLap += cw * fbm((uv + vec2(-px.x,  px.y)) * 6.0 + vec2(phase));
        uLap += cw * fbm((uv + vec2( px.x, -px.y)) * 6.0 + vec2(phase));
        uLap += cw * fbm((uv + vec2(-px.x, -px.y)) * 6.0 + vec2(phase));

        vLap += cw * (1.0 - fbm((uv + vec2( px.x,  px.y)) * 6.0 + vec2(5.3, 1.7 + phase)));
        vLap += cw * (1.0 - fbm((uv + vec2(-px.x,  px.y)) * 6.0 + vec2(5.3, 1.7 + phase)));
        vLap += cw * (1.0 - fbm((uv + vec2( px.x, -px.y)) * 6.0 + vec2(5.3, 1.7 + phase)));
        vLap += cw * (1.0 - fbm((uv + vec2(-px.x, -px.y)) * 6.0 + vec2(5.3, 1.7 + phase)));

        // Centre weight: -(4 * 1.0 + 4 * 0.707106781) = -6.828427
        uLap += -6.828427 * centre_u;
        vLap += -6.828427 * centre_v;

        // Gray-Scott reaction step
        float uvv = u * v * v;
        float du = diffU * uLap - uvv + f * (1.0 - u);
        float dv = diffV * vLap + uvv - (f + k) * v;

        u = clamp(u + du * 0.6, 0.0, 1.0);
        v = clamp(v + dv * 0.6, 0.0, 1.0);
    }

    return vec2(u, v);
}

// ── main ──────────────────────────────────────────────────────────────────────
void main() {
    // Normalise fragment to bbox-local UV [0,1]
    vec2 uv = (gl_FragCoord.xy - u_bbox.xy) / u_bbox.zw;

    // ── Physics coupling: velocity distorts UV ────────────────────────────────
    // High speed → pattern stretched along flow direction.
    // Velocity is in world-units/s; we normalise to a reasonable warp range.
    float speed = length(u_velocity);
    vec2  flowDir = speed > 0.001 ? normalize(u_velocity) : vec2(1.0, 0.0);
    // Project UV onto flow direction and warp
    float flow = dot(uv - 0.5, flowDir);
    float warpStr = clamp(speed * 0.08, 0.0, 0.35);
    vec2 uvWarped = uv + flowDir * flow * warpStr;
    // Clamp to avoid sampling outside cell
    uvWarped = clamp(uvWarped, 0.0, 1.0);

    // ── circleSDF boundary mask ───────────────────────────────────────────────
    // Fade the pattern out at the cell edges using the lygia circleSDF.
    float dist = circleSDF(uv);          // 0 = centre, 1 = boundary, >1 = outside
    float edgeMask = smoothstep(1.05, 0.75, dist);

    // ── Species (f,k) parameters ──────────────────────────────────────────────
    vec2  fk = speciesFeedKill(u_species);

    // ── Density modulates diffusion scale ─────────────────────────────────────
    // u_density in [0,1].  Low density → slower diffusion → coarser pattern.
    // High density → faster diffusion → finer, denser markings.
    float diffScale = 0.6 + u_density * 0.8;

    // ── Run RD evolution ──────────────────────────────────────────────────────
    vec2 rd = rdEvolve(uvWarped, fk, diffScale);
    float uConc = rd.x;   // activator  (chemical U)
    float vConc = rd.y;   // inhibitor  (chemical V)

    // ── Pattern visualisation ─────────────────────────────────────────────────
    // The inhibitor V concentration is the "ink" of the Turing pattern:
    //   high V  → dark markings (spots, stripes, worm heads)
    //   low  V  → bright background
    //
    // We use 1-U as a secondary channel that cleanly separates the pattern
    // from the background at the Pearson threshold ~0.5.

    float pattern = clamp(vConc * 2.0 - 0.5, 0.0, 1.0);

    // Species-specific colour tinting:
    //   spots  (0) → sharp high-contrast spots
    //   waves  (1) → softer continuous labyrinthine bands
    //   worms  (2) → elongated filaments, mid-contrast
    //   mitosis(3) → small oval pearls with halos
    float contrast = 1.0;
    float fi = float(u_species);
    contrast = mix(contrast, 2.2, step(0.5, fi - 0.0) * step(fi, 0.5));  // spots
    contrast = mix(contrast, 1.2, step(0.5, fi - 0.5) * step(fi, 1.5));  // waves
    contrast = mix(contrast, 1.6, step(0.5, fi - 1.5) * step(fi, 2.5));  // worms
    contrast = mix(contrast, 1.8, step(0.5, fi - 2.5) * step(fi, 3.5));  // mitosis

    pattern = clamp((pattern - 0.5) * contrast + 0.5, 0.0, 1.0);

    // Inner glow from activator U: bright halos around pattern edges
    float glow = smoothstep(0.3, 0.7, uConc) * 0.25 * edgeMask;

    // ── Final alpha compositing ───────────────────────────────────────────────
    // Pattern alpha: solid at pattern peaks, transparent at background.
    // The cell body colour is rendered by the host; this shader adds the
    // Turing overlay ON TOP with additive/multiply blending intent.
    float alpha = clamp(pattern * edgeMask + glow, 0.0, 1.0);

    // Pattern colour: dark markings = near-black over cell colour.
    // We output the concentration directly; the host renderer multiplies by
    // u_fillColor when compositing.
    float brightness = 1.0 - pattern * 0.85;
    vec3  col = vec3(brightness) + vec3(glow * 0.3, glow * 0.6, glow * 1.0);

    fragColor = vec4(col, alpha);
}
