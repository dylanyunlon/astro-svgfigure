# Fluid Shader Inventory — compiled.vs

**Source:** `upstream/activetheory-assets/compiled.vs`  
**Format:** `{@}shaderName{@}...source...` delimited (174 shaders total)  
**Task:** M972 — Extract and validate 9 core fluid simulation fragment shaders

---

## Core 9 Fluid Simulation Shaders

All 9 core shaders are **present** and **syntax-valid** (balanced braces, `void main` present, no `#require` dependencies).

| # | Shader | Chars | Lines | void main | Braces balanced | Notes |
|---|--------|-------|-------|-----------|-----------------|-------|
| 1 | `splatShader.fs` | 1081 | 40 | ✅ | ✅ (5/5) | Line splat with cubic easing + screen blend |
| 2 | `advectionShader.fs` | 315 | 11 | ✅ | ✅ (1/1) | Simple texture advection |
| 3 | `advectionManualFilteringShader.fs` | 801 | 22 | ✅ | ✅ (2/2) | Bilinear interpolation (bilerp) fallback |
| 4 | `divergenceShader.fs` | 588 | 19 | ✅ | ✅ (5/5) | Boundary-aware divergence field |
| 5 | `pressureShader.fs` | 758 | 23 | ✅ | ✅ (2/2) | Jacobi pressure solve iteration |
| 6 | `vorticityShader.fs` | 651 | 23 | ✅ | ✅ (1/1) | Curl-based vorticity confinement |
| 7 | `curlShader.fs` | 423 | 14 | ✅ | ✅ (1/1) | Velocity curl (vorticity field) |
| 8 | `gradientSubtractShader.fs` | 629 | 21 | ✅ | ✅ (2/2) | Pressure gradient subtraction |
| 9 | `clearShader.fs` | 136 | 6 | ✅ | ✅ (1/1) | Scalar field decay (dissipation) |

**Result: 9/9 core shaders found — all pass syntax validation.**

---

## Additional Fluid-Related Shaders

| Shader | Chars | Lines | void main | Notes |
|--------|-------|-------|-----------|-------|
| `displayShader.fs` | 175 | 7 | ✅ | RGB→RGBA with alpha = max channel |
| `colorShader.fs` | 62 | 4 | ✅ | Flat uniform color fill |
| `fluidBase.vs` | 336 | 15 | ✅ | Vertex shader — 5-tap neighbour UV setup |
| `mousefluid.fs` | 382 | 12 | ⚠️ no main | Utility functions only: `getFluidVelocity()` / `getFluidVelocityMask()` — intended as an include fragment |

---

## Shader Source Code

### 1. `splatShader.fs` (1081 chars, 40 lines)

```glsl
varying vec2 vUv;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec3 bgColor;
uniform vec2 point;
uniform vec2 prevPoint;
uniform float radius;
uniform float canRender;
uniform float uAdd;

float blendScreen(float base, float blend) {
    return 1.0-((1.0-base)*(1.0-blend));
}

vec3 blendScreen(vec3 base, vec3 blend) {
    return vec3(blendScreen(base.r, blend.r), blendScreen(base.g, blend.g), blendScreen(base.b, blend.b));
}

float l(vec2 uv, vec2 point1, vec2 point2) {
    vec2 pa = uv - point1, ba = point2 - point1;
    pa.x *= aspectRatio;
    ba.x *= aspectRatio;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float cubicOut(float t) {
    float f = t - 1.0;
    return f * f * f + 1.0;
}

void main () {
    vec3 splat = (1.0 - cubicOut(clamp(l(vUv, prevPoint.xy, point.xy) / radius, 0.0, 1.0))) * color;
    vec3 base = texture2D(uTarget, vUv).xyz;
    base *= canRender;

    vec3 outColor = mix(blendScreen(base, splat), base + splat, uAdd);
    gl_FragColor = vec4(outColor, 1.0);
}
```

### 2. `advectionShader.fs` (315 chars, 11 lines)

```glsl
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;
void main () {
    vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
    gl_FragColor = dissipation * texture2D(uSource, coord);
    gl_FragColor.a = 1.0;
}
```

### 3. `advectionManualFilteringShader.fs` (801 chars, 22 lines)

```glsl
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform vec2 dyeTexelSize;
uniform float dt;
uniform float dissipation;
vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
    vec2 st = uv / tsize - 0.5;
    vec2 iuv = floor(st);
    vec2 fuv = fract(st);
    vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
    vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
    vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
    vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
void main () {
    vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
    gl_FragColor = dissipation * bilerp(uSource, coord, dyeTexelSize);
    gl_FragColor.a = 1.0;
}
```

### 4. `divergenceShader.fs` (588 chars, 19 lines)

```glsl
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
    float L = texture2D(uVelocity, vL).x;
    float R = texture2D(uVelocity, vR).x;
    float T = texture2D(uVelocity, vT).y;
    float B = texture2D(uVelocity, vB).y;
    vec2 C = texture2D(uVelocity, vUv).xy;
   if (vL.x < 0.0) { L = -C.x; }
   if (vR.x > 1.0) { R = -C.x; }
   if (vT.y > 1.0) { T = -C.y; }
   if (vB.y < 0.0) { B = -C.y; }
    float div = 0.5 * (R - L + T - B);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
}
```

### 5. `pressureShader.fs` (758 chars, 23 lines)

```glsl
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
vec2 boundary (vec2 uv) {
    return uv;
    // uncomment if you use wrap or repeat texture mode
    // uv = min(max(uv, 0.0), 1.0);
    // return uv;
}
void main () {
    float L = texture2D(uPressure, boundary(vL)).x;
    float R = texture2D(uPressure, boundary(vR)).x;
    float T = texture2D(uPressure, boundary(vT)).x;
    float B = texture2D(uPressure, boundary(vB)).x;
    float C = texture2D(uPressure, vUv).x;
    float divergence = texture2D(uDivergence, vUv).x;
    float pressure = (L + R + B + T - divergence) * 0.25;
    gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
}
```

### 6. `vorticityShader.fs` (651 chars, 23 lines)

```glsl
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
void main () {
    float L = texture2D(uCurl, vL).x;
    float R = texture2D(uCurl, vR).x;
    float T = texture2D(uCurl, vT).x;
    float B = texture2D(uCurl, vB).x;
    float C = texture2D(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
//    force.y += 400.3;
    vec2 vel = texture2D(uVelocity, vUv).xy;
    gl_FragColor = vec4(vel + force * dt, 0.0, 1.0);
}
```

### 7. `curlShader.fs` (423 chars, 14 lines)

```glsl
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
    float L = texture2D(uVelocity, vL).y;
    float R = texture2D(uVelocity, vR).y;
    float T = texture2D(uVelocity, vT).x;
    float B = texture2D(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
}
```

### 8. `gradientSubtractShader.fs` (629 chars, 21 lines)

```glsl
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
vec2 boundary (vec2 uv) {
    return uv;
    // uv = min(max(uv, 0.0), 1.0);
    // return uv;
}
void main () {
    float L = texture2D(uPressure, boundary(vL)).x;
    float R = texture2D(uPressure, boundary(vR)).x;
    float T = texture2D(uPressure, boundary(vT)).x;
    float B = texture2D(uPressure, boundary(vB)).x;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
}
```

### 9. `clearShader.fs` (136 chars, 6 lines)

```glsl
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
void main () {
    gl_FragColor = value * texture2D(uTexture, vUv);
}
```

---

## Validation Summary

- **Delimiter format:** `{@}name{@}source` — parsed correctly across all 174 shaders
- **`#require` directives:** None found in any fluid shader — all are self-contained
- **Brace balance:** All 9 core shaders have perfectly balanced `{` / `}` counts
- **`void main` presence:** 9/9 core shaders confirmed ✅ (`mousefluid.fs` intentionally has no main — it is a utility include)
- **Uniforms/varyings:** All inputs are declared; no undeclared variables detected
- **`highp` precision:** Used in `divergenceShader`, `pressureShader`, `curlShader`, `gradientSubtractShader` on neighbour UVs — correct for mobile WebGL compatibility

**All 9 core fluid shaders are syntactically valid and ready for extraction.**

---

*Generated: M972 fluid shader inventory task*
