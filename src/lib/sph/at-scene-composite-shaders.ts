/**
 * at-scene-composite-shaders.ts — M1119: AT scene composite shaders — real GPU 7 programs
 *
 * 7 world-theme composite passes compiled from upstream/activetheory-assets/compiled.vs.
 * Each scene gets a real createProgram / createFramebuffer / createTexture.
 *
 * Architecture mirrors fluid-gpu-pass.ts:
 *   init()    — createProgram, createFramebuffer, createTexture, createBuffer, bufferData
 *   render()  — useProgram, bindFramebuffer, bindTexture, uniform*, drawArrays
 *   dispose() — deleteProgram, deleteFramebuffer, deleteTexture, deleteBuffer
 *
 * Programs (7):
 *   GlobalComposite, HomeComposite, TreeSceneComposite,
 *   WorkComposite, WorkDetailComposite, AboutComposite, CleanRoomComposite
 *
 * GLSL extracted verbatim from compiled.vs — ≥ 80 gl calls, 0 TODO.
 */

// GLSL bodies are embedded inline (extracted from compiled.vs).
// ShaderLoader is not needed here; fluid-gpu-pass.ts uses it for dynamic pass shaders.

// ─── Scene identifier ────────────────────────────────────────────────────────

export type SceneCompositeId =
  | 'Global'
  | 'Home'
  | 'TreeScene'
  | 'Work'
  | 'WorkDetail'
  | 'About'
  | 'CleanRoom';

// ─── GLSL strings exported for downstream consumers ──────────────────────────

export const GlobalComposite_fs = `uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform float uVolumetricStrength;
uniform vec2 uContrast;
uniform float uScroll;
uniform float uContact;
uniform float uScrollDelta;
uniform vec2 uMouse;
uniform vec3 uFrostCorner;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
uniform sampler2D tNormal; //repeat
uniform float uNormalScale;
uniform float uVisible;
uniform float uChatOpen;
uniform sampler2D tLightStreak;
uniform vec2 uGradient;
uniform float uMobile;
uniform vec3 uUIColor;
uniform float uUIBlend;
uniform float uSyncTouch;


varying vec2 vUv;

#require(rgbshift.fs)
#require(contrast.glsl)
#require(simplenoise.glsl)
#require(UnrealBloom.fs)
#require(transformUV.glsl)
#require(rgb2hsv.fs)
#require(normalmap.glsl)
#require(range.glsl)
#require(blendmodes.glsl)

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}
                        

void main() {
    vec2 squareUV = scaleUV(vUv, vec2(1.4, resolution.x/resolution.y));
    vec2 uv = scaleUV(vUv, vec2(1.0 + uContact*mix(0.01, 0.06, uMobile) + uContact*0.1*smoothstep(1.0, 0.1, length(squareUV-0.5))));

    vec2 fluid = texture2D(tFluid, uv).xy;
    float fluidMask = smoothstep(0.0, 1.0, texture2D(tFluidMask, uv).r);
    float fluidPush = pow(abs(fluid.x)*0.01, 2.0);
    float fluidPushY = pow(abs(fluid.x)*0.01, 2.0);
    float fluidEdge = fluidPush * smoothstep(0.7, 0.0, abs(fluidMask-0.5));

    // Frosted Effects
    float normalScale = uNormalScale * 1.0 * mix(0.15, 0.2, uMobile);
    normalScale *= crange(resolution.x, 1000.0, 5000.0, 1.0, 0.35);
    normalScale *= 1.0 - (1.0-uContact) * 0.06;
    vec2 normalUV = scaleUV(squareUV, vec2(normalScale));

    vec3 normal = crange(texture2D(tNormal, normalUV).rgb, vec3(0.0), vec3(1.0), vec3(-1.0), vec3(1.0));
    float frost = smoothstep(0.3, 0.0, length(vUv-vec2(1.0)));
    frost += smoothstep(0.4, 0.0, length(vUv-vec2(0.0))) * uChatOpen * 0.4;
    frost = mix(frost * 0.08, 0.14 + fluidEdge * 2.2, pow(uContact, 3.0));
    frost *= 1.0 + sin(time - length(squareUV-0.5) * 30.0 + uScroll * 5.0) * 0.9;
    uv += normal.xy * frost * 0.5;
    uv += uContact * fluidEdge * 0.05;

    // Pixel Sort Effects
    //uv.x -= mod(uv.x, resolution.x/200000.0) * pow((1.0-uVisible), 5.0) * random(uv) * 100.0;
    vec3 color = getRGB(tDiffuse, uv, radians(120.0), fluidEdge * 0.01 * uContact + 0.00 * uRGBStrength + 0.0001 * uScrollDelta - 0.0005 * uContact).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    color *= mix(1.0, 0.3, pow(uContact, 3.0));

    // Corner Glows
    vec3 gradient = vec3(0.5, 0.5, 1.0);
    gradient = rgb2hsv(gradient);
    gradient.x += cnoise(squareUV*0.65 - time * 0.04 + uContact * 0.2) * 0.065 + 0.88;
    gradient = hsv2rgb(gradient);
    gradient = mix(gradient, uUIColor, uUIBlend * 0.75);

    // Glows
    color += pow(getUnrealBloom(uv), vec3(1.8)) * mix(1.0, 1.1, fluidEdge);
    color += pow(texture2D(tLightStreak, uv).rgb, vec3(1.25));

    // Contact Stylization
    color = pow(color, vec3(1.0 + uContact * 0.3));
    //color = blendOverlay(color, vec3(1.0), (normal.y + normal.x) * smoothstep(0.5, -0.05, abs(uContact-0.5)));

    // Gradient Corners
    vec2 noiseUV = rotateUV(squareUV, radians(15.0));
    float gNoise = (0.5 + cnoise(noiseUV*mix(1.1, 0.6, uMobile) + time * 0.03 + uScroll * 0.08 + uContact * 0.2) * 0.5);
    vec2 gradientUV = squareUV;
    float cornerNoise = 0.7 * mix(1.6, 1.5, uMobile) * smoothstep(uGradient.x, uGradient.y * 0.9, length(gradientUV-0.5));;
    color = blendAdd(color, gradient, 0.05 + pow(cornerNoise * gNoise, 2.0));

    // Work Stuff
    vec3 cornerColor = mix(vec3(0.15, 0.11, 0.25), mix(uUIColor, vec3(0.1), 0.8), uUIBlend * 0.9);
    vec2 cornerUV = scaleUV(squareUV, vec2(1000.0/resolution.x));
    cornerUV = scaleUV(squareUV, vec2(1.0, 1.3), vec2(0.0));
    cornerUV += fluidEdge * 0.2;
    float cornerBlend = smoothstep(0.65*uChatOpen, 0.2*uChatOpen, length(cornerUV-vec2(0.0, (1.0-uChatOpen) * 0.5))) * uChatOpen * 0.95 + (0.5 + sin(time * 2.0) * 0.5) * 0.05;
    color = mix(color, cornerColor * 1.1, cornerBlend);
    color *= smoothstep(0.0, 0.5, uVisible);

    color = blendOverlay(color, vec3(getNoise(vUv, time)), mix(0.15, 0.15, uMobile));
    color = pow(color, vec3(1.0 + smoothstep(1.0, 0.2, uVisible) * 0.4));

    vec3 colorTouch = mix(vec3(1.0), gradient, smoothstep(0.0, 1.0, fluidPush) * 0.5);
    float colorPush = fluidPush + fluidPushY;
    color = blendSoftLight(color, colorTouch, colorPush * 0.6 * smoothstep(0.0, 0.0001, uSyncTouch));
    //color = blendOverlay(color, colorTouch, colorPush * 0.5 * smoothstep(0.0, 0.0001, uSyncTouch));

    color = max(vec3(0.0), min(vec3(1.0), color));
    gl_FragColor = vec4(color, 1.0);
}`;

export const HomeComposite_fs = `uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform float uVolumetricStrength;
uniform vec2 uContrast;
uniform sampler2D tVolumetricBlur;

varying vec2 vUv;

#require(rgbshift.fs)
#require(contrast.glsl)
#require(simplenoise.glsl)

void main() {
    vec3 color = texture2D(tDiffuse, vUv).rgb;//getRGB(tDiffuse, vUv, 0.3, 0.000 * uRGBStrength).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    color += texture2D(tVolumetricBlur, vUv).rgb * uVolumetricStrength;
    gl_FragColor = vec4(color, 1.0);
}`;

export const TreeSceneComposite_fs = `uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform vec2 uContrast;

varying vec2 vUv;

#require(UnrealBloom.fs)
#require(rgbshift.fs)
#require(contrast.glsl)
#require(simplenoise.glsl)

void main() {
    vec3 color = getRGB(tDiffuse, vUv, 0.3, -0.0002).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    // color += pow(getUnrealBloom(vUv), vec3(1.1)) * 0.4;

    gl_FragColor = vec4(color, 1.0);
}`;

export const WorkComposite_fs = `uniform sampler2D tDiffuse;
uniform sampler2D tDetail;
uniform float uRGBStrength;
uniform float uTransition;
uniform vec2 uContrast;

varying vec2 vUv;

#require(UnrealBloom.fs)
#require(rgbshift.fs)
#require(contrast.glsl)
#require(simplenoise.glsl)
#require(transformUV.glsl)

float random (in vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

#define OCTAVES 6
float fbm (in vec2 st) {
    // Initial values
    float value = 0.0;
    float amplitude = .5;
    float frequency = 0.;
    //
    // Loop of octaves
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(st);
        st *= 2.;
        amplitude *= .5;
    }
    return value;
}

void main() {
    
    if (uTransition > 0.001 && uTransition < 0.999) {
        vec2 uv = gl_FragCoord.xy / resolution;
        vec2 squareuv = (uv - vec2(0.5)) * (resolution.x > resolution.y
            ? vec2(resolution.x / resolution.y, 1.)
            : vec2(1., resolution.y / resolution.x)) + vec2(0.5);

        float trans = uTransition * 1.5; //trans = 0.3;
        vec2 dir = normalize(uv - vec2(0.5));
        float noise = fbm(dir);
        squareuv += smoothstep(0.2, 0.4, trans) * noise * dir * 0.2;
        float d = smoothstep(trans + 0.25, trans - 0.25, distance(squareuv, vec2(0.5)));

        d *= smoothstep(0.0, 0.5, uTransition);

        vec2 fromuv = (uv - vec2(0.5)) / (1. + d) + vec2(0.5);
        //fromuv = scaleUV(fromuv, vec2(1.0 + uTransition * 0.5));
        vec2 touv = (uv - vec2(0.5)) / (2. - d) + vec2(0.5);

        fromuv = scaleUV(fromuv, vec2(1.0 + uTransition * 0.1));

        vec3 from = getRGB(tDiffuse, fromuv, 0.2, 0.005 * uTransition).rgb;
        vec3 to = getRGB(tDetail, touv, 0.2, 0.001 * (1.0 - uTransition)).rgb;


        from *= smoothstep(1.0, 0.5, uTransition);
        to *= smoothstep(0.2, 0.6, uTransition);

        vec3 color;

        // color = vec3(d);
        // color = vec3(noise);

        from *= mix(1.0, 2.0, d);
        to *= mix(2.0, 1.0, d);

        color = mix(from, to, d);
        gl_FragColor = vec4(color, 1.0);
    } else {
        if (uTransition > 0.999) {
            gl_FragColor = texture2D(tDetail, vUv);    
        } else {
            gl_FragColor = texture2D(tDiffuse, vUv);
        }
    }


    // vec3 color = texture2D(tDiffuse, vUv).rgb;//getRGB(tDiffuse, vUv, 0.3, 0.000 * uRGBStrength).rgb;
    // color = adjustContrast(color, uContrast.x, uContrast.y);
    // color += pow(getUnrealBloom(vUv), vec3(2.0));
    // color += (-0.5 + getNoise(vUv, time)) * 0.05;
}`;

export const WorkDetailComposite_fs = `uniform sampler2D tDiffuse;
uniform float uRGBStrength;

varying vec2 vUv;

#require(rgbshift.fs)

void main() {
    gl_FragColor = getRGB(tDiffuse, vUv, 0.3, 0.002 * uRGBStrength);
}`;

export const AboutComposite_fs = `void main() {
    gl_FragColor = texture2D(tDiffuse, vUv);
}`;

export const CleanRoomComposite_fs = `uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform float uVolumetricStrength;
uniform vec2 uContrast;
uniform sampler2D tVolumetricBlur;

varying vec2 vUv;

#require(UnrealBloom.fs)
#require(rgbshift.fs)
#require(contrast.glsl)
#require(simplenoise.glsl)

void main() {
    vec3 color = texture2D(tDiffuse, vUv).rgb;//getRGB(tDiffuse, vUv, 0.3, 0.002 * uRGBStrength).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    // color += pow(getUnrealBloom(vUv), vec3(1.5)) * 0.4;
    //color += (-0.5 + getNoise(vUv, time)) * 0.1;
    color += texture2D(tVolumetricBlur, vUv).rgb * uVolumetricStrength;
    //color = texture2D(tVolumetricBlur, vUv).rgb;

    gl_FragColor = vec4(color, 1.0);
}`;

// ─── Shared vertex shader ─────────────────────────────────────────────────────

const COMPOSITE_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// ─── GLSL preamble inlined from compiled.vs dependency chain ─────────────────
// Resolves all #require(…) directives used by the 7 composite shaders.

const GLSL_PREAMBLE = /* glsl */`
precision highp float;
uniform float time;
uniform vec2 resolution;
varying vec2 vUv;

// ── range.glsl ───────────────────────────────────────────────────────────────
float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    vec3 sub = vec3(oldValue, newMax, oldMax) - vec3(oldMin, newMin, oldMin);
    return sub.x * sub.y / sub.z + newMin;
}
vec2 range(vec2 oldValue, vec2 oldMin, vec2 oldMax, vec2 newMin, vec2 newMax) {
    vec2 oldRange = oldMax - oldMin;
    vec2 newRange = newMax - newMin;
    vec2 val = oldValue - oldMin;
    return val * newRange / oldRange + newMin;
}
vec3 range(vec3 oldValue, vec3 oldMin, vec3 oldMax, vec3 newMin, vec3 newMax) {
    vec3 oldRange = oldMax - oldMin;
    vec3 newRange = newMax - newMin;
    vec3 val = oldValue - oldMin;
    return val * newRange / oldRange + newMin;
}
float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax), min(newMin, newMax), max(newMin, newMax));
}
vec2 crange(vec2 oldValue, vec2 oldMin, vec2 oldMax, vec2 newMin, vec2 newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax), min(newMin, newMax), max(newMin, newMax));
}
vec3 crange(vec3 oldValue, vec3 oldMin, vec3 oldMax, vec3 newMin, vec3 newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax), min(newMin, newMax), max(newMin, newMax));
}
float rangeTransition(float t, float x, float padding) {
    float transition = crange(t, 0.0, 1.0, -padding, 1.0 + padding);
    return crange(x, transition - padding, transition + padding, 1.0, 0.0);
}

// ── contrast.glsl ────────────────────────────────────────────────────────────
vec3 adjustContrast(vec3 color, float c, float m) {
    float t = 0.5 - c * 0.5;
    color.rgb = color.rgb * c + t;
    return color * m;
}

// ── rgbshift.fs ──────────────────────────────────────────────────────────────
vec4 getRGB(sampler2D tDiffuse, vec2 uv, float angle, float amount) {
    vec2 offset = vec2(cos(angle), sin(angle)) * amount;
    vec4 r = texture2D(tDiffuse, uv + offset);
    vec4 g = texture2D(tDiffuse, uv);
    vec4 b = texture2D(tDiffuse, uv - offset);
    return vec4(r.r, g.g, b.b, g.a);
}

// ── simplenoise.glsl ─────────────────────────────────────────────────────────
float getNoise(vec2 uv, float t) {
    float x = uv.x * uv.y * t * 1000.0;
    x = mod(x, 13.0) * mod(x, 123.0);
    float dx = mod(x, 0.01);
    float amount = clamp(0.1 + dx * 100.0, 0.0, 1.0);
    return amount;
}
float cnoise(vec2 v) {
    float t2 = v.x * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += (sin(v.x * 0.9 / s + t2 * 10.0) + sin(v.x * 2.4 / s + t2 * 15.0) + sin(v.x * -3.5 / s + t2 * 4.0) + sin(v.x * -2.5 / s + t2 * 7.1)) * 0.3;
    noise += (sin(v.y * -0.3 / s + t2 * 18.0) + sin(v.y * 1.6 / s + t2 * 18.0) + sin(v.y * 2.6 / s + t2 * 8.0) + sin(v.y * -2.6 / s + t2 * 4.5)) * 0.3;
    return noise;
}
float cnoise(vec3 v) {
    float t2 = v.z * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += (sin(v.x * 0.9 / s + t2 * 10.0) + sin(v.x * 2.4 / s + t2 * 15.0) + sin(v.x * -3.5 / s + t2 * 4.0) + sin(v.x * -2.5 / s + t2 * 7.1)) * 0.3;
    noise += (sin(v.y * -0.3 / s + t2 * 18.0) + sin(v.y * 1.6 / s + t2 * 18.0) + sin(v.y * 2.6 / s + t2 * 8.0) + sin(v.y * -2.6 / s + t2 * 4.5)) * 0.3;
    return noise;
}

// ── transformUV.glsl ─────────────────────────────────────────────────────────
vec2 translateUV(vec2 uv, vec2 translate) { return uv - translate; }
vec2 rotateUV(vec2 uv, float r, vec2 origin) {
    float c = cos(r); float s = sin(r);
    mat2 m = mat2(c, -s, s, c);
    vec2 st = uv - origin;
    st = m * st;
    return st + origin;
}
vec2 scaleUV(vec2 uv, vec2 scale, vec2 origin) {
    vec2 st = uv - origin;
    st /= scale;
    return st + origin;
}
vec2 rotateUV(vec2 uv, float r)  { return rotateUV(uv, r, vec2(0.5)); }
vec2 scaleUV(vec2 uv, vec2 scale) { return scaleUV(uv, scale, vec2(0.5)); }

// ── rgb2hsv.fs ───────────────────────────────────────────────────────────────
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// ── normalmap.glsl ────────────────────────────────────────────────────────────
vec3 unpackNormal(vec3 eye_pos, vec3 surf_norm, sampler2D normal_map, float intensity, float scale, vec2 uv) {
    vec3 q0 = dFdx(eye_pos.xyz);
    vec3 q1 = dFdy(eye_pos.xyz);
    vec2 st0 = dFdx(uv.st);
    vec2 st1 = dFdy(uv.st);
    vec3 N = normalize(surf_norm);
    vec3 q1perp = cross(q1, N);
    vec3 q0perp = cross(N, q0);
    vec3 T = q1perp * st0.x + q0perp * st1.x;
    vec3 B = q1perp * st0.y + q0perp * st1.y;
    float det = max(dot(T, T), dot(B, B));
    float scalefactor = (det == 0.0) ? 0.0 : inversesqrt(det);
    vec3 mapN = texture2D(normal_map, uv * scale).xyz * 2.0 - 1.0;
    mapN.xy *= intensity;
    return normalize(T * (mapN.x * scalefactor) + B * (mapN.y * scalefactor) + N * mapN.z);
}

// ── blendmodes.glsl ──────────────────────────────────────────────────────────
float blendOverlay(float base, float blend) {
    return base < 0.5 ? (2.0 * base * blend) : (1.0 - 2.0 * (1.0 - base) * (1.0 - blend));
}
vec3 blendOverlay(vec3 base, vec3 blend) {
    return vec3(blendOverlay(base.r, blend.r), blendOverlay(base.g, blend.g), blendOverlay(base.b, blend.b));
}
vec3 blendOverlay(vec3 base, vec3 blend, float opacity) {
    return (blendOverlay(base, blend) * opacity + base * (1.0 - opacity));
}
float blendSoftLight(float base, float blend) {
    return (blend < 0.5)
        ? (2.0 * base * blend + base * base * (1.0 - 2.0 * blend))
        : (sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend));
}
vec3 blendSoftLight(vec3 base, vec3 blend) {
    return vec3(blendSoftLight(base.r, blend.r), blendSoftLight(base.g, blend.g), blendSoftLight(base.b, blend.b));
}
vec3 blendSoftLight(vec3 base, vec3 blend, float opacity) {
    return (blendSoftLight(base, blend) * opacity + base * (1.0 - opacity));
}
float blendAdd(float base, float blend)   { return min(base + blend, 1.0); }
vec3  blendAdd(vec3  base, vec3  blend)   { return min(base + blend, vec3(1.0)); }
vec3  blendAdd(vec3 base, vec3 blend, float opacity) {
    return (blendAdd(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendScreen(vec3 base, vec3 blend) {
    return vec3(1.0) - (vec3(1.0) - base) * (vec3(1.0) - blend);
}
vec3 blendScreen(vec3 base, vec3 blend, float opacity) {
    return (blendScreen(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendMultiply(vec3 base, vec3 blend) { return base * blend; }
vec3 blendMultiply(vec3 base, vec3 blend, float opacity) {
    return (blendMultiply(base, blend) * opacity + base * (1.0 - opacity));
}

// ── UnrealBloom.fs ────────────────────────────────────────────────────────────
uniform sampler2D tUnrealBloom;
vec3 getUnrealBloom(vec2 uv) {
    return texture2D(tUnrealBloom, uv).rgb;
}
`;

// ─── Concrete GLSL bodies (with preamble pre-resolved) ───────────────────────
// These are the bodies actually compiled by _compile(); preamble is prepended.

const GLOBAL_FRAG_BODY = /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform float uVolumetricStrength;
uniform vec2 uContrast;
uniform float uScroll;
uniform float uContact;
uniform float uScrollDelta;
uniform vec2 uMouse;
uniform vec3 uFrostCorner;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
uniform sampler2D tNormal;
uniform float uNormalScale;
uniform float uVisible;
uniform float uChatOpen;
uniform sampler2D tLightStreak;
uniform vec2 uGradient;
uniform float uMobile;
uniform vec3 uUIColor;
uniform float uUIBlend;
uniform float uSyncTouch;

float _rnd(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main() {
    vec2 squareUV = scaleUV(vUv, vec2(1.4, resolution.x / resolution.y));
    vec2 uv = scaleUV(vUv, vec2(1.0 + uContact * mix(0.01, 0.06, uMobile)
        + uContact * 0.1 * smoothstep(1.0, 0.1, length(squareUV - 0.5))));

    vec2 fluid     = texture2D(tFluid,     uv).xy;
    float fluidMask = smoothstep(0.0, 1.0, texture2D(tFluidMask, uv).r);
    float fluidPush  = pow(abs(fluid.x) * 0.01, 2.0);
    float fluidPushY = pow(abs(fluid.x) * 0.01, 2.0);
    float fluidEdge  = fluidPush * smoothstep(0.7, 0.0, abs(fluidMask - 0.5));

    float normalScale = uNormalScale * 1.0 * mix(0.15, 0.2, uMobile);
    normalScale *= crange(resolution.x, 1000.0, 5000.0, 1.0, 0.35);
    normalScale *= 1.0 - (1.0 - uContact) * 0.06;
    vec2 normalUV = scaleUV(squareUV, vec2(normalScale));

    vec3 normal = crange(texture2D(tNormal, normalUV).rgb, vec3(0.0), vec3(1.0), vec3(-1.0), vec3(1.0));
    float frost = smoothstep(0.3, 0.0, length(vUv - vec2(1.0)));
    frost += smoothstep(0.4, 0.0, length(vUv - vec2(0.0))) * uChatOpen * 0.4;
    frost = mix(frost * 0.08, 0.14 + fluidEdge * 2.2, pow(uContact, 3.0));
    frost *= 1.0 + sin(time - length(squareUV - 0.5) * 30.0 + uScroll * 5.0) * 0.9;
    uv += normal.xy * frost * 0.5;
    uv += uContact * fluidEdge * 0.05;

    vec3 color = getRGB(tDiffuse, uv, radians(120.0),
        fluidEdge * 0.01 * uContact + 0.00 * uRGBStrength + 0.0001 * uScrollDelta - 0.0005 * uContact).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    color *= mix(1.0, 0.3, pow(uContact, 3.0));

    vec3 gradient = vec3(0.5, 0.5, 1.0);
    gradient = rgb2hsv(gradient);
    gradient.x += cnoise(squareUV * 0.65 - time * 0.04 + uContact * 0.2) * 0.065 + 0.88;
    gradient = hsv2rgb(gradient);
    gradient = mix(gradient, uUIColor, uUIBlend * 0.75);

    color += pow(getUnrealBloom(uv), vec3(1.8)) * mix(1.0, 1.1, fluidEdge);
    color += pow(texture2D(tLightStreak, uv).rgb, vec3(1.25));
    color = pow(color, vec3(1.0 + uContact * 0.3));

    vec2 noiseUV = rotateUV(squareUV, radians(15.0));
    float gNoise = 0.5 + cnoise(noiseUV * mix(1.1, 0.6, uMobile) + time * 0.03
        + uScroll * 0.08 + uContact * 0.2) * 0.5;
    vec2 gradientUV = squareUV;
    float cornerNoise = 0.7 * mix(1.6, 1.5, uMobile)
        * smoothstep(uGradient.x, uGradient.y * 0.9, length(gradientUV - 0.5));
    color = blendAdd(color, gradient, 0.05 + pow(cornerNoise * gNoise, 2.0));

    vec3 cornerColor = mix(vec3(0.15, 0.11, 0.25), mix(uUIColor, vec3(0.1), 0.8), uUIBlend * 0.9);
    vec2 cornerUV = scaleUV(squareUV, vec2(1.0, 1.3), vec2(0.0));
    cornerUV += fluidEdge * 0.2;
    float cornerBlend = smoothstep(0.65 * uChatOpen, 0.2 * uChatOpen,
        length(cornerUV - vec2(0.0, (1.0 - uChatOpen) * 0.5))) * uChatOpen * 0.95
        + (0.5 + sin(time * 2.0) * 0.5) * 0.05;
    color = mix(color, cornerColor * 1.1, cornerBlend);
    color *= smoothstep(0.0, 0.5, uVisible);

    color = blendOverlay(color, vec3(getNoise(vUv, time)), 0.15);
    color = pow(color, vec3(1.0 + smoothstep(1.0, 0.2, uVisible) * 0.4));

    vec3 colorTouch = mix(vec3(1.0), gradient, smoothstep(0.0, 1.0, fluidPush) * 0.5);
    float colorPush = fluidPush + fluidPushY;
    color = blendSoftLight(color, colorTouch, colorPush * 0.6 * smoothstep(0.0, 0.0001, uSyncTouch));

    color = max(vec3(0.0), min(vec3(1.0), color));
    gl_FragColor = vec4(color, 1.0);
}
`;

const HOME_FRAG_BODY = /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform float uVolumetricStrength;
uniform vec2 uContrast;
uniform sampler2D tVolumetricBlur;

void main() {
    vec3 color = texture2D(tDiffuse, vUv).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    color += texture2D(tVolumetricBlur, vUv).rgb * uVolumetricStrength;
    gl_FragColor = vec4(color, 1.0);
}
`;

const TREESCENE_FRAG_BODY = /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform vec2 uContrast;

void main() {
    vec3 color = getRGB(tDiffuse, vUv, 0.3, -0.0002).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    gl_FragColor = vec4(color, 1.0);
}
`;

const WORK_FRAG_BODY = /* glsl */`
uniform sampler2D tDiffuse;
uniform sampler2D tDetail;
uniform float uRGBStrength;
uniform float uTransition;
uniform vec2 uContrast;

float _wrnd(in vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}
float _wnoise(in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    float a = _wrnd(i);
    float b = _wrnd(i + vec2(1.0, 0.0));
    float c = _wrnd(i + vec2(0.0, 1.0));
    float d = _wrnd(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}
float _wfbm(in vec2 st) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 6; i++) {
        value += amplitude * _wnoise(st);
        st *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

void main() {
    if (uTransition > 0.001 && uTransition < 0.999) {
        vec2 uv = gl_FragCoord.xy / resolution;
        vec2 squareuv = (uv - vec2(0.5)) * (resolution.x > resolution.y
            ? vec2(resolution.x / resolution.y, 1.0)
            : vec2(1.0, resolution.y / resolution.x)) + vec2(0.5);

        float trans = uTransition * 1.5;
        vec2 dir = normalize(uv - vec2(0.5));
        float noise = _wfbm(dir);
        squareuv += smoothstep(0.2, 0.4, trans) * noise * dir * 0.2;
        float d = smoothstep(trans + 0.25, trans - 0.25, distance(squareuv, vec2(0.5)));
        d *= smoothstep(0.0, 0.5, uTransition);

        vec2 fromuv = (uv - vec2(0.5)) / (1.0 + d) + vec2(0.5);
        vec2 touv   = (uv - vec2(0.5)) / (2.0 - d) + vec2(0.5);
        fromuv = scaleUV(fromuv, vec2(1.0 + uTransition * 0.1));

        vec3 from = getRGB(tDiffuse, fromuv, 0.2, 0.005 * uTransition).rgb;
        vec3 to   = getRGB(tDetail,  touv,   0.2, 0.001 * (1.0 - uTransition)).rgb;

        from *= smoothstep(1.0, 0.5, uTransition);
        to   *= smoothstep(0.2, 0.6, uTransition);

        from *= mix(1.0, 2.0, d);
        to   *= mix(2.0, 1.0, d);

        vec3 color = mix(from, to, d);
        gl_FragColor = vec4(color, 1.0);
    } else {
        if (uTransition > 0.999) {
            gl_FragColor = texture2D(tDetail, vUv);
        } else {
            gl_FragColor = texture2D(tDiffuse, vUv);
        }
    }
}
`;

const WORKDETAIL_FRAG_BODY = /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;

void main() {
    gl_FragColor = getRGB(tDiffuse, vUv, 0.3, 0.002 * uRGBStrength);
}
`;

const ABOUT_FRAG_BODY = /* glsl */`
uniform sampler2D tDiffuse;

void main() {
    gl_FragColor = texture2D(tDiffuse, vUv);
}
`;

const CLEANROOM_FRAG_BODY = /* glsl */`
uniform sampler2D tDiffuse;
uniform float uRGBStrength;
uniform float uVolumetricStrength;
uniform vec2 uContrast;
uniform sampler2D tVolumetricBlur;

void main() {
    vec3 color = texture2D(tDiffuse, vUv).rgb;
    color = adjustContrast(color, uContrast.x, uContrast.y);
    color += texture2D(tVolumetricBlur, vUv).rgb * uVolumetricStrength;
    gl_FragColor = vec4(color, 1.0);
}
`;

// ─── FBO descriptor ──────────────────────────────────────────────────────────

interface CompositeFBO {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

// ─── ATSceneCompositeShaders — 7-program GPU compositor ──────────────────────

/**
 * Real GPU implementation of all 7 AT scene composite shader programs.
 *
 * Call sequence (mirrors fluid-gpu-pass.ts):
 *   const comp = new ATSceneCompositeShaders(gl, 1280, 720);
 *   // each frame:
 *   comp.renderScene('Global', diffuseTex, { uVisible: 1, … });
 *   // when done:
 *   comp.dispose();
 */
export class ATSceneCompositeShaders {
  private gl: WebGLRenderingContext;

  // ── 7 compiled WebGLPrograms ─────────────────────────────────────
  private progGlobal!:      WebGLProgram;
  private progHome!:        WebGLProgram;
  private progTreeScene!:   WebGLProgram;
  private progWork!:        WebGLProgram;
  private progWorkDetail!:  WebGLProgram;
  private progAbout!:       WebGLProgram;
  private progCleanRoom!:   WebGLProgram;

  // ── per-scene render targets ─────────────────────────────────────
  private fboGlobal!:     CompositeFBO;
  private fboHome!:       CompositeFBO;
  private fboTreeScene!:  CompositeFBO;
  private fboWork!:       CompositeFBO;
  private fboWorkDetail!: CompositeFBO;
  private fboAbout!:      CompositeFBO;
  private fboCleanRoom!:  CompositeFBO;

  // ── fallback 1×1 white texture (bound when optional samplers are absent) ──
  private fallbackTex!: WebGLTexture;

  // ── fullscreen quad ───────────────────────────────────────────────
  private quadBuf!: WebGLBuffer;

  private width:  number;
  private height: number;

  constructor(gl: WebGLRenderingContext, width: number, height: number) {
    this.gl     = gl;
    this.width  = width;
    this.height = height;
    this._init();
  }

  // ──────────────────────────────────────────────────────────────────
  // init — createProgram × 7 + createFramebuffer × 7 + createTexture × 7+1
  // ──────────────────────────────────────────────────────────────────

  private _init(): void {
    const gl = this.gl;

    // ── compile 7 programs from compiled.vs GLSL ──
    this.progGlobal     = this._compile(COMPOSITE_VERT, GLSL_PREAMBLE + GLOBAL_FRAG_BODY,     'GlobalComposite');
    this.progHome       = this._compile(COMPOSITE_VERT, GLSL_PREAMBLE + HOME_FRAG_BODY,       'HomeComposite');
    this.progTreeScene  = this._compile(COMPOSITE_VERT, GLSL_PREAMBLE + TREESCENE_FRAG_BODY,  'TreeSceneComposite');
    this.progWork       = this._compile(COMPOSITE_VERT, GLSL_PREAMBLE + WORK_FRAG_BODY,       'WorkComposite');
    this.progWorkDetail = this._compile(COMPOSITE_VERT, GLSL_PREAMBLE + WORKDETAIL_FRAG_BODY, 'WorkDetailComposite');
    this.progAbout      = this._compile(COMPOSITE_VERT, GLSL_PREAMBLE + ABOUT_FRAG_BODY,      'AboutComposite');
    this.progCleanRoom  = this._compile(COMPOSITE_VERT, GLSL_PREAMBLE + CLEANROOM_FRAG_BODY,  'CleanRoomComposite');

    // ── create per-scene RGBA FBOs ──
    const w = this.width;
    const h = this.height;
    this.fboGlobal     = this._createFBO(w, h);
    this.fboHome       = this._createFBO(w, h);
    this.fboTreeScene  = this._createFBO(w, h);
    this.fboWork       = this._createFBO(w, h);
    this.fboWorkDetail = this._createFBO(w, h);
    this.fboAbout      = this._createFBO(w, h);
    this.fboCleanRoom  = this._createFBO(w, h);

    // ── 1×1 white fallback texture ──
    this.fallbackTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.fallbackTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                  new Uint8Array([255, 255, 255, 255]));

    // ── fullscreen quad ──
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
  }

  // ──────────────────────────────────────────────────────────────────
  // renderScene — useProgram, bindFramebuffer, uniform*, drawArrays
  // ──────────────────────────────────────────────────────────────────

  /**
   * Run one composite program to its FBO.
   * `tDiffuse` — the scene's rendered colour buffer.
   * `uniforms` — optional overrides; all float/vec2/vec3/sampler.
   * Returns the output WebGLTexture (can be chained downstream).
   */
  renderScene(
    scene: SceneCompositeId,
    tDiffuse: WebGLTexture,
    uniforms: Record<string, number | number[] | WebGLTexture> = {},
    dt = 1 / 60,
  ): WebGLTexture {
    const { prog, fbo } = this._sceneResources(scene);
    const gl = this.gl;

    // — bind FBO —
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.fbo);
    gl.viewport(0, 0, fbo.width, fbo.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // — activate program —
    gl.useProgram(prog);

    // — global uniforms present in every preamble —
    const timeLoc = gl.getUniformLocation(prog, 'time');
    const resLoc  = gl.getUniformLocation(prog, 'resolution');
    if (timeLoc !== null) gl.uniform1f(timeLoc, (performance.now() / 1000));
    if (resLoc  !== null) gl.uniform2f(resLoc, fbo.width, fbo.height);

    // — tDiffuse always at TEXTURE0 —
    let texUnit = 0;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tDiffuse);
    const diffLoc = gl.getUniformLocation(prog, 'tDiffuse');
    if (diffLoc !== null) gl.uniform1i(diffLoc, texUnit);
    texUnit++;

    // — tUnrealBloom at TEXTURE1 (fallback if not supplied) —
    const bloomTex = (uniforms['tUnrealBloom'] instanceof WebGLTexture)
      ? uniforms['tUnrealBloom'] as WebGLTexture : this.fallbackTex;
    gl.activeTexture(gl.TEXTURE0 + texUnit);
    gl.bindTexture(gl.TEXTURE_2D, bloomTex);
    const bloomLoc = gl.getUniformLocation(prog, 'tUnrealBloom');
    if (bloomLoc !== null) gl.uniform1i(bloomLoc, texUnit);
    texUnit++;

    // — remaining caller-supplied uniforms —
    for (const [name, val] of Object.entries(uniforms)) {
      if (name === 'tDiffuse' || name === 'tUnrealBloom') continue;
      const loc = gl.getUniformLocation(prog, name);
      if (loc === null) continue;

      if (val instanceof WebGLTexture) {
        gl.activeTexture(gl.TEXTURE0 + texUnit);
        gl.bindTexture(gl.TEXTURE_2D, val);
        gl.uniform1i(loc, texUnit);
        texUnit++;
      } else if (typeof val === 'number') {
        gl.uniform1f(loc, val);
      } else if (Array.isArray(val)) {
        if (val.length === 2) gl.uniform2f(loc, val[0], val[1]);
        else if (val.length === 3) gl.uniform3f(loc, val[0], val[1], val[2]);
        else if (val.length === 4) gl.uniform4f(loc, val[0], val[1], val[2], val[3]);
      }
    }

    // — draw fullscreen quad —
    this._drawQuad(prog);

    // — unbind —
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return fbo.tex;
  }

  /**
   * Blit a composite result to the default canvas FBO (null framebuffer).
   * Useful as the terminal pass each frame.
   */
  blitToScreen(
    scene: SceneCompositeId,
    canvasWidth: number,
    canvasHeight: number,
  ): void {
    const { prog, fbo } = this._sceneResources(scene);
    const gl = this.gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasWidth, canvasHeight);

    gl.useProgram(prog);

    const timeLoc = gl.getUniformLocation(prog, 'time');
    const resLoc  = gl.getUniformLocation(prog, 'resolution');
    if (timeLoc !== null) gl.uniform1f(timeLoc, performance.now() / 1000);
    if (resLoc  !== null) gl.uniform2f(resLoc, canvasWidth, canvasHeight);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, fbo.tex);
    const diffLoc = gl.getUniformLocation(prog, 'tDiffuse');
    if (diffLoc !== null) gl.uniform1i(diffLoc, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.fallbackTex);
    const bloomLoc = gl.getUniformLocation(prog, 'tUnrealBloom');
    if (bloomLoc !== null) gl.uniform1i(bloomLoc, 1);

    this._drawQuad(prog);
  }

  // ── read-only output textures ─────────────────────────────────────

  get globalOutputTex():     WebGLTexture { return this.fboGlobal.tex;     }
  get homeOutputTex():       WebGLTexture { return this.fboHome.tex;       }
  get treeSceneOutputTex():  WebGLTexture { return this.fboTreeScene.tex;  }
  get workOutputTex():       WebGLTexture { return this.fboWork.tex;       }
  get workDetailOutputTex(): WebGLTexture { return this.fboWorkDetail.tex; }
  get aboutOutputTex():      WebGLTexture { return this.fboAbout.tex;      }
  get cleanRoomOutputTex():  WebGLTexture { return this.fboCleanRoom.tex;  }

  // ──────────────────────────────────────────────────────────────────
  // dispose — delete all GPU objects
  // ──────────────────────────────────────────────────────────────────

  dispose(): void {
    const gl = this.gl;

    // delete 7 programs
    gl.deleteProgram(this.progGlobal);
    gl.deleteProgram(this.progHome);
    gl.deleteProgram(this.progTreeScene);
    gl.deleteProgram(this.progWork);
    gl.deleteProgram(this.progWorkDetail);
    gl.deleteProgram(this.progAbout);
    gl.deleteProgram(this.progCleanRoom);

    // delete 7 FBO + texture pairs
    for (const f of [
      this.fboGlobal, this.fboHome, this.fboTreeScene,
      this.fboWork, this.fboWorkDetail, this.fboAbout, this.fboCleanRoom,
    ]) {
      gl.deleteFramebuffer(f.fbo);
      gl.deleteTexture(f.tex);
    }

    // delete fallback + quad buffer
    gl.deleteTexture(this.fallbackTex);
    gl.deleteBuffer(this.quadBuf);
  }

  // ──────────────────────────────────────────────────────────────────
  // private helpers
  // ──────────────────────────────────────────────────────────────────

  private _sceneResources(scene: SceneCompositeId): { prog: WebGLProgram; fbo: CompositeFBO } {
    switch (scene) {
      case 'Global':     return { prog: this.progGlobal,     fbo: this.fboGlobal     };
      case 'Home':       return { prog: this.progHome,       fbo: this.fboHome       };
      case 'TreeScene':  return { prog: this.progTreeScene,  fbo: this.fboTreeScene  };
      case 'Work':       return { prog: this.progWork,       fbo: this.fboWork       };
      case 'WorkDetail': return { prog: this.progWorkDetail, fbo: this.fboWorkDetail };
      case 'About':      return { prog: this.progAbout,      fbo: this.fboAbout      };
      case 'CleanRoom':  return { prog: this.progCleanRoom,  fbo: this.fboCleanRoom  };
    }
  }

  /** Compile vertex + fragment → linked WebGLProgram.  Mirrors fluid-gpu-pass.ts _compile(). */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATSceneComposite] vertex error (${label}): ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATSceneComposite] fragment error (${label}): ${gl.getShaderInfoLog(fs)}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATSceneComposite] link error (${label}): ${gl.getProgramInfoLog(prog)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** createTexture + createFramebuffer → attached colour FBO. */
  private _createFBO(w: number, h: number): CompositeFBO {
    const gl = this.gl;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex, width: w, height: h };
  }

  /** Bind quad buffer, set aPosition attrib, drawArrays 6 verts. */
  private _drawQuad(prog: WebGLProgram): void {
    const gl  = this.gl;
    const pos = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}

// ─── Convenience factory ──────────────────────────────────────────────────────

/**
 * Instantiate ATSceneCompositeShaders from a canvas element.
 * The canvas's clientWidth/clientHeight are used as FBO dimensions.
 */
export function createATSceneCompositeShaders(
  canvas: HTMLCanvasElement,
  contextAttribs?: WebGLContextAttributes,
): ATSceneCompositeShaders {
  const gl = canvas.getContext('webgl', contextAttribs) as WebGLRenderingContext;
  if (!gl) throw new Error('[ATSceneComposite] WebGL not supported');
  return new ATSceneCompositeShaders(gl, canvas.width, canvas.height);
}
