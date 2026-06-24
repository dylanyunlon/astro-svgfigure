/**
 * at-remaining-shaders.ts --- M870: Final AT shader sweep
 * ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * Remaining 24 shaders from compiled.vs to reach 100% coverage.
 * Includes: Loader BG variants, Nav UI, AR/VR, debug, occlusion,
 * work panel UI, triangle particles, and bloom gaussian.
 */

/** ARCameraQuad.glsl */








export const ARCameraQuad_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment
void main() {
    #test !!window.Metal
    vUv.y = 1.0 - vUv.y;
    #endtest

    gl_FragColor = texture2D(tMap, vUv);
}`;

/** ChatBGShader.glsl */
export const ChatBGShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uAlpha;
uniform float uScroll;
uniform float uScrollDelta;
uniform float uBottom;
uniform float uDisabled;
uniform float uHeight;
uniform float uActive;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(transformUV.glsl)
#require(simplenoise.glsl)
#require(range.glsl)
#require(rgb2hsv.fs)

float innerRoundedRectangle(vec2 p, vec2 size, float radius, float stepper) {
    vec2 q = abs(p) - size + radius;
    return step(stepper, length(max(q, 0.0)) - radius);
}

void main() {
    vec2 uv = vUv;
    vec4 color = vec4(0.0);

    //float innerClip = step(0.4, abs(uv.x-0.5));
    //innerClip = max(innerClip, step(0.3, abs(uv.y-0.5)));
    
    vec2 innerUV = uv;
    innerUV.y += smoothstep(1.5, 0.0, abs(uv.x-mix(0.35, 0.5, uActive))) * 0.03 * uScrollDelta;

    //innerUV += cnoise(uv*2.0+time*0.2) * 0.01;
    vec2 innerScale = vec2(mix(0.4, 0.75, uActive), uHeight);
    vec2 innerOffset = vec2(mix(0.7, 1.05, uActive), 1.0 + uScrollDelta * 0.03);
    float innerClip = innerRoundedRectangle(scaleUV(innerUV, vec2(0.5, 0.5), innerOffset), innerScale, innerScale.y, 0.0);
    float innerClip2 = innerRoundedRectangle(scaleUV(innerUV, vec2(0.5, 0.5), innerOffset), innerScale, innerScale.y, 0.01);

    vec2 bgUV = scaleUV(innerUV, vec2(mix(0.65, 1.0, uActive), 0.4), vec2(0.1, mix(0.42, 0.57, uBottom) + uScrollDelta * 0.015));

    vec3 rainbow = vec3(0.65, 1.0, 0.9);
    rainbow = rgb2hsv(rainbow);
    rainbow.x += cnoise(-bgUV*0.5-bgUV.y*0.5-time*0.05-uScroll*0.3+length(bgUV-0.2)*0.2) * 0.2;
    rainbow = hsv2rgb(rainbow);

    color = mix(color, vec4(rainbow, 0.5), smoothstep(0.65, abs(uScrollDelta * 0.02) - 0.2, length(bgUV-0.5)));
    color = mix(color, vec4(rainbow, 0.8), smoothstep(0.25, 0.0, length(bgUV-0.5)));
    color = mix(color, vec4(rainbow, 0.2 + abs(uScrollDelta * 0.08) + uActive * 0.4), 1.0-innerClip2);

    vec4 inner = vec4(uColor, 0.7);
    vec2 barUV = innerUV;
    barUV.y -= uScroll * 0.2 + time * 0.1;
    float bars = sin(barUV.x * 1000.0) * cnoise(barUV*30.0 + time * 0.2 + abs(barUV.y-0.5) * 4.0);
    color += vec4(mix(rainbow, vec3(1.0), 0.5), step(0.9, bars)) * mix(abs(uScrollDelta) * 0.03 + 0.2, 0.3, uActive) * innerClip2 * smoothstep(0.5, 0.2, length(bgUV-0.5));

    color = mix(color, inner, 1.0-innerClip);
    color.a *= mix(1.0, 0.0, uDisabled);

    gl_FragColor = color;
    gl_FragColor.a *= uAlpha;
}`;

/** Cube2Equi.glsl */
export const Cube2Equi_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform samplerCube tCube;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = vec2( 1.- uv.x, uv.y );
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#define M_PI 3.1415926535897932384626433832795

void main() {
    vec2 uv = vUv;
    float longitude = uv.x * 2. * M_PI - M_PI + M_PI / 2.;
    float latitude = uv.y * M_PI;

    vec3 dir = vec3(
        - sin( longitude ) * sin( latitude ),
        cos( latitude ),
        - cos( longitude ) * sin( latitude )
    );

    normalize(dir);
    gl_FragColor = textureCube(tCube, dir);
}`;

/** DebugCamera.glsl */
export const DebugCamera_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;

#!VARYINGS
varying vec3 vColor;

#!SHADER: DebugCamera.vs
void main() {
    vColor = mix(uColor, vec3(1.0, 0.0, 0.0), step(position.z, -0.1));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: DebugCamera.fs
void main() {
    gl_FragColor = vec4(vColor, 1.0);
}`;

/** LabLogoShader.glsl */
export const LabLogoShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(transformUV.glsl)

void main() {
    vec2 uv = vUv;
    vec4 color = texture2D(tMap, uv);

    color.a *= 0.7 + sin(-time + length(uv-0.5) * 10.0) * 0.3;

    color.a = mix(color.a, 1.0, smoothstep(0.8, 1.0, uAlpha));
    
    gl_FragColor = color;
    gl_FragColor.a *= uAlpha;
}`;

/** LightBlur.fs */
export const LightBlur_fs = `uniform vec2 uDir;

#require(gaussianblur.fs)

void main() {
    gl_FragColor = blur9(tDiffuse, vUv, resolution, uDir);
}`;

/** LitMaterial.glsl */
export const LitMaterial_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;

#!SHADER: Vertex

#require(lighting.vs)

void main() {
    vUv = uv;
    vPos = position;
    setupLight(position);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(lighting.fs)
#require(shadows.fs)

void main() {
    setupLight();

    vec3 color = texture2D(tMap, vUv).rgb;
    color *= getShadow(vPos);

    color += getCombinedColor();

    gl_FragColor = vec4(color, 1.0);
}`;

/** LoaderBGShader.glsl */
export const LoaderBGShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uAlpha;
uniform float uVisible;
uniform float uScrollDelta;
uniform float uBottom;
uniform float uProgress;
uniform float uHeight;
uniform float uMobile;
uniform float uBars;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(transformUV.glsl)
#require(simplenoise.glsl)
#require(range.glsl)
#require(rgb2hsv.fs)

// float waveform(vec2 uv, float t) {
//     vec2 waveformUV = uv;
//     // return smoothstep(0.0 + smoothstep(0.6, 0.0, abs(uv.x-0.5)) * 0.08, 0.0, abs(waveformUV.y-0.5)) * 0.5;
// }

const float PI = 3.141592653589793;

float drawLine(vec2 uv, float offset) {
    vec2 circleUv = uv - 0.5;
    // radius of circle
    float r = 0.2; 
    
    // thickness of circle
    float t = 0.0015; 
    
    // half angle
    float a = (35.0 + 17.0*cos(-3.0*t + offset));

    // vector from the circle origin to the middle of the arc
    vec2 up = vec2(cos(1.0*t + offset), sin(-1.0*t + offset));
        
    // cos(angle/2.0), where 'angle' is the full arc length
    float c = cos(a*3.1416/180.0); 
    // in particular:
    // c =  1.0 gives a 0 degree arc, 
    // c =  0.0 gives a 180 degree arc, 
    // c = -1.0 gives a 360 degree arc

    c = -0.4;
    
    // smoothing perpendicular to the arc
    float d1 = abs(length(circleUv) - r) - t;
    float w1 = 2.0*fwidth(d1); // proportional to how much 'd1' change between pixels
    float s1 = smoothstep(w1/2.0, -w1/2.0, d1); 

    // smoothing along the arc
    float d2 = dot(up, normalize(circleUv)) - c;
    float w2 = 2.0*fwidth(d2); // proportional to how much 'd2' changes between pixels
    float s2 = smoothstep(w2/2.0, -w2/2.0, d2); 

    // mix perpendicular and parallel smoothing
    float s = s1*(1.0 - s2);
    return s;
}

void main() {

    float t = time * 0.1 + uProgress * 4.0 + 1.6;
    vec2 uv = scaleUV(vUv, vec2(1.0, resolution.x/resolution.y));
    uv = scaleUV(uv, vec2(mix(0.9, 1.5, uMobile) + (1.0-uVisible) * 0.2));
    uv = rotateUV(uv, uVisible * 3.0);

    vec2 gradientUv = uv;
    gradientUv += cnoise(uv*2.0 - t * 0.1 + length(uv-0.5) * 2.0) * 0.01;
    vec4 color = vec4(vec3(0.0), 1.0);

    vec2 barUV = scaleUV(vUv, vec2(1.0, 1.0));
    barUV = scaleUV(barUV, vec2(1.0 + sin(t * 0.5 - length(barUV-0.5)*30.0) * mix(0.1, 0.4, smoothstep(0.3, 0.1, length(uv-0.5)))));
    barUV = rotateUV(barUV, radians(mix(0.0, 90.0, uMobile)));

    //barUV.x += (1.0/uBars) * 0.5;
    float bars = fract(barUV.x * uBars);
    bars *= (0.5 + sin(t + bars * 10.0 - length(uv-0.5) * 30.0 + t * 1.0) * 0.5);
    
    vec3 rainbow = vec3(0.5, 0.8, 1.0);
    rainbow = rgb2hsv(rainbow);
    rainbow.x += (0.5-bars) * 0.12 + 0.05 + sin(t * 1.0 - bars * 2.0) * 0.05;
    rainbow.x -= length(uv-0.5) * 0.2;
    rainbow.y *= 0.9;
    rainbow = hsv2rgb(rainbow);

    rainbow *= step(0.85, bars) * smoothstep(0.5, 0.3, length(uv-0.5));
    rainbow = mix(rainbow, vec3(1.0), step(0.98, bars) - step(0.9, bars));
    //rainbow *= step(0.92, bars);

    color.rgb += rainbow * step(0.02*uProgress, length(uv-0.5)) * step(0.08, length(uv-0.5)) * (1.0-step(0.4, length(uv-0.5))) * 0.9; 

    // Gradient Corners
    vec3 gradient = vec3(0.5, 0.4, 1.0);
    gradient = rgb2hsv(gradient);
    gradient.x += cnoise(vUv*2.5 - t * 0.04) * 0.05 + 0.87;
    gradient.y *= 0.9;
    gradient = hsv2rgb(gradient);
    //float gNoise = (0.5 + cnoise(uv*2.0 + t * 0.2) * 0.5);
    color.rgb += gradient * step(0.08, length(uv-0.5)) * smoothstep(0.09, 0.08, length(uv-0.5)) * (0.5 + sin(t * 4.0) * 0.3) * 0.5;
    //color.rgb += gradient * 0.5 * smoothstep(0.2, 0.7, length(uv-0.5));

    color.rgb += drawLine(scaleUV(uv, vec2(0.4)), 0.0) * gradient * 0.5;//
    color.rgb += drawLine(scaleUV(uv, vec2(0.4)), 2.0) * gradient * 0.5;///
    color.rgb += drawLine(scaleUV(uv, vec2(0.4)), 4.0) * gradient * 0.5;//
    color.rgb += step(0.2, bars) * 0.06 * smoothstep(0.5, 0.0, length(uv-0.5)) * step(0.08, length(uv-0.5));

    color.rgb *= smoothstep(0.0, 0.4, uProgress);
    color.rgb *= uVisible;
    color.rgb -= getNoise(uv, t) * 0.2;
    color.rgb += 0.03;

    color.rgb *= smoothstep(0.4, 0.0, length(vUv-0.5));

    gl_FragColor = color;
    gl_FragColor.a *= uAlpha;
}`;

/** LoaderBGShader2.glsl */
export const LoaderBGShader2_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uAlpha;
uniform float uVisible;
uniform float uScrollDelta;
uniform float uBottom;
uniform float uProgress;
uniform float uHeight;
uniform float uMobile;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(transformUV.glsl)
#require(simplenoise.glsl)
#require(range.glsl)
#require(rgb2hsv.fs)


void main() {
    vec2 uv = scaleUV(vUv, vec2(1.0, resolution.x/resolution.y));
    uv = scaleUV(uv, vec2(mix(0.8, 1.5, uMobile) - (1.0-uVisible) * 0.2));
    uv = scaleUV(uv, vec2(1.0 + sin(time * 4.0 - length(uv-0.5) * 20.0) * 0.05));


    vec3 color = vec3(0.0);
    float len = length(uv-0.5);

    float wave = 0.5 + sin(time * 2.0 - length(uv-0.5) * 20.0) * 0.5;


    float lines = step(0.9, fract(uv.x * resolution.x * 0.025));
    lines *= (0.5 + sin(time + fract(uv.y * resolution.y * 0.025 + time * 2.0 + cnoise(uv - time * 0.5 + len * 50.0)) * 10.0) * 0.5);

    color += wave * lines * step(0.1, len);
    color += smoothstep(0.105, 0.1, len) * 0.5;



    color *= step(0.1, len) * smoothstep(0.7, 0.0, len);



    gl_FragColor = vec4(color, uAlpha);
}`;

/** LoaderBGShader3.glsl */
export const LoaderBGShader3_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uAlpha;
uniform float uVisible;
uniform float uScrollDelta;
uniform float uBottom;
uniform float uProgress;
uniform float uHeight;
uniform float uMobile;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(transformUV.glsl)
#require(simplenoise.glsl)
#require(range.glsl)
#require(rgb2hsv.fs)

// float waveform(vec2 uv, float time) {
//     vec2 waveformUV = uv;
//     // return smoothstep(0.0 + smoothstep(0.6, 0.0, abs(uv.x-0.5)) * 0.08, 0.0, abs(waveformUV.y-0.5)) * 0.5;
// }

const float PI = 3.141592653589793;

float drawLine(vec2 uv, float offset) {
    vec2 circleUv = uv - 0.5;
    float r = 0.2; 
    float t = 0.003; 
    float a = (35.0 + 17.0*cos(-3.0*time + offset));
    vec2 up = vec2(cos(-3.0*time + offset), sin(-3.0*time + offset));
    float c = cos(time - radians(81.0 + time * 1.0)); 
    
    float d1 = abs(length(circleUv) - r) - t;
    float w1 = 2.0*fwidth(d1); // proportional to how much 'd1' change between pixels
    float s1 = smoothstep(w1/2.0, -w1/2.0, d1); 

    float d2 = dot(up, normalize(circleUv)) - c;
    float w2 = 2.0*fwidth(d2); // proportional to how much 'd2' changes between pixels
    float s2 = smoothstep(w2/2.0, -w2/2.0, d2); 

    float s = s1*(1.0 - s2);
    return s;
}

void main() {
    vec2 uv = scaleUV(vUv, vec2(1.0, resolution.x/resolution.y));
    uv = scaleUV(uv, vec2(mix(0.8, 1.5, uMobile) - (1.0-uVisible) * 0.2));
    uv = scaleUV(uv, vec2(1.0 + sin(time * 3.0 - length(uv-0.5) * 20.0) * 0.06));

    //uv = rotateUV(uv, uVisible * 3.0);

    // Gradient Corners
    vec3 gradient = vec3(0.5, 0.4, 1.0);
    gradient = rgb2hsv(gradient);
    gradient.x += cnoise(vUv*3.0 + length(uv-0.5) * 30.0 - time * 0.2) * 0.1 + 0.8;
    gradient.y *= 0.8;
    gradient = hsv2rgb(gradient);

    vec3 color = vec3(0.0);

    color += drawLine(scaleUV(uv, vec2(0.4)), 0.0) * gradient * 0.5;//
    color += drawLine(scaleUV(uv, vec2(0.4)), 2.0) * gradient * 0.5;///
    color += drawLine(scaleUV(uv, vec2(0.4)), 4.0) * gradient * 0.5;//

    float len = length(uv-0.5);
    float wave = 0.5 + sin(time * 1.0 - length(uv-0.5) * 20.0) * 0.5;
    float lines = step(0.85, fract(uv.x * resolution.x * 0.02));
    lines *= (0.5 + sin(time + fract(uv.y * resolution.y * 0.005 + time * 1.0 + cnoise(uv - time * 0.5 + len * 50.0) * 0.05) * 10.0) * 0.5);
    color += wave * lines * step(0.1, len) * gradient * smoothstep(0.7, 0.1, len);

    color.rgb *= smoothstep(0.0, 0.4, uProgress);
    color.rgb *= uVisible;
    color.rgb -= getNoise(uv, time) * 0.2;

    color.rgb += 0.05;

    gl_FragColor = vec4(color, uAlpha);
}`;

/** LoaderBGShader4.glsl */
export const LoaderBGShader4_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uAlpha;
uniform float uVisible;
uniform float uScrollDelta;
uniform float uBottom;
uniform float uProgress;
uniform float uHeight;
uniform float uMobile;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(transformUV.glsl)
#require(simplenoise.glsl)
#require(range.glsl)
#require(rgb2hsv.fs)
#require(blendmodes.glsl)

const float PI = 3.141592653589793;

float drawLine(vec2 uv, float offset) {
    vec2 circleUv = uv - 0.5;
    float r = 0.2; 
    float t = 0.003; 
    float a = (35.0 + 17.0*cos(-3.0*time + offset));
    vec2 up = vec2(cos(-3.0*time + offset), sin(-3.0*time + offset));
    float c = cos(time - radians(81.0 + time * 1.0)); 
    
    float d1 = abs(length(circleUv) - r) - t;
    float w1 = 2.0*fwidth(d1); // proportional to how much 'd1' change between pixels
    float s1 = smoothstep(w1/2.0, -w1/2.0, d1); 

    float d2 = dot(up, normalize(circleUv)) - c;
    float w2 = 2.0*fwidth(d2); // proportional to how much 'd2' changes between pixels
    float s2 = smoothstep(w2/2.0, -w2/2.0, d2); 

    float s = s1*(1.0 - s2);
    return s;
}

float waveform(vec2 uv, float time) {
    vec2 waveformUV = uv;
    waveformUV.y += sin(waveformUV.x * 10.0 + time * 4.0 + (1.0-uVisible) * 2.0) * mix(0.012, 0.022, smoothstep(0.5, 0.0, vUv.y));// * smoothstep(0.7, 0.0, abs(waveformUV.x-0.5));
    float wave = smoothstep(0.0 + smoothstep(0.6, 0.2, abs(uv.x-0.5)) * 0.03, 0.0, abs(waveformUV.y-0.5)) * 0.5;
    return wave;
}

void main() {
    vec3 color = vec3(0.0);

    vec2 uv = rotateUV(vUv, radians(90.0));
    uv.x += (0.63 - 0.3 * (1.0-uVisible));

    float t = time * 0.5;
    color += waveform(uv, t);
    color += waveform(uv, t + sin(t * 2.0 + uv.x * 1.0) * 0.3);
    color += waveform(uv, t + cos(t * 2.0 + uv.x * 1.0) * 0.3);

    // vec2 lineUV = vUv;
    // lineUV.y += 0.02 * (1.0-pow(uVisible, 5.0));
    // color += drawLine(scaleUV(lineUV, vec2(0.4)), 0.0) * 0.4 * pow(uVisible, 20.0);//
    // color += drawLine(scaleUV(lineUV, vec2(0.4)), 2.0) * 0.4 * pow(uVisible, 20.0);///
    // color += drawLine(scaleUV(lineUV, vec2(0.4)), 4.0) * 0.4 * pow(uVisible, 20.0);//

    color.rgb -= getNoise(uv, time) * 0.5;
    color.rgb += 0.1;
    vec3 gradient = vec3(0.1, 1.0, 1.0);

    gradient = rgb2hsv(gradient);
    gradient.x += color.r * 0.45 - 0.15;
    gradient = hsv2rgb(gradient);

    color.rgb = blendOverlay(color.rgb, gradient, 0.8);




    //color.rgb *= smoothstep(-0.5, 0.4, vUv.y * uAlpha);

    //color = mix(color, vec3(0.0, 1.0, 1.0), pow(color.r, 5.0));

    gl_FragColor = vec4(color, uAlpha);
}`;

/** NavAudioShader.glsl */
export const NavAudioShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uScroll;
uniform float uAmplitude;
uniform float uAlpha;
uniform float uHover;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(simplenoise.glsl)
#require(range.glsl)
#require(rgb2hsv.fs)


float waveform(vec2 uv, float time) {
    vec2 waveformUV = uv;
    waveformUV.y += sin(waveformUV.x * 6.0 + time * 4.0) * mix(0.06, 0.09, uHover) * uAmplitude - 0.03;// * smoothstep(0.7, 0.0, abs(waveformUV.x-0.5));

    float wave = smoothstep(0.0 + smoothstep(0.6, 0.2, abs(uv.x-0.5)) * 0.05, 0.0, abs(waveformUV.y-0.5)) * mix(0.5, 0.7, uHover);

    wave = mix(wave, smoothstep(0.01, 0.0, abs(waveformUV.y-0.5)), 1.0-uAmplitude);

    return wave;//* smoothstep(0.0, 1.0, uv.x);
    //return step(0.25, mod(waveformUV.y, 0.3));
}

void main() {
    vec2 uv = vUv;

    vec3 color = uColor;

    vec3 rainbow = vec3(0.7, 0.8, 1.0);
    rainbow = rgb2hsv(rainbow);
    rainbow.x += sin(uv.x * 5.0 + time * 3.0) * 0.08;
    rainbow = hsv2rgb(rainbow);

    float alpha = 0.0;
    float t = time * 0.5 + uScroll * 0.3;
    alpha += waveform(uv, t);
    alpha += waveform(uv, t + sin(t * 2.0 + uv.x * 1.0) * 0.4);
    alpha += waveform(uv, t + cos(t * 2.0 + uv.x * 1.0) * 0.4);
    //alpha += waveform(uv, time + 2.0);
    // alpha = 0.5 + sin(uv.x * 5.0 + time * 10.0) * 0.5;
    // alpha *= 0.5 + cos(abs(uv.y-0.5) * 20.0 + time * 10.0) * 0.5;

    alpha *= uAlpha;

    color = mix(color, rainbow, smoothstep(1.0, -1.0, abs(alpha-0.5)));

    gl_FragColor = vec4(color, alpha);
}`;

/** NavBGShader.glsl */
export const NavBGShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uAlpha;
uniform float uScroll;
uniform float uScrollDelta;
uniform float uBottom;
uniform float uDisabled;
uniform float uHeight;
uniform vec3 uUIColor;
uniform float uUIBlend;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(transformUV.glsl)
#require(simplenoise.glsl)
#require(range.glsl)
#require(rgb2hsv.fs)

float innerRoundedRectangle(vec2 p, vec2 size, float radius, float stepper) {
    vec2 q = abs(p) - size + radius;
    return step(stepper, length(max(q, 0.0)) - radius);
}

void main() {
    vec2 uv = vUv;
    vec4 color = vec4(0.0);

    //float innerClip = step(0.4, abs(uv.x-0.5));
    //innerClip = max(innerClip, step(0.3, abs(uv.y-0.5)));
    
    vec2 innerUV = uv;
    innerUV.y += smoothstep(1.1, 0.0, abs(uv.x-0.5)) * 0.03 * uScrollDelta;

    //innerUV += cnoise(uv*2.0+time*0.2) * 0.01;
    vec2 innerScale = vec2(0.7, uHeight);
    vec2 innerOffset = vec2(1.0, 1.0 + uScrollDelta * 0.03);
    float innerClip = innerRoundedRectangle(scaleUV(innerUV, vec2(0.5, 0.5), innerOffset), innerScale, innerScale.y, 0.0);
    float innerClip2 = innerRoundedRectangle(scaleUV(innerUV, vec2(0.5, 0.5), innerOffset), innerScale, innerScale.y, 0.01);

    vec2 bgUV = scaleUV(innerUV, vec2(1.0, 0.4), vec2(1.0, mix(0.45, 0.57, uBottom) + uScrollDelta * 0.025));

    vec3 rainbow = vec3(0.65, 1.0, 0.9);
    rainbow = rgb2hsv(rainbow);
    rainbow.x += cnoise(-bgUV*0.5-bgUV.y*0.5-time*0.05-uScroll*0.3+length(bgUV-0.2)*0.2) * 0.2;
    rainbow = hsv2rgb(rainbow);

    rainbow = mix(rainbow, mix(uUIColor * 1.2, vec3(1.0), 0.2), uUIBlend * 0.8);

    color = mix(color, vec4(rainbow, 0.5), smoothstep(0.65, abs(uScrollDelta * 0.02) - 0.2, length(bgUV-0.5)));
    color = mix(color, vec4(rainbow, 0.8), smoothstep(0.25, 0.0, length(bgUV-0.5)));
    color = mix(color, vec4(rainbow, 0.3 + abs(uScrollDelta * 0.08)), 1.0-innerClip2);

    vec4 inner = vec4(uColor, 0.7);
    vec2 barUV = scaleUV(vUv, vec2(1.0, 1.0));
    barUV.y -= uScroll * 0.2 - time * 0.02;
    float bars = sin(barUV.x * 500.0) * cnoise(barUV*30.0 + time * 0.2 + abs(barUV.y-0.5) * 4.0);
    color += vec4(mix(rainbow, vec3(1.0), 0.5), step(0.9, bars)) * (abs(uScrollDelta) * 0.05 + 0.2) * innerClip2 * smoothstep(0.7, 0.2, length(bgUV-0.5));

    color = mix(color, inner, 1.0-innerClip);
    color.a *= mix(1.0, 0.1, uDisabled);

    gl_FragColor = color;
    gl_FragColor.a *= uAlpha;
}`;

/** OcclusionMaterial.glsl */
export const OcclusionMaterial_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 bbMin;
uniform vec3 bbMax;

#!VARYINGS

#!SHADER: Vertex.vs
void main() {
    vec3 pos = position;
    pos *= bbMax - bbMin;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment.fs
void main() {
    gl_FragColor = vec4(1.0);
}`;

/** ScreenQuadVR.glsl */
export const ScreenQuadVR_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uEye;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex

vec2 scaleUV(vec2 uv, vec2 scale, vec2 origin) {
    vec2 st = uv - origin;
    st /= scale;
    return st + origin;
}

void main() {
    vUv = scaleUV(uv, vec2(2.0, 1.0), vec2(0.0)) - vec2(uEye, 0.0);
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment
void main() {
    gl_FragColor = texture2D(tMap, vUv);
}`;

/** ShadowInspector.glsl */
export const ShadowInspector_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(depthvalue.fs)

void main() {
    gl_FragColor = vec4(vec3(getDepthValue(tMap, vUv, 10.0, 51.0)), 1.0);
}`;

/** TestMaterial.glsl */
export const TestMaterial_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform float alpha;

#!VARYINGS
varying vec3 vNormal;

#!SHADER: TestMaterial.vs
void main() {
    vec3 pos = position;
    vNormal = normalMatrix * normal;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: TestMaterial.fs
void main() {
    gl_FragColor = vec4(vNormal, 1.0);
}`;

/** TextureMaterial.glsl */
export const TextureMaterial_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;

#!SHADER: TextureMaterial.vs
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: TextureMaterial.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.rgb /= gl_FragColor.a;
}`;

/** TriangleParticleShader.glsl */
export const TriangleParticleShader_glsl = `#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS

#!SHADER: Vertex
void main() {
    vec3 pos = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment
void main() {
    gl_FragColor = vec4(1.0);
}`;

/** UnrealBloomGaussian.glsl */
export const UnrealBloomGaussian_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D colorTexture;
uniform vec2 texSize;
uniform vec2 direction;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex.vs
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment.fs

float gaussianPdf(in float x, in float sigma) {
    return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

void main() {
    vec2 invSize = 1.0 / texSize;
    float fSigma = float(SIGMA);
    float weightSum = gaussianPdf(0.0, fSigma);
    vec3 diffuseSum = texture2D( colorTexture, vUv).rgb * weightSum;
    for(int i = 1; i < KERNEL_RADIUS; i ++) {
        float x = float(i);
        float w = gaussianPdf(x, fSigma);
        vec2 uvOffset = direction * invSize * x;
        vec3 sample1 = texture2D( colorTexture, vUv + uvOffset).rgb;
        vec3 sample2 = texture2D( colorTexture, vUv - uvOffset).rgb;
        diffuseSum += (sample1 + sample2) * w;
        weightSum += 2.0 * w;
    }
    gl_FragColor = vec4(diffuseSum/weightSum, 1.0);
}`;

/** UnrealBloomPass.fs */
export const UnrealBloomPass_fs = `#require(UnrealBloom.fs)

void main() {
    vec4 color = texture2D(tDiffuse, vUv);
    color.rgb += getUnrealBloom(vUv);
    gl_FragColor = color;
}`;

/** WorkItemUIShader.glsl */
export const WorkItemUIShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;
uniform vec3 uColor;
uniform float uHover;
uniform float uCamDistance;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;
varying vec3 vCameraPos;
varying vec3 vViewDir;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vec3 pos = position;

    pos.z += 0.1 + smoothstep(1.0, 0.0, abs(vUv.x-0.5)) * 0.3 + uHover * 0.2;

    pos.y -= (-0.5 + vUv.x) * 0.42;
    vPos = pos;
    vWorldPos = vec3(modelMatrix * vec4(pos, 1.0));
    vCameraPos = cameraPosition;
        vViewDir = -vec3(modelViewMatrix * vec4(pos, 1.0));

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(transformUV.glsl)
#require(range.glsl)
#require(rgbshift.fs)
#require(rgb2hsv.fs)


void main() {
    vec2 uv = scaleUV(vUv, vec2(0.42, 1.0));
    uv = scaleUV(uv, vec2(1.15));

    //uv += fract(uv.y * 20.0) * smoothstep(1.0, 0.0, length(vWorldPos-vCameraPos));
    //uv.x += length(vWorldPos.x-vCameraPos.x);

    uv.y -= (-0.5 + uv.x) * 0.15;
    uv.x -= (0.5 - vViewDir.x) * 0.1;
    uv.y += 0.02;

    float edges = smoothstep(0.9 + uHover * 1.0, 7.0, abs(vViewDir.x-0.5));
    uv.x += fract(uv.x * 15.0) * edges;
    uv.y -= uv.y * vViewDir.x * 0.15 * edges;

    vec3 color = getRGB(tMap, uv, radians(180.0), 0.001 - edges * 0.15).rgb;
    color *= smoothstep(0.5, 0.4, abs(uv.x-0.5));
    color *= smoothstep(0.5, 0.4, abs(uv.y-0.5));


    vec3 base = rgb2hsv(uColor);

    vec3 color2 = rgb2hsv(color);
    color2.x = color2.x * 0.1 + base.x - 0.25;
    color2.y *= base.y;
    color2 = hsv2rgb(color2);
    

    color = mix(color, color2, 0.05);

    vec2 lineUV = vUv;
    lineUV.y -= (-0.5 + uv.x) * 0.05;
    lineUV.x -= (0.5 - vViewDir.x) * 0.05;

    //color *= smoothstep(0.15, 0.3, abs(edges-0.5));

    // float lines = smoothstep(0.01, 0.0, abs(fract(lineUV.y*7.0)-0.5));
    // lines *= pow(0.5 + sin(lineUV.y * 10.0 + vViewDir.x * 2.0 - time * 2.0) * 0.5, 4.0);
    // lines *= smoothstep(0.5, 0.4, abs(lineUV.x - 0.5));
    // lines *= smoothstep(0.5, 0.0, abs(lineUV.y - 0.5));
    // color += vec3(lines) * 0.4;

    // color *= 0.2;

    color *= crange(uCamDistance, 5.0, 6.0, 1.0, 0.0);
    
    gl_FragColor = vec4(color * mix(0.65, 0.9, uHover), 1.0);
}`;

/** WorkPanelShader.glsl */
export const WorkPanelShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

void main() {
    vUv = uv;
    vPos = position;
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(blendmodes.glsl)
#require(transformUV.glsl)

float roundedBox(vec2 p, vec2 c, float r) {
  return length(max(abs(p-c),0.0))-r;
}

void main() {
    vec2 screenUV = gl_FragCoord.xy / resolution.xy;
    vec2 fluid = texture2D(tFluid, screenUV).xy;
    float fluidMask = smoothstep(0.0, 1.0, texture2D(tFluidMask, screenUV).r);
    float fluidEdge = pow(abs(fluid.x)*0.01, 2.0);

    vec2 uv = vUv;
    
    float rounded = roundedBox(uv, scaleUV(uv, vec2(0.55, 0.6)), 0.505);
    if (rounded > 0.0) discard;

    uv += fluidEdge * 0.1;
    

    vec3 color = texture2D(tMap, uv).rgb * 0.8;
    color *= mix(1.0, 1.6, smoothstep(0.2, 1.0, length(uv-0.5)));

    //color = blendOverlay(color, vec3(0.25), smoothstep(0.0, 2.0, border) + step(0.95, border));
    float alpha = uAlpha;

    alpha *= smoothstep(0.5, 0.0, fluidEdge);

    gl_FragColor = vec4(color * 0.8, alpha * 0.9);
}`;

/** gluimask.fs */
export const gluimask_fs = `uniform vec4 uMaskValues;

#require(range.glsl)

vec2 getMaskUV() {
    vec2 ores = gl_FragCoord.xy / resolution;
    vec2 uv;
    uv.x = range(ores.x, uMaskValues.x, uMaskValues.z, 0.0, 1.0);
    uv.y = 1.0 - range(1.0 - ores.y, uMaskValues.y, uMaskValues.w, 0.0, 1.0);
    return uv;
}`;
