/**
 * at-text-rendering-import.ts — M858: AT Text3D+GLUI
 * MSDF text + GLUI system for GPU text rendering.
 */









export const DefaultText_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;
uniform vec2 uMouse;

#!VARYINGS

varying vec2 vUv;
varying vec3 vWorldPos;

#!SHADER: DefaultText.vs

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
}

#!SHADER: DefaultText.fs

#require(msdf.glsl)

void main() {
    float transition = smoothstep(0.3, 0.8, uAlpha);
    float gridV = mix(50.0, 500.0, transition);
    vec2 gridSize = vec2(gridV*3.0, floor(gridV/(resolution.x/resolution.y)));
    vec2 uv = floor(vUv * gridSize) / gridSize;
    uv += (1.0-transition) * (1.0/gridV) * vec2(0.2, 0.5);
    uv = mix(uv, vUv,transition);

    float alpha = msdf(tMap, uv);
    alpha *= uAlpha;

    vec3 color = uColor;
    color = mix(color, vec3(0.5, 0.5, 1.0), 0.1 + sin(time - vWorldPos.x * 0.01 + vWorldPos.y * 0.005 + alpha * 10.0) * 0.1);

    alpha *= 0.9 + sin(time*40.0) * 0.1 * smoothstep(0.2, 0.15, abs(uAlpha-0.5));

    gl_FragColor = vec4(color, alpha);

}`;

export const Text3D_glsl = `#!ATTRIBUTES
attribute vec3 animation;

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;
uniform float uOpacity;
uniform vec3 uTranslate;
uniform vec3 uRotate;
uniform float uTransition;
uniform float uWordCount;
uniform float uLineCount;
uniform float uLetterCount;
uniform float uByWord;
uniform float uByLine;
uniform float uPadding;
uniform vec3 uBoundingMin;
uniform vec3 uBoundingMax;
uniform float uScrollDelta;
uniform vec2 uMouse;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;

#!VARYINGS
varying float vTrans;
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

#require(range.glsl)
#require(eases.glsl)
#require(rotation.glsl)
#require(conditionals.glsl)

void main() {
    vUv = uv;
    vTrans = 1.0;

    vec3 pos = position;

    if (uTransition > 0.0 && uTransition < 1.0) {
        float padding = uPadding;
        float letter = (animation.x + 1.0) / uLetterCount;
        float word = (animation.y + 1.0) / uWordCount;
        float line = (animation.z + 1.0) / uLineCount;

        float letterTrans = rangeTransition(uTransition, letter, padding);
        float wordTrans = rangeTransition(uTransition, word, padding);
        float lineTrans = rangeTransition(uTransition, line, padding);

        vTrans = mix(cubicOut(letterTrans), cubicOut(wordTrans), uByWord);
        vTrans = mix(vTrans, cubicOut(lineTrans), uByLine);

        float invTrans = (1.0 - vTrans);
        vec3 nRotate = normalize(uRotate);
        vec3 axisX = vec3(1.0, 0.0, 0.0);
        vec3 axisY = vec3(0.0, 1.0, 0.0);
        vec3 axisZ = vec3(0.0, 0.0, 1.0);
        vec3 axis = mix(axisX, axisY, when_gt(nRotate.y, nRotate.x));
        axis = mix(axis, axisZ, when_gt(nRotate.z, nRotate.x));
        pos = vec3(vec4(position, 1.0) * rotationMatrix(axis, radians(max(max(uRotate.x, uRotate.y), uRotate.z) * invTrans)));
        pos += uTranslate * invTrans;
    }

    vPos = pos;
	vWorldPos = vec3(modelMatrix * vec4(pos, 1.0));

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(msdf.glsl)
#require(simplenoise.glsl)
#require(transformUV.glsl)

vec2 getBoundingUV() {
    vec2 uv;
    uv.x = crange(vPos.x, uBoundingMin.x, uBoundingMax.x, 0.0, 1.0);
    uv.y = crange(vPos.y, uBoundingMin.y, uBoundingMax.y, 0.0, 1.0);
    return uv;
}

void main() {
    vec2 uv = vUv;
    vec2 screenuv = gl_FragCoord.xy / resolution;
    vec2 squareScreenuv = scaleUV(screenuv, vec2(1.0, resolution.x/resolution.y));
    vec2 mouse = scaleUV(vec2(uMouse.x, 1.0-uMouse.y), vec2(1.0, resolution.x/resolution.y));

    mouse += cnoise(screenuv*10.0 + time * 0.2 + length(mouse) * 5.0) * 0.01;

    vec2 fluid = texture2D(tFluid, screenuv).xy;
    float fluidMask = smoothstep(0.0, 1.0, texture2D(tFluidMask, screenuv).r);
    float fluidPush = pow(abs(fluid.x)*0.01, 2.5);
    float fluidEdge = fluidPush * smoothstep(0.0, 0.5, fluidMask) * smoothstep(1.0, 0.8, fluidMask);

    //uv.y -= uScrollDelta * 0.1 * mix(-1.0, 1.0, step(0.05, mod(uv.x, 0.5))) * mod(uv.y, 0.3);
    uv += fluidEdge * 0.1;

    float alpha = msdf(tMap, uv);

    //float noise = 0.5 + smoothstep(-1.0, 1.0, cnoise(vec3(vUv*50.0, time* 0.3))) * 0.5;

    vec4 color = vec4(uColor, alpha * uAlpha * uOpacity * vTrans);

    float mouseLen = (1.0-step(0.1, length(squareScreenuv-mouse)));

    // float lines = sin(screenuv.x * resolution.x * 0.5) * (0.5 + cnoise(screenuv*30.0 + time * 0.2));
    // lines = step(0.2, lines);

    vec2 lineUV = screenuv + fluidPush * 0.1;
    float lines = fract(screenuv.x * 300.0) * fract(screenuv.y * 300.0);
    lines = step(0.7, lines);
    color.a = mix(color.a, lines, fluidEdge);

    #drawbuffer Color gl_FragColor = color;
    #drawbuffer Refraction gl_FragColor = color;
}`;

export const GLUIBatch_glsl = `#!ATTRIBUTES
attribute vec3 offset;
attribute vec2 scale;
attribute float rotation;
//attributes

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;
//varyings

#!SHADER: Vertex

mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
    0.0,                                0.0,                                0.0,                                1.0);
}

void main() {
    vUv = uv;
    //vdefines

    vec3 pos = vec3(rotationMatrix(vec3(0.0, 0.0, 1.0), rotation) * vec4(position, 1.0));
    pos.xy *= scale;
    pos.xyz += offset;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment
void main() {
    gl_FragColor = vec4(1.0);
}`;

export const GLUIBatchText_glsl = `#!ATTRIBUTES
attribute vec3 offset;
attribute vec2 scale;
attribute float rotation;
//attributes

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;
//varyings

#!SHADER: Vertex

mat4 lrotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
    0.0,                                0.0,                                0.0,                                1.0);
}

void main() {
    vUv = uv;
    //vdefines

    vec3 pos = vec3(lrotationMatrix(vec3(0.0, 0.0, 1.0), rotation) * vec4(position, 1.0));

    //custommain

    pos.xy *= scale;
    pos += offset;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(msdf.glsl)

void main() {
    float alpha = msdf(tMap, vUv);

    gl_FragColor.rgb = v_uColor;
    gl_FragColor.a = alpha * v_uAlpha;
}`;

export const GLUIColor_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;

#!SHADER: GLUIColor.vs
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: GLUIColor.fs
void main() {
    vec2 uv = vUv;
    vec3 uvColor = vec3(uv, 1.0);
    gl_FragColor = vec4(mix(uColor, uvColor, 0.0), uAlpha);
}`;

export const GLUIObject_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;
varying vec3 vWorldPos;

#!SHADER: GLUIObject.vs
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
}

#!SHADER: GLUIObject.fs

#require(transformUV.glsl)

void main() {
    // float transition = smoothstep(0.0, 0.8, uAlpha);
    // float gridV = mix(20.0, 100.0, transition);
    // vec2 gridSize = vec2(gridV, floor(gridV/(resolution.x/resolution.y)));
    // vec2 uv = floor(vUv * gridSize) / gridSize;
    // uv += (1.0-transition) * (1.0/gridV) * 0.4;
    // uv = mix(uv, vUv,transition);

    vec4 color = texture2D(tMap, vUv);
    color.a *= 0.8 + sin(time * 2.0 + vUv.y * 2.0 - vWorldPos.x * 0.02) * 0.2;
    color.a *= uAlpha;
    gl_FragColor = color;
}`;
