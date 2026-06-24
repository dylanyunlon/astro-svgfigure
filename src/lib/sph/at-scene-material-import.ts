/**
 * at-scene-material-import.ts — M853: AT scene material shaders
 * Core materials from AT site — mapped to cell visual themes.
 */









export const JellyShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tMatcap;
uniform sampler2D tVideo;
uniform sampler2D tRefraction;
uniform vec2 uReflection;
uniform float uScroll;
uniform float uDirection;
uniform vec2 uMouse;

#!VARYINGS
varying vec3 vCameraPos;
varying vec3 vViewDir;
varying vec4 vWorldPos;
varying float vDist;
varying float vMouse;

#!SHADER: Vertex

#require(fbr.vs)
#require(simplenoise.glsl)

void main() {
    vec3 pos = position;

    pos.y += cnoise(pos * vec3(0.1, 0.5, 0.1) * 0.8 + time * 0.5 * 0.35) * 0.6;

    pos.x += sin(pos.y + time * 0.1 + uScroll) * 0.1;
    pos.z += cos(pos.y + time * 0.1 + uScroll) * 0.1;

    pos.x += sin(pos.y * 0.04 + time * 0.2) * 1.0;
    pos.z += cos(pos.y * 0.04 + time * 0.2) * 1.0;

    // pos.x += sin(pos.y * 0.04 + time * 0.1) * 5.0;
    // pos.z += cos(pos.y * 0.04 + time * 0.1) * 5.0;
    vWorldPos = modelMatrix * vec4(pos, 1.0);

    vMouse = smoothstep(2.0, 1.0, length(pos.xy-uMouse));

    setupFBR(pos);
    vNormal = normalMatrix * normal;
    vCameraPos = cameraPosition;
    vDist = length(vWorldPos.xyz - cameraPosition);
    vViewDir = -vec3(modelViewMatrix * vec4(pos, 1.0));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(fbr.fs)
#require(rgb2hsv.fs)
#require(transformUV.glsl)
#require(blendmodes.glsl)
#require(fresnel.glsl)

vec3 rainbowColor(float t) {
    t = mod(t, 1.0); // Wraps the t value between 0.0 and 1.0
    if (t < 0.03) return mix(vec3(0.5, 0.0, 0.5), vec3(0.5, 0.0, 1.0), t / 0.03); // violet to blue
    else if (t < 0.06) return mix(vec3(0.5, 0.0, 1.0), vec3(0.0, 0.0, 1.0), (t - 0.03) / 0.03); // blue to darker blue
    else if (t < 0.09) return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), (t - 0.06) / 0.03); // darker blue to cyan
    else if (t < 0.12) return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.09) / 0.03); // cyan to green
    else if (t < 0.18) return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.12) / 0.06); // green to yellow
    else if (t < 0.24) return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.18) / 0.06); // yellow to orange
    else return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.24) / 0.06); // orange to red
}

void main() {
    vec3 baseColor = texture2D(tMap, vUv).rgb;
    vec3 color = getFBR(baseColor, vUv);
    vec3 normal = unpackNormalFBR(vEyePos, vWorldNormal, tNormal, uNormalStrength, 1.0, vUv);
    //normal.y -= vWorldPos.y * 0.01;

    vec2 screenuv = gl_FragCoord.xy / resolution;
        screenuv.y = mix(1.0-screenuv.y, screenuv.y, uDirection);
    screenuv.x = mix(1.0-screenuv.x, screenuv.x, uDirection);

    screenuv += normal.xy * 0.01 * uReflection.x;


    float f = pow(getFresnel(vNormal + normal * 0.02, vViewDir, 1.0), 5.0);
    
    color += f * texture2D(tVideo, vUv + normal.xy * 0.1).rgb * 0.9;
    color += texture2D(tRefraction, screenuv).rgb * uReflection.y;
    color = blendSoftLight(color, vec3(1.0), 1.0);
    color = pow(color * 1.5, vec3(1.8));

    color = mix(color, vec3(1.0), vMouse);

    gl_FragColor = vec4(color, 1.0);
}`;

export const FloorShader_glsl = `#!ATTRIBUTES
attribute vec2 uv2;

#!UNIFORMS
uniform sampler2D tLightmap;
uniform sampler2D tMirrorReflection;
uniform mat4 uMirrorMatrix;
uniform float uMirrorStrength;
uniform float uDistortStrength;
uniform sampler2D tLightReflection;
uniform vec2 uRUVOffset;
uniform float uRUVScale;

#!VARYINGS
varying vec2 vUv2;
varying vec4 vMirrorCoord;
varying vec3 vWorldPos;

#!SHADER: Vertex

#require(fbr.vs)

void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vMirrorCoord = uMirrorMatrix * worldPos;
    vWorldPos = worldPos.xyz;

    setupFBR(position);
    vUv2 = uv2;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(fbr.fs)
#require(radialblur.fs)
#require(luma.fs)
#require(simplenoise.glsl)
#require(range.glsl)
#require(transformUV.glsl)

void main() {
    gl_FragColor = vec4(getFBR(vec3(1.0)), 1.0);
    gl_FragColor.rgb *= crange(getNoise(vUv, time), 0.0, 1.0, 0.5, 1.0);

    vec3 mro = texture2D(tMRO, vUv).rgb;
    vec3 normal = texture2D(tNormal, vUv).rgb;

    vec2 mirrorUV = vMirrorCoord.xy / vMirrorCoord.w;
    mirrorUV += crange(normal.xy, vec2(0.0), vec2(1.0), vec2(-1.0), vec2(1.0)) * uDistortStrength;

    float strength = crange(mro.y, 0.6, 0.7, 0.0, 1.0);
    vec3 reflectionColor = radialBlur(tMirrorReflection, mirrorUV, 15.0 * strength, 5.0) * uMirrorStrength;
    gl_FragColor.rgb += reflectionColor;

    vec3 lightmap = texture2D(tLightmap, vUv2).rgb;
    float lighting = lightmap.g;
    float ao = lightmap.r;
    gl_FragColor.rgb *= ao;
    gl_FragColor.rgb += lighting * 0.15;

    vec3 viewDir = normalize(vWorldPos - cameraPosition);
    vec3 viewProjection = viewDir - dot(viewDir, vNormal) * vNormal;
    float maxViewSkew = radians(30.0);
    vec2 viewSkew;
    viewSkew.x = clamp(viewProjection.x / maxViewSkew, -1.0, 1.0);
    viewSkew.y = -clamp(viewProjection.y / maxViewSkew, -1.0, 1.0);

    vec2 ruv = scaleUV(vUv, vec2(0.2 * uRUVScale));
    ruv += uRUVOffset + (viewSkew*0.2);

    gl_FragColor.rgb += texture2D(tLightReflection, ruv).rgb * 0.5 * crange(strength, 0.0, 1.0, 0.5, 1.0);

    // gl_FragColor = texture2D(tLightReflection, ruv);
}`;

export const WallShader_glsl = `#!ATTRIBUTES
attribute vec2 uv2;

#!UNIFORMS
uniform sampler2D tLightmap;
uniform sampler2D tLightReflection;
uniform vec2 uRUVOffset;
uniform float uRUVScale;

#!VARYINGS
varying vec2 vUv;
varying vec2 vUv2;
varying vec3 vONormal;
varying vec3 vWorldPos;

#!SHADER: Vertex

#require(fbr.vs)

void main() {
    vWorldPos = vec3(modelMatrix * vec4(position, 1.0));
    vUv2 = uv2;
    vUv = uv;
    vONormal = normal;
    setupFBR(position);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(fbr.fs)
#require(simplenoise.glsl)
#require(range.glsl)
#require(transformUV.glsl)

vec3 calculateFixedReflection(vec3 surfacePosition, vec3 lightPosition, vec3 surfaceNormal) {
    // Calculate the vector from the surface to the light source
    vec3 toLight = normalize(lightPosition - surfacePosition);

    // Reflect this vector around the surface normal
    vec3 reflected = reflect(-toLight, surfaceNormal);

    // Return the reflected vector, which will remain constant as long as the light position
    // and surface normal don't change, even if the camera moves
    return reflected;
}

void main() {
    vec3 viewDir = normalize(vWorldPos - cameraPosition);
    vec3 viewProjection = viewDir - dot(viewDir, vNormal) * vNormal;
    float maxViewSkew = radians(30.0);
    vec2 viewSkew;
    viewSkew.y = clamp(viewProjection.y / maxViewSkew, -1.0, 1.0);
    
    vec2 ruvscale = vec2(0.2 * uRUVScale);
    ruvscale.y += crange(viewSkew.y, -1.0, 1.0, 0.0, 1.0);
    vec2 ruv = scaleUV(vUv, ruvscale);
    ruv += uRUVOffset;

    vec3 mro = texture2D(tMRO, vUv).rgb;
    vec3 color = getFBR(vec3(1.0));

    vec3 lightmap = texture2D(tLightmap, vUv2).rgb;
    float lighting = lightmap.g;
    float ao = lightmap.r * 2.2;
    color *= ao;
    color += lighting * 0.25 * crange(mro.y, 0.6, 0.7, 0.8, 1.0);

    //gl_FragColor.rgb *= crange(getNoise(vUv, time), 0.0, 1.0, 0.5, 1.0);
    
    if (vONormal.z > 0.9) {
        color += texture2D(tLightReflection, ruv).rgb * crange(mro.y, 0.6, 0.7, 0.5, 1.0) * 0.025;
    }

    color *= 0.7;


    gl_FragColor = vec4(color, 1.0);
}`;

export const SpineShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tBaseColor;
uniform sampler2D tRefraction;
uniform vec2 uReflection;

#!VARYINGS
varying vec3 vCameraPos;
varying vec4 vWorldPos;

#!SHADER: Vertex

#require(fbr.vs)

void main() {
    vec3 pos = position;
    setupFBR(position);
    vNormal = normalMatrix * normal;
    vCameraPos = cameraPosition;
    vWorldPos = modelMatrix * vec4(position, 1.0);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(fbr.fs)
#require(rgb2hsv.fs)
#require(transformUV.glsl)

void main() {
    vec2 uv = vUv;
    uv.x += vWorldPos.x * 0.2;

    vec3 baseColor = texture2D(tBaseColor, uv).rgb;
    vec3 color = getFBR(baseColor, uv);

    vec3 normal = unpackNormalFBR(vEyePos, vWorldNormal, tNormal, uNormalStrength, 1.0, vUv);

    normal.y -= vWorldPos.y * 0.02;

    vec2 screenuv = gl_FragCoord.xy / resolution;
    screenuv += normal.xy * 0.1 * uReflection.x;

    color += texture2D(tRefraction, screenuv).rgb * uReflection.y;

    #drawbuffer Color gl_FragColor = vec4(color, 1.0);
    #drawbuffer WorkRefraction gl_FragColor = vec4(color, 1.0);
}`;

export const HomeLogoShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tVideo;
uniform sampler2D tNormal;
uniform sampler2D tRefraction;
uniform float uAlpha;
uniform float uNormalScale;
uniform float uScrollDelta;
uniform float uVisible;
uniform float uScroll;
uniform float uFooter;
uniform float uPhone;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec2 vMUV;
varying vec3 vCameraPos;

#!SHADER: Vertex
#require(matcap.vs)

void main() {
    vUv = uv;
    vec3 pos = position;

    pos.x += cos(pos.y * 6.0 + uScrollDelta * 2.0) * 0.005 * uScrollDelta;
    pos.z += sin(pos.y * 6.0 + uScrollDelta * 2.0) * 0.005 * uScrollDelta;

    vPos = pos;
    vWorldPos = vec3(modelMatrix * vec4(pos, 1.0));
    vNormal = normalMatrix * normal;
    vCameraPos = cameraPosition;
    vViewDir = -vec3(modelViewMatrix * vec4(pos, 1.0));
    vMUV = reflectMatcap(vWorldPos, vNormal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(fresnel.glsl)
#require(rgbshift.fs)
#require(radialblur.fs)
#require(normalmap.glsl)
#require(transformUV.glsl)
#require(blendmodes.glsl)
#require(rgb2hsv.fs)

vec3 rainbowColor(float t) {
    t = mod(t, 1.0); // Wraps the t value between 0.0 and 1.0
    if (t < 0.03) return mix(vec3(0.5, 0.0, 0.5), vec3(0.5, 0.0, 1.0), t / 0.03); // violet to blue
    else if (t < 0.06) return mix(vec3(0.5, 0.0, 1.0), vec3(0.0, 0.0, 1.0), (t - 0.03) / 0.03); // blue to darker blue
    else if (t < 0.09) return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), (t - 0.06) / 0.03); // darker blue to cyan
    else if (t < 0.12) return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.09) / 0.03); // cyan to green
    else if (t < 0.18) return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.12) / 0.06); // green to yellow
    else if (t < 0.24) return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.18) / 0.06); // yellow to orange
    else return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.24) / 0.06); // orange to red
}

void main() {
    vec2 uv = vUv;
    vec2 screenuv = gl_FragCoord.xy / resolution;

    vec2 normalUV = scaleUV(mix(screenuv, vUv, 0.5), vec2(0.5)) - vNormal.xy * 0.05 - vViewDir.xy * 0.001;
    //normalUV.y += vCameraPos.y * 0.015;
    normalUV -= time*0.01;
    vec3 normal = crange(texture2D(tNormal, normalUV).rgb, vec3(0.0), vec3(1.0), vec3(-1.0), vec3(1.0));
    uv = rotateUV(uv, vCameraPos.y * 0.2 - 1.5 - time * 0.2);
    uv += normal.xy * 0.02;

    float center = smoothstep(0.4, 0.25, length(vPos));
    float highlight = smoothstep(0.03, 0.0, abs(0.97-uVisible + vPos.y * 0.01));
    
    // Base Color
    vec3 color = texture2D(tRefraction, screenuv - vNormal.xy * 0.05 - normal.xy * 0.005).rgb;

    vec2 baseUv = scaleUV(uv, vec2(2.0)) - vViewDir.xy * 0.05 - vNormal.xy * 0.2;
    color += getRGB(tMap, baseUv, 0.2, 0.002).rgb * smoothstep(0.5, 0.4, abs(0.5-uv.x));

    color *= smoothstep(0.0, 0.1, vUv.x);
    color *= smoothstep(0.75, 0.65, vUv.x);

    // Video Add
    vec3 video = texture2D(tVideo, scaleUV(screenuv, vec2(0.5)) - normal.xy * 0.02).rgb;
    color = blendAdd(color, video, 0.1);
    color = blendSoftLight(color, video, 0.2);

    // Refraction

    // Stylizations
    float f = getFresnel(vNormal, vViewDir, 1.5 + sin(time * 0.1) * 0.3);
    //f += normal.x * 0.1;
    vec3 r = rainbowColor(f*3.0);
    if (r.r > 0.99) r *= 0.0;

    r = rgb2hsv(r);
    r.x += 0.5;
    r = hsv2rgb(r);

    color += r * f * mix(0.8, 2.0, highlight) * mix(0.0, 1.0, uVisible) * 0.4;
    //color *= 1.0 + f * 1.0;
    //color += pow(f, 2.0) * mix(0.4, 0.5, uFooter) * video;
    //color += f * mix(0.2, 0.5, uFooter);
    color += pow(f, 2.0) * mix(0.55, 1.0, uFooter) * mix(video, vec3(1.0), 0.5);
    color += highlight * 0.15;
    //color *= 1.0 + highlight * 5.0 + mix(-0.5, 2.0, center);

    vec3 hueShift = rgb2hsv(color);
    hueShift.y *= 0.9;
    color = hsv2rgb(hueShift);

    color *= mix(0.5, 1.0, uVisible);
    color = pow(color * mix(1.5, 2.5, highlight), vec3(1.8));

    //color *= highlight;
    
    gl_FragColor = vec4(color, uAlpha);
}`;

export const HomeBGShader_glsl = `#!ATTRIBUTES

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

void main() {
    vec2 uv = vUv;
    uv *= 0.5;
    vec4 color = texture2D(tMap, uv);

    color.rgb *= smoothstep(30.0, 0.0, abs(vWorldPos.y-5.0)) * 0.1;
    
    gl_FragColor = color;
    gl_FragColor.a *= uAlpha;
}`;

export const HomeColumnShader_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tVideo;
uniform sampler2D tRefraction;
uniform float uAlpha;
uniform float uVisible;
uniform float uOffset;
uniform float uDirection;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec3 vWorldPos;
varying vec3 vCameraPosition;
varying vec3 vNormal;
varying vec3 vViewDir;
varying float vTop;

#!SHADER: Vertex


void main() {
    vUv = uv;
    vec3 pos = position;

    vTop = smoothstep(9.8, 9.8-2.0*smoothstep(0.8, 1.0, uVisible), pos.y);

    pos.y -= pow((1.0-uVisible), 1.15) * 20.0;

    float radius = mix(1.9, 4.0, smoothstep(10.0, -10.0, pos.y));
    pos.x += cos(-pos.y * 0.32 * uDirection + uOffset) * radius;
    pos.z += sin(-pos.y * 0.32 * uDirection + uOffset) * radius;


    pos.x += cos(-pos.y * 10.0 * uDirection) * 0.1 * pow((1.0-uVisible), 1.25);
    pos.z += sin(-pos.y * 10.0 * uDirection) * 0.1 * pow((1.0-uVisible), 1.25);

    vWorldPos = vec3(modelMatrix * vec4(pos, 1.0));
    vCameraPosition = cameraPosition;
    vPos = pos;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(rgbshift.fs)
#require(blendmodes.glsl)
#require(transformUV.glsl)
#require(fresnel.glsl)

void main() {
    vec2 uv = vUv;
    vec2 screenuv = gl_FragCoord.xy / resolution;
    //uv.y += time * 0.1;
    uv.y *= 1.2;
    uv.y += uVisible;

    vec2 texUV = uv;
    texUV.y += time * 0.1 - cameraPosition.y * 0.03;

    vec4 color = texture2D(tRefraction, screenuv);
    vec3 video = texture2D(tVideo, screenuv).rgb;

    float highlight = smoothstep(0.03, 0.0, abs(0.97-uVisible)) * smoothstep(0.8, 1.0, vUv.y); 

    color.rgb += getRGB(tMap, texUV, 0.2, 0.00001).rgb * 0.5;
    color.rgb = pow(color.rgb * mix(0.8, 1.5, highlight), vec3(1.2));
    color.rgb = blendSoftLight(color.rgb, video, 0.7);
    //color.rgb = blendSoftLight(color.rgb, texture2D(tRefraction, screenuv).rgb, 1.0);
    color.rgb += pow(highlight, 2.0) * 0.5 * video;
    color.rgb *= mix(1.0, 1.5, highlight);

    color.a = mix(vTop, 1.0, highlight*0.3);
    color.rgb = pow(0.1 + color.rgb * mix(1.5, 2.5, highlight) * 1.5, vec3(1.5));

    color.a *= uAlpha;

    
    gl_FragColor = color;
    //gl_FragColor.a *= uAlpha;
}`;
