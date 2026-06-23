/**
 * at-glass-pbr-import.ts — M854: AT Glass+PBR materials
 * Glass reflections + PBR material system.
 */

export const CleanRoomGlass_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tRefraction;
uniform sampler2D tEnv;
uniform sampler2D tInner;
uniform float uFresnelPow;
uniform float uDistortStrength;
uniform float uRefractionRatio;

#!VARYINGS
varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflection;
varying vec3 vRefraction;
varying vec3 vPos;

#!SHADER: Vertex

#require(refl.vs)

void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);

    vReflection = reflection(worldPos);
    vRefraction = refraction(worldPos, uRefractionRatio);

    vPos = position;
    vWorldPos = worldPos.xyz;
    vNormal = normalMatrix * normal;
    vViewDir = -vec3(modelViewMatrix * vec4(position, 1.0));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(fresnel.glsl)
#require(eases.glsl)
#require(refl.fs)
#require(rgbshift.fs)

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
    float f = getFresnel(vNormal, vViewDir, uFresnelPow);
    vec3 r = rainbowColor(f * 4.0);
    if (r.r > 0.99) r *= 0.0;

    vec2 uv = gl_FragCoord.xy / resolution;
    uv += 0.1 * vNormal.xy * f * uDistortStrength;

    gl_FragColor = getRGB(tRefraction, uv, 0.3, 0.002);
    gl_FragColor.rgb += r;
    // gl_FragColor = envColorEqui(tEnv, vReflection);
    gl_FragColor += envColorEquiRGB(tEnv, vRefraction, 0.2, 1.0);
    gl_FragColor.rgb += cnoise(vViewDir + 2.0) * 0.1;
    gl_FragColor.rgb += texture2D(tInner, gl_FragCoord.xy / resolution).r;
    gl_FragColor.rgb += quarticIn(crange(abs(vPos.x), 0.5, 0.3, 1.0, 0.0) * crange(abs(vPos.z), 0.5, 0.3, 1.0, 0.0)) * 0.05;

    gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(1.5));

    if (vNormal.y > 0.8) gl_FragColor.rgb *= 1.8;

    // gl_FragColor = vec4(vec3(f), 1.0);
}`;

export const GlassInner_glsl = `#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vPos;

#!SHADER: Vertex
void main() {
    vNormal = normal;
    vViewDir = -vec3(modelViewMatrix * vec4(position, 1.0));
    vPos = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(eases.glsl)

void main() {
    gl_FragColor = mix(vec4(0.0), vec4(1.4), vNormal.y) * crange(cnoise(vViewDir*0.2 + 0.5), -1.0, 1.0, 0.0, 1.0);
    gl_FragColor.rgb += cnoise(vViewDir) * 0.05;
    gl_FragColor.rgb += quarticIn(crange(abs(vPos.x), 0.5, 0.3, 1.0, 0.0) * crange(abs(vPos.z), 0.5, 0.3, 1.0, 0.0)) * 0.1;
}`;

export const GlassReflection_glsl = `#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS

#!SHADER: Vertex
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment
void main() {
    gl_FragColor = vec4(vec3(1.0), 0.1);
}`;

export const PBR_glsl = `#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS

#!SHADER: Vertex

#require(pbr.vs)

void main() {
    vec3 pos = position;
    setupPBR(pos);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(pbr.fs)

void main() {
    gl_FragColor = getPBR();
}`;

export const RoomPBR_glsl = `#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS

#!SHADER: Vertex

#require(pbr.vs)

void main() {
    vec3 pos = position;
    setupPBR(pos);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(pbr.fs)

void main() {
    gl_FragColor = getPBR();
}`;

export const TreeFBR_glsl = `#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tBaseColor;
uniform sampler2D tVideo;
uniform float uWobble;
uniform float uScroll;

#!VARYINGS
varying vec3 vWorldPos;

#!SHADER: Vertex

#require(fbr.vs)

void main() {
    vec3 pos = position;
    setupFBR(position);
    vWorldPos = vec3(modelMatrix * vec4(pos, 1.0));

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    
}

#!SHADER: Fragment

#require(fbr.fs)
#require(rgb2hsv.fs)
#require(transformUV.glsl)
#require(blendmodes.glsl)

void main() {
    vec3 baseColor = texture2D(tBaseColor, vUv).rgb;

    vec3 color = getFBR(baseColor, vUv);
    vec3 video = texture2D(tVideo, (vWorldPos.xz * 0.025) + vec2(0.5)).rgb;

    color = rgb2hsv(color);
    video = rgb2hsv(video);
    //color.x *= 0.1;
    //color.x = color.x * 0.05 + video.x;
    //color.y = video.y * 0.8;

    color.x = 0.0;
    color.y = 0.0;
    //color.z *= mix(1.0, video.z, smoothstep(30.0, 0.0, length(vWorldPos)) * 0.1);
    float saturation = color.y;
    color = hsv2rgb(color);
    video = hsv2rgb(video);


    video *= smoothstep(16.0, 5.0, vWorldPos.y);

    color = blendOverlay(color, video, 0.5);
    color = blendSoftLight(color, video, 0.5);
    color = blendAdd(color, baseColor, 0.1);

    color = pow(color*1.0, vec3(1.0));
    color *= smoothstep(13.0, 4.0, length(vWorldPos-vec3(-3.0, 15.0 - uScroll * 20.0, 0.0)));
    color *= smoothstep(25.0, 10.0, length(vWorldPos-vec3(-3.0, -2.0, 0.0)));
    color *= 1.25;




    gl_FragColor = vec4(color, 1.0);
}`;
