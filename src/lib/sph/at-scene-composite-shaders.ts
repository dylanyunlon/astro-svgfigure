/**
 * at-scene-composite-shaders.ts — M852: AT scene composite shaders
 * 7 world-theme composite passes from AT production.
 */









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
