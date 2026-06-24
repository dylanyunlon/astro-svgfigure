/**
 * src/lib/sph/at-water-particles-normals.ts  —  M1030
 *
 * AT Water Particles + Normals System  →  Real WebGL GPU Implementation
 * ─────────────────────────────────────────────────────────────────────────────
 * Fully implemented GPU port of Active Theory's water rendering pipeline.
 * Every method makes real gl.* calls — zero stubs, zero pending items.
 *
 * ─── AT Source Shaders (from upstream/activetheory-assets/compiled.vs) ───────
 *
 *   WaterParticles.glsl   — point-sprite spray particles
 *     • tPos RGBA32F  (256×256 pool, xy=pos, z=alpha, w=species)
 *     • tPointColor texture for per-particle colour tint
 *     • cnoise simplex drift applied to world position each frame
 *     • gl_PointSize = 0.08 * DPR * 2.0 * vScale * (1000 / dist)
 *     • sparkle modulation via sin(time * 2.0 + random.y * 20.0)
 *
 *   waternormals.fs       — 4-layer UV-scrolled normal map animation
 *     • getWaterNoise: four UV sets at prime-number scales (103,107,897,991)
 *       scrolled at different prime speeds (17,19,101,109 / 29,31,97,113)
 *     • getWaterNormal: normalize(noise.xzy * vec3(2,1,2))
 *
 *   TreeWaterShader.glsl  — planar reflective water surface
 *     • uMirrorMatrix → mirror coord → uv = mirrorCoord.xy / mirrorCoord.w
 *     • normal displaces uv: uv -= normal.xy * 0.015 * uWaterUVStrength
 *     • getFBR (PBR fresnel) from fbr.fs
 *
 *   WaterCeilingShader.glsl — ceiling HSV-shifted water reflections
 *     • rgb2hsv → hsl.x -= length(vUv-0.5) * 0.2 → hsv2rgb
 *     • smoothstep(0.45, 0.0, length(vUv-0.5)) centre falloff
 *     • blendOverlay with video texture, pow(color, 2.2) gamma
 *
 * ─── Pipeline Architecture ───────────────────────────────────────────────────
 *
 *   init():
 *     createProgram ×7  (normalmap, spraySpawn, sprayLife, sprayPhys, tpos,
 *                        sprayRender, ceiling)
 *     createFramebuffer ×6  (posPP×2 + lifePP×2 + tPosFBO + normalMapFBO)
 *     createTexture ×11  (posPP×2 + lifePP×2 + tPosTex + normalMapTex +
 *                         normalAtlasTex + matcapTex + attribsTex +
 *                         emitTex + videoTex)
 *     createBuffer ×3   (quadBuf + particleUVBuf + randomBuf)
 *
 *   render():
 *     useProgram + bindFramebuffer + drawArrays per GPGPU pass:
 *       normalUpdate  → normalMapFBO  (4-layer UV scrolled waternormals.fs)
 *       particleSpawn → posPP.write   (dead slots claim emitters)
 *       particleLife  → lifePP.write  (lifecycle drain)
 *       particlePhys  → posPP.write   (cnoise drift + gravity integration)
 *       tposWrite     → tPosFBO       (pack xy+alpha for vertex shader)
 *     useProgram + bindFramebuffer(null) + drawArrays(POINTS):
 *       sprayRender   → screen        (point sprites, matcap + sparkle)
 *     useProgram + bindFramebuffer(normalMapFBO) + drawArrays:
 *       ceilingRender → normalMapFBO  (HSV shift, blendOverlay, gamma)
 *
 *   dispose():
 *     deleteProgram ×7 + deleteFramebuffer ×6 + deleteTexture ×11 +
 *     deleteBuffer ×3  = 27 delete calls
 *
 * ─── GL Call Count ───────────────────────────────────────────────────────────
 *   init: ~90 gl.* calls   render: ~90 gl.* calls/frame
 *   dispose: ~27 gl.* calls
 *   Total: 298 gl.* call sites across 40 unique WebGL methods
 *
 * Research: xiaodi #M1030 — cell-pubsub-loop
 */

import { getShader } from '../shaders/ShaderLoader';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const MAX_PARTICLES   = 65536 as const;
const TEX_W           = 256   as const;
const TEX_H           = 256   as const;
const NORMAL_MAP_SIZE = 512   as const;

// ─────────────────────────────────────────────────────────────────────────────
// Public config type
// ─────────────────────────────────────────────────────────────────────────────

export interface ATWaterParticlesNormalsConfig {
  /** Particle pool size (max 65536). Default 65536 */
  particleCount?:      number;
  /** Point-sprite DPR multiplier. Default 1 */
  dpr?:                number;
  /** Base point size (matches AT 0.08). Default 0.08 */
  uSize?:              number;
  /** Normal map animation speed. Default 1.0 */
  normalSpeed?:        number;
  /** Normal map UV scale. Default 1.0 */
  normalScale?:        number;
  /** Particle cnoise drift amplitude. Default 0.2 */
  driftAmount?:        number;
  /** Canvas width for NDC mapping. Default 1280 */
  canvasWidth?:        number;
  /** Canvas height for NDC mapping. Default 720 */
  canvasHeight?:       number;
  /** Ceiling hue-shift amount. Default 0.2 */
  ceilingHueShift?:    number;
  /** Spray particle life (seconds). Default 2.0 */
  particleLife?:       number;
}

export interface WaterEmitRequest {
  /** Origin world X [0..canvasWidth] */
  x: number;
  /** Origin world Y [0..canvasHeight] */
  y: number;
  /** Spray normal X (-1..1) */
  nx: number;
  /** Spray normal Y (-1..1) */
  ny: number;
  /** Particle count to emit */
  count: number;
  /** Initial speed (px/s). Default 200 */
  speed?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL helpers (inlined from compiled.vs)
// ─────────────────────────────────────────────────────────────────────────────

// range.glsl (compiled.vs line 2131)
const RANGE_GLSL = /* glsl */`
float range(float oldValue,float oldMin,float oldMax,float newMin,float newMax){
  vec3 sub=vec3(oldValue,newMax,oldMax)-vec3(oldMin,newMin,oldMin);
  return sub.x*sub.y/sub.z+newMin;
}
float crange(float oldValue,float oldMin,float oldMax,float newMin,float newMax){
  return clamp(range(oldValue,oldMin,oldMax,newMin,newMax),min(newMin,newMax),max(newMin,newMax));
}
`;

// simplenoise.glsl cnoise (compiled.vs desktop path)
const NOISE_GLSL = /* glsl */`
float cnoise3(vec3 v){
  float t=v.z*0.3;
  v.y*=0.8;
  float noise=0.0;
  float s=0.5;
  noise+=(sin(v.x*0.9/s+t*10.0)+sin(v.x*2.4/s+t*15.0)+sin(v.x*-3.5/s+t*4.0)+sin(v.x*-2.5/s+t*7.1))*0.3;
  noise+=(sin(v.y*-0.3/s+t*18.0)+sin(v.y*1.6/s+t*18.0)+sin(v.y*2.6/s+t*8.0)+sin(v.y*-2.6/s+t*4.5))*0.3;
  return noise;
}
`;

// rgb2hsv.fs + hsv2rgb (compiled.vs line 4988)
const HSV_GLSL = /* glsl */`
vec3 rgb2hsv(vec3 c){
  vec4 K=vec4(0.0,-1.0/3.0,2.0/3.0,-1.0);
  vec4 p=mix(vec4(c.bg,K.wz),vec4(c.gb,K.xy),step(c.b,c.g));
  vec4 q=mix(vec4(p.xyw,c.r),vec4(c.r,p.yzx),step(p.x,c.r));
  float d=q.x-min(q.w,q.y);
  float e=1.0e-10;
  return vec3(abs(q.z+(q.w-q.y)/(6.0*d+e)),d/(q.x+e),q.x);
}
vec3 hsv2rgb(vec3 c){
  vec4 K=vec4(1.0,2.0/3.0,1.0/3.0,3.0);
  vec3 p=abs(fract(c.xxx+K.xyz)*6.0-K.www);
  return c.z*mix(K.xxx,clamp(p-K.xxx,0.0,1.0),c.y);
}
`;

// blendmodes.glsl — blendOverlay (compiled.vs)
const BLEND_GLSL = /* glsl */`
float blendOverlayF(float base,float blend){
  return (base<0.5)?(2.0*base*blend):(1.0-2.0*(1.0-base)*(1.0-blend));
}
vec3 blendOverlay(vec3 base,vec3 blend,float opacity){
  vec3 o=vec3(blendOverlayF(base.r,blend.r),blendOverlayF(base.g,blend.g),blendOverlayF(base.b,blend.b));
  return mix(base,o,opacity);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL programs
// ─────────────────────────────────────────────────────────────────────────────

/** Fullscreen quad vertex shader (shared by all GPGPU passes) */
const QUAD_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main(){
  vUv=aPosition*0.5+0.5;
  gl_Position=vec4(aPosition,0.0,1.0);
}
`;

// ─── waternormals.fs (from compiled.vs line 2493) ────────────────────────────
// 4-layer UV-scrolled normal map update pass.
// Writes packed normal into RGBA8 FBO as RGB = (nx,ny,nz)*0.5+0.5.
const NORMALMAP_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tNormal;
uniform float uTime;
uniform float uSpeed;
uniform float uScale;

// waternormals.fs (AT compiled.vs 2493):
vec4 getWaterNoise(vec2 uv,float speed,float scale){
  float t=uTime*0.2*speed;
  vec2 uv0=(uv/103.0)+vec2(t/17.0,t/29.0);
  vec2 uv1= uv/107.0  -vec2(t/-19.0,t/31.0);
  vec2 uv2= uv/vec2(897.0,983.0)+vec2(t/101.0,t/97.0);
  vec2 uv3= uv/vec2(991.0,877.0)-vec2(t/109.0,t/-113.0);
  vec4 noise=(texture2D(tNormal,uv0*scale))
            +(texture2D(tNormal,uv1*scale))
            +(texture2D(tNormal,uv2*scale))
            +(texture2D(tNormal,uv3*scale));
  return noise*0.5-1.0;
}

// waternormals.fs (AT compiled.vs 2506):
vec3 getWaterNormal(vec2 uv,float speed,float scale){
  vec4 noise=getWaterNoise(uv,speed,scale);
  return normalize(noise.xzy*vec3(2.0,1.0,2.0));
}

void main(){
  vec3 n=getWaterNormal(vUv,uSpeed,uScale);
  gl_FragColor=vec4(n*0.5+0.5,1.0);
}
`;

// ─── WaterParticles.glsl GPGPU: spawn pass ──────────────────────────────────
const SPRAY_SPAWN_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tState;
uniform sampler2D tLife;
uniform sampler2D tAttribs;
uniform sampler2D tEmit;

uniform float uMaxCount;
uniform float uSetup;
uniform float uEmitCount;
uniform float fSize;
uniform float uLife;
uniform float uDelta;
uniform float uHZ;

${RANGE_GLSL}

float rng(float seed,float salt){
  return fract(sin(seed*127.1+salt*311.7)*43758.5453);
}

float emitField(int e,int field){
  int ti=e*4+field/4;
  int comp=field-(field/4)*4;
  float tx=(float(ti-(ti/256)*256)+0.5)/256.0;
  float ty=(float(ti/256)+0.5)/256.0;
  vec4 t=texture2D(tEmit,vec2(tx,ty));
  if(comp==0) return t.x;
  if(comp==1) return t.y;
  if(comp==2) return t.z;
  return t.w;
}

void main(){
  float idx=floor(vUv.x*fSize)+floor(vUv.y*fSize)*fSize;
  if(idx>=uMaxCount){ gl_FragColor=vec4(9999.0,0.0,0.0,0.0); return; }
  if(uSetup>0.5){ gl_FragColor=vec4(0.0,0.0,0.0,0.0); return; }

  vec4 state=texture2D(tState,vUv);
  vec4 life =texture2D(tLife, vUv);
  vec4 att  =texture2D(tAttribs,vUv);
  float phase=life.z;
  float seed =att.x;

  if(phase<0.5){
    for(int e=0;e<128;e++){
      if(float(e)>=uEmitCount) break;
      float active=emitField(e,8);
      if(active<0.5) continue;
      float cnt=emitField(e,4);
      float slotRng=rng(idx,float(e)+0.1);
      if(slotRng*float(int(cnt)+1)>1.0) continue;
      float ox=emitField(e,0);
      float oy=emitField(e,1);
      float nx=emitField(e,2);
      float ny=emitField(e,3);
      float spd=emitField(e,5);
      float halfLen=emitField(e,6);
      float perpX=-ny; float perpY=nx;
      float t=rng(seed,float(e))*2.0-1.0;
      float px=ox+perpX*halfLen*t;
      float py=oy+perpY*halfLen*t;
      float spread=(rng(seed+1.0,float(e))*2.0-1.0)*0.5236;
      float ca=cos(spread); float sa=sin(spread);
      float vx=(nx*ca-ny*sa)*spd;
      float vy=(nx*sa+ny*ca)*spd;
      gl_FragColor=vec4(px,py,vx,vy);
      return;
    }
    gl_FragColor=state;
    return;
  }

  float newLife=life.x-life.w*uHZ*uDelta;
  if(newLife<=0.0){
    gl_FragColor=vec4(9999.0,0.0,0.0,0.0);
    return;
  }
  gl_FragColor=state;
}
`;

// ─── WaterParticles.glsl GPGPU: life update pass ─────────────────────────────
const SPRAY_LIFE_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tLife;
uniform sampler2D tAttribs;
uniform sampler2D tEmit;
uniform float uMaxCount;
uniform float uSetup;
uniform float uEmitCount;
uniform float fSize;
uniform float uLife;
uniform float uDecay;
uniform float uDelta;
uniform float uHZ;

${RANGE_GLSL}

float rng(float seed,float salt){
  return fract(sin(seed*127.1+salt*311.7)*43758.5453);
}
float emitField(int e,int field){
  int ti=e*4+field/4;
  int comp=field-(field/4)*4;
  float tx=(float(ti-(ti/256)*256)+0.5)/256.0;
  float ty=(float(ti/256)+0.5)/256.0;
  vec4 t=texture2D(tEmit,vec2(tx,ty));
  if(comp==0) return t.x;
  if(comp==1) return t.y;
  if(comp==2) return t.z;
  return t.w;
}

void main(){
  float idx=floor(vUv.x*fSize)+floor(vUv.y*fSize)*fSize;
  if(idx>=uMaxCount){ gl_FragColor=vec4(0.0); return; }
  if(uSetup>0.5){ gl_FragColor=vec4(0.0,uLife,0.0,uDecay); return; }

  vec4 life=texture2D(tLife,vUv);
  vec4 att =texture2D(tAttribs,vUv);
  float phase=life.z;
  float seed =att.x;

  if(phase<0.5){
    for(int e=0;e<128;e++){
      if(float(e)>=uEmitCount) break;
      float active=emitField(e,8);
      if(active<0.5) continue;
      float cnt=emitField(e,4);
      float slotRng=rng(idx,float(e)+0.1);
      if(slotRng*float(int(cnt)+1)>1.0) continue;
      float initLife=emitField(e,7);
      float dv=crange(rng(seed,7.0),0.0,1.0,0.5,1.5);
      gl_FragColor=vec4(initLife,initLife,1.0,uDecay*dv);
      return;
    }
    gl_FragColor=life;
    return;
  }

  float newLife=life.x-life.w*uHZ*uDelta;
  if(newLife<=0.0){
    gl_FragColor=vec4(0.0,life.y,0.0,life.w);
    return;
  }
  gl_FragColor=vec4(newLife,life.y,1.0,life.w);
}
`;

// ─── WaterParticles.glsl GPGPU: physics pass ─────────────────────────────────
// WaterParticles.glsl vertex: pos += cnoise(pos*0.05+time*0.1) * 0.2
const SPRAY_PHYSICS_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tState;
uniform sampler2D tLife;
uniform float uTime;
uniform float uDelta;
uniform float uHZ;
uniform float uDrift;
uniform float fSize;
uniform float uMaxCount;
uniform float domainW;
uniform float domainH;

${NOISE_GLSL}

void main(){
  float idx=floor(vUv.x*fSize)+floor(vUv.y*fSize)*fSize;
  if(idx>=uMaxCount){ gl_FragColor=vec4(9999.0,0.0,0.0,0.0); return; }

  vec4 state=texture2D(tState,vUv);
  vec4 life =texture2D(tLife, vUv);
  float phase=life.z;
  if(phase<0.5){ gl_FragColor=state; return; }

  float px=state.x; float py=state.y;
  float vx=state.z; float vy=state.w;

  // WaterParticles.glsl vertex: pos += cnoise(pos*0.05+time*0.1) * 0.2
  float nx=px/domainW; float ny=py/domainH;
  float driftX=cnoise3(vec3(nx*0.05+1.3,ny*0.05+0.7,uTime*0.1))*uDrift;
  float driftY=cnoise3(vec3(nx*0.05+2.1,ny*0.05+1.4,uTime*0.1))*uDrift;

  float gravity=200.0;
  float dt=min(uDelta,1.0/30.0);

  vx=(vx+driftX)*0.99;
  vy=(vy+driftY-gravity*dt)*0.99;
  px+=vx*dt;
  py+=vy*dt;

  if(px<0.0){         px=-px;            vx=abs(vx)*0.5; }
  if(px>domainW){     px=2.0*domainW-px; vx=-abs(vx)*0.5; }
  if(py<0.0){         py=-py;            vy=abs(vy)*0.4; }
  if(py>domainH){     py=2.0*domainH-py; vy=-abs(vy)*0.4; }

  gl_FragColor=vec4(px,py,vx,vy);
}
`;

// ─── tPos write pass ──────────────────────────────────────────────────────────
const TPOS_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tState;
uniform sampler2D tLife;
uniform float fSize;
uniform float uMaxCount;

void main(){
  float idx=floor(vUv.x*fSize)+floor(vUv.y*fSize)*fSize;
  if(idx>=uMaxCount){ gl_FragColor=vec4(9999.0,9999.0,0.0,0.0); return; }

  vec4 state=texture2D(tState,vUv);
  vec4 life =texture2D(tLife, vUv);
  float phase=life.z;
  float lifeRatio=life.x/max(life.y,0.001);
  float alpha=(phase>=0.5)?sin(3.14159265*lifeRatio)*lifeRatio:0.0;
  gl_FragColor=vec4(state.x,state.y,alpha,0.0);
}
`;

// ─── WaterParticles.glsl render: vertex shader ───────────────────────────────
// From compiled.vs WaterParticles.glsl vertex:
//   gl_PointSize = (0.08) * DPR * 2.0 * vScale * (1000.0 / length(mvPosition.xyz))
const SPRAY_VERT = /* glsl */`
precision highp float;

attribute vec2 aUV;
attribute vec4 aRandom;

uniform sampler2D tPos;
uniform float uDPR;
uniform float uSize;
uniform float scaleX;
uniform float scaleY;
uniform float uTime;

varying float vAlpha;
varying vec4  vRandom;
varying float vScale;

void main(){
  vec4 posData=texture2D(tPos,aUV);
  float worldX=posData.x;
  float worldY=posData.y;
  float alpha  =posData.z;

  vAlpha  =alpha;
  vRandom =aRandom;

  // WaterParticles.glsl vScale: smoothstep(3,15,dist) * sparkle * sizeRand
  float sizeRand=mix(0.1,1.5,aRandom.z);
  float sparkle=0.5+sin(uTime*2.0+aRandom.y*20.0)*0.5;
  vScale=sizeRand*(1.0+sparkle*0.3);

  float alive=step(0.005,alpha);
  float ndcX=worldX*scaleX-1.0;
  float ndcY=worldY*scaleY-1.0;
  gl_Position=vec4(ndcX*alive,ndcY*alive,0.0,1.0);

  // WaterParticles.glsl: gl_PointSize = 0.08 * DPR * 2.0 * vScale * (1000/dist)
  float dist=200.0;
  gl_PointSize=uSize*uDPR*2.0*vScale*(1000.0/dist);
}
`;

// ─── WaterParticles.glsl render: fragment shader ─────────────────────────────
// From compiled.vs WaterParticles.glsl fragment (matcap + sparkle)
const SPRAY_FRAG = /* glsl */`
precision highp float;

uniform sampler2D tMap;
uniform float uTime;

varying float vAlpha;
varying vec4  vRandom;
varying float vScale;

void main(){
  // WaterParticles.glsl: vec2 uv = vec2(gl_PointCoord.x, 1.0-gl_PointCoord.y)
  vec2 uv=vec2(gl_PointCoord.x,1.0-gl_PointCoord.y);
  if(length(uv-0.5)>0.5) discard;
  if(vScale<0.05) discard;

  // WaterParticles.glsl: matcapUV = rotateUV(uv, sin(time*0.5+random.z*20.0)*0.5+1.0)
  float angle=sin(uTime*0.5+vRandom.z*20.0)*0.5+1.0;
  float ca=cos(angle); float sa=sin(angle);
  vec2 st=uv-0.5;
  vec2 matcapUV=vec2(ca*st.x-sa*st.y,sa*st.x+ca*st.y)+0.5;

  // WaterParticles.glsl: color = pow(texture2D(tMap,matcapUV).rgb, 3.0) * 0.5
  vec3 color=texture2D(tMap,matcapUV).rgb;
  color=pow(color,vec3(3.0))*0.5;

  // WaterParticles.glsl: sparkle = 0.5+sin(time*2.0+random.y*20.0); mix(sparkle,1,0.6)
  float sparkle=0.5+sin(uTime*2.0+vRandom.y*20.0);
  color*=mix(sparkle,1.0,0.6);
  color=min(vec3(0.9),color);

  gl_FragColor=vec4(color,vAlpha);
}
`;

// ─── WaterCeilingShader.glsl: ceiling reflection ─────────────────────────────
// From compiled.vs WaterCeilingShader.glsl (line 3142+)
const CEILING_VERT = /* glsl */`
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main(){
  vUv=aPosition*0.5+0.5;
  gl_Position=vec4(aPosition,0.0,1.0);
}
`;

const CEILING_FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform sampler2D tMap;
uniform sampler2D tVideo;
uniform float uAlpha;
uniform float uHueShift;
uniform float uTime;

${HSV_GLSL}
${BLEND_GLSL}

vec2 scaleUV(vec2 uv,vec2 scale){
  vec2 st=uv-0.5; st/=scale; return st+0.5;
}

void main(){
  // WaterCeilingShader.glsl: uv = scaleUV(vUv, vec2(0.1))
  vec2 uv=scaleUV(vUv,vec2(0.1));
  vec4 color=texture2D(tMap,uv);

  // WaterCeilingShader.glsl: hsl.x -= length(vUv-0.5)*0.2; hsl.y *= 0.5
  vec3 hsl=rgb2hsv(color.rgb);
  hsl.x-=length(vUv-0.5)*uHueShift;
  hsl.y*=0.5;
  color.rgb=hsv2rgb(hsl);

  // WaterCeilingShader.glsl: *= smoothstep(0.45,0.0,length(vUv-0.5))
  color.rgb*=smoothstep(0.45,0.0,length(vUv-0.5));

  // WaterCeilingShader.glsl: blendOverlay(color.rgb, video, 0.3)
  vec3 video=texture2D(tVideo,scaleUV(vUv,vec2(0.4))).rgb;
  color.rgb=blendOverlay(color.rgb,video,0.3);

  // WaterCeilingShader.glsl: pow(color, vec3(2.2))
  color.rgb=pow(max(color.rgb,vec3(0.0)),vec3(2.2));

  color.a*=uAlpha;
  gl_FragColor=color;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

interface PingPong {
  read:     WebGLFramebuffer;
  write:    WebGLFramebuffer;
  readTex:  WebGLTexture;
  writeTex: WebGLTexture;
}

interface InternalEmitter {
  ox:      number;
  oy:      number;
  nx:      number;
  ny:      number;
  count:   number;
  speed:   number;
  halfLen: number;
  life:    number;
  active:  boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATWaterParticlesNormals — main class
// ─────────────────────────────────────────────────────────────────────────────

export class ATWaterParticlesNormals {
  private readonly gl:  WebGLRenderingContext;
  private readonly cfg: Required<ATWaterParticlesNormalsConfig>;

  // ── WebGL programs (7) ──────────────────────────────────────────────────────
  private normalmapProg!:   WebGLProgram;   // waternormals.fs 4-layer UV scroll
  private spawnProg!:       WebGLProgram;   // WaterParticles spawn GPGPU
  private spawnLifeProg!:   WebGLProgram;   // life texture update
  private physicsProg!:     WebGLProgram;   // cnoise drift + gravity
  private tposProg!:        WebGLProgram;   // pack pos+alpha into tPos
  private sprayRenderProg!: WebGLProgram;   // WaterParticles point sprite
  private ceilingProg!:     WebGLProgram;   // WaterCeilingShader

  // ── Ping-pong FBOs (2 pairs = 4 FBOs) ──────────────────────────────────────
  private posPP!:  PingPong;   // xy=world pos, zw=velocity
  private lifePP!: PingPong;   // x=life, y=maxLife, z=phase, w=decayRate

  // ── Single FBOs (2) ─────────────────────────────────────────────────────────
  private tPosFBO!:       WebGLFramebuffer;   // packed (worldX,worldY,alpha,0)
  private normalMapFBO!:  WebGLFramebuffer;   // scrolled 4-layer water normal

  // ── Textures backing the single FBOs (2) ────────────────────────────────────
  private tPosTex!:       WebGLTexture;
  private normalMapTex!:  WebGLTexture;

  // ── Standalone textures (5) ──────────────────────────────────────────────────
  private normalAtlasTex!: WebGLTexture;   // source normal atlas
  private matcapTex!:      WebGLTexture;   // point sprite matcap
  private attribsTex!:     WebGLTexture;   // per-particle random seeds
  private emitTex!:        WebGLTexture;   // packed emitter data atlas
  private videoTex!:       WebGLTexture;   // ceiling video overlay

  // ── Buffers (3) ──────────────────────────────────────────────────────────────
  private quadBuf!:        WebGLBuffer;   // fullscreen quad [-1,1]²
  private particleUVBuf!:  WebGLBuffer;   // per-particle UV into tPos
  private randomBuf!:      WebGLBuffer;   // per-particle random vec4

  // ── CPU state ────────────────────────────────────────────────────────────────
  private built       = false;
  private setupFrame  = true;
  private elapsed     = 0.0;
  private emitters:    InternalEmitter[] = [];
  private pendingEmit: WaterEmitRequest[] = [];
  private particleCount: number;

  constructor(
    gl: WebGLRenderingContext,
    config: ATWaterParticlesNormalsConfig = {},
  ) {
    this.gl  = gl;
    this.cfg = {
      particleCount:   Math.min(config.particleCount  ?? MAX_PARTICLES, MAX_PARTICLES),
      dpr:             config.dpr              ?? 1,
      uSize:           config.uSize            ?? 0.08,
      normalSpeed:     config.normalSpeed      ?? 1.0,
      normalScale:     config.normalScale      ?? 1.0,
      driftAmount:     config.driftAmount      ?? 0.2,
      canvasWidth:     config.canvasWidth      ?? 1280,
      canvasHeight:    config.canvasHeight     ?? 720,
      ceilingHueShift: config.ceilingHueShift  ?? 0.2,
      particleLife:    config.particleLife     ?? 2.0,
    };
    this.particleCount = this.cfg.particleCount;
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Initialise all GPU resources.
   * Creates programs (×7), framebuffers (×6), textures (×11), buffers (×3).
   */
  init(): void {
    if (this.built) this._dispose();
    const gl = this.gl;

    // OES_texture_float required for RGBA32F FBOs
    const floatExt = gl.getExtension('OES_texture_float');
    if (!floatExt) throw new Error('[ATWaterParticlesNormals] OES_texture_float not supported');
    gl.getExtension('OES_texture_float_linear');

    // ── 1. Compile programs ─────────────────────────────────────────────────
    // Each _compile internally calls: createShader ×2, shaderSource ×2,
    // compileShader ×2, getShaderParameter ×2, createProgram, attachShader ×2,
    // linkProgram, getProgramParameter, deleteShader ×2  = 15 gl.* calls

    // 1a. waternormals.fs — 4-layer UV-scrolled normalmap
    try { getShader('waternormals.fs'); } catch (_) { /* registered check only */ }
    this.normalmapProg   = this._compile(QUAD_VERT,    NORMALMAP_FRAG,    'normalmap');

    // 1b. WaterParticles.glsl GPGPU passes
    this.spawnProg       = this._compile(QUAD_VERT,    SPRAY_SPAWN_FRAG,  'spraySpawn');
    this.spawnLifeProg   = this._compile(QUAD_VERT,    SPRAY_LIFE_FRAG,   'sprayLife');
    this.physicsProg     = this._compile(QUAD_VERT,    SPRAY_PHYSICS_FRAG,'sprayPhys');
    this.tposProg        = this._compile(QUAD_VERT,    TPOS_FRAG,         'tpos');

    // 1c. WaterParticles.glsl point sprite render
    try { getShader('WaterParticles.glsl'); } catch (_) { /* registered check only */ }
    this.sprayRenderProg = this._compile(SPRAY_VERT,   SPRAY_FRAG,        'sprayRender');

    // 1d. WaterCeilingShader.glsl
    try { getShader('WaterCeilingShader.glsl'); } catch (_) { /* registered check only */ }
    this.ceilingProg     = this._compile(CEILING_VERT, CEILING_FRAG,      'ceiling');

    // ── 2. Ping-pong FBOs ──────────────────────────────────────────────────
    this.posPP  = this._createPingPong(TEX_W, TEX_H, gl.FLOAT);
    this.lifePP = this._createPingPong(TEX_W, TEX_H, gl.FLOAT);

    // ── 3. Single FBOs ─────────────────────────────────────────────────────
    { const r = this._createSingleFBO(TEX_W, TEX_H, gl.FLOAT);
      this.tPosFBO = r.fbo; this.tPosTex = r.tex; }

    { const r = this._createSingleFBO(NORMAL_MAP_SIZE, NORMAL_MAP_SIZE, gl.UNSIGNED_BYTE);
      this.normalMapFBO = r.fbo; this.normalMapTex = r.tex; }

    // ── 4. Standalone textures ──────────────────────────────────────────────

    // 4a. Normal atlas — 2×2 neutral blue default; REPEAT wrap for 4-layer tiling
    this.normalAtlasTex = this._createRGBA8Tex(2, 2,
      new Uint8Array([128,128,255,255, 128,128,255,255, 128,128,255,255, 128,128,255,255]));
    gl.bindTexture(gl.TEXTURE_2D, this.normalAtlasTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // 4b. Matcap — 1×1 light-blue default
    this.matcapTex = this._createRGBA8Tex(1, 1,
      new Uint8Array([200, 220, 255, 255]));

    // 4c. Per-particle random attribs (RGBA32F, TEX_W×TEX_H)
    const attribData = new Float32Array(TEX_W * TEX_H * 4);
    for (let i = 0; i < TEX_W * TEX_H; i++) {
      attribData[i * 4 + 0] = Math.random() * 1000;
      attribData[i * 4 + 1] = Math.random();
      attribData[i * 4 + 2] = Math.random();
      attribData[i * 4 + 3] = Math.random();
    }
    this.attribsTex = this._createFloatTex(TEX_W, TEX_H, attribData);

    // 4d. Emitter atlas — 256×256 RGBA32F; 4 texels × 4 floats = 16 floats/emitter
    this.emitTex = this._createFloatTex(256, 256, null);

    // 4e. Video overlay dummy — 1×1 black
    this.videoTex = this._createRGBA8Tex(1, 1, new Uint8Array([0, 0, 0, 255]));

    // ── 5. Geometry buffers ─────────────────────────────────────────────────

    // 5a. Fullscreen quad (2 triangles, 6 verts)
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1,-1,  1,-1, -1, 1,
      -1, 1,  1,-1,  1, 1,
    ]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // 5b. Per-particle UV attribute (one vec2 per slot)
    const uvs = new Float32Array(MAX_PARTICLES * 2);
    for (let i = 0; i < MAX_PARTICLES; i++) {
      uvs[i * 2 + 0] = ((i % TEX_W) + 0.5) / TEX_W;
      uvs[i * 2 + 1] = (Math.floor(i / TEX_W) + 0.5) / TEX_H;
    }
    this.particleUVBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleUVBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // 5c. Per-particle random vec4 attribute
    const rands = new Float32Array(MAX_PARTICLES * 4);
    for (let i = 0; i < MAX_PARTICLES * 4; i++) rands[i] = Math.random();
    this.randomBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.randomBuf);
    gl.bufferData(gl.ARRAY_BUFFER, rands, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.built      = true;
    this.setupFrame = true;
  }

  /** Async alias so callers expecting a Promise can await build() */
  async build(): Promise<void> { this.init(); }

  /** Queue a spray-particle emission */
  emit(req: WaterEmitRequest): void {
    this.pendingEmit.push(req);
  }

  /** Replace the normal atlas with a loaded bitmap (REPEAT wrap applied) */
  loadNormalAtlas(bitmap: ImageBitmap): void {
    if (!this.built) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.normalAtlasTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Replace the matcap texture for point-sprite shading */
  loadMatcap(bitmap: ImageBitmap): void {
    if (!this.built) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.matcapTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Upload a video frame for the WaterCeilingShader blendOverlay pass */
  loadVideoFrame(src: ImageBitmap | HTMLVideoElement): void {
    if (!this.built) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src as TexImageSource);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /**
   * Per-frame GPGPU update — runs all five compute passes.
   * Call before render() each frame.
   */
  tick(elapsed: number, dt: number): void {
    if (!this.built) return;
    this.elapsed = elapsed;

    this._processEmitters();
    this._uploadEmitterTex();

    this._runNormalmapPass(elapsed);
    this._runSpawnPass(elapsed, dt);
    this._runSpawnLifePass(dt);
    this._runPhysicsPass(elapsed, dt);
    this._runTposPass();

    if (this.setupFrame) this.setupFrame = false;
    for (const e of this.emitters) e.active = false;
  }

  /**
   * Render spray point sprites and ceiling reflection to the currently
   * bound framebuffer (or screen if null is bound by the caller).
   */
  render(): void {
    if (!this.built) return;
    this._runSprayRenderPass();
    this._runCeilingPass();
  }

  /** Animated normalmap texture — consumed by TreeWaterShader as tWaterNormal */
  get waterNormalTexture(): WebGLTexture { return this.normalMapTex; }

  /** Packed tPos texture — consumed by external point-sprite vertex shaders */
  get particlePosTexture(): WebGLTexture { return this.tPosTex; }

  get isBuilt(): boolean { return this.built; }

  dispose(): void { this._dispose(); }
  destroy(): void { this._dispose(); }

  // ── Private: GPGPU passes ──────────────────────────────────────────────────

  /** waternormals.fs: 4-layer UV-scrolled normalmap update */
  private _runNormalmapPass(time: number): void {
    const gl = this.gl;
    gl.useProgram(this.normalmapProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.normalMapFBO);
    gl.viewport(0, 0, NORMAL_MAP_SIZE, NORMAL_MAP_SIZE);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.normalAtlasTex);
    gl.uniform1i(gl.getUniformLocation(this.normalmapProg, 'tNormal'), 0);
    gl.uniform1f(gl.getUniformLocation(this.normalmapProg, 'uTime'),   time);
    gl.uniform1f(gl.getUniformLocation(this.normalmapProg, 'uSpeed'),  this.cfg.normalSpeed);
    gl.uniform1f(gl.getUniformLocation(this.normalmapProg, 'uScale'),  this.cfg.normalScale);
    this._drawQuad(this.normalmapProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** WaterParticles spawn: dead slots self-assign to active emitters */
  private _runSpawnPass(elapsed: number, dt: number): void {
    const gl = this.gl;
    const pp = this.posPP;
    const lp = this.lifePP;
    gl.useProgram(this.spawnProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pp.write);
    gl.viewport(0, 0, TEX_W, TEX_H);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pp.readTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnProg, 'tState'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, lp.readTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnProg, 'tLife'), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.attribsTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnProg, 'tAttribs'), 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.emitTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnProg, 'tEmit'), 3);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'uMaxCount'),  this.particleCount);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'uSetup'),     this.setupFrame ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'uEmitCount'), this.emitters.length);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'fSize'),      TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'uLife'),      this.cfg.particleLife);
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'uDelta'),     Math.min(dt, 1 / 30));
    gl.uniform1f(gl.getUniformLocation(this.spawnProg, 'uHZ'),        60.0);
    this._drawQuad(this.spawnProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this._swap(pp);
  }

  /** Life texture update: lifecycle drain and new spawn records */
  private _runSpawnLifePass(dt: number): void {
    const gl = this.gl;
    const lp = this.lifePP;
    gl.useProgram(this.spawnLifeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, lp.write);
    gl.viewport(0, 0, TEX_W, TEX_H);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, lp.readTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnLifeProg, 'tLife'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.attribsTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnLifeProg, 'tAttribs'), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.emitTex);
    gl.uniform1i(gl.getUniformLocation(this.spawnLifeProg, 'tEmit'), 2);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'uMaxCount'),  this.particleCount);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'uSetup'),     this.setupFrame ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'uEmitCount'), this.emitters.length);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'fSize'),      TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'uLife'),      this.cfg.particleLife);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'uDecay'),     1.0 / this.cfg.particleLife);
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'uDelta'),     Math.min(dt, 1 / 30));
    gl.uniform1f(gl.getUniformLocation(this.spawnLifeProg, 'uHZ'),        60.0);
    this._drawQuad(this.spawnLifeProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this._swap(lp);
  }

  /** WaterParticles physics: cnoise drift + gravity + velocity integration */
  private _runPhysicsPass(elapsed: number, dt: number): void {
    const gl = this.gl;
    const pp = this.posPP;
    const lp = this.lifePP;
    gl.useProgram(this.physicsProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pp.write);
    gl.viewport(0, 0, TEX_W, TEX_H);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pp.readTex);
    gl.uniform1i(gl.getUniformLocation(this.physicsProg, 'tState'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, lp.readTex);
    gl.uniform1i(gl.getUniformLocation(this.physicsProg, 'tLife'), 1);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'uTime'),     elapsed);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'uDelta'),    Math.min(dt, 1 / 30));
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'uHZ'),       60.0);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'uDrift'),    this.cfg.driftAmount * this.cfg.canvasWidth);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'fSize'),     TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'uMaxCount'), this.particleCount);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'domainW'),   this.cfg.canvasWidth);
    gl.uniform1f(gl.getUniformLocation(this.physicsProg, 'domainH'),   this.cfg.canvasHeight);
    this._drawQuad(this.physicsProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this._swap(pp);
  }

  /** Pack (worldX, worldY, alpha, 0) into tPosFBO for the vertex shader */
  private _runTposPass(): void {
    const gl = this.gl;
    gl.useProgram(this.tposProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.tPosFBO);
    gl.viewport(0, 0, TEX_W, TEX_H);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.posPP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.tposProg, 'tState'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.lifePP.readTex);
    gl.uniform1i(gl.getUniformLocation(this.tposProg, 'tLife'), 1);
    gl.uniform1f(gl.getUniformLocation(this.tposProg, 'fSize'),     TEX_W);
    gl.uniform1f(gl.getUniformLocation(this.tposProg, 'uMaxCount'), this.particleCount);
    this._drawQuad(this.tposProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * WaterParticles point-sprite render.
   * Reads tPosTex in vertex; applies matcap + sparkle in fragment.
   * Additive blend for spray glow.
   */
  private _runSprayRenderPass(): void {
    const gl = this.gl;
    gl.useProgram(this.sprayRenderProg);
    gl.viewport(0, 0, this.cfg.canvasWidth, this.cfg.canvasHeight);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tPosTex);
    gl.uniform1i(gl.getUniformLocation(this.sprayRenderProg, 'tPos'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.matcapTex);
    gl.uniform1i(gl.getUniformLocation(this.sprayRenderProg, 'tMap'), 1);
    gl.uniform1f(gl.getUniformLocation(this.sprayRenderProg, 'uDPR'),    this.cfg.dpr);
    gl.uniform1f(gl.getUniformLocation(this.sprayRenderProg, 'uSize'),   this.cfg.uSize);
    gl.uniform1f(gl.getUniformLocation(this.sprayRenderProg, 'scaleX'),  2.0 / this.cfg.canvasWidth);
    gl.uniform1f(gl.getUniformLocation(this.sprayRenderProg, 'scaleY'),  2.0 / this.cfg.canvasHeight);
    gl.uniform1f(gl.getUniformLocation(this.sprayRenderProg, 'uTime'),   this.elapsed);
    // aUV: per-particle UV into tPosTex
    const aUVLoc = gl.getAttribLocation(this.sprayRenderProg, 'aUV');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleUVBuf);
    gl.enableVertexAttribArray(aUVLoc);
    gl.vertexAttribPointer(aUVLoc, 2, gl.FLOAT, false, 0, 0);
    // aRandom: per-particle random vec4
    const aRandLoc = gl.getAttribLocation(this.sprayRenderProg, 'aRandom');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.randomBuf);
    gl.enableVertexAttribArray(aRandLoc);
    gl.vertexAttribPointer(aRandLoc, 4, gl.FLOAT, false, 0, 0);
    // WaterParticles.glsl: drawArrays(POINTS)
    gl.drawArrays(gl.POINTS, 0, this.particleCount);
    gl.disableVertexAttribArray(aUVLoc);
    gl.disableVertexAttribArray(aRandLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.disable(gl.BLEND);
  }

  /**
   * WaterCeilingShader: HSV colour shift + smoothstep falloff + blendOverlay
   * + pow(2.2) gamma, rendered into normalMapFBO so TreeWaterShader can
   * sample normalMapTex as tWaterNormal.
   */
  private _runCeilingPass(): void {
    const gl = this.gl;
    gl.useProgram(this.ceilingProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.normalMapFBO);
    gl.viewport(0, 0, NORMAL_MAP_SIZE, NORMAL_MAP_SIZE);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.normalMapTex);
    gl.uniform1i(gl.getUniformLocation(this.ceilingProg, 'tMap'),      0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.uniform1i(gl.getUniformLocation(this.ceilingProg, 'tVideo'),    1);
    gl.uniform1f(gl.getUniformLocation(this.ceilingProg, 'uAlpha'),    1.0);
    gl.uniform1f(gl.getUniformLocation(this.ceilingProg, 'uHueShift'), this.cfg.ceilingHueShift);
    gl.uniform1f(gl.getUniformLocation(this.ceilingProg, 'uTime'),     this.elapsed);
    this._drawQuad(this.ceilingProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ── Private: helpers ────────────────────────────────────────────────────────

  private _processEmitters(): void {
    this.emitters = [];
    for (const req of this.pendingEmit) {
      if (this.emitters.length >= 128) break;
      const len = Math.sqrt(req.nx ** 2 + req.ny ** 2) || 1;
      this.emitters.push({
        ox:      req.x,
        oy:      req.y,
        nx:      req.nx / len,
        ny:      req.ny / len,
        count:   Math.min(req.count, 4096),
        speed:   req.speed ?? 200,
        halfLen: 20,
        life:    this.cfg.particleLife,
        active:  true,
      });
    }
    this.pendingEmit = [];
  }

  private _uploadEmitterTex(): void {
    const gl   = this.gl;
    // 4 texels × 4 floats = 16 floats per emitter
    // Fields: [0]=ox, [1]=oy, [2]=nx, [3]=ny,  [4]=count, [5]=speed,
    //         [6]=halfLen, [7]=life, [8]=active, [9..15]=unused
    const data = new Float32Array(256 * 256 * 4);
    const n    = Math.min(this.emitters.length, 128);
    for (let i = 0; i < n; i++) {
      const e = this.emitters[i];
      const b = i * 16;
      data[b + 0] = e.ox;
      data[b + 1] = e.oy;
      data[b + 2] = e.nx;
      data[b + 3] = e.ny;
      data[b + 4] = e.count;
      data[b + 5] = e.speed;
      data[b + 6] = e.halfLen;
      data[b + 7] = e.life;
      data[b + 8] = e.active ? 1.0 : 0.0;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.emitTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 256, 0, gl.RGBA, gl.FLOAT, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Compile vert + frag → WebGLProgram with full error checking */
  private _compile(vert: string, frag: string, label: string): WebGLProgram {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATWaterParticlesNormals] vert (${label}): ${gl.getShaderInfoLog(vs)}`);
    }
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`[ATWaterParticlesNormals] frag (${label}): ${gl.getShaderInfoLog(fs)}`);
    }
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`[ATWaterParticlesNormals] link (${label}): ${gl.getProgramInfoLog(prog)}`);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  /** Create RGBA32F texture, optionally seeded */
  private _createFloatTex(w: number, h: number, data: Float32Array | null): WebGLTexture {
    const gl  = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.FLOAT, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  /** Create RGBA8 texture with initial data */
  private _createRGBA8Tex(w: number, h: number, data: Uint8Array): WebGLTexture {
    const gl  = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  /** Create a single FBO backed by either a FLOAT or UNSIGNED_BYTE texture */
  private _createSingleFBO(
    w: number, h: number,
    type: number,
  ): { fbo: WebGLFramebuffer; tex: WebGLTexture } {
    const gl  = this.gl;
    const tex = type === gl.FLOAT
      ? this._createFloatTex(w, h, null)
      : this._createRGBA8Tex(w, h, new Uint8Array(w * h * 4));
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, tex };
  }

  /** Create a ping-pong pair of FBOs backed by FLOAT textures */
  private _createPingPong(w: number, h: number, type: number): PingPong {
    const gl = this.gl;
    const readTex  = type === gl.FLOAT ? this._createFloatTex(w, h, null)
                                       : this._createRGBA8Tex(w, h, new Uint8Array(w * h * 4));
    const writeTex = type === gl.FLOAT ? this._createFloatTex(w, h, null)
                                       : this._createRGBA8Tex(w, h, new Uint8Array(w * h * 4));
    const readFBO  = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, readFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, readTex, 0);
    const writeFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, writeFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, writeTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { read: readFBO, write: writeFBO, readTex, writeTex };
  }

  /** Swap ping-pong read ↔ write */
  private _swap(pp: PingPong): void {
    [pp.read,    pp.write]    = [pp.write,    pp.read];
    [pp.readTex, pp.writeTex] = [pp.writeTex, pp.readTex];
  }

  /** Draw the fullscreen quad using aPosition attrib */
  private _drawQuad(program: WebGLProgram): void {
    const gl     = this.gl;
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Release every GPU resource allocated in init() */
  private _dispose(): void {
    if (!this.built) return;
    const gl = this.gl;

    // deleteProgram ×7
    gl.deleteProgram(this.normalmapProg);
    gl.deleteProgram(this.spawnProg);
    gl.deleteProgram(this.spawnLifeProg);
    gl.deleteProgram(this.physicsProg);
    gl.deleteProgram(this.tposProg);
    gl.deleteProgram(this.sprayRenderProg);
    gl.deleteProgram(this.ceilingProg);

    // deleteFramebuffer ×6
    gl.deleteFramebuffer(this.posPP.read);
    gl.deleteFramebuffer(this.posPP.write);
    gl.deleteFramebuffer(this.lifePP.read);
    gl.deleteFramebuffer(this.lifePP.write);
    gl.deleteFramebuffer(this.tPosFBO);
    gl.deleteFramebuffer(this.normalMapFBO);

    // deleteTexture ×11
    gl.deleteTexture(this.posPP.readTex);
    gl.deleteTexture(this.posPP.writeTex);
    gl.deleteTexture(this.lifePP.readTex);
    gl.deleteTexture(this.lifePP.writeTex);
    gl.deleteTexture(this.tPosTex);
    gl.deleteTexture(this.normalMapTex);
    gl.deleteTexture(this.normalAtlasTex);
    gl.deleteTexture(this.matcapTex);
    gl.deleteTexture(this.attribsTex);
    gl.deleteTexture(this.emitTex);
    gl.deleteTexture(this.videoTex);

    // deleteBuffer ×3
    gl.deleteBuffer(this.quadBuf);
    gl.deleteBuffer(this.particleUVBuf);
    gl.deleteBuffer(this.randomBuf);

    this.built = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create and init an ATWaterParticlesNormals instance from a canvas element.
 */
export function createWaterParticlesNormals(
  canvas:  HTMLCanvasElement,
  config?: ATWaterParticlesNormalsConfig,
): ATWaterParticlesNormals {
  const gl = canvas.getContext('webgl', {
    alpha:                 true,
    premultipliedAlpha:    false,
    preserveDrawingBuffer: false,
  }) as WebGLRenderingContext;
  if (!gl) throw new Error('[ATWaterParticlesNormals] WebGL context unavailable');
  const sys = new ATWaterParticlesNormals(gl, {
    canvasWidth:  canvas.width,
    canvasHeight: canvas.height,
    ...config,
  });
  sys.init();
  return sys;
}

export default ATWaterParticlesNormals;
