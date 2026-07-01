/**
 * cell-mesh-renderer.ts — M1315: 3D SDF Ray March Runtime Mesh Generator
 *
 * Each cell is rendered as a per-cell quad. The fragment shader ray-marches
 * a 3D metaball SDF (base sphere + lobes from geometry.json) in the cell's
 * local [-1,1]³ space. Produces correct 3D normals, depth, lighting, and
 * subsurface scattering — all driven by geometry.json tick-runner output.
 *
 * No pre-built GLB, no procedural geometry, no mesh vertices.
 * The GPU IS the runtime mesh generator.
 */

import type { CellData } from './gpu-render-loop';

// ─── Vertex shader: per-cell positioned quad ─────────────────────────────────

const VERT = /* glsl */ `#version 300 es
precision highp float;
in vec2 aCorner;
uniform vec2 uCellPos;
uniform vec2 uCellSize;
out vec2 vUV;
void main() {
    vUV = aCorner * 0.5 + 0.5;
    gl_Position = vec4(uCellPos + aCorner * uCellSize, 0.0, 1.0);
}
`;

// ─── Fragment shader: 3D SDF ray march ───────────────────────────────────────

const FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUV;

uniform vec3  uAlbedo;
uniform float uOpacity;
uniform vec3  uGlowColor;
uniform float uTime;
uniform float uBaseRadius;
uniform int   uLobeCount;
uniform vec3  uLobes[8];
uniform float uNoiseAmp;
uniform float uNoiseFreq;
uniform float uRoughness;
uniform float uMetallic;

out vec4 fragColor;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x),
               mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y);
}
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}

float cellSDF(vec3 p) {
    float d = length(p) - uBaseRadius;
    for (int i = 0; i < 8; i++) {
        if (i >= uLobeCount) break;
        vec3 lc = vec3(
            cos(uLobes[i].x) * uLobes[i].y,
            sin(uLobes[i].x) * uLobes[i].y,
            sin(uTime * 0.5 + uLobes[i].x) * 0.08
        );
        d = smin(d, length(p - lc) - uLobes[i].z, 0.3);
    }
    d += (vnoise(p.xy * uNoiseFreq + uTime * 0.15) * 2.0 - 1.0) * uNoiseAmp;
    return d;
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.005, 0.0);
    return normalize(vec3(
        cellSDF(p+e.xyy) - cellSDF(p-e.xyy),
        cellSDF(p+e.yxy) - cellSDF(p-e.yxy),
        cellSDF(p+e.yyx) - cellSDF(p-e.yyx)
    ));
}

void main() {
    vec2 uv = vUV * 2.0 - 1.0;
    vec3 ro = vec3(uv * 1.2, 2.0);
    vec3 rd = vec3(0.0, 0.0, -1.0);

    float t = 0.0;
    bool hit = false;
    for (int i = 0; i < 64; i++) {
        float d = cellSDF(ro + rd * t);
        if (d < 0.002) { hit = true; break; }
        if (t > 5.0) break;
        t += d;
    }
    if (!hit) discard;

    vec3 p = ro + rd * t;
    vec3 N = calcNormal(p);
    vec3 L = normalize(vec3(-0.4, 0.6, 0.8));
    float NdotL = max(dot(N, L), 0.0);
    vec3 H = normalize(L + vec3(0,0,1));
    float spec = pow(max(dot(N, H), 0.0), mix(8.0, 64.0, 1.0 - uRoughness));

    vec3 diffuse  = uAlbedo * NdotL * 0.7;
    vec3 ambient  = uAlbedo * 0.35;
    vec3 specular = vec3(spec * mix(0.02, 0.4, uMetallic));
    float fresnel = pow(1.0 - max(dot(N, vec3(0,0,1)), 0.0), 3.0);
    vec3 rim      = uGlowColor * fresnel * 0.3;
    vec3 scatter  = uAlbedo * max(dot(-N, L), 0.0) * 0.1;

    vec3 color = ambient + diffuse + specular + rim + scatter;
    // Filmic tone map — preserves saturation better than Reinhard
    color = color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(clamp(color, 0.0, 1.0), vec3(1.0/2.2));

    fragColor = vec4(color, uOpacity);
}
`;

// ─── Renderer class ──────────────────────────────────────────────────────────

export class CellMeshRenderer {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private quadVAO: WebGLVertexArrayObject;
  private fbo: WebGLFramebuffer | null = null;
  private colorTex: WebGLTexture | null = null;
  private fboW = 0;
  private fboH = 0;
  private _time = 0;

  // Uniforms
  private loc: Record<string, WebGLUniformLocation> = {};
  private lobesLoc: WebGLUniformLocation[] = [];

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.prog = this._compile();

    // Resolve uniforms
    const names = ['uCellPos','uCellSize','uAlbedo','uOpacity','uGlowColor',
      'uTime','uBaseRadius','uLobeCount','uNoiseAmp','uNoiseFreq','uRoughness','uMetallic'];
    for (const n of names) this.loc[n] = gl.getUniformLocation(this.prog, n)!;
    for (let i = 0; i < 8; i++) this.lobesLoc.push(gl.getUniformLocation(this.prog, `uLobes[${i}]`)!);

    // Quad VAO
    this.quadVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.quadVAO);
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    const aCorner = gl.getAttribLocation(this.prog, 'aCorner');
    gl.enableVertexAttribArray(aCorner);
    gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    console.info('[CellMeshRenderer] 3D SDF ray march ready');
  }

  setTime(t: number) { this._time = t; }
  get outputTexture(): WebGLTexture | null { return this.colorTex; }

  render(cells: CellData[], camScale: number, camOffX: number, camOffY: number, W: number, H: number): void {
    const gl = this.gl;
    if (!this.fbo || this.fboW !== W || this.fboH !== H) this._initFBO(W, H);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.prog);
    gl.uniform1f(this.loc.uTime, this._time);
    gl.bindVertexArray(this.quadVAO);

    for (const cell of cells) {
      const cx = ((cell.x * camScale + camOffX + cell.w * camScale * 0.5) / W) * 2 - 1;
      const cy = 1 - ((cell.y * camScale + camOffY + cell.h * camScale * 0.5) / H) * 2;
      const hw = cell.w * camScale / W;
      const hh = cell.h * camScale / H;

      gl.uniform2f(this.loc.uCellPos, cx, cy);
      gl.uniform2f(this.loc.uCellSize, hw * 1.3, hh * 1.3);
      gl.uniform3f(this.loc.uAlbedo, cell.albedo[0], cell.albedo[1], cell.albedo[2]);
      gl.uniform1f(this.loc.uOpacity, cell.opacity ?? 0.9);
      gl.uniform3f(this.loc.uGlowColor,
        cell.glowColor?.[0] ?? cell.albedo[0],
        cell.glowColor?.[1] ?? cell.albedo[1],
        cell.glowColor?.[2] ?? cell.albedo[2]);
      gl.uniform1f(this.loc.uRoughness, cell.roughness ?? 0.5);
      gl.uniform1f(this.loc.uMetallic, cell.metallic ?? 0.1);

      // SDF data from geometry.json
      const lobes = (cell as any).sdfLobes as Array<{angle:number,distance:number,radius:number}> | undefined;
      const sz = Math.min(cell.w, cell.h) || 1;  // normalize by short axis — cells are often wide+short
      if (lobes && lobes.length > 0) {
        const rawR = (cell as any).sdfBaseRadius ?? 20;
        gl.uniform1f(this.loc.uBaseRadius, Math.min(rawR / (sz * 0.5), 1.2));
        gl.uniform1i(this.loc.uLobeCount, Math.min(lobes.length, 8));
        for (let i = 0; i < Math.min(lobes.length, 8); i++)
          gl.uniform3f(this.lobesLoc[i], lobes[i].angle, lobes[i].distance/(sz*0.5), lobes[i].radius/(sz*0.5));
        gl.uniform1f(this.loc.uNoiseAmp, (cell as any).sdfNoiseAmp ?? 0.02);
        gl.uniform1f(this.loc.uNoiseFreq, (cell as any).sdfNoiseFreq ?? 4.0);
      } else {
        gl.uniform1f(this.loc.uBaseRadius, 0.7);
        gl.uniform1i(this.loc.uLobeCount, 0);
        gl.uniform1f(this.loc.uNoiseAmp, 0.015);
        gl.uniform1f(this.loc.uNoiseFreq, 3.0);
      }

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private _initFBO(W: number, H: number): void {
    const gl = this.gl;
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    if (this.colorTex) gl.deleteTexture(this.colorTex);

    this.colorTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    this.fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.colorTex, 0);

    const rb = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, W, H);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rb);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.fboW = W;
    this.fboH = H;
  }

  private _compile(): WebGLProgram {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, VERT);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS))
      throw new Error('[CellMesh] vert: ' + gl.getShaderInfoLog(vs));

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, FRAG);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS))
      throw new Error('[CellMesh] frag: ' + gl.getShaderInfoLog(fs));

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
      throw new Error('[CellMesh] link: ' + gl.getProgramInfoLog(prog));

    return prog;
  }
}
