/**
 * gpu-error-guard.ts — WebGL error recovery + safe shader compile
 *
 * Wraps gl calls to prevent one failed pass from crashing the entire pipeline.
 * Every gpu-pass should use safeCompile() instead of raw gl.createShader().
 */

const FALLBACK_VERT = `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const FALLBACK_FRAG = `
precision highp float;
varying vec2 vUv;
void main() {
  gl_FragColor = vec4(vUv, 0.5, 1.0);
}`;

let _fallbackProgram: WebGLProgram | null = null;

export function safeCompile(
  gl: WebGLRenderingContext,
  vert: string,
  frag: string,
  label: string
): WebGLProgram {
  const vs = gl.createShader(gl.VERTEX_SHADER)!;
  gl.shaderSource(vs, vert);
  gl.compileShader(vs);
  if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
    console.error(`[GPU-GUARD] vertex compile FAIL (${label}):`, gl.getShaderInfoLog(vs));
    gl.deleteShader(vs);
    return _getFallback(gl);
  }

  const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
  gl.shaderSource(fs, 'precision highp float;\n' + frag);
  gl.compileShader(fs);
  if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
    console.error(`[GPU-GUARD] fragment compile FAIL (${label}):`, gl.getShaderInfoLog(fs));
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return _getFallback(gl);
  }

  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error(`[GPU-GUARD] link FAIL (${label}):`, gl.getProgramInfoLog(prog));
    gl.deleteProgram(prog);
    return _getFallback(gl);
  }

  console.log(`[GPU-GUARD] ✅ ${label} compiled OK`);
  return prog;
}

function _getFallback(gl: WebGLRenderingContext): WebGLProgram {
  if (_fallbackProgram) return _fallbackProgram;
  const vs = gl.createShader(gl.VERTEX_SHADER)!;
  gl.shaderSource(vs, FALLBACK_VERT);
  gl.compileShader(vs);
  const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
  gl.shaderSource(fs, FALLBACK_FRAG);
  gl.compileShader(fs);
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  _fallbackProgram = prog;
  return prog;
}

export function checkFBO(gl: WebGLRenderingContext, label: string): boolean {
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    console.error(`[GPU-GUARD] FBO incomplete (${label}): 0x${status.toString(16)}`);
    return false;
  }
  return true;
}

export function drainErrors(gl: WebGLRenderingContext): number {
  let count = 0;
  let err: number;
  while ((err = gl.getError()) !== gl.NO_ERROR) {
    console.warn(`[GPU-GUARD] gl error: 0x${err.toString(16)}`);
    count++;
  }
  return count;
}

export function setupContextLost(canvas: HTMLCanvasElement, onLost: () => void, onRestored: () => void): void {
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    console.error('[GPU-GUARD] WebGL context lost');
    onLost();
  });
  canvas.addEventListener('webglcontextrestored', () => {
    console.log('[GPU-GUARD] WebGL context restored');
    onRestored();
  });
}
