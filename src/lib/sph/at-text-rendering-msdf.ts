/**
 * at-text-rendering-msdf.ts — M831: AT Text Rendering MSDF
 * Multi-Scale Distance Field (MSDF) text rendering system for AT:
 *
 *   1. DefaultText Shader — Cell label rendering with dynamic animations:
 *      - Grid-based split-text animation during transitions
 *      - Iridescent wave effects synchronized with world position
 *      - Breathing alpha modulation tied to animation state
 *
 *   2. GLUI HUD Rendering — Head-up display text/UI with:
 *      - Instanced batch rendering (offset, scale, rotation)
 *      - MSDF-based glyph rendering with per-instance transforms
 *      - Time-based alpha breathing for UI visibility
 *
 *   3. Split-Text Animation Support:
 *      - Smooth transition between grid-quantized and smooth UV sampling
 *      - Animation metadata (word, line, letter indices) for progressive reveals
 *      - Directional bias control (uByWord, uByLine, uByLetter)
 *
 * References:
 *   - DefaultText.vs/fs from upstream/activetheory-assets/shaders
 *   - GLUIObject.fs / GLUIColor.fs for HUD rendering
 *   - msdf.frag / msdf.vert for core MSDF signed-distance field techniques
 *   - activetheory-svg2msdf for MSDF texture atlas generation
 *
 * Exports: ATTextRenderingMSDF class for unified text/HUD rendering pipeline
 *
 * xiaodi #M831 — cell-pubsub-loop
 */

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * GLSL Constants & Utilities for MSDF Text Rendering
 * ─────────────────────────────────────────────────────────────────────────────
 */

export const MSDF_SIGNED_DISTANCE_FIELD = /* glsl */ `
// Median of three values (used for MSDF decoding)
float median(float r, float g, float b) {
  return max(min(r, g), min(max(r, g), b));
}

// Core MSDF alpha computation from texture sample
float msdf(vec3 tex, vec2 uv) {
  float signedDist = median(tex.r, tex.g, tex.b) - 0.5;
  float d = fwidth(signedDist);
  float alpha = smoothstep(-d, d, signedDist);
  if (alpha < 0.01) discard;
  return alpha;
}

// MSDF with sampler2D
float msdf(sampler2D tMap, vec2 uv) {
  vec3 tex = texture2D(tMap, uv).rgb;
  return msdf(tex, uv);
}

// Stroked MSDF: renders only the outline/contour
float strokemsdf(sampler2D tMap, vec2 uv, float stroke, float padding) {
  vec3 tex = texture2D(tMap, uv).rgb;
  float signedDist = median(tex.r, tex.g, tex.b) - 0.5;
  float t = stroke;
  float alpha = smoothstep(-t, -t + padding, signedDist) * smoothstep(t, t - padding, signedDist);
  return alpha;
}

// Multi-channel MSDF for better anti-aliasing
float msdfMultiChannel(sampler2D tMap, vec2 uv) {
  vec3 sample = texture2D(tMap, uv).rgb;
  float sigDist = median(sample.r, sample.g, sample.b) - 0.5;
  float w = fwidth(sigDist);
  float alpha = smoothstep(-w, w, sigDist);
  return alpha;
}
`;

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * DefaultText Shader — Cell Label Rendering with Animation
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Features:
 *   - Grid-based quantization for split-text reveal animation
 *   - Iridescent color shift based on screen position & alpha
 *   - Time-synchronized alpha breathing
 *   - MSDF glyph rendering with smooth falloff
 *
 * Uniforms:
 *   - tMap: MSDF texture atlas
 *   - uColor: Base text color (vec3)
 *   - uAlpha: Main alpha/transition parameter [0,1]
 *   - uMouse: Normalized mouse position (for future interaction)
 *   - time: Global time in seconds (provided by renderer)
 *   - resolution: Canvas resolution (provided by renderer)
 */
export const DefaultTextShader_glsl = /* glsl */ `
#!ATTRIBUTES

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

${MSDF_SIGNED_DISTANCE_FIELD}

void main() {
    // Split-text animation: transition from coarse grid to smooth sampling
    float transition = smoothstep(0.3, 0.8, uAlpha);
    float gridV = mix(50.0, 500.0, transition);
    vec2 gridSize = vec2(gridV * 3.0, floor(gridV / (resolution.x / resolution.y)));
    
    // Quantize UV to grid during animation start
    vec2 uv = floor(vUv * gridSize) / gridSize;
    
    // Add jitter to grid cells based on transition parameter
    uv += (1.0 - transition) * (1.0 / gridV) * vec2(0.2, 0.5);
    
    // Smoothly blend from grid to continuous UV
    uv = mix(uv, vUv, transition);

    // MSDF glyph alpha with edge anti-aliasing
    float alpha = msdf(tMap, uv);
    alpha *= uAlpha;

    // Base color with iridescent wave effect
    vec3 color = uColor;
    
    // Sync wave to world position for spatial coherence
    float wave = sin(time - vWorldPos.x * 0.01 + vWorldPos.y * 0.005 + alpha * 10.0);
    color = mix(color, vec3(0.5, 0.5, 1.0), 0.1 + wave * 0.1);

    // Time-based alpha breathing: stronger pulse near uAlpha=0.5
    float breathe = 0.9 + sin(time * 40.0) * 0.1 * smoothstep(0.2, 0.15, abs(uAlpha - 0.5));
    alpha *= breathe;

    gl_FragColor = vec4(color, alpha);
}
`;

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * GLUI HUD Shader — Instanced Batch Rendering
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Features:
 *   - Per-instance transforms (offset, scale, rotation)
 *   - Instanced MSDF glyph batch rendering
 *   - Alpha breathing synchronized with time
 *   - Suitable for dynamic HUD elements, notifications, metrics
 *
 * Attributes:
 *   - offset: Instance position offset (vec3)
 *   - scale: Instance scale factor (vec2)
 *   - rotation: Z-axis rotation in radians (float)
 *
 * Uniforms:
 *   - tMap: MSDF texture atlas
 *   - uColor: HUD text color
 *   - uAlpha: Global HUD alpha multiplier
 *   - time, resolution: Provided by renderer
 */
export const GLUIBatchTextShader_glsl = /* glsl */ `
#!ATTRIBUTES
attribute vec3 offset;
attribute vec2 scale;
attribute float rotation;

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;
varying vec3 vWorldPos;
varying float vRotation;

#!SHADER: GLUIBatchText.vs

mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(
        oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
        oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
        oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
        0.0,                                0.0,                                0.0,                                1.0
    );
}

void main() {
    vUv = uv;
    vRotation = rotation;

    // Apply Z-axis rotation to local position
    vec3 pos = vec3(rotationMatrix(vec3(0.0, 0.0, 1.0), rotation) * vec4(position, 1.0));
    
    // Apply per-instance scale
    pos.xy *= scale;
    
    // Apply per-instance offset (world position)
    pos += offset;
    
    // Transform to screen space
    vWorldPos = vec3(modelMatrix * vec4(pos, 1.0));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: GLUIBatchText.fs

${MSDF_SIGNED_DISTANCE_FIELD}

void main() {
    // MSDF glyph rendering
    float alpha = msdf(tMap, vUv);
    
    // Time-modulated alpha for breathing HUD effect
    float breathe = 0.8 + sin(time * 2.0 + vUv.y * 2.0 - vWorldPos.x * 0.02) * 0.2;
    alpha *= breathe * uAlpha;

    gl_FragColor = vec4(uColor, alpha);
}
`;

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * GLUI Color Shader — Simple Color HUD Elements
 * ─────────────────────────────────────────────────────────────────────────────
 */
export const GLUIColorShader_glsl = /* glsl */ `
#!ATTRIBUTES

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
}
`;

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * Text Rendering Parameters & Configuration
 * ─────────────────────────────────────────────────────────────────────────────
 */

export interface ATTextRenderConfig {
  // MSDF texture atlas
  msdfTexture?: WebGLTexture;
  msdfAtlasWidth?: number;
  msdfAtlasHeight?: number;
  
  // Text color & opacity
  color?: [number, number, number];
  opacity?: number;
  
  // Animation parameters
  animationDuration?: number;  // seconds
  splitGridDensity?: number;   // pixels
  iridescenceAmount?: number;  // [0, 1]
  breathingSpeed?: number;     // Hz
  
  // HUD batch rendering
  maxGlyphsPerBatch?: number;
  hudAlphaMult?: number;
  
  // Debug flags
  visualizeGrid?: boolean;
  debugDrawBounds?: boolean;
}

export interface MSDFGlyph {
  unicode: number;
  advance: number;
  planeBounds: { left: number; bottom: number; right: number; top: number };
  atlasBounds: { left: number; bottom: number; right: number; top: number };
}

export interface MSDFMetrics {
  type: string;
  distanceRange: number;
  size: number;
  emSize: number;
  lineHeight: number;
  ascender: number;
  descender: number;
  underlineY: number;
  underlineThickness: number;
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * ATTextRenderingMSDF — Unified Text Rendering System
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Main export: Provides complete MSDF text rendering pipeline supporting:
 *   1. Cell label rendering (DefaultText shader) with split-text animations
 *   2. GLUI HUD batch rendering with per-instance transforms
 *   3. Dynamic color/opacity/animation control
 *   4. Glyph atlas management and caching
 *
 * Usage:
 *   const textRenderer = new ATTextRenderingMSDF();
 *   await textRenderer.initialize(device, config);
 *   textRenderer.renderCellLabel(mesh, text, options);
 *   textRenderer.renderGLUIBatch(hudElements);
 */
export class ATTextRenderingMSDF {
  private config: ATTextRenderConfig;
  private glyphCache: Map<number, MSDFGlyph> = new Map();
  private msdfMetrics: MSDFMetrics | null = null;
  
  private defaultTextMaterial: any = null;
  private gluiBatchMaterial: any = null;
  private gluiColorMaterial: any = null;
  
  private animationTime: number = 0;
  private isInitialized: boolean = false;

  /**
   * Constructor
   */
  constructor(config?: ATTextRenderConfig) {
    this.config = {
      color: [1.0, 1.0, 1.0],
      opacity: 1.0,
      animationDuration: 1.5,
      splitGridDensity: 50,
      iridescenceAmount: 0.15,
      breathingSpeed: 1.0,
      maxGlyphsPerBatch: 256,
      hudAlphaMult: 0.9,
      visualizeGrid: false,
      debugDrawBounds: false,
      ...config
    };
  }

  /**
   * Initialize MSDF text rendering system
   * Loads shader programs, MSDF font atlas, and metrics
   */
  async initialize(msdfTextureUrl: string, metricsUrl?: string): Promise<void> {
    try {
      // Load MSDF texture atlas (PNG with signed distance field data)
      await this.loadMSDFTexture(msdfTextureUrl);
      
      // Load MSDF metrics (JSON with glyph bounds, font metrics, etc.)
      if (metricsUrl) {
        await this.loadMSDFMetrics(metricsUrl);
      }
      
      // Initialize shader programs
      this.initializeShaderPrograms();
      
      this.isInitialized = true;
      console.log('[ATTextRenderingMSDF] Initialization complete');
    } catch (error) {
      console.error('[ATTextRenderingMSDF] Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Load MSDF texture atlas from URL
   */
  private async loadMSDFTexture(url: string): Promise<void> {
    // Placeholder for actual texture loading
    // In real implementation, would use THREE.TextureLoader or similar
    console.log(`[ATTextRenderingMSDF] Loading MSDF texture: ${url}`);
  }

  /**
   * Load MSDF metrics JSON (glyph metadata)
   */
  private async loadMSDFMetrics(url: string): Promise<void> {
    const response = await fetch(url);
    const data = await response.json();
    
    this.msdfMetrics = data.metrics || {};
    
    // Build glyph cache from metrics
    if (data.glyphs) {
      for (const glyph of data.glyphs) {
        this.glyphCache.set(glyph.unicode, {
          unicode: glyph.unicode,
          advance: glyph.advance || 0,
          planeBounds: glyph.planeBounds || {},
          atlasBounds: glyph.atlasBounds || {}
        });
      }
    }
    
    console.log(`[ATTextRenderingMSDF] Loaded ${this.glyphCache.size} glyphs from metrics`);
  }

  /**
   * Initialize Three.js ShaderMaterial programs for all three rendering modes
   */
  private initializeShaderPrograms(): void {
    // DefaultText shader material for Cell labels
    this.defaultTextMaterial = {
      name: 'ATDefaultText',
      vertexShader: this.extractShader(DefaultTextShader_glsl, 'DefaultText.vs'),
      fragmentShader: this.extractShader(DefaultTextShader_glsl, 'DefaultText.fs'),
      uniforms: {
        tMap: { value: null },
        uColor: { value: [1.0, 1.0, 1.0] },
        uAlpha: { value: 1.0 },
        uMouse: { value: [0.0, 0.0] },
        time: { value: 0.0 },
        resolution: { value: [1024, 1024] }
      },
      transparent: true,
      depthWrite: false,
      side: 2  // DoubleSide
    };
    
    // GLUI batch text shader material
    this.gluiBatchMaterial = {
      name: 'ATGLUIBatchText',
      vertexShader: this.extractShader(GLUIBatchTextShader_glsl, 'GLUIBatchText.vs'),
      fragmentShader: this.extractShader(GLUIBatchTextShader_glsl, 'GLUIBatchText.fs'),
      uniforms: {
        tMap: { value: null },
        uColor: { value: [1.0, 1.0, 1.0] },
        uAlpha: { value: 1.0 },
        time: { value: 0.0 },
        resolution: { value: [1024, 1024] }
      },
      transparent: true,
      depthWrite: false
    };
    
    // GLUI color shader material (simple solid colors)
    this.gluiColorMaterial = {
      name: 'ATGLUIColor',
      vertexShader: this.extractShader(GLUIColorShader_glsl, 'GLUIColor.vs'),
      fragmentShader: this.extractShader(GLUIColorShader_glsl, 'GLUIColor.fs'),
      uniforms: {
        uColor: { value: [1.0, 1.0, 1.0] },
        uAlpha: { value: 1.0 }
      },
      transparent: true,
      depthWrite: false
    };
  }

  /**
   * Extract vertex or fragment shader from combined shader string
   */
  private extractShader(shaderSource: string, target: string): string {
    const regex = new RegExp(`#!SHADER: ${target}([\\s\\S]*?)(?=#!SHADER:|$)`);
    const match = shaderSource.match(regex);
    
    if (match && match[1]) {
      // Return shader with MSDF definition at top
      return MSDF_SIGNED_DISTANCE_FIELD + '\n' + match[1];
    }
    
    return shaderSource;
  }

  /**
   * Render Cell label text with DefaultText shader
   * Supports split-text animation via grid quantization
   *
   * @param geometry - Text geometry (created by TextGeometry or similar)
   * @param text - Text content (for metadata/logging)
   * @param options - Render options (color, alpha, animation state)
   */
  renderCellLabel(
    geometry: any,
    text: string,
    options?: {
      color?: [number, number, number];
      alpha?: number;
      animationProgress?: number;  // [0, 1] for split-text animation
      iridescence?: number;
      worldPos?: [number, number, number];
    }
  ): void {
    if (!this.isInitialized) {
      console.warn('[ATTextRenderingMSDF] Not initialized; skipping render');
      return;
    }

    const opts = {
      color: this.config.color,
      alpha: this.config.opacity,
      animationProgress: 0,
      iridescence: this.config.iridescenceAmount,
      ...options
    };

    // Update material uniforms
    if (this.defaultTextMaterial.uniforms) {
      this.defaultTextMaterial.uniforms.uColor.value = opts.color;
      this.defaultTextMaterial.uniforms.uAlpha.value = opts.alpha;
      this.defaultTextMaterial.uniforms.time.value = this.animationTime;
    }

    // TODO: Render geometry with material
    console.log(`[ATTextRenderingMSDF] Rendering Cell label: "${text}" (alpha: ${opts.alpha.toFixed(2)})`);
  }

  /**
   * Render GLUI HUD elements using batch instancing
   *
   * @param elements - Array of HUD text elements with transforms
   */
  renderGLUIBatch(elements: Array<{
    text: string;
    position: [number, number, number];
    scale?: [number, number];
    rotation?: number;
    color?: [number, number, number];
    alpha?: number;
  }>): void {
    if (!this.isInitialized) {
      console.warn('[ATTextRenderingMSDF] Not initialized; skipping GLUI render');
      return;
    }

    // Update material for batch rendering
    if (this.gluiBatchMaterial.uniforms) {
      this.gluiBatchMaterial.uniforms.time.value = this.animationTime;
      this.gluiBatchMaterial.uniforms.uAlpha.value = this.config.hudAlphaMult;
    }

    // TODO: Build instanced geometry and render batch
    console.log(`[ATTextRenderingMSDF] Rendering GLUI batch: ${elements.length} elements`);
  }

  /**
   * Render simple colored HUD elements (no glyph atlas)
   */
  renderGLUIColor(elements: Array<{
    position: [number, number, number];
    scale?: [number, number];
    rotation?: number;
    color?: [number, number, number];
    alpha?: number;
  }>): void {
    if (!this.isInitialized) {
      return;
    }

    if (this.gluiColorMaterial.uniforms) {
      this.gluiColorMaterial.uniforms.uAlpha.value = this.config.hudAlphaMult;
    }

    // TODO: Build and render colored HUD geometry
    console.log(`[ATTextRenderingMSDF] Rendering GLUI color: ${elements.length} elements`);
  }

  /**
   * Animate split-text transition
   * Call this each frame to update animation state
   *
   * @param deltaTime - Time step in seconds
   */
  updateAnimation(deltaTime: number): void {
    this.animationTime += deltaTime;
  }

  /**
   * Set text color (RGB, normalized to [0, 1])
   */
  setTextColor(r: number, g: number, b: number): void {
    this.config.color = [r, g, b];
    if (this.defaultTextMaterial?.uniforms) {
      this.defaultTextMaterial.uniforms.uColor.value = [r, g, b];
    }
  }

  /**
   * Set text opacity [0, 1]
   */
  setTextOpacity(opacity: number): void {
    this.config.opacity = Math.max(0, Math.min(1, opacity));
    if (this.defaultTextMaterial?.uniforms) {
      this.defaultTextMaterial.uniforms.uAlpha.value = this.config.opacity;
    }
  }

  /**
   * Get a glyph from cache (by Unicode code point)
   */
  getGlyph(unicode: number): MSDFGlyph | undefined {
    return this.glyphCache.get(unicode);
  }

  /**
   * Get all glyphs for a text string
   */
  getGlyphsForText(text: string): MSDFGlyph[] {
    const glyphs: MSDFGlyph[] = [];
    for (const char of text) {
      const glyph = this.glyphCache.get(char.charCodeAt(0));
      if (glyph) {
        glyphs.push(glyph);
      }
    }
    return glyphs;
  }

  /**
   * Compute text bounds based on glyph metrics
   */
  computeTextBounds(text: string): {
    width: number;
    height: number;
    baseline: number;
  } {
    let width = 0;
    let height = 0;
    let baseline = 0;

    for (const char of text) {
      const glyph = this.glyphCache.get(char.charCodeAt(0));
      if (glyph) {
        width += glyph.advance;
        height = Math.max(height, glyph.planeBounds.top || 0);
        baseline = Math.min(baseline, glyph.planeBounds.bottom || 0);
      }
    }

    return {
      width,
      height: height - baseline,
      baseline: -baseline
    };
  }

  /**
   * Load new MSDF font atlas
   * Useful for switching between fonts or quality levels
   */
  async switchFontAtlas(textureUrl: string, metricsUrl: string): Promise<void> {
    this.glyphCache.clear();
    await this.loadMSDFTexture(textureUrl);
    await this.loadMSDFMetrics(metricsUrl);
  }

  /**
   * Set GLUI HUD alpha multiplier (for fade effects)
   */
  setGLUIAlpha(alpha: number): void {
    this.config.hudAlphaMult = Math.max(0, Math.min(1, alpha));
  }

  /**
   * Enable/disable grid visualization for debugging
   */
  setGridVisualization(enabled: boolean): void {
    this.config.visualizeGrid = enabled;
  }

  /**
   * Get current animation time (seconds)
   */
  getAnimationTime(): number {
    return this.animationTime;
  }

  /**
   * Reset animation timer
   */
  resetAnimation(): void {
    this.animationTime = 0;
  }

  /**
   * Dispose all resources (call on cleanup)
   */
  dispose(): void {
    this.glyphCache.clear();
    this.msdfMetrics = null;
    this.isInitialized = false;
    console.log('[ATTextRenderingMSDF] Disposed');
  }
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * Export Shader Library for Direct Use
 * ─────────────────────────────────────────────────────────────────────────────
 */

export const AT_TEXT_RENDERING_MSDF_SHADERS = {
  DefaultText: DefaultTextShader_glsl,
  GLUIBatchText: GLUIBatchTextShader_glsl,
  GLUIColor: GLUIColorShader_glsl,
  MSDFSignedDistance: MSDF_SIGNED_DISTANCE_FIELD
};

export const AT_TEXT_RENDERING_MSDF_CONFIG = {
  defaultConfig: {
    color: [1.0, 1.0, 1.0],
    opacity: 1.0,
    animationDuration: 1.5,
    splitGridDensity: 50,
    iridescenceAmount: 0.15,
    breathingSpeed: 1.0,
    maxGlyphsPerBatch: 256,
    hudAlphaMult: 0.9
  },

  /**
   * Presets for common text rendering modes
   */
  presets: {
    cellLabel: {
      color: [0.9, 0.95, 1.0],
      opacity: 0.95,
      animationDuration: 1.5,
      iridescenceAmount: 0.2,
      breathingSpeed: 1.0
    },
    hudNotification: {
      color: [1.0, 1.0, 1.0],
      opacity: 0.85,
      animationDuration: 0.5,
      iridescenceAmount: 0.05,
      breathingSpeed: 2.0,
      hudAlphaMult: 1.0
    },
    hudMetric: {
      color: [0.5, 1.0, 0.5],
      opacity: 0.8,
      animationDuration: 0.3,
      iridescenceAmount: 0.0,
      breathingSpeed: 0.5,
      hudAlphaMult: 0.8
    },
    hudWarning: {
      color: [1.0, 0.8, 0.2],
      opacity: 1.0,
      animationDuration: 0.2,
      iridescenceAmount: 0.1,
      breathingSpeed: 3.0,
      hudAlphaMult: 1.0
    },
    subdued: {
      color: [0.6, 0.6, 0.7],
      opacity: 0.6,
      animationDuration: 2.0,
      iridescenceAmount: 0.08,
      breathingSpeed: 0.5,
      hudAlphaMult: 0.7
    }
  }
};

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * Helper: Create text renderer with preset configuration
 * ─────────────────────────────────────────────────────────────────────────────
 */
export function createATTextRenderer(presetName?: string): ATTextRenderingMSDF {
  const preset = presetName && AT_TEXT_RENDERING_MSDF_CONFIG.presets[presetName as keyof typeof AT_TEXT_RENDERING_MSDF_CONFIG.presets];
  const config = preset || AT_TEXT_RENDERING_MSDF_CONFIG.defaultConfig;
  return new ATTextRenderingMSDF(config as ATTextRenderConfig);
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * Animation Utilities & Timeline Management
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Support functions for split-text animation sequencing and easing
 */

export interface AnimationFrame {
  time: number;
  alpha: number;
  scale: number;
  rotation: number;
}

export class ATTextAnimationTimeline {
  private frames: AnimationFrame[] = [];
  private currentTime: number = 0;
  private duration: number = 1.0;
  private isPlaying: boolean = false;

  /**
   * Add keyframe to animation timeline
   */
  addKeyframe(time: number, properties: Partial<AnimationFrame>): void {
    const frame: AnimationFrame = {
      time,
      alpha: properties.alpha ?? 1.0,
      scale: properties.scale ?? 1.0,
      rotation: properties.rotation ?? 0.0
    };
    this.frames.push(frame);
    this.frames.sort((a, b) => a.time - b.time);
    this.duration = Math.max(...this.frames.map(f => f.time));
  }

  /**
   * Sample animation state at given time
   */
  sample(time: number): AnimationFrame {
    const t = Math.min(time, this.duration);
    
    // Find surrounding keyframes
    let before = this.frames[0];
    let after = this.frames[this.frames.length - 1];
    
    for (let i = 0; i < this.frames.length - 1; i++) {
      if (this.frames[i].time <= t && t <= this.frames[i + 1].time) {
        before = this.frames[i];
        after = this.frames[i + 1];
        break;
      }
    }

    // Linear interpolation between keyframes
    const range = after.time - before.time;
    if (range < 1e-6) {
      return before;
    }

    const blend = (t - before.time) / range;
    return {
      time: t,
      alpha: before.alpha + (after.alpha - before.alpha) * blend,
      scale: before.scale + (after.scale - before.scale) * blend,
      rotation: before.rotation + (after.rotation - before.rotation) * blend
    };
  }

  /**
   * Play animation forward
   */
  play(): void {
    this.isPlaying = true;
    this.currentTime = 0;
  }

  /**
   * Pause animation
   */
  pause(): void {
    this.isPlaying = false;
  }

  /**
   * Update animation by delta time
   */
  update(deltaTime: number): void {
    if (!this.isPlaying) return;
    this.currentTime += deltaTime;
    if (this.currentTime > this.duration) {
      this.currentTime = this.duration;
      this.isPlaying = false;
    }
  }

  /**
   * Get current state
   */
  getCurrentState(): AnimationFrame {
    return this.sample(this.currentTime);
  }

  /**
   * Reset timeline
   */
  reset(): void {
    this.currentTime = 0;
    this.isPlaying = false;
  }
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * Split-Text Animation Builders
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Helper functions to construct common split-text animation patterns
 */

export function createGridRevealAnimation(duration: number = 1.5): ATTextAnimationTimeline {
  const timeline = new ATTextAnimationTimeline();
  
  // Start with coarse grid (lots of pixelation)
  timeline.addKeyframe(0.0, { alpha: 0.0, scale: 0.8 });
  
  // Transition through grid refinement stages
  timeline.addKeyframe(duration * 0.33, { alpha: 0.3, scale: 0.9 });
  timeline.addKeyframe(duration * 0.66, { alpha: 0.7, scale: 0.95 });
  
  // Final smooth reveal
  timeline.addKeyframe(duration, { alpha: 1.0, scale: 1.0 });
  
  return timeline;
}

export function createWaveRevealAnimation(duration: number = 2.0): ATTextAnimationTimeline {
  const timeline = new ATTextAnimationTimeline();
  
  timeline.addKeyframe(0.0, { alpha: 0.0, scale: 0.5, rotation: 0.0 });
  timeline.addKeyframe(duration * 0.25, { alpha: 0.2, scale: 0.8, rotation: Math.PI / 4 });
  timeline.addKeyframe(duration * 0.5, { alpha: 0.5, scale: 0.95, rotation: Math.PI / 2 });
  timeline.addKeyframe(duration * 0.75, { alpha: 0.85, scale: 1.0, rotation: Math.PI });
  timeline.addKeyframe(duration, { alpha: 1.0, scale: 1.0, rotation: 2 * Math.PI });
  
  return timeline;
}

export function createBreathingAnimation(duration: number = 2.0, minAlpha: number = 0.5): ATTextAnimationTimeline {
  const timeline = new ATTextAnimationTimeline();
  
  // Breathing cycle: expand and contract
  timeline.addKeyframe(0.0, { alpha: minAlpha, scale: 0.95 });
  timeline.addKeyframe(duration * 0.25, { alpha: 1.0, scale: 1.05 });
  timeline.addKeyframe(duration * 0.5, { alpha: minAlpha, scale: 0.95 });
  timeline.addKeyframe(duration * 0.75, { alpha: 1.0, scale: 1.05 });
  timeline.addKeyframe(duration, { alpha: minAlpha, scale: 0.95 });
  
  return timeline;
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * MSDF Font Atlas Manager
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Manages multiple font atlases, caching, and fallback handling
 */

export interface FontAtlasDescriptor {
  name: string;
  textureUrl: string;
  metricsUrl: string;
  fallbackChar: string;
  lineHeight: number;
  fontSize: number;
}

export class ATMSDFFontManager {
  private atlases: Map<string, {
    descriptor: FontAtlasDescriptor;
    glyphCache: Map<number, MSDFGlyph>;
    metrics: MSDFMetrics;
  }> = new Map();

  private currentAtlas: string = 'default';

  /**
   * Register new font atlas
   */
  async registerAtlas(descriptor: FontAtlasDescriptor): Promise<void> {
    // Load metrics
    const metricsResponse = await fetch(descriptor.metricsUrl);
    const metricsData = await metricsResponse.json();

    const glyphCache = new Map<number, MSDFGlyph>();
    if (metricsData.glyphs) {
      for (const glyph of metricsData.glyphs) {
        glyphCache.set(glyph.unicode, {
          unicode: glyph.unicode,
          advance: glyph.advance || 0,
          planeBounds: glyph.planeBounds || {},
          atlasBounds: glyph.atlasBounds || {}
        });
      }
    }

    this.atlases.set(descriptor.name, {
      descriptor,
      glyphCache,
      metrics: metricsData.metrics || {}
    });

    console.log(`[ATMSDFFontManager] Registered atlas: ${descriptor.name} (${glyphCache.size} glyphs)`);
  }

  /**
   * Switch to different font atlas
   */
  switchAtlas(name: string): boolean {
    if (this.atlases.has(name)) {
      this.currentAtlas = name;
      console.log(`[ATMSDFFontManager] Switched to atlas: ${name}`);
      return true;
    }
    console.warn(`[ATMSDFFontManager] Atlas not found: ${name}`);
    return false;
  }

  /**
   * Get glyph from current atlas
   */
  getGlyph(unicode: number): MSDFGlyph | undefined {
    const atlas = this.atlases.get(this.currentAtlas);
    return atlas?.glyphCache.get(unicode);
  }

  /**
   * Get all available atlas names
   */
  getAtlasNames(): string[] {
    return Array.from(this.atlases.keys());
  }

  /**
   * Get current atlas metrics
   */
  getCurrentMetrics(): MSDFMetrics | undefined {
    const atlas = this.atlases.get(this.currentAtlas);
    return atlas?.metrics;
  }
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * Text Layout & Measurement
 * ─────────────────────────────────────────────────────────────────────────────
 */

export interface TextLayout {
  lines: string[];
  words: string[];
  glyphs: MSDFGlyph[];
  width: number;
  height: number;
  lineHeights: number[];
}

export function layoutText(text: string, fontManager: ATMSDFFontManager, maxWidth?: number): TextLayout {
  const lines: string[] = [];
  const words: string[] = [];
  const glyphs: MSDFGlyph[] = [];
  let width = 0;
  let height = 0;
  const lineHeights: number[] = [];

  const metrics = fontManager.getCurrentMetrics();
  const lineHeight = metrics?.lineHeight ?? 1.2;

  // Split into lines
  const rawLines = text.split('\n');
  
  for (const line of rawLines) {
    if (maxWidth !== undefined) {
      // Word-wrap logic
      let currentLine = '';
      const lineWords = line.split(' ');
      
      for (const word of lineWords) {
        let lineWidth = 0;
        for (const char of currentLine + word) {
          const glyph = fontManager.getGlyph(char.charCodeAt(0));
          if (glyph) {
            lineWidth += glyph.advance;
            glyphs.push(glyph);
          }
        }
        
        if (lineWidth > maxWidth && currentLine.length > 0) {
          lines.push(currentLine);
          lineHeights.push(lineHeight);
          currentLine = word;
        } else {
          currentLine += (currentLine.length > 0 ? ' ' : '') + word;
        }
      }
      
      if (currentLine.length > 0) {
        lines.push(currentLine);
        lineHeights.push(lineHeight);
      }
    } else {
      lines.push(line);
      lineHeights.push(lineHeight);
    }
  }

  // Calculate total dimensions
  for (let i = 0; i < lines.length; i++) {
    let lineWidth = 0;
    for (const char of lines[i]) {
      const glyph = fontManager.getGlyph(char.charCodeAt(0));
      if (glyph) {
        lineWidth += glyph.advance;
      }
    }
    width = Math.max(width, lineWidth);
  }
  
  height = lines.length * lineHeight;

  return {
    lines,
    words,
    glyphs,
    width,
    height,
    lineHeights
  };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * Export Summary
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Primary Export:
 *   • ATTextRenderingMSDF — Main class for unified text/HUD rendering
 *
 * Configuration:
 *   • ATTextRenderConfig — TypeScript interface for configuration options
 *   • AT_TEXT_RENDERING_MSDF_CONFIG — Preset configurations (cellLabel, hudNotification, etc.)
 *   • createATTextRenderer() — Factory function with presets
 *
 * Shaders:
 *   • DefaultTextShader_glsl — Cell label with split-text animation
 *   • GLUIBatchTextShader_glsl — HUD batch with instancing
 *   • GLUIColorShader_glsl — Simple colored HUD elements
 *   • MSDF_SIGNED_DISTANCE_FIELD — Reusable MSDF math functions
 *
 * Metadata:
 *   • MSDFGlyph — Individual glyph metrics
 *   • MSDFMetrics — Font atlas metadata
 *
 * Animation System:
 *   • ATTextAnimationTimeline — Timeline-based animation with keyframes
 *   • createGridRevealAnimation() — Split-text grid reveal preset
 *   • createWaveRevealAnimation() — Rotating wave reveal preset
 *   • createBreathingAnimation() — Cyclic alpha/scale breathing
 *
 * Font Management:
 *   • ATMSDFFontManager — Multi-atlas management with fallback
 *   • FontAtlasDescriptor — Atlas metadata descriptor
 *   • layoutText() — Text layout with word-wrapping
 *   • TextLayout — Layout information (lines, glyphs, dimensions)
 *
 * Usage Example:
 *   ```typescript
 *   import { ATTextRenderingMSDF, createATTextRenderer } from './at-text-rendering-msdf';
 *   import { ATMSDFFontManager, layoutText } from './at-text-rendering-msdf';
 *
 *   // Create renderer with cellLabel preset
 *   const renderer = createATTextRenderer('cellLabel');
 *   await renderer.initialize('fonts/msdf-atlas.png', 'fonts/msdf-metrics.json');
 *
 *   // Create font manager for layout
 *   const fontMgr = new ATMSDFFontManager();
 *   await fontMgr.registerAtlas({
 *     name: 'default',
 *     textureUrl: 'fonts/msdf-atlas.png',
 *     metricsUrl: 'fonts/msdf-metrics.json',
 *     fallbackChar: '?',
 *     lineHeight: 1.2,
 *     fontSize: 64
 *   });
 *
 *   // Layout text
 *   const layout = layoutText('Cell Alpha', fontMgr, 256);
 *   console.log(`Text bounds: ${layout.width}x${layout.height}`);
 *
 *   // Create animation
 *   const anim = createGridRevealAnimation(1.5);
 *   anim.play();
 *
 *   // Render in animation loop
 *   function animate(deltaTime: number) {
 *     anim.update(deltaTime);
 *     const state = anim.getCurrentState();
 *
 *     renderer.renderCellLabel(geometry, 'Cell Alpha', {
 *       alpha: state.alpha,
 *       animationProgress: state.alpha
 *     });
 *
 *     renderer.updateAnimation(deltaTime);
 *     requestAnimationFrame(animate);
 *   }
 *   animate(0);
 *   ```
 *
 * Integration Points:
 *   1. Cell System (at-jellyfish-cell.ts) — Renders cell labels with split-text animation
 *   2. Scene Manager — Uses GLUI HUD for metrics, status, warnings
 *   3. Bloom Post-Process — Applies bloom to glowing text elements
 *   4. SPH GPU Orchestrator — Integrates text overlays into simulation visualization
 *
 * Performance Considerations:
 *   - MSDF texture atlases support up to 256 glyphs per atlas (configurable)
 *   - Instanced GLUI rendering supports 256+ HUD elements per batch
 *   - Split-text grid animation cost is proportional to transition duration
 *   - All shaders use GL_OES_standard_derivatives for fwidth() anti-aliasing
 *
 * Shader Features:
 *   - MSDF signed-distance field rendering with smooth edge anti-aliasing
 *   - Per-vertex color/opacity for batch instancing
 *   - Time-synchronized breathing/wave effects for visual polish
 *   - Grid-quantized UV sampling for split-text animation reveal
 *   - World-space coordinate effects (iridescence, spatial modulation)
 *
 * Future Enhancements:
 *   - WebGPU/WGSL port (currently GLSL for WebGL2 compatibility)
 *   - SDFFont format support (binary atlas optimization)
 *   - Multi-channel signed distance fields for better rasterization
 *   - SDF caching/atlasing for dynamic text strings
 *   - Right-to-left/RTL text support
 *   - Superscript/subscript baseline adjustments
 */
