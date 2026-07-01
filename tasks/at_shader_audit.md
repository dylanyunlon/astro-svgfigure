# AT Shader Audit — compiled.vs 可用 shader 清单

**任务**: M1317n  
**分支**: cell-pubsub-loop  
**来源**: `upstream/activetheory-assets/compiled.vs` (352 entries, 90 class shaders)

---

## 1. 所有 Class Shader 名称（含 uniform 列表）

Class shader = 含 `#!ATTRIBUTES` 块，可被 `getATProgram()` 直接编译为 vertex+fragment 程序。

### 1.1 Cell / Organic / Blob / Sphere 相关

| Shader | uniforms | 用途 |
|--------|----------|------|
| **JellyShader** | `tMap`, `tMatcap`, `tVideo`, `tRefraction`, `uReflection`, `uScroll`, `uDirection`, `uMouse` | **最相关**。顶点做 cnoise 脉动变形 + 正弦摆动；片元做 matcap + Fresnel + rainbowColor 彩虹色渐变。完全适合 cell 有机体外观。内部 `#require(fbr.vs)`, `#require(fbr.fs)`, `#require(simplenoise.glsl)`, `#require(fresnel.glsl)` |
| **PhysicalShader** | `tVideo`, `tBaseColor`, `tMRO`, `tNormal`, `tLUT`, `tFluid`, `tFluidMask`, `tEnvDiffuse`, `tEnvSpecular`, `tLightmap`, `uUseLightmap`, `uLightmapIntensity`, `uTime`, `uParams`, `uFogColor`, `uTint`, `uTiling`, `uOffset`, `uMRON`, `uEnv`, `uHDR`, `uLightmapAsDiffuse`, `uHold`, `uVisible`, `uRotation`, `uScroll`, `uUVScale` | **完整 PBR 管线**。全套 GGX+IBL+lightmap，可替换手写 `pbr-gpu-pass.ts` 的片元着色器。fragment 做 fresnelSphericalGaussianRoughness + unreal tonemap |
| **WorkDetailCube** | `tRefraction`, `tPrevFrame`, `tEnv`, `tNormal`, `uFresnelPow`, `uDistortStrength`, `uRefractionRatio`, `uNormalScale`, `uSideReflection`, `uParticleDarken`, `uColor`, `tFluid`, `tFluidMask` | 玻璃质感 cell。matcap + reflection + refraction + Fresnel + simplenoise。适合 ffn/glass 物种 |
| **WorkItemShader** | `tMap`, `tVideo`, `tRefraction`, `tEnv`, `tNormal`, `uFresnelPow`, `uDistortStrength`, `uRefractionRatio`, `uColor`, `uHover`, `uMouse`, `uVideoBlend`, `uScale`, `uPhone` | 玻璃面板 + 交互动画（hover/mouse），适合 embedding 物种 |
| **SpineShader** | `tBaseColor`, `tRefraction`, `uReflection` | 脊柱有机体形状，FBR 系 |
| **ChainShader** | `tBaseColor`, `tRefraction`, `uReflection`, `uScroll` | 链式有机体，FBR + 正弦位移 |
| **TreeFBR** | `tBaseColor`, `tVideo`, `uWobble`, `uScroll` | 树形有机体，FBR 系列，有摇摆动画 |
| **CleanRoomGlass** | `tRefraction`, `tEnv`, `tInner`, `uFresnelPow`, `uDistortStrength`, `uRefractionRatio` | 玻璃折射+Fresnel，适合 ffn 物种 |
| **PBR** | *(无 uniform，全靠 `#require(pbr.vs)` / `#require(pbr.fs)` 的内置)* | PBR 基础壳。`pbr.fs` 内部有 `tBaseColor`, `tMRO`, `tNormal`, `tLUT`, `tEnvDiffuse`, `tEnvSpecular`, `tLightmap` |
| **RoomPBR** | *(同上)* | 场景 PBR，与 PBR 相同结构 |
| **LitMaterial** | `tMap` | 最轻量带光照材质，`#require(lighting.vs)` |
| **OcclusionMaterial** | `bbMin`, `bbMax` | AABB 遮挡剔除材质 |
| **GlassInner** | *(无)* | 玻璃内面 |
| **GlassReflection** | *(无)* | 玻璃反射 pass |

### 1.2 Bloom / Glow / Postprocess 相关

| Shader | uniforms | 用途 |
|--------|----------|------|
| **UnrealBloomLuminosity** | `tDiffuse`, `defaultColor`, `defaultOpacity`, `luminosityThreshold`, `smoothWidth` | 亮度阈值提取（UE Bloom 第一步）。与手写 `LUMINOSITY_FRAG` 等价，但使用 AT 的 `luma.fs` 实现 |
| **BloomLuminosityPass** | `tDiffuse`, `defaultColor`, `defaultOpacity`, `luminosityThreshold`, `smoothWidth` | 同上（另一版本，uniformname 完全一致） |
| **UnrealBloomGaussian** | `colorTexture`, `texSize`, `direction` | 可分离高斯模糊。内含 `gaussianPdf` 函数，SIGMA/KERNEL_RADIUS define 控制。可直接替换手写 `BLUR_FRAG` |
| **UnrealBloomComposite** | `blurTexture1`, `bloomStrength`, `bloomRadius`, `bloomTintColor` | Bloom 合成（lerpBloomFactor 权重混合）。可替换手写 `COMPOSITE_FRAG` |
| **HydraBloomPass** | `tDiffuse` | Hydra 风格 bloom pass |
| **DownSample** | `tMap`, `uResolution`, `uRadius` | 13-tap 加权下采样（棋盘格权重）。与 `at-unreal-bloom-pipeline.ts` 中的 downsample 相同算法 |
| **UpSample** | `tMap`, `tNext`, `uResolution`, `uRadius`, `uIntensity`, `uTint` | 9-tap 帐篷滤波上采样 + tint 累积 |
| **LensFlarePrefilter** | `tMap`, `uThreshold`, `uRotate` | 镜头光晕预过滤（亮度提取+旋转） |
| **LensFlareDown** | `tMap`, `uResolution`, `uStretch` | 光晕下采样（streak 拉伸方向） |
| **LensFlareUp** | `tHigh`, `tScene`, `uStretch`, `uSoftenEdge`, `uResolution` | 光晕上采样 + 合成回场景 |
| **CompositeStreak** | `tHigh`, `tDown`, `tPrefiltered`, `uStreakColor`, `uStreakIntensity`, `uGlowIntensity`, `uDebugHalo`, `uFlareIntensity`, `uAspectCorrection`, `uHaloChroma`, `uHaloScale`, `uRotateStreak`, `uHaloSoftness`, `uHaloRotateSrc`, `uHaloConstant`, `uHaloColor`, `uHaloRing` | **光晕/glow 合成**。包含 halo ring SDF、streak 方向、色差。适合 cell glow 效果 |
| **FXAA** | `tMask` | FXAA 抗锯齿（基于 resolution 内置 varying） |

### 1.3 Composite / 最终合成

| Shader | uniforms | 用途 |
|--------|----------|------|
| **GlobalComposite** *(standalone .fs)* | `tDiffuse`, `uRGBStrength`, `uVolumetricStrength`, `uContrast`, `uScroll`, `uContact`, `uScrollDelta`, `uMouse`, `uFrostCorner`, `tFluid`, `tFluidMask`, `tNormal`, `uNormalScale`, `uVisible`, `uChatOpen`, `tLightStreak`, `uGradient`, `uMobile`, `uUIColor`, `uUIBlend`, `uSyncTouch` | AT 全局合成（最完整）。含 RGB shift + contrast + volumetric + fluid distort + light streak |
| **HomeComposite** *(standalone .fs)* | `tDiffuse`, `uRGBStrength`, `uVolumetricStrength`, `uContrast`, `tVolumetricBlur` | AT home 场景合成。含 RGB shift + contrast + volumetric |
| **CleanRoomComposite** *(standalone .fs)* | `tDiffuse`, `uRGBStrength`, `uVolumetricStrength`, `uContrast`, `tVolumetricBlur` | CleanRoom 场景合成，含 UnrealBloom + RGB shift + contrast |
| **WorkComposite** *(standalone .fs)* | `tDiffuse`, `tDetail`, `uRGBStrength`, `uTransition`, `uContrast` | Work 场景合成，含 UnrealBloom |
| **TreeSceneComposite** *(standalone .fs)* | `tDiffuse`, `uRGBStrength`, `uContrast` | Tree 场景合成，含 UnrealBloom + RGB shift |
| **WorkDetailComposite** *(standalone .fs)* | `tDiffuse`, `uRGBStrength` | 最简合成，只做 RGB shift |
| **AboutComposite** *(standalone .fs)* | *(无)* | 透传（`gl_FragColor = texture2D(tDiffuse, vUv)`) |

### 1.4 粒子系统相关

| Shader | uniforms | attributes | 用途 |
|--------|----------|-----------|------|
| **FloatingParticles** | `tPos`, `tPointColor`, `DPR` | `random` | 基于位置纹理的点粒子，含 simplenoise |
| **FlowerParticleShader** | `tPos`, `tLightTexture`, `tPointColor`, `tMap`, `uLightPos`, `uTint`, `DPR`, `uScroll`, `uSizeBias`, `uAnimate`, `uRotate`, `uSparkle` | `random` | 花朵粒子，带光照纹理 |
| **LogoParticleShader** | `tPos`, `tLightTexture`, `tPointColor`, `tMap`, `tVideo`, `uLightPos`, `uTint`, `DPR`, `uScroll` | `random` | Logo 粒子 |
| **ParticleTestShader** | `tPos`, `tLightTexture`, `tPointColor`, `tMap`, `tVideo`, `uLightPos`, `uTint`, `uLogoPos`, `DPR`, `uScroll`, `uVisible`, `uPulse`, `uSizeBias` | `random` | 测试粒子（含 pulse 动画） |
| **WaterParticles** | `tPos`, `tPointColor`, `tMap`, `DPR` | `random` | 水粒子 |
| **TreeParticleShader** | `tPos`, `tLightTexture`, `tPointColor`, `tMap`, `uLightPos`, `uTint`, `DPR`, `uScroll` | `random` | 树粒子 |
| **WorkDetailParticleShader** | `tPos`, `tMap`, `tVideo`, `DPR`, `uSize`, `uSizeBias` | - | Work 细节粒子 |
| **ProtonTube** | `tPos`, `tLife`, `radialSegments`, `thickness`, `taper` | `angle`, `tuv`, `cIndex`, `cNumber` | 管状粒子连接线 |
| **SplineParticleInstance** | `tPos` | `lookup` | Spline 粒子实例 |

### 1.5 光照 / 体积光 / 阴影

| Shader | uniforms | 用途 |
|--------|----------|------|
| **Lighting** | `tLTC1`, `tLTC2` | LTC 面积光照（Linearly Transformed Cosines），高质量 |
| **LightVolume** | `tMap`, `tMask`, `uScale`, `uSeparation`, `uAlpha`, `uMaskScale`, `uRotateSpeed`, `uRotateTexture`, `uNoiseScale`, `uNoiseSpeed`, `uNoiseRange`, `uOffset`, `uScrollX`, `uScrollY`, `uHueShift`, `uColor` | 体积光粒子群，instanced，hue shift 支持 |
| **ShadowDepth** | *(无)* | 深度 pass（阴影贴图生成） |
| **ShadowInspector** | `tMap` | 阴影 debug 可视化 |
| **OcclusionMaterial** | `bbMin`, `bbMax` | AABB 遮挡 |

### 1.6 UI / 文本 / 工具类

| Shader | uniforms | 用途 |
|--------|----------|------|
| **GLUIBatch** | `tMap`, `uColor`, `uAlpha` | GLUI batch renderer |
| **GLUIBatchText** | `tMap`, `uColor`, `uAlpha` | GLUI batch text |
| **GLUIColor** | `uColor`, `uAlpha` | GLUI 纯色 |
| **GLUIObject** | `tMap`, `uAlpha` | GLUI object |
| **DefaultText** | `tMap`, `uColor`, `uAlpha`, `uMouse` | 文本 shader |
| **Text3D** | `tMap`, `uColor`, `uAlpha`, `uOpacity`, `uTranslate`, `uRotate`, `uTransition`, `uWordCount`, `uLineCount`, `uLetterCount`, `uByWord`, `uByLine`, `uPadding`, `uBoundingMin`, `uBoundingMax`, `uScrollDelta`, `uMouse`, `tFluid`, `tFluidMask` | 3D 文本动画 |
| **ColorMaterial** | `color`, `alpha` | 纯色材质 |
| **TextureMaterial** | `tMap` | 纯贴图材质 |
| **ScreenQuad** | `tMap` | 全屏 blit |
| **Blit** | `tMap` | 简单 blit |
| **DownSample** / **UpSample** | 见上表 | — |
| **FXAA** | `tMask` | — |

### 1.7 其他场景 shader

| Shader | uniforms |
|--------|----------|
| **NavBGShader** | `uColor`, `uAlpha`, `uScroll`, `uScrollDelta`, `uBottom`, `uDisabled`, `uHeight`, `uUIColor`, `uUIBlend` |
| **NavAudioShader** | `uColor`, `uScroll`, `uAmplitude`, `uAlpha`, `uHover` |
| **HomeBGShader** / **HomeLogoShader** / **HomeVideoShader** | 各含 `tMap` + scene 参数 |
| **ChatBGShader** / **LoaderBGShader** (1-4) | 各含 `uColor`, `uAlpha`, `uVisible`, 等 |
| **GazeSelector** | `uTime`, `uColor`, `uAlpha`, `uAlpha2`, `uVisible` |
| **TubeShader** / **TubeOrbShader** | 管状装饰物 |
| **WallShader** / **FloorShader** / **WaterCeilingShader** | 场景几何材质 |
| **TreeWaterShader** | `tWaterNormal`, `tVideo`, `uSpeed`, `uScale`, ...水面 |

---

## 2. Library Chunks（`#require` 引用的非 class 片段）

这些不能直接 `getATProgram()` 编译，但被 ATShaderLoader `#require` 解析时内联到 class shader 中：

| Chunk | 内容 |
|-------|------|
| `fbr.vs` / `fbr.fs` | FBR (Fresnel-Based Rendering) 基础 varying 设置 + PBR-lite 材质计算。`fbr.fs` 含 `tMRO`, `tMatcap`, `tNormal`, `uLight`, `uColor` |
| `pbr.vs` / `pbr.fs` | 完整 PBR 设置（UV/normal/view dir）+ GGX 计算函数 |
| `refl.vs` / `refl.fs` | 反射/折射方向计算（Snell law） |
| `matcap.vs` | `reflectMatcap()` — view-space normal → matcap UV |
| `fresnel.glsl` | `getFresnel(normal, viewDir, power)` — Schlick 近似 |
| `simplenoise.glsl` | `cnoise()` — 3D Perlin noise（用于有机体形变） |
| `sdfs.glsl` | SDF 工具函数（含 AT logo SDF） |
| `blendmodes.glsl` | 完整混合模式库（dodge/burn/overlay/screen 等） |
| `contrast.glsl` | `adjustContrast()` |
| `rgbshift.fs` | `getRGB()` — RGB 色差偏移 |
| `gaussianblur.fs` | `blur9()` — 9-tap 高斯 |
| `luma.fs` | `luma()` — BT.601 亮度 |
| `normalmap.glsl` | normal map 解码 |
| `transformUV.glsl` | UV 变换工具 |
| `eases.glsl` | 缓动函数 |
| `range.glsl` | `map()`/`clamp01()` |
| `curl.glsl` | Curl noise |
| `UnrealBloom.fs` | `getUnrealBloom(uv)` — 从 `tUnrealBloom` 采样（作为 #require 片段被 composite 调用） |
| `VolumetricLight.fs` | 光线步进体积光（20步，`lightPos`, `fExposure`, `fDecay`, `fDensity`, `fWeight`, `fClamp`） |

---

## 3. 推荐替换方案

### 3.1 Cell 渲染（替换手写 PBR + matcap）

**当前问题**: `pbr-gpu-pass.ts` 和 `src/lib/shaders/pbr-cell-surface.frag` / `matcap-fresnel-cell.frag` 是手写的独立 shader，没有利用 AT 的 FBR/PBR 库。

**推荐替换**:

#### Option A — JellyShader（有机体 cell，★★★ 最高优先级）

JellyShader 天然适合 cell 有机体渲染：
- 顶点: `cnoise` 驱动的脉动变形 + 正弦摆动（生物感）
- 片元: matcap + Fresnel + rainbowColor（iridescent 彩虹感）
- uniforms 精简，只需 `tMatcap` + `tRefraction` + `uMouse`

```typescript
// 接入方案 — cell-material-system.ts 或 instanced-cell-renderer.ts
import { getATProgram, initATShaderPipeline } from './at-shader-pipeline-bridge';

// 在 initATShaderPipeline 完成后:
const jellyProg = getATProgram(gl, 'JellyShader');
// jellyProg.program 可直接用于 nanogl / raw WebGL draw call
// 设置 uniforms:
//   tMatcap     → 256×256 matcap 纹理
//   tRefraction → 折射 FBO 纹理
//   uScroll     → 0.0
//   uMouse      → vec2(mouseX, mouseY)
//   uDirection  → 1.0
```

**替换目标文件**: `src/lib/sph/cell-mesh-renderer.ts`, `src/lib/sph/instanced-cell-renderer.ts`

#### Option B — PhysicalShader（完整 PBR，★★ 高质量）

替换 `src/lib/sph/pbr-gpu-pass.ts` 的手写 GGX 片元：

```typescript
const physProg = getATProgram(gl, 'PhysicalShader');
// 必需纹理: tBaseColor, tMRO, tNormal, tLUT, tEnvDiffuse, tEnvSpecular
// 可选: tLightmap, tFluid, tFluidMask
// uTint → species color vec3
// uTime → elapsed seconds
// uMRON → vec4(metallic, roughness, occlusion, normalStrength)
```

**替换目标文件**: `src/lib/sph/pbr-gpu-pass.ts`（当前手写 GGX 实现）

#### Option C — WorkDetailCube（玻璃 cell，★★ ffn 物种）

适合需要透明折射效果的 ffn/embedding 物种：

```typescript
const glassProg = getATProgram(gl, 'WorkDetailCube');
// uniforms: tRefraction(折射FBO), tEnv(环境), tNormal(法线图)
// uFresnelPow=2.0, uDistortStrength=0.5, uRefractionRatio=0.9
// uColor → species tint
```

**替换目标文件**: `src/lib/sph/at-glass-material.ts`, `src/lib/sph/cell-material-system.ts`

---

### 3.2 Bloom（替换手写 bloom 金字塔）

**当前问题**: `bloom-gpu-pass.ts` 手写了 `LUMINOSITY_FRAG` + `BLUR_FRAG` + `COMPOSITE_FRAG`，与 AT 的 UnrealBloom shader 几乎等价但未使用 AT 版本。

**推荐**: 用 AT class shader 替换手写 GLSL 字符串。

```typescript
// bloom-gpu-pass.ts — 替换 LUMINOSITY_FRAG
const lumProg = getATProgram(gl, 'UnrealBloomLuminosity');
// uniforms: tDiffuse, luminosityThreshold, smoothWidth, defaultColor, defaultOpacity

// 替换 BLUR_FRAG (gaussian)
const gaussProg = getATProgram(gl, 'UnrealBloomGaussian');
// uniforms: colorTexture, texSize, direction

// 替换 COMPOSITE_FRAG
const compProg = getATProgram(gl, 'UnrealBloomComposite');
// uniforms: blurTexture1, bloomStrength, bloomRadius, bloomTintColor

// 替换 DownSample
const downProg = getATProgram(gl, 'DownSample');
// uniforms: tMap, uResolution, uRadius

// 替换 UpSample
const upProg = getATProgram(gl, 'UpSample');
// uniforms: tMap, tNext, uResolution, uRadius, uIntensity, uTint
```

**替换目标文件**: `src/lib/sph/bloom-gpu-pass.ts`, `src/lib/sph/at-unreal-bloom-pipeline.ts`

---

### 3.3 Composite（替换手写 final composite）

**当前问题**: `composite-gpu-pass.ts` 的 `COMPOSITE_FRAG` 手写了 vignette + grain + color grading，而 AT 有更完整的 `GlobalComposite.fs`（含 RGB shift + volumetric + fluid distort + streak）。

**推荐**:

```typescript
// composite-gpu-pass.ts 或 at-scene-compositor.ts
// 方案1: 直接用 GlobalComposite.fs 的原始 GLSL 通过 getATShaderSource
import { getATShaderSource } from './at-shader-pipeline-bridge';
const globalCompSrc = getATShaderSource('GlobalComposite.fs');
// 手动 compileShader(gl, AntimatterPass_vs, globalCompSrc)

// 方案2: 用 HomeComposite.fs（最简洁，只含 RGB shift + contrast）
const homeCompSrc = getATShaderSource('HomeComposite.fs');
// uniforms: tDiffuse, uRGBStrength(0.0-1.0), uContrast(vec2), tVolumetricBlur
```

**替换目标文件**: `src/lib/sph/composite-gpu-pass.ts`, `src/lib/sph/at-scene-compositor.ts`

---

### 3.4 Cell Glow / Lens Flare（新增效果）

目前 cell 没有 glow 后处理。AT 提供了完整 lens flare + streak 管线：

```typescript
// 新建 cell-glow-pass.ts，接入 3-pass lens flare:
const prefilterProg = getATProgram(gl, 'LensFlarePrefilter');
// uniforms: tMap(scene), uThreshold(0.6), uRotate(0.0)

const downProg = getATProgram(gl, 'LensFlareDown');
// uniforms: tMap(prefiltered), uResolution, uStretch(1.0 水平)

const upProg = getATProgram(gl, 'LensFlareUp');
// uniforms: tHigh(down result), tScene(original), uStretch, uSoftenEdge(0.3), uResolution

// 或完整 streak/halo 效果:
const streakProg = getATProgram(gl, 'CompositeStreak');
// uniforms: tHigh, tDown, tPrefiltered,
//           uStreakColor(vec3), uStreakIntensity, uGlowIntensity,
//           uHaloColor(vec3), uHaloScale, uHaloRing(vec4)
```

**接入文件**: `src/lib/sph/lens-flare.ts`（已有文件，补接 AT shader）

---

## 4. 具体接入方案（文件级改动）

### 4.1 `src/lib/sph/cell-material-system.ts`

**改动**: 在 `SpeciesMaterialDef` 中增加 `atShaderName` 字段，运行时从 bridge 拿程序。

```typescript
// 在文件顶部增加:
import { getATProgram } from './at-shader-pipeline-bridge';

// 在 species defs 中：
const SPECIES_AT_SHADERS: Record<CellSpecies, string> = {
  attention:  'JellyShader',     // 有机脉动
  ffn:        'WorkDetailCube',  // 玻璃折射
  layernorm:  'PhysicalShader',  // 完整 PBR
  embedding:  'JellyShader',     // 膜状有机体
  softmax:    'CompositeStreak', // 发光核心
};

// 在渲染时:
export function getSpeciesProgram(gl: WebGL2RenderingContext, species: CellSpecies) {
  return getATProgram(gl, SPECIES_AT_SHADERS[species]);
}
```

### 4.2 `src/lib/sph/bloom-gpu-pass.ts`

**改动**: 删除 `LUMINOSITY_FRAG`, `BLUR_FRAG`, `COMPOSITE_FRAG` 字符串常量，改用 bridge。

```typescript
// 删除手写 GLSL 常量后，在 BloomGPU.init() 中:
import { getATProgram } from './at-shader-pipeline-bridge';

this._lumProg    = getATProgram(gl, 'UnrealBloomLuminosity')?.program ?? null;
this._gaussProg  = getATProgram(gl, 'UnrealBloomGaussian')?.program ?? null;
this._compProg   = getATProgram(gl, 'UnrealBloomComposite')?.program ?? null;
this._downProg   = getATProgram(gl, 'DownSample')?.program ?? null;
this._upProg     = getATProgram(gl, 'UpSample')?.program ?? null;
// uniform 名称保持原来不变（AT uniform 名与手写版一致）
```

### 4.3 `src/lib/sph/composite-gpu-pass.ts`

**改动**: 替换手写 `COMPOSITE_FRAG` 为 AT HomeComposite / GlobalComposite。

```typescript
import { getATShaderSource } from './at-shader-pipeline-bridge';
import { AntimatterPassVert } from './at-shader-utils'; // 或直接用 AntimatterPass.vs

// 在 CompositeGPU.init() 中:
const fragSrc = getATShaderSource('HomeComposite.fs') ?? COMPOSITE_FRAG_FALLBACK;
// 注意: HomeComposite.fs 用 WebGL1 语法 (varying/texture2D)
// 需要 AntimatterPass.vs 作为顶点 shader
```

### 4.4 `src/lib/sph/instanced-cell-renderer.ts`

**改动**: 当前使用矩形 quad + 自定义片元，改为 JellyShader。

```typescript
import { getATProgram } from './at-shader-pipeline-bridge';

// 在 InstancedCellRenderer.init() 中:
const jelly = getATProgram(gl, 'JellyShader');
if (jelly) {
  this._program = jelly.program;
  // 绑定 tMatcap 纹理 (unit 1)
  // 绑定 tRefraction FBO 纹理 (unit 3)
}
```

---

## 5. 注意事项

1. **`#require` 解析**: standalone `.fs` 文件（GlobalComposite.fs 等）内部含 `#require(...)` 调用，必须通过 `ATShaderLoader.getProgram()` 路径（Path 1）才能正确解析；bridge 的 `_shaderMap`（Path 2）只存 class shader，不包含这些。用 `getATShaderSource('GlobalComposite.fs')` 拿到的是未展开的源码，需要 ATShaderLoader 才能内联依赖。

2. **WebGL1 语法**: AT shader 使用 WebGL1 GLSL（`attribute`/`varying`/`texture2D`/`gl_FragColor`），而项目手写 shader 用 WebGL2（`in`/`out`/`texture`/`fragColor`）。混用时需注意版本兼容。

3. **`time` uniform**: JellyShader 等 FBR 系列 shader 依赖 `time` built-in（Three.js 传入），nanogl 环境需手动设置。

4. **`resolution` uniform**: FXAA 等 shader 依赖 `resolution` built-in，同样需手动传入。

5. **matcap 纹理**: JellyShader 需要 `tMatcap` 纹理，AT 的 matcap 文件在 `upstream/activetheory-assets/textures/` 目录。

6. **测试顺序建议**: bloom pipeline（UnrealBloom 系）最易接入（uniform 名几乎与手写版相同）→ JellyShader（cell 有机体）→ GlobalComposite（final composite，最复杂因含 #require）。

---

## 6. 快速查找表

```
cell / organic / blob → JellyShader, WorkDetailCube, PhysicalShader
bloom luminosity      → UnrealBloomLuminosity, BloomLuminosityPass
bloom gaussian        → UnrealBloomGaussian
bloom composite       → UnrealBloomComposite
bloom downsample      → DownSample
bloom upsample        → UpSample
glow / streak / halo  → CompositeStreak, LensFlarePrefilter/Down/Up
final composite       → GlobalComposite.fs, HomeComposite.fs
antialias             → FXAA
glass / refraction    → CleanRoomGlass, WorkDetailCube, WorkItemShader
particles             → FloatingParticles, FlowerParticleShader, WaterParticles
volumetric light      → VolumetricLight.fs (standalone), LightVolume
area lights           → Lighting (LTC)
shadow                → ShadowDepth
PBR base              → PBR, RoomPBR, PhysicalShader
matcap                → JellyShader, WorkDetailCube (via fbr.fs + matcap.vs)
```
