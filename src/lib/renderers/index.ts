/**
 * renderers/index.ts — Cell rendering methods
 *
 * Method 1: PixiJS Graphics — cell outputs params, PixiJS draws everything.
 *           Bloom glow via BlurFilter, Bezier edges, GPU anti-aliasing.
 *
 * Method 3: SDF shader — each cell is a quad with fragment shader SDF.
 *           sdRoundBox, sdCircle, sin() zigzag. Resolution-independent.
 *           Built-in glow: exp(-d*d*0.008).
 *
 * Both accept the same CellDescriptor[] + EdgeDescriptor[] interface.
 * Cell (sub-Claude) only needs to produce the descriptor JSON.
 * All visual quality is determined by the GPU pipeline, not LLM code generation.
 */

// ── M023: cell-csp-shader — CSP 策略下 shader 编译方案 ───────────────────────
// installCSPShaderPolyfill  — 向 PixiJS 注入 eval-free shader sync polyfill
// installCSPShaderPolyfillSync — 同步语义别名，顶层 await 友好
// isCspBlocked              — 检测当前环境 CSP 是否禁止 unsafe-eval
// CSPShaderRegistry         — 预编译 WebGL program 缓存（无 new Function）
// CellCSPShaderProgram      — cell 渲染 program + uniform location 缓存包装
// createCellCSPShader       — 一步工厂：canvas → registry + cellProgram
// warmupCellShaders         — 预热所有 cell shader，消除首帧编译 stall
// SPECIES_ID_MAP            — species 名称 → uSpeciesId int 映射表
// GLSL 常量（tree-shakeable 静态字符串）:
//   CELL_VERT_GLSL / CELL_FRAG_GLSL       — cell body SDF shader
//   EDGE_PARTICLE_VERT_GLSL / _FRAG_GLSL  — edge 粒子 render shader
//   EDGE_SIM_VERT_GLSL / _FRAG_GLSL       — edge 粒子 Transform Feedback 模拟 shader
//
// Upstream 整合参考:
//   upstream/pixijs-engine/src/unsafe-eval/init.ts             — selfInstall() 注入模式
//   upstream/pixijs-engine/src/unsafe-eval/shader/generateShaderSyncPolyfill.ts
//   upstream/pixijs-engine/src/unsafe-eval/uniforms/generateUniformsSyncPolyfill.ts
//   upstream/pixijs-engine/src/unsafe-eval/ubo/generateUboSyncPolyfill.ts
//   upstream/pixijs-engine/src/rendering/renderers/gl/shader/GenerateShaderSyncCode.ts
export {
  installCSPShaderPolyfill,
  installCSPShaderPolyfillSync,
  isCspBlocked,
  CSPShaderRegistry,
  CellCSPShaderProgram,
  createCellCSPShader,
  warmupCellShaders,
  SPECIES_ID_MAP,
  CELL_VERT_GLSL,
  CELL_FRAG_GLSL,
  EDGE_PARTICLE_VERT_GLSL,
  EDGE_PARTICLE_FRAG_GLSL,
  EDGE_SIM_VERT_GLSL,
  EDGE_SIM_FRAG_GLSL,
} from './cell-csp-shader';
export type { CellCSPUniformSet } from './cell-csp-shader';

// ── M017: cell-culling — viewport frustum skip offscreen ─────────────────────
// CellCuller          — stateful culler class (margin-aware AABB frustum test)
// sharedCellCuller    — shared singleton (mirrors Culler.shared pattern)
// cullCells()         — free function wrapper for sharedCellCuller
// attachCullingToTicker() — register culling as a HIGH-priority ticker callback
// viewportFromStageTransform — derive world-space vp from panned/scaled stage
// viewportFromCamera  — derive 2-D cell-px vp from OGL CameraController params
export {
  CellCuller,
  sharedCellCuller,
  cullCells,
  attachCullingToTicker,
  viewportFromStageTransform,
  viewportFromCamera,
} from './cell-culling';
export type {
// [orphan3]   CellViewport,
// [orphan3]   CellBbox,
// [orphan3]   CullableLiveCell,
} from './cell-culling';

export { renderCellGraph } from './pixi-cell-renderer';
export type { CellDescriptor, EdgeDescriptor } from './pixi-cell-renderer';

// ── M016: PixiJS Ticker 驱动 epoch 动画帧循环 ─────────────────────────────────
// EpochTicker       — 封装 upstream/pixijs-engine Ticker，驱动 epoch 位置推进
// createEpochTicker — 工厂函数（接受 JSON snapshots 数组或 wrapper 对象）
// attachEpochTickerToApp — 挂载到已有 PixiJS Application ticker（避免重复 Ticker）
// wireEpochTickerToTimeline — 桥接 EpochTicker → Theatre.js EpochTimeline
// UPDATE_PRIORITY   — 上游 const 重导出（INTERACTION/HIGH/NORMAL/LOW/UTILITY）
// Ticker            — 上游 Ticker 类重导出（供调用方直接引用）
export {
  EpochTicker,
  createEpochTicker,
  attachEpochTickerToApp,
  wireEpochTickerToTimeline,
  normaliseEpochCell,
  Ticker,
  UPDATE_PRIORITY,
} from './epoch-ticker';
export type {
// [orphan3]   EpochCellState,
// [orphan3]   EpochSnapshot,
// [orphan3]   EpochTickFrame,
// [orphan3]   EpochTickCallback,
// [orphan3]   EpochLoopMode,
// [orphan3]   EpochTickerOptions,
// [orphan3]   SequenceRef,
// [orphan3]   TickerCallback,
} from './epoch-ticker';

export { renderCellGraphSDF } from './sdf-cell-renderer';

// ── M010: species-keyed geometry batcher — upstream/pixijs-engine batcher fusion ──
// Mirrors: Batcher.add/break/begin, BatcherPipe.buildStart/addToBatch/buildEnd,
//          GlBatchAdaptor.execute, DefaultBatcher.packQuadAttributes, BatchableSprite
export {
  CellSpeciesBatch,
  CellBatchManager,
  createCellBatchManager,
  formatCellBatchStats,
  SPECIES_PALETTE,
} from './cell-batch-renderer';
export type { BatchCell } from './cell-batch-renderer';

// ── M007: Cell blur module — upstream pixijs-engine BlurFilter adapted for bloom pipeline ──
export {
  BlurFilter,
  BlurFilterPass,
  CellBlurFilter,
  getCellBlurPreset,
  createCellBlurFilter,
  applyBloomPreBlur,
  acquireBlurPass,
  releaseBlurPass,
  buildBloomFilterChain,
} from './pixi-blur-cell';
export type {
// [orphan3]   BlurFilterOptions,
// [orphan3]   BlurFilterPassOptions,
// [orphan3]   CellBlurPreset,
// [orphan3]   CellBlurFilterOptions,
// [orphan3]   BloomPreBlurOptions,
} from './pixi-blur-cell';

export {
// [orphan3]   AntimatterAttribute,
// [orphan3]   AntimatterFBO,
// [orphan3]   AntimatterPass,
// [orphan3]   AntimatterSpawn,
// [orphan3]   AntimatterCellCompute,
// [orphan3]   CELL_PHYSICS_VERT,
// [orphan3]   AttributeKind,
} from './antimatter-compute';
export type {
// [orphan3]   AntimatterPassOptions,
// [orphan3]   CellSpawnDescriptor,
// [orphan3]   ForceFieldEntry,
} from './antimatter-compute';

export {
// [orphan3]   TweenManager,
// [orphan3]   VelocityTracker,
// [orphan3]   SplineInterpolation,
// [orphan3]   Easing,
} from '../tween-system';
export type {
// [orphan3]   EasingFn,
// [orphan3]   TweenHandle,
// [orphan3]   MathTweenHandle,
// [orphan3]   FrameTweenHandle,
// [orphan3]   Velocity,
// [orphan3]   Vec2,
} from '../tween-system';

export { HierarchyAnimation } from '../hierarchy-animation';
export type {
  CellPos,
  CellState,
  PositionSetter,
  HierarchyAnimationOptions,
} from '../hierarchy-animation';

export { MatrixWasm } from '../matrix-wasm';
export type {
  AABB,
  Viewport,
  AABBVisibilityResult,
} from '../matrix-wasm';

export {
// [orphan3]   GLUIObject,
// [orphan3]   GLUIElement,
// [orphan3]   GLUIText,
// [orphan3]   GLUIBatch,
// [orphan3]   GLUIStage,
// [orphan3]   createGLUIButton,
// [orphan3]   createGLUISlider,
} from './glui-system';
export type {
// [orphan3]   GLUIPoint,
// [orphan3]   GLUISize,
// [orphan3]   GLUIColor,
// [orphan3]   GLUIEventType,
// [orphan3]   GLUIPointerHandler,
// [orphan3]   GLUITextOptions,
// [orphan3]   BatchEntry,
// [orphan3]   BatchRectEntry,
// [orphan3]   BatchCircleEntry,
// [orphan3]   GLUIButtonOptions,
// [orphan3]   GLUISliderOptions,
// [orphan3]   GLUISliderHandle,
} from './glui-system';

// L3: InteractAI + GLSEO (AT: 36 + 9 refs)
export { InteractAI, GLSEO } from './interact-ai'
export type { ChatMessage, InteractAIOptions, CellSEOData } from './interact-ai'

// M012: CellEventSystem — PixiJS EventSystem adapter for cell pointer events
// Fuses upstream/pixijs-engine/src/events (EventSystem, EventBoundary,
// FederatedPointerEvent) into hover/click/drag interaction on cell Containers.
export { CellEventSystem, attachCellEvents, makeCellMeta } from './cell-event-system'
export type {
  CellMeta,
  CellEventSystemOptions,
  CellPointerEventDetail,
  CellSelectEventDetail,
  CellDragEventDetail,
} from './cell-event-system'

// M018: cell-a11y — ARIA labels + keyboard Tab navigation for cell Containers
// Fuses upstream/pixijs-engine/src/accessibility (AccessibilitySystem,
// accessibilityTarget) into the cell pipeline:
//   applyCellA11y()    — stamp accessible/accessibleHint/tabIndex on one container
//   attachCellA11y()   — post-render bulk helper
//   CellA11yManager    — lifecycle manager (register/unregister/update/destroy)
//   buildAriaLabel()   — species + cell_id → "Attention — self_attn_q"
//   buildAriaTitle()   — species → "Attention"
//   applyA11yToContainer() — one-shot helper from CellDescriptor
export {
  CellA11yManager,
  attachCellA11y,
  applyCellA11y,
  applyA11yToContainer,
  buildAriaLabel,
  buildAriaTitle,
} from './cell-a11y'
export type { CellA11yOptions } from './cell-a11y'

// L4: XR system (AT: xr-system)
export {
  XRDeviceManager,
  VRInput,
  VRAbstractHand,
  VRHandFingerTip,
  VRControllerBeam,
  UserInputVR,
  WEBVRPolyfill,
} from '../xr-system'
export type {
// [orphan3]   XRSessionConfig,
// [orphan3]   XRFrameCallback,
// [orphan3]   VRInputState,
// [orphan3]   HandJointPose,
// [orphan3]   FingerTipState,
// [orphan3]   BeamHit,
// [orphan3]   BeamOptions,
// [orphan3]   VRActionMap,
} from '../xr-system'

// L4: Multiplayer (AT: multiplayer)
export {
  SocketConnection,
  GameCenterPlayer,
  GameCenterRoom,
  GameCenter,
  Multiplayer,
} from '../multiplayer'
export type {
// [orphan3]   PlayerId,
// [orphan3]   RoomId,
// [orphan3]   MessageId,
// [orphan3]   PlayerMeta,
// [orphan3]   RoomMeta,
// [orphan3]   NetworkMessage,
// [orphan3]   MultiplayerOptions,
// [orphan3]   MultiplayerConfig,
// [orphan3]   GameCenterConfig,
// [orphan3]   RpcResult,
} from '../multiplayer'

// L4: Audio system (AT: audio-system)
export {
  SFXController,
  ResonanceAudioScene,
  SpeechInputManager,
  createAudioSystem,
} from '../audio-system'
export type {
// [orphan3]   SFXOptions,
// [orphan3]   SFXHandle,
// [orphan3]   ResonanceSourceOptions,
// [orphan3]   ResonanceSourceHandle,
// [orphan3]   ResonanceSceneOptions,
// [orphan3]   ResonanceRoomDimensions,
// [orphan3]   ResonanceMaterials,
// [orphan3]   SpeechGrammar,
// [orphan3]   SpeechResult,
// [orphan3]   SpeechInputOptions,
// [orphan3]   AudioSystemBundle,
} from '../audio-system'

// ── M011: pixijs-engine rendering/render-target fusion ───────────────────────
// PixiRenderTarget    offscreen FBO wrapper (mirrors upstream RenderTarget + GlRenderTarget)
// PixiRenderTargetPool acquire/release pool for bloom ping-pong (mirrors TexturePool + RTPool)
// BloomFBOPass        WebGL2 multi-pass bloom: extract → Kawase blur → composite
// BloomFBOPipeline    unified FBO + pixi-filters-registry AdvancedBloomFilter bridge
export {
  PixiRenderTarget,
  PixiRenderTargetPool,
  BloomFBOPass,
  BloomFBOPipeline,
  createOffscreenTarget,
  buildBloomPipeline,
} from './pixi-render-target';
export type {
// [orphan3]   PixiRenderTargetOptions,
// [orphan3]   BloomFBOPassOptions,
// [orphan3]   BloomFBOPipelineOptions,
} from './pixi-render-target';

// ── AT: full module coverage ──────────────────────────────────────────────────

// platform
export {
  Platform,
  ScreenLock,
  CookieNotice,
  Privacy,
  MetalDetector,
  NBArchitektStdFont,
} from '../platform';
export type {
// [orphan3]   PlatformInfo,
// [orphan3]   ScreenLockOrientation,
// [orphan3]   CookieNoticeOptions,
// [orphan3]   PrivacySettings,
// [orphan3]   MetalCapabilities,
// [orphan3]   FontLoadOptions,
} from '../platform';

// physics-animation
export {
  PhysicalSync,
  SkinAnimation,
  Mirror,
  PlayerModel,
  Bounce,
} from '../physics-animation';
export type {
// [orphan3]   BoneTransform,
// [orphan3]   PhysicsBody,
// [orphan3]   PhysicalSyncOptions,
// [orphan3]   SkinClip,
// [orphan3]   SkinAnimationOptions,
// [orphan3]   MirrorAxis,
// [orphan3]   MirrorOptions,
// [orphan3]   PlayerModelOptions,
// [orphan3]   PlayerState,
// [orphan3]   BounceOptions,
} from '../physics-animation';

// fx-extensions
export {
  FXLayer,
  FXStencil,
  FXAssetsController,
  FXScrollTransition,
  FXSceneVisibility,
  FXSceneCompositor,
  FragUIHelper,
  FXDhCwa,
} from '../fx-extensions';
export type {
// [orphan3]   FXLayerOptions,
// [orphan3]   StencilRegion,
// [orphan3]   FXAssetEntry,
// [orphan3]   FXAssetLoaded,
// [orphan3]   FXAssetProgress,
// [orphan3]   AssetKind,
// [orphan3]   ScrollTransitionKind,
// [orphan3]   FXScrollTransitionOptions,
// [orphan3]   FXScene,
// [orphan3]   FXSceneCompositorOptions,
// [orphan3]   FragUIHelperOptions,
// [orphan3]   FXDhCwaOptions,
} from '../fx-extensions';

// page-components
export {
  WorkItems,
  WorkDetail,
  WorkDetailContent,
  TubesInteraction,
  MoveNode,
  Contact,
  Footer,
  Playground,
  Theory,
  Player,
} from '../page-components';
export type {
// [orphan3]   WorkItem,
// [orphan3]   WorkItemFilter,
// [orphan3]   WorkItemsOptions,
// [orphan3]   WorkDetailData,
// [orphan3]   WorkDetailOptions,
// [orphan3]   WorkDetailContentOptions,
// [orphan3]   TubeNode,
// [orphan3]   TubeEdge,
// [orphan3]   TubesInteractionOptions,
// [orphan3]   MoveNodeOptions,
// [orphan3]   ContactFormData,
// [orphan3]   ContactOptions,
// [orphan3]   FooterLink,
// [orphan3]   FooterOptions,
// [orphan3]   PlaygroundModule,
// [orphan3]   TheorySection,
// [orphan3]   TheoryOptions,
// [orphan3]   PlayerTrack,
// [orphan3]   PlayerOptions,
} from '../page-components';

// threed-pipeline
export {
  GaussianSplats,
  DracoThread,
  GeomThread,
  GLTFLoader,
} from '../threed-pipeline';
export type {
// [orphan3]   SplatPoint,
// [orphan3]   GaussianSplatsOptions,
// [orphan3]   GaussianSplatsLoadResult,
// [orphan3]   GeometryData,
// [orphan3]   GLTFNode,
// [orphan3]   GLTFScene,
// [orphan3]   GLTFAnimation,
// [orphan3]   GLTFAnimationChannel,
// [orphan3]   DracoDecodeResult,
// [orphan3]   DracoThreadOptions,
// [orphan3]   GeomTask,
// [orphan3]   GeomTaskResult,
// [orphan3]   GLTFLoaderOptions,
} from '../threed-pipeline';

// engine-utils
export {
  ListNode,
  LinkedList,
  SnapshotFrame,
  OptimizationProfiler,
  CleanRoom,
  Quaternion,
  Interpolation,
} from '../engine-utils';
export type {
// [orphan3]   FrameSnapshot,
// [orphan3]   SnapshotFrameOptions,
// [orphan3]   ProfileSample,
// [orphan3]   ProfileReport,
// [orphan3]   Quat4,
// [orphan3]   EasingName,
} from '../engine-utils';

// asset-pipeline
export {
  AssetLoader,
  AssetList,
  CMSData,
  Config,
} from '../asset-pipeline';
export type {
// [orphan3]   AssetType,
// [orphan3]   AssetDescriptor,
// [orphan3]   AssetResult,
// [orphan3]   AssetLoaderEvent,
// [orphan3]   AssetGroup,
// [orphan3]   CMSEntry,
// [orphan3]   CMSCollection,
// [orphan3]   CMSDataOptions,
// [orphan3]   ConfigValue,
// [orphan3]   ConfigSchema,
} from '../asset-pipeline';

// interaction
export {
  DragAndDrop,
  ScrollController,
  Keyboard,
  ContextMenu,
  UserInput,
} from '../interaction';
export type {
// [orphan3]   DragItem,
// [orphan3]   DropTarget,
// [orphan3]   DragAndDropOptions,
// [orphan3]   ScrollControllerOptions,
// [orphan3]   KeyCombo,
// [orphan3]   KeyBinding,
// [orphan3]   ContextMenuItem,
// [orphan3]   ContextMenuOptions,
// [orphan3]   PointerState,
// [orphan3]   UserInputOptions,
} from '../interaction';

// renderers/hydra-css
export {
  HydraObject,
  HydraCSS,
  FXController,
  FXScroll,
} from './hydra-css';
export type {
// [orphan3]   CSSTransform,
// [orphan3]   HydraProp,
// [orphan3]   HydraObjectOptions,
// [orphan3]   HydraCSSOptions,
// [orphan3]   FXControllerOptions,
// [orphan3]   FXPhase,
// [orphan3]   FXControllerEvent,
// [orphan3]   FXScrollBinding,
// [orphan3]   FXScrollOptions,
} from './hydra-css';

// renderers/rendering-utils
export {
  ScreenProjection,
  ScreenQuad,
  Frustum,
  Lighting,
  LightVolume,
  RTPool,
} from './rendering-utils';
export type {
// [orphan3]   ScreenProjectionOptions,
// [orphan3]   FrustumPlane,
// [orphan3]   AABB,
// [orphan3]   Sphere,
// [orphan3]   LightKind,
// [orphan3]   Light,
// [orphan3]   LightingUniforms,
// [orphan3]   LightVolumeOptions,
// [orphan3]   RTDescriptor,
// [orphan3]   RenderTarget,
} from './rendering-utils';

// ── M019: cell-env-detect — WebGL2/WebGPU capability probe + Canvas2D fallback ─
// detectCellEnv()         — async probe → CellEnvReport (cached after first call)
// buildRendererPreference — CellRenderBackend → PixiJS autoDetectRenderer preference[]
// createCellCanvas2D      — one-shot Canvas2D cell graph render (no WebGL)
// CellCanvas2DRenderer    — stateful class-based Canvas2D fallback renderer
// CellEnvAdapter          — PixiJS Adapter shim for the detected environment
// Probe priority:  WebGPU → WebGL2 → WebGL1 → Canvas2D
// Upstream fusion: environment-browser/BrowserAdapter, utils/browser/isWebGLSupported,
//                  utils/browser/isWebGPUSupported, renderers/autoDetectRenderer,
//                  renderers/canvas/CanvasContextSystem
export {
  detectCellEnv,
  buildRendererPreference,
  createCellCanvas2D,
  CellCanvas2DRenderer,
  CellEnvAdapter,
  _resetCellEnvCache,
} from './cell-env-detect';
export type {
// [orphan3]   CellRenderBackend,
// [orphan3]   CellGpuFeatures,
// [orphan3]   CellEnvReport,
// [orphan3]   C2DCellDescriptor,
// [orphan3]   C2DEdgeDescriptor,
} from './cell-env-detect';

// ── M020: cell-compressed-tex — ASTC/ETC2/BC 压缩纹理 for 大 topology 图 ────────
// CompressedTexProbe     — WebGL 扩展探测，缓存 ASTC/ETC2/BC 能力
// CompressedTexManager   — 生命周期：alloc atlas / loadKTX / loadDDS / destroy
// TopologyTexAtlas       — bin-packer: cell 列表 → atlas layout (AtlasTile[])
// parseKTXBuffer()       — KTX ArrayBuffer → TextureSourceOptions (ASTC/ETC2)
// parseDDSBuffer()       — DDS ArrayBuffer → TextureSourceOptions (BC7/BC3/BC1)
// createTopologyTex()    — RGBA8 pixels → CompressedSource (software encoder)
// selectBestFormat()     — 根据 CompressedTexCapabilities 选最佳 TEXTURE_FORMATS
// estimateMemorySaving() — 压缩 vs RGBA8 显存节省率计算
// buildCompressedTopology() — 一键工厂：probe + select + alloc，返回 AtlasRecord
// LARGE_TOPOLOGY_THRESHOLD — 启用压缩的最小 cell 数量 (64)
export {
  CompressedTexProbe,
  CompressedTexManager,
  TopologyTexAtlas,
  parseKTXBuffer,
  parseDDSBuffer,
  createTopologyTex,
  selectBestFormat,
  estimateTextureMemory,
  estimateMemorySaving,
  buildCompressedTopology,
  LARGE_TOPOLOGY_THRESHOLD,
  ASTC_BLOCK_4x4,
  COMPRESSED_BLOCK_BYTES,
  COMPRESSED_BLOCK_DIM,
} from './cell-compressed-tex';
export type {
// [orphan3]   CompressedTexCapabilities,
// [orphan3]   AtlasTile,
// [orphan3]   TopologyAtlasLayout,
// [orphan3]   TopologyTexOptions,
// [orphan3]   AtlasRecord,
} from './cell-compressed-tex';

// ── M022: cell-prepare — PixiJS PrepareSystem GPU 首帧预上传 ─────────────────
// prepareCellGPU    — topology 加载后批量上传所有 cell 纹理/Graphics/Text 到 GPU
// prepareStageGPU   — 任意 Container 子树 GPU upload（通用辅助）
// warmCellAssets    — loadCellAssets() 完成后立即预热 SpeciesAssets Texture 列表
// withGPUPrepare    — HOF 包装器：渲染工厂函数 + 自动 prepare，一步到位
//
// 消除首帧 texture upload 延迟：
//   loadCellAssets() → warmCellAssets() → buildCellContainer() × N
//   → prepareCellGPU() → app.ticker.start() → 首帧零 stall
//
// Upstream fusion:
//   upstream/pixijs-engine/src/prepare/PrepareSystem.ts   (PrepareSystem.upload)
//   upstream/pixijs-engine/src/prepare/PrepareBase.ts     (upload → Promise<void>)
//   upstream/pixijs-engine/src/prepare/PrepareQueue.ts    (resolveQueueItem)
//   upstream/pixijs-engine/src/prepare/PrepareUpload.ts   (uploadTextureSource)
export {
  prepareCellGPU,
  prepareStageGPU,
  warmCellAssets,
  withGPUPrepare,
} from './cell-prepare';
export type { PrepareResult, CellPrepareOptions } from './cell-prepare';

// ── M058: Theatre.js epoch timeline ↔ PixiJS Ticker 桥接 ───────────────────────
//
// EpochCellBridge  — 核心桥接类：
//   registerContainer(cellId, container, origSize?) — 注册 PixiJS Container
//   advanceEpoch(rawCells, rate?)  — 触发 Theatre.js sequence.play() N→N+1 过渡
//   play / pause / stop / seek     — 代理到 EpochTimeline 播放控制
//   attachToApp(app)               — 挂载到 PixiJS Application Ticker
//   jumpToEpoch(n)                 — 无动画跳转到指定 epoch
//   destroy()                      — 取消订阅 + 解除 Ticker
//
// createEpochCellBridge(data, opts) — 工厂函数
//
// renderCellGraphWithEpochBridge()  — 一键高层 API：初始化 PixiJS + 桥接，
//                                     返回 { bridge, stop }
//
// buildContainerRegistry()          — 批量注册已有 Container Map
//
// EpochPubSubBridge — pubsub emitter 自动接线：emitter.on('epoch') → advanceEpoch()
//
// Props 插值（每帧从 SheetObject 读取）:
//   x, y         → container.position.set()     bbox lerp
//   w, h         → container.scale              (bboxScale=true 时)
//   opacity      → container.alpha              opacity fade
//   r, g, b      → container.tint               color interpolation
//   bloomStrength → glow.__bloomFilter.bloomScale  bloom 强度
//
// Upstream 参考:
//   src/lib/renderers/theatre-epoch-timeline.ts  — EpochTimeline / CellState
//   src/lib/renderers/pixi-cell-renderer.ts      — buildCellContainer / __bloomFilter
//   upstream/theatre-js/core/src/coreExports.ts  — val(), onChange()
//   upstream/pixijs-engine/src/ticker/Ticker.ts  — Ticker
export {
  EpochCellBridge,
  EpochPubSubBridge,
  createEpochCellBridge,
  renderCellGraphWithEpochBridge,
  buildContainerRegistry,
} from './theatre-epoch-cell-bridge';
export type {
// [orphan3]   EpochCellBridgeOptions,
} from './theatre-epoch-cell-bridge';

// ── M042/M067: Theatre.js epoch timeline (re-export for convenience) ──────────
export {
  createEpochTimeline,
  hexToRgb,
  rgbToHex,
  normaliseCellState,
  projectFromTopology,
} from './theatre-epoch-timeline';
export type {
  CellState as EpochCellStateTheatre,
// [orphan3]   EpochSheet,
// [orphan3]   EpochSnapshotsJSON,
// [orphan3]   RawCellState,
// [orphan3]   EpochFrame,
// [orphan3]   FrameCallback,
// [orphan3]   EasingPreset,
// [orphan3]   EpochTimelineOptions,
} from './theatre-epoch-timeline';

// ── M705: curl-particle-field — AT curl.glsl analytic derivatives → cell decor ─
export {
  CurlParticleField,
  attachCurlParticleField,
} from './curl-particle-field';
export type {
// [orphan3]   CurlParticleFieldConfig,
// [orphan3]   CellDecorDesc,
} from './curl-particle-field';

// ── M746: caustics-background — AT caustic_plane analytic Jacobian caustics ──
export {
  mountCausticsBackground,
} from './caustics-background';
export type {
// [orphan3]   CausticsBackgroundOptions,
// [orphan3]   CausticsBackgroundHandle,
} from './caustics-background';

// ── M765: cell-label-renderer — MSDF text labels for cell bbox + pubsub bridge ──
// CellLabelRenderer       — instanced MSDF label 渲染器 (GLText wrapper)
// createCellLabelRenderer — 一步工厂: init atlas + syncFromCells
// CellLabelPubSubBridge   — CellEventSource → CellLabelRenderer 自动接线
// extractLabelsFromTopology — composite_params.json → 带 label 的 CellParamsJson[]
export {
  CellLabelRenderer,
  createCellLabelRenderer,
  CellLabelPubSubBridge,
  extractLabelsFromTopology,
} from './cell-label-renderer';
export type {
// [orphan3]   CellLabelRendererOptions,
// [orphan3]   CellLabelDrawParams,
} from './cell-label-renderer';
