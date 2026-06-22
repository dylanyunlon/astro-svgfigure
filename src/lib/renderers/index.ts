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
  CellViewport,
  CellBbox,
  CullableLiveCell,
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
  EpochCellState,
  EpochSnapshot,
  EpochTickFrame,
  EpochTickCallback,
  EpochLoopMode,
  EpochTickerOptions,
  SequenceRef,
  TickerCallback,
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
  BlurFilterOptions,
  BlurFilterPassOptions,
  CellBlurPreset,
  CellBlurFilterOptions,
  BloomPreBlurOptions,
} from './pixi-blur-cell';

export {
  AntimatterAttribute,
  AntimatterFBO,
  AntimatterPass,
  AntimatterSpawn,
  AntimatterCellCompute,
  CELL_PHYSICS_VERT,
  AttributeKind,
} from './antimatter-compute';
export type {
  AntimatterPassOptions,
  CellSpawnDescriptor,
  ForceFieldEntry,
} from './antimatter-compute';

export {
  TweenManager,
  VelocityTracker,
  SplineInterpolation,
  Easing,
} from '../tween-system';
export type {
  EasingFn,
  TweenHandle,
  MathTweenHandle,
  FrameTweenHandle,
  Velocity,
  Vec2,
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
  GLUIObject,
  GLUIElement,
  GLUIText,
  GLUIBatch,
  GLUIStage,
  createGLUIButton,
  createGLUISlider,
} from './glui-system';
export type {
  GLUIPoint,
  GLUISize,
  GLUIColor,
  GLUIEventType,
  GLUIPointerHandler,
  GLUITextOptions,
  BatchEntry,
  BatchRectEntry,
  BatchCircleEntry,
  GLUIButtonOptions,
  GLUISliderOptions,
  GLUISliderHandle,
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
  XRSessionConfig,
  XRFrameCallback,
  VRInputState,
  HandJointPose,
  FingerTipState,
  BeamHit,
  BeamOptions,
  VRActionMap,
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
  PlayerId,
  RoomId,
  MessageId,
  PlayerMeta,
  RoomMeta,
  NetworkMessage,
  MultiplayerOptions,
  MultiplayerConfig,
  GameCenterConfig,
  RpcResult,
} from '../multiplayer'

// L4: Audio system (AT: audio-system)
export {
  SFXController,
  ResonanceAudioScene,
  SpeechInputManager,
  createAudioSystem,
} from '../audio-system'
export type {
  SFXOptions,
  SFXHandle,
  ResonanceSourceOptions,
  ResonanceSourceHandle,
  ResonanceSceneOptions,
  ResonanceRoomDimensions,
  ResonanceMaterials,
  SpeechGrammar,
  SpeechResult,
  SpeechInputOptions,
  AudioSystemBundle,
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
  PixiRenderTargetOptions,
  BloomFBOPassOptions,
  BloomFBOPipelineOptions,
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
  PlatformInfo,
  ScreenLockOrientation,
  CookieNoticeOptions,
  PrivacySettings,
  MetalCapabilities,
  FontLoadOptions,
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
  BoneTransform,
  PhysicsBody,
  PhysicalSyncOptions,
  SkinClip,
  SkinAnimationOptions,
  MirrorAxis,
  MirrorOptions,
  PlayerModelOptions,
  PlayerState,
  BounceOptions,
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
  FXLayerOptions,
  StencilRegion,
  FXAssetEntry,
  FXAssetLoaded,
  FXAssetProgress,
  AssetKind,
  ScrollTransitionKind,
  FXScrollTransitionOptions,
  FXScene,
  FXSceneCompositorOptions,
  FragUIHelperOptions,
  FXDhCwaOptions,
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
  WorkItem,
  WorkItemFilter,
  WorkItemsOptions,
  WorkDetailData,
  WorkDetailOptions,
  WorkDetailContentOptions,
  TubeNode,
  TubeEdge,
  TubesInteractionOptions,
  MoveNodeOptions,
  ContactFormData,
  ContactOptions,
  FooterLink,
  FooterOptions,
  PlaygroundModule,
  TheorySection,
  TheoryOptions,
  PlayerTrack,
  PlayerOptions,
} from '../page-components';

// threed-pipeline
export {
  GaussianSplats,
  DracoThread,
  GeomThread,
  GLTFLoader,
} from '../threed-pipeline';
export type {
  SplatPoint,
  GaussianSplatsOptions,
  GaussianSplatsLoadResult,
  GeometryData,
  GLTFNode,
  GLTFScene,
  GLTFAnimation,
  GLTFAnimationChannel,
  DracoDecodeResult,
  DracoThreadOptions,
  GeomTask,
  GeomTaskResult,
  GLTFLoaderOptions,
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
  FrameSnapshot,
  SnapshotFrameOptions,
  ProfileSample,
  ProfileReport,
  Quat4,
  EasingName,
} from '../engine-utils';

// asset-pipeline
export {
  AssetLoader,
  AssetList,
  CMSData,
  Config,
} from '../asset-pipeline';
export type {
  AssetType,
  AssetDescriptor,
  AssetResult,
  AssetLoaderEvent,
  AssetGroup,
  CMSEntry,
  CMSCollection,
  CMSDataOptions,
  ConfigValue,
  ConfigSchema,
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
  DragItem,
  DropTarget,
  DragAndDropOptions,
  ScrollControllerOptions,
  KeyCombo,
  KeyBinding,
  ContextMenuItem,
  ContextMenuOptions,
  PointerState,
  UserInputOptions,
} from '../interaction';

// renderers/hydra-css
export {
  HydraObject,
  HydraCSS,
  FXController,
  FXScroll,
} from './hydra-css';
export type {
  CSSTransform,
  HydraProp,
  HydraObjectOptions,
  HydraCSSOptions,
  FXControllerOptions,
  FXPhase,
  FXControllerEvent,
  FXScrollBinding,
  FXScrollOptions,
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
  ScreenProjectionOptions,
  FrustumPlane,
  AABB,
  Sphere,
  LightKind,
  Light,
  LightingUniforms,
  LightVolumeOptions,
  RTDescriptor,
  RenderTarget,
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
  CellRenderBackend,
  CellGpuFeatures,
  CellEnvReport,
  C2DCellDescriptor,
  C2DEdgeDescriptor,
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
  CompressedTexCapabilities,
  AtlasTile,
  TopologyAtlasLayout,
  TopologyTexOptions,
  AtlasRecord,
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
  EpochCellBridgeOptions,
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
  EpochSheet,
  EpochSnapshotsJSON,
  RawCellState,
  EpochFrame,
  FrameCallback,
  EasingPreset,
  EpochTimelineOptions,
} from './theatre-epoch-timeline';

// ── M705: curl-particle-field — AT curl.glsl analytic derivatives → cell decor ─
export {
  CurlParticleField,
  attachCurlParticleField,
} from './curl-particle-field';
export type {
  CurlParticleFieldConfig,
  CellDecorDesc,
} from './curl-particle-field';

// ── M746: caustics-background — AT caustic_plane analytic Jacobian caustics ──
export {
  mountCausticsBackground,
} from './caustics-background';
export type {
  CausticsBackgroundOptions,
  CausticsBackgroundHandle,
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
  CellLabelRendererOptions,
  CellLabelDrawParams,
} from './cell-label-renderer';
