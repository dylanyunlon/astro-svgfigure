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

export { renderCellGraph } from './pixi-cell-renderer';
export type { CellDescriptor, EdgeDescriptor } from './pixi-cell-renderer';

export { renderCellGraphSDF } from './sdf-cell-renderer';

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
