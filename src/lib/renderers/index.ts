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
