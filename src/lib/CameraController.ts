/**
 * CameraController.ts
 * AT Camera Parameters — 66 parameters from channels/physics/at_uil_categorized.json → "camera"
 * xiaodi #67
 */

export interface CameraPreset {
  position?: [number, number, number];
  lookAt?: [number, number, number];
  groupPos?: [number, number, number];
  moveXY?: [number, number];
  fov?: number;
  far?: number;
  near?: number;
  lerpSpeed?: number;
  lerpSpeed2?: number;
  rotation?: [number, number, number];
  cameraRotation?: [number, number, number];
  viewportFocus?: [number, number];
  deltaRotate?: number;
  wobbleStrength?: number;
}

// ─── All 66 AT Camera Parameters (exact values) ──────────────────────────────

export const AT_CAMERA_PARAMS = {
  // Element_0 / Footer
  CAMERA_Element_0_FootergroupPos:        [0, 0, 0]            as [number,number,number],
  CAMERA_Element_0_FooterlookAt:          [0, 0, 0]            as [number,number,number],
  CAMERA_Element_0_FootermoveXY:          [0, 0]               as [number,number],
  CAMERA_Element_0_Footerposition:        [0, -1.03, 40]       as [number,number,number],

  // Element_10 / CleanRoom
  CAMERA_Element_10_CleanRoomfov:         30,
  CAMERA_Element_10_CleanRoomgroupPos:    [0, 0, 0]            as [number,number,number],
  CAMERA_Element_10_CleanRoomlerpSpeed:   0.08,
  CAMERA_Element_10_CleanRoomlookAt:      [0, 0.8, -0.3]       as [number,number,number],
  CAMERA_Element_10_CleanRoommoveXY:      [0, 0]               as [number,number],
  CAMERA_Element_10_CleanRoomposition:    [0, 1, 5]            as [number,number,number],
  CAMERA_Element_10_CleanRoomrotation:    [0.99, 0, 0]         as [number,number,number],

  // Element_10 / TreeScene
  CAMERA_Element_10_TreeScenegroupPos:    [0, 0, 0]            as [number,number,number],
  CAMERA_Element_10_TreeScenelookAt:      [-2, 4.74, 6]        as [number,number,number],
  CAMERA_Element_10_TreeScenemoveXY:      [0, 0]               as [number,number],
  CAMERA_Element_10_TreeSceneposition:    [0, 5, 32]           as [number,number,number],

  // Element_1 / About
  CAMERA_Element_1_AboutmoveXY:           [0, 0]               as [number,number],
  CAMERA_Element_1_Aboutposition:         [0, 0, 6]            as [number,number,number],

  // Element_1 / ContactUs
  CAMERA_Element_1_ContactUsposition:     [0, 0, 6]            as [number,number,number],

  // Element_1 / Contact
  CAMERA_Element_1_ContactgroupPos:       [0, 0, 0.004]        as [number,number,number],
  CAMERA_Element_1_Contactposition:       [0, 0, 6]            as [number,number,number],

  // Element_1 / Home
  CAMERA_Element_1_Homefov:               30,
  CAMERA_Element_1_HomegroupPos:          [0, 0, 0]            as [number,number,number],
  CAMERA_Element_1_HomelerpSpeed:         0.1,
  CAMERA_Element_1_HomelerpSpeed2:        1,
  CAMERA_Element_1_HomelookAt:            [0, 4.59, 0]         as [number,number,number],
  CAMERA_Element_1_HomemoveXY:            [0, 0]               as [number,number],
  CAMERA_Element_1_Homeposition:          [0, 2, 40]           as [number,number,number],

  // Element_1 / JellyfishDemo
  CAMERA_Element_1_JellyfishDemomoveXY:   [1, 1]               as [number,number],
  CAMERA_Element_1_JellyfishDemoposition: [0, 0, 6]            as [number,number,number],

  // Element_1 / WorkDetailParticles
  CAMERA_Element_1_WorkDetailParticleslookAt:   [0, 0, 5]      as [number,number,number],
  CAMERA_Element_1_WorkDetailParticlesmoveXY:   [-4, 4]        as [number,number],
  CAMERA_Element_1_WorkDetailParticlesposition: [0, 0, 22]     as [number,number,number],

  // Element_1 / WorkDetail
  CAMERA_Element_1_WorkDetailgroupPos:    [0, 0, 9.33]         as [number,number,number],
  CAMERA_Element_1_WorkDetaillerpSpeed:   0.07,
  CAMERA_Element_1_WorkDetaillookAt:      [0, 0, -10]          as [number,number,number],
  CAMERA_Element_1_WorkDetailmoveXY:      [-1, 0.5]            as [number,number],
  CAMERA_Element_1_WorkDetailposition:    [0, 0, 1]            as [number,number,number],
  CAMERA_Element_1_WorkDetailviewportFocus: [0, 0]             as [number,number],

  // Element_1 / homeScene
  CAMERA_Element_1_homeScenefar:          100,
  CAMERA_Element_1_homeScenefov:          20,
  CAMERA_Element_1_homeScenelookAt:       [0, 3, 0]            as [number,number,number],
  CAMERA_Element_1_homeSceneposition:     [0, 3, 15]           as [number,number,number],

  // Element_1 / particleTest
  CAMERA_Element_1_particleTestposition:  [0, 0, 10]           as [number,number,number],

  // Element_2 / Work
  CAMERA_Element_2_WorkcameraRotation:    [0, 0, 0]            as [number,number,number],
  CAMERA_Element_2_Workfov:               35,
  CAMERA_Element_2_WorkgroupPos:          [0, 0, 0]            as [number,number,number],
  CAMERA_Element_2_WorklerpSpeed:         0.07,
  CAMERA_Element_2_WorklerpSpeed2:        1,
  CAMERA_Element_2_WorklookAt:            [0, 0, -4]           as [number,number,number],
  CAMERA_Element_2_WorkmoveXY:            [0, 0]               as [number,number],
  CAMERA_Element_2_Workposition:          [0, 0, 2]            as [number,number,number],
  CAMERA_Element_2_Workrotation:          [0, 196.07, 0]       as [number,number,number],

  // Element_2 / work_page
  CAMERA_Element_2_work_pagemoveXY:       [0.5, 0.5]           as [number,number],
  CAMERA_Element_2_work_pageposition:     [0, 0, 2]            as [number,number,number],

  // Element_3 / home_scene
  CAMERA_Element_3_home_scenedeltaRotate:     3,
  CAMERA_Element_3_home_scenefov:             30,
  CAMERA_Element_3_home_scenegroupPos:        [0, 1.95, 8.020000000000001] as [number,number,number],
  CAMERA_Element_3_home_scenemoveXY:          [0.4, 0.2]       as [number,number],
  CAMERA_Element_3_home_sceneposition:        [0, 0, 8]        as [number,number,number],
  CAMERA_Element_3_home_scenerotation:        [0, 0, 0]        as [number,number,number],
  CAMERA_Element_3_home_scenewobbleStrength:  0.1,

  // Element_4 / ParticleTest
  CAMERA_Element_4_ParticleTestgroupPos:  [0, 0, 0]            as [number,number,number],
  CAMERA_Element_4_ParticleTestlookAt:    [0, 6.2, 2]          as [number,number,number],
  CAMERA_Element_4_ParticleTestmoveXY:    [0, 0]               as [number,number],
  CAMERA_Element_4_ParticleTestposition:  [0, 8, 35]           as [number,number,number],
  CAMERA_Element_4_ParticleTestrotation:  [0, 0, 0]            as [number,number,number],
} as const;

// ─── Scene/Species Presets (derived from AT params) ──────────────────────────

export const CAMERA_PRESETS: Record<string, CameraPreset> = {
  Footer: {
    groupPos:   AT_CAMERA_PARAMS.CAMERA_Element_0_FootergroupPos,
    lookAt:     AT_CAMERA_PARAMS.CAMERA_Element_0_FooterlookAt,
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_0_FootermoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_0_Footerposition,
  },
  CleanRoom: {
    fov:        AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomfov,
    groupPos:   AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomgroupPos,
    lerpSpeed:  AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomlerpSpeed,
    lookAt:     AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomlookAt,
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoommoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomposition,
    rotation:   AT_CAMERA_PARAMS.CAMERA_Element_10_CleanRoomrotation,
  },
  TreeScene: {
    groupPos:   AT_CAMERA_PARAMS.CAMERA_Element_10_TreeScenegroupPos,
    lookAt:     AT_CAMERA_PARAMS.CAMERA_Element_10_TreeScenelookAt,
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_10_TreeScenemoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_10_TreeSceneposition,
  },
  About: {
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_1_AboutmoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_Aboutposition,
  },
  ContactUs: {
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_ContactUsposition,
  },
  Contact: {
    groupPos:   AT_CAMERA_PARAMS.CAMERA_Element_1_ContactgroupPos,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_Contactposition,
  },
  Home: {
    fov:        AT_CAMERA_PARAMS.CAMERA_Element_1_Homefov,
    groupPos:   AT_CAMERA_PARAMS.CAMERA_Element_1_HomegroupPos,
    lerpSpeed:  AT_CAMERA_PARAMS.CAMERA_Element_1_HomelerpSpeed,
    lerpSpeed2: AT_CAMERA_PARAMS.CAMERA_Element_1_HomelerpSpeed2,
    lookAt:     AT_CAMERA_PARAMS.CAMERA_Element_1_HomelookAt,
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_1_HomemoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_Homeposition,
  },
  JellyfishDemo: {
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_1_JellyfishDemomoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_JellyfishDemoposition,
  },
  WorkDetailParticles: {
    lookAt:     AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailParticleslookAt,
    moveXY:     AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailParticlesmoveXY,
    position:   AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailParticlesposition,
  },
  WorkDetail: {
    groupPos:       AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailgroupPos,
    lerpSpeed:      AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetaillerpSpeed,
    lookAt:         AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetaillookAt,
    moveXY:         AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailmoveXY,
    position:       AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailposition,
    viewportFocus:  AT_CAMERA_PARAMS.CAMERA_Element_1_WorkDetailviewportFocus,
  },
  homeScene: {
    far:      AT_CAMERA_PARAMS.CAMERA_Element_1_homeScenefar,
    fov:      AT_CAMERA_PARAMS.CAMERA_Element_1_homeScenefov,
    lookAt:   AT_CAMERA_PARAMS.CAMERA_Element_1_homeScenelookAt,
    position: AT_CAMERA_PARAMS.CAMERA_Element_1_homeSceneposition,
  },
  particleTest: {
    position: AT_CAMERA_PARAMS.CAMERA_Element_1_particleTestposition,
  },
  Work: {
    cameraRotation: AT_CAMERA_PARAMS.CAMERA_Element_2_WorkcameraRotation,
    fov:            AT_CAMERA_PARAMS.CAMERA_Element_2_Workfov,
    groupPos:       AT_CAMERA_PARAMS.CAMERA_Element_2_WorkgroupPos,
    lerpSpeed:      AT_CAMERA_PARAMS.CAMERA_Element_2_WorklerpSpeed,
    lerpSpeed2:     AT_CAMERA_PARAMS.CAMERA_Element_2_WorklerpSpeed2,
    lookAt:         AT_CAMERA_PARAMS.CAMERA_Element_2_WorklookAt,
    moveXY:         AT_CAMERA_PARAMS.CAMERA_Element_2_WorkmoveXY,
    position:       AT_CAMERA_PARAMS.CAMERA_Element_2_Workposition,
    rotation:       AT_CAMERA_PARAMS.CAMERA_Element_2_Workrotation,
  },
  work_page: {
    moveXY:   AT_CAMERA_PARAMS.CAMERA_Element_2_work_pagemoveXY,
    position: AT_CAMERA_PARAMS.CAMERA_Element_2_work_pageposition,
  },
  home_scene: {
    deltaRotate:    AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenedeltaRotate,
    fov:            AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenefov,
    groupPos:       AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenegroupPos,
    moveXY:         AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenemoveXY,
    position:       AT_CAMERA_PARAMS.CAMERA_Element_3_home_sceneposition,
    rotation:       AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenerotation,
    wobbleStrength: AT_CAMERA_PARAMS.CAMERA_Element_3_home_scenewobbleStrength,
  },
  ParticleTest: {
    groupPos: AT_CAMERA_PARAMS.CAMERA_Element_4_ParticleTestgroupPos,
    lookAt:   AT_CAMERA_PARAMS.CAMERA_Element_4_ParticleTestlookAt,
    moveXY:   AT_CAMERA_PARAMS.CAMERA_Element_4_ParticleTestmoveXY,
    position: AT_CAMERA_PARAMS.CAMERA_Element_4_ParticleTestposition,
    rotation: AT_CAMERA_PARAMS.CAMERA_Element_4_ParticleTestrotation,
  },
};

// ─── Controller ──────────────────────────────────────────────────────────────

export class CameraController {
  private current: CameraPreset | null = null;

  getPreset(scene: string): CameraPreset | undefined {
    return CAMERA_PRESETS[scene];
  }

  applyPreset(scene: string): CameraPreset | null {
    const preset = CAMERA_PRESETS[scene];
    if (!preset) {
      console.warn(`[CameraController] No preset for scene: "${scene}"`);
      return null;
    }
    this.current = preset;
    return preset;
  }

  getCurrent(): CameraPreset | null {
    return this.current;
  }

  listScenes(): string[] {
    return Object.keys(CAMERA_PRESETS);
  }
}

export default new CameraController();
