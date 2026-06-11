# astro-svgfigure 渲染引擎开发计划

## 里程碑分配

| Claude | 里程碑 | 状态 | 任务 | 改动 |
|--------|--------|------|------|------|
| 第一位 | M001-M025 | ✅ 完成 | DeferredShading→ConstraintSolver, CyberRT debug, BasePass, AllLatest fusion | 5 files +266 lines |
| 第二位 | M026-M050 | ✅ 完成 | SceneRendering cell registry, Visibility epoch culling, Occlusion z-layer, Core lifecycle | 5 files +48 lines |
| 第三位 | M051-M075 | ✅ 完成 | LightRendering/ShadowRendering: 光照→约束解算权重, 阴影→碰撞检测 | |
| 第四位 | M076-M100 | ✅ 完成 | VelocityRendering epoch delta, DistortionRendering force field, Bloom→视觉强调, ToneMapping→色彩协调, AA→边缘抗锯齿 | 5 files +160 lines |
| 第五位 | M101-M125 | ✅ 完成 | FogRendering: 雾→深度衰减 | 1 file +23 lines |
| 第六位 | M126-M150 | ✅ 完成 | Apollo transport: pub/sub→Git channel, scheduler→epoch控制, processor→epoch executor | 3 files +46 lines |

## Debug Tag 统计 (60+ total)

| Tag | Count | 来源 |
|-----|-------|------|
| [ASTRO-RENDER] | 8 | DeferredShadingRenderer.cpp |
| [ASTRO-SCENE] | 4 | SceneRendering.cpp/h |
| [ASTRO-FUSION] | 3 | all_latest.h |
| [ASTRO-CHANNEL] | 3 | reader.h, writer.h |
| [ASTRO-BASEPASS] | 3 | BasePassRendering.cpp |
| [ASTRO-VISIBILITY] | 2 | SceneVisibility.cpp |
| [ASTRO-OCCLUSION] | 2 | SceneOcclusion.cpp |
| [ASTRO-CORE] | 2 | SceneCore.cpp |
| [ASTRO-CELL] | 1 | DeferredShadingRenderer.cpp |
| [ASTRO-LIGHT] | 3 | LightRendering.cpp |
| [ASTRO-SHADOW] | 3 | ShadowRendering.cpp |
| [ASTRO-TRANSLUCENT] | 3 | TranslucentRendering.cpp |
| [ASTRO-VELOCITY] | 4 | VelocityRendering.cpp |
| [ASTRO-DISTORTION] | 4 | DistortionRendering.cpp |
| [ASTRO-BLOOM] | 3 | PostProcessBloomSetup.cpp |
| [ASTRO-TONEMAP] | 1 | PostProcessTonemap.cpp |
| [ASTRO-AA] | 1 | PostProcessAA.cpp |
| [ASTRO-FOG] | 3 | FogRendering.cpp |
| [ASTRO-TRANSPORT] | 3 | transport.cc |
| [ASTRO-SCHEDULER] | 3 | scheduler.cc |
| [ASTRO-PROCESSOR] | 2 | processor.cc |

## 已改文件清单 (21 files, +520 lines)

```
upstream/unreal-renderer/DeferredShadingRenderer.cpp        +128  M001-M010
upstream/unreal-renderer/BasePassRendering.cpp               +43  M016-M020
upstream/unreal-renderer/SceneRendering.h                    +12  M026-M030
upstream/unreal-renderer/SceneRendering.cpp                   +9  M031-M035
upstream/unreal-renderer/SceneVisibility.cpp                 +10  M036-M042
upstream/unreal-renderer/SceneOcclusion.cpp                   +9  M043-M047
upstream/unreal-renderer/SceneCore.cpp                        +8  M048-M050
upstream/apollo-cyber/node/reader.h                          +32  M011-M013
upstream/apollo-cyber/node/writer.h                          +15  M014-M015
upstream/apollo-cyber/data/fusion/all_latest.h               +49  M021-M025
upstream/unreal-renderer/LightRendering.cpp                  +18  M051-M055
upstream/unreal-renderer/ShadowRendering.cpp                 +15  M056-M060
upstream/unreal-renderer/TranslucentRendering.cpp            +12  M061-M075
upstream/unreal-renderer/VelocityRendering.cpp               +28  M076-M080
upstream/unreal-renderer/DistortionRendering.cpp             +21  M076-M080
upstream/unreal-renderer/PostProcess/PostProcessBloomSetup.cpp +17 M081-M090
upstream/unreal-renderer/PostProcess/PostProcessTonemap.cpp  +14  M091-M095
upstream/unreal-renderer/PostProcess/PostProcessAA.cpp       +11  M096-M100
upstream/unreal-renderer/FogRendering.cpp                    +23  M101-M110
upstream/apollo-cyber/transport/transport.cc                 +17  M126-M135
upstream/apollo-cyber/scheduler/scheduler.cc                 +18  M136-M145
upstream/apollo-cyber/scheduler/processor.cc                 +11  M146-M150
```

## 规则

- 在原文件上改 20% 算法, 不改文件名, 不加后缀
- 所有 debug 用 [ASTRO-*] fprintf, 便于 grep
- 作者: dylanyunlon <dogechat@163.com>
- 分支: cell-pubsub-loop
