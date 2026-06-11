# astro-svgfigure 渲染引擎开发计划

## 里程碑分配

| Claude | 里程碑 | 状态 | 任务 | 改动 |
|--------|--------|------|------|------|
| 第一位 | M001-M025 | ✅ 完成 | DeferredShading→ConstraintSolver, CyberRT debug, BasePass, AllLatest fusion | 5 files +266 lines |
| 第二位 | M026-M050 | ✅ 完成 | SceneRendering cell registry, Visibility epoch culling, Occlusion z-layer, Core lifecycle | 5 files +48 lines |
| 第三位 | M051-M075 | 待派发 | LightRendering/ShadowRendering: 光照→约束解算权重, 阴影→碰撞检测 | |
| 第四位 | M076-M100 | 待派发 | PostProcess/: Bloom→视觉强调, ToneMapping→色彩协调, AA→边缘抗锯齿 | |
| 第五位 | M101-M125 | 待派发 | TranslucentRendering/FogRendering: 半透明→z层叠加, 雾→深度衰减 | |
| 第六位 | M126-M150 | 待派发 | Apollo transport/scheduler: pub/sub→Git channel, 调度器→epoch控制 | |

## Debug Tag 统计 (28 total)

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

## 已改文件清单 (10 files, +314 lines)

```
upstream/unreal-renderer/DeferredShadingRenderer.cpp  +128  M001-M010
upstream/unreal-renderer/BasePassRendering.cpp         +43  M016-M020
upstream/unreal-renderer/SceneRendering.h              +12  M026-M030
upstream/unreal-renderer/SceneRendering.cpp             +9  M031-M035
upstream/unreal-renderer/SceneVisibility.cpp           +10  M036-M042
upstream/unreal-renderer/SceneOcclusion.cpp             +9  M043-M047
upstream/unreal-renderer/SceneCore.cpp                  +8  M048-M050
upstream/apollo-cyber/node/reader.h                    +32  M011-M013
upstream/apollo-cyber/node/writer.h                    +15  M014-M015
upstream/apollo-cyber/data/fusion/all_latest.h         +49  M021-M025
```

## 规则

- 在原文件上改 20% 算法, 不改文件名, 不加后缀
- 所有 debug 用 [ASTRO-*] fprintf, 便于 grep
- 作者: dylanyunlon <dogechat@163.com>
- 分支: cell-pubsub-loop
