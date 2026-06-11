# astro-svgfigure 渲染引擎开发计划

## 总览

- 54 个 upstream C++ 文件改动
- +2602 行
- 419 个 [ASTRO-*] debug tags, 50 种 tag 类型
- 27 个 commits, 16+ 位小弟 Claude 完成

## 已完成里程碑

| # | 里程碑 | commit | 文件 |
|---|--------|--------|------|
| 1 | M001-M025 | c8a63b9 | DeferredShadingRenderer, BasePass, reader/writer, all_latest |
| 2 | M026-M050 | c5e6574 | SceneRendering/Visibility/Occlusion/Core |
| 3 | M051-M055 | 4aad4e3 | LightRendering |
| 4 | M056-M060 | 1c4195d | ShadowRendering |
| 5 | M061-M065 | 91a717b | Component 4-channel fusion |
| 6 | M066-M070 | 0e33d60 | TranslucentRendering |
| 7 | M071-M075 | 9670c02 | FogRendering + GammaCorrection |
| 8 | M076-M080 | c430892 | VelocityRendering + DistortionRendering |
| 9 | M081-M085 | a44de10 | MeshPassProcessor + MeshDrawCommands |
| 10 | M086-M090 | f2c5d36 | DataVisitor + Dispatcher + ChannelBuffer |
| 11 | M091-M095 | ad90841 | RendererScene |
| 12 | M096-M100 | 6d145b2 | ShadowDepth + ShadowSetup |
| 13 | M101-M105 | d9cbac3 | TranslucentLighting + AtmosphereRendering |
| 14 | M106-M110 | f5f2446 | ReflectionCapture + GlobalDistanceField |
| 15 | M111-M115 | 29fe176 | CellNode + TopologyManager |
| 16 | M116-M120 | 5583652 | VolumetricFog + LightShaft |
| 17 | M121-M125 | 25cd034 | PrimitiveSceneInfo + IndirectLightingCache |
| 18 | M126-M135 | 71692b6 | VolumetricFog + LightShaft + Apollo scheduler/transport/notifier |
| 19 | M141-M150 | 853ee71 | DistanceField + PostProcessing + CompositionGraph |

## Upstream 来源

- upstream/unreal-renderer/ — Unreal Engine 4.22 Renderer/Private (330 files)
- upstream/apollo-cyber/ — Apollo CyberRT (200 files)

## 渲染管线映射

```
Unreal                          astro-svgfigure
─────                          ───────────────
Actor/Component          →     Cell (sub-Claude)
Scene Graph              →     ELK Topology
BasePass → GBuffer       →     Constraint Collect → constraints.json
Lighting Pass            →     Constraint Solver (relative → absolute)
PostProcess              →     SVG post-processing (edge softening, highlight)
Final Composite          →     SVG Assembly (z-layer ordering)
```

## 规则

- 在原文件上改 20% 算法, 不改文件名, 不加后缀
- 所有 debug 用 [ASTRO-*] fprintf, 便于 grep
- 作者: dylanyunlon <dogechat@163.com>
- 分支: cell-pubsub-loop
