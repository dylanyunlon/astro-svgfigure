# astro-svgfigure 渲染引擎开发计划

## 里程碑分配

| Claude | 里程碑 | 状态 | 任务 |
|--------|--------|------|------|
| 第一位 | M001-M025 | ✅ 完成 | DeferredShading→ConstraintSolver, CyberRT debug, BasePass constraint write, AllLatest fusion |
| 第二位 | M026-M050 | 待派发 | SceneRendering/SceneVisibility 改造: InitViews→约束收集, 可见性裁剪→z-layer过滤 |
| 第三位 | M051-M075 | 待派发 | LightRendering/ShadowRendering 改造: 光照→约束解算权重, 阴影→遮挡检测 |
| 第四位 | M076-M100 | 待派发 | PostProcess/ 改造: Bloom→视觉强调, ToneMapping→色彩协调, AA→边缘抗锯齿 |
| 第五位 | M101-M125 | 待派发 | TranslucentRendering/FogRendering 改造: 半透明→z层叠加, 雾→深度衰减 |
| 第六位 | M126-M150 | 待派发 | Apollo transport/scheduler 改造: pub/sub通道→Git channel, 调度器→epoch控制 |

## 文件改动追踪

### M001-M025 (第一位 Claude, ✅ 已完成)
```
upstream/unreal-renderer/DeferredShadingRenderer.cpp  +128 lines
  - FConstraintBufferData 结构体 (替代 GBuffer)
  - Render() 管线每个阶段加 [ASTRO-RENDER] debug fprintf
  - ConstraintSolver 逻辑嵌入 DeferredLighting 阶段

upstream/unreal-renderer/BasePassRendering.cpp         +43 lines
  - ConstraintBuffer 写入逻辑
  - [ASTRO-BASEPASS] slot tracking debug

upstream/apollo-cyber/node/reader.h                    +32 lines
  - [ASTRO-CHANNEL] Reader observe debug

upstream/apollo-cyber/node/writer.h                    +15 lines
  - [ASTRO-CHANNEL] Writer publish debug

upstream/apollo-cyber/data/fusion/all_latest.h         +49 lines
  - [ASTRO-FUSION] AllLatest fusion trigger debug
```

## 规则

- 不改文件名, 不加后缀, 在原文件上改 20% 算法
- 所有 debug 输出用 [ASTRO-*] tag, 便于 grep 过滤
- 作者: dylanyunlon <dogechat@163.com>
- 分支: cell-pubsub-loop
- 不允许 v10/port/copy/old/new/backup 等后缀
