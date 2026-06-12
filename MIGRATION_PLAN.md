# MIGRATION PLAN — astro-svgfigure cell-pubsub-loop

## 六位 Claude 并行开发计划

### Claude-1: PixiJS Engine 融合 (M001-M026)
将 upstream/pixijs-engine 的 26 个核心模块逐个融合到 src/lib/renderers/。
每个模块 = 1 个 M 里程碑。鲁迅式：mv upstream 文件，改 20% 算法适配 cell 渲染。

- M001: app (Application 初始化 + resize 策略)
- M002: scene/graphics (Graphics API → cell body 绘制)
- M003: scene/text (Text + TextStyle → cell label 渲染)
- M004: scene/mesh (MeshGeometry → SDF quad 几何)
- M005: scene/container (Container 层级 → z-layer 管理)
- M006: scene/sprite (Sprite → cell icon/图标资源)
- M007: filters/blur (BlurFilter → bloom glow 管线)
- M008: filters/color-matrix (ColorMatrixFilter → species 色调)
- M009: filters/mask (MaskFilter → group 裁剪)
- M010: rendering/batcher (Batcher → cell 批量渲染)
- M011: rendering/render-target (RenderTarget → offscreen buffer)
- M012: events (EventSystem → cell 交互/hover/click)
- M013: assets (Assets loader → 纹理/字体加载)
- M014: math (Matrix/Point/Rectangle → bbox 变换)
- M015: color (Color → species palette 管理)
- M016: ticker (Ticker → epoch 动画帧循环)
- M017: culling (视锥剔除 → SceneVisibility 前端版)
- M018: accessibility (A11y → cell ARIA 标签)
- M019: environment-browser (浏览器适配层)
- M020: compressed-textures (纹理压缩 → 大图优化)
- M021: spritesheet (图集 → species icon 图集)
- M022: prepare (资源预热 → 首帧加速)
- M023: unsafe-eval (CSP 策略 → shader 编译)
- M024: advanced-blend-modes (混合模式 → glow/additive)
- M025: dom (DOM 桥接 → Astro 组件集成)
- M026: bundle (最终打包 → tree-shaking 优化)

### Claude-2: PixiJS Filters 融合 (M027-M064)
将 upstream/pixijs-filters 的 38 个 filter 融合到渲染管线。
重点: bloom, glow, godray, kawase-blur, outline, drop-shadow。

- M027: advanced-bloom (AT HydraBloom 等价)
- M028: bloom (基础 bloom)
- M029: kawase-blur (AT downSample/upSample)
- M030: godray (光束效果)
- M031: glow (外发光)
- M032: outline (描边)
- M033: drop-shadow (投影)
- M034: motion-blur (运动模糊)
- M035: adjustment (亮度/对比度/饱和度)
- M036: color-gradient (渐变映射)
- M037: color-overlay (颜色叠加)
- M038: color-replace (颜色替换)
- M039: color-map (色彩映射)
- M040: crt (CRT 扫描线效果)
- M041: dot (网点效果)
- M042: ascii (ASCII 艺术)
- M043: glitch (故障效果)
- M044: pixelate (像素化)
- M045: emboss (浮雕)
- M046: bevel (斜面)
- M047: bulge-pinch (凸出/缩放)
- M048: convolution (卷积滤镜)
- M049: cross-hatch (交叉影线)
- M050: twist (扭曲)
- M051: zoom-blur (径向模糊)
- M052: old-film (旧胶片)
- M053: reflection (水面反射)
- M054: rgb-split (RGB 分离)
- M055: shockwave (冲击波)
- M056: tilt-shift (移轴)
- M057: radial-blur (旋转模糊)
- M058: grayscale (灰度)
- M059: backdrop-blur (背景模糊)
- M060: multi-color-replace (多色替换)
- M061: simplex-noise (单纯噪声)
- M062: hsl-adjustment (HSL 调整)
- M063: defaults (默认 filter 配置)
- M064: filter pipeline 集成测试

### Claude-3: Theatre.js 动画融合 (M065-M090)
将 upstream/theatre-js 融合到 epoch loop 可视化。
让每个 epoch 的 cell 状态变化映射到时间线上。

- M065: core/types (类型定义)
- M066: core/sequences (序列播放器)
- M067: core/sheets (Sheet → epoch 状态表)
- M068: core/projects (Project → topology 项目)
- M069: core/propTypes (属性类型 → cell 属性)
- M070: core/keyframes (关键帧 → epoch 快照)
- M071: dataverse/atoms (响应式原子 → cell 状态)
- M072: dataverse/derivations (派生 → constraint 计算)
- M073: dataverse/prism (棱镜 → 多 cell 聚合)
- M074-M080: studio UI 面板集成
- M081-M085: epoch timeline 编辑器
- M086-M090: 动画导出 + 回放

### Claude-4: UE5 算法深度融合 (M091-M140)
将 upstream/unreal-renderer-ue5 的高级算法融合到 channels/ Python 代码。
重点: Nanite, Lumen, Virtual Shadow Maps, TSR。

- M091-M100: Nanite (自适应网格细分 → cell 自适应 LOD)
- M101-M110: Lumen (全局光照 → cell 风格一致性传播)
- M111-M120: Virtual Shadow Maps → cell 阴影精度
- M121-M130: TSR (时间超分 → epoch 帧插值)
- M131-M140: Substrate (材质系统 → species 材质属性)

### Claude-5: Cell Agent 自主发育 (M141-M180)
让每个 cell 成为真正的 Claude 小弟，自主发育。
channel_runtime.py 的 pub/sub 驱动多轮 epoch。

- M141-M150: cell agent prompt 模板 (每个 species 的发育规则)
- M151-M160: 多轮 epoch 信号传递 (resize/push/pull)
- M161-M170: cell 自主 web search (学术特征 → svgwrite 算法)
- M171-M180: convergence 控制 + divergence rollback

### Claude-6: Astro 前端集成 (M181-M210)
把 PixiJS 渲染器集成到 Astro 页面组件。
SvgPreview.astro → PixiPreview.astro。

- M181-M190: PixiPreview.astro 组件 (替换 SvgPreview)
- M191-M200: TopologyEditor 组件 (ELK → PixiJS 实时编辑)
- M201-M210: 导出管线 (PixiJS canvas → PNG/SVG/MP4)

---

## 当前进度

| Claude | 范围 | 状态 |
|--------|------|------|
| Claude-1 | M001-M026 PixiJS Engine | 待开始 |
| Claude-2 | M027-M064 PixiJS Filters | 待开始 |
| Claude-3 | M065-M090 Theatre.js | 待开始 |
| Claude-4 | M091-M140 UE5 算法 | 待开始 |
| Claude-5 | M141-M180 Cell Agent | 待开始 |
| Claude-6 | M181-M210 Astro 前端 | 待开始 |

总计: 210 个里程碑, 6 位 Claude 并行
