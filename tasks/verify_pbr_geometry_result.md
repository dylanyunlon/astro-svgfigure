# verify-pbr.html — geometry.json 接入结果

**任务**: M1314a  
**完成时间**: 2026-07-01  
**执行者**: claude-worker-a  
**分支**: cell-pubsub-loop

---

## 修改摘要

`public/verify-pbr.html` 从依赖 `composite_params.json` 改为直接 fetch 3个 cell 的真实 geometry.json。

### 主要改动

| 项目 | 旧版本 | 新版本 |
|------|--------|--------|
| 数据源 | `composite_params.json` (可能不存在) | `channels/cell/{id}/geometry.json` × 3 |
| cell 列表 | 动态从 composite_params 读取 | 固定: `['input', 'self_attn', 'ffn']` |
| SDF 参数 | hardcoded fallback (0.5, 0.03, 4.0) | 直接用 geometry.json 真实值 |
| layout | 依赖 bbox 坐标系 | NDC 水平三等分排列 |
| shader 入口 | `mapScene()` 通用调度 | `cellSDF()` (GEOMETRY_FORMAT.md 实现) |
| 图例 | 无 | 右上角显示每个 cell 的 albedo / baseRadius / tick |

---

## 数据映射

### geometry.json → uniform

```
sdf.base_radius    → uBaseRadius  (归一化: rawR/30.0, clamp[0.3,1.0])
sdf.lobes[i].angle    → uLobes[i].x
sdf.lobes[i].distance → uLobes[i].y  (同比缩放: dist/rawR * normR)
sdf.lobes[i].radius   → uLobes[i].z  (同比缩放: rad/rawR * normR)
sdf.noise_amplitude   → uNoiseAmp  (×8.0 放大, 原值极小)
sdf.noise_frequency   → uNoiseFreq
surface.albedo        → uAlbedo    (RGB [0,1])
surface.roughness     → uRoughness
surface.metallic      → uMetallic
surface.opacity       → uOpacity
surface.glow_color    → uGlowColor
surface.glow_intensity → uGlowIntens
```

### 实际加载的 geometry.json 字段值 (tick=16)

| cell | base_radius | lobes | albedo | opacity |
|------|-------------|-------|--------|---------|
| input | 17.683 | 1个 (angle=-0.370, dist=13.0, r=11.3) | [0.38, 0.49, 0.55] | 0.65 |
| self_attn | 15.718 | 1个 (angle=0.070, dist=11.6, r=10.0) | [0.12, 0.58, 0.82] | 0.65 |
| ffn | 17.662 | 1个 (angle=0.026, dist=13.0, r=11.2) | [0.093, 0.59, 0.83] | 1.00 |

---

## cellSDF() 实现

来自 `channels/cell/GEOMETRY_FORMAT.md`, 移植到 GLSL fragment shader:

```glsl
float cellSDF(vec3 p, float baseR) {
    float d = sdSphere(p, baseR);
    for (int i = 0; i < uLobeCount; i++) {
        vec3 lobeCenter = vec3(
            cos(uLobes[i].x) * uLobes[i].y,
            sin(uLobes[i].x) * uLobes[i].y,
            sin(uTime * 0.4 + uLobes[i].x) * 0.04
        );
        float lobeDist = sdSphere(p - lobeCenter, uLobes[i].z);
        float k = 0.18;  // smooth union blend
        float h = clamp(0.5 + 0.5 * (lobeDist - d) / k, 0.0, 1.0);
        d = mix(lobeDist, d, h) - k * h * (1.0 - h);
    }
    d += noise2(p.xy * uNoiseFreq + uTime * 0.3) * uNoiseAmp;
    return d;
}
```

---

## 渲染验证

- ✅ 3个 geometry.json 成功 fetch (input / self_attn / ffn)
- ✅ base_radius 差异体现在3个球的大小不同 (input≈ffn > self_attn)
- ✅ lobe 方向不同: input lobe 偏左下 (-0.37rad), self_attn 偏右 (0.07rad), ffn 接近水平 (0.03rad)  
- ✅ albedo 颜色不同: input 偏灰蓝, self_attn / ffn 偏亮蓝
- ✅ ffn opacity=1.0 完全不透明, input/self_attn opacity≈0.65 半透明
- ✅ PBR Cook-Torrance + rim glow + 膜边缘高光全部正常
- ✅ 动画: lobe Z轴 wobble (sin(time)) + 表面噪声动画运行流畅
- ✅ 不再依赖 composite_params.json

## 文件行数

旧: 304 行 → 新: 425 行
