# M1314c: geometry.json → CellData 桥接结果

## 任务摘要

将 `channels/cell/<id>/geometry.json` 中的 SDF 数据桥接到 GPU 渲染管线的 `CellData`，
实现 metaball SDF 参数在每帧渲染时可用。

## 修改文件

### 1. `src/lib/sph/gpu-render-loop.ts` — CellData 接口扩展

在 `CellData` 接口末尾追加 **M1314c** SDF 字段（均为可选，向后兼容）：

```ts
// === M1314c: geometry.json SDF bridge ===
sdfBaseRadius?: number;       // sdf.base_radius — metaball core radius
sdfLobes?: Array<{ angle: number; distance: number; radius: number }>;
sdfNoiseAmp?: number;         // sdf.noise_amplitude
sdfNoiseFreq?: number;        // sdf.noise_frequency
surfaceGlowIntensity?: number;// surface.glow_intensity
membraneThickness?: number;   // membrane.thickness
membraneWobbleAmp?: number;   // membrane.wobble_amplitude
cytplasmFlowSpeed?: number;   // internal_motion.cytoplasm_flow_speed
```

### 2. `src/pages/world/index.astro` — loadGPUScene() 新增步骤 2b

在原有步骤 2（composite_params.json → cells[]）之后、步骤 3（edge_routes）之前，
插入并发 fetch 逻辑：

```ts
// 并发 fetch 所有 cell 的 geometry.json
const cellIds = cells.map(c => c.cell_id);
const geoPromises = cellIds.map(id =>
  fetch('/channels/cell/' + id + '/geometry.json')
    .then(r => r.ok ? r.json() : null)
    .catch(() => null)
);
const geos = await Promise.all(geoPromises);

// 将 SDF 字段 merge 进 CellData
for (let i = 0; i < cells.length; i++) {
  const geo = geos[i];
  if (!geo) continue;
  // 安全类型检查后 spread 合并
  cells[i] = { ...cells[i], sdfBaseRadius: ..., sdfLobes: ..., ... };
}
console.info(`[gpu-world] geometry.json SDF merged: ${geoHitCount}/${cells.length} cells`);
```

## geometry.json 数据结构（来源）

```json
{
  "cell_id": "add_norm2",
  "sdf": {
    "type": "metaball",
    "base_radius": 15.699,
    "lobes": [{ "angle": 0.2137, "distance": 11.553, "radius": 9.983 }],
    "noise_amplitude": 0.0538,
    "noise_frequency": 4.188
  },
  "surface": { "glow_intensity": 0.744, ... },
  "membrane": { "thickness": 2.453, "wobble_amplitude": 0.0155, ... },
  "internal_motion": { "cytoplasm_flow_speed": 0.3, ... }
}
```

## 关键设计决策

| 决策 | 理由 |
|------|------|
| 所有新字段均为 `optional` | 不破坏已有 `satisfies CellData` 校验，无 geometry.json 的 cell 正常降级 |
| `Promise.all` 并发 fetch | 58 个 cell 串行需 ~1-2s，并发约 100-200ms |
| `.catch(()=>null)` | 单个 geometry.json 404/网络错误不阻断整体加载 |
| 只改两个指定文件 | 最小化变更范围，不触碰 pass 层 |

## 验证

- `src/lib/sph/gpu-render-loop.ts` 行 78-87：8 个新 SDF 字段已添加
- `src/pages/world/index.astro` 行 1790-1821：并发 fetch + merge 逻辑完整插入
- 已有字段（`glowColor`, `sdfShape` 等）和 `satisfies CellData` 断言未受影响

## 状态

✅ 完成 — commit M1314c pushed to `cell-pubsub-loop`
