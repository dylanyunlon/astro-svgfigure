# Cell Geometry Channel Format

GPU 渲染循环每帧从 `/channels/cell/{id}/geometry.json` 读取几何数据。
Cell agent (Claude 对话) 每 tick 写入此文件。这是 cell ↔ GPU 的唯一接口。

## 数据流

```
Cell Agent (Claude 对话)          GPU Render Loop (browser)
         │                                 │
  感知环境 (read physics/*.json)            │
         │                                 │
  决定形变                                  │
         │                                 │
  写 geometry.json  ──── SSE/poll ────→  读 geometry.json
         │                                 │
         │                          SDF metaball shader 渲染
         │                                 │
  写 status.json                      下一帧
         │                                 │
  下一个 tick                               │
```

## geometry.json Schema

```typescript
interface CellGeometry {
  cell_id: string;
  tick: number;
  timestamp_ms: number;

  // 世界空间位置 + 变换
  transform: {
    x: number;        // 世界像素坐标
    y: number;
    scale: number;    // 1.0 = 原始大小
    rotation: number; // 弧度
  };

  // SDF 形状定义 — shader 用这个渲染轮廓
  sdf: {
    type: 'metaball';
    base_radius: number;  // 基础球半径 (像素)
    lobes: Array<{
      angle: number;      // 弧度, 从中心出发的方向
      distance: number;   // 离中心的距离 (像素)
      radius: number;     // lobe 球的半径
    }>;
    noise_amplitude: number;  // 表面噪声振幅 [0, 0.1]
    noise_frequency: number;  // 表面噪声频率 [1, 10]
  };

  // PBR 材质属性
  surface: {
    albedo: [number, number, number];  // RGB [0,1]
    roughness: number;                  // [0,1]
    metallic: number;                   // [0,1]
    opacity: number;                    // [0,1]
    glow_color: [number, number, number];
    glow_intensity: number;             // [0,2]
  };

  // 细胞膜参数
  membrane: {
    thickness: number;          // 膜厚度 (像素)
    wobble_amplitude: number;   // 膜波动振幅
    wobble_frequency: number;   // 膜波动频率
    permeability_visual: number; // 膜通透性视觉效果 [0,1]
  };

  // 伪足 — 向邻居伸出的突起
  pseudopods: Array<{
    target_cell: string;  // 目标 cell id
    length: number;       // 伪足长度 (像素)
    width: number;        // 伪足宽度
    curl: number;         // 弯曲度 [-1, 1]
  }>;

  // 内部运动 — cytoplasm 流动
  internal_motion: {
    cytoplasm_flow_angle: number;  // 流动方向 (弧度)
    cytoplasm_flow_speed: number;  // 流动速度 [0,1]
    organelle_drift: number;       // 细胞器随机漂移量
  };
}
```

## GPU Shader 如何使用

PBR fragment shader 中的 SDF 计算:

```glsl
// metaball SDF: 基础球 + lobes 叠加
float cellSDF(vec2 uv, vec2 center, float baseR, Lobe lobes[MAX_LOBES], int lobeCount) {
    float d = length(uv - center) - baseR;

    for (int i = 0; i < lobeCount; i++) {
        vec2 lobeCenter = center + vec2(
            cos(lobes[i].angle) * lobes[i].distance,
            sin(lobes[i].angle) * lobes[i].distance
        );
        float lobeDist = length(uv - lobeCenter) - lobes[i].radius;
        // smooth union (metaball blend)
        float k = 0.3; // blend smoothness
        float h = clamp(0.5 + 0.5 * (lobeDist - d) / k, 0.0, 1.0);
        d = mix(lobeDist, d, h) - k * h * (1.0 - h);
    }

    // surface noise
    d += noise(uv * noiseFreq) * noiseAmp;

    return d;
}
```

## 默认值 (cell 还没写 geometry.json 时)

如果某个 cell 没有 geometry.json, GPU 用 params.json 的 bbox 画一个简单圆角矩形。
一旦 cell agent 写了第一个 geometry.json, 就切换到 metaball SDF 渲染。
