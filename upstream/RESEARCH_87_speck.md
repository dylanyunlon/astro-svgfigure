# RESEARCH #87 — Speck: GPU Impostor Spheres for Cell Rendering

**Researcher:** 小弟#87 (xiaodi)  
**Source:** https://github.com/wwwtyro/speck  
**Branch:** cell-pubsub-loop  
**Date:** 2026-06-15

---

## 核心洞察：不用多边形，用 Impostor 画球

Speck 渲染分子不用传统的二十面体 mesh。每个原子是一个 **billboard quad**（一张朝向摄像机的矩形面片），fragment shader 里做 **ray-sphere intersection** 来决定每个像素是否命中球体、命中哪个点、法线是什么。

### 优势
- **零几何细分**：一个原子 = 1 个 quad（6 个顶点），不管球有多大，细节全在 fragment 阶段
- **完美轮廓**：pixel 级精度，不存在低多边形球的锯齿
- **正确深度**：通过 `gl_FragDepthEXT` 写入真实交点深度，AO 和 outline 都能正确工作

---

## atoms.glsl 解析

### Vertex Shader
```glsl
attribute vec3 aImposter;   // billboard 顶点偏移（单位正方形的角）
attribute vec3 aPosition;   // 原子世界坐标
attribute float aRadius;    // 原子半径

void main() {
    vRadius = uAtomScale * (1.0 + (aRadius - 1.0) * uRelativeAtomScale);
    // 关键：把 billboard quad 按半径缩放，平移到原子位置
    gl_Position = uProjection * uView * uModel * vec4(vRadius * aImposter + aPosition, 1.0);
}
```

Billboard quad 以原子为中心、以半径为尺度展开，正好包住整个球的投影。

### Fragment Shader — Ray-Sphere Intersection

```glsl
float raySphereIntersect(vec3 r0, vec3 rd) {
    float a = dot(rd, rd);
    vec3 s0_r0 = r0 - vPosition;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (vRadius * vRadius);
    float disc = b*b - 4.0*a*c;
    if (disc <= 0.0) return -1.0;            // 射线未命中球
    return (-b - sqrt(disc)) / (2.0*a);      // 取近交点
}

void main() {
    // 从 NDC fragment 坐标重建世界空间射线（正交投影）
    vec3 r0 = vec3(uBottomLeft + (gl_FragCoord.xy/res) * (uTopRight - uBottomLeft), 0.0);
    vec3 rd = vec3(0, 0, -1);   // 正交投影：射线方向恒定

    float t = raySphereIntersect(r0, rd);
    if (t < 0.0) discard;                    // 未命中 → 丢弃像素

    vec3 coord  = r0 + rd * t;              // 交点世界坐标
    vec3 normal = normalize(coord - vPosition); // 球面法线

    gl_FragDepthEXT = -coord.z / uDepth;    // 写入真实深度！
}
```

**数学原理**：标准二次方程 `|r0 + t*rd - center|² = r²`，展开得 `at² + bt + c = 0`，判别式 `disc < 0` 即无交点。

---

## Ambient Occlusion 管线 (accumulator.glsl)

Speck 的 AO 不是 SSAO，而是 **多帧随机旋转累积**：

1. 每帧把整个场景从一个随机旋转方向重新渲染一遍深度图（`uRandRotDepth`）
2. `accumulator.glsl` 比较当前像素深度与旋转后深度，判断该方向是否被遮挡
3. 结果累积进 RGBA 四通道（每通道 256 次采样，共 1024 次）
4. `ao.glsl` 最终 `shade = pow(1 - avgOcclusion, 2.0)` 乘到颜色上

```glsl
// accumulator.glsl 核心
float ao = step(dRandRot, depth * 0.99);   // 该方向被遮挡？
float mag = dot(dir, normal);               // 背面剔除
float sampled = step(0.0, mag);
ao *= sampled;
// 写入累积缓冲（每通道 255 次）
acc.r += ao / 255.0;
```

---

## Depth-Aware Outlines (ao.glsl)

```glsl
// 读 4 个相邻像素深度，取最大差值
float d0 = abs(texture2D(uSceneDepth, p + vec2(-r, 0)).r - depth);
// ...
float d = max(d0, max(d1, max(d2, d3)));
sceneColor.rgb *= pow(1.0 - d, uOutlineStrength * 32.0);  // 边缘变暗
```

纯深度差值，不依赖法线，对 impostor 球天然有效。

---

## 对 Cell 渲染的迁移方案

### 为什么 Cell 适合 Impostor？

| 对比维度 | Mesh Sphere | Impostor Sphere |
|---------|------------|-----------------|
| 顶点数/cell | ~600（二十面体） | 6（1 个 quad） |
| 轮廓精度 | 受分辨率限制 | 像素级精确 |
| 深度正确性 | 自动 | 需手写 gl_FragDepth |
| 支持 AO | 需要复杂设置 | 天然（有精确法线） |
| 实现复杂度 | 简单 | 中等（shader 数学） |

### 迁移要点

1. **每个 Cell = 1 billboard quad**，传入 `position`（细胞中心）和 `radius`
2. **正交 vs 透视**：Speck 用正交（`rd = (0,0,-1)`）。透视摄像机需改为：
   ```glsl
   vec3 rd = normalize(r0 - uCameraPos);
   ```
3. **深度写入**：必须用 `gl_FragDepthEXT`（WebGL1）或 `gl_FragDepth`（WebGL2）
4. **PubSub 集成**：cell 位置/半径通过 pubsub 更新 → 只需更新 GPU buffer，shader 不变
5. **AO 可选**：Speck 的多帧累积 AO 对静态分子很好，但 cell 动态场景需考虑帧数收敛问题

### 伪代码（WebGL2）

```glsl
// cell_vert.glsl
uniform mat4 uMVP;
attribute vec3 aBillboard;  // [-1,-1,0], [1,-1,0], etc.
attribute vec3 aCenter;
attribute float aRadius;
varying vec3 vCenter;
varying float vRadius;

void main() {
    vCenter = aCenter;
    vRadius = aRadius;
    gl_Position = uMVP * vec4(aRadius * aBillboard + aCenter, 1.0);
}

// cell_frag.glsl
void main() {
    vec3 ro = reconstructWorldRay();   // 从 gl_FragCoord 重建
    vec3 rd = normalize(ro - uCamera);
    
    // 标准 ray-sphere
    vec3 oc = ro - vCenter;
    float a = dot(rd, rd);
    float b = 2.0 * dot(oc, rd);
    float c = dot(oc, oc) - vRadius * vRadius;
    float disc = b*b - 4.0*a*c;
    if (disc < 0.0) discard;
    
    float t = (-b - sqrt(disc)) / (2.0 * a);
    vec3 hit = ro + t * rd;
    vec3 normal = normalize(hit - vCenter);
    
    gl_FragDepth = computeDepth(hit);   // 真实深度
    gl_FragColor = shadingFunction(normal, uCellColor);
}
```

---

## 文件索引

| 文件 | 作用 |
|------|------|
| `src/shaders/atoms.glsl` | **核心**：billboard vert + ray-sphere frag，写 gl_FragDepth |
| `src/shaders/bonds.glsl` | 圆柱 impostor（ray-cylinder intersection） |
| `src/shaders/accumulator.glsl` | 多帧随机旋转 AO 累积 |
| `src/shaders/ao.glsl` | AO 合成 + 深度差 outline |
| `src/shaders/blur.glsl` | AO 模糊 |
| `src/shaders/dof.glsl` | 景深后处理 |
| `src/renderer.js` | WebGL 管线编排，多 pass 渲染顺序 |

---

## 结论

Speck 证明了 **impostor sphere 在 WebGL 中完全可行**，代码极简（~50 行 shader），视觉质量远超同等多边形球。对于 cell-pubsub-loop 中每帧可能有大量 cell 实例的场景，impostor 方案可以把顶点数降低 100x，同时获得更好的 AO 支持。

**推荐下一步**：在 cell renderer 里实现一个 `CellImpostorPass`，替换现有的 `SphereGeometry` mesh 方案。
