# vtk.js — Scientific Visualization WebGL Implementation
**小弟 #92 研究报告 | xiaodi**

## 项目概览

- **来源**: https://github.com/Kitware/vtk-js
- **定位**: VTK (Visualization Toolkit) 的 Web/WebGL 移植版，科学可视化工业标准
- **开发者**: Kitware Inc. — ParaView、CMake 同一团队
- **License**: BSD-3-Clause

---

## 架构分层

```
Sources/
├── Common/          # 核心数据结构 (DataModel, Core, Transform)
├── Filters/         # 数据处理管道 (Core, General, Modeling, Sources)
├── IO/              # 数据读写 (XML, Legacy VTK, Geometry, Image)
├── Imaging/         # 图像处理
├── Interaction/     # 交互 (Widgets, Manipulators, Style)
├── Proxy/           # 高层代理 (Representations, Animation)
├── Rendering/       # 渲染核心 ← 重点
│   ├── Core/        # 渲染器抽象接口
│   ├── OpenGL/      # WebGL 实现 ← 关键
│   ├── WebGPU/      # 新一代 WebGPU 实现
│   └── SceneGraph/  # 场景图管理
└── Widgets/         # 3D 交互控件
```

---

## Rendering/OpenGL/ — WebGL 渲染器详解

### 核心渲染对象
| 模块 | 作用 |
|------|------|
| `RenderWindow` | WebGL canvas 管理，帧循环控制 |
| `Renderer` | 场景渲染器，管理 Camera + Actors |
| `Camera` | 投影矩阵，视图变换 |
| `Actor` / `Actor2D` | 3D/2D 场景对象 |
| `Framebuffer` | 离屏渲染目标 |

### Mapper 体系（数据→WebGL 的桥梁）
| Mapper | 用途 |
|--------|------|
| `PolyDataMapper` | 多边形网格（2056行，核心） |
| `VolumeMapper` | 体渲染（1953行，光线投射） |
| `ImageMapper` | 2D/3D 图像切片 |
| `ImageResliceMapper` | 斜切面重采样 |
| `SphereMapper` | 粒子/点云球体 |
| `GlyphMapper` | 矢量场 Glyph |
| `PixelSpaceCallbackMapper` | 屏幕空间回调 |
| `CutterMapper` | 等值面切割 |
| `PolyDataMapper2D` | 2D 叠加层 |

### GPU 资源管理
| 模块 | 作用 |
|------|------|
| `ShaderProgram` | GLSL 编译/链接 |
| `ShaderCache` | Shader 缓存复用 |
| `Shader` | 顶点/片元 shader 对象 |
| `BufferObject` | VBO/IBO GPU 缓冲 |
| `VertexArrayObject` | VAO 状态封装 |
| `Texture` / `TextureUnitManager` | 纹理管理 |
| `CellArrayBufferObject` | VTK Cell 数组→GPU |

### 渲染通道（Pass 系统）
| Pass | 用途 |
|------|------|
| `ForwardPass` | 标准前向渲染 |
| `OrderIndependentTranslucentPass` | OIT 透明物体渲染 |
| `Convolution2DPass` | 后处理卷积滤波 |
| `RadialDistortionPass` | VR/镜头畸变校正 |

### GLSL Shaders (`glsl/`)
```
vtkPolyDataVS.glsl        # 网格顶点着色器
vtkPolyDataFS.glsl        # 网格片元着色器
vtkVolumeVS.glsl          # 体渲染顶点
vtkVolumeFS.glsl          # 体渲染光线投射（核心）
vtkSphereMapperVS.glsl    # 球形粒子 billboard
vtkStickMapperVS.glsl     # 棒状 Glyph
vtkPolyData2DVS/FS.glsl   # 2D 叠加渲染
vtkImageResliceMapper*.glsl # 斜切面重采样
```

---

## 科学可视化核心技术

### 体渲染（Volume Rendering）
- **技术**: GPU 光线投射（Ray Casting），单张 pass 完成
- `VolumeMapper/index.js`: 1953行，管理传输函数、不透明度、梯度
- 支持: CT/MRI 医学影像、流体仿真标量场

### 多边形渲染
- `PolyDataMapper`: 支持点、线、三角面、四边面
- 支持: Phong 光照、纹理贴图、标量着色（ColorMap）
- 自动处理 VTK 的 CellArray 格式→WebGL IBO

### 透明度处理
- `OrderIndependentTranslucentPass`: 实现 OIT
- 解决经典 WebGL 透明排序问题

### 硬件拾取（HardwareSelector）
- GPU 拾取：将 Actor/Cell ID 编码到颜色缓冲
- 支持大规模数据集的精确点击检测

### SurfaceLIC（表面线积分卷积）
- 流场可视化：在网格表面展示向量场流线纹理

---

## 与我们项目的关联

### SVG Figure / Cell PubSub 视角
1. **渲染管道模式** — vtk.js 的 Mapper→Actor→Renderer 分层
   与 Cell 的 producer→subscriber 数据流有结构相似性
2. **Pass 系统** — 多 Pass 渲染通道 ≈ Cell 多步骤处理流
3. **SceneGraph** — 场景图的节点订阅父节点变换，天然 PubSub
4. **HardwareSelector** — GPU 拾取技术可应用于 SVG 图形的精确交互

### 可借鉴的设计模式
- **ShaderCache**: 基于 hash 的着色器去重 → Cell 计算结果缓存
- **TextureUnitManager**: 资源槽位管理 → 订阅者槽位分配
- **ForwardPass + OIT Pass**: 有序渲染通道 → Cell 依赖拓扑排序

---

## 关键文件速查

```
Sources/Rendering/OpenGL/
├── VolumeMapper/index.js      # 体渲染核心 (1953行)
├── PolyDataMapper/index.js    # 网格渲染核心 (2056行)
├── glsl/vtkVolumeFS.glsl      # 光线投射 GLSL
├── ShaderProgram/index.js     # GLSL 编译管理
├── HardwareSelector/index.js  # GPU 拾取
└── ForwardPass/index.js       # 渲染主通道
```

---

*小弟 #92 — xiaodi | vtk.js scientific visualization research*
