# REFACTOR PLAN: 消灭 SVG 硬编码，走渲染库管线

## 现状诊断

### 冗余系统
前端 `pixi-cell-renderer.ts` 已经用 PixiJS Graphics API 实现了全部 10 个 species。
后端 `species_port.py` 用 f-string 拼 SVG 做了同样的事。两套并行，后端那套是废的。

### SVG 硬编码文件清单

| # | 文件 | 函数/区域 | 硬编码内容 | 依赖方 |
|---|------|----------|-----------|--------|
| 1 | `channels/rendering/species/species_port.py` | `generate_svg_cil_*` ×10 | 全部 species 的 SVG body | `cell_component.py` |
| 2 | `channels/cell_component.py` L132-337 | `proc()` 中 svg 拼接 + `write_channel(svg.svg)` | cell wrapper `<g>` + svg body | `loop_orchestrator.py`, `msdf_gen.py` |
| 3 | `channels/loop_orchestrator.py` L2467-2964 | `assemble_final_svg()` | `<svg>` root + defs + z-layer `<g>` + edge `<path>` | 最终输出 `output_cell_loop.svg` |
| 4 | `backend/pipeline/nanobanana_bridge.py` L236-290 | `generate_skeleton_svg()` | 骨架 `<rect>` + `<text>` + arrow `<path>` | pipeline fallback |
| 5 | `channels/rendering/lighting/lighting.py` L1556-1734 | `generate_svg_water_overlay()` | 水面叠加 `<rect>` 层 | compositor |
| 6 | `channels/rendering/decoration/decoration_extra.py` L175-310 | 装饰叠加 | `<circle>` `<line>` `<path>` 装饰 | species_port |
| 7 | `channels/rendering/compositor/compositor_core.py` | `<g>` + `<svg>` 拼合 | 合成层 | loop_orchestrator |
| 8 | `channels/rendering/nanite/composition.py` | `<g>` + `<svg>` 拼合 | Nanite 合成 | loop_orchestrator |

---

## 重构路线 (按依赖序)

### Phase 1: cell_component.py — 停止写 svg.svg，只写 params.json

**目标**: `proc()` 不再调用 `SPECIES_GENERATORS`，不再生成 SVG 字符串，不再写 `svg.svg`。
只写 `params.json`（已经在做了 L446-470），让前端 PixiJS 渲染器消费 params。

**改动**:
- `cell_component.py` L131-133: 删除 `generator = SPECIES_GENERATORS.get(...)` 和 `svg_content, actual_bbox = generator(...)`
- `cell_component.py` L185-210: 删除 svg_content 字符串替换（颜色注入）
- `cell_component.py` L319-337: 删除 `full_svg` 拼接和 `write_channel(svg.svg)`
- 保留 L446-470 的 `params.json` 写入（这是正确的输出）
- 保留所有 rendering 模块调用（StyleProbe、shadow、crowding、registry）

**验证**: `proc()` 运行后只产出 `params.json`，不产出 `svg.svg`

### Phase 2: species_port.py — 删除全部 generate_svg_* 函数

**目标**: 这个文件只保留 `_species_f0()`、`_species_to_index()` 和 `SPECIES_GENERATORS` dict（改为导出 species metadata 而非 SVG 生成器）。

**改动**:
- 删除 `generate_svg_cil_eye` ~ `generate_svg_graph` 全部 10 个函数 (~470 行)
- `SPECIES_GENERATORS` 改名为 `SPECIES_METADATA`，值改为 species 元数据 dict（颜色、F0、roughness），不再是函数
- 保留 `_species_f0()`（被 rendering 模块引用）
- 保留 `_species_to_index()`（被 cell_component 引用）

**验证**: import species_port 不再有任何 SVG 字符串

### Phase 3: msdf_gen.py — 从 params.json 生成 MSDF，不依赖 svg.svg

**目标**: MSDF 生成从 species 元数据 + params.json 中的几何参数计算 SDF path，不再读 svg.svg。

**改动**:
- `msdf_gen.py` L298-303: 改为读 `params.json` 中的 `species_params`
- 根据 species 类型，用算法生成 MSDF 需要的 path（和 species_port 做的一样，但输出是 path d-string 给 msdfgen binary，不是 SVG 字符串）
- 或者：用 `tiny-sdf` (upstream) 在前端直接生成 SDF，完全绕过后端 msdfgen

**验证**: `python msdf_gen.py self_attn` 生成 msdf.png 不依赖 svg.svg

### Phase 4: assemble_final_svg() — 用 PixiJS extract 导出，不拼 SVG

**目标**: 最终输出走前端 PixiJS renderer.extract → Canvas → SVG/PNG，不走后端字符串拼接。

**改动**:
- `loop_orchestrator.py` L2467-2964: `assemble_final_svg()` 改为：
  1. 收集所有 cell 的 `params.json`
  2. POST 到前端 `/api/export`（已有此 route）
  3. 前端 PixiJS 渲染所有 cell → `renderer.extract.canvas()` → `canvas.toDataURL('image/svg+xml')`
  4. 或者用 `modern-screenshot` (upstream/RESEARCH_128) 的 `domToForeignObjectSvg()` 导出
- 后端只负责收集 params 和触发导出，不拼 SVG

**备选方案**: 如果纯前端导出复杂度太高，可以用 headless PixiJS (node-canvas) 在后端渲染：
```
npm install @pixi/node  # PixiJS Node.js adapter
```
后端 Python 调 Node.js 子进程渲染 → 输出 PNG/SVG。

**验证**: `output_cell_loop.svg` 是渲染结果导出，不是字符串拼接

### Phase 5: nanobanana_bridge.py — skeleton_svg 走 ELK → PixiJS 管线

**改动**:
- `generate_skeleton_svg()` 删除
- 骨架预览走前端 `TopologyPreview.astro` 的 PixiJS 渲染
- fallback 时返回 `params.json` 数据让前端渲染，不返回 SVG 字符串

### Phase 6: 清理装饰/光照/合成层的 SVG 拼接

- `decoration_extra.py`: 装饰参数化，输出到 `params.json` 的 `decoration` 字段
- `lighting.py`: 水面叠加参数化，前端用 PixiJS filter (DisplacementFilter) 实现
- `compositor_core.py` / `nanite/composition.py`: 合成逻辑移到前端 PixiJS Container 层级

---

## 不动的部分

- `cell_agent.py` — 已经只输出 JSON params，不生成 SVG ✓
- `pixi-cell-renderer.ts` — 已经用 PixiJS Graphics API 渲染 ✓
- `sdf-cell-renderer.ts` — SDF 着色器渲染 ✓
- `src/lib/shaders/cil-*.frag` — Species SDF 着色器 ✓
- `src/lib/shaders/msdf.frag/vert` — MSDF 文字渲染 ✓
- `src/lib/shaders/edge-spline.*` — Edge 渲染 ✓
- 全部 `channels/data/` pub/sub 通信层 ✓
- 全部 `channels/transport/` 传输层 ✓

---

## 执行顺序

Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6

Phase 1 和 2 可以一起做（同一次 commit）。
Phase 3 依赖 Phase 1 完成（不再有 svg.svg）。
Phase 4 是最大的改动，需要前后端联调。
Phase 5 和 6 是清理工作。
