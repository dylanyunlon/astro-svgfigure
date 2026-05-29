# NEXT_CLAUDE_HANDOFF.md

## Branch: `fix/cfc0672-enhancements`
## Latest commit: `2e6f8ab` (on top of `8e9538e`)

---

## 已完成

| Commit | 内容 | 文件 |
|--------|------|------|
| `c0e6191` | metaball blob renderer | `to-svg.ts` |
| `7b78b67` | 完整学术图渲染: 骨架色块、sprite-as-body、family 色板 | `to-svg.ts` |
| `d126d50` | sprite_injector.py + layered_pipeline Stage 2.8 | `sprite_injector.py`, `layered_pipeline.py` |
| `8e9538e` | tldraw 集成层 M300-M305 | `src/lib/tldraw/` 4 files |
| `2e6f8ab` | **关键接线**: inject_sprites 接入 generate_scientific_figure | `gemini_image_gen.py` |

## 完整调用链（已通）

```
/generate 页面 Step 2→3
  → /api/generate-image (src/pages/api/generate-image.ts)
    → Python backend /api/generate-image (server.py L591)
      → generate_scientific_figure() (gemini_image_gen.py L1196)
        → classify_nodes(elk_graph)          ← 标记 renderMode
        → inject_sprites(elk_graph)          ← Gemini 逐格子生成
          → design_prompts_for_classified()  ← 族内一致 prompt
          → plan_sheets()                    ← ≤16 cells/sheet
          → generate_sheets_concurrent()     ← Gemini API 调用
          → split_and_clean()                ← rembg 去背景
          → stamp_sprite_ref()               ← 写入 spriteRef
        → generate_image_with_gemini()       ← 最终 Gemini polish
```

## 未完成 / 已知问题

### 1. tldraw 未接入页面（第四位 Claude M318-M323）

- `tldraw` npm 包未安装
- `ElkCanvas.tsx` 没有被任何 `.astro` 页面引用
- `/generate` 页面仍用 `SvgPreview` Astro 组件（静态 SVG innerHTML）
- **需要做**: 替换 `<SvgPreview />` 为 React island `<ElkCanvas />`

### 2. Gemini interleaved output 未利用（第三位 Claude M312-M317）

- 当前 `sprite_batch_generator.py` 用 sprite sheet 网格方案（一张大图裁切）
- Gemini `responseModalities: ['TEXT','IMAGE']` 支持单次请求返回多张独立图片
- **应该替换为**: 一个 prompt 列出所有 sprite → Gemini 一次返回 N 张 `parts[].inline_data`
- 优势: 不需要 rembg 去背景、不需要 sheet 裁切、风格天然一致

### 3. tldraw ElkCanvas 缺少 sprite 实时更新

- 当 `spriteRef` 在后端生成后，前端 tldraw 画布不会自动刷新
- **需要**: M308 单节点重生 + M310 SpriteProgressIndicator

### 4. 颜色一致性已验证 ✓

ElkNodeShapeUtil 和 to-svg.ts 的 SKELETON_FILLS 和 FAMILY_PALETTES 完全一致。

## 下一步任务优先级

1. **P0**: `npm install tldraw` + 把 `ElkCanvas` 接入 `/generate` 页面（替换 SvgPreview）
2. **P0**: 改 `sprite_batch_generator.py` 用 Gemini interleaved output（替代 sheet 方案）
3. **P1**: 后端 `/api/sprite-batch/stream` SSE 端点（M313）
4. **P1**: 前端 Step 2 改为 "Generate Sprites" 按钮（M320）
5. **P2**: SpriteInspectTool + SpritePanelOverlay（M306-M307）

## 参考代码库位置

| 参考 | 路径 | 用途 |
|------|------|------|
| tldraw FlowchartShapeUtil | `/home/claude/tldraw/apps/examples/.../customMermaidShapeUtil.tsx` | ElkNodeShapeUtil 的 1:1 参考 |
| tldraw CustomShapeMermaids | `/home/claude/tldraw/apps/examples/.../CustomShapeMermaids.tsx` | elkToTldraw/ElkCanvas 的参考 |
| g-harel/blobs gen.ts | `/home/claude/blobs/internal/gen.ts` | 有机 blob 算法 |
| Paper.js Boolean.js | `/home/claude/paper.js/src/path/PathItem.Boolean.js` | 路径并集（未使用） |
| Inkscape drawing-shape | `/home/claude/inkscape/src/display/drawing-shape.cpp` | SVG 渲染参考 |
