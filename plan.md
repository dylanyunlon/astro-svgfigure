# plan.md — astro-svgfigure v2: ELK.js + NanoBanana 正向 Pipeline

> **核心理念**: AutoFigure-Edit 是逆向流程（图→SAM3分割→SVG），我们做正向流程（文本→拓扑JSON→ELK布局→NanoBanana SVG）  
> **技术栈**: Astro 5 + astro-pure 主题 + UnoCSS + ELK.js + Gemini NanoBanana + Python 后端  
> **关键洞察**: NanoBanana 生成的图片如此完美以至于让拓扑学家像个小丑——ELK.js 只负责骨架坐标，最终以 JSON 脚手架向 NanoBanana 请求神经网络级 SVG

---

## 0. 项目现状（基于 6 commits / main 分支）

```
astro-svgfigure/
├── packages/pure/           # astro-pure 主题包（本地 link）
├── preset/                  # 预设配置脚本
├── public/                  # 静态资源
├── src/                     # Astro 前端源码
│   ├── components/          # 已有组件
│   ├── content/             # 内容集合
│   ├── layouts/             # 布局模板
│   ├── pages/               # 页面路由
│   ├── plugins/             # Astro 插件
│   └── assets/              # 静态资源
├── autofigure2.py           # AutoFigure-Edit 逆向脚本
├── server.py                # Python 后端
├── astro.config.ts / package.json / uno.config.ts / tsconfig.json
├── requirements.txt / bun.lock
├── eslint.config.mjs / prettier.config.mjs / .editorconfig
├── CODE_OF_CONDUCT.md / LICENSE / README.md
├── plan.md                  # 本文件
└── tree.txt                 # 结构快照
```

**语言构成**: Astro 26.7% | TypeScript 25.2% | Python 18.1% | XSLT 12.6% | MDX 12.6% | JS 3.7% | CSS 1.1%

---

## 1. 架构总览：四步正向 Pipeline

```
[用户输入 method text]
         │
         ▼
┌─── Step 1: LLM 拓扑推理 ───┐     GitHub 背书: ResearAI/AutoFigure
│  Gemini 解析文本             │
│  输出 ELK JSON (零坐标)      │
└──────────┬──────────────────┘
           │ topology.json
           ▼
┌─── Step 2: ELK.js 约束布局 ─┐     GitHub 背书: kieler/elkjs
│  elkjs 分层/树形/力导向算法   │                  EmilStenstrom/elkjs-svg
│  计算每个节点 (x,y,w,h)     │                  xyflow/xyflow
└──────────┬──────────────────┘
           │ layouted.json
           ▼
┌─── Step 3: NanoBanana 美化 ─┐     GitHub 背书: gemini-cli-extensions/nanobanana
│  layouted.json → JSON 脚手架 │                  ZeroLu/awesome-nanobanana-pro
│  Gemini 生成学术级 SVG       │
└──────────┬──────────────────┘
           │ final.svg
           ▼
┌─── Step 4: Astro 前端展示 ──┐     GitHub 背书: cworld1/astro-theme-pure
│  astro-pure 主题组件渲染     │                  withastro/astro
│  SVG 预览 + 导出             │
└─────────────────────────────┘
```

---

## 2. 前端界面设计（纯 astro-pure 组件，不自创样式）

### 使用的 astro-pure 组件

| 组件 | import 路径 | 页面用途 |
|------|------------|---------|
| Header | `astro-pure/components/basic` | 全站顶栏 |
| Footer | `astro-pure/components/basic` | 全站页脚 |
| ThemeProvider | `astro-pure/components/basic` | 暗色模式 |
| Button | `astro-pure/user` | 生成/导出/复制按钮 |
| Card | `astro-pure/user` | 特性卡片/示例卡片/结果卡片 |
| Collapse | `astro-pure/user` | JSON 折叠展示 |
| Aside | `astro-pure/user` | 状态提示/错误/警告框 |
| Steps | `astro-pure/user` | Pipeline 四步指示器 |
| Tab | `astro-pure/user` | 算法切换/视图切换面板 |
| Timeline | `astro-pure/user` | 生成历史时间线 |
| Icon | `astro-pure/user` | SVG 图标 |
| Spoiler | `astro-pure/user` | JSON 脚手架细节 |
| GithubCard | `astro-pure/advanced` | 相关项目展示 |
| LinkPreview | `astro-pure/advanced` | 外部链接预览 |
| Toast | `astro-pure/advanced` | 操作通知 |
| MediumZoom | `astro-pure/advanced` | SVG 图片缩放 |

---

## 3. 100 文件变更清单

**标记**: 🆕 新增 | ✏️ 修改 | 🗑️ 删除 | 📌 不动

### A. 根目录配置 (10 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 001 | `package.json` | ✏️ | +elkjs +elkjs-svg +@google/generative-ai +zod |
| 002 | `astro.config.ts` | ✏️ | +hybrid output +vercel adapter +vite optimizeDeps |
| 003 | `uno.config.ts` | ✏️ | +pipeline UI shortcuts |
| 004 | `tsconfig.json` | ✏️ | +paths: @elk, @pipeline, @components |
| 005 | `requirements.txt` | ✏️ | +google-generativeai +lxml +cairosvg |
| 006 | `plan.md` | ✏️ | 替换为本文件 |
| 007 | `tree.txt` | ✏️ | 更新结构快照 |
| 008 | `README.md` | ✏️ | 更新说明+架构图+部署命令 |
| 009 | `bun.lock` | ✏️ | 自动更新 |
| 010 | `.env.example` | 🆕 | GEMINI_API_KEY= / PYTHON_BACKEND_URL= |

### B. Python 后端 (8 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 011 | `server.py` | ✏️ | +/api/topology +/api/beautify +/api/export 端点 |
| 012 | `autofigure2.py` | 📌 | 逆向流程参考，不动 |
| 013 | `pipeline/__init__.py` | 🆕 | Python pipeline 包 |
| 014 | `pipeline/topology_gen.py` | 🆕 | Gemini 拓扑生成 |
| 015 | `pipeline/nanobanana_bridge.py` | 🆕 | NanoBanana SVG 桥接 |
| 016 | `pipeline/svg_validator.py` | 🆕 | lxml 验证 + LLM 修复 |
| 017 | `pipeline/scaffold_builder.py` | 🆕 | ELK layouted → JSON 脚手架 |
| 018 | `pipeline/svg_scaler.py` | 🆕 | SVG 缩放（移植 inter_rl_figure.py） |

### C. Astro 页面 (12 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 019 | `src/pages/index.astro` | ✏️ | 重构: Hero + 特性卡片 + 快速入口 |
| 020 | `src/pages/generate.astro` | 🆕 | **核心** 四步 Pipeline 生成页 |
| 021 | `src/pages/gallery.astro` | 🆕 | 画廊: 示例 + 历史 |
| 022 | `src/pages/playground.astro` | 🆕 | ELK 布局实验场 |
| 023 | `src/pages/about.astro` | ✏️ | +技术架构说明 |
| 024 | `src/pages/api/topology.ts` | 🆕 | POST: text → topology JSON |
| 025 | `src/pages/api/layout.ts` | 🆕 | POST: topology → ELK layouted |
| 026 | `src/pages/api/beautify.ts` | 🆕 | POST: layouted → NanoBanana SVG |
| 027 | `src/pages/api/export.ts` | 🆕 | POST: SVG → PNG/PDF |
| 028 | `src/pages/api/validate.ts` | 🆕 | POST: SVG 语法校验 |
| 029 | `src/pages/docs/[...slug].astro` | ✏️ | +svgfigure 文档路由 |
| 030 | `src/pages/404.astro` | 📌 | 不动 |

### D. Astro 布局 (5 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 031 | `src/layouts/BaseLayout.astro` | ✏️ | +elkjs CDN, +generate 页面 meta |
| 032 | `src/layouts/GenerateLayout.astro` | 🆕 | 生成页三栏面板布局 |
| 033 | `src/layouts/GalleryLayout.astro` | 🆕 | 画廊瀑布流网格布局 |
| 034 | `src/layouts/PlaygroundLayout.astro` | 🆕 | Playground 全屏布局 |
| 035 | `src/layouts/DocLayout.astro` | 📌 | 不动 |

### E. 组件 — Pipeline 核心 (15 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 036 | `src/components/pipeline/PipelineSteps.astro` | 🆕 | 四步指示器 (Steps) |
| 037 | `src/components/pipeline/TextInput.astro` | 🆕 | Method text 输入 (Card) |
| 038 | `src/components/pipeline/TopologyPreview.astro` | 🆕 | 拓扑 JSON 预览 (Collapse) |
| 039 | `src/components/pipeline/ElkPreview.astro` | 🆕 | ELK 骨架 SVG 预览 |
| 040 | `src/components/pipeline/SvgPreview.astro` | 🆕 | 最终 SVG 预览 (MediumZoom) |
| 041 | `src/components/pipeline/GenerateButton.astro` | 🆕 | 生成按钮 (Button variant) |
| 042 | `src/components/pipeline/ExportPanel.astro` | 🆕 | 导出面板 (Button 组) |
| 043 | `src/components/pipeline/StatusBar.astro` | 🆕 | 状态栏 (Aside) |
| 044 | `src/components/pipeline/JsonEditor.astro` | 🆕 | JSON 编辑器 (textarea) |
| 045 | `src/components/pipeline/ElkOptions.astro` | 🆕 | ELK 参数面板 (Tab) |
| 046 | `src/components/pipeline/HistoryList.astro` | 🆕 | 生成历史 (Timeline) |
| 047 | `src/components/pipeline/ErrorDisplay.astro` | 🆕 | 错误展示 (Aside danger) |
| 048 | `src/components/pipeline/PromptPreview.astro` | 🆕 | NanoBanana prompt 预览 |
| 049 | `src/components/pipeline/ScaffoldView.astro` | 🆕 | JSON 脚手架 (Spoiler) |
| 050 | `src/components/pipeline/CompareView.astro` | 🆕 | 骨架 vs 最终对比 |

### F. 组件 — Gallery & 展示 (8 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 051 | `src/components/gallery/FigureCard.astro` | 🆕 | 图片卡片 (Card) |
| 052 | `src/components/gallery/FigureGrid.astro` | 🆕 | 网格布局 |
| 053 | `src/components/gallery/FilterBar.astro` | 🆕 | 分类筛选 (Tab) |
| 054 | `src/components/gallery/ExampleList.astro` | 🆕 | 内置示例 |
| 055 | `src/components/gallery/StylePicker.astro` | 🆕 | 风格选择 |
| 056 | `src/components/gallery/RepoCards.astro` | 🆕 | GitHub 仓库卡片 (GithubCard) |
| 057 | `src/components/gallery/LinkCards.astro` | 🆕 | 链接预览 (LinkPreview) |
| 058 | `src/components/gallery/ZoomableSvg.astro` | 🆕 | 可缩放 SVG (MediumZoom) |

### G. 组件 — 首页 & 通用 (8 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 059 | `src/components/home/Hero.astro` | 🆕 | 首页 Hero (Button CTA) |
| 060 | `src/components/home/FeatureCards.astro` | 🆕 | 特性卡片 (Card 网格) |
| 061 | `src/components/home/ArchDiagram.astro` | 🆕 | 架构图 (内联 SVG) |
| 062 | `src/components/home/QuickStart.astro` | 🆕 | 快速开始 (Steps) |
| 063 | `src/components/home/TechStack.astro` | 🆕 | 技术栈 (Icon + Card) |
| 064 | `src/components/common/SvgRenderer.astro` | 🆕 | 通用 SVG 渲染器 |
| 065 | `src/components/common/LoadingSpinner.astro` | 🆕 | 加载动画 |
| 066 | `src/components/common/CopyButton.astro` | 🆕 | 复制按钮 (Button + Toast) |

### H. TS 库 — ELK (10 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 067 | `src/lib/elk/index.ts` | 🆕 | ELK 模块入口 |
| 068 | `src/lib/elk/layout.ts` | 🆕 | ELK.js 布局封装 |
| 069 | `src/lib/elk/algorithms.ts` | 🆕 | 算法选项 (layered/mrtree/force/stress) |
| 070 | `src/lib/elk/to-svg.ts` | 🆕 | ELK → 骨架 SVG |
| 071 | `src/lib/elk/to-scaffold.ts` | 🆕 | ELK → NanoBanana 脚手架 |
| 072 | `src/lib/elk/presets.ts` | 🆕 | 预设参数 (学术图/流程图/架构图) |
| 073 | `src/lib/elk/types.ts` | 🆕 | TS 类型定义 |
| 074 | `src/lib/elk/validator.ts` | 🆕 | 拓扑 JSON 验证 (zod) |
| 075 | `src/lib/elk/examples.ts` | 🆕 | 内置示例拓扑 |
| 076 | `src/lib/elk/constants.ts` | 🆕 | ELK 常量 |

### I. TS 库 — Pipeline (10 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 077 | `src/lib/pipeline/index.ts` | 🆕 | Pipeline 编排入口 |
| 078 | `src/lib/pipeline/topology-gen.ts` | 🆕 | LLM 拓扑生成 |
| 079 | `src/lib/pipeline/nanobanana.ts` | 🆕 | NanoBanana API 封装 |
| 080 | `src/lib/pipeline/svg-validator.ts` | 🆕 | SVG 验证 (DOMParser) |
| 081 | `src/lib/pipeline/svg-scaler.ts` | 🆕 | SVG 缩放变换 |
| 082 | `src/lib/pipeline/svg-optimizer.ts` | 🆕 | SVG 优化 (SVGO) |
| 083 | `src/lib/pipeline/export.ts` | 🆕 | 导出逻辑 (PNG/PDF) |
| 084 | `src/lib/pipeline/types.ts` | 🆕 | Pipeline 类型定义 |
| 085 | `src/lib/pipeline/prompts.ts` | 🆕 | LLM prompt 模板集 |
| 086 | `src/lib/pipeline/cache.ts` | 🆕 | 结果缓存 |

### J. 内容集合 & 文档 (8 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 087 | `src/content/config.ts` | ✏️ | +gallery 集合 schema |
| 088 | `src/content/docs/elkjs-guide.mdx` | 🆕 | ELK.js 使用指南 |
| 089 | `src/content/docs/pipeline-overview.mdx` | 🆕 | Pipeline 架构文档 |
| 090 | `src/content/docs/nanobanana-integration.mdx` | 🆕 | NanoBanana 集成文档 |
| 091 | `src/content/docs/api-reference.mdx` | 🆕 | API 接口文档 |
| 092 | `src/content/docs/examples.mdx` | 🆕 | 使用示例 |
| 093 | `src/content/gallery/transformer.json` | 🆕 | 示例数据: Transformer |
| 094 | `src/content/gallery/diffusion.json` | 🆕 | 示例数据: Diffusion |

### K. 静态资源 & 部署 (6 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 095 | `public/examples/transformer.svg` | 🆕 | 示例 SVG |
| 096 | `public/examples/diffusion.svg` | 🆕 | 示例 SVG |
| 097 | `public/examples/gan.svg` | 🆕 | 示例 SVG |
| 098 | `src/assets/styles/generate.css` | 🆕 | 生成页最小化补充样式 |
| 099 | `src/site.config.ts` | ✏️ | +header menu: Generate/Gallery/Playground |
| 100 | `Dockerfile` | 🆕 | Docker 一键部署 |

---

### 统计汇总

| 操作 | 数量 |
|------|------|
| 🆕 新增 | 73 |
| ✏️ 修改 | 18 |
| 📌 不动 | 4 |
| 🗑️ 删除 | 0 |
| **合计** | **100** (含 5 个 📌) |

---

## 4. 修改文件 diff 要点

### package.json

```diff
  "dependencies": {
+   "elkjs": "^0.9.3",
+   "elkjs-svg": "^0.1.0",
+   "@google/generative-ai": "^0.21.0",
+   "zod": "^3.23.0",
    "astro": "...",
    "astro-pure": "...",
    // 保留所有现有依赖不变
  }
```

### astro.config.ts

```diff
+ import vercel from '@astrojs/vercel';
  export default defineConfig({
+   output: 'hybrid',
+   adapter: vercel(),
    integrations: [ /* 保留全部现有 */ ],
+   vite: {
+     optimizeDeps: { include: ['elkjs/lib/elk.bundled.js'] }
+   }
  })
```

### src/site.config.ts

```diff
  header: {
    menu: [
      { title: 'Blog', link: '/blog' },
      { title: 'Docs', link: '/docs' },
+     { title: 'Generate', link: '/generate' },
+     { title: 'Gallery', link: '/gallery' },
+     { title: 'Playground', link: '/playground' },
    ],
  },
```

### server.py

```diff
+ from pipeline.topology_gen import generate_topology
+ from pipeline.nanobanana_bridge import beautify_with_nanobanana
+ from pipeline.svg_validator import validate_svg
+
+ @app.route('/api/topology', methods=['POST'])
+ def api_topology(): ...
+
+ @app.route('/api/beautify', methods=['POST'])
+ def api_beautify(): ...
+
+ @app.route('/api/export', methods=['POST'])
+ def api_export(): ...
  # 保留全部现有端点
```

### requirements.txt

```diff
+ google-generativeai>=0.8.0
+ lxml>=5.0.0
+ cairosvg>=2.7.0
+ Pillow>=10.0.0
  # 保留全部现有依赖
```

---

## 5. 部署命令

```bash
# 开发
git clone https://github.com/dylanyunlon/astro-svgfigure.git && cd astro-svgfigure
bun install && bun add elkjs elkjs-svg @google/generative-ai zod
pip install -r requirements.txt
cp .env.example .env  # 填入 GEMINI_API_KEY
bun run dev            # 终端 1: Astro
python server.py       # 终端 2: Python

# Docker
docker build -t astro-svgfigure .
docker run -p 4321:4321 -p 8000:8000 -e GEMINI_API_KEY=your-key astro-svgfigure

# Vercel
npx vercel --prod
```

---

## 6. 全部 GitHub 背书 (15 个)

| # | 项目 | Step | 用途 |
|---|------|------|------|
| 1 | [ResearAI/AutoFigure](https://github.com/ResearAI/AutoFigure) | 1 | LLM 学术图生成 (ICLR 2026) |
| 2 | [ResearAI/AutoFigure-Edit](https://github.com/ResearAI/AutoFigure-Edit) | 1 | 逆向流程参考 |
| 3 | [kieler/elkjs](https://github.com/kieler/elkjs) | 2 | 核心布局引擎 |
| 4 | [EmilStenstrom/elkjs-svg](https://github.com/EmilStenstrom/elkjs-svg) | 2 | ELK → SVG |
| 5 | [xyflow/xyflow](https://github.com/xyflow/xyflow) | 2 | React Flow + ELK |
| 6 | [davidthings/hdelk](https://github.com/davidthings/hdelk) | 2 | ELK + SVG.js |
| 7 | [eclipse.org/elk](https://www.eclipse.org/elk/reference.html) | 2 | 算法参考 |
| 8 | [gemini-cli-extensions/nanobanana](https://github.com/gemini-cli-extensions/nanobanana) | 3 | NanoBanana CLI |
| 9 | [ZeroLu/awesome-nanobanana-pro](https://github.com/ZeroLu/awesome-nanobanana-pro) | 3 | Prompt 工程 |
| 10 | [aaronkwhite/nanobanana-studio-web](https://github.com/aaronkwhite/nanobanana-studio-web) | 3 | 自托管生成 |
| 11 | [GeminiGenAI/Free-Nano-Banana-Pro-API](https://github.com/GeminiGenAI/Free-Nano-Banana-Pro-API-Ultimate-AI-Image-Generator) | 3 | API 参考 |
| 12 | [cworld1/astro-theme-pure](https://github.com/cworld1/astro-theme-pure) | 4 | 主题源码 |
| 13 | [withastro/astro](https://github.com/withastro/astro) | 4 | 框架 |
| 14 | [natemoo-re/astro-icon](https://github.com/natemoo-re/astro-icon) | 4 | SVG 内联 |
| 15 | [rtsys.informatik.uni-kiel.de/elklive](https://rtsys.informatik.uni-kiel.de/elklive/) | 2 | ELK 在线 Demo |