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




03更新：
# plan.md — astro-svgfigure v2: ELK.js + NanoBanana 正向 Pipeline

> **核心理念**: AutoFigure-Edit 是逆向流程（图→SAM3分割→SVG），我们做正向流程（文本→拓扑JSON→ELK布局→NanoBanana SVG）
> **技术栈**: Astro 5 + astro-pure 主题 + UnoCSS + ELK.js + Gemini NanoBanana + Python 后端
> **关键洞察**: NanoBanana 生成的图片如此完美以至于让拓扑学家像个小丑——ELK.js 只负责骨架坐标，最终以 JSON 脚手架向 NanoBanana 请求神经网络级 SVG
> **AI请求模式**: 参照 skynetCheapBuy 的 `app/core/ai_engine.py` 多 Provider 抽象工厂模式

---

## 0. 项目现状（基于 main 分支 7 commits）

```
astro-svgfigure/  (commit: 3f5faab)
├── packages/pure/           # astro-pure 主题包（本地 workspace link）
│   ├── components/
│   │   ├── basic/          # Header, Footer, ThemeProvider
│   │   ├── user/           # Button, Card, Collapse, Aside, Steps, Tab, Timeline, Icon, Spoiler...
│   │   ├── advanced/       # GithubCard, LinkPreview, MediumZoom, Quote, QRCode
│   │   └── pages/          # Hero, PostPreview, Paginator, TOC, BackToTop, PFSearch...
│   ├── utils/              # clsx, toast, theme, date, reading-time...
│   ├── libs/               # icons
│   ├── schemas/            # 主题配置 schemas
│   ├── plugins/            # rehype/remark 插件
│   └── types/              # TS 类型定义
├── preset/                  # 预设配置（icons, components, scripts）
├── public/                  # 静态资源（favicon, images, icons, scripts）
├── src/
│   ├── components/
│   │   ├── home/           # Section.astro, SkillLayout.astro, ProjectCard.astro
│   │   ├── projects/       # Sponsors.astro, ProjectSection.astro
│   │   ├── links/          # FriendList.astro
│   │   ├── about/          # ToolSection.astro, Substats.astro
│   │   └── waline/         # 评论系统
│   ├── content/
│   │   ├── blog/           # 博客文章
│   │   └── docs/           # 文档 (setup/, integrations/, advanced/)
│   ├── layouts/            # BaseLayout, ContentLayout, BlogPost, CommonPage, IndividualPage
│   ├── pages/              # index, blog, docs, projects, links, about, search, tags, archives, 404, terms
│   ├── plugins/            # shiki 插件
│   ├── assets/             # 图片, 样式(global.css, app.css), tools SVG
│   ├── content.config.ts   # 内容集合定义 (blog + docs)
│   ├── site.config.ts      # 站点配置
│   └── type.d.ts
├── autofigure2.py           # AutoFigure-Edit 逆向脚本 (2793行, 参考用)
├── server.py                # FastAPI 后端 (522行)
├── astro.config.ts          # 已有 vercel adapter + server output
├── package.json             # astro 5 + astro-pure + sharp + katex
├── requirements.txt         # Python 依赖
├── uno.config.ts / tsconfig.json / prettier / eslint
└── plan.md                  # 本文件
```

**语言构成**: Astro 26.7% | TypeScript 25.2% | Python 18.1% | XSLT 12.6% | MDX 12.6% | JS 3.7% | CSS 1.1%

**skynetCheapBuy 参考**: AI Provider 抽象工厂 (`AIProvider` → `OpenAIProvider` / `ClaudeCompatibleProvider` / `AnthropicProvider` / `GoogleProvider`)，以 `AIEngine._get_provider(model)` 自动路由，支持 streaming / non-streaming / function calling。

---

## 1. 架构总览：四步正向 Pipeline

```
[用户输入 method text]
         │
         ▼
┌─── Step 1: LLM 拓扑推理 ───┐     GitHub: ResearAI/AutoFigure
│  Gemini/Claude 解析文本      │             ResearAI/AutoFigure-Edit
│  输出 ELK JSON (零坐标)      │
│  AI Engine 多 Provider 路由   │     GitHub: dylanyunlon/skynetCheapBuy
└──────────┬──────────────────┘              (ai_engine.py 参考)
           │ topology.json
           ▼
┌─── Step 2: ELK.js 约束布局 ─┐     GitHub: kieler/elkjs
│  elkjs layered/mrtree/force  │             EmilStenstrom/elkjs-svg
│  计算每个节点 (x,y,w,h)     │             xyflow/xyflow (ReactFlow ELK)
│  + 边的路由点                │             cytoscape/cytoscape.js-elk
└──────────┬──────────────────┘             davidthings/hdelk
           │ layouted.json                   eclipse.org/elk (算法参考)
           ▼
┌─── Step 3: NanoBanana 美化 ─┐     GitHub: gemini-cli-extensions/nanobanana
│  layouted.json → JSON 脚手架 │             ZeroLu/awesome-nanobanana-pro
│  json_example_user1 模板     │             aaronkwhite/nanobanana-studio-web
│  Gemini NanoBanana 神经网络  │             GeminiGenAI/Free-Nano-Banana-Pro-API
│  级 SVG 生成                 │
└──────────┬──────────────────┘
           │ final.svg
           ▼
┌─── Step 4: Astro 前端展示 ──┐     GitHub: cworld1/astro-theme-pure
│  astro-pure 组件渲染         │             withastro/astro
│  SVG 预览 + 导出 + 画廊     │             natemoo-re/astro-icon
│  ELK Playground              │             svgdotjs/svg.js
└─────────────────────────────┘

后端 AI 请求:
┌─── AI Engine (仿 skynetCheapBuy) ──┐
│  AIProvider 抽象基类                  │  GitHub: dylanyunlon/skynetCheapBuy
│  ├── OpenAIProvider     (openai SDK) │          openai/openai-python
│  ├── AnthropicProvider  (anthropic)  │          anthropics/anthropic-sdk-python
│  ├── GoogleProvider     (genai)      │          google/generative-ai-python
│  └── ClaudeCompatible  (/v1/messages)│
│  AIEngine._get_provider(model) 路由  │
└──────────────────────────────────────┘
```

---

## 2. 前端界面设计（纯 astro-pure 组件，不自创样式）

### 已确认可用的 astro-pure 组件 (packages/pure/components/)

| 组件 | import 路径 | 页面用途 |
|------|------------|---------|
| Header | `astro-pure/components/basic` | 全站顶栏 (已用于 BaseLayout) |
| Footer | `astro-pure/components/basic` | 全站页脚 (已用于 BaseLayout) |
| ThemeProvider | `astro-pure/components/basic` | 暗色模式 (已用于 BaseLayout) |
| Button | `astro-pure/user` | 生成/导出/复制/CTA 按钮 |
| Card | `astro-pure/user` | 特性卡片/结果卡片/示例卡片 |
| CardList/Children | `astro-pure/user` | 多卡片列表 |
| Collapse | `astro-pure/user` | JSON 折叠展示 |
| Aside | `astro-pure/user` | 状态提示/错误/警告框 |
| Steps | `astro-pure/user` | Pipeline 四步指示器 |
| Tabs/TabItem | `astro-pure/user` | 算法切换/视图切换面板 |
| Timeline | `astro-pure/user` | 生成历史时间线 |
| Icon | `astro-pure/user` | SVG 图标 |
| Label | `astro-pure/user` | 标签展示 |
| Spoiler | `astro-pure/user` | JSON 脚手架细节折叠 |
| Svg | `astro-pure/user` | 内联 SVG 渲染 |
| FormattedDate | `astro-pure/user` | 日期格式化 |
| GithubCard | `astro-pure/advanced` | GitHub 仓库卡片展示 |
| LinkPreview | `astro-pure/advanced` | 外部链接预览 |
| MediumZoom | `astro-pure/advanced` | SVG/图片缩放 |
| Quote | `astro-pure/advanced` | 引用块 |
| QRCode | `astro-pure/advanced` | 二维码生成 |
| Hero | `astro-pure/components/pages` | 首页 Hero 区域 |
| PostPreview | `astro-pure/components/pages` | 内容预览 |
| Paginator | `astro-pure/components/pages` | 分页 |
| TOC | `astro-pure/components/pages` | 文档目录 |
| BackToTop | `astro-pure/components/pages` | 回到顶部 |
| PFSearch | `astro-pure/components/pages` | 搜索 |

### 工具函数 (packages/pure/utils/)

| 工具 | 用途 |
|------|------|
| `toast` | 操作通知 (复制成功/导出完成) |
| `clsx` / `class-merge` | 样式合并 |
| `theme` | 主题工具 |
| `date` | 日期格式化 |
| `reading-time` | 阅读时间 |

---

## 3. 111 文件变更清单

**标记**: 🆕 新增 | ✏️ 修改 | 🗑️ 删除 | 📌 不动

### A. 根目录配置 (10 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 001 | `package.json` | ✏️ | +elkjs +@google/generative-ai +zod +openai +@anthropic-ai/sdk |
| 002 | `astro.config.ts` | ✏️ | +vite.optimizeDeps.include: ['elkjs/lib/elk.bundled.js'] (已有 vercel+server output) |
| 003 | `uno.config.ts` | ✏️ | +pipeline 相关 shortcut classes |
| 004 | `tsconfig.json` | ✏️ | +paths: @elk→src/lib/elk, @pipeline→src/lib/pipeline, @ai→src/lib/ai |
| 005 | `requirements.txt` | ✏️ | +google-generativeai +lxml +cairosvg +openai +anthropic +httpx +pydantic-settings |
| 006 | `plan.md` | ✏️ | 替换为本文件 |
| 007 | `tree.txt` | ✏️ | 更新结构快照 |
| 008 | `README.md` | ✏️ | 更新说明+架构图+部署命令 |
| 009 | `bun.lock` | ✏️ | 自动更新 (bun install 后) |
| 010 | `.env.example` | 🆕 | GEMINI_API_KEY= / OPENAI_API_KEY= / ANTHROPIC_API_KEY= / OPENAI_API_BASE= / PYTHON_BACKEND_URL= |

### B. Python 后端 — AI Engine (仿 skynetCheapBuy) (12 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 011 | `server.py` | ✏️ | +/api/topology +/api/beautify +/api/export +/api/validate 端点; +ai_engine 初始化 |
| 012 | `autofigure2.py` | 📌 | 逆向流程参考，不动 |
| 013 | `backend/__init__.py` | 🆕 | Python 后端包入口 |
| 014 | `backend/config.py` | 🆕 | Settings (pydantic-settings) 仿 skynetCheapBuy/app/config.py |
| 015 | `backend/ai_engine.py` | 🆕 | **核心** AIProvider基类 + OpenAIProvider + AnthropicProvider + GoogleProvider + ClaudeCompatibleProvider + AIEngine 工厂 |
| 016 | `backend/schemas.py` | 🆕 | Pydantic models: TopologyRequest, BeautifyRequest, ExportRequest, StreamChunk |
| 017 | `backend/pipeline/__init__.py` | 🆕 | Pipeline 子包 |
| 018 | `backend/pipeline/topology_gen.py` | 🆕 | Gemini/Claude 拓扑生成 (调用 ai_engine) |
| 019 | `backend/pipeline/nanobanana_bridge.py` | 🆕 | NanoBanana SVG 桥接 (json_example_user1 脚手架 → Gemini 生成) |
| 020 | `backend/pipeline/svg_validator.py` | 🆕 | lxml 验证 + LLM 修复 (调用 ai_engine) |
| 021 | `backend/pipeline/scaffold_builder.py` | 🆕 | ELK layouted.json → NanoBanana JSON 脚手架转换器 |
| 022 | `backend/pipeline/svg_scaler.py` | 🆕 | SVG 缩放（移植 autofigure2.py 中的缩放逻辑） |

### C. Astro 页面 (14 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 023 | `src/pages/index.astro` | ✏️ | 重构: Hero + 特性卡片(Card) + Pipeline Steps + 快速入口(Button) + GitHub 仓库(GithubCard) |
| 024 | `src/pages/generate/index.astro` | 🆕 | **核心** 四步 Pipeline 生成页 |
| 025 | `src/pages/gallery/index.astro` | 🆕 | 画廊: 示例 SVG + 历史作品 |
| 026 | `src/pages/playground/index.astro` | 🆕 | ELK 布局实验场 |
| 027 | `src/pages/about/index.astro` | ✏️ | +技术架构说明 +ToolSection 展示技术栈 |
| 028 | `src/pages/api/topology.ts` | 🆕 | POST: text → topology JSON |
| 029 | `src/pages/api/layout.ts` | 🆕 | POST: topology → ELK layouted |
| 030 | `src/pages/api/beautify.ts` | 🆕 | POST: layouted → NanoBanana SVG |
| 031 | `src/pages/api/export.ts` | 🆕 | POST: SVG → PNG/PDF |
| 032 | `src/pages/api/validate.ts` | 🆕 | POST: SVG 语法校验 |
| 033 | `src/pages/api/models.ts` | 🆕 | GET: 可用模型列表 |
| 034 | `src/pages/docs/[...id].astro` | ✏️ | 确保兼容新增 docs |
| 035 | `src/pages/projects/index.astro` | ✏️ | +SVGFigure 项目卡片 |
| 036 | `src/pages/404.astro` | 📌 | 不动 |

### D. Astro 布局 (4 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 037 | `src/layouts/BaseLayout.astro` | ✏️ | +elkjs CDN fallback in head (可选) |
| 038 | `src/layouts/GenerateLayout.astro` | 🆕 | 生成页三栏面板布局 |
| 039 | `src/layouts/FullWidthLayout.astro` | 🆕 | Playground/Gallery 全宽布局 |
| 040 | `src/layouts/ContentLayout.astro` | 📌 | 不动 |

### E. 组件 — Pipeline 核心 (16 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 041 | `src/components/pipeline/PipelineSteps.astro` | 🆕 | 四步指示器 (Steps) |
| 042 | `src/components/pipeline/TextInput.astro` | 🆕 | Method text 输入区 (Card + textarea) |
| 043 | `src/components/pipeline/TopologyPreview.astro` | 🆕 | 拓扑 JSON 预览 (Collapse + Spoiler) |
| 044 | `src/components/pipeline/ElkPreview.astro` | 🆕 | ELK 骨架 SVG 预览 (Svg + MediumZoom) |
| 045 | `src/components/pipeline/SvgPreview.astro` | 🆕 | 最终 SVG 预览 (MediumZoom) |
| 046 | `src/components/pipeline/GenerateButton.astro` | 🆕 | 生成按钮 (Button) |
| 047 | `src/components/pipeline/ExportPanel.astro` | 🆕 | 导出面板 (Button 组) |
| 048 | `src/components/pipeline/StatusBar.astro` | 🆕 | 状态栏 (Aside) |
| 049 | `src/components/pipeline/JsonEditor.astro` | 🆕 | JSON 编辑器 |
| 050 | `src/components/pipeline/ElkOptions.astro` | 🆕 | ELK 参数面板 (Tabs + TabItem) |
| 051 | `src/components/pipeline/HistoryList.astro` | 🆕 | 生成历史 (Timeline) |
| 052 | `src/components/pipeline/ErrorDisplay.astro` | 🆕 | 错误展示 (Aside danger) |
| 053 | `src/components/pipeline/PromptPreview.astro` | 🆕 | NanoBanana prompt 预览 (Spoiler) |
| 054 | `src/components/pipeline/ScaffoldView.astro` | 🆕 | JSON 脚手架查看 (Collapse) |
| 055 | `src/components/pipeline/CompareView.astro` | 🆕 | 骨架 vs 最终对比 |
| 056 | `src/components/pipeline/ModelSelector.astro` | 🆕 | AI 模型选择器 (Tabs) |

### F. 组件 — Gallery & 展示 (8 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 057 | `src/components/gallery/FigureCard.astro` | 🆕 | 图片卡片 (Card + MediumZoom) |
| 058 | `src/components/gallery/FigureGrid.astro` | 🆕 | 网格布局 (CardList) |
| 059 | `src/components/gallery/FilterBar.astro` | 🆕 | 分类筛选 (Tabs) |
| 060 | `src/components/gallery/ExampleList.astro` | 🆕 | 内置示例列表 |
| 061 | `src/components/gallery/StylePicker.astro` | 🆕 | 风格选择 (Tabs) |
| 062 | `src/components/gallery/RepoCards.astro` | 🆕 | GitHub 仓库 (GithubCard) |
| 063 | `src/components/gallery/LinkCards.astro` | 🆕 | 外部链接 (LinkPreview) |
| 064 | `src/components/gallery/ZoomableSvg.astro` | 🆕 | 可缩放 SVG (MediumZoom) |

### G. 组件 — 首页 & 通用 (10 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 065 | `src/components/home/Hero.astro` | 🆕 | 首页 Hero (Button CTA) |
| 066 | `src/components/home/FeatureCards.astro` | 🆕 | 四大特性卡片 (Card 网格) |
| 067 | `src/components/home/ArchDiagram.astro` | 🆕 | 架构流程图 (内联 SVG) |
| 068 | `src/components/home/QuickStart.astro` | 🆕 | 快速开始 (Steps) |
| 069 | `src/components/home/TechStack.astro` | 🆕 | 技术栈 (Icon + Label) |
| 070 | `src/components/home/Section.astro` | ✏️ | +variant prop 支持 full-width |
| 071 | `src/components/common/SvgRenderer.astro` | 🆕 | 通用 SVG 安全渲染器 |
| 072 | `src/components/common/LoadingSpinner.astro` | 🆕 | 加载动画 |
| 073 | `src/components/common/CopyButton.astro` | 🆕 | 复制按钮 (Button + toast) |
| 074 | `src/components/common/CodeBlock.astro` | 🆕 | JSON/SVG 代码块 |

### H. TS 库 — ELK 布局引擎 (10 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 075 | `src/lib/elk/index.ts` | 🆕 | ELK 模块入口 |
| 076 | `src/lib/elk/layout.ts` | 🆕 | **核心** elkjs 布局封装 |
| 077 | `src/lib/elk/algorithms.ts` | 🆕 | 算法选项 (layered/mrtree/force/stress/radial) |
| 078 | `src/lib/elk/to-svg.ts` | 🆕 | ELK → 骨架 SVG |
| 079 | `src/lib/elk/to-scaffold.ts` | 🆕 | ELK → NanoBanana 脚手架 |
| 080 | `src/lib/elk/presets.ts` | 🆕 | 预设参数 (学术图/流程图/架构图/神经网络) |
| 081 | `src/lib/elk/types.ts` | 🆕 | TS 类型定义 |
| 082 | `src/lib/elk/validator.ts` | 🆕 | 拓扑 JSON 验证 (zod) |
| 083 | `src/lib/elk/examples.ts` | 🆕 | 内置示例拓扑 |
| 084 | `src/lib/elk/constants.ts` | 🆕 | ELK 常量 |

### I. TS 库 — Pipeline 编排 (10 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 085 | `src/lib/pipeline/index.ts` | 🆕 | Pipeline 编排入口 |
| 086 | `src/lib/pipeline/topology-gen.ts` | 🆕 | LLM 拓扑生成客户端 |
| 087 | `src/lib/pipeline/nanobanana.ts` | 🆕 | NanoBanana API 封装 |
| 088 | `src/lib/pipeline/svg-validator.ts` | 🆕 | SVG 验证 (DOMParser) |
| 089 | `src/lib/pipeline/svg-scaler.ts` | 🆕 | SVG 缩放变换 |
| 090 | `src/lib/pipeline/svg-optimizer.ts` | 🆕 | SVG 优化 |
| 091 | `src/lib/pipeline/export.ts` | 🆕 | 导出逻辑 (SVG/PNG/PDF) |
| 092 | `src/lib/pipeline/types.ts` | 🆕 | Pipeline 类型定义 |
| 093 | `src/lib/pipeline/prompts.ts` | 🆕 | LLM prompt 模板集 |
| 094 | `src/lib/pipeline/cache.ts` | 🆕 | 结果缓存 |

### J. TS 库 — AI Client (仿 skynetCheapBuy) (3 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 095 | `src/lib/ai/client.ts` | 🆕 | 前端 AI API 客户端 (SSE streaming) |
| 096 | `src/lib/ai/types.ts` | 🆕 | AI 请求/响应类型 |
| 097 | `src/lib/ai/models.ts` | 🆕 | 可用模型配置 |

### K. 内容集合 & 文档 (9 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 098 | `src/content.config.ts` | ✏️ | +gallery 集合 schema |
| 099 | `src/content/docs/svgfigure/getting-started.mdx` | 🆕 | 快速开始 |
| 100 | `src/content/docs/svgfigure/elkjs-guide.mdx` | 🆕 | ELK.js 使用指南 |
| 101 | `src/content/docs/svgfigure/pipeline-overview.mdx` | 🆕 | Pipeline 架构文档 |
| 102 | `src/content/docs/svgfigure/nanobanana-integration.mdx` | 🆕 | NanoBanana 集成 |
| 103 | `src/content/docs/svgfigure/api-reference.mdx` | 🆕 | API 接口文档 |
| 104 | `src/content/docs/svgfigure/ai-engine.mdx` | 🆕 | AI Engine 多 Provider 文档 |
| 105 | `src/content/gallery/transformer.json` | 🆕 | 示例: Transformer |
| 106 | `src/content/gallery/diffusion.json` | 🆕 | 示例: Diffusion |

### L. 静态资源 & 部署 (5 files)

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 107 | `public/examples/transformer.svg` | 🆕 | 示例 SVG |
| 108 | `public/examples/diffusion.svg` | 🆕 | 示例 SVG |
| 109 | `public/examples/gan.svg` | 🆕 | 示例 SVG |
| 110 | `src/site.config.ts` | ✏️ | +header menu: Generate/Gallery/Playground |
| 111 | `Dockerfile` | 🆕 | 多阶段构建 |

---

### 统计汇总

| 操作 | 数量 |
|------|------|
| 🆕 新增 | 80 |
| ✏️ 修改 | 20 |
| 📌 不动 | 3 |
| 🗑️ 删除 | 0 |
| **合计** | **111** |

---

## 4. 修改文件 diff 要点

### 4.1 package.json

```diff
  "dependencies": {
+   "elkjs": "^0.9.3",
+   "@google/generative-ai": "^0.21.0",
+   "zod": "^3.23.0",
+   "openai": "^4.70.0",
    "@astrojs/check": "^0.9.6",
    "@astrojs/rss": "^4.0.14",
    "@astrojs/vercel": "^9.0.2",
    // ... 保留所有现有依赖不变
  }
```

### 4.2 astro.config.ts

```diff
  export default defineConfig({
    site: 'https://astro-pure.js.org',
    // ... 保留全部现有配置
+   vite: {
+     optimizeDeps: {
+       include: ['elkjs/lib/elk.bundled.js']
+     }
+   }
  })
```

### 4.3 tsconfig.json

```diff
  {
    "compilerOptions": {
+     "paths": {
+       "@elk/*": ["./src/lib/elk/*"],
+       "@pipeline/*": ["./src/lib/pipeline/*"],
+       "@ai/*": ["./src/lib/ai/*"]
+     }
    }
  }
```

### 4.4 src/site.config.ts

```diff
  header: {
    menu: [
      { title: 'Blog', link: '/blog' },
      { title: 'Docs', link: '/docs' },
+     { title: 'Generate', link: '/generate' },
+     { title: 'Gallery', link: '/gallery' },
+     { title: 'Playground', link: '/playground' },
      { title: 'Projects', link: '/projects' },
-     { title: 'Links', link: '/links' },
      { title: 'About', link: '/about' }
    ],
  },
```

### 4.5 server.py (关键改动，保留现有全部代码)

```diff
+ import sys
+ sys.path.insert(0, str(BASE_DIR))
+ from backend.config import get_settings
+ from backend.ai_engine import AIEngine
+ from backend.schemas import TopologyRequest, BeautifyRequest, ExportRequest
+ from backend.pipeline.topology_gen import generate_topology
+ from backend.pipeline.nanobanana_bridge import beautify_with_nanobanana
+ from backend.pipeline.svg_validator import validate_svg
+
+ # 初始化 AI Engine (仿 skynetCheapBuy)
+ app_settings = get_settings()
+ ai_engine = AIEngine()
+
+ @app.post('/api/topology')
+ async def api_topology(req: TopologyRequest):
+     result = await generate_topology(ai_engine, req.text, req.model)
+     return JSONResponse(content=result)
+
+ @app.post('/api/beautify')
+ async def api_beautify(req: BeautifyRequest):
+     result = await beautify_with_nanobanana(ai_engine, req.layouted, req.scaffold)
+     return JSONResponse(content=result)
+
+ @app.post('/api/export')
+ async def api_export(req: ExportRequest): ...
+
+ @app.post('/api/validate')
+ async def api_validate(req: dict): ...
+
+ @app.get('/api/models')
+ async def api_models():
+     return JSONResponse(content=app_settings.AVAILABLE_MODELS)
+
  # ============ 以下保留全部现有代码 ============
  # Job / autofigure 相关端点完全不动
```

### 4.6 requirements.txt

```diff
+ # AI Providers (仿 skynetCheapBuy)
+ openai>=1.50.0
+ anthropic>=0.37.0
+ google-generativeai>=0.8.0
+ httpx>=0.27.0
+ pydantic-settings>=2.5.0
+ # SVG Processing
+ lxml>=5.0.0
+ cairosvg>=2.7.0
+ Pillow>=10.0.0
  # 保留全部现有依赖
```

### 4.7 src/content.config.ts

```diff
+ const gallery = defineCollection({
+   loader: glob({ base: './src/content/gallery', pattern: '**/*.json' }),
+   schema: () =>
+     z.object({
+       title: z.string(),
+       description: z.string(),
+       svg_path: z.string(),
+       topology: z.record(z.unknown()),
+       tags: z.array(z.string()).default([]),
+       created: z.coerce.date(),
+       algorithm: z.enum(['layered', 'mrtree', 'force', 'stress', 'radial']).default('layered')
+     })
+ })
+
- export const collections = { blog, docs }
+ export const collections = { blog, docs, gallery }
```

---

## 5. AI Engine 详细设计 (仿 skynetCheapBuy/app/core/ai_engine.py)

### backend/ai_engine.py 核心结构

```python
"""
AI Engine — 多 Provider 抽象工厂
完全参照 skynetCheapBuy/app/core/ai_engine.py 的设计模式

参考: https://github.com/dylanyunlon/skynetCheapBuy/blob/main/app/core/ai_engine.py
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Optional
import openai
from anthropic import AsyncAnthropic
import google.generativeai as genai

# ====== 辅助函数 (与 skynetCheapBuy 一致) ======
def is_claude_model(model: str) -> bool:
    return model.lower().startswith(("claude-", "claude_"))

def is_openai_model(model: str) -> bool:
    return model.lower().startswith(("gpt-", "o1-", "o3-"))

def is_gemini_model(model: str) -> bool:
    return model.lower().startswith("gemini")

# ====== Provider 抽象基类 ======
class AIProvider(ABC):
    @abstractmethod
    async def get_completion(self, messages, model, **kwargs) -> Dict[str, Any]: ...
    @abstractmethod
    async def stream_completion(self, messages, model, **kwargs) -> AsyncGenerator: ...

# ====== 具体 Provider ======
class OpenAIProvider(AIProvider):
    """OpenAI (gpt-*) — openai.AsyncOpenAI"""
    ...

class AnthropicProvider(AIProvider):
    """Anthropic Claude (原生 SDK) — AsyncAnthropic"""
    ...

class GoogleProvider(AIProvider):
    """Google Gemini — genai.GenerativeModel"""
    ...

class ClaudeCompatibleProvider(AIProvider):
    """Claude via /v1/messages (兼容接口) — httpx"""
    ...

# ====== Engine 工厂 ======
class AIEngine:
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self):
        self.providers = {}
        self._init_providers()

    def _get_provider(self, model, api_key=None, api_url=None) -> AIProvider:
        """根据模型名自动路由到对应 Provider (与 skynetCheapBuy 逻辑一致)"""
        if api_key:
            if is_claude_model(model):
                return ClaudeCompatibleProvider(api_key, api_url)
            elif is_openai_model(model):
                return OpenAIProvider(api_key, api_url)
            elif is_gemini_model(model):
                return GoogleProvider(api_key)
            else:
                return OpenAIProvider(api_key, api_url)  # 默认 OpenAI 兼容
        # 使用默认 Provider
        ...

    async def generate(self, prompt, model=None, **kwargs) -> Dict[str, Any]: ...
    async def stream(self, prompt, model=None, **kwargs) -> AsyncGenerator: ...
```

---

## 6. ELK.js 核心用法 (基于 kieler/elkjs)

### 输入格式 (LLM 输出的拓扑 JSON, 零坐标)

```json
{
  "id": "root",
  "layoutOptions": {
    "elk.algorithm": "layered",
    "elk.direction": "DOWN",
    "elk.layered.spacing.nodeNodeBetweenLayers": "100",
    "elk.spacing.nodeNode": "80"
  },
  "children": [
    { "id": "input", "width": 150, "height": 50, "labels": [{"text": "Input"}] },
    { "id": "encoder", "width": 150, "height": 50, "labels": [{"text": "Encoder"}] },
    { "id": "decoder", "width": 150, "height": 50, "labels": [{"text": "Decoder"}] },
    { "id": "output", "width": 150, "height": 50, "labels": [{"text": "Output"}] }
  ],
  "edges": [
    { "id": "e1", "sources": ["input"], "targets": ["encoder"] },
    { "id": "e2", "sources": ["encoder"], "targets": ["decoder"] },
    { "id": "e3", "sources": ["decoder"], "targets": ["output"] }
  ]
}
```

### ELK 布局调用 (src/lib/elk/layout.ts)

```typescript
import ELK from 'elkjs/lib/elk.bundled.js'

const elk = new ELK()

export async function layoutGraph(topology: ElkGraph): Promise<LayoutResult> {
  const layouted = await elk.layout(topology)
  // layouted.children[i] 现在有 x, y 坐标
  // layouted.edges[i] 有 sections[].startPoint/endPoint/bendPoints
  return layouted
}
```

### 布局结果 → NanoBanana 脚手架 (src/lib/elk/to-scaffold.ts)

```typescript
export function toScaffold(layouted: ElkGraph): NanoBananaScaffold {
  return {
    figure_type: "academic_architecture",
    canvas: {
      width: computeCanvasWidth(layouted),
      height: computeCanvasHeight(layouted)
    },
    elements: layouted.children.map(node => ({
      id: node.id,
      type: "box",
      label: node.labels?.[0]?.text || node.id,
      x: node.x, y: node.y,
      width: node.width, height: node.height,
      style: "rounded_rect",
      fill: getColorByType(node)
    })),
    connections: layouted.edges.map(edge => ({
      from: edge.sources[0],
      to: edge.targets[0],
      style: "arrow",
      points: extractRoutePoints(edge)
    }))
  }
}
```

---

## 7. 部署命令

### 开发环境

```bash
# 1. 克隆项目
git clone https://github.com/dylanyunlon/astro-svgfigure.git
cd astro-svgfigure

# 2. 安装前端依赖
bun install
bun add elkjs @google/generative-ai zod openai

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env:
#   GEMINI_API_KEY=your-gemini-key
#   OPENAI_API_KEY=your-openai-key (可选)
#   ANTHROPIC_API_KEY=your-anthropic-key (可选)
#   OPENAI_API_BASE=https://api.openai.com/v1

# 5. 启动
bun run dev          # 终端 1: Astro (port 4321)
python server.py     # 终端 2: FastAPI (port 8000)
```

### Docker 部署

```bash
docker build -t astro-svgfigure .
docker run -p 4321:4321 -p 8000:8000 \
  -e GEMINI_API_KEY=your-key \
  -e OPENAI_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  astro-svgfigure
```

### Vercel 部署

```bash
npx vercel --prod
# Python 后端: Railway / Fly.io / 自建
```

---

## 8. 全部 GitHub 背书 (20 个)

| # | 项目 | Step | 用途 |
|---|------|------|------|
| 1 | [ResearAI/AutoFigure](https://github.com/ResearAI/AutoFigure) | 1 | LLM 学术图生成 (ICLR 2026) |
| 2 | [ResearAI/AutoFigure-Edit](https://github.com/ResearAI/AutoFigure-Edit) | 1 | 逆向流程参考 |
| 3 | [dylanyunlon/skynetCheapBuy](https://github.com/dylanyunlon/skynetCheapBuy) | 后端 | **AI Engine 多 Provider 参考** |
| 4 | [kieler/elkjs](https://github.com/kieler/elkjs) | 2 | **核心布局引擎** |
| 5 | [EmilStenstrom/elkjs-svg](https://github.com/EmilStenstrom/elkjs-svg) | 2 | ELK → SVG 参考 |
| 6 | [xyflow/xyflow](https://github.com/xyflow/xyflow) | 2 | React Flow + ELK 参考 |
| 7 | [cytoscape/cytoscape.js-elk](https://github.com/cytoscape/cytoscape.js-elk) | 2 | ELK 算法适配 |
| 8 | [davidthings/hdelk](https://github.com/davidthings/hdelk) | 2 | ELK + SVG.js |
| 9 | [eclipse.org/elk](https://www.eclipse.org/elk/reference.html) | 2 | 算法参考 |
| 10 | [rtsys.informatik.uni-kiel.de/elklive](https://rtsys.informatik.uni-kiel.de/elklive/) | 2 | ELK 在线 Demo |
| 11 | [gemini-cli-extensions/nanobanana](https://github.com/gemini-cli-extensions/nanobanana) | 3 | NanoBanana CLI |
| 12 | [ZeroLu/awesome-nanobanana-pro](https://github.com/ZeroLu/awesome-nanobanana-pro) | 3 | Prompt 工程 |
| 13 | [aaronkwhite/nanobanana-studio-web](https://github.com/aaronkwhite/nanobanana-studio-web) | 3 | 自托管生成 |
| 14 | [GeminiGenAI/Free-Nano-Banana-Pro-API](https://github.com/GeminiGenAI/Free-Nano-Banana-Pro-API-Ultimate-AI-Image-Generator) | 3 | API 参考 |
| 15 | [cworld1/astro-theme-pure](https://github.com/cworld1/astro-theme-pure) | 4 | **主题源码** |
| 16 | [withastro/astro](https://github.com/withastro/astro) | 4 | Astro 框架 |
| 17 | [natemoo-re/astro-icon](https://github.com/natemoo-re/astro-icon) | 4 | SVG 内联 |
| 18 | [svgdotjs/svg.js](https://github.com/svgdotjs/svg.js) | 2/3 | SVG 操作库 |
| 19 | [openai/openai-python](https://github.com/openai/openai-python) | 后端 | OpenAI SDK |
| 20 | [anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) | 后端 | Anthropic SDK |

---

## 9. 正向 vs 逆向流程对比

| 维度 | 逆向 (autofigure-edit) | 正向 (本项目) |
|------|----------------------|-------------|
| 输入 | 文本 → Gemini 生成图片 | 文本 → Gemini 生成拓扑 JSON |
| 分割 | SAM3 逆向分割图片 | ELK.js 正向约束布局 |
| 去背景 | RMBG2 去背景 | 不需要 (纯 SVG) |
| SVG | Gemini 看 samed.png 生成 | NanoBanana 看 JSON 脚手架生成 |
| 修复 | lxml + LLM 修复 | lxml + LLM 修复 (复用) |
| 替换 | 图标替换到占位符 | 不需要 (一步到位) |
| 优势 | 能处理已有图片 | 更快、更精确、无需 GPU |
| 依赖 | SAM3, RMBG2 (大模型) | elkjs (纯 JS, 43KB) |

---

## 10. 下一步执行顺序

1. **Phase 0**: 配置文件更新 — 001~010
2. **Phase 1**: AI Engine 后端 — 013~022 (仿 skynetCheapBuy)
3. **Phase 2**: ELK TS 库 — 075~084
4. **Phase 3**: Pipeline TS 库 — 085~097
5. **Phase 4**: 前端组件 — 041~074
6. **Phase 5**: 页面路由 — 023~036
7. **Phase 6**: 布局模板 — 037~040
8. **Phase 7**: 内容 & 文档 — 098~106
9. **Phase 8**: 静态资源 & 部署 — 107~111
10. **Phase 9**: 集成测试 + diff 校验


04更新：

# plan.md — astro-svgfigure v2: ELK.js + NanoBanana 正向 Pipeline

> **核心理念**: AutoFigure-Edit 是逆向流程（图→SAM3分割→SVG），我们做正向流程（文本→拓扑JSON→ELK布局→NanoBanana SVG）
> **技术栈**: Astro 5 + astro-pure 主题 + UnoCSS + ELK.js + Gemini NanoBanana + Python 后端
> **关键洞察**: NanoBanana 生成的图片如此完美以至于让拓扑学家像个小丑——ELK.js 只负责骨架坐标，最终以 JSON 脚手架向 NanoBanana 请求神经网络级 SVG
> **AI请求模式**: 参照 skynetCheapBuy 的 `app/core/ai_engine.py` 多 Provider 抽象工厂模式

---

## 0. 项目现状（基于 main 分支）

```
astro-svgfigure/  (base: 3f5faab)
├── packages/pure/           # astro-pure 主题包（workspace link）
│   ├── components/basic/    # Header, Footer, ThemeProvider
│   ├── components/user/     # Button, Card, Collapse, Aside, Steps, Tabs, Timeline, Icon, Spoiler, Svg, Label
│   ├── components/advanced/ # GithubCard, LinkPreview, MediumZoom, Quote, QRCode
│   ├── components/pages/    # Hero, PostPreview, Paginator, TOC, BackToTop, PFSearch
│   ├── utils/               # clsx, toast, theme, date, reading-time
│   └── types/               # TS 类型
├── src/
│   ├── components/home/     # Section, SkillLayout, ProjectCard
│   ├── content/blog/        # 博客文章
│   ├── content/docs/        # 文档
│   ├── layouts/             # BaseLayout, ContentLayout, BlogPost, CommonPage
│   ├── pages/               # index, blog, docs, projects, links, about, search, tags, 404
│   ├── assets/styles/       # global.css, app.css
│   └── site.config.ts
├── autofigure2.py           # 逆向脚本参考 (2793行, 不动)
├── server.py                # FastAPI 后端 (522行)
├── astro.config.ts          # vercel adapter + server output
├── package.json             # astro 5 + astro-pure
└── requirements.txt
```

---

## 1. 四步正向 Pipeline 架构

```
[用户输入 method text]
       │
       ▼
Step 1: LLM 拓扑推理 ──────── GitHub: ResearAI/AutoFigure, dylanyunlon/skynetCheapBuy
       │ topology.json (零坐标 ELK graph)
       ▼
Step 2: ELK.js 约束布局 ───── GitHub: kieler/elkjs, xyflow/xyflow, EmilStenstrom/elkjs-svg
       │ layouted.json (精确像素 x,y,w,h)
       ▼
Step 3: NanoBanana 美化 ───── GitHub: gemini-cli-extensions/nanobanana, ZeroLu/awesome-nanobanana-pro
       │ final.svg (神经网络级学术图)
       ▼
Step 4: Astro 前端展示 ────── GitHub: cworld1/astro-theme-pure, withastro/astro
```

后端 AI Engine (仿 skynetCheapBuy):
- AIProvider 基类 → OpenAIProvider / AnthropicProvider / GoogleProvider / ClaudeCompatibleProvider
- AIEngine._get_provider(model) 自动路由

---

## 2. 111 文件变更清单

🆕=新增 ✏️=修改 📌=不动

### A. 根目录配置 (10)

| # | 文件 | Op | 说明 |
|---|------|---|------|
| 001 | package.json | ✏️ | +elkjs +@google/generative-ai +zod +openai |
| 002 | astro.config.ts | ✏️ | +vite.optimizeDeps elkjs |
| 003 | uno.config.ts | ✏️ | +pipeline shortcuts |
| 004 | tsconfig.json | ✏️ | +paths @elk @pipeline @ai |
| 005 | requirements.txt | ✏️ | +google-generativeai +openai +anthropic +lxml +cairosvg +httpx +pydantic-settings |
| 006 | plan.md | ✏️ | 本文件 |
| 007 | tree.txt | ✏️ | 更新结构 |
| 008 | README.md | ✏️ | 更新说明 |
| 009 | bun.lock | ✏️ | 自动 |
| 010 | .env.example | 🆕 | API keys |

### B. Python 后端 AI Engine (12)

| # | 文件 | Op | 说明 |
|---|------|---|------|
| 011 | server.py | ✏️ | +/api/topology +/api/beautify +/api/export +/api/validate +/api/models |
| 012 | autofigure2.py | 📌 | 不动 |
| 013 | backend/__init__.py | 🆕 | 包入口 |
| 014 | backend/config.py | 🆕 | Settings (pydantic-settings) 仿 skynetCheapBuy |
| 015 | backend/ai_engine.py | 🆕 | 多Provider工厂 |
| 016 | backend/schemas.py | 🆕 | Pydantic models |
| 017 | backend/pipeline/__init__.py | 🆕 | 子包 |
| 018 | backend/pipeline/topology_gen.py | 🆕 | 拓扑生成 |
| 019 | backend/pipeline/nanobanana_bridge.py | 🆕 | NanoBanana桥接 |
| 020 | backend/pipeline/svg_validator.py | 🆕 | SVG验证+LLM修复 |
| 021 | backend/pipeline/scaffold_builder.py | 🆕 | ELK→脚手架 |
| 022 | backend/pipeline/svg_scaler.py | 🆕 | SVG缩放 |

### C. Astro 页面 (14)

| # | 文件 | Op | 说明 |
|---|------|---|------|
| 023 | src/pages/index.astro | ✏️ | Hero+特性+Pipeline Steps |
| 024 | src/pages/generate/index.astro | 🆕 | 核心Pipeline生成页 |
| 025 | src/pages/gallery/index.astro | 🆕 | 画廊 |
| 026 | src/pages/playground/index.astro | 🆕 | ELK实验场 |
| 027 | src/pages/about/index.astro | ✏️ | +技术架构 |
| 028 | src/pages/api/topology.ts | 🆕 | POST text→topology |
| 029 | src/pages/api/layout.ts | 🆕 | POST topology→ELK |
| 030 | src/pages/api/beautify.ts | 🆕 | POST layouted→SVG |
| 031 | src/pages/api/export.ts | 🆕 | POST SVG→PNG/PDF |
| 032 | src/pages/api/validate.ts | 🆕 | POST SVG校验 |
| 033 | src/pages/api/models.ts | 🆕 | GET 模型列表 |
| 034 | src/pages/docs/[...id].astro | ✏️ | 兼容新docs |
| 035 | src/pages/projects/index.astro | ✏️ | +项目卡片 |
| 036 | src/pages/404.astro | 📌 | 不动 |

### D. 布局 (4)

| # | 文件 | Op | 说明 |
|---|------|---|------|
| 037 | src/layouts/BaseLayout.astro | ✏️ | 可选elkjs CDN |
| 038 | src/layouts/GenerateLayout.astro | 🆕 | 三栏面板 |
| 039 | src/layouts/FullWidthLayout.astro | 🆕 | 全宽 |
| 040 | src/layouts/ContentLayout.astro | 📌 | 不动 |

### E. Pipeline组件 (16)

| # | 文件 | Op |
|---|------|---|
| 041 | src/components/pipeline/PipelineSteps.astro | 🆕 |
| 042 | src/components/pipeline/TextInput.astro | 🆕 |
| 043 | src/components/pipeline/TopologyPreview.astro | 🆕 |
| 044 | src/components/pipeline/ElkPreview.astro | 🆕 |
| 045 | src/components/pipeline/SvgPreview.astro | 🆕 |
| 046 | src/components/pipeline/GenerateButton.astro | 🆕 |
| 047 | src/components/pipeline/ExportPanel.astro | 🆕 |
| 048 | src/components/pipeline/StatusBar.astro | 🆕 |
| 049 | src/components/pipeline/JsonEditor.astro | 🆕 |
| 050 | src/components/pipeline/ElkOptions.astro | 🆕 |
| 051 | src/components/pipeline/HistoryList.astro | 🆕 |
| 052 | src/components/pipeline/ErrorDisplay.astro | 🆕 |
| 053 | src/components/pipeline/PromptPreview.astro | 🆕 |
| 054 | src/components/pipeline/ScaffoldView.astro | 🆕 |
| 055 | src/components/pipeline/CompareView.astro | 🆕 |
| 056 | src/components/pipeline/ModelSelector.astro | 🆕 |

### F. Gallery组件 (8)

| # | 文件 | Op |
|---|------|---|
| 057-064 | src/components/gallery/*.astro | 🆕 |



### G. 首页&通用组件 (10)

| # | 文件 | Op |
|---|------|---|
| 065-074 | src/components/home/*.astro + src/components/common/*.astro | 🆕/✏️ |

### H. ELK TS库 (10)

| # | 文件 | Op |
|---|------|---|
| 075 | src/lib/elk/index.ts | 🆕 |
| 076 | src/lib/elk/layout.ts | 🆕 |
| 077 | src/lib/elk/algorithms.ts | 🆕 |
| 078 | src/lib/elk/to-svg.ts | 🆕 |
| 079 | src/lib/elk/to-scaffold.ts | 🆕 |
| 080 | src/lib/elk/presets.ts | 🆕 |
| 081 | src/lib/elk/types.ts | 🆕 |
| 082 | src/lib/elk/validator.ts | 🆕 |
| 083 | src/lib/elk/examples.ts | 🆕 |
| 084 | src/lib/elk/constants.ts | 🆕 |

### I. Pipeline TS库 (10)

| # | 文件 | Op |
|---|------|---|
| 085-094 | src/lib/pipeline/*.ts | 🆕 |

### J. AI Client (3)

| # | 文件 | Op |
|---|------|---|
| 095-097 | src/lib/ai/*.ts | 🆕 |

### K. 内容&文档 (9)

| # | 文件 | Op |
|---|------|---|
| 098 | src/content.config.ts | ✏️ |
| 099-106 | src/content/docs/svgfigure/*.mdx + src/content/gallery/*.json | 🆕 |

### L. 静态资源&部署 (5)

| # | 文件 | Op |
|---|------|---|
| 107-109 | public/examples/*.svg | 🆕 |
| 110 | src/site.config.ts | ✏️ |
| 111 | Dockerfile | 🆕 |

统计: 🆕80 ✏️20 📌3 = 111 total

---

## 3. 部署命令

```bash
# 开发
git clone https://github.com/dylanyunlon/astro-svgfigure.git && cd astro-svgfigure
bun install && pip install -r requirements.txt
cp .env.example .env  # 填入 API keys
bun run dev            # Astro :4321
python server.py       # FastAPI :8000

# Docker
docker build -t astro-svgfigure .
docker run -p 4321:4321 -p 8000:8000 -e GEMINI_API_KEY=xxx astro-svgfigure

# Vercel (前端) + Railway/Fly.io (后端)
npx vercel --prod
```

---

## 4. GitHub 背书 (20)

| Step | 项目 |
|------|------|
| 1 拓扑 | ResearAI/AutoFigure, ResearAI/AutoFigure-Edit, dylanyunlon/skynetCheapBuy |
| 2 ELK | kieler/elkjs, EmilStenstrom/elkjs-svg, xyflow/xyflow, cytoscape/cytoscape.js-elk, davidthings/hdelk, eclipse.org/elk, elklive |
| 3 NanoBanana | gemini-cli-extensions/nanobanana, ZeroLu/awesome-nanobanana-pro, aaronkwhite/nanobanana-studio-web, GeminiGenAI/Free-Nano-Banana-Pro-API |
| 4 前端 | cworld1/astro-theme-pure, withastro/astro, natemoo-re/astro-icon, svgdotjs/svg.js |
| 后端SDK | openai/openai-python, anthropics/anthropic-sdk-python |

---

## 5. 并行开发分工

**Claude (本轮)**: Phase 0 配置 (001-010) + ELK核心 (076,081,084)

  git commit -m  "完成的 10 个任务:,111文件清单+并行分工,.env.example,GEMINI/OPENAI/ANTHROPIC API keys 模板,package.json, >+elkjs, @google/generative-ai, zod, openai, 4,astro.config.ts, >+vite.optimizeDeps: elkjs/lib/elk.bundled.js, 5,tsconfig.json, >+paths: @elk/<em>, @pipeline/</em>, @ai/*, 6,+anthropic, httpx, pydantic-settings, google-generativeai, 7,src/site.config.ts, >+Generate/Gallery/Playground 菜单, 8,src/lib/elk/types.ts,ElkGraph, LayoutResult, NanoBananaScaffold, PipelineState 等完整类型, 9,src/lib/elk/constants.ts,9种算法配置, 默认布局选项, SVG骨架常量, 配色方案, 10,src/lib/elk/layout.ts, + ,presets.ts,核心,: layoutGraph(), layoutWithPreset(), quickLayout() + 7个预设 "

**Codex (并行)**: Phase 1 后端 AI Engine (013-022)

后续 Phase 按需分配。每个 Phase 完成后 `git diff` 校验。
