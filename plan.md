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

05更新：
# plan.md — astro-svgfigure v2: ELK.js + NanoBanana 正向 Pipeline

> **核心理念**: AutoFigure-Edit 是逆向流程（图→SAM3分割→SVG），我们做正向流程（文本→拓扑JSON→ELK布局→NanoBanana SVG）  
> **技术栈**: Astro 5 + astro-pure 主题 + UnoCSS + ELK.js + Gemini NanoBanana + Python 后端  
> **关键洞察**: NanoBanana 生成的图片如此完美以至于让拓扑学家像个小丑——ELK.js 只负责骨架坐标，最终以 JSON 脚手架向 NanoBanana 请求神经网络级 SVG

---

## 0. 项目现状（基于 main 分支最新提交树）

```
astro-svgfigure/
├── packages/pure/           # astro-pure 主题包（本地 link）
│   ├── components/basic/    # Header, Footer, ThemeProvider
│   ├── components/user/     # Button, Card, Collapse, Steps, Tabs, Icon, Label...
│   └── components/pages/    # Hero, PostPreview, Paginator, TOC...
├── backend/                 # Python AI Engine (Phase 1 已完成)
│   ├── ai_engine.py         # 多Provider工厂 (OpenAI/Claude/Gemini)
│   ├── config.py            # pydantic-settings
│   ├── schemas.py           # Pydantic models
│   └── pipeline/            # 5个pipeline模块
│       ├── topology_gen.py      # Step 1: LLM→拓扑
│       ├── scaffold_builder.py  # ELK→脚手架
│       ├── nanobanana_bridge.py # Step 3: NanoBanana美化
│       ├── svg_validator.py     # SVG验证+LLM修复
│       └── svg_scaler.py       # SVG缩放
├── src/
│   ├── components/          # Astro组件
│   │   ├── home/            # Section, ProjectCard, SkillLayout
│   │   ├── about/           # Substats, ToolSection
│   │   └── pipeline/        # 🆕 Pipeline组件 (Phase 2 新增)
│   ├── layouts/             # BaseLayout, BlogPost, CommonPage...
│   ├── lib/elk/             # ELK TS库 (Phase 0 已完成)
│   │   ├── layout.ts        # layoutGraph(), layoutWithPreset()
│   │   ├── presets.ts       # 7个预设
│   │   ├── types.ts         # 完整类型
│   │   └── constants.ts     # 算法配置
│   ├── pages/               # 路由
│   │   ├── index.astro      # ✏️ 首页 → 产品着陆页+Generate入口
│   │   ├── generate/        # 🆕 核心生成页
│   │   ├── blog/            # 博客 (保留)
│   │   └── api/             # 🆕 API端点
│   └── site.config.ts       # 导航 (已有Generate/Gallery/Playground)
├── server.py                # FastAPI 后端
├── autofigure2.py           # AutoFigure-Edit 逆向脚本 (保留不动)
└── package.json             # 依赖 (已有elkjs, @google/generative-ai, zod)
```

**已完成**: Phase 0 (配置+ELK核心 001-010) ✅ / Phase 1 (Backend AI Engine 011-022) ✅

---

## 1. 架构总览：四步正向 Pipeline

```
[用户输入 method text]
         │
         ▼
┌─── Step 1: LLM 拓扑推理 ───┐     GitHub: ResearAI/AutoFigure
│  Gemini 解析文本             │
│  输出 ELK JSON (零坐标)      │
└──────────┬──────────────────┘
           │ topology.json
           ▼
┌─── Step 2: ELK.js 约束布局 ─┐     GitHub: kieler/elkjs, xyflow/xyflow
│  elkjs 分层/树形/力导向算法   │            EmilStenstrom/elkjs-svg
│  计算每个节点 (x,y,w,h)     │
└──────────┬──────────────────┘
           │ layouted.json
           ▼
┌─── Step 3: NanoBanana 美化 ─┐     GitHub: gemini-cli-extensions/nanobanana
│  layouted.json → JSON 脚手架 │            ZeroLu/awesome-nanobanana-pro
│  Gemini 生成学术级 SVG       │
└──────────┬──────────────────┘
           │ final.svg
           ▼
┌─── Step 4: Astro 前端展示 ──┐     GitHub: cworld1/astro-theme-pure
│  astro-pure 主题组件渲染     │            withastro/astro
│  SVG 预览 + 导出             │
└─────────────────────────────┘
```

---

## 2. astro-pure 组件使用清单（不自创样式）

| 组件 | import 路径 | 页面用途 |
|------|------------|---------|
| Header | `astro-pure/components/basic` | 全站顶栏 |
| Footer | `astro-pure/components/basic` | 全站页脚 |
| ThemeProvider | `astro-pure/components/basic` | 暗色模式 |
| Button | `astro-pure/user` | 生成/导出/复制按钮 |
| Card | `astro-pure/user` | 特性卡片/步骤卡片/结果卡片 |
| Collapse | `astro-pure/user` | JSON 折叠展示 |
| Steps | `astro-pure/user` | Pipeline 步骤展示 |
| Tabs/TabItem | `astro-pure/user` | 输入/预览/JSON 标签切换 |
| Label | `astro-pure/user` | 状态标签 |
| Icon | `astro-pure/user` | 图标 |

---

## 3. Phase 2: 前端 Pipeline 集成 — Claude 本轮 10 任务

> **核心问题**: 用户 `bun dev` 后访问 :4321 看到博客首页，**Generate 功能不可见**  
> **解决**: 首页改为产品着陆页 + Generate CTA; 新建 /generate 页

| # | 文件 | Op | 说明 | GitHub 背书 |
|---|------|---|------|-------------|
| 023 | `src/pages/index.astro` | ✏️ | 首页→产品着陆页: Hero+Pipeline四步+CTA+博客降级 | cworld1/astro-theme-pure |
| 024 | `src/pages/generate/index.astro` | 🆕 | 核心Generate页: 输入→Pipeline→预览→导出 | withastro/astro |
| 025 | `src/components/pipeline/TextInput.astro` | 🆕 | 文本输入: textarea+示例+字数提示 | cworld1/astro-theme-pure |
| 026 | `src/components/pipeline/PipelineSteps.astro` | 🆕 | 四步进度条: 拓扑→ELK→NanoBanana→SVG | cworld1/astro-theme-pure |
| 027 | `src/components/pipeline/SvgPreview.astro` | 🆕 | SVG预览面板: 渲染+缩放+暗色 | svgdotjs/svg.js |
| 028 | `src/components/pipeline/ExportPanel.astro` | 🆕 | 导出: SVG/PNG下载+复制 | cworld1/astro-theme-pure |
| 029 | `src/pages/api/topology.ts` | 🆕 | POST text→topology (→Python backend) | withastro/astro API Routes |
| 030 | `src/pages/api/layout.ts` | 🆕 | POST topology→ELK布局 (前端elkjs) | kieler/elkjs |
| 031 | `src/pages/api/beautify.ts` | 🆕 | POST layouted→NanoBanana SVG (→Python) | gemini-cli-extensions/nanobanana |
| 032 | `src/layouts/GenerateLayout.astro` | 🆕 | Generate全宽布局 | cworld1/astro-theme-pure |

### 修改文件 diff 要点

**023 - src/pages/index.astro (✏️)**
- 保留: BaseLayout, Section, ProjectCard, astro-pure imports, Quote
- 新增: Pipeline Hero区域, 四步特性Card, Generate CTA Button
- 降级: 博客列表→底部section, 移除 Education/Certifications/Skills 占位
- 关键: import { Button, Card, Icon, Label } from 'astro-pure/user' 不变

---

## 4. Phase 3: Gallery + Playground + Docs — Codex 并行

| # | 文件 | Op | GitHub 背书 |
|---|------|---|-------------|
| 033 | `src/pages/gallery/index.astro` | 🆕 | cworld1/astro-theme-pure |
| 034 | `src/pages/playground/index.astro` | 🆕 | kieler/elkjs, xyflow/xyflow |
| 035 | `src/components/gallery/GalleryGrid.astro` | 🆕 | cworld1/astro-theme-pure |
| 036 | `src/components/gallery/GalleryCard.astro` | 🆕 | cworld1/astro-theme-pure |
| 037 | `src/components/pipeline/JsonEditor.astro` | 🆕 | cworld1/astro-theme-pure |
| 038 | `src/components/pipeline/ElkOptions.astro` | 🆕 | kieler/elkjs |
| 039 | `src/components/pipeline/TopologyPreview.astro` | 🆕 | EmilStenstrom/elkjs-svg |
| 040 | `src/pages/api/export.ts` | 🆕 | withastro/astro |
| 041 | `src/pages/api/validate.ts` | 🆕 | withastro/astro |
| 042 | `src/content/docs/svgfigure/getting-started.mdx` | 🆕 | withastro/astro |

---

## 5. 部署命令

```bash
# 开发 (双服务)
git clone https://github.com/dylanyunlon/astro-svgfigure.git && cd astro-svgfigure
bun install && pip install -r requirements.txt
cp .env.example .env  # 填入 GEMINI_API_KEY, OPENAI_API_KEY 等
bun run dev            # Astro :4321
python server.py       # FastAPI :8000 (新终端)

# Docker 一键启动
docker build -t astro-svgfigure .
docker run -p 4321:4321 -p 8000:8000 --env-file .env astro-svgfigure

# 生产部署: Vercel (前端) + Railway (后端)
npx vercel --prod
```

---

## 6. GitHub 背书汇总

| Step | 项目 | 用途 |
|------|------|------|
| 拓扑 | ResearAI/AutoFigure | LLM→学术图拓扑 |
| 拓扑 | ResearAI/AutoFigure-Edit | SAM3逆向参考 |
| 后端 | dylanyunlon/skynetCheapBuy | AI Engine多Provider |
| ELK | kieler/elkjs | 约束布局引擎 |
| ELK | EmilStenstrom/elkjs-svg | ELK→SVG渲染 |
| ELK | xyflow/xyflow | ReactFlow ELK集成 |
| ELK | eclipse/elk | ELK算法参考 |
| NanoBanana | gemini-cli-extensions/nanobanana | 核心 |
| NanoBanana | ZeroLu/awesome-nanobanana-pro | prompt工程 |
| NanoBanana | aaronkwhite/nanobanana-studio-web | Web版 |
| 前端 | cworld1/astro-theme-pure | Astro主题 |
| 前端 | withastro/astro | Astro框架 |
| SVG | svgdotjs/svg.js | SVG操作 |
| SDK | openai/openai-python | OpenAI |
| SDK | anthropics/anthropic-sdk-python | Claude |
| SDK | google/generative-ai-python | Gemini |

---

## 7. PR 说明

**Phase 2 PR**: `feat(frontend): Generate页面+Pipeline组件+API端点 (Tasks 023-032)`

Changes:
 `src/pages/index.astro` — 首页改为产品着陆页, Generate CTA入口
 `src/pages/generate/index.astro` — 核心Pipeline生成页
- 🆕 `src/components/pipeline/TextInput.astro` — 文本输入组件
- 🆕 `src/components/pipeline/PipelineSteps.astro` — 四步进度条
- 🆕 `src/components/pipeline/SvgPreview.astro` — SVG预览面板
- 🆕 `src/components/pipeline/ExportPanel.astro` — 导出面板
- 🆕 `src/pages/api/topology.ts` — 拓扑生成API
- 🆕 `src/pages/api/layout.ts` — ELK布局API
- 🆕 `src/pages/api/beautify.ts` — NanoBanana美化API
- 🆕 `src/layouts/GenerateLayout.astro` — 全宽布局

所有样式使用 astro-pure 组件 (Button/Card/Collapse/Label/Icon), 不自创文件和样式。


06更新：
# plan.md — astro-svgfigure v2: Phase 3 更新 (06版)

> **核心理念**: AutoFigure-Edit 是逆向流程（图→SAM3分割→SVG），我们做正向流程（文本→拓扑JSON→ELK布局→NanoBanana SVG）
> **技术栈**: Astro 5 + astro-pure 主题 + UnoCSS + ELK.js + Gemini NanoBanana + Python 后端
> **关键洞察**: NanoBanana 生成的图片如此完美以至于让拓扑学家像个小丑——ELK.js 只负责骨架坐标，最终以 JSON 脚手架向 NanoBanana 请求神经网络级 SVG

---

## 0. 项目现状 (基于 commit bb857aa)

### 已完成 Phase

| Phase | 内容 | Commit | 状态 |
|-------|------|--------|------|
| Phase 0 | 配置+ELK核心 (001-010) | af3fb7a | ✅ |
| Phase 1 | Backend AI Engine (013-022) | 45df633 | ✅ |
| Phase 2 | Generate页面+Pipeline组件 (023-032) | bb857aa | ✅ (有502 bug) |

### 🔴 已知 Bug: POST /api/topology → 502

**根因**: Astro 前端 `src/pages/api/topology.ts` 代理到 `http://localhost:8000/api/topology`，但 `server.py` 中没有 `/api/topology`、`/api/beautify`、`/api/models` 端点。`server.py` 只有旧的 autofigure 端点 (`/api/run`, `/api/config`, `/api/events/{job_id}` 等)。

**修复**: 在 `server.py` 中添加新的 pipeline API 端点，调用 `backend/` 已有的模块。

---

## 1. 四步正向 Pipeline 架构

```
[用户输入 method text]
         │
         ▼
┌─── Step 1: LLM 拓扑推理 ───┐     GitHub: ResearAI/AutoFigure
│  Gemini 解析文本             │
│  输出 ELK JSON (零坐标)      │
└──────────┬──────────────────┘
           │ topology.json
           ▼
┌─── Step 2: ELK.js 约束布局 ─┐     GitHub: kieler/elkjs
│  elkjs 分层算法               │
│  计算每个节点 (x,y,w,h)     │
└──────────┬──────────────────┘
           │ layouted.json
           ▼
┌─── Step 3: NanoBanana 美化 ─┐     GitHub: gemini-cli-extensions/nanobanana
│  layouted.json → JSON 脚手架 │
│  Gemini 生成学术级 SVG       │
└──────────┬──────────────────┘
           │ final.svg
           ▼
┌─── Step 4: Astro 前端展示 ──┐     GitHub: cworld1/astro-theme-pure
│  astro-pure 主题组件渲染     │
│  SVG 预览 + 导出             │
└─────────────────────────────┘
```

---

## 2. Phase 3: Claude 本轮 10 任务 — 修复502 + 完善 Pipeline

### 核心问题
1. **502 Bug**: server.py 缺少 /api/topology, /api/beautify, /api/models 端点
2. **前端→后端不通**: Astro API proxy → Python backend 断裂
3. **缺少 CORS**: Python backend 未配置 CORS 允许 :4321 调用
4. **缺少启动脚本**: 需要 concurrently 同时启动 Astro + Python

### 10 任务清单

| # | 文件 | Op | 说明 | GitHub 背书 |
|---|------|---|------|-------------|
| T1 | `server.py` | ✏️ | **修复502**: +/api/topology +/api/beautify +/api/validate +/api/models +CORS | ResearAI/AutoFigure |
| T2 | `backend/pipeline/nanobanana_bridge.py` | ✏️ | 补全 beautify_with_nanobanana() 缺失的 scaffold_builder 调用 | gemini-cli-extensions/nanobanana |
| T3 | `backend/pipeline/scaffold_builder.py` | ✏️ | 补全 build_scaffold() → NanoBananaScaffold 完整逻辑 | kieler/elkjs |
| T4 | `src/pages/api/topology.ts` | ✏️ | 增强错误处理+超时+重试提示 | withastro/astro |
| T5 | `src/pages/generate/index.astro` | ✏️ | 修复前端 pipeline 状态管理+错误展示 | cworld1/astro-theme-pure |
| T6 | `package.json` | ✏️ | +dev:all 脚本同时启动 Astro+Python | withastro/astro |
| T7 | `src/components/pipeline/PipelineSteps.astro` | ✏️ | 增强步骤状态样式+错误态 | cworld1/astro-theme-pure |
| T8 | `src/components/pipeline/SvgPreview.astro` | ✏️ | 增强SVG渲染+错误回退+loading | svgdotjs/svg.js |
| T9 | `src/components/pipeline/TextInput.astro` | ✏️ | 增加示例预填+placeholder优化 | cworld1/astro-theme-pure |
| T10 | `plan.md` | ✏️ | 更新为本文件 (v6) | - |

### T1 diff — server.py (关键修复)

```diff
+ import asyncio
+ from fastapi.middleware.cors import CORSMiddleware
+ from backend.config import get_settings
+ from backend.ai_engine import AIEngine
+ from backend.schemas import TopologyRequest, BeautifyRequest, ValidateRequest
+ from backend.pipeline.topology_gen import generate_topology
+ from backend.pipeline.nanobanana_bridge import beautify_with_nanobanana
+ from backend.pipeline.scaffold_builder import build_scaffold
+ from backend.pipeline.svg_validator import validate_svg
+
+ settings = get_settings()
+ ai_engine = AIEngine(settings)
+
+ app.add_middleware(
+     CORSMiddleware,
+     allow_origins=settings.CORS_ORIGINS,
+     allow_methods=["*"],
+     allow_headers=["*"],
+ )
+
+ @app.post("/api/topology")
+ async def api_topology(req: TopologyRequest): ...
+
+ @app.post("/api/beautify")
+ async def api_beautify(req: BeautifyRequest): ...
+
+ @app.post("/api/validate")
+ async def api_validate_svg(req: ValidateRequest): ...
+
+ @app.get("/api/models")
+ async def api_models(): ...
  # 保留全部现有 autofigure 端点 (/api/run, /api/config, /api/events/...)
```

---

## 3. Phase 4: Codex 并行 10 任务

| # | 文件 | Op | 说明 | GitHub 背书 |
|---|------|---|------|-------------|
| T11 | `src/pages/gallery/index.astro` | 🆕 | 画廊页: GalleryGrid + Card | cworld1/astro-theme-pure |
| T12 | `src/pages/playground/index.astro` | 🆕 | ELK 实验场 | kieler/elkjs |
| T13 | `src/components/gallery/GalleryGrid.astro` | 🆕 | 画廊网格 | cworld1/astro-theme-pure |
| T14 | `src/components/gallery/GalleryCard.astro` | 🆕 | 画廊卡片 | cworld1/astro-theme-pure |
| T15 | `src/components/pipeline/ElkOptions.astro` | 🆕 | ELK参数面板 | kieler/elkjs |
| T16 | `src/components/pipeline/TopologyPreview.astro` | 🆕 | 拓扑JSON预览 | EmilStenstrom/elkjs-svg |
| T17 | `src/pages/api/export.ts` | 🆕 | 导出 API | withastro/astro |
| T18 | `src/pages/api/validate.ts` | 🆕 | 验证 API | withastro/astro |
| T19 | `src/content/docs/svgfigure/getting-started.mdx` | 🆕 | 文档 | withastro/astro |
| T20 | `Dockerfile` | 🆕 | 部署 | docker/docker |

---

## 4. 部署命令

```bash
# 开发 (推荐: 使用 dev:all 一键启动)
git clone https://github.com/dylanyunlon/astro-svgfigure.git && cd astro-svgfigure
bun install && pip install -r requirements.txt
cp .env.example .env  # 填入 GEMINI_API_KEY
bun run dev:all       # 同时启动 Astro:4321 + FastAPI:8000

# 或分别启动
bun run dev            # 终端 1: Astro :4321
python server.py       # 终端 2: FastAPI :8000

# Docker
docker build -t astro-svgfigure .
docker run -p 4321:4321 -p 8000:8000 --env-file .env astro-svgfigure

# Vercel (前端) + Railway (后端)
npx vercel --prod
```

---

## 5. GitHub 背书 (20)

| Step | 项目 | 用途 |
|------|------|------|
| 拓扑 | ResearAI/AutoFigure | LLM→学术图拓扑 |
| 拓扑 | ResearAI/AutoFigure-Edit | SAM3逆向参考 |
| 后端 | dylanyunlon/skynetCheapBuy | AI Engine多Provider |
| ELK | kieler/elkjs | 约束布局引擎 |
| ELK | EmilStenstrom/elkjs-svg | ELK→SVG渲染 |
| ELK | xyflow/xyflow | ReactFlow ELK集成 |
| ELK | eclipse/elk | ELK算法参考 |
| NanoBanana | gemini-cli-extensions/nanobanana | 核心 |
| NanoBanana | ZeroLu/awesome-nanobanana-pro | prompt工程 |
| NanoBanana | aaronkwhite/nanobanana-studio-web | Web版 |
| 前端 | cworld1/astro-theme-pure | Astro主题 |
| 前端 | withastro/astro | Astro框架 |
| SVG | svgdotjs/svg.js | SVG操作 |
| SDK | openai/openai-python | OpenAI |
| SDK | anthropics/anthropic-sdk-python | Claude |
| SDK | google/generative-ai-python | Gemini |
| 部署 | docker/docker | 容器化 |
| CI/CD | actions/checkout | GitHub Actions |
| 测试 | pytest-dev/pytest | Python测试 |
| 启动 | open-cli-tools/concurrently | 双服务启动 |

---

## 6. PR 说明

**Phase 3 PR**: `fix(backend): 修复502错误 + Pipeline端到端联调 (Tasks T1-T10)`

Changes:
- ✏️ `server.py` — **修复502**: 添加 /api/topology, /api/beautify, /api/models 端点 + CORS
- ✏️ `backend/pipeline/nanobanana_bridge.py` — 补全 beautify 逻辑
- ✏️ `backend/pipeline/scaffold_builder.py` — 补全 scaffold 构建
- ✏️ `src/pages/api/topology.ts` — 增强错误处理
- ✏️ `src/pages/generate/index.astro` — 修复前端状态管理
- ✏️ `package.json` — 添加 dev:all 启动脚本
- ✏️ `src/components/pipeline/PipelineSteps.astro` — 增强步骤状态
- ✏️ `src/components/pipeline/SvgPreview.astro` — 增强预览+loading
- ✏️ `src/components/pipeline/TextInput.astro` — 示例预填
- ✏️ `plan.md` — 更新 v6

**Codex 并行工作请注意**: 本 PR 修改了 server.py 核心文件, Codex 的 Phase 4 应基于此 PR merge 后的代码。


07更新：
# plan.md — astro-svgfigure v2: Phase 4 更新 (07版)

> **核心理念**: AutoFigure-Edit 是逆向流程（图→SAM3分割→SVG），我们做正向流程（文本→拓扑JSON→ELK布局→NanoBanana SVG）
> **技术栈**: Astro 5 + astro-pure 主题 + UnoCSS + ELK.js + Gemini NanoBanana + Python 后端
> **关键洞察**: NanoBanana 生成的图片如此完美以至于让拓扑学家像个小丑——ELK.js 只负责骨架坐标，最终以 JSON 脚手架向 NanoBanana 请求神经网络级 SVG

---

## 0. 项目现状 (基于 commit 6633c01)

### 已完成 Phase

| Phase | 内容 | Commit | 状态 |
|-------|------|--------|------|
| Phase 0 | 配置+ELK核心 (001-010) | af3fb7a | ✅ |
| Phase 1 | Backend AI Engine (013-022) | 45df633 | ✅ |
| Phase 2 | Generate页面+Pipeline组件 (023-032) | bb857aa | ✅ |
| Phase 3 | 修复502+Pipeline联调 (T1-T10) | 6633c01 | ✅ |
| **Phase 4** | **Gallery+Playground+补全API (T1-T10)** | **本轮** | **🔄 进行中** |

### 🔴 已修复 Bug: server.py "web directory does not exist"

**根因**: `server.py` 第672行 `app.mount("/", StaticFiles(directory=WEB_DIR))` 硬编码挂载 `web/` 目录，但该目录仅在 `bun run build` 后才存在。开发模式下 Astro 自带 dev server (:4321)，不需要 Python 后端 serve 静态文件。

**修复**: 改为条件挂载 — `if WEB_DIR.is_dir()` 才挂载，否则打印提示信息。
同时添加 `build:web` 脚本: `astro build && cp -r dist/* web/` 用于生产部署。

---

## 1. Phase 4: Claude 本轮 10 任务清单

| # | 文件 | Op | 说明 | GitHub 背书 |
|---|------|---|------|-------------|
| T1 | `server.py` | ✏️ | **修复**: web/ 目录不存在时跳过 StaticFiles 挂载 | withastro/astro |
| T2a | `src/pages/api/export.ts` | 🆕 | SVG→PNG/PDF 导出 API 代理 | withastro/astro |
| T2b | `src/pages/api/validate.ts` | 🆕 | SVG 语法校验 API 代理 | withastro/astro |
| T3 | `src/pages/api/models.ts` | 🆕 | GET 可用 AI 模型列表 (fallback 支持) | dylanyunlon/skynetCheapBuy |
| T4 | `src/pages/gallery/index.astro` | 🆕 | 画廊页: 示例SVG展示+分类筛选+Card+Label | cworld1/astro-theme-pure |
| T5 | `src/pages/playground/index.astro` | 🆕 | ELK Playground: JSON编辑+实时布局+SVG渲染 | kieler/elkjs |
| T6 | `src/components/pipeline/TopologyPreview.astro` | 🆕 | 拓扑JSON预览: 语法高亮+统计+校验 | EmilStenstrom/elkjs-svg |
| T7 | `src/components/pipeline/ElkOptions.astro` | 🆕 | ELK参数面板: 算法/方向/间距配置 | kieler/elkjs, eclipse/elk |
| T8 | `src/components/pipeline/ModelSelector.astro` | 🆕 | AI模型选择器: 从/api/models动态加载 | dylanyunlon/skynetCheapBuy |
| T9 | `src/layouts/FullWidthLayout.astro` | 🆕 | 全宽布局: Gallery/Playground共用 | cworld1/astro-theme-pure |
| T10 | `plan.md` | ✏️ | 更新为本文件 (v7) | - |
| T10b | `package.json` | ✏️ | +build:web 脚本 (Astro build → web/) | withastro/astro |

### diff 汇总

**server.py (T1)**: 
```diff
-app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
+if WEB_DIR.is_dir():
+    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
+else:
+    logger.info("web/ directory not found — skipping static file mount.")
```

**package.json (T10b)**:
```diff
+    "build:web": "astro build && cp -r dist/* web/ 2>/dev/null || (mkdir -p web && cp -r dist/* web/)",
```

---

## 2. Phase 5: Codex 并行 10 任务 (下一轮)

| # | 文件 | Op | 说明 | GitHub 背书 |
|---|------|---|------|-------------|
| T11 | `src/components/pipeline/JsonEditor.astro` | 🆕 | JSON编辑器: 编辑拓扑/脚手架 | cworld1/astro-theme-pure |
| T12 | `src/components/pipeline/CompareView.astro` | 🆕 | 骨架vs最终SVG对比 | cworld1/astro-theme-pure |
| T13 | `src/components/pipeline/ScaffoldView.astro` | 🆕 | NanoBanana脚手架预览 | gemini-cli-extensions/nanobanana |
| T14 | `src/components/pipeline/HistoryList.astro` | 🆕 | 生成历史 (Timeline) | cworld1/astro-theme-pure |
| T15 | `src/components/pipeline/ErrorDisplay.astro` | 🆕 | 错误展示 (Aside danger) | cworld1/astro-theme-pure |
| T16 | `src/components/pipeline/PromptPreview.astro` | 🆕 | NanoBanana prompt预览 | ZeroLu/awesome-nanobanana-pro |
| T17 | `src/content/docs/svgfigure/getting-started.mdx` | 🆕 | 快速开始文档 | withastro/astro |
| T18 | `src/content/docs/svgfigure/pipeline-overview.mdx` | 🆕 | Pipeline架构文档 | withastro/astro |
| T19 | `public/examples/transformer.svg` | 🆕 | 示例SVG: Transformer | - |
| T20 | `Dockerfile` | 🆕 | 多阶段构建 | docker/docker |

---

## 3. 部署命令

```bash
# 开发 (推荐: 双服务)
git clone https://github.com/dylanyunlon/astro-svgfigure.git && cd astro-svgfigure
bun install && pip install -r requirements.txt
cp .env.example .env  # 填入 GEMINI_API_KEY
bun run dev            # 终端 1: Astro :4321
python server.py       # 终端 2: FastAPI :8000

# 一键启动 (需要 concurrently)
bun run dev:all

# 生产构建 (Astro输出到web/供Python server serve)
bun run build:web && python server.py

# Docker
docker build -t astro-svgfigure .
docker run -p 4321:4321 -p 8000:8000 --env-file .env astro-svgfigure

# Vercel (前端) + Railway/Fly.io (后端)
npx vercel --prod
```

---

## 4. PR 说明

**Phase 4 PR**: `feat(frontend): Gallery页+Playground页+Pipeline组件补全+web目录修复 (Tasks T1-T10)`

Changes:
- ✏️ `server.py` — **修复**: web/目录不存在时条件跳过StaticFiles挂载
- ✏️ `package.json` — +build:web脚本
- 🆕 `src/pages/api/export.ts` — SVG→PNG/PDF导出API
- 🆕 `src/pages/api/validate.ts` — SVG语法校验API
- 🆕 `src/pages/api/models.ts` — AI模型列表API (带fallback)
- 🆕 `src/pages/gallery/index.astro` — 画廊页 (Card+Label+Button)
- 🆕 `src/pages/playground/index.astro` — ELK Playground (实时布局+SVG)
- 🆕 `src/components/pipeline/TopologyPreview.astro` — 拓扑JSON预览
- 🆕 `src/components/pipeline/ElkOptions.astro` — ELK算法参数面板
- 🆕 `src/components/pipeline/ModelSelector.astro` — AI模型选择器
- 🆕 `src/layouts/FullWidthLayout.astro` — 全宽布局
- ✏️ `plan.md` — 更新v7

所有样式使用 astro-pure 组件 (Card/Button/Collapse/Label/Icon/Tabs), 不自创文件和样式。
**Codex 并行注意**: 本 PR 不修改 backend/ 目录任何文件(除 server.py 一行条件挂载), Codex 可安全并行开发 T11-T20。

---

## 5. GitHub 背书 (20)

| Step | 项目 | 用途 |
|------|------|------|
| 拓扑 | ResearAI/AutoFigure | LLM→学术图拓扑 |
| 拓扑 | ResearAI/AutoFigure-Edit | SAM3逆向参考 |
| 后端 | dylanyunlon/skynetCheapBuy | AI Engine多Provider |
| ELK | kieler/elkjs | 约束布局引擎 |
| ELK | EmilStenstrom/elkjs-svg | ELK→SVG渲染 |
| ELK | xyflow/xyflow | ReactFlow ELK集成 |
| ELK | eclipse/elk | ELK算法参考 |
| NanoBanana | gemini-cli-extensions/nanobanana | 核心 |
| NanoBanana | ZeroLu/awesome-nanobanana-pro | prompt工程 |
| NanoBanana | aaronkwhite/nanobanana-studio-web | Web版 |
| 前端 | cworld1/astro-theme-pure | Astro主题 |
| 前端 | withastro/astro | Astro框架 |
| SVG | svgdotjs/svg.js | SVG操作 |
| SDK | openai/openai-python | OpenAI |
| SDK | anthropics/anthropic-sdk-python | Claude |
| SDK | google/generative-ai-python | Gemini |
| 部署 | docker/docker | 容器化 |
| CI/CD | actions/checkout | GitHub Actions |
| 测试 | pytest-dev/pytest | Python测试 |
| 启动 | open-cli-tools/concurrently | 双服务启动 |

08更新：

**Claude 的 PR** (T01-T10):
```
Title: fix(502) + feat(pipeline): 修复代理错误 + 前端组件增强 (T01-T10)
Branch: claude/fix-502-and-enhance-pipeline

Changes:
- T01: 修复 IPv6 502 错误 (localhost → 127.0.0.1)
- T02: 修复 Pydantic model_ 命名空间警告
- T03: /api/layout 增加骨架 SVG 输出
- T04: SvgPreview 双视图 Tab 切换
- T05: 结构化错误显示组件
- T06: scaffold_builder 完善
- T07: nanobanana_bridge 实现
- T08: Gallery 页面重构
- T09: Playground 页面重构
- T10: 首页重构
```

**Codex 的 PR** (T11-T20):
```
Title: feat(backend+docs): 验证/导出/文档/Docker (T11-T20)
Branch: codex/backend-and-docs

Changes:
- T11: svg_validator 完善
- T12: svg_scaler 完善
- T13-T14: 导出功能前后端
- T15: ELK 类型/常量/预设
- T16: 文档页面
- T17: Docker
- T18: 暗色模式
- T19: 示例数据

09更新：
# astro-svgfigure — 第二阶段开发计划 (plan.md)

> **基于最新提交树** `31f830b feat: T01-T20 大批量更新`
> **目标**: 修复当前 Bug + 完善正向 Pipeline + 前端界面打磨
> **分工**: Claude 完成 T21-T30, Codex 并行完成 T31-T40

---

## 一、当前问题诊断

### Bug 1: `[500] POST /api/layout 860ms` — ELK layout failed

**根因**: `src/pages/api/layout.ts` 中 `elkToSvg(layouted)` 导入路径使用 `@/lib/elk/to-svg`，但 ELK 的 `layout()` 返回结果的 `edges` 可能没有 `sections` 字段（当拓扑 JSON 的 edge 格式不规范时），导致 `elkToSvg` 内部抛异常。虽然有 `try/catch` 包裹 skeletonSvg 生成，但真正的问题在于 **topology → ELK 的 graph 格式不兼容**：LLM 返回的 `layoutOptions` 键名与 ELK 期望的不一致（如 `elk.algorithm` vs `"elk.algorithm"` 字符串键），或 `children` 节点缺少必要的 `width/height`。

**修复方案**:
1. `layout.ts` 增加更健壮的 graph 预处理（确保 children 有 width/height，edges 有正确格式）
2. `topology_gen.py` 的 LLM prompt 增加更严格的格式要求
3. `layout.ts` 的 `elkToSvg` 调用增加 null check

### Bug 2: 前端 Pipeline 步骤 2 报 "ELK layout failed"

**关联 Bug 1**: 是同一个问题的前端表现。topology API 成功返回后，layout API 收到的 graph 可能不符合 ELK 要求。

---

## 二、文件变更清单

### 修改文件 (MODIFY) — 必须与源文件 diff 确认

| # | 文件路径 | 变更说明 |
|---|---------|---------|
| M1 | `src/pages/api/layout.ts` | 修复 ELK layout 500 错误：增强 graph 预处理、容错处理 |
| M2 | `backend/pipeline/topology_gen.py` | 优化 LLM prompt，确保输出严格 ELK 格式；增加后处理验证 |
| M3 | `backend/pipeline/nanobanana_bridge.py` | 修复 `scaffold.elements` 属性访问（dict vs object）|
| M4 | `src/pages/generate/index.astro` | 修复 Pipeline 错误处理、增加 loading 状态、改善 UX |
| M5 | `src/components/pipeline/PipelineSteps.astro` | 增加动画过渡、步骤详情展示 |
| M6 | `src/components/pipeline/SvgPreview.astro` | 增加缩放控制、skeleton → final 过渡动画 |
| M7 | `src/components/pipeline/ErrorDisplay.astro` | 增加具体错误诊断提示和重试按钮 |
| M8 | `src/components/pipeline/ExportPanel.astro` | 修复导出按钮状态、增加 PNG 导出 |
| M9 | `server.py` | 增加 `/api/health` 端点；增加启动时 provider 检测日志 |
| M10 | `backend/config.py` | 增加 `GEMINI_API_BASE` 的 tryallai 代理支持说明 |
| M11 | `src/pages/playground/index.astro` | 修复 playground 与 layout API 的交互 |
| M12 | `src/lib/elk/to-svg.ts` | 增加 null/undefined 防御性检查 |
| M13 | `package.json` | 增加 `concurrently` devDependency |
| M14 | `.env.example` | 增加 `GEMINI_API_BASE` 示例和 tryallai 说明 |

### 新增文件 (ADD)

| # | 文件路径 | 说明 |
|---|---------|------|
| A1 | `src/pages/api/health.ts` | 健康检查端点：检测 Python 后端连通性 |
| A2 | `src/components/pipeline/HealthCheck.astro` | 后端连接状态指示器组件 |
| A3 | `src/components/pipeline/ModelStatus.astro` | 当前模型/API 状态显示 |
| A4 | `backend/pipeline/__tests__/test_topology.py` | topology_gen 单元测试 |
| A5 | `backend/pipeline/__tests__/test_scaffold.py` | scaffold_builder 单元测试 |
| A6 | `backend/pipeline/__tests__/__init__.py` | 测试包初始化 |
| A7 | `docs/ARCHITECTURE.md` | 项目架构文档 |
| A8 | `docs/API.md` | API 接口文档 |
| A9 | `scripts/dev.sh` | 一键启动开发环境脚本 |


---

## 三、任务分解 (T21-T40)

### Claude 负责: T21-T30

#### T21: 修复 ELK Layout 500 错误 (Critical Bug Fix)
- **修改**: `src/pages/api/layout.ts`
- **内容**:
  1. 增强 `processedGraph` 预处理：验证 children 数组非空、每个 node 有 id/width/height
  2. edges 预处理：确保 sources/targets 是数组、id 唯一
  3. layoutOptions 规范化：统一使用 `elk.` 前缀字符串键
  4. `elkToSvg` 调用前检查 layouted 有效性
  5. 错误响应增加 `debug` 字段返回实际收到的 graph 结构
- **PR 说明**: `fix(api): 修复 /api/layout ELK 500 错误 — 增强 graph 预处理和容错`

#### T22: 优化 Topology LLM Prompt
- **修改**: `backend/pipeline/topology_gen.py`
- **内容**:
  1. prompt 增加更严格的 JSON Schema 示例
  2. 增加 `_validate_topology()` 后处理函数：验证 id 唯一性、edge 引用有效性
  3. 增加 `_fix_topology()` 自动修复常见问题（缺少 width/height 的 node、重复 edge id）
  4. 增加 few-shot 示例（transformer、diffusion 各一个）
- **PR 说明**: `fix(backend): 优化 topology LLM prompt + 增加后处理验证`

#### T23: 修复 NanoBanana Bridge dict/object 兼容
- **修改**: `backend/pipeline/nanobanana_bridge.py`
- **内容**:
  1. `beautify_with_nanobanana()` 内 `scaffold.elements` 属性访问改为安全访问
  2. 统一 scaffold 为 dict 处理路径，避免 `hasattr` 判断不准确
  3. fallback SVG 生成路径修复 NanoBananaScaffold 构造
- **PR 说明**: `fix(backend): 修复 nanobanana_bridge scaffold dict/object 兼容性`

#### T24: 增加 /api/health 健康检查
- **新增**: `src/pages/api/health.ts`
- **新增**: `src/components/pipeline/HealthCheck.astro`
- **修改**: `src/pages/generate/index.astro` — 引入 HealthCheck 组件
- **内容**:
  1. `/api/health` 检测 Python 后端连通性（fetch `BACKEND_URL/api/models`）
  2. 返回 `{ astro: true, backend: true/false, models: [...] }`
  3. HealthCheck 组件：页面加载时自动检测，显示绿/红状态灯
  4. 后端不可用时禁用 Generate 按钮并提示用户启动 `python server.py`
- **PR 说明**: `feat(api): 增加 /api/health 健康检查 + 前端状态指示器`

#### T25: 改善 Generate 页面 UX
- **修改**: `src/pages/generate/index.astro`
- **修改**: `src/components/pipeline/ErrorDisplay.astro`
- **内容**:
  1. Pipeline 错误时显示具体诊断：区分"后端未启动"、"API Key 缺失"、"ELK 格式错误"
  2. 增加"重试"按钮
  3. 成功时增加完成动画
  4. textarea 增加 placeholder 示例文本
  5. 步骤进度增加百分比或时间估算
- **PR 说明**: `improve(frontend): Generate 页面 UX 增强 — 错误诊断、重试、进度指示`

#### T26: 增加 to-svg.ts 防御性检查
- **修改**: `src/lib/elk/to-svg.ts`
- **内容**:
  1. `renderEdge` 增加 section.startPoint/endPoint null 检查
  2. `renderNode` 增加 label 安全转义
  3. `elkToSvg` 增加 graph.children/edges 空数组检查
  4. 增加最大节点数限制（防止 LLM 生成过大拓扑导致 SVG 过大）
- **PR 说明**: `fix(elk): to-svg.ts 增加防御性检查，防止 undefined 属性访问`

#### T27: 修复 ExportPanel SVG/PNG 导出
- **修改**: `src/components/pipeline/ExportPanel.astro`
- **修改**: `src/pages/api/export.ts`
- **内容**:
  1. SVG 导出：从 preview 容器提取 SVG 内容，创建 Blob 下载
  2. PNG 导出：使用 Canvas API 将 SVG 渲染为 PNG
  3. 导出按钮状态跟随 pipeline 完成状态
  4. 增加尺寸选择（1x, 2x, 4x）
- **PR 说明**: `feat(export): 修复并增强 SVG/PNG 导出功能`


#### T29: 清理调试文件 + 更新 .env.example
- **删除**: `test_502.py`, `test_502.log`, `tree.txt`
- **修改**: `.env.example`
- **内容**:
  1. 移除已完成使命的调试文件
  2. `.env.example` 增加 `GEMINI_API_BASE=https://api.tryallai.com/v1` 示例
  3. 增加各 provider 的 tryallai 代理配置说明
- **PR 说明**: `chore: 清理调试文件 + 完善 .env.example 配置说明`

#### T30: 增加 Pipeline 单元测试
- **新增**: `backend/pipeline/__tests__/test_topology.py`
- **新增**: `backend/pipeline/__tests__/test_scaffold.py`
- **新增**: `backend/pipeline/__tests__/__init__.py`
- **内容**:
  1. test_topology: 测试 `_parse_topology_json`、`_validate_topology`（新增）、`create_example_topology`
  2. test_scaffold: 测试 `build_scaffold` 输入 dict 和 ElkGraph 两种情况
  3. 测试 edge case: 空 children、无 edges、嵌套子图
- **PR 说明**: `test: 增加 topology_gen + scaffold_builder 单元测试`

---

### Codex 负责: T31-T40

#### T31: server.py 增加 /api/health + 启动检测
- **修改**: `server.py`
- **PR 说明**: `feat(server): 增加 /api/health + 启动时 provider 检测`

#### T32: Playground 页面修复
- **修改**: `src/pages/playground/index.astro`
- **PR 说明**: `fix(playground): 修复 Playground 页面 layout 交互`

#### T33: Gallery 页面展示生成历史
- **修改**: `src/pages/gallery/index.astro`
- **PR 说明**: `feat(gallery): Gallery 页面展示生成历史`

#### T34: ModelSelector 组件联动
- **修改**: `src/components/pipeline/ModelSelector.astro`
- **PR 说明**: `feat(component): ModelSelector 动态模型列表`

#### T35: TopologyPreview 可视化
- **修改**: `src/components/pipeline/TopologyPreview.astro`
- **PR 说明**: `feat(component): TopologyPreview 拓扑 JSON 可视化`

#### T36: ElkOptions 高级选项面板
- **修改**: `src/components/pipeline/ElkOptions.astro`
- **PR 说明**: `feat(component): ElkOptions 高级选项面板`

#### T37: SVG 验证 + 自动修复前端集成
- **修改**: `src/pages/api/validate.ts`
- **PR 说明**: `feat(pipeline): SVG 验证 + 自动修复前端集成`


#### T39: 响应式布局优化
- **修改**: `src/pages/generate/index.astro`, `src/pages/playground/index.astro`
- **PR 说明**: `style: Generate + Playground 响应式布局优化`

#### T40: E2E Pipeline 集成测试
- **新增**: `tests/e2e/pipeline.test.ts`
- **PR 说明**: `test: E2E pipeline 集成测试`

---

## 四、Git 操作规范

### 分支策略
```
main
 ├── fix/elk-layout-500          (T21, T22, T23, T26) ← Claude
 ├── feat/health-check           (T24) ← Claude
 ├── improve/generate-ux         (T25, T27) ← Claude
 ├── docs/architecture           (T28, T29) ← Claude
 ├── test/pipeline-unit          (T30) ← Claude
 ├── feat/server-health          (T31) ← Codex
 ├── fix/playground              (T32) ← Codex
 ├── feat/gallery-history        (T33) ← Codex
 ├── feat/model-selector         (T34) ← Codex
 ├── feat/topology-preview       (T35) ← Codex
 ├── feat/elk-options            (T36) ← Codex
 ├── feat/svg-validate-ui        (T37) ← Codex
 ├── style/dark-mode             (T38) ← Codex
 ├── style/responsive            (T39) ← Codex
 └── test/e2e                    (T40) ← Codex
```

### Commit 格式
```
<type>(<scope>): <subject>

Types: fix, feat, improve, docs, test, chore, style
Scopes: api, backend, frontend, elk, pipeline, component, server
```

### PR 流程
1. 每个任务（或相关任务组）一个 PR
2. PR 标题使用上面各任务的 "PR 说明"
3. PR 描述包含：变更文件列表、测试方法、与源文件的 diff 摘要
4. Claude 和 Codex 的 PR 不应有文件冲突（按上面分工划分）

### Diff 检查 (必须！)
**每次修改文件后，运行:**
```bash
git diff <filename>
```
确保：
- 没有误删原有功能代码
- 没有覆盖其他任务的修改
- import 语句完整
- 类型定义一致

---

## 五、架构概览

```
正向 Pipeline 流程:

  用户输入 (Paper Method Text)
       │
       ▼
  ┌─────────────────┐
  │ Step 1: Topology │  POST /api/topology → Python backend
  │ LLM → ELK JSON  │  (topology_gen.py → AIEngine → Gemini)
  └────────┬────────┘
           │ ELK Graph JSON (零坐标)
           ▼
  ┌─────────────────┐
  │ Step 2: Layout   │  POST /api/layout → Astro SSR
  │ ELK.js 约束求解  │  (elkjs → computed x,y)
  └────────┬────────┘
           │ Layouted Graph + Skeleton SVG
           ▼
  ┌─────────────────┐
  │ Step 3: Beautify │  POST /api/beautify → Python backend
  │ NanoBanana SVG   │  (scaffold_builder → nanobanana_bridge → Gemini)
  └────────┬────────┘
           │ Publication-quality SVG
           ▼
  ┌─────────────────┐
  │ Step 4: Render   │  前端渲染 + 导出
  │ SVG Preview      │
  └─────────────────┘

前端技术栈:
  - Astro 5 + astro-pure theme (Card, Button, Collapse, Label, Tabs)
  - ELK.js (约束布局引擎)
  - UnoCSS (样式)
  - 组件全部使用 astro-pure，不自创文件自创样式

后端技术栈:
  - FastAPI + Uvicorn (server.py, port 8000)
  - AIEngine (multi-provider: Gemini/OpenAI/Anthropic/Claude-Compatible)
  - pydantic-settings (.env 配置)
  - tryallai.com 代理支持
```

---

## 六、优先级

| 优先级 | 任务 | 原因 |
|--------|------|------|
| P0 (立即) | T21, T22, T23, T26 | 修复 Pipeline 核心 Bug，当前无法正常运行 |
| P1 (重要) | T24, T25, T31 | 改善开发体验和错误诊断 |
| P2 (正常) | T27, T32, T33, T34, T35 | 功能完善 |
| P3 (低) | T28, T29, T30, T36-T40 | 文档、测试、打磨 |


10更新：
# plan.md — astro-svgfigure v2: Phase 5 更新 (10版)

> **核心理念**: AutoFigure-Edit 是逆向流程（图→SAM3分割→SVG），我们做正向流程（文本→拓扑JSON→ELK布局→NanoBanana SVG→Gemini 3 科研图）
> **技术栈**: Astro 5 + astro-pure 主题 + UnoCSS + ELK.js + Grok 4 + Gemini 3 Pro Image + Python 后端
> **关键洞察**: NanoBanana 生成的图片如此完美以至于让拓扑学家像个小丑——ELK.js 只负责骨架坐标，最终用 Grok 4 反推 prompt + Gemini 3 生成科研级别图片

---

## 0. 项目现状 (基于 commit 2631634)

### 已完成 Phase

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 配置+ELK核心 (001-010) | ✅ |
| Phase 1 | Backend AI Engine (013-022) | ✅ |
| Phase 2 | Generate页面+Pipeline组件 (023-032) | ✅ |
| Phase 3 | 修复502+Pipeline联调 (T1-T10) | ✅ |
| Phase 4 | Gallery+Playground+补全API (T1-T10) | ✅ |
| Phase 4.5 | 修复ELK layout 500 + topology验证 (T21-T26) | ✅ |
| **Phase 5** | **Step 5: Grok 4 + Gemini 3 Image (本轮)** | **🔄 进行中** |

---

## 1. 五步正向 Pipeline 架构 (更新)

```
[用户输入 method text]
         │
         ▼
┌─── Step 1: LLM 拓扑推理 ───┐
│  Claude Opus 4.6 解析文本    │
│  输出 ELK JSON (零坐标)      │
└──────────┬──────────────────┘
           │ topology.json
           ▼
┌─── Step 2: ELK.js 约束布局 ─┐
│  elkjs 分层算法               │
│  计算每个节点 (x,y,w,h)     │
└──────────┬──────────────────┘
           │ layouted.json + skeleton SVG
           ▼
┌─── Step 3: NanoBanana 美化 ─┐
│  Grok 4 生成 JSON 脚手架     │
│  Gemini NanoBanana → SVG     │
└──────────┬──────────────────┘
           │ final.svg
           ▼
┌─── Step 4: SVG 渲染展示 ────┐
│  astro-pure 组件渲染         │
│  用户确认 SVG 无缺失组件     │
└──────────┬──────────────────┘
           │ 用户确认 ✅
           ▼
┌─── Step 5: 科研级图片生成 ──┐   ← 🆕 本轮新增
│  a) Grok 4 反推专业 prompt   │
│  b) Gemini 3 Pro Image 生图  │
│  输出: publication-quality   │
│  科研级 PNG (4K)             │
└─────────────────────────────┘
```

### Step 5 详细流程

```
Grok 4 + Gemini 3 结合高效生成科研绘图:

Step a: Grok 4 反推专业提示词 (Prompt)
  - 输入: paper method text + ELK SVG 布局 + (可选) 参考图片
  - Grok 4 分析 SVG 结构,从画面风格、布局、核心元素、配色方案等多维度描述
  - 输出: 200-400 词的详细 AI 绘图提示词

Step b: Gemini 3 Pro Image 生成科研图
  - 输入: Grok 4 生成的 prompt + SVG 布局 (作为空间参考)
  - API: Gemini native v1beta/models/gemini-3-pro-image-preview:generateContent
  - responseModalities: ["TEXT", "IMAGE"]
  - imageConfig: { aspectRatio: "16:9", imageSize: "4K" }
  - 输出: base64 encoded PNG 图片

tryallai.com 代理配置:
  - Gemini Image: GEMINI_API_BASE + /v1beta/models/{model}:generateContent
  - Grok 4: OPENAI_API_BASE + /v1/chat/completions (model: grok-4)
```

---

## 2. Phase 5: Claude 本轮 10 任务清单

| # | 文件 | Op | 说明 |
|---|------|---|------|
| T1 | `backend/pipeline/gemini_image_gen.py` | 🆕 | **核心**: Grok 4 prompt + Gemini 3 Image 生成模块 |
| T2 | `server.py` | ✏️ | +/api/generate-image +/api/generate-prompt 端点 |
| T3 | `src/pages/api/generate-image.ts` | 🆕 | Astro API proxy → Python /api/generate-image |
| T4 | `src/pages/api/generate-prompt.ts` | 🆕 | Astro API proxy → Python /api/generate-prompt |
| T5 | `src/components/pipeline/ImageGenPanel.astro` | 🆕 | Step 5 前端组件: 确认→prompt→生成→预览→下载 |
| T6 | `src/pages/generate/index.astro` | ✏️ | 集成 ImageGenPanel + pipeline 状态联动 |
| T7 | `src/components/pipeline/PipelineSteps.astro` | ✏️ | +Step 5 进度条 |
| T8 | `backend/config.py` | ✏️ | +DEFAULT_PROMPT_MODEL +DEFAULT_IMAGE_MODEL +新模型列表 |
| T9 | `.env.example` | ✏️ | +tryallai 说明 +Step 5 配置 |
| T10 | `plan.md` | ✏️ | 更新为本文件 (v10) |

---

## 3. 新增文件位置

```
astro-svgfigure/
├── backend/pipeline/
│   └── gemini_image_gen.py          ← 🆕 Step 5 核心模块
├── src/pages/api/
│   ├── generate-image.ts            ← 🆕 Astro API proxy
│   └── generate-prompt.ts           ← 🆕 Astro API proxy
├── src/components/pipeline/
│   └── ImageGenPanel.astro          ← 🆕 Step 5 前端组件
└── (修改的文件保持原位)
```

---

## 4. API 接口文档

### POST /api/generate-image

Step 5 完整流程: Grok 4 prompt → Gemini 3 Image

Request:
```json
{
  "svg_content": "<svg ...>...</svg>",
  "method_text": "Our model uses a transformer encoder...",
  "reference_image_b64": null,
  "prompt_model": "grok-4",
  "image_model": "gemini-3-pro-image-preview",
  "aspect_ratio": "16:9",
  "image_size": "4K",
  "custom_prompt": null
}
```

Response:
```json
{
  "success": true,
  "image_b64": "iVBORw0KGgo...",
  "mime_type": "image/png",
  "prompt": "Create a high-quality scientific...",
  "prompt_model_used": "grok-4",
  "image_model_used": "gemini-3-pro-image-preview"
}
```

### POST /api/generate-prompt

Step a only: Grok 4 prompt engineering

---

## 5. 部署命令


---

## 6. PR 说明

**Phase 5 PR**: `feat(step5): Grok 4 + Gemini 3 Pro Image `

Branch: `claude/step5-gemini-image-gen`

Changes:
- 🆕 `backend/pipeline/gemini_image_gen.py` — Step 5 核心
- ✏️ `server.py` — +/api/generate-image +/api/generate-prompt
- 🆕 `src/pages/api/generate-image.ts` — Astro API proxy
- 🆕 `src/pages/api/generate-prompt.ts` — Astro API proxy
- 🆕 `src/components/pipeline/ImageGenPanel.astro` — Step 5 前端组件
- ✏️ `src/pages/generate/index.astro` — 集成 ImageGenPanel
- ✏️ `src/components/pipeline/PipelineSteps.astro` — +Step 5
- ✏️ `backend/config.py` — +新模型配置
- ✏️ `.env.example` — +tryallai + Step 5
- ✏️ `plan.md` — v10

**Codex 并行注意**: 本 PR 新增文件不与现有 Steps 1-4 冲突。server.py 仅新增端点。

---

## 7. GitHub 背书 (22)

| Step | 项目 | 用途 |
|------|------|------|
| 拓扑 | ResearAI/AutoFigure | LLM→学术图拓扑 |
| 后端 | dylanyunlon/skynetCheapBuy | AI Engine多Provider |
| ELK | kieler/elkjs | 约束布局引擎 |
| NanoBanana | gemini-cli-extensions/nanobanana | 核心 |
| NanoBanana | ZeroLu/awesome-nanobanana-pro | prompt工程 |
| 前端 | cworld1/astro-theme-pure | Astro主题 |
| 前端 | withastro/astro | Astro框架 |
| Step 5 | xAI/grok | Grok 4 prompt engineering |
| Step 5 | google/gemini-api | Gemini 3 Pro Image |
| 代理 | tryallai.com | API 代理 (无需翻墙) |

11更新：
Pipeline 重构 + 修复 503 model_not_found

## 问题

1. **503 错误**: 前端硬编码发送 `gemini-2.5-flash`，tryallai 代理无此渠道
2. **流程不对**: 旧 5-step Pipeline 中 Step 3 (NanoBanana/beautify) 只是用 LLM 重写 SVG 标签，本质还是矩形+箭头，并未实现"科研级图片生成"
3. **多处残留**: `index.astro`、`ai_engine.py`、`config.py`、`models.ts` 仍引用旧流程和不存在的模型

## 改动

### 新 3-Step Pipeline

| Step | 模型 | 输入 → 输出 |
|------|------|------------|
| 1. Topology + ELK Layout | `claude-opus-4-6` (.env) | 论文 method text → 拓扑 JSON → ELK.js 约束布局 → 骨架 SVG |
| 2. Grok 4 Prompt | `grok-4` (.env) | 骨架 SVG + method text → 200-400 词专业绘图 prompt |
| 3. Gemini 3 Image | `gemini-3-pro-image-preview` (.env) | prompt → 4K PNG 科研级图片 |

用户在 Step 1 后确认骨架组件无缺失，再触发 Step 2→3。Step 2 生成的 prompt 可编辑后再送 Step 3。

### 修改文件 (9个)

**前端 (5个)**

| 文件 | 改动 |
|------|------|
| `src/pages/generate/index.astro` | 全面重写: 去掉 model 下拉框和 NanoBanana 调用, 改为两个按钮 (Step 1 / Step 2→3), 新增 Grok prompt 编辑区 + 最终图片预览/下载 |
| `src/components/pipeline/PipelineSteps.astro` | 5 步 → 3 步 (Topology+ELK / Grok 4 Prompt / Gemini 3 Image) |
| `src/pages/index.astro` | 删除旧 4-step 流程副本, 改为重定向到 `/generate` |
| `src/pages/api/topology.ts` | 移除硬编码 `model: 'gemini-2.5-flash'`, 改为 `undefined` 由后端 .env 决定 |
| `src/pages/api/beautify.ts` | 同上 |
| `src/pages/api/models.ts` | fallback 模型列表从 `gemini-2.5-flash` 改为 `claude-opus-4-6` / `grok-4` / `gemini-3-pro-image-preview` |

**后端 (3个)**

| 文件 | 改动 |
|------|------|
| `backend/config.py` | `DEFAULT_AI_MODEL` / `DEFAULT_TOPOLOGY_MODEL` 从 `gemini-2.5-flash` → `claude-opus-4-6`; 注释更新为 3-Step |
| `backend/ai_engine.py` | `DEFAULT_MODEL` 改为从 `settings.DEFAULT_AI_MODEL` 读取; `is_openai_model()` 新增 `grok-` 前缀路由 |
| `backend/pipeline/gemini_image_gen.py` | 修复 tryallai endpoint URL (加 trailing `/`); 修复 Bearer auth 逻辑; 新增 `_extract_svg_structure()` 避免给 image model 喂 raw SVG XML |

### 未改动的文件

`server.py`、`generate-image.ts`、`generate-prompt.ts`、`ImageGenPanel.astro` 等上一轮已提交的文件保持不变，本次不涉及。

## 503 根因

```
topology.ts 发送 { model: "gemini-2.5-flash" }
  → server.py 用该 model 调 ai_engine
    → ai_engine 路由到 gemini provider
      → tryallai 代理返回 503: 分组 画图分组 下模型 gemini-2.5-flash 无可用渠道
```

修复后: 前端不再发送 model 字段 → 后端从 .env 读取 `DEFAULT_TOPOLOGY_MODEL=claude-opus-4-6`

12更新：

> **PR Title**: feat(edge-routing): Neural-network level advanced edge routing system + Grok/topology prompt enhancement
> **Parallel Note**: Claude (this PR) handles Tasks T1-T10 (edge routing core). Codex handles T11-T20 (frontend polish + docs).

---

## Completed in This PR (T1-T10)

### T1: Enhanced Edge Schema (types.ts) -- DONE
- **File**: `src/lib/elk/types.ts` (MODIFIED)
- Added: `EdgeRoutingMode`, `EdgeLineStyle`, `EdgeArrowType`, `EdgeDirectionality`, `EdgeSemanticType`
- Added: `AdvancedEdgeProperties` interface (routing, lineStyle, strokeDasharray, edgeLabels, curvature, etc.)
- Added: `EdgeLabelConfig` interface for math labels on arrows
- Added: `EDGE_SEMANTIC_DEFAULTS` mapping semantic types to visual styles
- Enhanced `ElkEdge` with optional `advanced?: AdvancedEdgeProperties`
- Enhanced `ScaffoldConnection` with optional `advanced?: AdvancedEdgeProperties`

### T2: Advanced Topology System Prompt -- DONE
- **File**: `backend/pipeline/edge_routing_prompts.py` (NEW)
- **File**: `backend/pipeline/topology_gen.py` (MODIFIED)
- Created `ADVANCED_EDGE_ROUTING_SYSTEM_PROMPT` with:
  - 12 semantic edge types (data_flow, gradient_flow, skip_connection, fan_out, fan_in, etc.)
  - Full "advanced" field schema documentation
  - JSON examples for skip connections, gradient flow, labeled edges, bidirectional, fan-out
  - Compound node (group) with hierarchy instructions
  - Rules: every non-trivial edge MUST have advanced field
- Integrated into `generate_topology()` via `get_topology_prompt_with_edge_routing()`

### T3: Enhanced Grok Prompt Engineering -- DONE
- **File**: `backend/pipeline/edge_routing_prompts.py` (NEW)
- **File**: `backend/pipeline/gemini_image_gen.py` (MODIFIED)
- Created `GROK_EDGE_ROUTING_SYSTEM_ADDON` with precise arrow rendering instructions:
  - Orthogonal routing description for image prompts
  - Fan-out/fan-in visual descriptions
  - Dashed/dotted arrow descriptions
  - Bidirectional arrow descriptions
  - Curved/spline arrow descriptions
  - Labeled arrow descriptions with positioning
  - Cross-boundary arrow descriptions
- Integrated into `generate_prompt_with_grok()` via `get_grok_prompt_with_edge_routing()`

### T4: ELK Layout API Enhancement -- DONE
- **File**: `src/pages/api/layout.ts` (MODIFIED)
- Added `elk.hierarchyHandling: INCLUDE_CHILDREN` for cross-boundary edges
- Added `elk.layered.crossingMinimization.strategy: LAYER_SWEEP`
- Added `elk.layered.nodePlacement.strategy: NETWORK_SIMPLEX`
- Added `elk.layered.considerModelOrder.strategy: NODES_AND_EDGES`
- Edge sanitization now preserves `advanced` and `labels` properties

### T5: Enhanced Skeleton SVG Renderer (to-svg.ts) -- DONE
- **File**: `src/lib/elk/to-svg.ts` (MODIFIED - full rewrite)
- Renders dashed edges (gradient_flow, optional_path, inference_only)
- Renders dotted edges (attention)
- Renders bidirectional arrows (double arrowheads)
- Renders curved edges using quadratic bezier (skip_connection, residual)
- Renders edge labels with background rectangles
- Dynamic SVG marker generation per edge color
- Semantic type -> visual style mapping

### T6: Scaffold Builder Edge Properties -- DONE
- **File**: `backend/pipeline/scaffold_builder.py` (MODIFIED)
- Extracts `advanced` properties from topology edges
- Maps semantic types to scaffold connection styles (arrow/dashed/dotted/bidirectional)
- Extracts edge labels for scaffold connections

### T7: Verification & Diff Review -- DONE
- All 6 modified files verified with `git diff --stat`
- 1 new file (`edge_routing_prompts.py`) verified
- No content from previous version lost

12更新：
astro-svgfigure v3: 3-Step Pipeline 修复 + 完善

> **核心流程**: Text → LLM Topology JSON → ELK.js Layout → Skeleton SVG → (用户确认) → Grok 4 Prompt → Gemini 3 Pro Image
> **技术栈**: Astro 5 + astro-pure 主题 + UnoCSS + ELK.js + Grok 4 + Gemini 3 Pro Image + Python FastAPI
> **分支**: `fix/skeleton-render-and-plan` — Claude Opus 负责前 10 个任务, Codex 并行后 10 个任务

---

## 0. 当前已确认的架构 (commit ba6a4cd 起)

3-Step Pipeline:
1. **Step 1**: 用户输入 method text → Claude Opus LLM 生成拓扑 JSON → ELK.js 约束布局 → Skeleton SVG (用户确认组件无缺失)
2. **Step 2**: Grok 4 从 Skeleton SVG + method text 反推专业 AI drawing prompt (用户可编辑)
3. **Step 3**: Gemini 3 Pro Image 根据 prompt 生成科研级图片 (4K PNG)

两个按钮:
- `Step 1: Generate Skeleton` — 执行 Step 1
- `Step 2→3: Generate Figure` — 执行 Step 2 + Step 3

---

## 1. 关键 Bug 修复 (已完成 ✅)

### BUG-1: Skeleton SVG 完全没渲染
**症状**: 运行 Step 1 后, SVG Preview 区域仍显示 "Skeleton SVG will appear after ELK layout (Step 2)", skeleton 从未出现在界面上
**根因**:
- `generate/index.astro` 的 Step 1 完成后调用 `preview?.show(currentSkeletonSvg)`, 但 `show()` 设置的是 `finalSvg` (Final tab 内容)
- `SvgPreview.astro` 默认 `activeTab = 'final'`, skeleton 被存到了 Final tab
- 用户切到 Skeleton tab 时看到空白, 因为 `showSkeleton()` 从未被调用
- 空白提示文字 "Step 2" 应为 "Step 1" (3-step pipeline 里 ELK layout 是 Step 1)

**修复** (3 个文件):
| 文件 | 改动 |
|------|------|
| `src/pages/generate/index.astro` | Step 1 完成后同时调用 `showSkeleton()` 和 `show()`, 并自动切换到 Skeleton tab |
| `src/components/pipeline/SvgPreview.astro` | 默认 `activeTab = 'skeleton'`, Skeleton tab 设为默认高亮 |
| `src/components/pipeline/SvgPreview.astro` | 空白提示文字 "Step 2" → "Step 1", "NanoBanana" → "Gemini 3" |

---

## 2. 接下来的 20 个任务 (Claude Opus: T1-T10, Codex: T11-T20)

### Claude Opus 负责: T1 ~ T10

| # | 任务 | 涉及文件 | 说明 |
|---|------|---------|------|
| T1 | ✅ 修复 Skeleton SVG 渲染 | `src/pages/generate/index.astro`, `src/components/pipeline/SvgPreview.astro` | 见上方 BUG-1 |
| T2 | SVG 手动编辑功能 | **新增** `src/components/pipeline/SvgEditor.astro`, **修改** `src/pages/generate/index.astro` | 用户可在 skeleton 预览上拖拽节点、调整连线, 使用内联 contenteditable 或轻量 SVG 交互 (不引入第三方 SVG editor) |
| T3 | Step 2→3 流程完善 | **修改** `src/pages/generate/index.astro` | Step 3 完成后自动切换到 Final tab, 显示 Gemini 3 生成的图片 |
| T4 | Grok prompt 编辑区域增强 | **修改** `src/pages/generate/index.astro` | prompt 区域加载时显示 "waiting for Step 2...", Step 2 完成后高亮提示用户可编辑, 编辑后 regenerate 只重跑 Step 3 |
| T5 | 导出功能完善 | **修改** `src/components/pipeline/ExportPanel.astro` | 支持导出 Skeleton SVG, Final Image (PNG), Topology JSON, Prompt Text 四种格式 |
| T6 | 错误处理增强 | **修改** `src/components/pipeline/ErrorDisplay.astro`, `src/pages/generate/index.astro` | 各 step 错误提示区分: API key 缺失 / 后端未启动 / 超时 / 模型不可用 |
| T7 | 后端 topology prompt 优化 | **修改** `backend/pipeline/topology_gen.py` | 优化 LLM prompt 使输出 ELK JSON 的 edge 格式更准确 (sources/targets 数组) |
| T8 | ELK to-svg 渲染增强 | **修改** `src/lib/elk/to-svg.ts` | 支持 group/compound node 渲染, 改善 label 截断, 添加 legend |
| T9 | 健康检查页面完善 | **修改** `src/components/pipeline/HealthCheck.astro`, `src/pages/api/health.ts` | 检查后端 + API key 状态 |

likec4/likec4 Issues 参考清单 (批判性需求调研)

# # #  高相关性 (直接影响 ELK 布局 / SVG 渲染)

| Issue | 类型 | 标题 | 启示 |
|-------|------|------|------|
| # 1629 | feature | ELK Playground | 我们 Playground 应提供类似交互 |
| # 751 | wanted | Larger systems render poorly | 大图性能, 限制节点数 |
| # 1224 | bug | Relationship text overlapping | edge label 重叠, T8 需解决 |
| # 1210 | feature | Reposition relationship label | T2 SVG 编辑应支持 |
| # 663 | feature | Multiple relationships | 多边渲染 |
| # 86 | feature | Add hexagon shape | to-svg.ts 应支持更多 shape |
| # 1447 | feature | Snap to grid | SVG 编辑对齐 |
| # 1476 | bug | Overlapping text with self-call | 自循环边文字重叠 |
| # 1985 | bug | Memory out of bounds | 大图内存问题 |
| # 2033 | bug | Incorrect self-reference | 自引用边 |

# # #  中相关性 (UX / 功能设计参考)

| Issue | 类型 | 标题 |
|-------|------|------|
| # 2679 | bug | Draw.io export: person → ellipse |
| # 2674 | wanted | Logical grouping (loops, conditions) |
| # 2672 | bug | View title newlines |
| # 2671 | wanted | Secondary icons |
| # 2636 | wanted | Claude/Agent skill |
| # 2625 | bug | Editing manual layout → reference issues |
| # 2594 | wanted | Decentralized Model Resolution |
| # 2567 | wanted | Notes elements and info popovers |
| # 2566 | wanted | Animated diagrams |
| # 2562 | bug | watch: true HMR issue |
| # 2553 | bug | codegen react manual layouts |
| # 2505 | bug | Images in markdown |
| # 2475 | wanted | Notes on sequence diagrams |
| # 2468 | wanted | Rank constraints for layout |
| # 2465 | bug | Nested element rank positioning |
| # 2462 | bug | Relationship wrong nesting level |
| # 2422 | bug | URLs don't open in preview |
| # 2402 | wanted | Lighter syntax for beginners |
| # 2399 | bug | Node not visualized |
| # 2378 | wanted | Import via URLs |
| # 2375 | wanted | Conditional relation |
| # 2308 | wanted | Sequence diagrams in PlantUML |
| # 2286 | wanted | Export PNG from command palette |
| # 2271 | wanted | Glob/regexp predicates |
| # 2264 | wanted | Custom element properties |
| # 2251 | wanted | Alphabetise views |
| # 2238 | wanted | Element in two parents |
| # 2165 | wanted | Icon override specificity |
| # 2143 | wanted | Accurate tag text color |
| # 2109 | wanted | Default relationship styling |
| # 2107 | bug | Groups ignored in rendering |
| # 2101 | wanted | Accurate background color |
| # 2100 | wanted | Text contrast adaptation |
| # 2094 | bug | Empty popup for typed relationships |
| # 2091 | wanted | Choose views for landing page |
| # 2090 | wanted | Instance count on deployment view |
| # 2088 | wanted | Brand Identity & Design System |
| # 2087 | wanted | GitHub Language Support |
| # 2051 | feature | Text color |
| # 2048 | wanted | View styling by metadata |
| # 2009 | wanted | Self-relationship |
| # 1998 | wanted | IF Logic for routing |
| # 1993 | wanted | Direct link to element |
| # 1987 | wanted | Reusable constants |
| # 1946 | bug | Multi-file merge ambiguity |
| # 1919 | wanted | Showcase projects |
| # 1915 | bug | Relationship styling not applied |
| # 1828 | wanted | Global groups |
| # 1724 | feature | FontAwesome icons |
| # 1722 | feature | Include without parent |
| # 1718 | feature | Reset control points → straight lines |
| # 1717 | feature | Custom icon size |
| # 1709 | feature | Multiple versions/iterations |
| # 1708 | feature | Tags not inherited |
| # 1705 | feature | Include/reference element in another |
| # 1676 | bug | Unexpected crossing lines |
| # 1675 | feature | Inline tag declaration |
| # 1640 | feature | Metadata viewer: yaml + links |
| # 1639 | feature | Export: background, filter |
| # 1608 | bug | Exclude issue in deployment view |
| # 1599 | feature | Dynamic deployment views |
| # 1595 | bug | Residual elements |
| # 1593 | feature | Fuzzy search |
| # 1569 | feature | Border radius style |
| # 1568 | feature | Border width style |
| # 1565 | bug | Build warnings with dot-bin |
| # 1483 | bug | Many descendants handling |
| # 1480 | feature | Global styles for relationships |
| # 1474 | feature | Import via git/file ref |
| # 1465 | feature | Data flow visualization |
| # 1460 | bug | GraphvizWasm build error |
| # 1459 | feature | JetBrains support |
| # 1453 | feature | Mermaid → dynamic view |
| # 1414 | bug | Exclusion rules nested instances |
| # 1365 | feature | Export relationships overlay |
| # 1349 | feature | Specification inheritance |
| # 1343 | bug | No filesystem in browser |
| # 1276 | feature | Colors limitation |
| # 1257 | feature | More style customization |
| # 1234 | wanted | Style relationships by tag |
| # 1209 | feature | Drawing tunnels |
| # 1194 | docs | Deploy with GitLab |
| # 1148 | bug | Memory access out of bounds |
| # 1003 | feature | Nest in multiple parents |
| # 988 | feature | Error reporting for parallel steps |
| # 960 | feature | Externally declared element rendering |
| # 853 | bug | Dynamic view relationship hidden |



13更新：
astro-svgfigure v3: Phase 6 — Interactive ELK Editor

> **核心理念**: 正向 Pipeline（文本→拓扑JSON→ELK布局→可交互骨架SVG→科研图）  
> **本轮重点**: 让用户能**手动调整** ELK skeleton SVG（拖拽节点、编辑标签、调整大小），借鉴 ReactFlow + ELK.js 的交互灵活性  
> **技术栈**: Astro 5 + astro-pure 主题 + UnoCSS + ELK.js (前端交互) + Grok 4 + Gemini 3 + Python 后端  
> **关键修复**: skeleton SVG 渲染失败 + 用户无法编辑骨架

---

## 0. 问题诊断（基于最新 commit 3d40437）

### 已发现的 BUG

| # | 问题 | 严重度 | 原因分析 |
|---|------|--------|---------|
| B1 | skeleton SVG 完全没渲染 | 🔴 Critical | `@elk/to-svg` 导入路径可能在 Vite/Astro SSR 中失败；`elkToSvg()` 返回空字符串但无错误上报 |
| B2 | 用户无法编辑 skeleton SVG | 🔴 Critical | `SvgPreview` 仅用 `innerHTML` 静态渲染，没有交互层（无拖拽、无编辑、无 resize） |
| B3 | ELK.js 没有在前端体现 | 🟡 Major | ELK 只在 `/api/layout` 后端跑，前端无 re-layout 能力；与 ReactFlow+ELK 的交互模式差距巨大 |
| B4 | Step 1 后只能点 Step 2, 无编辑流程 | 🟡 Major | Pipeline 设计缺少 "编辑→确认→继续" 的循环 |

### ReactFlow + ELK.js 参考思路 (reactflow.dev/examples/layout/elkjs)

ReactFlow 的做法：
1. ELK.js 异步计算布局 → 返回 `{ children: [{x, y, width, height}], edges: [{sections}] }`
2. 将 ELK 坐标映射为 ReactFlow 的 `position: {x, y}` → 渲染为可拖拽节点
3. 用户拖拽节点后，可重新触发 ELK re-layout
4. 支持多种布局方向 (horizontal/vertical) 切换

**我们的 Astro 适配方案**: 不用 React，而是用 **原生 SVG + 事件监听** 实现类似交互：
- SVG `<rect>` 节点支持 mousedown → drag
- 双击 `<text>` 进入 inline 编辑（`<foreignObject>` + `<input>`）
- 节点边角拖拽 resize
- 边自动跟随节点移动（重新计算 bendPoints）
- "Re-layout" 按钮: 将修改后的 JSON 重新提交给 ELK

---

## 1. Phase 6 — Claude 本轮 10 任务清单

| # | 文件 | Op | 说明 | 优先级 |
|---|------|----|------|--------|
| T1 | `src/lib/elk/to-svg.ts` | ✏️ | **修复** skeleton SVG 渲染: 确保 `elkToSvg()` 正确生成 SVG 且不返回空串 | P0 |
| T2 | `src/lib/elk/interactive-svg.ts` | 🆕 | **核心**: 交互式 SVG 引擎 — 拖拽节点、编辑标签、resize、边跟随、zoom/pan | P0 |
| T3 | `src/components/pipeline/SvgPreview.astro` | ✏️ | 集成交互 SVG: 替换静态 innerHTML 为 interactive-svg 引擎 | P0 |
| T4 | `src/components/pipeline/SvgEditor.astro` | 🆕 | SVG 编辑工具栏: 添加节点、删除节点、修改颜色、导出 JSON、Re-layout 按钮 | P0 |
| T5 | `src/pages/generate/index.astro` | ✏️ | 集成编辑器+编辑→确认→继续流程 | P1 |
| T6 | `src/pages/api/layout.ts` | ✏️ | 修复 `@elk/to-svg` 导入、增加 re-layout 端点支持用户修改后的 JSON | P0 |
| T7 | `src/lib/elk/elk-to-interactive.ts` | 🆕 | ELK layouted JSON ↔ Interactive SVG 双向转换 | P1 |
| T8 | `src/components/pipeline/PipelineSteps.astro` | ✏️ | 添加 "Edit Skeleton" 步骤指示 | P2 |
| T9 | `src/components/pipeline/ElkOptions.astro` | ✏️ | ELK 参数实时调整面板（算法、间距、方向）+ re-layout 联动 | P2 |
| T10 | `plan.md` | ✏️ | 更新为本文件 (v11) + likec4 issues 调研清单 | P1 |

### Codex 并行 10 任务（建议）

| # | 文件 | Op | 说明 |
|---|------|----|------|
| C1 | `src/pages/playground/index.astro` | ✏️ | Playground 页面使用交互 SVG 引擎 |
| C2 | `src/components/gallery/FigureCard.astro` | ✏️ | Gallery 卡片展示 |
| C3 | `backend/pipeline/topology_gen.py` | ✏️ | 优化拓扑 JSON 生成质量 |
| C4 | `src/lib/elk/presets.ts` | ✏️ | 丰富 ELK 预设 |
| C5 | `src/lib/elk/examples.ts` | ✏️ | 添加更多示例 |
| C6 | `backend/pipeline/svg_validator.py` | ✏️ | SVG 验证增强 |
| C7 | `src/components/pipeline/HealthCheck.astro` | ✏️ | 健康检查改进 |
| C8 | `README.md` | ✏️ | 更新文档 |
| C9 | `src/pages/index.astro` | ✏️ | 首页展示交互 Demo |
| C10 | 单元测试 | 🆕 | ELK + Interactive SVG 测试 |

---

## 2. 文件变更清单（Phase 6）

**标记**: 🆕 新增 | ✏️ 修改 | 🗑️ 删除 | 📌 不动

### 修改的文件

| # | 文件路径 | Op | 变更内容 |
|---|----------|----|---------|
| 1 | `src/lib/elk/to-svg.ts` | ✏️ | 修复空串 bug、增加错误处理、viewBox 修正 |
| 2 | `src/components/pipeline/SvgPreview.astro` | ✏️ | 集成 interactive-svg 引擎替换静态 innerHTML |
| 3 | `src/pages/generate/index.astro` | ✏️ | 添加 SvgEditor 组件、编辑→确认→继续流程 |
| 4 | `src/pages/api/layout.ts` | ✏️ | 修复导入路径、添加 re-layout 支持 |
| 5 | `src/components/pipeline/PipelineSteps.astro` | ✏️ | Step 1.5: Edit Skeleton 指示 |
| 6 | `src/components/pipeline/ElkOptions.astro` | ✏️ | re-layout 联动 |
| 7 | `plan.md` | ✏️ | 本文件 |

### 新增的文件

| # | 文件路径 | Op | 说明 |
|---|----------|----|------|
| 8 | `src/lib/elk/interactive-svg.ts` | 🆕 | 交互式 SVG 引擎核心 |
| 9 | `src/lib/elk/elk-to-interactive.ts` | 🆕 | ELK ↔ Interactive 转换 |
| 10 | `src/components/pipeline/SvgEditor.astro` | 🆕 | SVG 编辑工具栏组件 |

---

## 3. 交互式 SVG 编辑器架构设计

```
┌─── ELK Layout Engine (后端/api/layout) ───┐
│  topology JSON → elk.layout() → layouted   │
│  生成初始 {x,y,w,h} 坐标                   │
└──────────┬────────────────────────────────┘
           │ layouted JSON
           ▼
┌─── interactive-svg.ts (前端) ────────────┐
│  渲染 SVG → 每个节点是可交互元素          │
│                                           │
│  ┌─── Node ────────────────────┐         │
│  │ <rect> 可拖拽               │         │
│  │ <text> 双击编辑标签         │         │
│  │ resize handle 四角拖拽      │         │
│  └─────────────────────────────┘         │
│                                           │
│  ┌─── Edge ────────────────────┐         │
│  │ <path> 自动跟随源/目标节点  │         │
│  │ bendPoints 自动更新         │         │
│  └─────────────────────────────┘         │
│                                           │
│  API:                                     │
│  - initInteractiveSvg(container, layout)  │
│  - getModifiedLayout() → JSON             │
│  - reLayout(options) → fetch /api/layout  │
│  - toStaticSvg() → SVG string             │
│  - addNode() / removeNode()               │
│  - setZoom() / fitView()                  │
└───────────────────────────────────────────┘
           │
           ▼
┌─── SvgEditor.astro (工具栏) ─────────────┐
│  [Re-Layout] [Add Node] [Delete]          │
│  [Export SVG] [Export JSON] [Confirm→S2]  │
│  ELK Options: algorithm, direction, etc.  │
└───────────────────────────────────────────┘
```

---

## 4. likec4/likec4 Issues 调研清单（竞品分析 + 我们可借鉴的教训）

> 来源: github.com/likec4/likec4/issues pages 1-6, 共 130 open issues
> 目的: 批判性分析，避免踩同样的坑，吸收好的 feature request

### Page 1 (最新)

| # | Issue | 类型 | 与我们的关联 |
|---|-------|------|-------------|
| # 2679 | Draw.io export: shape "person" → ellipse | bug | SVG 导出时形状映射要正确 |
| # 2674 | Support logical grouping (loops, conditions) in Dynamic Views | feature | 我们的 ELK 也需支持条件/循环分组 |
| # 2673 | Standalone Language Server fails to run | bug | N/A (LSP 相关) |
| # 2672 | "View title cannot contain newlines" with implicitViews | bug | 文本处理要严格转义 |
| # 2671 | Add secondary icons to specialize types icons | feature | 节点图标叠加功能 |
| # 2669 | View title cannot contain newlines | bug | 同 #2672 |
| # 2640 | Predicate to add all ancestors of visible elements | feature | 视图过滤逻辑 |
| # 2637 | Custom color broken for tags since 1.49.0 | bug | 升级后颜色系统回归测试 |
| # 2636 | Claude/Agent skill | feature | LLM 集成到架构工具 |
| # 2626 | Follow up: iconColor does not apply to custom icons | bug | 自定义颜色体系要覆盖全面 |
| # 2625 | Editing manual layout → reference resolvement issues | bug | 手动布局编辑的稳定性 |
| # 2594 | Support Decentralized Model via Component `src` attribute | feature | N/A |

### Page 2

| # | Issue | 类型 | 与我们的关联 |
|---|-------|------|-------------|
| # 2533 | Support tags in relationships kind | feature | 边的元数据/标签系统 |
| # 2505 | Images don't display in markdown | bug | Markdown 中图片路径处理 |
| # 2475 | Notes directly displayed on sequence diagrams | feature | 图上注释功能 |
| # 2468 | Support rank constraints for layout in dynamic view | feature | ELK rank 约束 |
| # 2467 | Inheriting label from model in dynamic view | feature | 标签继承机制 |
| # 2465 | Rank for nested elements moves it outside the box | bug | ELK 嵌套布局 bug |
| # 2462 | Deployment view: Relationship to wrong nesting level | bug | 层级关系边的正确连接 |
| # 2422 | Extension Diagram Preview URLs do not open | bug | N/A (VSCode 扩展) |
| # 2402 | Lighter-weight syntax for beginner | feature | 降低使用门槛 |
| # 2399 | Node not visualized on deployment view | bug | 节点渲染缺失 |
| # 2378 | Allow importing files via URLs | feature | 远程资源加载 |
| # 2375 | Conditional relation in dynamic views | feature | 条件逻辑 |

### Page 3

| # | Issue | 类型 | 与我们的关联 |
|---|-------|------|-------------|
| # 2143 | Accurate text color for tags | feature | 文本颜色对比度 |
| # 2109 | Default (no kind) relationship styling | feature | 默认边样式 |
| # 2107 | Groups in view containers ignored during rendering | bug | 分组渲染 |
| # 2101 | Accurate background color for elements | feature | 颜色准确性 |
| # 2100 | Adapt text color based on contrast with background | feature | WCAG 对比度自动计算 |
| # 2094 | Relationships with kind show empty popup | bug | N/A |
| # 2091 | Specify which views in landing page | feature | 视图选择器 |
| # 2090 | Number of instances on deployment view | feature | N/A |
| # 2088 | Create Brand Identity & Design System | feature | 设计系统化 |
| # 2087 | Add GitHub Language Support | feature | N/A |
| # 2051 | Text color | feature | 同 #2143 |
| # 2048 | View styling by metadata | feature | 基于数据的样式 |

### Page 4

| # | Issue | 类型 | 与我们的关联 |
|---|-------|------|-------------|
| # 1919 | Showcase projects | feature | Gallery/案例展示 |
| # 1915 | Styling of relationships not correctly applied | bug | 边样式一致性 |
| # 1828 | Global groups | feature | 全局分组 |
| # 1817 | Search deploymentNode | feature | 搜索功能 |
| # 1775 | Searchable metadata | feature | 元数据搜索 |
| # 1724 | Support fontawesome icons | feature | 图标系统扩展 |
| # 1722 | Include predicate without parent | feature | N/A |
| # 1718 | Resetting control points → straight line relationships | bug | 控制点重置 bug |
| # 1717 | Customise icon size | feature | 图标尺寸可配 |
| # 1709 | Multiple versions/iterations of architecture | feature | 版本历史 |
| # 1708 | Tags are not inherited | bug | 标签继承 |
| # 1705 | Include/reference element inside another | feature | 元素引用 |

### Page 5

| # | Issue | 类型 | 与我们的关联 |
|---|-------|------|-------------|
| # 1565 | Warnings on build --use-dot-bin | bug | N/A |
| # 1483 | Handle lot of descendants in deployment view | feature | 大量节点性能 |
| # 1480 | Support global styles in relationship browser | feature | 全局样式 |
| # 1476 | Overlapping text in self-call dynamic view | bug | 文本重叠处理 |
| # 1475 | Unable to include relation into deployment view | bug | N/A |
| # 1474 | Import via git or file reference | feature | 外部资源导入 |
| # 1465 | Support for visualizing data flows | feature | 数据流可视化 |
| # 1460 | GraphvizWasmAdapter Build error | bug | WASM 构建兼容性 |
| # 1459 | Likec4 support for Jetbrains | feature | N/A |
| # 1453 | Generate dynamic view from mermaid sequence | feature | 从其他格式生成 |
| # 1447 | Snap to grid | feature | 网格对齐 |
| # 1414 | Exclusion rules not work for nested deployment | bug | N/A |

### Page 6

| # | Issue | 类型 | 与我们的关联 |
|---|-------|------|-------------|
| # 960 | Rendering externally declared element in container | bug | 跨容器渲染 |
| # 853 | Dynamic view relationship hidden | bug | 边被隐藏 |
| # 751 | Larger systems render poorly | bug | 大规模布局性能 |
| # 663 | Multiple relationships | feature | 多重边处理 |
| # 86 | Add hexagon shape | feature | 更多形状支持 |

### 从 likec4 issues 中提取的关键教训

1. **手动布局编辑是核心痛点** (#2625, #1718, #1447) — 我们必须做好交互编辑
2. **大规模渲染性能** (#751, #1483) — 超过 50 个节点时要考虑虚拟化
3. **颜色/文本对比度** (#2143, #2100, #2101) — WCAG 对比度自动调整
4. **分组/嵌套渲染** (#2107, #2465) — ELK 嵌套子图要正确渲染
5. **网格对齐** (#1447) — Snap to grid 是高频需求
6. **文本重叠** (#1476) — 布局算法要防止标签遮盖
7. **边样式一致性** (#1915, #2109) — 边的样式系统要统一

---

## 5. PR 说明

**Branch**: `claude/phase6-interactive-elk-editor`

**PR Title**: `feat(editor): Phase 6 — Interactive ELK Skeleton Editor + Fix Rendering`

**Changes**:
- FIX: `src/lib/elk/to-svg.ts` — 修复 skeleton SVG 空串渲染 bug
- FIX: `src/pages/api/layout.ts` — 修复 `@elk/to-svg` 导入路径
- NEW: `src/lib/elk/interactive-svg.ts` — 交互式 SVG 引擎 (拖拽/编辑/resize)
- NEW: `src/lib/elk/elk-to-interactive.ts` — ELK ↔ Interactive 双向转换
- NEW: `src/components/pipeline/SvgEditor.astro` — 编辑器工具栏
- MOD: `src/components/pipeline/SvgPreview.astro` — 集成交互引擎
- MOD: `src/pages/generate/index.astro` — 编辑→确认→继续流程
- MOD: `src/components/pipeline/PipelineSteps.astro` — Edit 步骤
- MOD: `src/components/pipeline/ElkOptions.astro` — Re-layout 联动
- MOD: `plan.md` — v11 含 likec4 issues 调研

**Codex 并行注意**:
- 本 PR 主要修改 `src/lib/elk/`, `src/components/pipeline/`, `src/pages/generate/`
- 不修改 `backend/`, `packages/pure/`, `public/`
- server.py 不动
- Codex 可安全并行修改: `src/pages/playground/`, `src/pages/gallery/`, `README.md`, 后端文件

---

## 6. 历史 Phase 记录

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 配置+ELK核心 (001-010) | ✅ |
| Phase 1 | Backend AI Engine (013-022) | ✅ |
| Phase 2 | Generate页面+Pipeline组件 (023-032) | ✅ |
| Phase 3 | 修复502+Pipeline联调 (T1-T10) | ✅ |
| Phase 4 | Gallery+Playground+补全API (T1-T10) | ✅ |
| Phase 4.5 | 修复ELK layout 500 + topology验证 | ✅ |
| Phase 5 | Grok 4 + Gemini 3 Image (步骤5) | ✅ |
| **Phase 6** | **Interactive ELK Editor (本轮)** | **🔄 进行中** |


14更新：
编辑确认 + 主题色统一

涉及: interactive-svg.ts, elk-to-interactive.ts, to-svg.ts, SvgPreview.astro, SvgEditor.astro, PipelineSteps.astro, generate/index.astro
