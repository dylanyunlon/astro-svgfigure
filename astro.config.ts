import { rehypeHeadingIds } from '@astrojs/markdown-remark'
import vercel from '@astrojs/vercel'
import AstroPureIntegration from 'astro-pure'
import { defineConfig, fontProviders } from 'astro/config'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'

// Local integrations
import rehypeAutolinkHeadings from './src/plugins/rehype-auto-link-headings.ts'
// Shiki
import {
  addCollapse,
  addCopyButton,
  addLanguage,
  addTitle,
  updateStyle
} from './src/plugins/shiki-custom-transformers.ts'
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerRemoveNotationEscape
} from './src/plugins/shiki-official/transformers.ts'
import config from './src/site.config.ts'

// https://astro.build/config
export default defineConfig({
  // [Basic]
  site: 'https://astro-pure.js.org',
  // Deploy to a sub path
  // https://astro-pure.js.org/docs/setup/deployment#platform-with-base-path
  // base: '/astro-pure/',
  trailingSlash: 'never',
  // root: './my-project-directory',
  server: { host: true },

  // [Adapter]
  // https://docs.astro.build/en/guides/deploy/
  adapter: vercel(),
  output: 'server',
  // Local (standalone)
  // adapter: node({ mode: 'standalone' }),
  // output: 'server',

  // [Assets]
  image: {
    responsiveStyles: true,
    service: {
      entrypoint: 'astro/assets/services/sharp'
    }
  },

  // [Markdown]
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      [rehypeKatex, {}],
      rehypeHeadingIds,
      [
        rehypeAutolinkHeadings,
        {
          behavior: 'append',
          properties: { className: ['anchor'] },
          content: { type: 'text', value: '#' }
        }
      ]
    ],
    // https://docs.astro.build/en/guides/syntax-highlighting/
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark'
      },
      transformers: [
        // Two copies of @shikijs/types (one under node_modules
        // and another nested under @astrojs/markdown-remark → shiki).
        // Official transformers
        // @ts-ignore this happens due to multiple versions of shiki types
        transformerNotationDiff(),
        // @ts-ignore this happens due to multiple versions of shiki types
        transformerNotationHighlight(),
        // @ts-ignore this happens due to multiple versions of shiki types
        transformerRemoveNotationEscape(),
        // Custom transformers
        // @ts-ignore this happens due to multiple versions of shiki types
        updateStyle(),
        // @ts-ignore this happens due to multiple versions of shiki types
        addTitle(),
        // @ts-ignore this happens due to multiple versions of shiki types
        addLanguage(),
        // @ts-ignore this happens due to multiple versions of shiki types
        addCopyButton(2000), // timeout in ms
        // @ts-ignore this happens due to multiple versions of shiki types
        addCollapse(15) // max lines that needs to collapse
      ]
    }
  },

  // [Integrations]
  integrations: [
    // astro-pure will automatically add sitemap, mdx & unocss
    // sitemap(),
    // mdx(),
    AstroPureIntegration(config)
  ],

  // [Experimental]
  experimental: {
    // Allow compatible editors to support intellisense features for content collection entries
    // https://docs.astro.build/en/reference/experimental-flags/content-intellisense/
    contentIntellisense: true,
    // Enable SVGO optimization for SVG assets
    // https://docs.astro.build/en/reference/experimental-flags/svg-optimization/
    svgo: true,
    // Enable font preloading and optimization
    // https://docs.astro.build/en/reference/experimental-flags/fonts/
    fonts: [
      {
        provider: fontProviders.fontshare(),
        name: 'Satoshi',
        cssVariable: '--font-satoshi',
        // Default included:
        // weights: [400],
        // styles: ["normal", "italics"],
        // subsets: ["cyrillic-ext", "cyrillic", "greek-ext", "greek", "vietnamese", "latin-ext", "latin"],
        // fallbacks: ["sans-serif"],
        styles: ['normal', 'italic'],
        weights: [400, 500],
        subsets: ['latin']
      }
    ]
  },

  // [Vite] ELK.js bundled needs explicit optimizeDeps
  // Ref: https://github.com/kieler/elkjs
  vite: {
    plugins: [
      (await import('vite-plugin-glsl')).default(),
    ],
    optimizeDeps: {
      include: ['elkjs/lib/elk.bundled.js'],
      exclude: ['pixi.js']
    },
    build: {
      rollupOptions: {
        external: ['tweedle.js', 'fs', 'path', 'lodash-es', /^@theatre\/.*/, /^@pixi\/.*/],
        onwarn(warning, warn) {
          // 忽略 unresolved import warnings (upstream 依赖)
          if (warning.code === 'UNRESOLVED_IMPORT') return;
          warn(warning);
        },
      }
    },
    resolve: {
      alias: {
        '@theatre/core': './upstream/theatre-js/core/src',
        '@theatre/utils': './upstream/theatre-js/utils/src',
        '@theatre/dataverse': './upstream/theatre-js/dataverse/src',
        'worker:./basis.worker.ts': './src/lib/stubs/empty-worker.ts',
        'worker:./ktx.worker.ts': './src/lib/stubs/empty-worker.ts',
        'worker:./checkImageBitmap.worker.ts': './src/lib/stubs/empty-worker.ts',
        'worker:./loadImageBitmap.worker.ts': './src/lib/stubs/empty-worker.ts',
      }
    }
  }
})
