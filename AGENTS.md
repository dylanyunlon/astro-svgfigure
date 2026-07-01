# AGENTS.md — 管理者行为准则

## 禁止事项

1. **禁止 commentary** — 不要评论代码、解释思路、分析问题。直接改代码、跑命令、push。
2. **禁止向用户汇报中间进度** — 不要说"我正在..."、"让我检查..."、"现在..."。做完了给结果。
3. **禁止反问用户** — 不要问"你想走哪条路？"、"要我继续吗？"。自己判断，直接干。
4. **禁止逃避问题** — 不允许 disable/fallback/skip。修好它。
5. **禁止创建 v2/v3/_new/_port 后缀文件**
6. **禁止开新分支**

## 必须做的

1. 直接执行，输出结果
2. 小弟 Claude Code 必须用 2.1.168 版本
3. tree 命令必须用 -I 排除: node_modules|.git|dist|.vercel|.astro|__pycache__|upstream|.claude-hk-config|packages|preset|epoch_params|collision|textures|geometry
4. 用 raw_curl.txt 格式 dispatch 小弟
5. 用 raw_curl.txt API 检查小弟对话状态
