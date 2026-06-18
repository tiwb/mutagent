# CodeBlock 代码语法高亮

**状态**：✅ 已完成
**日期**：2026-06-18
**类型**：功能设计

## 需求

1. `CodeBlockItem` 渲染时对代码做语法高亮，按语言选择对应 grammar
2. 颜色体系适配 mutagent 默认暗色主题，后续可扩展到亮色主题
3. 流式增量场景下，代码内容变化时高亮结果实时更新
4. 不认识的语言降级为纯文本渲染（不改现有行为）
5. 不引入重量级依赖，包体积可控

## 关键参考

- `mutagent/frontend/src/components/CodeBlock.tsx` — 当前纯文本渲染，需改造
- `mutagent/frontend/src/components/CodeBlock.css` — 当前样式，需追加 token class 规则
- `mutagent/frontend/package.json` — 依赖清单（React 19 + TypeScript + Vite）
- `mutagent/src/mutagent/webui/_messages.py` — `CodeBlockItem` dataclass（`code: str`、`language: str`）
- `mutagent/src/mutagent/webui/_markdown_parser.py` — `IncrementalMarkdownParser`，代码块拆分点
- `mutagent/src/mutagent/webui/_conversation.py` — 流式 text_delta 路径，code 字段增量更新
- `mutagent/docs/specifications/feature-incremental-markdown-parser.md` — 前置依赖（已完成）
- `mutgui/frontend/src/plugins/theme-dark/dark.css` — 暗色主题 token 定义
- `mutgui/frontend/src/core/base.css` — 亮色默认 token（fallback）

## 设计方案

### 一、库选型：highlight.js

选择 **highlight.js v11**（`npm: highlight.js`），理由：

| 维度 | highlight.js | Prism.js | Shiki | react-syntax-highlighter |
|------|-------------|----------|-------|--------------------------|
| 体积（按需） | ~60KB / ~20KB gzip | ~30KB / ~10KB gzip | 数 MB | ~150KB+ |
| 流式兼容 | ✅ `hljs.highlight()` 同步，不完整代码也能高亮 | ✅ 需手动注册 | ❌ 异步 WASM 初始化 | ✅ 底层同 Prism/hljs |
| React 集成 | `dangerouslySetInnerHTML` | 需自行封装 | `codeToHtml` 异步 | 开箱即用但重 |
| 语言覆盖 | 190+ 种 | 200+ 种（需手动 import） | VS Code 全部 | 同底层 |
| 维护状态 | 活跃（18 年历史） | 活跃 | 活跃 | 维护缓慢 |

**关键决策**：highlight.js 用 `hljs.highlight(code, { language })` 纯数据 API（不是 DOM API `highlightElement`），在 React 中用 `useMemo` + `dangerouslySetInnerHTML` 渲染，避免 DOM 操作与 React 并发模式冲突。

### 二、集成方式

```tsx
// CodeBlock.tsx 改造示意
import hljs from 'highlight.js/lib/core';
// tree-shake: 只打包需要的语言
import python from 'highlight.js/lib/languages/python';
import javascript from 'highlight.js/lib/languages/javascript';
// ...

hljs.registerLanguage('python', python);
hljs.registerLanguage('javascript', javascript);
// ...

function CodeBlock({ code, language }: CodeBlockProps) {
  const html = useMemo(() => {
    const lang = language?.toLowerCase() || '';
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    if (lang === '') {
      // 未指定语言但 code 非空，走 auto 检测
      return hljs.highlightAuto(code).value;
    }
    // 不认识的语言：纯文本 + HTML 转义
    return hljs.highlight(code, { language: 'plaintext' }).value;
  }, [code, language]);

  return (
    <div className="mutagent-code-block">
      <div className="mutagent-code-block__fence">
        <span className="mutagent-code-block__fence-mark">```</span>
        {language && <span className="mutagent-code-block__fence-lang">{language}</span>}
      </div>
      <pre className="mutagent-code-block__code">
        <code
          className={language ? `language-${language}` : undefined}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </pre>
      <div className="mutagent-code-block__fence">
        <span className="mutagent-code-block__fence-mark">```</span>
      </div>
    </div>
  );
}
```

**降级策略**：
- 语言已知且已注册 → `hljs.highlight(code, { language })` 精准高亮
- 语言未指定（空字符串） → `hljs.highlightAuto(code)` 自动检测
- 语言不认识（已注册列表中无此语言） → `hljs.highlight(code, { language: 'plaintext' })` 纯文本 + 转义

### 三、语言选择

按需打包 10 种语言，覆盖常见场景：

| 语言 | highlight.js 标识 | 体积（约） | 理由 |
|------|-------------------|-----------|------|
| Python | `python` | ~7KB | 最常用 |
| JavaScript | `javascript` | ~12KB | Web 开发 |
| TypeScript | `typescript` | ~12KB | 前端主力 |
| Bash | `bash` | ~5KB | 命令行 |
| JSON | `json` | ~2KB | 数据格式 |
| XML/HTML | `xml` | ~6KB | 标记语言 |
| CSS | `css` | ~5KB | 样式 |
| SQL | `sql` | ~5KB | 数据库 |
| YAML | `yaml` | ~4KB | 配置 |
| Markdown | `markdown` | ~5KB | 文档 |

总计约 63KB（gzip 后约 20KB）。`highlight.js/lib/core` 本身约 18KB。

### 四、颜色表（暗色主题）

基于 mutgui dark theme（背景 `oklch(0.2393 0 0)` ≈ `#1F1F1F`），参考 VS Code Dark+ 的色彩语义设计，调整为与 mutgui accent（蓝 `oklch(0.55 0.17 240)`）协调的色板。

所有颜色通过 CSS 变量暴露，外部可覆盖。

| Token 类型 | CSS 变量 | Hex | OKLCH | 语义 | 对比度(AA) |
|-----------|----------|-----|-------|------|-----------|
| 基础文字 | `--mutagent-hl-text` | `#d4d4d4` | `oklch(0.84 0 0)` | 高亮容器内主文字，略亮于 mutgui-text | ~10:1 |
| Keyword | `--mutagent-hl-keyword` | `#6ba9e6` | `oklch(0.70 0.13 245)` | 语法关键词（if/else/return/def/import），偏冷蓝 | ~6:1 |
| String | `--mutagent-hl-string` | `#dca080` | `oklch(0.70 0.08 45)` | 字符串字面量，暖橙 | ~6:1 |
| Number | `--mutagent-hl-number` | `#b2d4a0` | `oklch(0.78 0.07 140)` | 数字字面量，柔和绿 | ~9:1 |
| Comment | `--mutagent-hl-comment` | `#7d8a75` | `oklch(0.58 0.04 140)` | 注释，低调灰绿，刻意低对比度 | ~4.5:1 |
| Function | `--mutagent-hl-function` | `#dcdaa0` | `oklch(0.83 0.07 100)` | 函数/方法名，暖黄高亮 | ~9:1 |
| Type/Class | `--mutagent-hl-type` | `#5ac8b8` | `oklch(0.72 0.08 180)` | 类型/类名，青绿 | ~7:1 |
| Builtin | `--mutagent-hl-builtin` | `#6ba9e6` | `oklch(0.70 0.13 245)` | 内置常量/函数，同 keyword | ~6:1 |
| Literal | `--mutagent-hl-literal` | `#6ba9e6` | `oklch(0.70 0.13 245)` | 布尔/None/null，同 keyword | ~6:1 |
| Variable | `--mutagent-hl-variable` | `#88c8f0` | `oklch(0.76 0.08 235)` | 参数/变量名，浅蓝 | ~8:1 |
| Regex | `--mutagent-hl-regex` | `#d47070` | `oklch(0.62 0.11 25)` | 正则表达式，偏红 | ~5.5:1 |
| Punctuation | `--mutagent-hl-punctuation` | `#cccccc` | `oklch(0.82 0 0)` | 标点/括号，同 mutgui-text | ~9:1 |
| Operator | `--mutagent-hl-operator` | `#cccccc` | `oklch(0.82 0 0)` | 运算符，同 punctuation | ~9:1 |
| Title | `--mutagent-hl-title` | `#5ac8b8` | `oklch(0.72 0.08 180)` | 类/函数定义名，同 type | ~7:1 |
| Attribute | `--mutagent-hl-attr` | `#88c8f0` | `oklch(0.76 0.08 235)` | 对象属性访问，同 variable | ~8:1 |
| Meta/Pragma | `--mutagent-hl-meta` | `#7d8a75` | `oklch(0.58 0.04 140)` | shebang/encoding 声明，同 comment | ~4.5:1 |
| Emphasis | `--mutagent-hl-emphasis` | `#d4d4d4` | `oklch(0.84 0 0)` | markdown **粗体** 等高亮文本 | ~10:1 |

**配色原则**：
- 冷色系（蓝/青/绿）= 语法结构：keywords、types、numbers
- 暖色系（橙/黄）= 数据与行为：strings、functions
- 灰色系 = 元信息：comments、pragmas
- 所有 hex 颜色在 `#1F1F1F` 背景上满足 WCAG AA 对比度（≥4.5:1），comment 刻意压到 4.5:1 边界以保持"低存在感"的语义
- 颜色值同时提供 hex 和 oklch 表示，hex 作为 CSS 实际值，oklch 记录设计意图便于后续调整

### 五、CSS 架构

highlight.js 高亮后产出的 HTML 带 `hljs-*` class，通过 CSS 后代选择器映射到 `--mutagent-hl-*` 变量：

```css
/* CodeBlock.css 新增部分 */
.mutagent-code-block__code code {
  color: var(--mutagent-hl-text, oklch(0.84 0 0));
  background: transparent;
}

.mutagent-code-block__code .hljs-keyword    { color: var(--mutagent-hl-keyword); }
.mutagent-code-block__code .hljs-string     { color: var(--mutagent-hl-string); }
.mutagent-code-block__code .hljs-number     { color: var(--mutagent-hl-number); }
.mutagent-code-block__code .hljs-comment    { color: var(--mutagent-hl-comment); font-style: italic; }
.mutagent-code-block__code .hljs-function   { color: var(--mutagent-hl-function); }
/* ... 完整映射见实施 */
```

**不引入 highlight.js 内置主题 CSS**：内置主题（如 `github-dark.css`）使用固定颜色值，不经过 CSS 变量。自定义 class 映射方案可以用 `--mutagent-hl-*` 变量覆盖，与 mutgui 主题体系深度集成。

**亮色主题适配**：当前只实现暗色。后续如需亮色主题，只需在亮色 token 作用域下覆盖 `--mutagent-hl-*` 变量值即可，不需改组件或 JS 逻辑。

### 六、流式兼容

`hljs.highlight()` 是同步纯函数，输入不完整代码也能正常产出 HTML（未闭合的 token 按 plaintext 处理）。流式更新路径：

```
text_delta → parser.feed() → CodeBlockItem.code 更新
  → CodeBlock 组件 re-render
  → useMemo 重新调用 hljs.highlight()
  → 新版 <code dangerouslySetInnerHTML> 渲染
```

`useMemo` 依赖 `[code, language]`，code 变化时重新计算。highlight.js 处理 30KB 代码约 ≤1ms，对 60fps 渲染无影响。

### 七、不做的事

- **不做行号** — 阅读场景不需要，和复制按钮同理
- **不做复制按钮** — 可后续单独迭代
- **不做亮色主题色板** — 当前只出暗色。亮色可后续在亮色 token 作用域下追加变量覆盖
- **不做 Shiki/WASM** — 太重，体积不适合前端嵌入场景
- **不做主题切换** — 不在 CodeBlock 内加亮/暗切换，跟随全局 mutgui 主题自动适配
- **不做流式 token 级增量高亮** — 每次 code 变化整体重新高亮，性能足够

## 待定问题

### QUEST Q1: highlight.js 内置 CSS vs 自定义 class 映射

**问题**：highlight.js 提供 90+ 种内置 CSS 主题（如 `github-dark.css`、`monokai.css`），可以直接 import 然后用固定颜色。自定义 class 映射方案需要用 CSS 变量逐个覆盖 `hljs-*` class，维护成本稍高但更灵活。

**建议**：采用自定义 class 映射方案。理由：
1. CSS 变量支持运行时覆盖（插件系统、用户自定义主题）
2. 颜色与 `--mutgui-*` token 体系一致，跨组件协调
3. highlight.js 内置主题颜色值与 mutgui 色系不匹配，硬套会显得突兀
4. 10+ 个 class 映射量不大，维护成本低

### QUEST Q2: 语言别名映射

**问题**：AI 输出的语言标识可能不规范（如 `py` 而非 `python`、`js` 而非 `javascript`、`sh` 而非 `bash`）。highlight.js 内部有别名表但 `hljs.getLanguage()` 不识别。

**建议**：在 `CodeBlock` 组件内加一个简单的别名映射表（~10 条），在调用 `hljs.getLanguage()` 前先做规范化：

```ts
const LANGUAGE_ALIAS: Record<string, string> = {
  py: 'python', js: 'javascript', ts: 'typescript',
  sh: 'bash', zsh: 'bash', yml: 'yaml',
  md: 'markdown', htm: 'xml',
};
```

highlight.js 自己维护完整的别名表在 `highlight.js/lib/languages/` 各自定义中，但 tree-shake 后不暴露。手动维护 10 条别名表成本低。

### QUEST Q3: inline code 高亮

**问题**：`MarkdownContent` 组件中内联 code（`` `code` ``）目前只有颜色 `oklch(0.75 0.15 160)`（浅绿），没有语法高亮。是否需要？

**建议**：不做。内联 code 通常只有 1-3 个 token，完整语法高亮收益低但实现复杂（需要知道上下文语言）。保持单色样式即可。

## 实施步骤清单

- [x] 安装 highlight.js npm 包
- [x] 新建 `frontend/src/components/highlight.ts`：语言注册 + 别名映射
- [x] 改造 `CodeBlock.tsx`：集成 hljs.highlight() + dangerouslySetInnerHTML
- [x] 改造 `CodeBlock.css`：hljs-* class → --mutagent-hl-* CSS 变量映射
- [x] 构建验证：`npm run build` 无错误，检查产出大小
