/**
 * highlight.js 语言注册 + 别名映射
 *
 * tree-shake：只打包需要的语言，减小 bundle 体积。
 */

import hljs from 'highlight.js/lib/core';

// ── 基础语言（必须注册，用于降级和未匹配场景） ──
import plaintext from 'highlight.js/lib/languages/plaintext';

// ── 语言 grammar（tree-shake：只打包这些） ──
import python from 'highlight.js/lib/languages/python';
import javascript from 'highlight.js/lib/languages/javascript';
import typescript from 'highlight.js/lib/languages/typescript';
import bash from 'highlight.js/lib/languages/bash';
import json from 'highlight.js/lib/languages/json';
import xml from 'highlight.js/lib/languages/xml';
import css from 'highlight.js/lib/languages/css';
import sql from 'highlight.js/lib/languages/sql';
import yaml from 'highlight.js/lib/languages/yaml';
import markdown from 'highlight.js/lib/languages/markdown';
import toml from 'highlight.js/lib/languages/ini';      // toml 与 ini 共用 grammar
import diff from 'highlight.js/lib/languages/diff';

hljs.registerLanguage('plaintext', plaintext);
hljs.registerLanguage('python', python);
hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('json', json);
hljs.registerLanguage('xml', xml);
hljs.registerLanguage('css', css);
hljs.registerLanguage('sql', sql);
hljs.registerLanguage('yaml', yaml);
hljs.registerLanguage('markdown', markdown);
hljs.registerLanguage('ini', toml);      // 含 toml 别名
hljs.registerLanguage('diff', diff);

// ── 别名映射：AI 可能输出缩写 ──
const ALIAS: Record<string, string> = {
  py: 'python',
  js: 'javascript',
  ts: 'typescript',
  sh: 'bash',
  zsh: 'bash',
  yml: 'yaml',
  md: 'markdown',
  htm: 'xml',
  html: 'xml',
  jsx: 'javascript',
  tsx: 'typescript',
  toml: 'ini',
  patch: 'diff',
};

/**
 * 对代码做语法高亮，返回 HTML 字符串。
 *
 * @param code    原始代码文本
 * @param language  语言标识（可能是别名）；空字符串走 auto 检测
 * @returns  高亮后的 HTML 字符串（hljs-* class 标注 token）
 */
export function highlightCode(code: string, language: string): string {
  const lang = normalizeLanguage(language);

  if (lang && hljs.getLanguage(lang)) {
    return hljs.highlight(code, { language: lang }).value;
  }

  if (language === '') {
    // 未指定语言，自动检测
    return hljs.highlightAuto(code).value;
  }

  // 不认识的语言 → plaintext（内部做 HTML 转义）
  return hljs.highlight(code, { language: 'plaintext' }).value;
}

/** 规范化语言标识：别名 → 正式名；大小写不敏感 */
function normalizeLanguage(language: string): string | null {
  const lower = language.toLowerCase();
  return ALIAS[lower] || lower || null;
}

export { hljs };
