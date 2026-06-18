import { useMemo } from 'react';
import { highlightCode } from './highlight';
import './CodeBlock.css';

/* ──── CodeBlock ──── */

export interface CodeBlockProps {
  code: string;
  language: string;
  fenceCount?: number;
}

/** Fenced code block — 语法高亮 + 语言名。
 *
 * 通过 highlight.js 对代码做语法高亮，不支持的语言降级为纯文本。
 * 上下 ``` 和语言名是独立元素，可分别通过
 * ``mutagent-code-block__fence-mark`` / ``mutagent-code-block__fence-lang`` 定制。
 */
export function CodeBlock({ code, language, fenceCount = 3 }: CodeBlockProps) {
  const html = useMemo(
    () => highlightCode(code, language),
    [code, language],
  );

  const fence = '`'.repeat(fenceCount);

  return (
    <div className="mutagent-code-block">
      <div className="mutagent-code-block__fence">
        <span className="mutagent-code-block__fence-mark">{fence}</span>
        {language && <span className="mutagent-code-block__fence-lang">{language}</span>}
      </div>
      <pre className="mutagent-code-block__code">
        <code
          className={language ? `language-${language.toLowerCase()}` : undefined}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </pre>
      <div className="mutagent-code-block__fence">
        <span className="mutagent-code-block__fence-mark">{fence}</span>
      </div>
    </div>
  );
}
