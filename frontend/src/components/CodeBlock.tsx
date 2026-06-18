import './CodeBlock.css';

/* ──── CodeBlock ──── */

export interface CodeBlockProps {
  code: string;
  language: string;
}

/** Fenced code block — 纯文本风格，无边框无背景。
 *
 * 上下 ``` 和语言名是独立元素，可分别通过
 * ``mutagent-code-block__fence-mark`` / ``mutagent-code-block__fence-lang`` 定制。
 */
export function CodeBlock({ code, language }: CodeBlockProps) {
  return (
    <div className="mutagent-code-block">
      <div className="mutagent-code-block__fence">
        <span className="mutagent-code-block__fence-mark">```</span>
        {language && <span className="mutagent-code-block__fence-lang">{language}</span>}
      </div>
      <pre className="mutagent-code-block__code">{code}</pre>
      <div className="mutagent-code-block__fence">
        <span className="mutagent-code-block__fence-mark">```</span>
      </div>
    </div>
  );
}
