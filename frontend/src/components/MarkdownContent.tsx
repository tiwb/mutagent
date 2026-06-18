import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './MarkdownContent.css';

/* ──── MarkdownContent ──── */

export interface MarkdownContentProps {
  text: string;
}

/** Markdown 文本内容块。使用 react-markdown 渲染，fenced code block 由 CodeBlock 独立处理。
 *
 * 样式集中在 ``MarkdownContent.css``，通过 ``.mutagent-markdown-content`` 及其
 * 后代选择器（code / pre / p / table 等）控制，可直接覆盖。
 */
export function MarkdownContent({ text }: MarkdownContentProps) {
  return (
    <div className="mutagent-markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        allowedElements={[
          'p', 'strong', 'em', 'code', 'a', 'del',
          'ul', 'ol', 'li', 'blockquote',
          'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
          'table', 'thead', 'tbody', 'tr', 'th', 'td',
          'hr', 'br', 'img',
        ]}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
