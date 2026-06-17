import type { ReactNode } from 'react';
import './ThinkingBlock.css';

export interface ThinkingBlockProps {
  thinking: string;
  expanded: boolean;
  children?: ReactNode; /* antd.Button for toggle — rendered from Python Callback */
}

export function ThinkingBlock({ thinking, expanded, children }: ThinkingBlockProps) {
  return (
    <div className="mutagent-thinking-shell">
      <div className="mutagent-thinking-header">
        <span className="mutagent-thinking-title">思考过程</span>
        {children}
      </div>
      {expanded && (
        <pre className="mutagent-thinking-body">{thinking}</pre>
      )}
    </div>
  );
}
