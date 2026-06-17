import './ToolCard.css';

export interface ToolResultProps {
  result: string;
  isError: boolean;
}

export function ToolResult({ result, isError }: ToolResultProps) {
  if (!result) return null;
  return (
    <pre className={`mutagent-tool-pre ${isError ? 'mutagent-tool-pre--error' : ''}`}>
      {result}
    </pre>
  );
}
