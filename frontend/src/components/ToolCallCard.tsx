import { ToolResult } from './ToolResult';
import './ToolCard.css';

type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

const STATUS_COLORS: Record<string, string> = {
  pending: '#d4a72c',
  success: '#2fb171',
  error: '#e5534b',
  cancelled: '#8b949e',
};

function toPythonRepr(value: JsonValue, indent: number): string {
  if (value === null) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (typeof value === 'number') return String(value);
  if (typeof value === 'string') {
    if (value.includes("'") && !value.includes('"')) return `"${value}"`;
    return `'${value}'`;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]';
    const items = value.map(v => toPythonRepr(v, indent + 1));
    const joined = items.join(', ');
    if (joined.length <= 60) return `[${joined}]`;
    const pad = '  '.repeat(indent + 1);
    const inner = items.map(i => `${pad}${i}`).join(',\n');
    return `[\n${inner},\n${'  '.repeat(indent)}]`;
  }
  const keys = Object.keys(value);
  if (keys.length === 0) return '{}';
  const entries = keys.map(k =>
    `${'  '.repeat(indent + 1)}'${k}': ${toPythonRepr((value as Record<string, JsonValue>)[k], indent + 1)}`
  );
  return `{\n${entries.join(',\n')},\n${'  '.repeat(indent)}}`;
}

function renderInput(name: string, input: Record<string, JsonValue>): string {
  const entries = Object.entries(input);
  if (entries.length === 0) return `${name}()`;

  const oneLineParts = entries.map(([k, v]) => `${k}=${toPythonRepr(v, 0)}`);
  const oneLine = `${name}(${oneLineParts.join(', ')})`;
  const useMultiLine = entries.length > 3 || oneLine.length > 80;

  if (useMultiLine) {
    const lines = entries.map(([k, v]) => `    ${k}=${toPythonRepr(v, 1)}`);
    return `${name}(\n${lines.join(',\n')},\n)`;
  }
  return oneLine;
}

export interface ToolCallCardProps {
  name: string;
  status: 'pending' | 'success' | 'error' | 'cancelled';
  input: Record<string, JsonValue>;
  resultText?: string;
  isError?: boolean;
}

export function ToolCallCard({ name, status, input, resultText, isError }: ToolCallCardProps) {
  const color = STATUS_COLORS[status] || STATUS_COLORS.cancelled;
  const code = renderInput(name, input);

  return (
    <div className="mutagent-tool-card" style={{ borderColor: color }}>
      <pre className="mutagent-tool-pre">{code}</pre>
      {resultText && (
        <>
          <div className="mutagent-tool-divider" />
          <ToolResult result={resultText} isError={isError || false} />
        </>
      )}
    </div>
  );
}
