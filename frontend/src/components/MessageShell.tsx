import type { ReactNode } from 'react';
import './MessageShell.css';

function formatTime(timestamp: number): string {
  if (!timestamp) return '';
  const d = new Date(timestamp * 1000);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

function joinMeta(...parts: string[]): string {
  return parts.filter(Boolean).join(' · ');
}

/* ──── UserMessage ──── */

export interface UserMessageProps {
  role: string;
  timestamp: number;
  text: string;
}

export function UserMessage({ role, timestamp, text }: UserMessageProps) {
  const meta = joinMeta(role, formatTime(timestamp));
  return (
    <div className="mutagent-message-shell">
      <div className="mutagent-bubble mutagent-bubble--user">
        <div className="mutagent-meta">{meta}</div>
        <div className="mutagent-message-text">{text}</div>
      </div>
    </div>
  );
}

/* ──── AssistantMessage ──── */

export interface AssistantMessageProps {
  role: string;
  model: string;
  timestamp: number;
  children?: ReactNode;
}

export function AssistantMessage({ role, model, timestamp, children }: AssistantMessageProps) {
  const meta = joinMeta(role, model, formatTime(timestamp));
  return (
    <div className="mutagent-message-shell">
      <div className="mutagent-bubble mutagent-bubble--assistant">
        <div className="mutagent-meta">{meta}</div>
        {children}
      </div>
    </div>
  );
}

/* ──── AssistantError ──── */

export interface AssistantErrorProps {
  role: string;
  timestamp: number;
  error: string;
}

export function AssistantError({ role, timestamp, error }: AssistantErrorProps) {
  const meta = joinMeta(role, formatTime(timestamp));
  return (
    <div className="mutagent-message-shell">
      <div className="mutagent-bubble mutagent-bubble--error">
        <div className="mutagent-meta">{meta}</div>
        <pre className="mutagent-error-text">{error}</pre>
      </div>
    </div>
  );
}

/* ──── TurnSeparator ──── */

export interface TurnSeparatorProps {
  inputTokens: number;
  outputTokens: number;
  duration: number;
}

export function TurnSeparator({ inputTokens, outputTokens, duration }: TurnSeparatorProps) {
  const detail = (
    duration || inputTokens || outputTokens
  ) ? `${duration.toFixed(1)}s · in ${inputTokens} · out ${outputTokens}` : 'turn done';

  return (
    <div className="mutagent-turn-separator">
      <div className="mutagent-turn-line" />
      <div className="mutagent-turn-detail">{detail}</div>
      <div className="mutagent-turn-line" />
    </div>
  );
}
