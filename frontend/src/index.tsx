import { registerComponents } from '@mutgui/core';

import './components/ChatInput.css';
import { ChatInput } from './components/ChatInput';
import {
  UserMessage,
  AssistantMessage,
  AssistantError,
  TurnSeparator,
} from './components/MessageShell';
import { ThinkingBlock } from './components/ThinkingBlock';
import { ToolCallCard } from './components/ToolCallCard';
import { ToolResult } from './components/ToolResult';
import { SectionHeading } from './components/SectionHeading';
import { MarkdownContent } from './components/MarkdownContent';
import { CodeBlock } from './components/CodeBlock';

registerComponents({
  __name__: 'mutagent',
  ChatInput,
  UserMessage,
  AssistantMessage,
  AssistantError,
  TurnSeparator,
  ThinkingBlock,
  ToolCallCard,
  ToolResult,
  SectionHeading,
  MarkdownContent,
  CodeBlock,
});

export { ChatInput };
