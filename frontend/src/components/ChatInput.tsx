import { Input } from 'antd';
import type { ReactNode } from 'react';

export interface ChatInputProps {
  value?: string;
  sendMode?: string;
  disabled?: boolean;
  placeholder?: string;
  children?: ReactNode;
  onChange?: (value: string) => void;
  onSubmit?: () => void;
}

export function ChatInput({
  value = '',
  sendMode = 'enter',
  disabled = false,
  placeholder = '',
  children = null,
  onChange,
  onSubmit,
}: ChatInputProps) {
  const submitDisabled = disabled || value.trim().length === 0;

  const handlePress = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const native = event.nativeEvent as KeyboardEvent & { isComposing?: boolean };
    if (event.key !== 'Enter') return;
    if (native.isComposing || native.keyCode === 229) return;
    const shouldSubmit = sendMode === 'enter'
      ? !event.shiftKey && !event.ctrlKey && !event.metaKey
      : (event.ctrlKey || event.metaKey) && !event.shiftKey;
    if (!shouldSubmit || submitDisabled) return;
    event.preventDefault();
    event.stopPropagation();
    onSubmit?.();
  };

  return (
    <div className="mutagent-chat-input-shell" data-send-mode={sendMode}>
      <Input.TextArea
        value={value}
        variant="borderless"
        autoSize={{ minRows: 1, maxRows: 8 }}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(event) => onChange?.(event.target.value)}
        onKeyDown={handlePress}
      />
      <div className="mutagent-chat-input-toolbar">
        {children}
      </div>
    </div>
  );
}
